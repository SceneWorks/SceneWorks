use super::*;

/// Maximum wall-clock lifetime of one FFmpeg child. Timeline exports and CPU transcodes can
/// legitimately run much longer than model inference, so the default is deliberately generous;
/// the important invariant is that a wedged child is finite. Operators can lower or raise it with
/// `SCENEWORKS_FFMPEG_TIMEOUT_SECONDS`. Zero/invalid values fail closed to this bounded default
/// rather than disabling the watchdog.
const FFMPEG_EXECUTION_TIMEOUT: Duration = Duration::from_secs(12 * 60 * 60);
/// An operator override remains finite and representable as a Tokio deadline. Seven days is far
/// beyond a legitimate SceneWorks export while preventing a typo such as `u64::MAX` from either
/// panicking in `Instant + Duration` or effectively disabling the watchdog.
pub(crate) const FFMPEG_EXECUTION_TIMEOUT_MAX: Duration = Duration::from_secs(7 * 24 * 60 * 60);
const FFMPEG_TEARDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const FFMPEG_TIMEOUT_SECONDS_ENV: &str = "SCENEWORKS_FFMPEG_TIMEOUT_SECONDS";

pub(crate) fn ffmpeg_execution_timeout_from(raw: Option<&str>) -> Duration {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(FFMPEG_EXECUTION_TIMEOUT)
        .min(FFMPEG_EXECUTION_TIMEOUT_MAX)
}

fn ffmpeg_execution_timeout() -> Duration {
    ffmpeg_execution_timeout_from(std::env::var(FFMPEG_TIMEOUT_SECONDS_ENV).ok().as_deref())
}

fn ffmpeg_timeout_io_error(timeout: Duration) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        format!("FFmpeg execution exceeded its {timeout:?} deadline."),
    )
}

fn checked_deadline_from(now: tokio::time::Instant, timeout: Duration) -> tokio::time::Instant {
    // If an exotic runtime Instant cannot represent even the capped duration, expiring immediately
    // is safer than panicking or silently running without a deadline.
    now.checked_add(timeout).unwrap_or(now)
}

pub(crate) fn ffmpeg_execution_deadline(requested: Duration) -> (tokio::time::Instant, Duration) {
    let effective = if requested.is_zero() {
        FFMPEG_EXECUTION_TIMEOUT
    } else {
        requested.min(FFMPEG_EXECUTION_TIMEOUT_MAX)
    };
    (
        checked_deadline_from(tokio::time::Instant::now(), effective),
        effective,
    )
}

#[derive(Debug, Clone)]
pub(crate) struct TimelineExportRequest {
    project_id: String,
    timeline_id: String,
    timeline_name: String,
    timeline_path: String,
    resolution: u32,
    fps: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct FfmpegContext<'a> {
    api: &'a ApiClient,
    settings: &'a Settings,
    job_id: &'a str,
    cancel_message: &'a str,
}

impl<'a> FfmpegContext<'a> {
    /// Build a context so callers in sibling modules (e.g. video generation,
    /// `video_jobs`) can drive [`run_ffmpeg`] with the same periodic-heartbeat +
    /// cooperative-cancel loop the in-module callers use.
    pub(crate) fn new(
        api: &'a ApiClient,
        settings: &'a Settings,
        job_id: &'a str,
        cancel_message: &'a str,
    ) -> Self {
        Self {
            api,
            settings,
            job_id,
            cancel_message,
        }
    }
}

pub(crate) async fn run_frame_extract_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing frame extraction.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Frame extraction canceled before reading media.",
    )
    .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Extracting,
            0.25,
            "Extracting timeline frame.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let result = run_frame_extract(api, settings, job).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Timeline frame saved as an asset.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

/// The resolved source of a frame-extract / person-detect job: the project store + path, the source
/// asset JSON, and the on-disk media path (existence-checked). Shared prologue of both frame handlers
/// (sc-8914 / F-112) — they resolve the same `projectId` + `sourceAssetId` identically.
struct FrameSource {
    store: ProjectStore,
    project_path: PathBuf,
    source_asset: Value,
    source_media_path: PathBuf,
}

/// Resolve the `projectId` + `sourceAssetId` a frame job carries into a [`FrameSource`], confining the
/// source media path with `safe_project_path` and erroring if it is missing.
fn resolve_frame_source(settings: &Settings, job: &JobSnapshot) -> WorkerResult<FrameSource> {
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let source_asset_id = required_payload_string(&job.payload, "sourceAssetId")?;
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    let project_path = PathBuf::from(project.path);
    let source_asset = store.get_asset(project_id, source_asset_id)?;
    let source_media_rel = required_value_str(
        source_asset.get("file").ok_or_else(|| {
            WorkerError::InvalidPayload("Source asset file is missing.".to_owned())
        })?,
        "path",
    )?;
    let source_media_path = safe_project_path(&project_path, source_media_rel)?;
    if !source_media_path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "Source media not found: {}",
            source_media_path.display()
        )));
    }
    Ok(FrameSource {
        store,
        project_path,
        source_asset,
        source_media_path,
    })
}

/// The per-frame identity handed to the asset-builder closure of [`render_frame_asset`]: the freshly
/// rendered frame's on-disk media path, its ids, and the source path relative to the project — enough
/// for the closure to build the `"type":"frame"` asset JSON (its recipe/lineage/detection payload).
struct RenderedFrame {
    asset_id: String,
    created_at: String,
    media_rel: String,
    media_path: PathBuf,
    source_rel: String,
}

/// The single resolve→render→tmp-rename→asset-JSON→sidecar+recipe→index flow shared by both frame
/// handlers (sc-8914 / F-112). Renders one frame at `timestamp` (dims `width×height`) into
/// `assets/frames/{date}_{filename_kind}_{suffix}.png`, promotes it out of its `.tmp.png`, posts the
/// `Saving` progress update, guards the promotion with a cancel check that cleans up the frame on
/// cancel, then lets `build_asset` produce the full asset JSON (it runs *after* the frame exists, so
/// person-detect can run its detector on the rendered PNG). Writes the sidecar + recipe and indexes
/// the asset. Returns the asset id, the built asset JSON, and any per-handler `extra` the builder
/// produced alongside it (the detection payload for person-detect, `()` for frame-extract) — passed
/// back through the return value rather than stashed in an interior-mutability cell, so the handler
/// future stays `Send` for the job dispatcher's `tokio::spawn`.
#[allow(clippy::too_many_arguments)]
async fn render_frame_asset<F, Fut, T>(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    source: &FrameSource,
    timestamp: f64,
    width: u32,
    height: u32,
    filename_kind: &str,
    ffmpeg_cancel_message: &str,
    saving_progress: f64,
    saving_message: &str,
    promotion_cancel_message: &str,
    build_asset: F,
) -> WorkerResult<(String, Value, T)>
where
    F: FnOnce(RenderedFrame) -> Fut,
    Fut: std::future::Future<Output = WorkerResult<(Value, T)>>,
{
    let project_path = &source.project_path;
    tokio::fs::create_dir_all(project_path.join("assets").join("frames")).await?;
    tokio::fs::create_dir_all(project_path.join("recipes")).await?;
    let asset_id = fresh_asset_id();
    let created_at = now_rfc3339();
    let filename = format!(
        "{}_{filename_kind}_{}.png",
        &created_at[..10],
        asset_suffix(&asset_id)
    );
    let media_rel = format!("assets/frames/{filename}");
    let media_path = project_path.join(&media_rel);
    let temp_path = media_path.with_extension("tmp.png");

    let ffmpeg_context = FfmpegContext {
        api,
        settings,
        job_id: &job.id,
        cancel_message: ffmpeg_cancel_message,
    };
    // sc-19549: seek to a frame that actually EXISTS. `timestamp` is a playhead, which can sit past
    // the end of the source (a still has one frame at t=0; a timeline item can outlive its clip),
    // and `-ss` past the end encodes nothing while ffmpeg still exits 0 — so this used to produce no
    // file at all, silently, for every image asset extracted at a playhead > 0.
    let seek = resolve_frame_seek(
        "ffmpeg",
        &source.source_asset,
        &source.source_media_path,
        timestamp,
    )
    .await?;
    render_frame_png(
        "ffmpeg",
        &source.source_media_path,
        &temp_path,
        seek,
        width,
        height,
        Some(ffmpeg_context),
    )
    .await?;
    tokio::fs::rename(&temp_path, &media_path).await?;

    let source_rel = relative_path(project_path, &source.source_media_path)?;
    // Build the asset JSON now that the frame exists on disk (person-detect runs its detector here).
    let (asset, extra) = build_asset(RenderedFrame {
        asset_id: asset_id.clone(),
        created_at,
        media_rel,
        media_path: media_path.clone(),
        source_rel,
    })
    .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Saving,
            ProgressStage::Saving,
            saving_progress,
            saving_message,
            None,
            None,
            None,
        ),
    )
    .await?;
    if let Err(error) = check_cancel(api, &job.id, promotion_cancel_message).await {
        let _ = tokio::fs::remove_file(&media_path).await;
        return Err(error);
    }

    let sidecar_path = media_path.with_extension("sceneworks.json");
    let project_id = required_payload_string(&job.payload, "projectId")?;
    source
        .store
        .persist_native_asset_sidecar(project_id, &sidecar_path, &asset)?;

    Ok((asset_id, asset, extra))
}

pub(crate) async fn run_frame_extract(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<JsonObject> {
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let source_asset_id = required_payload_string(&job.payload, "sourceAssetId")?;
    let timestamp = payload_f64(&job.payload, "sourceTimestamp", 0.0).clamp(0.0, 3600.0);
    let source = resolve_frame_source(settings, job)?;

    let intended_use = optional_payload_string(&job.payload, "intendedUse").unwrap_or("reuse");
    let source_display_name = source
        .source_asset
        .get("displayName")
        .and_then(Value::as_str)
        .unwrap_or("clip")
        .to_owned();
    let (asset_id, asset, ()) = render_frame_asset(
        api,
        settings,
        job,
        &source,
        timestamp,
        1920,
        1080,
        "frame",
        "Frame extraction canceled by user.",
        0.85,
        "Saving extracted frame asset.",
        "Frame extraction canceled before asset promotion.",
        |frame| async move {
            let timeline_id = job
                .payload
                .get("timelineId")
                .cloned()
                .unwrap_or(Value::Null);
            let timeline_item_id = job
                .payload
                .get("timelineItemId")
                .cloned()
                .unwrap_or(Value::Null);
            let playhead_seconds = job
                .payload
                .get("playheadSeconds")
                .cloned()
                .unwrap_or(Value::Null);
            Ok((
                json!({
                "schemaVersion": 1,
                "id": frame.asset_id,
                "projectId": project_id,
                "generationSetId": Value::Null,
                "type": "frame",
                "displayName": format!("Frame {timestamp:.2}s from {source_display_name}"),
                "createdAt": frame.created_at,
                "file": {
                    "path": frame.media_rel,
                    "mimeType": "image/png",
                    "width": 1920,
                    "height": 1080,
                    "duration": Value::Null,
                    "fps": Value::Null
                },
                "status": {
                    "favorite": false,
                    "rating": 0,
                    "rejected": false,
                    "trashed": false
                },
                "recipe": {
                    "mode": "frame_extract",
                    "model": "timeline-frame-extract",
                    "adapter": "ffmpeg-frame-extract",
                    "prompt": format!("Extract frame at {timestamp:.2}s"),
                    "negativePrompt": "",
                    "seed": 0,
                    "loras": [],
                    "stylePreset": "none",
                    "normalizedSettings": {
                        "timelineId": timeline_id,
                        "timelineItemId": timeline_item_id,
                        "playheadSeconds": playhead_seconds,
                        "sourceTimestamp": timestamp,
                        "intendedUse": intended_use
                    },
                    "rawAdapterSettings": { "sourcePath": frame.source_rel }
                },
                "lineage": {
                    "parents": [source_asset_id],
                    "sourceAssetId": source_asset_id,
                    "sourceTimestamp": timestamp,
                    "timelineId": job.payload.get("timelineId").cloned().unwrap_or(Value::Null),
                    "timelineItemId": job.payload.get("timelineItemId").cloned().unwrap_or(Value::Null),
                    "intendedUse": intended_use,
                    "jobId": job.id
                }
                }),
                (),
            ))
        },
    )
    .await?;

    let mut result = JsonObject::new();
    result.insert("assetIds".to_owned(), json!([asset_id]));
    result.insert("assets".to_owned(), json!([asset]));
    result.insert(
        "sourceAssetId".to_owned(),
        Value::String(source_asset_id.to_owned()),
    );
    result.insert("sourceTimestamp".to_owned(), json!(timestamp));
    result.insert(
        "timelineId".to_owned(),
        job.payload
            .get("timelineId")
            .cloned()
            .unwrap_or(Value::Null),
    );
    result.insert(
        "timelineItemId".to_owned(),
        job.payload
            .get("timelineItemId")
            .cloned()
            .unwrap_or(Value::Null),
    );
    Ok(result)
}

pub(crate) async fn run_person_detect_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing representative frame analysis.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Person detection canceled before frame extraction.",
    )
    .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Extracting,
            0.25,
            "Extracting representative frame.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let result = run_person_detect(api, settings, job).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Person candidates detected.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

/// The largest seek (seconds) a person-detect frame extraction will ever accept.
const PERSON_DETECT_MAX_TIMESTAMP_SECONDS: f64 = 3600.0;
pub(crate) const PERSON_DETECTOR_MODEL: &str = "yolo11m";
#[cfg(target_os = "macos")]
pub(crate) const PERSON_DETECTOR_ADAPTER: &str = "yolo11_mlx";
#[cfg(not(target_os = "macos"))]
pub(crate) const PERSON_DETECTOR_ADAPTER: &str = "yolo11_ort";
#[cfg(target_os = "macos")]
pub(crate) const PERSON_DETECTOR_BACKEND: &str = "mlx";
#[cfg(not(target_os = "macos"))]
pub(crate) const PERSON_DETECTOR_BACKEND: &str = "ort";

/// Clamp the requested `sourceTimestamp` into a seek that lands inside the source clip.
///
/// The upper bound is `min(duration, 3600)`, not `max(duration, 3600)`: a timestamp past the
/// end of a short clip is pulled back to the clip's duration so `ffmpeg -ss` still produces a
/// frame, instead of sailing past the end and failing with a generic "no frame output" error
/// (sc-8831). The 3600s ceiling only ever tightens the bound for absurdly long clips.
fn person_detect_source_timestamp(timestamp: f64, duration: f64) -> f64 {
    timestamp.clamp(0.0, duration.min(PERSON_DETECT_MAX_TIMESTAMP_SECONDS))
}

pub(crate) async fn run_person_detect(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<JsonObject> {
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let source_asset_id = required_payload_string(&job.payload, "sourceAssetId")?;
    let source = resolve_frame_source(settings, job)?;

    let duration = source
        .source_asset
        .get("file")
        .and_then(|file| file.get("duration"))
        .map_or(6.0, |value| value_f64(value, 6.0))
        .clamp(0.0, 3600.0);
    let timestamp = person_detect_source_timestamp(
        payload_f64(
            &job.payload,
            "sourceTimestamp",
            if duration > 0.0 { duration * 0.25 } else { 0.0 },
        ),
        duration,
    );

    // Preview jobs (`preview: true`, claimed via the CPU worker's
    // person_detect_preview capability) keep the procedural placeholder. Real
    // jobs run the YOLO11 onnx detector (epic 3482, sc-3633) — model-backed,
    // `personDetectionActive: true`, and erroring honestly when the detector
    // can't run rather than silently degrading to boxes.
    let is_preview = job
        .payload
        .get("preview")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let confidence = job
        .payload
        .get("advanced")
        .and_then(|advanced| advanced.get("confidence"))
        .or_else(|| job.payload.get("confidence"))
        .map_or(0.25, |value| value_f64(value, 0.25))
        .clamp(0.01, 1.0);
    let source_display_name = source
        .source_asset
        .get("displayName")
        .and_then(Value::as_str)
        .unwrap_or("clip")
        .to_owned();

    // The detector runs inside the asset-builder (it needs the rendered frame on disk); its outputs
    // also feed the outer result payload, so the builder returns them as the `extra` tuple alongside
    // the asset JSON (keeping the handler future `Send` for the dispatcher's `tokio::spawn`).
    let (asset_id, asset, (detections, detection_active)) = render_frame_asset(
        api,
        settings,
        job,
        &source,
        timestamp,
        1280,
        720,
        "person-frame",
        "Person detection canceled by user.",
        0.78,
        "Saving representative frame and candidate boxes.",
        "Person detection canceled before asset promotion.",
        |frame| async move {
            let (detections, detector_model, detector_adapter, detection_active, detector_meta) =
                if is_preview {
                    (
                        candidate_people(1280, 720, source_asset_id, timestamp),
                        "procedural-person-detector".to_owned(),
                        "procedural_person_tracking",
                        false,
                        Value::Null,
                    )
                } else {
                    let (boxes, device) = run_yolo11_person_detect(
                        api,
                        settings,
                        job,
                        frame.media_path.clone(),
                        confidence,
                    )
                    .await?;
                    (
                        boxes,
                        PERSON_DETECTOR_MODEL.to_owned(),
                        PERSON_DETECTOR_ADAPTER,
                        true,
                        json!({
                            "backend": PERSON_DETECTOR_BACKEND,
                            "device": device,
                            "model": PERSON_DETECTOR_MODEL
                        }),
                    )
                };
            let asset = json!({
                "schemaVersion": 1,
                "id": frame.asset_id,
                "projectId": project_id,
                "generationSetId": Value::Null,
                "type": "frame",
                "displayName": format!("Person selection frame from {source_display_name}"),
                "createdAt": frame.created_at,
                "file": {
                    "path": frame.media_rel,
                    "mimeType": "image/png",
                    "width": 1280,
                    "height": 720,
                    "duration": Value::Null,
                    "fps": Value::Null
                },
                "status": {
                    "favorite": false,
                    "rating": 0,
                    "rejected": false,
                    "trashed": false
                },
                "recipe": {
                    "mode": "person_detect",
                    "model": detector_model,
                    "adapter": detector_adapter,
                    "prompt": "Detect selectable people in representative frame",
                    "negativePrompt": "",
                    "seed": 0,
                    "loras": [],
                    "stylePreset": "none",
                    "normalizedSettings": {
                        "sourceTimestamp": timestamp,
                        "detectionCount": detections.len(),
                        "confidence": confidence,
                        "personDetectionActive": detection_active,
                        "detector": detector_meta
                    },
                    "rawAdapterSettings": { "sourcePath": frame.source_rel }
                },
                "lineage": {
                    "parents": [source_asset_id],
                    "sourceAssetId": source_asset_id,
                    "sourceTimestamp": timestamp,
                    "jobId": job.id
                }
            });
            Ok((asset, (detections, detection_active)))
        },
    )
    .await?;

    let mut result = JsonObject::new();
    result.insert("frameAssetId".to_owned(), Value::String(asset_id));
    result.insert("frameAsset".to_owned(), asset);
    result.insert(
        "sourceAssetId".to_owned(),
        Value::String(source_asset_id.to_owned()),
    );
    result.insert("sourceTimestamp".to_owned(), json!(timestamp));
    result.insert("detections".to_owned(), Value::Array(detections));
    result.insert(
        "personDetectionActive".to_owned(),
        Value::Bool(detection_active),
    );
    result.insert(
        "limits".to_owned(),
        json!({
            "maskStorage": "deferred",
            "correction": "single selected box corrections can be added to the track sidecar later"
        }),
    );
    Ok(result)
}

/// Run the YOLO11 person detector on a rendered frame, returning the normalized detection
/// array (Python `run_person_detect` shape) + the device the model ran on. Available on Mac
/// (native MLX, epic 3482 / sc-3633) AND the off-Mac candle GPU-worker lane (`ort`/CUDA,
/// sc-5498); identical body — the platform backend is chosen inside `person_jobs`. A
/// candle-disabled off-Mac build refuses the job (the stub below).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn run_yolo11_person_detect(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    frame_path: PathBuf,
    confidence: f64,
) -> WorkerResult<(Vec<Value>, &'static str)> {
    let weights = crate::person_jobs::require_detector_weights(settings)?;
    let conf = confidence as f32;
    // Keep the worker heartbeat alive across the blocking YOLO11 detect (cold weight load +
    // inference) so a slow detection never trips the API's 90s stale-sweep (sc-8390). Cancel stays
    // `None` by explicit per-engine decision (sc-9123): YOLO11 person-detect is one bounded forward
    // pass on one frame — no loop for a flag to interrupt, so the bounded join already gives
    // everything a cancel flag would.
    let result = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        None,
        "",
        "person detect task",
        crate::no_cancel_ack(),
        tokio::task::spawn_blocking(move || {
            crate::person_jobs::detect_people_blocking(weights, frame_path, conf)
        }),
    )
    .await?;
    let boxes =
        crate::person_jobs::detections_to_json(&result.detections, result.width, result.height);
    Ok((boxes, result.device))
}

#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
async fn run_yolo11_person_detect(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
    _frame_path: PathBuf,
    _confidence: f64,
) -> WorkerResult<(Vec<Value>, &'static str)> {
    Err(WorkerError::InvalidPayload(
        "Real person detection requires the candle GPU worker on this platform.".to_owned(),
    ))
}

/// Seconds the final frame-extraction seek is held inside the clip. `sample_timestamps`
/// is inclusive of both ends, so its last sample is exactly `duration` — but a video has
/// no frame at `duration` (the last decodable frame sits ~`1/fps` before it), and an
/// `ffmpeg -ss duration` accurate seek then yields no output and fails the whole track.
/// 0.2 s clears one frame for any clip ≥ 5 fps without meaningfully moving the sample.
///
/// UNGATED (sc-19549): this was `cfg(macos | backend-candle)` while its second caller,
/// `render_frame_asset`, is compiled on EVERY target. Leaving the guard gated while the caller
/// is not is precisely the "one half of a pair moved" break — the plain-Linux worker that the
/// `parity-rust` lane builds would not have had the constant at all.
const FRAME_SEEK_GUARD_SECONDS: f64 = 0.2;

/// Clamp a sample timestamp to a frame-extraction seek that always lands on a real frame:
/// never past `duration - FRAME_SEEK_GUARD_SECONDS`. Only the final inclusive-end sample is
/// affected; every interior sample passes through unchanged. The tracker still records the
/// logical sample time — only the seek used to pull pixels is clamped.
///
/// This is the ONE clamping rule for every `-ss` frame seek in this module — the person-track
/// sampler, the single-pass `select` threshold, and (since sc-19549) the single-frame
/// `render_frame_asset` path all route through it rather than spelling their own bound.
pub(crate) fn frame_seek_timestamp(timestamp: f64, duration: f64) -> f64 {
    timestamp.min((duration - FRAME_SEEK_GUARD_SECONDS).max(0.0))
}

/// Is this asset a still picture rather than a timed clip? The ONE spelling of that question in
/// this module — `render_item_segment` (which loops a still with `-loop 1`) and the frame-extract
/// seek resolver below both call it, so the two can never disagree about what a still is.
pub(crate) fn asset_is_image_source(asset: &Value) -> bool {
    let media_type = asset.get("type").and_then(Value::as_str);
    let mime_type = asset
        .get("file")
        .and_then(|file| file.get("mimeType"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    media_type != Some("video") && (media_type == Some("image") || mime_type.starts_with("image/"))
}

/// Resolve the ffmpeg binary a probe should spawn, honouring `SCENEWORKS_FFMPEG` exactly the way
/// the render paths do. Shared by both probes so a lane that points at a bundled ffmpeg cannot end
/// up probing one binary and rendering with another.
fn resolve_probe_program(ffmpeg: &str) -> String {
    let configured = match std::env::var("SCENEWORKS_FFMPEG") {
        Ok(path) if ffmpeg == "ffmpeg" && !path.trim().is_empty() => path,
        _ => ffmpeg.to_owned(),
    };
    sceneworks_core::media_convert::resolve_ffmpeg_program(&configured).into_owned()
}

/// Parse an ffmpeg `HH:MM:SS.ms` clock into seconds.
fn parse_ffmpeg_clock(clock: &str) -> Option<f64> {
    let mut parts = clock.trim().split(':');
    let hours: f64 = parts.next()?.trim().parse().ok()?;
    let minutes: f64 = parts.next()?.trim().parse().ok()?;
    let seconds: f64 = parts.next()?.trim().parse().ok()?;
    Some(hours * 3600.0 + minutes * 60.0 + seconds)
}

/// The container-header duration ffmpeg prints as `Duration: 00:00:02.00, start: ...`. `None` when
/// the header says `Duration: N/A` (image pipes and raw elementary streams both do).
fn parse_ffmpeg_header_duration(stderr: &str) -> Option<f64> {
    let idx = stderr.find("Duration:")?;
    let rest = &stderr[idx + "Duration:".len()..];
    let clock = rest.split(',').next()?;
    parse_ffmpeg_clock(clock)
}

/// The last `time=HH:MM:SS.ms` progress token of a completed ffmpeg pass — the MEASURED end of the
/// decoded stream, which is what a source with no usable container header has instead.
fn parse_ffmpeg_measured_time(stderr: &str) -> Option<f64> {
    stderr.rmatch_indices("time=").find_map(|(idx, _)| {
        let rest = &stderr[idx + "time=".len()..];
        let clock = rest.split_whitespace().next()?;
        parse_ffmpeg_clock(clock)
    })
}

/// Measure the seekable duration of `source_path` so [`frame_seek_timestamp`] has a real bound to
/// clamp against (sc-19549).
///
/// **Failure is an `Err`, never a quiet `None`.** That is deliberate and is the lesson of
/// [`probe_source_frame_count`] directly above, whose `Option` return made "ffmpeg 6 printed no
/// `frame=` token" indistinguishable from "this source legitimately has no frame count", so every
/// caller on every ffmpeg-6 host silently took the fallback and nobody could see it. Every timed
/// source HAS a duration, so there is no legitimate absence here to be confused with: if we cannot
/// measure one, the job fails with ffmpeg's own stderr tail in the message.
///
/// Two tiers, cheapest first:
/// 1. the container header (`ffmpeg -i` alone, which exits non-zero after printing it — no decode);
/// 2. when the header says `N/A`, a real decode pass (`-f null -`) and its final `time=` token.
///
/// Tier 2 decodes rather than `-c copy`-walks on purpose: `-c copy` is exactly the form that
/// emitted no progress token on ffmpeg 6.1.1. `time=` on a decoding pass is emitted by every
/// ffmpeg we support.
async fn probe_source_duration(ffmpeg: &str, source_path: &Path) -> WorkerResult<f64> {
    let program = resolve_probe_program(ffmpeg);
    let path = source_path.display().to_string();

    let mut header = Command::new(&program);
    header.args(["-hide_banner", "-i", &path]);
    let header = run_ffmpeg_probe_command(header).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::TimedOut {
            WorkerError::Io(error)
        } else {
            WorkerError::InvalidPayload(format!(
                "Could not run ffmpeg to measure the duration of {path}: {error}"
            ))
        }
    })?;
    let header_stderr = String::from_utf8_lossy(&header.stderr).into_owned();
    if let Some(duration) = parse_ffmpeg_header_duration(&header_stderr) {
        return Ok(duration);
    }

    // No usable header (`Duration: N/A`) — measure it by decoding.
    let mut measured = Command::new(&program);
    measured.args([
        "-hide_banner",
        "-i",
        &path,
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]);
    let measured = run_ffmpeg_probe_command(measured).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::TimedOut {
            WorkerError::Io(error)
        } else {
            WorkerError::InvalidPayload(format!(
                "Could not run ffmpeg to measure the duration of {path}: {error}"
            ))
        }
    })?;
    let measured_stderr = String::from_utf8_lossy(&measured.stderr).into_owned();
    parse_ffmpeg_measured_time(&measured_stderr).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Could not determine the duration of {path}: ffmpeg reported neither a container \
             duration nor a measured time. ffmpeg said: {}",
            ffmpeg_stderr_tail(&measured_stderr)
        ))
    })
}

/// The last few ffmpeg stderr lines, for an error message that a human can act on.
fn ffmpeg_stderr_tail(stderr: &str) -> String {
    let lines: Vec<&str> = stderr
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    lines[lines.len().saturating_sub(4)..].join(" | ")
}

/// The `-ss` seek a single-frame extraction must use so it always lands on a real frame (sc-19549).
///
/// `ffmpeg -ss T -i src -frames:v 1` past the end of a source encodes NOTHING and still **exits 0**,
/// so before this existed any extraction at a playhead beyond the source silently produced no file.
///
/// - **Still pictures clamp to 0.** A still is one frame at t=0 and has no timeline of its own, so
///   every playhead over it denotes that same single frame — returning it is the right answer, not
///   a consolation prize. (This is *not* the "just retry at 0" shortcut, which is wrong for video
///   precisely because a video's frames differ; see the next bullet.)
/// - **Timed sources clamp to the last real frame** via [`frame_seek_timestamp`] against a measured
///   duration, so a playhead past the end yields the END of the clip, never the beginning.
async fn resolve_frame_seek(
    ffmpeg: &str,
    source_asset: &Value,
    source_path: &Path,
    timestamp: f64,
) -> WorkerResult<f64> {
    if asset_is_image_source(source_asset) {
        return Ok(0.0);
    }
    let duration = probe_source_duration(ffmpeg, source_path).await?;
    Ok(frame_seek_timestamp(timestamp, duration))
}

/// The native-MLX person segmenter used by the macOS person-track masking pass.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PersonSegmenter {
    /// SAM3 text-concept (PCS) — the box-prompt-free default (sc-4926).
    Sam3,
    /// Legacy SAM2 box-prompt video predictor (epic 3704) — kept for A/B parity + fallback.
    Sam2,
}

#[cfg(target_os = "macos")]
impl PersonSegmenter {
    /// The sidecar `tracker_meta.segmenter` id for this backend.
    fn meta_label(self) -> &'static str {
        match self {
            PersonSegmenter::Sam3 => "sam3",
            PersonSegmenter::Sam2 => "sam2.1_hiera_large",
        }
    }
}

/// Select the person segmenter: SAM3 PCS by default, the legacy SAM2 box-prompt path under
/// `SCENEWORKS_PERSON_SEGMENTER=sam2` (A/B parity validation + cutover fallback, sc-4926).
#[cfg(target_os = "macos")]
fn person_segmenter_kind() -> PersonSegmenter {
    match std::env::var("SCENEWORKS_PERSON_SEGMENTER").ok().as_deref() {
        Some("sam2") => PersonSegmenter::Sam2,
        _ => PersonSegmenter::Sam3,
    }
}

/// The real (model-backed) outcome of `assemble_real_person_track`: the resampled track frames
/// plus the metadata `run_person_track` folds into the sidecar.
struct RealPersonTrack {
    frames: Vec<Value>,
    average_confidence: f64,
    /// Sidecar `maskState` from the SAM2 segmentation pass (sc-3709): active / generated /
    /// degraded (segmenter unavailable → box-mask fallback) / missing (segmentation off).
    mask_state: &'static str,
    quality: Value,
    tracker_meta: Value,
}

/// The resolved person-track fields the sidecar assembly consumes, produced by ONE of the two tracking
/// paths — the procedural CPU-preview placeholder or the real native detector+tracker run. Replaces the
/// 8-tuple `run_person_track` used to bind out of its preview/real `if/else` (sc-8921, F-119): every
/// consumer now reads a named field, and the two paths must fill the same shape rather than agreeing on
/// positional order. `person_active` distinguishes the paths (false = preview placeholder).
struct PersonTrackOutcome {
    frames: Vec<Value>,
    average_confidence: f64,
    /// Sidecar `maskState` (`active` / `generated` / `degraded` / `missing`, or `deferred` for preview).
    mask_state: &'static str,
    /// Whether a real detector/tracker ran (`true`) vs the procedural preview placeholder (`false`).
    person_active: bool,
    /// `recipe.model` provenance label (e.g. `yolo11m` / `procedural-person-tracker`).
    tracker_model: String,
    /// `recipe.adapter` provenance label (e.g. `yolo11_bytetrack` / `procedural_person_tracking`).
    tracker_adapter: &'static str,
    /// `status.quality` payload (real run only; `Null` for preview).
    quality: Value,
    /// `status.tracker` / `recipe.normalizedSettings.tracker` payload (real run only; `Null` for preview).
    tracker_meta: Value,
}

/// The per-backend seam of [`assemble_real_person_track`] (sc-8833): everything the shared
/// orchestrator can't infer from the tracking result. `device_default` is the fallback device label
/// used only if no frame reports one (zero-frame case); every frame's real device overrides it.
/// `backend_label` / `segmenter_label` populate `tracker_meta`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct PersonTrackBackend {
    /// Fallback `tracker_meta.device` (macOS = `"mlx"`, off-Mac candle = `"cuda"`), overridden by
    /// the detector's real per-frame device.
    device_default: &'static str,
    /// `tracker_meta.backend` (`"mlx"` / `"candle"`).
    backend_label: &'static str,
    /// `tracker_meta.segmenter` when segmentation is enabled (macOS = active SAM2/SAM3 id, off-Mac =
    /// `"sam3"`); `"disabled"` is substituted by the orchestrator when `segment_enabled` is false.
    segmenter_label: fn() -> &'static str,
}

/// Track the selected person through real source content: sample frames at the 2-FPS cadence, run
/// the YOLO11 detector (native-MLX on Mac sc-3633 / `ort`-CUDA off-Mac sc-5498) on each, associate
/// the boxes into track identities with the pure-Rust SORT/ByteTrack tracker, resample the chosen
/// identity onto the sample cadence (sc-3634), then fill per-frame masks with the SAM segmenter.
///
/// One cfg-free orchestrator shared by the macOS (native-MLX) and off-Mac candle lanes (sc-8833).
/// A candle-disabled off-Mac build refuses the job. The only per-backend seams are the `backend`
/// descriptor (device/backend/segmenter labels) and the `run_segmenter` closure —
/// every rendered frame's real device overrides `device_default`, so the default only shows through
/// in the (unreachable) zero-frame case. The work dir is cleaned up on both the not-found error path
/// and the success path.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn assemble_real_person_track_shared<F, Fut>(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_path: &std::path::Path,
    source_media_path: &std::path::Path,
    detection: &Value,
    track_id: &str,
    selected_timestamp: f64,
    duration: f64,
    confidence: f64,
    segment_enabled: bool,
    backend: PersonTrackBackend,
    run_segmenter: F,
) -> WorkerResult<RealPersonTrack>
where
    F: FnOnce(SegmentClip) -> Fut,
    Fut: std::future::Future<Output = SegmentOutcome>,
{
    use crate::person_track as pt;

    let selected_box = pt::NormalizedBox::from_json(detection.get("box").unwrap_or(&Value::Null));
    let weights = crate::person_jobs::require_detector_weights(settings)?;
    let conf = confidence as f32;
    let timestamps = pt::sample_timestamps(duration);

    // Sanitize the job id before it becomes a temp-dir path component (F-111): a hostile id
    // (`../…`, absolute, separators) would otherwise escape `temp_dir()` and stage frames at an
    // attacker-chosen path. `safe_download_dir` slugs every non-`[A-Za-z0-9_.-]` char, matching
    // the timeline-export convention (`sceneworks_export_{safe_download_dir(job.id)}` above).
    let work_dir =
        std::env::temp_dir().join(format!("sw-person-track-{}", safe_download_dir(&job.id)));
    tokio::fs::create_dir_all(&work_dir).await?;

    // Run the fallible sampling/assembly/segmentation body, then remove `work_dir` on EVERY
    // outcome — success, the not-found error, a cancel, or a render/detect failure (sc-8912).
    // Previously the cleanup lived only on the not-found and success paths, so an intervening
    // `?` leaked the staged frame PNGs in the system temp dir. The body borrows the outer locals
    // and consumes the `run_segmenter` FnOnce, so it's an inline `async` block whose result is
    // captured before cleanup.
    let outcome = assemble_real_person_track_body(
        api,
        settings,
        job,
        source_media_path,
        project_path,
        track_id,
        &weights,
        conf,
        &timestamps,
        selected_box,
        selected_timestamp,
        duration,
        segment_enabled,
        &backend,
        run_segmenter,
        &work_dir,
    )
    .await;
    let _ = tokio::fs::remove_dir_all(&work_dir).await;
    outcome
}

/// The fallible body of [`assemble_real_person_track_shared`], factored out so the caller can
/// remove the staged-frame `work_dir` on every exit path (success or error, sc-8912). Returns the
/// assembled track; the not-found case is still a typed `InvalidPayload` error (cleanup now happens
/// in the caller, so this no longer removes the dir itself).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn assemble_real_person_track_body<F, Fut>(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    source_media_path: &std::path::Path,
    project_path: &std::path::Path,
    track_id: &str,
    weights: &std::path::Path,
    conf: f32,
    timestamps: &[f64],
    selected_box: crate::person_track::NormalizedBox,
    selected_timestamp: f64,
    duration: f64,
    segment_enabled: bool,
    backend: &PersonTrackBackend,
    run_segmenter: F,
    work_dir: &std::path::Path,
) -> WorkerResult<RealPersonTrack>
where
    F: FnOnce(SegmentClip) -> Fut,
    Fut: std::future::Future<Output = SegmentOutcome>,
{
    use crate::person_track as pt;

    // Extract every sample frame in ONE ffmpeg pass (sc-8915 / F-113) rather than spawning a
    // process per sample. `frame_paths[i]` ↔ `timestamps[i]` ↔ the segmentation pass's sample index,
    // exactly as the prior per-frame loop produced. The frames are kept (not deleted in-loop): the
    // segmentation pass re-reads the detected target frames by the same index.
    let (track_frame_width, track_frame_height) = TRACK_FRAME_SIZE;
    let frame_paths = render_track_frames(
        "ffmpeg",
        source_media_path,
        work_dir,
        timestamps,
        duration,
        track_frame_width,
        track_frame_height,
        FfmpegContext {
            api,
            settings,
            job_id: &job.id,
            cancel_message: "Person tracking canceled by user.",
        },
    )
    .await?;

    // The detector reports its real execution device per frame; `device_default` is the fallback for
    // the zero-frame case. Detection stays per-frame (each is one `spawn_blocking` YOLO11 forward).
    let mut device = backend.device_default;
    let mut per_frame: Vec<(f64, Vec<(pt::NormalizedBox, f64)>)> =
        Vec::with_capacity(timestamps.len());
    for (&timestamp, frame_path) in timestamps.iter().zip(&frame_paths) {
        check_cancel(api, &job.id, "Person tracking canceled during sampling.").await?;
        let weights_for_frame = weights.to_path_buf();
        let frame_for_task = frame_path.clone();
        let result = tokio::task::spawn_blocking(move || {
            crate::person_jobs::detect_people_blocking(weights_for_frame, frame_for_task, conf)
        })
        .await
        .map_err(|error| task_join_error("person track detect task", error))??;
        device = result.device;
        let boxes = result
            .detections
            .iter()
            .map(|d| {
                (
                    pt::xyxy_to_normalized(
                        d.x1 as f64,
                        d.y1 as f64,
                        d.x2 as f64,
                        d.y2 as f64,
                        result.width,
                        result.height,
                    ),
                    d.score as f64,
                )
            })
            .collect::<Vec<_>>();
        per_frame.push((timestamp, boxes));
    }

    let observations = pt::observe(per_frame);
    let assembly = pt::assemble_track(&observations, selected_box, selected_timestamp, timestamps);
    if assembly.target_track_id.is_none() || assembly.detected_frames == 0 {
        // `work_dir` is removed by the caller on this (and every) exit path (sc-8912).
        return Err(WorkerError::InvalidPayload(
            "Selected person was not found in the source video. Re-run detection or adjust the selection."
                .to_owned(),
        ));
    }

    // SAM2/SAM3 segmentation pass (sc-3709): write a per-frame mask for each detected target
    // frame and fold the result into `maskState`. Any segmenter unavailability degrades
    // gracefully to box-derived masks (handled by the replacement loader), never failing
    // a track that already located the person.
    let mut frames_json = pt::frames_to_json(&assembly.frames);
    let mask_state = segment_assembly_frames(
        api,
        project_path,
        job,
        track_id,
        &assembly.frames,
        &frame_paths,
        &mut frames_json,
        segment_enabled,
        run_segmenter,
    )
    .await?;

    Ok(RealPersonTrack {
        frames: frames_json,
        average_confidence: pt::average_confidence(&assembly.frames),
        mask_state,
        quality: assembly.quality,
        tracker_meta: json!({
            "backend": backend.backend_label,
            "device": device,
            "model": "yolo11m",
            "tracker": "sort_bytetrack",
            "segmenter": if segment_enabled { (backend.segmenter_label)() } else { "disabled" },
        }),
    })
}

/// Render dimensions of the sampled track frames (`render_frame_png` above), and therefore the
/// size of the masks the SAM segmenter emits. Shared by the macOS (SAM2/SAM3) and off-Mac candle
/// (SAM3, sc-6247) segmentation passes.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const TRACK_FRAME_SIZE: (u32, u32) = (1280, 720);

/// The clip a segmenter backend is asked to mask: the `first..=last` detected-span frame paths and
/// their per-frame ByteTrack box anchors (`None` on the gap frames a video predictor fills from its
/// memory bank). Handed to the backend closure so it can drive its own model without knowing the
/// span math, sidecar layout, or PNG contract.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) struct SegmentClip {
    /// Rendered frame paths for `first..=last`, index-aligned with `anchors`.
    pub(crate) clip_paths: Vec<PathBuf>,
    /// `Some(x, y, width, height)` on detected frames, `None` on gap frames.
    pub(crate) anchors: Vec<Option<(f64, f64, f64, f64)>>,
}

/// What a segmenter backend returns for a clip. Kept separate from `WorkerResult<Vec<_>>` so the
/// shared orchestrator can distinguish the three outcomes without overloading `Ok(vec![])`:
/// a user cancel is terminal for the whole job, a weights/engine failure degrades to box masks, and
/// masks are the success path.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) enum SegmentOutcome {
    /// One binary mask per clip frame (detected + gap), index-aligned with `SegmentClip::clip_paths`.
    Masks(Vec<Vec<u8>>),
    /// Segmenter failed (engine error, task join, corrupt weights) → fall back to box-derived masks
    /// at replacement time. Nothing the user can act on, so it degrades quietly as it always has.
    Degraded,
    /// The segmenter weights are **not installed** — carries the actionable install message
    /// (sc-17629 / sc-17635).
    ///
    /// Distinct from [`SegmentOutcome::Degraded`] on purpose. Before sc-17629 a missing segmenter
    /// meant an automatic mid-job download, so "unavailable" was always a genuine failure; now it
    /// is the ordinary state of a fresh install, and folding it into `Degraded` would silently
    /// hand the user box masks with nothing anywhere telling them to install SAM3. It still
    /// degrades rather than failing — a track that already located the person is never failed by
    /// the mask pass — but the reason reaches the job.
    Unavailable(String),
    /// The user canceled the job mid-segmentation; terminal for the whole person-track job.
    Canceled(String),
}

/// Roll the per-frame segmentation outcome into the sidecar `maskState` (Python `segment_track`):
/// `generated` masks out of `detected_total` detected target frames → `degraded` (none written →
/// box-mask fallback), `active` (every detected frame segmented), or `generated` (a partial subset).
/// Kept cfg-free next to the shared orchestrator so both lanes roll up identically without reaching
/// into a per-backend, per-lane segmenter module (`person_segment` is Mac-only; the candle rollup
/// lives in `person_segment_sam3_candle`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn mask_rollup_state(generated: usize, detected_total: usize) -> &'static str {
    if generated == 0 {
        "degraded"
    } else if generated >= detected_total {
        "active"
    } else {
        "generated"
    }
}

/// Generate the selected person's track masks, then roll the outcome into a sidecar `maskState`
/// (Python `segment_track`): prompt/propagate across the `first..=last` detected span so non-detected
/// gap frames inside the span still get a mask (the "survives weak-detection frames" win). Masks are
/// written under `person-tracks/{track_id}/masks/frame_{index:06}.png`, each frame's `mask` is set,
/// and the outcome rolls up to `missing` (disabled / no detected frame), `degraded` (segmenter
/// unavailable / failed → box-mask fallback at replacement time), `generated` (some detected frames
/// masked), or `active` (all detected frames masked). The `generated`/`detected_total` rollup counts
/// only detected frames, keeping the contract identical to the per-frame path; gap-frame masks are
/// additive coverage.
///
/// This is the single cfg-free orchestrator shared by the macOS (native-MLX SAM2/SAM3) and off-Mac
/// candle (SAM3) person-track masking passes (sc-8833). Everything but the model call — span math,
/// bounds guard, clip/anchor assembly, cancel checks, the `p > 127` mask threshold, the PNG-write
/// dispatch, and the rollup — lives here exactly once. The `run_segmenter` closure is the only
/// per-backend seam: it resolves that backend's weights and drives the blocking segment step under
/// `run_blocking_with_heartbeat`, returning a [`SegmentOutcome`]. A cold-start propagate (multi-GB
/// checkpoint parse + quantize + per-frame 1008² propagation) can exceed the API's 90s stale-sweep,
/// so the keepalive inside the closure pings `Busy` every interval AND polls for a user cancel;
/// that cancel is terminal for the whole job (never a degrade), any other failure keeps the degrade
/// contract.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn segment_assembly_frames<F, Fut>(
    api: &ApiClient,
    project_path: &std::path::Path,
    job: &JobSnapshot,
    track_id: &str,
    frames: &[crate::person_track::TrackFrame],
    frame_paths: &[PathBuf],
    frames_json: &mut [Value],
    segment_enabled: bool,
    run_segmenter: F,
) -> WorkerResult<&'static str>
where
    F: FnOnce(SegmentClip) -> Fut,
    Fut: std::future::Future<Output = SegmentOutcome>,
{
    let detected_total = frames.iter().filter(|frame| frame.detected).count();
    if detected_total == 0 || !segment_enabled {
        return Ok("missing");
    }

    let masks_dir = project_path
        .join("person-tracks")
        .join(track_id)
        .join("masks");
    if tokio::fs::create_dir_all(&masks_dir).await.is_err() {
        return Ok("degraded");
    }

    // The track spans first..=last detected frame. Propagate across that contiguous clip; frames
    // outside it (before the person appears / after they leave) are left mask-less as before.
    let Some(first) = frames.iter().position(|f| f.detected) else {
        return Ok("missing");
    };
    let last = frames.iter().rposition(|f| f.detected).unwrap_or(first);

    // Clip frame paths + per-frame ByteTrack box anchors (None on the gap frames the predictor
    // fills from memory). The shared sample index ties frame `i` ↔ `frame_paths[i]` ↔ mask `i + 1`.
    // The bounds guard is load-bearing: a short `frame_paths` (fewer rendered frames than detected
    // assembly frames) would otherwise slice past the end and panic (OOB).
    if frame_paths.len() <= last {
        return Ok("degraded");
    }
    let mut clip_paths = Vec::with_capacity(last - first + 1);
    let mut anchors = Vec::with_capacity(last - first + 1);
    for (frame, path) in frames[first..=last].iter().zip(&frame_paths[first..=last]) {
        clip_paths.push(path.clone());
        anchors.push(frame.detected.then_some((
            frame.box_.x,
            frame.box_.y,
            frame.box_.width,
            frame.box_.height,
        )));
    }

    check_cancel(
        api,
        &job.id,
        "Person tracking canceled during segmentation.",
    )
    .await?;

    // The only per-backend seam: resolve weights + drive the blocking segment step (macOS = native
    // MLX SAM2/SAM3 per `SCENEWORKS_PERSON_SEGMENTER`; off-Mac = candle SAM3, sc-6247). Either path's
    // failure degrades to box masks (handled by the replacement loader) — a track that already
    // located the person is never failed by the mask pass.
    let masks = match run_segmenter(SegmentClip {
        clip_paths,
        anchors,
    })
    .await
    {
        SegmentOutcome::Masks(masks) => masks,
        // The keepalive posted the terminal `Canceled` already; propagate it (job canceled).
        SegmentOutcome::Canceled(message) => return Err(WorkerError::Canceled(message)),
        // The segmenter is not installed. Still degrade to box masks — the track already found the
        // person and must not be failed by the mask pass — but SAY SO, on the job and in the log.
        // Otherwise a fresh install silently produces worse masks forever with no hint that
        // installing "SAM3 Person Segmenter" would fix it (sc-17629).
        SegmentOutcome::Unavailable(message) => {
            tracing::warn!(
                event = "person_track_segmenter_not_installed",
                track_id,
                detail = %message
            );
            update_job(
                api,
                &job.id,
                progress_payload(
                    JobStatus::Running,
                    ProgressStage::Tracking,
                    0.9,
                    &format!("Using box masks: {message}"),
                    None,
                    None,
                    None,
                ),
            )
            .await?;
            return Ok("degraded");
        }
        // Any other failure (engine error, task join) degrades to box masks.
        SegmentOutcome::Degraded => return Ok("degraded"),
    };

    // Write every clip frame's non-empty mask (detected + gap) and set its sidecar `mask`. All the
    // PNG encoding is blocking, so it runs in one `spawn_blocking`.
    let pending: Vec<(usize, String, PathBuf, Vec<u8>)> = masks
        .into_iter()
        .enumerate()
        .filter(|(_, pixels)| pixels.iter().any(|&p| p > 127))
        .map(|(clip_idx, pixels)| {
            let assembly_idx = first + clip_idx;
            let rel = format!(
                "person-tracks/{track_id}/masks/frame_{:06}.png",
                assembly_idx + 1
            );
            let out_path = project_path.join(&rel);
            (assembly_idx, rel, out_path, pixels)
        })
        .collect();
    let (width, height) = TRACK_FRAME_SIZE;
    let written =
        match tokio::task::spawn_blocking(move || write_track_mask_pngs(width, height, pending))
            .await
        {
            Ok(written) => written,
            Err(_) => return Ok("degraded"),
        };

    let mut generated = 0usize;
    for (assembly_idx, rel) in written {
        if let Some(entry) = frames_json.get_mut(assembly_idx) {
            entry["mask"] = Value::String(rel);
        }
        if frames
            .get(assembly_idx)
            .map(|f| f.detected)
            .unwrap_or(false)
        {
            generated += 1;
        }
    }

    Ok(mask_rollup_state(generated, detected_total))
}

/// The native-MLX person-segmenter closure (sc-8833): dispatch to the active backend (SAM3 text-
/// concept PCS by default, the legacy SAM2 box-prompt path under `SCENEWORKS_PERSON_SEGMENTER=sam2`,
/// kept for A/B parity + fallback during the cutover, sc-4926) and drive the blocking segment step
/// under `run_blocking_with_heartbeat` (sc-8390 / sc-8807). Both return one binary mask per clip
/// frame; a user cancel is terminal, any other failure degrades to box masks.
#[cfg(target_os = "macos")]
async fn run_macos_segmenter(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    clip: SegmentClip,
) -> SegmentOutcome {
    let SegmentClip {
        clip_paths,
        anchors,
    } = clip;
    let cancel = gen_core::CancelFlag::new();
    let cancel_message = "Person tracking canceled during segmentation.";
    let outcome = match person_segmenter_kind() {
        PersonSegmenter::Sam3 => {
            let (model, tokenizer) =
                match crate::person_segment_sam3::require_segmenter_weights(settings) {
                    Ok(pair) => pair,
                    Err(WorkerError::Canceled(message)) => {
                        return SegmentOutcome::Canceled(message)
                    }
                    // `require_*` reports a missing install as `InvalidPayload` carrying the
                    // actionable message; anything else is a genuine failure (sc-17629).
                    Err(WorkerError::InvalidPayload(message)) => {
                        return SegmentOutcome::Unavailable(message)
                    }
                    Err(_) => return SegmentOutcome::Degraded,
                };
            let flag = cancel.clone();
            run_blocking_with_heartbeat(
                api,
                settings,
                &job.id,
                Some(cancel),
                cancel_message,
                "person segment task",
                crate::no_cancel_ack(),
                tokio::task::spawn_blocking(move || {
                    crate::person_segment_sam3::segment_track_blocking(
                        model,
                        tokenizer,
                        clip_paths,
                        anchors,
                        Some(flag),
                        Some(Box::new(|frame, total| {
                            tracing::debug!(event = "sam3_propagate_progress", frame, total);
                        })),
                    )
                }),
            )
            .await
        }
        PersonSegmenter::Sam2 => {
            let weights = match crate::person_segment::require_segmenter_weights(settings) {
                Ok(path) => path,
                Err(WorkerError::Canceled(message)) => return SegmentOutcome::Canceled(message),
                Err(WorkerError::InvalidPayload(message)) => {
                    return SegmentOutcome::Unavailable(message)
                }
                Err(_) => return SegmentOutcome::Degraded,
            };
            let flag = cancel.clone();
            run_blocking_with_heartbeat(
                api,
                settings,
                &job.id,
                Some(cancel),
                cancel_message,
                "person segment task",
                crate::no_cancel_ack(),
                tokio::task::spawn_blocking(move || {
                    crate::person_segment::propagate_track_blocking(
                        weights,
                        clip_paths,
                        anchors,
                        Some(flag),
                        Some(Box::new(|frame, total| {
                            tracing::debug!(event = "sam2_propagate_progress", frame, total);
                        })),
                    )
                }),
            )
            .await
        }
    };
    match outcome {
        Ok(masks) => SegmentOutcome::Masks(masks),
        Err(WorkerError::Canceled(message)) => SegmentOutcome::Canceled(message),
        Err(_) => SegmentOutcome::Degraded,
    }
}

/// The off-Mac candle SAM3 person-segmenter closure (sc-6247 / sc-8833): SAM3 is the only off-Mac
/// segmenter (no SAM2 box-prompt fallback — that's the native-MLX `mlx-gen-sam2`). Resolves the
/// candle SAM3 weights and drives the blocking segment step under `run_blocking_with_heartbeat`
/// (sc-8390 / sc-8807), mirroring the macOS closure: the keepalive's cancel poll trips `cancel`,
/// which the engine's per-frame propagate contract (sc-8972, the candle sibling of gen-core
/// d8038beb) observes between frames — not just at the coarse seams (cold parse / model build).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn run_candle_segmenter(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    clip: SegmentClip,
) -> SegmentOutcome {
    let SegmentClip {
        clip_paths,
        anchors,
    } = clip;
    let (model, tokenizer) =
        match crate::person_segment_sam3_candle::require_segmenter_weights(settings) {
            Ok(pair) => pair,
            Err(WorkerError::Canceled(message)) => return SegmentOutcome::Canceled(message),
            Err(WorkerError::InvalidPayload(message)) => {
                return SegmentOutcome::Unavailable(message)
            }
            Err(_) => return SegmentOutcome::Degraded,
        };
    let cancel = gen_core::CancelFlag::new();
    let flag = cancel.clone();
    let outcome = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        "Person tracking canceled during segmentation.",
        "person segment task",
        crate::no_cancel_ack(),
        tokio::task::spawn_blocking(move || {
            crate::person_segment_sam3_candle::segment_track_blocking(
                model,
                tokenizer,
                clip_paths,
                anchors,
                Some(flag),
                Some(Box::new(|frame, total| {
                    tracing::debug!(event = "sam3_propagate_progress", frame, total);
                })),
            )
        }),
    )
    .await;
    match outcome {
        Ok(masks) => SegmentOutcome::Masks(masks),
        Err(WorkerError::Canceled(message)) => SegmentOutcome::Canceled(message),
        Err(_) => SegmentOutcome::Degraded,
    }
}

/// Encode each `(assembly_idx, rel, out_path, pixels)` mask as an `L` (8-bit grayscale) PNG,
/// returning the `(assembly_idx, rel)` of the frames that were written. A single frame's failure is
/// non-fatal (matches Python's per-frame `except: continue`); it keeps `mask: null` and falls back
/// to a box mask at replacement time. Shared by the macOS and off-Mac candle segmentation passes.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn write_track_mask_pngs(
    width: u32,
    height: u32,
    pending: Vec<(usize, String, PathBuf, Vec<u8>)>,
) -> Vec<(usize, String)> {
    let mut written = Vec::with_capacity(pending.len());
    for (assembly_idx, rel, out_path, pixels) in pending {
        let Some(gray) = image::GrayImage::from_raw(width, height, pixels) else {
            continue;
        };
        if gray.save(&out_path).is_ok() {
            written.push((assembly_idx, rel));
        }
    }
    written
}

/// macOS entry point for [`assemble_real_person_track_shared`] (sc-8833): supplies the native-MLX
/// backend descriptor (device `mlx`, active SAM2/SAM3 segmenter id) and the MLX segmenter closure.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn assemble_real_person_track(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_path: &std::path::Path,
    source_media_path: &std::path::Path,
    detection: &Value,
    track_id: &str,
    selected_timestamp: f64,
    duration: f64,
    confidence: f64,
    segment_enabled: bool,
) -> WorkerResult<RealPersonTrack> {
    assemble_real_person_track_shared(
        api,
        settings,
        job,
        project_path,
        source_media_path,
        detection,
        track_id,
        selected_timestamp,
        duration,
        confidence,
        segment_enabled,
        PersonTrackBackend {
            device_default: "mlx",
            backend_label: "mlx",
            segmenter_label: || person_segmenter_kind().meta_label(),
        },
        |clip| run_macos_segmenter(api, settings, job, clip),
    )
    .await
}

/// Off-Mac candle GPU-worker entry point for [`assemble_real_person_track_shared`] (sc-5498 /
/// sc-8833): supplies the candle backend descriptor (device `cuda`, SAM3-only segmenter, sc-6247)
/// and the candle SAM3 segmenter closure. A candle-disabled build has no person-track lane.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[allow(clippy::too_many_arguments)]
async fn assemble_real_person_track(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_path: &std::path::Path,
    source_media_path: &std::path::Path,
    detection: &Value,
    track_id: &str,
    selected_timestamp: f64,
    duration: f64,
    confidence: f64,
    segment_enabled: bool,
) -> WorkerResult<RealPersonTrack> {
    assemble_real_person_track_shared(
        api,
        settings,
        job,
        project_path,
        source_media_path,
        detection,
        track_id,
        selected_timestamp,
        duration,
        confidence,
        segment_enabled,
        PersonTrackBackend {
            device_default: "cuda",
            backend_label: "candle",
            segmenter_label: || "sam3",
        },
        |clip| run_candle_segmenter(api, settings, job, clip),
    )
    .await
}

#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
#[allow(clippy::too_many_arguments)]
async fn assemble_real_person_track(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
    _project_path: &std::path::Path,
    _source_media_path: &std::path::Path,
    _detection: &Value,
    _track_id: &str,
    _selected_timestamp: f64,
    _duration: f64,
    _confidence: f64,
    _segment_enabled: bool,
) -> WorkerResult<RealPersonTrack> {
    Err(WorkerError::InvalidPayload(
        "Real person tracking requires the candle GPU worker on this platform.".to_owned(),
    ))
}

pub(crate) async fn run_person_track_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing selected-person tracking.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(api, &job.id, "Person tracking canceled before saving.").await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Tracking,
            0.35,
            "Tracking selected person through sampled frames.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let result = run_person_track(api, settings, job).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Reusable person track saved.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

pub(crate) async fn run_person_track(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<JsonObject> {
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let source_asset_id = required_payload_string(&job.payload, "sourceAssetId")?;
    let detection = job
        .payload
        .get("detection")
        .cloned()
        .filter(Value::is_object)
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Selected detection metadata is required".to_owned())
        })?;
    if detection
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .is_none()
    {
        return Err(WorkerError::InvalidPayload(
            "Selected detection metadata is required".to_owned(),
        ));
    }
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    let project_path = PathBuf::from(project.path);
    let source_asset = store.get_asset(project_id, source_asset_id)?;
    let source_file = source_asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Source asset file is missing.".to_owned()))?;
    let duration = source_file
        .get("duration")
        .map_or(6.0, |value| value_f64(value, 6.0))
        .clamp(1.0, 3600.0);
    let confidence = job
        .payload
        .get("advanced")
        .and_then(|advanced| advanced.get("confidence"))
        .or_else(|| job.payload.get("confidence"))
        .map_or(0.25, |value| value_f64(value, 0.25))
        .clamp(0.01, 1.0);
    let is_preview = job
        .payload
        .get("preview")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // The selection frame's source timestamp (recorded in the representative frame's lineage).
    let selected_timestamp = job
        .payload
        .get("representativeFrameAssetId")
        .and_then(Value::as_str)
        .and_then(|asset_id| store.get_asset(project_id, asset_id).ok())
        .and_then(|asset| {
            asset
                .get("lineage")
                .and_then(|lineage| lineage.get("sourceTimestamp"))
                .map(|value| value_f64(value, 0.0))
        })
        .unwrap_or(0.0);

    // Segmentation is on by default (Python `advanced.segment`); a per-frame SAM2 mask is
    // written for each detected target frame. `segment: false` skips it (maskState=missing).
    let segment_enabled = job
        .payload
        .get("advanced")
        .and_then(|advanced| advanced.get("segment"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    // The track id is generated up front so the SAM2 segmentation pass can write masks under
    // `person-tracks/{track_id}/masks/` before the sidecar is assembled.
    let track_id = format!("track_{}", Uuid::new_v4().simple());

    // Preview jobs (CPU worker) keep the procedural placeholder. Real jobs run the native-MLX
    // YOLO11 detector (sc-3633) per sampled frame + the SORT/ByteTrack tracker (sc-3634), then
    // segment each detected frame with the native-MLX SAM2 segmenter (sc-3709) → maskState
    // active / generated / degraded / missing.
    let outcome: PersonTrackOutcome = if is_preview {
        let frames = track_frames_from_detection(&detection, duration);
        let avg = frames
            .iter()
            .map(|frame| {
                frame
                    .get("confidence")
                    .map_or(0.0, |value| value_f64(value, 0.0))
            })
            .sum::<f64>()
            / (frames.len().max(1) as f64);
        PersonTrackOutcome {
            frames,
            average_confidence: avg,
            mask_state: "deferred",
            person_active: false,
            tracker_model: "procedural-person-tracker".to_owned(),
            tracker_adapter: "procedural_person_tracking",
            quality: Value::Null,
            tracker_meta: Value::Null,
        }
    } else {
        let source_media_rel = required_value_str(source_file, "path")?;
        let source_media_path = safe_project_path(&project_path, source_media_rel)?;
        if !source_media_path.exists() {
            return Err(WorkerError::InvalidPayload(format!(
                "Source media not found: {}",
                source_media_path.display()
            )));
        }
        let real = assemble_real_person_track(
            api,
            settings,
            job,
            &project_path,
            &source_media_path,
            &detection,
            &track_id,
            selected_timestamp,
            duration,
            confidence,
            segment_enabled,
        )
        .await?;
        PersonTrackOutcome {
            frames: real.frames,
            average_confidence: real.average_confidence,
            mask_state: real.mask_state,
            person_active: true,
            tracker_model: "yolo11m".to_owned(),
            tracker_adapter: "yolo11_bytetrack",
            quality: real.quality,
            tracker_meta: real.tracker_meta,
        }
    };
    let PersonTrackOutcome {
        frames,
        average_confidence,
        mask_state,
        person_active,
        tracker_model,
        tracker_adapter,
        quality: quality_value,
        tracker_meta,
    } = outcome;

    let track_name =
        optional_payload_string(&job.payload, "trackName").unwrap_or("Selected person");
    let representative_frame_asset_id = job
        .payload
        .get("representativeFrameAssetId")
        .cloned()
        .unwrap_or(Value::Null);
    let raw_selected_detection = detection.clone();
    let created_at = now_rfc3339();
    let source_display_name = source_asset
        .get("displayName")
        .cloned()
        .unwrap_or(Value::Null);

    let mut status = serde_json::Map::new();
    status.insert(
        "sampleRateFps".to_owned(),
        json!(PERSON_TRACK_SAMPLE_RATE_FPS),
    );
    status.insert("maskState".to_owned(), json!(mask_state));
    status.insert(
        "averageConfidence".to_owned(),
        json!(round_to(average_confidence, 4)),
    );
    status.insert(
        "correctionState".to_owned(),
        json!("ready_for_box_corrections"),
    );
    status.insert("personTrackingActive".to_owned(), json!(person_active));
    if person_active {
        status.insert("quality".to_owned(), quality_value);
        status.insert("tracker".to_owned(), tracker_meta.clone());
    }

    let mut normalized = serde_json::Map::new();
    normalized.insert(
        "sampleRateFps".to_owned(),
        json!(PERSON_TRACK_SAMPLE_RATE_FPS),
    );
    normalized.insert("personDetectionActive".to_owned(), json!(person_active));
    normalized.insert("personTrackingActive".to_owned(), json!(person_active));
    if person_active {
        normalized.insert("maskState".to_owned(), json!(mask_state));
        normalized.insert("tracker".to_owned(), tracker_meta);
    }

    let track = json!({
        "schemaVersion": 1,
        "id": track_id.clone(),
        "projectId": project_id,
        "name": track_name,
        "createdAt": created_at,
        "sourceAssetId": source_asset_id,
        "sourceDisplayName": source_display_name,
        "representativeFrameAssetId": representative_frame_asset_id,
        "selectedDetection": detection,
        "frames": frames,
        "corrections": [],
        "status": Value::Object(status),
        "recipe": {
            "mode": "person_track",
            "model": tracker_model,
            "adapter": tracker_adapter,
            "prompt": format!("Track {track_name}"),
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": Value::Object(normalized),
            "rawAdapterSettings": { "selectedDetection": raw_selected_detection }
        },
        "lineage": {
            "jobId": job.id,
            "parents": [source_asset_id, job.payload.get("representativeFrameAssetId").cloned().unwrap_or(Value::Null)]
        }
    });

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.82,
            "Saving reusable person track metadata.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Person tracking canceled before sidecar write.",
    )
    .await?;
    let track_path = project_path
        .join("person-tracks")
        .join(format!("{track_id}.sceneworks.person-track.json"));
    write_json_value(&track_path, &track).await?;
    let relative = relative_path(&project_path, &track_path)?;
    let mut result = JsonObject::new();
    result.insert("trackId".to_owned(), Value::String(track_id));
    result.insert("track".to_owned(), track);
    result.insert("path".to_owned(), Value::String(relative));
    Ok(result)
}

pub(crate) async fn run_timeline_export_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.06,
            "Preparing timeline export.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(api, &job.id, "Timeline export canceled before rendering.").await?;
    let request = export_request_from_job(job)?;
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let timeline_path = safe_project_path(&project_path, &request.timeline_path)?;
    let timeline = read_json_value(&timeline_path).await?;
    let (width, height) = output_dimensions(
        timeline
            .get("aspectRatio")
            .and_then(Value::as_str)
            .unwrap_or("16:9"),
        request.resolution,
    );
    let mut items = main_track_items(&timeline);
    items.sort_by(|left, right| {
        item_f64(left, "timelineStart", 0.0).total_cmp(&item_f64(right, "timelineStart", 0.0))
    });
    if items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Timeline has no main video items to export.".to_owned(),
        ));
    }
    // The plan's own duration is the timeline's span; the EXPORT's duration is read off the
    // rendered segments in `finalize`, because a crossfade makes the two different numbers.
    let (plan, _) = plan_segments(&items)?;

    let temp_dir = tempfile::Builder::new()
        .prefix(&format!(
            "sceneworks_export_{}_",
            safe_download_dir(&job.id)
        ))
        .tempdir()?;
    let tmp_path = temp_dir.path().to_path_buf();

    let spec = RenderSpec {
        width,
        height,
        fps: request.fps,
    };
    let export = TimelineExport {
        context: FfmpegContext {
            api,
            settings,
            job_id: &job.id,
            cancel_message: "Timeline export canceled by user.",
        },
        store: &store,
        request,
        project_path,
        timeline,
        spec,
    };
    let segments = export.render(&plan, &tmp_path).await?;
    export.finalize(&segments, &tmp_path).await?;
    Ok(())
}

/// A timeline item resolved to its render plan: the optional black gap to emit
/// before it (when items leave a hole in the timeline) plus the transition it
/// carries in. Pure data, so `plan_segments` can build it without touching
/// ffmpeg or the store and the planning logic stays unit-testable.
#[derive(Debug)]
pub(crate) struct PlannedItem<'a> {
    pub(crate) leading_gap: Option<f64>,
    pub(crate) item: &'a Value,
    pub(crate) transition: Option<String>,
    pub(crate) transition_duration: f64,
}

/// Walk the sorted main-track items into an ordered render plan, inserting a
/// black gap wherever an item does not abut the previous one, and return the
/// plan together with the total timeline duration. No I/O: the gap, transition,
/// and span-validation logic can be exercised in isolation.
pub(crate) fn plan_segments(items: &[Value]) -> WorkerResult<(Vec<PlannedItem<'_>>, f64)> {
    let mut plan = Vec::with_capacity(items.len());
    let mut cursor = 0.0_f64;
    for item in items {
        let start = item_f64(item, "timelineStart", 0.0);
        let item_end = item_f64(item, "timelineEnd", start);
        if item_end <= start {
            return Err(WorkerError::InvalidPayload(
                "timelineEnd must be greater than timelineStart.".to_owned(),
            ));
        }
        let leading_gap = if start > cursor {
            let gap = start - cursor;
            cursor = start;
            Some(gap)
        } else {
            None
        };
        let transition_in = item.get("transitionIn").unwrap_or(&Value::Null);
        plan.push(PlannedItem {
            leading_gap,
            item,
            transition: transition_in
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_owned),
            transition_duration: value_f64(
                transition_in.get("duration").unwrap_or(&Value::Null),
                DEFAULT_TRANSITION_DURATION_SECONDS,
            ),
        });
        cursor = cursor.max(item_end);
    }
    Ok((plan, cursor))
}

/// Per-job state for a timeline MP4 export, resolved once so the rendering and
/// finalization steps can share it without threading a long argument list.
struct TimelineExport<'a> {
    context: FfmpegContext<'a>,
    store: &'a ProjectStore,
    request: TimelineExportRequest,
    project_path: PathBuf,
    timeline: Value,
    spec: RenderSpec,
}

impl TimelineExport<'_> {
    /// Render each planned segment (gaps then source items) into `tmp_path`,
    /// reporting progress and honoring cancellation between items.
    async fn render(
        &self,
        plan: &[PlannedItem<'_>],
        tmp_path: &Path,
    ) -> WorkerResult<Vec<TimelineSegment>> {
        let mut segments = Vec::new();
        let total = plan.len().max(1);
        for (index, planned) in plan.iter().enumerate() {
            check_cancel(
                self.context.api,
                self.context.job_id,
                "Timeline export canceled by user.",
            )
            .await?;
            if let Some(gap_duration) = planned.leading_gap {
                let gap_path = tmp_path.join(format!("segment_{:04}_gap.mp4", segments.len()));
                render_black_segment(
                    "ffmpeg",
                    &gap_path,
                    gap_duration,
                    self.spec,
                    Some(self.context),
                )
                .await?;
                segments.push(TimelineSegment {
                    path: gap_path,
                    duration: gap_duration,
                    transition: None,
                    transition_duration: 0.0,
                });
            }

            let asset_id = required_value_str(planned.item, "assetId")?;
            let asset = self.store.get_asset(&self.request.project_id, asset_id)?;
            let display_name = planned
                .item
                .get("displayName")
                .and_then(Value::as_str)
                .unwrap_or("item");
            let segment_path = tmp_path.join(format!(
                "segment_{:04}_{}.mp4",
                segments.len(),
                slugify(display_name, "timeline-export", Some(48))
            ));
            let duration = render_item_segment(
                "ffmpeg",
                &self.project_path,
                planned.item,
                &asset,
                &segment_path,
                self.spec,
                Some(self.context),
            )
            .await?;
            segments.push(TimelineSegment {
                path: segment_path,
                duration,
                transition: planned.transition.clone(),
                transition_duration: planned.transition_duration,
            });
            update_job(
                self.context.api,
                self.context.job_id,
                progress_payload(
                    JobStatus::Running,
                    ProgressStage::Rendering,
                    0.12 + (((index + 1) as f64 / total as f64) * 0.58),
                    "Rendering timeline segments.",
                    None,
                    None,
                    None,
                ),
            )
            .await?;
        }
        Ok(segments)
    }

    /// Resolve every audio source the timeline places into a file on disk (sc-22712).
    ///
    /// A placement whose asset is missing from the project, whose media file is gone, or which
    /// carries no decodable audio stream is DROPPED rather than fatal: the picture is the
    /// deliverable, and losing the whole export because one clip turned out to be silent would be
    /// a worse answer than an export with one fewer layer. Everything dropped is logged with the
    /// track it came from AND returned as a [`DroppedAudioLayer`] (sc-22715), so a missing layer is
    /// diagnosable from the job RESULT and the render sidecar — not only from a log line nobody
    /// reads after the fact.
    async fn resolve_audio_sources(
        &self,
        timing: &PictureTiming,
    ) -> (Vec<ResolvedAudioSource>, Vec<DroppedAudioLayer>) {
        let mut resolved = Vec::new();
        let mut dropped = Vec::new();
        for placement in audio_placements(&self.timeline, timing) {
            let Ok(asset) = self
                .store
                .get_asset(&self.request.project_id, &placement.asset_id)
            else {
                tracing::warn!(
                    asset_id = %placement.asset_id,
                    track_id = %placement.track_id,
                    "timeline export: audio source asset is missing; dropping it from the mix"
                );
                dropped.push(DroppedAudioLayer::new(&placement, "asset_missing"));
                continue;
            };
            let media_rel = asset
                .get("file")
                .and_then(|file| file.get("path"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let Ok(media_path) = safe_project_path(&self.project_path, media_rel) else {
                tracing::warn!(
                    asset_id = %placement.asset_id,
                    "timeline export: audio source path is unsafe; dropping it from the mix"
                );
                dropped.push(DroppedAudioLayer::new(&placement, "path_unsafe"));
                continue;
            };
            if !media_path.exists() {
                tracing::warn!(
                    asset_id = %placement.asset_id,
                    path = %media_path.display(),
                    "timeline export: audio source file is missing; dropping it from the mix"
                );
                dropped.push(DroppedAudioLayer::new(&placement, "file_missing"));
                continue;
            }
            if !source_has_audio_stream("ffmpeg", &media_path).await {
                tracing::info!(
                    asset_id = %placement.asset_id,
                    track_id = %placement.track_id,
                    generated = placement.generated,
                    "timeline export: source carries no audio stream; dropping it from the mix"
                );
                dropped.push(DroppedAudioLayer::new(&placement, "no_audio_stream"));
                continue;
            }
            resolved.push(ResolvedAudioSource {
                placement,
                media_path,
            });
        }
        (resolved, dropped)
    }

    /// Mux the rendered segments into the project's render directory, write the
    /// asset sidecar and recipe, index the asset, and report completion.
    ///
    /// # No workflow travels in an export (sc-15956)
    ///
    /// **A timeline export is not a generation.** Its sidecar recipe is `mode: timeline_export,
    /// model: ffmpeg` — there is no model, no seed and no prompt to replay, and what sits in the
    /// recipe's `prompt` slot is the TIMELINE'S NAME, a piece of the user's own project structure
    /// with no business leaving in a file. An envelope built from that would reproduce nothing
    /// while disclosing something, which is worse than an absence in both directions.
    ///
    /// So this function never touches a `WorkflowShare`, and therefore is deliberately NOT in
    /// `WORKFLOW_WRITE_SEAMS`: that registry lists functions an envelope can REACH, and an entry
    /// whose function touches none fails the lint as a stale claim. `SeamDisposition::Declines` is
    /// for a seam that writes a file with an envelope available to it —
    /// `segment_jobs::run_image_segment_job` — which is not this.
    ///
    /// **What does need enforcing here is inheritance**, and it is enforced where it happens. This
    /// export's inputs are generated clips that now DO carry envelopes, and ffmpeg's default on a
    /// multi-input command is to copy input 0's metadata to the output — so a crossfaded export
    /// would have published its first clip's recipe as its own. Both mux paths pass
    /// `-map_metadata -1` explicitly; `mux_with_crossfades_args` records the measurement, and
    /// `a_crossfaded_export_does_not_inherit_a_clips_recipe` in this file
    /// (`export_metadata_tests`) fails if either regresses.
    ///
    /// # The picture's own clock, not the plan's (sc-22712)
    ///
    /// The length is read off the SEGMENTS through [`PictureTiming`] rather than taken from
    /// `plan_segments`, which sums item spans and knows nothing about crossfades. A crossfaded
    /// picture is shorter than its timeline by the crossfade duration per transition, so the plan's
    /// number would give the mix a ceiling that is too generous, delay every clip past the
    /// transition, and put a duration in the sidecar that the file does not have.
    async fn finalize(&self, segments: &[TimelineSegment], tmp_path: &Path) -> WorkerResult<()> {
        let timing = PictureTiming::from_segments(segments);
        let duration = timing.duration();
        let output_rel = format!(
            "assets/renders/{}_{}_{}.mp4",
            &now_rfc3339()[..10],
            slugify(&self.request.timeline_name, "timeline-export", Some(48)),
            asset_suffix(self.context.job_id)
        );
        let output_path = self.project_path.join(&output_rel);
        tokio::fs::create_dir_all(output_path.parent().ok_or_else(|| {
            WorkerError::InvalidPayload("Render output has no parent directory.".to_owned())
        })?)
        .await?;
        update_job(
            self.context.api,
            self.context.job_id,
            progress_payload(
                JobStatus::Saving,
                ProgressStage::Muxing,
                0.78,
                "Muxing MP4 export.",
                None,
                None,
                None,
            ),
        )
        .await?;
        // sc-22712. With nothing to mix this is byte-for-byte the export it always was: one mux
        // straight to the deliverable, still `-c copy` on the concat path. Sound is a SECOND pass
        // over the finished picture rather than a wider first pass, because the picture pass joins
        // pre-rendered segments whose timing has nothing to do with where a bed or a line sits —
        // a bed spanning three cuts cannot be expressed as a property of any one segment.
        let (audio, dropped) = self.resolve_audio_sources(&timing).await;
        if audio.is_empty() {
            mux_segments(
                "ffmpeg",
                segments,
                tmp_path,
                &output_path,
                Some(self.context),
            )
            .await?;
        } else {
            let picture_path = tmp_path.join("picture.mp4");
            mux_segments(
                "ffmpeg",
                segments,
                tmp_path,
                &picture_path,
                Some(self.context),
            )
            .await?;
            mux_audio(
                "ffmpeg",
                &picture_path,
                &audio,
                &output_path,
                Some(self.context),
            )
            .await?;
        }

        let dropped_layers: Vec<Value> = dropped.iter().map(DroppedAudioLayer::to_json).collect();
        let asset = build_render_asset(
            &self.request,
            &self.timeline,
            self.context.job_id,
            &output_rel,
            self.spec.width,
            self.spec.height,
            duration,
            &audio,
            &dropped_layers,
        );
        let sidecar_path = output_path.with_extension("sceneworks.json");
        let asset_id = required_value_str(&asset, "id")?.to_owned();
        self.store
            .persist_native_asset_sidecar(&self.request.project_id, &sidecar_path, &asset)?;

        let mut result = JsonObject::new();
        result.insert("assetIds".to_owned(), json!([asset_id]));
        result.insert("assets".to_owned(), json!([asset]));
        result.insert(
            "timelineId".to_owned(),
            Value::String(self.request.timeline_id.clone()),
        );
        result.insert("renderPath".to_owned(), Value::String(output_rel));
        // Every layer the mix went without, in the RESULT (sc-22715): the film harness copies it
        // into the run record's export entry, so "why is the music missing" is answerable from
        // `run.json` alone.
        result.insert(
            "droppedAudioLayers".to_owned(),
            Value::Array(dropped_layers),
        );
        result.insert(
            "adapter".to_owned(),
            Value::String("ffmpeg_timeline".to_owned()),
        );
        update_job(
            self.context.api,
            self.context.job_id,
            progress_payload(
                JobStatus::Completed,
                ProgressStage::Completed,
                1.0,
                "Timeline MP4 export saved.",
                None,
                Some(result),
                None,
            ),
        )
        .await?;
        Ok(())
    }
}

pub(crate) fn candidate_people(
    width: u32,
    height: u32,
    source_asset_id: &str,
    timestamp: f64,
) -> Vec<Value> {
    let seed = format!("{source_asset_id}:{timestamp:.3}:{width}x{height}");
    let digest = Sha256::digest(seed.as_bytes());
    let templates = [
        (0.34, 0.16, 0.24, 0.68, 0.91),
        (0.58, 0.20, 0.20, 0.58, 0.78),
        (0.14, 0.26, 0.17, 0.50, 0.66),
    ];
    templates
        .iter()
        .enumerate()
        .map(|(index, (x, y, box_width, box_height, confidence))| {
            let jitter = ((digest[index] % 13) as f64 - 6.0) / 1000.0;
            json!({
                "id": format!("person_{}", index + 1),
                "label": format!("Person {}", index + 1),
                "confidence": round_to(*confidence - index as f64 * 0.04, 2),
                "box": {
                    "x": (*x + jitter).clamp(0.02, 0.92),
                    "y": *y,
                    "width": *box_width,
                    "height": *box_height
                },
                "maskState": "deferred",
                "frameWidth": width,
                "frameHeight": height
            })
        })
        .collect()
}

pub(crate) fn track_frames_from_detection(detection: &Value, duration: f64) -> Vec<Value> {
    let sample_count = ((duration.max(1.0) * PERSON_TRACK_SAMPLE_RATE_FPS).round() as usize)
        .clamp(3, PERSON_TRACK_MAX_SAMPLES);
    let base_confidence =
        value_f64(detection.get("confidence").unwrap_or(&Value::Null), 0.82).clamp(0.0, 1.0);
    (0..sample_count)
        .map(|index| {
            let t = index as f64 / (sample_count.saturating_sub(1).max(1) as f64);
            json!({
                "timestamp": round_to(t * duration.max(0.0), 3),
                "box": {
                    "x": round_to(detection_box_f64(detection, "x", 0.35, 0.0, 1.0) + (t - 0.5) * PERSON_TRACK_X_DRIFT, 4),
                    "y": round_to(detection_box_f64(detection, "y", 0.16, 0.0, 1.0), 4),
                    "width": round_to(detection_box_f64(detection, "width", 0.24, 0.01, 1.0), 4),
                    "height": round_to(detection_box_f64(detection, "height", 0.68, 0.01, 1.0), 4)
                },
                "confidence": 0.5_f64.max(round_to(base_confidence - index as f64 * 0.006, 3)),
                "mask": Value::Null
            })
        })
        .collect()
}

pub(crate) fn detection_box_f64(
    detection: &Value,
    field: &str,
    default: f64,
    min_value: f64,
    max_value: f64,
) -> f64 {
    detection
        .get("box")
        .and_then(|value| value.get(field))
        .map_or(default, |value| value_f64(value, default))
        .clamp(min_value, max_value)
}

pub(crate) fn round_to(value: f64, places: u32) -> f64 {
    let factor = 10_f64.powi(i32::try_from(places).unwrap_or(0));
    (value * factor).round() / factor
}

pub(crate) fn export_request_from_job(job: &JobSnapshot) -> WorkerResult<TimelineExportRequest> {
    Ok(TimelineExportRequest {
        project_id: required_payload_string(&job.payload, "projectId")?.to_owned(),
        timeline_id: required_payload_string(&job.payload, "timelineId")?.to_owned(),
        timeline_name: optional_payload_string(&job.payload, "timelineName")
            .unwrap_or("Timeline")
            .to_owned(),
        timeline_path: required_payload_string(&job.payload, "timelinePath")?.to_owned(),
        resolution: payload_u32(&job.payload, "resolution", 720).clamp(240, 2160),
        fps: payload_u32(&job.payload, "fps", 30).clamp(1, 60),
    })
}

/// The scale→pad→rgb24 filter chain every person-track / frame-extract sample shares. Downscales
/// into `width×height` preserving aspect (letterboxed on the app ink color), then forces rgb24 so
/// the detector always sees a 3-channel frame. Shared by the single-frame `render_frame_png` and the
/// single-pass `render_track_frames` so both produce identical frame geometry for a given source
/// frame (the two paths can still pick DIFFERENT source frames — see `render_track_frames`).
fn frame_scale_pad_filter(width: u32, height: u32) -> String {
    format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x12110f,format=rgb24"
    )
}

/// Count the decodable video frames in `source_path` with a single ffmpeg null-mux pass, so the
/// caller can decide whether the single-pass `select` optimization is provably equivalent to the
/// old per-frame accurate-seek loop (sc-8915 / F-113). Returns `Ok(Some(n))` on a clean probe and
/// `Ok(None)` for ordinary unavailable/unparseable probes (unknown → the caller must take the SAFE
/// per-frame path), but preserves a typed timeout: silently starting up to 24 more FFmpeg children
/// after the probe itself wedged would defeat the shared execution deadline. We deliberately probe
/// with `ffmpeg` and not `ffprobe`: the desktop app ships only the imageio-ffmpeg binary (no
/// `ffprobe`), so `ffprobe` is not guaranteed present.
///
/// `-f null -` decodes every frame and prints the running `frame=<N>` counter to stderr; the last
/// value is the exact decodable frame count. This is VFR-safe (it counts real frames rather than
/// trusting an `avg_frame_rate` metadata field that lies on VFR/screen-record sources). Ordinary
/// unavailable/unparseable probes are non-fatal: `None` routes to the accurate-seek fallback. A
/// deadline expiry remains a typed timeout and aborts the wrapper instead of multiplying a hang.
///
/// The pass must NOT add `-c copy`, which is what it used to do. MEASURED on ffmpeg 6.1.1-3ubuntu5:
/// under stream copy that build prints no `frame=` token at all (`size=N/A time=00:00:01.87
/// bitrate=N/A speed=1.28e+03x`), so this probe returned `None` on every input and silently routed
/// EVERY caller to the slow accurate-seek fallback — a real regression that hid behind the
/// documented non-fatal `None`, because the bundled 7.1 and current 9.0.1 both print `frame=`
/// either way. Decoding reports `frame=   48` on all three builds (sc-19549).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn probe_source_frame_count(ffmpeg: &str, source_path: &Path) -> WorkerResult<Option<usize>> {
    let program = resolve_probe_program(ffmpeg);
    let mut command = Command::new(&program);
    command.args([
        "-hide_banner",
        "-i",
        &source_path.display().to_string(),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]);
    probe_source_frame_count_command_with_timeout(command, ffmpeg_execution_timeout()).await
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn probe_source_frame_count_command_with_timeout(
    command: Command,
    execution_timeout: Duration,
) -> WorkerResult<Option<usize>> {
    let output = match run_ffmpeg_probe_command_with_timeout(command, execution_timeout).await {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::TimedOut => {
            return Err(WorkerError::Io(error));
        }
        Err(_) => return Ok(None),
    };
    // ffmpeg writes `frame=<N>` progress lines to stderr; the last one is the final decoded count.
    let stderr = String::from_utf8_lossy(&output.stderr);
    Ok(parse_ffmpeg_frame_count(&stderr))
}

/// Extract the last `frame=<N>` counter ffmpeg prints to stderr (the progress token may be padded
/// with spaces, e.g. `frame=  12`). Split into its own pure fn so it is unit-testable without ffmpeg.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn parse_ffmpeg_frame_count(stderr: &str) -> Option<usize> {
    stderr.rmatch_indices("frame=").find_map(|(idx, _)| {
        stderr[idx + "frame=".len()..]
            .trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .filter(|digits| !digits.is_empty())
            .and_then(|digits| digits.parse::<usize>().ok())
    })
}

/// Extract the person-track sample frames in ONE ffmpeg pass (sc-8915 / F-113) — replacing up to
/// `MAX_SAMPLES` (24) per-sample accurate-seek spawns — WHENEVER that single pass is provably
/// equivalent to the old per-sample loop, and FALLING BACK to the per-frame accurate seeks when it
/// is not.
///
/// `sample_timestamps` is a uniform cadence — evenly spaced `duration·i/(count-1)` for the interior
/// samples, with only the final inclusive-end sample pulled inside the clip by
/// `FRAME_SEEK_GUARD_SECONDS`. A single `select` filter over that same grid picks, for the
/// `selected_n`-th output frame, the first source frame whose `pts ≥ TH(selected_n)`, where
/// `TH(n) = format!("{:.3}", frame_seek_timestamp(timestamps[n], duration).max(0.0))` is the byte-
/// identical seek STRING the per-frame accurate path passes to `-ss` for sample `n` (same formatting,
/// no separate pre-round — see [`render_track_frames_single_pass`] for why a pre-round diverged at
/// `{:.3}` tie points). Feeding the filter that per-sample threshold lookup (rather than a
/// `min(selected_n·interval, last_seek)` recurrence) is what makes the single pass byte-identical,
/// sample-for-sample, to the accurate-seek path over its whole regime.
///
/// GUARANTEE (what this function actually delivers):
/// - **High-fps / frame-dense sources** (`source_frame_count ≥ count`): the single pass is taken and
///   is BYTE-IDENTICAL to the per-frame accurate-seek reference. Both mechanisms compare `pts` against
///   the identical `{:.3}` threshold, so both select the identical frame — including at grid points
///   that align to a source frame's pts, which an earlier `{:.6}`-truncated `selected_n·interval`
///   recurrence got wrong by an adjacent (±1) source frame (e.g. 5/16 samples off on 30fps@8s).
/// - **Sub-cadence-fps / frame-starved sources** (`source_frame_count < count`, or an unknown probe):
///   the forward-only single `select` would exhaust the clip early (two close grid points with no
///   distinct source frame between them collapse to one forward frame, whereas `-ss` re-selects the
///   same earlier frame independently), diverging grossly — reproduced with real ffmpeg at 19–21/24
///   frames wrong on a 12s@1fps clip. So we DO NOT run the single pass there: we fall back to
///   per-frame accurate seeks, which reproduce the exact pre-PR frames by construction.
///
/// We probe the real decodable frame count with [`probe_source_frame_count`] (an ffmpeg null-mux
/// count, VFR-safe, no `ffprobe` dependency). Single pass ONLY when the probe succeeds AND
/// `frame_count ≥ count`; otherwise (including an unknown/failed probe) fall back to per-frame
/// accurate seeks. Either way the person-detector sees the same frames the pre-optimization code did.
///
/// In the single-pass path the frames land as a 1-based `seq_%04d.png` image2 sequence, then are
/// renamed to the 0-based `frame_{index:04}.png` names the segmentation pass re-reads by sample
/// index. A single-sample cadence (`count ≤ 1`, i.e. `duration ≤ 0`) has no interval, so it always
/// takes one accurate-seek `render_frame_png` — the exact frame the old loop's sole iteration
/// produced. In the single-pass path, if the source is shorter than the grid implies and ffmpeg
/// emits fewer than `count` frames, the last real frame is cloned forward so every sample index has
/// a readable frame (the tracker records the logical `timestamps[i]` regardless).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn render_track_frames(
    ffmpeg: &str,
    source_path: &Path,
    work_dir: &Path,
    timestamps: &[f64],
    duration: f64,
    width: u32,
    height: u32,
    context: FfmpegContext<'_>,
) -> WorkerResult<Vec<PathBuf>> {
    let count = timestamps.len();
    let frame_path = |index: usize| work_dir.join(format!("frame_{index:04}.png"));

    // Degenerate single-frame cadence: no interval → one accurate-seek frame (the old loop's only
    // iteration). `frame_seek_timestamp` keeps the seek inside the clip exactly as before.
    if count <= 1 {
        let path = frame_path(0);
        render_frame_png(
            ffmpeg,
            source_path,
            &path,
            frame_seek_timestamp(timestamps.first().copied().unwrap_or(0.0), duration),
            width,
            height,
            Some(context),
        )
        .await?;
        return Ok(vec![path]);
    }

    // Gate the single-pass optimization on the equivalence condition above: only when the source has
    // at least `count` decodable frames does each grid point map to a distinct forward frame that
    // matches the independent accurate seek. An ordinary unknown probe (`None`) is treated as unsafe
    // and routes to the fallback; a typed timeout propagates through `?`. This keeps the perf win for
    // well-behaved clips without converting one wedged probe into up to 24 more FFmpeg launches.
    let source_frame_count = probe_source_frame_count(ffmpeg, source_path).await?;
    let single_pass_equivalent = source_frame_count.is_some_and(|frames| frames >= count);
    if !single_pass_equivalent {
        return render_track_frames_per_frame(
            ffmpeg,
            source_path,
            work_dir,
            timestamps,
            duration,
            width,
            height,
            context,
        )
        .await;
    }

    // The single pass is an OPTIMIZATION, not a correctness dependency: if this ffmpeg build rejects the
    // long `select` filter (or the pass fails for any other reason), degrade gracefully to the exact
    // per-frame accurate-seek path rather than hard-failing the job. That path produces the identical
    // frame selection (it is the reference the single pass is gated against), so the fallback is safe and
    // lossless — it just spends more ffmpeg spawns. We log a warn so the degradation is observable.
    match render_track_frames_single_pass(
        ffmpeg,
        source_path,
        work_dir,
        timestamps,
        duration,
        width,
        height,
        context,
    )
    .await
    {
        Ok(frames) => Ok(frames),
        Err(error) => {
            tracing::warn!(
                error = %error,
                source = %source_path.display(),
                "single-pass person-track frame sampling failed; falling back to per-frame accurate seeks"
            );
            render_track_frames_per_frame(
                ffmpeg,
                source_path,
                work_dir,
                timestamps,
                duration,
                width,
                height,
                context,
            )
            .await
        }
    }
}

/// The single ffmpeg-pass sampling path — the perf win. Split out from [`render_track_frames`] (which
/// owns the frame-count gate) so the byte-identity of this path can be exercised directly AND so a test
/// can drive it UNGATED on a low-fps clip to prove the gate is load-bearing (gross divergence there).
///
/// Builds the select predicate so the single pass lands on the SAME source frame the per-sample
/// accurate-seek path ([`render_track_frames_per_frame`]) would, for every sample — byte-identically.
/// The accurate path pulls sample `i` with `-ss {frame_seek_timestamp(timestamps[i], duration):.3}`
/// (i.e. `format!("{:.3}", seek.max(0.0))`), snapping to the first frame whose `pts ≥` that rendered
/// threshold STRING. The single `select` filter increments `selected_n` (0-based output-frame counter)
/// each time it fires, so we drive it off a per-sample threshold LOOKUP keyed on `selected_n` —
/// `gte(t, TH(selected_n))` where `TH(n)` is the byte-identical `format!("{:.3}", seek.max(0.0))` string
/// interpolated verbatim into the filter — instead of a `min(selected_n·interval, last_seek)` recurrence.
///
/// Why interpolate the rendered string and not a re-derived number: the round-1 recurrence fed the filter
/// a `{:.6}`-truncated `selected_n·interval`, a DIFFERENT number than the `-ss` threshold, so at grid
/// points that align to a source frame's pts the two mechanisms landed on ADJACENT frames (swept: 30fps@8s
/// picked +1 source frame on 5/16 samples). Round 2 pre-rounded the threshold `f64` with
/// `(x*1000.0).round()/1000.0` before formatting — but `f64::round` is round-half-AWAY-from-zero while
/// `{:.3}` is round-half-to-EVEN, so the two disagreed at `{:.3}` tie points (a `…5` third-decimal seek,
/// e.g. duration 2.25s → sample[1] seek 0.5625 → `-ss 0.562` vs pre-round `0.563`) and again picked
/// ADJACENT frames (swept: 66/660 tie durations diverged). Round 3 drops the pre-round and formats the raw
/// seek ONCE with the exact `{:.3}` the accurate path uses, then compares against that identical string, so
/// both select the identical frame. Verified byte-identical across the high-fps sweep (30/24/60/25/29.97
/// fps), at the 24-sample cap, and at the 2.25s `{:.3}` tie point (0/660 divergences).
///
/// NOTE: byte-identity here holds ONLY for frame-dense sources (`source_frame_count ≥ count`); on
/// sub-cadence-fps sources the forward-only `select` exhausts the clip early and diverges grossly, which
/// is exactly why [`render_track_frames`] gates this path behind the probe. Callers other than the gated
/// wrapper (i.e. tests) must respect that.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn render_track_frames_single_pass(
    ffmpeg: &str,
    source_path: &Path,
    work_dir: &Path,
    timestamps: &[f64],
    duration: f64,
    width: u32,
    height: u32,
    context: FfmpegContext<'_>,
) -> WorkerResult<Vec<PathBuf>> {
    let count = timestamps.len();
    let frame_path = |index: usize| work_dir.join(format!("frame_{index:04}.png"));
    // Render each per-sample seek threshold to a string with the EXACT SAME `{:.3}` formatting the
    // accurate path applies to `-ss` (`try_render_frame_png`/`render_frame_png`:
    // `format!("{:.3}", timestamp.max(0.0))`). We interpolate these rendered strings verbatim into the
    // `select` predicate so the filter compares `pts` against the byte-identical seek the accurate path
    // seeks to. Do NOT pre-round the f64 first (e.g. `(x*1000.0).round()/1000.0`): that is round-half-
    // AWAY-from-zero, whereas Rust's `{:.3}` is round-half-to-EVEN, so the two disagree at `{:.3}` tie
    // points (a `…5` third-decimal-boundary seek) and pick ADJACENT source frames (swept: 66/660 tie
    // durations diverged byte-for-byte). Formatting the raw seek once, the same way, is bit-for-bit
    // identical to the accurate seek.
    let thresholds: Vec<String> = timestamps
        .iter()
        .map(|&t| format!("{:.3}", frame_seek_timestamp(t, duration).max(0.0)))
        .collect();
    // No shell here (`run_ffmpeg` execs the binary directly), so the select expression carries no
    // wrapping quotes — only the intra-expression commas are backslash-escaped so ffmpeg does not
    // read them as `-vf` filter-chain separators. The lookup is a nested `if(eq(selected_n,n), th_n, …)`
    // chain with the final threshold as the default tail (already selected once `selected_n == count-1`).
    let mut lookup = thresholds[count - 1].clone();
    for n in (0..count - 1).rev() {
        lookup = format!(
            "if(eq(selected_n\\,{n})\\,{th}\\,{lookup})",
            th = thresholds[n]
        );
    }
    let select = format!("select=gte(t\\,{lookup})");
    let filters = format!(
        "{select},{scale_pad}",
        scale_pad = frame_scale_pad_filter(width, height)
    );
    let seq_pattern = work_dir.join("seq_%04d.png");
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            source_path.display().to_string(),
            "-vf".to_owned(),
            filters,
            "-frames:v".to_owned(),
            count.to_string(),
            "-fps_mode".to_owned(),
            "passthrough".to_owned(),
            "-f".to_owned(),
            "image2".to_owned(),
            seq_pattern.display().to_string(),
        ],
        Some(context),
    )
    .await?;

    // Rename the 1-based sequence into the 0-based sample-index names, cloning the last real frame
    // forward for any tail index ffmpeg didn't reach (short source).
    let mut frame_paths = Vec::with_capacity(count);
    let mut last_written: Option<PathBuf> = None;
    for index in 0..count {
        let seq_path = work_dir.join(format!("seq_{:04}.png", index + 1));
        let dest = frame_path(index);
        if tokio::fs::try_exists(&seq_path).await? {
            tokio::fs::rename(&seq_path, &dest).await?;
            last_written = Some(dest.clone());
        } else if let Some(previous) = &last_written {
            tokio::fs::copy(previous, &dest).await?;
        } else {
            return Err(WorkerError::InvalidPayload(format!(
                "FFmpeg produced no sampled frames for {}",
                source_path.display()
            )));
        }
        frame_paths.push(dest);
    }
    Ok(frame_paths)
}

/// The pre-PR person-track sampling path (sc-8915 / F-113 fallback): one `-ss T` accurate-seek
/// spawn per sample. This is the correctness reference the single-pass path is gated against — for
/// sub-cadence-fps / frame-starved sources it selects the exact frames the code shipped before the
/// single-pass optimization, including re-SELECTING the same earlier frame for close grid points
/// (which the forward-only single pass cannot). `frame_paths[i]` ↔ `timestamps[i]`, the same 0-based
/// `frame_{index:04}.png` names the segmentation pass re-reads.
///
/// One robustness improvement over the literal pre-PR loop: when a tail sample's seek lands past the
/// last real frame (the `FRAME_SEEK_GUARD_SECONDS` guard assumes ~5fps-dense frames, so a very
/// low-fps clip can seek past EOF and some ffmpeg builds then emit NO frame), the last successfully
/// rendered frame is cloned forward instead of hard-erroring. That mirrors the single-pass path's own
/// short-source tail handling and matches pre-PR frame *selection* for every sample pre-PR produced,
/// while never failing the job on a short clip (the tracker still records the logical `timestamps[i]`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn render_track_frames_per_frame(
    ffmpeg: &str,
    source_path: &Path,
    work_dir: &Path,
    timestamps: &[f64],
    duration: f64,
    width: u32,
    height: u32,
    context: FfmpegContext<'_>,
) -> WorkerResult<Vec<PathBuf>> {
    let mut frame_paths = Vec::with_capacity(timestamps.len());
    let mut last_written: Option<PathBuf> = None;
    for (index, &timestamp) in timestamps.iter().enumerate() {
        let path = work_dir.join(format!("frame_{index:04}.png"));
        let produced = try_render_frame_png(
            ffmpeg,
            source_path,
            &path,
            frame_seek_timestamp(timestamp, duration),
            width,
            height,
            Some(context),
        )
        .await?;
        if produced {
            last_written = Some(path.clone());
        } else if let Some(previous) = &last_written {
            // Seek landed past the last real frame on a short/low-fps clip → clone the last frame
            // forward rather than erroring, matching the single-pass tail behavior.
            tokio::fs::copy(previous, &path).await?;
        } else {
            return Err(WorkerError::InvalidPayload(format!(
                "FFmpeg produced no sampled frames for {}",
                source_path.display()
            )));
        }
        frame_paths.push(path);
    }
    Ok(frame_paths)
}

/// Accurate-seek one frame like [`render_frame_png`], but instead of erroring when ffmpeg emits no
/// output (a seek past the last real frame on a short/low-fps clip), return `Ok(false)` so the caller
/// can decide whether to clone the previous frame forward. `Ok(true)` means the frame was written.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn try_render_frame_png(
    ffmpeg: &str,
    source_path: &Path,
    output_path: &Path,
    timestamp: f64,
    width: u32,
    height: u32,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<bool> {
    let filters = frame_scale_pad_filter(width, height);
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-ss".to_owned(),
            format!("{:.3}", timestamp.max(0.0)),
            "-i".to_owned(),
            source_path.display().to_string(),
            "-frames:v".to_owned(),
            "1".to_owned(),
            "-vf".to_owned(),
            filters,
            "-f".to_owned(),
            "image2".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await?;
    Ok(tokio::fs::try_exists(output_path).await?)
}

// `main` fixed this same defect independently, as an `-sseof -0.100` retry inside this function
// after a failed render (`f24c9227c`). It did NOT conflict with sc-19549's fix and both landed in
// the merged tree; the retry is dropped here and sc-19549's clamp is the one that survives.
//
// Not a preference — the two do not compose:
//   * Every production caller of `render_frame_png` already routes its seek through
//     `frame_seek_timestamp` (`render_frame_asset` via `resolve_frame_seek`, the degenerate
//     single-frame cadence arm, and the person-track e2e path), so the retry is unreachable.
//   * `resolve_frame_seek` covers BOTH cases the retry's comment cited, and covers them earlier:
//     a still short-circuits to t=0 via `asset_is_image_source`, and a clip clamps to
//     `duration - FRAME_SEEK_GUARD_SECONDS`. The retry would have spawned ffmpeg a second time to
//     reach a frame the first spawn already got right.
//   * It actively breaks `still_at_a_nonzero_playhead_*`'s CONTROL arm, which asserts an unclamped
//     seek MUST fail with "did not produce frame output" and says in as many words that if it
//     starts succeeding the test below stops proving anything. A silent retry makes it succeed.
//   * sc-19549 also picks the RIGHT frame rather than whatever sits 0.1 s before EOF, and its
//     `video_past_its_end_yields_the_last_frame_not_the_first` test asserts that on PIXELS.
//
// The user-visible defect main fixed stays fixed — more precisely, one ffmpeg spawn earlier, and
// on the three seek paths rather than one.
pub(crate) async fn render_frame_png(
    ffmpeg: &str,
    source_path: &Path,
    output_path: &Path,
    timestamp: f64,
    width: u32,
    height: u32,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let filters = frame_scale_pad_filter(width, height);
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-ss".to_owned(),
            format!("{:.3}", timestamp.max(0.0)),
            "-i".to_owned(),
            source_path.display().to_string(),
            "-frames:v".to_owned(),
            "1".to_owned(),
            "-vf".to_owned(),
            filters,
            "-f".to_owned(),
            "image2".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await?;
    if !tokio::fs::try_exists(output_path).await? {
        return Err(WorkerError::InvalidPayload(format!(
            "FFmpeg did not produce frame output: {}",
            output_path.display()
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct TimelineSegment {
    path: PathBuf,
    duration: f64,
    transition: Option<String>,
    transition_duration: f64,
}

/// Index of the track whose items become the PICTURE — `track_main` by id, or the first
/// `kind: "video"` track. One definition, because two places need the same answer: the render plan
/// (what is drawn) and [`audio_placements`] (whose generated audio may be mixed). An item on any
/// other track is not rendered into the picture, so its own audio has nothing to sit behind.
pub(crate) fn picture_track_index(timeline: &Value) -> Option<usize> {
    timeline
        .get("tracks")
        .and_then(Value::as_array)
        .and_then(|tracks| {
            tracks.iter().position(|track| {
                track.get("id").and_then(Value::as_str) == Some("track_main")
                    || track.get("kind").and_then(Value::as_str) == Some("video")
            })
        })
}

pub(crate) fn main_track_items(timeline: &Value) -> Vec<Value> {
    picture_track_index(timeline)
        .and_then(|index| timeline["tracks"][index].get("items"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// How the saved timeline's seconds map onto the EXPORTED PICTURE's seconds (sc-22712).
///
/// They are not the same clock the moment a shot carries a crossfade. `crossfade_filter_complex`
/// overlaps each crossfaded segment with the one before it, so the picture comes out SHORTER than
/// the timeline by the crossfade duration once per transition, and every shot after a transition
/// begins that much EARLIER in the file than the timeline says.
///
/// Measured with ffmpeg 9.0.1: two 2 s clips with a 1.0 s crossfade at offset 1.0 produce a 3.000 s
/// file over a timeline that spans 4.0 s. Handing that 4.0 to [`audio_placements`] gave the mix a
/// ceiling a second too generous, `adelay`ed a line keyed to the second shot a second after the
/// shot it belongs to, and `-shortest` then cut the tail of the mix off — three symptoms of the one
/// missing conversion.
///
/// The accumulation is deliberately the same arithmetic `crossfade_filter_complex` performs
/// (`current_duration += segment.duration - duration`);
/// `picture_timing_matches_the_crossfade_graphs_own_arithmetic` fails if the two ever disagree.
#[derive(Debug, Clone)]
pub(crate) struct PictureTiming {
    duration: f64,
    /// `(timeline second a segment starts at, crossfade seconds absorbed before it)`, ascending.
    absorbed: Vec<(f64, f64)>,
}

impl PictureTiming {
    /// A picture exactly as long as the timeline says: no crossfades, nothing absorbed.
    #[cfg(test)]
    pub(crate) fn flat(duration: f64) -> Self {
        Self {
            duration: duration.max(0.0),
            absorbed: Vec::new(),
        }
    }

    /// Read the real picture clock off the segments the mux is about to join.
    pub(crate) fn from_segments(segments: &[TimelineSegment]) -> Self {
        let mut absorbed = Vec::new();
        let mut timeline_cursor = 0.0_f64;
        let mut total = 0.0_f64;
        for (index, segment) in segments.iter().enumerate() {
            if index > 0 {
                if segment.transition.as_deref() == Some("crossfade") {
                    total += crossfade_duration(segment.transition_duration);
                }
                absorbed.push((timeline_cursor, total));
            }
            timeline_cursor += segment.duration;
        }
        Self {
            duration: (timeline_cursor - total).max(0.0),
            absorbed,
        }
    }

    /// Length of the picture the mux actually produces — the mix's hard ceiling.
    pub(crate) fn duration(&self) -> f64 {
        self.duration
    }

    /// Where a timeline second lands in the exported picture.
    pub(crate) fn picture_time(&self, timeline_time: f64) -> f64 {
        let absorbed = self
            .absorbed
            .iter()
            .take_while(|(start, _)| *start <= timeline_time + 1e-9)
            .map(|(_, absorbed)| *absorbed)
            .last()
            .unwrap_or(0.0);
        (timeline_time - absorbed).max(0.0)
    }
}

/// Sample format every mixed source is converted to before `amix` sees it. Fixed rather than
/// negotiated so a mono 22 kHz dialogue take and a stereo 48 kHz music bed mix to the same thing on
/// every host — an `amix` over inputs with different layouts is a source of host-dependent output,
/// and this export is supposed to be reproducible from the saved timeline alone.
const AUDIO_MIX_FORMAT: &str = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo";

/// The headroom guard on the summed mix (sc-22712).
///
/// `amix=normalize=0` is a straight sum, so N buses at the gains the timeline asked for can — and
/// with the film harness's own defaults (dialogue 1.0 + ambience 0.35 + music 0.2, peaking at 1.55)
/// routinely do — exceed full scale, and the AAC encoder answers that by hard-clipping. Measured
/// with ffmpeg 9.0.1: a two-bus mix peaking at 1.24 round-trips through AAC with 180 samples pinned
/// at ±1.0; the same mix through this filter peaks at 0.961.
///
/// `level=disabled` turns off `alimiter`'s auto-level, which is ON by default and scales the output
/// by `1/limit` — putting the ceiling back at exactly 1.0, which is the value the encoder clips at.
/// Measured on the same two-bus mix: 0.968 with auto-level, 0.961 without. It is a gain applied to
/// the whole mix either way, so it changes no bus's level RELATIVE to another; what it costs is the
/// headroom this filter is here to buy.
const MIX_LIMITER: &str = "alimiter=level=disabled:limit=0.98";

/// One audio source placed on the timeline, resolved to the numbers ffmpeg needs (sc-22712).
///
/// A placement is produced for every item on a non-muted `kind: "audio"` track, and for a PICTURE
/// item only when it explicitly opts in with `generatedAudio: "include"`. That asymmetry is the
/// doubling guard: a generated take whose model spoke the line contributes nothing to the mix
/// unless the timeline says out loud that it should.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AudioPlacement {
    pub(crate) asset_id: String,
    pub(crate) track_id: String,
    pub(crate) role: String,
    /// Range taken from the SOURCE file, in source seconds.
    pub(crate) source_in: f64,
    pub(crate) source_out: f64,
    /// Where the clip lands in the EXPORTED PICTURE, and how long its slot is.
    ///
    /// Picture seconds, not timeline seconds: a crossfade pulls everything after it earlier in the
    /// file, and this is the number `adelay` is given, so it has to be the one the picture keeps.
    /// See [`PictureTiming`].
    pub(crate) picture_start: f64,
    pub(crate) span: f64,
    pub(crate) speed: f64,
    /// Track gain multiplied by the item's own `volume`.
    pub(crate) gain: f64,
    pub(crate) fade_in: f64,
    pub(crate) fade_out: f64,
    /// True when this is a picture item's own audio rather than a placed sound clip.
    pub(crate) generated: bool,
}

/// An [`AudioPlacement`] whose asset has been resolved to a file on disk.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedAudioSource {
    pub(crate) placement: AudioPlacement,
    pub(crate) media_path: PathBuf,
}

/// An [`AudioPlacement`] the export could NOT resolve and therefore mixed without (sc-22715).
///
/// `reason` is one of `asset_missing`, `path_unsafe`, `file_missing`, `no_audio_stream` — the four
/// branches of `TimelineExport::resolve_audio_sources`, named so the job result and the sidecar say
/// which one, not merely that a layer went missing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DroppedAudioLayer {
    pub(crate) asset_id: String,
    pub(crate) track_id: String,
    pub(crate) role: String,
    pub(crate) generated: bool,
    pub(crate) reason: &'static str,
}

impl DroppedAudioLayer {
    fn new(placement: &AudioPlacement, reason: &'static str) -> Self {
        Self {
            asset_id: placement.asset_id.clone(),
            track_id: placement.track_id.clone(),
            role: placement.role.clone(),
            generated: placement.generated,
            reason,
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        json!({
            "assetId": self.asset_id,
            "trackId": self.track_id,
            "role": self.role,
            "generated": self.generated,
            "reason": self.reason,
        })
    }
}

/// Walk the saved timeline into the ordered list of audio sources the export must mix.
///
/// Pure: no store, no ffmpeg. `timing` is the clock of the picture the video pass ACTUALLY
/// produced, and its duration is a HARD CEILING — a clip that starts past the last frame is dropped
/// and one that overruns it is shortened. That is what makes "the exported duration and the audio
/// synchronisation match the saved timeline" a single claim rather than two: the mix cannot extend
/// the file, so the export is exactly as long as the picture the timeline describes.
///
/// Every position is converted through [`PictureTiming::picture_time`] rather than used raw,
/// because a crossfaded picture is shorter than its timeline and everything after the transition
/// sits earlier in the file than the timeline says. Skipping that conversion is the whole of the
/// sc-22712 crossfade drift: the ceiling is too generous, the sound bed lands late by the
/// accumulated crossfade time, and `-shortest` cuts the overhang off the end.
///
/// Order is `(pictureStart, trackId, assetId)` so the generated filter graph — and therefore the
/// exported bytes — do not depend on the order tracks happen to sit in the document.
pub(crate) fn audio_placements(timeline: &Value, timing: &PictureTiming) -> Vec<AudioPlacement> {
    let mut placements = Vec::new();
    let picture_duration = timing.duration();
    let Some(tracks) = timeline.get("tracks").and_then(Value::as_array) else {
        return placements;
    };
    let picture_track = picture_track_index(timeline);
    for (track_index, track) in tracks.iter().enumerate() {
        if track.get("muted").and_then(Value::as_bool).unwrap_or(false) {
            continue;
        }
        let track_id = track
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let kind = track.get("kind").and_then(Value::as_str).unwrap_or("video");
        let role = track
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or(match kind {
                "overlay" => "overlay",
                "audio" => "sound",
                _ => "picture",
            })
            .to_owned();
        let track_gain = item_f64(track, "gain", 1.0).clamp(0.0, 4.0);
        let is_audio_track = kind == "audio";
        // Only the PICTURE track's items can contribute their own generated audio. An overlay item
        // is never rendered into the picture (`main_track_items` reads one track), so mixing its
        // take's audio in would put sound in the export with no picture behind it.
        let is_picture_track = picture_track == Some(track_index);
        if !is_audio_track && !is_picture_track {
            continue;
        }
        let Some(items) = track.get("items").and_then(Value::as_array) else {
            continue;
        };
        for item in items {
            let generated = !is_audio_track;
            if generated && item.get("generatedAudio").and_then(Value::as_str) != Some("include") {
                continue;
            }
            let Some(asset_id) = item.get("assetId").and_then(Value::as_str) else {
                continue;
            };
            let timeline_start = item_f64(item, "timelineStart", 0.0).max(0.0);
            let timeline_end = item_f64(item, "timelineEnd", 0.0);
            if timeline_end <= timeline_start {
                continue;
            }
            let picture_start = timing.picture_time(timeline_start);
            if picture_start >= picture_duration {
                continue;
            }
            // The clip itself is not retimed by a crossfade — only moved — so its slot keeps its
            // own length, cut short only by the end of the picture.
            let span = (timeline_end - timeline_start)
                .min(picture_duration - picture_start)
                .max(0.0);
            if span <= 0.0 {
                continue;
            }
            let speed = item_f64(item, "speed", 1.0).clamp(0.1, 8.0);
            let source_in = item_f64(item, "sourceIn", 0.0).max(0.0);
            let declared_out = item_f64(item, "sourceOut", 0.0);
            // A clip whose declared source range is shorter than its slot still plays only what it
            // has; one with no usable range at all takes the slot's worth of source.
            let source_out = if declared_out > source_in {
                declared_out
            } else {
                source_in + span * speed
            };
            let gain = (track_gain * item_f64(item, "volume", 1.0).clamp(0.0, 2.0)).clamp(0.0, 8.0);
            let fade_in = item_f64(item, "fadeInSeconds", 0.0).clamp(0.0, span);
            let fade_out = item_f64(item, "fadeOutSeconds", 0.0).clamp(0.0, span);
            placements.push(AudioPlacement {
                asset_id: asset_id.to_owned(),
                track_id: track_id.clone(),
                role: role.clone(),
                source_in,
                source_out,
                picture_start,
                span,
                speed,
                gain,
                fade_in,
                fade_out,
                generated,
            });
        }
    }
    placements.sort_by(|left, right| {
        left.picture_start
            .total_cmp(&right.picture_start)
            .then_with(|| left.track_id.cmp(&right.track_id))
            .then_with(|| left.asset_id.cmp(&right.asset_id))
    });
    placements
}

/// Decompose a playback rate into a chain of `atempo` filters.
///
/// `atempo` is specified for `0.5..=2.0` per instance; the timeline admits `0.1..=8.0`. Chaining
/// powers of two around a final residual covers the whole range, and it is the only way a
/// speed-changed clip's audio stays in step with the `setpts`-rescaled picture the video pass
/// produced. Returns an empty chain at unit speed so the common case adds no filters at all.
pub(crate) fn atempo_chain(speed: f64) -> Vec<String> {
    let mut remaining = speed.clamp(0.1, 8.0);
    let mut filters = Vec::new();
    while remaining > 2.0 {
        filters.push("atempo=2.000000".to_owned());
        remaining /= 2.0;
    }
    while remaining < 0.5 {
        filters.push("atempo=0.500000".to_owned());
        remaining *= 2.0;
    }
    if (remaining - 1.0).abs() > 1e-6 {
        filters.push(format!("atempo={remaining:.6}"));
    }
    filters
}

/// The filter chain for one mixed source. `input` is its ffmpeg input index (input 0 is always the
/// already-muxed picture), and the chain ends in the `[aN]` label `audio_mix_filter` mixes.
///
/// Order matters and is not arbitrary: trim the source range, rebase timestamps, retime, cap at the
/// slot length, normalise the format, apply gain, apply the fades **while they are still relative
/// to the clip**, and only then delay the whole thing to its place on the timeline.
pub(crate) fn audio_source_filter(input: usize, source: &ResolvedAudioSource) -> String {
    let placement = &source.placement;
    let mut chain = vec![
        format!(
            "atrim=start={:.3}:end={:.3}",
            placement.source_in, placement.source_out
        ),
        "asetpts=PTS-STARTPTS".to_owned(),
    ];
    chain.extend(atempo_chain(placement.speed));
    chain.push(format!("atrim=end={:.3}", placement.span));
    chain.push("asetpts=PTS-STARTPTS".to_owned());
    chain.push(AUDIO_MIX_FORMAT.to_owned());
    chain.push(format!("volume={:.4}", placement.gain));
    if placement.fade_in > 0.0 {
        chain.push(format!("afade=t=in:st=0:d={:.3}", placement.fade_in));
    }
    if placement.fade_out > 0.0 {
        chain.push(format!(
            "afade=t=out:st={:.3}:d={:.3}",
            (placement.span - placement.fade_out).max(0.0),
            placement.fade_out
        ));
    }
    let delay_ms = (placement.picture_start * 1000.0).round().max(0.0) as i64;
    if delay_ms > 0 {
        chain.push(format!("adelay={delay_ms}:all=1"));
    }
    format!("[{input}:a]{}[a{input}]", chain.join(","))
}

/// Build the whole `-filter_complex` graph: one chain per source, then the mix.
///
/// `normalize=0` is load-bearing. `amix`'s default renormalises by input count, so adding a quiet
/// music bed would silently drop the dialogue by 6 dB and the gains written in the timeline would
/// mean nothing. With it off, `volume=` is the only thing that sets a level — which is the whole
/// point of giving dialogue, ambience and music independent faders.
///
/// `apad` closes the other half: the mix is padded with silence so it always outlasts the picture,
/// and `-shortest` then cuts the file at the last video frame. Without the pad, `-shortest` would
/// end the FILE when the audio ran out and truncate the picture.
///
/// [`MIX_LIMITER`] is the third leg. `normalize=0` sums N inputs, and the harness's own defaults
/// (dialogue 1.0 + ambience 0.35 + music 0.2) peak at 1.55 — the AAC encoder hard-clips anything
/// above full scale, and clipping distortion is not "their gain choices, audibly" (sc-22712).
pub(crate) fn audio_mix_filter(sources: &[ResolvedAudioSource]) -> String {
    let mut chains: Vec<String> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| audio_source_filter(index + 1, source))
        .collect();
    let labels: String = (1..=sources.len())
        .map(|index| format!("[a{index}]"))
        .collect();
    if sources.len() == 1 {
        chains.push(format!("{labels}{MIX_LIMITER},apad[aout]"));
    } else {
        chains.push(format!(
            "{labels}amix=inputs={}:normalize=0:dropout_transition=0,{MIX_LIMITER},apad[aout]",
            sources.len()
        ));
    }
    chains.join(";")
}

/// Arguments for the audio pass: take the finished picture, mix every placed source over it, and
/// write the deliverable.
///
/// The picture is copied, never re-encoded — the video pass already produced exactly the frames the
/// timeline describes, and a second encode would cost quality for nothing.
///
/// **`-map_metadata -1` is load-bearing here for the same reason it is in
/// [`mux_with_crossfades_args`] (sc-15956), and more so:** this command has the picture as input 0
/// and every sound clip after it, so ffmpeg's multi-input default would republish the muxed
/// picture's container metadata as the export's own. `an_audio_mix_does_not_inherit_a_clips_recipe`
/// fails if this regresses.
pub(crate) fn audio_mix_args(
    ffmpeg: &str,
    picture_path: &Path,
    sources: &[ResolvedAudioSource],
    output_path: &Path,
) -> WorkerResult<Vec<String>> {
    if sources.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Timeline audio mix has no sources.".to_owned(),
        ));
    }
    let mut args = vec![
        ffmpeg.to_owned(),
        "-y".to_owned(),
        "-i".to_owned(),
        picture_path.display().to_string(),
    ];
    for source in sources {
        args.push("-i".to_owned());
        args.push(source.media_path.display().to_string());
    }
    args.extend([
        "-filter_complex".to_owned(),
        audio_mix_filter(sources),
        "-map".to_owned(),
        "0:v".to_owned(),
        "-map".to_owned(),
        "[aout]".to_owned(),
        "-c:v".to_owned(),
        "copy".to_owned(),
        "-c:a".to_owned(),
        "aac".to_owned(),
        "-b:a".to_owned(),
        "192k".to_owned(),
        "-ar".to_owned(),
        "48000".to_owned(),
        "-ac".to_owned(),
        "2".to_owned(),
        "-shortest".to_owned(),
        "-map_metadata".to_owned(),
        "-1".to_owned(),
        output_path.display().to_string(),
    ]);
    Ok(args)
}

/// Mix `sources` over `picture_path` into `output_path`.
pub(crate) async fn mux_audio(
    ffmpeg: &str,
    picture_path: &Path,
    sources: &[ResolvedAudioSource],
    output_path: &Path,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    run_ffmpeg(
        audio_mix_args(ffmpeg, picture_path, sources, output_path)?,
        context,
    )
    .await
}

/// Whether `source_path` actually carries a decodable audio stream.
///
/// Asked of every source before it enters the graph, because `[N:a]` against a file with no audio
/// stream does not degrade — it fails the WHOLE command, so one silent generated take would take
/// the entire export down with it. The sidecar is not trusted for this: `hasAudio` is written by
/// whatever produced the clip, and the question here is what ffmpeg can actually read now.
///
/// Probed with `ffmpeg`, not `ffprobe`, for the reason spelled out on [`probe_source_frame_count`]:
/// the desktop app ships the imageio-ffmpeg binary and there is no `ffprobe` beside it.
async fn source_has_audio_stream(ffmpeg: &str, source_path: &Path) -> bool {
    let program = resolve_probe_program(ffmpeg);
    let mut command = Command::new(&program);
    command.args(["-hide_banner", "-i", &source_path.display().to_string()]);
    let Ok(output) = run_ffmpeg_probe_command(command).await else {
        return false;
    };
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .any(|line| line.contains("Stream #") && line.contains(": Audio:"))
}

pub(crate) fn output_dimensions(aspect_ratio: &str, resolution: u32) -> (u32, u32) {
    let resolution = resolution.max(2);
    let (width, height) = match aspect_ratio {
        "9:16" => (resolution, ((resolution as f64) * 16.0 / 9.0).ceil() as u32),
        "1:1" => (resolution, resolution),
        _ => (((resolution as f64) * 16.0 / 9.0).ceil() as u32, resolution),
    };
    (even(width), even(height))
}

pub(crate) fn even(value: u32) -> u32 {
    if value.is_multiple_of(2) {
        value
    } else {
        value + 1
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderSpec {
    width: u32,
    height: u32,
    fps: u32,
}

pub(crate) async fn render_black_segment(
    ffmpeg: &str,
    output_path: &Path,
    duration: f64,
    spec: RenderSpec,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-f".to_owned(),
            "lavfi".to_owned(),
            "-i".to_owned(),
            format!(
                "color=c=black:s={}x{}:r={}",
                spec.width, spec.height, spec.fps
            ),
            "-t".to_owned(),
            format!("{duration:.3}"),
            "-pix_fmt".to_owned(),
            "yuv420p".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await
}

pub(crate) async fn render_item_segment(
    ffmpeg: &str,
    project_path: &Path,
    item: &Value,
    asset: &Value,
    output_path: &Path,
    spec: RenderSpec,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<f64> {
    let file = asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Timeline asset file is missing.".to_owned()))?;
    let media_rel = required_value_str(file, "path")?;
    let media_path = safe_project_path(project_path, media_rel)?;
    if !media_path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "Timeline source file is missing: {}",
            media_path.display()
        )));
    }

    let source_in = item_f64(item, "sourceIn", 0.0);
    let source_out = item_f64(item, "sourceOut", item_f64(item, "timelineEnd", 4.0));
    let timeline_duration =
        item_f64(item, "timelineEnd", 4.0) - item_f64(item, "timelineStart", 0.0);
    let source_duration = (source_out - source_in).max(0.1);
    let speed = item_f64(item, "speed", 1.0).max(0.1);
    let duration = if timeline_duration > 0.0 {
        timeline_duration.max(0.1)
    } else {
        (source_duration / speed).max(0.1)
    };
    let mut vf = vec![
        format!(
            "scale={}:{}:force_original_aspect_ratio=decrease",
            spec.width, spec.height
        ),
        format!(
            "pad={}:{}:(ow-iw)/2:(oh-ih)/2:color=black",
            spec.width, spec.height
        ),
        format!("fps={}", spec.fps),
        "format=yuv420p".to_owned(),
    ];
    let transition_in = item.get("transitionIn").unwrap_or(&Value::Null);
    let transition_out = item.get("transitionOut").unwrap_or(&Value::Null);
    if transition_in.get("type").and_then(Value::as_str) == Some("fade_from_black") {
        let fade_duration = duration.min(value_f64(
            transition_in.get("duration").unwrap_or(&Value::Null),
            0.5,
        ));
        vf.push(format!("fade=t=in:st=0:d={fade_duration:.3}"));
    }
    if transition_out.get("type").and_then(Value::as_str) == Some("fade_to_black") {
        let fade_duration = duration.min(value_f64(
            transition_out.get("duration").unwrap_or(&Value::Null),
            0.5,
        ));
        vf.push(format!(
            "fade=t=out:st={:.3}:d={fade_duration:.3}",
            (duration - fade_duration).max(0.0)
        ));
    }

    let is_image_source = asset_is_image_source(asset);
    if is_image_source {
        run_ffmpeg(
            vec![
                ffmpeg.to_owned(),
                "-y".to_owned(),
                "-loop".to_owned(),
                "1".to_owned(),
                "-framerate".to_owned(),
                spec.fps.to_string(),
                "-i".to_owned(),
                media_path.display().to_string(),
                "-t".to_owned(),
                format!("{duration:.3}"),
                "-vf".to_owned(),
                vf.join(","),
                "-an".to_owned(),
                output_path.display().to_string(),
            ],
            context,
        )
        .await?;
        return Ok(duration);
    }

    let args = video_segment_args(
        ffmpeg,
        &media_path,
        output_path,
        source_in,
        speed,
        duration,
        vf,
    );
    run_ffmpeg(args, context).await?;
    Ok(duration)
}

/// Builds the ffmpeg args for the slow/fast video branch of [`render_item_segment`].
///
/// The `setpts={1/speed}*PTS` filter rescales the source timeline so the output
/// length becomes `source_duration / speed`, which is exactly the timeline
/// `duration` the function returns and that `crossfade_filter_complex` uses to
/// compute xfade offsets. The `-t` output limit MUST therefore be `duration`
/// (the declared timeline length), not the pre-rescale `source_duration`.
///
/// - speed < 1 (slow-mo): duration > source_duration; `-t source_duration` would
///   truncate the stretched output and desync every later crossfade (sc-8832).
/// - speed > 1 (fast): duration < source_duration; `-t duration` correctly caps
///   the shortened output.
/// - speed == 1: duration == source_duration; unchanged.
fn video_segment_args(
    ffmpeg: &str,
    media_path: &Path,
    output_path: &Path,
    source_in: f64,
    speed: f64,
    duration: f64,
    vf: Vec<String>,
) -> Vec<String> {
    let setpts = format!("setpts={:.6}*PTS", 1.0 / speed);
    let filters = std::iter::once(setpts)
        .chain(vf)
        .collect::<Vec<_>>()
        .join(",");
    vec![
        ffmpeg.to_owned(),
        "-y".to_owned(),
        "-ss".to_owned(),
        format!("{source_in:.3}"),
        "-i".to_owned(),
        media_path.display().to_string(),
        "-t".to_owned(),
        format!("{duration:.3}"),
        "-vf".to_owned(),
        filters,
        "-an".to_owned(),
        output_path.display().to_string(),
    ]
}

pub(crate) async fn mux_segments(
    ffmpeg: &str,
    segments: &[TimelineSegment],
    tmp_path: &Path,
    output_path: &Path,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    if segments
        .iter()
        .skip(1)
        .any(|segment| segment.transition.as_deref() == Some("crossfade"))
    {
        return mux_with_crossfades(ffmpeg, segments, output_path, context).await;
    }
    let list_path = tmp_path.join("concat.txt");
    tokio::fs::write(
        &list_path,
        concat_file_contents(segments.iter().map(|segment| &segment.path)),
    )
    .await?;
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-f".to_owned(),
            "concat".to_owned(),
            "-safe".to_owned(),
            "0".to_owned(),
            "-i".to_owned(),
            list_path.display().to_string(),
            "-c".to_owned(),
            "copy".to_owned(),
            // The concat demuxer already drops per-segment metadata, so this changes nothing
            // today; it is here because "the default happens to be right" is not a property
            // anyone maintains, and its crossfade sibling proves the default is NOT right on a
            // multi-input command. See `mux_with_crossfades_args` (sc-15956).
            "-map_metadata".to_owned(),
            "-1".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await
}

/// Mux crossfade timelines in one `xfade`/`concat` filter graph (sc-8955 / F-153: dropped the unused
/// `_tmp_path` — the crossfade path builds no intermediate concat list, unlike the plain `mux_segments`
/// copy path, so it never needed the scratch dir).
pub(crate) async fn mux_with_crossfades(
    ffmpeg: &str,
    segments: &[TimelineSegment],
    output_path: &Path,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    run_ffmpeg(
        mux_with_crossfades_args(ffmpeg, segments, output_path)?,
        context,
    )
    .await
}

pub(crate) fn mux_with_crossfades_args(
    ffmpeg: &str,
    segments: &[TimelineSegment],
    output_path: &Path,
) -> WorkerResult<Vec<String>> {
    if segments.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Timeline has no rendered segments to mux.".to_owned(),
        ));
    }
    let (filter, output_label) = crossfade_filter_complex(segments);
    let mut args = vec![ffmpeg.to_owned(), "-y".to_owned()];
    for segment in segments {
        args.push("-i".to_owned());
        args.push(segment.path.display().to_string());
    }
    args.extend([
        "-filter_complex".to_owned(),
        filter,
        "-map".to_owned(),
        format!("[{output_label}]"),
        // **Load-bearing since sc-15956.** With several `-i` inputs and no `-map_metadata`, ffmpeg
        // defaults to input 0 — so the moment generated clips started carrying a workflow tag, a
        // crossfaded export silently inherited THE FIRST CLIP'S recipe and published it as its own.
        // Measured, not theorised: the plain `-f concat` path below drops metadata and this one
        // does not, which is exactly the kind of asymmetry nobody would guess at.
        //
        // An export is a mux of several clips and is not the output of any one of their recipes;
        // presenting one of them as the recipe for the whole is the failure the envelope contract
        // exists to prevent. So the export carries NO workflow — see `TimelineExport::finalize`.
        "-map_metadata".to_owned(),
        "-1".to_owned(),
        output_path.display().to_string(),
    ]);
    Ok(args)
}

fn crossfade_filter_complex(segments: &[TimelineSegment]) -> (String, String) {
    let mut filters = Vec::with_capacity(segments.len() * 2);
    for (index, _) in segments.iter().enumerate() {
        filters.push(format!(
            "[{index}:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[v{index}]"
        ));
    }

    let mut current_label = "v0".to_owned();
    let mut current_duration = segments
        .first()
        .map(|segment| segment.duration)
        .unwrap_or(0.0);
    for (index, segment) in segments.iter().enumerate().skip(1) {
        let next_label = format!("mix{index}");
        if segment.transition.as_deref() == Some("crossfade") {
            let duration = crossfade_duration(segment.transition_duration);
            let offset = (current_duration - duration).max(0.0);
            filters.push(format!(
                "[{current_label}][v{index}]xfade=transition=fade:duration={duration:.3}:offset={offset:.3},format=yuv420p[{next_label}]"
            ));
            current_duration += segment.duration - duration;
        } else {
            filters.push(format!(
                "[{current_label}][v{index}]concat=n=2:v=1:a=0,format=yuv420p[{next_label}]"
            ));
            current_duration += segment.duration;
        }
        current_label = next_label;
    }
    (filters.join(";"), current_label)
}

pub(crate) fn crossfade_duration(duration: f64) -> f64 {
    duration.clamp(0.1, 1.5)
}

pub(crate) fn concat_file_contents<'a>(paths: impl Iterator<Item = &'a PathBuf>) -> String {
    paths
        .map(|path| {
            let path = path
                .display()
                .to_string()
                .replace('\\', "/")
                .replace('\'', "'\\''");
            format!("file '{path}'\n")
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_render_asset(
    request: &TimelineExportRequest,
    timeline: &Value,
    job_id: &str,
    media_rel: &str,
    width: u32,
    height: u32,
    duration: f64,
    audio: &[ResolvedAudioSource],
    dropped_audio_layers: &[Value],
) -> Value {
    let asset_id = fresh_asset_id();
    let created_at = now_rfc3339();
    let source_asset_ids = timeline
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|track| track.get("items").and_then(Value::as_array))
        .flatten()
        .filter_map(|item| item.get("assetId").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let aspect_ratio = timeline
        .get("aspectRatio")
        .and_then(Value::as_str)
        .unwrap_or("16:9");
    // What actually reached the mix, layer by layer (sc-22712). Recorded rather than recomputed
    // because `resolve_audio_sources` DROPS sources it cannot read, so the timeline alone does not
    // say what was audible — and "why is the music missing" is exactly the question this sidecar
    // has to be able to answer after the fact.
    let audio_layers = audio
        .iter()
        .map(|source| {
            let placement = &source.placement;
            json!({
                "trackId": placement.track_id,
                "role": placement.role,
                "assetId": placement.asset_id,
                "generated": placement.generated,
                // Where the layer sits in the EXPORTED file. Not the item's `timelineStart`: a
                // crossfade absorbs time out of the picture, so the two differ by design.
                "pictureStart": (placement.picture_start * 1000.0).round() / 1000.0,
                "durationSeconds": (placement.span * 1000.0).round() / 1000.0,
                "gain": (placement.gain * 10000.0).round() / 10000.0,
                "fadeInSeconds": placement.fade_in,
                "fadeOutSeconds": placement.fade_out,
            })
        })
        .collect::<Vec<_>>();
    let has_audio = !audio_layers.is_empty();
    json!({
        "schemaVersion": 1,
        "id": asset_id,
        "projectId": request.project_id,
        "generationSetId": Value::Null,
        "type": "render",
        "displayName": format!("{} export", request.timeline_name),
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "video/mp4",
            "width": width,
            "height": height,
            "duration": (duration * 1000.0).round() / 1000.0,
            "fps": request.fps,
            "hasAudio": has_audio
        },
        "status": {
            "favorite": false,
            "rating": 0,
            "rejected": false,
            "trashed": false
        },
        "recipe": {
            "mode": "timeline_export",
            "model": "ffmpeg",
            "adapter": "ffmpeg_timeline",
            "prompt": request.timeline_name,
            "negativePrompt": "",
            "seed": Value::Null,
            "loras": [],
            "normalizedSettings": {
                "timelineId": request.timeline_id,
                "resolution": request.resolution,
                "width": width,
                "height": height,
                "fps": request.fps,
                "aspectRatio": aspect_ratio
            },
            "rawAdapterSettings": {
                "timelinePath": request.timeline_path,
                "renderer": if has_audio {
                    "ffmpeg segment concat + audio mix"
                } else {
                    "ffmpeg segment concat"
                },
                "audioLayers": audio_layers,
                // The layers that were PLACED but never reached the mix (sc-22715), beside the
                // ones that did, with the reason each one was dropped.
                "droppedAudioLayers": dropped_audio_layers
            }
        },
        "lineage": {
            "parents": source_asset_ids,
            "sourceAssetId": request.timeline_id,
            "sourceTimestamp": Value::Null,
            "jobId": job_id
        }
    })
}

pub(crate) async fn run_ffmpeg(
    args: Vec<String>,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    run_ffmpeg_capture_stderr(args, context).await.map(|_| ())
}

/// Run FFmpeg while streaming pre-split binary chunks into stdin. The chunks are consumed one at a
/// time, so callers can move already-owned frame buffers into the writer without concatenating or
/// cloning a second whole-video buffer. Heartbeat, cancellation, binary resolution, stderr bounds,
/// and kill-on-drop behavior are identical to [`run_ffmpeg`].
pub(crate) async fn run_ffmpeg_with_stdin_chunks(
    args: Vec<String>,
    chunks: Vec<Vec<u8>>,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    run_ffmpeg_capture_stderr_inner(args, context, Some(chunks), ffmpeg_execution_timeout())
        .await
        .map(|_| ())
}

/// Run FFmpeg with the shared heartbeat/cancellation lifecycle and return its stderr on success.
/// Most callers use [`run_ffmpeg`]; media probes need FFmpeg's progress/stream metadata without
/// introducing an `ffprobe` dependency (the desktop bundle ships only FFmpeg).
pub(crate) async fn run_ffmpeg_capture_stderr(
    args: Vec<String>,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<String> {
    run_ffmpeg_capture_stderr_inner(args, context, None, ffmpeg_execution_timeout()).await
}

async fn run_ffmpeg_capture_stderr_inner(
    args: Vec<String>,
    context: Option<FfmpegContext<'_>>,
    stdin_chunks: Option<Vec<Vec<u8>>>,
    execution_timeout: Duration,
) -> WorkerResult<String> {
    let Some((program, arguments)) = args.split_first() else {
        return Err(WorkerError::InvalidPayload(
            "FFmpeg command is empty.".to_owned(),
        ));
    };
    // Let the host override the default ffmpeg binary via SCENEWORKS_FFMPEG. The
    // desktop app sets this to the venv's bundled imageio-ffmpeg (it ships no
    // system ffmpeg); the server stack / Docker leave it unset and use the
    // caller's "ffmpeg" on PATH.
    let configured_program = match std::env::var("SCENEWORKS_FFMPEG") {
        Ok(path) if program.as_str() == "ffmpeg" && !path.trim().is_empty() => path,
        _ => program.clone(),
    };
    // On Windows an ffmpeg reached through a symlink/reparse point (e.g. WinGet's
    // `…\WinGet\Links\ffmpeg.exe` shim) cannot be launched under redirection-trust enforcement:
    // CreateProcess returns ERROR_UNTRUSTED_MOUNT_POINT (os error 448), the same wall HF-cache
    // symlinks hit in the model loaders. Resolve it to the real (non-reparse) target first; a
    // no-op for a plain binary and on every other platform.
    let resolved_program =
        sceneworks_core::media_convert::resolve_ffmpeg_program(&configured_program);
    let mut command = Command::new(resolved_program.as_ref());
    command
        .args(arguments)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if stdin_chunks.is_some() {
        command.stdin(Stdio::piped());
    } else {
        command.stdin(Stdio::null());
    }
    let mut child = command
        // sc-8804/sc-21742: every ordinary timeout/heartbeat/cancel return explicitly kills and
        // reaps below. Keep kill-on-drop as the panic/unwind backstop: a Tokio child otherwise keeps
        // running after its handle is dropped and could continue writing a partial output file.
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| {
            WorkerError::Engine(format!(
                "Failed to start FFmpeg. Ensure ffmpeg is installed and on PATH: {error}"
            ))
        })?;

    let mut stdin_task = stdin_chunks.map(|chunks| {
        let mut stdin = child
            .stdin
            .take()
            .expect("piped FFmpeg stdin is available after spawn");
        tokio::spawn(async move {
            for chunk in chunks {
                stdin.write_all(&chunk).await?;
            }
            stdin.shutdown().await
        })
    });
    let mut stderr = child.stderr.take();
    let mut stderr_task = Some(tokio::spawn(async move {
        let mut bytes = Vec::new();
        if let Some(stderr) = stderr.as_mut() {
            let _ = stderr.read_to_end(&mut bytes).await;
        }
        bytes
    }));

    let (deadline, execution_timeout) = ffmpeg_execution_deadline(execution_timeout);

    let status = if let Some(context) = context {
        let mut interval = tokio::time::interval(progress_report_interval(context.settings));
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                status = child.wait() => match status {
                    Ok(status) => break status,
                    Err(error) => {
                        let cleanup = stop_ffmpeg_child(
                            &mut child,
                            &mut stdin_task,
                            &mut stderr_task,
                        ).await;
                        return Err(worker_error_with_cleanup(WorkerError::Io(error), cleanup));
                    }
                },
                _ = tokio::time::sleep_until(deadline) => {
                    let cleanup = stop_ffmpeg_child(
                        &mut child,
                        &mut stdin_task,
                        &mut stderr_task,
                    ).await;
                    return Err(worker_error_with_cleanup(
                        WorkerError::Io(ffmpeg_timeout_io_error(execution_timeout)),
                        cleanup,
                    ));
                }
                _ = interval.tick() => {
                    let control = tokio::time::timeout_at(deadline, async {
                        heartbeat(
                            context.api,
                            context.settings,
                            WorkerStatus::Busy,
                            Some(context.job_id),
                        ).await?;
                        Ok::<bool, WorkerError>(
                            cancel_requested_peek(context.api, context.job_id).await,
                        )
                    }).await;
                    match control {
                        Err(_) => {
                            let cleanup = stop_ffmpeg_child(
                                &mut child,
                                &mut stdin_task,
                                &mut stderr_task,
                            ).await;
                            return Err(worker_error_with_cleanup(
                                WorkerError::Io(ffmpeg_timeout_io_error(execution_timeout)),
                                cleanup,
                            ));
                        }
                        Ok(Err(error)) => {
                            let cleanup = stop_ffmpeg_child(
                                &mut child,
                                &mut stdin_task,
                                &mut stderr_task,
                            ).await;
                            return Err(worker_error_with_cleanup(error, cleanup));
                        }
                        Ok(Ok(true)) => {
                            let cleanup = stop_ffmpeg_child(
                                &mut child,
                                &mut stdin_task,
                                &mut stderr_task,
                            ).await;
                            if let Err(error) = mark_job_canceled(
                                context.api,
                                context.job_id,
                                context.cancel_message,
                            ).await {
                                return Err(worker_error_with_cleanup(error, cleanup));
                            }
                            return Err(worker_error_with_cleanup(
                                WorkerError::Canceled(context.cancel_message.to_owned()),
                                cleanup,
                            ));
                        }
                        Ok(Ok(false)) => {}
                    }
                }
            }
        }
    } else {
        match tokio::time::timeout_at(deadline, child.wait()).await {
            Ok(Ok(status)) => status,
            Ok(Err(error)) => {
                let cleanup =
                    stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
                return Err(worker_error_with_cleanup(WorkerError::Io(error), cleanup));
            }
            Err(_) => {
                let cleanup =
                    stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
                return Err(worker_error_with_cleanup(
                    WorkerError::Io(ffmpeg_timeout_io_error(execution_timeout)),
                    cleanup,
                ));
            }
        }
    };

    let stderr = match join_ffmpeg_task_until(&mut stderr_task, deadline).await {
        Ok(stderr) => stderr,
        Err(FfmpegTaskDeadlineError::TimedOut) => {
            let cleanup = stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
            return Err(worker_error_with_cleanup(
                WorkerError::Io(ffmpeg_timeout_io_error(execution_timeout)),
                cleanup,
            ));
        }
        Err(FfmpegTaskDeadlineError::Join(error)) => {
            let cleanup = stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
            return Err(worker_error_with_cleanup(
                task_join_error("ffmpeg stderr reader task", error),
                cleanup,
            ));
        }
    };
    let stderr = String::from_utf8_lossy(&stderr);
    if status.success() {
        if stdin_task.is_some() {
            let result = match join_ffmpeg_task_until(&mut stdin_task, deadline).await {
                Ok(result) => result,
                Err(FfmpegTaskDeadlineError::TimedOut) => {
                    let cleanup =
                        stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
                    return Err(worker_error_with_cleanup(
                        WorkerError::Io(ffmpeg_timeout_io_error(execution_timeout)),
                        cleanup,
                    ));
                }
                Err(FfmpegTaskDeadlineError::Join(error)) => {
                    let cleanup =
                        stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
                    return Err(worker_error_with_cleanup(
                        task_join_error("ffmpeg stdin writer task", error),
                        cleanup,
                    ));
                }
            };
            result.map_err(WorkerError::Io)?;
        }
        return Ok(stderr.into_owned());
    }
    if stdin_task.is_some() {
        match abort_ffmpeg_task_until(&mut stdin_task, deadline).await {
            Ok(()) => {}
            Err(FfmpegTaskDeadlineError::TimedOut) => {
                let cleanup =
                    stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
                return Err(worker_error_with_cleanup(
                    WorkerError::Io(ffmpeg_timeout_io_error(execution_timeout)),
                    cleanup,
                ));
            }
            Err(FfmpegTaskDeadlineError::Join(error)) => {
                let cleanup =
                    stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
                return Err(worker_error_with_cleanup(
                    task_join_error("ffmpeg stdin writer task", error),
                    cleanup,
                ));
            }
        }
    }
    let bounded = bounded_tail(&stderr, 10, 2000);
    if bounded.trim().is_empty() {
        Err(WorkerError::Engine(
            "FFmpeg command failed without stderr output.".to_owned(),
        ))
    } else {
        Err(WorkerError::Engine(bounded))
    }
}

enum FfmpegTaskDeadlineError {
    TimedOut,
    Join(tokio::task::JoinError),
}

async fn join_ffmpeg_task_until<T>(
    task: &mut Option<tokio::task::JoinHandle<T>>,
    deadline: tokio::time::Instant,
) -> Result<T, FfmpegTaskDeadlineError> {
    let mut handle = task
        .take()
        .expect("FFmpeg pipe task exists until joined or aborted");
    match tokio::time::timeout_at(deadline, &mut handle).await {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(error)) => Err(FfmpegTaskDeadlineError::Join(error)),
        Err(_) => {
            *task = Some(handle);
            Err(FfmpegTaskDeadlineError::TimedOut)
        }
    }
}

async fn abort_ffmpeg_task_until<T>(
    task: &mut Option<tokio::task::JoinHandle<T>>,
    deadline: tokio::time::Instant,
) -> Result<(), FfmpegTaskDeadlineError> {
    let Some(mut handle) = task.take() else {
        return Ok(());
    };
    handle.abort();
    match tokio::time::timeout_at(deadline, &mut handle).await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) if error.is_cancelled() => Ok(()),
        Ok(Err(error)) => Err(FfmpegTaskDeadlineError::Join(error)),
        Err(_) => {
            *task = Some(handle);
            Err(FfmpegTaskDeadlineError::TimedOut)
        }
    }
}

fn task_cleanup_detail(label: &str, error: FfmpegTaskDeadlineError) -> String {
    match error {
        FfmpegTaskDeadlineError::TimedOut => {
            format!("{label} did not stop within {FFMPEG_TEARDOWN_TIMEOUT:?}")
        }
        FfmpegTaskDeadlineError::Join(error) => format!("{label} failed to join: {error}"),
    }
}

/// Explicitly terminate and reap the child, then stand down both pipe tasks. `kill_on_drop` remains
/// a panic/unwind backstop, but ordinary timeout/heartbeat/cancel paths never rely on drop semantics
/// and never leave a detached stdin or stderr task behind.
async fn stop_ffmpeg_child(
    child: &mut Child,
    stdin_task: &mut Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    stderr_task: &mut Option<tokio::task::JoinHandle<Vec<u8>>>,
) -> Option<String> {
    let deadline = checked_deadline_from(tokio::time::Instant::now(), FFMPEG_TEARDOWN_TIMEOUT);
    let mut details = Vec::new();
    if let Err(error) = abort_ffmpeg_task_until(stdin_task, deadline).await {
        details.push(task_cleanup_detail("FFmpeg stdin writer", error));
    }
    if let Err(error) = abort_ffmpeg_task_until(stderr_task, deadline).await {
        details.push(task_cleanup_detail("FFmpeg stderr reader", error));
    }

    let mut child_reaped = false;
    match child.try_wait() {
        Ok(Some(_)) => child_reaped = true,
        Ok(None) => {}
        Err(error) => details.push(format!("could not inspect FFmpeg child status: {error}")),
    }
    if !child_reaped {
        let kill_error = child.start_kill().err();
        match tokio::time::timeout_at(deadline, child.wait()).await {
            // `start_kill` can lose a benign child-exited race. A successful wait proves that race
            // was safe and that the child was reaped, so the transient kill error is not reported.
            Ok(Ok(_)) => {}
            Ok(Err(wait_error)) => {
                if let Some(kill_error) = kill_error {
                    details.push(format!("could not kill FFmpeg child: {kill_error}"));
                }
                details.push(format!("could not reap FFmpeg child: {wait_error}"));
            }
            Err(_) => {
                if let Some(kill_error) = kill_error {
                    details.push(format!("could not kill FFmpeg child: {kill_error}"));
                }
                details.push(format!(
                    "FFmpeg child was not reaped within {FFMPEG_TEARDOWN_TIMEOUT:?}"
                ));
            }
        }
    }

    (!details.is_empty()).then(|| details.join("; "))
}

fn worker_error_with_cleanup(error: WorkerError, cleanup: Option<String>) -> WorkerError {
    let Some(cleanup) = cleanup else {
        return error;
    };
    tracing::error!(cleanup = %cleanup, "FFmpeg teardown reported an additional failure");
    let suffix = |detail: String| format!("{detail} FFmpeg cleanup also failed: {cleanup}");
    match error {
        WorkerError::Io(error) => {
            WorkerError::Io(std::io::Error::new(error.kind(), suffix(error.to_string())))
        }
        WorkerError::Api {
            status,
            detail,
            code,
        } => WorkerError::Api {
            status,
            detail: suffix(detail),
            code,
        },
        WorkerError::InvalidPayload(detail) => WorkerError::InvalidPayload(suffix(detail)),
        WorkerError::ExternalLibraryUnavailable(detail) => {
            WorkerError::ExternalLibraryUnavailable(suffix(detail))
        }
        WorkerError::Engine(detail) => WorkerError::Engine(suffix(detail)),
        WorkerError::Canceled(detail) => WorkerError::Canceled(suffix(detail)),
        // These wrapped errors cannot be reconstructed with extra context. The structured error log
        // above keeps the cleanup failure observable while preserving the primary classification.
        other => other,
    }
}

fn io_error_with_cleanup(error: std::io::Error, cleanup: Option<String>) -> std::io::Error {
    let Some(cleanup) = cleanup else {
        return error;
    };
    tracing::error!(cleanup = %cleanup, "FFmpeg teardown reported an additional failure");
    std::io::Error::new(
        error.kind(),
        format!("{error} FFmpeg cleanup also failed: {cleanup}"),
    )
}

#[cfg(test)]
mod ffmpeg_cleanup_error_tests {
    use super::*;

    #[test]
    fn cleanup_detail_preserves_primary_timeout_and_cancel_classification() {
        let timeout = worker_error_with_cleanup(
            WorkerError::Io(ffmpeg_timeout_io_error(Duration::from_secs(1))),
            Some("could not reap child".to_owned()),
        );
        match timeout {
            WorkerError::Io(error) => {
                assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
                assert!(error.to_string().contains("could not reap child"));
            }
            other => panic!("cleanup detail changed timeout classification: {other:?}"),
        }

        let canceled = worker_error_with_cleanup(
            WorkerError::Canceled("Canceled by user.".to_owned()),
            Some("could not inspect child".to_owned()),
        );
        match canceled {
            WorkerError::Canceled(detail) => {
                assert!(detail.starts_with("Canceled by user."));
                assert!(detail.contains("could not inspect child"));
            }
            other => panic!("cleanup detail changed cancel classification: {other:?}"),
        }
    }
}

async fn run_ffmpeg_probe_command(command: Command) -> std::io::Result<std::process::Output> {
    run_ffmpeg_probe_command_with_timeout(command, ffmpeg_execution_timeout()).await
}

/// The duration/frame-count probes used to call `Command::output()` directly. Keep their output
/// contract while exposing the same explicit timeout → kill → reap lifecycle as the shared runner.
pub(crate) async fn run_ffmpeg_probe_command_with_timeout(
    mut command: Command,
    execution_timeout: Duration,
) -> std::io::Result<std::process::Output> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn()?;
    let mut stdin_task: Option<tokio::task::JoinHandle<std::io::Result<()>>> = None;
    let mut stderr = child.stderr.take();
    let mut stderr_task = Some(tokio::spawn(async move {
        let mut bytes = Vec::new();
        if let Some(stderr) = stderr.as_mut() {
            let _ = stderr.read_to_end(&mut bytes).await;
        }
        bytes
    }));

    let (deadline, execution_timeout) = ffmpeg_execution_deadline(execution_timeout);
    let status = match tokio::time::timeout_at(deadline, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            let cleanup = stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
            return Err(io_error_with_cleanup(error, cleanup));
        }
        Err(_) => {
            let cleanup = stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
            return Err(io_error_with_cleanup(
                ffmpeg_timeout_io_error(execution_timeout),
                cleanup,
            ));
        }
    };
    let stderr = match join_ffmpeg_task_until(&mut stderr_task, deadline).await {
        Ok(stderr) => stderr,
        Err(FfmpegTaskDeadlineError::TimedOut) => {
            let cleanup = stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
            return Err(io_error_with_cleanup(
                ffmpeg_timeout_io_error(execution_timeout),
                cleanup,
            ));
        }
        Err(FfmpegTaskDeadlineError::Join(error)) => {
            let cleanup = stop_ffmpeg_child(&mut child, &mut stdin_task, &mut stderr_task).await;
            return Err(io_error_with_cleanup(
                std::io::Error::other(format!("ffmpeg stderr reader task: {error}")),
                cleanup,
            ));
        }
    };
    Ok(std::process::Output {
        status,
        stdout: Vec::new(),
        stderr,
    })
}

#[cfg(test)]
pub(crate) async fn run_ffmpeg_capture_stderr_with_timeout(
    args: Vec<String>,
    context: Option<FfmpegContext<'_>>,
    execution_timeout: Duration,
) -> WorkerResult<String> {
    run_ffmpeg_capture_stderr_inner(args, context, None, execution_timeout).await
}

#[cfg(test)]
pub(crate) async fn run_ffmpeg_with_stdin_chunks_and_timeout(
    args: Vec<String>,
    chunks: Vec<Vec<u8>>,
    context: Option<FfmpegContext<'_>>,
    execution_timeout: Duration,
) -> WorkerResult<()> {
    run_ffmpeg_capture_stderr_inner(args, context, Some(chunks), execution_timeout)
        .await
        .map(|_| ())
}

#[cfg(all(test, target_os = "macos"))]
mod frame_seek_tests {
    use super::*;
    use crate::person_track::sample_timestamps;

    #[test]
    fn frame_seek_clamps_only_the_final_inclusive_end_sample() {
        // sample_timestamps is inclusive of both ends → the last sample == duration, which
        // is not a decodable frame. The seek clamp must pull exactly that sample inside the
        // clip and leave every interior sample (and t=0) untouched.
        let duration = 8.0;
        let stamps = sample_timestamps(duration);
        let last = *stamps.last().expect("non-empty");
        assert_eq!(last, duration, "final sample sits on the inclusive end");

        for &ts in &stamps {
            let seek = frame_seek_timestamp(ts, duration);
            assert!(
                seek <= duration - FRAME_SEEK_GUARD_SECONDS + 1e-9,
                "seek {seek} must stay a frame inside the {duration}s clip"
            );
            if ts < duration - FRAME_SEEK_GUARD_SECONDS {
                assert_eq!(seek, ts, "interior samples pass through unchanged");
            }
        }
        // The final sample is the one that gets clamped.
        assert_eq!(
            frame_seek_timestamp(last, duration),
            duration - FRAME_SEEK_GUARD_SECONDS
        );
    }

    #[test]
    fn frame_seek_never_goes_negative_on_tiny_clips() {
        // A clip shorter than the guard clamps to 0 rather than a negative seek.
        assert_eq!(frame_seek_timestamp(0.1, 0.1), 0.0);
        assert_eq!(frame_seek_timestamp(0.0, 0.0), 0.0);
    }

    /// F-112 (sc-8914): the sidecar sample-rate the media handlers record and the cadence the
    /// `person_track` sampler uses are ONE value. The `PERSON_TRACK_*` crate constants must stay
    /// aliases of the `person_track` module constants so the two can never drift out of sync.
    #[test]
    fn person_track_sample_constants_are_a_single_source_of_truth() {
        assert_eq!(
            PERSON_TRACK_SAMPLE_RATE_FPS,
            crate::person_track::SAMPLE_RATE_FPS,
            "sidecar sampleRateFps must equal the sampler cadence"
        );
        assert_eq!(
            PERSON_TRACK_MAX_SAMPLES,
            crate::person_track::MAX_SAMPLES,
            "the placeholder-track sample cap must equal the sampler cap"
        );
    }
}

#[cfg(test)]
mod person_detect_timestamp_tests {
    use super::*;

    #[test]
    fn timestamp_past_short_clip_end_is_clamped_to_duration_not_3600() {
        // sc-8831: a request past the end of a 5s clip must land at the clip's duration so
        // ffmpeg -ss still pulls a frame — the old `duration.max(3600.0)` upper bound left
        // this at 999.0 (>=3600 ceiling), which produced a generic "no frame output" failure.
        let clamped = person_detect_source_timestamp(999.0, 5.0);
        assert_eq!(
            clamped, 5.0,
            "out-of-range seek clamps to the clip duration"
        );
        assert!(
            clamped < PERSON_DETECT_MAX_TIMESTAMP_SECONDS,
            "clamp must not float up to the 3600s ceiling for a short clip"
        );
    }

    #[test]
    fn in_range_timestamp_passes_through_unchanged() {
        assert_eq!(person_detect_source_timestamp(2.5, 5.0), 2.5);
        assert_eq!(person_detect_source_timestamp(0.0, 5.0), 0.0);
        assert_eq!(person_detect_source_timestamp(5.0, 5.0), 5.0);
    }

    #[test]
    fn negative_timestamp_clamps_to_zero() {
        assert_eq!(person_detect_source_timestamp(-3.0, 5.0), 0.0);
    }

    #[test]
    fn very_long_clip_is_capped_at_the_3600s_ceiling() {
        // The 3600s ceiling only tightens the bound for absurdly long clips.
        assert_eq!(
            person_detect_source_timestamp(5000.0, 7200.0),
            PERSON_DETECT_MAX_TIMESTAMP_SECONDS
        );
    }
}

/// Real-Mac end-to-end validation for the native-MLX Replace-Person pipeline
/// (epic 3704 / sc-3709): a short person video → ffmpeg frame sampling → MLX YOLO11
/// detect (sc-3633) → SORT/ByteTrack assembly (sc-3634) → MLX SAM2 segment (sc-3706/3709)
/// → per-frame masks on disk → `maskState`. This is the model-path twin of
/// [`assemble_real_person_track`] / [`segment_assembly_frames`], driven through the same
/// blocking seams but without the ApiClient/ProjectStore plumbing (that plumbing carries no
/// MLX work). It cannot run in CI — there is no GPU or weights on the runner — so it is
/// `#[ignore]` and skips cleanly unless a clip is staged. macOS-only, like the pipeline.
///
/// Run on Apple Silicon with a short clip that contains a clearly-visible person:
///
/// ```text
/// SCENEWORKS_PERSON_E2E_VIDEO=/path/to/person_clip.mp4 \
/// SCENEWORKS_PERSON_E2E_DURATION=4 \
///   cargo test -p sceneworks-worker --lib \
///   person_track_e2e -- --ignored --nocapture
/// ```
///
/// The YOLO11 and SAM2 weights must already be installed from the Model Manager (sc-17629); the
/// smoke resolves them from the public `SceneWorks/*` HF
/// repos (or pin them with `SCENEWORKS_PERSON_DETECTOR_WEIGHTS` / `SCENEWORKS_SAM2_WEIGHTS`).
#[cfg(all(test, target_os = "macos"))]
mod person_track_e2e_tests {
    use super::*;
    use std::path::PathBuf;

    /// The staged source clip (`SCENEWORKS_PERSON_E2E_VIDEO`), or `None` to skip.
    fn staged_video() -> Option<PathBuf> {
        let path = PathBuf::from(std::env::var("SCENEWORKS_PERSON_E2E_VIDEO").ok()?);
        path.exists().then_some(path)
    }

    /// Clip duration in seconds (`SCENEWORKS_PERSON_E2E_DURATION`, default 4.0), used to pick
    /// the 2-FPS sample cadence — the same `sample_timestamps` the real job uses.
    fn staged_duration() -> f64 {
        std::env::var("SCENEWORKS_PERSON_E2E_DURATION")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(4.0)
            .clamp(1.0, 3600.0)
    }

    /// The assembled, ready-to-segment clip both segmenter E2E tests share (sc-8955 / F-153): the
    /// YOLO11 detect → ByteTrack assemble result plus the `first..=last` detected-span clip paths and
    /// per-frame box anchors the SAM2 / SAM3 propagation calls consume.
    struct AssembledClip {
        detected_total: usize,
        first: usize,
        last: usize,
        clip_paths: Vec<PathBuf>,
        anchors: Vec<Option<(f64, f64, f64, f64)>>,
    }

    /// Provision the real YOLO11 detector, sample + detect every cadence frame exactly as
    /// `assemble_real_person_track` does, select the single highest-confidence detection (the clear
    /// person a user would click), and assemble the ByteTrack identity. Returns the `first..=last`
    /// detected-span clip paths + anchors ready to hand to a segmenter. Extracted from the ~100
    /// identical setup lines the SAM2 and SAM3 E2E tests each carried (sc-8955 / F-153).
    async fn detect_and_assemble(
        settings: &crate::Settings,
        video: &std::path::Path,
        duration: f64,
        timestamps: &[f64],
    ) -> AssembledClip {
        let det_weights = crate::person_jobs::require_detector_weights(settings)
            .expect("yolo11 weights provisioned");
        let mut per_frame: Vec<(f64, Vec<(crate::person_track::NormalizedBox, f64)>)> =
            Vec::with_capacity(timestamps.len());
        let mut frame_paths: Vec<PathBuf> = Vec::with_capacity(timestamps.len());
        let frames_dir = settings.data_dir.join("person-e2e-frames");
        std::fs::create_dir_all(&frames_dir).expect("frames dir");
        for (index, &timestamp) in timestamps.iter().enumerate() {
            let frame_path = frames_dir.join(format!("frame_{index:04}.png"));
            render_frame_png(
                "ffmpeg",
                video,
                &frame_path,
                frame_seek_timestamp(timestamp, duration),
                1280,
                720,
                None,
            )
            .await
            .expect("ffmpeg renders the sample frame");
            let weights_for_frame = det_weights.clone();
            let frame_for_task = frame_path.clone();
            let result = tokio::task::spawn_blocking(move || {
                crate::person_jobs::detect_people_blocking(weights_for_frame, frame_for_task, 0.25)
            })
            .await
            .expect("detect task joins")
            .expect("yolo11 detection runs");
            let boxes = result
                .detections
                .iter()
                .map(|d| {
                    (
                        crate::person_track::xyxy_to_normalized(
                            d.x1 as f64,
                            d.y1 as f64,
                            d.x2 as f64,
                            d.y2 as f64,
                            result.width,
                            result.height,
                        ),
                        d.score as f64,
                    )
                })
                .collect::<Vec<_>>();
            per_frame.push((timestamp, boxes));
            frame_paths.push(frame_path);
        }

        // Per-frame detection summary, so a failed lock is legible.
        for (timestamp, boxes) in &per_frame {
            let max_conf = boxes.iter().map(|b| b.1).fold(0.0_f64, f64::max);
            eprintln!(
                "  t={timestamp:>6.3}s  detections={}  maxConf={max_conf:.3}",
                boxes.len()
            );
        }
        let total_detections: usize = per_frame.iter().map(|(_, boxes)| boxes.len()).sum();
        assert!(
            total_detections > 0,
            "YOLO11 found no people in any sampled frame — is the clip a person video?"
        );

        // The single highest-confidence detection across all frames — the tracker only confirms a
        // new identity from a high-confidence box, so the selection must be one.
        let (selected_timestamp, selected_box, selected_conf) = per_frame
            .iter()
            .flat_map(|(timestamp, boxes)| boxes.iter().map(move |b| (*timestamp, b.0, b.1)))
            .max_by(|a, b| {
                a.2.partial_cmp(&b.2)
                    .expect("detection confidence is finite")
            })
            .expect("at least one detection exists");
        eprintln!(
            "selection: t={selected_timestamp:.3}s conf={selected_conf:.3} \
             box=({:.3},{:.3},{:.3},{:.3})",
            selected_box.x, selected_box.y, selected_box.width, selected_box.height
        );
        assert!(
            selected_conf >= 0.5,
            "best detection conf {selected_conf:.3} is below the tracker's high-confidence \
             floor (0.5) — the clip never yields a confidently-trackable person"
        );

        let observations = crate::person_track::observe(per_frame);
        let assembly = crate::person_track::assemble_track(
            &observations,
            selected_box,
            selected_timestamp,
            timestamps,
        );
        assert!(
            assembly.target_track_id.is_some(),
            "tracker failed to lock onto the selected person"
        );
        let detected_total = assembly.frames.iter().filter(|f| f.detected).count();
        assert!(detected_total > 0, "no detected target frames to segment");

        let first = assembly
            .frames
            .iter()
            .position(|f| f.detected)
            .expect("a detected frame exists");
        let last = assembly
            .frames
            .iter()
            .rposition(|f| f.detected)
            .unwrap_or(first);
        let mut clip_paths = Vec::new();
        let mut anchors = Vec::new();
        for (frame, path) in assembly.frames[first..=last]
            .iter()
            .zip(&frame_paths[first..=last])
        {
            clip_paths.push(path.clone());
            anchors.push(frame.detected.then(|| {
                let b = &frame.box_;
                (b.x, b.y, b.width, b.height)
            }));
        }
        AssembledClip {
            detected_total,
            first,
            last,
            clip_paths,
            anchors,
        }
    }

    #[test]
    #[ignore = "real Mac E2E: set SCENEWORKS_PERSON_E2E_VIDEO to a person clip; downloads YOLO11 + SAM2 weights; Apple Silicon only"]
    fn person_track_e2e_detect_track_segment_writes_masks_and_active_state() {
        let Some(video) = staged_video() else {
            eprintln!(
                "skipping: set SCENEWORKS_PERSON_E2E_VIDEO to a short clip containing a person"
            );
            return;
        };

        // Isolated scratch: a throwaway data dir (weights + frame cache resolve under it).
        let scratch_guard = tempfile::Builder::new()
            .prefix("sw-person-track-e2e-")
            .tempdir()
            .expect("temp dir");
        let scratch = scratch_guard.path();
        // `ensure_*_weights` resolve their cache under `settings.data_dir`. Through the crate-wide
        // seam so the value is RESTORED on drop: this used to `set_var` with no restore at all, which
        // leaked the scratch dir into every later `Settings::from_env()` in the process (sc-12380).
        let _env = crate::test_env::EnvVars::set(&[(
            "SCENEWORKS_DATA_DIR",
            scratch.join("data").to_str().expect("utf-8 scratch dir"),
        )]);
        let settings = crate::Settings::from_env();

        let duration = staged_duration();
        let timestamps = crate::person_track::sample_timestamps(duration);
        assert!(!timestamps.is_empty(), "sample cadence produced no frames");

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            // 1-2. Detect + assemble the selected person (shared with the SAM3 E2E, sc-8955).
            let clip = detect_and_assemble(&settings, &video, duration, &timestamps).await;
            let AssembledClip {
                detected_total,
                first,
                last,
                clip_paths,
                anchors,
            } = clip;
            // Which clip frames are detected (Some anchor) — used to count the rollup after the
            // anchors move into the propagate task.
            let clip_detected: Vec<bool> = anchors.iter().map(Option::is_some).collect();
            let gap_frames = clip_detected.iter().filter(|d| !**d).count();

            // 3. Provision the real SAM2 weights and propagate the person's mask across the
            //    detected span with the video predictor (sc-3715).
            let seg_weights = crate::person_segment::require_segmenter_weights(&settings)
                .expect("sam2 weights provisioned");
            let masks = tokio::task::spawn_blocking(move || {
                crate::person_segment::propagate_track_blocking(
                    seg_weights,
                    clip_paths,
                    anchors,
                    None,
                    None,
                )
            })
            .await
            .expect("propagate task joins")
            .expect("sam2 propagation runs");
            assert_eq!(masks.len(), last - first + 1, "one mask per clip frame");

            // Every detected frame's propagated mask is non-empty (SAM2 actually tracked the
            // person, not a blank map); count detected frames masked for the rollup.
            let mut generated = 0usize;
            for (clip_idx, pixels) in masks.iter().enumerate() {
                let foreground = pixels.iter().filter(|&&p| p > 127).count();
                assert_eq!(
                    pixels.len(),
                    (1280 * 720) as usize,
                    "mask must match the rendered frame size"
                );
                if clip_detected[clip_idx] {
                    assert!(
                        foreground > 0,
                        "clip frame {clip_idx} mask has no foreground — propagation lost the person"
                    );
                    generated += 1;
                }
            }

            // 4. The maskState rollup must report `active` — every detected frame masked.
            let mask_state = mask_rollup_state(generated, detected_total);
            eprintln!(
                "person-track E2E: detected={} span={}..={} gapFrames={} segmented={} maskState={}",
                detected_total, first, last, gap_frames, generated, mask_state
            );
            assert_eq!(
                generated, detected_total,
                "every detected frame should be masked by propagation"
            );
            assert_eq!(
                mask_state, "active",
                "all detected frames masked → maskState=active"
            );
        });
    }

    /// SAM3 cutover E2E (sc-4926): the same detect → track → assemble flow as the SAM2 test
    /// above, then segment the selected person with the **SAM3 text-concept (PCS)** path
    /// (`person_segment_sam3::segment_track_blocking`, prompt `"person"`, **no box prompt**) and
    /// the legacy **SAM2 box-prompt** path on the identical clip + anchors, and compare. Proves
    /// the box-free pipeline produces an `active` mask track end-to-end and reports SAM3-vs-SAM2
    /// mask agreement (the "parity/quality vs the SAM2 baseline" acceptance).
    ///
    /// ```text
    /// SCENEWORKS_SAM3_WEIGHTS=<facebook/sam3 snapshot dir> \  # or omit to download SceneWorks/sam3-mlx
    /// SCENEWORKS_PERSON_E2E_VIDEO=/path/to/person_clip.mp4 \
    /// SCENEWORKS_PERSON_E2E_DURATION=4 \
    ///   cargo test -p sceneworks-worker --lib person_track_e2e_sam3 -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "real Mac E2E: SAM3 + SAM2 weights; set SCENEWORKS_PERSON_E2E_VIDEO; Apple Silicon only"]
    fn person_track_e2e_sam3_segments_and_matches_sam2_baseline() {
        let Some(video) = staged_video() else {
            eprintln!(
                "skipping: set SCENEWORKS_PERSON_E2E_VIDEO to a short clip containing a person"
            );
            return;
        };
        let scratch_guard = tempfile::Builder::new()
            .prefix("sw-person-track-e2e-sam3-")
            .tempdir()
            .expect("temp dir");
        let scratch = scratch_guard.path();
        // Restored on drop — see the twin above (sc-12380).
        let _env = crate::test_env::EnvVars::set(&[(
            "SCENEWORKS_DATA_DIR",
            scratch.join("data").to_str().expect("utf-8 scratch dir"),
        )]);
        let settings = crate::Settings::from_env();
        let duration = staged_duration();
        let timestamps = crate::person_track::sample_timestamps(duration);
        assert!(!timestamps.is_empty(), "sample cadence produced no frames");

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {

            // detect → track → assemble (shared with the SAM2 E2E, sc-8955).
            let AssembledClip {
                detected_total,
                first,
                last,
                clip_paths,
                anchors,
            } = detect_and_assemble(&settings, &video, duration, &timestamps)
                .await;
            let clip_detected: Vec<bool> = anchors.iter().map(Option::is_some).collect();

            // SAM3 text-concept segmentation (no box prompt) — the path under test.
            let (sam3_model, sam3_tok) =
                crate::person_segment_sam3::require_segmenter_weights(&settings)
                    .expect("sam3 weights provisioned");
            let (cp3, an3) = (clip_paths.clone(), anchors.clone());
            let sam3 = tokio::task::spawn_blocking(move || {
                crate::person_segment_sam3::segment_track_blocking(
                    sam3_model, sam3_tok, cp3, an3, None, None,
                )
            })
            .await
            .expect("sam3 task joins")
            .expect("sam3 segmentation runs");
            assert_eq!(sam3.len(), last - first + 1, "one SAM3 mask per clip frame");
            let mut sam3_generated = 0usize;
            for (i, px) in sam3.iter().enumerate() {
                assert_eq!(px.len(), 1280 * 720, "SAM3 mask must match the frame size");
                if clip_detected[i] {
                    assert!(
                        px.iter().any(|&p| p > 127),
                        "SAM3 clip frame {i} has no foreground — concept segmentation lost the person"
                    );
                    sam3_generated += 1;
                }
            }
            let sam3_state = mask_rollup_state(sam3_generated, detected_total);
            // SAM3 masking every detected frame is the hard acceptance gate for the cutover.
            assert_eq!(
                sam3_state, "active",
                "every detected frame masked by SAM3 → maskState=active"
            );

            // SAM2 box-prompt baseline on the identical clip + anchors. Provisioning the SAM2
            // video-predictor weights needs the download API (or a `SCENEWORKS_SAM2_WEIGHTS` pin);
            // when it is unavailable the comparison is skipped — the SAM3 gate above still holds.
            let sam2_weights =
                match crate::person_segment::require_segmenter_weights(&settings)
                {
                    Ok(path) => path,
                    Err(e) => {
                        eprintln!(
                            "SAM3 E2E: sam3_state={sam3_state} sam3_generated={sam3_generated}; \
                             SAM2 baseline skipped (weights unavailable: {e}). Pin \
                             SCENEWORKS_SAM2_WEIGHTS to run the parity comparison."
                        );
                        return;
                    }
                };
            let (cp2, an2) = (clip_paths.clone(), anchors.clone());
            let sam2 = tokio::task::spawn_blocking(move || {
                crate::person_segment::propagate_track_blocking(sam2_weights, cp2, an2, None, None)
            })
            .await
            .expect("sam2 task joins")
            .expect("sam2 propagation runs");

            // Per-detected-frame mask IoU between the two segmenters.
            let mut ious = Vec::new();
            for i in 0..sam3.len() {
                if !clip_detected[i] {
                    continue;
                }
                let (a, b) = (&sam3[i], &sam2[i]);
                if a.is_empty() || b.is_empty() {
                    continue;
                }
                let (mut inter, mut union) = (0u64, 0u64);
                for k in 0..a.len() {
                    let (pa, pb) = (a[k] > 127, b[k] > 127);
                    if pa || pb {
                        union += 1;
                        if pa && pb {
                            inter += 1;
                        }
                    }
                }
                if union > 0 {
                    ious.push(inter as f64 / union as f64);
                }
            }
            let mean_iou = if ious.is_empty() {
                0.0
            } else {
                ious.iter().sum::<f64>() / ious.len() as f64
            };
            let sam3_cov = sam3.iter().map(|m| m.iter().filter(|&&p| p > 127).count()).sum::<usize>()
                as f64
                / (sam3.len() * 1280 * 720) as f64;
            eprintln!(
                "SAM3 E2E: detected={detected_total} sam3_generated={sam3_generated} \
                 sam3_state={sam3_state} meanIoU(sam3,sam2)={mean_iou:.3} sam3_coverage={sam3_cov:.3}"
            );
            assert!(
                mean_iou > 0.5,
                "SAM3 vs SAM2 mean IoU {mean_iou:.3} below the parity floor (0.5)"
            );
        });
    }

    /// Extract every `timestamps[i]` two ways for one synthetic `rate`fps × `duration`s clip and
    /// return `(gated_frames, accurate_seek_reference_frames)` as raw PNG bytes, so a caller can assert
    /// they match. The first element is the SHIPPED `render_track_frames` (probe-gated: single-pass
    /// where the source is frame-dense, per-frame accurate-seek fallback otherwise); the reference is
    /// the pre-PR per-frame path built directly with `try_render_frame_png`. Byte-identity is the honest
    /// expectation for BOTH regimes here — the gate routes each to a mechanism that lands on the same
    /// source frame the reference does (see `render_track_frames_single_pass`). Returns `None` when
    /// ffmpeg can't build the fixture (so the test degrades to a skip rather than a false failure).
    async fn extract_both_ways(
        scratch: &std::path::Path,
        label: &str,
        rate: &str,
        duration: f64,
    ) -> Option<(Vec<Vec<u8>>, Vec<Vec<u8>>)> {
        use crate::person_track::sample_timestamps;

        let case_dir = scratch.join(label);
        let cut_dir = case_dir.join("cut"); // code under test (render_track_frames)
        let ref_dir = case_dir.join("ref"); // per-frame accurate-seek reference
        std::fs::create_dir_all(&cut_dir).expect("cut dir");
        std::fs::create_dir_all(&ref_dir).expect("ref dir");

        // A deterministic testsrc2 clip at the requested cadence. `render_frame_png` accurate seek
        // needs a real container, so materialize the file first.
        let src = case_dir.join("src.mp4");
        let make = run_ffmpeg(
            vec![
                "ffmpeg".to_owned(),
                "-y".to_owned(),
                "-f".to_owned(),
                "lavfi".to_owned(),
                "-i".to_owned(),
                format!("testsrc2=size=320x240:rate={rate}:duration={duration}"),
                "-pix_fmt".to_owned(),
                "yuv420p".to_owned(),
                src.display().to_string(),
            ],
            None,
        )
        .await;
        if make.is_err() {
            eprintln!("skipping {label}: ffmpeg unavailable to build the fixture");
            return None;
        }

        let timestamps = sample_timestamps(duration);
        assert!(
            timestamps.len() > 1,
            "{label}: fixture yields a multi-sample cadence"
        );

        let cut_paths = render_track_frames(
            "ffmpeg",
            &src,
            &cut_dir,
            &timestamps,
            duration,
            1280,
            720,
            FfmpegContext {
                api: &ApiClient::new(&crate::Settings::from_env()),
                settings: &crate::Settings::from_env(),
                job_id: "render-track-frames-verify",
                cancel_message: "",
            },
        )
        .await
        .expect("render_track_frames extraction");
        assert_eq!(
            cut_paths.len(),
            timestamps.len(),
            "{label}: one frame per sample"
        );

        // Build the accurate-seek reference the SAME way the fallback does: one independent `-ss`
        // seek per sample, cloning the last produced frame forward for any tail sample whose seek
        // lands past EOF on a low-fps clip. This is exactly the pre-PR frame *selection*, made robust
        // against the short-clip tail. `render_track_frames` must match it byte-for-byte.
        let mut cut_bytes = Vec::with_capacity(timestamps.len());
        let mut ref_bytes = Vec::with_capacity(timestamps.len());
        let mut last_ref: Option<Vec<u8>> = None;
        for (index, &timestamp) in timestamps.iter().enumerate() {
            let ref_path = ref_dir.join(format!("frame_{index:04}.png"));
            let produced = try_render_frame_png(
                "ffmpeg",
                &src,
                &ref_path,
                frame_seek_timestamp(timestamp, duration),
                1280,
                720,
                None,
            )
            .await
            .expect("per-frame reference extraction");
            let bytes = if produced {
                let b = std::fs::read(&ref_path).expect("read reference frame");
                last_ref = Some(b.clone());
                b
            } else {
                last_ref
                    .clone()
                    .expect("a reference frame exists before any EOF tail sample")
            };
            ref_bytes.push(bytes);
            cut_bytes.push(std::fs::read(&cut_paths[index]).expect("read code-under-test frame"));
        }
        Some((cut_bytes, ref_bytes))
    }

    /// Run the UNGATED single-pass path ([`render_track_frames_single_pass`], no frame-count probe)
    /// against the accurate-seek reference for one synthetic clip, returning `(single_pass, reference)`
    /// PNG bytes. This is how the test proves the frame-count gate in [`render_track_frames`] is
    /// load-bearing: on a low-fps clip the ungated single pass diverges grossly from the reference, so
    /// deleting the gate would ship wrong frames. `None` on fixture-build failure (ffmpeg missing).
    async fn single_pass_ungated_vs_reference(
        scratch: &std::path::Path,
        label: &str,
        rate: &str,
        duration: f64,
    ) -> Option<(Vec<Vec<u8>>, Vec<Vec<u8>>)> {
        use crate::person_track::sample_timestamps;

        let case_dir = scratch.join(format!("{label}_ungated"));
        let cut_dir = case_dir.join("cut");
        std::fs::create_dir_all(&cut_dir).expect("cut dir");
        let src = case_dir.join("src.mp4");
        let make = run_ffmpeg(
            vec![
                "ffmpeg".to_owned(),
                "-y".to_owned(),
                "-f".to_owned(),
                "lavfi".to_owned(),
                "-i".to_owned(),
                format!("testsrc2=size=320x240:rate={rate}:duration={duration}"),
                "-pix_fmt".to_owned(),
                "yuv420p".to_owned(),
                src.display().to_string(),
            ],
            None,
        )
        .await;
        if make.is_err() {
            eprintln!("skipping {label}: ffmpeg unavailable to build the fixture");
            return None;
        }

        let timestamps = sample_timestamps(duration);
        // Drive the single-pass path DIRECTLY, skipping `render_track_frames`'s probe gate — this is the
        // "single-pass always" behavior the gate exists to prevent on sub-cadence-fps sources.
        let cut_paths = render_track_frames_single_pass(
            "ffmpeg",
            &src,
            &cut_dir,
            &timestamps,
            duration,
            1280,
            720,
            FfmpegContext {
                api: &ApiClient::new(&crate::Settings::from_env()),
                settings: &crate::Settings::from_env(),
                job_id: "render-track-frames-ungated",
                cancel_message: "",
            },
        )
        .await
        .expect("ungated single-pass extraction");

        let mut cut_bytes = Vec::with_capacity(timestamps.len());
        let mut ref_bytes = Vec::with_capacity(timestamps.len());
        let mut last_ref: Option<Vec<u8>> = None;
        for (index, &timestamp) in timestamps.iter().enumerate() {
            let ref_path = case_dir.join(format!("ref_{index:04}.png"));
            let produced = try_render_frame_png(
                "ffmpeg",
                &src,
                &ref_path,
                frame_seek_timestamp(timestamp, duration),
                1280,
                720,
                None,
            )
            .await
            .expect("per-frame reference extraction");
            let bytes = if produced {
                let b = std::fs::read(&ref_path).expect("read reference frame");
                last_ref = Some(b.clone());
                b
            } else {
                last_ref
                    .clone()
                    .expect("a reference frame exists before any EOF tail sample")
            };
            ref_bytes.push(bytes);
            cut_bytes
                .push(std::fs::read(&cut_paths[index]).expect("read ungated single-pass frame"));
        }
        Some((cut_bytes, ref_bytes))
    }

    /// F-113 (sc-8915) — the HONEST guarantee this PR ships, verified against real ffmpeg. There is no
    /// "byte-identical across all inputs" claim: `-ss` input-seek and decode-side `select` are different
    /// mechanisms, so this test proves exactly what the code delivers, regime by regime:
    ///
    /// 1. SINGLE-PASS REGIME (source fps ≥ sample cadence): the gated `render_track_frames` takes the
    ///    single pass and is BYTE-IDENTICAL to the per-frame accurate-seek reference. We build the
    ///    `select` predicate from the SAME `{:.3}`-rounded per-sample seeks the accurate path passes to
    ///    `-ss`, so both mechanisms land on the identical source frame — including at grid points that
    ///    ALIGN to a source frame's pts, the boundary case the round-1 recurrence got wrong.
    ///      - `hi_30fps_4s`: high-fps, grid points do NOT align → round-1 shipped this green by luck.
    ///      - `aligned_30fps_8s`: 16 samples on a 30fps clip, interval 8/15 s, grid points DO align to
    ///        source pts. Under the round-1 `{:.6}`-truncated `selected_n·interval` recurrence this was
    ///        5/16 samples off by one ADJACENT source frame (PSNR ~12–21 dB); the per-sample rounded
    ///        threshold lookup makes it byte-identical. This case is the whole point of round-2.
    ///      - `tie_2_25s_48fps`: 5 samples, sample[1] seek 0.5625 is a `{:.3}` round-half boundary. The
    ///        round-2 pre-round (`(x*1000.0).round()/1000.0`, round-half-AWAY-from-zero) rendered the
    ///        threshold `0.563` while the accurate `-ss` (`{:.3}`, round-half-to-EVEN) renders `0.562`.
    ///        48fps is chosen so a real source frame (frame 27, pts `27/48 = 0.5625`) sits inside that
    ///        `[0.562, 0.563)` gap: `-ss 0.562` lands on it, `select=gte(t,0.563)` skips it, so the two
    ///        select ADJACENT frames and this sample diverged byte-for-byte (60/90/120fps have no frame
    ///        in the gap — why the round-2 sweep missed it). Round-3 formats the raw seek once with the
    ///        same `{:.3}` and interpolates that exact string, so both compare `pts` against `0.562` and
    ///        match. This case FAILS under the pre-round, PASSES after; it is the whole point of round-3
    ///        (the 4/8/12/3s durations coincidentally dodge every tie point).
    ///
    /// 2. FALLBACK REGIME (source fps < sample cadence, frame-starved): the single pass is NOT safe
    ///    here — a forward-only `select` exhausts the clip early and diverges grossly — so the gate
    ///    routes to per-frame accurate seeks, which reproduce the pre-PR frames by construction and are
    ///    byte-identical to the reference.
    ///      - `lo_1fps_12s`, `lo_1_5fps_12s`: 12 / 18 real frames < 24 samples.
    ///      - `starved_1fps_3s`: 3 real frames, 6 samples.
    ///
    /// 3. GATE IS LOAD-BEARING: on a low-fps clip the UNGATED single pass (bypassing the probe) must
    ///    diverge grossly from the accurate-seek reference. We assert a large fraction of samples differ
    ///    — so if someone deletes the frame-count gate and lets single-pass run everywhere, THIS test
    ///    fails. That is the regression guard the round-1 gate needs and this test now enforces directly
    ///    (round 1 only proved the gated path matches, never that the gate itself was necessary).
    ///
    /// `#[ignore]` because it needs an ffmpeg binary (unset on CI runners); run with the bundled or a
    /// system ffmpeg:
    ///
    /// ```text
    /// SCENEWORKS_FFMPEG=apps/desktop/ffmpeg/ffmpeg \
    ///   cargo test -p sceneworks-worker --lib render_track_frames -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs an ffmpeg binary (SCENEWORKS_FFMPEG or ffmpeg on PATH)"]
    fn render_track_frames_honest_guarantee_across_fps_regimes() {
        let scratch_guard = tempfile::Builder::new()
            .prefix("sw-render-track-frames-verify-")
            .tempdir()
            .expect("temp dir");
        let scratch = scratch_guard.path();

        // (label, rate, duration): the non-aligned AND frame-ALIGNED high-fps single-pass cases, plus
        // the sub-cadence-fps and frame-starved fallback regimes. `aligned_30fps_8s` is the boundary
        // case the round-1 test missed (grid points land on source frame pts).
        let gated_cases: &[(&str, &str, f64)] = &[
            ("hi_30fps_4s", "30", 4.0),
            ("aligned_30fps_8s", "30", 8.0),
            // `{:.3}` TIE-POINT case (round-3): duration 2.25s → 5 samples → sample[1] seek 0.5625, a
            // third-decimal round-half boundary. The round-2 `(x*1000.0).round()/1000.0` pre-round
            // (round-half-AWAY-from-zero) rendered the threshold `0.563`, while the accurate `-ss`
            // (`{:.3}`, round-half-to-EVEN) renders `0.562`. That 1ms gap only mis-selects a frame when a
            // SOURCE frame's pts lands inside `[0.562, 0.563)`, so the fps must be chosen deliberately:
            // 48fps has frame 27 at exactly `27/48 = 0.5625` s, dead in the gap → `select=gte(t,0.563)`
            // skips it but `-ss 0.562` lands on it, so the pre-round diverges byte-for-byte here (60/90/
            // 120fps happen to have NO frame in that gap, which is why the round-2 sweep missed this).
            // At 48fps × 2.25s the source has 108 frames (≥ 5), so the gate takes the single pass; this
            // case FAILS under the round-2 pre-round and PASSES once the threshold is the raw seek
            // formatted with the same `{:.3}` (verified with real ffmpeg: fixed `select=gte(t,0.562)` ==
            // `-ss 0.562`; buggy `select=gte(t,0.563)` != `-ss 0.562`). The existing 4/8/12/3s durations
            // coincidentally dodge every tie point, so without this case the divergence shipped green.
            ("tie_2_25s_48fps", "48", 2.25),
            ("lo_1fps_12s", "1", 12.0),
            ("lo_1_5fps_12s", "1.5", 12.0),
            ("starved_1fps_3s", "1", 3.0),
        ];

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            // (1)+(2): the shipped, gated path is byte-identical to the accurate-seek reference in every
            // regime (single-pass where dense, per-frame fallback where sparse).
            for &(label, rate, duration) in gated_cases {
                let Some((cut, reference)) =
                    extract_both_ways(scratch, label, rate, duration).await
                else {
                    return; // ffmpeg missing → skip the whole test
                };
                assert_eq!(cut.len(), reference.len(), "{label}: one frame per sample");
                for (index, (cut_frame, ref_frame)) in cut.iter().zip(&reference).enumerate() {
                    assert_eq!(
                        cut_frame, ref_frame,
                        "{label}: gated render_track_frames sample {index} differs from the \
                         accurate-seek reference (sc-8915). For the single-pass regime this is the \
                         adjacent-frame boundary bug; for the low-fps regime it means the fallback gate \
                         regressed."
                    );
                }
            }

            // (3): prove the gate is load-bearing. UNGATED single-pass on a low-fps clip must diverge
            // grossly from the reference — otherwise the gate would be pointless and its removal would
            // go unnoticed. 1fps@12s is 12 real frames for 24 samples; the forward-only select exhausts
            // the clip and mis-selects the large majority of samples.
            let Some((ungated, reference)) =
                single_pass_ungated_vs_reference(scratch, "gate_proof_1fps_12s", "1", 12.0).await
            else {
                return;
            };
            let differing = ungated
                .iter()
                .zip(&reference)
                .filter(|(u, r)| u != r)
                .count();
            assert!(
                differing >= reference.len() / 2,
                "ungated single-pass on 1fps@12s must diverge grossly from accurate-seek to prove the \
                 frame-count gate is necessary, but only {differing}/{} samples differed — if this drops \
                 to ~0 the gate has become a no-op and low-fps sources would silently regress (sc-8915)",
                reference.len()
            );
        });
    }

    /// The single pass is an OPTIMIZATION, not a correctness dependency: if this ffmpeg build rejects
    /// the long `select` filter, `render_track_frames` must degrade to the exact per-frame accurate-seek
    /// path rather than hard-failing the job (sc-8915, reviewer [minor]). This test forces the single
    /// pass to fail on an otherwise frame-DENSE clip (so the frame-count gate passes and the runtime
    /// reaches the single pass) and asserts the job still succeeds AND lands on the byte-identical frames
    /// the per-frame path would — i.e. graceful, lossless degradation.
    ///
    /// The failure is injected with a tiny wrapper "ffmpeg" that exits non-zero whenever it sees the
    /// single-pass `select=gte` filter and otherwise `exec`s the real ffmpeg. Passing this wrapper as the
    /// `ffmpeg` argument (an absolute path, NOT the literal `"ffmpeg"`) bypasses the `SCENEWORKS_FFMPEG`
    /// override, so the probe and every per-frame seek run the real binary while only the single pass is
    /// sabotaged. `#[ignore]` because it needs a real ffmpeg (via `SCENEWORKS_FFMPEG` or on PATH).
    #[test]
    #[ignore = "needs an ffmpeg binary (SCENEWORKS_FFMPEG or ffmpeg on PATH)"]
    fn render_track_frames_falls_back_to_per_frame_when_single_pass_fails() {
        use crate::person_track::sample_timestamps;
        use std::os::unix::fs::PermissionsExt;

        let scratch_guard = tempfile::Builder::new()
            .prefix("sw-render-track-frames-fallback-")
            .tempdir()
            .expect("temp dir");
        let scratch = scratch_guard.path();

        // Resolve the real ffmpeg the wrapper should delegate to (bundled override or PATH default).
        let real_ffmpeg = std::env::var("SCENEWORKS_FFMPEG")
            .ok()
            .filter(|p| !p.trim().is_empty())
            .unwrap_or_else(|| "ffmpeg".to_owned());

        // Wrapper "ffmpeg": fail on the single-pass `select=gte` filter, otherwise exec the real binary.
        let wrapper = scratch.join("fake-ffmpeg.sh");
        std::fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\nfor a in \"$@\"; do\n  case \"$a\" in\n    *select=gte*)\n      echo 'fake-ffmpeg: single-pass filter rejected' 1>&2\n      exit 1\n      ;;\n  esac\ndone\nexec \"{real_ffmpeg}\" \"$@\"\n"
            ),
        )
        .expect("write wrapper ffmpeg");
        let mut perms = std::fs::metadata(&wrapper)
            .expect("wrapper metadata")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&wrapper, perms).expect("chmod wrapper");
        let wrapper_str = wrapper.display().to_string();

        let duration = 4.0_f64; // 30fps → 120 frames, gate passes → runtime reaches the single pass
        let src = scratch.join("src.mp4");

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            // Build a dense fixture with the REAL ffmpeg (wrapper would allow it too, but keep it simple).
            let make = run_ffmpeg(
                vec![
                    real_ffmpeg.clone(),
                    "-y".to_owned(),
                    "-f".to_owned(),
                    "lavfi".to_owned(),
                    "-i".to_owned(),
                    format!("testsrc2=size=320x240:rate=30:duration={duration}"),
                    "-pix_fmt".to_owned(),
                    "yuv420p".to_owned(),
                    src.display().to_string(),
                ],
                None,
            )
            .await;
            if make.is_err() {
                eprintln!("skipping fallback test: ffmpeg unavailable to build the fixture");
                return;
            }

            let timestamps = sample_timestamps(duration);

            // (a) The gated wrapper: single pass fails (wrapper rejects `select=gte`) → runtime fallback.
            let fb_dir = scratch.join("fallback");
            std::fs::create_dir_all(&fb_dir).expect("fallback dir");
            let fb_paths = render_track_frames(
                &wrapper_str,
                &src,
                &fb_dir,
                &timestamps,
                duration,
                1280,
                720,
                FfmpegContext {
                    api: &ApiClient::new(&crate::Settings::from_env()),
                    settings: &crate::Settings::from_env(),
                    job_id: "render-track-frames-fallback",
                    cancel_message: "",
                },
            )
            .await
            .expect("render_track_frames must SUCCEED via the per-frame fallback when single-pass fails");
            assert_eq!(fb_paths.len(), timestamps.len(), "one frame per sample after fallback");

            // (b) The per-frame path run DIRECTLY with the real ffmpeg — the byte reference the fallback
            // must reproduce (the fallback simply calls this same fn on single-pass Err).
            let ref_dir = scratch.join("per_frame_ref");
            std::fs::create_dir_all(&ref_dir).expect("ref dir");
            let ref_paths = render_track_frames_per_frame(
                &real_ffmpeg,
                &src,
                &ref_dir,
                &timestamps,
                duration,
                1280,
                720,
                FfmpegContext {
                    api: &ApiClient::new(&crate::Settings::from_env()),
                    settings: &crate::Settings::from_env(),
                    job_id: "render-track-frames-fallback-ref",
                    cancel_message: "",
                },
            )
            .await
            .expect("per-frame reference extraction");

            for (index, (fb, reference)) in fb_paths.iter().zip(&ref_paths).enumerate() {
                let fb_bytes = std::fs::read(fb).expect("read fallback frame");
                let ref_bytes = std::fs::read(reference).expect("read per-frame reference");
                assert_eq!(
                    fb_bytes, ref_bytes,
                    "sample {index}: single-pass-failure fallback must be byte-identical to the \
                     per-frame accurate-seek path (sc-8915 graceful degradation)"
                );
            }
        });
    }
}

#[cfg(all(test, target_os = "macos"))]
mod render_track_frame_probe_tests {
    use super::*;

    #[test]
    fn parse_ffmpeg_frame_count_reads_the_last_padded_progress_token() {
        // ffmpeg pads the counter and prints many progress lines; the LAST frame= wins.
        let stderr = "frame=    1 fps=0.0 q=-1.0 size=N/A\nframe=   12 fps=0.0 q=-1.0 Lsize=N/A\n";
        assert_eq!(parse_ffmpeg_frame_count(stderr), Some(12));
    }

    #[test]
    fn parse_ffmpeg_frame_count_handles_no_space_and_missing_token() {
        assert_eq!(parse_ffmpeg_frame_count("frame=120 other"), Some(120));
        assert_eq!(parse_ffmpeg_frame_count("no counter here"), None);
    }
}

#[cfg(test)]
mod crossfade_mux_tests {
    use super::*;

    fn segment(
        path: &str,
        duration: f64,
        transition: Option<&str>,
        transition_duration: f64,
    ) -> TimelineSegment {
        TimelineSegment {
            path: PathBuf::from(path),
            duration,
            transition: transition.map(str::to_owned),
            transition_duration,
        }
    }

    #[test]
    fn crossfade_mux_builds_one_filter_graph_for_all_segments() {
        let segments = vec![
            segment("a.mp4", 2.0, None, 0.5),
            segment("b.mp4", 3.0, Some("crossfade"), 0.5),
            segment("c.mp4", 4.0, None, 0.5),
        ];

        let args = mux_with_crossfades_args("ffmpeg", &segments, Path::new("out.mp4")).unwrap();
        let filter_index = args
            .iter()
            .position(|arg| arg == "-filter_complex")
            .expect("filter_complex present")
            + 1;
        let filter = &args[filter_index];

        assert_eq!(args.iter().filter(|arg| arg.as_str() == "-i").count(), 3);
        assert_eq!(
            args.iter()
                .filter(|arg| arg.as_str() == "-filter_complex")
                .count(),
            1
        );
        assert!(filter.contains("[v0][v1]xfade=transition=fade:duration=0.500:offset=1.500"));
        assert!(filter.contains("[mix1][v2]concat=n=2:v=1:a=0"));
        assert!(args.windows(2).any(|pair| pair == ["-map", "[mix2]"]));
        assert!(!args.iter().any(|arg| arg.contains("xfade_")));
        assert!(!args.iter().any(|arg| arg.contains("concat_")));
    }

    #[test]
    fn crossfade_mux_rejects_empty_segments() {
        let error = mux_with_crossfades_args("ffmpeg", &[], Path::new("out.mp4"))
            .expect_err("empty timeline rejects");

        assert!(matches!(error, WorkerError::InvalidPayload(_)));
        assert!(error
            .to_string()
            .contains("Timeline has no rendered segments"));
    }

    fn t_value(args: &[String]) -> f64 {
        let idx = args.iter().position(|arg| arg == "-t").expect("-t present") + 1;
        args[idx].parse().expect("-t is a number")
    }

    #[test]
    fn slow_motion_segment_output_limit_matches_timeline_duration() {
        // speed=0.5 stretches a 2s source to a 4s timeline via setpts=2.0*PTS.
        // -t must be the 4s timeline duration, NOT the 2s source_duration, or the
        // stretched output is truncated and every later crossfade desyncs (sc-8832).
        let source_duration = 2.0_f64;
        let speed = 0.5_f64;
        let duration = source_duration / speed; // 4.0

        let args = video_segment_args(
            "ffmpeg",
            Path::new("in.mp4"),
            Path::new("out.mp4"),
            0.0,
            speed,
            duration,
            vec!["format=yuv420p".to_owned()],
        );

        assert!(
            (t_value(&args) - duration).abs() < 1e-6,
            "-t must equal timeline duration {duration}, got {}",
            t_value(&args)
        );
        assert!(
            (t_value(&args) - source_duration).abs() > 1e-6,
            "-t must NOT be the truncating source_duration {source_duration}"
        );
        // setpts stretches: 1/0.5 = 2.0
        assert!(args.iter().any(|arg| arg.contains("setpts=2.000000*PTS")));
    }

    #[test]
    fn fast_motion_segment_output_limit_matches_timeline_duration() {
        // speed=2.0 compresses a 2s source to a 1s timeline via setpts=0.5*PTS.
        // -t must be the 1s timeline duration (source_duration/speed).
        let source_duration = 2.0_f64;
        let speed = 2.0_f64;
        let duration = source_duration / speed; // 1.0

        let args = video_segment_args(
            "ffmpeg",
            Path::new("in.mp4"),
            Path::new("out.mp4"),
            0.0,
            speed,
            duration,
            vec!["format=yuv420p".to_owned()],
        );

        assert!(
            (t_value(&args) - duration).abs() < 1e-6,
            "-t must equal timeline duration {duration}, got {}",
            t_value(&args)
        );
        assert!(args.iter().any(|arg| arg.contains("setpts=0.500000*PTS")));
    }

    #[test]
    fn unit_speed_segment_output_limit_equals_source_duration() {
        // speed=1.0: duration == source_duration; behaviour unchanged.
        let source_duration = 3.0_f64;
        let speed = 1.0_f64;
        let duration = source_duration / speed; // 3.0

        let args = video_segment_args(
            "ffmpeg",
            Path::new("in.mp4"),
            Path::new("out.mp4"),
            0.5,
            speed,
            duration,
            vec!["format=yuv420p".to_owned()],
        );

        assert!((t_value(&args) - duration).abs() < 1e-6);
        assert!((t_value(&args) - source_duration).abs() < 1e-6);
        // source_in preserved as -ss.
        let ss_idx = args.iter().position(|arg| arg == "-ss").unwrap() + 1;
        assert_eq!(args[ss_idx], "0.500");
    }
}

/// The timeline export's relationship to the workflow envelope (sc-15956).
///
/// An export embeds nothing — see [`TimelineExport::finalize`] — but that is only half the job.
/// The other half is that it must not INHERIT one, and inheritance is ffmpeg's default rather than
/// something anyone wrote.
#[cfg(test)]
mod export_metadata_tests {
    use super::*;

    fn segment(path: &str, duration: f64, transition: Option<&str>) -> TimelineSegment {
        TimelineSegment {
            path: PathBuf::from(path),
            duration,
            transition: transition.map(str::to_owned),
            transition_duration: 0.5,
        }
    }

    /// The position of `-map_metadata`'s value in an argument list, if it is there at all.
    fn map_metadata_value(args: &[String]) -> Option<&str> {
        args.iter()
            .position(|arg| arg == "-map_metadata")
            .and_then(|at| args.get(at + 1))
            .map(String::as_str)
    }

    /// **The measured surprise this guard exists for.**
    ///
    /// `mux_with_crossfades_args` passes every segment as its own `-i`. With several inputs and no
    /// `-map_metadata`, ffmpeg copies input 0's container metadata to the output — so once
    /// generated clips began carrying a workflow tag, a crossfaded export silently published THE
    /// FIRST CLIP'S recipe as its own. Verified against a real ffmpeg before this was written: the
    /// output carried clip 1's comment verbatim.
    ///
    /// An export is a mux of several clips and is not the output of any one of their recipes.
    /// Presenting one of them as the recipe for the whole is precisely the writer-and-reader
    /// disagreement the envelope contract exists to prevent.
    #[test]
    fn a_crossfaded_export_does_not_inherit_a_clips_recipe() {
        let segments = [
            segment("a.mp4", 2.0, None),
            segment("b.mp4", 3.0, Some("crossfade")),
            segment("c.mp4", 4.0, None),
        ];
        let args = mux_with_crossfades_args("ffmpeg", &segments, Path::new("out.mp4"))
            .expect("three segments mux");
        assert_eq!(
            map_metadata_value(&args),
            Some("-1"),
            "the crossfade mux MUST strip metadata: ffmpeg's default on a multi-input command is \
             to copy input 0's, which would publish the first clip's recipe as the export's own. \
             Args: {args:?}"
        );
    }

    /// The plain concat path says the same thing, though it did not have to.
    ///
    /// The concat demuxer drops per-segment metadata on its own, so this changes no behaviour — it
    /// is here because its crossfade sibling proves the default is not reliable, and an asymmetry
    /// between two paths that produce the same kind of file is how one of them regresses unnoticed.
    #[test]
    fn the_concat_export_strips_metadata_explicitly_too() {
        // `mux_segments` builds its arguments inline around a concat list it has already written,
        // so there is no pure args builder to call the way the crossfade path has one. Read the
        // constant off the source instead — the same technique the seam lint uses, and it is the
        // flag's PRESENCE that is the claim here, not a runtime behaviour.
        let source = include_str!("media_jobs.rs");
        let concat_body = source
            .split("pub(crate) async fn mux_segments(")
            .nth(1)
            .expect("mux_segments is still here")
            .split("pub(crate) async fn mux_with_crossfades(")
            .next()
            .expect("the function ends");
        assert!(
            concat_body.contains("\"-map_metadata\".to_owned()"),
            "`mux_segments` must pass `-map_metadata` explicitly — see \
             `a_crossfaded_export_does_not_inherit_a_clips_recipe` for why the default is not \
             something to rely on"
        );
    }

    /// The export writes no envelope of its own, and the lint's registry agrees.
    ///
    /// `TimelineExport::finalize` is deliberately NOT in `WORKFLOW_WRITE_SEAMS`: that registry
    /// lists functions an envelope can reach, and an entry whose function touches none fails as a
    /// stale claim. What keeps the decision honest is that the body must stay free of the envelope
    /// surface, which is what this reads.
    #[test]
    fn the_timeline_export_builds_no_envelope() {
        let source = include_str!("media_jobs.rs");
        let finalize = source
            .split("    async fn finalize(")
            .nth(1)
            .expect("finalize is still here")
            .split("\n    }\n}")
            .next()
            .expect("the function ends");
        for forbidden in [
            "embeddable_workflow_share",
            "embeddable_video_workflow_share",
            "build_workflow_share",
            "write_workflow_metadata_file",
            "write_workflow_chunk",
        ] {
            assert!(
                !finalize.contains(forbidden),
                "`TimelineExport::finalize` calls `{forbidden}`, so it now EMBEDS. A timeline \
                 export is a mux of several clips rather than the output of any one recipe, and \
                 its sidecar `prompt` slot holds the user's own timeline name. If this is \
                 deliberate, it needs a `WORKFLOW_WRITE_SEAMS` entry naming the builders behind it."
            );
        }
    }
}

/// sc-19549: the single-frame extraction seek. `ffmpeg -ss T -i src -frames:v 1` past the end of a
/// source encodes NOTHING and **exits 0**, so an exit-code assertion proves nothing here — every
/// test below asserts on the FILE ffmpeg was supposed to produce, and the video case asserts on its
/// CONTENT (a container header can lie about duration; the pixels cannot).
///
/// Deliberately ungated `#[cfg(test)]` — not `cfg(macos | backend-candle)` — because the code under
/// test is compiled on every target and the lane that caught this defect (`parity-rust`) is plain
/// Linux. These run there, under `SCENEWORKS_REQUIRE_FFMPEG`, rather than soft-skipping.
#[cfg(test)]
mod frame_extract_seek_tests {
    use super::*;
    use crate::video_jobs::tests::ffmpeg_reachable;

    fn image_asset() -> Value {
        json!({"type": "image", "file": {"path": "still.png", "mimeType": "image/png"}})
    }

    fn video_asset() -> Value {
        json!({"type": "video", "file": {"path": "clip.mp4", "mimeType": "video/mp4"}})
    }

    fn scratch(prefix: &str) -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(prefix)
            .tempdir()
            .expect("temp dir")
    }

    /// Build a deterministic fixture with lavfi. Returns false when ffmpeg cannot build it.
    async fn build_fixture(args: Vec<String>) -> bool {
        run_ffmpeg(args, None).await.is_ok()
    }

    fn still_fixture_args(out: &Path) -> Vec<String> {
        vec![
            "ffmpeg".to_owned(),
            "-y".to_owned(),
            "-f".to_owned(),
            "lavfi".to_owned(),
            "-i".to_owned(),
            "testsrc2=size=320x240:rate=1".to_owned(),
            "-frames:v".to_owned(),
            "1".to_owned(),
            out.display().to_string(),
        ]
    }

    fn video_fixture_args(out: &Path, duration: f64) -> Vec<String> {
        vec![
            "ffmpeg".to_owned(),
            "-y".to_owned(),
            "-f".to_owned(),
            "lavfi".to_owned(),
            "-i".to_owned(),
            format!("testsrc2=size=320x240:rate=10:duration={duration}"),
            "-pix_fmt".to_owned(),
            "yuv420p".to_owned(),
            out.display().to_string(),
        ]
    }

    /// A still is one frame at t=0, so EVERY playhead over it denotes that frame → seek 0.
    /// A timed source clamps to the last real frame, which is NOT 0 — the distinction the naive
    /// "just retry at 0" fix erases.
    #[test]
    fn still_clamps_to_zero_while_a_timed_source_clamps_to_its_last_frame() {
        assert_eq!(
            frame_seek_timestamp(10.0, 2.0),
            2.0 - FRAME_SEEK_GUARD_SECONDS
        );
        assert!(
            frame_seek_timestamp(10.0, 2.0) > 0.0,
            "a video past its end must land at the END of the clip, never at the first frame"
        );
        // Interior playheads are untouched.
        assert_eq!(frame_seek_timestamp(0.5, 2.0), 0.5);
        // A source shorter than the guard collapses to 0 rather than a negative seek.
        assert_eq!(frame_seek_timestamp(0.5, 0.04), 0.0);
    }

    #[test]
    fn image_source_predicate_matches_the_timeline_renderers_own_spelling() {
        assert!(asset_is_image_source(&image_asset()));
        assert!(!asset_is_image_source(&video_asset()));
        // mimeType alone is enough (the `type` key is absent on older sidecars).
        assert!(asset_is_image_source(
            &json!({"file": {"mimeType": "image/jpeg"}})
        ));
        // An explicit video type wins over an image-ish mime.
        assert!(!asset_is_image_source(
            &json!({"type": "video", "file": {"mimeType": "image/png"}})
        ));
    }

    #[test]
    fn duration_parsers_read_ffmpegs_two_shapes() {
        let header = "  Duration: 00:00:02.00, start: 0.000000, bitrate: 42 kb/s";
        assert_eq!(parse_ffmpeg_header_duration(header), Some(2.0));
        assert_eq!(
            parse_ffmpeg_header_duration("  Duration: N/A, bitrate: N/A"),
            None
        );
        assert_eq!(
            parse_ffmpeg_header_duration("  Duration: 00:01:30.50,"),
            Some(90.5)
        );

        let progress = "frame=   20 fps=0.0 q=-0.0 Lsize=N/A time=00:00:02.00 bitrate=N/A";
        assert_eq!(parse_ffmpeg_measured_time(progress), Some(2.0));
        // The LAST token wins (ffmpeg reprints progress as it goes).
        let two = "time=00:00:01.00 x\ntime=00:00:02.50 y";
        assert_eq!(parse_ffmpeg_measured_time(two), Some(2.5));
        assert_eq!(parse_ffmpeg_measured_time("no progress here"), None);
    }

    /// THE DEFECT. A still image extracted at a playhead > 0 must produce a file.
    ///
    /// The first half is the CONTROL that pins the mechanism: the raw playhead ffmpeg used to be
    /// handed produces NO file while ffmpeg exits 0. The second half is the fix.
    #[test]
    fn still_image_at_a_nonzero_playhead_produces_a_frame() {
        if !ffmpeg_reachable() {
            eprintln!("skipping: no ffmpeg on this lane");
            return;
        }
        let guard = scratch("sw-still-seek-");
        let dir = guard.path();
        let src = dir.join("still.png");

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            assert!(
                build_fixture(still_fixture_args(&src)).await,
                "fixture built"
            );

            // CONTROL: the pre-fix behaviour — seek at the raw playhead. ffmpeg EXITS 0 and writes
            // nothing, which is exactly why this was silent, so the assertion is on the FILE.
            // `render_frame_png` is the ungated twin of `try_render_frame_png` (which is gated to
            // macos|candle and so cannot be named from this ungated test module): it runs the same
            // command and turns "no output" into an error, so its Err IS the missing file.
            let unclamped = dir.join("unclamped.png");
            let error = render_frame_png("ffmpeg", &src, &unclamped, 0.5, 320, 240, None)
                .await
                .expect_err(
                    "CONTROL: seeking a still at 0.5s must produce nothing — if this succeeds, \
                     the defect's mechanism changed and the test below no longer proves anything",
                );
            assert!(
                error.to_string().contains("did not produce frame output"),
                "the control must fail for the empty-output reason, not some other error: {error}"
            );
            assert!(!unclamped.exists(), "ffmpeg wrote no file at all");

            // THE FIX: the resolved seek lands on the still's only frame.
            let seek = resolve_frame_seek("ffmpeg", &image_asset(), &src, 0.5)
                .await
                .expect("seek resolves for a still");
            assert_eq!(seek, 0.0, "a still clamps to its single frame at t=0");

            let out = dir.join("out.png");
            render_frame_png("ffmpeg", &src, &out, seek, 320, 240, None)
                .await
                .expect("still extraction at a non-zero playhead must succeed");
            let bytes = std::fs::metadata(&out).expect("frame file exists").len();
            assert!(bytes > 0, "the extracted still frame must not be empty");
        });
    }

    /// THE CASE THE NAIVE FIX GETS WRONG. A video asked for past its end must yield the LAST frame.
    /// Asserted on PIXELS: byte-identical to a render at the clamped end, and DIFFERENT from t=0.
    #[test]
    fn video_past_its_end_yields_the_last_frame_not_the_first() {
        if !ffmpeg_reachable() {
            eprintln!("skipping: no ffmpeg on this lane");
            return;
        }
        let guard = scratch("sw-video-seek-");
        let dir = guard.path();
        let src = dir.join("clip.mp4");
        let duration = 2.0;

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            assert!(
                build_fixture(video_fixture_args(&src, duration)).await,
                "fixture built"
            );

            // The probe must MEASURE the clip, not guess.
            let probed = probe_source_duration("ffmpeg", &src)
                .await
                .expect("duration probe succeeds on a real clip");
            assert!(
                (probed - duration).abs() < 0.25,
                "probed duration {probed} must match the {duration}s fixture"
            );

            // A playhead far past the end.
            let seek = resolve_frame_seek("ffmpeg", &video_asset(), &src, 10.0)
                .await
                .expect("seek resolves for a video");
            assert_eq!(seek, frame_seek_timestamp(10.0, probed));
            assert!(seek > 0.0, "must not collapse to the first frame");

            let actual = dir.join("actual.png");
            let first = dir.join("first.png");
            let last = dir.join("last.png");
            render_frame_png("ffmpeg", &src, &actual, seek, 320, 240, None)
                .await
                .expect("clamped extraction produces a frame");
            render_frame_png("ffmpeg", &src, &first, 0.0, 320, 240, None)
                .await
                .expect("reference first frame");
            render_frame_png(
                "ffmpeg",
                &src,
                &last,
                probed - FRAME_SEEK_GUARD_SECONDS,
                320,
                240,
                None,
            )
            .await
            .expect("reference last frame");

            let actual_bytes = std::fs::read(&actual).expect("actual frame");
            let first_bytes = std::fs::read(&first).expect("first frame");
            let last_bytes = std::fs::read(&last).expect("last frame");

            assert!(
                !actual_bytes.is_empty(),
                "extraction produced an empty file"
            );
            assert_ne!(
                actual_bytes, first_bytes,
                "a playhead past the end must NOT hand back the first frame — that is the silent \
                 lie the `retry at 0` fix would introduce"
            );
            assert_eq!(
                actual_bytes, last_bytes,
                "a playhead past the end must hand back the clip's LAST real frame"
            );
        });
    }

    /// The probe's failure must be OBSERVABLE — an `Err` carrying ffmpeg's own words, never a
    /// quiet `None` that a caller cannot tell apart from a legitimate absence (the shape that let
    /// `probe_source_frame_count` return `None` on every ffmpeg-6 host unnoticed).
    #[test]
    fn duration_probe_fails_loudly_rather_than_returning_a_quiet_absence() {
        if !ffmpeg_reachable() {
            eprintln!("skipping: no ffmpeg on this lane");
            return;
        }
        let guard = scratch("sw-probe-loud-");
        let dir = guard.path();
        let junk = dir.join("not-media.mp4");
        std::fs::write(&junk, b"this is not a media file").expect("write junk");

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let error = probe_source_duration("ffmpeg", &junk)
                .await
                .expect_err("an unreadable source must be an error, not a silent default");
            let message = error.to_string();
            assert!(
                message.contains("Could not determine the duration"),
                "probe failure must name itself: {message}"
            );
            assert!(
                message.contains("ffmpeg said:"),
                "probe failure must carry ffmpeg's own stderr so it is diagnosable: {message}"
            );

            // And that failure PROPAGATES — it does not degrade into a seek of 0.
            resolve_frame_seek("ffmpeg", &video_asset(), &junk, 5.0)
                .await
                .expect_err("an unprobeable video must fail the extraction, not guess a seek");
        });
    }

    /// The still short-circuit's OWN value, distinct from the measured fallback that happens to
    /// agree with it: a declared still resolves to its single frame WITHOUT probing the source at
    /// all. Pinned against a path that cannot be probed, so only the short-circuit can answer.
    /// (Without this the short-circuit is a mutation survivor — the fallback quietly covers for it.)
    #[test]
    fn a_declared_still_resolves_without_probing_the_source() {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let missing = Path::new("/nonexistent/sw-19549/not-here.png");
            let seek = resolve_frame_seek("ffmpeg", &image_asset(), missing, 0.5)
                .await
                .expect("a declared still needs no probe to know its only frame is at t=0");
            assert_eq!(seek, 0.0);

            // Same unprobeable path, declared a video → this MUST fail, proving the success above
            // came from the still short-circuit and not from a probe that quietly tolerates junk.
            resolve_frame_seek("ffmpeg", &video_asset(), missing, 0.5)
                .await
                .expect_err("an unprobeable video must not resolve a seek");
        });
    }

    /// A still whose sidecar does not declare it an image still converges on 0 — the measured
    /// fallback reports a sub-guard duration, which `frame_seek_timestamp` clamps to 0. Belt and
    /// braces: the fix does not depend on the asset metadata being right.
    #[test]
    fn a_still_with_no_image_metadata_still_converges_on_the_only_frame() {
        if !ffmpeg_reachable() {
            eprintln!("skipping: no ffmpeg on this lane");
            return;
        }
        let guard = scratch("sw-still-nometa-");
        let dir = guard.path();
        let src = dir.join("still.png");

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            assert!(
                build_fixture(still_fixture_args(&src)).await,
                "fixture built"
            );

            // No `type`, no image mime → the image short-circuit does NOT fire.
            let bare = json!({"file": {"path": "still.png"}});
            assert!(!asset_is_image_source(&bare));

            let seek = resolve_frame_seek("ffmpeg", &bare, &src, 0.5)
                .await
                .expect("the measured fallback resolves a still with no metadata");
            assert_eq!(
                seek, 0.0,
                "measured sub-guard duration clamps to the only frame"
            );

            let out = dir.join("out.png");
            render_frame_png("ffmpeg", &src, &out, seek, 320, 240, None)
                .await
                .expect("extraction succeeds");
            assert!(std::fs::metadata(&out).expect("frame file").len() > 0);
        });
    }
}

/// sc-22712: dialogue, ambience and music as three independently controlled buses, mixed into the
/// timeline export.
///
/// **Every claim here is measured off the exported MP4's own samples**, never off the ffmpeg
/// command that was supposed to produce them. An argument-shape assertion cannot tell the
/// difference between a filter graph that mutes a track and one that merely fails to unmute it,
/// and "the gains in the timeline mean what they say" is not a property of a string. So each
/// fixture bed is a tone at a known frequency, the export is decoded back to PCM, and a Goertzel
/// probe asks what is actually audible in the window the timeline says it should be.
///
/// Deliberately ungated `#[cfg(test)]` for the same reason as `frame_extract_seek_tests` directly
/// above: the code under test compiles on every target, and the lane that runs it is plain Linux.
#[cfg(test)]
mod timeline_audio_mix_tests {
    use super::*;
    use crate::video_jobs::tests::ffmpeg_reachable;

    const SAMPLE_RATE: f64 = 48_000.0;

    fn scratch(prefix: &str) -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(prefix)
            .tempdir()
            .expect("temp dir")
    }

    fn lavfi(inputs: &[&str], tail: &[&str], out: &Path) -> Vec<String> {
        let mut args = vec!["ffmpeg".to_owned(), "-y".to_owned()];
        for input in inputs {
            args.push("-f".to_owned());
            args.push("lavfi".to_owned());
            args.push("-i".to_owned());
            args.push((*input).to_owned());
        }
        args.extend(tail.iter().map(|arg| (*arg).to_owned()));
        args.push(out.display().to_string());
        args
    }

    /// A picture clip whose OWN audio is a tone at `tone_hz` — the "generated clip audio" every
    /// doubling assertion below is about.
    async fn picture_clip(out: &Path, seconds: f64, tone_hz: u32) -> bool {
        run_ffmpeg(
            lavfi(
                &[
                    &format!("testsrc2=size=320x180:rate=24:duration={seconds}"),
                    &format!("sine=frequency={tone_hz}:duration={seconds}:sample_rate=48000"),
                ],
                &["-shortest", "-pix_fmt", "yuv420p", "-c:a", "aac"],
                out,
            ),
            None,
        )
        .await
        .is_ok()
    }

    /// A picture clip with no audio stream at all, for the drop-rather-than-fail path.
    async fn silent_picture_clip(out: &Path, seconds: f64) -> bool {
        run_ffmpeg(
            lavfi(
                &[&format!("testsrc2=size=320x180:rate=24:duration={seconds}")],
                &["-pix_fmt", "yuv420p", "-an"],
                out,
            ),
            None,
        )
        .await
        .is_ok()
    }

    async fn tone_wav(out: &Path, seconds: f64, tone_hz: u32) -> bool {
        run_ffmpeg(
            lavfi(
                &[&format!(
                    "sine=frequency={tone_hz}:duration={seconds}:sample_rate=48000"
                )],
                &["-c:a", "pcm_s16le"],
                out,
            ),
            None,
        )
        .await
        .is_ok()
    }

    /// A bed whose instantaneous frequency rises as `200 + 150t` Hz.
    ///
    /// This is the shape that can tell "one continuous bed" apart from "the same bed restarted at
    /// every cut", which a constant tone cannot: a restart puts the STARTING frequency back on the
    /// other side of the cut, and the probe below reads the frequency rather than the level.
    async fn chirp_wav(out: &Path, seconds: f64) -> bool {
        run_ffmpeg(
            lavfi(
                &[&format!(
                    "aevalsrc=exprs=sin(2*PI*(200*t+75*t*t)):d={seconds}:s=48000"
                )],
                &["-c:a", "pcm_s16le"],
                out,
            ),
            None,
        )
        .await
        .is_ok()
    }

    /// Decode `[start, start + seconds)` of `path` to mono 48 kHz PCM.
    fn decode_window(path: &Path, start: f64, seconds: f64) -> Vec<f64> {
        let program = resolve_probe_program("ffmpeg");
        let output = std::process::Command::new(program)
            .args([
                "-v",
                "error",
                "-ss",
                &format!("{start:.3}"),
                "-i",
                &path.display().to_string(),
                "-t",
                &format!("{seconds:.3}"),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-f",
                "s16le",
                "-",
            ])
            .output()
            .expect("ffmpeg decodes the export");
        output
            .stdout
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f64 / 32768.0)
            .collect()
    }

    fn rms(samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|value| value * value).sum::<f64>() / samples.len() as f64).sqrt()
    }

    /// Peak sample of the whole export, decoded at the file's OWN channel count.
    ///
    /// Deliberately NOT [`decode_window`]'s `-ac 1`: swresample normalises the downmix matrix when
    /// the output format is an integer one, so a mix that is pinned at ±1.0 in stereo comes back
    /// from a mono decode politely scaled down and the clipping is invisible. Measured both ways on
    /// the same unlimited two-bus export: peak 1.000 with 360 clipped samples at `-ac 2`, peak
    /// 0.959 and not one clipped sample at `-ac 1`. A headroom assertion has to ask the first
    /// question, not the second.
    fn decoded_peak(path: &Path) -> f64 {
        let program = resolve_probe_program("ffmpeg");
        let output = std::process::Command::new(program)
            .args([
                "-v",
                "error",
                "-i",
                &path.display().to_string(),
                "-vn",
                "-f",
                "s16le",
                "-",
            ])
            .output()
            .expect("ffmpeg decodes the export");
        output
            .stdout
            .chunks_exact(2)
            .map(|pair| (i16::from_le_bytes([pair[0], pair[1]]) as f64 / 32768.0).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Normalised magnitude of `frequency` in `samples` — a single-bin DFT (Goertzel).
    fn tone_level(samples: &[f64], frequency: f64) -> f64 {
        let count = samples.len();
        if count < 64 {
            return 0.0;
        }
        let bin = (0.5 + count as f64 * frequency / SAMPLE_RATE).floor();
        let omega = 2.0 * std::f64::consts::PI * bin / count as f64;
        let coefficient = 2.0 * omega.cos();
        let (mut previous, mut older) = (0.0_f64, 0.0_f64);
        for sample in samples {
            let current = sample + coefficient * previous - older;
            older = previous;
            previous = current;
        }
        (previous * previous + older * older - coefficient * previous * older)
            .max(0.0)
            .sqrt()
            / (count as f64 / 2.0)
    }

    /// The strongest frequency in `samples`, searched on a 5 Hz grid.
    fn dominant_frequency(samples: &[f64], low: u32, high: u32) -> f64 {
        (low..=high)
            .step_by(5)
            .map(|frequency| (frequency as f64, tone_level(samples, frequency as f64)))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(frequency, _)| frequency)
            .unwrap_or(0.0)
    }

    fn track(id: &str, kind: &str, role: &str, muted: bool, gain: f64, items: Value) -> Value {
        json!({
            "id": id,
            "name": id,
            "kind": kind,
            "role": role,
            "locked": false,
            "muted": muted,
            "gain": gain,
            "items": items,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn item(
        id: &str,
        track_id: &str,
        asset_id: &str,
        kind: &str,
        source_in: f64,
        source_out: f64,
        start: f64,
        end: f64,
        extra: Value,
    ) -> Value {
        let mut value = json!({
            "id": id,
            "trackId": track_id,
            "assetId": asset_id,
            "type": kind,
            "displayName": id,
            "sourceIn": source_in,
            "sourceOut": source_out,
            "timelineStart": start,
            "timelineEnd": end,
            "speed": 1.0,
            "fit": "fit",
            "volume": 1.0,
            "generatedAudio": "mute",
            "fadeInSeconds": 0.0,
            "fadeOutSeconds": 0.0,
        });
        let object = value.as_object_mut().expect("item object");
        for (key, entry) in extra.as_object().expect("extra object") {
            object.insert(key.clone(), entry.clone());
        }
        value
    }

    fn asset(kind: &str, path: &str, mime: &str) -> Value {
        json!({"type": kind, "file": {"path": path, "mimeType": mime}})
    }

    /// Run the production export sequence over a hand-built timeline: plan the main track, render
    /// each segment, mux the picture, then mix the audio. Returns the exported file and the
    /// PICTURE's duration — the same number `finalize` writes into the sidecar, which is not the
    /// plan's span once a crossfade absorbs time out of it.
    ///
    /// This is the real path minus the store and the API — `plan_segments`, `render_item_segment`,
    /// `PictureTiming::from_segments`, `audio_placements`, the `source_has_audio_stream` filter
    /// `resolve_audio_sources` applies, `mux_segments` and `mux_audio` are the same functions
    /// `TimelineExport` calls, in the same order, with the same arguments.
    async fn export(
        dir: &Path,
        timeline: &Value,
        assets: &[(&str, Value)],
        spec: RenderSpec,
    ) -> (PathBuf, f64) {
        let mut items = main_track_items(timeline);
        items.sort_by(|left, right| {
            item_f64(left, "timelineStart", 0.0).total_cmp(&item_f64(right, "timelineStart", 0.0))
        });
        let (plan, _) = plan_segments(&items).expect("main track plans");

        let mut segments = Vec::new();
        for (index, planned) in plan.iter().enumerate() {
            if let Some(gap) = planned.leading_gap {
                let gap_path = dir.join(format!("segment_{index:04}_gap.mp4"));
                render_black_segment("ffmpeg", &gap_path, gap, spec, None)
                    .await
                    .expect("gap renders");
                segments.push(TimelineSegment {
                    path: gap_path,
                    duration: gap,
                    transition: None,
                    transition_duration: 0.0,
                });
            }
            let asset_id = required_value_str(planned.item, "assetId").expect("assetId");
            let asset = assets
                .iter()
                .find(|(id, _)| *id == asset_id)
                .map(|(_, value)| value.clone())
                .expect("asset is in the fixture set");
            let segment_path = dir.join(format!("segment_{index:04}.mp4"));
            let segment_duration = render_item_segment(
                "ffmpeg",
                dir,
                planned.item,
                &asset,
                &segment_path,
                spec,
                None,
            )
            .await
            .expect("segment renders");
            segments.push(TimelineSegment {
                path: segment_path,
                duration: segment_duration,
                transition: planned.transition.clone(),
                transition_duration: planned.transition_duration,
            });
        }

        let timing = PictureTiming::from_segments(&segments);
        let duration = timing.duration();
        let mut sources: Vec<ResolvedAudioSource> = Vec::new();
        for placement in audio_placements(timeline, &timing) {
            let asset = assets
                .iter()
                .find(|(id, _)| *id == placement.asset_id)
                .map(|(_, value)| value.clone())
                .expect("audio asset is in the fixture set");
            let relative = asset["file"]["path"]
                .as_str()
                .expect("asset path")
                .to_owned();
            let media_path = dir.join(relative);
            // The same drop-rather-than-fail filter `resolve_audio_sources` applies. Without it
            // these tests would never reach the guard, and one silent opted-in take fails the
            // WHOLE ffmpeg command — `[N:a]` against a file with no audio stream does not degrade.
            if !source_has_audio_stream("ffmpeg", &media_path).await {
                continue;
            }
            sources.push(ResolvedAudioSource {
                placement,
                media_path,
            });
        }

        let output = dir.join("export.mp4");
        if sources.is_empty() {
            mux_segments("ffmpeg", &segments, dir, &output, None)
                .await
                .expect("picture muxes");
        } else {
            let picture = dir.join("picture.mp4");
            mux_segments("ffmpeg", &segments, dir, &picture, None)
                .await
                .expect("picture muxes");
            mux_audio("ffmpeg", &picture, &sources, &output, None)
                .await
                .expect("audio mixes");
        }
        (output, duration)
    }

    // -----------------------------------------------------------------------------------------
    // Pure planning: which items reach the mix at all, and with what numbers.
    // -----------------------------------------------------------------------------------------

    /// The doubling guard, stated as a selection rule before any ffmpeg runs.
    ///
    /// A picture item is in the mix only when it says `generatedAudio: "include"`. The default the
    /// store writes is `"mute"`, so a shot that has both a generated take with a spoken line AND a
    /// placed dialogue clip contributes exactly ONE of them unless the timeline explicitly asks
    /// for both — which is the whole content of the policy.
    #[test]
    fn generated_picture_audio_joins_the_mix_only_when_the_item_opts_in() {
        let timeline = json!({
            "aspectRatio": "16:9",
            "tracks": [
                track("track_main", "video", "picture", false, 1.0, json!([
                    item("item_a", "track_main", "asset_a", "video", 0.0, 2.0, 0.0, 2.0, json!({})),
                    item("item_b", "track_main", "asset_b", "video", 0.0, 2.0, 2.0, 4.0,
                         json!({"generatedAudio": "include"})),
                ])),
                track("track_dialogue", "audio", "dialogue", false, 1.0, json!([
                    item("item_d", "track_dialogue", "asset_d", "audio", 0.0, 1.0, 0.5, 1.5, json!({})),
                ])),
            ],
        });
        let placements = audio_placements(&timeline, &PictureTiming::flat(4.0));
        let selected: Vec<(&str, bool)> = placements
            .iter()
            .map(|placement| (placement.asset_id.as_str(), placement.generated))
            .collect();
        assert_eq!(
            selected,
            vec![("asset_d", false), ("asset_b", true)],
            "the muted-by-default picture item must stay OUT of the mix while the one that opted \
             in joins it; the dialogue clip is always in. Placements: {placements:#?}"
        );
    }

    /// A muted track contributes nothing, and gains multiply track by item.
    #[test]
    fn track_mute_removes_a_bus_and_track_gain_multiplies_item_volume() {
        let timeline = json!({
            "tracks": [
                track("track_music", "audio", "music", true, 1.0, json!([
                    item("item_m", "track_music", "asset_m", "audio", 0.0, 4.0, 0.0, 4.0, json!({})),
                ])),
                track("track_ambience", "audio", "ambience", false, 0.5, json!([
                    item("item_x", "track_ambience", "asset_x", "audio", 0.0, 4.0, 0.0, 4.0,
                         json!({"volume": 0.5})),
                ])),
            ],
        });
        let placements = audio_placements(&timeline, &PictureTiming::flat(4.0));
        assert_eq!(placements.len(), 1, "the muted music bus must be dropped");
        assert_eq!(placements[0].asset_id, "asset_x");
        assert!(
            (placements[0].gain - 0.25).abs() < 1e-9,
            "track gain 0.5 times item volume 0.5 is 0.25, got {}",
            placements[0].gain
        );
    }

    /// The picture is the ceiling: the mix can never make the file longer than the timeline says.
    #[test]
    fn placements_are_clamped_to_the_picture_duration() {
        let timeline = json!({
            "tracks": [
                track("track_ambience", "audio", "ambience", false, 1.0, json!([
                    item("item_over", "track_ambience", "asset_o", "audio", 0.0, 30.0, 0.0, 30.0, json!({})),
                    item("item_past", "track_ambience", "asset_p", "audio", 0.0, 2.0, 9.0, 11.0, json!({})),
                ])),
            ],
        });
        let placements = audio_placements(&timeline, &PictureTiming::flat(4.0));
        assert_eq!(
            placements.len(),
            1,
            "a clip that starts after the last frame has no slot at all: {placements:#?}"
        );
        assert!((placements[0].span - 4.0).abs() < 1e-9, "{placements:#?}");
    }

    /// An overlay item is never drawn into the picture, so its take's audio has nothing to sit
    /// behind — `generatedAudio: include` on one must not put sound into the export.
    ///
    /// `main_track_items` reads ONE track, so the picture is only ever the main track's items; the
    /// generated branch has to be keyed on that same track rather than on "not an audio track".
    #[test]
    fn an_overlay_items_generated_audio_never_reaches_the_mix() {
        let timeline = json!({
            "tracks": [
                track("track_main", "video", "picture", false, 1.0, json!([
                    item("item_a", "track_main", "asset_a", "video", 0.0, 2.0, 0.0, 2.0,
                         json!({"generatedAudio": "include"})),
                ])),
                track("track_overlay", "overlay", "overlay", false, 1.0, json!([
                    item("item_o", "track_overlay", "asset_o", "video", 0.0, 2.0, 0.0, 2.0,
                         json!({"generatedAudio": "include"})),
                ])),
            ],
        });
        let placements = audio_placements(&timeline, &PictureTiming::flat(4.0));
        let selected: Vec<&str> = placements
            .iter()
            .map(|placement| placement.asset_id.as_str())
            .collect();
        assert_eq!(
            selected,
            vec!["asset_a"],
            "only the PICTURE track's opted-in item may contribute its own audio; the overlay \
             item's take is never rendered into the picture"
        );
    }

    /// The picture clock is the crossfade graph's own arithmetic, not the plan's span.
    ///
    /// `crossfade_filter_complex` overlaps each crossfaded segment with the one before it
    /// (`current_duration += segment.duration - duration`), so two 2 s shots with a 1 s crossfade
    /// are a 3 s picture and everything in the second shot happens a second earlier than the
    /// timeline says. Both halves are read here: the length, and the mapping.
    #[test]
    fn picture_timing_matches_the_crossfade_graphs_own_arithmetic() {
        let segment = |duration: f64, transition: Option<&str>| TimelineSegment {
            path: PathBuf::from("segment.mp4"),
            duration,
            transition: transition.map(str::to_owned),
            transition_duration: 1.0,
        };
        let plain = PictureTiming::from_segments(&[segment(2.0, None), segment(2.0, Some("cut"))]);
        assert!(
            (plain.duration() - 4.0).abs() < 1e-9 && (plain.picture_time(2.0) - 2.0).abs() < 1e-9,
            "a straight cut absorbs nothing: {plain:?}"
        );

        let faded =
            PictureTiming::from_segments(&[segment(2.0, None), segment(2.0, Some("crossfade"))]);
        assert!(
            (faded.duration() - 3.0).abs() < 1e-9,
            "two 2s shots with a 1s crossfade are a 3s picture (measured with ffmpeg 9.0.1 in \
             `a_crossfaded_export_places_sound_against_the_picture_it_actually_produced`); this \
             timing says {}",
            faded.duration()
        );
        for (timeline_second, expected) in [(0.0, 0.0), (1.5, 1.5), (2.0, 1.0), (4.0, 3.0)] {
            let measured = faded.picture_time(timeline_second);
            assert!(
                (measured - expected).abs() < 1e-9,
                "timeline {timeline_second}s lands at {expected}s in the picture, not {measured}s"
            );
        }

        // Two crossfades accumulate, and the shortening applies to everything after each one.
        let twice = PictureTiming::from_segments(&[
            segment(2.0, None),
            segment(2.0, Some("crossfade")),
            segment(2.0, Some("crossfade")),
        ]);
        assert!(
            (twice.duration() - 4.0).abs() < 1e-9
                && (twice.picture_time(4.0) - 2.0).abs() < 1e-9
                && (twice.picture_time(2.0) - 1.0).abs() < 1e-9,
            "{twice:?}"
        );
    }

    /// `atempo` is specified for 0.5..=2.0 per instance; the timeline admits 0.1..=8.0.
    #[test]
    fn atempo_chains_cover_the_speed_range_the_timeline_admits() {
        assert!(atempo_chain(1.0).is_empty(), "unit speed adds no filters");
        for speed in [0.1_f64, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 8.0] {
            let chain = atempo_chain(speed);
            let product: f64 = chain
                .iter()
                .map(|filter| {
                    filter
                        .trim_start_matches("atempo=")
                        .parse::<f64>()
                        .expect("atempo factor parses")
                })
                .product();
            assert!(
                (product - speed).abs() < 1e-4,
                "atempo chain for {speed} multiplies to {product}: {chain:?}"
            );
            assert!(
                chain.iter().all(|filter| {
                    let factor: f64 = filter
                        .trim_start_matches("atempo=")
                        .parse()
                        .expect("factor");
                    (0.5..=2.0).contains(&factor)
                }),
                "every atempo factor must stay inside 0.5..=2.0: {chain:?}"
            );
        }
    }

    /// `amix` renormalises by input count unless told not to, which would make every gain in the
    /// timeline a lie the moment a second bus appeared.
    #[test]
    fn the_mix_never_renormalises_by_input_count() {
        let sources: Vec<ResolvedAudioSource> = ["a", "b"]
            .iter()
            .map(|id| ResolvedAudioSource {
                placement: AudioPlacement {
                    asset_id: (*id).to_owned(),
                    track_id: "track_dialogue".to_owned(),
                    role: "dialogue".to_owned(),
                    source_in: 0.0,
                    source_out: 1.0,
                    picture_start: 0.0,
                    span: 1.0,
                    speed: 1.0,
                    gain: 1.0,
                    fade_in: 0.0,
                    fade_out: 0.0,
                    generated: false,
                },
                media_path: PathBuf::from(format!("{id}.wav")),
            })
            .collect();
        let filter = audio_mix_filter(&sources);
        assert!(
            filter.contains("amix=inputs=2:normalize=0"),
            "the mix must not renormalise: {filter}"
        );
        assert!(
            filter.ends_with("apad[aout]"),
            "the mix must be padded so -shortest cuts on the PICTURE, not on the audio: {filter}"
        );
    }

    /// The same inheritance hazard `a_crossfaded_export_does_not_inherit_a_clips_recipe` measured,
    /// on the newer multi-input command (sc-15956 / sc-22712).
    #[test]
    fn an_audio_mix_does_not_inherit_a_clips_recipe() {
        let sources = vec![ResolvedAudioSource {
            placement: AudioPlacement {
                asset_id: "a".to_owned(),
                track_id: "track_music".to_owned(),
                role: "music".to_owned(),
                source_in: 0.0,
                source_out: 1.0,
                picture_start: 0.0,
                span: 1.0,
                speed: 1.0,
                gain: 1.0,
                fade_in: 0.0,
                fade_out: 0.0,
                generated: false,
            },
            media_path: PathBuf::from("a.wav"),
        }];
        let args = audio_mix_args(
            "ffmpeg",
            Path::new("picture.mp4"),
            &sources,
            Path::new("out.mp4"),
        )
        .expect("one source mixes");
        let metadata = args
            .iter()
            .position(|arg| arg == "-map_metadata")
            .and_then(|at| args.get(at + 1))
            .map(String::as_str);
        assert_eq!(
            metadata,
            Some("-1"),
            "the audio mix takes the picture as input 0 and every clip after it, so ffmpeg's \
             multi-input default would republish the picture's metadata as the export's own. \
             Args: {args:?}"
        );
        assert!(
            args.windows(2).any(|pair| pair == ["-c:v", "copy"]),
            "the picture is already exactly the frames the timeline describes; re-encoding it \
             costs quality for nothing. Args: {args:?}"
        );
    }

    // -----------------------------------------------------------------------------------------
    // Measured: what the exported MP4 actually sounds like.
    // -----------------------------------------------------------------------------------------

    /// The story's whole second acceptance criterion, measured in one export.
    ///
    /// Three buses over a two-shot cut: a dialogue line inside shot 1 only, an ambience bed at a
    /// quarter gain spanning both shots, and a muted music bus. The picture clips carry their own
    /// 900 Hz generated audio and do NOT opt in.
    #[tokio::test]
    async fn each_bus_lands_where_the_timeline_says_at_the_gain_it_says() {
        if !ffmpeg_reachable() {
            return;
        }
        let dir = scratch("sceneworks_mix_buses_");
        let root = dir.path();
        assert!(picture_clip(&root.join("shot_a.mp4"), 2.0, 900).await);
        assert!(picture_clip(&root.join("shot_b.mp4"), 2.0, 900).await);
        assert!(tone_wav(&root.join("line.wav"), 1.0, 440).await);
        assert!(tone_wav(&root.join("room.wav"), 4.0, 300).await);
        assert!(tone_wav(&root.join("theme.wav"), 4.0, 1200).await);

        let timeline = json!({
            "aspectRatio": "16:9",
            "tracks": [
                track("track_main", "video", "picture", false, 1.0, json!([
                    item("item_a", "track_main", "shot_a", "video", 0.0, 2.0, 0.0, 2.0, json!({})),
                    item("item_b", "track_main", "shot_b", "video", 0.0, 2.0, 2.0, 4.0, json!({})),
                ])),
                track("track_dialogue", "audio", "dialogue", false, 1.0, json!([
                    item("item_line", "track_dialogue", "line", "audio", 0.0, 1.0, 0.5, 1.5, json!({})),
                ])),
                track("track_ambience", "audio", "ambience", false, 0.25, json!([
                    item("item_room", "track_ambience", "room", "audio", 0.0, 4.0, 0.0, 4.0, json!({})),
                ])),
                track("track_music", "audio", "music", true, 1.0, json!([
                    item("item_theme", "track_music", "theme", "audio", 0.0, 4.0, 0.0, 4.0, json!({})),
                ])),
            ],
        });
        let assets = [
            ("shot_a", asset("video", "shot_a.mp4", "video/mp4")),
            ("shot_b", asset("video", "shot_b.mp4", "video/mp4")),
            ("line", asset("audio", "line.wav", "audio/wav")),
            ("room", asset("audio", "room.wav", "audio/wav")),
            ("theme", asset("audio", "theme.wav", "audio/wav")),
        ];
        let spec = RenderSpec {
            width: 320,
            height: 180,
            fps: 24,
        };
        let (export_path, duration) = export(root, &timeline, &assets, spec).await;
        assert!((duration - 4.0).abs() < 1e-6, "planned duration {duration}");

        let inside = decode_window(&export_path, 0.6, 0.3);
        let outside = decode_window(&export_path, 2.5, 0.3);
        assert!(
            !inside.is_empty() && !outside.is_empty(),
            "the export must carry a decodable audio stream"
        );

        let line_inside = tone_level(&inside, 440.0);
        let line_outside = tone_level(&outside, 440.0);
        assert!(
            line_inside > 0.02,
            "the dialogue line must be audible in 0.5..1.5s, measured {line_inside:.5}"
        );
        assert!(
            line_outside < line_inside / 10.0,
            "the dialogue line must be GONE after its clip ends: {line_outside:.5} inside the \
             window was {line_inside:.5}"
        );

        // The muted music bus and the opted-out generated audio are both absent everywhere.
        for (label, frequency) in [("muted music bus", 1200.0), ("generated clip audio", 900.0)] {
            for (window_label, window) in [("during", &inside), ("after", &outside)] {
                let level = tone_level(window, frequency);
                assert!(
                    level < 0.01,
                    "{label} must be silent ({window_label} window), measured {level:.5}"
                );
            }
        }

        // The ambience bed is present on both sides of the cut at 2.0s, at the same level.
        let before_cut = decode_window(&export_path, 1.7, 0.25);
        let after_cut = decode_window(&export_path, 2.1, 0.25);
        let ambience_before = tone_level(&before_cut, 300.0);
        let ambience_after = tone_level(&after_cut, 300.0);
        assert!(
            ambience_before > 0.02 && ambience_after > 0.02,
            "the ambience bed must span the cut: {ambience_before:.5} before, \
             {ambience_after:.5} after"
        );
        assert!(
            (ambience_before - ambience_after).abs() < ambience_before * 0.35,
            "the bed's level must not step at the cut: {ambience_before:.5} -> {ambience_after:.5}"
        );

        // The window with dialogue over the bed is louder than the bed alone.
        assert!(
            rms(&inside) > rms(&outside),
            "the window with dialogue over the bed must be louder than the bed alone: {:.5} vs \
             {:.5}",
            rms(&inside),
            rms(&outside)
        );
    }

    /// A continuous bed is CONTINUOUS, not re-triggered per shot.
    ///
    /// The bed's frequency rises with time, so the probe reads the bed's own clock: if the export
    /// had restarted it at the cut, the frequency just after 2.0s would be the STARTING frequency
    /// again instead of the one two seconds in.
    #[tokio::test]
    async fn a_bed_placed_once_does_not_restart_at_a_cut() {
        if !ffmpeg_reachable() {
            return;
        }
        let dir = scratch("sceneworks_mix_bed_");
        let root = dir.path();
        assert!(picture_clip(&root.join("shot_a.mp4"), 2.0, 900).await);
        assert!(picture_clip(&root.join("shot_b.mp4"), 2.0, 900).await);
        assert!(chirp_wav(&root.join("bed.wav"), 4.0).await);

        let timeline = json!({
            "aspectRatio": "16:9",
            "tracks": [
                track("track_main", "video", "picture", false, 1.0, json!([
                    item("item_a", "track_main", "shot_a", "video", 0.0, 2.0, 0.0, 2.0, json!({})),
                    item("item_b", "track_main", "shot_b", "video", 0.0, 2.0, 2.0, 4.0, json!({})),
                ])),
                track("track_ambience", "audio", "ambience", false, 1.0, json!([
                    item("item_bed", "track_ambience", "bed", "audio", 0.0, 4.0, 0.0, 4.0, json!({})),
                ])),
            ],
        });
        let assets = [
            ("shot_a", asset("video", "shot_a.mp4", "video/mp4")),
            ("shot_b", asset("video", "shot_b.mp4", "video/mp4")),
            ("bed", asset("audio", "bed.wav", "audio/wav")),
        ];
        let (export_path, _) = export(
            root,
            &timeline,
            &assets,
            RenderSpec {
                width: 320,
                height: 180,
                fps: 24,
            },
        )
        .await;

        // Instantaneous frequency of the fixture is 200 + 150t Hz; probe the middle of each window.
        for (start, expected) in [(0.1_f64, 230.0_f64), (2.1, 530.0), (3.5, 740.0)] {
            let window = decode_window(&export_path, start, 0.2);
            let dominant = dominant_frequency(&window, 150, 900);
            assert!(
                (dominant - expected).abs() <= 25.0,
                "at {start}s the bed should be near {expected} Hz (it is one continuous take, not \
                 one restarted at each cut); measured {dominant} Hz"
            );
        }
    }

    /// The picture's length is the file's length, with or without a mix, and the mix does not
    /// nudge it either way.
    #[tokio::test]
    async fn the_exported_duration_matches_the_timeline_with_and_without_sound() {
        if !ffmpeg_reachable() {
            return;
        }
        let dir = scratch("sceneworks_mix_duration_");
        let root = dir.path();
        assert!(picture_clip(&root.join("shot_a.mp4"), 3.0, 900).await);
        // A bed deliberately LONGER than the picture: it must not extend the file.
        assert!(tone_wav(&root.join("room.wav"), 10.0, 300).await);

        let picture_only = json!({
            "aspectRatio": "16:9",
            "tracks": [
                track("track_main", "video", "picture", false, 1.0, json!([
                    item("item_a", "track_main", "shot_a", "video", 0.5, 3.0, 0.0, 2.5, json!({})),
                ])),
            ],
        });
        let mut with_sound = picture_only.clone();
        with_sound["tracks"]
            .as_array_mut()
            .expect("tracks")
            .push(track(
                "track_ambience",
                "audio",
                "ambience",
                false,
                0.5,
                json!([item(
                    "item_room",
                    "track_ambience",
                    "room",
                    "audio",
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                    json!({})
                )]),
            ));
        let assets = [
            ("shot_a", asset("video", "shot_a.mp4", "video/mp4")),
            ("room", asset("audio", "room.wav", "audio/wav")),
        ];
        let spec = RenderSpec {
            width: 320,
            height: 180,
            fps: 24,
        };

        let silent_dir = root.join("silent");
        std::fs::create_dir_all(&silent_dir).expect("scratch dir");
        for name in ["shot_a.mp4", "room.wav"] {
            std::fs::copy(root.join(name), silent_dir.join(name)).expect("fixture copy");
        }
        let (silent_export, silent_duration) =
            export(&silent_dir, &picture_only, &assets, spec).await;
        let (mixed_export, mixed_duration) = export(root, &with_sound, &assets, spec).await;

        assert!((silent_duration - 2.5).abs() < 1e-6);
        assert!((mixed_duration - 2.5).abs() < 1e-6);

        let measured_silent = probe_source_duration("ffmpeg", &silent_export)
            .await
            .expect("silent export has a duration");
        let measured_mixed = probe_source_duration("ffmpeg", &mixed_export)
            .await
            .expect("mixed export has a duration");
        // One frame at 24 fps is 41.7 ms; allow two, which is the container's own rounding.
        assert!(
            (measured_silent - 2.5).abs() < 0.09,
            "picture-only export measured {measured_silent}s against a 2.5s timeline"
        );
        assert!(
            (measured_mixed - 2.5).abs() < 0.09,
            "a 10s bed must not extend a 2.5s timeline; measured {measured_mixed}s"
        );

        // And the sound really is in there, up to the very end.
        let tail = decode_window(&mixed_export, 2.2, 0.25);
        assert!(
            tone_level(&tail, 300.0) > 0.02,
            "the bed must still be playing at the last cut point"
        );
    }

    /// Both halves of the doubling policy, measured: `include` really does bring the generated
    /// line in, and it mixes ALONGSIDE the placed dialogue only because the timeline asked.
    #[tokio::test]
    async fn generated_audio_doubles_with_dialogue_only_when_explicitly_included() {
        if !ffmpeg_reachable() {
            return;
        }
        let dir = scratch("sceneworks_mix_double_");
        let root = dir.path();
        assert!(picture_clip(&root.join("shot_a.mp4"), 2.0, 900).await);
        assert!(tone_wav(&root.join("line.wav"), 2.0, 440).await);
        let assets = [
            ("shot_a", asset("video", "shot_a.mp4", "video/mp4")),
            ("line", asset("audio", "line.wav", "audio/wav")),
        ];
        let spec = RenderSpec {
            width: 320,
            height: 180,
            fps: 24,
        };

        let timeline_for = |policy: &str| {
            json!({
                "aspectRatio": "16:9",
                "tracks": [
                    track("track_main", "video", "picture", false, 1.0, json!([
                        item("item_a", "track_main", "shot_a", "video", 0.0, 2.0, 0.0, 2.0,
                             json!({"generatedAudio": policy})),
                    ])),
                    track("track_dialogue", "audio", "dialogue", false, 1.0, json!([
                        item("item_line", "track_dialogue", "line", "audio", 0.0, 2.0, 0.0, 2.0, json!({})),
                    ])),
                ],
            })
        };

        for (policy, subdir) in [("mute", "muted"), ("include", "included")] {
            let case_dir = root.join(subdir);
            std::fs::create_dir_all(&case_dir).expect("scratch dir");
            for name in ["shot_a.mp4", "line.wav"] {
                std::fs::copy(root.join(name), case_dir.join(name)).expect("fixture copy");
            }
            let (export_path, _) = export(&case_dir, &timeline_for(policy), &assets, spec).await;
            let window = decode_window(&export_path, 0.5, 0.5);
            let dialogue = tone_level(&window, 440.0);
            let generated = tone_level(&window, 900.0);
            assert!(
                dialogue > 0.02,
                "the placed dialogue clip is always in the mix ({policy}), measured {dialogue:.5}"
            );
            if policy == "mute" {
                assert!(
                    generated < 0.01,
                    "a shot with BOTH a generated take and a placed dialogue clip must not mix \
                     both by default — that is the doubling this policy exists to prevent. \
                     Generated tone measured {generated:.5}"
                );
            } else {
                assert!(
                    generated > 0.02,
                    "`generatedAudio: include` is an explicit request for the take's own audio; \
                     measured {generated:.5}"
                );
            }
        }
    }

    /// The defect the sc-22710 GPU smoke measured, stated as a test.
    ///
    /// Both MiniMax-H3 takes in that run carried a real AAC track (H3 renders `t2va`/`fl2va`:
    /// video AND audio), and the exported MP4 came out with ONE stream. The picture pass renders
    /// every segment `-an` and the concat mux copies what it is given, so a take's audio could
    /// never survive no matter what the timeline said.
    ///
    /// This is the narrowest statement of the fix: one take with an AAC track, no placed sound at
    /// all, and the ONLY thing that differs between the two exports is the policy on the item.
    /// Under `mute` the file has no audio stream — not a silent one, none, exactly as before —
    /// and under `include` the take's own tone is audible in the export.
    #[tokio::test]
    async fn a_take_with_an_aac_track_exports_with_audio_only_when_the_policy_includes_it() {
        if !ffmpeg_reachable() {
            return;
        }
        let dir = scratch("sceneworks_mix_aac_");
        let root = dir.path();
        assert!(picture_clip(&root.join("take.mp4"), 2.0, 900).await);
        assert!(
            source_has_audio_stream("ffmpeg", &root.join("take.mp4")).await,
            "the fixture take must carry an audio track, like a real H3 render"
        );
        let assets = [("take", asset("video", "take.mp4", "video/mp4"))];
        let spec = RenderSpec {
            width: 320,
            height: 180,
            fps: 24,
        };
        let timeline_for = |policy: &str| {
            json!({
                "aspectRatio": "16:9",
                "tracks": [
                    track("track_main", "video", "picture", false, 1.0, json!([
                        item("item_a", "track_main", "take", "video", 0.0, 2.0, 0.0, 2.0,
                             json!({"generatedAudio": policy})),
                    ])),
                ],
            })
        };

        for name in ["muted", "included"] {
            let case_dir = root.join(name);
            std::fs::create_dir_all(&case_dir).expect("scratch dir");
            std::fs::copy(root.join("take.mp4"), case_dir.join("take.mp4")).expect("fixture copy");
        }

        let (muted, _) = export(&root.join("muted"), &timeline_for("mute"), &assets, spec).await;
        assert!(
            !source_has_audio_stream("ffmpeg", &muted).await,
            "with the policy at `mute` the export carries no audio stream at all — the picture \
             pass is untouched and there is nothing to mix"
        );

        let (included, _) = export(
            &root.join("included"),
            &timeline_for("include"),
            &assets,
            spec,
        )
        .await;
        assert!(
            source_has_audio_stream("ffmpeg", &included).await,
            "`generatedAudio: include` must put the take's own AAC track into the export — this is \
             the sc-22710 smoke's video-only MP4"
        );
        let window = decode_window(&included, 0.5, 0.5);
        let level = tone_level(&window, 900.0);
        assert!(
            level > 0.02,
            "the take's own tone must be AUDIBLE, not merely present as an empty stream; measured \
             {level:.5}"
        );
    }

    /// A source with no audio stream is detected AND the export survives it (sc-22712).
    ///
    /// `[N:a]` against a file with no audio stream fails the WHOLE ffmpeg command, so a single
    /// silent take that opted in would otherwise take the entire export down with it. The predicate
    /// is read first, then the same silent take is put on a real two-shot timeline with
    /// `generatedAudio: include` — the export must still be produced, and must still carry the
    /// dialogue clip that had nothing to do with the silent take.
    #[tokio::test]
    async fn a_silent_source_is_detected_rather_than_failing_the_whole_export() {
        if !ffmpeg_reachable() {
            return;
        }
        let dir = scratch("sceneworks_mix_silent_");
        let root = dir.path();
        assert!(silent_picture_clip(&root.join("silent.mp4"), 1.0).await);
        assert!(picture_clip(&root.join("voiced.mp4"), 1.0, 900).await);
        assert!(
            !source_has_audio_stream("ffmpeg", &root.join("silent.mp4")).await,
            "a clip encoded with -an has no audio stream"
        );
        assert!(
            source_has_audio_stream("ffmpeg", &root.join("voiced.mp4")).await,
            "a clip with a sine track does"
        );

        assert!(tone_wav(&root.join("line.wav"), 1.0, 440).await);
        let timeline = json!({
            "aspectRatio": "16:9",
            "tracks": [
                track("track_main", "video", "picture", false, 1.0, json!([
                    // The opted-in take that turns out to have no audio at all.
                    item("item_silent", "track_main", "silent", "video", 0.0, 1.0, 0.0, 1.0,
                         json!({"generatedAudio": "include"})),
                    item("item_voiced", "track_main", "voiced", "video", 0.0, 1.0, 1.0, 2.0,
                         json!({"generatedAudio": "include"})),
                ])),
                track("track_dialogue", "audio", "dialogue", false, 1.0, json!([
                    item("item_line", "track_dialogue", "line", "audio", 0.0, 1.0, 0.0, 1.0, json!({})),
                ])),
            ],
        });
        let assets = [
            ("silent", asset("video", "silent.mp4", "video/mp4")),
            ("voiced", asset("video", "voiced.mp4", "video/mp4")),
            ("line", asset("audio", "line.wav", "audio/wav")),
        ];
        let (export_path, _) = export(
            root,
            &timeline,
            &assets,
            RenderSpec {
                width: 320,
                height: 180,
                fps: 24,
            },
        )
        .await;
        assert!(
            export_path.exists(),
            "one silent opted-in take must cost the export that layer, NOT the whole file"
        );
        let window = decode_window(&export_path, 0.2, 0.6);
        let line = tone_level(&window, 440.0);
        assert!(
            line > 0.02,
            "the layers that CAN be read must still be in the export; the dialogue clip measured \
             {line:.5}"
        );
        let voiced = tone_level(&decode_window(&export_path, 1.2, 0.6), 900.0);
        assert!(
            voiced > 0.02,
            "so must the second take's own audio, which is readable; measured {voiced:.5}"
        );
    }

    /// AC3 on a crossfaded sequence: sound is placed against the picture that was ACTUALLY built.
    ///
    /// `crossfade_filter_complex` overlaps the shots, so this 4 s timeline exports as a 3 s picture
    /// and the second shot begins at 1.0 s in the file rather than 2.0 s. A line keyed to that shot
    /// has to travel with it. Before sc-22712's fix the mix was handed the plan's 4.0: the line was
    /// `adelay`ed to 2.0 s — a full second after the shot it belongs to, and inside the last second
    /// that `-shortest` then cut off.
    #[tokio::test]
    async fn a_crossfaded_export_places_sound_against_the_picture_it_actually_produced() {
        if !ffmpeg_reachable() {
            return;
        }
        let dir = scratch("sceneworks_mix_crossfade_");
        let root = dir.path();
        assert!(picture_clip(&root.join("shot_a.mp4"), 2.0, 900).await);
        assert!(picture_clip(&root.join("shot_b.mp4"), 2.0, 900).await);
        assert!(tone_wav(&root.join("line.wav"), 1.0, 440).await);

        let timeline = json!({
            "aspectRatio": "16:9",
            "tracks": [
                track("track_main", "video", "picture", false, 1.0, json!([
                    item("item_a", "track_main", "shot_a", "video", 0.0, 2.0, 0.0, 2.0, json!({})),
                    item("item_b", "track_main", "shot_b", "video", 0.0, 2.0, 2.0, 4.0,
                         json!({"transitionIn": {"type": "crossfade", "duration": 1.0}})),
                ])),
                track("track_dialogue", "audio", "dialogue", false, 1.0, json!([
                    // Keyed to the top of the second shot.
                    item("item_line", "track_dialogue", "line", "audio", 0.0, 1.0, 2.0, 3.0, json!({})),
                ])),
            ],
        });
        let assets = [
            ("shot_a", asset("video", "shot_a.mp4", "video/mp4")),
            ("shot_b", asset("video", "shot_b.mp4", "video/mp4")),
            ("line", asset("audio", "line.wav", "audio/wav")),
        ];
        let (export_path, duration) = export(
            root,
            &timeline,
            &assets,
            RenderSpec {
                width: 320,
                height: 180,
                fps: 24,
            },
        )
        .await;
        assert!(
            (duration - 3.0).abs() < 1e-6,
            "the picture the export reports is the one the crossfade graph builds: 2 + 2 - 1 = 3, \
             got {duration}"
        );
        let measured = probe_source_duration("ffmpeg", &export_path)
            .await
            .expect("the crossfaded export has a duration");
        assert!(
            (measured - 3.0).abs() < 0.09,
            "two 2s shots with a 1.0s crossfade measure 3.000s, not the timeline's 4.0; measured \
             {measured}s"
        );

        let at_shot_b = tone_level(&decode_window(&export_path, 1.1, 0.6), 440.0);
        let at_timeline_start = tone_level(&decode_window(&export_path, 0.2, 0.6), 440.0);
        assert!(
            at_shot_b > 0.02,
            "the line must play where its SHOT is — 1.0s into the picture, not 2.0s; measured \
             {at_shot_b:.5} there"
        );
        assert!(
            at_timeline_start < at_shot_b / 10.0,
            "and nowhere else: {at_timeline_start:.5} at the head of the film against \
             {at_shot_b:.5} under its own shot"
        );
    }

    /// The mix has headroom: a deliberately hot two-bus sum does not reach the encoder's ceiling.
    ///
    /// `amix=normalize=0` is a straight sum — the thing that makes the timeline's gains mean
    /// something — so N loud buses can exceed full scale and the AAC encoder answers by clipping.
    /// Measured without [`MIX_LIMITER`], this mix round-trips with 180 samples pinned at ±1.0.
    /// Clipping distortion is not "their gain choices, audibly" (AC2).
    #[tokio::test]
    async fn a_hot_two_bus_mix_stays_below_full_scale() {
        if !ffmpeg_reachable() {
            return;
        }
        let dir = scratch("sceneworks_mix_hot_");
        let root = dir.path();
        assert!(picture_clip(&root.join("shot_a.mp4"), 2.0, 900).await);
        assert!(tone_wav(&root.join("line.wav"), 2.0, 440).await);
        assert!(tone_wav(&root.join("room.wav"), 2.0, 300).await);

        let timeline = json!({
            "aspectRatio": "16:9",
            "tracks": [
                track("track_main", "video", "picture", false, 1.0, json!([
                    item("item_a", "track_main", "shot_a", "video", 0.0, 2.0, 0.0, 2.0, json!({})),
                ])),
                // Gain 4.0 x volume 2.0 = 8, and 4.0 x 1.5 = 6: the top of what the timeline
                // admits on two buses at once.
                track("track_dialogue", "audio", "dialogue", false, 4.0, json!([
                    item("item_line", "track_dialogue", "line", "audio", 0.0, 2.0, 0.0, 2.0,
                         json!({"volume": 2.0})),
                ])),
                track("track_ambience", "audio", "ambience", false, 4.0, json!([
                    item("item_room", "track_ambience", "room", "audio", 0.0, 2.0, 0.0, 2.0,
                         json!({"volume": 1.5})),
                ])),
            ],
        });
        let assets = [
            ("shot_a", asset("video", "shot_a.mp4", "video/mp4")),
            ("line", asset("audio", "line.wav", "audio/wav")),
            ("room", asset("audio", "room.wav", "audio/wav")),
        ];
        let (export_path, _) = export(
            root,
            &timeline,
            &assets,
            RenderSpec {
                width: 320,
                height: 180,
                fps: 24,
            },
        )
        .await;

        let samples = decode_window(&export_path, 0.2, 1.5);
        assert!(!samples.is_empty(), "the hot mix must decode");
        let peak = decoded_peak(&export_path);
        assert!(
            peak < 0.999,
            "the summed mix must be limited BEFORE the encoder sees it — these two buses sum to \
             1.24 and an unlimited export decodes with 360 samples pinned at ±1.0. Peak measured \
             {peak:.5}"
        );
        // The limiter is a gain on the whole mix, so the louder bus is still the louder one.
        let line = tone_level(&samples, 440.0);
        let room = tone_level(&samples, 300.0);
        assert!(
            line > room * 1.1,
            "headroom control must not flatten the gains it is protecting: dialogue {line:.5} \
             against ambience {room:.5}"
        );
    }
}
