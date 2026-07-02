//! Native dataset face-embedding analysis (epic 6529 P3, sc-6538).
//!
//! The `dataset_face_analysis` job runs the native SCRFD+ArcFace face stack over a (Person) training
//! dataset's images, takes the LARGEST face in each, and POSTs every image's raw ArcFace embedding +
//! frame fraction to rust-api's face sidecar (`.../face-embeddings`) — the face-stack analog of the
//! CLIP `dataset_analysis` job. MLX on macOS (`mlx-gen-face`); candle off-Mac with `--features
//! backend-candle` (`candle-gen-face`); on a platform with neither, a precise unsupported error.
//!
//! The one correctness linchpin: an image with no detectable face yields an **empty** embedding + a
//! `0.0` fraction record — the explicit "examined, no face" signal the readiness fold turns into a
//! `NoFace` finding (an *absent* record means "not processed"). The two backends differ exactly here —
//! MLX's `detect` returns an empty list on no face while candle's `largest_face` *errors* — so both
//! legs return an `Option` and funnel through the pure [`face_record_fields`], whose `None` branch
//! emits that empty record.

use super::*;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{CancelFlag, Image};

// The face-embedding space (glintr100 / ArcFace ResNet100, 512-d). Like the CLIP `EMBEDDING_SPACE`,
// the readiness reader ignores it; it only guards the sidecar ingest merge, so the one requirement is
// that it is stable across runs (a drifting value would replace instead of merge a partial re-run).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const FACE_EMBEDDING_SPACE: &str = "arcface-glintr100";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const CANCEL_MESSAGE: &str = "Dataset face analysis canceled by user.";

// MLX face stack (macOS): the same SCRFD + ArcFace the InstantID/kps paths use, loaded directly (not
// through a registry — the `FaceEmbedder` contract has no gen-core registration).
#[cfg(target_os = "macos")]
use mlx_gen::weights::Weights;
#[cfg(target_os = "macos")]
use mlx_gen_face::FaceAnalysis;
// candle face stack (off-Mac): `candle_gen_face::load` builds an `impl FaceEmbedder`; importing the
// trait brings `analyze` into scope.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use gen_core::FaceEmbedder;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug)]
struct FaceAnalysisItem {
    image_path: PathBuf,
    content_hash: String,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug)]
struct FaceAnalysisRecord {
    content_hash: String,
    /// Raw (un-normalized) ArcFace embedding of the largest face. Empty ⇒ no face detected.
    embedding: Vec<f32>,
    /// Largest-face bbox area as a fraction of the frame, in `[0, 1]`. `0.0` when no face.
    face_fraction: f64,
}

/// The largest face's bounding-box area as a fraction of the frame (sc-6538). Pure. SCRFD can return
/// boxes that extend past the image edge, so the fraction is clamped to `[0, 1]` rather than trusting
/// `area ≤ frame`. A zero-area frame (degenerate metadata) yields `0.0`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn face_fraction(bbox: [f32; 4], width: u32, height: u32) -> f64 {
    let frame = f64::from(width) * f64::from(height);
    if frame <= 0.0 {
        return 0.0;
    }
    let w = f64::from((bbox[2] - bbox[0]).max(0.0));
    let h = f64::from((bbox[3] - bbox[1]).max(0.0));
    ((w * h) / frame).clamp(0.0, 1.0)
}

/// The no-face funnel (sc-6538). Pure. Maps the largest detected face (its bbox + raw embedding) — or
/// `None` when the image has no detectable face — to the `(embedding, face_fraction)` the sidecar
/// stores. The `None` branch emits the empty/`0.0` "examined, no face" record both backends rely on.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn face_record_fields(
    largest: Option<([f32; 4], Vec<f32>)>,
    width: u32,
    height: u32,
) -> (Vec<f32>, f64) {
    match largest {
        Some((bbox, embedding)) => (embedding, face_fraction(bbox, width, height)),
        None => (Vec::new(), 0.0),
    }
}

/// MLX leg: detect every face (largest-first), then ArcFace-embed only the largest — one recognition
/// forward, not N. `None` when no face. Runs the `!Send` MLX work; call inside `spawn_blocking`.
#[cfg(target_os = "macos")]
fn largest_face_mlx(
    analysis: &FaceAnalysis,
    image: &Image,
) -> WorkerResult<Option<([f32; 4], Vec<f32>)>> {
    let (h, w) = (image.height as usize, image.width as usize);
    let dets = analysis
        .detect(&image.pixels, h, w)
        .map_err(|error| WorkerError::Engine(format!("face detect: {error}")))?;
    // `detect` returns largest-first; embed just `[0]`.
    let Some(det) = dets.first() else {
        return Ok(None);
    };
    let face = analysis
        .embed(&image.pixels, h, w, det)
        .map_err(|error| WorkerError::Engine(format!("face embed: {error}")))?;
    Ok(Some((face.bbox, face.embedding)))
}

/// candle leg: detect+embed every face (largest-first) and take `[0]`. Deliberately NOT `largest_face`,
/// which *errors* on no face — that would fail the whole job on the first faceless image; `analyze`
/// returns an empty list instead, so `None` flows to the empty record. Call inside `spawn_blocking`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn largest_face_candle(
    analysis: &dyn FaceEmbedder,
    image: &Image,
) -> WorkerResult<Option<([f32; 4], Vec<f32>)>> {
    let faces = analysis
        .analyze(image)
        .map_err(|error| WorkerError::Engine(format!("face analyze: {error}")))?;
    Ok(faces.into_iter().next().map(|f| (f.bbox, f.embedding)))
}

/// Load the face stack from `weights_dir` and embed the largest face of every item, reporting per-item
/// progress on `tx`. The `!Send` model work; call inside `spawn_blocking`. MLX leg (macOS).
#[cfg(target_os = "macos")]
fn analyze_faces(
    items: Vec<FaceAnalysisItem>,
    weights_dir: PathBuf,
    cancel: CancelFlag,
    tx: tokio::sync::mpsc::Sender<usize>,
) -> WorkerResult<Vec<FaceAnalysisRecord>> {
    let scrfd = Weights::from_file(weights_dir.join(crate::image_jobs::INSTANTID_SCRFD_FILE))
        .map_err(|error| WorkerError::Engine(format!("SCRFD weights: {error}")))?;
    let arcface = Weights::from_file(weights_dir.join(crate::image_jobs::INSTANTID_ARCFACE_FILE))
        .map_err(|error| WorkerError::Engine(format!("ArcFace weights: {error}")))?;
    let analysis = FaceAnalysis::load(&scrfd, &arcface)
        .map_err(|error| WorkerError::Engine(format!("face stack load: {error}")))?;
    embed_largest_faces(
        &items,
        |image| largest_face_mlx(&analysis, image),
        &cancel,
        &tx,
    )
}

/// candle leg (off-Mac): same as the MLX [`analyze_faces`], loading the candle SCRFD/ArcFace stack from
/// `weights_dir` by its canonical file names.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn analyze_faces(
    items: Vec<FaceAnalysisItem>,
    weights_dir: PathBuf,
    cancel: CancelFlag,
    tx: tokio::sync::mpsc::Sender<usize>,
) -> WorkerResult<Vec<FaceAnalysisRecord>> {
    let analysis = candle_gen_face::load(&weights_dir)
        .map_err(|error| WorkerError::Engine(format!("face stack load: {error}")))?;
    embed_largest_faces(
        &items,
        |image| largest_face_candle(&analysis, image),
        &cancel,
        &tx,
    )
}

/// The per-item loop shared by both backends: decode → largest face → record, with cancel checks +
/// per-item progress. `largest` is the backend's largest-face extractor.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn embed_largest_faces(
    items: &[FaceAnalysisItem],
    largest: impl Fn(&Image) -> WorkerResult<Option<([f32; 4], Vec<f32>)>>,
    cancel: &CancelFlag,
    tx: &tokio::sync::mpsc::Sender<usize>,
) -> WorkerResult<Vec<FaceAnalysisRecord>> {
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
        }
        let image = load_face_image(&item.image_path)?;
        let (embedding, face_fraction) =
            face_record_fields(largest(&image)?, image.width, image.height);
        out.push(FaceAnalysisRecord {
            content_hash: item.content_hash.clone(),
            embedding,
            face_fraction,
        });
        // A closed channel means the consumer loop returned early (POST failure / 409); trip the
        // engine flag so face analysis bails instead of running unheard, matching
        // dataset_analysis_jobs (sc-8804, F-003 — the swallowed-closed-channel leak).
        if tx.blocking_send(index).is_err() {
            cancel.cancel();
        }
    }
    Ok(out)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_dataset_face_analysis_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let items = face_analysis_items(settings, &job.payload)?;
    if items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Dataset face analysis job has no items to analyze.".to_owned(),
        ));
    }
    let backend = backend_label(&settings.gpu_id);
    let total = items.len();

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        face_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.04,
            "Preparing dataset face analysis job.",
            None,
            backend,
        ),
    )
    .await?;
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;

    // Stage the SCRFD + ArcFace bundle (download-on-first-use; a prior InstantID / kps run leaves it
    // cached). The same dir the candle stack loads, and the dir the MLX path joins the two file names in.
    let weights_dir = crate::image_jobs::ensure_face_stack_dir(api, settings, job).await?;
    update_job(
        api,
        &job.id,
        face_progress(
            JobStatus::LoadingModel,
            ProgressStage::LoadingModel,
            0.08,
            "Loading face stack (SCRFD + ArcFace).",
            None,
            backend,
        ),
    )
    .await?;

    let cancel = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<usize>(64);
    let blocking_cancel = cancel.clone();
    let blocking_items = items;
    let job_id = job.id.clone();
    let blocking = tokio::task::spawn_blocking(move || -> WorkerResult<Vec<FaceAnalysisRecord>> {
        emit_event(
            "dataset_face_analysis_load_start",
            json!({ "jobId": job_id, "space": FACE_EMBEDDING_SPACE }),
        );
        let records = analyze_faces(blocking_items, weights_dir, blocking_cancel, tx)?;
        emit_event(
            "dataset_face_analysis_load_complete",
            json!({ "jobId": job_id, "space": FACE_EMBEDDING_SPACE }),
        );
        Ok(records)
    });

    // Bind the blocking face-analysis task to its cancel flag (sc-8804, F-003): every `update_job`/
    // `heartbeat` `?` below returns early on a transient POST failure or a 409 (stale-sweep
    // reclaim); on that early return this guard trips `cancel` and aborts the analysis thread
    // instead of leaving it running on a job nobody is consuming. `cancel` is kept alongside (it's
    // `Clone`) for the in-loop cancel poll; the guard drives only the drop-time teardown.
    let mut guard = CancelJoinGuard::new(cancel.clone(), blocking);
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // Run the stream loop capturing its Result so any `?`-error path performs the explicit awaited
    // bounded-join teardown BEFORE returning, instead of drop-and-run (sc-8804, F-003).
    let loop_result: WorkerResult<()> = async {
        loop {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Some(index) => {
                            let progress = 0.12 + 0.78 * ((index + 1) as f64 / total as f64);
                            update_job(
                                api,
                                &job.id,
                                face_progress(
                                    JobStatus::Running,
                                    ProgressStage::Running,
                                    progress,
                                    &format!("Analyzed face {} of {}.", index + 1, total),
                                    None,
                                    backend,
                                ),
                            )
                            .await?;
                        }
                        None => break,
                    }
                }
                _ = interval.tick() => {
                    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                    match check_cancel(api, &job.id, CANCEL_MESSAGE).await {
                        Ok(()) => {}
                        Err(WorkerError::Canceled(_)) => cancel.cancel(),
                        Err(error) => return Err(error),
                    }
                }
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = loop_result {
        guard.cancel_and_join().await;
        return Err(error);
    }

    // Loop exited cleanly (channel closed) — reclaim the handle (disarming the drop-guard) and join.
    let records = guard
        .into_handle()
        .await
        .map_err(|error| task_join_error("dataset face analysis task join", error))??;

    update_job(
        api,
        &job.id,
        face_progress(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.94,
            "Saving face records.",
            None,
            backend,
        ),
    )
    .await?;
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let dataset_id = required_payload_string(&job.payload, "datasetId")?;
    let payload = face_records_payload(&records);
    let stored: Value = api
        .post_json(
            &format!(
                "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/face-embeddings"
            ),
            &json!({ "space": FACE_EMBEDDING_SPACE, "items": payload }),
        )
        .await?;
    let with_face = records.iter().filter(|r| !r.embedding.is_empty()).count();
    update_job(
        api,
        &job.id,
        face_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!(
                "Analyzed {} image(s); {with_face} with a face.",
                records.len()
            ),
            Some(face_result(dataset_id, records.len(), with_face, stored)),
            backend,
        ),
    )
    .await?;
    Ok(())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn face_records_payload(records: &[FaceAnalysisRecord]) -> Vec<Value> {
    records
        .iter()
        .map(|record| {
            json!({
                "contentHash": record.content_hash,
                "embedding": record.embedding,
                "faceFraction": record.face_fraction,
            })
        })
        .collect()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn face_analysis_items(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<Vec<FaceAnalysisItem>> {
    let dataset_root = payload
        .get("datasetRoot")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Dataset face analysis payload.datasetRoot must be an app-managed dataset path."
                    .to_owned(),
            )
        })?;
    let items = payload
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Dataset face analysis payload.items must be an array.".to_owned(),
            )
        })?;
    items
        .iter()
        .map(|item| {
            let object = item.as_object().ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "Dataset face analysis item must be an object.".to_owned(),
                )
            })?;
            let item_id = object
                .get("itemId")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(
                        "Dataset face analysis item is missing itemId.".to_owned(),
                    )
                })?;
            let content_hash = object
                .get("contentHash")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "Dataset face analysis item {item_id} is missing contentHash."
                    ))
                })?
                .to_owned();
            let image_path = object
                .get("imagePath")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "Dataset face analysis item {item_id} is missing imagePath."
                    ))
                })?;
            let image_path = resolve_dataset_item_path(
                settings,
                dataset_root,
                image_path,
                &format!("Dataset face analysis item {item_id} imagePath"),
            )?;
            Ok(FaceAnalysisItem {
                image_path,
                content_hash,
            })
        })
        .collect()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn load_face_image(path: &Path) -> WorkerResult<Image> {
    let decoded = crate::image_decode::decode_image_any(path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("face image {}: {error}", path.display()))
        })?
        .to_rgb8();
    Ok(Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn face_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn face_result(dataset_id: &str, analyzed: usize, with_face: usize, stored: Value) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("space".to_owned(), json!(FACE_EMBEDDING_SPACE));
    result.insert("datasetId".to_owned(), json!(dataset_id));
    result.insert("analyzedItemCount".to_owned(), json!(analyzed));
    result.insert("withFaceCount".to_owned(), json!(with_face));
    result.insert(
        "stored".to_owned(),
        stored.get("stored").cloned().unwrap_or(Value::Null),
    );
    result
}

#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
pub(crate) async fn run_dataset_face_analysis_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Dataset face analysis (SCRFD + ArcFace) needs the macOS MLX backend or the candle backend \
         (build with --features backend-candle)."
            .to_owned(),
    ))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn face_fraction_is_box_area_over_frame() {
        // A 50×40 box in a 100×100 frame → 2000 / 10000 = 0.2.
        let f = face_fraction([10.0, 10.0, 60.0, 50.0], 100, 100);
        assert!((f - 0.2).abs() < 1e-9, "{f}");
    }

    #[test]
    fn face_fraction_clamps_an_out_of_frame_box_and_a_degenerate_frame() {
        // SCRFD can return a box past the edge — the fraction never exceeds 1.0.
        assert_eq!(face_fraction([-10.0, -10.0, 200.0, 200.0], 100, 100), 1.0);
        // An inverted / empty box has no area.
        assert_eq!(face_fraction([60.0, 60.0, 10.0, 10.0], 100, 100), 0.0);
        // A zero-area frame (bad metadata) never divides by zero.
        assert_eq!(face_fraction([0.0, 0.0, 10.0, 10.0], 0, 100), 0.0);
    }

    #[test]
    fn no_face_funnels_to_the_empty_examined_record() {
        // The linchpin: None (no detectable face) must produce an EMPTY embedding + 0.0 fraction — the
        // "examined, no face" record the readiness fold turns into a NoFace finding. Skipping the item
        // (no record) would instead read as "not processed" and the finding would silently never fire.
        let (embedding, fraction) = face_record_fields(None, 512, 512);
        assert!(embedding.is_empty());
        assert_eq!(fraction, 0.0);
    }

    #[test]
    fn a_detected_face_carries_its_embedding_and_fraction() {
        let (embedding, fraction) = face_record_fields(
            Some(([0.0, 0.0, 50.0, 50.0], vec![0.1, 0.2, 0.3])),
            100,
            100,
        );
        assert_eq!(embedding, vec![0.1, 0.2, 0.3]);
        assert!((fraction - 0.25).abs() < 1e-9, "{fraction}");
    }

    #[test]
    fn records_payload_encodes_the_face_sidecar_shape() {
        let payload = face_records_payload(&[
            FaceAnalysisRecord {
                content_hash: "h_face".to_owned(),
                embedding: vec![1.0, 0.0],
                face_fraction: 0.3,
            },
            FaceAnalysisRecord {
                content_hash: "h_noface".to_owned(),
                embedding: Vec::new(),
                face_fraction: 0.0,
            },
        ]);
        assert_eq!(
            Value::Array(payload),
            json!([
                { "contentHash": "h_face", "embedding": [1.0, 0.0], "faceFraction": 0.3 },
                { "contentHash": "h_noface", "embedding": [], "faceFraction": 0.0 }
            ])
        );
    }

    /// Real-weights worker integration (sc-6538): proves the *worker binary* links + loads the MLX
    /// `FaceAnalysis` (SCRFD + ArcFace) and runs the real forward through `largest_face_mlx`. Two legs:
    /// a synthetic gradient (no face) must funnel to the empty "examined, no face" record — the linchpin
    /// against the REAL detector, not just the pure funnel; and, when `SCENEWORKS_TEST_FACE` points at a
    /// real face photo, the largest face yields a 512-d embedding + a fraction in `(0, 1]`. `#[ignore]`
    /// per convention — the weights live outside CI. Run on a Mac with the bundle staged + Metal:
    ///   SCENEWORKS_INSTANTID_WEIGHTS=/path/to/instantid-mlx \
    ///     cargo test -p sceneworks-worker --lib -- --ignored face_pass_real_weights --nocapture
    #[test]
    #[ignore = "real-weight: needs the SceneWorks/instantid-mlx SCRFD+ArcFace bundle + Metal"]
    fn face_pass_real_weights_embeds_largest_and_reports_no_face() {
        let home = std::env::var("HOME").expect("HOME");
        let bundle = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(&home)
                    .join("Library/Application Support/SceneWorks/data/cache/instantid-mlx")
            });
        let scrfd = Weights::from_file(bundle.join(crate::image_jobs::INSTANTID_SCRFD_FILE))
            .expect("SCRFD weights staged");
        let arcface = Weights::from_file(bundle.join(crate::image_jobs::INSTANTID_ARCFACE_FILE))
            .expect("ArcFace weights staged");
        let analysis = FaceAnalysis::load(&scrfd, &arcface).expect("face stack loads");

        // No-face leg: a synthetic gradient has no face → the REAL detector returns empty → the funnel
        // produces the empty record. This is the linchpin proven against the model.
        let mut gradient = image::RgbImage::new(128, 128);
        for (x, y, px) in gradient.enumerate_pixels_mut() {
            *px = image::Rgb([(x * 2) as u8, (y * 2) as u8, 96]);
        }
        let gradient = Image {
            width: gradient.width(),
            height: gradient.height(),
            pixels: gradient.into_raw(),
        };
        let largest = largest_face_mlx(&analysis, &gradient).expect("detect runs");
        assert!(largest.is_none(), "a gradient has no detectable face");
        let (embedding, fraction) = face_record_fields(largest, gradient.width, gradient.height);
        assert!(
            embedding.is_empty() && fraction == 0.0,
            "empty no-face record"
        );

        // Face leg (optional): a real photo yields the largest face's 512-d embedding + a real fraction.
        if let Ok(face_path) = std::env::var("SCENEWORKS_TEST_FACE") {
            let decoded = image::open(&face_path)
                .unwrap_or_else(|e| panic!("face {face_path}: {e}"))
                .to_rgb8();
            let image = Image {
                width: decoded.width(),
                height: decoded.height(),
                pixels: decoded.into_raw(),
            };
            let largest = largest_face_mlx(&analysis, &image).expect("detect+embed runs");
            let (embedding, fraction) = face_record_fields(largest, image.width, image.height);
            assert_eq!(embedding.len(), 512, "ArcFace embedding is 512-d");
            assert!(
                embedding.iter().all(|v| v.is_finite()) && embedding.iter().any(|&v| v != 0.0),
                "embedding is finite + non-degenerate"
            );
            assert!(
                fraction > 0.0 && fraction <= 1.0,
                "face fraction in (0, 1]: {fraction}"
            );
            println!(
                "face pass ok: dim={} fraction={fraction:.4}",
                embedding.len()
            );
        }
    }
}
