//! CPU-only SheetSage2 transcription for YuE2 recording jobs. The backend seam keeps the job
//! lifecycle testable without downloading the cover closure or running a multi-GiB model.

use super::*;
use sceneworks_core::yue2_score::jobs::{self as contract, Yue2JobSpec};
use sha2::{Digest, Sha256};
use std::sync::{OnceLock, RwLock};

const CANCEL_MESSAGE: &str = "YuE2 transcription canceled by user.";
const OWNER_FILE: &str = "sceneworks-owner.json";

fn artifact_owner(
    job_id: &str,
    run_id: &str,
    asset_id: &str,
    sha256: &str,
    manifest: &str,
) -> Value {
    json!({"schema":"sceneworks-yue2-transcription-owner-v1", "originJobId":job_id,
        "runId":run_id, "sourceAudioAssetId":asset_id, "recordingSha256":sha256,
        "manifestSha256":manifest})
}

#[cfg(any(target_os = "macos", feature = "backend-candle", test))]
fn owner_matches(dir: &Path, run_id: &str, asset_id: &str, sha256: &str) -> WorkerResult<String> {
    let path = dir.join(OWNER_FILE);
    let file = std::fs::File::open(&path)?;
    if file.metadata()?.len() > 4096 {
        return Err(WorkerError::InvalidPayload(
            "yue2: transcription owner record is oversized".into(),
        ));
    }
    let owner: Value = serde_json::from_reader(file).map_err(|e| {
        WorkerError::InvalidPayload(format!("yue2: transcription owner record: {e}"))
    })?;
    if owner["schema"] != "sceneworks-yue2-transcription-owner-v1"
        || owner["runId"] != run_id
        || owner["sourceAudioAssetId"] != asset_id
        || owner["recordingSha256"] != sha256
    {
        return Err(WorkerError::InvalidPayload(
            "yue2: existing transcription belongs to a different recording asset or run".into(),
        ));
    }
    owner["manifestSha256"]
        .as_str()
        .filter(|digest| digest.len() == 64)
        .map(str::to_owned)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "yue2: transcription owner record has no manifest digest".into(),
            )
        })
}

/// One guarded decode directory per attempt. Random allocation prevents a retry or another
/// process from deleting or following a pre-existing path with the same job id.
fn decode_scratch(job_id: &str) -> std::io::Result<tempfile::TempDir> {
    let prefix = format!(
        "sw-yue2-transcription-{}-",
        super::safe_download_dir(job_id)
    );
    tempfile::Builder::new().prefix(&prefix).tempdir()
}

async fn verified_source_sha256(path: &Path, expected: &str) -> WorkerResult<()> {
    use tokio::io::AsyncReadExt;
    let mut file = tokio::fs::File::open(path).await?;
    let mut digest = Sha256::new();
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        digest.update(&chunk[..read]);
    }
    if format!("{:x}", digest.finalize()) != expected {
        return Err(WorkerError::InvalidPayload(
            "yue2: source recording no longer matches its queued SHA-256".into(),
        ));
    }
    Ok(())
}

fn planned_frames(
    info: crate::video_jobs::reference_audio::DecodedWavInfo,
    max: Option<f64>,
) -> u64 {
    max.map(|seconds| ((seconds + 1.0) * f64::from(info.sample_rate)).ceil() as u64)
        .map_or(info.frames, |crop| info.frames.min(crop))
}

/// Serialize the CPU transcriber against YuE2 generator residency in this process. The generator
/// holds a read guard through generation; a transcription holds a write guard through unload.
pub(crate) fn residency_lock() -> &'static RwLock<()> {
    static LOCK: OnceLock<RwLock<()>> = OnceLock::new();
    LOCK.get_or_init(|| RwLock::new(()))
}

pub(crate) fn require_unloaded(live: usize) -> WorkerResult<()> {
    if live == 0 {
        Ok(())
    } else {
        Err(WorkerError::InvalidPayload(format!(
            "yue2: refusing generator load while {live} SheetSage2 model(s) are resident"
        )))
    }
}

/// The narrow injectable boundary around the native cover closure and its explicit unload.
// The neither build keeps the trait for job routing but has no native transcriber to read these
// fields; its implementation rejects the request before touching any input.
#[cfg_attr(
    not(any(target_os = "macos", feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) struct TranscriptionInput<'a> {
    pub snapshots: &'a [(String, PathBuf)],
    pub audio: gen_core::AudioTrack,
    pub source_sha256: &'a str,
    pub spec: &'a Yue2JobSpec,
    pub artifact_dir: &'a Path,
    pub canceled: &'a gen_core::CancelFlag,
}

pub(crate) trait TranscriptionBackend: Send + Sync + 'static {
    fn transcribe(
        &self,
        input: TranscriptionInput<'_>,
        progress: &mut dyn FnMut(f64, String),
    ) -> WorkerResult<Value>;

    fn live_models(&self) -> usize;
}

pub(crate) struct NativeTranscription;

#[cfg(any(target_os = "macos", feature = "backend-candle"))]
impl TranscriptionBackend for NativeTranscription {
    fn live_models(&self) -> usize {
        crate::inference_runtime::audio_providers::candle_audio_sheetsage2::provider::live_models()
    }

    fn transcribe(
        &self,
        input: TranscriptionInput<'_>,
        progress: &mut dyn FnMut(f64, String),
    ) -> WorkerResult<Value> {
        let TranscriptionInput {
            snapshots,
            audio,
            source_sha256,
            spec,
            artifact_dir,
            canceled,
        } = input;
        use crate::inference_runtime::audio_providers::{
            candle_audio_sheetsage2 as ss2, candle_audio_yue2 as yue2,
        };
        let _exclusive = residency_lock()
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if self.live_models() != 0 {
            return Err(WorkerError::InvalidPayload(
                "yue2: another SheetSage2 transcriber is still resident".into(),
            ));
        }
        let dirs = snapshots
            .iter()
            .fold(yue2::snapshot::SnapshotDirs::new(), |dirs, (repo, path)| {
                dirs.with(repo.clone(), path.clone())
            });
        let closure =
            yue2::snapshot::resolve_closure(yue2::Closure::Cover, &dirs).map_err(|e| {
                WorkerError::InvalidPayload(format!("yue2: cover closure does not verify: {e}"))
            })?;
        if canceled.is_cancelled() {
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.into()));
        }
        let settings = spec.transcription.unwrap_or_default();
        let settings = ss2::review::TranscriptionSettings {
            overlap_seconds: settings.resolved_overlap_seconds(),
            lookahead_seconds: settings.resolved_lookahead_seconds(),
            max_seconds: settings.max_seconds,
            ..Default::default()
        };
        let source = ss2::provider::SourceAudio::interleaved(
            &audio.samples,
            audio.sample_rate,
            audio.channels,
        )
        .map_err(|e| WorkerError::InvalidPayload(format!("yue2: source recording: {e}")))?
        .with_original_sha256(source_sha256);
        let source = if let Some(name) = spec
            .sources
            .as_ref()
            .and_then(|s| s.recording.as_ref())
            .and_then(|r| r.name.as_ref())
        {
            source.with_name(name)
        } else {
            source
        };
        // SheetSage2's GRN needs float64 reduction, which Metal cannot perform. The provider
        // validates Device::Cpu before reading weights; this is the only production device.
        let transcriber =
            ss2::provider::Transcriber::load(&closure, &ss2::candle_core::Device::Cpu)
                .map_err(|e| WorkerError::Engine(format!("yue2: SheetSage2 load failed: {e}")))?;
        let result = transcriber.transcribe(&source, &settings, |step| {
            if canceled.is_cancelled() {
                Err(ss2::Error::Request(CANCEL_MESSAGE.into()))
            } else {
                match step {
                    ss2::provider::Progress::Encoding { window, windows } => progress(
                        0.1 + 0.7 * (window.saturating_sub(1) as f64 / windows.max(1) as f64),
                        format!("Encoding transcription window {window} of {windows}."),
                    ),
                    ss2::provider::Progress::Decoding {
                        window,
                        windows,
                        tokens,
                    } => progress(
                        0.1 + 0.7 * (window as f64 / windows.max(1) as f64),
                        format!("Transcribing window {window} of {windows}: {tokens} tokens."),
                    ),
                    ss2::provider::Progress::Notation => {
                        progress(0.85, "Building score and review artifacts.".into())
                    }
                }
                Ok(())
            }
        });
        // Save while the tokenizer is available, then unload on success, failure and cancel.
        let saved = result.and_then(|transcription| {
            transcription.save(artifact_dir, transcriber.model().tokenizer())
        });
        let unload = transcriber.unload();
        if !unload.released || unload.live_models_after != 0 || self.live_models() != 0 {
            return Err(WorkerError::Engine(format!(
                "yue2: SheetSage2 unload failed (released={}, live_models_after={})",
                unload.released, unload.live_models_after
            )));
        }
        if canceled.is_cancelled() {
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.into()));
        }
        let digest =
            saved.map_err(|e| WorkerError::Engine(format!("yue2: transcription failed: {e}")))?;
        let artifact = ss2::review::ReviewArtifact::open(artifact_dir).map_err(|e| {
            WorkerError::Engine(format!("yue2: saved transcription failed replay: {e}"))
        })?;
        if artifact.manifest_sha256() != digest
            || artifact
                .source()
                .map_err(|e| WorkerError::Engine(format!("yue2: transcription source: {e}")))?
                .original_sha256
                .as_deref()
                != Some(source_sha256)
        {
            return Err(WorkerError::Engine(
                "yue2: transcription identity changed during publication".into(),
            ));
        }
        Ok(json!({
            "manifestSha256": digest,
            "device": "cpu",
            "replay": {"artifactsMatched": artifact.report().artifacts_matched, "windows": artifact.report().windows},
            "unload": {"released": unload.released, "parameterBytes": unload.parameter_bytes, "liveModelsAfter": unload.live_models_after},
            "warnings": artifact.manifest()["review"]["warnings"],
            "readiness": artifact.manifest()["review"]["cover"],
        }))
    }
}

#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
impl TranscriptionBackend for NativeTranscription {
    fn live_models(&self) -> usize {
        0
    }
    fn transcribe(
        &self,
        _: TranscriptionInput<'_>,
        _: &mut dyn FnMut(f64, String),
    ) -> WorkerResult<Value> {
        Err(WorkerError::InvalidPayload(
            "yue2: this worker has no native audio runtime".into(),
        ))
    }
}

pub(crate) fn live_models() -> usize {
    NativeTranscription.live_models()
}

/// A completed artifact may outlive a failed terminal progress POST. A retried job replays that
/// same immutable artifact instead of loading a second transcriber or overwriting its files.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
async fn published_details(
    dir: &Path,
    source_sha256: &str,
    source_asset_id: &str,
    spec: &Yue2JobSpec,
    gpu_id: &str,
) -> WorkerResult<Option<Value>> {
    use crate::inference_runtime::audio_providers::candle_audio_sheetsage2 as ss2;
    if !dir.exists() {
        return Ok(None);
    }
    let owner_manifest = owner_matches(
        dir,
        spec.run_id.as_deref().unwrap_or_default(),
        source_asset_id,
        source_sha256,
    )?;
    let persisted_bytes = std::fs::read_dir(dir)?.try_fold(0u64, |bytes, entry| {
        let entry = entry?;
        let metadata = entry.metadata()?;
        Ok::<u64, std::io::Error>(bytes.saturating_add(metadata.len()))
    })?;
    let _replay_lease =
        crate::yue2_admission::check_transcription_replay(persisted_bytes, gpu_id).await?;
    let artifact = ss2::review::ReviewArtifact::open(dir).map_err(|e| {
        WorkerError::InvalidPayload(format!(
            "yue2: existing transcription artifact does not verify: {e}"
        ))
    })?;
    if artifact.manifest_sha256() != owner_manifest {
        return Err(WorkerError::InvalidPayload(
            "yue2: existing transcription owner digest changed".into(),
        ));
    }
    let source = artifact.source().map_err(|e| {
        WorkerError::InvalidPayload(format!("yue2: existing transcription source: {e}"))
    })?;
    let settings = artifact.settings().map_err(|e| {
        WorkerError::InvalidPayload(format!("yue2: existing transcription settings: {e}"))
    })?;
    let requested = spec.transcription.unwrap_or_default();
    if source.original_sha256.as_deref() != Some(source_sha256)
        || settings.max_seconds != requested.max_seconds
        || settings.overlap_seconds != requested.resolved_overlap_seconds()
        || settings.lookahead_seconds != requested.resolved_lookahead_seconds()
        || artifact.manifest()["closure"]["device"] != "cpu"
    {
        return Err(WorkerError::InvalidPayload(
            "yue2: existing transcription belongs to a different recording or settings".into(),
        ));
    }
    require_unloaded(live_models())?;
    Ok(Some(json!({
        "manifestSha256": artifact.manifest_sha256(),
        "device": "cpu",
        "replay": {"artifactsMatched": artifact.report().artifacts_matched, "windows": artifact.report().windows},
        "unload": {"released": true, "parameterBytes": 0, "liveModelsAfter": 0},
        "warnings": artifact.manifest()["review"]["warnings"],
        "readiness": artifact.manifest()["review"]["cover"],
        "reusedVerifiedArtifact": true,
    })))
}

#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
async fn published_details(
    dir: &Path,
    _: &str,
    _: &str,
    _: &Yue2JobSpec,
    _: &str,
) -> WorkerResult<Option<Value>> {
    if dir.exists() {
        Err(WorkerError::InvalidPayload(
            "yue2: this worker cannot replay a native transcription artifact".into(),
        ))
    } else {
        Ok(None)
    }
}

fn completion_result(
    spec: &Yue2JobSpec,
    rel: &str,
    mut details: Value,
    usage_policy: &Value,
) -> JsonObject {
    details["id"] = json!(contract::transcription_id(
        spec.run_id.as_deref().unwrap_or_default()
    ));
    details["dir"] = json!(rel);
    let readiness = details["readiness"].clone();
    let mut result = JsonObject::new();
    result.insert(
        "yue2".into(),
        json!({
            "status": "completed", "kind": "transcribe", "runId": spec.run_id,
            "transcription": details, "readiness": readiness, "usagePolicy": usage_policy,
        }),
    );
    result
}

fn cover_snapshots(settings: &Settings, entry: &Value) -> WorkerResult<Vec<(String, PathBuf)>> {
    let rows = entry
        .get("conditionalComponents")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            WorkerError::InvalidPayload("yue2: the catalog declares no cover closure".into())
        })?;
    ["yue2_sheetsage2", "yue2_mert_v2_fullsong"]
        .iter()
        .map(|id| {
            let row = rows
                .iter()
                .find(|row| row["componentId"] == *id)
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!("yue2: cover closure lacks {id}"))
                })?;
            let repo = row["repo"]
                .as_str()
                .ok_or_else(|| WorkerError::InvalidPayload(format!("yue2: {id} has no repo")))?;
            let revision = row["revision"].as_str().ok_or_else(|| {
                WorkerError::InvalidPayload(format!("yue2: {id} has no revision"))
            })?;
            let path = crate::model_jobs::huggingface_pinned_snapshot_dir(
                &settings.data_dir,
                repo,
                revision,
            )
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "yue2: {id} ({repo}@{revision}) is not installed"
                ))
            })?;
            Ok((repo.to_owned(), path))
        })
        .collect()
}

/// Execute after the job's live eligibility check. Its terminal result is exactly the pointer the
/// API side effect reopens, so a forged or incomplete artifact cannot be imported as a score.
pub(crate) struct TranscriptionJob<'a> {
    pub api: &'a ApiClient,
    pub settings: &'a Settings,
    pub job: &'a JobSnapshot,
    pub spec: &'a Yue2JobSpec,
    pub project_id: &'a str,
    pub project_path: &'a Path,
    pub entry: &'a Value,
    pub usage_policy: &'a Value,
}

pub(crate) async fn run<B: TranscriptionBackend>(
    context: TranscriptionJob<'_>,
    backend: B,
) -> WorkerResult<()> {
    let TranscriptionJob {
        api,
        settings,
        job,
        spec,
        project_id,
        project_path,
        entry,
        usage_policy,
    } = context;
    let source_id = spec.source_audio_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload("yue2: transcribe has no sourceAudioAssetId".into())
    })?;
    let source = spec
        .sources
        .as_ref()
        .and_then(|s| s.recording.as_ref())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("yue2: transcribe has no recorded source identity".into())
        })?;
    if source.asset_id != source_id {
        return Err(WorkerError::InvalidPayload(
            "yue2: source recording changed after enqueue".into(),
        ));
    }
    let source_path = crate::video_jobs::ltx::resolve_clip_media_path(
        settings,
        project_id,
        source_id,
        project_path,
    )?;
    verified_source_sha256(&source_path, &source.sha256).await?;
    let run_id = spec.run_id.as_deref().unwrap_or_default();
    let rel = contract::transcription_dir(run_id);
    let dir = project_path.join(&rel);
    let _claim = super::yue2_jobs::RunClaim::acquire(&dir)?;
    if let Some(details) =
        published_details(&dir, &source.sha256, source_id, spec, &settings.gpu_id).await?
    {
        return super::update_job(
            api,
            &job.id,
            super::yue2_jobs::yue2_progress(
                JobStatus::Completed,
                ProgressStage::Completed,
                1.0,
                "Published a reviewable YuE2 transcription.",
                Some(completion_result(spec, &rel, details, usage_policy)),
                "cpu",
            ),
        )
        .await
        .map(|_| ());
    }
    let snapshots = cover_snapshots(settings, entry)?;
    let max_seconds = spec.transcription.and_then(|s| s.max_seconds);
    let source_layout = crate::video_jobs::reference_audio::probe_source_wav(&source_path)?;
    let planned = source_layout.map(|info| planned_frames(info, max_seconds));
    let preflight_seconds = source_layout.map_or(300, |info| {
        planned
            .unwrap_or(0)
            .div_ceil(u64::from(info.sample_rate.max(1)))
    });
    let source_cost = source_layout.map_or(0, |info| {
        crate::yue2_admission::transcription_source_bytes(
            planned.unwrap_or(0),
            info.sample_rate,
            info.channels,
        )
    });
    // Reserve the model and, for WAV, all source buffers before ffmpeg writes or RAM is allocated.
    // Compressed inputs use a budget-derived disk cap; the decoded layout is re-priced below.
    let preflight = crate::yue2_admission::check_transcription(
        entry,
        preflight_seconds,
        &settings.gpu_id,
        source_cost,
    )
    .await?;
    let output_cap = if let Some(info) = source_layout {
        planned
            .unwrap_or(0)
            .saturating_mul(u64::from(info.channels))
            .saturating_mul(4)
            .saturating_add(2 << 20)
    } else {
        crate::yue2_admission::transcription_unknown_decode_cap(&settings.gpu_id).await?
    };
    let audio = {
        let scratch = decode_scratch(&job.id)?;
        let disk_cap = fs2::available_space(scratch.path())?.saturating_sub(64 << 20);
        if disk_cap < (2 << 20) || (source_layout.is_some() && output_cap > disk_cap) {
            return Err(WorkerError::InvalidPayload(format!(
                "yue2: transcription scratch disk cannot admit {:.2} GiB of decoded recording",
                output_cap as f64 / 1_073_741_824.0
            )));
        }
        let (wav, decoded) = crate::video_jobs::reference_audio::write_audio_normalized_bounded(
            api,
            settings,
            &job.id,
            CANCEL_MESSAGE,
            &source_path,
            scratch.path(),
            crate::video_jobs::reference_audio::AudioDecodeBounds {
                max_seconds: max_seconds.map(|s| s + 1.0),
                max_output_bytes: output_cap.min(disk_cap),
            },
        )
        .await?;
        if let Some(expected) = planned {
            if decoded.frames.saturating_add(2) < expected {
                return Err(WorkerError::InvalidPayload(
                    "yue2: decoded recording ended before its admitted source duration".into(),
                ));
            }
        }
        // The asset can be replaced between the first digest check and ffmpeg's open. Refuse any
        // changed bytes before assigning their decoded samples the queued recording identity.
        verified_source_sha256(&source_path, &source.sha256).await?;
        drop(preflight);
        let seconds = decoded
            .frames
            .div_ceil(u64::from(decoded.sample_rate.max(1)));
        let source_bytes = crate::yue2_admission::transcription_source_bytes(
            decoded.frames,
            decoded.sample_rate,
            decoded.channels,
        );
        let lease = crate::yue2_admission::check_transcription(
            entry,
            seconds,
            &settings.gpu_id,
            source_bytes,
        )
        .await?;
        let audio = crate::audio_jobs::read_wav_f32(&wav)?;
        (audio, lease)
    };
    let (audio, lease) = audio;
    if backend.live_models() != 0 {
        return Err(WorkerError::InvalidPayload(
            "yue2: a SheetSage2 transcriber is already resident".into(),
        ));
    }
    let partial = super::yue2_jobs::partial_dir(&dir);
    if partial.exists() {
        std::fs::remove_dir_all(&partial)?;
    }
    std::fs::create_dir_all(&partial)?;
    let cancel = gen_core::CancelFlag::new();
    let task_cancel = cancel.clone();
    let spec_copy = spec.clone();
    let source_sha = source.sha256.clone();
    let partial_for_cleanup = partial.clone();
    let (tx, mut rx) = tokio::sync::watch::channel::<Option<(f64, String)>>(None);
    let handle = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        let mut progress = |fraction, message| {
            tx.send_replace(Some((fraction, message)));
        };
        let result = backend.transcribe(
            TranscriptionInput {
                snapshots: &snapshots,
                audio,
                source_sha256: &source_sha,
                spec: &spec_copy,
                artifact_dir: &partial,
                canceled: &task_cancel,
            },
            &mut progress,
        );
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&partial);
        }
        result
    });
    let pump = {
        let api = api.clone();
        let job_id = job.id.clone();
        tokio::spawn(async move {
            let mut last = None::<Instant>;
            while rx.changed().await.is_ok() {
                if let Some(at) = last {
                    let wait = std::time::Duration::from_millis(500).saturating_sub(at.elapsed());
                    if !wait.is_zero() {
                        tokio::time::sleep(wait).await;
                    }
                }
                let latest = rx.borrow_and_update().clone();
                if let Some((fraction, message)) = latest {
                    let _ = super::update_job(
                        &api,
                        &job_id,
                        super::yue2_jobs::yue2_progress(
                            JobStatus::Running,
                            ProgressStage::Generating,
                            fraction,
                            &message,
                            None,
                            "cpu",
                        ),
                    )
                    .await;
                    last = Some(Instant::now());
                }
            }
        })
    };
    let outcome = super::run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel.clone()),
        CANCEL_MESSAGE,
        "YuE2 transcription",
        super::no_cancel_ack(),
        handle,
    )
    .await;
    let _ = pump.await;
    if outcome.is_err() || cancel.is_cancelled() {
        let _ = std::fs::remove_dir_all(&partial_for_cleanup);
    }
    let details = outcome?;
    if cancel.is_cancelled() {
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.into()));
    }
    // Publish only a fully replay-verified artifact. Rename is atomic within the project volume.
    if dir.exists() {
        let _ = std::fs::remove_dir_all(super::yue2_jobs::partial_dir(&dir));
        return Err(WorkerError::InvalidPayload(
            "yue2: this transcription artifact already exists".into(),
        ));
    }
    let publish = (|| -> WorkerResult<()> {
        let manifest_sha = details["manifestSha256"]
            .as_str()
            .filter(|digest| {
                digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "yue2: transcription backend returned no valid manifest digest".into(),
                )
            })?;
        let owner = artifact_owner(&job.id, run_id, source_id, &source.sha256, manifest_sha);
        std::fs::write(
            super::yue2_jobs::partial_dir(&dir).join(OWNER_FILE),
            serde_json::to_vec(&owner)?,
        )?;
        std::fs::rename(super::yue2_jobs::partial_dir(&dir), &dir)?;
        Ok(())
    })();
    if publish.is_err() {
        let _ = std::fs::remove_dir_all(super::yue2_jobs::partial_dir(&dir));
    }
    publish?;
    let result = completion_result(spec, &rel, details, usage_policy);
    super::update_job(
        api,
        &job.id,
        super::yue2_jobs::yue2_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Published a reviewable YuE2 transcription.",
            Some(result),
            "cpu",
        ),
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_owner_binds_same_bytes_to_the_original_asset_and_run() {
        let dir = tempfile::tempdir().unwrap();
        let sha = "a".repeat(64);
        let manifest = "b".repeat(64);
        let owner = artifact_owner("job-original", "yue2run_one", "asset_one", &sha, &manifest);
        std::fs::write(
            dir.path().join(OWNER_FILE),
            serde_json::to_vec(&owner).unwrap(),
        )
        .unwrap();
        assert_eq!(
            owner_matches(dir.path(), "yue2run_one", "asset_one", &sha).unwrap(),
            manifest
        );
        assert!(owner_matches(dir.path(), "yue2run_one", "asset_other_same_bytes", &sha).is_err());
        assert!(owner_matches(dir.path(), "yue2run_other", "asset_one", &sha).is_err());
    }

    #[test]
    fn long_wav_and_compressed_source_are_probed_without_reading_payload() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("long.wav");
        let frames = 600u64 * 48_000;
        let bytes = frames * 2 * 2;
        let mut file = std::fs::File::create(&wav).unwrap();
        file.write_all(b"RIFF").unwrap();
        file.write_all(&(36u32 + bytes as u32).to_le_bytes())
            .unwrap();
        file.write_all(b"WAVEfmt ").unwrap();
        file.write_all(&16u32.to_le_bytes()).unwrap();
        file.write_all(&1u16.to_le_bytes()).unwrap();
        file.write_all(&2u16.to_le_bytes()).unwrap();
        file.write_all(&48_000u32.to_le_bytes()).unwrap();
        file.write_all(&192_000u32.to_le_bytes()).unwrap();
        file.write_all(&4u16.to_le_bytes()).unwrap();
        file.write_all(&16u16.to_le_bytes()).unwrap();
        file.write_all(b"data").unwrap();
        file.write_all(&(bytes as u32).to_le_bytes()).unwrap();
        file.set_len(44 + bytes).unwrap();
        let info = crate::video_jobs::reference_audio::probe_source_wav(&wav)
            .unwrap()
            .unwrap();
        assert_eq!(info.frames, frames);
        assert_eq!(planned_frames(info, None), frames);
        assert_eq!(planned_frames(info, Some(10.0)), 11 * 48_000);
        let compressed = dir.path().join("take.webm");
        std::fs::write(&compressed, b"\x1a\x45\xdf\xa3webm header").unwrap();
        assert!(
            crate::video_jobs::reference_audio::probe_source_wav(&compressed)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn decode_attempts_with_the_same_job_id_never_share_or_delete_scratch() {
        let first = decode_scratch("job_1").unwrap();
        let second = decode_scratch("job_1").unwrap();
        let first_path = first.path().to_path_buf();
        let second_path = second.path().to_path_buf();
        assert_ne!(first_path, second_path);
        std::fs::write(first_path.join("reference.wav"), b"first").unwrap();
        std::fs::write(second_path.join("reference.wav"), b"second").unwrap();
        drop(first);
        assert!(!first_path.exists());
        assert_eq!(
            std::fs::read(second_path.join("reference.wav")).unwrap(),
            b"second"
        );
        drop(second);
        assert!(!second_path.exists());
    }

    /// Mutation evidence: changing `live == 0` to accept positive counts fails both assertions.
    #[test]
    fn generator_requires_all_transcribers_unloaded() {
        assert!(require_unloaded(0).is_ok());
        assert!(require_unloaded(1)
            .unwrap_err()
            .to_string()
            .contains("1 SheetSage2"));
        assert!(require_unloaded(2).is_err());
    }

    /// A catalog without either pinned cover component is refused before the backend is entered.
    #[test]
    fn cover_snapshot_resolution_fails_closed_on_missing_component() {
        let settings = Settings::from_env();
        let entry = json!({"conditionalComponents": []});
        let error = cover_snapshots(&settings, &entry).unwrap_err().to_string();
        assert!(error.contains("yue2_sheetsage2"), "{error}");
    }
}
