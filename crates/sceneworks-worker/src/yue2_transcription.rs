//! CPU-only SheetSage2 transcription for YuE2 recording jobs. The backend seam keeps the job
//! lifecycle testable without downloading the cover closure or running a multi-GiB model.

use super::*;
use sceneworks_core::yue2_score::jobs::{self as contract, Yue2JobSpec};
use std::sync::{OnceLock, RwLock};

const CANCEL_MESSAGE: &str = "YuE2 transcription canceled by user.";

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
pub(crate) trait TranscriptionBackend: Send + Sync + 'static {
    fn transcribe(
        &self,
        snapshots: &[(String, PathBuf)],
        audio: gen_core::AudioTrack,
        source_sha256: &str,
        spec: &Yue2JobSpec,
        artifact_dir: &Path,
        canceled: &gen_core::CancelFlag,
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
        snapshots: &[(String, PathBuf)],
        audio: gen_core::AudioTrack,
        source_sha256: &str,
        spec: &Yue2JobSpec,
        artifact_dir: &Path,
        canceled: &gen_core::CancelFlag,
        progress: &mut dyn FnMut(f64, String),
    ) -> WorkerResult<Value> {
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
        _: &[(String, PathBuf)],
        _: gen_core::AudioTrack,
        _: &str,
        _: &Yue2JobSpec,
        _: &Path,
        _: &gen_core::CancelFlag,
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
fn published_details(
    dir: &Path,
    source_sha256: &str,
    spec: &Yue2JobSpec,
) -> WorkerResult<Option<Value>> {
    use crate::inference_runtime::audio_providers::candle_audio_sheetsage2 as ss2;
    if !dir.exists() {
        return Ok(None);
    }
    let artifact = ss2::review::ReviewArtifact::open(dir).map_err(|e| {
        WorkerError::InvalidPayload(format!(
            "yue2: existing transcription artifact does not verify: {e}"
        ))
    })?;
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
fn published_details(dir: &Path, _: &str, _: &Yue2JobSpec) -> WorkerResult<Option<Value>> {
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
pub(crate) async fn run<B: TranscriptionBackend>(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    spec: &Yue2JobSpec,
    project_id: &str,
    project_path: &Path,
    entry: &Value,
    usage_policy: &Value,
    backend: B,
) -> WorkerResult<()> {
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
    let encoded = tokio::fs::read(&source_path).await?;
    if super::yue2_jobs::sha256_hex(&encoded) != source.sha256 {
        return Err(WorkerError::InvalidPayload(
            "yue2: source recording no longer matches its queued SHA-256".into(),
        ));
    }
    let run_id = spec.run_id.as_deref().unwrap_or_default();
    let rel = contract::transcription_dir(run_id);
    let dir = project_path.join(&rel);
    let _claim = super::yue2_jobs::RunClaim::acquire(&dir)?;
    if let Some(details) = published_details(&dir, &source.sha256, spec)? {
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
    let scratch = std::env::temp_dir().join(format!(
        "sw-yue2-transcription-{}",
        super::safe_download_dir(&job.id)
    ));
    let _ = tokio::fs::remove_dir_all(&scratch).await;
    tokio::fs::create_dir_all(&scratch).await?;
    let decoded = crate::video_jobs::reference_audio::decode_audio_normalized(
        api,
        settings,
        &job.id,
        CANCEL_MESSAGE,
        &source_path,
        &scratch,
        crate::video_jobs::reference_audio::AudioDecode {
            float32: true,
            ..Default::default()
        },
    )
    .await;
    let _ = tokio::fs::remove_dir_all(&scratch).await;
    let audio = decoded?;
    let duration = audio.samples.len() as f64
        / f64::from(audio.sample_rate.max(1))
        / f64::from(audio.channels.max(1));
    let seconds = spec
        .transcription
        .and_then(|s| s.max_seconds)
        .map_or(duration, |max| duration.min(max))
        .ceil() as u64;
    let lease =
        crate::yue2_admission::check_transcription(entry, seconds, &settings.gpu_id).await?;
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
            &snapshots,
            audio,
            &source_sha,
            &spec_copy,
            &partial,
            &task_cancel,
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
        return Err(WorkerError::InvalidPayload(
            "yue2: this transcription artifact already exists".into(),
        ));
    }
    std::fs::rename(super::yue2_jobs::partial_dir(&dir), &dir)?;
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
