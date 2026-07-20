//! Pure audio generation — the SceneWorks Audio Studio job path (epic 13400 / sc-13404).
//!
//! The audio analogue of [`crate::video_jobs::run_video_generate_job`], deliberately much smaller:
//! audio has no fit gate, no VRAM gate, no mode/route ladder. `run_audio_generate_job` resolves the
//! model's cached snapshot dir, loads the generator from the runtime's **candle audio registry**
//! (`crate::inference_runtime::load_audio` → `catalog().audio()`, a separate lane from the mlx media
//! graph — sc-12835), builds a [`GenerationRequest`] with the typed [`AudioParams`] sub-block, runs
//! it on a blocking thread, writes the produced [`AudioTrack`] to a WAV with the shared
//! [`write_wav_pcm16`] writer (the same one LTX synchronized audio uses), and registers the result
//! as a `type: "audio"` asset through the ordinary `assetWrites` streaming-result contract — exactly
//! how a video job registers its clip, so `resolveJobResultAssets` / the library see it as audio.
//!
//! The **language-casing seam** (sc-13404): the manifest declares languages in BCP-47 display casing
//! (`"en-US"`) but the Generator's advertised `audio_languages` are lowercase (`"en"` / `"en-us"` /
//! `"en-gb"`). [`normalize_audio_language`] lowercases the request's language so the shared gen-core
//! validation floor accepts it instead of rejecting an advertised value.

use super::*;

use gen_core::{
    AudioParams, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

use crate::video_jobs::{write_wav_pcm16, AudioTrack};

const CANCEL_MESSAGE: &str = "Audio generation canceled by user.";
/// Adapter id recorded on the asset when the manifest declares no family — the audio twin of the
/// video/image adapter labels.
const AUDIO_ADAPTER_FALLBACK: &str = "audio";

/// The parsed audio job payload — the audio twin of `VideoRequest`, kept local to the worker (audio
/// has far fewer knobs than video, so it needs no shared-core struct). Infallible parse: missing
/// fields fall back to sane defaults and `project_id` may be empty — the preflight rejects that.
struct AudioRequest {
    project_id: String,
    model: String,
    prompt: String,
    voice: Option<String>,
    language: Option<String>,
    target_duration_secs: Option<f32>,
    seed: Option<i64>,
    model_manifest_entry: Value,
}

impl AudioRequest {
    fn from_payload(payload: &JsonObject) -> Self {
        let string = |key: &str| {
            payload
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        };
        Self {
            project_id: string("projectId").unwrap_or_default(),
            model: string("model").unwrap_or_else(|| "kokoro_82m".to_owned()),
            prompt: payload
                .get("prompt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            voice: string("voice"),
            language: string("language"),
            target_duration_secs: payload
                .get("targetDurationSecs")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            seed: payload.get("seed").and_then(Value::as_i64),
            model_manifest_entry: payload
                .get("modelManifestEntry")
                .cloned()
                .unwrap_or_else(|| json!({})),
        }
    }

    /// The asset's audio family, from the resolved manifest entry when present (Kokoro's `"kokoro"`),
    /// else a neutral fallback — parity with `resolve_family` on the video path.
    fn family(&self) -> String {
        self.model_manifest_entry
            .get("family")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| "audio".to_owned())
    }
}

/// Per-job invariants for the single audio clip this job produces — the audio twin of `VideoPlan`.
struct AudioPlan {
    genset_id: String,
    asset_id: String,
    created_at: String,
    family: String,
    /// `assets/audios/<genset>/<date>_<model>_<slug>.wav` (project-relative).
    media_rel: String,
    /// Absolute path to the media file.
    media_path: PathBuf,
}

impl AudioPlan {
    fn new(request: &AudioRequest, project_path: &Path) -> Self {
        let genset_id = format!("genset_{}", Uuid::new_v4().simple());
        let asset_id = fresh_asset_id();
        let created_at = now_rfc3339();
        let family = request.family();
        let slug = slugify(&request.prompt, "audio", Some(42));
        // Sanitize the untrusted model id before it becomes a path component (F-003 / sc-11159) —
        // slugify collapses any separator/`..` to a single readable component, mirroring VideoPlan.
        let model_slug = slugify(&request.model, "model", None);
        let media_rel = format!(
            "assets/audios/{genset_id}/{}_{}_{slug}.wav",
            &created_at[..10],
            model_slug
        );
        let media_path = project_path.join(&media_rel);
        Self {
            genset_id,
            asset_id,
            created_at,
            family,
            media_rel,
            media_path,
        }
    }
}

/// Map a manifest-cased language (`"en-US"`) to what the audio Generator advertises (`"en"` /
/// `"en-us"` / `"en-gb"` — lowercase). The shared gen-core validation floor gates the request's
/// language against the descriptor's `audio_languages` (all lowercase), so passing the manifest's
/// display casing verbatim would be rejected as an unadvertised value (sc-13404).
fn normalize_audio_language(language: &str) -> String {
    language.trim().to_lowercase()
}

/// Resolve the model's cached snapshot directory from its manifest entry — the Hugging Face repo the
/// model downloads from (e.g. `hexgrad/Kokoro-82M`, whose snapshot carries `config.json` +
/// `kokoro-v1_0.pth` + `voices/`). Uses the same `resolve_app_managed_model_dir` seam the other
/// worker jobs use, so it finds the HF-cache snapshot the manifest's normal install path populated.
fn resolve_audio_model_dir(settings: &Settings, request: &AudioRequest) -> WorkerResult<PathBuf> {
    let repo = audio_model_repo(&request.model_manifest_entry).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{}: the model manifest entry declares no Hugging Face download repo, so its audio \
             weights cannot be resolved.",
            request.model
        ))
    })?;
    crate::paths::resolve_app_managed_model_dir(settings, &repo, "Audio model")
}

/// The Hugging Face repo hosting an audio model's weights — the first `huggingface` download entry,
/// falling back to stripping the `${HF_CACHE}/` prefix off `paths.model`.
fn audio_model_repo(entry: &Value) -> Option<String> {
    if let Some(downloads) = entry.get("downloads").and_then(Value::as_array) {
        for download in downloads {
            // A download entry with no explicit provider is treated as huggingface (the manifest
            // default), so a repo without the key is still resolvable.
            let is_hf = match download.get("provider").and_then(Value::as_str) {
                Some(provider) => provider == "huggingface",
                None => true,
            };
            if is_hf {
                if let Some(repo) = download
                    .get("repo")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|repo| !repo.is_empty())
                {
                    return Some(repo.to_owned());
                }
            }
        }
    }
    entry
        .get("paths")
        .and_then(|paths| paths.get("model"))
        .and_then(Value::as_str)
        .and_then(|model| model.strip_prefix("${HF_CACHE}/"))
        .map(str::trim)
        .filter(|repo| !repo.is_empty())
        .map(str::to_owned)
}

fn audio_preflight(request: &AudioRequest) -> WorkerResult<()> {
    if request.project_id.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "projectId is required.".to_owned(),
        ));
    }
    if request.prompt.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "prompt (the script text) is required.".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) async fn run_audio_generate_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = AudioRequest::from_payload(&job.payload);
    audio_preflight(&request)?;
    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let plan = AudioPlan::new(&request, &project_path);
    if let Some(parent) = plan.media_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let backend = backend_label(&settings.gpu_id);
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        audio_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Preparing audio.",
            None,
            backend,
        ),
    )
    .await?;

    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    let model_dir = resolve_audio_model_dir(settings, &request)?;

    update_job(
        api,
        &job.id,
        audio_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.2,
            "Synthesizing audio.",
            None,
            backend,
        ),
    )
    .await?;

    // Synthesis (load + generate) is CPU/GPU-bound and synchronous — run it on the blocking pool so
    // the worker's async runtime stays responsive, and emit periodic keepalive heartbeats + progress
    // while it runs so a long synthesis (a cold pipeline build, a 30 s clip, or a slow host) is never
    // flagged stale and marked `interrupted` mid-flight. Mirrors the video path's interval keepalive
    // for the no-progress cold-load phase.
    let track = run_audio_synthesis(api, settings, job, backend, &request, model_dir).await?;

    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    update_job(
        api,
        &job.id,
        audio_progress(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.9,
            "Saving audio.",
            None,
            backend,
        ),
    )
    .await?;

    // Measure the produced clip BEFORE the samples move into the WAV writer, so the asset fact
    // records the honest length/rate of the file on disk (the audio twin of `EncodedClip::measure`).
    let sample_rate = track.sample_rate.max(1);
    let channels = track.channels.max(1);
    let sample_count = track.samples.len();
    let duration_secs = sample_count as f64 / (sample_rate as f64 * channels as f64);
    let peak = track
        .samples
        .iter()
        .fold(0.0f32, |max, &sample| max.max(sample.abs()));
    if sample_count == 0 || peak == 0.0 {
        return Err(WorkerError::Engine(format!(
            "{}: the audio generator produced silence ({} samples, peak {peak}) — refusing to \
             register an empty clip.",
            request.model, sample_count
        )));
    }

    let wav = AudioTrack {
        samples: track.samples,
        sample_rate,
        channels,
    };
    let media_path = plan.media_path.clone();
    tokio::task::spawn_blocking(move || write_wav_pcm16(&wav, &media_path))
        .await
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;

    let fact = audio_asset_fact(&plan, &request, sample_rate, channels, duration_secs);
    let result = audio_streaming_result(&plan, &request, &fact);
    update_job(
        api,
        &job.id,
        audio_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Generated audio.",
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// Keepalive cadence while the blocking synthesis runs (seconds). Comfortably under the API's
/// default 90 s worker-timeout, so a long synthesis is never swept to `interrupted`.
const AUDIO_KEEPALIVE_SECS: u64 = 15;

/// Load the audio generator and run one synthesis on a blocking thread, emitting periodic keepalive
/// heartbeats + progress updates while it runs. Returns the engine [`AudioTrack`] (`gen_core`'s) for
/// the caller to write + register.
async fn run_audio_synthesis(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    backend: &str,
    request: &AudioRequest,
    model_dir: PathBuf,
) -> WorkerResult<gen_core::AudioTrack> {
    let model_id = request.model.clone();
    let prompt = request.prompt.clone();
    let voice = request.voice.clone();
    let language = request.language.as_deref().map(normalize_audio_language);
    let target_duration = request.target_duration_secs;
    let seed = request.seed.map(|seed| seed as u64);
    let handle = tokio::task::spawn_blocking(move || -> WorkerResult<gen_core::AudioTrack> {
        let spec = LoadSpec::new(WeightsSource::Dir(model_dir));
        let generator = crate::inference_runtime::load_audio(&model_id, &spec)
            .map_err(|error| crate::classify_engine_error("audio model load failed", error))?;
        let req = GenerationRequest {
            prompt,
            seed,
            audio: Some(AudioParams {
                voice,
                language,
                target_duration,
                ..Default::default()
            }),
            cancel: CancelFlag::new(),
            ..Default::default()
        };
        // The Audio Studio progress is driven around this call (Preparing → Generating → Saving);
        // the per-stage engine callback is a no-op here — the async keepalive below is what keeps the
        // job alive during synthesis, so the engine progress doesn't need to be forwarded.
        let mut on_progress = |_progress: Progress| {};
        let output = generator
            .generate(&req, &mut on_progress)
            .map_err(|error| crate::classify_engine_error("audio generation failed", error))?;
        match output {
            GenerationOutput::Audio(track) => Ok(track),
            GenerationOutput::Images(_) => Err(WorkerError::Engine(
                "audio model returned images, expected an audio track".to_owned(),
            )),
            GenerationOutput::Video { .. } => Err(WorkerError::Engine(
                "audio model returned video, expected an audio track".to_owned(),
            )),
        }
    });
    tokio::pin!(handle);
    let mut ticker = tokio::time::interval(Duration::from_secs(AUDIO_KEEPALIVE_SECS));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    // The first tick fires immediately; skip it so the first keepalive lands one interval in (the
    // job was already marked Generating by the caller just before this).
    ticker.tick().await;
    loop {
        tokio::select! {
            result = &mut handle => {
                return result.map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
            }
            _ = ticker.tick() => {
                // Best-effort keepalive: a failed heartbeat/update must not abort a synthesis that is
                // otherwise progressing (the job completes on the next successful post).
                let _ = heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await;
                let _ = update_job(
                    api,
                    &job.id,
                    audio_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        0.2,
                        "Synthesizing audio.",
                        None,
                        backend,
                    ),
                )
                .await;
            }
        }
    }
}

/// The `type: "audio"` asset fact the API persists into a sidecar (via
/// `project_store::build_audio_sidecar_parts`) — the audio twin of `video_asset_fact`. Carries the
/// MEASURED clip facts (duration / sampleRate / channels off the produced track) plus the REQUESTED
/// knobs (voice / language / targetDurationSecs) the replay path round-trips.
fn audio_asset_fact(
    plan: &AudioPlan,
    request: &AudioRequest,
    sample_rate: u32,
    channels: u16,
    duration_secs: f64,
) -> Value {
    let title: String = request.prompt.chars().take(56).collect();
    let title = title.trim();
    let display_name = if title.is_empty() {
        "Generated audio".to_owned()
    } else {
        title.to_owned()
    };
    let adapter = request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(AUDIO_ADAPTER_FALLBACK);
    json!({
        "type": "audio",
        "assetId": plan.asset_id,
        "mediaPath": plan.media_rel,
        "mimeType": "audio/wav",
        // MEASURED off the produced track — the honest running time + PCM shape of the WAV on disk.
        "duration": duration_secs,
        "sampleRate": sample_rate,
        "channels": channels,
        "family": plan.family,
        "displayName": display_name,
        "createdAt": plan.created_at,
        "mode": "text_to_audio",
        "model": request.model,
        "adapter": adapter,
        "prompt": request.prompt,
        // REQUESTED knobs (the replay record). `null` for an omitted voice/language/duration — the
        // studio falls back to the model's own defaults, exactly as the video recipe does.
        "voice": request.voice,
        "language": request.language,
        "targetDurationSecs": request.target_duration_secs,
        "seed": request.seed,
        "rawAdapterSettings": {
            "model": request.model,
            "voice": request.voice,
            "language": request.language,
            "targetDurationSecs": request.target_duration_secs,
            "sampleRate": sample_rate,
        },
    })
}

/// The job-result shape the API streams from: `assetWrites` + the `generationSet` fact — the audio
/// twin of `streaming_result`. An audio job always reports exactly one asset (`expectedCount` 1).
fn audio_streaming_result(plan: &AudioPlan, request: &AudioRequest, fact: &Value) -> JsonObject {
    json!({
        "generationSetId": plan.genset_id,
        "expectedCount": 1,
        "adapter": fact.get("adapter").cloned().unwrap_or(Value::Null),
        "model": request.model,
        "generationSet": {
            "id": plan.genset_id,
            "mode": "text_to_audio",
            "model": request.model,
            "prompt": request.prompt,
            "count": 1,
            "createdAt": plan.created_at,
        },
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// Progress payload with the worker's real backend label — mirrors `video_progress`.
fn audio_progress(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn kokoro_entry() -> Value {
        json!({
            "id": "kokoro_82m",
            "type": "audio",
            "family": "kokoro",
            "downloads": [{ "provider": "huggingface", "repo": "hexgrad/Kokoro-82M", "files": ["config.json"] }],
            "paths": { "model": "${HF_CACHE}/hexgrad/Kokoro-82M" },
        })
    }

    fn payload(extra: Value) -> JsonObject {
        let mut base = json!({
            "projectId": "project-1",
            "model": "kokoro_82m",
            "prompt": "Hello from SceneWorks audio.",
            "modelManifestEntry": kokoro_entry(),
        });
        if let (Some(base), Some(extra)) = (base.as_object_mut(), extra.as_object()) {
            for (key, value) in extra {
                base.insert(key.clone(), value.clone());
            }
        }
        base.as_object().cloned().unwrap()
    }

    #[test]
    fn from_payload_reads_the_audio_knobs() {
        let request = AudioRequest::from_payload(&payload(json!({
            "voice": "bm_george",
            "language": "en-GB",
            "targetDurationSecs": 4.5,
            "seed": 7,
        })));
        assert_eq!(request.project_id, "project-1");
        assert_eq!(request.model, "kokoro_82m");
        assert_eq!(request.prompt, "Hello from SceneWorks audio.");
        assert_eq!(request.voice.as_deref(), Some("bm_george"));
        assert_eq!(request.language.as_deref(), Some("en-GB"));
        assert_eq!(request.target_duration_secs, Some(4.5));
        assert_eq!(request.seed, Some(7));
        assert_eq!(request.family(), "kokoro");
    }

    #[test]
    fn language_casing_seam_lowercases_for_the_generator() {
        // The manifest declares "en-US"/"en-GB"; the Generator advertises lowercase — normalize so
        // the shared validation floor accepts an advertised value (sc-13404).
        assert_eq!(normalize_audio_language("en-US"), "en-us");
        assert_eq!(normalize_audio_language("en-GB"), "en-gb");
        assert_eq!(normalize_audio_language("en"), "en");
        assert_eq!(normalize_audio_language("  EN-US "), "en-us");
    }

    #[test]
    fn model_repo_prefers_the_huggingface_download_then_paths_model() {
        assert_eq!(
            audio_model_repo(&kokoro_entry()).as_deref(),
            Some("hexgrad/Kokoro-82M")
        );
        // Falls back to `paths.model` (stripping the `${HF_CACHE}/` prefix) when downloads are absent.
        let paths_only = json!({ "paths": { "model": "${HF_CACHE}/hexgrad/Kokoro-82M" } });
        assert_eq!(
            audio_model_repo(&paths_only).as_deref(),
            Some("hexgrad/Kokoro-82M")
        );
        // No download repo and no paths.model → None (the handler turns this into a clear error).
        assert_eq!(audio_model_repo(&json!({ "type": "audio" })), None);
    }

    #[test]
    fn preflight_requires_project_and_prompt() {
        assert!(audio_preflight(&AudioRequest::from_payload(&payload(json!({})))).is_ok());
        let no_project = AudioRequest::from_payload(&payload(json!({ "projectId": "" })));
        assert!(audio_preflight(&no_project).is_err());
        let no_prompt = AudioRequest::from_payload(&payload(json!({ "prompt": "   " })));
        assert!(audio_preflight(&no_prompt).is_err());
    }

    #[test]
    fn asset_fact_carries_measured_and_requested_facts() {
        let request = AudioRequest::from_payload(&payload(json!({
            "voice": "af_heart",
            "language": "en-US",
            "targetDurationSecs": 3.0,
            "seed": 42,
        })));
        let plan = AudioPlan::new(&request, Path::new("/tmp/project"));
        let fact = audio_asset_fact(&plan, &request, 24_000, 1, 3.25);
        assert_eq!(fact["type"], "audio");
        assert_eq!(fact["mimeType"], "audio/wav");
        assert_eq!(fact["sampleRate"], 24_000);
        assert_eq!(fact["channels"], 1);
        assert_eq!(fact["duration"], 3.25);
        assert_eq!(fact["voice"], "af_heart");
        assert_eq!(fact["language"], "en-US");
        assert_eq!(fact["targetDurationSecs"], 3.0);
        assert_eq!(fact["seed"], 42);
        assert_eq!(fact["model"], "kokoro_82m");
        assert_eq!(fact["adapter"], "kokoro");
        assert!(fact["mediaPath"]
            .as_str()
            .is_some_and(|path| path.starts_with("assets/audios/") && path.ends_with(".wav")));
        // The streaming result wraps the fact as the sole assetWrite.
        let result = audio_streaming_result(&plan, &request, &fact);
        assert_eq!(result["expectedCount"], 1);
        assert_eq!(result["assetWrites"].as_array().map(Vec::len), Some(1));
        assert_eq!(result["assetWrites"][0]["type"], "audio");
    }
}
