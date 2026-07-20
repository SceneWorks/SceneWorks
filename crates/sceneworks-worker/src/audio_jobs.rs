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
    AudioEditMode, AudioParams, AudioTransform, AudioTransformRequest, CancelFlag, Conditioning,
    GenerationOutput, GenerationRequest, Generator, LoadSpec, Progress, TimeRegion, WeightsSource,
};

use crate::video_jobs::{write_wav_pcm16, AudioTrack};

const CANCEL_MESSAGE: &str = "Audio generation canceled by user.";
/// Adapter id recorded on the asset when the manifest declares no family — the audio twin of the
/// video/image adapter labels.
const AUDIO_ADAPTER_FALLBACK: &str = "audio";

/// Classify an error surfacing from a synthesis `generate` / `apply` call (sc-13469). A cooperative
/// mid-synthesis bail on the tripped [`CancelFlag`] arrives as [`gen_core::Error::Canceled`], which
/// MUST surface as [`WorkerError::Canceled`] so [`run_blocking_with_heartbeat`] posts the terminal
/// `Canceled` and the job reads as canceled — [`crate::classify_engine_error`] would otherwise bucket
/// it as a generic `Engine` failure (its match only special-cases `Unsupported`), turning a user
/// cancel into a spurious job failure. Every other engine error keeps the ordinary classification.
fn classify_audio_synthesis_error(context: &str, error: gen_core::Error) -> WorkerError {
    if matches!(error, gen_core::Error::Canceled) {
        WorkerError::Canceled(CANCEL_MESSAGE.to_owned())
    } else {
        crate::classify_engine_error(context, error)
    }
}

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
    /// CFG guidance scale for diffusion-audio models (Sound FX / MOSS-SoundEffect). Rides the
    /// top-level `GenerationRequest::guidance` — NOT `AudioParams` (sc-13409). `None` ⇒ the model's
    /// sampler default.
    guidance: Option<f32>,
    /// Solver step count for diffusion-audio models. Rides the top-level `GenerationRequest::steps`.
    /// `None` ⇒ the model's sampler default.
    steps: Option<u32>,
    /// Negative prompt for music models that advertise it. Rides the top-level
    /// `GenerationRequest::negative_prompt` (sc-13410). `None` ⇒ unconditional. The guidance-distilled
    /// ACE-Step turbo advertises no support, so the studio never sends one to it.
    negative_prompt: Option<String>,
    /// Musical tempo (BPM) — rides `AudioParams::bpm` (music models). `None` ⇒ the model's own tempo.
    bpm: Option<f32>,
    /// Musical key (e.g. `"C minor"`) — rides `AudioParams::musical_key`. `None` ⇒ the model's own.
    musical_key: Option<String>,
    /// Lyrics — rides `AudioParams::lyrics` (empty ⇒ instrumental).
    lyrics: Option<String>,
    /// Extend/edit source track asset id — resolved to a WAV and built into a
    /// `Conditioning::AudioEdit` (sc-13410). `None` ⇒ plain text-to-music.
    source_audio_asset_id: Option<String>,
    /// Edit operation for the source track (`inpaint` / `repaint` / `extend` / `cover`).
    edit_mode: Option<String>,
    /// Edit-region start (seconds) — inpaint/repaint window start; for `extend` the worker defaults it
    /// to the source clip's own length.
    edit_region_start_secs: Option<f32>,
    /// Edit-region end (seconds) — inpaint/repaint window end, OR (for `extend`) the new total length.
    edit_region_end_secs: Option<f32>,
    /// Edit strength (0..=1). `None` ⇒ the model default.
    edit_strength: Option<f32>,
    /// Voice Clone (sc-13411 C4): the reference-voice library `type: "audio"` asset whose timbre is
    /// transferred onto the base TTS clip. Its presence is the discriminator that routes this job onto
    /// the two-call Kokoro→OpenVoice chain (`run_voice_clone_synthesis`) instead of the single-generator
    /// path. `None` ⇒ an ordinary Speech/SFX/Music generation.
    reference_audio_asset_id: Option<String>,
    /// Voice Clone base TTS model id (the "content" generator whose speech OpenVoice re-timbres). The API
    /// injects its manifest entry as `baseModelManifestEntry`. Defaults to Kokoro (`kokoro_82m`).
    base_model: String,
    /// Voice Clone match strength — overrides OpenVoice V2's posterior-sampling temperature τ (rides
    /// `AudioTransformRequest::strength`). `None` ⇒ the converter's own default (τ = 0.3).
    match_strength: Option<f32>,
    /// The resolved manifest entry for [`base_model`] (the base TTS), injected by the API on a voice-clone
    /// request so the worker resolves the base generator's snapshot without re-parsing the jsonc. `{}` on a
    /// non-voice-clone job.
    base_model_manifest_entry: Value,
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
            guidance: payload
                .get("guidance")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            steps: payload
                .get("steps")
                .and_then(Value::as_u64)
                .map(|value| value as u32),
            negative_prompt: string("negativePrompt"),
            bpm: payload
                .get("bpm")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            musical_key: string("musicalKey"),
            // Lyrics are free-form and may legitimately be multi-line/whitespace-led ([verse] tags),
            // so read them verbatim (unlike the trimmed `string()` helper) — only an entirely-absent
            // key ⇒ None (instrumental).
            lyrics: payload
                .get("lyrics")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(str::to_owned),
            source_audio_asset_id: string("sourceAudioAssetId"),
            edit_mode: string("editMode").map(|mode| mode.to_lowercase()),
            edit_region_start_secs: payload
                .get("editRegionStartSecs")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            edit_region_end_secs: payload
                .get("editRegionEndSecs")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            edit_strength: payload
                .get("editStrength")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            reference_audio_asset_id: string("referenceAudioAssetId"),
            base_model: string("baseModel").unwrap_or_else(|| "kokoro_82m".to_owned()),
            match_strength: payload
                .get("matchStrength")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            base_model_manifest_entry: payload
                .get("baseModelManifestEntry")
                .cloned()
                .unwrap_or_else(|| json!({})),
            seed: payload.get("seed").and_then(Value::as_i64),
            model_manifest_entry: payload
                .get("modelManifestEntry")
                .cloned()
                .unwrap_or_else(|| json!({})),
        }
    }

    /// A voice-clone job carries a non-empty reference-voice asset id — the discriminator that routes it
    /// onto the two-call Kokoro→OpenVoice chain rather than the single-generator path.
    fn is_voice_clone(&self) -> bool {
        self.reference_audio_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
    }

    /// The generation mode recorded on the asset + generation set — `"voice_clone"` for a reference-
    /// driven job, else `"text_to_audio"` (Speech / SFX / Music).
    fn mode(&self) -> &'static str {
        if self.is_voice_clone() {
            "voice_clone"
        } else {
            "text_to_audio"
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
    resolve_audio_model_dir_for(settings, &request.model_manifest_entry, &request.model)
}

/// Resolve an audio model's cached snapshot directory from an explicit manifest entry + id — the
/// generalization of [`resolve_audio_model_dir`] the voice-clone chain uses to resolve BOTH its base
/// TTS model (`baseModelManifestEntry`) and its converter (`modelManifestEntry`).
fn resolve_audio_model_dir_for(
    settings: &Settings,
    entry: &Value,
    model_id: &str,
) -> WorkerResult<PathBuf> {
    let repo = audio_model_repo(entry).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{model_id}: the model manifest entry declares no Hugging Face download repo, so its \
             audio weights cannot be resolved."
        ))
    })?;
    crate::paths::resolve_app_managed_model_dir(settings, &repo, "Audio model")
}

/// OpenVoice V2's converter weights (`config.json` + `checkpoint.pth`) live under a `converter/`
/// subdirectory of its `myshell-ai/OpenVoiceV2` snapshot (the manifest downloads them as
/// `converter/*`), whereas the transform's `load` expects the directory that directly holds those two
/// files. Descend into `converter/` when it carries the checkpoint; otherwise pass the root through
/// (a future flat-layout repack still loads).
fn openvoice_converter_dir(root: PathBuf) -> PathBuf {
    let converter = root.join("converter");
    if converter.join("checkpoint.pth").is_file() {
        converter
    } else {
        root
    }
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

    // Voice Clone routing (sc-13411 C4 → sc-13412). A reference-voice job renders natively when the
    // selected clone model is a Generator that advertises `ReferenceAudio` conditioning — Chatterbox
    // `chatterbox_tts`: a SINGLE generator call clones from the script + reference. The choice is gated
    // purely on the audio catalog's registration + Capabilities (not a hardcoded id), so the native
    // path lights up automatically the moment such a generator is linked in; otherwise the two-call
    // Kokoro→OpenVoice conversion chain remains the fallback. Every other mode runs the single-generator
    // path. All resolve their snapshot dir(s) up front so a missing install fails with a clear error
    // before the job is marked Running.
    let synthesis: AudioSynthesis = if request.is_voice_clone() {
        if crate::inference_runtime::audio_generator_clones_from_reference(&request.model) {
            let plan = resolve_native_voice_clone_plan(settings, &request, &project_path)?;
            AudioSynthesis::NativeVoiceClone(plan)
        } else {
            let plan = resolve_voice_clone_plan(settings, &request, &project_path)?;
            AudioSynthesis::VoiceClone(plan)
        }
    } else {
        let model_dir = resolve_audio_model_dir(settings, &request)?;
        // Extend/edit SOURCE band (sc-13410): resolve + decode the source track and build the
        // `Conditioning::AudioEdit` here (in async, where the project path + store are in scope), then move
        // it into the blocking synthesis. `None` ⇒ plain text-to-music. The per-model gates (edit mode ∈
        // advertised, region inside the clip, 48 kHz source) run in the generator's `validate` at synthesis.
        let audio_edit = build_audio_edit(settings, &request, &project_path)?;
        AudioSynthesis::Single {
            model_dir,
            audio_edit,
        }
    };

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
    // the worker's async runtime stays responsive, and emit periodic keepalive heartbeats while it
    // runs so a long synthesis (a cold pipeline build, a 30 s clip, or a slow host) is never flagged
    // stale and marked `interrupted` mid-flight. Mirrors the video path's interval keepalive for the
    // no-progress cold-load phase. The voice-clone chain runs TWO backend calls (base TTS then
    // conversion) inside one blocking task, so the single keepalive loop spans both.
    // Whether this run took the native single-call clone path — threaded into the asset fact so the
    // replay record omits the (unused) base-TTS model that only the conversion chain carries.
    let native_clone = matches!(synthesis, AudioSynthesis::NativeVoiceClone(_));
    let track = match synthesis {
        AudioSynthesis::Single {
            model_dir,
            audio_edit,
        } => run_audio_synthesis(api, settings, job, &request, model_dir, audio_edit).await?,
        AudioSynthesis::VoiceClone(plan) => {
            run_voice_clone_synthesis(api, settings, job, &request, plan).await?
        }
        AudioSynthesis::NativeVoiceClone(plan) => {
            run_native_voice_clone_synthesis(api, settings, job, &request, plan).await?
        }
    };

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

    let fact = audio_asset_fact(
        &plan,
        &request,
        sample_rate,
        channels,
        duration_secs,
        native_clone,
    );
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

/// Load the audio generator and run one synthesis on a blocking thread, honoring a mid-synthesis
/// cancel through the shared [`run_blocking_with_heartbeat`] watcher (sc-13469). Returns the engine
/// [`AudioTrack`] (`gen_core`'s) for the caller to write + register.
async fn run_audio_synthesis(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &AudioRequest,
    model_dir: PathBuf,
    audio_edit: Option<Conditioning>,
) -> WorkerResult<gen_core::AudioTrack> {
    run_audio_synthesis_using(
        api,
        settings,
        job,
        request,
        model_dir,
        audio_edit,
        crate::inference_runtime::load_audio,
    )
    .await
}

/// [`run_audio_synthesis`] with the generator loader injected (sc-13469), the audio sibling of
/// [`crate::video_jobs::generate_video_using`]: with the load threaded in, a test drives the REAL
/// synthesis path against a stub [`Generator`] and asserts the shared engine [`CancelFlag`] that
/// reaches [`GenerationRequest::cancel`] is tripped mid-generation (not only by the post-synthesis
/// `check_cancel` after the whole clip has rendered).
#[allow(clippy::too_many_arguments)]
async fn run_audio_synthesis_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &AudioRequest,
    model_dir: PathBuf,
    audio_edit: Option<Conditioning>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<gen_core::AudioTrack> {
    let model_id = request.model.clone();
    let prompt = request.prompt.clone();
    let voice = request.voice.clone();
    let language = request.language.as_deref().map(normalize_audio_language);
    let target_duration = request.target_duration_secs;
    // Diffusion-audio sampling knobs (Sound FX / MOSS-SoundEffect, sc-13409). These live on the
    // top-level GenerationRequest — the flow-matching pipeline reads `req.guidance` (CFG scale) and
    // `req.steps`, not `AudioParams`. `None` ⇒ the generator's own sampler default; the shared
    // gen-core floor range-checks any value we pass. A TTS model (Kokoro) ignores them entirely, so
    // Speech jobs — which never carry them — are unaffected.
    let guidance = request.guidance;
    let steps = request.steps;
    // Music describe-the-music sub-block (ACE-Step, sc-13410). BPM/key/lyrics ride the AudioParams
    // music fields; negative_prompt rides the top-level request. A model that doesn't consume one
    // ignores it; a model that advertises no negative-prompt support rejects a supplied one at the
    // gen-core floor (the studio only sends one to a model that advertises support).
    let negative_prompt = request.negative_prompt.clone();
    let bpm = request.bpm;
    let musical_key = request.musical_key.clone();
    let lyrics = request.lyrics.clone();
    let seed = request.seed.map(|seed| seed as u64);
    // sc-13469: ONE shared engine CancelFlag — cloned into the request the blocking synthesis runs
    // AND handed to `run_blocking_with_heartbeat`, whose watcher polls the API cancel state (and a
    // worker shutdown) and trips it MID-synthesis. Replaces the degenerate inline `CancelFlag::new()`
    // that was never tripped, so a user cancel is honored DURING the clip render — not only by the
    // post-synthesis `check_cancel` after the whole clip has already rendered.
    let cancel = CancelFlag::new();
    let handle = {
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<gen_core::AudioTrack> {
            let spec = LoadSpec::new(WeightsSource::Dir(model_dir));
            let generator = load_generator(&model_id, &spec)
                .map_err(|error| crate::classify_engine_error("audio model load failed", error))?;
            let req = GenerationRequest {
                prompt,
                negative_prompt,
                seed,
                steps,
                guidance,
                audio: Some(AudioParams {
                    voice,
                    language,
                    target_duration,
                    bpm,
                    musical_key,
                    lyrics,
                    ..Default::default()
                }),
                // Extend/edit source-audio conditioning (Conditioning::AudioEdit), when a source band
                // was supplied; empty for plain text-to-music.
                conditioning: audio_edit.into_iter().collect(),
                // The shared, watcher-tripped flag (sc-13469) — NOT a fresh `CancelFlag::new()`.
                cancel,
                ..Default::default()
            };
            // The Audio Studio progress is driven around this call (Preparing → Generating → Saving);
            // the per-stage engine callback is a no-op here — the shared keepalive watcher is what
            // keeps the job alive during synthesis, so the engine progress doesn't need forwarding.
            let mut on_progress = |_progress: Progress| {};
            let output = generator
                .generate(&req, &mut on_progress)
                .map_err(|error| {
                    classify_audio_synthesis_error("audio generation failed", error)
                })?;
            match output {
                GenerationOutput::Audio(track) => Ok(track),
                GenerationOutput::Images(_) => Err(WorkerError::Engine(
                    "audio model returned images, expected an audio track".to_owned(),
                )),
                GenerationOutput::Video { .. } => Err(WorkerError::Engine(
                    "audio model returned video, expected an audio track".to_owned(),
                )),
            }
        })
    };
    // The shared blocking keepalive + cancel watcher (sc-13469): pings the worker heartbeat while the
    // synthesis runs (so a long/cold render is never swept to `interrupted`), polls the API cancel
    // state each interval and trips the SAME `cancel` the request carries, and on completion tears the
    // watcher down cleanly (bounded-join, no leaked task, no false-trip on normal completion). Mirrors
    // the video path's in-loop cancel watcher.
    run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        CANCEL_MESSAGE,
        "audio synthesis",
        no_cancel_ack(),
        handle,
    )
    .await
}

/// Which synthesis path a resolved audio job takes — the single-generator lane (Speech / SFX / Music)
/// or the two-call voice-clone chain (sc-13411 C4). Resolved before the job is marked Running so a
/// missing install / reference surfaces as a clear preflight error.
enum AudioSynthesis {
    Single {
        model_dir: PathBuf,
        audio_edit: Option<Conditioning>,
    },
    VoiceClone(VoiceClonePlan),
    /// Native cloned-voice TTS (sc-13412): a single-generator clone from the script + reference,
    /// chosen when the selected clone model is a Generator advertising `ReferenceAudio` conditioning
    /// (Chatterbox `chatterbox_tts`) instead of the two-call OpenVoice conversion chain.
    NativeVoiceClone(NativeVoiceClonePlan),
}

/// A resolved native clone-TTS job (sc-13412): the clone generator's snapshot dir + the decoded
/// reference-voice clip. Unlike [`VoiceClonePlan`] there is NO base-TTS/converter pair — the single
/// generator renders directly from the script + reference. Resolved in async (settings + project path
/// in scope) so the blocking synthesis gets ready-to-load inputs.
struct NativeVoiceClonePlan {
    model_dir: PathBuf,
    reference: gen_core::AudioTrack,
}

/// A resolved voice-clone job: the base TTS snapshot dir, the OpenVoice converter snapshot dir, and
/// the decoded reference-voice clip whose timbre is transferred. All three are resolved in async (where
/// the settings + project path are in scope) so the blocking chain gets ready-to-load inputs.
struct VoiceClonePlan {
    base_model_dir: PathBuf,
    converter_dir: PathBuf,
    reference: gen_core::AudioTrack,
}

/// Resolve the base TTS snapshot, the OpenVoice converter snapshot, and the reference-voice clip for a
/// voice-clone job (sc-13411 C4). Fails with a clear error when the reference asset can't be resolved to
/// a decodable WAV, the base/converter isn't installed, or the manifest entries are missing — all before
/// the job is marked Running.
fn resolve_voice_clone_plan(
    settings: &Settings,
    request: &AudioRequest,
    project_path: &Path,
) -> WorkerResult<VoiceClonePlan> {
    let base_model_dir = resolve_audio_model_dir_for(
        settings,
        &request.base_model_manifest_entry,
        &request.base_model,
    )?;
    let converter_root = resolve_audio_model_dir(settings, request)?;
    let converter_dir = openvoice_converter_dir(converter_root);
    let reference_id = request
        .reference_audio_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "a voice-clone job needs a referenceAudioAssetId.".to_owned(),
            )
        })?;
    // Resolve the reference through the same project-scoped guard the extend/edit source clip uses, then
    // decode its PCM-16 WAV into the host AudioTrack OpenVoice consumes as its tone-color target.
    let reference_path = crate::video_jobs::resolve_clip_media_path(
        settings,
        &request.project_id,
        reference_id,
        project_path,
    )?;
    let reference = read_wav_pcm16(&reference_path)?;
    Ok(VoiceClonePlan {
        base_model_dir,
        converter_dir,
        reference,
    })
}

/// Run the two-call voice-clone chain on a blocking thread (sc-13411 C4): base TTS (Kokoro) speaks the
/// script, then OpenVoice V2 transfers the reference clip's tone color onto that speech. This is the
/// product-layer orchestration of two backend calls — a single worker job so the library sees exactly
/// one asset (the converted clip). Returns the converted `gen_core::AudioTrack` for the caller to write
/// + register, exactly like [`run_audio_synthesis`].
async fn run_voice_clone_synthesis(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &AudioRequest,
    plan: VoiceClonePlan,
) -> WorkerResult<gen_core::AudioTrack> {
    run_voice_clone_synthesis_using(
        api,
        settings,
        job,
        request,
        plan,
        crate::inference_runtime::load_audio,
        crate::inference_runtime::load_audio_transform,
    )
    .await
}

/// [`run_voice_clone_synthesis`] with the base-TTS + converter loaders injected (sc-13469). Both
/// backend calls run inside ONE blocking task under ONE shared engine [`CancelFlag`] threaded into
/// BOTH the base [`GenerationRequest::cancel`] and the [`AudioTransformRequest::cancel`], so a cancel
/// requested while EITHER call is in-flight trips promptly — the two-call trap (the converter request
/// used to default `cancel` to a fresh, never-tripped flag via `..Default::default()`).
#[allow(clippy::too_many_arguments)]
async fn run_voice_clone_synthesis_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &AudioRequest,
    plan: VoiceClonePlan,
    load_base: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>> + Send + 'static,
    load_converter: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn AudioTransform>>
        + Send
        + 'static,
) -> WorkerResult<gen_core::AudioTrack> {
    let VoiceClonePlan {
        base_model_dir,
        converter_dir,
        reference,
    } = plan;
    let base_model = request.base_model.clone();
    let converter_model = request.model.clone();
    let script = request.prompt.clone();
    // The base TTS voice (which Kokoro voice speaks the script) + language ride the base generator's
    // AudioParams exactly as a Speech job does; OpenVoice then re-timbres the result toward the reference.
    let voice = request.voice.clone();
    let language = request.language.as_deref().map(normalize_audio_language);
    // Match strength overrides OpenVoice's posterior-sampling temperature τ; `None` ⇒ the converter's
    // own default (0.3). The converter range-checks it (finite, >= 0).
    let strength = request.match_strength;
    let seed = request.seed.map(|seed| seed as u64);
    // sc-13469: ONE shared engine CancelFlag spanning BOTH backend calls, tripped mid-synthesis by the
    // `run_blocking_with_heartbeat` watcher (see [`run_audio_synthesis_using`]).
    let cancel = CancelFlag::new();
    let handle = {
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<gen_core::AudioTrack> {
            // Call 1 — base TTS: synthesize the script in the requested base voice.
            let base_spec = LoadSpec::new(WeightsSource::Dir(base_model_dir));
            let base_generator = load_base(&base_model, &base_spec).map_err(|error| {
                crate::classify_engine_error("voice-clone base TTS load failed", error)
            })?;
            let base_req = GenerationRequest {
                prompt: script,
                audio: Some(AudioParams {
                    voice,
                    language,
                    ..Default::default()
                }),
                // The shared, watcher-tripped flag (sc-13469).
                cancel: cancel.clone(),
                ..Default::default()
            };
            let mut on_progress = |_progress: Progress| {};
            let base_track = match base_generator
                .generate(&base_req, &mut on_progress)
                .map_err(|error| {
                    classify_audio_synthesis_error("voice-clone base TTS failed", error)
                })? {
                GenerationOutput::Audio(track) => track,
                _ => {
                    return Err(WorkerError::Engine(
                        "voice-clone base TTS returned non-audio output".to_owned(),
                    ))
                }
            };

            // Call 2 — OpenVoice V2 tone-color conversion: transfer the reference clip's timbre onto the
            // base speech. The source (`audio`) carries content + prosody; `target_reference` is the voice.
            let converter_spec = LoadSpec::new(WeightsSource::Dir(converter_dir));
            let transform = load_converter(&converter_model, &converter_spec).map_err(|error| {
                crate::classify_engine_error("voice-clone converter load failed", error)
            })?;
            let transform_req = AudioTransformRequest {
                audio: base_track,
                target_reference: Some(reference),
                strength,
                seed,
                // The SAME shared flag also drives the converter call (sc-13469) — never the
                // `..Default::default()` fresh flag that would ignore a mid-conversion cancel.
                cancel,
                ..Default::default()
            };
            transform
                .apply(&transform_req, &mut on_progress)
                .map_err(|error| classify_audio_synthesis_error("voice conversion failed", error))?
                .into_iter()
                .next()
                .ok_or_else(|| WorkerError::Engine("voice conversion produced no track".to_owned()))
        })
    };
    run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        CANCEL_MESSAGE,
        "voice-clone synthesis",
        no_cancel_ack(),
        handle,
    )
    .await
}

/// Resolve the native clone-TTS snapshot dir + the reference-voice clip for a Voice Clone job that
/// routes onto the single-call native path (sc-13412). The clone generator (`chatterbox_tts`) renders
/// directly from the script + reference, so — unlike [`resolve_voice_clone_plan`] — there is NO
/// separate base-TTS model to resolve. Fails with a clear error before the job is marked Running when
/// the model isn't installed or the reference asset can't be resolved to a decodable WAV.
fn resolve_native_voice_clone_plan(
    settings: &Settings,
    request: &AudioRequest,
    project_path: &Path,
) -> WorkerResult<NativeVoiceClonePlan> {
    let model_dir = resolve_audio_model_dir(settings, request)?;
    let reference_id = request
        .reference_audio_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "a voice-clone job needs a referenceAudioAssetId.".to_owned(),
            )
        })?;
    // Resolve the reference through the same project-scoped guard the conversion chain + the
    // extend/edit source clip use, then decode its PCM-16 WAV into the host AudioTrack the clone
    // generator consumes as its `Conditioning::ReferenceAudio` voice.
    let reference_path = crate::video_jobs::resolve_clip_media_path(
        settings,
        &request.project_id,
        reference_id,
        project_path,
    )?;
    let reference = read_wav_pcm16(&reference_path)?;
    Ok(NativeVoiceClonePlan {
        model_dir,
        reference,
    })
}

/// Run the native clone-TTS synthesis on a blocking thread (sc-13412): a SINGLE Chatterbox generator
/// call renders the script in the reference voice. The reference clip rides as
/// [`Conditioning::ReferenceAudio`] — the provider derives the 256-d speaker embedding from it (the
/// `VoiceEmbedding` path) AND consumes the clip as S3Gen's reference mel / prompt tokens / speaker
/// x-vector, so this one call produces the full cloned WAV with no base-TTS + conversion pass. Returns
/// the produced `gen_core::AudioTrack` for the caller to write + register, exactly like
/// [`run_audio_synthesis`].
async fn run_native_voice_clone_synthesis(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &AudioRequest,
    plan: NativeVoiceClonePlan,
) -> WorkerResult<gen_core::AudioTrack> {
    run_native_voice_clone_synthesis_using(
        api,
        settings,
        job,
        request,
        plan,
        crate::inference_runtime::load_audio,
    )
    .await
}

/// [`run_native_voice_clone_synthesis`] with the clone-generator loader injected (sc-13469). The
/// shared engine [`CancelFlag`] reaches [`GenerationRequest::cancel`] and is tripped mid-synthesis by
/// the `run_blocking_with_heartbeat` watcher (see [`run_audio_synthesis_using`]).
async fn run_native_voice_clone_synthesis_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &AudioRequest,
    plan: NativeVoiceClonePlan,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<gen_core::AudioTrack> {
    let NativeVoiceClonePlan {
        model_dir,
        reference,
    } = plan;
    let model_id = request.model.clone();
    let script = request.prompt.clone();
    let seed = request.seed.map(|seed| seed as u64);
    // sc-13469: ONE shared engine CancelFlag, tripped mid-synthesis by the keepalive watcher.
    let cancel = CancelFlag::new();
    let handle = {
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<gen_core::AudioTrack> {
            let spec = LoadSpec::new(WeightsSource::Dir(model_dir));
            let generator = load_generator(&model_id, &spec).map_err(|error| {
                crate::classify_engine_error("clone-TTS model load failed", error)
            })?;
            let req = GenerationRequest {
                prompt: script,
                seed,
                // The reference clip is the sole voice conditioning — one native call renders the clone.
                // No `voice` (the model has no named voice bank), no `language`/`target_duration` (Voice
                // Clone carries none; the utterance is text-proportional), and no `matchStrength` (that τ
                // is the OpenVoice converter's, not this generator's).
                conditioning: vec![Conditioning::ReferenceAudio {
                    audio: reference,
                    strength: None,
                }],
                // The shared, watcher-tripped flag (sc-13469).
                cancel,
                ..Default::default()
            };
            let mut on_progress = |_progress: Progress| {};
            match generator
                .generate(&req, &mut on_progress)
                .map_err(|error| {
                    classify_audio_synthesis_error("clone-TTS generation failed", error)
                })? {
                GenerationOutput::Audio(track) => Ok(track),
                GenerationOutput::Images(_) => Err(WorkerError::Engine(
                    "clone-TTS model returned images, expected an audio track".to_owned(),
                )),
                GenerationOutput::Video { .. } => Err(WorkerError::Engine(
                    "clone-TTS model returned video, expected an audio track".to_owned(),
                )),
            }
        })
    };
    run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        CANCEL_MESSAGE,
        "clone-TTS synthesis",
        no_cancel_ack(),
        handle,
    )
    .await
}

/// Parse an edit-mode token (`inpaint` / `repaint` / `extend` / `cover`) into the gen-core
/// [`AudioEditMode`]. The API already rejects an unknown token up front; this is the worker-side
/// mirror so a raw-enqueued job still fails cleanly rather than mis-routing.
fn parse_audio_edit_mode(mode: &str) -> WorkerResult<AudioEditMode> {
    match mode {
        "inpaint" => Ok(AudioEditMode::Inpaint),
        "repaint" => Ok(AudioEditMode::Repaint),
        "extend" => Ok(AudioEditMode::Extend),
        "cover" => Ok(AudioEditMode::Cover),
        other => Err(WorkerError::InvalidPayload(format!(
            "unknown audio edit mode {other:?} (expected inpaint / repaint / extend / cover)"
        ))),
    }
}

/// Build the prompted source-audio-edit conditioning (sc-13410) from the request's source band, or
/// `None` for plain text-to-music. Resolves the source track asset to its WAV (guarded through
/// `resolve_clip_media_path`, the same project-scoped resolver the video source-clip path uses),
/// decodes it into a [`gen_core::AudioTrack`], and assembles the [`Conditioning::AudioEdit`] the
/// ACE-Step generator consumes: the source clip + the edit mode + a time region.
///
/// Region policy mirrors the [`AudioEditMode`] contract: `Extend` begins the appended tail at the
/// source clip's own length (unless the request overrides `start`) and reads `end` as the new total
/// length; `Inpaint`/`Repaint` carry the request's bounded window (`None` ⇒ the generator's own
/// default region check fires); `Cover` is whole-clip (no region). The per-model gates (mode ∈
/// advertised `audio_edit_modes`, region inside the clip duration, 48 kHz source) run in the
/// generator's `validate` at synthesis, so a bad region surfaces as a clear engine error there.
fn build_audio_edit(
    settings: &Settings,
    request: &AudioRequest,
    project_path: &Path,
) -> WorkerResult<Option<Conditioning>> {
    let Some(source_id) = request
        .source_audio_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let mode = request.edit_mode.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "an audio edit source was supplied without an editMode.".to_owned(),
        )
    })?;
    let mode = parse_audio_edit_mode(mode)?;
    let source_path = crate::video_jobs::resolve_clip_media_path(
        settings,
        &request.project_id,
        source_id,
        project_path,
    )?;
    let track = read_wav_pcm16(&source_path)?;
    // The clip's running length in seconds (interleaved: total samples / (rate · channels)).
    let src_secs = track.samples.len() as f32
        / (track.sample_rate.max(1) as f32 * track.channels.max(1) as f32);
    let region = audio_edit_region(
        mode,
        src_secs,
        request.edit_region_start_secs,
        request.edit_region_end_secs,
    );
    Ok(Some(Conditioning::AudioEdit {
        audio: track,
        mode,
        region,
        strength: request.edit_strength,
    }))
}

/// The edit-region policy (sc-13410), factored out of [`build_audio_edit`] so it is unit-testable
/// without a project store or a real clip. `Extend` begins the appended tail at the source clip's own
/// length when the request omits a start, and reads the request `end` as the new total length;
/// `Inpaint`/`Repaint` carry the request's bounded window (a missing start ⇒ `None`, so the
/// generator's own "region required" check fires); `Cover` is whole-clip.
fn audio_edit_region(
    mode: AudioEditMode,
    src_secs: f32,
    start: Option<f32>,
    end: Option<f32>,
) -> Option<TimeRegion> {
    match mode {
        AudioEditMode::Extend => Some(TimeRegion {
            start_secs: start.unwrap_or(src_secs),
            end_secs: end,
        }),
        AudioEditMode::Inpaint | AudioEditMode::Repaint => start.map(|start_secs| TimeRegion {
            start_secs,
            end_secs: end,
        }),
        AudioEditMode::Cover => None,
    }
}

/// Decode a canonical PCM-16 WAV (the format both `write_wav_pcm16` writers emit) into a
/// [`gen_core::AudioTrack`] — the source-track reader for the extend/edit path (sc-13410). Iterates
/// the RIFF chunk list so a file carrying extra chunks (LIST/fact) still reads, requires PCM
/// (`audio_format == 1`) 16-bit samples, and converts interleaved `i16` to `f32` in `[-1, 1)`.
/// Non-PCM / non-16-bit inputs are a clear `Unsupported` rather than a silent mis-decode.
pub(crate) fn read_wav_pcm16(path: &Path) -> WorkerResult<gen_core::AudioTrack> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WorkerError::InvalidPayload(format!(
            "source audio {} is not a RIFF/WAVE file",
            path.display()
        )));
    }
    let le16 = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let le32 = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (audio_format, channels, sample_rate, bits)
    let mut data: Option<(usize, usize)> = None; // (start, end)
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = le32(&bytes, pos + 4) as usize;
        let body = pos + 8;
        let end = body.saturating_add(size).min(bytes.len());
        if id == b"fmt " && size >= 16 && body + 16 <= bytes.len() {
            fmt = Some((
                le16(&bytes, body),
                le16(&bytes, body + 2),
                le32(&bytes, body + 4),
                le16(&bytes, body + 14),
            ));
        } else if id == b"data" {
            data = Some((body, end));
        }
        // RIFF chunks are word-aligned: an odd body is followed by a pad byte.
        pos = body + size + (size & 1);
    }
    let (audio_format, channels, sample_rate, bits) = fmt.ok_or_else(|| {
        WorkerError::InvalidPayload(format!("source audio {} has no fmt chunk", path.display()))
    })?;
    if audio_format != 1 || bits != 16 {
        return Err(WorkerError::InvalidPayload(format!(
            "source audio {} must be PCM 16-bit (got format {audio_format}, {bits}-bit)",
            path.display()
        )));
    }
    if channels == 0 || sample_rate == 0 {
        return Err(WorkerError::InvalidPayload(format!(
            "source audio {} declares {channels} channels @ {sample_rate} Hz",
            path.display()
        )));
    }
    let (start, end) = data.ok_or_else(|| {
        WorkerError::InvalidPayload(format!("source audio {} has no data chunk", path.display()))
    })?;
    let samples: Vec<f32> = bytes[start..end]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32_768.0)
        .collect();
    if samples.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "source audio {} decoded to zero samples",
            path.display()
        )));
    }
    Ok(gen_core::AudioTrack {
        samples,
        sample_rate,
        channels,
        stems: Vec::new(),
    })
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
    // Whether this run took the native single-call clone path (sc-13412): the native generator has no
    // base-TTS model, so `baseModel` is omitted for it (it is only meaningful to the OpenVoice chain).
    native_clone: bool,
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
    // Build the `rawAdapterSettings` sub-object separately (not inline in the outer `json!`): the audio
    // knob set has grown wide enough (TTS + SFX + the sc-13410 music/edit fields) that a single nested
    // `json!` literal blows past the macro recursion limit. A separate expansion keeps each within it.
    let raw_adapter_settings = json!({
        "model": request.model,
        "voice": request.voice,
        "language": request.language,
        "targetDurationSecs": request.target_duration_secs,
        "guidance": request.guidance,
        "steps": request.steps,
        "negativePrompt": request.negative_prompt,
        "bpm": request.bpm,
        "musicalKey": request.musical_key,
        "lyrics": request.lyrics,
        "sourceAudioAssetId": request.source_audio_asset_id,
        "editMode": request.edit_mode,
        "editRegionStartSecs": request.edit_region_start_secs,
        "editRegionEndSecs": request.edit_region_end_secs,
        "editStrength": request.edit_strength,
        // Voice Clone (sc-13411 C4): the reference-voice asset, base TTS model, and match strength (τ) so a
        // re-generate reconstructs the exact chain. `null` on a non-voice-clone run.
        "referenceAudioAssetId": request.reference_audio_asset_id,
        "baseModel": if request.is_voice_clone() && !native_clone { Some(request.base_model.as_str()) } else { None },
        "matchStrength": request.match_strength,
        "sampleRate": sample_rate,
    });
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
        "mode": request.mode(),
        "model": request.model,
        "adapter": adapter,
        "prompt": request.prompt,
        // REQUESTED knobs (the replay record). `null` for an omitted voice/language/duration — the
        // studio falls back to the model's own defaults, exactly as the video recipe does. `guidance`
        // / `steps` are the Sound FX (diffusion) sampling knobs; `null` for a TTS run that carries none.
        // The music sub-block (bpm / musicalKey / lyrics / negativePrompt) and the extend/edit source
        // band (sourceAudioAssetId / editMode / editRegion* / editStrength) round-trip for replay too —
        // `null` on a run that carries none (sc-13410).
        "voice": request.voice,
        "language": request.language,
        "targetDurationSecs": request.target_duration_secs,
        "guidance": request.guidance,
        "steps": request.steps,
        "negativePrompt": request.negative_prompt,
        "bpm": request.bpm,
        "musicalKey": request.musical_key,
        "lyrics": request.lyrics,
        "sourceAudioAssetId": request.source_audio_asset_id,
        "editMode": request.edit_mode,
        "editRegionStartSecs": request.edit_region_start_secs,
        "editRegionEndSecs": request.edit_region_end_secs,
        "editStrength": request.edit_strength,
        // Voice Clone replay record (sc-13411 C4) — `null` on a non-voice-clone run.
        "referenceAudioAssetId": request.reference_audio_asset_id,
        "baseModel": if request.is_voice_clone() && !native_clone { Some(request.base_model.as_str()) } else { None },
        "matchStrength": request.match_strength,
        "seed": request.seed,
        "rawAdapterSettings": raw_adapter_settings,
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
            "mode": request.mode(),
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
        // A TTS payload carries no diffusion sampling knobs → None (the generator's own defaults).
        assert_eq!(request.guidance, None);
        assert_eq!(request.steps, None);
    }

    #[test]
    fn from_payload_reads_the_sfx_sampling_knobs() {
        // The Sound FX path (MOSS-SoundEffect, sc-13409) additionally carries guidance (CFG scale) +
        // steps, which the worker maps onto the top-level GenerationRequest — not AudioParams.
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "moss_sfx_v2",
            "prompt": "a heavy wooden door creaking open",
            "language": "en",
            "targetDurationSecs": 3.0,
            "guidance": 6.5,
            "steps": 60,
            "seed": 11,
        })));
        assert_eq!(request.model, "moss_sfx_v2");
        assert_eq!(request.prompt, "a heavy wooden door creaking open");
        assert_eq!(request.language.as_deref(), Some("en"));
        assert_eq!(request.target_duration_secs, Some(3.0));
        assert_eq!(request.guidance, Some(6.5));
        assert_eq!(request.steps, Some(60));
        assert_eq!(request.seed, Some(11));
        // SFX carries no voice — MOSS advertises no voice surface.
        assert_eq!(request.voice, None);
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
        let fact = audio_asset_fact(&plan, &request, 24_000, 1, 3.25, false);
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
        // A TTS run carries no diffusion sampling knobs → null in the replay record.
        assert!(fact["guidance"].is_null());
        assert!(fact["steps"].is_null());
        assert!(fact["mediaPath"]
            .as_str()
            .is_some_and(|path| path.starts_with("assets/audios/") && path.ends_with(".wav")));
        // The streaming result wraps the fact as the sole assetWrite.
        let result = audio_streaming_result(&plan, &request, &fact);
        assert_eq!(result["expectedCount"], 1);
        assert_eq!(result["assetWrites"].as_array().map(Vec::len), Some(1));
        assert_eq!(result["assetWrites"][0]["type"], "audio");
    }

    #[test]
    fn from_payload_reads_the_music_and_edit_fields() {
        // The Music path (ACE-Step, sc-13410) carries the describe-the-music sub-block (bpm / key /
        // lyrics), a negative prompt, and the extend/edit source band (source id + mode + region +
        // strength). All ride verbatim onto the AudioRequest.
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "acestep_v15_turbo",
            "prompt": "gentle lofi piano loop",
            "language": "en",
            "targetDurationSecs": 8.0,
            "steps": 8,
            "bpm": 92.0,
            "musicalKey": "C minor",
            "lyrics": "  [verse] la la la  ",
            "negativePrompt": "harsh distortion",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "Extend",
            "editRegionStartSecs": 3.0,
            "editRegionEndSecs": 20.0,
            "editStrength": 0.5,
            "seed": 5,
        })));
        assert_eq!(request.model, "acestep_v15_turbo");
        assert_eq!(request.bpm, Some(92.0));
        assert_eq!(request.musical_key.as_deref(), Some("C minor"));
        // Lyrics are read verbatim (interior whitespace / tags preserved), only trimmed of nothing —
        // they may legitimately be multi-line; a purely-blank value would be None (instrumental).
        assert_eq!(request.lyrics.as_deref(), Some("  [verse] la la la  "));
        assert_eq!(request.negative_prompt.as_deref(), Some("harsh distortion"));
        assert_eq!(
            request.source_audio_asset_id.as_deref(),
            Some("audio-src-1")
        );
        // The edit mode is lowercased at the parse seam so a mixed-case token still routes.
        assert_eq!(request.edit_mode.as_deref(), Some("extend"));
        assert_eq!(request.edit_region_start_secs, Some(3.0));
        assert_eq!(request.edit_region_end_secs, Some(20.0));
        assert_eq!(request.edit_strength, Some(0.5));
        assert_eq!(request.steps, Some(8));
        assert_eq!(request.seed, Some(5));
        // A blank lyrics value is dropped to None (instrumental), not carried as an empty string.
        let instrumental = AudioRequest::from_payload(&payload(json!({ "lyrics": "   " })));
        assert_eq!(instrumental.lyrics, None);
    }

    #[test]
    fn parse_audio_edit_mode_maps_the_tokens_and_rejects_garbage() {
        assert_eq!(
            parse_audio_edit_mode("inpaint").unwrap(),
            AudioEditMode::Inpaint
        );
        assert_eq!(
            parse_audio_edit_mode("repaint").unwrap(),
            AudioEditMode::Repaint
        );
        assert_eq!(
            parse_audio_edit_mode("extend").unwrap(),
            AudioEditMode::Extend
        );
        assert_eq!(
            parse_audio_edit_mode("cover").unwrap(),
            AudioEditMode::Cover
        );
        assert!(parse_audio_edit_mode("bogus").is_err());
    }

    #[test]
    fn audio_edit_region_policy_matches_the_mode_contract() {
        // Extend: begins the appended tail at the source clip's own length when no start is given, and
        // reads `end` as the new total length (gen_core AudioEditMode::Extend contract).
        let region = audio_edit_region(AudioEditMode::Extend, 10.0, None, Some(20.0))
            .expect("extend yields a region");
        assert_eq!(region.start_secs, 10.0);
        assert_eq!(region.end_secs, Some(20.0));
        // An explicit start overrides the source-length default.
        let region = audio_edit_region(AudioEditMode::Extend, 10.0, Some(8.0), Some(20.0))
            .expect("extend region");
        assert_eq!(region.start_secs, 8.0);

        // Inpaint / Repaint: carry the request's bounded window; a MISSING start ⇒ None (so the
        // generator's own "region required" check fires rather than the worker inventing a window).
        let region = audio_edit_region(AudioEditMode::Inpaint, 10.0, Some(2.0), Some(5.0))
            .expect("inpaint region");
        assert_eq!(region.start_secs, 2.0);
        assert_eq!(region.end_secs, Some(5.0));
        assert!(audio_edit_region(AudioEditMode::Repaint, 10.0, None, None).is_none());

        // Cover: whole-clip, no region.
        assert!(audio_edit_region(AudioEditMode::Cover, 10.0, Some(1.0), Some(2.0)).is_none());
    }

    #[test]
    fn build_audio_edit_is_none_without_a_source() {
        // No source id ⇒ plain text-to-music (no Conditioning), regardless of the other edit fields.
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "acestep_v15_turbo",
            "editMode": "extend",
        })));
        let settings = crate::test_env::offline_settings();
        assert!(
            build_audio_edit(&settings, &request, Path::new("/tmp/project"))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn read_wav_pcm16_roundtrips_a_written_wav() {
        // A clip written by the shared `write_wav_pcm16` (48 kHz stereo, the ACE-Step output shape) must
        // decode back through `read_wav_pcm16` into a gen_core::AudioTrack with the same rate / channels
        // / sample count — the source-track reader the extend/edit path resolves through.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("source.wav");
        let track = AudioTrack {
            samples: vec![0.0, 0.25, -0.5, 0.75, -0.1, 0.4], // 3 interleaved stereo frames
            sample_rate: 48_000,
            channels: 2,
        };
        write_wav_pcm16(&track, &path).expect("write wav");
        let decoded = read_wav_pcm16(&path).expect("read wav");
        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.samples.len(), 6);
        assert!(decoded.stems.is_empty());
        // Values land in the normalized [-1, 1) range (peak-normalized on write), and are finite.
        assert!(decoded
            .samples
            .iter()
            .all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));

        // A non-RIFF blob is a clear error, not a silent mis-decode.
        let junk = dir.path().join("junk.wav");
        std::fs::write(&junk, b"not a wav file at all").expect("write junk");
        assert!(read_wav_pcm16(&junk).is_err());
    }

    #[test]
    fn music_asset_fact_records_the_music_and_edit_fields_for_replay() {
        // A Music run records the describe-the-music sub-block + the extend/edit source band in both the
        // top-level replay record and rawAdapterSettings, so a re-generate round-trips exactly (sc-13410).
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "acestep_v15_turbo",
            "prompt": "gentle lofi piano loop",
            "targetDurationSecs": 8.0,
            "steps": 8,
            "bpm": 92.0,
            "musicalKey": "C minor",
            "lyrics": "[verse] la la la",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "extend",
            "editRegionEndSecs": 20.0,
            "editStrength": 0.5,
            "seed": 5,
        })));
        let plan = AudioPlan::new(&request, Path::new("/tmp/project"));
        let fact = audio_asset_fact(&plan, &request, 48_000, 2, 8.0, false);
        assert_eq!(fact["sampleRate"], 48_000);
        assert_eq!(fact["channels"], 2);
        assert_eq!(fact["bpm"], 92.0);
        assert_eq!(fact["musicalKey"], "C minor");
        assert_eq!(fact["lyrics"], "[verse] la la la");
        assert_eq!(fact["sourceAudioAssetId"], "audio-src-1");
        assert_eq!(fact["editMode"], "extend");
        assert_eq!(fact["editRegionEndSecs"], 20.0);
        assert_eq!(fact["editStrength"], 0.5);
        // ...and mirrored into rawAdapterSettings so the exact request is reconstructable.
        assert_eq!(fact["rawAdapterSettings"]["bpm"], 92.0);
        assert_eq!(fact["rawAdapterSettings"]["editMode"], "extend");
        assert_eq!(fact["rawAdapterSettings"]["editStrength"], 0.5);
        // A music run carries no voice; a distilled-turbo run carries no guidance/negative.
        assert!(fact["voice"].is_null(), "music carries no voice");
    }

    #[test]
    fn sfx_asset_fact_records_the_sampling_knobs_for_replay() {
        // A Sound FX run records its CFG guidance + steps in both the top-level replay record and
        // rawAdapterSettings, so a re-generate round-trips the exact sampler settings (sc-13409).
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "moss_sfx_v2",
            "prompt": "distant rolling thunder over a field",
            "language": "en",
            "targetDurationSecs": 5.0,
            "guidance": 7.0,
            "steps": 80,
            "seed": 99,
        })));
        let plan = AudioPlan::new(&request, Path::new("/tmp/project"));
        let fact = audio_asset_fact(&plan, &request, 48_000, 1, 5.0, false);
        assert_eq!(fact["sampleRate"], 48_000);
        assert_eq!(fact["guidance"], 7.0);
        assert_eq!(fact["steps"], 80);
        assert!(fact["voice"].is_null(), "SFX carries no voice");
        assert_eq!(fact["rawAdapterSettings"]["guidance"], 7.0);
        assert_eq!(fact["rawAdapterSettings"]["steps"], 80);
    }

    // ── Voice Clone (sc-13411 C4) ────────────────────────────────────────────────────────────────

    #[test]
    fn from_payload_reads_the_voice_clone_fields() {
        // A voice-clone payload carries the reference-voice asset, an optional base TTS model, and the
        // match strength (τ). Their presence flips `is_voice_clone` / `mode` onto the conversion path.
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "openvoice_v2",
            "prompt": "Clone this into my reference voice.",
            "referenceAudioAssetId": "ref-voice-1",
            "baseModel": "kokoro_82m",
            "matchStrength": 0.5,
            "seed": 3,
        })));
        assert_eq!(request.model, "openvoice_v2");
        assert_eq!(
            request.reference_audio_asset_id.as_deref(),
            Some("ref-voice-1")
        );
        assert_eq!(request.base_model, "kokoro_82m");
        assert_eq!(request.match_strength, Some(0.5));
        assert!(request.is_voice_clone());
        assert_eq!(request.mode(), "voice_clone");

        // A base model defaults to Kokoro when the payload omits it.
        let defaulted = AudioRequest::from_payload(&payload(json!({
            "model": "openvoice_v2",
            "referenceAudioAssetId": "ref-voice-1",
        })));
        assert_eq!(defaulted.base_model, "kokoro_82m");
        assert!(defaulted.is_voice_clone());

        // No reference ⇒ an ordinary (non-voice-clone) run: a blank/whitespace id does not route.
        let none = AudioRequest::from_payload(&payload(json!({ "model": "kokoro_82m" })));
        assert!(!none.is_voice_clone());
        assert_eq!(none.mode(), "text_to_audio");
        let blank = AudioRequest::from_payload(&payload(json!({ "referenceAudioAssetId": "   " })));
        assert!(!blank.is_voice_clone());
    }

    #[test]
    fn openvoice_converter_dir_descends_into_the_converter_subdir() {
        // The OpenVoice snapshot downloads its converter weights under `converter/`; the transform's
        // `load` wants the dir that directly holds `checkpoint.pth`, so the resolver descends when the
        // subdir carries the checkpoint and passes the root through otherwise.
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().to_path_buf();
        // No converter/ ⇒ pass the root through unchanged.
        assert_eq!(openvoice_converter_dir(root.clone()), root);
        // converter/checkpoint.pth present ⇒ descend into converter/.
        let converter = root.join("converter");
        std::fs::create_dir_all(&converter).unwrap();
        std::fs::write(converter.join("checkpoint.pth"), b"stub").unwrap();
        assert_eq!(openvoice_converter_dir(root.clone()), converter);
    }

    #[test]
    fn voice_clone_asset_fact_records_the_reference_base_and_strength_for_replay() {
        // A voice-clone run records the reference asset, base model, and match strength in both the
        // top-level replay record and rawAdapterSettings, and stamps mode = "voice_clone" (sc-13411).
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "openvoice_v2",
            "prompt": "Clone this into my reference voice.",
            "referenceAudioAssetId": "ref-voice-1",
            "baseModel": "kokoro_82m",
            "matchStrength": 0.5,
            "seed": 3,
        })));
        let plan = AudioPlan::new(&request, Path::new("/tmp/project"));
        let fact = audio_asset_fact(&plan, &request, 22_050, 1, 4.2, false);
        assert_eq!(fact["mode"], "voice_clone");
        assert_eq!(fact["model"], "openvoice_v2");
        assert_eq!(fact["referenceAudioAssetId"], "ref-voice-1");
        assert_eq!(fact["baseModel"], "kokoro_82m");
        assert_eq!(fact["matchStrength"], 0.5);
        // ...mirrored into rawAdapterSettings so the exact conversion request is reconstructable.
        assert_eq!(
            fact["rawAdapterSettings"]["referenceAudioAssetId"],
            "ref-voice-1"
        );
        assert_eq!(fact["rawAdapterSettings"]["baseModel"], "kokoro_82m");
        assert_eq!(fact["rawAdapterSettings"]["matchStrength"], 0.5);
        // The streaming set carries the voice_clone mode too.
        let result = audio_streaming_result(&plan, &request, &fact);
        assert_eq!(result["generationSet"]["mode"], "voice_clone");

        // A non-voice-clone run leaves the voice-clone fields null and baseModel null (not "kokoro_82m").
        let plain = AudioRequest::from_payload(&payload(json!({ "model": "kokoro_82m" })));
        let plain_plan = AudioPlan::new(&plain, Path::new("/tmp/project"));
        let plain_fact = audio_asset_fact(&plain_plan, &plain, 24_000, 1, 3.0, false);
        assert_eq!(plain_fact["mode"], "text_to_audio");
        assert!(plain_fact["referenceAudioAssetId"].is_null());
        assert!(
            plain_fact["baseModel"].is_null(),
            "baseModel is null on a non-voice-clone run"
        );
        assert!(plain_fact["matchStrength"].is_null());
    }

    #[test]
    fn native_voice_clone_asset_fact_omits_the_base_model() {
        // A native clone-TTS run (sc-13412) is still a `voice_clone` mode with a reference, but it has
        // NO base-TTS model — the single generator renders directly. So `baseModel` is null even though
        // `is_voice_clone()` is true (native_clone = true), while the reference is still recorded for
        // replay. This is the ONE field that differs from the OpenVoice conversion chain's replay record.
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "chatterbox_tts",
            "prompt": "Render this in the cloned voice, in one step.",
            "referenceAudioAssetId": "ref-voice-1",
            "seed": 9,
        })));
        let plan = AudioPlan::new(&request, Path::new("/tmp/project"));
        let fact = audio_asset_fact(
            &plan, &request, 24_000, 1, 4.0, /* native_clone */ true,
        );
        assert_eq!(fact["mode"], "voice_clone");
        assert_eq!(fact["model"], "chatterbox_tts");
        assert_eq!(fact["referenceAudioAssetId"], "ref-voice-1");
        // No base-TTS pass on the native path — baseModel is omitted in both the top-level record and
        // rawAdapterSettings (the conversion chain records "kokoro_82m" here; the native clone does not).
        assert!(
            fact["baseModel"].is_null(),
            "native clone has no base-TTS model"
        );
        assert!(fact["rawAdapterSettings"]["baseModel"].is_null());
        // The reference still round-trips for replay.
        assert_eq!(
            fact["rawAdapterSettings"]["referenceAudioAssetId"],
            "ref-voice-1"
        );

        // Contrast: the SAME request scored as the conversion chain (native_clone = false) DOES record
        // the base model — proving the flag is the only lever and the default (conversion) is unchanged.
        let conversion_fact = audio_asset_fact(&plan, &request, 24_000, 1, 4.0, false);
        assert_eq!(conversion_fact["baseModel"], "kokoro_82m");
    }

    // -------------------------------------------------------------------------------------------
    // sc-13469 — mid-synthesis cancellation (parity with the video cancel watcher). Each blocking
    // synthesis path now runs under ONE shared engine `CancelFlag` that reaches BOTH the request(s)
    // the engine sees AND `run_blocking_with_heartbeat`, whose interval watcher polls the API cancel
    // state and trips the flag while generation is IN FLIGHT — not only via the post-synthesis
    // `check_cancel` after the whole clip has already rendered.
    //
    // These tests drive the REAL `_using` synthesis functions against stub engines that read
    // `req.cancel`, so the regression the story fixes (a fresh `CancelFlag::new()` in the request,
    // never tripped) FAILS them: the stub would never observe a trip and its wait-for-cancel loop
    // would time out. This is the deterministic watcher test the DoD prefers over a timing-based
    // real-inference abort (the seeded models render short clips, making a real mid-flight cancel
    // flaky).
    // -------------------------------------------------------------------------------------------

    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use axum::{
        extract::{Path as AxumPath, State},
        response::{IntoResponse, Response},
        routing::{get, post},
        Json, Router,
    };

    fn stub_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "stub_audio",
            family: "stub",
            backend: "mlx",
            modality: gen_core::Modality::Audio,
            capabilities: gen_core::Capabilities::default(),
        }
    }

    enum StubBehavior {
        /// Block until `req.cancel` is tripped, then surface the engine's typed cancellation — the
        /// cooperative mid-synthesis bail a MOSS-SFX / ACE-Step / chatterbox step loop performs.
        WaitForCancel,
        /// Return a short non-silent track promptly (the clean-completion control).
        CompleteOk,
    }

    /// A stub [`Generator`] whose `generate` reads `req.cancel` — so it can prove the SHARED,
    /// watcher-tripped flag actually reached [`GenerationRequest::cancel`].
    struct StubGenerator {
        descriptor: gen_core::ModelDescriptor,
        behavior: StubBehavior,
        observed_cancel: Arc<AtomicBool>,
    }

    impl gen_core::Generator for StubGenerator {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            match self.behavior {
                StubBehavior::WaitForCancel => {
                    let start = Instant::now();
                    while !req.cancel.is_cancelled() {
                        if start.elapsed() > Duration::from_secs(30) {
                            return Err(gen_core::Error::Msg(
                                "req.cancel was never tripped mid-synthesis".to_owned(),
                            ));
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    // Observed the shared flag trip DURING generate() — the proof this path wires it.
                    self.observed_cancel.store(true, Ordering::SeqCst);
                    Err(gen_core::Error::Canceled)
                }
                StubBehavior::CompleteOk => {
                    if req.cancel.is_cancelled() {
                        self.observed_cancel.store(true, Ordering::SeqCst);
                    }
                    Ok(GenerationOutput::Audio(gen_core::AudioTrack {
                        samples: vec![0.1, -0.1, 0.1, -0.1],
                        sample_rate: 24_000,
                        channels: 1,
                        stems: Vec::new(),
                    }))
                }
            }
        }
    }

    fn stub_transform_descriptor() -> gen_core::AudioTransformDescriptor {
        gen_core::AudioTransformDescriptor {
            id: "stub_converter",
            family: "audio",
            backend: "mlx",
            capabilities: gen_core::AudioTransformCapabilities::default(),
        }
    }

    /// A converter stub that blocks until `req.cancel` trips — proving the shared flag reaches the
    /// SECOND (converter) call of the voice-clone chain, whose request used to default `cancel` to a
    /// fresh, never-tripped flag via `..Default::default()`.
    struct StubTransform {
        descriptor: gen_core::AudioTransformDescriptor,
        observed_cancel: Arc<AtomicBool>,
    }

    impl gen_core::AudioTransform for StubTransform {
        fn descriptor(&self) -> &gen_core::AudioTransformDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &AudioTransformRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn apply(
            &self,
            req: &AudioTransformRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<Vec<gen_core::AudioTrack>> {
            let start = Instant::now();
            while !req.cancel.is_cancelled() {
                if start.elapsed() > Duration::from_secs(30) {
                    return Err(gen_core::Error::Msg(
                        "converter req.cancel was never tripped mid-conversion".to_owned(),
                    ));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            self.observed_cancel.store(true, Ordering::SeqCst);
            Err(gen_core::Error::Canceled)
        }
    }

    fn audio_test_job(job_id: &str) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": job_id,
            "type": "audio_generate",
            "status": "running",
            "projectId": null,
            "projectName": null,
            "payload": {},
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": "test-worker",
            "progress": 0.2,
            "stage": "generating",
            "message": "running",
            "error": null,
            "etaSeconds": null,
            "elapsedSeconds": null,
            "attempts": 1,
            "sourceJobId": null,
            "duplicateOfJobId": null,
            "cancelRequested": false,
            "createdAt": "2026-07-20T00:00:00Z",
            "updatedAt": "2026-07-20T00:00:00Z",
            "startedAt": null,
            "completedAt": null,
            "canceledAt": null,
            "lastHeartbeatAt": null
        }))
        .expect("audio job snapshot deserializes")
    }

    fn cancel_job_json(job_id: &str, cancel_requested: bool) -> Value {
        json!({
            "id": job_id, "type": "audio_generate", "status": "running",
            "projectId": null, "projectName": null, "payload": {}, "result": {},
            "requestedGpu": "auto", "assignedGpu": null, "workerId": "test-worker",
            "progress": 0.2, "stage": "generating", "message": "running", "error": null,
            "etaSeconds": null, "elapsedSeconds": null, "attempts": 1,
            "sourceJobId": null, "duplicateOfJobId": null,
            "cancelRequested": cancel_requested,
            "createdAt": "2026-07-20T00:00:00Z", "updatedAt": "2026-07-20T00:00:00Z",
            "startedAt": null, "completedAt": null, "canceledAt": null, "lastHeartbeatAt": null
        })
    }

    #[derive(Clone)]
    struct CancelStubState {
        cancel_requested: bool,
        progress: Arc<Mutex<Vec<Value>>>,
    }

    /// Spawn an API stub whose job GET + progress POST report `cancel_requested` (so the
    /// `run_blocking_with_heartbeat` interval watcher trips the shared flag on its first, immediate
    /// tick when `true`), records every progress body (so the terminal `Canceled` write is
    /// observable), and answers worker heartbeats.
    async fn spawn_audio_cancel_stub(cancel_requested: bool) -> (String, Arc<Mutex<Vec<Value>>>) {
        async fn job_route(
            State(state): State<CancelStubState>,
            AxumPath(job_id): AxumPath<String>,
        ) -> Response {
            Json(cancel_job_json(&job_id, state.cancel_requested)).into_response()
        }
        async fn progress_route(
            State(state): State<CancelStubState>,
            AxumPath(job_id): AxumPath<String>,
            Json(body): Json<Value>,
        ) -> Response {
            state.progress.lock().expect("progress lock").push(body);
            Json(cancel_job_json(&job_id, state.cancel_requested)).into_response()
        }
        async fn heartbeat_route() -> Response {
            // The body does not parse as a WorkerSnapshot; heartbeat() tolerates the decode failure as
            // a transport error, so the keepalive ping still succeeds without modeling the snapshot.
            Json(json!({})).into_response()
        }
        let progress = Arc::new(Mutex::new(Vec::new()));
        let state = CancelStubState {
            cancel_requested,
            progress: progress.clone(),
        };
        let app = Router::new()
            .route("/api/v1/jobs/:job_id", get(job_route))
            .route("/api/v1/jobs/:job_id/progress", post(progress_route))
            .route(
                "/api/v1/workers/:worker_id/heartbeat",
                post(heartbeat_route),
            )
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener binds");
        let address = listener.local_addr().expect("listener has address");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("stub serves");
        });
        (format!("http://{address}"), progress)
    }

    fn cancel_test_settings(base_url: String) -> Settings {
        let mut settings = Settings::from_env();
        settings.api_url = base_url;
        settings.worker_id = "test-worker".to_owned();
        // Shortest interval the clamp allows so the watcher polls promptly (the first interval tick is
        // immediate regardless).
        settings.heartbeat_seconds = 5;
        settings
    }

    fn stub_load(
        behavior: StubBehavior,
        observed: Arc<AtomicBool>,
    ) -> impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>> + Send + 'static {
        move |_id: &str, _spec: &LoadSpec| {
            Ok(Box::new(StubGenerator {
                descriptor: stub_descriptor(),
                behavior,
                observed_cancel: observed,
            }) as Box<dyn Generator>)
        }
    }

    /// Single path (Speech / SFX / Music): a cancel requested while synthesis is in flight must trip
    /// the SHARED flag the request carries — observed from inside the stub's `generate` — and surface
    /// as a terminal `Canceled`, not a failure.
    #[tokio::test]
    async fn single_synthesis_trips_shared_cancel_flag_during_generation() {
        let (base_url, progress) = spawn_audio_cancel_stub(true).await;
        let settings = cancel_test_settings(base_url);
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-cancel-single");
        let request = AudioRequest::from_payload(&payload(json!({})));

        let observed = Arc::new(AtomicBool::new(false));
        let load = stub_load(StubBehavior::WaitForCancel, observed.clone());

        let result = run_audio_synthesis_using(
            &api,
            &settings,
            &job,
            &request,
            PathBuf::from("unused"),
            None,
            load,
        )
        .await;

        assert!(
            matches!(result, Err(WorkerError::Canceled(_))),
            "a mid-synthesis cancel must surface as WorkerError::Canceled, got {result:?}"
        );
        assert!(
            observed.load(Ordering::SeqCst),
            "the stub generator must have observed its req.cancel tripped DURING generate() — proving \
             the shared, watcher-tripped flag reached GenerationRequest.cancel, not a fresh \
             CancelFlag::new()"
        );
        let posts = progress.lock().expect("progress lock");
        assert!(
            posts.iter().any(|p| p["status"] == "canceled"),
            "the terminal Canceled must be posted, got {posts:?}"
        );
    }

    /// Control: a clean completion must NOT trip the flag and must NOT leak the watcher. The call
    /// returning at all proves the watcher tore down when the blocking task resolved; the flag stays
    /// untripped and no terminal `Canceled` is posted.
    #[tokio::test]
    async fn single_synthesis_completes_cleanly_without_a_false_trip_or_leak() {
        let (base_url, progress) = spawn_audio_cancel_stub(false).await;
        let settings = cancel_test_settings(base_url);
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-ok-single");
        let request = AudioRequest::from_payload(&payload(json!({})));

        let observed = Arc::new(AtomicBool::new(false));
        let load = stub_load(StubBehavior::CompleteOk, observed.clone());

        let track = run_audio_synthesis_using(
            &api,
            &settings,
            &job,
            &request,
            PathBuf::from("unused"),
            None,
            load,
        )
        .await
        .expect("a clean synthesis returns the produced track");

        assert!(
            !track.samples.is_empty(),
            "the produced track carries samples"
        );
        assert!(
            !observed.load(Ordering::SeqCst),
            "a normal completion must NOT trip the request's cancel flag (no false-trip)"
        );
        let posts = progress.lock().expect("progress lock");
        assert!(
            posts.iter().all(|p| p["status"] != "canceled"),
            "a clean completion posts no terminal Canceled, got {posts:?}"
        );
    }

    /// Voice-clone chain: base TTS completes, then the CONVERTER call blocks. A cancel observed inside
    /// the converter stub proves the shared flag reached `AudioTransformRequest.cancel` (the two-call
    /// trap the fix closes — that request previously defaulted `cancel` to a fresh flag).
    #[tokio::test]
    async fn voice_clone_trips_shared_flag_during_the_converter_call() {
        let (base_url, progress) = spawn_audio_cancel_stub(true).await;
        let settings = cancel_test_settings(base_url);
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-cancel-voiceclone");
        let request = AudioRequest::from_payload(&payload(json!({})));

        let base_observed = Arc::new(AtomicBool::new(false));
        let base_load = stub_load(StubBehavior::CompleteOk, base_observed);
        let converter_observed = Arc::new(AtomicBool::new(false));
        let converter_load = {
            let converter_observed = converter_observed.clone();
            move |_id: &str, _spec: &LoadSpec| {
                Ok(Box::new(StubTransform {
                    descriptor: stub_transform_descriptor(),
                    observed_cancel: converter_observed,
                }) as Box<dyn AudioTransform>)
            }
        };
        let plan = VoiceClonePlan {
            base_model_dir: PathBuf::from("unused-base"),
            converter_dir: PathBuf::from("unused-converter"),
            reference: gen_core::AudioTrack {
                samples: vec![0.05, -0.05, 0.05],
                sample_rate: 24_000,
                channels: 1,
                stems: Vec::new(),
            },
        };

        let result = run_voice_clone_synthesis_using(
            &api,
            &settings,
            &job,
            &request,
            plan,
            base_load,
            converter_load,
        )
        .await;

        assert!(
            matches!(result, Err(WorkerError::Canceled(_))),
            "a mid-conversion cancel must surface as WorkerError::Canceled, got {result:?}"
        );
        assert!(
            converter_observed.load(Ordering::SeqCst),
            "the SHARED flag must reach the converter's AudioTransformRequest.cancel and trip \
             mid-conversion — the two-call trap"
        );
        let posts = progress.lock().expect("progress lock");
        assert!(
            posts.iter().any(|p| p["status"] == "canceled"),
            "the terminal Canceled must be posted, got {posts:?}"
        );
    }

    /// Native single-call clone-TTS path (`chatterbox_tts`): the shared flag reaches the one
    /// `GenerationRequest.cancel` and trips mid-generation.
    #[tokio::test]
    async fn native_voice_clone_trips_shared_cancel_flag_during_generation() {
        let (base_url, progress) = spawn_audio_cancel_stub(true).await;
        let settings = cancel_test_settings(base_url);
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-cancel-native");
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "chatterbox_tts",
            "referenceAudioAssetId": "ref-1",
        })));

        let observed = Arc::new(AtomicBool::new(false));
        let load = stub_load(StubBehavior::WaitForCancel, observed.clone());
        let plan = NativeVoiceClonePlan {
            model_dir: PathBuf::from("unused"),
            reference: gen_core::AudioTrack {
                samples: vec![0.05, -0.05],
                sample_rate: 24_000,
                channels: 1,
                stems: Vec::new(),
            },
        };

        let result =
            run_native_voice_clone_synthesis_using(&api, &settings, &job, &request, plan, load)
                .await;

        assert!(
            matches!(result, Err(WorkerError::Canceled(_))),
            "a mid-synthesis cancel must surface as WorkerError::Canceled, got {result:?}"
        );
        assert!(
            observed.load(Ordering::SeqCst),
            "the native clone path must trip its req.cancel mid-generation (shared flag reached the \
             request)"
        );
        let posts = progress.lock().expect("progress lock");
        assert!(
            posts.iter().any(|p| p["status"] == "canceled"),
            "the terminal Canceled must be posted, got {posts:?}"
        );
    }
}
