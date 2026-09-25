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
    GenerationOutput, GenerationRequest, Generator, LoadSpec, Progress, SpeechSegment, TimeRegion,
    WeightsSource,
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
    model_provided: bool,
    prompt: String,
    voice: Option<String>,
    language: Option<String>,
    target_duration_secs: Option<f32>,
    /// Multi-speaker / long-form dialogue script (sc-13676) — an ordered list of spoken segments that
    /// rides [`AudioParams::script`]. A model advertising `supports_multi_speaker` (MOSS-TTSD) renders
    /// each segment in its own voice into one clip; the gen-core floor rejects a script sent to a
    /// model that does NOT advertise it. `None` (no `script` key, or an empty array) ⇒ an ordinary
    /// single-voice request, byte-for-byte unaffected.
    script: Option<Vec<SpeechSegment>>,
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
    /// injects its manifest entry as `baseModelManifestEntry`.
    base_model: String,
    base_model_provided: bool,
    /// Voice Clone match strength — overrides OpenVoice V2's posterior-sampling temperature τ (rides
    /// `AudioTransformRequest::strength`). `None` ⇒ the converter's own default (τ = 0.3).
    match_strength: Option<f32>,
    /// The resolved manifest entry for [`base_model`] (the base TTS), injected by the API on a voice-clone
    /// request so the worker resolves the base generator's snapshot without re-parsing the jsonc. `{}` on a
    /// non-voice-clone job.
    base_model_manifest_entry: Value,
    seed: Option<i64>,
    model_manifest_entry: Value,
    /// Segmented-song controls (YuE, sc-19384) — ride `AudioParams::segments` /
    /// `max_new_tokens_per_segment` / `repetition_penalty`. `None` ⇒ the model default; the gen-core
    /// floor refuses each on a model that does not advertise reading it.
    segments: Option<u32>,
    max_new_tokens_per_segment: Option<u32>,
    repetition_penalty: Option<f32>,
    /// Guidance on/off (sc-19384). `Some(false)` sends guidance `0.0` — YuE reads `0..=1` as CFG off —
    /// and the API refuses a scale alongside it. `None`/`Some(true)` send `guidance` as given (`None`
    /// ⇒ the model's own schedule).
    guidance_enabled: Option<bool>,
    /// In-context-learning reference (YuE `_icl` checkpoints, sc-19384): `single` (one mixed clip,
    /// `icl_reference_asset_id`) or `dual` (`icl_vocal_asset_id` + `icl_instrumental_asset_id`), over
    /// the `icl_start_secs..icl_end_secs` window. Deliberately NOT `referenceAudioAssetId`, which
    /// routes a job onto the voice-clone chain.
    icl_mode: Option<String>,
    icl_reference_asset_id: Option<String>,
    icl_vocal_asset_id: Option<String>,
    icl_instrumental_asset_id: Option<String>,
    icl_start_secs: Option<f32>,
    icl_end_secs: Option<f32>,
    /// Requested weight tier (`bf16` / `q8` / `q4`) for a model that ships physical per-tier
    /// downloads (YuE, sc-19384). `None` ⇒ the manifest's default tier when installed, else the first
    /// installed tier.
    quant_tier: Option<String>,
    /// Output limiter (YuE `save_audio`, upstream `--rescale`, sc-19384): `clamp` (the default) or
    /// `rescale`. `None` ⇒ the model default.
    output_limiter: Option<String>,
}

/// Parse a `script` payload value into a multi-speaker dialogue script (sc-13676). The wire shape is
/// an array of `{ text, speaker?, style? }` objects (the API's `SpeechSegmentDto`, camelCase). Returns
/// `None` when the key is absent, not an array, or empty — so a single-voice request (which never
/// carries `script`) builds the identical `AudioParams { script: None, .. }` as before. A segment with
/// empty/whitespace text is dropped defensively; the API's `validate_audio_job` already rejects one,
/// and the gen-core floor gates the whole script against the model's `supports_multi_speaker` /
/// `max_speakers`.
fn parse_speech_segments(value: Option<&Value>) -> Option<Vec<SpeechSegment>> {
    let array = value.and_then(Value::as_array)?;
    let opt_string = |segment: &Value, key: &str| {
        segment
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    let segments: Vec<SpeechSegment> = array
        .iter()
        .filter_map(|segment| {
            let text = segment.get("text").and_then(Value::as_str)?;
            if text.trim().is_empty() {
                return None;
            }
            Some(SpeechSegment {
                text: text.to_owned(),
                speaker: opt_string(segment, "speaker"),
                style: opt_string(segment, "style"),
            })
        })
        .collect();
    if segments.is_empty() {
        None
    } else {
        Some(segments)
    }
}

/// Serialize a parsed multi-speaker script back to the wire shape (`[{ text, speaker?, style? }]`) for
/// the asset's replay record (sc-13676) — the inverse of [`parse_speech_segments`], so a re-generate
/// reconstructs the exact dialogue. `None` ⇒ JSON `null` (a single-voice run carries no script).
/// gen-core's `SpeechSegment` is not `Serialize`, so this maps it by hand.
fn script_to_json(script: &Option<Vec<SpeechSegment>>) -> Value {
    match script {
        Some(segments) => Value::Array(
            segments
                .iter()
                .map(|segment| {
                    json!({
                        "text": segment.text,
                        "speaker": segment.speaker,
                        "style": segment.style,
                    })
                })
                .collect(),
        ),
        None => Value::Null,
    }
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
        let model = string("model");
        let base_model = string("baseModel");
        Self {
            project_id: string("projectId").unwrap_or_default(),
            model_provided: model.is_some(),
            model: model.unwrap_or_else(|| "kokoro_82m".to_owned()),
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
            // Multi-speaker dialogue script (sc-13676): parse the `script` array into gen-core
            // SpeechSegments. Absent or empty ⇒ `None`, so a single-voice request builds the identical
            // `AudioParams { script: None, .. }` it did before this field existed.
            script: parse_speech_segments(payload.get("script")),
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
            base_model_provided: base_model.is_some(),
            base_model: base_model.unwrap_or_else(|| "kokoro_82m".to_owned()),
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
            segments: payload
                .get("segments")
                .and_then(Value::as_u64)
                .map(|value| value.min(u64::from(u32::MAX)) as u32),
            max_new_tokens_per_segment: payload
                .get("maxNewTokensPerSegment")
                .and_then(Value::as_u64)
                .map(|value| value.min(u64::from(u32::MAX)) as u32),
            repetition_penalty: payload
                .get("repetitionPenalty")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            guidance_enabled: payload.get("guidanceEnabled").and_then(Value::as_bool),
            icl_mode: string("iclMode").map(|mode| mode.to_lowercase()),
            icl_reference_asset_id: string("iclReferenceAssetId"),
            icl_vocal_asset_id: string("iclVocalAssetId"),
            icl_instrumental_asset_id: string("iclInstrumentalAssetId"),
            icl_start_secs: payload
                .get("iclStartSecs")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            icl_end_secs: payload
                .get("iclEndSecs")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            quant_tier: string("quantTier").map(|tier| tier.to_lowercase()),
            output_limiter: string("outputLimiter").map(|limiter| limiter.to_lowercase()),
        }
    }

    /// The top-level `GenerationRequest::guidance` this job sends: `0.0` when guidance is switched
    /// off (a segmented-song model reads `0..=1` as CFG off), else the requested scale.
    fn effective_guidance(&self) -> Option<f32> {
        if self.guidance_enabled == Some(false) {
            Some(0.0)
        } else {
            self.guidance
        }
    }

    /// The ICL reference window (`AudioParams::reference_region`). `None` when neither end is set, so
    /// the model's own default window applies. A missing end mirrors upstream YuE's
    /// `prompt_end_time` default ([`ICL_DEFAULT_END_SECS`]), and a missing start is `0`.
    fn reference_region(&self) -> Option<TimeRegion> {
        if self.icl_start_secs.is_none() && self.icl_end_secs.is_none() {
            return None;
        }
        Some(TimeRegion {
            start_secs: self.icl_start_secs.unwrap_or(0.0),
            end_secs: Some(self.icl_end_secs.unwrap_or(ICL_DEFAULT_END_SECS)),
        })
    }

    /// The ICL reference asset ids in `(stem name, asset id)` order: one unnamed clip for `single`,
    /// the `vocals` + `instrumental` pair for `dual`. Empty when the job carries no ICL mode.
    fn icl_references(&self) -> WorkerResult<Vec<(Option<&'static str>, &str)>> {
        let mode = self.icl_mode.as_deref().unwrap_or_default();
        let required = |value: &Option<String>, field: &str| -> WorkerResult<()> {
            if value.is_none() {
                return Err(WorkerError::InvalidPayload(format!(
                    "iclMode {mode:?} requires {field}."
                )));
            }
            Ok(())
        };
        let refuse = |value: &Option<String>, field: &str| {
            if value.is_some() {
                Err(WorkerError::InvalidPayload(format!(
                    "{field} does not apply to iclMode {mode:?}."
                )))
            } else {
                Ok(())
            }
        };
        match self.icl_mode.as_deref() {
            None => {
                if self.icl_reference_asset_id.is_some()
                    || self.icl_vocal_asset_id.is_some()
                    || self.icl_instrumental_asset_id.is_some()
                    || self.icl_start_secs.is_some()
                    || self.icl_end_secs.is_some()
                {
                    return Err(WorkerError::InvalidPayload(
                        "ICL reference fields need an iclMode (single or dual).".to_owned(),
                    ));
                }
                Ok(Vec::new())
            }
            Some("single") => {
                refuse(&self.icl_vocal_asset_id, "iclVocalAssetId")?;
                refuse(&self.icl_instrumental_asset_id, "iclInstrumentalAssetId")?;
                required(&self.icl_reference_asset_id, "iclReferenceAssetId")?;
                Ok(vec![(
                    None,
                    self.icl_reference_asset_id.as_deref().unwrap_or_default(),
                )])
            }
            Some("dual") => {
                refuse(&self.icl_reference_asset_id, "iclReferenceAssetId")?;
                required(&self.icl_vocal_asset_id, "iclVocalAssetId")?;
                required(&self.icl_instrumental_asset_id, "iclInstrumentalAssetId")?;
                Ok(vec![
                    (
                        Some("vocals"),
                        self.icl_vocal_asset_id.as_deref().unwrap_or_default(),
                    ),
                    (
                        Some("instrumental"),
                        self.icl_instrumental_asset_id
                            .as_deref()
                            .unwrap_or_default(),
                    ),
                ])
            }
            Some(other) => Err(WorkerError::InvalidPayload(format!(
                "iclMode must be \"single\" or \"dual\", got {other:?}."
            ))),
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
    if !request.model_provided {
        return Err(WorkerError::InvalidPayload("model is required.".to_owned()));
    }
    // A single-voice request needs a prompt; a multi-speaker request (sc-13676) carries its text in
    // the `script` instead, so one of the two must be present. `script` is `None` for every
    // single-voice request, so this is byte-for-byte the original "prompt required" check for them.
    let has_script = request.script.as_ref().is_some_and(|s| !s.is_empty());
    if request.prompt.trim().is_empty() && !has_script {
        return Err(WorkerError::InvalidPayload(
            "prompt (or a multi-speaker script) is required.".to_owned(),
        ));
    }
    if let Some(limiter) = request.output_limiter.as_deref() {
        output_limiter(limiter)?;
    }
    if let Some(TimeRegion {
        start_secs,
        end_secs: Some(end_secs),
    }) = request.reference_region()
    {
        if !(start_secs.is_finite() && end_secs.is_finite() && start_secs >= 0.0)
            || end_secs <= start_secs
        {
            return Err(WorkerError::InvalidPayload(format!(
                "the ICL reference window {start_secs}..{end_secs} s must satisfy 0 <= start < end \
                 (the end defaults to {ICL_DEFAULT_END_SECS} s)."
            )));
        }
    }
    // ICL reference (sc-19384): the mode/asset-id pairing is well-formed, and it is the job's ONE
    // reference-audio source — never combined with a voice-clone reference or an edit source.
    if !request.icl_references()?.is_empty()
        && (request.is_voice_clone() || request.source_audio_asset_id.is_some())
    {
        return Err(WorkerError::InvalidPayload(
            "an ICL reference cannot be combined with referenceAudioAssetId or \
             sourceAudioAssetId."
                .to_owned(),
        ));
    }
    Ok(())
}

/// Parse an `outputLimiter` token (`clamp` / `rescale`, sc-19384) into the engine's
/// [`gen_core::OutputLimiter`]. The API rejects an unknown token up front; this is the worker mirror.
fn output_limiter(token: &str) -> WorkerResult<gen_core::OutputLimiter> {
    match token {
        "clamp" => Ok(gen_core::OutputLimiter::Clamp),
        "rescale" => Ok(gen_core::OutputLimiter::Rescale),
        other => Err(WorkerError::InvalidPayload(format!(
            "outputLimiter must be \"clamp\" or \"rescale\", got {other:?}."
        ))),
    }
}

/// The rate every YuE ICL reference reaches the engine at: xcodec's 16 kHz mono input (the upstream
/// pipeline loads the prompt audio mono and resamples it to 16 kHz before encoding). sc-19384.
const ICL_REFERENCE_SAMPLE_RATE: u32 = 16_000;
/// Upstream YuE's `prompt_end_time` default — the window end when a request sets only a start.
const ICL_DEFAULT_END_SECS: f32 = 30.0;
const ICL_REFERENCE_CHANNELS: u16 = 1;

/// A weight tier resolved for a model that ships physical per-tier downloads (`<tier>/*`, sc-19384):
/// the tier name (also the selector for its per-tier co-requisites) and the `LoadSpec::quantize`
/// the load asserts (`None` for bf16 — the unquantized load).
#[derive(Clone, Debug, PartialEq)]
struct AudioTier {
    name: String,
    quantize: Option<gen_core::Quant>,
}

/// The primary (non-`coRequisite`) per-tier downloads a manifest entry declares, as
/// `(variant, subdir, default)` in manifest order. Empty for every untiered audio model.
fn audio_tier_rows(entry: &Value) -> Vec<(String, String, bool)> {
    entry
        .get("downloads")
        .and_then(Value::as_array)
        .map(|downloads| {
            downloads
                .iter()
                .filter(|download| {
                    download.get("coRequisite").and_then(Value::as_bool) != Some(true)
                })
                .filter_map(|download| {
                    let variant = download.get("variant").and_then(Value::as_str)?.trim();
                    if variant.is_empty() {
                        return None;
                    }
                    let subdir = download
                        .get("subdir")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|subdir| !subdir.is_empty())
                        .unwrap_or(variant);
                    Some((
                        variant.to_lowercase(),
                        subdir.to_owned(),
                        download.get("default").and_then(Value::as_bool) == Some(true),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Resolve the weights dir + tier for `request` under the model's snapshot `root`. An untiered model
/// passes `root` through with no tier (every audio model before YuE). A tiered model loads from
/// `root/<tier subdir>`: the requested `quantTier` when installed (else a clear error naming it),
/// otherwise the manifest's default tier when installed, else the first installed tier — never a
/// silent substitute for an explicit pick.
fn resolve_audio_tier(
    request: &AudioRequest,
    root: PathBuf,
) -> WorkerResult<(PathBuf, Option<AudioTier>)> {
    let rows = audio_tier_rows(&request.model_manifest_entry);
    if rows.is_empty() {
        if let Some(tier) = &request.quant_tier {
            return Err(WorkerError::InvalidPayload(format!(
                "{}: quantTier {tier:?} was requested, but the model ships no per-tier weights.",
                request.model
            )));
        }
        return Ok((root, None));
    }
    let installed = |subdir: &str| root.join(subdir).is_dir();
    let (name, subdir) = match &request.quant_tier {
        Some(tier) => {
            let (variant, subdir, _) = rows
                .iter()
                .find(|(variant, _, _)| variant == tier)
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "{}: quantTier {tier:?} is not one of the model's tiers ({}).",
                        request.model,
                        rows.iter()
                            .map(|(variant, _, _)| variant.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?;
            if !installed(subdir) {
                return Err(WorkerError::InvalidPayload(format!(
                    "{}: the {variant} tier is not installed — download it first.",
                    request.model
                )));
            }
            (variant.clone(), subdir.clone())
        }
        None => rows
            .iter()
            .filter(|(_, _, default)| *default)
            .chain(rows.iter().filter(|(_, _, default)| !*default))
            .find(|(_, subdir, _)| installed(subdir))
            .map(|(variant, subdir, _)| (variant.clone(), subdir.clone()))
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{}: no weight tier is installed — download one first.",
                    request.model
                ))
            })?,
    };
    let quantize = match name.as_str() {
        "q4" => Some(gen_core::Quant::Q4),
        "q8" => Some(gen_core::Quant::Q8),
        _ => None,
    };
    Ok((root.join(subdir), Some(AudioTier { name, quantize })))
}

/// The facts the YuE memory gate prices (sc-19386), read off the job's own parsed request and its
/// resolved tier — the values the synthesis arm sends, never a second parse of the payload.
fn yue_request_facts<'a>(
    request: &'a AudioRequest,
    tier: Option<&AudioTier>,
) -> crate::yue_admission::YueRequestFacts<'a> {
    crate::yue_admission::YueRequestFacts {
        tier: tier.and_then(|tier| crate::yue_admission::YueTier::from_key(&tier.name)),
        segments: request.segments,
        max_new_tokens: request.max_new_tokens_per_segment,
        guidance: request.effective_guidance(),
        icl_mode: request.icl_mode.as_deref(),
        icl_start_secs: request.icl_start_secs,
        icl_end_secs: request.icl_end_secs,
        prompt: &request.prompt,
        lyrics: request.lyrics.as_deref().unwrap_or_default(),
    }
}

/// Resolve + decode a job's ICL reference clip(s) into ONE [`Conditioning::ReferenceAudio`]
/// (sc-19384): a `single` mix rides as the track itself; a `dual` pair rides as a track carrying
/// `vocals` + `instrumental` stems (the engine's dual-track carrier), with the mix field their sum.
/// Every clip goes through the shared asset guard ([`crate::video_jobs::ltx::resolve_clip_media_path`])
/// and the shared ffmpeg normalization onto [`ICL_REFERENCE_SAMPLE_RATE`] mono, so any container the
/// library holds is admissible. Scratch lives in a [`tempfile::TempDir`] that is removed on EVERY exit
/// — refusal, cancel (the ffmpeg runner returns `Canceled`), panic, or the future being dropped.
async fn resolve_icl_reference(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &AudioRequest,
    project_path: &Path,
) -> WorkerResult<Option<Conditioning>> {
    let references = request.icl_references()?;
    if references.is_empty() {
        return Ok(None);
    }
    let mut decoded: Vec<(Option<&'static str>, gen_core::AudioTrack)> = Vec::new();
    for (stem, asset_id) in references {
        let source = crate::video_jobs::ltx::resolve_clip_media_path(
            settings,
            &request.project_id,
            asset_id,
            project_path,
        )?;
        let track = decode_icl_clip(api, settings, &job.id, &source).await?;
        decoded.push((stem, track));
    }
    Ok(Some(Conditioning::ReferenceAudio {
        audio: assemble_icl_track(decoded),
        strength: None,
    }))
}

/// Decode one ICL clip inside a job-scoped scratch dir under the system temp root (see
/// [`resolve_icl_reference`] for the cleanup contract).
async fn decode_icl_clip(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    source: &Path,
) -> WorkerResult<gen_core::AudioTrack> {
    // `scratch` drops on success AND on every early return, removing the directory.
    let scratch = icl_scratch_dir(job_id)?;
    crate::video_jobs::reference_audio::decode_audio_normalized(
        api,
        settings,
        job_id,
        CANCEL_MESSAGE,
        source,
        scratch.path(),
        ICL_REFERENCE_SAMPLE_RATE,
        ICL_REFERENCE_CHANNELS,
    )
    .await
}

/// The prefix of every ICL scratch dir for `job_id` — the handle the cleanup test sweeps for.
fn icl_scratch_prefix(job_id: &str) -> String {
    format!("sw-yue-icl-{}-", safe_download_dir(job_id))
}

fn icl_scratch_dir(job_id: &str) -> WorkerResult<tempfile::TempDir> {
    Ok(tempfile::Builder::new()
        .prefix(&icl_scratch_prefix(job_id))
        .tempdir()?)
}

/// Build the engine's ICL reference track from decoded clips: one unnamed clip passes through; a
/// named pair becomes stems over a common length, with the mix field their sample-wise sum.
fn assemble_icl_track(
    mut decoded: Vec<(Option<&'static str>, gen_core::AudioTrack)>,
) -> gen_core::AudioTrack {
    if decoded.len() == 1 && decoded[0].0.is_none() {
        return decoded.remove(0).1;
    }
    let len = decoded
        .iter()
        .map(|(_, track)| track.samples.len())
        .min()
        .unwrap_or(0);
    let (sample_rate, channels) = decoded
        .first()
        .map(|(_, track)| (track.sample_rate, track.channels))
        .unwrap_or((ICL_REFERENCE_SAMPLE_RATE, ICL_REFERENCE_CHANNELS));
    let mut samples = vec![0.0f32; len];
    let stems = decoded
        .into_iter()
        .map(|(name, mut track)| {
            track.samples.truncate(len);
            for (sum, sample) in samples.iter_mut().zip(&track.samples) {
                *sum += sample;
            }
            gen_core::AudioStem {
                name: name.unwrap_or("reference").to_owned(),
                samples: track.samples,
            }
        })
        .collect();
    gen_core::AudioTrack {
        samples,
        sample_rate,
        channels,
        stems,
    }
}

pub(crate) async fn run_audio_generate_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    run_audio_generate_job_using(api, settings, job, crate::inference_runtime::load_audio).await
}

/// [`run_audio_generate_job`] with the single-generator lane's loader injected (sc-19384) — the same
/// seam [`run_audio_synthesis_using`] exposes, lifted to the whole job so a test drives the REAL
/// preflight → tier/ICL resolution → synthesis → persistence path against a stub [`Generator`] and
/// asserts what lands in the project (the mix plus every stem the track carries). The voice-clone
/// chains keep their own production loaders; they are not reachable from a stubbed Single job.
async fn run_audio_generate_job_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<()> {
    let request = AudioRequest::from_payload(&job.payload);
    audio_preflight(&request)?;
    // YuE whole-render memory admission (sc-19386): refuse a render that cannot fit before the
    // project, weights, reference clip or source track is touched. It prices the tier THIS job will
    // load — the same `resolve_audio_tier` the synthesis arm runs, so a missing install or tier is
    // refused here with that arm's own message. Skipped for non-YuE models.
    if crate::yue_admission::is_yue(&request.model_manifest_entry) {
        let (_, tier) = resolve_audio_tier(&request, resolve_audio_model_dir(settings, &request)?)?;
        crate::yue_admission::check(
            &request.model,
            &request.model_manifest_entry,
            &yue_request_facts(&request, tier.as_ref()),
            &settings.gpu_id,
        )
        .await?;
    }
    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let plan = AudioPlan::new(&request, &project_path);

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
        // A model shipping physical per-tier weights (YuE, sc-19384) loads from its tier subdir and
        // asserts that tier on the load; every other audio model passes the snapshot through.
        let (model_dir, tier) =
            resolve_audio_tier(&request, resolve_audio_model_dir(settings, &request)?)?;
        // Extend/edit SOURCE band (sc-13410): resolve + decode the source track and build the
        // `Conditioning::AudioEdit` here (in async, where the project path + store are in scope), then move
        // it into the blocking synthesis. `None` ⇒ plain text-to-music. The per-model gates (edit mode ∈
        // advertised, region inside the clip, 48 kHz source) run in the generator's `validate` at synthesis.
        // The ICL reference (sc-19384) is the other, mutually exclusive, conditioning source.
        let conditioning = match build_audio_edit(settings, &request, &project_path)? {
            Some(edit) => Some(edit),
            None => resolve_icl_reference(api, settings, job, &request, &project_path).await?,
        };
        AudioSynthesis::Single(SinglePlan {
            model_dir,
            tier,
            conditioning,
        })
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
    let mut resolved_tier = None;
    let track = match synthesis {
        AudioSynthesis::Single(single) => {
            resolved_tier = single.tier.as_ref().map(|tier| tier.name.clone());
            run_audio_synthesis_with(api, settings, job, &request, single, load_generator).await?
        }
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
    // Dead-render guard. An exact `peak == 0.0` test is too weak: a collapsed diffusion render
    // returns a residual noise floor, not true zeros (a MOSS-SoundEffect run at its default 100
    // solver steps came back at peak 0.02 / RMS 0.0002 — inaudible, but not zero), so it slipped
    // through and was registered as a real asset. Gate on RMS instead: -60 dBFS is far below any
    // usable render while still admitting deliberately quiet content, which a peak test cannot
    // distinguish because one stray sample lifts the peak arbitrarily. Applied to the MIX only: a
    // source-separated stem may legitimately be near-silent (an instrumental passage's vocal stem).
    const MIN_RMS: f32 = 1e-3; // -60 dBFS
    let rms = if sample_count == 0 {
        0.0
    } else {
        (track.samples.iter().map(|s| s * s).sum::<f32>() / sample_count as f32).sqrt()
    };
    if sample_count == 0 || peak == 0.0 || rms < MIN_RMS {
        return Err(WorkerError::Engine(format!(
            "{}: the audio generator produced silence ({sample_count} samples, peak {peak:.5}, \
             RMS {rms:.6} < {MIN_RMS}) — refusing to register an empty clip.",
            request.model
        )));
    }

    // Every source-separated stem the model genuinely emitted (YuE: `vocals` + `instrumental`,
    // sc-19384) is persisted as its own library asset beside the mix. A stem shares the mix's rate
    // and channel layout (the gen-core `AudioStem` contract); an empty one is refused rather than
    // registered as a zero-length asset.
    let stem_plans: Vec<StemPlan> = track
        .stems
        .iter()
        .map(|stem| StemPlan::new(&plan, &stem.name))
        .collect();
    for stem in &track.stems {
        if stem.samples.is_empty() {
            return Err(WorkerError::Engine(format!(
                "{}: the audio generator returned an empty `{}` stem.",
                request.model, stem.name
            )));
        }
    }
    let mut writes = vec![(
        plan.media_path.clone(),
        AudioTrack {
            samples: track.samples,
            sample_rate,
            channels,
        },
    )];
    let mut stem_durations = Vec::with_capacity(stem_plans.len());
    for (stem, stem_plan) in track.stems.into_iter().zip(&stem_plans) {
        stem_durations.push(stem.samples.len() as f64 / (sample_rate as f64 * channels as f64));
        writes.push((
            stem_plan.media_path.clone(),
            AudioTrack {
                samples: stem.samples,
                sample_rate,
                channels,
            },
        ));
    }
    // The generation-set directory is created only now, at the first write — a job that fails or is
    // canceled before this point leaves nothing behind in the project — and removed again if any
    // write fails, so a half-written set never lingers.
    let genset_dir = plan.media_path.parent().map(Path::to_path_buf);
    let written = tokio::task::spawn_blocking(move || -> WorkerResult<()> {
        for (path, wav) in &writes {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            write_wav_pcm16(wav, path)?;
        }
        Ok(())
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))
    .and_then(|result| result);
    if let Err(error) = written {
        if let Some(dir) = genset_dir {
            let _ = tokio::fs::remove_dir_all(dir).await;
        }
        return Err(error);
    }

    let mut fact = audio_asset_fact(
        &plan,
        &request,
        sample_rate,
        channels,
        duration_secs,
        native_clone,
    );
    record_song_settings(&mut fact, &request, resolved_tier.as_deref());
    let mut facts = Vec::with_capacity(1 + stem_plans.len());
    if !stem_plans.is_empty() {
        insert_fact_extra(
            &mut fact,
            json!({
                "audioStem": "mix",
                "stemAssetIds": stem_plans
                    .iter()
                    .map(|stem| json!({ "stem": stem.name, "assetId": stem.asset_id }))
                    .collect::<Vec<_>>(),
            }),
        );
    }
    for (stem_plan, duration) in stem_plans.iter().zip(stem_durations) {
        facts.push(stem_asset_fact(&fact, &plan, stem_plan, duration));
    }
    facts.insert(0, fact);
    let result = audio_streaming_result(&plan, &request, facts);
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

/// One source-separated stem's asset slot beside the mix (sc-19384): its own asset id and a media
/// path in the mix's generation-set directory, suffixed with the stem name.
struct StemPlan {
    name: String,
    asset_id: String,
    media_rel: String,
    media_path: PathBuf,
}

impl StemPlan {
    fn new(plan: &AudioPlan, name: &str) -> Self {
        // The stem name comes from the engine; slugify it before it becomes a path component.
        let stem_slug = slugify(name, "stem", Some(24));
        let media_rel = match plan.media_rel.strip_suffix(".wav") {
            Some(stem) => format!("{stem}_{stem_slug}.wav"),
            None => format!("{}_{stem_slug}.wav", plan.media_rel),
        };
        let media_path = plan
            .media_path
            .parent()
            .map(|parent| {
                parent.join(
                    Path::new(&media_rel)
                        .file_name()
                        .expect("stem media path has a file name"),
                )
            })
            .unwrap_or_else(|| PathBuf::from(&media_rel));
        Self {
            name: name.to_owned(),
            asset_id: fresh_asset_id(),
            media_rel,
            media_path,
        }
    }
}

/// Merge `extra` into the fact's `extra` object — the free-form block the sidecar persists verbatim.
fn insert_fact_extra(fact: &mut Value, extra: Value) {
    let Some(object) = fact.as_object_mut() else {
        return;
    };
    let slot = object
        .entry("extra".to_owned())
        .or_insert_with(|| json!({}));
    if let (Some(slot), Some(extra)) = (slot.as_object_mut(), extra.as_object()) {
        for (key, value) in extra {
            slot.insert(key.clone(), value.clone());
        }
    }
}

/// A stem's asset fact: the mix's replay record (same recipe, same knobs), re-addressed to the stem's
/// own asset id + WAV, named for the stem, with the mix as its lineage parent.
fn stem_asset_fact(mix: &Value, plan: &AudioPlan, stem: &StemPlan, duration_secs: f64) -> Value {
    let mut fact = mix.clone();
    let display = mix
        .get("displayName")
        .and_then(Value::as_str)
        .unwrap_or("Generated audio");
    if let Some(object) = fact.as_object_mut() {
        object.insert("assetId".to_owned(), json!(stem.asset_id));
        object.insert("mediaPath".to_owned(), json!(stem.media_rel));
        object.insert("duration".to_owned(), json!(duration_secs));
        object.insert(
            "displayName".to_owned(),
            json!(format!("{display} ({})", stem.name)),
        );
        object.insert("parents".to_owned(), json!([plan.asset_id]));
        object.insert(
            "extra".to_owned(),
            json!({ "audioStem": stem.name, "mixAssetId": plan.asset_id }),
        );
    }
    fact
}

/// Record the segmented-song / ICL / tier knobs (sc-19384) in the asset's replay record, so a
/// re-generate reconstructs the exact request. Written into `rawAdapterSettings` (persisted verbatim)
/// after the fact is built — the base `json!` literal is already at the macro recursion limit.
fn record_song_settings(fact: &mut Value, request: &AudioRequest, tier: Option<&str>) {
    let settings = json!({
        "segments": request.segments,
        "maxNewTokensPerSegment": request.max_new_tokens_per_segment,
        "repetitionPenalty": request.repetition_penalty,
        "guidanceEnabled": request.guidance_enabled,
        "iclMode": request.icl_mode,
        "iclReferenceAssetId": request.icl_reference_asset_id,
        "iclVocalAssetId": request.icl_vocal_asset_id,
        "iclInstrumentalAssetId": request.icl_instrumental_asset_id,
        "iclStartSecs": request.icl_start_secs,
        "iclEndSecs": request.icl_end_secs,
        "quantTier": tier,
        "outputLimiter": request.output_limiter,
    });
    if let Some(raw) = fact
        .get_mut("rawAdapterSettings")
        .and_then(Value::as_object_mut)
    {
        for (key, value) in settings.as_object().expect("json! object literal") {
            raw.insert(key.clone(), value.clone());
        }
    }
    // The ICL reference clips are this render's lineage parents.
    let parents: Vec<&str> = [
        &request.icl_reference_asset_id,
        &request.icl_vocal_asset_id,
        &request.icl_instrumental_asset_id,
    ]
    .into_iter()
    .filter_map(|id| id.as_deref())
    .collect();
    if !parents.is_empty() {
        if let Some(object) = fact.as_object_mut() {
            object.insert("parents".to_owned(), json!(parents));
        }
    }
}

/// A resolved single-generator job: the weights dir, the resolved tier (tiered models only), and the
/// one conditioning the request carries (an extend/edit source OR an ICL reference).
struct SinglePlan {
    model_dir: PathBuf,
    tier: Option<AudioTier>,
    conditioning: Option<Conditioning>,
}

/// Coalesced progress-post cadence for the synthesis pump (sc-19384 review; the sc-11189 F-016
/// pattern `caption_jobs` / `prompt_refine_jobs` use). Engine `Progress` and streamed chunks can fire
/// per token / per frame (MOSS-TTS, Chatterbox T3), so the synthesis thread publishes into a
/// latest-wins `watch` channel and the pump posts at most once per interval — always including the
/// final state — instead of one awaited `update_job` per event.
const PROGRESS_POST_INTERVAL: Duration = Duration::from_millis(250);

/// One Running job update the pump may post: `(stage, fraction, message)`.
type SynthesisUpdate = (ProgressStage, f64, String);

/// Folds streamed chunks (sc-13675) and engine [`Progress`] events (sc-19384) into the latest job
/// update inside the (Generating 0.2 → Saving 0.9) band. The fraction is monotone: every event takes
/// `max(previous, candidate)`, so a model that interleaves `Decoding` and `Step`s (or chunks and
/// steps) never moves the bar backwards.
#[derive(Debug)]
struct SynthesisProgress {
    chunks: usize,
    fraction: f64,
}

impl SynthesisProgress {
    fn new() -> Self {
        Self {
            chunks: 0,
            fraction: 0.2,
        }
    }

    /// A streamed chunk (1-based running count). The total is unknown ahead of time (the AR loop
    /// decides its own length via EOS), so the fraction approaches but never reaches 0.9.
    fn chunk(&mut self, latest: usize) -> SynthesisUpdate {
        self.chunks = self.chunks.max(latest);
        let count = self.chunks;
        self.fraction = self
            .fraction
            .max(0.25 + 0.6 * (count as f64 / (count as f64 + 6.0)));
        (
            ProgressStage::Generating,
            self.fraction,
            format!(
                "Streaming audio… ({count} chunk{})",
                if count == 1 { "" } else { "s" }
            ),
        )
    }

    /// An engine event: `Step { current, total }` advances proportionally (YuE: one step per lyric
    /// segment and one per stage-2 track), `Loading` names the stage being loaded, `Decoding` marks
    /// the codec/vocoder pass. `None` for a degenerate `total == 0` step.
    fn engine(&mut self, progress: Progress) -> Option<SynthesisUpdate> {
        let (stage, candidate, message) = match progress {
            Progress::Step { current, total } => {
                if total == 0 {
                    return None;
                }
                let done = f64::from(current.min(total)) / f64::from(total);
                (
                    ProgressStage::Generating,
                    0.2 + 0.65 * done,
                    format!("Generating audio ({current}/{total})."),
                )
            }
            Progress::Loading(phase) => (
                ProgressStage::LoadingModel,
                self.fraction,
                match phase {
                    gen_core::LoadPhase::TextEncoder => "Loading the text encoder.".to_owned(),
                    gen_core::LoadPhase::Renderer => "Loading model weights.".to_owned(),
                },
            ),
            Progress::Decoding => (
                ProgressStage::Generating,
                0.85,
                "Decoding audio.".to_owned(),
            ),
        };
        self.fraction = self.fraction.max(candidate);
        Some((stage, self.fraction, message))
    }
}

/// [`run_audio_synthesis_with`] for an untiered model with at most an extend/edit conditioning — the
/// pre-sc-19384 seam the cancel/streaming/component tests drive.
#[cfg(test)]
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
    run_audio_synthesis_with(
        api,
        settings,
        job,
        request,
        SinglePlan {
            model_dir,
            tier: None,
            conditioning: audio_edit,
        },
        load_generator,
    )
    .await
}

/// Load the audio generator and run one synthesis on a blocking thread, honoring a mid-synthesis
/// cancel through the shared [`run_blocking_with_heartbeat`] watcher (sc-13469), with the generator
/// loader injected — the audio sibling of [`crate::video_jobs::generate_video_using`]: with the load
/// threaded in, a test drives the REAL synthesis path against a stub [`Generator`] and asserts the
/// shared engine [`CancelFlag`] that reaches [`GenerationRequest::cancel`] is tripped mid-generation
/// (not only by the post-synthesis `check_cancel` after the whole clip has rendered). Returns the
/// engine [`AudioTrack`] (`gen_core`'s, stems included) for the caller to write + register.
async fn run_audio_synthesis_with(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &AudioRequest,
    single: SinglePlan,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<gen_core::AudioTrack> {
    let SinglePlan {
        model_dir,
        tier,
        conditioning,
    } = single;
    let model_id = request.model.clone();
    let prompt = request.prompt.clone();
    let voice = request.voice.clone();
    let language = request.language.as_deref().map(normalize_audio_language);
    let target_duration = request.target_duration_secs;
    // Multi-speaker dialogue script (sc-13676): rides AudioParams.script. `None` for a single-voice
    // request, so the built AudioParams is identical to the pre-sc-13676 one for those; a model
    // advertising `supports_multi_speaker` (MOSS-TTSD) renders each segment in its own voice, and the
    // gen-core floor rejects a script sent to a model that does not advertise it (typed Unsupported).
    let script = request.script.clone();
    // Diffusion-audio sampling knobs (Sound FX / MOSS-SoundEffect, sc-13409). These live on the
    // top-level GenerationRequest — the flow-matching pipeline reads `req.guidance` (CFG scale) and
    // `req.steps`, not `AudioParams`. `None` ⇒ the generator's own sampler default; the shared
    // gen-core floor range-checks any value we pass. A TTS model (Kokoro) ignores them entirely, so
    // Speech jobs — which never carry them — are unaffected. Guidance switched OFF (sc-19384) rides
    // as `0.0`, which a segmented-song model reads as "no CFG".
    let guidance = request.effective_guidance();
    let steps = request.steps;
    // Music describe-the-music sub-block (ACE-Step, sc-13410). BPM/key/lyrics ride the AudioParams
    // music fields; negative_prompt rides the top-level request. A model that doesn't consume one
    // ignores it; a model that advertises no negative-prompt support rejects a supplied one at the
    // gen-core floor (the studio only sends one to a model that advertises support).
    let negative_prompt = request.negative_prompt.clone();
    let bpm = request.bpm;
    let musical_key = request.musical_key.clone();
    let lyrics = request.lyrics.clone();
    // Segmented-song controls (YuE, sc-19384): each is `None` unless the request set it, and the
    // gen-core floor refuses one sent to a model that does not advertise reading it.
    let segments = request.segments;
    let max_new_tokens_per_segment = request.max_new_tokens_per_segment;
    let repetition_penalty = request.repetition_penalty;
    let reference_region = request.reference_region();
    let output_limiter = request
        .output_limiter
        .as_deref()
        .map(output_limiter)
        .transpose()?;
    let seed = request.seed.map(|seed| seed as u64);
    // sc-13469: ONE shared engine CancelFlag — cloned into the request the blocking synthesis runs
    // AND handed to `run_blocking_with_heartbeat`, whose watcher polls the API cancel state (and a
    // worker shutdown) and trips it MID-synthesis. Replaces the degenerate inline `CancelFlag::new()`
    // that was never tripped, so a user cancel is honored DURING the clip render — not only by the
    // post-synthesis `check_cancel` after the whole clip has already rendered.
    let cancel = CancelFlag::new();
    // Incremental progress (sc-13675 streaming chunks; sc-19384 engine Progress events). A
    // streaming-capable Generator drives `generate_streaming` and emits an `AudioChunk` per PCM block
    // as the AR loop decodes it; any generator may report `Progress` (YuE: one `Step` per lyric
    // segment and per stage-2 track, `Loading` per LM stage, `Decoding` before the codec/vocoder).
    // Both fold (monotone) into the LATEST job update, published into a latest-wins `watch` channel
    // that the concurrent async pump below posts at most once per [`PROGRESS_POST_INTERVAL`] (always
    // including the final state), so the Audio Studio's WorkerProgressCard advances THROUGH the
    // render without a per-token POST storm. The reassembled chunks equal the returned one-shot track
    // (the gen-core reassembly law), so the library asset is still the full `AudioTrack`.
    let (update_tx, mut update_rx) = tokio::sync::watch::channel::<Option<SynthesisUpdate>>(None);
    // Named model components (epic 13657, sc-13679): resolve every coRequisite-provisioned component
    // this model's descriptor advertises (`chatterbox_tts` → `perth` + `voice_embedding`; YuE →
    // `stage2` + `xcodec`; most audio models advertise none → an empty map, a no-op) BEFORE the
    // blocking load, so a missing co-requisite fails the JOB here with an actionable error rather
    // than a mid-render hub fetch. A per-tier component (YuE's `stage2`, sc-19384) resolves to the
    // SAME tier as the primary weights. The resolved paths are staged in `LoadSpec::components` for
    // the generator's load-time `require_component` gate. Weights-free registry read; no-op when this
    // build ships no audio lane.
    let mut components = match crate::inference_runtime::audio_descriptor(&model_id) {
        Some(descriptor) => crate::model_jobs::resolve_co_requisites_for_tier(
            &descriptor,
            &request.model_manifest_entry,
            settings,
            tier.as_ref().map(|tier| tier.name.as_str()),
        )?,
        None => BTreeMap::new(),
    };
    // Optional, on-demand Cover component (sc-13821): acestep's ~8.76 GB `sft_cover` snapshot — the
    // non-distilled reference cover DiT plus the FSQ audio_tokenizer/detokenizer — is staged ONLY for a
    // Cover restyle request. It is deliberately NOT a `required_components` id (Cover-only, lazy,
    // mirroring LTX `uncensored_enhancer`), so the generic `resolve_co_requisites` above never stages
    // it; text2music and the Inpaint/Repaint/Extend region edits load without it. The id matches
    // `candle_audio_acestep::COVER_COMPONENT_ID` ("sft_cover"); resolve it from the model's own soft
    // `sft_cover` coRequisite in the manifest. At the adopted inference pin (sc-13756) the Cover path
    // reads this snapshot ONLY from `LoadSpec::components["sft_cover"]` (the env-var/self-fetch read was
    // removed), so this attach is the sole offline Cover source. An absent snapshot stages nothing; the
    // engine then surfaces the actionable Cover-needs-`sft_cover` error on the request that needs it.
    if request.edit_mode.as_deref() == Some("cover") {
        if let Some(source) = crate::model_jobs::resolve_optional_component(
            &request.model_manifest_entry,
            "sft_cover",
            settings,
        ) {
            components.insert("sft_cover".to_string(), source);
        }
    }
    let quantize = tier.and_then(|tier| tier.quantize);
    let handle = {
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<gen_core::AudioTrack> {
            let mut spec = components.into_iter().fold(
                LoadSpec::new(WeightsSource::Dir(model_dir)),
                |spec, (id, source)| spec.with_component(id, source),
            );
            // A tiered model asserts the resolved tier on the load (YuE: Q8 / Q4; bf16 is the
            // unquantized load and asserts nothing).
            if let Some(quant) = quantize {
                spec = spec.with_quant(quant);
            }
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
                    script,
                    segments,
                    max_new_tokens_per_segment,
                    repetition_penalty,
                    reference_region,
                    output_limiter,
                    ..Default::default()
                }),
                // The request's one conditioning: an extend/edit source band
                // (Conditioning::AudioEdit) or an ICL reference (Conditioning::ReferenceAudio);
                // empty for plain generation.
                conditioning: conditioning.into_iter().collect(),
                // The shared, watcher-tripped flag (sc-13469) — NOT a fresh `CancelFlag::new()`.
                cancel,
                ..Default::default()
            };
            // Engine progress and streamed chunks fold into one latest update (sc-19384). Publishing
            // never blocks and never fails — synthesis must not depend on the progress sink.
            let fold = std::cell::RefCell::new(SynthesisProgress::new());
            let publish = |update: SynthesisUpdate| {
                update_tx.send_replace(Some(update));
            };
            let mut on_progress = |progress: Progress| {
                let update = fold.borrow_mut().engine(progress);
                if let Some(update) = update {
                    publish(update);
                }
            };
            // Gate PURELY on the loaded generator's advertised capability (sc-13675), never a hardcoded
            // id: a `supports_streaming` model streams incremental chunks; every other model keeps the
            // exact one-shot `generate` path unchanged. `generate_streaming` also returns the same
            // aggregate `GenerationOutput` as `generate`, so the written asset is identical either way.
            let output = if generator.descriptor().capabilities.supports_streaming {
                let mut on_chunk = |chunk: gen_core::AudioChunk| {
                    // The 1-based running chunk count.
                    let update = fold.borrow_mut().chunk(chunk.index.saturating_add(1));
                    publish(update);
                };
                generator.generate_streaming(&req, &mut on_chunk, &mut on_progress)
            } else {
                generator.generate(&req, &mut on_progress)
            }
            .map_err(|error| classify_audio_synthesis_error("audio generation failed", error))?;
            match output {
                GenerationOutput::Audio(track) => Ok(track),
                // Either channel count is the same defect: an audio job asked for a track and
                // got pixels. `ImagesRgba` (sc-24111) is unreachable here — this request never
                // sets `output_channels` — but the match must name it.
                GenerationOutput::Images(_) | GenerationOutput::ImagesRgba(_) => {
                    Err(WorkerError::Engine(
                        "audio model returned images, expected an audio track".to_owned(),
                    ))
                }
                GenerationOutput::Video { .. } => Err(WorkerError::Engine(
                    "audio model returned video, expected an audio track".to_owned(),
                )),
            }
        })
    };
    // Coalescing progress pump (sc-13675 / sc-19384): waits for the synthesis thread to publish a
    // new latest update, then posts it — at most once per [`PROGRESS_POST_INTERVAL`], so a model
    // reporting per token/frame costs a handful of POSTs per second, never one per event. When
    // synthesis ends the sender drops (the blocking closure owns it), and the pump posts the final
    // unposted state (if any) and exits, so draining it below is bounded by one interval + one POST.
    // It runs ALONGSIDE `run_blocking_with_heartbeat` (which owns worker heartbeats + the cancel
    // watcher on a DIFFERENT endpoint) and stops posting once the shared cancel flag is observed; a
    // POST already in flight when a cancel lands gets a 409 from jobs_store (no nonterminal write
    // after `Canceled`), which is deliberately ignored. A generator that reports nothing never
    // publishes, so the pump exits with zero posts.
    let pump = {
        let api = api.clone();
        let job_id = job.id.clone();
        let backend = backend_label(&settings.gpu_id).to_owned();
        let cancel = cancel.clone();
        tokio::spawn(async move {
            let mut last_sent: Option<SynthesisUpdate> = None;
            let mut last_post: Option<tokio::time::Instant> = None;
            let mut posts = 0usize;
            loop {
                let closed = update_rx.changed().await.is_err();
                if cancel.is_cancelled() {
                    break;
                }
                if !closed {
                    if let Some(at) = last_post {
                        tokio::time::sleep_until(at + PROGRESS_POST_INTERVAL).await;
                    }
                    if cancel.is_cancelled() {
                        break;
                    }
                }
                let latest = update_rx.borrow_and_update().clone();
                if let Some(update) = latest {
                    if last_sent.as_ref() != Some(&update) {
                        let (stage, fraction, message) = update.clone();
                        let _ = update_job(
                            &api,
                            &job_id,
                            audio_progress(
                                JobStatus::Running,
                                stage,
                                fraction,
                                &message,
                                None,
                                &backend,
                            ),
                        )
                        .await;
                        posts += 1;
                        last_post = Some(tokio::time::Instant::now());
                        last_sent = Some(update);
                    }
                }
                if closed {
                    break;
                }
            }
            posts
        })
    };
    // The shared blocking keepalive + cancel watcher (sc-13469): pings the worker heartbeat while the
    // synthesis runs (so a long/cold render is never swept to `interrupted`), polls the API cancel
    // state each interval and trips the SAME `cancel` the request carries, and on completion tears the
    // watcher down cleanly (bounded-join, no leaked task, no false-trip on normal completion). Mirrors
    // the video path's in-loop cancel watcher.
    let result = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        CANCEL_MESSAGE,
        "audio synthesis",
        no_cancel_ack(),
        handle,
    )
    .await;
    // Drain the pump. Synthesis has finished, so its sender is gone: the pump posts at most the one
    // final unposted update (after at most one interval) and exits — bounded, never hangs.
    let _ = pump.await;
    result
}

/// Which synthesis path a resolved audio job takes — the single-generator lane (Speech / SFX / Music)
/// or the two-call voice-clone chain (sc-13411 C4). Resolved before the job is marked Running so a
/// missing install / reference surfaces as a clear preflight error.
enum AudioSynthesis {
    Single(SinglePlan),
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
    if !request.base_model_provided {
        return Err(WorkerError::InvalidPayload(
            "baseModel is required for converter-based voice cloning.".to_owned(),
        ));
    }
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
    let reference_path = crate::video_jobs::ltx::resolve_clip_media_path(
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
    let reference_path = crate::video_jobs::ltx::resolve_clip_media_path(
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
    // Named model components (epic 13657, sc-13679/13686): the native clone generator advertises the
    // coRequisite-provisioned components it gates on at load — `chatterbox_tts` requires `perth` +
    // `voice_embedding` (its descriptor's `required_components`, enforced by `require_component` at the
    // top of the provider's `load`). Resolve each from the model's manifest entry and stage it in
    // `LoadSpec::components` BEFORE the blocking load — the SAME seam `run_audio_synthesis_using` uses
    // (sc-13679), so a missing co-requisite fails the JOB here with an actionable error naming the
    // component id + repo rather than at the engine's `require_component` gate (or a mid-render hub
    // fetch), and — crucially — an INSTALLED co-requisite is actually staged so the render can load.
    // Without this the native clone path built a component-less `LoadSpec`, so at the sc-13680 pin
    // (generator no longer self-fetches ve/perth) chatterbox_tts could not load even fully installed.
    // A generator that advertises no components (a future native clone) yields an empty map — a no-op.
    let components = match crate::inference_runtime::audio_descriptor(&model_id) {
        Some(descriptor) => {
            resolve_co_requisites(&descriptor, &request.model_manifest_entry, settings)?
        }
        None => BTreeMap::new(),
    };
    // sc-13469: ONE shared engine CancelFlag, tripped mid-synthesis by the keepalive watcher.
    let cancel = CancelFlag::new();
    let handle = {
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<gen_core::AudioTrack> {
            let spec = components.into_iter().fold(
                LoadSpec::new(WeightsSource::Dir(model_dir)),
                |spec, (id, source)| spec.with_component(id, source),
            );
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
                GenerationOutput::Images(_) | GenerationOutput::ImagesRgba(_) => {
                    Err(WorkerError::Engine(
                        "clone-TTS model returned images, expected an audio track".to_owned(),
                    ))
                }
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
    let source_path = crate::video_jobs::ltx::resolve_clip_media_path(
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
        // Multi-speaker dialogue script (sc-13676) — `null` on a single-voice run.
        "script": script_to_json(&request.script),
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
        // Multi-speaker dialogue script (sc-13676): the ordered segments round-trip for replay so a
        // re-generate reconstructs the same dialogue. `null` on a single-voice run.
        "script": script_to_json(&request.script),
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
fn audio_streaming_result(
    plan: &AudioPlan,
    request: &AudioRequest,
    facts: Vec<Value>,
) -> JsonObject {
    // One asset per written WAV: the mix, then each source-separated stem (sc-19384).
    let count = facts.len();
    let adapter = facts
        .first()
        .and_then(|fact| fact.get("adapter"))
        .cloned()
        .unwrap_or(Value::Null);
    json!({
        "generationSetId": plan.genset_id,
        "expectedCount": count,
        "adapter": adapter,
        "model": request.model,
        "generationSet": {
            "id": plan.genset_id,
            "mode": request.mode(),
            "model": request.model,
            "prompt": request.prompt,
            "count": count,
            "createdAt": plan.created_at,
        },
        "assetWrites": facts,
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

        let mut no_model_payload = payload(json!({}));
        no_model_payload.remove("model");
        let no_model = AudioRequest::from_payload(&no_model_payload);
        assert!(matches!(
            audio_preflight(&no_model),
            Err(WorkerError::InvalidPayload(message)) if message == "model is required."
        ));
    }

    #[test]
    fn parse_speech_segments_reads_the_wire_shape() {
        // Absent / non-array / empty → None (so a single-voice request builds AudioParams.script:
        // None, byte-for-byte unaffected).
        assert!(parse_speech_segments(None).is_none());
        assert!(parse_speech_segments(Some(&json!("not-an-array"))).is_none());
        assert!(parse_speech_segments(Some(&json!([]))).is_none());
        // A whitespace-only-text segment is dropped defensively; if that leaves nothing → None.
        assert!(parse_speech_segments(Some(&json!([{ "text": "   " }]))).is_none());
        // Well-formed segments map text + optional speaker/style; blank speaker/style → None.
        let script = parse_speech_segments(Some(&json!([
            { "text": "Hi there.", "speaker": "S1", "style": "cheerful" },
            { "text": "Hello!", "speaker": "  ", "style": "" },
        ])))
        .expect("a non-empty script parses");
        assert_eq!(script.len(), 2);
        assert_eq!(script[0].text, "Hi there.");
        assert_eq!(script[0].speaker.as_deref(), Some("S1"));
        assert_eq!(script[0].style.as_deref(), Some("cheerful"));
        assert_eq!(script[1].text, "Hello!");
        assert_eq!(
            script[1].speaker, None,
            "a blank speaker is dropped to None"
        );
        assert_eq!(script[1].style, None);
    }

    #[test]
    fn from_payload_reads_the_multi_speaker_script() {
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "moss_ttsd_v05",
            "prompt": "",
            "script": [
                { "text": "Line one.", "speaker": "S1" },
                { "text": "Line two.", "speaker": "S2" },
            ],
        })));
        let script = request.script.as_ref().expect("script parsed");
        assert_eq!(script.len(), 2);
        assert_eq!(script[1].speaker.as_deref(), Some("S2"));
    }

    #[test]
    fn preflight_accepts_a_script_only_request_and_still_gates_the_empty_case() {
        // A multi-speaker request carries its text in the script — an empty prompt is fine THEN.
        let script_only = AudioRequest::from_payload(&payload(json!({
            "model": "moss_ttsd_v05",
            "prompt": "",
            "script": [{ "text": "Hello.", "speaker": "S1" }],
        })));
        assert!(script_only.script.is_some());
        assert!(audio_preflight(&script_only).is_ok());
        // But an empty prompt AND no script is still rejected (single-voice unaffected).
        let neither = AudioRequest::from_payload(&payload(json!({ "prompt": "" })));
        assert!(neither.script.is_none());
        assert!(audio_preflight(&neither).is_err());
    }

    #[test]
    fn asset_fact_records_the_script_for_replay_and_null_for_single_voice() {
        // Multi-speaker: the ordered dialogue round-trips as an array in the replay record.
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "moss_ttsd_v05",
            "prompt": "",
            "script": [
                { "text": "First turn.", "speaker": "S1" },
                { "text": "Second turn.", "speaker": "S2", "style": "warm" },
            ],
        })));
        let plan = AudioPlan::new(&request, Path::new("/tmp/project"));
        let fact = audio_asset_fact(&plan, &request, 24_000, 1, 5.0, false);
        let script = fact["script"]
            .as_array()
            .expect("script recorded as an array");
        assert_eq!(script.len(), 2);
        assert_eq!(script[0]["text"], "First turn.");
        assert_eq!(script[0]["speaker"], "S1");
        assert_eq!(script[1]["speaker"], "S2");
        assert_eq!(script[1]["style"], "warm");
        assert_eq!(fact["rawAdapterSettings"]["script"], fact["script"]);

        // Single-voice: `script` is null (the byte-for-byte-unaffected replay record).
        let single = AudioRequest::from_payload(&payload(json!({ "voice": "af_heart" })));
        let single_plan = AudioPlan::new(&single, Path::new("/tmp/project"));
        let single_fact = audio_asset_fact(&single_plan, &single, 24_000, 1, 3.0, false);
        assert!(single_fact["script"].is_null());
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
        let result = audio_streaming_result(&plan, &request, vec![fact.clone()]);
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

        // Parsing retains the legacy value for replay helpers, but the converter-plan resolver rejects
        // the omission so a direct worker payload can never silently select Kokoro.
        let defaulted = AudioRequest::from_payload(&payload(json!({
            "model": "openvoice_v2",
            "referenceAudioAssetId": "ref-voice-1",
        })));
        assert_eq!(defaulted.base_model, "kokoro_82m");
        assert!(defaulted.is_voice_clone());
        assert!(audio_preflight(&defaulted).is_ok());
        assert!(matches!(
            resolve_voice_clone_plan(
                &Settings::from_env(),
                &defaulted,
                Path::new("/unused-project"),
            ),
            Err(WorkerError::InvalidPayload(message))
                if message == "baseModel is required for converter-based voice cloning."
        ));

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
        let result = audio_streaming_result(&plan, &request, vec![fact.clone()]);
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
        assert!(audio_preflight(&request).is_ok());
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
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &[],
            control_kinds: None,
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

    /// A descriptor that advertises `supports_streaming: true` — so `run_audio_synthesis_using`
    /// takes the `generate_streaming` path (the gate reads the loaded generator's capability, never a
    /// hardcoded id). The audio twin of the real moss_tts_realtime descriptor's streaming flag.
    fn streaming_stub_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "stub_streaming_audio",
            family: "stub",
            backend: "mlx",
            modality: gen_core::Modality::Audio,
            capabilities: gen_core::Capabilities {
                supports_streaming: true,
                ..Default::default()
            },
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &[],
            control_kinds: None,
        }
    }

    /// A streaming [`Generator`] that emits `chunks` incremental [`AudioChunk`]s of `per_chunk`
    /// interleaved samples through `on_chunk` (with a small inter-chunk delay so the chunks arrive
    /// incrementally, not all at once), and returns the SAME aggregate track — proving the reassembly
    /// law (`concat(chunks) == generate()`). Its one-shot `generate` returns that same concatenation.
    struct StreamingStubGenerator {
        descriptor: gen_core::ModelDescriptor,
        chunks: usize,
        per_chunk: usize,
    }

    impl StreamingStubGenerator {
        fn chunk_samples(&self, index: usize) -> Vec<f32> {
            // Deterministic, non-silent, per-chunk-distinct PCM so a reassembly bug is observable.
            (0..self.per_chunk)
                .map(|s| {
                    let n = (index * self.per_chunk + s) as f32;
                    ((n * 0.25).sin() * 0.5).clamp(-1.0, 1.0)
                })
                .collect()
        }

        fn aggregate(&self) -> Vec<f32> {
            (0..self.chunks)
                .flat_map(|i| self.chunk_samples(i))
                .collect()
        }
    }

    impl gen_core::Generator for StreamingStubGenerator {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            Ok(GenerationOutput::Audio(gen_core::AudioTrack {
                samples: self.aggregate(),
                sample_rate: 24_000,
                channels: 1,
                stems: Vec::new(),
            }))
        }
        fn generate_streaming(
            &self,
            req: &GenerationRequest,
            on_chunk: &mut dyn FnMut(gen_core::AudioChunk),
            _on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            let mut all = Vec::new();
            for index in 0..self.chunks {
                if req.cancel.is_cancelled() {
                    return Err(gen_core::Error::Canceled);
                }
                let samples = self.chunk_samples(index);
                all.extend(samples.iter().copied());
                on_chunk(gen_core::AudioChunk {
                    samples,
                    sample_rate: 24_000,
                    channels: 1,
                    index,
                });
                std::thread::sleep(Duration::from_millis(8));
            }
            Ok(GenerationOutput::Audio(gen_core::AudioTrack {
                samples: all,
                sample_rate: 24_000,
                channels: 1,
                stems: Vec::new(),
            }))
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

    /// Neutralize the ambient HF cache env so a co-requisite snapshot resolves under the test's own
    /// `data_dir` rather than a developer's real `HF_HOME` (sc-13679 trap) — the audio twin of the
    /// model_jobs co_requisite seam tests' `isolate_hf_cache`.
    fn isolate_hf_cache() -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ])
    }

    /// Stage a co-requisite component file at `models--<repo>/snapshots/<revision>/<file>` under
    /// `data_dir`, mirroring the model_jobs seam tests' `stage_snapshot_file`.
    fn stage_snapshot_file(data_dir: &Path, repo: &str, revision: &str, file: &str) {
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, repo)
            .expect("repo cache path resolves")
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).expect("create snapshot dir");
        std::fs::write(snapshot.join(file), b"weights").expect("write staged component file");
    }

    /// The staged `codec` component for a MOSS TTS model: the isolated HF-cache env guard + tempdir
    /// (both must be kept alive for the whole test) and the `modelManifestEntry` — carrying the codec
    /// `coRequisite` download — the payload must advertise.
    struct StagedCodec {
        _env: crate::test_env::EnvVars,
        _data_dir: tempfile::TempDir,
        manifest_entry: Value,
    }

    /// Stage a MOSS model's `codec` component under an isolated HF cache, point `settings.data_dir` at
    /// it, and return the matching `modelManifestEntry`. After the inference codec-component pin bump
    /// (sc-13681) the real MOSS descriptors advertise `required_components: ["codec"]`, so
    /// `run_audio_synthesis_using` calls `resolve_co_requisites`, which resolves the codec from its
    /// cached pinned-SHA snapshot; without a staged snapshot the JOB fails with `InvalidPayload`
    /// BEFORE the behavior under test runs. This mirrors the model_jobs co_requisite seam tests'
    /// `isolate_hf_cache` / `stage_snapshot_file` pattern — codec resolution actually SUCCEEDS here
    /// (the seam is exercised, not stubbed), and the loaded generator is still the test's own stub, so
    /// the staged bytes are never read. The returned guard + tempdir must be kept alive for the whole
    /// test.
    fn stage_moss_codec(
        settings: &mut Settings,
        model_id: &str,
        codec_repo: &str,
        codec_revision: &str,
        codec_files: &[&str],
    ) -> StagedCodec {
        let env = isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        for file in codec_files {
            stage_snapshot_file(data_dir.path(), codec_repo, codec_revision, file);
        }
        settings.data_dir = data_dir.path().to_path_buf();
        let manifest_entry = json!({
            "id": model_id,
            "type": "audio",
            "downloads": [{
                "provider": "huggingface",
                "repo": codec_repo,
                "revision": codec_revision,
                "coRequisite": true,
                "componentId": "codec",
                "files": codec_files,
            }],
        });
        StagedCodec {
            _env: env,
            _data_dir: data_dir,
            manifest_entry,
        }
    }

    /// Stage `chatterbox_tts`'s two component coRequisites (`voice_embedding` + `perth`) under an
    /// isolated HF cache, point `settings.data_dir` at it, and return the matching `modelManifestEntry`.
    /// After the inference component pin (sc-13680) the native clone path resolves these via
    /// `resolve_co_requisites` and stages them in `LoadSpec::components` BEFORE the load; without a
    /// staged snapshot the JOB fails with `InvalidPayload` before the behavior under test runs. Mirrors
    /// [`stage_moss_codec`] — the pinned SHAs are chatterbox's live `ve.safetensors` @ 5bb1f6ee and
    /// `perth_implicit.safetensors` @ 80b60f9c. The returned guard + tempdir must be kept alive for the
    /// whole test.
    struct StagedChatterboxComponents {
        _env: crate::test_env::EnvVars,
        _data_dir: tempfile::TempDir,
        manifest_entry: Value,
    }

    fn stage_chatterbox_components(settings: &mut Settings) -> StagedChatterboxComponents {
        let env = isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        stage_snapshot_file(
            data_dir.path(),
            "ResembleAI/chatterbox",
            "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
            "ve.safetensors",
        );
        stage_snapshot_file(
            data_dir.path(),
            "SceneWorks/perth-implicit",
            "80b60f9caead09b8d3b512bda0b24038f28c08ec",
            "perth_implicit.safetensors",
        );
        settings.data_dir = data_dir.path().to_path_buf();
        let manifest_entry = json!({
            "id": "chatterbox_tts",
            "type": "audio",
            "downloads": [
                { "provider": "huggingface", "repo": "ResembleAI/chatterbox",
                  "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", "coRequisite": true,
                  "componentId": "voice_embedding", "files": ["ve.safetensors"] },
                { "provider": "huggingface", "repo": "SceneWorks/perth-implicit",
                  "revision": "80b60f9caead09b8d3b512bda0b24038f28c08ec", "coRequisite": true,
                  "componentId": "perth", "files": ["perth_implicit.safetensors"] }
            ],
        });
        StagedChatterboxComponents {
            _env: env,
            _data_dir: data_dir,
            manifest_entry,
        }
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

    // acestep's Cover snapshot pin (sc-13821), matching the `sft_cover` soft coRequisite in
    // config/manifests/builtin.models.jsonc and candle-audio-acestep's SFT_HUB_REVISION.
    const SFT_COVER_REPO: &str = "ACE-Step/acestep-v15-xl-sft-diffusers";
    const SFT_COVER_REVISION: &str = "4bf7b60a63b27144f539f980927eeb89f5f912b0";

    /// Stage acestep's `sft_cover` Cover snapshot (sc-13821) under an isolated HF cache with the three
    /// component subdirs the provider joins (`transformer/`, `audio_tokenizer/`,
    /// `audio_token_detokenizer/`), point `settings.data_dir` at it, and return the acestep
    /// `modelManifestEntry` carrying the soft `sft_cover` coRequisite. Mirrors
    /// [`stage_chatterbox_components`]; the staged bytes are never read (the loaded generator is a stub)
    /// — the test asserts the RESOLVED PATH reaches the LoadSpec. Guard + tempdir must outlive the test.
    struct StagedSftCover {
        _env: crate::test_env::EnvVars,
        _data_dir: tempfile::TempDir,
        manifest_entry: Value,
    }

    fn stage_sft_cover(settings: &mut Settings) -> StagedSftCover {
        let env = isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let snapshot =
            sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir.path(), SFT_COVER_REPO)
                .expect("repo cache path resolves")
                .join("snapshots")
                .join(SFT_COVER_REVISION);
        for (subdir, file) in [
            (
                "transformer",
                "diffusion_pytorch_model.safetensors.index.json",
            ),
            ("audio_tokenizer", "config.json"),
            ("audio_token_detokenizer", "config.json"),
        ] {
            let dir = snapshot.join(subdir);
            std::fs::create_dir_all(&dir).expect("create component subdir");
            std::fs::write(dir.join(file), b"weights").expect("write staged component file");
        }
        settings.data_dir = data_dir.path().to_path_buf();
        let manifest_entry = json!({
            "id": "acestep_v15_turbo",
            "type": "audio",
            "downloads": [
                { "provider": "huggingface", "repo": "ACE-Step/acestep-v15-xl-turbo-diffusers",
                  "revision": "200ba991ae448051e14b0183157e35c2d27c9fb0" },
                { "provider": "huggingface", "repo": SFT_COVER_REPO, "revision": SFT_COVER_REVISION,
                  "coRequisite": true, "required": "soft", "componentId": "sft_cover",
                  "files": ["transformer/*", "audio_tokenizer/*", "audio_token_detokenizer/*"] }
            ],
        });
        StagedSftCover {
            _env: env,
            _data_dir: data_dir,
            manifest_entry,
        }
    }

    /// A loader that records the `LoadSpec::components` it receives before returning a completing stub —
    /// the audio twin of [`stub_load`], used to assert which components a request stages on the spec the
    /// engine load actually sees (the seam the multi-entrypoint bug hid, sc-13686).
    fn spec_capture_load(
        captured: Arc<Mutex<Option<BTreeMap<String, gen_core::WeightsSource>>>>,
    ) -> impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>> + Send + 'static {
        move |_id: &str, spec: &LoadSpec| {
            *captured.lock().expect("spec capture lock") = Some(spec.components.clone());
            Ok(Box::new(StubGenerator {
                descriptor: stub_descriptor(),
                behavior: StubBehavior::CompleteOk,
                observed_cancel: Arc::new(AtomicBool::new(false)),
            }) as Box<dyn Generator>)
        }
    }

    /// sc-13821: a Cover request stages the pinned `sft_cover` snapshot into the LoadSpec the loader
    /// receives — the OPTIONAL, on-demand component the generic `resolve_co_requisites` seam does NOT
    /// stage (sft_cover is deliberately not a `required_components` id). This drives the REAL
    /// `run_audio_synthesis_using` entrypoint (not the resolver in isolation), matching
    /// [[coreq_seam_has_multiple_loadspec_entrypoints]].
    #[tokio::test]
    async fn cover_request_stages_the_sft_cover_component_on_the_loadspec() {
        let (base_url, _progress) = spawn_audio_cancel_stub(false).await;
        let mut settings = cancel_test_settings(base_url);
        let staged = stage_sft_cover(&mut settings);
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-sft-cover");
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "acestep_v15_turbo",
            "editMode": "cover",
            "modelManifestEntry": staged.manifest_entry.clone(),
        })));

        let captured = Arc::new(Mutex::new(None));
        run_audio_synthesis_using(
            &api,
            &settings,
            &job,
            &request,
            PathBuf::from("unused"),
            None,
            spec_capture_load(captured.clone()),
        )
        .await
        .expect("the stub Cover synthesis completes");

        let comps = captured
            .lock()
            .expect("capture lock")
            .take()
            .expect("the loader captured a LoadSpec");
        match comps.get("sft_cover") {
            Some(gen_core::WeightsSource::Dir(dir)) => assert!(
                dir.ends_with(SFT_COVER_REVISION),
                "sft_cover must resolve to the PINNED snapshot dir (snapshots/<sha>/), got {dir:?}"
            ),
            other => panic!(
                "a Cover request must stage sft_cover as a Dir on the LoadSpec, got {other:?}"
            ),
        }
    }

    /// Mutation guard for [`cover_request_stages_the_sft_cover_component_on_the_loadspec`]: with the SAME
    /// model + manifest and the snapshot present, a NON-Cover (text-to-music) request must NOT stage
    /// `sft_cover` — an unconditional attach (or one via the always-run required-components seam) would
    /// false-green the positive test.
    #[tokio::test]
    async fn non_cover_request_does_not_stage_the_sft_cover_component() {
        let (base_url, _progress) = spawn_audio_cancel_stub(false).await;
        let mut settings = cancel_test_settings(base_url);
        let staged = stage_sft_cover(&mut settings);
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-sft-no-cover");
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "acestep_v15_turbo",
            "modelManifestEntry": staged.manifest_entry.clone(),
        })));

        let captured = Arc::new(Mutex::new(None));
        run_audio_synthesis_using(
            &api,
            &settings,
            &job,
            &request,
            PathBuf::from("unused"),
            None,
            spec_capture_load(captured.clone()),
        )
        .await
        .expect("the stub text-to-music synthesis completes");

        let comps = captured
            .lock()
            .expect("capture lock")
            .take()
            .expect("the loader captured a LoadSpec");
        assert!(
            !comps.contains_key("sft_cover"),
            "a non-Cover request must NOT stage sft_cover, got {:?}",
            comps.get("sft_cover")
        );
    }

    /// A stub [`Generator`] that records the [`AudioParams::script`] it receives (sc-13676) — so a
    /// test can prove the parsed multi-speaker script actually reaches `GenerationRequest.audio.script`,
    /// and that a single-voice request carries `None` there (byte-for-byte unaffected). The outer
    /// `Option` records whether `generate` ran at all; the inner is the observed script value.
    struct ScriptCaptureGenerator {
        descriptor: gen_core::ModelDescriptor,
        captured: Arc<Mutex<Option<Option<Vec<SpeechSegment>>>>>,
    }

    impl gen_core::Generator for ScriptCaptureGenerator {
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
            *self.captured.lock().expect("capture lock") =
                Some(req.audio.as_ref().and_then(|audio| audio.script.clone()));
            Ok(GenerationOutput::Audio(gen_core::AudioTrack {
                samples: vec![0.1, -0.1, 0.1, -0.1],
                sample_rate: 24_000,
                channels: 1,
                stems: Vec::new(),
            }))
        }
    }

    /// A multi-speaker descriptor (supports_multi_speaker + max_speakers = 2), the audio twin of the
    /// real moss_ttsd_v05 descriptor's dialogue flags.
    fn multi_speaker_stub_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "stub_multi_speaker_audio",
            family: "stub",
            backend: "mlx",
            modality: gen_core::Modality::Audio,
            capabilities: gen_core::Capabilities {
                supports_multi_speaker: true,
                max_speakers: Some(2),
                ..Default::default()
            },
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &[],
            control_kinds: None,
        }
    }

    fn script_capture_load(
        captured: Arc<Mutex<Option<Option<Vec<SpeechSegment>>>>>,
    ) -> impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>> + Send + 'static {
        move |_id: &str, _spec: &LoadSpec| {
            Ok(Box::new(ScriptCaptureGenerator {
                descriptor: multi_speaker_stub_descriptor(),
                captured,
            }) as Box<dyn Generator>)
        }
    }

    /// Multi-speaker path (sc-13676): a parsed `script` must ride `GenerationRequest.audio.script`
    /// through the SAME one-shot synthesis seam every Speech job uses — proving the segmented dialogue
    /// reaches the generator (the model's own `validate` is the capability gate at the gen-core floor).
    #[tokio::test]
    async fn single_synthesis_forwards_the_multi_speaker_script_to_audio_params() {
        let (base_url, _progress) = spawn_audio_cancel_stub(false).await;
        let mut settings = cancel_test_settings(base_url);
        // moss_ttsd_v05 now advertises `required_components: ["codec"]` (sc-13681), so the synthesis
        // seam resolves the XY_Tokenizer codec co-requisite before the stub load — stage it (and
        // advertise it on the payload) so resolution SUCCEEDS and the script-forwarding behavior runs.
        let staged = stage_moss_codec(
            &mut settings,
            "moss_ttsd_v05",
            "OpenMOSS-Team/XY_Tokenizer_TTSD_V0",
            "c83433728e698ed0698e88cb5096bc221fb8f8c5",
            &["xy_tokenizer.ckpt"],
        );
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-script");
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "moss_ttsd_v05",
            "prompt": "",
            "script": [
                { "text": "Hello, how are you today?", "speaker": "S1" },
                { "text": "I'm doing great, thanks for asking!", "speaker": "S2" },
            ],
            "modelManifestEntry": staged.manifest_entry.clone(),
        })));

        let captured = Arc::new(Mutex::new(None));
        let load = script_capture_load(captured.clone());

        run_audio_synthesis_using(
            &api,
            &settings,
            &job,
            &request,
            PathBuf::from("unused"),
            None,
            load,
        )
        .await
        .expect("a clean multi-speaker synthesis returns the produced track");

        let observed = captured.lock().expect("capture lock").clone();
        let script = observed
            .expect("generate() must have run")
            .expect("the multi-speaker script must reach AudioParams.script");
        assert_eq!(
            script.len(),
            2,
            "both dialogue segments reach the generator"
        );
        assert_eq!(script[0].text, "Hello, how are you today?");
        assert_eq!(script[0].speaker.as_deref(), Some("S1"));
        assert_eq!(script[1].speaker.as_deref(), Some("S2"));
    }

    /// Single-voice control (sc-13676): a request with no `script` must build the identical
    /// `AudioParams { script: None, .. }` — the byte-for-byte-unaffected guarantee for every existing
    /// Speech / SFX / Music mode.
    #[tokio::test]
    async fn single_voice_request_carries_no_script() {
        let (base_url, _progress) = spawn_audio_cancel_stub(false).await;
        let settings = cancel_test_settings(base_url);
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-no-script");
        let request = AudioRequest::from_payload(&payload(json!({ "voice": "af_heart" })));
        assert!(
            request.script.is_none(),
            "a request with no script parses to script: None"
        );

        let captured = Arc::new(Mutex::new(None));
        let load = script_capture_load(captured.clone());

        run_audio_synthesis_using(
            &api,
            &settings,
            &job,
            &request,
            PathBuf::from("unused"),
            None,
            load,
        )
        .await
        .expect("a clean single-voice synthesis returns the produced track");

        let observed = captured.lock().expect("capture lock").clone();
        assert_eq!(
            observed.expect("generate() must have run"),
            None,
            "a single-voice request must reach the generator with AudioParams.script == None"
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
        let mut settings = cancel_test_settings(base_url);
        // chatterbox_tts advertises `required_components: [perth, voice_embedding]`, so the native clone
        // path now resolves them before the (stub) load (sc-13686); stage them + advertise them on the
        // payload so resolution SUCCEEDS and the cancel behavior under test runs.
        let staged = stage_chatterbox_components(&mut settings);
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-cancel-native");
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "chatterbox_tts",
            "referenceAudioAssetId": "ref-1",
            "modelManifestEntry": staged.manifest_entry.clone(),
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

    /// sc-13686 walking-skeleton (chatterbox voice clone, the epic's original PR-note case): the native
    /// clone JOB must resolve the generator's `required_components` and STAGE them in
    /// `LoadSpec::components` BEFORE the load. This is the gap this verification story found — the native
    /// path built a component-less `LoadSpec`, so at the sc-13680 pin (the generator no longer self-
    /// fetches ve/perth) `chatterbox_tts` could not load even fully installed, failing at the engine's
    /// `require_component` gate. Drives the REAL entrypoint (`run_native_voice_clone_synthesis_using`)
    /// with both components staged and captures the spec the loader receives: BOTH `perth` and
    /// `voice_embedding` must be present. FAILS on the pre-fix component-less spec (keys empty).
    ///
    /// Gated to the audio lane's own cfg (`inference_runtime::audio()`): the descriptor these tests
    /// resolve exists only on macOS or a `backend-candle` build, so this runs on the macOS + candle test
    /// lanes (windows-candle's `cargo test --features backend-candle`) and is compiled out on the
    /// no-candle Linux `parity` lane, where `audio_descriptor` returns None and there is nothing to stage.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[tokio::test]
    async fn native_voice_clone_stages_the_required_components_into_the_load_spec() {
        let (base_url, _progress) = spawn_audio_cancel_stub(false).await;
        let mut settings = cancel_test_settings(base_url);
        let staged = stage_chatterbox_components(&mut settings);
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-native-components");
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "chatterbox_tts",
            "referenceAudioAssetId": "ref-1",
            "modelManifestEntry": staged.manifest_entry.clone(),
        })));

        // Capture the component keys the loader sees in the LoadSpec — proof the worker staged them.
        let observed_components = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = observed_components.clone();
        let load = move |_id: &str, spec: &LoadSpec| {
            *sink.lock().expect("component lock") = spec.components.keys().cloned().collect();
            Ok(Box::new(StubGenerator {
                descriptor: stub_descriptor(),
                behavior: StubBehavior::CompleteOk,
                observed_cancel: Arc::new(AtomicBool::new(false)),
            }) as Box<dyn Generator>)
        };
        let plan = NativeVoiceClonePlan {
            model_dir: PathBuf::from("unused"),
            reference: gen_core::AudioTrack {
                samples: vec![0.05, -0.05],
                sample_rate: 24_000,
                channels: 1,
                stems: Vec::new(),
            },
        };

        run_native_voice_clone_synthesis_using(&api, &settings, &job, &request, plan, load)
            .await
            .expect("with both components staged the native clone render must succeed");

        let mut keys = observed_components.lock().expect("component lock").clone();
        keys.sort();
        assert_eq!(
            keys,
            vec!["perth".to_owned(), "voice_embedding".to_owned()],
            "the worker must stage BOTH required components in the LoadSpec the generator loads with"
        );
    }

    /// sc-13794 (epic 13678) VERIFICATION harness — the POSITIVE full offline render through the FIXED
    /// production entrypoint `run_native_voice_clone_synthesis_using`, with REAL Chatterbox weights, from
    /// a FRESH custom HF cache where `ResembleAI/chatterbox` + `SceneWorks/perth-implicit` start ABSENT,
    /// and with NETWORK DISABLED for the render. Complements the (non-ignored) staging regression above
    /// (which stubs the loader) by driving the SAME entrypoint with the REAL `inference_runtime::load_audio`
    /// loader end to end, proving: (a) `resolve_co_requisites` stages BOTH `perth` + `voice_embedding` into
    /// `LoadSpec::components`, and (b) chatterbox_tts loads from the local primary dir + those staged files
    /// and renders a valid cloned WAV with ZERO network. `#[ignore]`d, macOS real-weight (downloads
    /// ~3.2 GB into an ISOLATED temp cache):
    /// ```text
    /// cargo test -p sceneworks-worker --release native_voice_clone_offline_render_through_entrypoint -- --ignored --nocapture
    /// ```
    #[cfg(target_os = "macos")]
    #[ignore = "sc-13794 real-weight offline render through the fixed entrypoint; run by hand on Apple Silicon"]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_voice_clone_offline_render_through_entrypoint() {
        use sha2::{Digest, Sha256};

        const VE_REVISION: &str = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18";
        const PERTH_REVISION: &str = "80b60f9caead09b8d3b512bda0b24038f28c08ec";

        // The REAL model-download executor (the exact seam a Models-screen install runs), into the
        // isolated hub. Returns the resolved commit SHA.
        async fn install_snapshot(
            settings: &Settings,
            repo: &str,
            revision: &str,
            files: &[&str],
        ) -> String {
            use crate::downloads::{
                download_snapshot_into_cache, DownloadContext, DownloadProgress,
                HuggingFaceSnapshot,
            };
            let client = crate::downloads::streaming_download_client();
            let api = ApiClient::new(settings);
            let repo_dir =
                sceneworks_core::hf_home::huggingface_repo_cache_path(&settings.data_dir, repo)
                    .expect("resolve hub cache path");
            let file_patterns: Vec<String> = files.iter().map(|f| (*f).to_owned()).collect();
            let snapshot =
                HuggingFaceSnapshot::resolve(&client, settings, repo, revision, &file_patterns)
                    .await
                    .expect("resolve HF snapshot listing");
            assert!(
                !snapshot.files.is_empty(),
                "{repo}@{revision} resolved zero files for {file_patterns:?}"
            );
            let context = DownloadContext {
                api: &api,
                client: &client,
                settings,
                job_id: "sc-13794-offline-entrypoint",
                cancel_message: "canceled",
                fresh_download: false,
            };
            let mut progress =
                DownloadProgress::new(repo, 0, snapshot.total_bytes(), Duration::from_secs(86_400));
            download_snapshot_into_cache(&context, &repo_dir, revision, &snapshot, &mut progress)
                .await
                .unwrap_or_else(|e| panic!("materialize {repo}@{revision}: {e}"))
        }

        // A short synthetic speech-like reference clip (4 s @ 24 kHz mono) — enough for the provider to
        // derive a speaker embedding + prompt tokens without pulling in a second TTS model.
        fn synthetic_reference() -> gen_core::AudioTrack {
            let sample_rate = 24_000u32;
            let n = (sample_rate as f32 * 4.0) as usize;
            let mut samples = Vec::with_capacity(n);
            let mut seed = 0x2b41_53c7u32;
            for i in 0..n {
                let t = i as f32 / sample_rate as f32;
                let vibrato = 1.0 + 0.02 * (2.0 * std::f32::consts::PI * 5.0 * t).sin();
                let f0 = 165.0 * vibrato;
                let mut s = 0.6 * (2.0 * std::f32::consts::PI * f0 * t).sin();
                s += 0.25 * (2.0 * std::f32::consts::PI * 2.0 * f0 * t).sin();
                s += 0.15 * (2.0 * std::f32::consts::PI * 3.0 * f0 * t).sin();
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
                s += 0.05 * noise;
                let env = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * 0.5 * t).cos();
                samples.push(0.8 * env * s);
            }
            gen_core::AudioTrack {
                samples,
                sample_rate,
                channels: 1,
                stems: Vec::new(),
            }
        }

        // ── Step 1: isolate the cache the RUNTIME RESOLVER reads (HOME + HF_HOME at a fresh temp root),
        //    clearing every other HF cache + proxy var so neither install nor resolution can diverge. ──
        let home_root = tempfile::tempdir().expect("temp HOME root");
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let hf_home = home_root.path().join(".cache").join("huggingface");
        let hub = hf_home.join("hub");
        let _env = crate::test_env::EnvVars::set(&[
            ("HOME", home_root.path().to_str().expect("utf-8 HOME")),
            ("HF_HOME", hf_home.to_str().expect("utf-8 HF_HOME")),
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HUB_OFFLINE", ""),
            ("TRANSFORMERS_OFFLINE", ""),
            ("HF_ENDPOINT", ""),
            ("PERTH_SNAPSHOT", ""),
            ("ALL_PROXY", ""),
            ("all_proxy", ""),
            ("HTTPS_PROXY", ""),
            ("https_proxy", ""),
            ("HTTP_PROXY", ""),
            ("http_proxy", ""),
            ("NO_PROXY", ""),
            ("no_proxy", ""),
        ]);
        assert!(
            !hub.join("models--ResembleAI--chatterbox").exists()
                && !hub.join("models--SceneWorks--perth-implicit").exists(),
            "the custom HF cache must start with BOTH repos ABSENT: {}",
            hub.display()
        );

        // ── Step 2: heartbeat stub API + settings pointing the resolver at the isolated hub ──────────
        let (base_url, _progress) = spawn_audio_cancel_stub(false).await;
        let mut settings = cancel_test_settings(base_url);
        settings.data_dir = data_dir.path().to_path_buf();

        // ── Step 3: install (online) the primary + the two pinned companion co-requisites ───────────
        let main_commit = install_snapshot(
            &settings,
            "ResembleAI/chatterbox",
            "main",
            &["t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"],
        )
        .await;
        install_snapshot(
            &settings,
            "ResembleAI/chatterbox",
            VE_REVISION,
            &["ve.safetensors"],
        )
        .await;
        install_snapshot(
            &settings,
            "SceneWorks/perth-implicit",
            PERTH_REVISION,
            &["perth_implicit.safetensors"],
        )
        .await;

        let chatterbox_dir = hub.join("models--ResembleAI--chatterbox");
        let main_snapshot_dir = chatterbox_dir.join("snapshots").join(&main_commit);
        assert!(
            main_snapshot_dir.join("s3gen.safetensors").exists(),
            "primary chatterbox snapshot must hold the generator weights"
        );
        assert!(
            chatterbox_dir
                .join("snapshots")
                .join(VE_REVISION)
                .join("ve.safetensors")
                .exists(),
            "ve co-requisite must materialize at its pinned snapshot"
        );
        assert!(
            hub.join("models--SceneWorks--perth-implicit")
                .join("snapshots")
                .join(PERTH_REVISION)
                .join("perth_implicit.safetensors")
                .exists(),
            "perth co-requisite must materialize at its pinned snapshot"
        );

        // ── Step 4: DISABLE NETWORK for the render — offline flags PLUS a black-hole proxy that fails
        //    any residual hub agent (belt and suspenders; at this pin a cache HIT never builds one).
        //    All five keys are pre-registered by the `_env` guard above, so they restore on drop —
        //    no leak (sc-13909). ──
        std::env::set_var("HF_HUB_OFFLINE", "1");
        std::env::set_var("TRANSFORMERS_OFFLINE", "1");
        std::env::set_var("ALL_PROXY", "http://127.0.0.1:1");
        std::env::set_var("HTTPS_PROXY", "http://127.0.0.1:1");
        std::env::set_var("HTTP_PROXY", "http://127.0.0.1:1");

        // ── Step 5: drive the FIXED production entrypoint with the REAL audio loader ─────────────────
        let manifest_entry = json!({
            "id": "chatterbox_tts",
            "type": "audio",
            "downloads": [
                { "provider": "huggingface", "repo": "ResembleAI/chatterbox", "revision": VE_REVISION,
                  "coRequisite": true, "componentId": "voice_embedding", "files": ["ve.safetensors"] },
                { "provider": "huggingface", "repo": "SceneWorks/perth-implicit", "revision": PERTH_REVISION,
                  "coRequisite": true, "componentId": "perth", "files": ["perth_implicit.safetensors"] }
            ],
        });
        let api = ApiClient::new(&settings);
        let job = audio_test_job("sc-13794-native-clone-offline");
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "chatterbox_tts",
            "prompt": "This cloned voice was rendered fully offline through the fixed entrypoint.",
            "seed": 13794,
            "referenceAudioAssetId": "ref-1",
            "modelManifestEntry": manifest_entry,
        })));

        // Wrap the REAL loader so we can capture the component keys the LoadSpec carries — the proof the
        // entrypoint's `resolve_co_requisites` staged both components before the real load.
        let observed_components = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = observed_components.clone();
        let load = move |id: &str, spec: &LoadSpec| {
            let mut keys: Vec<String> = spec.components.keys().cloned().collect();
            keys.sort();
            *sink.lock().expect("component lock") = keys;
            crate::inference_runtime::load_audio(id, spec)
        };

        let plan = NativeVoiceClonePlan {
            model_dir: main_snapshot_dir.clone(),
            reference: synthetic_reference(),
        };

        let clone =
            run_native_voice_clone_synthesis_using(&api, &settings, &job, &request, plan, load)
                .await
                .expect(
                    "offline native clone render MUST succeed through the fixed entrypoint from the \
                     installed isolated cache (resolve_co_requisites stages ve + perth cache-first; the \
                     dead proxy proves no fetch)",
                );

        // ── Assert: BOTH components staged + a valid clone clip ──────────────────────────────────────
        let keys = observed_components.lock().expect("component lock").clone();
        assert_eq!(
            keys,
            vec!["perth".to_owned(), "voice_embedding".to_owned()],
            "the entrypoint must stage BOTH required components into the LoadSpec the generator loads with"
        );
        assert!(!clone.samples.is_empty(), "clone is empty");
        let peak = clone.samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(peak > 1e-2, "clone is (near-)silent: peak {peak:.6}");
        assert_eq!(clone.sample_rate, 24_000, "clone must be 24 kHz");
        assert_eq!(clone.channels, 1, "clone must be mono");

        let out_dir = PathBuf::from(
            std::env::var("VOICECLONE_OUT_DIR")
                .unwrap_or_else(|_| "/tmp/voiceclone_smoke".to_owned()),
        );
        std::fs::create_dir_all(&out_dir).expect("create out dir");
        let wav_path = out_dir.join("entrypoint_offline_clone.wav");
        let wav = AudioTrack {
            samples: clone.samples.clone(),
            sample_rate: clone.sample_rate,
            channels: clone.channels,
        };
        write_wav_pcm16(&wav, &wav_path).expect("write clone wav");
        let bytes = std::fs::read(&wav_path).expect("read clone wav");
        let sha = Sha256::digest(&bytes);
        let duration =
            clone.samples.len() as f32 / (clone.sample_rate as f32 * clone.channels.max(1) as f32);
        eprintln!(
            "[sc-13794] PASS — entrypoint offline render: {} samples @ {} Hz, {:.2}s, peak {:.4}",
            clone.samples.len(),
            clone.sample_rate,
            duration,
            peak
        );
        eprintln!("[sc-13794] staged components: {keys:?}");
        eprintln!(
            "[sc-13794] custom HF cache (repos started ABSENT): {}",
            hub.display()
        );
        eprintln!(
            "[sc-13794] wav: {} sha256={:x} bytes={}",
            wav_path.display(),
            sha,
            bytes.len()
        );
    }

    /// sc-13686 end-to-end negative path (chatterbox voice clone): a native clone JOB whose `perth`
    /// component is NOT installed must fail at `resolve_co_requisites` — BEFORE the generator load —
    /// with an actionable `InvalidPayload` naming the missing component id + repo, never at the engine's
    /// `require_component` gate or a mid-render fetch. Drives the REAL entrypoint; the load closure flips
    /// a flag it must NEVER set, proving the job fails at resolution, not inside the engine. (perth is
    /// first in `required_components`, so with only ve staged the surfaced miss is perth.) Gated to the
    /// audio lane's cfg — compiled out on the no-candle Linux `parity` lane, run on macOS + candle.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[tokio::test]
    async fn native_voice_clone_job_fails_before_load_when_a_component_is_absent() {
        let (base_url, _progress) = spawn_audio_cancel_stub(false).await;
        let mut settings = cancel_test_settings(base_url);
        // Stage ONLY voice_embedding; perth's snapshot stays absent.
        let _env = isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        stage_snapshot_file(
            data_dir.path(),
            "ResembleAI/chatterbox",
            "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
            "ve.safetensors",
        );
        settings.data_dir = data_dir.path().to_path_buf();
        let manifest_entry = json!({
            "id": "chatterbox_tts",
            "type": "audio",
            "downloads": [
                { "provider": "huggingface", "repo": "ResembleAI/chatterbox",
                  "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", "coRequisite": true,
                  "componentId": "voice_embedding", "files": ["ve.safetensors"] },
                { "provider": "huggingface", "repo": "SceneWorks/perth-implicit",
                  "revision": "80b60f9caead09b8d3b512bda0b24038f28c08ec", "coRequisite": true,
                  "componentId": "perth", "files": ["perth_implicit.safetensors"] }
            ],
        });
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-native-missing-perth");
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "chatterbox_tts",
            "referenceAudioAssetId": "ref-1",
            "modelManifestEntry": manifest_entry,
        })));

        let load_ran = Arc::new(AtomicBool::new(false));
        let sink = load_ran.clone();
        let load = move |_id: &str, _spec: &LoadSpec| {
            sink.store(true, Ordering::SeqCst);
            Ok(Box::new(StubGenerator {
                descriptor: stub_descriptor(),
                behavior: StubBehavior::CompleteOk,
                observed_cancel: Arc::new(AtomicBool::new(false)),
            }) as Box<dyn Generator>)
        };
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

        let Err(WorkerError::InvalidPayload(message)) = result else {
            panic!(
                "a missing component must fail the JOB with InvalidPayload before the load, got {result:?}"
            );
        };
        assert!(
            message.contains("perth") && message.contains("SceneWorks/perth-implicit"),
            "the error must name the missing component id + repo, got: {message}"
        );
        assert!(
            !load_ran.load(Ordering::SeqCst),
            "the generator load must NEVER run when a required component is absent — the job must fail \
             at resolution, not inside the engine"
        );
    }

    /// sc-13686 end-to-end negative path (MOSS TTS): a MOSS synthesis JOB whose `codec` component is NOT
    /// installed must fail at the worker's `resolve_co_requisites` seam — BEFORE the generator load —
    /// with the actionable `InvalidPayload` naming the `codec` component and its repo, never a mid-render
    /// hub fetch. Drives the REAL job entrypoint (`run_audio_synthesis_using`, which resolves the
    /// descriptor's `required_components` before the blocking load). The load closure records whether it
    /// ran: the discrimination is that it must NOT. Complements the resolver-seam unit test
    /// (model_jobs::…a_missing_moss_codec_fails…) and the catalog install-state gate
    /// (models::moss_tts_install_state_gates…) with the integration link. Gated to the audio lane's cfg —
    /// compiled out on the no-candle Linux `parity` lane, run on macOS + candle.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[tokio::test]
    async fn moss_synthesis_job_fails_before_load_when_the_codec_is_absent() {
        let (base_url, _progress) = spawn_audio_cancel_stub(false).await;
        let mut settings = cancel_test_settings(base_url);
        // Isolate the HF cache at a FRESH empty data_dir — the XY_Tokenizer codec is NOT staged, so the
        // descriptor's `codec` required_component cannot resolve.
        let _env = isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        settings.data_dir = data_dir.path().to_path_buf();
        // A moss_ttsd_v05 request whose manifest entry advertises the (absent) codec coRequisite — the
        // exact pinned XY_Tokenizer snapshot the live catalog declares (sc-13681).
        let manifest_entry = json!({
            "id": "moss_ttsd_v05",
            "type": "audio",
            "downloads": [{
                "provider": "huggingface",
                "repo": "OpenMOSS-Team/XY_Tokenizer_TTSD_V0",
                "revision": "c83433728e698ed0698e88cb5096bc221fb8f8c5",
                "coRequisite": true,
                "componentId": "codec",
                "files": ["xy_tokenizer.ckpt"],
            }],
        });
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-missing-codec");
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "moss_ttsd_v05",
            "prompt": "This dialogue can never render without its codec.",
            "modelManifestEntry": manifest_entry,
        })));

        let load_ran = Arc::new(AtomicBool::new(false));
        let sink = load_ran.clone();
        let load = move |_id: &str, _spec: &LoadSpec| {
            sink.store(true, Ordering::SeqCst);
            Ok(Box::new(StubGenerator {
                descriptor: stub_descriptor(),
                behavior: StubBehavior::CompleteOk,
                observed_cancel: Arc::new(AtomicBool::new(false)),
            }) as Box<dyn Generator>)
        };

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

        let Err(WorkerError::InvalidPayload(message)) = result else {
            panic!(
                "a missing codec must fail the JOB with InvalidPayload before the engine load, got {result:?}"
            );
        };
        assert!(
            message.contains("codec") && message.contains("OpenMOSS-Team/XY_Tokenizer_TTSD_V0"),
            "the error must name the `codec` component + its repo, got: {message}"
        );
        assert!(
            !load_ran.load(Ordering::SeqCst),
            "the generator load must NEVER run when a hard component co-requisite is absent — the job \
             must fail at resolution, not inside the engine"
        );
    }

    /// Streaming path (sc-13675): a `supports_streaming` Generator must be driven through
    /// `generate_streaming`, the worker must forward an incremental job update per chunk (so the UI
    /// advances through the stream), and the returned track must equal the reassembly of every chunk
    /// (the gen-core reassembly law → the library asset is the full clip). Gated purely on the loaded
    /// generator's capability, so no model id is hardcoded.
    #[tokio::test]
    async fn streaming_synthesis_forwards_per_chunk_progress_and_returns_reassembled_track() {
        let (base_url, progress) = spawn_audio_cancel_stub(false).await;
        let mut settings = cancel_test_settings(base_url);
        // moss_tts_realtime now advertises `required_components: ["codec"]` (sc-13681), so the
        // synthesis seam resolves the MOSS-Audio-Tokenizer codec co-requisite before the stub load —
        // stage it (and advertise it on the payload) so resolution SUCCEEDS and the streaming behavior
        // runs.
        let staged = stage_moss_codec(
            &mut settings,
            "moss_tts_realtime",
            "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            "3cd226ba2947efa357ef453bcad111b6eafba782",
            &[
                "config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-stream");
        let request = AudioRequest::from_payload(&payload(json!({
            "model": "moss_tts_realtime",
            "modelManifestEntry": staged.manifest_entry.clone(),
        })));

        let expected = StreamingStubGenerator {
            descriptor: streaming_stub_descriptor(),
            chunks: 5,
            per_chunk: 4,
        }
        .aggregate();
        let load = move |_id: &str, _spec: &LoadSpec| {
            Ok(Box::new(StreamingStubGenerator {
                descriptor: streaming_stub_descriptor(),
                chunks: 5,
                per_chunk: 4,
            }) as Box<dyn Generator>)
        };

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
        .expect("a streaming synthesis returns the reassembled track");

        // The returned one-shot track == concat of the 5×4 streamed chunks (the reassembly law).
        assert_eq!(
            track.samples, expected,
            "the returned track must equal the reassembly of every streamed chunk"
        );
        assert_eq!(track.sample_rate, 24_000);
        assert_eq!(track.channels, 1);

        // The pump forwarded ≥2 incremental "Streaming audio…" job updates BEFORE the clip finished —
        // the observable proof the worker streams (a non-streaming model posts none; see below).
        let posts = progress.lock().expect("progress lock");
        let streaming_posts = posts
            .iter()
            .filter(|p| {
                p["message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with("Streaming audio"))
            })
            .count();
        assert!(
            streaming_posts >= 2,
            "expected ≥2 incremental streaming progress posts, got {streaming_posts}: {posts:?}"
        );
        // None of the incremental posts is terminal — the pump only ever emits Running/Generating.
        assert!(
            posts.iter().all(|p| p["status"] != "canceled"),
            "a clean stream posts no terminal Canceled, got {posts:?}"
        );
    }

    /// Control (sc-13675): a NON-streaming Generator (Capabilities default → `supports_streaming:
    /// false`) must keep the one-shot `generate` path and emit ZERO incremental "Streaming audio…"
    /// posts — proving the streaming wiring does not perturb Kokoro/MOSS-SFX/ACE-Step/clone modes.
    #[tokio::test]
    async fn non_streaming_synthesis_emits_no_incremental_streaming_posts() {
        let (base_url, progress) = spawn_audio_cancel_stub(false).await;
        let settings = cancel_test_settings(base_url);
        let api = ApiClient::new(&settings);
        let job = audio_test_job("audio-nonstream");
        let request = AudioRequest::from_payload(&payload(json!({})));

        let observed = Arc::new(AtomicBool::new(false));
        let load = stub_load(StubBehavior::CompleteOk, observed);
        run_audio_synthesis_using(
            &api,
            &settings,
            &job,
            &request,
            PathBuf::from("unused"),
            None,
            load,
        )
        .await
        .expect("a non-streaming synthesis completes");

        let posts = progress.lock().expect("progress lock");
        assert!(
            posts.iter().all(|p| {
                p["message"]
                    .as_str()
                    .is_none_or(|m| !m.starts_with("Streaming audio"))
            }),
            "a non-streaming model must emit no incremental streaming posts, got {posts:?}"
        );
    }

    /// DoD walking skeleton (sc-13675) — REAL MOSS-TTS-Realtime streaming through the generator seam.
    /// `#[ignore]` because it needs the ~4.66 GB AR checkpoint + ~7.1 GB MOSS-Audio-Tokenizer codec
    /// (not in CI) and is pathologically slow in a debug build. Run it in RELEASE:
    ///
    /// ```text
    /// SCENEWORKS_MOSS_TTS_AR_DIR=~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-TTS-Realtime/\
    ///   snapshots/6acbc7f161a0db71c291f2d0aaa9eee59334cab2 \
    /// SCENEWORKS_MOSS_TTS_CODEC_DIR=~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer/\
    ///   snapshots/3cd226ba2947efa357ef453bcad111b6eafba782 \
    /// HF_HUB_OFFLINE=1 \
    /// cargo test --release -p sceneworks-worker -- --ignored --nocapture \
    ///   moss_tts_realtime_streams_a_real_clip
    /// ```
    ///
    /// It loads the real generator through the worker's `load_audio` seam, drives `generate_streaming`
    /// capturing every chunk + timing, then asserts the DoD: ≥2 chunks, the first chunk arrives before
    /// the full clip finishes, `concat(chunks) == returned track` (reassembly law), 24 kHz, non-silent
    /// — and writes the assembled WAV + an evidence JSON to `SCENEWORKS_DOD_OUT` (or the temp dir).
    #[test]
    #[ignore = "requires the ~12GB MOSS-TTS-Realtime + MOSS-Audio-Tokenizer weights; DoD, run manually in release"]
    fn moss_tts_realtime_streams_a_real_clip_through_the_generator_seam() {
        let ar_dir = std::env::var("SCENEWORKS_MOSS_TTS_AR_DIR").expect(
            "set SCENEWORKS_MOSS_TTS_AR_DIR to the cached MOSS-TTS-Realtime snapshot directory",
        );
        // moss_tts_realtime advertises `required_components: ["codec"]` (sc-13681); on the self-fetch-free
        // inference pin (sc-13818) the loader HARD-FAILS unless the MOSS-Audio-Tokenizer codec is staged in
        // the LoadSpec — it no longer self-fetches it. The codec is a multi-file component, so it resolves
        // to `WeightsSource::Dir` (the whole snapshot dir), matching production `resolve_co_requisites`
        // (model_jobs.rs:1685). Silently broken by the pin bump because the smoke is `#[ignore]`d.
        let codec_dir = std::env::var("SCENEWORKS_MOSS_TTS_CODEC_DIR").expect(
            "set SCENEWORKS_MOSS_TTS_CODEC_DIR to the cached MOSS-Audio-Tokenizer snapshot directory \
             (moss_tts_realtime's required `codec` component)",
        );
        let generator = crate::inference_runtime::load_audio(
            "moss_tts_realtime",
            &LoadSpec::new(WeightsSource::Dir(PathBuf::from(&ar_dir)))
                .with_component("codec", WeightsSource::Dir(PathBuf::from(&codec_dir))),
        )
        .expect(
            "load the real moss_tts_realtime generator from the cached snapshot + staged codec",
        );
        assert!(
            generator.descriptor().capabilities.supports_streaming,
            "moss_tts_realtime must advertise supports_streaming"
        );

        let req = GenerationRequest {
            prompt:
                "SceneWorks streaming speech is now live. This clip was produced incrementally, \
                     one chunk at a time."
                    .to_owned(),
            audio: Some(AudioParams {
                language: Some("en".to_owned()),
                target_duration: Some(6.0),
                ..Default::default()
            }),
            ..Default::default()
        };

        let start = Instant::now();
        let mut chunk_lengths: Vec<usize> = Vec::new();
        let mut reassembled: Vec<f32> = Vec::new();
        let mut first_chunk_at: Option<Duration> = None;
        let mut sample_rate = 0u32;
        let mut channels = 0u16;
        {
            let mut on_chunk = |chunk: gen_core::AudioChunk| {
                if first_chunk_at.is_none() {
                    first_chunk_at = Some(start.elapsed());
                }
                assert_eq!(
                    chunk.index,
                    chunk_lengths.len(),
                    "chunk indices are gapless"
                );
                sample_rate = chunk.sample_rate;
                channels = chunk.channels;
                chunk_lengths.push(chunk.samples.len());
                reassembled.extend(chunk.samples.iter().copied());
            };
            let mut on_progress = |_p: Progress| {};
            let output = generator
                .generate_streaming(&req, &mut on_chunk, &mut on_progress)
                .expect("real streaming synthesis succeeds");
            let total = start.elapsed();
            let track = match output {
                GenerationOutput::Audio(track) => track,
                other => panic!("expected audio, got {other:?}"),
            };

            let first = first_chunk_at.expect("at least one chunk streamed");
            // (a) ≥2 chunks, first before the full clip finished.
            assert!(
                chunk_lengths.len() >= 2,
                "streaming must emit ≥2 chunks, got {}",
                chunk_lengths.len()
            );
            assert!(
                first < total,
                "the first chunk ({first:?}) must arrive before the full synthesis finishes ({total:?})"
            );
            // (b) reassembly law: concat(chunks) == the returned one-shot track.
            assert_eq!(
                reassembled, track.samples,
                "concatenating the streamed chunks must equal the returned track"
            );
            // (c) real speech at the expected rate, non-silent.
            assert_eq!(
                track.sample_rate, 24_000,
                "MOSS-TTS-Realtime renders 24 kHz"
            );
            assert_eq!(sample_rate, 24_000);
            assert!(channels >= 1);
            let peak = track.samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
            assert!(peak > 0.0, "the produced clip must be audible (non-silent)");

            // Save the artifact + evidence to SCENEWORKS_DOD_OUT (or the temp dir).
            let out_dir = std::env::var("SCENEWORKS_DOD_OUT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| std::env::temp_dir());
            std::fs::create_dir_all(&out_dir).ok();
            let wav_path = out_dir.join("moss_tts_realtime_stream_dod.wav");
            let wav = AudioTrack {
                samples: track.samples.clone(),
                sample_rate: track.sample_rate,
                channels: track.channels.max(1),
            };
            write_wav_pcm16(&wav, &wav_path).expect("write the assembled WAV");
            let evidence = json!({
                "model": "moss_tts_realtime",
                "chunkCount": chunk_lengths.len(),
                "chunkSampleLengths": chunk_lengths,
                "firstChunkMs": first.as_millis(),
                "fullSynthesisMs": total.as_millis(),
                "firstChunkBeforeFull": first < total,
                "sampleRate": track.sample_rate,
                "channels": track.channels,
                "totalSamples": track.samples.len(),
                "durationSecs": track.samples.len() as f64
                    / (track.sample_rate.max(1) as f64 * track.channels.max(1) as f64),
                "peakAmplitude": peak,
                "reassemblyEqualsOneShot": reassembled == track.samples,
                "wavPath": wav_path.display().to_string(),
            });
            let evidence_path = out_dir.join("moss_tts_realtime_stream_dod.json");
            std::fs::write(
                &evidence_path,
                serde_json::to_string_pretty(&evidence).unwrap(),
            )
            .expect("write the evidence JSON");
            eprintln!(
                "DoD evidence: {}\nWAV: {}",
                evidence_path.display(),
                wav_path.display()
            );
        }
    }

    /// DoD walking skeleton (sc-13676) — REAL MOSS-TTSD multi-speaker dialogue through the worker's
    /// generator seam (`load_audio`), with objective cross-speaker-vs-self voice-distinctness measured
    /// by the real `chatterbox_ve` speaker embedder. `#[ignore]` because it needs the ~4.1 GB AR
    /// checkpoint + ~2.1 GB XY_Tokenizer codec + the 5.7 MB chatterbox_ve weights (not in CI) and is
    /// pathologically slow in a debug build. Run it in RELEASE:
    ///
    /// ```text
    /// SCENEWORKS_MOSS_TTSD_AR_DIR=~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-TTSD-v0.5/\
    ///   snapshots/8527b9136b6afefe2252ae597cecea2e80e7ebeb \
    /// SCENEWORKS_CHATTERBOX_VE_FILE=~/.cache/huggingface/hub/models--ResembleAI--chatterbox/\
    ///   snapshots/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18/ve.safetensors \
    /// SCENEWORKS_MOSS_TTSD_CODEC_FILE=~/.cache/huggingface/hub/models--OpenMOSS-Team--XY_Tokenizer_TTSD_V0/\
    ///   snapshots/c83433728e698ed0698e88cb5096bc221fb8f8c5/xy_tokenizer.ckpt \
    /// HF_HUB_OFFLINE=1 \
    /// cargo test --release -p sceneworks-worker -- --ignored --nocapture \
    ///   moss_ttsd_renders_a_multi_speaker_dialogue
    /// ```
    ///
    /// It loads the real generator through the worker's `load_audio` seam, renders (a) the 2-speaker
    /// dialogue as ONE clip, (b) three probe clips whose chatterbox_ve embeddings give a cross-speaker
    /// cosine materially below the same-speaker (self) cosine — objective proof the two turns are
    /// DIFFERENT voices — and (c) a single-voice (no-script) control. It asserts the DoD: both segments
    /// rendered (the dialogue is longer than one turn alone), 24 kHz, non-silent, cross < self, and the
    /// single-voice control still renders. Writes the dialogue WAV + an evidence JSON to
    /// `SCENEWORKS_DOD_OUT` (or the temp dir).
    #[test]
    #[ignore = "requires the ~6GB MOSS-TTSD + XY_Tokenizer weights + chatterbox_ve; DoD, run manually in release"]
    fn moss_ttsd_renders_a_multi_speaker_dialogue_in_distinct_voices() {
        let ar_dir = std::env::var("SCENEWORKS_MOSS_TTSD_AR_DIR").expect(
            "set SCENEWORKS_MOSS_TTSD_AR_DIR to the cached MOSS-TTSD-v0.5 snapshot directory",
        );
        let ve_file = std::env::var("SCENEWORKS_CHATTERBOX_VE_FILE").expect(
            "set SCENEWORKS_CHATTERBOX_VE_FILE to the cached chatterbox ve.safetensors path",
        );
        // moss_ttsd_v05 advertises `required_components: ["codec"]` (sc-13681); on the self-fetch-free
        // inference pin (sc-13818) the loader HARD-FAILS unless the XY_Tokenizer codec is staged in the
        // LoadSpec — it no longer self-fetches it. Stage it exactly as production `resolve_co_requisites`
        // does: a single-file co-requisite (`xy_tokenizer.ckpt`) resolves to `WeightsSource::File`
        // (model_jobs.rs:1684). This was silently broken by the pin bump because the smoke is `#[ignore]`d.
        let codec_file = std::env::var("SCENEWORKS_MOSS_TTSD_CODEC_FILE").expect(
            "set SCENEWORKS_MOSS_TTSD_CODEC_FILE to the cached XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt path \
             (moss_ttsd_v05's required `codec` component)",
        );

        let generator = crate::inference_runtime::load_audio(
            "moss_ttsd_v05",
            &LoadSpec::new(WeightsSource::Dir(PathBuf::from(&ar_dir)))
                .with_component("codec", WeightsSource::File(PathBuf::from(&codec_file))),
        )
        .expect("load the real moss_ttsd_v05 generator from the cached snapshot + staged codec");
        assert!(
            generator.descriptor().capabilities.supports_multi_speaker,
            "moss_ttsd_v05 must advertise supports_multi_speaker"
        );
        assert_eq!(
            generator.descriptor().capabilities.max_speakers,
            Some(2),
            "moss_ttsd_v05 advertises max_speakers = 2"
        );

        // Render one request (script OR plain prompt) into a track through the generator seam.
        let render =
            |script: Option<Vec<SpeechSegment>>, prompt: &str, secs: f32| -> gen_core::AudioTrack {
                let req = GenerationRequest {
                    prompt: prompt.to_owned(),
                    audio: Some(AudioParams {
                        language: Some("en".to_owned()),
                        target_duration: Some(secs),
                        script,
                        ..Default::default()
                    }),
                    ..Default::default()
                };
                let mut on_progress = |_p: Progress| {};
                match generator
                    .generate(&req, &mut on_progress)
                    .expect("real MOSS-TTSD synthesis succeeds")
                {
                    GenerationOutput::Audio(track) => track,
                    other => panic!("expected audio, got {other:?}"),
                }
            };
        let seg = |text: &str, speaker: &str| SpeechSegment {
            text: text.to_owned(),
            speaker: Some(speaker.to_owned()),
            style: None,
        };
        // Two contrasting lines. The turns are UNBALANCED (S1's line is much shorter than S2's), so a
        // blind 50% midpoint split of the dialogue would NOT separate the voices — the midpoint lands
        // deep inside S2. Instead we isolate each voice explicitly (below).
        let line_a = "Hello, how are you today?";
        let line_b = "I'm doing great, thanks for asking!";

        // (a) Single-voice control — NO script, one plain-prompt line. It doubles as the both-turns
        // length reference (one line spoken) and proves the multi-speaker script field is additive
        // (single-voice byte-for-byte unaffected).
        let single_voice = render(None, line_a, 4.0);
        let single_rms = (single_voice.samples.iter().map(|s| s * s).sum::<f32>()
            / single_voice.samples.len().max(1) as f32)
            .sqrt();
        assert_eq!(single_voice.sample_rate, 24_000, "MOSS-TTSD renders 24 kHz");
        assert!(
            !single_voice.samples.is_empty() && single_rms > 1e-3,
            "the single-voice control must render non-silent audio (rms={single_rms})"
        );

        // (b) A pure voice-1 reference clip for the SAME line the second turn speaks. A single-segment
        // script maps its sole (first-seen) speaker onto MOSS-TTSD's `[S1]` turn tag, so this renders
        // entirely in the FIRST voice — a 100%-one-speaker clip (used for the cross comparison + the
        // same-speaker self baseline, where an intra-clip split is valid because it is one voice).
        let v1_line_b = render(Some(vec![seg(line_b, "S1")]), "", 4.0);

        // (c) The real 2-speaker dialogue — S1 then S2 — rendered into ONE clip. `[S1]line_a[S2]line_b`
        // renders line_a in voice 1 then line_b in voice 2.
        let dialogue = render(Some(vec![seg(line_a, "S1"), seg(line_b, "S2")]), "", 6.0);
        assert_eq!(dialogue.sample_rate, 24_000, "MOSS-TTSD renders 24 kHz");
        assert!(
            !dialogue.samples.is_empty() && dialogue.samples.iter().all(|s| s.is_finite()),
            "the dialogue clip must be a non-empty, finite waveform"
        );
        let dialogue_peak = dialogue.samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        let dialogue_rms = (dialogue.samples.iter().map(|s| s * s).sum::<f32>()
            / dialogue.samples.len() as f32)
            .sqrt();
        assert!(
            dialogue_rms > 1e-3,
            "the dialogue clip must be audible (rms={dialogue_rms})"
        );
        let dialogue_secs = dialogue.samples.len() as f32 / 24_000.0;

        // BOTH turns must actually render: the 2-turn dialogue must be materially LONGER than one line
        // spoken alone (the single-voice control). A single dropped turn would leave a ~one-line clip,
        // so this rejects that; the speaker-distinctness check below is the semantic both-turns guard
        // (a dropped 2nd turn leaves the tail in voice 1 → cross ≈ self → the distinctness assert fails).
        assert!(
            dialogue.samples.len() > single_voice.samples.len() * 2,
            "the 2-turn dialogue ({} samples) must be materially longer than a single-line control \
             ({} samples) — both segments must render",
            dialogue.samples.len(),
            single_voice.samples.len()
        );

        // (c) Voice distinctness — with the speakers ISOLATED (never a blind midpoint split of the
        // unbalanced dialogue). S1's turn is the SHORT first turn, so the dialogue's TAIL is purely
        // voice 2 saying line_b; `v1_line_b` is purely voice 1 saying the SAME line. Comparing them
        // holds the TEXT constant and varies ONLY the speaker, so a low cosine is a genuine speaker
        // difference (chatterbox_ve is a text-independent speaker-identity encoder). The same-speaker
        // baseline is measured within a SINGLE-speaker clip (halves of `v1_line_b`), where the split is
        // valid because it is one voice throughout. The AUTHORITATIVE distinctness proof is the upstream
        // inference conformance DoD (cross 0.4953 vs self 0.749, real weights); this SceneWorks test
        // confirms both turns present + non-silent + a correctly-ordered, speaker-isolated cross < self.
        let embedder = crate::inference_runtime::load_voice_embedder(
            "chatterbox_ve",
            &LoadSpec::new(WeightsSource::File(PathBuf::from(&ve_file))),
        )
        .expect("load the real chatterbox_ve embedder");
        let ve_embed = |samples: &[f32]| -> Vec<f32> {
            embedder
                .embed(&gen_core::AudioTrack {
                    samples: samples.to_vec(),
                    sample_rate: dialogue.sample_rate,
                    channels: dialogue.channels.max(1),
                    stems: Vec::new(),
                })
                .expect("embed a waveform slice")
        };
        let cosine = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                dot / (na * nb)
            }
        };
        // Voice-2 window: the last 45% of the dialogue. S1's (short) turn is FIRST, so this tail is deep
        // inside S2's turn — guaranteed pure voice 2 (no midpoint contamination from voice 1).
        let v2_start = (dialogue.samples.len() as f32 * 0.55) as usize;
        let v2_line_b = &dialogue.samples[v2_start..];
        // Same-speaker self-similarity: two halves of the pure voice-1 clip (one voice throughout, so
        // the split is valid — the exact flaw the old 50%-of-unbalanced-dialogue split had).
        let (v1b_first, v1b_second) = v1_line_b.samples.split_at(v1_line_b.samples.len() / 2);
        let self_cosine = cosine(&ve_embed(v1b_first), &ve_embed(v1b_second));
        // Cross-speaker: the SAME line (line_b), voice 1 (`v1_line_b`) vs voice 2 (dialogue tail) —
        // text held constant, only the speaker varies.
        let cross_cosine = cosine(&ve_embed(&v1_line_b.samples), &ve_embed(v2_line_b));
        assert!(
            cross_cosine < self_cosine - 0.05,
            "the two turns must be acoustically distinct: speaker-isolated cross cosine \
             {cross_cosine:.4} (voice 1 vs voice 2, same line) is not materially below the \
             same-speaker self-similarity {self_cosine:.4}"
        );

        // Save the dialogue artifact + evidence.
        let out_dir = std::env::var("SCENEWORKS_DOD_OUT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());
        std::fs::create_dir_all(&out_dir).ok();
        let wav_path = out_dir.join("moss_ttsd_dialogue_dod.wav");
        let wav = AudioTrack {
            samples: dialogue.samples.clone(),
            sample_rate: dialogue.sample_rate,
            channels: dialogue.channels.max(1),
        };
        write_wav_pcm16(&wav, &wav_path).expect("write the dialogue WAV");
        let evidence = json!({
            "model": "moss_ttsd_v05",
            "script": [{ "speaker": "S1", "text": line_a }, { "speaker": "S2", "text": line_b }],
            "sampleRate": dialogue.sample_rate,
            "channels": dialogue.channels,
            "dialogueSamples": dialogue.samples.len(),
            "dialogueDurationSecs": dialogue_secs,
            "dialoguePeakAmplitude": dialogue_peak,
            "dialogueRms": dialogue_rms,
            "singleVoiceControlSamples": single_voice.samples.len(),
            "singleVoiceControlRms": single_rms,
            "bothTurnsRendered": dialogue.samples.len() > single_voice.samples.len() * 2,
            "pureV1LineBSamples": v1_line_b.samples.len(),
            "v2WindowStartSample": v2_start,
            "distinctnessMethod": "speaker-isolated: cross = cosine(pure voice-1 line_b, dialogue voice-2 tail [last 45%], SAME line line_b); self = cosine(halves of the pure voice-1 line_b clip)",
            "crossCosineSameLineDifferentSpeaker": cross_cosine,
            "selfCosineSameSpeakerHalves": self_cosine,
            "crossMateriallyBelowSelf": cross_cosine < self_cosine - 0.05,
            "authoritativeUpstreamDoD": "inference conformance: cross 0.4953 vs self 0.749",
            "wavPath": wav_path.display().to_string(),
        });
        let evidence_path = out_dir.join("moss_ttsd_dialogue_dod.json");
        std::fs::write(
            &evidence_path,
            serde_json::to_string_pretty(&evidence).unwrap(),
        )
        .expect("write the evidence JSON");
        eprintln!(
            "DoD evidence: {}\nWAV: {}\nself={self_cosine:.4} cross={cross_cosine:.4}",
            evidence_path.display(),
            wav_path.display()
        );
    }
}

/// The YuE job surface (sc-19384): the full R5 control set reaches the engine request, a completed
/// job persists the mix plus both stems, engine progress maps onto job events, and a mid-job cancel
/// leaves no generation-set directory or ICL scratch behind. Every test drives the REAL job path
/// through the injected-loader seam (`run_audio_generate_job_using` / `run_audio_synthesis_with`)
/// against a stub [`Generator`] — never a replica of an engine loader.
#[cfg(test)]
mod yue_job_surface_tests {
    use super::*;

    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use axum::{
        extract::{Path as AxumPath, State},
        response::{IntoResponse, Response},
        routing::{get, post},
        Json, Router,
    };

    /// A model id no audio registry knows, so `audio_descriptor` is `None` and the test stages no
    /// co-requisites — the loaded generator is the stub whatever inference pin is linked.
    const STUB_MODEL: &str = "yue_stub_song";
    const STUB_REPO: &str = "SceneWorks/yue-stub-song-candle";
    const STUB_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
    const LYRICS: &str = "[verse]\nwalking down the empty street\n[chorus]\nsing it loud";

    fn tiered_entry() -> Value {
        json!({
            "id": STUB_MODEL,
            "family": "yue",
            "type": "audio",
            "audio": { "supportsSegmentedLyrics": true, "supportsRepetitionPenalty": true,
                       "supportsReferenceRegion": true, "conditioning": ["ReferenceAudio"] },
            "downloads": [
                { "provider": "huggingface", "repo": STUB_REPO, "revision": STUB_REVISION,
                  "variant": "q4", "default": true, "files": ["q4/*"],
                  "estimatedSizeBytes": 1_000_000 },
                { "provider": "huggingface", "repo": STUB_REPO, "revision": STUB_REVISION,
                  "variant": "q8", "files": ["q8/*"], "estimatedSizeBytes": 2_000_000 },
                { "provider": "huggingface", "repo": STUB_REPO, "revision": STUB_REVISION,
                  "variant": "bf16", "files": ["bf16/*"], "estimatedSizeBytes": 4_000_000 },
                // Sized co-requisites so the YuE memory gate (sc-19386) can price the stub render;
                // the stub model registers no descriptor, so nothing stages them.
                { "provider": "huggingface", "repo": "SceneWorks/yue-stub-stage2", "coRequisite": true,
                  "componentId": "stage2", "variant": "q4", "subdir": "q4", "files": ["q4/*"],
                  "estimatedSizeBytes": 500_000 },
                { "provider": "huggingface", "repo": "SceneWorks/yue-stub-stage2", "coRequisite": true,
                  "componentId": "stage2", "variant": "q8", "subdir": "q8", "files": ["q8/*"],
                  "estimatedSizeBytes": 700_000 },
                { "provider": "huggingface", "repo": "SceneWorks/yue-stub-stage2", "coRequisite": true,
                  "componentId": "stage2", "variant": "bf16", "subdir": "bf16", "files": ["bf16/*"],
                  "estimatedSizeBytes": 900_000 },
                { "provider": "huggingface", "repo": "SceneWorks/yue-stub-xcodec", "coRequisite": true,
                  "componentId": "xcodec", "files": ["final_ckpt/*"], "estimatedSizeBytes": 300_000 }
            ],
        })
    }

    fn full_payload(project_id: &str) -> Value {
        json!({
            "projectId": project_id,
            "model": STUB_MODEL,
            "prompt": "uplifting pop female vocal airy",
            "lyrics": LYRICS,
            "segments": 3,
            "maxNewTokensPerSegment": 1500,
            "repetitionPenalty": 1.25,
            "seed": 11,
            "guidanceEnabled": true,
            "guidance": 1.5,
            "outputLimiter": "Rescale",
            "modelManifestEntry": tiered_entry(),
        })
    }

    fn job_snapshot(job_id: &str, payload: Value) -> JobSnapshot {
        serde_json::from_value(job_value(job_id, payload, false))
            .expect("audio job snapshot deserializes")
    }

    fn job_value(job_id: &str, payload: Value, cancel: bool) -> Value {
        json!({
            "id": job_id, "type": "audio_generate", "status": "running",
            "projectId": null, "projectName": null, "payload": payload, "result": {},
            "requestedGpu": "auto", "assignedGpu": null, "workerId": "test-worker",
            "progress": 0.0, "stage": "queued", "message": "queued", "error": null,
            "etaSeconds": null, "elapsedSeconds": null, "attempts": 1,
            "sourceJobId": null, "duplicateOfJobId": null, "cancelRequested": cancel,
            "createdAt": "2026-09-24T00:00:00Z", "updatedAt": "2026-09-24T00:00:00Z",
            "startedAt": null, "completedAt": null, "canceledAt": null, "lastHeartbeatAt": null
        })
    }

    #[derive(Clone)]
    struct StubApi {
        /// Reported as the job's `cancelRequested` — flipped mid-job by a test.
        cancel: Arc<AtomicBool>,
        progress: Arc<Mutex<Vec<Value>>>,
    }

    fn job_json(job_id: &str, cancel: bool) -> Value {
        job_value(job_id, json!({}), cancel)
    }

    /// An API stub answering the job GET (cancel state), progress POST (recorded) and worker
    /// heartbeat routes the audio job path calls.
    async fn spawn_stub_api() -> (String, StubApi) {
        async fn job_route(
            State(state): State<StubApi>,
            AxumPath(job_id): AxumPath<String>,
        ) -> Response {
            Json(job_json(&job_id, state.cancel.load(Ordering::SeqCst))).into_response()
        }
        async fn progress_route(
            State(state): State<StubApi>,
            AxumPath(job_id): AxumPath<String>,
            Json(body): Json<Value>,
        ) -> Response {
            state.progress.lock().expect("progress lock").push(body);
            Json(job_json(&job_id, state.cancel.load(Ordering::SeqCst))).into_response()
        }
        async fn heartbeat_route() -> Response {
            Json(json!({})).into_response()
        }
        let state = StubApi {
            cancel: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(Mutex::new(Vec::new())),
        };
        let app = Router::new()
            .route("/api/v1/jobs/:job_id", get(job_route))
            .route("/api/v1/jobs/:job_id/progress", post(progress_route))
            .route(
                "/api/v1/workers/:worker_id/heartbeat",
                post(heartbeat_route),
            )
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener binds");
        let address = listener.local_addr().expect("listener has address");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("stub serves");
        });
        (format!("http://{address}"), state)
    }

    /// Settings over an isolated data dir holding a project and the stub model's staged snapshot
    /// (`q4/` and `q8/` installed, `bf16/` not). The env guard + tempdir must outlive the test.
    struct Staged {
        _env: crate::test_env::EnvVars,
        _data_dir: tempfile::TempDir,
        settings: Settings,
        project_id: String,
        project_path: PathBuf,
        snapshot: PathBuf,
    }

    fn stage(base_url: String) -> Staged {
        let env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ]);
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let snapshot =
            sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir.path(), STUB_REPO)
                .expect("repo cache path resolves")
                .join("snapshots")
                .join(STUB_REVISION);
        for tier in ["q4", "q8"] {
            std::fs::create_dir_all(snapshot.join(tier)).expect("tier dir");
            std::fs::write(snapshot.join(tier).join("model.safetensors"), b"weights")
                .expect("tier weights");
        }
        let mut settings = Settings::from_env();
        settings.api_url = base_url;
        settings.worker_id = "test-worker".to_owned();
        settings.heartbeat_seconds = 5;
        settings.data_dir = data_dir.path().to_path_buf();
        let project = ProjectStore::new(settings.data_dir.clone(), "worker")
            .create_project("Song Project")
            .expect("project creates");
        Staged {
            _env: env,
            _data_dir: data_dir,
            settings,
            project_id: project.id,
            project_path: PathBuf::from(project.path),
            snapshot,
        }
    }

    /// What the stub generator saw: the request fields the job built and the load it was handed.
    #[derive(Default)]
    struct Seen {
        weights: Option<PathBuf>,
        quantize: Option<gen_core::Quant>,
        audio: Option<AudioParams>,
        prompt: String,
        seed: Option<u64>,
        guidance: Option<f32>,
        conditioning: Vec<Conditioning>,
        components: BTreeMap<String, WeightsSource>,
    }

    #[derive(Clone, Copy)]
    enum Behavior {
        /// Emit YuE-shaped progress, then return a mix + `vocals` + `instrumental`.
        Song,
        /// Flag `started`, then block until the request's cancel flag trips.
        WaitForCancel,
        /// Emit this many `Step`s as fast as a per-token engine would (then return a song).
        ManySteps(u32),
        /// Emit `Decoding` then `Step { 2, 3 }` (a model interleaving the two), then return a song.
        DecodeThenStep,
    }

    struct StubSong {
        descriptor: gen_core::ModelDescriptor,
        behavior: Behavior,
        seen: Arc<Mutex<Seen>>,
        started: Arc<AtomicBool>,
    }

    fn tone(len: usize, freq: f32, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 * freq * 0.001).sin() * amp)
            .collect()
    }

    /// A 0.1 s mix of two tones plus the two stems it is the sum of.
    fn song_output() -> GenerationOutput {
        let len = 4_410;
        let vocals = tone(len, 440.0, 0.3);
        let instrumental = tone(len, 110.0, 0.3);
        let mix = vocals
            .iter()
            .zip(&instrumental)
            .map(|(a, b)| a + b)
            .collect();
        GenerationOutput::Audio(gen_core::AudioTrack {
            samples: mix,
            sample_rate: 44_100,
            channels: 1,
            stems: vec![
                gen_core::AudioStem {
                    name: "vocals".to_owned(),
                    samples: vocals,
                },
                gen_core::AudioStem {
                    name: "instrumental".to_owned(),
                    samples: instrumental,
                },
            ],
        })
    }

    impl gen_core::Generator for StubSong {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            req: &GenerationRequest,
            on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            {
                let mut seen = self.seen.lock().expect("seen lock");
                seen.audio = req.audio.clone();
                seen.prompt = req.prompt.clone();
                seen.seed = req.seed;
                seen.guidance = req.guidance;
                seen.conditioning = req.conditioning.clone();
            }
            match self.behavior {
                Behavior::ManySteps(total) => {
                    for current in 1..=total {
                        on_progress(Progress::Step { current, total });
                        std::thread::sleep(Duration::from_micros(100));
                    }
                    Ok(song_output())
                }
                Behavior::DecodeThenStep => {
                    on_progress(Progress::Decoding);
                    on_progress(Progress::Step {
                        current: 2,
                        total: 3,
                    });
                    Ok(song_output())
                }
                Behavior::Song => {
                    // The YuE provider's progress contract: Loading per LM, one Step per lyric
                    // segment and per stage-2 track (total = segments + 2), one Decoding. Each event
                    // is spaced past the pump's post interval, the way a real multi-second segment
                    // is, so every one is observable as its own job event.
                    let mut emit = |progress: Progress| {
                        on_progress(progress);
                        std::thread::sleep(PROGRESS_POST_INTERVAL + Duration::from_millis(40));
                    };
                    emit(Progress::Loading(gen_core::LoadPhase::Renderer));
                    for current in 1..=2 {
                        emit(Progress::Step { current, total: 4 });
                    }
                    emit(Progress::Loading(gen_core::LoadPhase::Renderer));
                    for current in 3..=4 {
                        emit(Progress::Step { current, total: 4 });
                    }
                    emit(Progress::Decoding);
                    Ok(song_output())
                }
                Behavior::WaitForCancel => {
                    self.started.store(true, Ordering::SeqCst);
                    let start = Instant::now();
                    while !req.cancel.is_cancelled() {
                        if start.elapsed() > Duration::from_secs(30) {
                            return Err(gen_core::Error::Msg("cancel never tripped".to_owned()));
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(gen_core::Error::Canceled)
                }
            }
        }
    }

    fn stub_loader(
        behavior: Behavior,
        seen: Arc<Mutex<Seen>>,
        started: Arc<AtomicBool>,
    ) -> impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>> + Send + 'static {
        move |_id: &str, spec: &LoadSpec| {
            {
                let mut record = seen.lock().expect("seen lock");
                record.weights = match &spec.weights {
                    WeightsSource::Dir(dir) | WeightsSource::File(dir) => Some(dir.clone()),
                };
                record.quantize = spec.quantize;
                record.components = spec.components.clone();
            }
            Ok(Box::new(StubSong {
                descriptor: gen_core::ModelDescriptor {
                    id: "yue_stub_song",
                    family: "yue",
                    backend: "candle",
                    modality: gen_core::Modality::Audio,
                    capabilities: gen_core::Capabilities::default(),
                    encoder_contract: None,
                    denoiser_output_latent_space: None,
                    required_components: &[],
                    control_kinds: None,
                },
                behavior,
                seen,
                started,
            }) as Box<dyn Generator>)
        }
    }

    #[test]
    fn from_payload_reads_the_full_yue_control_set() {
        let mut payload = full_payload("project-1");
        payload["iclMode"] = json!("Dual");
        payload["iclVocalAssetId"] = json!("asset_v");
        payload["iclInstrumentalAssetId"] = json!("asset_i");
        payload["iclStartSecs"] = json!(5.0);
        payload["iclEndSecs"] = json!(25.0);
        payload["quantTier"] = json!("Q8");
        let request = AudioRequest::from_payload(payload.as_object().expect("object"));
        assert_eq!(request.lyrics.as_deref(), Some(LYRICS));
        assert_eq!(request.segments, Some(3));
        assert_eq!(request.max_new_tokens_per_segment, Some(1500));
        assert_eq!(request.repetition_penalty, Some(1.25));
        assert_eq!(request.seed, Some(11));
        assert_eq!(request.guidance_enabled, Some(true));
        assert_eq!(request.effective_guidance(), Some(1.5));
        assert_eq!(request.icl_mode.as_deref(), Some("dual"));
        assert_eq!(request.quant_tier.as_deref(), Some("q8"));
        assert_eq!(request.output_limiter.as_deref(), Some("rescale"));
        assert_eq!(
            request.reference_region(),
            Some(TimeRegion {
                start_secs: 5.0,
                end_secs: Some(25.0)
            })
        );
        assert_eq!(
            request.icl_references().expect("dual resolves"),
            vec![
                (Some("vocals"), "asset_v"),
                (Some("instrumental"), "asset_i")
            ]
        );

        // Guidance OFF rides as a 0.0 scale (YuE reads 0..=1 as CFG off).
        let mut off = full_payload("project-1");
        off["guidanceEnabled"] = json!(false);
        off.as_object_mut().expect("object").remove("guidance");
        let off = AudioRequest::from_payload(off.as_object().expect("object"));
        assert_eq!(off.effective_guidance(), Some(0.0));
        // No window ⇒ the model's default window.
        assert_eq!(off.reference_region(), None);
        // A start with no end mirrors upstream's `prompt_end_time` default: 5–30 s.
        let mut start_only = full_payload("project-1");
        start_only["iclMode"] = json!("single");
        start_only["iclReferenceAssetId"] = json!("a");
        start_only["iclStartSecs"] = json!(5.0);
        let start_only = AudioRequest::from_payload(start_only.as_object().expect("object"));
        assert_eq!(
            start_only.reference_region(),
            Some(TimeRegion {
                start_secs: 5.0,
                end_secs: Some(30.0)
            })
        );
        audio_preflight(&start_only).expect("a 5–30 s window is well-formed");

        // Malformed ICL pairings are refused worker-side too (a raw-enqueued job).
        for (extra, needle) in [
            (
                json!({ "iclMode": "single" }),
                "requires iclReferenceAssetId",
            ),
            (
                json!({ "iclMode": "dual", "iclVocalAssetId": "v" }),
                "requires iclInstrumentalAssetId",
            ),
            (
                json!({ "iclMode": "single", "iclReferenceAssetId": "a", "iclVocalAssetId": "v" }),
                "does not apply",
            ),
            (json!({ "iclReferenceAssetId": "a" }), "need an iclMode"),
            (json!({ "iclMode": "triple" }), "must be"),
            (
                json!({ "iclMode": "single", "iclReferenceAssetId": "a", "iclStartSecs": 30.0 }),
                "must satisfy 0 <= start < end",
            ),
            (
                json!({ "outputLimiter": "normalize" }),
                "outputLimiter must be",
            ),
        ] {
            let mut payload = full_payload("project-1");
            for (key, value) in extra.as_object().expect("object") {
                payload[key] = value.clone();
            }
            let request = AudioRequest::from_payload(payload.as_object().expect("object"));
            let error = audio_preflight(&request).expect_err(needle);
            assert!(error.to_string().contains(needle), "{needle}: {error}");
        }
    }

    /// sc-19386: the YuE memory gate prices the tier `resolve_audio_tier` resolves — the SAME
    /// function the synthesis arm loads through — and the request fields the job parsed.
    #[test]
    fn an_unset_tier_prices_the_tier_the_job_will_load() {
        use crate::yue_admission::{YueRenderShape, YueTier};
        let root = tempfile::tempdir().expect("snapshot root");
        let root_path = root.path().to_path_buf();
        let request = |tier: Option<&str>| {
            let mut payload = full_payload("p");
            payload["modelManifestEntry"] = builtin_entry("yue_en_cot");
            if let Some(tier) = tier {
                payload["quantTier"] = json!(tier);
            }
            AudioRequest::from_payload(payload.as_object().expect("object"))
        };
        let priced = |request: &AudioRequest| {
            let (_, tier) =
                resolve_audio_tier(request, root_path.clone()).expect("a tier resolves");
            YueRenderShape::new(&yue_request_facts(request, tier.as_ref()))
                .expect("priced")
                .tier
        };
        // Only q8 installed: the default (q4) is absent, so the first installed tier is priced.
        std::fs::create_dir_all(root.path().join("q8")).expect("q8");
        assert_eq!(priced(&request(None)), YueTier::Q8);
        // The default wins once installed; an explicit pick is priced as asked.
        std::fs::create_dir_all(root.path().join("q4")).expect("q4");
        std::fs::create_dir_all(root.path().join("bf16")).expect("bf16");
        assert_eq!(priced(&request(None)), YueTier::Q4);
        assert_eq!(priced(&request(Some("BF16"))), YueTier::Bf16);
        // The rest of the shape is the job's own parse: segments / budget / guidance.
        let request = request(None);
        let (_, tier) = resolve_audio_tier(&request, root_path.clone()).unwrap();
        let shape = YueRenderShape::new(&yue_request_facts(&request, tier.as_ref())).unwrap();
        assert_eq!(shape.max_new_tokens, 1500);
        assert!(shape.cfg, "guidance 1.5 keeps CFG on");
        assert_eq!(
            shape.segments,
            3.min(crate::yue_admission::lyric_section_count(LYRICS) as u32)
        );
        let mut off = full_payload("p");
        off["guidanceEnabled"] = json!(false);
        let off = AudioRequest::from_payload(off.as_object().unwrap());
        let shape = YueRenderShape::new(&yue_request_facts(&off, tier.as_ref())).unwrap();
        assert!(!shape.cfg, "guidanceEnabled=false sends 0.0 ⇒ CFG off");
    }

    /// sc-19386: the YuE gate runs right after preflight, before the project, the weights, any
    /// reference clip or source track is touched.
    #[test]
    fn the_audio_job_runs_the_yue_gate_before_touching_the_project_or_weights() {
        let source = include_str!("audio_jobs.rs");
        let body = source
            .split_once("async fn run_audio_generate_job_using(")
            .expect("audio job body")
            .1;
        let gate = body
            .find("crate::yue_admission::check(")
            .expect("the audio job must run the YuE admission gate");
        let preflight = body.find("audio_preflight(&request)?").expect("preflight");
        assert!(preflight < gate, "the gate runs after preflight");
        for later in [
            "get_project(",
            "build_audio_edit(",
            "resolve_icl_reference(",
            "resolve_voice_clone_plan(",
            "run_audio_synthesis_with(",
        ] {
            let at = body
                .find(later)
                .unwrap_or_else(|| panic!("{later} in the job"));
            assert!(gate < at, "the gate must run before {later}");
        }
    }

    #[test]
    fn tier_resolution_honors_the_request_then_the_default_then_what_is_installed() {
        let root = tempfile::tempdir().expect("snapshot root");
        std::fs::create_dir_all(root.path().join("q8")).expect("q8");
        let request = |tier: Option<&str>, entry: Value| {
            let mut payload = full_payload("p");
            payload["modelManifestEntry"] = entry;
            if let Some(tier) = tier {
                payload["quantTier"] = json!(tier);
            }
            AudioRequest::from_payload(payload.as_object().expect("object"))
        };
        let root_path = root.path().to_path_buf();

        // Explicit, installed: that tier's subdir, asserted on the load.
        let (dir, tier) =
            resolve_audio_tier(&request(Some("q8"), tiered_entry()), root_path.clone())
                .expect("q8 resolves");
        assert_eq!(dir, root_path.join("q8"));
        assert_eq!(
            tier,
            Some(AudioTier {
                name: "q8".to_owned(),
                quantize: Some(gen_core::Quant::Q8)
            })
        );
        // Explicit, NOT installed: a clear refusal, never a silent substitute.
        let error = resolve_audio_tier(&request(Some("q4"), tiered_entry()), root_path.clone())
            .expect_err("q4 is not installed");
        assert!(
            error.to_string().contains("q4 tier is not installed"),
            "{error}"
        );
        // Unrequested: the default (q4) is not installed, so the first installed tier (q8) loads.
        let (dir, _) = resolve_audio_tier(&request(None, tiered_entry()), root_path.clone())
            .expect("an installed tier resolves");
        assert_eq!(dir, root_path.join("q8"));
        // ...and the default wins once it is installed; bf16 asserts nothing on the load.
        std::fs::create_dir_all(root.path().join("q4")).expect("q4");
        std::fs::create_dir_all(root.path().join("bf16")).expect("bf16");
        let (dir, tier) = resolve_audio_tier(&request(None, tiered_entry()), root_path.clone())
            .expect("default resolves");
        assert_eq!(dir, root_path.join("q4"));
        assert_eq!(tier.map(|t| t.quantize), Some(Some(gen_core::Quant::Q4)));
        let (_, tier) =
            resolve_audio_tier(&request(Some("bf16"), tiered_entry()), root_path.clone())
                .expect("bf16 resolves");
        assert_eq!(tier.map(|t| t.quantize), Some(None));

        // An untiered model passes the snapshot through untouched — and refuses a tier request.
        let untiered =
            json!({ "id": "kokoro_82m", "downloads": [{ "repo": "hexgrad/Kokoro-82M" }] });
        let (dir, tier) = resolve_audio_tier(&request(None, untiered.clone()), root_path.clone())
            .expect("untiered passes through");
        assert_eq!((dir, tier), (root_path.clone(), None));
        assert!(resolve_audio_tier(&request(Some("q4"), untiered), root_path).is_err());
    }

    /// AC1: every R5 knob the job carries reaches the engine request / load — lyrics, genre tags,
    /// segments, per-segment budget, repetition penalty, seed, guidance scale, the ICL reference
    /// (dual-track: vocals + instrumental stems) and its window, and the tier (weights dir + quant).
    #[tokio::test]
    async fn synthesis_carries_every_yue_knob_to_the_engine() {
        let (base_url, _api_state) = spawn_stub_api().await;
        let mut settings = Settings::from_env();
        settings.api_url = base_url;
        settings.worker_id = "test-worker".to_owned();
        settings.heartbeat_seconds = 5;
        let api = ApiClient::new(&settings);
        let mut payload = full_payload("project-1");
        payload["iclMode"] = json!("dual");
        payload["iclVocalAssetId"] = json!("asset_v");
        payload["iclInstrumentalAssetId"] = json!("asset_i");
        payload["iclStartSecs"] = json!(5.0);
        payload["iclEndSecs"] = json!(25.0);
        let job = job_snapshot("yue-knobs", payload.clone());
        let request = AudioRequest::from_payload(payload.as_object().expect("object"));
        let clip = |value: f32| gen_core::AudioTrack {
            samples: vec![value; 1_600],
            sample_rate: ICL_REFERENCE_SAMPLE_RATE,
            channels: ICL_REFERENCE_CHANNELS,
            stems: Vec::new(),
        };
        let reference = assemble_icl_track(vec![
            (Some("vocals"), clip(0.25)),
            (Some("instrumental"), clip(-0.5)),
        ]);
        let seen = Arc::new(Mutex::new(Seen::default()));
        let track = run_audio_synthesis_with(
            &api,
            &settings,
            &job,
            &request,
            SinglePlan {
                model_dir: PathBuf::from("/staged/yue/q8"),
                tier: Some(AudioTier {
                    name: "q8".to_owned(),
                    quantize: Some(gen_core::Quant::Q8),
                }),
                conditioning: Some(Conditioning::ReferenceAudio {
                    audio: reference,
                    strength: None,
                }),
            },
            stub_loader(
                Behavior::Song,
                seen.clone(),
                Arc::new(AtomicBool::new(false)),
            ),
        )
        .await
        .expect("synthesis completes");
        assert_eq!(track.stems.len(), 2);

        let seen = seen.lock().expect("seen lock");
        assert_eq!(seen.weights.as_deref(), Some(Path::new("/staged/yue/q8")));
        assert_eq!(seen.quantize, Some(gen_core::Quant::Q8));
        assert_eq!(seen.prompt, "uplifting pop female vocal airy");
        assert_eq!(seen.seed, Some(11));
        assert_eq!(seen.guidance, Some(1.5));
        let audio = seen.audio.as_ref().expect("audio params");
        assert_eq!(audio.lyrics.as_deref(), Some(LYRICS));
        assert_eq!(audio.segments, Some(3));
        assert_eq!(audio.max_new_tokens_per_segment, Some(1500));
        assert_eq!(audio.repetition_penalty, Some(1.25));
        assert_eq!(audio.output_limiter, Some(gen_core::OutputLimiter::Rescale));
        assert_eq!(
            audio.reference_region,
            Some(TimeRegion {
                start_secs: 5.0,
                end_secs: Some(25.0)
            })
        );
        match seen.conditioning.as_slice() {
            [Conditioning::ReferenceAudio { audio, strength }] => {
                assert!(strength.is_none());
                let names: Vec<&str> = audio.stems.iter().map(|s| s.name.as_str()).collect();
                assert_eq!(names, ["vocals", "instrumental"]);
                assert_eq!(audio.sample_rate, ICL_REFERENCE_SAMPLE_RATE);
                assert!(audio.samples.iter().all(|&s| (s - (-0.25)).abs() < 1e-6));
            }
            other => panic!(
                "expected one dual-track ReferenceAudio, got {} items",
                other.len()
            ),
        }
    }

    /// AC2 + AC3 (progress): a completed job persists THREE audio assets — the mix, the vocal stem
    /// and the instrumental stem, each a WAV on disk and an `assetWrite` with its own id — and the
    /// engine's per-segment / per-stage progress is posted as Running job events.
    #[tokio::test]
    async fn completed_job_persists_the_mix_and_both_stems_with_segment_progress() {
        let (base_url, api_state) = spawn_stub_api().await;
        let staged = stage(base_url);
        let api = ApiClient::new(&staged.settings);
        let job = job_snapshot("yue-complete", full_payload(&staged.project_id));
        let seen = Arc::new(Mutex::new(Seen::default()));

        run_audio_generate_job_using(
            &api,
            &staged.settings,
            &job,
            stub_loader(
                Behavior::Song,
                seen.clone(),
                Arc::new(AtomicBool::new(false)),
            ),
        )
        .await
        .expect("the job completes");

        // The default tier (q4) was installed, so the load came from its subdir asserting Q4.
        {
            let seen = seen.lock().expect("seen lock");
            assert_eq!(
                seen.weights.as_deref(),
                Some(staged.snapshot.join("q4").as_path())
            );
            assert_eq!(seen.quantize, Some(gen_core::Quant::Q4));
        }

        let posts = api_state.progress.lock().expect("progress lock").clone();
        let completed = posts
            .iter()
            .find(|post| post["status"] == "completed")
            .expect("a Completed post");
        let result = &completed["result"];
        assert_eq!(result["expectedCount"], 3);
        assert_eq!(result["generationSet"]["count"], 3);
        let writes = result["assetWrites"].as_array().expect("assetWrites");
        assert_eq!(writes.len(), 3, "mix + vocals + instrumental");
        let mix = &writes[0];
        let mix_id = mix["assetId"].as_str().expect("mix id");
        assert_eq!(mix["extra"]["audioStem"], "mix");
        assert_eq!(mix["rawAdapterSettings"]["quantTier"], "q4");
        assert_eq!(mix["rawAdapterSettings"]["segments"], 3);
        assert_eq!(mix["rawAdapterSettings"]["outputLimiter"], "rescale");
        let mut ids = std::collections::BTreeSet::new();
        let mut stems = Vec::new();
        for write in writes {
            assert_eq!(write["type"], "audio");
            assert!(ids.insert(write["assetId"].as_str().expect("id").to_owned()));
            let rel = write["mediaPath"].as_str().expect("media path");
            let wav = staged.project_path.join(rel);
            let decoded = read_wav_pcm16(&wav).expect("the WAV is on disk and decodes");
            assert_eq!(decoded.sample_rate, 44_100);
            assert_eq!(decoded.samples.len(), 4_410);
            if write["assetId"] != mix_id {
                assert_eq!(
                    write["parents"],
                    json!([mix_id]),
                    "a stem descends from its mix"
                );
                assert_eq!(write["extra"]["mixAssetId"], mix_id);
                stems.push(
                    write["extra"]["audioStem"]
                        .as_str()
                        .expect("stem")
                        .to_owned(),
                );
            }
        }
        assert_eq!(stems, ["vocals", "instrumental"]);

        // Engine progress → job events: model-stage loads, each segment/track step, the decode.
        let running: Vec<(String, String)> = posts
            .iter()
            .filter(|post| post["status"] == "running")
            .map(|post| {
                (
                    post["stage"].as_str().unwrap_or_default().to_owned(),
                    post["message"].as_str().unwrap_or_default().to_owned(),
                )
            })
            .collect();
        assert!(
            running
                .iter()
                .any(|(stage, message)| stage == "loading_model"
                    && message == "Loading model weights."),
            "{running:?}"
        );
        for step in 1..=4 {
            let want = format!("Generating audio ({step}/4).");
            assert!(
                running
                    .iter()
                    .any(|(stage, message)| stage == "generating" && *message == want),
                "missing {want}: {running:?}"
            );
        }
        assert!(running
            .iter()
            .any(|(_, message)| message == "Decoding audio."));
        let fractions: Vec<f64> = posts
            .iter()
            .filter(|post| post["status"] == "running")
            .filter_map(|post| post["progress"].as_f64())
            .collect();
        assert!(
            fractions.windows(2).all(|pair| pair[0] <= pair[1]),
            "progress must not go backwards: {fractions:?}"
        );
    }

    /// AC3 (cancel): a cancel requested while the engine is rendering trips the request's flag,
    /// the job ends Canceled, and nothing is left behind — no generation-set directory in the
    /// project (it is created only at the first write) and no ICL scratch dir.
    #[tokio::test]
    async fn mid_job_cancel_leaves_no_generation_dir_or_scratch() {
        let (base_url, api_state) = spawn_stub_api().await;
        let staged = stage(base_url);
        let api = ApiClient::new(&staged.settings);
        let job_id = "yue-cancel";
        let job = job_snapshot(job_id, full_payload(&staged.project_id));
        let started = Arc::new(AtomicBool::new(false));
        // Flip the API's cancel state once the engine is mid-render.
        {
            let started = started.clone();
            let cancel = api_state.cancel.clone();
            tokio::spawn(async move {
                while !started.load(Ordering::SeqCst) {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                cancel.store(true, Ordering::SeqCst);
            });
        }
        let result = run_audio_generate_job_using(
            &api,
            &staged.settings,
            &job,
            stub_loader(
                Behavior::WaitForCancel,
                Arc::new(Mutex::new(Seen::default())),
                started.clone(),
            ),
        )
        .await;
        assert!(
            matches!(result, Err(WorkerError::Canceled(_))),
            "a mid-render cancel ends the job Canceled, got {result:?}"
        );
        assert!(started.load(Ordering::SeqCst), "the engine was mid-render");
        let audios = staged.project_path.join("assets/audios");
        let leftovers: Vec<_> = std::fs::read_dir(&audios)
            .map(|entries| entries.flatten().map(|entry| entry.path()).collect())
            .unwrap_or_default();
        assert!(
            leftovers.is_empty(),
            "no generation-set dir survives a cancel: {leftovers:?}"
        );
        assert!(icl_scratch_dirs(job_id).is_empty());
        let posts = api_state.progress.lock().expect("progress lock");
        assert!(
            posts.iter().any(|post| post["status"] == "canceled"),
            "{posts:?}"
        );
        assert!(posts.iter().all(|post| post["status"] != "completed"));
    }

    fn icl_scratch_dirs(job_id: &str) -> Vec<PathBuf> {
        let prefix = icl_scratch_prefix(job_id);
        std::fs::read_dir(std::env::temp_dir())
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
                    .map(|entry| entry.path())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The ICL decode's scratch dir is removed on every exit (sc-19384): a refused decode (a source
    /// ffmpeg cannot read, or no ffmpeg at all), a decode whose job is already canceled, and — where
    /// ffmpeg exists — a successful decode, which also lands on xcodec's 16 kHz mono.
    #[tokio::test]
    async fn icl_decode_scratch_is_removed_on_every_exit() {
        let (base_url, api_state) = spawn_stub_api().await;
        let mut settings = Settings::from_env();
        settings.api_url = base_url;
        settings.worker_id = "test-worker".to_owned();
        settings.heartbeat_seconds = 5;
        let api = ApiClient::new(&settings);
        let dir = tempfile::tempdir().expect("source dir");

        let garbage = dir.path().join("not-audio.wav");
        std::fs::write(&garbage, b"this is not audio").expect("garbage writes");
        let job_id = "yue-icl-scratch-refused";
        assert!(decode_icl_clip(&api, &settings, job_id, &garbage)
            .await
            .is_err());
        assert!(
            icl_scratch_dirs(job_id).is_empty(),
            "a refused decode leaves no scratch"
        );

        if !crate::video_jobs::tests::ffmpeg_reachable() {
            eprintln!("icl_decode_scratch_is_removed_on_every_exit: ffmpeg not found, skipping the decode arms");
            return;
        }
        let source = dir.path().join("voice.wav");
        write_wav_pcm16(
            &AudioTrack {
                samples: (0..9_600).flat_map(|_| [0.25f32, -0.25f32]).collect(),
                sample_rate: 48_000,
                channels: 2,
            },
            &source,
        )
        .expect("source writes");
        let job_id = "yue-icl-scratch-ok";
        let track = decode_icl_clip(&api, &settings, job_id, &source)
            .await
            .expect("a real clip decodes");
        assert_eq!(track.sample_rate, ICL_REFERENCE_SAMPLE_RATE);
        assert_eq!(track.channels, ICL_REFERENCE_CHANNELS);
        // The stereo [0.25, -0.25] source downmixes to its channel mean, 0.0 (not one channel).
        assert!(!track.samples.is_empty());
        assert!(
            track.samples.iter().all(|sample| sample.abs() < 1e-3),
            "a stereo reference must downmix to mono (channel mean ≈ 0)"
        );
        assert!(
            icl_scratch_dirs(job_id).is_empty(),
            "a successful decode leaves no scratch"
        );

        // A long source, so the runner's immediate first cancel poll lands while ffmpeg is still
        // decoding rather than racing a decode that already finished.
        let long_source = dir.path().join("long.wav");
        write_wav_pcm16(
            &AudioTrack {
                samples: tone(48_000 * 2 * 20, 440.0, 0.3),
                sample_rate: 48_000,
                channels: 2,
            },
            &long_source,
        )
        .expect("long source writes");
        api_state.cancel.store(true, Ordering::SeqCst);
        let job_id = "yue-icl-scratch-canceled";
        let canceled = decode_icl_clip(&api, &settings, job_id, &long_source).await;
        assert!(
            matches!(canceled, Err(WorkerError::Canceled(_))),
            "a decode whose job is canceled ends Canceled, got {canceled:?}"
        );
        assert!(
            icl_scratch_dirs(job_id).is_empty(),
            "a canceled decode leaves no scratch"
        );
    }

    /// Review fix 2: the progress fold is monotone — `Decoding` (0.85) followed by a `Step { 2, 3 }`
    /// (0.63 on its own) must not move the bar backwards, nor may a later chunk or `Loading`.
    #[test]
    fn progress_fraction_never_goes_backwards() {
        let mut fold = SynthesisProgress::new();
        let mut fractions = vec![fold.engine(Progress::Decoding).expect("decoding").1];
        fractions.push(
            fold.engine(Progress::Step {
                current: 2,
                total: 3,
            })
            .expect("step")
            .1,
        );
        fractions.push(fold.chunk(1).1);
        fractions.push(
            fold.engine(Progress::Loading(gen_core::LoadPhase::Renderer))
                .expect("loading")
                .1,
        );
        assert_eq!(fractions[0], 0.85);
        assert!(
            fractions.windows(2).all(|pair| pair[0] <= pair[1]),
            "progress must not go backwards: {fractions:?}"
        );
    }

    fn pump_test_settings(base_url: String) -> Settings {
        let mut settings = Settings::from_env();
        settings.api_url = base_url;
        settings.worker_id = "test-worker".to_owned();
        settings.heartbeat_seconds = 5;
        settings
    }

    fn running_posts(state: &StubApi) -> Vec<Value> {
        state
            .progress
            .lock()
            .expect("progress lock")
            .iter()
            .filter(|post| post["status"] == "running")
            .cloned()
            .collect()
    }

    /// Review fix 1: a per-token engine (5,000 `Step`s) costs a coalesced handful of progress POSTs —
    /// at most one per `PROGRESS_POST_INTERVAL` plus the final state — never one POST per event, and
    /// the final state is always posted.
    #[tokio::test]
    async fn per_token_progress_is_coalesced_to_the_post_interval() {
        let (base_url, api_state) = spawn_stub_api().await;
        let settings = pump_test_settings(base_url);
        let api = ApiClient::new(&settings);
        let payload = full_payload("project-1");
        let job = job_snapshot("yue-coalesce", payload.clone());
        let request = AudioRequest::from_payload(payload.as_object().expect("object"));
        let started = Instant::now();
        run_audio_synthesis_with(
            &api,
            &settings,
            &job,
            &request,
            SinglePlan {
                model_dir: PathBuf::from("/staged/yue"),
                tier: None,
                conditioning: None,
            },
            stub_loader(
                Behavior::ManySteps(5_000),
                Arc::new(Mutex::new(Seen::default())),
                Arc::new(AtomicBool::new(false)),
            ),
        )
        .await
        .expect("synthesis completes");
        let elapsed = started.elapsed();
        let posts = running_posts(&api_state);
        let budget = (elapsed.as_millis() / PROGRESS_POST_INTERVAL.as_millis()) as usize + 3;
        assert!(
            posts.len() <= budget,
            "{} running posts for 5,000 steps in {elapsed:?} (budget {budget})",
            posts.len()
        );
        assert_eq!(
            posts.last().and_then(|post| post["message"].as_str()),
            Some("Generating audio (5000/5000)."),
            "the final state is always posted"
        );
    }

    /// Review fix 2, end to end: a model that reports `Decoding` then a `Step` posts non-decreasing
    /// fractions through the real pump.
    #[tokio::test]
    async fn interleaved_decode_and_step_posts_never_go_backwards() {
        let (base_url, api_state) = spawn_stub_api().await;
        let settings = pump_test_settings(base_url);
        let api = ApiClient::new(&settings);
        let payload = full_payload("project-1");
        let job = job_snapshot("yue-monotone", payload.clone());
        let request = AudioRequest::from_payload(payload.as_object().expect("object"));
        run_audio_synthesis_with(
            &api,
            &settings,
            &job,
            &request,
            SinglePlan {
                model_dir: PathBuf::from("/staged/yue"),
                tier: None,
                conditioning: None,
            },
            stub_loader(
                Behavior::DecodeThenStep,
                Arc::new(Mutex::new(Seen::default())),
                Arc::new(AtomicBool::new(false)),
            ),
        )
        .await
        .expect("synthesis completes");
        let fractions: Vec<f64> = running_posts(&api_state)
            .iter()
            .filter_map(|post| post["progress"].as_f64())
            .collect();
        assert!(!fractions.is_empty());
        assert!(
            fractions.iter().all(|fraction| *fraction >= 0.85),
            "no post may fall below the 0.85 Decoding mark once reached: {fractions:?}"
        );
    }

    /// The shipped manifest entry for `model_id`.
    fn builtin_entry(model_id: &str) -> Value {
        let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, contents)| *contents)
            .expect("builtin.models.jsonc present");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin.models.jsonc parses");
        manifest["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
            .cloned()
            .unwrap_or_else(|| panic!("builtin entry {model_id} present"))
    }

    /// Review fix 3: the resolved tier reaches the co-requisite resolver. With the REAL `yue_en_cot`
    /// descriptor (which requires `stage2` + `xcodec`) and its REAL manifest rows (three per-tier
    /// `stage2` rows), a q8 job stages the q8 stage-2 subdir — resolving without the tier would be
    /// refused ("no tier was resolved"), and would fail every real YuE job.
    #[tokio::test]
    async fn resolved_tier_selects_the_matching_stage2_component() {
        let descriptor = crate::inference_runtime::audio_descriptor("yue_en_cot")
            .expect("the linked audio registry serves yue_en_cot (run at the YuE inference pin)");
        assert!(descriptor.required_components.contains(&"stage2"));
        let (base_url, _api_state) = spawn_stub_api().await;
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ]);
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let entry = builtin_entry("yue_en_cot");
        let mut stage2_q8 = None;
        for download in entry["downloads"].as_array().expect("downloads") {
            if download.get("coRequisite").and_then(Value::as_bool) != Some(true) {
                continue;
            }
            let is_stage2 = download["componentId"] == "stage2";
            if is_stage2 && download["variant"] != "q8" {
                continue;
            }
            let repo = download["repo"].as_str().expect("repo");
            let revision = download["revision"].as_str().expect("revision");
            let snapshot =
                sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir.path(), repo)
                    .expect("repo cache path resolves")
                    .join("snapshots")
                    .join(revision);
            for file in download["files"].as_array().expect("files") {
                let file = file
                    .as_str()
                    .expect("file")
                    .replace('*', "weights.safetensors");
                let path = snapshot.join(&file);
                std::fs::create_dir_all(path.parent().expect("parent")).expect("dir");
                std::fs::write(&path, b"weights").expect("file");
            }
            if is_stage2 {
                stage2_q8 = Some(snapshot.join(download["subdir"].as_str().expect("subdir")));
            }
        }
        let stage2_q8 = stage2_q8.expect("the manifest declares a q8 stage2 row");

        let mut settings = pump_test_settings(base_url);
        settings.data_dir = data_dir.path().to_path_buf();
        let api = ApiClient::new(&settings);
        let payload = json!({
            "projectId": "project-1",
            "model": "yue_en_cot",
            "prompt": "pop",
            "lyrics": LYRICS,
            "quantTier": "q8",
            "modelManifestEntry": entry,
        });
        let job = job_snapshot("yue-stage2-tier", payload.clone());
        let request = AudioRequest::from_payload(payload.as_object().expect("object"));
        let seen = Arc::new(Mutex::new(Seen::default()));
        run_audio_synthesis_with(
            &api,
            &settings,
            &job,
            &request,
            SinglePlan {
                model_dir: PathBuf::from("/staged/yue-s1/q8"),
                tier: Some(AudioTier {
                    name: "q8".to_owned(),
                    quantize: Some(gen_core::Quant::Q8),
                }),
                conditioning: None,
            },
            stub_loader(
                Behavior::Song,
                seen.clone(),
                Arc::new(AtomicBool::new(false)),
            ),
        )
        .await
        .expect("synthesis completes with the q8 components staged");
        let seen = seen.lock().expect("seen lock");
        assert_eq!(
            seen.components.get("stage2"),
            Some(&WeightsSource::Dir(stage2_q8)),
            "the q8 job stages the q8 stage-2 subdir"
        );
        assert!(seen.components.contains_key("xcodec"));
    }
}
