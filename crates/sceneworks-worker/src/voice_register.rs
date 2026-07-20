//! The **embed path** for the Voice Clone "register a voice" flow (sc-13517).
//!
//! A single blocking entry point, [`embed_reference_clip`], that the rust-api calls when a user
//! registers a saved voice: it resolves the Chatterbox-VE speaker-encoder weights from the local HF
//! cache, decodes the chosen reference clip to PCM-16, and runs the embedder
//! ([`crate::inference_runtime::load_voice_embedder`]) to produce the raw 256-d speaker vector. That
//! vector is stored as the voice's identity and consumed by the registry's near-duplicate detection
//! (`sceneworks_core::voice_store`) — it is NOT fed into the clone Generator (VoiceEmbedding-only
//! errors at S3Gen; both clone paths drive off the reference AUDIO clip, sc-13412).
//!
//! Lives in the worker crate because that is where the inference runtime (candle audio lane) is
//! composed. The rust-api links this crate and invokes the function on a blocking thread; on a build
//! with no audio lane (non-macOS without `backend-candle`) it returns [`VoiceEmbedError::Embed`]
//! rather than silently succeeding.

use std::fmt;
use std::path::{Path, PathBuf};

use gen_core::{LoadSpec, WeightsSource};

/// The Hugging Face repo hosting the Chatterbox voice encoder weights (shared with `chatterbox_tts`).
pub const CHATTERBOX_VE_REPO: &str = "ResembleAI/chatterbox";
/// The single-file weights the Chatterbox-VE embedder loads (`WeightsSource::File`, not a snapshot dir).
pub const CHATTERBOX_VE_WEIGHTS_FILE: &str = "ve.safetensors";
/// The registry id of the Chatterbox voice embedder.
pub const CHATTERBOX_VE_MODEL_ID: &str = "chatterbox_ve";

/// Why a reference clip couldn't be embedded. The rust-api maps these to HTTP status codes: a missing
/// install is a client-actionable 400 ("install the encoder"), a bad clip is a 400, and an embedder
/// runtime failure is a 500.
#[derive(Debug)]
pub enum VoiceEmbedError {
    /// The Chatterbox-VE weights are not present in the local model cache.
    WeightsMissing(String),
    /// The reference clip couldn't be decoded to a PCM-16 WAV [`gen_core::AudioTrack`].
    Decode(String),
    /// The embedder failed to load or run (including "no audio lane in this build").
    Embed(String),
}

impl fmt::Display for VoiceEmbedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightsMissing(message) => write!(formatter, "{message}"),
            Self::Decode(message) => write!(formatter, "{message}"),
            Self::Embed(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for VoiceEmbedError {}

/// Resolve the Chatterbox-VE `ve.safetensors` weights file inside the local HF hub cache under
/// `data_dir`. The file rides in the shared `ResembleAI/chatterbox` snapshot (a co-requisite of the
/// clone TTS install, sc-13541). Returns [`VoiceEmbedError::WeightsMissing`] when the snapshot or the
/// file is absent, with an actionable message.
pub fn resolve_chatterbox_ve_weights(data_dir: &Path) -> Result<PathBuf, VoiceEmbedError> {
    let snapshot = crate::model_jobs::huggingface_snapshot_dir(data_dir, CHATTERBOX_VE_REPO)
        .ok_or_else(|| {
            VoiceEmbedError::WeightsMissing(format!(
                "the Chatterbox voice encoder ({CHATTERBOX_VE_MODEL_ID}) is not installed — install \
                 it from the Model Manager before registering a voice."
            ))
        })?;
    let weights = snapshot.join(CHATTERBOX_VE_WEIGHTS_FILE);
    if !weights.is_file() {
        return Err(VoiceEmbedError::WeightsMissing(format!(
            "the Chatterbox voice encoder weights ({CHATTERBOX_VE_WEIGHTS_FILE}) were not found in \
             the cached {CHATTERBOX_VE_REPO} snapshot — reinstall the voice encoder."
        )));
    }
    Ok(weights)
}

/// Compute the raw 256-d Chatterbox-VE speaker embedding for the reference clip at `wav_path`.
///
/// `data_dir` is the app data dir (the HF cache root is derived from it, exactly as the worker's own
/// model loading does), and `wav_path` is the already-resolved absolute path to a PCM-16 WAV library
/// asset. Blocking (mmaps weights, runs the GE2E encoder) — call it on a blocking thread.
pub fn embed_reference_clip(data_dir: &Path, wav_path: &Path) -> Result<Vec<f32>, VoiceEmbedError> {
    let weights = resolve_chatterbox_ve_weights(data_dir)?;
    let track = crate::audio_jobs::read_wav_pcm16(wav_path)
        .map_err(|error| VoiceEmbedError::Decode(error.to_string()))?;
    let embedder = crate::inference_runtime::load_voice_embedder(
        CHATTERBOX_VE_MODEL_ID,
        &LoadSpec::new(WeightsSource::File(weights)),
    )
    .map_err(|error| VoiceEmbedError::Embed(error.to_string()))?;
    let embedding = embedder
        .embed(&track)
        .map_err(|error| VoiceEmbedError::Embed(error.to_string()))?;
    Ok(embedding)
}
