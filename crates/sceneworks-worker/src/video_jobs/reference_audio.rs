#[allow(unused_imports)]
use super::prelude::*;

// ---------------------------------------------------------------------------
// Standalone audio references (`referenceAudioAssetIds`, sc-17160 / sc-18650).
//
// Cross-family and cross-platform: this is the shared resolver every video model's audio
// references go through, not an engine family, which is why it is ungated and lives beside the
// families rather than inside one. MiniMax-H3 Ref2VA is the only consumer today (every other
// model's `limits.maxReferenceAudioAssets` defaults to 0, so `video_preflight` refuses a non-empty
// list before this is reached), but nothing here is H3-specific except the engine's declared rate.
// ---------------------------------------------------------------------------

/// The rate every standalone audio reference reaches an engine at — MiniMax-H3's audio VAE rate
/// (`mlx-gen-minimax-h3::audio_config::AUDIO_SAMPLE_RATE`).
///
/// **This is a hard engine boundary, not a preference.** `Ref2VaReferences::check_audio` rejects any
/// reference soundtrack whose `sample_rate` is not exactly this number, with the message "this
/// engine has no resampler — resample it before sending it". This crate is the caller it is talking
/// to, so [`resolve_reference_audio_conditioning`] resamples here.
///
/// Ungated because that resolver is: [`super::minimax_h3::MINIMAX_H3_AUDIO_SAMPLE_RATE`] is an alias of it
/// so a video reference's own soundtrack and a standalone audio reference are normalized onto ONE
/// number rather than two literals that happen to agree today.
pub(super) const REFERENCE_AUDIO_SAMPLE_RATE: u32 = 32_000;

/// Resolve a video request's `referenceAudioAssetIds` (sc-17160) into the engine conditioning:
/// one [`gen_core::Conditioning::ReferenceAudio`] per id, in submission order.
///
/// The audio sibling of [`super::bernini::resolve_bernini_conditioning`]'s reference-image handling, and
/// deliberately built on the SAME primitives the rest of the crate uses
/// ([`super::ltx::resolve_clip_media_path`] + an ffmpeg normalization + [`crate::audio_jobs::read_wav_pcm16`])
/// rather than a second implementation of any of them.
///
/// **The path handling is the load-bearing part.** An asset id is not a filename: the id resolves
/// through [`ProjectStore::get_asset`] to a PROJECT-RELATIVE `file.path` in the sidecar, which is
/// then joined under the project root by `safe_project_path`. Reading the id as a bare name — or
/// joining the sidecar path without the guard — is the sc-4278 / F-MLXW-14 escape, and the
/// training-preview class of bug where a relative path was assumed absolute. Going through
/// `resolve_clip_media_path` inherits both the lookup and the guard.
///
/// # Why every reference goes through ffmpeg (sc-18650)
///
/// Handing the asset's own WAV header straight to the engine made this capability UNREACHABLE:
/// `Ref2VaReferences::check_audio` hard-errors on any rate but
/// [`REFERENCE_AUDIO_SAMPLE_RATE`], and no shipped SceneWorks producer emits that rate — every TTS
/// and music entry in `builtin.models.jsonc` declares `audio.sampleRates` of 22050 / 24000 / 48000.
/// So `referenceAudioAssetIds` had zero successful paths: a caller could name a reference, pass
/// preflight, and only learn at the engine boundary that no asset in their library was admissible.
/// The reference-CLIP path already honoured the same contract with an `-ar` pass
/// ([`super::minimax_h3::MINIMAX_H3_AUDIO_SAMPLE_RATE`]); this is that same pass, for the standalone half.
///
/// It runs UNCONDITIONALLY rather than behind a "is it already 32 kHz?" probe. One code path means
/// the conversion is exercised by every input rather than by whichever inputs happen to be
/// off-rate, and it is what lets a reference be any container ffmpeg reads (mp3 / m4a / flac /
/// non-PCM WAV) instead of only the canonical PCM-16 RIFF `read_wav_pcm16` decodes.
///
/// The CHANNEL layout is deliberately left alone (no `-ac`). The engine accepts any positive channel
/// count, so channels are not a contract the caller can violate — and both directions of "fixing"
/// them substitute a different reference than the one supplied: upmixing a mono voice clip doubles
/// its packed row cost for a duplicated channel, downmixing a stereo one discards its stereo image.
/// This differs from the clip path's `-ac 2` on purpose: there the soundtrack is a byproduct of a
/// VIDEO reference and stereo is the joint model's own emitted shape, while here the waveform IS the
/// reference.
///
/// Async since sc-18650, and it takes the `api`/`job` the shared
/// [`crate::media_jobs::run_ffmpeg`] runner needs for its heartbeat + cooperative-cancel loop —
/// exactly how [`super::minimax_h3::resolve_minimax_h3_clip_conditioning`] drives the same runner. A
/// private `Command` would give this one call site a different lifecycle from every other media
/// decode in the crate.
///
/// Ungated (compiled in every feature/target config) for the same reason
/// `resolve_clip_media_path` is: the body depends only on cross-platform helpers, and the
/// preflight caller below is itself cross-platform.
pub(crate) async fn resolve_reference_audio_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<gen_core::Conditioning>> {
    let mut conditioning = Vec::with_capacity(request.reference_audio_asset_ids.len());
    for (index, asset_id) in request.reference_audio_asset_ids.iter().enumerate() {
        let asset_id = asset_id.trim();
        if asset_id.is_empty() {
            return Err(WorkerError::InvalidPayload(
                "referenceAudioAssetIds must not contain blank ids".to_owned(),
            ));
        }
        let path = super::ltx::resolve_clip_media_path(
            settings,
            &request.project_id,
            asset_id,
            project_path,
        )?;
        // Sanitize the job id before it becomes a path component (F-111), and index the directory
        // so three concurrent references in one job never share one — the same shape the reference
        // clip scratch dirs use.
        let work_dir = std::env::temp_dir().join(format!(
            "sw-reference-audio-{}-{index}",
            safe_download_dir(&job.id)
        ));
        let _ = tokio::fs::remove_dir_all(&work_dir).await;
        tokio::fs::create_dir_all(&work_dir).await?;
        // Split so the scratch directory is dropped on EVERY exit, refusals included.
        let decoded = decode_reference_audio(api, settings, job, &path, &work_dir).await;
        let _ = tokio::fs::remove_dir_all(&work_dir).await;
        conditioning.push(gen_core::Conditioning::ReferenceAudio {
            audio: decoded?,
            // No per-reference strength knob today: the request carries one flat list, and
            // inventing a weight the caller cannot set would be a knob that silently does
            // nothing. `None` is what the voice-clone reference passes for the same reason.
            strength: None,
        });
    }
    Ok(conditioning)
}

/// Decode one reference asset into the engine's [`gen_core::AudioTrack`], normalized onto
/// [`REFERENCE_AUDIO_SAMPLE_RATE`].
///
/// `-map 0:a:0 -vn` takes the first audio stream and nothing else, so an asset with no audio stream
/// is a refusal rather than an empty track, and `-c:a pcm_s16le` writes the one WAV encoding
/// [`crate::audio_jobs::read_wav_pcm16`] decodes. Errors propagate exactly as the clip path's do —
/// ffmpeg's own bounded stderr tail, which is the only place the real reason ("Invalid data found
/// when processing input", "Stream map '0:a:0' matches no streams") is written down.
async fn decode_reference_audio(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    source: &Path,
    work_dir: &Path,
) -> WorkerResult<gen_core::AudioTrack> {
    let wav = work_dir.join("reference.wav");
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            source.display().to_string(),
            "-map".to_owned(),
            "0:a:0".to_owned(),
            "-vn".to_owned(),
            // The engine ships no resampler and refuses anything but its audio VAE's rate.
            "-ar".to_owned(),
            REFERENCE_AUDIO_SAMPLE_RATE.to_string(),
            // `read_wav_pcm16` reads PCM s16 only.
            "-c:a".to_owned(),
            "pcm_s16le".to_owned(),
            wav.display().to_string(),
        ],
        Some(ctx),
    )
    .await?;
    crate::audio_jobs::read_wav_pcm16(&wav)
}
