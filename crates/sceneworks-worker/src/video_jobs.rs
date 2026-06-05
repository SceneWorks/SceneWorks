//! Native video-generation jobs — runtime pipeline + procedural stub (epic 3018, sc-3033).
//!
//! Parses the job into a [`VideoRequest`], produces a single video (one mp4 asset,
//! unlike images which batch `count`), and reports a flat "fact" the Rust API turns
//! into an indexed asset (mirroring `video_generation_result` in the Python worker's
//! `video_adapters.py`). The shared encode pipeline takes the engine's video output
//! shape — RGB8 `frames` + `fps` + an optional synchronized `audio` track — writes
//! the frames to an mp4 (libx264), muxes a 16-bit-PCM WAV as AAC when audio is present
//! (`-shortest`), remuxes `+faststart` (WKWebView range-seek), and extracts a poster
//! frame. It reuses [`crate::media_jobs::run_ffmpeg`] (binary resolution + the
//! periodic-heartbeat / cooperative-cancel loop).
//!
//! sc-3033 ships only the **procedural stub** generator (a moving gradient + a quiet
//! synchronized tone for the LTX family, mirroring the engine: LTX emits audio, Wan
//! does not). The real in-process MLX video models — Wan2.2 (sc-3034) and LTX-2.3 +
//! audio (sc-3035) — link the `mlx-gen-wan` / `mlx-gen-ltx` provider crates and decode
//! `mlx_gen::GenerationOutput::Video { frames, fps, audio }` into the same
//! [`DecodedVideo`], so the encode/mux/poster path below is unchanged for them.

use std::f32::consts::PI;
use std::path::Path;

use sceneworks_core::video_request::{is_ltx_model, VideoRequest};

use super::*;
use crate::media_jobs::{run_ffmpeg, FfmpegContext};

/// Stub adapter id recorded on generated assets — matches the Python
/// `ProceduralVideoAdapter.id` so the asset sidecar reads identically.
const STUB_ADAPTER: &str = "procedural_video";
const CANCEL_MESSAGE: &str = "Video generation canceled by user.";

/// Decoded video ready for muxing — the worker-side shape both the procedural stub
/// (sc-3033) and the real engine output (`mlx_gen::GenerationOutput::Video`,
/// sc-3034/3035) feed into [`encode_media`]. Mirrors the engine contract: `frames`
/// are RGB8, `audio` is `Some` for LTX (a synchronized track) and `None` for Wan.
/// The frames are held in memory (the engine returns them that way); the duration
/// clamp (≤30s) in [`VideoRequest`] bounds the footprint.
struct DecodedVideo {
    frames: Vec<RgbFrame>,
    fps: u32,
    audio: Option<AudioTrack>,
}

/// One RGB8 frame, row-major, `pixels.len() == width * height * 3` (the engine's
/// `Image`).
struct RgbFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Interleaved PCM audio — the engine's `AudioTrack` (LTX-2.3 synchronized audio).
struct AudioTrack {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

/// Dispatch handler for `JobType::VideoGenerate`: generate, encode, and stream a
/// single video asset through the Rust GPU worker.
pub(crate) async fn run_video_generate_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = VideoRequest::from_payload(&job.payload);
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
    }
    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let plan = VideoPlan::new(&request, &project_path);
    if let Some(parent) = plan.media_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let backend = backend_label(&settings.gpu_id);
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Preparing video.",
            None,
            backend,
        ),
    )
    .await?;

    // sc-3033 ships the procedural stub only; the real MLX video models (Wan sc-3034,
    // LTX+audio sc-3035) decode `GenerationOutput::Video` into a `DecodedVideo` here.
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    let seed = resolve_video_seed(&request);
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.2,
            "Rendering frames.",
            None,
            backend,
        ),
    )
    .await?;
    let decoded = generate_stub_video(&request, seed);
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;

    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Muxing,
            0.6,
            "Encoding video.",
            None,
            backend,
        ),
    )
    .await?;
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    encode_media(&plan.media_path, decoded, Some(ctx)).await?;

    let fact = video_asset_fact(&plan, seed);
    let result = streaming_result(&plan, &fact);
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Generated video.",
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// Per-job invariants for the single video this job produces.
struct VideoPlan {
    request: VideoRequest,
    genset_id: String,
    asset_id: String,
    created_at: String,
    family: String,
    /// `assets/videos/<genset>/<date>_<model>_<slug>.mp4` (project-relative).
    media_rel: String,
    /// Absolute path to the media file.
    media_path: PathBuf,
}

impl VideoPlan {
    fn new(request: &VideoRequest, project_path: &Path) -> Self {
        let genset_id = format!("genset_{}", Uuid::new_v4().simple());
        let asset_id = fresh_asset_id();
        let created_at = now_rfc3339();
        let family = resolve_family(request);
        let slug = slugify(&request.prompt, "video", Some(42));
        // Nest under the per-generation id so two renders sharing date+model+slug
        // cannot collide on a flat path (mirrors the image + Python video adapters).
        let media_rel = format!(
            "assets/videos/{genset_id}/{}_{}_{slug}.mp4",
            &created_at[..10],
            request.model
        );
        let media_path = project_path.join(&media_rel);
        Self {
            request: request.clone(),
            genset_id,
            asset_id,
            created_at,
            family,
            media_rel,
            media_path,
        }
    }
}

/// Resolve the seed, matching the Python `resolve_seed(seed, prompt)`: an explicit
/// seed wins, else the first 4 bytes of `sha256(prompt)` (its `hexdigest()[:8]`).
fn resolve_video_seed(request: &VideoRequest) -> i64 {
    if let Some(seed) = request.seed {
        return seed;
    }
    let digest = Sha256::digest(request.prompt.as_bytes());
    u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as i64
}

/// The asset's video family, from the resolved manifest entry when present, else
/// inferred from the model id (parity with the Python `VIDEO_MODEL_TARGETS` family).
fn resolve_family(request: &VideoRequest) -> String {
    if let Some(family) = request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
    {
        if !family.trim().is_empty() {
            return family.to_owned();
        }
    }
    if is_ltx_model(&request.model) {
        "ltx-video".to_owned()
    } else if request.model.starts_with("wan") {
        "wan-video".to_owned()
    } else {
        "video".to_owned()
    }
}

// ---------------------------------------------------------------------------
// Procedural stub generator (sc-3033). Real MLX models land in sc-3034/3035.
// ---------------------------------------------------------------------------

/// Build a deterministic placeholder clip: `frame_count` moving-gradient frames at
/// the request fps, plus a quiet synchronized tone for the LTX family (the engine
/// emits audio for LTX and none for Wan, so the stub mirrors that split — exercising
/// both the audio-mux and video-only encode paths).
fn generate_stub_video(request: &VideoRequest, seed: i64) -> DecodedVideo {
    let frame_count = request.frame_count();
    let fps = request.fps.max(1);
    let (width, height) = (request.width, request.height);
    let frames = (0..frame_count)
        .map(|index| RgbFrame {
            width,
            height,
            pixels: stub_video_rgb8(width, height, seed, index, frame_count),
        })
        .collect();
    let audio = is_ltx_model(&request.model).then(|| stub_audio_track(frame_count, fps));
    DecodedVideo { frames, fps, audio }
}

/// Deterministic per-frame pixels: a vertical gradient from a per-seed base colour to
/// white, with a bright vertical band that sweeps left→right across the clip so frames
/// differ (visible motion). Exactly `width * height * 3` RGB8 bytes.
fn stub_video_rgb8(width: u32, height: u32, seed: i64, index: u32, frame_count: u32) -> Vec<u8> {
    let seed = seed as u64;
    let base = [
        (seed & 0xFF) as u8,
        ((seed >> 8) & 0xFF) as u8,
        ((seed >> 16) & 0xFF) as u8,
    ];
    let v_span = height.saturating_sub(1).max(1) as f32;
    // The sweeping band's centre column for this frame.
    let progress = index as f32 / frame_count.max(1) as f32;
    let band_centre = progress * width.saturating_sub(1).max(1) as f32;
    let band_half = (width as f32 * 0.06).max(1.0);
    let mut buffer = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for y in 0..height {
        let t = y as f32 / v_span;
        let row = [lerp(base[0], t), lerp(base[1], t), lerp(base[2], t)];
        for x in 0..width {
            let dist = (x as f32 - band_centre).abs();
            if dist <= band_half {
                // Brighten toward white inside the band (1.0 at centre → 0 at edge).
                let highlight = 1.0 - dist / band_half;
                buffer.push(lerp(row[0], highlight));
                buffer.push(lerp(row[1], highlight));
                buffer.push(lerp(row[2], highlight));
            } else {
                buffer.extend_from_slice(&row);
            }
        }
    }
    buffer
}

fn lerp(a: u8, t: f32) -> u8 {
    let a = a as f32;
    (a + (255.0 - a) * t).round().clamp(0.0, 255.0) as u8
}

/// A quiet 220 Hz mono tone matching the clip length (`frame_count / fps` seconds) at
/// 48 kHz — enough to exercise the WAV-write + AAC-mux + `-shortest` path end to end.
fn stub_audio_track(frame_count: u32, fps: u32) -> AudioTrack {
    let sample_rate = 48_000u32;
    let duration = frame_count as f32 / fps.max(1) as f32;
    let n = (sample_rate as f32 * duration).round().max(1.0) as usize;
    let freq = 220.0f32;
    let samples = (0..n)
        .map(|i| (2.0 * PI * freq * (i as f32 / sample_rate as f32)).sin() * 0.2)
        .collect();
    AudioTrack {
        samples,
        sample_rate,
        channels: 1,
    }
}

// ---------------------------------------------------------------------------
// Encode pipeline: frames → mp4 (+ optional AAC audio) → faststart → poster.
// Reuses `media_jobs::run_ffmpeg`. Pure of the API except the optional `ctx`
// (heartbeat/cancel), so it is exercisable in tests with a real ffmpeg.
// ---------------------------------------------------------------------------

/// Write `decoded` to `media_path` as an mp4: frames → libx264, an optional 16-bit
/// PCM WAV muxed as AAC (`-shortest`), then a best-effort `+faststart` remux and
/// `.poster.jpg`. `media_path` is created (atomically renamed from a temp) only on
/// success; all intermediates are removed regardless of outcome.
async fn encode_media(
    media_path: &Path,
    decoded: DecodedVideo,
    ctx: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let frames_dir = media_path.with_extension("frames");
    let enc_tmp = media_path.with_extension("enc.mp4");
    let wav_tmp = media_path.with_extension("audio.wav");
    let mux_tmp = media_path.with_extension("mux.mp4");
    let result = encode_inner(
        media_path,
        decoded,
        ctx,
        &frames_dir,
        &enc_tmp,
        &wav_tmp,
        &mux_tmp,
    )
    .await;
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    let _ = tokio::fs::remove_file(&enc_tmp).await;
    let _ = tokio::fs::remove_file(&wav_tmp).await;
    let _ = tokio::fs::remove_file(&mux_tmp).await;
    if result.is_err() {
        // A failure (or cancel) before the atomic rename leaves no media_path; if the
        // rename itself half-completed, drop the partial so the asset never points at it.
        let _ = tokio::fs::remove_file(media_path).await;
    }
    result
}

#[allow(clippy::too_many_arguments)]
async fn encode_inner(
    media_path: &Path,
    decoded: DecodedVideo,
    ctx: Option<FfmpegContext<'_>>,
    frames_dir: &Path,
    enc_tmp: &Path,
    wav_tmp: &Path,
    mux_tmp: &Path,
) -> WorkerResult<()> {
    let fps = decoded.fps.max(1);
    let audio = decoded.audio;
    let frames = decoded.frames;
    if frames.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "video generation produced no frames".to_owned(),
        ));
    }

    // 1. Write the frame sequence (blocking PNG encodes off the async runtime).
    tokio::fs::create_dir_all(frames_dir).await?;
    let dir = frames_dir.to_path_buf();
    tokio::task::spawn_blocking(move || -> WorkerResult<()> {
        for (index, frame) in frames.into_iter().enumerate() {
            let RgbFrame {
                width,
                height,
                pixels,
            } = frame;
            let image = image::RgbImage::from_raw(width, height, pixels).ok_or_else(|| {
                WorkerError::InvalidPayload("video frame buffer size mismatch".to_owned())
            })?;
            let path = dir.join(format!("frame_{index:05}.png"));
            image
                .save_with_format(&path, image::ImageFormat::Png)
                .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
        }
        Ok(())
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;

    // 2. Frames → mp4 (libx264, yuv420p — request dims are multiples of 32, so even).
    let pattern = frames_dir.join("frame_%05d.png");
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-framerate".to_owned(),
            fps.to_string(),
            "-start_number".to_owned(),
            "0".to_owned(),
            "-i".to_owned(),
            pattern.to_string_lossy().into_owned(),
            "-c:v".to_owned(),
            "libx264".to_owned(),
            "-pix_fmt".to_owned(),
            "yuv420p".to_owned(),
            "-r".to_owned(),
            fps.to_string(),
            enc_tmp.to_string_lossy().into_owned(),
        ],
        ctx,
    )
    .await?;

    // 3. Mux the audio track (LTX) as AAC, else the video-only mp4 is the result.
    let finished_tmp = if let Some(audio) = audio {
        write_wav_pcm16(&audio, wav_tmp)?;
        run_ffmpeg(
            vec![
                "ffmpeg".to_owned(),
                "-nostdin".to_owned(),
                "-y".to_owned(),
                "-i".to_owned(),
                enc_tmp.to_string_lossy().into_owned(),
                "-i".to_owned(),
                wav_tmp.to_string_lossy().into_owned(),
                "-c:v".to_owned(),
                "copy".to_owned(),
                "-c:a".to_owned(),
                "aac".to_owned(),
                "-shortest".to_owned(),
                mux_tmp.to_string_lossy().into_owned(),
            ],
            ctx,
        )
        .await?;
        mux_tmp
    } else {
        enc_tmp
    };

    // 4. Publish atomically, then best-effort faststart + poster (mirrors Python).
    tokio::fs::rename(finished_tmp, media_path).await?;
    faststart_mp4(media_path).await;
    write_poster_frame(media_path).await;
    Ok(())
}

/// Peak-normalize the f32 PCM to 16-bit and write a canonical WAV. Silence (peak 0)
/// stays silent rather than dividing by zero.
fn write_wav_pcm16(audio: &AudioTrack, path: &Path) -> WorkerResult<()> {
    let peak = audio
        .samples
        .iter()
        .fold(0.0f32, |max, &sample| max.max(sample.abs()));
    let scale = if peak > 0.0 {
        i16::MAX as f32 / peak
    } else {
        0.0
    };
    let mut pcm = Vec::with_capacity(audio.samples.len() * 2);
    for &sample in &audio.samples {
        let value = (sample * scale)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        pcm.extend_from_slice(&value.to_le_bytes());
    }

    let channels = audio.channels.max(1);
    let bits_per_sample = 16u16;
    let block_align = channels * bits_per_sample / 8;
    let byte_rate = audio.sample_rate * block_align as u32;
    let data_len = pcm.len() as u32;

    let mut buffer = Vec::with_capacity(44 + pcm.len());
    buffer.extend_from_slice(b"RIFF");
    buffer.extend_from_slice(&(36 + data_len).to_le_bytes());
    buffer.extend_from_slice(b"WAVE");
    buffer.extend_from_slice(b"fmt ");
    buffer.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buffer.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    buffer.extend_from_slice(&channels.to_le_bytes());
    buffer.extend_from_slice(&audio.sample_rate.to_le_bytes());
    buffer.extend_from_slice(&byte_rate.to_le_bytes());
    buffer.extend_from_slice(&block_align.to_le_bytes());
    buffer.extend_from_slice(&bits_per_sample.to_le_bytes());
    buffer.extend_from_slice(b"data");
    buffer.extend_from_slice(&data_len.to_le_bytes());
    buffer.extend_from_slice(&pcm);
    std::fs::write(path, buffer)?;
    Ok(())
}

/// Best-effort `+faststart` remux (moov atom to the front so WKWebView can start
/// playback without a tail byte-range seek). A missing/failing ffmpeg leaves the
/// original untouched — the API's byte-range support is the load-bearing guarantee.
async fn faststart_mp4(media_path: &Path) {
    if !media_path.exists() {
        return;
    }
    let remuxed = media_path.with_extension("faststart.mp4");
    let ok = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.to_string_lossy().into_owned(),
            "-c".to_owned(),
            "copy".to_owned(),
            "-movflags".to_owned(),
            "+faststart".to_owned(),
            remuxed.to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .is_ok();
    if ok {
        let _ = tokio::fs::rename(&remuxed, media_path).await;
    } else {
        let _ = tokio::fs::remove_file(&remuxed).await;
    }
}

/// Best-effort poster extraction to `<name>.poster.jpg` (WKWebView does not paint a
/// `<video>`'s first frame on its own). A missing/failing ffmpeg leaves no poster.
async fn write_poster_frame(media_path: &Path) {
    if !media_path.exists() {
        return;
    }
    let poster = media_path.with_extension("poster.jpg");
    let ok = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.to_string_lossy().into_owned(),
            "-frames:v".to_owned(),
            "1".to_owned(),
            "-q:v".to_owned(),
            "3".to_owned(),
            poster.to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .is_ok();
    if !ok {
        let _ = tokio::fs::remove_file(&poster).await;
    }
}

// ---------------------------------------------------------------------------
// Asset fact + streaming result (mirrors `video_generation_result`).
// ---------------------------------------------------------------------------

/// The flat per-asset fact the Rust API turns into an indexed video asset (every key
/// is consumed by the API's video sidecar builder). Mirrors `video_generation_result`.
fn video_asset_fact(plan: &VideoPlan, seed: i64) -> Value {
    let request = &plan.request;
    let title: String = request.prompt.chars().take(56).collect();
    let title = title.trim();
    let display_name = if title.is_empty() {
        "Generated video".to_owned()
    } else {
        title.to_owned()
    };
    let timeline_context = request
        .advanced
        .get("timelineContext")
        .cloned()
        .unwrap_or_else(|| json!({}));
    json!({
        "type": "video",
        "assetId": plan.asset_id,
        "mediaPath": plan.media_rel,
        "mimeType": "video/mp4",
        "width": request.width,
        "height": request.height,
        "duration": request.duration,
        "fps": request.fps,
        "quality": request.quality,
        "family": plan.family,
        "seed": seed,
        "displayName": display_name,
        "createdAt": plan.created_at,
        "mode": request.mode,
        "model": request.model,
        "adapter": STUB_ADAPTER,
        "prompt": request.prompt,
        "negativePrompt": request.negative_prompt,
        "loras": request.loras,
        "rawAdapterSettings": stub_raw_settings(plan),
        "sourceAssetId": request.source_asset_id,
        "lastFrameAssetId": request.last_frame_asset_id,
        "sourceClipAssetId": request.source_clip_asset_id,
        "bridgeRightClipAssetId": request.bridge_right_clip_asset_id,
        "characterId": request.character_id,
        "characterLookId": request.character_look_id,
        "personTrackId": request.person_track_id,
        "replacementMode": request.replacement_mode,
        "timelineContext": timeline_context,
    })
}

fn stub_raw_settings(plan: &VideoPlan) -> Value {
    let request = &plan.request;
    json!({
        "model": request.model,
        "frameCount": request.frame_count(),
        "fps": request.fps,
        "duration": request.duration,
        "quality": request.quality,
        "stub": true,
    })
}

/// The job-result shape the API streams from: `assetWrites` + the `generationSet`
/// fact. A video job always reports exactly one asset (`expectedCount` 1).
fn streaming_result(plan: &VideoPlan, fact: &Value) -> JsonObject {
    let request = &plan.request;
    json!({
        "generationSetId": plan.genset_id,
        "expectedCount": 1,
        "adapter": STUB_ADAPTER,
        "model": request.model,
        "generationSet": {
            "id": plan.genset_id,
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": 1,
            "createdAt": plan.created_at,
        },
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// Progress payload with the worker's real backend label (mirrors `image_progress`).
fn video_progress(
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
        extra: BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(value: Value) -> VideoRequest {
        VideoRequest::from_payload(&value.as_object().cloned().unwrap())
    }

    #[test]
    fn plan_builds_nested_media_path() {
        let request = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "A red fox runs"
        }));
        let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
        assert!(plan
            .media_rel
            .starts_with(&format!("assets/videos/{}/", plan.genset_id)));
        assert!(plan.media_rel.ends_with(".mp4"));
        assert!(plan.media_rel.contains("_ltx_2_3_"));
        assert!(plan.asset_id.starts_with("asset_"));
        assert_eq!(plan.family, "ltx-video");
        assert_eq!(
            plan.media_path,
            Path::new("/tmp/project").join(&plan.media_rel)
        );
    }

    #[test]
    fn family_prefers_manifest_then_infers_from_model() {
        let manifest = request(json!({
            "projectId": "p", "model": "ltx_2_3",
            "modelManifestEntry": { "family": "ltx-custom" }
        }));
        assert_eq!(resolve_family(&manifest), "ltx-custom");
        let wan = request(json!({ "projectId": "p", "model": "wan_2_2_t2v_14b" }));
        assert_eq!(resolve_family(&wan), "wan-video");
        let other = request(json!({ "projectId": "p", "model": "mystery" }));
        assert_eq!(resolve_family(&other), "video");
    }

    #[test]
    fn resolve_seed_prefers_explicit_then_hashes_prompt() {
        let explicit = request(json!({ "projectId": "p", "seed": 123 }));
        assert_eq!(resolve_video_seed(&explicit), 123);
        // No seed → deterministic from the prompt (re-run reproduces).
        let a = request(json!({ "projectId": "p", "prompt": "sunset" }));
        let b = request(json!({ "projectId": "p", "prompt": "sunset" }));
        assert_eq!(resolve_video_seed(&a), resolve_video_seed(&b));
        let c = request(json!({ "projectId": "p", "prompt": "sunrise" }));
        assert_ne!(resolve_video_seed(&a), resolve_video_seed(&c));
    }

    #[test]
    fn stub_video_frames_have_correct_size_and_audio_split() {
        // LTX → audio present; the frame buffers are exactly width*height*3.
        let ltx = request(json!({
            "projectId": "p", "model": "ltx_2_3", "width": 256, "height": 256,
            "duration": 1.0, "fps": 9
        }));
        let decoded = generate_stub_video(&ltx, 7);
        assert_eq!(decoded.frames.len(), ltx.frame_count() as usize);
        assert_eq!(decoded.fps, 9);
        for frame in &decoded.frames {
            assert_eq!(frame.pixels.len(), 256 * 256 * 3);
        }
        let audio = decoded.audio.expect("LTX stub emits audio");
        assert_eq!(audio.sample_rate, 48_000);
        assert_eq!(audio.channels, 1);
        assert!(!audio.samples.is_empty());

        // Wan → no audio track (mirrors the engine).
        let wan = request(json!({
            "projectId": "p", "model": "wan_2_2_t2v_14b", "duration": 1.0, "fps": 16
        }));
        assert!(generate_stub_video(&wan, 7).audio.is_none());
    }

    #[test]
    fn stub_frames_differ_across_time() {
        // The sweeping band makes frame 0 and a later frame differ (real motion).
        let request = request(json!({
            "projectId": "p", "model": "wan_2_2", "width": 256, "height": 64,
            "duration": 1.0, "fps": 16
        }));
        let decoded = generate_stub_video(&request, 3);
        assert!(decoded.frames.len() >= 2);
        assert_ne!(
            decoded.frames[0].pixels,
            decoded.frames[decoded.frames.len() - 1].pixels
        );
    }

    #[test]
    fn wav_header_is_canonical_and_peak_normalized() {
        let audio = AudioTrack {
            samples: vec![0.0, 0.5, -0.25, 0.5],
            sample_rate: 48_000,
            channels: 1,
        };
        let dir = std::env::temp_dir().join(format!("sw_wav_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.wav");
        write_wav_pcm16(&audio, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
        // 4 mono 16-bit samples → 8 bytes of PCM, 44-byte header.
        assert_eq!(bytes.len(), 44 + 8);
        // Peak (0.5) maps to i16::MAX; the matching trough (-0.25) is half-scale negative.
        let first = i16::from_le_bytes([bytes[44], bytes[45]]);
        let peak = i16::from_le_bytes([bytes[46], bytes[47]]);
        let trough = i16::from_le_bytes([bytes[48], bytes[49]]);
        assert_eq!(first, 0);
        assert_eq!(peak, i16::MAX);
        assert_eq!(trough, -(i16::MAX / 2) - 1); // -0.25/0.5 * 32767, rounded
    }

    #[test]
    fn silent_audio_does_not_divide_by_zero() {
        let audio = AudioTrack {
            samples: vec![0.0; 16],
            sample_rate: 48_000,
            channels: 1,
        };
        let dir = std::env::temp_dir().join(format!("sw_wav_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("silent.wav");
        write_wav_pcm16(&audio, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes[44..].iter().all(|&b| b == 0));
    }

    #[test]
    fn asset_fact_and_streaming_result_shape() {
        let request = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "A red fox",
            "duration": 4.0, "fps": 24, "width": 768, "height": 512,
            "sourceAssetId": "asset_src", "personTrackId": "track_1"
        }));
        let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
        let fact = video_asset_fact(&plan, 42);
        assert_eq!(fact["type"], json!("video"));
        assert_eq!(fact["mimeType"], json!("video/mp4"));
        assert_eq!(fact["mediaPath"], json!(plan.media_rel));
        assert_eq!(fact["adapter"], json!("procedural_video"));
        assert_eq!(fact["seed"], json!(42));
        assert_eq!(fact["duration"], json!(4.0));
        assert_eq!(fact["fps"], json!(24));
        assert_eq!(fact["sourceAssetId"], json!("asset_src"));
        assert_eq!(fact["personTrackId"], json!("track_1"));
        assert_eq!(fact["displayName"], json!("A red fox"));

        let result = streaming_result(&plan, &fact);
        assert_eq!(result["expectedCount"], json!(1));
        assert_eq!(result["assetWrites"].as_array().unwrap().len(), 1);
        assert_eq!(result["generationSet"]["count"], json!(1));
    }

    /// Full encode → mp4 + poster, exercised against a real ffmpeg. Skips when no
    /// ffmpeg is reachable (SCENEWORKS_FFMPEG or `ffmpeg` on PATH) so it never fails a
    /// host without the binary; CI with ffmpeg runs it for real.
    #[tokio::test]
    async fn encode_stub_to_mp4_with_audio_and_poster() {
        if !ffmpeg_reachable() {
            eprintln!("skipping encode_stub_to_mp4_with_audio_and_poster: ffmpeg not found");
            return;
        }
        let request = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "fox",
            "duration": 1.0, "fps": 9, "width": 128, "height": 128
        }));
        let decoded = generate_stub_video(&request, 11);
        assert!(decoded.audio.is_some());
        let dir = std::env::temp_dir().join(format!("sw_vid_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let media_path = dir.join("clip.mp4");
        encode_media(&media_path, decoded, None).await.unwrap();
        assert!(media_path.exists(), "mp4 must be written");
        assert!(media_path.metadata().unwrap().len() > 0);
        assert!(
            media_path.with_extension("poster.jpg").exists(),
            "poster must be extracted"
        );
        // Intermediates are cleaned up.
        assert!(!media_path.with_extension("frames").exists());
        assert!(!media_path.with_extension("enc.mp4").exists());
        assert!(!media_path.with_extension("audio.wav").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn ffmpeg_reachable() -> bool {
        if let Ok(path) = std::env::var("SCENEWORKS_FFMPEG") {
            if !path.trim().is_empty() && Path::new(&path).exists() {
                return true;
            }
        }
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }
}
