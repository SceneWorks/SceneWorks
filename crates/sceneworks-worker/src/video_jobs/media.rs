use std::path::Path;

use super::prelude::*;
use super::run_ffmpeg_with_stdin_chunks;
use super::write_wav_pcm16;

// ---------------------------------------------------------------------------
// Encode pipeline: frames → mp4 (+ optional AAC audio) → faststart → poster.
// Reuses `media_jobs::run_ffmpeg`. Pure of the API except the optional `ctx`
// (heartbeat/cancel), so it is exercisable in tests with a real ffmpeg.
/// Write `decoded` to `media_path` as an mp4: raw RGB frames streamed to libx264, an optional 16-bit
/// PCM WAV muxed as AAC and bounded at the picture's length ([`audio_mux_args`]), then a
/// best-effort `+faststart` remux and
/// `.poster.jpg`. `media_path` is created (atomically renamed from a temp) only on
/// success; all intermediates are removed regardless of outcome.
pub(super) async fn encode_media(
    media_path: &Path,
    decoded: DecodedVideo,
    workflow_metadata: Option<&Path>,
    ctx: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let enc_tmp = media_path.with_extension("enc.mp4");
    let wav_tmp = media_path.with_extension("audio.wav");
    let mux_tmp = media_path.with_extension("mux.mp4");
    let result = encode_inner(
        media_path,
        decoded,
        workflow_metadata,
        ctx,
        &enc_tmp,
        &wav_tmp,
        &mux_tmp,
    )
    .await;
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
    workflow_metadata: Option<&Path>,
    ctx: Option<FfmpegContext<'_>>,
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

    let width = frames[0].width;
    let height = frames[0].height;
    let expected_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| WorkerError::InvalidPayload("video frame dimensions overflow".to_owned()))?;
    for frame in &frames {
        if frame.width != width || frame.height != height {
            return Err(WorkerError::InvalidPayload(
                "video frames must all have the same dimensions".to_owned(),
            ));
        }
        if frame.pixels.len() != expected_bytes {
            return Err(WorkerError::InvalidPayload(
                "video frame buffer size mismatch".to_owned(),
            ));
        }
    }

    // Captured before `frames` is consumed below: step 2 bounds the file at the PICTURE's length,
    // and that length is this count — never anything read back off the soundtrack (sc-19425).
    let frame_count = frames.len();

    // 1. Stream the engine-owned RGB buffers directly into FFmpeg. This moves one existing frame
    // buffer at a time through the pipe: no per-frame PNG encode, no multi-GB scratch tree, and no
    // second whole-video concatenation.
    //
    // The workflow envelope (sc-15956) rides in HERE rather than in a post-hoc rewrite, so the tag
    // is part of the file from the moment it exists: there is no window in which a clip is on disk
    // without its recipe, and no extra pass over what can be gigabytes. It survives the two steps
    // below — measured, and pinned by `the_envelope_survives_the_whole_encode_chain` in
    // `crates/sceneworks-worker/src/video_jobs/tests.rs`.
    let chunks = frames.into_iter().map(|frame| frame.pixels).collect();
    let mut args = vec![
        "ffmpeg".to_owned(),
        "-nostdin".to_owned(),
        "-y".to_owned(),
        "-f".to_owned(),
        "rawvideo".to_owned(),
        "-pix_fmt".to_owned(),
        "rgb24".to_owned(),
        "-video_size".to_owned(),
        format!("{width}x{height}"),
        "-framerate".to_owned(),
        fps.to_string(),
        "-i".to_owned(),
        "pipe:0".to_owned(),
    ];
    if let Some(metadata_path) = workflow_metadata {
        // Input 1. The frames are input 0 and stay mapped by ffmpeg's own stream selection (an
        // `ffmetadata` input carries no streams to compete with), so only the metadata source has
        // to be named.
        args.extend(sceneworks_core::workflow_mp4::ffmetadata_input_args(
            metadata_path,
        ));
    }
    args.extend([
        "-c:v".to_owned(),
        "libx264".to_owned(),
        "-pix_fmt".to_owned(),
        "yuv420p".to_owned(),
        "-r".to_owned(),
        fps.to_string(),
    ]);
    if workflow_metadata.is_some() {
        args.extend(sceneworks_core::workflow_mp4::ffmetadata_map_args(1));
    }
    args.push(enc_tmp.to_string_lossy().into_owned());
    run_ffmpeg_with_stdin_chunks(args, chunks, ctx).await?;

    // 2. Mux the audio track (LTX, MiniMax-H3) as AAC, else the video-only mp4 is the result.
    let finished_tmp = if let Some(audio) = audio {
        write_wav_pcm16(&audio, wav_tmp)?;
        run_ffmpeg(
            audio_mux_args(enc_tmp, wav_tmp, mux_tmp, frame_count, fps),
            ctx,
        )
        .await?;
        mux_tmp
    } else {
        enc_tmp
    };

    // 3. Publish atomically, then best-effort faststart + poster (mirrors Python).
    tokio::fs::rename(finished_tmp, media_path).await?;
    faststart_mp4(media_path).await;
    write_poster_frame(media_path).await;
    Ok(())
}

/// **The AV mux policy's one number**: the picture's own length in seconds, `frame_count / fps`,
/// rendered to the microsecond for ffmpeg's `-t`. The picture is the exact quantity; the soundtrack
/// and the container are fitted to it.
///
/// Extracted (sc-19549) so the two muxing call sites cannot drift into two spellings of the same
/// rule: the generation mux [`audio_mux_args`] and the SeedVR2 upscale mux
/// `seedvr2::seedvr2_audio_mux_args` (cfg-gated to the lanes that ship it, so it is named rather
/// than linked). Their argument vectors legitimately differ — the upscale
/// maps a source clip's optional audio and writes `+faststart` in the same pass — but the BOUND is
/// one policy, spelled once by [`format_picture_bound`] (SeedVR2 feeds it its measured length,
/// sc-24391). The measurements behind `-t` over `-shortest` live on [`audio_mux_args`].
///
/// `fps.max(1)` mirrors `encode_inner`'s own clamp rather than dividing by zero.
pub(crate) fn picture_bound_seconds(frame_count: usize, fps: u32) -> String {
    format_picture_bound(frame_count as f64 / f64::from(fps.max(1)))
}

/// The one spelling of the bound, to the microsecond; SeedVR2 passes its measured length (sc-24391).
pub(crate) fn format_picture_bound(seconds: f64) -> String {
    format!("{seconds:.6}")
}

/// The step-2 mux command: copy the encoded picture, encode the WAV as AAC, and **bound the file at
/// the picture's own length** — [`picture_bound_seconds`].
///
/// # Why `-t` and not `-shortest` (sc-19425)
///
/// `-shortest` makes the SOUNDTRACK a candidate for deciding the clip's length, and when it wins it
/// does so by **discarding video frames** — silently, and out of proportion to the mismatch.
/// Measured with the bundled ffmpeg 7.1 on this exact two-step command, at 24 fps, MiniMax-H3's 14
/// legal frame counts, with a soundtrack already fitted to the picture to within a third of one
/// sample (`round(frames / 24 · 32000)` samples per channel — the engine's own mux policy):
///
/// | frames | `-shortest` | no flag | `-t frames/fps` |
/// |---|---|---|---|
/// | 124 | **121** | 124 | 124 |
/// | 175 | **173** | 175 | 175 |
/// | 226 | **225** | 226 | 226 |
/// | 277 | **275** | 277 | 277 |
/// | 328 | **327** | 328 | 328 |
/// | the other 9 | exact | exact | exact |
///
/// Five of the fourteen lost picture — up to three frames for a 10 µs audio deficit — while the
/// container went on advertising the AUDIO's duration (5.17 s for a file holding 121 frames = 5.04 s
/// of picture). That is precisely "the container claims a duration the file does not have", the
/// defect sc-12371 measured `EncodedClip` for, reappearing one layer down where measuring the
/// `DecodedVideo` cannot see it.
///
/// Dropping the flag outright fixes the frame loss but gives up what it was for: an audio track
/// LONGER than the picture then extends the file (measured: a 2× track produced a 16.0 s container
/// around 8.0 s of picture). `-t` is the only one of the three that is right in both directions, and
/// it is the container-level spelling of the same rule the MiniMax-H3 engine applies to the
/// waveform: **the picture is the exact quantity, so the picture is what everything else is fitted
/// to.** It is a strict improvement for LTX too, whose vocoder output is not length-matched to the
/// frame count at all and which today loses picture whenever that output lands short.
///
/// # Why `-t` cannot do to the picture what `-shortest` does
///
/// MEASURED: under `-c:v copy` the bound carries a consistent **two frames of slack at the tail** —
/// `-t 4.0` on a 24 fps clip keeps 98 frames, not the 96 the arithmetic suggests, and `-t 2.5` keeps
/// 62, not 60. (Consistent with the bound being applied to decode timestamps, which lag presentation
/// by libx264's default B-frame reorder delay; the two-frame figure is measured, that attribution is
/// not.) So the bound is inclusive at its own end and errs toward keeping picture, which is the
/// direction that matters here: at
/// `frame_count / fps` there is nothing past it to keep, and a bound short by a microsecond of
/// `{:.6}` rounding is nowhere near a frame interval (41 667 µs at 24 fps) of the last frame's
/// presentation time at `(frame_count - 1) / fps`. Trimming a long soundtrack is unaffected — the
/// audio is re-encoded, not copied (measured: a 2× track lands at 5.17 s, not 10.33 s).
pub(crate) fn audio_mux_args(
    video: &Path,
    wav: &Path,
    out: &Path,
    frame_count: usize,
    fps: u32,
) -> Vec<String> {
    vec![
        "ffmpeg".to_owned(),
        "-nostdin".to_owned(),
        "-y".to_owned(),
        "-i".to_owned(),
        video.to_string_lossy().into_owned(),
        "-i".to_owned(),
        wav.to_string_lossy().into_owned(),
        "-c:v".to_owned(),
        "copy".to_owned(),
        "-c:a".to_owned(),
        "aac".to_owned(),
        "-t".to_owned(),
        picture_bound_seconds(frame_count, fps),
        // Explicit, though it is also ffmpeg's default for a multi-input command: the
        // container metadata — including the sc-15956 workflow tag written in step 1 —
        // comes from the VIDEO, never from the WAV. Stated because "the default happens to
        // be right" is not a property anyone maintains, and the step below it depends on
        // this one having carried the tag through.
        "-map_metadata".to_owned(),
        "0".to_owned(),
        out.to_string_lossy().into_owned(),
    ]
}

/// Best-effort `+faststart` remux (moov atom to the front so WKWebView can start
/// playback without a tail byte-range seek). A missing/failing ffmpeg leaves the
/// original untouched — the API's byte-range support is the load-bearing guarantee.
pub(crate) async fn faststart_mp4(media_path: &Path) {
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
pub(crate) async fn write_poster_frame(media_path: &Path) {
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
