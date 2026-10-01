#[allow(unused_imports)]
use super::prelude::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::vace::rgb_image_to_engine;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::wan::{generate_video, VideoGenInput};

// ---------------------------------------------------------------------------
// SeedVR2 video upscale (epic 4811, sc-4816): the net-new `video_upscale` job —
// SceneWorks' first video upscaler. Decode the source clip -> native-MLX SeedVR2
// one-step super-resolution (temporal chunking + overlap is internal to the engine)
// -> encode + source-audio passthrough. Native MLX on Mac and Candle/CUDA off-Mac (no Python path).
// Reuses the shared
// encode pipeline (`encode_media`) + the streaming engine driver (`generate_video`).
// ---------------------------------------------------------------------------

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) use runtime_cuda::providers::seedvr2::video as seedvr2_video;
/// The SeedVR2 provider's pure temporal-chunk planning/blend module (`video::plan_chunks`,
/// `video::assemble_overlap`, `DEFAULT_OVERLAP`, `Chunk`), reused ONE LEVEL UP for worker-window
/// streaming (sc-9595). Both provider crates expose an identical `video` module over `gen_core::Image`
/// (byte-identical seam math), so the worker-window cross-fade is the engine's own — the worker never
/// reimplements the blend. Aliased per platform: MLX on Mac, candle on the Windows/CUDA lane.
#[cfg(target_os = "macos")]
pub(super) use runtime_macos::providers::seedvr2::video as seedvr2_video;

// SeedVR2 video upscale runs on Mac (native MLX) AND the Windows/CUDA candle lane (sc-5928); these
// constants/helpers are backend-neutral (gen_core + ffmpeg + the shared streaming driver).
//
// The repo/revision/filename constants that used to live here are GONE (sc-17632). They were a
// verbatim duplicate of the image-upscale lane's, and each lane downloaded the same ~7.3 GB
// checkpoint into its OWN `<data_dir>/cache` subtree. Both lanes now resolve the ONE installed copy
// through `upscale_jobs::require_seedvr2_checkpoint_dir`, which owns the repo pin, the two operator
// dir pins and the read-only legacy roots — so the pins cannot drift between the lanes at all,
// rather than being held together by an agreement test.
/// The engine registry id wired for video upscale (3B; 7B = sc-5197 / sc-5927).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_ENGINE_ID: &str = "seedvr2_3b";
/// Adapter id recorded on the result asset for provenance (mirrors the other `mlx_*` video adapters;
/// SeedVR2 itself takes no LoRA — this is metadata only).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_ADAPTER: &str = "mlx_seedvr2";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_CANCEL_MESSAGE: &str = "Video upscale canceled by user.";

/// A failed post-encode step must not leave the intermediate, silent output, or derived poster
/// behind. The API publishes the asset only after the final completion update succeeds.
#[cfg(any(target_os = "macos", feature = "backend-candle", test))]
pub(super) async fn cleanup_failed_seedvr2_mux(media_path: &Path, mux_tmp: &Path) {
    let _ = tokio::fs::remove_file(mux_tmp).await;
    let _ = tokio::fs::remove_file(media_path).await;
    let _ = tokio::fs::remove_file(media_path.with_extension("poster.jpg")).await;
}

#[cfg(any(target_os = "macos", feature = "backend-candle", test))]
pub(super) struct Seedvr2OutputGuard {
    media_path: PathBuf,
    mux_tmp: PathBuf,
    armed: bool,
}

#[cfg(any(target_os = "macos", feature = "backend-candle", test))]
impl Seedvr2OutputGuard {
    pub(super) fn new(media_path: &Path, mux_tmp: &Path) -> Self {
        Self {
            media_path: media_path.to_path_buf(),
            mux_tmp: mux_tmp.to_path_buf(),
            armed: true,
        }
    }

    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

#[cfg(any(target_os = "macos", feature = "backend-candle", test))]
impl Drop for Seedvr2OutputGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.mux_tmp);
            let _ = std::fs::remove_file(&self.media_path);
            let _ = std::fs::remove_file(self.media_path.with_extension("poster.jpg"));
        }
    }
}

/// Snap a dimension to the SeedVR2 VAE/patch stride (a multiple of 16, the engine's hard
/// requirement), rounding to nearest and clamping to the engine's `[16, 4096]` size range.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn snap_seedvr2_dim(value: u32) -> u32 {
    let rounded = value.saturating_add(8) / 16 * 16;
    rounded.clamp(16, 4096)
}

// ---------------------------------------------------------------------------
// Disk-space guard for the streaming SeedVR2 output (sc-9646, sc-9595 follow-up)
// ---------------------------------------------------------------------------
// sc-9595 removed the sc-8829 host-RAM cap by streaming the upscale in temporal windows, so peak host
// RAM is now bounded to ~one window regardless of clip length. But the constraint MOVED from RAM to
// DISK: the full upscaled PNG sequence is written to a worker scratch dir before the final encode, so
// a multi-minute / 4K clip can now write many GB with NO guard (the RAM cap previously bounded the
// whole operation). This mirrors the removed `check_seedvr2_host_ram`: a generous, machine-derived,
// fail-loud-before-the-window-loop preflight that estimates the output PNG footprint and rejects a
// clip that would fill the scratch volume, so the disk is not silently exhausted mid-run.

/// Fraction of the scratch volume's CURRENTLY-AVAILABLE space the streamed output PNG sequence is
/// allowed to occupy. Deliberately generous — the estimate itself uses raw RGB8 per frame (a real
/// upper bound on PNG-compressed output), and we still leave headroom for the source frames already on
/// disk, the eventual encoded MP4, and everything else sharing the volume.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_DISK_OUTPUT_FRACTION: f64 = 0.8;

/// Estimated peak on-disk bytes for the streamed output: `frame_count` PNG frames at
/// `out_w × out_h`, sized as raw RGB8 (`w·h·3`) per frame. PNG compression only ever makes the real
/// footprint SMALLER, so this is a safe upper bound (matching the generous shape of the removed
/// `seedvr2_estimated_host_bytes`). Pure so the estimate is unit-testable without a filesystem.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn seedvr2_estimated_output_bytes(frame_count: u64, out_w: u64, out_h: u64) -> u64 {
    let per_frame = out_w.saturating_mul(out_h).saturating_mul(3);
    frame_count.saturating_mul(per_frame)
}

/// Peak scratch footprint while SeedVR2 is streaming: the native-resolution source PNG sequence and
/// the upscaled output sequence coexist until the final encode. Both use a raw-RGB upper bound.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn seedvr2_estimated_scratch_bytes(
    frame_count: u64,
    src_w: u64,
    src_h: u64,
    out_w: u64,
    out_h: u64,
) -> u64 {
    seedvr2_estimated_output_bytes(frame_count, src_w, src_h)
        .saturating_add(seedvr2_estimated_output_bytes(frame_count, out_w, out_h))
}

/// Bytes currently AVAILABLE on the volume backing `path`, best-effort and portable with NO new crate
/// dependency: macOS + Linux run POSIX `df -k -P <path>` and read the 4th column (available 1K
/// blocks); Windows (candle lane) uses `fs2::available_space` (a safe wrapper over the Win32
/// `GetDiskFreeSpaceExW`) for the free bytes available to this process. Returns `None` if the probe
/// fails; the caller then skips the guard rather than falsely rejecting a job.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn available_disk_bytes(path: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        // `df -k -P` forces POSIX one-line-per-filesystem output in 1024-byte blocks, so the columns
        // are stable regardless of locale/long device names:
        //   Filesystem 1024-blocks Used Available Capacity Mounted on
        let out = std::process::Command::new("df")
            .args(["-k", "-P"])
            .arg(path)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        // The data row is the 2nd line (after the header); `-P` guarantees it is a single line.
        let row = text.lines().nth(1)?;
        let available_kib = row.split_whitespace().nth(3)?.parse::<u64>().ok()?;
        Some(available_kib.saturating_mul(1024))
    }
    #[cfg(not(unix))]
    {
        // Windows (candle lane): `fs2::available_space` wraps `GetDiskFreeSpaceExW`, returning the
        // free bytes available to this process on the volume backing `path` (honoring any per-user
        // quota — the same figure the old code's "avail" line meant to read). Safe (the crate forbids
        // `unsafe` and fs2 is already a worker dependency) and — unlike scraping localized `fsutil`
        // text — locale-independent, so the sc-9646 guard is armed on every Windows install (sc-13585:
        // the old branch matched an `fsutil` line with both "avail" and "free bytes", which no modern
        // Windows 11 layout — "Total free bytes" / "Total quota free bytes" — prints, so the probe
        // returned `None` and the guard silently fail-opened into a no-op). A probe error (e.g. the
        // path does not exist) becomes `None`, so the guard fail-opens rather than falsely rejecting.
        fs2::available_space(path).ok()
    }
}

/// Fail loud (before the window loop / before any GPU work) when the estimated streamed-output PNG
/// footprint would exceed the generous fraction of the scratch volume's currently-available space.
/// `Ok(())` when it fits, when the frame count / dimensions are unknown (0), or when the free-space
/// probe is unavailable (we do not falsely reject a job on a probe failure). The error names the
/// estimate AND the available space so the user knows exactly what to trim. Mirrors the shape of the
/// removed `check_seedvr2_host_ram` (sc-9646).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(dead_code)] // retained as the output-only regression seam for sc-9646/sc-13585 tests
pub(crate) fn check_seedvr2_output_disk(
    scratch_dir: &Path,
    frame_count: u64,
    out_w: u64,
    out_h: u64,
) -> WorkerResult<()> {
    if frame_count == 0 || out_w == 0 || out_h == 0 {
        return Ok(());
    }
    let Some(available) = available_disk_bytes(scratch_dir) else {
        // Probe failed (unknown platform / error): skip the guard rather than block a valid job.
        return Ok(());
    };
    let needed = seedvr2_estimated_output_bytes(frame_count, out_w, out_h);
    let budget = ((available as f64) * SEEDVR2_DISK_OUTPUT_FRACTION) as u64;
    if needed > budget {
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        // Largest frame count that fits the budget, so the message is actionable.
        let per_frame = seedvr2_estimated_output_bytes(1, out_w, out_h).max(1);
        let max_frames = budget / per_frame;
        return Err(WorkerError::InvalidPayload(format!(
            "Not enough disk space to upscale this clip: {frame_count} output frames at \
             {out_w}×{out_h} would write ~{needed:.1} GB of PNG frames to the scratch volume, over \
             the ~{budget:.1} GB usable of ~{available:.1} GB free. Trim the clip to about \
             {max_frames} frames (or fewer), lower the target resolution, or free up disk space.",
            needed = needed as f64 / GIB,
            budget = budget as f64 / GIB,
            available = available as f64 / GIB,
        )));
    }
    Ok(())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn check_seedvr2_scratch_disk(
    scratch_dir: &Path,
    frame_count: u64,
    src_w: u64,
    src_h: u64,
    out_w: u64,
    out_h: u64,
) -> WorkerResult<()> {
    if frame_count == 0 || src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0 {
        return Ok(());
    }
    let Some(available) = available_disk_bytes(scratch_dir) else {
        return Ok(());
    };
    let needed = seedvr2_estimated_scratch_bytes(frame_count, src_w, src_h, out_w, out_h);
    let budget = ((available as f64) * SEEDVR2_DISK_OUTPUT_FRACTION) as u64;
    if needed > budget {
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        let per_frame = seedvr2_estimated_scratch_bytes(1, src_w, src_h, out_w, out_h).max(1);
        let max_frames = budget / per_frame;
        return Err(WorkerError::InvalidPayload(format!(
            "Not enough disk space to upscale this clip: {frame_count} source frames at \
             {src_w}×{src_h} plus {frame_count} output frames at {out_w}×{out_h} need ~{needed:.1} GB \
             of PNG scratch, over the ~{budget:.1} GB usable of ~{available:.1} GB free. Trim the clip \
             to about {max_frames} frames (or fewer), lower the target resolution, or free disk space.",
            needed = needed as f64 / GIB,
            budget = budget as f64 / GIB,
            available = available as f64 / GIB,
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Streaming worker-window chunking for the SeedVR2 upscale (sc-9595, removes the sc-8829 host-RAM cap)
// ---------------------------------------------------------------------------
// sc-8829 (F-027) bounded host RGB8 RAM with a machine-derived frame cap that FAILED LOUD before decode:
// `decode_seedvr2_source_frames` materialized EVERY source frame into a `Vec<Image>` up front, and the
// up-to-4× output was likewise held whole before encode (the engine's whole-clip API returns the entire
// `Vec<Image>`), so a few-minute 1080p clip meant tens of GB of RGB8 → OOM. The cap rejected such clips
// — a capability narrowing.
//
// This story removes the cap by streaming the upscale in temporal WORKER WINDOWS: we plan windows over
// the real frame count with the engine's own `video::plan_chunks` (a valid chunk length + the
// `DEFAULT_OVERLAP=4` cross-fade), decode + upscale ONE window at a time through the shared
// `generate_video` funnel, and cross-fade across worker-window boundaries with the engine's own
// `video::assemble_overlap` (fed a local 2-window plan) so the seam handling is byte-identical to the
// engine's internal chunking. Finalized frames stream straight to a numbered PNG sequence on disk and
// are encoded once at the end (same ffmpeg args as the old whole-clip path), so peak host RAM is bounded
// to ~one worker window's frames + a ≤4-frame overlap tail regardless of clip length.
//
// Seam identity: `plan_chunks`/`assemble_overlap` are the SAME pure functions the engine uses
// internally, and each engine chunk's output is a deterministic function of its source pixel window +
// seed (`pipeline::preprocess_chunk`). When a worker window is processed by the engine as a single
// internal chunk (the common case — `SEEDVR2_WORKER_CHUNK_FRAMES` fits the engine's budget-sized chunk),
// the streamed output is bit-identical to the whole-clip run. If the engine sub-chunks a worker window
// under tight GPU budget, the cross-fade still closes every seam (identical blend math) but the
// upscaled pixels near an internal boundary can differ slightly from a whole-clip run's internal
// boundary — a real-weights fidelity nuance that only a GPU golden run can measure (see the PR notes).

/// The worker-level temporal window size (pixel frames). A valid engine chunk length (mult of 4, ≥8),
/// chosen at the engine's `MAX_CHUNK_FRAMES` ceiling so that on any machine whose budget-sized chunk is
/// ≥ this, the engine processes a whole worker window as ONE internal chunk (`plan_chunks(64,64,4)` = a
/// single chunk) → the streamed output is bit-identical to a whole-clip run. Larger windows mean fewer
/// worker-window seams and a closer match to whole-clip chunking, traded against a larger per-window
/// host footprint (≈ `64 · out_w · out_h · 3` bytes of RGB8, ~1.6 GB at 4×-of-1080p — bounded).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const SEEDVR2_WORKER_CHUNK_FRAMES: i32 = 64;

/// Streaming cross-window assembler (sc-9595): feeds each upscaled worker window into the engine's own
/// `video::assemble_overlap` and emits finalized frames in order, holding only a ≤`ov`-frame tail plus
/// the current window in memory. The cross-fade at each worker-window boundary is byte-identical to the
/// engine's internal chunk assembly — proven equivalent to a single whole-clip `assemble_overlap` over
/// synthetic frames (see `seedvr2_stream_matches_whole_clip_assembly`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) struct Seedvr2StreamAssembler {
    /// Cross-fade overlap (frames) between adjacent worker windows — the engine's `DEFAULT_OVERLAP`.
    ov: i32,
    /// Total real output frame count (== source frame count); trailing chunk padding past this is dropped.
    n: i32,
    /// The retained, blended-but-not-yet-final tail: the assembled frames from `retained_start` onward
    /// that may still be cross-faded by the NEXT window. Its absolute start is `retained_start`.
    pub(super) retained: Vec<Image>,
    /// Absolute frame index of `retained[0]`.
    retained_start: i32,
    /// Whether `push_window` has been called yet (the first window seeds `retained` with no blend).
    seeded: bool,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl Seedvr2StreamAssembler {
    pub(super) fn new(n: i32, ov: i32) -> Self {
        Self {
            ov,
            n,
            retained: Vec::new(),
            retained_start: 0,
            seeded: false,
        }
    }

    /// Feed the upscaled frames for the worker window at absolute `start`. Returns the newly finalized
    /// frames (in order) that no later window can touch; the caller streams them to disk immediately.
    /// Uses the engine's `video::assemble_overlap` over a local `[retained, window]` 2-window plan so
    /// the blend is the engine's own.
    pub(super) fn push_window(&mut self, start: i32, mut frames: Vec<Image>) -> Vec<Image> {
        // Clip trailing chunk padding past the real frame count.
        let visible = (self.n - start).clamp(0, frames.len() as i32);
        frames.truncate(visible.max(0) as usize);
        if !self.seeded {
            self.seeded = true;
            self.retained_start = start;
            self.retained = frames;
        } else {
            let off = start - self.retained_start;
            let local_plan = [
                seedvr2_video::Chunk {
                    start: 0,
                    len: self.retained.len() as i32,
                },
                seedvr2_video::Chunk {
                    start: off,
                    len: frames.len() as i32,
                },
            ];
            let n_local = off + frames.len() as i32;
            let inputs = [std::mem::take(&mut self.retained), frames];
            // The engine's own cross-fade closes the worker-window seam (identical to its internal one).
            self.retained = seedvr2_video::assemble_overlap(&local_plan, &inputs, n_local, self.ov);
        }
        // Finalize everything except the last `ov` frames (the next window may still blend them).
        let keep = self.ov.max(0) as usize;
        self.drain_prefix(keep)
    }

    /// Flush the remaining tail once every window has been pushed.
    pub(super) fn finish(&mut self) -> Vec<Image> {
        self.drain_prefix(0)
    }

    /// Emit finalized frames from the front of `retained`, keeping `keep` frames retained (and never
    /// emitting past the real frame count `n`).
    fn drain_prefix(&mut self, keep: usize) -> Vec<Image> {
        // How many front frames are finalized this call: everything past `keep`, but never emitting
        // past the real frame count `n` (`n - retained_start` remaining slots). `Vec::drain` shifts
        // the tail once, so this is O(n) rather than the O(n²) of a `remove(0)`-per-frame loop.
        let by_keep = self.retained.len().saturating_sub(keep);
        let by_count = (self.n - self.retained_start).max(0) as usize;
        let boundary = by_keep.min(by_count);
        let out: Vec<Image> = self.retained.drain(..boundary).collect();
        self.retained_start += boundary as i32;
        out
    }
}

/// Resolve a project-relative asset path safely under `project_path` (reject `..` / absolute
/// components — same guard as `upscale_jobs::resolve_source`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn safe_join(project_path: &Path, rel: &str) -> Option<PathBuf> {
    let mut path = project_path.to_path_buf();
    for component in Path::new(rel).components() {
        match component {
            std::path::Component::Normal(value) => path.push(value),
            _ => return None,
        }
    }
    Some(path)
}

/// The source decode command: every frame, in presentation order, to `in_%05d.png` (numbered from
/// 1). It maps the same stream the probe measures (`0:V:0`, the first video stream that is not cover
/// art or a thumbnail), and `-fps_mode passthrough` neither drops nor duplicates, so PNG `k` is the
/// probe's frame `k-1` and the probe's timestamps describe it (sc-24391). Kept pure so tests run
/// this exact command.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn seedvr2_source_decode_args(source: &Path, frames_dir: &Path) -> Vec<String> {
    vec![
        "ffmpeg".to_owned(),
        "-nostdin".to_owned(),
        "-y".to_owned(),
        "-i".to_owned(),
        source.to_string_lossy().into_owned(),
        "-map".to_owned(),
        "0:V:0".to_owned(),
        "-fps_mode".to_owned(),
        "passthrough".to_owned(),
        frames_dir
            .join("in_%05d.png")
            .to_string_lossy()
            .into_owned(),
    ]
}

/// Decode every frame of `source` to a numbered PNG sequence ON DISK (native resolution — the engine
/// bicubic-upscales internally to the target) and return the ordered PNG paths, WITHOUT loading any
/// pixels into RAM (sc-9595). Uses the bundled ffmpeg (`run_ffmpeg`); `-fps_mode passthrough` keeps the
/// exact source frame count. The caller loads each temporal WINDOW of these paths on demand via
/// [`load_seedvr2_window`], so host RGB8 RAM is bounded to one window instead of the whole clip. The
/// returned paths live under a job-scoped temp dir the caller is responsible for removing.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn decode_seedvr2_source_to_disk(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    source: &Path,
    frames_dir: &Path,
) -> WorkerResult<Vec<PathBuf>> {
    let _ = tokio::fs::remove_dir_all(frames_dir).await;
    tokio::fs::create_dir_all(frames_dir).await?;
    let ctx = FfmpegContext::new(api, settings, job_id, SEEDVR2_CANCEL_MESSAGE);
    run_ffmpeg(seedvr2_source_decode_args(source, frames_dir), Some(ctx)).await?;
    let dir = frames_dir.to_path_buf();
    let paths = tokio::task::spawn_blocking(move || -> WorkerResult<Vec<PathBuf>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
            .collect();
        paths.sort();
        Ok(paths)
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;
    if paths.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "source video produced no frames to upscale".to_owned(),
        ));
    }
    Ok(paths)
}

#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Seedvr2SourceProbe {
    pub(super) frame_count: u64,
    pub(super) width: u32,
    pub(super) height: u32,
}

/// Parse FFmpeg's mapped output-stream geometry and final `frame=` counter. The output stream is
/// authoritative because FFmpeg applies input rotation metadata before the null mux (and before our
/// PNG decode), so the first/input `Video:` line can report the opposite orientation. Kept pure so
/// the admission plan is tested without real media. We intentionally ignore encoder/status tokens
/// outside sane dimensions.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn parse_seedvr2_source_probe(stderr: &str) -> Option<Seedvr2SourceProbe> {
    let frame_count = stderr.rmatch_indices("frame=").find_map(|(idx, _)| {
        stderr[idx + 6..]
            .trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .filter(|digits| !digits.is_empty())
            .and_then(|digits| digits.parse::<u64>().ok())
    })?;
    let video_line = stderr.lines().rev().find(|line| line.contains("Video:"))?;
    let (width, height) = video_line
        .split(|c: char| c.is_whitespace() || c == ',')
        .find_map(|token| {
            let token = token.trim_matches(|c: char| !c.is_ascii_digit() && c != 'x');
            let (w, h) = token.split_once('x')?;
            let (w, h) = (w.parse::<u32>().ok()?, h.parse::<u32>().ok()?);
            (w > 0 && h > 0 && w <= 65_535 && h <= 65_535).then_some((w, h))
        })?;
    (frame_count > 0).then_some(Seedvr2SourceProbe {
        frame_count,
        width,
        height,
    })
}

/// The probe command: one FFmpeg null-mux pass that decodes every frame of the stream the decode
/// maps (`0:V:0`). `showinfo` logs each decoded frame's presentation timestamp and the stream time
/// base, which is where the upscale's output timing comes from ([`parse_seedvr2_source_timing`],
/// sc-24391). `checksum=0` skips the per-plane checksums the filter would otherwise compute on every
/// frame; the option exists in every FFmpeg the worker ships with (5.1+). Kept pure so tests run
/// this exact command.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn seedvr2_source_probe_args(source: &Path) -> Vec<String> {
    vec![
        "ffmpeg".to_owned(),
        "-nostdin".to_owned(),
        "-hide_banner".to_owned(),
        "-i".to_owned(),
        source.to_string_lossy().into_owned(),
        "-map".to_owned(),
        "0:V:0".to_owned(),
        "-vf".to_owned(),
        "showinfo=checksum=0".to_owned(),
        "-f".to_owned(),
        "null".to_owned(),
        "-".to_owned(),
    ]
}

/// VFR-safe source probe using the bundled FFmpeg null mux: frame count, output geometry and the
/// frame timing the upscaled clip must keep. The shared runner preserves the normal
/// heartbeat/cancellation lifecycle while FFmpeg walks the source.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn probe_seedvr2_source(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    source: &Path,
) -> WorkerResult<(Seedvr2SourceProbe, Seedvr2SourceTiming)> {
    let ctx = FfmpegContext::new(api, settings, job_id, SEEDVR2_CANCEL_MESSAGE);
    let stderr =
        crate::media_jobs::run_ffmpeg_capture_stderr(seedvr2_source_probe_args(source), Some(ctx))
            .await?;
    let probe = parse_seedvr2_source_probe(&stderr).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Could not determine the source video's frame count and dimensions before upscale."
                .to_owned(),
        )
    })?;
    let timing = parse_seedvr2_source_timing(&stderr, probe.frame_count).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Could not determine the source video's frame timing before upscale.".to_owned(),
        )
    })?;
    Ok((probe, timing))
}

// ---------------------------------------------------------------------------
// Source frame timing (sc-24391)
// ---------------------------------------------------------------------------
// The upscale keeps every source frame, so the output is only right if it also keeps WHEN each frame
// is shown. That timing used to come from the asset sidecar's `fps`, which an imported clip does not
// have: `import_asset` writes `fps: null` for every video upload, and this lane fell back to 24 fps.
// MEASURED (sc-24391): a 10 s, 29.97 fps clip came out as 12.46 s of picture at 24 fps against its
// own 10 s soundtrack. A sidecar that did carry a rate was rounded to an integer (29.97 -> 30). The
// worker now measures the timing off the decoded source frames in the probe pass it already makes,
// and the sidecar's `fps` is not read at all.

/// How far a measured frame may sit off a constant-rate grid and still be encoded on that grid: one
/// millisecond, or one source tick when the time base is coarser than that. A millisecond time base
/// (Matroska) cannot store 30000/1001 exactly, so its "29.97 fps" frames sit up to half a tick off the
/// ideal grid. A frame further off than this belongs to a genuinely variable clip, which keeps its own
/// timestamps.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_CFR_TOLERANCE_SECONDS: f64 = 0.001;

/// The finest tick the variable-rate encode carries. FFmpeg's concat demuxer reads each frame's
/// `duration` in microseconds (`AV_TIME_BASE`), so a finer source time base is rescaled to
/// microseconds, far below anything a viewer or a soundtrack can resolve.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_MAX_TIMESCALE: u64 = 1_000_000;

/// When the upscaled frames are shown (sc-24391). The upscale is 1:1 in frames, so this is the
/// source's own timing as measured by [`parse_seedvr2_source_timing`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
pub(super) enum Seedvr2Timing {
    /// Every frame sits on one uniform grid: encode at exactly `num/den` frames per second
    /// (`30000/1001`, never a rounded 30).
    Constant { num: u64, den: u64 },
    /// A genuinely variable-rate source. `offsets[i]` is frame `i`'s presentation time after the
    /// first frame, in `1/timescale` s ticks, and `last_duration` is how long the final frame shows.
    Variable {
        timescale: u64,
        offsets: Vec<u64>,
        last_duration: u64,
    },
}

#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
impl Seedvr2Timing {
    /// Constant-rate timing at a whole `fps`, clamped to at least 1 fps like the other encode paths.
    #[cfg(test)]
    pub(super) fn from_fps(fps: u32) -> Self {
        Self::Constant {
            num: u64::from(fps.max(1)),
            den: 1,
        }
    }

    /// The picture's length in seconds for `frame_count` frames: the number the audio mux bounds
    /// the file at.
    pub(super) fn picture_seconds(&self, frame_count: usize) -> f64 {
        match self {
            Self::Constant { num, den } => frame_count as f64 * *den as f64 / *num as f64,
            Self::Variable {
                timescale,
                offsets,
                last_duration,
            } => {
                let end = offsets.last().copied().unwrap_or(0) + last_duration;
                end as f64 / *timescale as f64
            }
        }
    }

    /// The rate FFmpeg is handed for a constant-rate encode: `30000/1001`, or a bare `24` for a
    /// whole rate so a whole-rate clip gets the same encode arguments it always did.
    fn rate_arg(num: u64, den: u64) -> String {
        if den == 1 {
            num.to_string()
        } else {
            format!("{num}/{den}")
        }
    }

    /// The whole-number rate handed to the engine request, which takes an integer and does not use
    /// it to time anything (SeedVR2 upscales frames, not seconds).
    pub(super) fn nominal_fps(&self, frame_count: usize) -> u32 {
        let fps = self.average_fps(frame_count).round();
        if fps.is_finite() {
            (fps as u32).max(1)
        } else {
            1
        }
    }

    /// Mean frames per second over the picture.
    fn average_fps(&self, frame_count: usize) -> f64 {
        match self {
            Self::Constant { num, den } => *num as f64 / *den as f64,
            Self::Variable { .. } => frame_count as f64 / self.picture_seconds(frame_count),
        }
    }

    /// The `fps` recorded on the upscaled asset: a whole number for a whole rate (the JSON integer
    /// it has always been), the exact mean otherwise (`29.97002997…`, not a rounded 30).
    pub(super) fn fps_json(&self, frame_count: usize) -> Value {
        match self {
            Self::Constant { num, den: 1 } => json!(num),
            _ => json!(self.average_fps(frame_count)),
        }
    }

    /// The mean rate of a variable-rate clip as an exact `num/den` (frames over its picture length),
    /// for x264. Passed through, the encoder would otherwise take the concat stream's time base
    /// (`1/timescale`) as its frame rate and size the H.264 level for thousands of frames per second:
    /// MEASURED in review on ffmpeg 6.1.1 and 8.1.1, 1920x1088 at timescale 90000 was stamped level
    /// 6.2 against 4.0 for the same frames encoded constant-rate. `None` for a constant-rate clip.
    pub(super) fn variable_mean_rate(&self) -> Option<String> {
        let Self::Variable {
            timescale,
            offsets,
            last_duration,
        } = self
        else {
            return None;
        };
        let end = offsets.last()? + last_duration;
        let (num, den) = reduced_ratio(
            offsets.len() as u128 * u128::from(*timescale),
            u128::from(end),
        )?;
        Some(format!("{num}/{den}"))
    }
}

/// `num/den` in lowest terms, or `None` for a zero term or a result that does not fit in `u64`.
#[cfg_attr(
    not(any(
        test,
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
fn reduced_ratio(num: u128, den: u128) -> Option<(u64, u64)> {
    fn gcd(a: u128, b: u128) -> u128 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    if num == 0 || den == 0 {
        return None;
    }
    let divisor = gcd(num, den);
    Some((
        u64::try_from(num / divisor).ok()?,
        u64::try_from(den / divisor).ok()?,
    ))
}

/// The measured source timing the upscaled clip keeps (sc-24391): when each frame is shown relative
/// to the first ([`Seedvr2Timing`]), and how far into the source that first frame is.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Seedvr2SourceTiming {
    pub(super) frames: Seedvr2Timing,
    /// The first frame's presentation time, in microseconds after the source's own start (the zero
    /// FFmpeg measures every stream of an input from). The upscaled picture starts at 0, so the mux
    /// skips this much of the source's audio to keep sound and picture in step: a capture whose
    /// video starts 0.48 s after its audio would otherwise play 480 ms out of sync (MEASURED in
    /// review). 0 when the first frame carries no timestamp.
    pub(super) start_micros: u64,
}

#[cfg(test)]
impl Seedvr2SourceTiming {
    /// Constant-rate timing at a whole `fps`, starting at the source's start.
    pub(super) fn from_fps(fps: u32) -> Self {
        Self {
            frames: Seedvr2Timing::from_fps(fps),
            start_micros: 0,
        }
    }
}

/// Read the source's frame timing off the probe's `showinfo` log (sc-24391): the stream time base
/// from the filter's `config in` line and every frame's `pts`, in order. A mid-stream filter
/// reconfiguration (the source changes resolution or pixel format) prints a fresh `config in` and
/// restarts the filter's frame numbering at 0; on the same time base the frames simply continue.
/// `None` when the log does not describe exactly `frame_count` frames, numbered in order, in one time
/// base; the caller then fails the job rather than guess a rate, because a guessed rate is the bug
/// this replaces.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn parse_seedvr2_source_timing(
    stderr: &str,
    frame_count: u64,
) -> Option<Seedvr2SourceTiming> {
    fn ratio_after(text: &str, key: &str) -> Option<(u64, u64)> {
        let value = text.split_once(key)?.1.trim_start();
        let value = value.split([',', ' ']).next()?;
        let (num, den) = value.split_once('/')?;
        Some((num.parse().ok()?, den.parse().ok()?))
    }

    let mut time_base: Option<(u64, u64)> = None;
    let mut frame_rate: Option<(u64, u64)> = None;
    let mut pts: Vec<Option<i64>> = Vec::new();
    // How many frames earlier filter configurations already logged; the current one numbers its
    // frames from 0 again.
    let mut segment_base = 0;
    for line in stderr.lines() {
        // `[Parsed_showinfo_0 @ 0x…] <body>`; every other line (and showinfo's own side-data and
        // colour continuation lines) is skipped.
        let Some((_, body)) = line.split_once("[Parsed_showinfo_0 @ ") else {
            continue;
        };
        let Some((_, body)) = body.split_once("] ") else {
            continue;
        };
        if let Some(config) = body.strip_prefix("config in ") {
            let base = ratio_after(config, "time_base:")?;
            if base.0 == 0 || base.1 == 0 {
                return None;
            }
            match time_base {
                // A mid-stream reconfiguration onto a different time base would make the pts
                // before and after it incomparable.
                Some(previous) if previous != base => return None,
                Some(_) => segment_base = pts.len(),
                None => {
                    time_base = Some(base);
                    frame_rate = ratio_after(config, "frame_rate:");
                }
            }
        } else if let Some(frame) = body.strip_prefix("n:") {
            let frame = frame.trim_start();
            let index: usize = frame.split_whitespace().next()?.parse().ok()?;
            if segment_base + index != pts.len() {
                return None;
            }
            let value = frame.split_once("pts:")?.1.split_whitespace().next()?;
            // `NOPTS` (a frame the demuxer could not time) parses as `None`.
            pts.push(value.parse().ok());
        }
    }
    if pts.len() as u64 != frame_count {
        return None;
    }
    let (tb_num, tb_den) = time_base?;
    let frames = classify_seedvr2_timing((tb_num, tb_den), frame_rate, &pts)?;
    let start_micros = match pts.first().copied().flatten() {
        Some(first) if first > 0 => {
            let scaled = u128::from(first.unsigned_abs()) * u128::from(tb_num) * 1_000_000;
            let tb_den = u128::from(tb_den);
            u64::try_from((scaled + tb_den / 2) / tb_den).ok()?
        }
        _ => 0,
    };
    Some(Seedvr2SourceTiming {
        frames,
        start_micros,
    })
}

/// Decide how the measured frames are laid out in time (sc-24391). Constant-rate when every frame
/// sits within [`SEEDVR2_CFR_TOLERANCE_SECONDS`] of one uniform grid: first the rate the stream
/// declares (`frame_rate`, FFmpeg's own guess, which is the tidy `30000/1001` a millisecond-timebase
/// file cannot store exactly), then the grid through the first and last frames. Anything else is
/// variable-rate and keeps every frame's own timestamp. The declared rate is also the answer when
/// per-frame timing is unusable (a frame without a timestamp, timestamps that do not increase, or a
/// single frame).
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn classify_seedvr2_timing(
    time_base: (u64, u64),
    frame_rate: Option<(u64, u64)>,
    pts: &[Option<i64>],
) -> Option<Seedvr2Timing> {
    let (tb_num, tb_den) = time_base;
    if tb_num == 0 || tb_den == 0 {
        return None;
    }
    let declared =
        frame_rate.and_then(|(num, den)| reduced_ratio(u128::from(num), u128::from(den)));
    let declared_timing = declared.map(|(num, den)| Seedvr2Timing::Constant { num, den });

    let Some(pts) = pts.iter().copied().collect::<Option<Vec<i64>>>() else {
        return declared_timing;
    };
    let first = *pts.first()?;
    if pts.len() == 1 || pts.windows(2).any(|pair| pair[1] <= pair[0]) {
        return declared_timing;
    }
    let offsets: Vec<u64> = pts
        .iter()
        .map(|&p| u64::try_from(i128::from(p) - i128::from(first)).ok())
        .collect::<Option<_>>()?;

    let tick = tb_num as f64 / tb_den as f64;
    let tolerance = tick.max(SEEDVR2_CFR_TOLERANCE_SECONDS);
    let on_grid = |num: u64, den: u64| {
        let frame_seconds = den as f64 / num as f64;
        offsets.iter().enumerate().all(|(index, &offset)| {
            (offset as f64 * tick - index as f64 * frame_seconds).abs() <= tolerance
        })
    };
    if let Some((num, den)) = declared {
        if on_grid(num, den) {
            return declared_timing;
        }
    }
    let last = *offsets.last()?;
    if let Some((num, den)) = reduced_ratio(
        (offsets.len() as u128 - 1) * u128::from(tb_den),
        u128::from(last) * u128::from(tb_num),
    ) {
        if on_grid(num, den) {
            return Some(Seedvr2Timing::Constant { num, den });
        }
    }

    // Variable-rate: carry the offsets in the source's own ticks when those are no finer than a
    // microsecond, else in microseconds. `tb_num/tb_den` s per tick is `tb_num` ticks of `1/tb_den`.
    let timescale = tb_den.min(SEEDVR2_MAX_TIMESCALE);
    let rescaled: Vec<u64> = offsets
        .iter()
        .map(|&offset| {
            let scaled = u128::from(offset) * u128::from(tb_num) * u128::from(timescale);
            let tb_den = u128::from(tb_den);
            u64::try_from((scaled + tb_den / 2) / tb_den).ok()
        })
        .collect::<Option<_>>()?;
    if rescaled.windows(2).any(|pair| pair[1] <= pair[0]) {
        // Two frames collapsed onto one microsecond; per-frame timing cannot be written.
        return declared_timing;
    }
    // The final frame shows for as long as the one before it. The per-frame `duration` showinfo
    // prints comes from packet durations and is not trustworthy on variable-rate files (MEASURED:
    // 3000 ticks for a frame shown for 6000 on a 30-then-15 fps clip).
    let last_duration = rescaled[rescaled.len() - 1] - rescaled[rescaled.len() - 2];
    Some(Seedvr2Timing::Variable {
        timescale,
        offsets: rescaled,
        last_duration,
    })
}

/// The FFmpeg concat list that encodes a variable-rate clip with every frame at its own time
/// (sc-24391): one entry per numbered `frame_%05d.png`, each with the demuxer's time base raised to
/// the timing's timescale (`option framerate`) and its display duration. Durations are written as
/// the difference of each frame's start ROUNDED to the microsecond, not rounded one at a time, so
/// the rounding never accumulates: frame `i` lands within half a microsecond of its source time, and
/// FFmpeg's rescale to `1/timescale` puts it back on the exact source tick. `None` unless `timing`
/// is variable-rate and describes exactly `frame_count` frames.
#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
pub(super) fn seedvr2_concat_list(timing: &Seedvr2Timing, frame_count: usize) -> Option<String> {
    let Seedvr2Timing::Variable {
        timescale,
        offsets,
        last_duration,
    } = timing
    else {
        return None;
    };
    if offsets.len() != frame_count || *timescale == 0 {
        return None;
    }
    let micros = |ticks: u64| -> u64 {
        let timescale = u128::from(*timescale);
        ((u128::from(ticks) * 1_000_000 + timescale / 2) / timescale) as u64
    };
    let end = offsets.last()? + last_duration;
    let mut list = String::from("ffconcat version 1.0\n");
    for (index, &offset) in offsets.iter().enumerate() {
        let next = offsets.get(index + 1).copied().unwrap_or(end);
        let duration = micros(next) - micros(offset);
        list.push_str(&format!(
            "file frame_{index:05}.png\noption framerate {timescale}\nduration {}.{:06}\n",
            duration / 1_000_000,
            duration % 1_000_000
        ));
    }
    Some(list)
}

/// Load the temporal window `paths[start .. start+len]` (clamped to the sequence end) into engine
/// [`Image`]s on demand (sc-9595). Real frames only — the engine's `preprocess_chunk` pads a partial
/// trailing window with last-frame repeats internally, matching the whole-clip path. Runs the blocking
/// PNG decode off the async runtime.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn load_seedvr2_window(
    paths: &[PathBuf],
    start: usize,
    len: usize,
) -> WorkerResult<Vec<Image>> {
    let end = start.saturating_add(len).min(paths.len());
    let window: Vec<PathBuf> = paths.get(start..end).unwrap_or(&[]).to_vec();
    tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
        let mut frames = Vec::with_capacity(window.len());
        for path in window {
            let image = crate::image_decode::decode_image_any(&path)
                .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
                .to_rgb8();
            frames.push(rgb_image_to_engine(image));
        }
        Ok(frames)
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
}

/// Append RGB8 frames to a numbered PNG sequence on disk, starting at `next_index`, off the async
/// runtime (sc-9595). Returns the next free index. The shared frames dir is later encoded once by
/// [`encode_seedvr2_stream`] with the exact ffmpeg args the whole-clip `encode_media` used, so the
/// output is byte-identical while peak host RAM stays bounded to one worker window.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn append_seedvr2_frames(
    frames_dir: &Path,
    next_index: usize,
    frames: Vec<Image>,
) -> WorkerResult<usize> {
    if frames.is_empty() {
        return Ok(next_index);
    }
    let dir = frames_dir.to_path_buf();
    let count = frames.len();
    tokio::task::spawn_blocking(move || -> WorkerResult<()> {
        for (offset, frame) in frames.into_iter().enumerate() {
            let index = next_index + offset;
            let img = image::RgbImage::from_raw(frame.width, frame.height, frame.pixels)
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("video frame buffer size mismatch".to_owned())
                })?;
            let path = dir.join(format!("frame_{index:05}.png"));
            img.save_with_format(&path, image::ImageFormat::Png)
                .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
        }
        Ok(())
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;
    Ok(next_index + count)
}

/// Encode a pre-written numbered PNG sequence (`frame_%05d.png`, `frame_count` frames from index 0) to
/// the final mp4 (silent) + faststart + poster (sc-9595), laid out in time by the source's measured
/// `timing` (sc-24391). A constant-rate source encodes exactly as the whole-clip `encode_media` path
/// does (`libx264` / `yuv420p` / `-framerate` / `-r`), at the exact rational rate (`30000/1001`); a
/// whole rate keeps the bare-integer arguments it always had. A variable-rate source reads the
/// frames through a concat list ([`seedvr2_concat_list`]) that puts each one at its source time
/// relative to the first frame, and the encode passes those timestamps through in the source's own
/// time base. The picture starts at 0; the first frame's offset into the source is applied to the
/// audio by the caller's source passthrough mux ([`Seedvr2SourceTiming::start_micros`]; SeedVR2
/// emits no audio of its own).
///
/// `workflow_metadata` is the sanitized envelope as an `ffmetadata` document, or `None` for
/// "encode exactly as before" (sc-15956 review). It rides in HERE for the same reason
/// `encode_media` takes it: the tag is part of the file from the moment it exists, with no window
/// in which an upscaled clip is on disk without its recipe and no second pass over what can be
/// gigabytes. This is the SECOND libx264 site in the worker; the review that found it noted the
/// seam lint structurally cannot — it discovers by `WorkflowShare` mentions, and before this change
/// nothing in this file mentioned one.
///
/// **Not cfg-gated**, unlike the SeedVR2 engine work above it. It is pure ffmpeg plumbing over
/// ungated helpers, and gating it would put the one thing that has to be *proved* — that an
/// upscaled clip really carries its recipe — behind a platform no CI lane runs `cargo test` on.
/// The lane that calls it is still gated; the neither build allows the resulting dead code
/// explicitly rather than by making it unreachable to a test.
#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
pub(super) async fn encode_seedvr2_stream(
    media_path: &Path,
    frames_dir: &Path,
    frame_count: usize,
    timing: &Seedvr2Timing,
    workflow_metadata: Option<&Path>,
    ctx: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    if frame_count == 0 {
        return Err(WorkerError::InvalidPayload(
            "video generation produced no frames".to_owned(),
        ));
    }
    let enc_tmp = media_path.with_extension("enc.mp4");
    let mut args = vec!["ffmpeg".to_owned(), "-nostdin".to_owned(), "-y".to_owned()];
    // Input 0 (the frames) and the output options that time them.
    let timing_output_args = match timing {
        Seedvr2Timing::Constant { num, den } => {
            let rate = Seedvr2Timing::rate_arg(*num, *den);
            args.extend([
                "-framerate".to_owned(),
                rate.clone(),
                "-start_number".to_owned(),
                "0".to_owned(),
                "-i".to_owned(),
                frames_dir
                    .join("frame_%05d.png")
                    .to_string_lossy()
                    .into_owned(),
            ]);
            vec!["-r".to_owned(), rate]
        }
        Seedvr2Timing::Variable { timescale, .. } => {
            let list = seedvr2_concat_list(timing, frame_count).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "the measured source timing does not describe the {frame_count} upscaled frames"
                ))
            })?;
            let mean_rate = timing.variable_mean_rate().ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "the measured source timing has no picture length".to_owned(),
                )
            })?;
            let list_path = frames_dir.join("frames.ffconcat");
            tokio::fs::write(&list_path, list).await?;
            args.extend([
                "-f".to_owned(),
                "concat".to_owned(),
                "-safe".to_owned(),
                "0".to_owned(),
                "-i".to_owned(),
                list_path.to_string_lossy().into_owned(),
            ]);
            vec![
                "-fps_mode".to_owned(),
                "passthrough".to_owned(),
                "-enc_time_base".to_owned(),
                format!("1/{timescale}"),
                "-video_track_timescale".to_owned(),
                timescale.to_string(),
                // Size the H.264 level for the clip's real mean rate, not the time base
                // (`Seedvr2Timing::variable_mean_rate`). Timestamps still pass through untouched.
                "-x264-params".to_owned(),
                format!("fps={mean_rate}"),
            ]
        }
    };
    if let Some(metadata_path) = workflow_metadata {
        // Input 1, exactly as `encode_media` does it: the frames are input 0 and stay mapped by
        // ffmpeg's own stream selection, because an `ffmetadata` input carries no streams to
        // compete with.
        args.extend(sceneworks_core::workflow_mp4::ffmetadata_input_args(
            metadata_path,
        ));
    }
    args.extend([
        "-c:v".to_owned(),
        "libx264".to_owned(),
        "-pix_fmt".to_owned(),
        "yuv420p".to_owned(),
    ]);
    args.extend(timing_output_args);
    if workflow_metadata.is_some() {
        args.extend(sceneworks_core::workflow_mp4::ffmetadata_map_args(1));
    }
    args.push(enc_tmp.to_string_lossy().into_owned());
    let result = run_ffmpeg(args, ctx).await;
    match result {
        Ok(()) => {
            // Publish atomically, then best-effort faststart + poster (mirrors `encode_media`).
            tokio::fs::rename(&enc_tmp, media_path).await?;
            faststart_mp4(media_path).await;
            write_poster_frame(media_path).await;
            Ok(())
        }
        Err(error) => {
            let _ = tokio::fs::remove_file(&enc_tmp).await;
            let _ = tokio::fs::remove_file(media_path).await;
            Err(error)
        }
    }
}

/// Build the sanitized workflow envelope for the UPSCALED clip and write it beside the media as an
/// `ffmetadata` document, returning whether the encoder should attach it (sc-15956 review).
///
/// **The video-upscale write seam** — the exact counterpart of
/// `upscale_jobs::run_image_upscale_job`'s `standalone_upscale_workflow_share` call, and declared in
/// `WORKFLOW_WRITE_SEAMS` for the same reason. sc-15956 shipped with the seam prose claiming
/// `encode_media` was "the ONE funnel every generated clip is encoded through". It was not: this
/// file has a libx264 site of its own, and it embedded nothing, while the image lane's analogue
/// embedded. The two lanes now make the same call.
///
/// # A distinct job, so a distinct envelope
///
/// A video upscale is its own job with its own payload, and the envelope describes THIS pass — it
/// inherits nothing from whatever generated the source clip, which is the rule sc-15948 set for the
/// image upscale and the same rule `media_jobs`'s export refuses to break in the other direction.
/// The source clip's own embedded recipe is not read, not copied and not merged: an upscale of
/// somebody else's video must not acquire their prompt.
///
/// Three overlays onto the job payload, and nothing else:
///
/// * `upscale` — the APPLIED engine, factor and (for the one engine that has the knob) softness,
///   validated above rather than the requested values. It is the only record of what this pass did;
/// * `sourceClipAssetId` replaces `sourceAssetId`, so `describe_inputs` records the shape as
///   [`INPUT_KIND_SOURCE_CLIP`](sceneworks_core::workflow_share::INPUT_KIND_SOURCE_CLIP) — "this
///   recipe needs a CLIP to start from" — rather than as a still. The id itself never travels
///   either way; only the kind and the count do;
/// * the geometry is the SOURCE geometry, for the reason the image lane states: the envelope is a
///   recipe, and "take this clip and upscale it 4x" replays to this file where the upscaled
///   dimensions would replay to something four times larger again.
///
/// `displayName` is in the payload and is not an envelope field, so the source clip's file name —
/// routinely a person's name — is left behind by the field list being closed. `prompt` is empty:
/// an upscale has none.
///
/// Returns `false` for the same three reasons `video_jobs::video_workflow_metadata` does, logged at
/// `debug` for the same reason, and a write failure degrades to no-workflow rather than failing a
/// clip that upscaled fine.
///
/// **Not cfg-gated**, for the reason [`encode_seedvr2_stream`] is not: this is the function whose
/// OUTPUT the acceptance criterion is about, and a test that cannot call it proves nothing.
#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
pub(super) fn seedvr2_workflow_metadata(
    settings: &Settings,
    job: &JobSnapshot,
    req: &sceneworks_core::contracts::VideoUpscaleRequest,
    engine_id: &str,
    factor: u32,
    seed: i64,
    src_w: u32,
    src_h: u32,
    metadata_path: &Path,
) -> bool {
    if job.payload.is_empty() {
        tracing::debug!(
            reason = "empty_job_payload",
            "not embedding a workflow: the job carries no payload to describe"
        );
        return false;
    }
    if !sceneworks_core::app_paths::embed_workflow_in_images(&settings.config_dir) {
        tracing::debug!(
            reason = "preference_off",
            config_dir = %settings.config_dir.display(),
            "not embedding a workflow: `embedWorkflowInImages` did not resolve to true"
        );
        return false;
    }
    let mut upscale = json!({ "enabled": true, "engine": engine_id, "factor": factor });
    if engine_id == "seedvr2" {
        // SeedVR2 is a generative one-step upscaler, so its detail knob changes the output. The
        // image lane records it for the same reason and for the same engine only.
        upscale["softness"] = json!(req.softness);
    }
    let mut overlay = job.payload.clone();
    overlay.insert("upscale".to_owned(), upscale);
    if let Some(source) = overlay.remove("sourceAssetId") {
        overlay.insert("sourceClipAssetId".to_owned(), source);
    }
    let facts = sceneworks_core::workflow_share::WorkflowAssetFacts {
        mode: "video_upscale".to_owned(),
        // The payload's own `model` (`seedvr2_3b`) wins inside the builder, so this is the
        // fallback for a payload that somehow has none. Either way it agrees with the sidecar
        // fact's `model`, which is the property that keeps a shared file and its record honest.
        model: req.model.clone(),
        prompt: String::new(),
        negative_prompt: String::new(),
        seed,
        width: Some(src_w),
        height: Some(src_h),
    };
    let Some(share) =
        sceneworks_core::workflow_share::embeddable_video_workflow_share(&facts, &overlay)
    else {
        tracing::debug!(
            reason = "over_recording_ceiling",
            "not embedding a workflow: the envelope is larger than the recording ceiling"
        );
        return false;
    };
    match sceneworks_core::workflow_mp4::write_workflow_metadata_file(&share, metadata_path) {
        Ok(()) => {
            tracing::debug!("embedding the sanitized workflow in the upscaled clip");
            true
        }
        Err(error) => {
            tracing::warn!(
                reason = "metadata_write_failed",
                %error,
                "not embedding a workflow: the metadata document could not be written"
            );
            false
        }
    }
}

/// Result of the streamed SeedVR2 upscale (sc-9595): the metadata the caller needs to build the asset
/// fact + encode the pre-written PNG sequence. The upscaled frames themselves are already on disk
/// (`out_frames_dir`), never held whole in RAM.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct Seedvr2Stream {
    /// Real output frame count (== source frame count).
    frame_count: usize,
    /// When those frames are shown: the source's own measured timing (sc-24391).
    timing: Seedvr2SourceTiming,
    out_w: u32,
    out_h: u32,
    src_w: u32,
    src_h: u32,
    seed: u64,
}

/// RAII guard for a worker-owned scratch directory (sc-9595). The streamed 4× PNG sequence can be many
/// GB, and it now outlives the stream call while the caller runs an `update_job` progress POST + an
/// `ffmpeg` encode — a span where any `?` (a transient POST failure, a 409 stale-sweep reclaim, an
/// encode error, or a between-step cancel) would otherwise leak the whole dir on disk. `Drop` removes
/// it on EVERY exit path (success, encode/create_dir_all/update_job failure, cancel, panic) so cleanup
/// can never be skipped. Call [`ScratchDir::disarm`] after the caller has already removed the dir to
/// avoid a redundant (harmless) second removal. `Drop` must use the sync `std::fs` API.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) struct ScratchDir {
    path: std::path::PathBuf,
    armed: bool,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl ScratchDir {
    /// Guard `path`. Does not create it — the caller populates it; the guard only guarantees removal.
    pub(super) fn new(path: std::path::PathBuf) -> Self {
        Self { path, armed: true }
    }

    /// Stop guarding: the caller has already removed the dir (e.g. right after a successful encode, to
    /// free disk before the mux step). A no-op `Drop` follows.
    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl Drop for ScratchDir {
    fn drop(&mut self) {
        if self.armed {
            // Best-effort: a missing dir yields a benign NotFound we intentionally ignore.
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// Stream the SeedVR2 upscale in temporal worker windows (sc-9595). Decodes the source to disk, plans
/// worker windows over the real frame count with the engine's `video::plan_chunks`
/// (`SEEDVR2_WORKER_CHUNK_FRAMES` + `DEFAULT_OVERLAP`), upscales ONE window at a time through the shared
/// `generate_video` funnel, cross-fades across worker-window boundaries with the engine's own
/// `video::assemble_overlap`, and appends each finalized frame to a numbered PNG sequence on disk. Peak
/// host RGB8 RAM is bounded to ~one worker window + a ≤`ov`-frame tail. Each per-window `generate_video`
/// call is independently heartbeat-covered + cancellable (the funnel's watchdog), and cancel is polled
/// between windows too — no long silent blocking span.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn run_seedvr2_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    backend: &str,
    req: &sceneworks_core::contracts::VideoUpscaleRequest,
    source_path: &Path,
    src_frames_dir: &Path,
    out_frames_dir: &Path,
    factor: u32,
    weights_dir: PathBuf,
) -> WorkerResult<Seedvr2Stream> {
    let seed = req.seed.unwrap_or(0);
    // Probe BEFORE creating either PNG sequence. The source and output sequences coexist at peak, so
    // admission must account for both before the first source frame can consume scratch. The same
    // pass measures the frame timing the output keeps (sc-24391).
    let (probe, timing) = probe_seedvr2_source(api, settings, &job.id, source_path).await?;
    let engine_fps = timing.frames.nominal_fps(probe.frame_count as usize);
    let (target_w, target_h) = match (req.target_width, req.target_height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
        _ => (
            probe.width.saturating_mul(factor),
            probe.height.saturating_mul(factor),
        ),
    };
    let target_w = snap_seedvr2_dim(target_w);
    let target_h = snap_seedvr2_dim(target_h);
    {
        let guard_dir = out_frames_dir
            .parent()
            .unwrap_or(out_frames_dir)
            .to_path_buf();
        let p = probe;
        tokio::task::spawn_blocking(move || {
            check_seedvr2_scratch_disk(
                &guard_dir,
                p.frame_count,
                u64::from(p.width),
                u64::from(p.height),
                u64::from(target_w),
                u64::from(target_h),
            )
        })
        .await
        .map_err(|error| task_join_error("seedvr2 disk preflight", error))??;
    }

    // Only an admitted job may materialize the native-resolution source PNG sequence.
    let paths =
        decode_seedvr2_source_to_disk(api, settings, &job.id, source_path, src_frames_dir).await?;
    let n = paths.len();
    if n as u64 != probe.frame_count {
        return Err(WorkerError::InvalidPayload(format!(
            "Source video changed while preparing upscale (probed {} frames, decoded {n}). Retry the job.",
            probe.frame_count
        )));
    }
    let first = load_seedvr2_window(&paths, 0, 1).await?;
    let src_w = first[0].width;
    let src_h = first[0].height;
    drop(first);
    if (src_w, src_h) != (probe.width, probe.height) {
        return Err(WorkerError::InvalidPayload(
            "Source video dimensions changed while preparing upscale. Retry the job.".to_owned(),
        ));
    }

    tokio::fs::create_dir_all(out_frames_dir).await?;

    // Plan the worker windows over the REAL frame count with the engine's own planner: a valid chunk
    // length + `DEFAULT_OVERLAP` cross-fade, so the worker-window seam handling matches the engine's
    // internal chunking exactly.
    let ov = seedvr2_video::DEFAULT_OVERLAP;
    let plan = seedvr2_video::plan_chunks(n as i32, SEEDVR2_WORKER_CHUNK_FRAMES, ov);
    let mut assembler = Seedvr2StreamAssembler::new(n as i32, ov);
    let mut next_index = 0usize;
    let mut out_w = target_w;
    let mut out_h = target_h;
    let window_total = plan.len().max(1);

    for (window_idx, chunk) in plan.iter().enumerate() {
        check_cancel(api, &job.id, SEEDVR2_CANCEL_MESSAGE).await?;
        // Real frames only for this window; the engine's preprocess_chunk pads a partial tail internally.
        let start = chunk.start.max(0) as usize;
        if start >= n {
            break;
        }
        let len = chunk.len.max(0) as usize;
        let window_frames = load_seedvr2_window(&paths, start, len).await?;
        if window_frames.is_empty() {
            continue;
        }
        let window_len = window_frames.len() as u32;

        // Per-window progress spanning the Generating band (0.18 → 0.55) so the bar advances per window
        // even though each window's own denoise progress is reported inside `generate_video`.
        let frac = 0.18 + 0.37 * (window_idx as f64 / window_total as f64);
        update_job(
            api,
            &job.id,
            video_progress(
                JobStatus::Running,
                ProgressStage::Generating,
                frac,
                &format!("Upscaling window {}/{window_total}.", window_idx + 1),
                None,
                backend,
            ),
        )
        .await?;

        // Upscale this window through the shared streaming driver (generator cache + stall watchdog +
        // cancel + per-step progress). Same seed for every window → deterministic per-chunk blend.
        let input = VideoGenInput {
            engine_id: SEEDVR2_ENGINE_ID,
            model_dir: weights_dir.clone(),
            conditioning: vec![Conditioning::VideoClip {
                frames: window_frames,
                frame_idx: 0,
                strength: 1.0,
            }],
            width: target_w,
            height: target_h,
            frames: window_len,
            fps: engine_fps,
            seed,
            softness: Some(req.softness),
            ..Default::default()
        };
        // SeedVR2 upscale is one-step with no per-generation sampler/scheduler knobs, so the
        // advanced block generate_video reads for those is empty here (F-118).
        let decoded =
            generate_video(api, settings, job, backend, &JsonObject::new(), input).await?;
        if let Some(frame) = decoded.frames.first() {
            out_w = frame.width;
            out_h = frame.height;
        }
        let upscaled: Vec<Image> = decoded
            .frames
            .into_iter()
            .map(|frame| Image {
                width: frame.width,
                height: frame.height,
                pixels: frame.pixels,
            })
            .collect();

        // Cross-fade this window against the retained tail (the engine's own blend) and stream the
        // now-finalized frames to disk.
        let finalized = assembler.push_window(chunk.start, upscaled);
        next_index = append_seedvr2_frames(out_frames_dir, next_index, finalized).await?;
    }

    // Flush the final tail.
    let tail = assembler.finish();
    next_index = append_seedvr2_frames(out_frames_dir, next_index, tail).await?;

    if next_index == 0 {
        return Err(WorkerError::InvalidPayload(
            "source video produced no frames to upscale".to_owned(),
        ));
    }

    Ok(Seedvr2Stream {
        frame_count: next_index,
        timing,
        out_w,
        out_h,
        src_w,
        src_h,
        seed,
    })
}

/// Validate a requested video-upscale factor. SeedVR2 supports only 2x and 4x, so any other value
/// (3x, 8x, 1x, 0) is rejected with a clear error rather than silently coerced (F-118). Returns the
/// factor widened to `u32` for the downstream dimension math.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_video_upscale_factor(factor: u8) -> WorkerResult<u32> {
    match factor {
        2 | 4 => Ok(u32::from(factor)),
        other => Err(WorkerError::InvalidPayload(format!(
            "Video upscale supports only factor 2 or 4 (got {other})."
        ))),
    }
}

/// The source-audio passthrough mux: copy the upscaled picture untouched, re-encode the source
/// clip's optional audio as AAC, and **bound the file at the upscaled picture's own length** —
/// `timing`'s picture length, spelled by [`format_picture_bound`] exactly as the generation mux
/// (`audio_mux_args`) spells its bound. `-map 1:a:0?` keeps the audio optional, so a silent source
/// yields a clean video-only file rather than an error.
///
/// # Why `-t` and not `-shortest` (sc-19549)
///
/// This path muxes the user's OWN source audio onto the user's OWN upscaled picture, so `-shortest`
/// spends the user's frames to satisfy a soundtrack they supplied and never asked to be
/// authoritative. MEASURED on this exact argument vector with the bundled ffmpeg 7.1, a 48-frame
/// 24 fps picture (2.000000 s) against a source clip carrying 1.5 s of audio:
///
/// | flag | frames kept | container `Duration:` |
/// |---|---|---|
/// | `-shortest` | **33 of 48** | 1.51 s — for 1.375 s of picture |
/// | none | 48 | 2.00 s |
/// | `-t 2.000000` | 48 | 2.00 s |
///
/// Fifteen frames gone, and the container went on advertising a duration the file does not have —
/// so a check of the reported duration passes on the damaged file and only a decoded frame count
/// sees it. Dropping the flag outright fixes that direction and loses the other: with 4.0 s of
/// source audio the unbounded command produced a 4.01 s container around 2.00 s of picture
/// (measured, same vector). `-t` is the only one of the three that is right in both directions.
///
/// # Source timing (sc-24391)
///
/// The bound is the length of input 0, the upscaled picture, as `encode_seedvr2_stream` laid it out
/// from the source's measured `timing.frames`: exactly `frame_count` frames, on the source's exact
/// constant rate or each at its own time relative to the first. That picture starts at 0 wherever the
/// source's first frame sat, so when that frame starts `timing.start_micros` into the source, the
/// same span of the source's audio is skipped (`-ss` on input 1) to keep sound on its frames. Input 1
/// contributes only `0:a` here; its video stream is not mapped. (The sc-19549 measurements above
/// were taken when the picture was re-timed to a constant rate; the bound policy is unchanged.)
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn seedvr2_audio_mux_args(
    upscaled: &Path,
    source: &Path,
    out: &Path,
    frame_count: usize,
    timing: &Seedvr2SourceTiming,
) -> Vec<String> {
    let mut source_input = Vec::new();
    if timing.start_micros > 0 {
        source_input.extend([
            "-ss".to_owned(),
            format!(
                "{}.{:06}",
                timing.start_micros / 1_000_000,
                timing.start_micros % 1_000_000
            ),
        ]);
    }
    source_input.extend(["-i".to_owned(), source.to_string_lossy().into_owned()]);
    let mut args = vec![
        "ffmpeg".to_owned(),
        "-nostdin".to_owned(),
        "-y".to_owned(),
        "-i".to_owned(),
        upscaled.to_string_lossy().into_owned(),
    ];
    args.extend(source_input);
    args.extend([
        "-map".to_owned(),
        "0:v:0".to_owned(),
        "-map".to_owned(),
        "1:a:0?".to_owned(),
        "-c:v".to_owned(),
        "copy".to_owned(),
        "-c:a".to_owned(),
        "aac".to_owned(),
        "-t".to_owned(),
        format_picture_bound(timing.frames.picture_seconds(frame_count)),
        // Explicit, though it is also ffmpeg's default for a multi-input command: the container
        // metadata — including the sc-15956 workflow tag the encode above wrote — comes from the
        // UPSCALED VIDEO (input 0), never from the source clip (input 1). Input 1 is the user's own
        // source file and may carry container tags of its own; inheriting those is the failure
        // `media_jobs`'s export exists to refuse. Stated because "the default happens to be right"
        // is not a property anyone maintains.
        "-map_metadata".to_owned(),
        "0".to_owned(),
        "-movflags".to_owned(),
        "+faststart".to_owned(),
        out.to_string_lossy().into_owned(),
    ]);
    args
}

/// Dispatch handler for `JobType::VideoUpscale`: decode the source clip, run the SeedVR2 upscaler
/// (native MLX on Mac / candle CUDA on Windows, sc-5928), re-encode, pass the source audio through,
/// and stream a single upscaled video asset.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_video_upscale_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let req: sceneworks_core::contracts::VideoUpscaleRequest =
        serde_json::from_value(Value::Object(job.payload.clone())).map_err(|error| {
            WorkerError::InvalidPayload(format!("Invalid video_upscale payload: {error}"))
        })?;
    if req.source_asset_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Video upscale jobs require a source video asset.".to_owned(),
        ));
    }
    let engine = req.engine.trim().to_ascii_lowercase();
    if !matches!(engine.as_str(), "" | "seedvr2" | "seedvr2_3b") {
        return Err(WorkerError::InvalidPayload(format!(
            "This video upscaler supports only engine=seedvr2 (got {engine})."
        )));
    }
    let project_id = req
        .project_id
        .clone()
        .or_else(|| job.project_id.clone())
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| WorkerError::InvalidPayload("Missing payload.projectId".to_owned()))?;
    // Reject an unsupported factor early rather than silently coercing it to 2 (F-118).
    let factor = resolve_video_upscale_factor(req.factor)?;
    let backend = backend_label(&settings.gpu_id);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Loading source video.",
            None,
            backend,
        ),
    )
    .await?;

    // Resolve the source video asset (on-disk path + display name) from its sidecar. Its `fps` is
    // deliberately NOT read: an imported clip has none, and the probe measures the real timing off
    // the file itself (sc-24391).
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(&project_id)?;
    let project_path = PathBuf::from(project.path);
    let asset = store
        .get_asset(&project_id, &req.source_asset_id)
        .map_err(|_| WorkerError::InvalidPayload("Source video asset not found.".to_owned()))?;
    let file = asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Source asset has no media file.".to_owned()))?;
    let rel = file.get("path").and_then(Value::as_str).ok_or_else(|| {
        WorkerError::InvalidPayload("Source asset media path missing.".to_owned())
    })?;
    let source_path = safe_join(&project_path, rel)
        .filter(|path| path.exists())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Source media file is unavailable.".to_owned())
        })?;
    let source_display = asset
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned);

    // sc-9595: no host-RAM cap. The upscale streams in temporal worker windows (decode → upscale →
    // append-encode one window at a time), so peak host RGB8 RAM is bounded to ~one window regardless
    // of clip length — the sc-8829 whole-clip frame cap that rejected long/high-res clips is gone.

    check_cancel(api, &job.id, SEEDVR2_CANCEL_MESSAGE).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Loading SeedVR2 weights.",
            None,
            backend,
        ),
    )
    .await?;
    // Cache-only since sc-17632: the SAME resolver the image-upscale lane uses, so the checkpoint
    // is installed once from the Model Manager (`seedvr2_upscaler`) and read from the HF cache — no
    // 7.3 GB fetch inside the render, and no second `<data_dir>/cache` copy of the same repo.
    let weights_dir = crate::upscale_jobs::require_seedvr2_checkpoint_dir(settings)?;

    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.18,
            "Decoding source frames.",
            None,
            backend,
        ),
    )
    .await?;
    // Decode the whole source to a numbered PNG sequence ON DISK (disk-bounded, no RAM), then load each
    // temporal window on demand (sc-9595). `-fps_mode passthrough` preserves the exact source frame
    // count / order; the output keeps that count and order, and the probe's measured timing (sc-24391).
    // Sanitize the job id before it becomes a temp-dir path component (F-111): a hostile id would
    // otherwise escape `temp_dir()`. Mirrors the person-track work dir sanitization.
    let safe_job = safe_download_dir(&job.id);
    let src_frames_dir = std::env::temp_dir().join(format!("sceneworks_seedvr2_src_{safe_job}"));
    let out_frames_dir = std::env::temp_dir().join(format!("sceneworks_seedvr2_out_{safe_job}"));
    let _ = tokio::fs::remove_dir_all(&out_frames_dir).await;
    // RAII-guard the output PNG scratch so it is removed on EVERY exit after the stream: not just the
    // stream error arm, but the create_dir_all / update_job progress POST / encode span below, any of
    // which can `?`-return (transient POST failure, 409 stale-sweep reclaim, encode error, cancel).
    // Without this the full multi-GB 4× sequence would leak on those paths (sc-9595 review).
    let mut out_scratch = ScratchDir::new(out_frames_dir.clone());
    let stream_result = run_seedvr2_stream(
        api,
        settings,
        job,
        backend,
        &req,
        &source_path,
        &src_frames_dir,
        &out_frames_dir,
        factor,
        weights_dir,
    )
    .await;
    // Always drop the source PNG scratch (it's disk-only; the output dir is owned by `out_scratch`).
    let _ = tokio::fs::remove_dir_all(&src_frames_dir).await;
    // On stream failure, `out_scratch`'s Drop removes the output dir as the function returns.
    let stream = stream_result?;
    let Seedvr2Stream {
        frame_count: out_count,
        timing,
        out_w,
        out_h,
        src_w,
        src_h,
        seed,
    } = stream;
    let duration = timing.frames.picture_seconds(out_count);

    // Plan the output asset path (nested under the per-generation id, like VideoPlan).
    let genset_id = format!("genset_{}", Uuid::new_v4().simple());
    let asset_id = fresh_asset_id();
    let created_at = now_rfc3339();
    let media_rel = format!(
        "assets/videos/{genset_id}/{}_seedvr2_upscale.mp4",
        &created_at[..10]
    );
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Muxing,
            0.6,
            "Encoding upscaled video.",
            None,
            backend,
        ),
    )
    .await?;
    // Encode the streamed PNG sequence to a (silent) mp4 + poster + faststart (byte-identical ffmpeg
    // args to the old whole-clip `encode_media` path), then drop the output scratch dir.
    //
    // sc-15956 review: the sanitized workflow for THIS pass, written beside the clip for the
    // encoder to read. Written here rather than earlier so the only thing between the document
    // appearing and being removed is the encode itself.
    let workflow_metadata = media_path.with_extension("workflow.ffmeta");
    let embedded = seedvr2_workflow_metadata(
        settings,
        job,
        &req,
        "seedvr2",
        factor,
        seed as i64,
        src_w,
        src_h,
        &workflow_metadata,
    );
    let ctx = FfmpegContext::new(api, settings, &job.id, SEEDVR2_CANCEL_MESSAGE);
    let encode_result = encode_seedvr2_stream(
        &media_path,
        &out_frames_dir,
        out_count,
        &timing.frames,
        embedded.then_some(workflow_metadata.as_path()),
        Some(ctx),
    )
    .await;
    // Free the multi-GB PNG scratch as soon as the encode returns (before the mux step), on BOTH the
    // ok and err arms; then disarm the guard so its Drop doesn't redundantly re-remove. `encode_result`
    // is propagated AFTER cleanup — an encode error still leaves no scratch behind (and if this early
    // removal is itself skipped by an unwind, the still-armed guard's Drop is the backstop).
    let _ = tokio::fs::remove_dir_all(&out_frames_dir).await;
    out_scratch.disarm();
    // The metadata document goes on every path too, and before the `?`. It holds the whole envelope
    // with the prompt in plaintext, and a failed or cancelled encode must not leave one sitting in
    // the user's project beside no video.
    let _ = tokio::fs::remove_file(&workflow_metadata).await;
    encode_result?;
    let mux_tmp = media_path.with_extension("audiomux.mp4");
    let mut output_guard = Seedvr2OutputGuard::new(&media_path, &mux_tmp);

    // Source-audio passthrough: remux the source's audio onto the upscaled video, bounded at the
    // UPSCALED PICTURE's own length — see `seedvr2_audio_mux_args`.
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Muxing,
            0.85,
            "Muxing source audio.",
            None,
            backend,
        ),
    )
    .await?;
    let ctx = FfmpegContext::new(api, settings, &job.id, SEEDVR2_CANCEL_MESSAGE);
    let mux_result: WorkerResult<()> = async {
        run_ffmpeg(
            seedvr2_audio_mux_args(&media_path, &source_path, &mux_tmp, out_count, &timing),
            Some(ctx),
        )
        .await?;
        tokio::fs::rename(&mux_tmp, &media_path).await?;
        Ok(())
    }
    .await;
    if let Err(error) = mux_result {
        cleanup_failed_seedvr2_mux(&media_path, &mux_tmp).await;
        return Err(error);
    }

    let display_name = req
        .display_name
        .clone()
        .unwrap_or_else(|| match &source_display {
            Some(name) => format!("{name} ({factor}x upscaled)"),
            None => format!("Upscaled video ({factor}x)"),
        });
    let raw_settings = json!({
        "engine": "seedvr2",
        "model": req.model,
        "factor": factor,
        "softness": req.softness,
        "sourceAssetId": req.source_asset_id,
        "sourceWidth": src_w,
        "sourceHeight": src_h,
        "targetWidth": out_w,
        "targetHeight": out_h,
        "frameCount": out_count,
    });
    let fact = json!({
        "type": "video",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "video/mp4",
        "width": out_w,
        "height": out_h,
        "duration": duration,
        "fps": timing.frames.fps_json(out_count),
        "quality": "best",
        "family": "video",
        "seed": seed as i64,
        "displayName": display_name,
        "createdAt": created_at,
        "mode": "video_upscale",
        "model": req.model,
        "adapter": SEEDVR2_ADAPTER,
        "prompt": "",
        "negativePrompt": Value::Null,
        "loras": [],
        "rawAdapterSettings": raw_settings,
        "sourceAssetId": req.source_asset_id,
        "parents": [req.source_asset_id],
        "extra": {
            "isUpscaled": true,
            "upscaledFromAssetId": req.source_asset_id,
            "factor": factor,
            "engine": "seedvr2",
        },
        "timelineContext": json!({}),
    });
    let result = json!({
        "generationSetId": genset_id,
        "expectedCount": 1,
        "adapter": SEEDVR2_ADAPTER,
        "model": req.model,
        "generationSet": {
            "id": genset_id,
            "mode": "video_upscale",
            "model": req.model,
            "prompt": "",
            "negativePrompt": Value::Null,
            "count": 1,
            "createdAt": created_at,
        },
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal");

    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Video upscale complete.",
            Some(result),
            backend,
        ),
    )
    .await?;
    output_guard.disarm();
    Ok(())
}
