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
// -> encode + source-audio passthrough. macOS-only (no torch path). Reuses the shared
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

/// HF repo hosting the raw SeedVR2 checkpoint (`numz/SeedVR2_comfyUI`); the engine converts it
/// in-memory at load (no Python). Override the staged dir with `SCENEWORKS_SEEDVR2_DIR`.
// SeedVR2 video upscale runs on Mac (native MLX) AND the Windows/CUDA candle lane (sc-5928); these
// constants/helpers are backend-neutral (gen_core + ffmpeg + the shared streaming driver).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const SEEDVR2_REPO: &str = "numz/SeedVR2_comfyUI";
/// Pinned SeedVR2 checkpoint revision (sc-8879 / sc-9879). `numz/SeedVR2_comfyUI` is a
/// third-party mirror with a fixed (non-overridable) repo here — fetching the mutable `main`
/// branch would let an upstream re-push silently swap the 3B fp16 DiT + VAE weights we load.
/// Pin the exact commit so downloads are reproducible; HF's tree API still reports each file's
/// `lfs.oid`, which `ensure_hf_cached_file` verifies the content against. MUST equal the
/// image-upscale lane's `upscale_jobs::SEEDVR2_REVISION` (same repo + files) — the
/// `seedvr2_video_revision_matches_image_lane` agreement test locks them together so they
/// can't drift.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const SEEDVR2_REVISION: &str = "09ced71023636e9bc8cdf9cdecfb2625d1e691e8";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_VAE_FILE: &str = "ema_vae_fp16.safetensors";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_DIT_3B_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";
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

/// Provision the SeedVR2 checkpoint dir: an env-pinned dir (pre-staged for local validation) wins,
/// else the app cache (download the VAE + 3B DiT from `numz/SeedVR2_comfyUI` on first use). Returns
/// the dir to hand the engine as `WeightsSource::Dir`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn ensure_seedvr2_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let dir = std::env::var("SCENEWORKS_SEEDVR2_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("seedvr2-mlx"));
    let client = crate::downloads::streaming_download_client();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "Video upscale canceled while fetching SeedVR2 weights.",
        fresh_download: false,
    };
    for file in [SEEDVR2_VAE_FILE, SEEDVR2_DIT_3B_FILE] {
        crate::downloads::ensure_hf_cached_file(
            &context,
            SEEDVR2_REPO,
            SEEDVR2_REVISION,
            file,
            &dir.join(file),
        )
        .await?;
    }
    Ok(dir)
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
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            source.to_string_lossy().into_owned(),
            "-fps_mode".to_owned(),
            "passthrough".to_owned(),
            frames_dir
                .join("in_%05d.png")
                .to_string_lossy()
                .into_owned(),
        ],
        Some(ctx),
    )
    .await?;
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

/// Decode-free, VFR-safe source probe using the bundled FFmpeg null mux. The shared runner preserves
/// the normal heartbeat/cancellation lifecycle while FFmpeg walks the source.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn probe_seedvr2_source(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    source: &Path,
) -> WorkerResult<Seedvr2SourceProbe> {
    let ctx = FfmpegContext::new(api, settings, job_id, SEEDVR2_CANCEL_MESSAGE);
    let stderr = crate::media_jobs::run_ffmpeg_capture_stderr(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-hide_banner".to_owned(),
            "-i".to_owned(),
            source.to_string_lossy().into_owned(),
            "-map".to_owned(),
            "0:v:0".to_owned(),
            "-f".to_owned(),
            "null".to_owned(),
            "-".to_owned(),
        ],
        Some(ctx),
    )
    .await?;
    parse_seedvr2_source_probe(&stderr).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Could not determine the source video's frame count and dimensions before upscale."
                .to_owned(),
        )
    })
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
/// the final mp4 (silent) + faststart + poster (sc-9595). The ffmpeg encode args are byte-identical to
/// the whole-clip `encode_media` path (`libx264` / `yuv420p` / `-framerate fps` / `-r fps`), so the
/// streamed output matches the old path frame-for-frame. Audio muxing stays the caller's source
/// passthrough step (SeedVR2 emits no audio).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn encode_seedvr2_stream(
    media_path: &Path,
    frames_dir: &Path,
    frame_count: usize,
    fps: u32,
    ctx: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    if frame_count == 0 {
        return Err(WorkerError::InvalidPayload(
            "video generation produced no frames".to_owned(),
        ));
    }
    let fps = fps.max(1);
    let enc_tmp = media_path.with_extension("enc.mp4");
    let pattern = frames_dir.join("frame_%05d.png");
    let result = run_ffmpeg(
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
    .await;
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
    fps: u32,
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
    source_fps: u32,
    weights_dir: PathBuf,
) -> WorkerResult<Seedvr2Stream> {
    let seed = req.seed.unwrap_or(0);
    // Probe BEFORE creating either PNG sequence. The source and output sequences coexist at peak, so
    // admission must account for both before the first source frame can consume scratch.
    let probe = probe_seedvr2_source(api, settings, &job.id, source_path).await?;
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
            fps: source_fps,
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
        fps: source_fps,
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

    // Resolve the source video asset (on-disk path + fps + display name) from its sidecar.
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
    let source_fps = file
        .get("fps")
        .and_then(Value::as_f64)
        .map(|fps| fps.round() as u32)
        .filter(|fps| *fps > 0)
        .unwrap_or(24);
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
            "Fetching SeedVR2 weights.",
            None,
            backend,
        ),
    )
    .await?;
    let weights_dir = ensure_seedvr2_weights(api, settings, job).await?;

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
    // count / order; the output frame count/order/fps therefore stay identical to the whole-clip path.
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
        source_fps,
        weights_dir,
    )
    .await;
    // Always drop the source PNG scratch (it's disk-only; the output dir is owned by `out_scratch`).
    let _ = tokio::fs::remove_dir_all(&src_frames_dir).await;
    // On stream failure, `out_scratch`'s Drop removes the output dir as the function returns.
    let stream = stream_result?;
    let Seedvr2Stream {
        frame_count: out_count,
        fps: out_fps,
        out_w,
        out_h,
        src_w,
        src_h,
        seed,
    } = stream;
    let duration = out_count as f64 / out_fps.max(1) as f64;

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
    let ctx = FfmpegContext::new(api, settings, &job.id, SEEDVR2_CANCEL_MESSAGE);
    let encode_result =
        encode_seedvr2_stream(&media_path, &out_frames_dir, out_count, out_fps, Some(ctx)).await;
    // Free the multi-GB PNG scratch as soon as the encode returns (before the mux step), on BOTH the
    // ok and err arms; then disarm the guard so its Drop doesn't redundantly re-remove. `encode_result`
    // is propagated AFTER cleanup — an encode error still leaves no scratch behind (and if this early
    // removal is itself skipped by an unwind, the still-armed guard's Drop is the backstop).
    let _ = tokio::fs::remove_dir_all(&out_frames_dir).await;
    out_scratch.disarm();
    encode_result?;
    let mux_tmp = media_path.with_extension("audiomux.mp4");
    let mut output_guard = Seedvr2OutputGuard::new(&media_path, &mux_tmp);

    // Source-audio passthrough: remux the source's audio onto the upscaled video. `-map 1:a:0?`
    // makes audio optional, so a source with no audio yields a clean video-only file (no error);
    // `-c:v copy` keeps the upscaled video stream untouched, `+faststart` is preserved.
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
            vec![
                "ffmpeg".to_owned(),
                "-nostdin".to_owned(),
                "-y".to_owned(),
                "-i".to_owned(),
                media_path.to_string_lossy().into_owned(),
                "-i".to_owned(),
                source_path.to_string_lossy().into_owned(),
                "-map".to_owned(),
                "0:v:0".to_owned(),
                "-map".to_owned(),
                "1:a:0?".to_owned(),
                "-c:v".to_owned(),
                "copy".to_owned(),
                "-c:a".to_owned(),
                "aac".to_owned(),
                "-movflags".to_owned(),
                "+faststart".to_owned(),
                "-shortest".to_owned(),
                mux_tmp.to_string_lossy().into_owned(),
            ],
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
        "fps": out_fps,
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
