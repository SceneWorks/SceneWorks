use super::*;
use std::net::SocketAddr;
use std::time::Duration;

/// Connect timeout applied to every outbound worker HTTP client. A peer that
/// completes the TCP handshake but never sends response headers must not wedge the
/// worker: reqwest's default is *no* timeout, so `send().await` would block forever.
/// This is especially load-bearing for the pre-heartbeat HEAD probes
/// (`remote_content_length` / `lora_source_content_length`), which run before any
/// cancel checkpoint exists — without a bound the process would never heartbeat or
/// claim again (sc-11149).
pub(crate) const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Chunk-level read timeout: the longest gap tolerated between received bytes on a
/// streaming download. This bounds a stalled mid-stream peer WITHOUT a total timeout,
/// so legitimate multi-GB weight streams still complete as long as bytes keep
/// arriving (sc-11149).
pub(crate) const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(60);

/// Apply the standard connect + chunk-level read timeouts to a download client
/// builder. Streaming downloads deliberately get NO total `timeout` (that would cap a
/// large weight transfer); a stalled peer is caught by `read_timeout` instead
/// (sc-11149).
pub(crate) fn apply_download_timeouts(builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
    builder
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .read_timeout(HTTP_READ_TIMEOUT)
}

/// A reqwest client for streaming weight/asset downloads, pre-configured with the
/// worker's standard connect + read timeouts (sc-11149). Replaces bare
/// `reqwest::Client::new()` at download call sites so no outbound download can hang
/// the worker indefinitely.
pub(crate) fn streaming_download_client() -> reqwest::Client {
    apply_download_timeouts(reqwest::Client::builder())
        .build()
        .expect("static download client config is always valid")
}

/// Forward-progress watchdog window: the longest a streaming download may run WITHOUT
/// gaining at least [`DOWNLOAD_STALL_MIN_PROGRESS`] bytes before the worker abandons
/// the current connection and resumes it with a fresh HTTP Range request. This catches
/// the Hugging Face failure mode the chunk-level [`HTTP_READ_TIMEOUT`] cannot: a CDN
/// edge that keeps the connection alive and dribbles bytes just often enough to reset
/// the 60s read timeout, so the transfer never *fails* but effectively never
/// *progresses*. Because the bytes already on disk are kept and continued via Range,
/// the abort is lossless and the reconnect usually lands on a healthy edge. Default 10
/// minutes; `SCENEWORKS_DOWNLOAD_STALL_TIMEOUT_SECONDS` overrides it, and `0` disables
/// the watchdog entirely.
pub(crate) const DOWNLOAD_STALL_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// Minimum bytes a download must gain within a [`DOWNLOAD_STALL_TIMEOUT`] window to
/// count as "making progress". Any real transfer clears 1 MiB in milliseconds, so a
/// window that fails to move even this much is hung, not slow-but-live. This threshold
/// is what makes the watchdog robust to a byte-at-a-time trickle that a naive
/// "any byte resets the clock" check would never flag.
pub(crate) const DOWNLOAD_STALL_MIN_PROGRESS: u64 = 1024 * 1024;

/// Give up after this many CONSECUTIVE stall-triggered resumes that each fail to
/// advance the on-disk byte count. A resume that makes real progress resets the
/// counter, so a genuinely slow-but-live transfer keeps going indefinitely; only an
/// endpoint wedged at the same offset trips this and surfaces a clear timeout error
/// instead of looping forever.
const MAX_STALL_RESUMES_WITHOUT_PROGRESS: u32 = 3;

/// Resolve the stall-watchdog window: [`DOWNLOAD_STALL_TIMEOUT`] by default,
/// overridable via `SCENEWORKS_DOWNLOAD_STALL_TIMEOUT_SECONDS`. An explicit `0`
/// disables the watchdog (returns `None`), matching the "no total timeout" posture of
/// the streaming clients.
fn download_stall_timeout() -> Option<Duration> {
    let seconds = crate::settings::env_u64_any(
        &["SCENEWORKS_DOWNLOAD_STALL_TIMEOUT_SECONDS"],
        DOWNLOAD_STALL_TIMEOUT.as_secs(),
    );
    (seconds > 0).then(|| Duration::from_secs(seconds))
}

// The cross-process download lock (F-098 / sc-8900) is only reachable from
// `ensure_cached_file_verified`, which is gated to the macOS MLX runtime and the
// off-Mac candle InstantID lane; gate the whole apparatus the same way so the bare
// (non-macOS, non-candle) lib build — which still compiles `download_source_url` —
// doesn't drag in an unused `fs2` import or dead lock helpers.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use download_lock::DownloadLock;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod download_lock {
    use super::{task_join_error, WorkerError, WorkerResult};
    use fs2::FileExt as _;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    /// Max time to block waiting for the cross-process download lock before giving up.
    /// A peer legitimately holding it is streaming a (potentially multi-GB) weight
    /// file, so this is far longer than the manifest lock's timeout — long enough to
    /// outlast a real download, short enough that a crashed/stuck peer surfaces a
    /// clear error rather than hanging the job forever (sc-8900).
    const DOWNLOAD_LOCK_TIMEOUT: Duration = Duration::from_secs(3600);
    /// Poll cadence while spin-waiting on `try_lock_exclusive` (fs2 has no timed
    /// blocking-lock API, so we retry rather than block indefinitely). Coarser than
    /// the manifest poll because a download hold is seconds-to-minutes, not sub-ms.
    const DOWNLOAD_LOCK_POLL: Duration = Duration::from_millis(200);

    /// RAII holder for a cross-process advisory *exclusive* lock on a `<target>.lock`
    /// sibling, serializing the download of one cache target across the separate
    /// utility-worker processes (F-098 / sc-8900). The default utility pool is 4
    /// SEPARATE PROCESSES, so an in-process mutex cannot serialize them — two jobs
    /// resolving the same runtime-weight file would each open the target and
    /// interleave/append their writes, producing a corrupt file that can slip past
    /// the size check when no sha256 is available. The lock releases when the
    /// underlying handle drops. Mirrors `manifest::ManifestLock`.
    pub(crate) struct DownloadLock {
        _file: std::fs::File,
    }

    impl DownloadLock {
        /// Acquire the exclusive lock for `target`, creating the parent dir and the
        /// `.lock` sibling as needed. Blocking (spin-waits on the advisory lock), so
        /// the caller runs it on the blocking pool.
        pub(crate) fn acquire(target: &Path) -> WorkerResult<Self> {
            let lock_path = download_lock_path(target);
            if let Some(parent) = lock_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&lock_path)?;
            let deadline = Instant::now() + DOWNLOAD_LOCK_TIMEOUT;
            // fs2 signals contention with a platform-specific error (`EWOULDBLOCK` on
            // Unix, `ERROR_LOCK_VIOLATION` on Windows); compare by RAW OS CODE against
            // fs2's own contention error so retry-vs-fail is correct on every platform
            // (same posture as `manifest::ManifestLock`, sc-8843).
            let contended = fs2::lock_contended_error().raw_os_error();
            loop {
                match file.try_lock_exclusive() {
                    Ok(()) => return Ok(Self { _file: file }),
                    Err(error) if error.raw_os_error() == contended => {
                        if Instant::now() >= deadline {
                            return Err(WorkerError::Io(std::io::Error::new(
                                std::io::ErrorKind::TimedOut,
                                format!(
                                    "timed out after {DOWNLOAD_LOCK_TIMEOUT:?} waiting for download lock {}",
                                    lock_path.display()
                                ),
                            )));
                        }
                        std::thread::sleep(DOWNLOAD_LOCK_POLL);
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }

        /// Acquire the lock on the blocking pool (the spin-wait must not stall the
        /// async runtime), then hold the guard across the async download. The
        /// `std::fs::File` handle is `Send`, so the guard lives across `.await` and
        /// the advisory lock is held for the whole transfer.
        pub(crate) async fn acquire_async(target: &Path) -> WorkerResult<Self> {
            let target = target.to_path_buf();
            tokio::task::spawn_blocking(move || DownloadLock::acquire(&target))
                .await
                .map_err(|error| task_join_error("download lock", error))?
        }
    }

    /// The `.lock` sibling path for a download target. Kept alongside the target so
    /// the lock scope is the exact file being written (per-file, not global), and two
    /// downloads of *different* files never contend.
    fn download_lock_path(target: &Path) -> PathBuf {
        let mut name = target
            .file_name()
            .map(std::ffi::OsString::from)
            .unwrap_or_default();
        name.push(".download.lock");
        target.with_file_name(name)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// sc-8900 / F-098: two holders of the same target's download lock are
        /// mutually exclusive — the second `try_lock_exclusive` sees contention while
        /// the first guard is alive, then succeeds once it drops. This is the
        /// cross-process serialization primitive the utility-worker pool relies on
        /// (exercised here across handles within one process).
        #[test]
        fn download_lock_is_exclusive_per_target_and_releases_on_drop() {
            let dir = tempfile::tempdir().expect("tempdir");
            let target = dir.path().join("weights").join("model.safetensors");

            let first = DownloadLock::acquire(&target).expect("first lock acquires");

            // A second exclusive lock on the SAME target's lock file must be contended
            // while the first is held. Probe the raw fs2 primitive directly so the test
            // doesn't block on the (1 hour) acquire timeout.
            let lock_path = download_lock_path(&target);
            let probe = std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&lock_path)
                .expect("probe opens lock file");
            let contended = fs2::lock_contended_error().raw_os_error();
            let held = probe.try_lock_exclusive();
            assert_eq!(
                held.as_ref().err().and_then(|e| e.raw_os_error()),
                contended,
                "second exclusive lock must be contended while the first is held"
            );

            // Once the first guard drops, the same target locks cleanly again.
            drop(first);
            DownloadLock::acquire(&target).expect("lock re-acquires after release");

            // A DIFFERENT target never contends with the first.
            let other = dir.path().join("weights").join("other.safetensors");
            let _first_other = DownloadLock::acquire(&other).expect("distinct target locks");
        }

        /// The lock file is a `.download.lock` sibling of the target (per-file scope),
        /// so distinct targets get distinct lock paths.
        #[test]
        fn download_lock_path_is_per_file_sibling() {
            let a = download_lock_path(Path::new("/data/models/a.safetensors"));
            let b = download_lock_path(Path::new("/data/models/b.safetensors"));
            assert_eq!(a, Path::new("/data/models/a.safetensors.download.lock"));
            assert_ne!(a, b);
        }
    }
}

/// Download `url` to `target` on first use. Existing complete files are reused;
/// partial files resume with HTTP Range when the caller can provide `expected_size`.
/// The transfer shares model-download progress/cancel plumbing instead of buffering
/// the response body in memory.
// Shared by the macOS MLX runtime-weight downloads AND the candle InstantID lane (sc-5491): the
// off-Mac InstantID provider stages its SCRFD/ArcFace/IP-Adapter/ControlNet files via this same
// download-on-first-use path, so it broadened from macOS-only. (All helpers it calls — download_file,
// DownloadProgress, DownloadContext, HuggingFaceSnapshot — already build on every platform.)
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn ensure_cached_file(
    context: &DownloadContext<'_>,
    url: &str,
    target: &Path,
    label: &str,
    expected_size: Option<u64>,
) -> WorkerResult<PathBuf> {
    ensure_cached_file_verified(context, url, target, label, expected_size, None).await
}

/// [`ensure_cached_file`] that additionally verifies the completed file against an
/// `expected_sha256` (a content digest — e.g. an HF `lfs.oid`) before returning
/// (sc-8879). A malformed/absent digest is skipped by `verify_file_sha256`, so callers
/// can pass whatever the source advertises; a mismatch removes the file and errors.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn ensure_cached_file_verified(
    context: &DownloadContext<'_>,
    url: &str,
    target: &Path,
    label: &str,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
) -> WorkerResult<PathBuf> {
    // Serialize the whole cache-check + transfer for this target across the separate
    // utility-worker processes so two jobs can't interleave/append writes to the same
    // partial file and leave a corrupt result (F-098 / sc-8900). The cache-hit
    // short-circuit lives inside the lock too, so a peer mid-transfer can't be read as
    // "already complete". The guard is held for the whole function.
    let _lock = DownloadLock::acquire_async(target).await?;
    let expected_size = match expected_size {
        Some(size) => Some(size),
        None => remote_content_length(context.client, url).await?,
    };
    if let Ok(metadata) = tokio::fs::metadata(target).await {
        if expected_size
            .map(|expected| metadata.len() == expected)
            .unwrap_or(true)
        {
            // A cached file that already matches the expected length still gets its
            // digest checked so a size-colliding tampered artifact can't be reused.
            if let Some(expected) = expected_sha256 {
                verify_file_sha256(target, expected, label).await?;
            }
            return Ok(target.to_path_buf());
        }
    }
    if expected_size.is_none() && target.exists() {
        if let Some(expected) = expected_sha256 {
            verify_file_sha256(target, expected, label).await?;
        }
        return Ok(target.to_path_buf());
    }
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let started_bytes = existing_download_bytes(target, expected_size).await?;
    let mut progress = DownloadProgress::new(
        label,
        started_bytes,
        expected_size,
        progress_report_interval(context.settings),
    );
    download_file(
        context,
        url,
        target,
        expected_size,
        expected_sha256,
        label,
        &mut progress,
    )
    .await?;
    Ok(target.to_path_buf())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn remote_content_length(client: &reqwest::Client, url: &str) -> WorkerResult<Option<u64>> {
    // `url` is built from trusted operator/runtime configuration
    // (`Settings::huggingface_base_url`) plus validated HF path pieces. User-provided source URLs
    // use the separate `download_source_url` path with SSRF checks.
    let response = match client.head(url).send().await {
        Ok(response) => response,
        Err(_) => return Ok(None),
    };
    if response.status().is_success() {
        Ok(response.content_length().filter(|value| *value > 0))
    } else {
        Ok(None)
    }
}

/// Resolve a single Hugging Face file and stream it into an app cache target with
/// size-aware resume/progress. This is for first-use runtime weights that are not
/// installed through the full model-download flow.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn ensure_hf_cached_file(
    context: &DownloadContext<'_>,
    repo: &str,
    revision: &str,
    file: &str,
    target: &Path,
) -> WorkerResult<PathBuf> {
    let snapshot = HuggingFaceSnapshot::resolve(
        context.client,
        context.settings,
        repo,
        revision,
        &[file.to_owned()],
    )
    .await?;
    let Some(snapshot_file) = snapshot
        .files
        .into_iter()
        .find(|candidate| candidate.path == file)
    else {
        return Err(WorkerError::InvalidPayload(format!(
            "Hugging Face file {file} not found in {repo}."
        )));
    };
    ensure_cached_file_verified(
        context,
        &snapshot_file.download_url,
        target,
        &snapshot_file.path,
        snapshot_file.size,
        // Verify the content against HF's `lfs.oid` (present for the LFS-tracked
        // weights) so a pinned-revision download is integrity-checked, not just
        // length-checked (sc-8879).
        snapshot_file.sha256.as_deref(),
    )
    .await
}

#[derive(Debug, Clone)]
pub(crate) struct SnapshotFile {
    pub(crate) path: String,
    pub(crate) size: Option<u64>,
    pub(crate) download_url: String,
    /// SHA-256 of the file content from Hugging Face's `lfs.oid` (tree API
    /// `expand=1`). Present for LFS-tracked files (the weights); `None` for small
    /// non-LFS files (configs/tokenizers), whose integrity rides on the size check.
    pub(crate) sha256: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct HuggingFaceSnapshot {
    pub(crate) files: Vec<SnapshotFile>,
}

impl HuggingFaceSnapshot {
    pub(crate) async fn resolve(
        client: &reqwest::Client,
        settings: &Settings,
        repo: &str,
        revision: &str,
        files: &[String],
    ) -> WorkerResult<Self> {
        let base_url = settings.huggingface_base_url.trim_end_matches('/');
        // The HF tree API paginates the `expand=1` listing (default limit 50) and returns the next
        // page as a `Link: <…?cursor=…>; rel="next"` header. A single request therefore sees only the
        // first ~50 files, so for a multi-tier repo (bf16/q4/q8 subdirs) the later tiers' files fall
        // past page 1 and go MISSING — a q8 download then resolves zero files and silently produces an
        // empty cache (sc-9909). Follow `rel="next"` until exhausted so every file is seen. The page
        // cap is a runaway backstop far above any real repo's file count.
        let mut next_url = Some(format!(
            "{base_url}/api/models/{}/tree/{}?recursive=1&expand=1",
            quote_path(repo),
            quote_path(revision)
        ));
        let mut snapshot_files = Vec::new();
        for _ in 0..10_000 {
            let Some(url) = next_url.take() else {
                break;
            };
            let response = with_hf_auth(settings, client.get(&url))
                .await
                .send()
                .await?
                .error_for_status()?;
            next_url = next_page_url(response.headers());
            let payload = response.json::<Value>().await?;
            let entries = if let Some(entries) = payload.as_array() {
                entries.clone()
            } else {
                payload
                    .get("siblings")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
            };
            snapshot_files.extend(
                entries
                    .iter()
                    .filter_map(|entry| snapshot_file_from_entry(base_url, repo, revision, entry))
                    .filter(|file| allow_pattern_matches(&file.path, files)),
            );
        }
        Ok(Self {
            files: snapshot_files,
        })
    }

    pub(crate) fn total_bytes(&self) -> Option<u64> {
        self.files
            .iter()
            .try_fold(0_u64, |total, file| Some(total.saturating_add(file.size?)))
    }
}

/// Extract the `rel="next"` target from an RFC 5988 `Link` header, if present. The HF tree API
/// paginates its `expand=1` listing this way — the header looks like
/// `<https://…/tree/main?expand=true&recursive=true&limit=50&cursor=…>; rel="next"`. Returns the
/// absolute next-page URL (the server preserves the `expand`/`recursive` params in it).
fn next_page_url(headers: &reqwest::header::HeaderMap) -> Option<String> {
    let link = headers.get(reqwest::header::LINK)?.to_str().ok()?;
    for part in link.split(',') {
        let mut segments = part.split(';');
        let Some(url) = segments.next() else {
            continue;
        };
        let is_next = segments.any(|attribute| {
            let attribute = attribute.trim();
            attribute == "rel=\"next\"" || attribute == "rel=next"
        });
        if is_next {
            let url = url
                .trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .trim();
            if !url.is_empty() {
                return Some(url.to_owned());
            }
        }
    }
    None
}

pub(crate) fn snapshot_file_from_entry(
    base_url: &str,
    repo: &str,
    revision: &str,
    entry: &Value,
) -> Option<SnapshotFile> {
    let kind = entry.get("type").and_then(Value::as_str);
    if kind.is_some_and(|kind| kind != "file") {
        return None;
    }
    let path = entry
        .get("path")
        .or_else(|| entry.get("rfilename"))
        .and_then(Value::as_str)?;
    Some(SnapshotFile {
        path: path.to_owned(),
        size: entry.get("size").and_then(json_size_to_u64),
        download_url: format!(
            "{base_url}/{}/resolve/{}/{}",
            quote_path(repo),
            // Revisions are pre-validated by `model_jobs::validate_hf_revision`;
            // quote_path is the direct-download path's final URL-segment guard.
            quote_path(revision),
            quote_path(path)
        ),
        sha256: entry
            .get("lfs")
            .and_then(|lfs| lfs.get("oid"))
            .and_then(Value::as_str)
            .and_then(normalize_sha256),
    })
}

/// Normalize a candidate SHA-256 digest (from `lfs.oid` or an HF ETag): strip an
/// optional `sha256:` prefix and surrounding whitespace, lowercase it, and accept it
/// only if it is exactly 64 hex characters. Returns `None` for anything else (e.g. a
/// git blob SHA-1 ETag, a fallback blob name) so callers verify only real content
/// digests.
pub(crate) fn normalize_sha256(value: &str) -> Option<String> {
    let digest = value
        .trim()
        .trim_start_matches("sha256:")
        .trim()
        .to_ascii_lowercase();
    if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Some(digest)
    } else {
        None
    }
}

pub(crate) struct DownloadContext<'a> {
    pub(crate) api: &'a ApiClient,
    pub(crate) client: &'a reqwest::Client,
    pub(crate) settings: &'a Settings,
    pub(crate) job_id: &'a str,
    pub(crate) cancel_message: &'a str,
    pub(crate) fresh_download: bool,
}

const AUTO_RESUME_ATTEMPTS: usize = 1;

/// Download a single file to `dest` (resumable via HTTP Range), rejecting a truncated
/// response (size mismatch) and, when `expected_sha256` is provided, a corrupt one
/// (content-digest mismatch). On a digest mismatch the file is removed so a corrupt
/// artifact is never left behind (sc-6137). `label` names the file in the error.
async fn download_file(
    context: &DownloadContext<'_>,
    url: &str,
    dest: &Path,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
    label: &str,
    progress: &mut DownloadProgress<'_>,
) -> WorkerResult<()> {
    download_file_inner(context, url, dest, expected_size, label, progress).await?;
    if let Some(expected) = expected_sha256 {
        verify_file_sha256(dest, expected, label).await?;
    }
    Ok(())
}

/// Verify `path`'s SHA-256 equals `expected`; on mismatch, remove the file and return
/// an actionable error. A malformed/absent `expected` (not 64 hex) is treated as "no
/// digest available" and skipped — only a real content digest is enforced.
pub(crate) async fn verify_file_sha256(
    path: &Path,
    expected: &str,
    label: &str,
) -> WorkerResult<()> {
    let Some(expected) = normalize_sha256(expected) else {
        return Ok(());
    };
    let actual = sha256_file(path).await?;
    if actual != expected {
        let _ = tokio::fs::remove_file(path).await;
        return Err(WorkerError::InvalidPayload(format!(
            "{label} failed its integrity check (sha256 {actual}, but the source declares {expected}); \
             the download was corrupted. Re-download the file."
        )));
    }
    Ok(())
}

/// Stream `path` through SHA-256 on the blocking pool (weights are multi-GB; hashing
/// on the async runtime would stall heartbeats/cancel checks).
async fn sha256_file(path: &Path) -> WorkerResult<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> std::io::Result<String> {
        use std::io::Read;
        let mut file = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    })
    .await
    .map_err(|error| {
        WorkerError::Io(std::io::Error::other(format!(
            "sha256 task failed: {error}"
        )))
    })?
    .map_err(WorkerError::Io)
}

/// Download a single file to `dest` (resumable via HTTP Range), reporting transfer
/// progress and rejecting a truncated response. `label` names the file in the
/// size-mismatch error.
async fn download_file_inner(
    context: &DownloadContext<'_>,
    url: &str,
    dest: &Path,
    expected_size: Option<u64>,
    label: &str,
    progress: &mut DownloadProgress<'_>,
) -> WorkerResult<()> {
    if context.fresh_download {
        let removed_bytes = remove_incomplete_download(dest, expected_size).await?;
        if removed_bytes > 0 {
            progress.discard_started_bytes(removed_bytes);
        }
    }
    let mut resume_attempts_remaining = if context.fresh_download {
        0
    } else {
        AUTO_RESUME_ATTEMPTS
    };
    // Forward-progress watchdog (independent of the network-error resume budget above):
    // a hung-but-alive stream is abort-and-resumed as often as it keeps advancing, and
    // only a source wedged at the same offset for MAX_STALL_RESUMES_WITHOUT_PROGRESS
    // consecutive windows gives up. `last_stall_bytes` starts at 0 so the FIRST stall
    // never counts as no-progress (any partial file already beats it).
    let stall_timeout = download_stall_timeout();
    let mut stall_resumes_without_progress = 0_u32;
    let mut last_stall_bytes = 0_u64;
    loop {
        let existing_bytes = existing_download_bytes(dest, expected_size).await?;
        if expected_size.is_some_and(|size| existing_bytes == size) {
            return Ok(());
        }
        let mut request = context.client.get(url);
        if existing_bytes > 0 {
            request = request.header(header::RANGE, format!("bytes={existing_bytes}-"));
        }
        let response = with_hf_auth(context.settings, request).await.send().await?;
        let status = response.status();
        if status == StatusCode::RANGE_NOT_SATISFIABLE && existing_bytes > 0 {
            if let Some(expected) = expected_size {
                return Err(WorkerError::InvalidPayload(download_size_mismatch_message(
                    label,
                    existing_bytes,
                    expected,
                )));
            }
        }
        if !status.is_success() {
            return Err(WorkerError::Http(response.error_for_status().unwrap_err()));
        }
        let appending = existing_bytes > 0 && status == StatusCode::PARTIAL_CONTENT;
        if existing_bytes > 0 && !appending {
            progress.discard_started_bytes(existing_bytes);
        }
        let mut response = response;
        let mut output = if appending {
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dest)
                .await?
        } else {
            tokio::fs::File::create(dest).await?
        };
        let mut interval = tokio::time::interval(progress.report_interval());
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // A tokio interval's first tick is immediate; consume it so the first chunk
        // doesn't spuriously fire a zero-byte progress report before any transfer.
        interval.tick().await;
        // Give this connection a fresh stall window: the time spent (re)issuing the
        // request must not count against forward progress.
        progress.reset_stall_clock();
        let mut transfer_error = None;
        let mut stalled = false;
        loop {
            tokio::select! {
                chunk = response.chunk() => {
                    match chunk {
                        Ok(Some(chunk)) => {
                            output.write_all(&chunk).await?;
                            progress.record_transferred(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
                        }
                        Ok(None) => break,
                        Err(error) => {
                            transfer_error = Some(WorkerError::from(error));
                            break;
                        }
                    }
                }
                _ = interval.tick() => {
                    report_download_progress(context, progress).await?;
                    // A live-but-hung stream (bytes trickling in under HTTP_READ_TIMEOUT
                    // yet never advancing) is caught here and abort-and-resumed below.
                    if stall_timeout
                        .is_some_and(|timeout| progress.is_stalled(timeout, DOWNLOAD_STALL_MIN_PROGRESS))
                    {
                        stalled = true;
                        break;
                    }
                }
            }
        }
        output.flush().await?;
        if stalled {
            // Resume from the bytes already on disk. As long as each stalled attempt
            // advances the on-disk count, keep going; only a source stuck at the same
            // offset for consecutive windows is fatal.
            let on_disk = existing_download_bytes(dest, expected_size).await?;
            if on_disk > last_stall_bytes {
                stall_resumes_without_progress = 0;
            } else {
                stall_resumes_without_progress += 1;
            }
            last_stall_bytes = on_disk;
            if stall_resumes_without_progress >= MAX_STALL_RESUMES_WITHOUT_PROGRESS {
                let timeout_secs = stall_timeout.map_or(0, |timeout| timeout.as_secs());
                return Err(WorkerError::Io(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!(
                        "{label} download stalled: no forward progress for \
                         {MAX_STALL_RESUMES_WITHOUT_PROGRESS} consecutive {timeout_secs}s windows \
                         (stuck at {}). The source appears hung; try again.",
                        format_bytes_with_exact(on_disk),
                    ),
                )));
            }
            continue;
        }
        if let Some(error) = transfer_error {
            if let Some(expected) = expected_size {
                let written = tokio::fs::metadata(dest).await?.len();
                if written == expected {
                    return Ok(());
                }
                if written < expected && resume_attempts_remaining > 0 {
                    resume_attempts_remaining -= 1;
                    continue;
                }
            }
            return Err(error);
        }
        // A truncated transfer (e.g. the server closes the stream at what looks like a
        // clean EOF) would otherwise be treated as success and the bad file only surface
        // as an opaque load failure later. When the expected size is known, verify it.
        // Short files are preserved so a later retry can resume them; overlong files are
        // discarded because appending would only move them farther away from the target.
        if let Some(expected) = expected_size {
            let written = tokio::fs::metadata(dest).await?.len();
            if written == expected {
                return Ok(());
            }
            if written < expected && resume_attempts_remaining > 0 {
                resume_attempts_remaining -= 1;
                continue;
            }
            if written > expected {
                let _ = tokio::fs::remove_file(dest).await;
            }
            return Err(WorkerError::InvalidPayload(download_size_mismatch_message(
                label, written, expected,
            )));
        }
        return Ok(());
    }
}

async fn remove_incomplete_download(path: &Path, expected_size: Option<u64>) -> WorkerResult<u64> {
    let Ok(metadata) = tokio::fs::metadata(path).await else {
        return Ok(0);
    };
    let existing_bytes = metadata.len();
    if expected_size
        .map(|expected| metadata.len() != expected)
        .unwrap_or(true)
    {
        tokio::fs::remove_file(path).await?;
        return Ok(existing_bytes);
    }
    Ok(0)
}

fn format_bytes_with_exact(value: u64) -> String {
    format!("{} ({value} bytes)", format_bytes(value))
}

fn download_size_mismatch_message(label: &str, actual: u64, expected: u64) -> String {
    let delta = actual.abs_diff(expected);
    let direction = if actual < expected {
        "missing"
    } else {
        "extra"
    };
    format!(
        "{label} download ended at {} but expected {}; {} {}.",
        format_bytes_with_exact(actual),
        format_bytes_with_exact(expected),
        format_bytes_with_exact(delta),
        direction
    )
}

/// Download a Hugging Face snapshot as a flat file tree under `target_dir`. Used by
/// the model-import flow, which intentionally populates the app's import store.
pub(crate) async fn download_snapshot(
    context: &DownloadContext<'_>,
    target_dir: &Path,
    snapshot: &HuggingFaceSnapshot,
    progress: &mut DownloadProgress<'_>,
) -> WorkerResult<()> {
    tokio::fs::create_dir_all(target_dir).await?;
    for file in &snapshot.files {
        check_download_cancel(context).await?;
        let target_path = safe_join(target_dir, &file.path)?;
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        download_file(
            context,
            &file.download_url,
            &target_path,
            file.size,
            file.sha256.as_deref(),
            &file.path,
            progress,
        )
        .await?;
    }
    Ok(())
}

/// Download a Hugging Face snapshot into the standard hub cache layout under
/// `repo_dir` (`models--<org>--<name>`): content lands in `blobs/<etag>`, the
/// checkpoint is materialized as `snapshots/<commit>/<path>` (a relative symlink to
/// its blob on Unix, a hardlink to the blob on Windows — see [`link_blob`] — or a
/// copy where neither is available), and `refs/<rev>` records the
/// commit. This matches `huggingface_hub`, so HF-sourced downloads dedupe with other
/// tools and the Python loader instead of duplicating into the private app store
/// (sc-1904).
pub(crate) async fn download_snapshot_into_cache(
    context: &DownloadContext<'_>,
    repo_dir: &Path,
    revision: &str,
    snapshot: &HuggingFaceSnapshot,
    progress: &mut DownloadProgress<'_>,
) -> WorkerResult<()> {
    let blobs_dir = repo_dir.join("blobs");
    tokio::fs::create_dir_all(&blobs_dir).await?;
    // A no-redirect client so the metadata HEAD reads huggingface.co's headers
    // (X-Repo-Commit, and X-Linked-Etag for LFS) rather than the CDN's after a
    // redirect — exactly how huggingface_hub resolves an etag/commit.
    let meta_client = apply_download_timeouts(
        reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()),
    )
    .build()?;
    let mut commit: Option<String> = None;
    let mut placements: Vec<(String, String)> = Vec::with_capacity(snapshot.files.len());

    for file in &snapshot.files {
        check_download_cancel(context).await?;
        let head = with_hf_auth(context.settings, meta_client.head(&file.download_url))
            .await
            .send()
            .await?;
        if commit.is_none() {
            commit = header_value(&head, "x-repo-commit");
        }
        let etag = header_value(&head, "x-linked-etag")
            .or_else(|| header_value(&head, "etag"))
            .map(|value| normalize_etag(&value))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| blob_fallback_name(&file.path));
        download_file(
            context,
            &file.download_url,
            &blobs_dir.join(&etag),
            file.size,
            // The blob is named by its etag (= the LFS sha256), and the tree API
            // reports the same digest as `lfs.oid`; verify the content against it.
            file.sha256.as_deref(),
            &file.path,
            progress,
        )
        .await?;
        placements.push((file.path.clone(), etag));
    }

    // Materialize the snapshot once every blob is present: refs/<rev> -> commit and
    // snapshots/<commit>/<path> -> ../../blobs/<etag>.
    let commit = commit.unwrap_or_else(|| revision.to_owned());
    let refs_dir = repo_dir.join("refs");
    tokio::fs::create_dir_all(&refs_dir).await?;
    tokio::fs::write(refs_dir.join(revision), commit.as_bytes()).await?;
    let snapshot_dir = repo_dir.join("snapshots").join(&commit);
    for (relpath, etag) in &placements {
        let link = safe_join(&snapshot_dir, relpath)?;
        if let Some(parent) = link.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        if tokio::fs::symlink_metadata(&link).await.is_ok() {
            let _ = tokio::fs::remove_file(&link).await;
        }
        let depth = link
            .parent()
            .and_then(|parent| parent.strip_prefix(repo_dir).ok())
            .map(|relative| relative.components().count())
            .unwrap_or(2);
        let mut rel_target = PathBuf::new();
        for _ in 0..depth {
            rel_target.push("..");
        }
        rel_target.push("blobs");
        rel_target.push(etag);
        if !link_blob(&blobs_dir.join(etag), &rel_target, &link).await {
            tokio::fs::copy(blobs_dir.join(etag), &link).await?;
        }
    }
    Ok(())
}

fn header_value(response: &reqwest::Response, name: &str) -> Option<String> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Strip the surrounding quotes and any weak-validator prefix HTTP/HF put around an
/// ETag, leaving the bare blob name huggingface_hub uses.
fn normalize_etag(raw: &str) -> String {
    raw.trim()
        .trim_start_matches("W/")
        .trim_matches('"')
        .to_owned()
}

/// Blob name when the server returns no etag (a non-HF stub or an endpoint that
/// omits ETag): a filesystem-safe rendering of the repo path. Keeps the download
/// working; only weakens cross-app dedup for that one file.
fn blob_fallback_name(path: &str) -> String {
    path.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

/// Materialize a snapshot entry pointing at its blob, returning whether it
/// succeeded (the caller copies when it does not). On Unix this is a relative
/// symlink, mirroring huggingface_hub so HF tools dedupe with this cache. On
/// Windows it is a **hardlink** to the absolute blob instead: the candle worker
/// process cannot traverse the relative `snapshots/<rev>/… -> ../blobs/<etag>`
/// symlinks — the open fails with `ERROR_UNTRUSTED_MOUNT_POINT` (os error 448, see
/// [`crate::model_jobs::downloaded_model_detection_io_error_is_inconclusive`]) — so
/// every model load died at the first file read. A hardlink is a plain directory
/// entry to the same blob data (no reparse point, same volume by construction, still
/// deduped) and reads fine. (`model_jobs::huggingface_snapshot_dir` repairs caches
/// that were already downloaded as symlinks, e.g. by `huggingface_hub`.)
async fn link_blob(blob_abs: &Path, rel_target: &Path, link: &Path) -> bool {
    #[cfg(windows)]
    {
        let _ = rel_target;
        tokio::fs::hard_link(blob_abs, link).await.is_ok()
    }
    #[cfg(unix)]
    {
        let _ = blob_abs;
        tokio::fs::symlink(rel_target, link).await.is_ok()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (blob_abs, rel_target, link);
        false
    }
}

pub(crate) async fn download_lora_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
) -> WorkerResult<()> {
    download_source_url(
        context,
        source_url,
        target_dir,
        "LoRA",
        context.settings.max_lora_url_bytes,
    )
    .await
}

pub(crate) async fn download_model_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
) -> WorkerResult<()> {
    download_source_url(
        context,
        source_url,
        target_dir,
        "Model",
        context.settings.max_model_url_bytes,
    )
    .await
}

pub(crate) async fn download_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
    source_label: &str,
    max_bytes: u64,
) -> WorkerResult<()> {
    let url =
        parse_lora_source_url_with_private(source_url, context.settings.allow_private_lora_urls)
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
    let file_name = lora_source_url_file_name(source_url)
        .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
    tokio::fs::create_dir_all(target_dir).await?;
    let target_path = target_dir.join(file_name);

    // Attach a stored credential matching the source host. Bearer tokens ride an
    // Authorization header (dropped on cross-host redirects below); query tokens
    // are baked into the request URL and never carried onto a redirect target. The
    // secret is resolved lazily (env/file-store, or the macOS desktop socket on
    // first use) so a no-credential install never touches the keychain (sc-5891).
    let credential = crate::credentials_ipc::resolve_credential_for_host(
        context.settings,
        url.host_str().unwrap_or_default(),
    )
    .await;
    let request_url = match &credential {
        Some(cred) if cred.scheme == CredentialScheme::Query => {
            let mut authed = url.clone();
            authed.query_pairs_mut().append_pair("token", &cred.token);
            authed.to_string()
        }
        _ => source_url.to_owned(),
    };
    let bearer = match &credential {
        Some(cred) if cred.scheme == CredentialScheme::Bearer => Some(cred.token.as_str()),
        _ => None,
    };

    let client = source_url_client_for_request(context.settings, &request_url).await?;
    let total_bytes = lora_source_content_length(&client, &request_url, bearer).await?;
    if total_bytes.is_some_and(|total| total > max_bytes) {
        return Err(WorkerError::InvalidPayload(format!(
            "{source_label} sourceUrl exceeds the {} limit",
            format_bytes(max_bytes)
        )));
    }
    let existing_bytes = existing_download_bytes(&target_path, total_bytes).await?;
    if total_bytes.is_some_and(|total| total > 0 && existing_bytes == total) {
        return Ok(());
    }
    let range_header = (existing_bytes > 0).then(|| format!("bytes={existing_bytes}-"));
    let mut response = send_source_url_with_redirects(
        context.settings,
        &request_url,
        &client,
        bearer,
        range_header.as_deref(),
    )
    .await?;
    if response.status() == StatusCode::RANGE_NOT_SATISFIABLE {
        let range_total = response
            .headers()
            .get(header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(content_range_total);
        if total_bytes
            .or(range_total)
            .is_some_and(|total| total > 0 && existing_bytes == total)
        {
            return Ok(());
        }
    }
    response = response.error_for_status()?;
    let appending = existing_bytes > 0 && response.status() == StatusCode::PARTIAL_CONTENT;
    let expected_bytes = total_bytes.or_else(|| {
        response.content_length().map(|remaining| {
            if appending {
                existing_bytes + remaining
            } else {
                remaining
            }
        })
    });
    if expected_bytes.is_some_and(|total| total > max_bytes) {
        return Err(WorkerError::InvalidPayload(format!(
            "{source_label} sourceUrl exceeds the {} limit",
            format_bytes(max_bytes)
        )));
    }
    let mut progress = DownloadProgress::new(
        source_url,
        if appending { existing_bytes } else { 0 },
        expected_bytes,
        progress_report_interval(context.settings),
    );
    let mut output = if appending {
        tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target_path)
            .await?
    } else {
        tokio::fs::File::create(&target_path).await?
    };
    let mut interval = tokio::time::interval(progress.report_interval());
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    interval.tick().await;
    loop {
        tokio::select! {
            chunk = response.chunk() => {
                let Some(chunk) = chunk? else {
                    break;
                };
                // No per-chunk cancel poll here (sc-8806): a GET per received HTTP
                // chunk turned a multi-GB download into tens of thousands of API
                // round-trips and serialized the transfer on them. The interval arm
                // below heartbeats + cancel-checks every report tick, exactly like
                // `download_file_inner`, so cancel latency is the tick interval.
                output.write_all(&chunk).await?;
                progress.record_transferred(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
                if progress.downloaded_bytes() > max_bytes {
                    return Err(WorkerError::InvalidPayload(format!(
                        "{source_label} sourceUrl exceeds the {} limit",
                        format_bytes(max_bytes)
                    )));
                }
            }
            _ = interval.tick() => {
                report_download_progress(context, &progress).await?;
            }
        }
    }
    output.flush().await?;
    if expected_bytes.is_some_and(|expected| progress.downloaded_bytes() != expected) {
        return Err(WorkerError::InvalidPayload(download_size_mismatch_message(
            &format!("{source_label} sourceUrl"),
            progress.downloaded_bytes(),
            expected_bytes.unwrap_or_default(),
        )));
    }
    Ok(())
}

/// Maximum redirect hops to follow on an authenticated source-URL download.
const MAX_SOURCE_URL_REDIRECTS: usize = 5;

/// The stored credential whose host matches `host` (case-insensitive exact match),
/// or `None` when nothing matches.
pub(crate) fn credential_for_host<'a>(
    settings: &'a Settings,
    host: &str,
) -> Option<&'a WorkerCredential> {
    let host = host.trim().to_ascii_lowercase();
    if host.is_empty() {
        return None;
    }
    settings
        .credentials
        .iter()
        .find(|credential| credential.host == host)
}

/// GET `initial_url`, manually following up to `MAX_SOURCE_URL_REDIRECTS` hops
/// (the download client uses `Policy::none()` so we control each hop). Every
/// redirect target is re-validated for SSRF (scheme + host/DNS), then fetched
/// with a client pinned to the validated socket addresses. The bearer
/// `Authorization` header is dropped on any cross-host hop so a token never
/// leaks to a CDN. Returns the final non-redirect response without
/// `error_for_status`, so the caller can still inspect
/// `RANGE_NOT_SATISFIABLE`.
async fn send_source_url_with_redirects(
    settings: &Settings,
    initial_url: &str,
    initial_client: &reqwest::Client,
    bearer: Option<&str>,
    range_header: Option<&str>,
) -> WorkerResult<reqwest::Response> {
    let mut current_url = initial_url.to_owned();
    let mut current_host = reqwest::Url::parse(&current_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase));
    let mut bearer = bearer.map(str::to_owned);
    let mut client = initial_client.clone();
    for _ in 0..=MAX_SOURCE_URL_REDIRECTS {
        let mut request = client.get(&current_url);
        if let Some(token) = &bearer {
            request = request.bearer_auth(token);
        }
        if let Some(range) = range_header {
            request = request.header(header::RANGE, range);
        }
        let response = request.send().await?;
        if !response.status().is_redirection() {
            return Ok(response);
        }
        let location = response
            .headers()
            .get(header::LOCATION)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "sourceUrl redirect was missing a Location header".to_owned(),
                )
            })?;
        let base = reqwest::Url::parse(&current_url)
            .map_err(|_| WorkerError::InvalidPayload("sourceUrl was invalid".to_owned()))?;
        let next = base.join(location).map_err(|_| {
            WorkerError::InvalidPayload("sourceUrl redirect target was invalid".to_owned())
        })?;
        if !matches!(next.scheme(), "http" | "https") {
            return Err(WorkerError::InvalidPayload(
                "sourceUrl redirect must use http or https".to_owned(),
            ));
        }
        let next_host = next.host_str().map(str::to_ascii_lowercase);
        if next_host != current_host {
            // Cross-host redirect: never carry the bearer token to a new origin.
            bearer = None;
        }
        current_host = next_host;
        current_url = next.to_string();
        client = source_url_client_for_url(settings, &next).await?;
    }
    Err(WorkerError::InvalidPayload(
        "sourceUrl exceeded the redirect limit".to_owned(),
    ))
}

async fn source_url_client_for_request(
    settings: &Settings,
    request_url: &str,
) -> WorkerResult<reqwest::Client> {
    let url = reqwest::Url::parse(request_url)
        .map_err(|_| WorkerError::InvalidPayload("sourceUrl was invalid".to_owned()))?;
    source_url_client_for_url(settings, &url).await
}

async fn source_url_client_for_url(
    settings: &Settings,
    url: &reqwest::Url,
) -> WorkerResult<reqwest::Client> {
    let validated_addrs = validate_lora_url_dns(settings, url).await?;
    build_source_url_client(url, validated_addrs.as_deref())
}

pub(crate) fn build_source_url_client(
    url: &reqwest::Url,
    validated_addrs: Option<&[SocketAddr]>,
) -> WorkerResult<reqwest::Client> {
    let mut builder = apply_download_timeouts(
        reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()),
    );
    if let (Some(host), Some(addrs)) = (url.host_str(), validated_addrs) {
        builder = builder.resolve_to_addrs(host, addrs);
    }
    Ok(builder.build()?)
}

pub(crate) async fn lora_source_content_length(
    client: &reqwest::Client,
    request_url: &str,
    bearer: Option<&str>,
) -> WorkerResult<Option<u64>> {
    let mut request = client.head(request_url);
    if let Some(token) = bearer {
        request = request.bearer_auth(token);
    }
    let response = request.send().await?;
    if response.status().is_success() {
        return Ok(response.content_length().filter(|value| *value > 0));
    }
    // A redirecting or auth-gated download endpoint (e.g. Civit.ai) can't report a
    // size via HEAD; fall back to the streamed GET response's content length.
    if response.status().is_redirection() {
        return Ok(None);
    }
    if matches!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED
            | StatusCode::NOT_IMPLEMENTED
            | StatusCode::FORBIDDEN
            | StatusCode::UNAUTHORIZED
    ) {
        return Ok(None);
    }
    response.error_for_status()?;
    Ok(None)
}

pub(crate) fn content_range_total(value: &str) -> Option<u64> {
    value
        .rsplit_once('/')
        .and_then(|(_, total)| total.trim().parse::<u64>().ok())
}

pub(crate) async fn validate_lora_url_dns(
    settings: &Settings,
    url: &reqwest::Url,
) -> WorkerResult<Option<Vec<SocketAddr>>> {
    if settings.allow_private_lora_urls {
        return Ok(None);
    }
    let Some(host) = url.host_str() else {
        return Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl host is not allowed".to_owned(),
        ));
    };
    let port = url.port_or_known_default().unwrap_or(443);
    if let Ok(address) = host.parse::<IpAddr>() {
        validate_public_ip(address)
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
        return Ok(Some(vec![SocketAddr::new(address, port)]));
    }
    let mut addrs = Vec::new();
    for address in tokio::net::lookup_host((host, port)).await? {
        validate_public_ip(address.ip())
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
        addrs.push(address);
    }
    if addrs.is_empty() {
        Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl host did not resolve".to_owned(),
        ))
    } else {
        Ok(Some(addrs))
    }
}

pub(crate) async fn report_download_progress(
    context: &DownloadContext<'_>,
    progress: &DownloadProgress<'_>,
) -> WorkerResult<()> {
    heartbeat(
        context.api,
        context.settings,
        WorkerStatus::Busy,
        Some(context.job_id),
    )
    .await?;
    // The progress POST already returns the job snapshot; read `cancel_requested`
    // off it instead of issuing a separate GET per tick (sc-8806). Cancel only
    // trips on a successful POST that confirms the request, so a transient API
    // failure can never be misread as a user cancel (same posture as sc-4174).
    let job = update_job(context.api, context.job_id, progress.payload()).await?;
    if job.cancel_requested {
        mark_job_canceled(context.api, context.job_id, context.cancel_message).await?;
        return Err(WorkerError::Canceled(context.cancel_message.to_owned()));
    }
    Ok(())
}

async fn check_download_cancel(context: &DownloadContext<'_>) -> WorkerResult<()> {
    if cancel_requested_peek(context.api, context.job_id).await {
        mark_job_canceled(context.api, context.job_id, context.cancel_message).await?;
        return Err(WorkerError::Canceled(context.cancel_message.to_owned()));
    }
    Ok(())
}

pub(crate) struct DownloadProgress<'a> {
    repo: &'a str,
    started_bytes: u64,
    transferred_bytes: u64,
    total_bytes: Option<u64>,
    started_at: Instant,
    report_interval: Duration,
    /// Downloaded-byte count at the last point the stall watchdog observed at least
    /// [`DOWNLOAD_STALL_MIN_PROGRESS`] of forward progress.
    stall_checkpoint_bytes: u64,
    /// When [`Self::stall_checkpoint_bytes`] was last advanced; its `elapsed()` is the
    /// time this transfer has gone without meaningful progress.
    stall_checkpoint_at: Instant,
}

impl<'a> DownloadProgress<'a> {
    pub(crate) fn new(
        repo: &'a str,
        started_bytes: u64,
        total_bytes: Option<u64>,
        report_interval: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            repo,
            started_bytes,
            transferred_bytes: 0,
            total_bytes,
            started_at: now,
            report_interval,
            stall_checkpoint_bytes: started_bytes,
            stall_checkpoint_at: now,
        }
    }

    fn downloaded_bytes(&self) -> u64 {
        self.started_bytes.saturating_add(self.transferred_bytes)
    }

    fn record_transferred(&mut self, bytes: u64) {
        self.transferred_bytes = self.transferred_bytes.saturating_add(bytes);
    }

    /// Reset the stall watchdog to "made progress just now", granting the next
    /// (re)connect a full stall window before it can be judged hung. Called at the
    /// start of each attempt so the idle gap spent re-issuing the request is not
    /// counted against the transfer.
    fn reset_stall_clock(&mut self) {
        self.stall_checkpoint_bytes = self.downloaded_bytes();
        self.stall_checkpoint_at = Instant::now();
    }

    /// Advance the stall checkpoint when at least `min_progress` bytes have moved since
    /// it was last set, then report whether the transfer has now gone a full `timeout`
    /// without that much progress — i.e. it is hung and should be abort-and-resumed. A
    /// byte-at-a-time trickle never clears `min_progress`, so it is correctly flagged
    /// as stalled, unlike a naive "time since last byte" check.
    fn is_stalled(&mut self, timeout: Duration, min_progress: u64) -> bool {
        let current = self.downloaded_bytes();
        if current.saturating_sub(self.stall_checkpoint_bytes) >= min_progress {
            self.stall_checkpoint_bytes = current;
            self.stall_checkpoint_at = Instant::now();
        }
        self.stall_checkpoint_at.elapsed() >= timeout
    }

    fn discard_started_bytes(&mut self, bytes: u64) {
        self.started_bytes = self.started_bytes.saturating_sub(bytes);
    }

    fn report_interval(&self) -> Duration {
        self.report_interval
    }

    fn payload(&self) -> ProgressRequest {
        download_progress_payload(
            self.repo,
            self.downloaded_bytes(),
            self.total_bytes,
            self.started_bytes,
            self.started_at.elapsed(),
        )
    }
}

pub fn download_progress_payload(
    repo: &str,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    started_bytes: u64,
    elapsed: Duration,
) -> ProgressRequest {
    let transferred_bytes = downloaded_bytes.saturating_sub(started_bytes);
    let elapsed_seconds = elapsed.as_secs_f64().max(0.001);
    let rate = transferred_bytes as f64 / elapsed_seconds;
    let eta_seconds = total_bytes.and_then(|total| {
        if rate > 0.0 {
            let remaining = total.saturating_sub(downloaded_bytes) as f64;
            Some(number_from_f64((remaining / rate).max(0.0)))
        } else {
            None
        }
    });

    let (progress, message) = if let Some(total) = total_bytes {
        let ratio = if total == 0 {
            1.0
        } else {
            (downloaded_bytes as f64 / total as f64).clamp(0.0, 1.0)
        };
        let remaining = total.saturating_sub(downloaded_bytes);
        (
            0.1 + ratio * 0.85,
            format!(
                "Downloading {repo}: {} of {} ({} left).",
                format_bytes(downloaded_bytes),
                format_bytes(total),
                format_bytes(remaining)
            ),
        )
    } else {
        (
            0.1,
            format!(
                "Downloading {repo}: {} written.",
                format_bytes(downloaded_bytes)
            ),
        )
    };

    progress_payload(
        JobStatus::Downloading,
        ProgressStage::Downloading,
        progress,
        &message,
        None,
        None,
        eta_seconds,
    )
}

#[cfg(all(test, target_os = "macos"))]
mod ensure_cached_file_tests {
    use super::{ensure_cached_file, DownloadContext};
    use crate::{
        ApiClient, Settings, DEFAULT_HUGGINGFACE_BASE_URL, DEFAULT_MAX_LORA_URL_BYTES,
        DEFAULT_MAX_MODEL_URL_BYTES,
    };

    /// sc-4283 / F-MLXW-22: when the target already exists, `ensure_cached_file`
    /// returns it without any network access (the cache-hit short-circuit shared
    /// by all the download-on-first-use weight fetchers). A bogus URL proves no
    /// request is made.
    #[tokio::test]
    async fn returns_existing_target_without_downloading() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("weights").join("model.safetensors");
        tokio::fs::create_dir_all(target.parent().unwrap())
            .await
            .expect("parent dir");
        tokio::fs::write(&target, b"already here")
            .await
            .expect("seed target");

        let client = reqwest::Client::new();
        let settings = Settings {
            api_url: "http://127.0.0.1:1".to_owned(),
            access_token: None,
            data_dir: dir.path().join("data"),
            config_dir: dir.path().join("config"),
            worker_id: "test-worker".to_owned(),
            gpu_id: "cpu".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 5,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
            max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: true,
            backend_candle_enabled: false,
            gpu_memory_limit_bytes: 0,
            external_model_roots: Vec::new(),
        };
        let api = ApiClient::new(&settings);
        let context = DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        };
        let resolved = ensure_cached_file(
            &context,
            "http://invalid.invalid/should-not-fetch",
            &target,
            "test weights",
            Some(12),
        )
        .await
        .expect("cache hit returns without downloading");
        assert_eq!(resolved, target);
        // Content untouched (no overwrite).
        assert_eq!(
            tokio::fs::read(&target).await.expect("read"),
            b"already here"
        );
    }
}

/// sc-11149 (F-001): every outbound download client MUST carry a connect + read
/// timeout so a server that accepts the TCP connection but never responds cannot
/// wedge the worker at `send().await` forever. reqwest exposes no getter for its
/// configured timeouts, so these tests pin the intent (the constants) and prove the
/// read-timeout mechanism actually fires against a stalled peer.
#[cfg(test)]
mod http_timeout_tests {
    use super::{
        apply_download_timeouts, streaming_download_client, HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT,
    };
    use std::time::Duration;

    #[test]
    fn download_timeout_constants_are_bounded_and_ordered() {
        // Non-zero: a zero Duration would mean "no timeout" for connect and is a
        // footgun for read — guard against an accidental regression to that.
        assert!(HTTP_CONNECT_TIMEOUT > Duration::ZERO);
        assert!(HTTP_READ_TIMEOUT > Duration::ZERO);
        // The chunk-level read window is generous enough to outlast a slow-but-live
        // multi-GB stream, hence longer than the connect handshake budget.
        assert!(HTTP_READ_TIMEOUT >= HTTP_CONNECT_TIMEOUT);
    }

    #[test]
    fn streaming_client_builds() {
        // The `.expect(...)` in the constructor must never fire for the static config.
        let _client = streaming_download_client();
    }

    /// A server that accepts the connection and then goes silent must not hang the
    /// caller: with a read timeout, `send().await` returns an error rather than
    /// blocking forever (the exact pre-heartbeat wedge from the finding). The
    /// production read timeout is 60s; here we build the same client via the shared
    /// helper and override only the read window to keep the test fast.
    #[tokio::test]
    async fn request_to_a_stalled_server_times_out_instead_of_hanging() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback listener");
        let addr = listener.local_addr().expect("local addr");

        // Accept the connection but never send a byte of response.
        let server = tokio::spawn(async move {
            if let Ok((stream, _)) = listener.accept().await {
                // Hold the socket open, silent, well past the client's read timeout.
                tokio::time::sleep(Duration::from_secs(30)).await;
                drop(stream);
            }
        });

        let client = apply_download_timeouts(reqwest::Client::builder())
            .read_timeout(Duration::from_millis(400))
            .build()
            .expect("client builds");

        let started = std::time::Instant::now();
        let result = client.get(format!("http://{addr}/")).send().await;
        let elapsed = started.elapsed();

        assert!(
            result.is_err(),
            "a stalled server must surface a timeout error, not a response"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "the request should time out promptly (took {elapsed:?})"
        );

        server.abort();
    }
}

/// sc-download-hang: the forward-progress watchdog. `HTTP_READ_TIMEOUT` only catches a
/// *fully idle* stream (any received byte resets it), so a Hugging Face CDN edge that
/// dribbles bytes under that window keeps the transfer alive yet never advancing. These
/// tests pin the decision logic — advance-on-progress, flag-on-stall, reset-per-attempt
/// — that drives the abort-and-resume in `download_file_inner`.
#[cfg(test)]
mod stall_watchdog_tests {
    use super::{DownloadProgress, DOWNLOAD_STALL_MIN_PROGRESS};
    use std::time::Duration;

    fn progress() -> DownloadProgress<'static> {
        DownloadProgress::new("repo", 0, Some(1_000_000_000), Duration::from_millis(10))
    }

    /// A transfer that receives nothing is flagged stalled once the window elapses.
    #[test]
    fn flags_a_fully_idle_transfer() {
        let mut progress = progress();
        std::thread::sleep(Duration::from_millis(10));
        assert!(progress.is_stalled(Duration::from_millis(1), DOWNLOAD_STALL_MIN_PROGRESS));
    }

    /// The load-bearing case: a trickle *below* the progress threshold still counts as
    /// stalled, where a naive "time since last byte" check would be reset by every drip.
    #[test]
    fn flags_a_sub_threshold_trickle() {
        let mut progress = progress();
        // Far less than 1 MiB — exactly what a wedged CDN edge dribbles out.
        progress.record_transferred(64);
        std::thread::sleep(Duration::from_millis(10));
        assert!(progress.is_stalled(Duration::from_millis(1), DOWNLOAD_STALL_MIN_PROGRESS));
    }

    /// Real progress (≥ the threshold) advances the checkpoint, so a healthy transfer is
    /// never flagged even after a long elapsed time.
    #[test]
    fn ignores_a_healthy_transfer() {
        let mut progress = progress();
        std::thread::sleep(Duration::from_millis(10));
        progress.record_transferred(DOWNLOAD_STALL_MIN_PROGRESS);
        // The checkpoint advances on this call; with a generous window it is not stalled.
        assert!(!progress.is_stalled(Duration::from_secs(3600), DOWNLOAD_STALL_MIN_PROGRESS));
    }

    /// `reset_stall_clock` grants a resumed attempt a fresh window — the idle gap spent
    /// re-issuing the request must not immediately re-trip the watchdog.
    #[test]
    fn reset_grants_a_fresh_window() {
        let mut progress = progress();
        std::thread::sleep(Duration::from_millis(10));
        assert!(progress.is_stalled(Duration::from_millis(1), DOWNLOAD_STALL_MIN_PROGRESS));
        progress.reset_stall_clock();
        assert!(!progress.is_stalled(Duration::from_secs(3600), DOWNLOAD_STALL_MIN_PROGRESS));
    }
}
