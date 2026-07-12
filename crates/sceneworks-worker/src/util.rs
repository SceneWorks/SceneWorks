//! Small worker-wide utilities: byte/time formatting, asset ids, HF auth, and directory sizing.
use super::*;

pub(crate) async fn directory_size(path: &Path) -> u64 {
    let mut total = 0_u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(path) = stack.pop() {
        let mut entries = match tokio::fs::read_dir(&path).await {
            Ok(entries) => entries,
            Err(error) => {
                // A missing directory is the normal start-of-a-fresh-download state (the HF
                // `blobs/` dir does not exist until the first file lands), so it means "0 bytes
                // so far", not a failure — don't log it at error level. Only surface genuine I/O
                // problems (permissions, etc.).
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        event = "rust_worker_directory_size_failed",
                        path = %path.display(),
                        error = %error,
                        "failed to read a directory while sizing a download"
                    );
                }
                continue;
            }
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let Ok(file_type) = entry.file_type().await else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if file_type.is_file() && entry.file_name() != INSTALL_MARKER {
                if let Ok(metadata) = entry.metadata().await {
                    total = total.saturating_add(metadata.len());
                }
            }
        }
    }
    total
}

pub fn format_bytes(value: u64) -> String {
    let mut size = value as f64;
    for unit in ["B", "KB", "MB", "GB", "TB"] {
        if size < 1024.0 || unit == "TB" {
            if unit == "B" {
                return format!("{} {unit}", size as u64);
            }
            return format!("{size:.1} {unit}");
        }
        size /= 1024.0;
    }
    format!("{size:.1} TB")
}

pub(crate) fn quote_path(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            output.push(char::from(byte));
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

pub(crate) fn now_rfc3339() -> String {
    format_unix_seconds(now_unix_seconds())
}

pub(crate) fn bounded_tail(value: &str, max_lines: usize, max_chars: usize) -> String {
    let mut lines = value.lines().rev().take(max_lines).collect::<Vec<_>>();
    lines.reverse();
    let mut output = lines.join("\n");
    if output.len() > max_chars {
        let start = output
            .char_indices()
            .rev()
            .nth(max_chars)
            .map_or(0, |(index, _)| index);
        output = output[start..].to_owned();
    }
    output
}

pub(crate) fn fresh_asset_id() -> String {
    format!("asset_{}", Uuid::new_v4().simple())
}

pub(crate) fn asset_suffix(value: &str) -> String {
    let safe = safe_download_dir(value);
    let chars = safe.chars().rev().take(8).collect::<Vec<_>>();
    chars.into_iter().rev().collect::<String>()
}

pub(crate) async fn existing_download_bytes(
    path: &Path,
    expected_size: Option<u64>,
) -> WorkerResult<u64> {
    let Ok(metadata) = tokio::fs::metadata(path).await else {
        return Ok(0);
    };
    let existing = metadata.len();
    if expected_size.is_some_and(|expected_size| existing > expected_size) {
        tokio::fs::remove_file(path).await?;
        return Ok(0);
    }
    Ok(existing)
}

pub(crate) async fn with_hf_auth(
    settings: &Settings,
    request: reqwest::RequestBuilder,
) -> reqwest::RequestBuilder {
    // Resolves the HF token lazily: the env `HF_TOKEN` (server/Docker/Windows) or,
    // on the macOS desktop, a one-time pull of the recorded `huggingface.co`
    // credential from the desktop socket (sc-5891). `None` ⇒ unauthenticated.
    match credentials_ipc::resolve_hf_token(settings).await {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

/// Resolve an explicit env-pinned weight *file* (sc-8911, sc-11175/F-011). Unset →
/// `Ok(None)` (fall through to cache/download). Set + existing → `Ok(Some(path))`. Set but
/// missing → an `InvalidPayload` error so a typo fails loudly instead of silently loading
/// whatever the download resolves. `what` names the expected file in the error. Takes the
/// raw value explicitly so it's unit-testable without mutating the process environment.
///
/// This is the worker-wide house helper the loud-env-pin fixes route through: the
/// upscaler (`SCENEWORKS_REALESRGAN_*_ONNX`, sc-8911), the person detector
/// (`SCENEWORKS_PERSON_DETECTOR_WEIGHTS`), and SAM2/SAM3
/// (`SCENEWORKS_SAM2_WEIGHTS`/`SCENEWORKS_SAM3_WEIGHTS`, sc-11175/F-011).
pub(crate) fn resolve_env_file_pin(
    key: &str,
    value: Option<std::ffi::OsString>,
    what: &str,
) -> WorkerResult<Option<PathBuf>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let path = PathBuf::from(&value);
    if path.exists() {
        return Ok(Some(path));
    }
    Err(WorkerError::InvalidPayload(format!(
        "{key} is set to {} but that path does not exist. Point it at {what}, or unset it to download on first use.",
        path.display()
    )))
}
