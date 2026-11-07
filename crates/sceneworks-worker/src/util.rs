//! Small worker-wide utilities: byte/time formatting, asset ids, and HF auth.
use super::*;

pub use sceneworks_core::display_util::format_bytes;

pub(crate) use sceneworks_core::display_util::quote_path;

pub(crate) use sceneworks_core::time::utc_now as now_rfc3339;

#[cfg(test)]
pub(crate) use sceneworks_core::media_process::bounded_tail;

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
///
/// Gated to the union of its callers' cfgs (upscaler / person detector / SAM2 / SAM3),
/// all of which compile only on macOS OR the off-Mac candle lane. On the Linux `parity`
/// lane (non-macOS, default features) none of those callers compile, so leaving this
/// helper ungated makes it dead code under `-D warnings` (sc-11175, Linux parity fix).
#[cfg(native_lane)]
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

#[cfg(native_lane)]
pub(crate) fn resolved_cache_paths_match<const N: usize>(
    cached: [&Path; N],
    requested: [&Path; N],
) -> bool {
    cached == requested
}

#[cfg(all(test, native_lane))]
mod cache_path_tests {
    use super::*;

    #[test]
    fn resolved_cache_identity_changes_with_any_weight_path() {
        let det_a = Path::new("/weights/a/det.onnx");
        let det_b = Path::new("/weights/b/det.onnx");
        let pose_a = Path::new("/weights/a/pose.onnx");
        let pose_b = Path::new("/weights/b/pose.onnx");
        assert!(resolved_cache_paths_match([det_a, pose_a], [det_a, pose_a]));
        assert!(!resolved_cache_paths_match(
            [det_a, pose_a],
            [det_b, pose_a]
        ));
        assert!(!resolved_cache_paths_match(
            [det_a, pose_a],
            [det_a, pose_b]
        ));
    }
}

pub(crate) fn gradient_rgb8(width: u32, height: u32, seed: i64) -> Vec<u8> {
    let seed = seed as u64;
    let base = [
        (seed & 0xFF) as u8,
        ((seed >> 8) & 0xFF) as u8,
        ((seed >> 16) & 0xFF) as u8,
    ];
    let span = height.saturating_sub(1).max(1) as f32;
    let mut buffer = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for y in 0..height {
        let t = y as f32 / span;
        let row = [lerp(base[0], t), lerp(base[1], t), lerp(base[2], t)];
        for _ in 0..width {
            buffer.extend_from_slice(&row);
        }
    }
    buffer
}

pub(crate) fn lerp(a: u8, t: f32) -> u8 {
    let a = a as f32;
    (a + (255.0 - a) * t).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod gradient_tests {
    use super::*;
    #[test]
    fn gradient_preserves_seed_bytes_white_endpoint_and_degenerate_dimensions() {
        assert_eq!(gradient_rgb8(1, 1, 0x030201), vec![1, 2, 3]);
        assert_eq!(
            gradient_rgb8(2, 2, 0x030201),
            vec![1, 2, 3, 1, 2, 3, 255, 255, 255, 255, 255, 255]
        );
        assert!(gradient_rgb8(0, 8, 7).is_empty());
        assert!(gradient_rgb8(8, 0, 7).is_empty());
    }
}
