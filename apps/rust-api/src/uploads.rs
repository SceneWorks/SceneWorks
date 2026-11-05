//! Multipart staging ownership; callers retain their accepted fields, limits and error messages.
use crate::{stream_multipart_field_to_file, ApiError, AppState, Multipart, MAX_UPLOAD_BYTES};
use axum::extract::multipart::Field;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// `filename` selects a private per-upload directory. Without it, the caller's cache directory
/// owns uniquely named temporary files and must survive cleanup (assets/training/pose/keypoint).
pub(crate) async fn stage_upload(
    state: &AppState,
    field: Field<'_>,
    subdir: &str,
    filename: Option<&str>,
    limit: usize,
    limit_message: impl FnOnce() -> String,
) -> Result<PathBuf, ApiError> {
    let root = state.settings.data_dir.join("cache").join(subdir);
    let (directory, name) = match filename {
        Some(name) => (
            root.join(format!("upload-{}", Uuid::new_v4().simple())),
            name.to_owned(),
        ),
        None => (root, format!("upload-{}.tmp", Uuid::new_v4().simple())),
    };
    tokio::fs::create_dir_all(&directory)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let path = directory.join(name);
    stream_multipart_field_to_file(field, &path, limit, limit_message, || {
        cleanup_staged_upload(&path)
    })
    .await?;
    Ok(path)
}

/// Best-effort cleanup of a staged file and its empty private upload directory. Shared cache
/// roots have no `upload-<uuid>` name and remain available to the other in-flight requests.
pub(crate) async fn cleanup_staged_upload(path: &Path) {
    let _ = tokio::fs::remove_file(path).await;
    if let Some(parent) = path.parent().filter(|parent| {
        parent
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("upload-"))
    }) {
        let _ = tokio::fs::remove_dir(parent).await;
    }
}

/// The pose and keypoint source routes accept the same ordered repeated `file` fields.
pub(crate) async fn stage_image_sources(
    state: &AppState,
    multipart: &mut Multipart,
    subdir: &str,
) -> Result<Vec<Value>, ApiError> {
    let mut sources = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?
    {
        if field.name() != Some("file") {
            continue;
        }
        let display_name = field.file_name().unwrap_or("image").to_owned();
        let path = stage_upload(state, field, subdir, None, MAX_UPLOAD_BYTES, || {
            "Uploaded file is too large".to_owned()
        })
        .await?;
        sources.push(json!({"path": path.to_string_lossy(), "displayName": display_name}));
    }
    if sources.is_empty() {
        return Err(ApiError::bad_request("At least one image file is required"));
    }
    Ok(sources)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cleanup_removes_private_staging_but_retains_shared_and_nonempty_directories() {
        let temp = tempfile::tempdir().unwrap();
        let private = temp.path().join("upload-0123456789abcdef");
        let shared = temp.path().join("pose-uploads");
        for directory in [&private, &shared] {
            tokio::fs::create_dir_all(directory).await.unwrap();
            let file = directory.join("source.png");
            tokio::fs::write(&file, b"metadata-only staged bytes")
                .await
                .unwrap();
            cleanup_staged_upload(&file).await;
            assert!(!file.exists());
        }
        assert!(!private.exists());
        assert!(shared.exists());
        let private = temp.path().join("upload-still-owned");
        tokio::fs::create_dir_all(&private).await.unwrap();
        let retained = private.join("another-file");
        tokio::fs::write(&retained, b"another owner").await.unwrap();
        let file = private.join("aborted");
        tokio::fs::write(&file, b"aborted field").await.unwrap();
        cleanup_staged_upload(&file).await;
        assert!(retained.is_file());
    }
}
