//! Key Point Library API (epic 4422, sc-4434): face-angle presets (kps + retained source image)
//! and user angle-set collections, the face-angle sibling of the Pose Library (`poses.rs`).
//! Storage lives in the reserved global keypoints project; the built-in 11 presets + the default
//! collection are virtual (served from `sceneworks_core::angle_kps`, never stored).

use super::*;

/// Persist a user keypoint preset from a previously-extracted kps + a staged source image.
/// Body: `{ name, kps:[[x,y]×5], sourceUploadPath, sourceAssetId? }` (validated by
/// `ProjectStore::create_keypoint_asset`). Returns the created asset.
pub(crate) async fn create_keypoint(
    State(state): State<AppState>,
    ApiJson(spec): ApiJson<serde_json::Value>,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    let created = project_call(state, move |store| store.create_keypoint_asset(&spec)).await?;
    Ok((StatusCode::CREATED, Json(created)))
}

/// Stage a File-Upload source image for kps extraction / preset capture as a TEMPORARY file
/// (mirrors `create_pose_sources`): the UI uploads the photo here, runs a `kps_extract` job on
/// the returned path, then saves a preset referencing it (the save copies it into the library).
/// A startup sweep is the backstop for paths that are never saved.
pub(crate) async fn create_keypoint_sources(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
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
        let path = write_upload_field_to_dir(&state, field, "keypoint-uploads").await?;
        sources.push(serde_json::json!({
            "path": path.to_string_lossy(),
            "displayName": display_name,
        }));
    }
    if sources.is_empty() {
        return Err(ApiError::bad_request("At least one image file is required"));
    }
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "sources": sources })),
    ))
}

/// The full preset list the library renders: the built-in 11 + the user's stored presets.
pub(crate) async fn list_keypoint_presets(
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let presets = project_call(state, move |store| store.list_keypoint_presets()).await?;
    Ok(Json(presets))
}

/// All angle-set collections: the virtual built-in default + the user's collections.
pub(crate) async fn list_keypoint_collections(
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let collections = project_call(state, move |store| store.list_keypoint_collections()).await?;
    Ok(Json(collections))
}

/// Create or update a user angle-set collection. Body:
/// `{ id?, name, orderedPresetIds[], isDefault? }`. Returns the saved collection.
pub(crate) async fn upsert_keypoint_collection(
    State(state): State<AppState>,
    ApiJson(spec): ApiJson<serde_json::Value>,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    let saved = project_call(state, move |store| store.upsert_keypoint_collection(&spec)).await?;
    Ok((StatusCode::CREATED, Json(saved)))
}

/// Mark a collection the default (the built-in id resets to the built-in default). Returns the
/// updated collection list.
pub(crate) async fn set_default_keypoint_collection(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let collections = project_call(state, move |store| {
        store.set_default_keypoint_collection(&collection_id)
    })
    .await?;
    Ok(Json(collections))
}

/// Delete a user collection (the built-in default cannot be deleted). Returns the updated list.
pub(crate) async fn delete_keypoint_collection(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let collections = project_call(state, move |store| {
        store.delete_keypoint_collection(&collection_id)?;
        store.list_keypoint_collections()
    })
    .await?;
    Ok(Json(collections))
}

/// Remove stale keypoint-source temp uploads at startup (the backstop for captures that were
/// never saved). Mirrors `sweep_stale_pose_uploads`.
pub(crate) fn sweep_stale_keypoint_uploads(data_dir: &FsPath) -> std::io::Result<usize> {
    let cutoff = SystemTime::now() - Duration::from_secs(STALE_LORA_UPLOAD_SECONDS);
    let upload_root = data_dir.join("cache").join("keypoint-uploads");
    let entries = match std::fs::read_dir(upload_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut removed = 0usize;
    for entry in entries {
        let entry = entry?;
        let filename = entry.file_name();
        if !filename.to_string_lossy().starts_with("upload-") {
            continue;
        }
        let modified = entry.metadata()?.modified().unwrap_or(UNIX_EPOCH);
        if modified <= cutoff {
            let path = entry.path();
            let _ = if entry.file_type()?.is_dir() {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
            removed += 1;
        }
    }
    Ok(removed)
}
