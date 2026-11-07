use super::*;

pub(super) const GRID_THUMBNAIL_SIZE: u32 = 384;

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ProjectFileQuery {
    pub(super) thumbnail: Option<u32>,
    /// Serve the file with any embedded `sceneworks:workflow` chunk removed (sc-15953).
    ///
    /// A query param on the existing route rather than a new endpoint, because the browser
    /// "Save As without the workflow" is a bare `<a download>` and cannot transform bytes or set
    /// headers. That constraint also decides the shape: in remote/LAN mode the anchor authenticates
    /// with a `?ticket=` media ticket, whose allow-list (`auth::is_ticketed_media_path`) matches on
    /// the PATH and is GET-only — so keeping the strip on this path means the download works
    /// remotely with no change to the ticket rules at all. A separate endpoint would have needed a
    /// new entry in that allow-list, which is the surface hardest to widen safely.
    pub(super) strip_workflow: Option<bool>,
}

pub(super) async fn get_project_file(
    State(state): State<AppState>,
    Path((project_id, relative_path)): Path<(String, String)>,
    Query(query): Query<ProjectFileQuery>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let project_file = project_call(state.clone(), {
        let project_id = project_id.clone();
        let relative_path = relative_path.clone();
        move |store| store.project_file(&project_id, &relative_path)
    })
    .await
    .inspect_err(|error| {
        // The generic 4xx logger (error.rs) records only `status` + `detail`, so a
        // bare "File not found" line can't be traced back to a file. Name the missing
        // resource here — mirroring the `auth_rejected`/`auth_throttled` structured
        // logs — so operators can see which asset the web UI requested. The common
        // startup culprits are a video's `<name>.poster.jpg` that was never generated
        // and an asset purged from disk but still referenced by the project; the web
        // UI degrades both to a placeholder (assetMedia.jsx), so the only trace is here.
        if error.status == StatusCode::NOT_FOUND {
            tracing::debug!(
                event = "project_file_missing",
                project_id = %project_id,
                relative_path = %relative_path,
                status = error.status.as_u16(),
                "requested project file not found"
            );
        }
    })?;

    // "Download without the workflow" (sc-15953). Answered here, before the thumbnail and
    // streaming branches, because it is the one variant whose body is not the file on disk.
    if query.strip_workflow.unwrap_or(false) {
        if query.thumbnail.is_some() {
            return Err(ApiError::bad_request(
                "stripWorkflow cannot be combined with thumbnail",
            ));
        }
        return stripped_project_file_response(&state, &project_file, &headers).await;
    }

    // One fixed representation prevents unbounded cache variants. Derivatives
    // are generated on first use, so assets written before this route existed
    // backfill without a migration.
    let force_download = project_file
        .content_type
        .eq_ignore_ascii_case("image/svg+xml");
    if force_download && query.thumbnail.is_some() {
        // Do not let the generic thumbnail branch become a second inline SVG preview route. Vector
        // sidecars point at the worker-rendered PNG instead.
        return Err(ApiError::bad_request(
            "SVG thumbnails are not supported; use the vector PNG preview",
        ));
    }
    let (served_path, content_type) = if let Some(size) = query.thumbnail {
        if size != GRID_THUMBNAIL_SIZE {
            return Err(ApiError::bad_request(format!(
                "thumbnail must be {GRID_THUMBNAIL_SIZE}"
            )));
        }
        if !project_file.content_type.starts_with("image/") {
            return Err(ApiError::bad_request(
                "thumbnail is only available for image media",
            ));
        }
        let permit = state
            .thumbnail_generation_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let source_path = project_file.path.clone();
        let cache_root = state
            .settings
            .data_dir
            .join("cache")
            .join("media-thumbnails")
            .join("v1");
        let thumbnail_path = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            ensure_grid_thumbnail(&source_path, &cache_root)
        })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(|error| {
            tracing::debug!(
                event = "project_thumbnail_unavailable",
                project_id = %project_id,
                relative_path = %relative_path,
                error = %error,
                "could not create bounded project thumbnail"
            );
            ApiError::bad_request("Thumbnail unavailable for this media")
        })?;
        (thumbnail_path, "image/png".to_owned())
    } else {
        (project_file.path, project_file.content_type)
    };

    let mut file = tokio::fs::File::open(&served_path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let metadata = file
        .metadata()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let total = metadata.len();
    let etag = project_file_etag(&metadata);
    let last_modified = metadata.modified().ok();
    let base_headers = project_file_response_headers(&content_type, total, &etag, last_modified)?;

    if project_file_is_not_modified(&headers, &etag, last_modified) {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = StatusCode::NOT_MODIFIED;
        *response.headers_mut() = base_headers;
        response.headers_mut().remove(header::CONTENT_LENGTH);
        return Ok(response);
    }

    // WebKit/WKWebView (the macOS desktop webview) requires HTTP byte-range
    // responses to play <video>: it probes with `Range: bytes=0-1` and treats
    // any 200 reply as a non-seekable source it won't play. Honor a single
    // range with 206 Partial Content; advertise Accept-Ranges otherwise.
    if let Some(range_header) = headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
        match parse_single_byte_range(range_header, total) {
            Some((start, end)) => {
                let len = end - start + 1;
                file.seek(SeekFrom::Start(start))
                    .await
                    .map_err(|error| ApiError::internal(error.to_string()))?;
                let stream = ReaderStream::new(file.take(len));
                let mut response = (
                    StatusCode::PARTIAL_CONTENT,
                    [
                        (header::CONTENT_TYPE, content_type.clone()),
                        (header::ACCEPT_RANGES, "bytes".to_string()),
                        (
                            header::CONTENT_RANGE,
                            format!("bytes {start}-{end}/{total}"),
                        ),
                        (header::CONTENT_LENGTH, len.to_string()),
                        // sc-9674 (sc-8872 follow-up): forbid MIME sniffing so a
                        // user-controlled project file can't be reinterpreted by the
                        // browser as a different (e.g. active) content type than the
                        // Content-Type we derived. Kept inline (no attachment
                        // disposition) so <img>/<video> preview and byte-range
                        // playback still work — the assets are served for inline
                        // display, not forced download.
                        (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()),
                    ],
                    Body::from_stream(stream),
                )
                    .into_response();
                response.headers_mut().extend(base_headers);
                response.headers_mut().insert(
                    header::CONTENT_LENGTH,
                    HeaderValue::from_str(&len.to_string())
                        .map_err(|error| ApiError::internal(error.to_string()))?,
                );
                return Ok(response);
            }
            None => {
                let mut response = (
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    [(header::CONTENT_RANGE, format!("bytes */{total}"))],
                )
                    .into_response();
                response.headers_mut().extend(base_headers);
                response.headers_mut().remove(header::CONTENT_LENGTH);
                return Ok(response);
            }
        }
    }

    let stream = ReaderStream::new(file);
    let mut response = (
        [
            (header::CONTENT_TYPE, content_type),
            (header::ACCEPT_RANGES, "bytes".to_string()),
            (header::CONTENT_LENGTH, total.to_string()),
            // sc-9674: forbid MIME sniffing (see the range branch above). Inline
            // disposition is kept intentionally so image/video preview still works.
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()),
        ],
        Body::from_stream(stream),
    )
        .into_response();
    response.headers_mut().extend(base_headers);
    Ok(response)
}

pub(super) fn ensure_grid_thumbnail(
    source_path: &FsPath,
    cache_root: &FsPath,
) -> Result<PathBuf, String> {
    let metadata = std::fs::metadata(source_path).map_err(|error| error.to_string())?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    let mut hash = Sha256::new();
    hash.update(source_path.to_string_lossy().as_bytes());
    hash.update(metadata.len().to_le_bytes());
    hash.update(modified.as_nanos().to_le_bytes());
    hash.update(GRID_THUMBNAIL_SIZE.to_le_bytes());
    let key = format!("{:x}", hash.finalize());
    let target = cache_root.join(format!("{key}.png"));
    if target.is_file() {
        return Ok(target);
    }

    std::fs::create_dir_all(cache_root).map_err(|error| error.to_string())?;
    let decoded = match image::open(source_path) {
        Ok(decoded) => decoded,
        Err(decode_error) => {
            // Project imports normalize these formats today, but old projects
            // can still contain AVIF/HEIC/HEIF (and other platform-decodable
            // raster files). Reuse the worker's cross-platform compatibility
            // path before declaring the thumbnail unavailable.
            let converted =
                cache_root.join(format!("{key}.{}.converted.png", Uuid::new_v4().simple()));
            let transcode_result =
                sceneworks_core::media_convert::transcode_to_png(source_path, &converted);
            let decoded = transcode_result
                .map_err(|error| format!("{decode_error}; {error}"))
                .and_then(|()| image::open(&converted).map_err(|error| error.to_string()));
            let _ = std::fs::remove_file(&converted);
            decoded?
        }
    };
    let thumbnail = decoded.thumbnail(GRID_THUMBNAIL_SIZE, GRID_THUMBNAIL_SIZE);
    let temporary = cache_root.join(format!("{key}.{}.tmp", Uuid::new_v4().simple()));
    thumbnail
        .save_with_format(&temporary, image::ImageFormat::Png)
        .map_err(|error| error.to_string())?;
    if let Err(error) = std::fs::rename(&temporary, &target) {
        if target.is_file() {
            let _ = std::fs::remove_file(&temporary);
        } else {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.to_string());
        }
    }
    Ok(target)
}

pub(super) async fn project_path_for_id(
    state: AppState,
    project_id: &str,
) -> Result<PathBuf, ApiError> {
    let project_id = project_id.to_owned();
    let project = project_call(state, move |store| store.get_project(&project_id)).await?;
    Ok(PathBuf::from(project.path))
}

/// Resolve a project asset record to an on-disk path without allowing absolute
/// paths or traversal outside the owning project.
pub(super) async fn resolve_project_confined_asset_path(
    state: AppState,
    project_id: &str,
    asset_id: &str,
    _project_path: &FsPath,
) -> Result<PathBuf, ApiError> {
    // Store resolves the owning root, never a caller-supplied path.
    let project_id = project_id.to_owned();
    let asset_id = asset_id.to_owned();
    project_call(state, move |store| {
        let asset = store.get_asset(&project_id, &asset_id)?;
        let relative_path = asset
            .get("file")
            .and_then(|file| file.get("path"))
            .and_then(Value::as_str)
            .ok_or_else(|| ProjectStoreError::BadRequest("Asset has no file path".into()))?;
        store
            .project_file(&project_id, relative_path)
            .map(|file| file.path)
    })
    .await
}
