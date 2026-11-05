use super::*;

pub(super) const PROJECT_MEDIA_CACHE_CONTROL: &str = "private, max-age=31536000, immutable";

/// How many `?stripWorkflow=true` downloads may be in flight at once (sc-15953).
///
/// Two, mirroring `thumbnail_generation_slots` — the other derived representation of this same
/// route, and the precedent for gating one. A strip is not CPU-bound the way a thumbnail is; what
/// it holds is MEMORY, for as long as the response body is being written to a client that may be
/// on a slow link. Ungated, the ceiling was the product of the file size and the number of
/// simultaneous callers, with nothing in the path to say no.
pub(super) const WORKFLOW_STRIP_SLOTS: usize = 2;

/// Largest PNG this route will rewrite in memory to remove a workflow chunk. 128 MiB.
///
/// The walk needs the tail of the file, so the whole thing is read; the cost therefore follows the
/// asset. A generated image is nowhere near this — measured through the repo's own writer on an
/// INCOMPRESSIBLE render, the worst case for a PNG, 1024² is 3.0 MiB, 2048² is 12.0 MiB and 4096²
/// is 48.0 MiB, and 4096² is the largest thing SceneWorks renders. An IMPORTED asset is bounded
/// only by `MAX_UPLOAD_BYTES` = 2 GiB, which is what this exists to refuse. 128 MiB is ~2.7× the
/// largest file the writer produces, so nothing SceneWorks made is ever turned away, and with
/// [`WORKFLOW_STRIP_SLOTS`] it bounds the route at 256 MiB rather than at multiple gigabytes.
///
/// Refused rather than silently served whole: "here is your copy without the workflow" answered
/// with the original file is the one outcome this feature must never produce, so an image too
/// large to rewrite gets a 413 that says why.
pub(super) const MAX_WORKFLOW_STRIP_BYTES: u64 = 128 * 1024 * 1024;

/// Serve a project file with any embedded workflow chunk excised (sc-15953).
///
/// The stripping itself is `sceneworks_core::workflow_png::strip_workflow_chunk` — a byte-level
/// walk of the PNG chunk framing that copies IHDR, every IDAT and IEND through unchanged, so the
/// served body is the file minus one slice rather than a re-encode. Nothing here decodes a pixel.
///
/// Five things this branch has to get right that the streaming path does not:
///
/// * **Only a PNG is read into memory.** The workflow chunk only ever exists in a PNG, so anything
///   else is served by the ordinary path — a video does not get buffered because someone appended
///   the flag to its URL. A non-PNG asked for stripped is not an error either: it genuinely carries
///   no workflow, and saying so by serving it is the honest answer.
/// * **The ETag must differ from the full file's.** `project_file_etag` is derived from the source
///   metadata, and the route sets `immutable` caching — so reusing it would let a cache answer a
///   strip request with the unstripped body it already holds. The variant tag is folded in.
/// * **Revalidation is ETag-ONLY.** The distinct ETag is half a fix on its own, and the missing
///   half was a live hole: both representations are derived from the same file, so they carry the
///   same `Last-Modified`, and a client that had done a plain GET could hand that date back on a
///   strip request and be told `304 Not Modified` — reusing the full body, workflow intact. The
///   date cannot distinguish the two representations, so it does not get a vote here; only the
///   variant tag does. `project_file_is_not_modified` still owns the ordinary path, where one URL
///   means one body and a date is a sound answer.
/// * **No byte ranges.** The body is a rewritten buffer whose offsets do not match the file on
///   disk, so `Accept-Ranges: none` and a `Range` header is ignored rather than answered against
///   the wrong length. This is a download, not a `<video src>`.
/// * **The memory is bounded, twice over.** The whole file is read at once — a walk of the chunk
///   framing has to see the tail — so the cost scales with the asset, and an imported PNG is
///   bounded only by `MAX_UPLOAD_BYTES` = 2 GiB. [`WORKFLOW_STRIP_SLOTS`] caps the concurrency
///   the way `thumbnail_generation_slots` does for the sibling derived representation on this same
///   route, and [`MAX_WORKFLOW_STRIP_BYTES`] caps the per-request size. Between them the route's
///   ceiling is a fixed number rather than "however many clients ask at once".
///
/// The body is assembled without a second copy. `strip_workflow_chunk` would build a new buffer
/// of its own — ~2N peak for a file of N bytes — so this calls the walk that underlies it
/// (`workflow_chunk_spans`) and serves the KEPT spans as `Bytes` slices of the one buffer already
/// read. `Bytes` slices are refcounted views, so the peak is N and the excision costs nothing.
pub(super) async fn stripped_project_file_response(
    state: &AppState,
    project_file: &sceneworks_core::project_store::ProjectFile,
    request_headers: &HeaderMap,
) -> Result<Response, ApiError> {
    let path = project_file.path.clone();
    let metadata = tokio::fs::metadata(&path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let last_modified = metadata.modified().ok();

    if project_file.content_type != "image/png" {
        // Nothing of ours can be in it. Fall back to the ordinary streaming response so a video or
        // a JPEG is served exactly as it always was, under its own ETag — and under the ordinary
        // revalidation rules too, because this is not a variant: the bytes are the file's.
        let etag = project_file_etag(&metadata);
        let headers = project_file_response_headers(
            &project_file.content_type,
            metadata.len(),
            &etag,
            last_modified,
        )?;
        if project_file_is_not_modified(request_headers, &etag, last_modified) {
            return Ok(not_modified_response(headers));
        }
        let file = tokio::fs::File::open(&path)
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let mut response = Body::from_stream(ReaderStream::new(file)).into_response();
        response.headers_mut().extend(headers);
        return Ok(response);
    }

    let etag = variant_etag(&project_file_etag(&metadata), "nw");
    if if_none_match_matches(request_headers, &etag) {
        let headers = project_file_response_headers(
            &project_file.content_type,
            metadata.len(),
            &etag,
            last_modified,
        )?;
        return Ok(not_modified_response(headers));
    }

    if metadata.len() > MAX_WORKFLOW_STRIP_BYTES {
        return Err(ApiError::payload_too_large(format!(
            "This image is {} MiB, over the {} MiB SceneWorks will rewrite in memory to remove a \
             workflow. Save it normally and remove the block with another tool.",
            metadata.len() / (1024 * 1024),
            MAX_WORKFLOW_STRIP_BYTES / (1024 * 1024)
        )));
    }

    let permit = state
        .workflow_strip_slots
        .clone()
        .acquire_owned()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;

    let (bytes, spans) = tokio::task::spawn_blocking(move || {
        let original = std::fs::read(&path)?;
        // A PNG that cannot be accounted for chunk by chunk is an ERROR, never the original bytes:
        // "here is your copy without the workflow" must not be answered with a file that may still
        // carry one. An empty span list is the honest "nothing of ours is in it".
        let spans = sceneworks_core::workflow_png::workflow_chunk_spans(&original)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok::<_, std::io::Error>((original, spans))
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))?
    .map_err(|error| {
        tracing::debug!(
            event = "project_file_strip_failed",
            error = %error,
            "could not serve a project file without its workflow"
        );
        ApiError::bad_request("This image could not be rewritten without its workflow.")
    })?;

    // One allocation for the whole response: `Bytes::from` takes the read buffer over, and every
    // slice below is a refcounted view into it rather than a copy.
    let bytes = axum::body::Bytes::from(bytes);
    let kept = sceneworks_core::workflow_png::kept_spans(bytes.len(), &spans);
    let served: Vec<axum::body::Bytes> = kept
        .iter()
        .map(|span| bytes.slice(span.start..span.end))
        .collect();
    let served_len: u64 = served.iter().map(|slice| slice.len() as u64).sum();

    let mut headers = project_file_response_headers(
        &project_file.content_type,
        served_len,
        &etag,
        last_modified,
    )?;
    headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("none"));
    // The permit rides with the stream rather than being dropped here: the buffer is alive until
    // the last slice has been written, so releasing the slot at this line would let the next
    // request in while this one's memory is still held.
    let mut response = Body::from_stream(stream::iter(served.into_iter().map(move |slice| {
        let _permit = &permit;
        Ok::<_, std::convert::Infallible>(slice)
    })))
    .into_response();
    response.headers_mut().extend(headers);
    Ok(response)
}

/// Whether `If-None-Match` names `etag` (or `*`).
///
/// The ETag half of [`project_file_is_not_modified`], on its own, for the derived representations
/// whose `Last-Modified` is the source file's and therefore cannot tell one representation from
/// another.
pub(super) fn if_none_match_matches(request_headers: &HeaderMap, etag: &str) -> bool {
    request_headers
        .get(header::IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(',')
                .map(str::trim)
                .any(|candidate| candidate == "*" || candidate == etag)
        })
}

/// A 304 carrying `headers` minus the content length, which a body-less response must not declare.
pub(super) fn not_modified_response(headers: HeaderMap) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = StatusCode::NOT_MODIFIED;
    *response.headers_mut() = headers;
    response.headers_mut().remove(header::CONTENT_LENGTH);
    response
}

/// Distinguish a derived representation's ETag from the source file's.
///
/// The route caches `immutable`, so two representations of one URL sharing a tag is a correctness
/// bug and not a missed optimization: a client holding the full file would revalidate a strip
/// request as `304 Not Modified` and reuse the body with the workflow still in it.
pub(super) fn variant_etag(base: &str, variant: &str) -> String {
    match base.strip_suffix('"') {
        Some(head) => format!("{head}-{variant}\""),
        None => format!("{base}-{variant}"),
    }
}

pub(super) fn project_file_etag(metadata: &std::fs::Metadata) -> String {
    let modified = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    // Weak avoids reading and hashing a multi-gigabyte original before it can
    // stream; project media is immutable after publication.
    format!(
        "W/\"{:x}-{:x}-{:x}\"",
        metadata.len(),
        modified.as_secs(),
        modified.subsec_nanos()
    )
}

pub(super) fn project_file_response_headers(
    content_type: &str,
    total: u64,
    etag: &str,
    last_modified: Option<SystemTime>,
) -> Result<HeaderMap, ApiError> {
    let mut headers = HeaderMap::new();
    // Every representation, including stripWorkflow, ranges and 304 responses, shares this
    // boundary. SVG source is download-only; only the worker-rendered PNG is inline media.
    let content_type = if content_type.eq_ignore_ascii_case("image/svg+xml") {
        headers.insert(
            header::CONTENT_DISPOSITION,
            HeaderValue::from_static("attachment"),
        );
        "application/octet-stream"
    } else {
        content_type
    };
    for (name, value) in [
        (header::CONTENT_TYPE, content_type.to_owned()),
        (header::ACCEPT_RANGES, "bytes".to_owned()),
        (header::CONTENT_LENGTH, total.to_string()),
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_owned()),
        (
            header::CACHE_CONTROL,
            PROJECT_MEDIA_CACHE_CONTROL.to_owned(),
        ),
        (header::ETAG, etag.to_owned()),
    ] {
        headers.insert(
            name,
            HeaderValue::from_str(&value).map_err(|error| ApiError::internal(error.to_string()))?,
        );
    }
    if let Some(modified) = last_modified {
        headers.insert(
            header::LAST_MODIFIED,
            HeaderValue::from_str(&httpdate::fmt_http_date(modified))
                .map_err(|error| ApiError::internal(error.to_string()))?,
        );
    }
    Ok(headers)
}

/// Ordinary revalidation for a representation whose bytes ARE the file's: the ETag when the client
/// sent one, the modification date otherwise.
///
/// Deliberately not used by the strip variant. `If-Modified-Since` is an answer about the FILE, and
/// a derived representation shares its file with the original — see
/// [`stripped_project_file_response`], which revalidates on the tag alone.
pub(super) fn project_file_is_not_modified(
    request_headers: &HeaderMap,
    etag: &str,
    last_modified: Option<SystemTime>,
) -> bool {
    if request_headers.contains_key(header::IF_NONE_MATCH) {
        return if_none_match_matches(request_headers, etag);
    }
    let Some(modified) = last_modified else {
        return false;
    };
    request_headers
        .get(header::IF_MODIFIED_SINCE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| httpdate::parse_http_date(value).ok())
        .is_some_and(|since| {
            modified
                .duration_since(since)
                .map_or(true, |delta| delta < Duration::from_secs(1))
        })
}

/// Parse a single HTTP byte range (`bytes=start-end`, `bytes=start-`, or
/// `bytes=-suffix`) against a known total size, returning an inclusive
/// `(start, end)` clamped to the file. Returns `None` for unsatisfiable or
/// multi-range requests (callers answer 416).
pub(super) fn parse_single_byte_range(value: &str, total: u64) -> Option<(u64, u64)> {
    let spec = value.strip_prefix("bytes=")?.trim();
    if spec.is_empty() || spec.contains(',') || total == 0 {
        return None;
    }
    let (start_str, end_str) = spec.split_once('-')?;
    let (start, end) = if start_str.is_empty() {
        // Suffix range: last `suffix` bytes.
        let suffix: u64 = end_str.parse().ok()?;
        if suffix == 0 {
            return None;
        }
        let start = total.saturating_sub(suffix);
        (start, total - 1)
    } else {
        let start: u64 = start_str.parse().ok()?;
        let end = if end_str.is_empty() {
            total - 1
        } else {
            end_str.parse::<u64>().ok()?.min(total - 1)
        };
        (start, end)
    };
    if start > end || start >= total {
        return None;
    }
    Some((start, end))
}

/// Router fallback. With `embed-web`, non-API paths are served from the embedded
/// web bundle (SPA fallback); API paths and all default-feature builds keep the
/// existing JSON not-found behavior.
pub(super) async fn app_fallback(request: Request<axum::body::Body>) -> Response {
    #[cfg(feature = "embed-web")]
    {
        if !request.uri().path().starts_with("/api/") {
            return web_assets::serve(request.uri(), request.headers()).await;
        }
    }
    route_not_found(request).await
}

pub(super) async fn route_not_found(request: Request<axum::body::Body>) -> Response {
    let path = request.uri().path();
    let lower_path = path.to_ascii_lowercase();
    if path.contains("/files/")
        && (path.contains("..")
            || lower_path.contains("%2e")
            || lower_path.contains("%2f")
            || lower_path.contains("%5c"))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "detail": "Invalid project file path" })),
        )
            .into_response();
    }
    if path.contains("/person-tracks/")
        && (path.contains("..")
            || lower_path.contains("%2e")
            || lower_path.contains("%2f")
            || lower_path.contains("%5c"))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "detail": "Invalid person track ID" })),
        )
            .into_response();
    }
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "detail": "Not Found" })),
    )
        .into_response()
}
