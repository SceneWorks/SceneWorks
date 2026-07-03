use super::*;
use axum::extract::ConnectInfo;
use std::net::SocketAddr;

pub(crate) async fn access_control(
    State(state): State<AppState>,
    // `Option<…>` so unit tests that drive the router via `oneshot` (no connect info)
    // still resolve the extractor — absent peer ⇒ not loopback-trusted, falls through
    // to the token check.
    connect_info: Option<ConnectInfo<SocketAddr>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let peer = connect_info.map(|ConnectInfo(addr)| addr);
    if request.method() == Method::OPTIONS
        || !requires_token(request.method(), request.uri().path())
        || loopback_trusted(state.settings.trust_loopback, peer)
        || is_authorized(request.headers(), &state.settings)
        || media_ticket_authorized(&state, &request)
    {
        return next.run(request).await;
    }

    // Make auth rejections visible to operators (they previously returned 401 with no
    // server-side trace). Log the path + reason + status only — never the token/secret
    // (and `uri().path()` excludes any query string).
    tracing::warn!(
        event = "auth_rejected",
        path = %request.uri().path(),
        reason = "missing_or_invalid_token",
        status = StatusCode::UNAUTHORIZED.as_u16(),
        "rejected unauthenticated API request"
    );

    (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "detail": "SceneWorks access token required",
            "authRequired": true
        })),
    )
        .into_response()
}

pub(crate) fn cors_layer(settings: &Settings) -> CorsLayer {
    let origins = settings
        .cors_origins
        .iter()
        .filter_map(|origin| HeaderValue::from_str(origin).ok())
        .collect::<Vec<_>>();

    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            HeaderName::from_static("x-sceneworks-token"),
        ])
}

/// Whether a request is gated by the access token. Only `/api/*` routes are
/// protected (minus the explicitly public ones); everything else is the
/// embedded web bundle / SPA fallback, which a browser must be able to load
/// before it can attach the token header.
///
/// The exemption is method-aware, not path-only (sc-8869, F-067): a public path
/// is only exempt for the read-safe method(s) it was whitelisted for. The theme
/// preferences path (`/api/v1/ui-preferences`) shares its route between a
/// pre-auth GET (the theme read that loads before auth, to avoid a flash) and a
/// PUT that writes `ui-preferences.json` to disk. Only the GET stays public; the
/// disk-writing PUT must still present the token when one is configured, so an
/// unauthenticated LAN caller can't overwrite the file (epic 4484: every write is
/// authenticated). All other `PUBLIC_PATHS` entries expose a single route method,
/// so a path-only exemption for them is already method-correct.
pub(crate) fn requires_token(method: &Method, path: &str) -> bool {
    if !path.starts_with("/api/") {
        return false;
    }
    if path == UI_PREFERENCES_PATH {
        // Only the pre-auth theme READ is public; the disk-writing PUT is gated.
        return method != Method::GET;
    }
    !PUBLIC_PATHS.contains(&path)
}

/// Whether a request should bypass the access token because it originates from this
/// machine. When LAN remote access is on, the desktop launcher binds `0.0.0.0` and sets
/// the password as the API's access token — but the embedded desktop UI and the local
/// GPU worker(s) reach the API over loopback and have no password to send. Trusting
/// loopback peers keeps local use password-free while still gating LAN callers (other
/// source IPs).
///
/// Opt-in via `SCENEWORKS_TRUST_LOOPBACK` (the desktop sets it; Docker/server does NOT),
/// so a server deployment fronted by a reverse proxy — where every request would appear
/// to come from loopback — stays fail-closed. Pure so the decision is unit-tested without
/// a live listener; mirrors `should_warn_open_bind`.
pub(crate) fn loopback_trusted(trust_loopback: bool, peer: Option<SocketAddr>) -> bool {
    trust_loopback && peer.is_some_and(|addr| addr.ip().is_loopback())
}

/// Whether a request may bypass the header-token check because it carries a valid
/// media ticket (sc-8810). Browsers cannot attach headers to element-driven requests
/// (`<img src>`, `<video src>`, `<a download>`), so — mirroring the SSE ticket — an
/// authenticated client mints a short-lived ticket (POST /api/v1/files/ticket) and
/// appends it as a `?ticket=` query param. The bypass is scoped hard: GET only, and
/// only the read-only media routes (project files + pose previews); every other
/// route still requires the real token, and an SSE event ticket is never accepted
/// here (separate store).
fn media_ticket_authorized(state: &AppState, request: &Request<axum::body::Body>) -> bool {
    if request.method() != Method::GET {
        return false;
    }
    if !is_ticketed_media_path(request.uri().path()) {
        return false;
    }
    match ticket_from_query(request.uri().query().unwrap_or_default()) {
        Some(ticket) => state.media_tickets.validate(ticket),
        None => false,
    }
}

/// The exact route families a media ticket unlocks:
///   GET /api/v1/projects/:project_id/files/*relative_path
///   GET /api/v1/poses/preview/:job_id/:file_name
/// Matched on the raw request path (same shape the router matches); the handlers
/// keep their own traversal/validity checks, the ticket only answers "is this
/// caller allowed", identically to a header-token caller on these routes.
pub(crate) fn is_ticketed_media_path(path: &str) -> bool {
    if let Some(rest) = path.strip_prefix("/api/v1/projects/") {
        let mut segments = rest.split('/');
        let has_project = segments.next().is_some_and(|s| !s.is_empty());
        let files_literal = segments.next() == Some("files");
        let has_file = segments.next().is_some_and(|s| !s.is_empty());
        return has_project && files_literal && has_file;
    }
    if let Some(rest) = path.strip_prefix("/api/v1/poses/preview/") {
        return !rest.is_empty();
    }
    false
}

/// Extract the raw `ticket` query-param value. Tickets are hex UUIDs, so no
/// percent-decoding is needed; a decoded-away match simply fails validation.
fn ticket_from_query(query: &str) -> Option<&str> {
    query
        .split('&')
        .find_map(|pair| pair.strip_prefix("ticket="))
        .filter(|value| !value.is_empty())
}

pub(crate) fn is_authorized(headers: &HeaderMap, settings: &Settings) -> bool {
    if settings.access_token.is_empty() {
        return true;
    }
    constant_time_eq(
        token_from_headers(headers).as_bytes(),
        settings.access_token.as_bytes(),
    )
}

fn token_from_headers(headers: &HeaderMap) -> String {
    if let Some(token) = headers
        .get("x-sceneworks-token")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return token.to_owned();
    }
    headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .unwrap_or_default()
        .to_owned()
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .fold(0, |difference, (left, right)| difference | (left ^ right))
        == 0
}
