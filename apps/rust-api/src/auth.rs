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
        || !requires_token(request.uri().path())
        || loopback_trusted(state.settings.trust_loopback, peer)
        || is_authorized(request.headers(), &state.settings)
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

/// Whether a path is gated by the access token. Only `/api/*` routes are
/// protected (minus the explicitly public ones); everything else is the
/// embedded web bundle / SPA fallback, which a browser must be able to load
/// before it can attach the token header.
pub(crate) fn requires_token(path: &str) -> bool {
    path.starts_with("/api/") && !PUBLIC_PATHS.contains(&path)
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
