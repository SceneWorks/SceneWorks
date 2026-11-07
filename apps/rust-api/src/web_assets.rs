use std::collections::HashSet;
use std::sync::OnceLock;

use axum::body::Body;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "../web/dist"]
struct WebAssets;

#[cfg(test)]
pub(super) fn first_static_asset_path() -> String {
    WebAssets::iter()
        .find(|path| {
            path.starts_with("assets/") && !path.ends_with(".br") && !path.ends_with(".gz")
        })
        .expect("production web bundle contains a static asset")
        .into_owned()
}

#[cfg(test)]
pub(super) fn first_compressible_static_asset_path() -> String {
    WebAssets::iter()
        .find(|path| {
            path.starts_with("assets/")
                && !path.ends_with(".br")
                && !path.ends_with(".gz")
                && WebAssets::get(&format!("{path}.br")).is_some()
                && WebAssets::get(&format!("{path}.gz")).is_some()
        })
        .expect("production web bundle contains a precompressed static asset")
        .into_owned()
}

#[cfg(test)]
pub(super) fn first_uncompressed_static_asset_path() -> String {
    WebAssets::iter()
        .find(|path| {
            !path.ends_with(".br")
                && !path.ends_with(".gz")
                && path.as_ref() != IMMUTABLE_ASSET_MANIFEST
                && WebAssets::get(&format!("{path}.br")).is_none()
                && WebAssets::get(&format!("{path}.gz")).is_none()
        })
        .expect("production web bundle contains an uncompressed static asset")
        .into_owned()
}

// The desktop shell navigates its privileged webview to this server, so the embedded
// UI runs from this origin and its CSP must come from here (tauri.conf.json only
// governs the bundled setup screen). Kept narrow: scripts only from this origin (the
// theme bootstrap was moved to /theme-init.js so no inline script is needed), fonts
// self-hosted from this origin (no third-party font host — sc-8956), images/media as
// self/data/blob, IPC for the Tauri webview. Same-origin API + SSE are covered by
// connect-src 'self'.
pub(super) const CONTENT_SECURITY_POLICY: &str =
    include_str!("../../web/content-security-policy.txt");

const IMMUTABLE_CACHE: &str = "public, max-age=31536000, immutable";
const REVALIDATE_CACHE: &str = "no-cache";
const IMMUTABLE_ASSET_MANIFEST: &str = ".sceneworks-immutable-assets";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Encoding {
    Brotli,
    Gzip,
    Identity,
}

impl Encoding {
    fn suffix(self) -> &'static str {
        match self {
            Self::Brotli => ".br",
            Self::Gzip => ".gz",
            Self::Identity => "",
        }
    }

    fn header_value(self) -> Option<&'static str> {
        match self {
            Self::Brotli => Some("br"),
            Self::Gzip => Some("gzip"),
            Self::Identity => None,
        }
    }
}

#[derive(Default)]
struct EncodingPreferences {
    header_present: bool,
    brotli: Option<f32>,
    gzip: Option<f32>,
    identity: Option<f32>,
    wildcard: Option<f32>,
}

impl EncodingPreferences {
    fn parse(headers: &HeaderMap) -> Self {
        let mut preferences = Self::default();
        for value in headers.get_all(header::ACCEPT_ENCODING) {
            preferences.header_present = true;
            let Ok(value) = value.to_str() else {
                continue;
            };
            for item in value.split(',') {
                let mut parts = item.trim().split(';');
                let name = parts.next().unwrap_or_default().trim();
                if name.is_empty() {
                    continue;
                }
                let mut quality = 1.0;
                for parameter in parts {
                    let Some((key, value)) = parameter.trim().split_once('=') else {
                        continue;
                    };
                    if key.trim().eq_ignore_ascii_case("q") {
                        quality = value
                            .trim()
                            .parse::<f32>()
                            .ok()
                            .filter(|quality| (0.0..=1.0).contains(quality))
                            .unwrap_or(0.0);
                    }
                }
                if name.eq_ignore_ascii_case("br") {
                    preferences.brotli = Some(quality);
                } else if name.eq_ignore_ascii_case("gzip") {
                    preferences.gzip = Some(quality);
                } else if name.eq_ignore_ascii_case("identity") {
                    preferences.identity = Some(quality);
                } else if name == "*" {
                    preferences.wildcard = Some(quality);
                }
            }
        }
        preferences
    }

    fn quality(&self, encoding: Encoding) -> f32 {
        if !self.header_present {
            return if encoding == Encoding::Identity {
                1.0
            } else {
                0.0
            };
        }
        match encoding {
            Encoding::Brotli => self.brotli.or(self.wildcard).unwrap_or(0.0),
            Encoding::Gzip => self.gzip.or(self.wildcard).unwrap_or(0.0),
            // RFC 9110: identity is acceptable by default unless it is
            // explicitly excluded, or a wildcard q=0 excludes every
            // unlisted coding.
            Encoding::Identity => match self.identity {
                Some(quality) => quality,
                None if self.wildcard == Some(0.0) => 0.0,
                None => 1.0,
            },
        }
    }
}

fn select_encoding(headers: &HeaderMap, requested: &str) -> Option<Encoding> {
    let preferences = EncodingPreferences::parse(headers);
    // Deterministic tie-break: Brotli, then gzip, then identity. A client
    // advertising a coding at q=1 therefore receives compression rather
    // than an equally acceptable identity response.
    let candidates = [
        (
            Encoding::Brotli,
            WebAssets::get(&format!("{requested}.br")).is_some(),
        ),
        (
            Encoding::Gzip,
            WebAssets::get(&format!("{requested}.gz")).is_some(),
        ),
        (Encoding::Identity, WebAssets::get(requested).is_some()),
    ];
    let mut selected = None;
    for (encoding, available) in candidates {
        let quality = preferences.quality(encoding);
        let is_better = match selected {
            Some((_, best_quality)) => quality > best_quality,
            None => true,
        };
        if available && quality > 0.0 && is_better {
            selected = Some((encoding, quality));
        }
    }
    selected.map(|(encoding, _)| encoding)
}

fn immutable_assets() -> &'static HashSet<String> {
    static ASSETS: OnceLock<HashSet<String>> = OnceLock::new();
    ASSETS.get_or_init(|| {
        WebAssets::get(IMMUTABLE_ASSET_MANIFEST)
            .map(|manifest| {
                String::from_utf8_lossy(&manifest.data)
                    .lines()
                    .map(str::trim)
                    .filter(|path| !path.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    })
}

fn is_immutable_asset(path: &str) -> bool {
    immutable_assets().contains(path)
}

#[cfg(test)]
pub(super) fn immutable_asset_manifest_path() -> &'static str {
    IMMUTABLE_ASSET_MANIFEST
}

fn apply_common_headers(
    response: &mut axum::http::response::Builder,
    mime: &str,
    cache_control: &'static str,
) {
    let headers = response
        .headers_mut()
        .expect("embedded response headers are available");
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(mime).expect("embedded MIME type is a valid header"),
    );
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CONTENT_SECURITY_POLICY),
    );
    headers.insert(header::VARY, HeaderValue::from_static("Accept-Encoding"));
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(cache_control),
    );
}

fn not_acceptable_response(mime: &str, cache_control: &'static str) -> Response {
    let mut response = Response::builder().status(StatusCode::NOT_ACCEPTABLE);
    apply_common_headers(&mut response, mime, cache_control);
    response
        .body(Body::empty())
        .expect("not-acceptable embedded response constructs")
}

fn etag(hash: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut value = String::with_capacity(66);
    value.push('"');
    for byte in hash {
        value.push(HEX[(byte >> 4) as usize] as char);
        value.push(HEX[(byte & 0x0f) as usize] as char);
    }
    value.push('"');
    value
}

fn if_none_match(headers: &HeaderMap, current: &str) -> bool {
    headers
        .get_all(header::IF_NONE_MATCH)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .any(|candidate| {
            candidate == "*"
                || candidate == current
                || candidate
                    .strip_prefix("W/")
                    .is_some_and(|candidate| candidate == current)
        })
}

fn embedded_response(
    requested: &str,
    mime: &str,
    fallback: bool,
    request_headers: &HeaderMap,
) -> Option<Response> {
    let cache_control = if !fallback && is_immutable_asset(requested) {
        IMMUTABLE_CACHE
    } else {
        REVALIDATE_CACHE
    };
    let Some(encoding) = select_encoding(request_headers, requested) else {
        return Some(not_acceptable_response(mime, cache_control));
    };
    let representation_path = format!("{requested}{}", encoding.suffix());
    let file = WebAssets::get(&representation_path)?;
    let etag = etag(file.metadata.sha256_hash());
    let status = if if_none_match(request_headers, &etag) {
        StatusCode::NOT_MODIFIED
    } else {
        StatusCode::OK
    };
    let mut response = Response::builder().status(status);
    apply_common_headers(&mut response, mime, cache_control);
    let headers = response
        .headers_mut()
        .expect("embedded response headers are available");
    headers.insert(
        header::ETAG,
        HeaderValue::from_str(&etag).expect("SHA-256 ETag is a valid header"),
    );
    if let Some(content_encoding) = encoding.header_value() {
        headers.insert(
            header::CONTENT_ENCODING,
            HeaderValue::from_static(content_encoding),
        );
    }
    let body = if status == StatusCode::NOT_MODIFIED {
        Body::empty()
    } else {
        Body::from(file.data.into_owned())
    };
    Some(
        response
            .body(body)
            .expect("embedded response body constructs"),
    )
}

pub(super) async fn serve(uri: &Uri, request_headers: &HeaderMap) -> Response {
    let requested = uri.path().trim_start_matches('/');
    let requested = if requested.is_empty() {
        "index.html"
    } else {
        requested
    };
    // Encoded siblings and the cache allowlist are internal build
    // metadata, never public paths.
    if !requested.ends_with(".br")
        && !requested.ends_with(".gz")
        && requested != IMMUTABLE_ASSET_MANIFEST
        && WebAssets::get(requested).is_some()
    {
        let mime = mime_guess::from_path(requested).first_or_octet_stream();
        if let Some(response) = embedded_response(requested, mime.as_ref(), false, request_headers)
        {
            return response;
        }
    }
    // Single-page app: unknown non-API paths resolve to index.html so
    // client-side deep links (e.g. project routes) load correctly.
    embedded_response(
        "index.html",
        "text/html; charset=utf-8",
        true,
        request_headers,
    )
    .unwrap_or_else(|| StatusCode::NOT_FOUND.into_response())
}
