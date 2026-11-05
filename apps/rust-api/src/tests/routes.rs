//! Route/method/authentication baseline captured before area router extraction.
//! Only unauthenticated requests and method discovery run here; no job is submitted.
use super::support::*;
use axum::extract::MatchedPath;
use std::collections::{BTreeMap, BTreeSet};

const ROUTES: &str = include_str!("route_contract.tsv");

fn route_contract() -> Vec<(&'static str, &'static str, &'static str)> {
    ROUTES
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| {
            let parts: Vec<_> = line.split('\t').collect();
            assert_eq!(parts.len(), 3);
            (parts[0], parts[1], parts[2])
        })
        .collect()
}

fn sample_path(pattern: &str) -> String {
    pattern
        .split('/')
        .map(|segment| {
            if segment.starts_with(':') {
                "route_fixture_id"
            } else if segment.starts_with('*') {
                "route-fixture.bin"
            } else {
                segment
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

async fn report_matched_route(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let matched = request
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str().to_owned());
    let mut response = next.run(request).await;
    if let Some(matched) = matched {
        response
            .headers_mut()
            .insert("x-tested-route", matched.parse().unwrap());
    }
    response
}

#[tokio::test]
async fn every_declared_route_preserves_public_or_protected_authentication() {
    let temp = tempfile::tempdir().unwrap();
    let mut settings = test_settings(&temp);
    settings.access_token = "route-contract-secret".into();
    let app = create_app(settings)
        .unwrap()
        .layer(axum::middleware::from_fn(report_matched_route));
    for (method, path, authentication) in route_contract() {
        let (status, headers, _) = request_raw(
            app.clone(),
            method,
            &sample_path(path),
            Body::from(r#"{"token":""}"#),
            &[("content-type", "application/json")],
        )
        .await;
        assert_eq!(
            headers
                .get("x-tested-route")
                .and_then(|value| value.to_str().ok()),
            Some(path),
            "missing {method} {path}"
        );
        let expected = if authentication == "public" {
            StatusCode::OK
        } else {
            StatusCode::UNAUTHORIZED
        };
        assert_eq!(status, expected, "{method} {path}: {authentication}");
        assert_eq!(
            crate::auth::requires_token(&method.parse().unwrap(), &sample_path(path)),
            authentication == "token",
            "auth contract {method} {path}"
        );
    }
    let (status, _, _) = request_raw(app, "POST", "/mcp", Body::empty(), &[]).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn every_declared_route_preserves_methods_and_browser_boundary() {
    let temp = tempfile::tempdir().unwrap();
    let mut settings = test_settings(&temp);
    settings.access_token = "route-contract-secret".into();
    let app = create_app(settings).unwrap();
    let mut methods: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (method, path, _) in route_contract() {
        assert!(methods.entry(path).or_default().insert(method));
    }
    for (path, expected) in methods {
        let path = sample_path(path);
        // CORS owns OPTIONS before MethodRouter; an unregistered TRACE reaches only
        // method discovery and cannot invoke any mutation handler.
        let (status, headers, _) = request_raw(
            app.clone(),
            "TRACE",
            &path,
            Body::empty(),
            &[("x-sceneworks-token", "route-contract-secret")],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::METHOD_NOT_ALLOWED,
            "method discovery {path}"
        );
        let actual: BTreeSet<_> = headers
            .get("allow")
            .expect("method router supplies Allow")
            .to_str()
            .unwrap()
            .split(',')
            .map(str::trim)
            .collect();
        assert_eq!(actual, expected, "method set {path}");
        for browser_headers in [
            vec![
                ("host", "attacker.invalid"),
                ("x-sceneworks-token", "route-contract-secret"),
            ],
            vec![
                ("origin", "null"),
                ("x-sceneworks-token", "route-contract-secret"),
            ],
        ] {
            let (status, _, _) = request_raw(
                app.clone(),
                "OPTIONS",
                &path,
                Body::empty(),
                &browser_headers,
            )
            .await;
            assert_eq!(status, StatusCode::FORBIDDEN, "browser boundary {path}");
        }
    }
}
