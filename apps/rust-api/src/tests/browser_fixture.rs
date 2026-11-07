//! Explicitly manual browser fixture for a real, isolated Rust API router.
//!
//! This is intentionally ignored: it is a long-lived development HTTP server rather than an
//! assertion. The owner launches the compiled libtest binary directly and stops it through the
//! owning preview process after browser verification. It never starts the production `main`, a
//! worker, in-process utility loop, model load, or GPU work.

use super::support::{create_app, isolate_hf_cache, test_settings};

/// Fixed, non-user token for the manual browser fixture. It exists only inside the ignored test
/// binary and exercises the ordinary access gate; it is not a production credential.
const BROWSER_FIXTURE_TOKEN: &str = "sceneworks-moss-s3-browser-fixture";
const BROWSER_FIXTURE_ORIGIN: &str = "http://127.0.0.1:5173";
const BROWSER_FIXTURE_ADDR: &str = "127.0.0.1:18000";

/// Serve a real API router backed by one owned temporary root for manual Browser-pane checks.
///
/// The guard is held for the whole server lifetime so model/cache resolution stays below the temp
/// root. The fixture starts empty on purpose: browser checks may create ordinary projects and read
/// the real target registry, but must not submit training or generation jobs.
#[tokio::test]
#[ignore = "manual owned Rust API fixture for Browser-pane verification"]
async fn serve_owned_manual_browser_fixture() {
    let _hf_cache = isolate_hf_cache();
    let root = tempfile::Builder::new()
        .prefix("sceneworks-moss-s3-browser-")
        .tempdir()
        .expect("fixture temp root");
    let mut settings = test_settings(&root);
    settings.port = 18_000;
    settings.access_token = BROWSER_FIXTURE_TOKEN.to_owned();
    settings.cors_origins = vec![BROWSER_FIXTURE_ORIGIN.to_owned()];
    let app = create_app(settings).expect("fixture app");
    let listener = tokio::net::TcpListener::bind(BROWSER_FIXTURE_ADDR)
        .await
        .expect("fixture listener");

    eprintln!(
        "manual_browser_fixture listening={} origin={} temp_root={} token={}",
        BROWSER_FIXTURE_ADDR,
        BROWSER_FIXTURE_ORIGIN,
        root.path().display(),
        BROWSER_FIXTURE_TOKEN,
    );
    axum::serve(listener, app)
        .await
        .expect("manual browser fixture server");
}
