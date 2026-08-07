use super::support::*;

/// The pre-readiness surface: a bootstrap router whose `ready_app` is never set,
/// i.e. the exact window between `api_listening` and `api_ready` (sc-14788).
fn bootstrap_app(settings: Settings) -> axum::Router {
    crate::server::bootstrap_router(
        settings,
        std::sync::Arc::new(std::sync::OnceLock::new()),
        crate::startup::StartupMaintenance::pending(),
    )
}

/// Like `request_with_headers`, but returns the raw body + content type instead of
/// parsing JSON — the holding page is HTML, so a JSON-parsing helper would panic.
async fn raw_request(
    app: axum::Router,
    uri: &str,
    headers: &[(&str, &str)],
) -> (StatusCode, String, String) {
    let mut builder = Request::builder().method("GET").uri(uri);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let request = builder.body(Body::empty()).expect("request builds");
    let response = app.oneshot(request).await.expect("response returns");
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body buffers");
    (
        status,
        content_type,
        String::from_utf8(bytes.to_vec()).expect("body is utf-8"),
    )
}

fn write_stale_asset_upload(data_dir: &std::path::Path) -> std::path::PathBuf {
    let upload_root = data_dir.join("cache/uploads");
    std::fs::create_dir_all(&upload_root).expect("upload root creates");
    let path = upload_root.join("upload-stale.tmp");
    std::fs::write(&path, b"stale").expect("stale upload writes");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("stale upload opens");
    file.set_times(
        std::fs::FileTimes::new()
            .set_modified(SystemTime::now() - Duration::from_secs(25 * 60 * 60)),
    )
    .expect("stale upload mtime writes");
    path
}

#[tokio::test]
async fn slow_background_upload_sweep_keeps_health_and_access_ready_then_cleans_up() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    // Public health diagnostics must not add a second path disclosure when auth
    // is configured; the pre-existing `directories` field is omitted in this mode.
    settings.access_token = "secret".to_owned();
    let stale = write_stale_asset_upload(&settings.data_dir);
    let hook = crate::startup::test_startup_maintenance_hook(settings.data_dir.clone(), false);
    let (app, state) =
        crate::create_app_with_deferred_startup_maintenance(settings, Duration::from_secs(2))
            .expect("deferred app creates");
    tokio::time::timeout(Duration::from_secs(1), hook.wait_until_reached())
        .await
        .expect("background sweep reaches deterministic block");

    let started = std::time::Instant::now();
    let (health_status, running) = request(app.clone(), "GET", "/api/v1/health", Value::Null).await;
    let (access_status, _) = request(app.clone(), "GET", "/api/v1/access", Value::Null).await;
    assert_eq!(health_status, StatusCode::OK);
    assert_eq!(access_status, StatusCode::OK);
    assert_eq!(running["startupMaintenance"]["status"], "running");
    assert_eq!(
        running["startupMaintenance"]["criticality"],
        "background-safe"
    );
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "background filesystem work must not hold health/access readiness"
    );
    eprintln!(
        "SC-14788 local readiness timing: blocked background upload sweep, /health + /access={:?}",
        started.elapsed()
    );
    assert!(stale.exists(), "the blocked sweep has not run yet");
    assert!(
        !state
            .startup_maintenance
            .start(temp_dir.path().join("must-not-run"), Duration::from_secs(2)),
        "startup maintenance is single-flight"
    );

    hook.release();
    let completed = tokio::time::timeout(
        Duration::from_secs(2),
        state.startup_maintenance.wait_for_terminal(),
    )
    .await
    .expect("background sweep completes");
    assert_eq!(completed.status, "complete");
    assert_eq!(completed.removed_uploads, 1);
    assert_eq!(completed.failure_count, 0);
    assert!(!stale.exists(), "eventual cleanup removes the stale upload");

    let (_, health) = request(app, "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(health["startupMaintenance"]["status"], "complete");
    assert_eq!(health["startupMaintenance"]["removedUploads"], 1);
    assert!(
        !serde_json::to_string(&health)
            .expect("health serializes")
            .contains(temp_dir.path().to_string_lossy().as_ref()),
        "background maintenance diagnostics do not leak filesystem paths"
    );
}

#[tokio::test]
async fn background_upload_sweep_failure_is_reported_without_blocking_readiness() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret".to_owned();
    let hook = crate::startup::test_startup_maintenance_hook(settings.data_dir.clone(), true);
    let (app, state) =
        crate::create_app_with_deferred_startup_maintenance(settings, Duration::from_secs(2))
            .expect("deferred app creates");
    tokio::time::timeout(Duration::from_secs(1), hook.wait_until_reached())
        .await
        .expect("background sweep reaches deterministic failure");

    let (status, running) = request(app.clone(), "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(running["startupMaintenance"]["status"], "running");
    hook.release();

    let failed = tokio::time::timeout(
        Duration::from_secs(2),
        state.startup_maintenance.wait_for_terminal(),
    )
    .await
    .expect("background failure is reported");
    assert_eq!(failed.status, "failed");
    assert_eq!(failed.failure_count, 1);
    let (_, health) = request(app, "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(health["startupMaintenance"]["status"], "failed");
    assert_eq!(health["startupMaintenance"]["failureCount"], 1);
    assert!(
        !serde_json::to_string(&health)
            .expect("health serializes")
            .contains(temp_dir.path().to_string_lossy().as_ref()),
        "failure diagnostics do not leak filesystem paths"
    );
}

#[tokio::test]
async fn background_upload_sweep_timeout_is_bounded_and_cancellation_aware() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let hook = crate::startup::test_startup_maintenance_hook(settings.data_dir.clone(), false);
    let (app, state) =
        crate::create_app_with_deferred_startup_maintenance(settings, Duration::from_millis(50))
            .expect("deferred app creates");
    tokio::time::timeout(Duration::from_secs(1), hook.wait_until_reached())
        .await
        .expect("background sweep reaches deterministic block");

    let timed_out = tokio::time::timeout(
        Duration::from_secs(1),
        state.startup_maintenance.wait_for_terminal(),
    )
    .await
    .expect("maintenance deadline is bounded");
    assert_eq!(timed_out.status, "timed-out");
    let (status, health) = request(app, "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["startupMaintenance"]["status"], "timed-out");
    hook.release();
}

#[tokio::test]
async fn deferred_upload_sweeps_preserve_orphan_first_list_and_asset_mutation() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let store = sceneworks_core::project_store::ProjectStore::new(
        &settings.data_dir,
        settings.app_version.clone(),
    );
    let project = store
        .create_project("Startup integrity")
        .expect("project creates");
    store
        .persist_generated_asset(
            &project.id,
            "old-job",
            "old-set",
            &json!({
                "assetId": "orphan",
                "mediaPath": "assets/images/old-set/orphan.png",
                "mimeType": "image/png",
                "type": "image",
                "displayName": "Orphan",
                "createdAt": "2026-07-25T00:00:00Z",
                "mode": "text_to_image",
                "model": "test",
                "adapter": "test",
                "prompt": "orphan"
            }),
        )
        .expect("orphan row persists");

    let hook = crate::startup::test_startup_maintenance_hook(settings.data_dir.clone(), false);
    let (app, state) =
        crate::create_app_with_deferred_startup_maintenance(settings, Duration::from_secs(2))
            .expect("deferred app creates");
    tokio::time::timeout(Duration::from_secs(1), hook.wait_until_reached())
        .await
        .expect("background upload sweep blocks");

    // Orphan pruning remains readiness-critical and uses the same per-project
    // lock as list/mutation. The first list is clean even while unrelated upload
    // staging cleanup is still running.
    let assets_uri = format!(
        "/api/v1/projects/{}/assets?includeRejected=true&includeTrashed=true",
        project.id
    );
    let (status, first) = request(app.clone(), "GET", &assets_uri, Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(first, json!([]));

    let project_path = std::path::PathBuf::from(&project.path);
    let media = project_path.join("assets/images/new-set/current.png");
    std::fs::create_dir_all(media.parent().expect("media parent")).expect("media dir creates");
    std::fs::write(&media, PNG_32X32).expect("media writes");
    state
        .project_store
        .persist_generated_asset(
            &project.id,
            "new-job",
            "new-set",
            &json!({
                "assetId": "current",
                "mediaPath": "assets/images/new-set/current.png",
                "mimeType": "image/png",
                "type": "image",
                "displayName": "Current",
                "createdAt": "2026-07-25T00:01:00Z",
                "mode": "text_to_image",
                "model": "test",
                "adapter": "test",
                "prompt": "current"
            }),
        )
        .expect("concurrent asset mutation persists");
    let (_, after_mutation) = request(app, "GET", &assets_uri, Value::Null).await;
    assert_eq!(after_mutation.as_array().expect("assets array").len(), 1);
    assert_eq!(after_mutation[0]["id"], "current");
    hook.release();
}

/// sc-15591 (GH #1937). A browser that navigates to the app root before readiness
/// finishes must get a page that becomes the app on its own. Before this, `/`
/// answered the `api_initializing` JSON to everyone — so the desktop window (and
/// any LAN/Compose tab) rendered that JSON as text and stayed there forever, since
/// nothing re-navigates a webview.
#[tokio::test]
async fn browser_navigation_before_readiness_gets_a_self_refreshing_page() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = bootstrap_app(test_settings(&temp_dir));

    for uri in ["/", "/projects/some-deep-link"] {
        let (status, content_type, body) = raw_request(
            app.clone(),
            uri,
            // What Chromium/WebView2 actually sends on a top-level navigation.
            &[(
                "accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )],
        )
        .await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{uri} stays a 503");
        assert!(
            content_type.starts_with("text/html"),
            "{uri} is rendered as a page, not as text: {content_type}"
        );
        assert!(
            body.contains(r#"http-equiv="refresh""#),
            "{uri} retries on its own, so the tab becomes the app once startup ends"
        );
        assert!(
            !body.contains("api_initializing"),
            "{uri} never shows the raw error payload to a human"
        );
    }
}

/// The other half of the negotiation: an API client is unchanged. A `*/*`-only
/// caller (curl, the GPU worker, a health probe) must keep the machine-readable
/// 503 + `retry-after` rather than being handed a page it cannot parse.
#[tokio::test]
async fn api_clients_before_readiness_keep_the_machine_readable_error() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = bootstrap_app(test_settings(&temp_dir));

    for accept in [None, Some("*/*"), Some("application/json")] {
        // `None` sends no Accept header at all.
        let headers: Vec<(&str, &str)> =
            accept.into_iter().map(|value| ("accept", value)).collect();
        let (status, content_type, body) =
            raw_request(app.clone(), "/api/v1/projects", &headers).await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "accept={accept:?}");
        assert!(
            content_type.starts_with("application/json"),
            "accept={accept:?} keeps JSON: {content_type}"
        );
        let payload: Value = serde_json::from_str(&body).expect("json body parses");
        assert_eq!(payload["code"], "api_initializing", "accept={accept:?}");
    }
}

/// The bootstrap health body is what the desktop's window gate classifies, so it
/// has to be distinguishable from the ready router's. `status`/`readiness.status`
/// are the two markers `probe_health` reads; if either stopped moving, the gate
/// would navigate early again and land on the holding page instead of the app.
#[tokio::test]
async fn bootstrap_health_is_distinguishable_from_ready_health() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let bootstrap = bootstrap_app(settings.clone());

    let (status, starting) = request(bootstrap, "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "the trust probe still gets a 200");
    // The three fields the desktop uses to decide the responder is genuinely ours.
    assert_eq!(starting["service"], "sceneworks-api");
    assert_eq!(starting["runtime"], "rust");
    // …and the two that say it is not ready to be rendered yet.
    assert_eq!(starting["status"], "starting");
    assert_eq!(starting["readiness"]["status"], "initializing");

    let (ready_app, _state) = create_app_with_state(settings).expect("ready app creates");
    let (status, ready) = request(ready_app, "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ready["service"], "sceneworks-api");
    assert_eq!(ready["runtime"], "rust");
    assert_eq!(ready["status"], "ok");
    assert_eq!(ready["readiness"]["status"], "ready");
}
