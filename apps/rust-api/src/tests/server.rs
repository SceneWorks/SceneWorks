//! rust-api server tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

#[test]
fn default_api_host_is_loopback() {
    // sc-4201 (F-API-1): an out-of-the-box bind must not expose the API to the LAN.
    let ip: std::net::IpAddr = DEFAULT_API_HOST.parse().expect("default host parses");
    assert!(ip.is_loopback(), "default API host must be loopback");
}

/// F-003 / sc-11159: the enqueue gate must reject a path-unsafe `model` id (traversal /
/// separators / absolute), since it flows verbatim into the worker's asset filename. A
/// plain single-component id — including an uncatalogued one the stub lane serves — is
/// accepted (path-safety only, not catalog membership).
#[test]
fn validate_model_id_rejects_traversal_and_separators() {
    // Path-unsafe ids are rejected.
    for evil in [
        "../../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "/etc/cron.d/pwn",
        "a/b",
        "a\\b",
        "..",
        ".",
        "",
        "   ",
    ] {
        assert!(
            validate_model_id(evil).is_err(),
            "expected {evil:?} to be rejected as an unsafe model id"
        );
    }
    // Legitimate single-component ids (incl. uncatalogued stub-lane ids) are accepted.
    for ok in [
        "z_image_turbo",
        "ltx_2_3",
        "ideogram_4",
        "vid-model",
        "external_base_my-comfy-model",
        "unknown_stub_model",
    ] {
        assert!(
            validate_model_id(ok).is_ok(),
            "expected {ok:?} to be accepted as a safe model id"
        );
    }
}

#[test]
fn seed_mode_refreshes_the_app_owned_default_but_not_an_explicit_config_dir() {
    use sceneworks_core::builtin_manifests::SeedMode;
    // sc-10212: an explicit, non-empty SCENEWORKS_CONFIG_DIR marks an operator-owned dir (repo
    // checkout / Compose bind mount) → stay authoritative (IfMissing), never dirty a checkout.
    assert_eq!(
        seed_mode_for_config_dir(Some("/srv/sceneworks/config")),
        SeedMode::IfMissing
    );
    assert_eq!(
        seed_mode_for_config_dir(Some("./config")),
        SeedMode::IfMissing
    );
    // Unset, or blank/whitespace (env_path_or treats those as unset) → the platform-default
    // app-owned dir → Overwrite so a directly-launched API refreshes its builtin catalog on launch.
    assert_eq!(seed_mode_for_config_dir(None), SeedMode::Overwrite);
    assert_eq!(seed_mode_for_config_dir(Some("")), SeedMode::Overwrite);
    assert_eq!(seed_mode_for_config_dir(Some("   ")), SeedMode::Overwrite);
}

#[tokio::test]
async fn malformed_manifest_returns_stable_server_error() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "models": [ /*"#,
    )
    .expect("manifest writes");
    std::fs::write(config_dir.join("user.models.jsonc"), r#"{ "models": [] }"#)
        .expect("manifest writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, error) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.starts_with("Failed to parse manifest")));
}

#[test]
fn inprocess_worker_defaults_to_cpu_utility() {
    // Default (and blank override) → cpu so utility capabilities are
    // advertised regardless of the ambient SCENEWORKS_GPU_ID.
    assert_eq!(inprocess_worker_gpu_id(None), "cpu");
    assert_eq!(inprocess_worker_gpu_id(Some("   ".to_owned())), "cpu");
    // Explicit override is honored.
    assert_eq!(inprocess_worker_gpu_id(Some("auto".to_owned())), "auto");
    assert_eq!(inprocess_worker_gpu_id(Some("0".to_owned())), "0");
}

#[test]
fn inprocess_utility_worker_count_defaults_to_two() {
    // Unset / blank / unparseable → 2 so desktop downloads run two-at-a-time
    // (sc-10744) instead of serializing behind a single worker.
    assert_eq!(parse_inprocess_utility_worker_count(None), 2);
    assert_eq!(parse_inprocess_utility_worker_count(Some(String::new())), 2);
    assert_eq!(
        parse_inprocess_utility_worker_count(Some("   ".to_owned())),
        2
    );
    assert_eq!(
        parse_inprocess_utility_worker_count(Some("two".to_owned())),
        2
    );
    // A set, parseable value wins; surrounding whitespace is tolerated.
    assert_eq!(
        parse_inprocess_utility_worker_count(Some("1".to_owned())),
        1
    );
    assert_eq!(
        parse_inprocess_utility_worker_count(Some("4".to_owned())),
        4
    );
    assert_eq!(
        parse_inprocess_utility_worker_count(Some(" 3 ".to_owned())),
        3
    );
    // Never a zero-worker pool: 0 clamps up to 1.
    assert_eq!(
        parse_inprocess_utility_worker_count(Some("0".to_owned())),
        1
    );
}

#[test]
fn inprocess_utility_worker_ids_are_distinct_and_stable() {
    // Index 0 keeps the configured id unchanged (a single-worker setup registers
    // exactly as before); each extra worker gets a unique `-N` suffix so two loops
    // never collide on registration.
    assert_eq!(
        inprocess_utility_worker_id("rust-utility-worker", 0),
        "rust-utility-worker"
    );
    assert_eq!(
        inprocess_utility_worker_id("rust-utility-worker", 1),
        "rust-utility-worker-1"
    );
    assert_eq!(
        inprocess_utility_worker_id("rust-utility-worker", 2),
        "rust-utility-worker-2"
    );
    // Distinctness across a small pool.
    let ids: std::collections::HashSet<_> = (0..4)
        .map(|index| inprocess_utility_worker_id("host-a", index))
        .collect();
    assert_eq!(ids.len(), 4);
}

/// The watchdog must stay pending while the parent lives and resolve once it
/// exits — the desktop-sidecar orphan fix. Mirrors the Python worker's
/// parent-death test: spawn a dummy parent, confirm it isn't flagged alive
/// falsely, kill it, and assert the future then resolves promptly.
#[cfg(unix)]
#[tokio::test]
async fn parent_death_resolves_when_watched_parent_exits() {
    let mut parent = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .expect("spawn dummy parent");
    let pid = parent.id() as i32;

    assert!(
        crate::pid_alive(pid),
        "freshly spawned parent reads as dead"
    );
    // Still alive -> the watchdog must not resolve within a poll cycle.
    assert!(
        tokio::time::timeout(Duration::from_millis(200), crate::parent_death(Some(pid)),)
            .await
            .is_err(),
        "watchdog resolved while the parent was still alive"
    );

    parent.kill().expect("kill dummy parent");
    parent.wait().expect("reap dummy parent");
    assert!(!crate::pid_alive(pid), "reaped parent still reads as alive");

    // Now gone -> the watchdog resolves on its next check.
    tokio::time::timeout(
        crate::PARENT_POLL_INTERVAL + Duration::from_secs(2),
        crate::parent_death(Some(pid)),
    )
    .await
    .expect("watchdog did not resolve after the parent exited");
}

/// The Windows analogue: the parent-death watchdog is no longer unix-only (the
/// candle `auto` supervisor's per-GPU children orphaned on every quit because
/// `run_worker`/`shutdown_signal` never raced it off-unix). Spawn a dummy parent,
/// confirm `pid_alive` (a `tasklist` probe) tracks it, kill it, and assert the
/// watchdog then resolves.
#[cfg(windows)]
#[tokio::test]
async fn parent_death_resolves_when_watched_parent_exits_windows() {
    // `ping -n 30` lives ~30s and spawns no children to orphan; its stdout is
    // discarded so it doesn't bleed into the test output.
    let mut parent = std::process::Command::new("ping")
        .args(["-n", "30", "127.0.0.1"])
        .stdout(std::process::Stdio::null())
        .spawn()
        .expect("spawn dummy parent");
    let pid = parent.id() as i32;

    assert!(
        crate::pid_alive(pid),
        "freshly spawned parent reads as dead"
    );
    // Still alive -> the watchdog must not resolve within a poll cycle.
    assert!(
        tokio::time::timeout(Duration::from_millis(500), crate::parent_death(Some(pid)))
            .await
            .is_err(),
        "watchdog resolved while the parent was still alive"
    );

    parent.kill().expect("kill dummy parent");
    parent.wait().expect("reap dummy parent");
    assert!(!crate::pid_alive(pid), "reaped parent still reads as alive");

    // Now gone -> the watchdog resolves on its next check.
    tokio::time::timeout(
        crate::PARENT_POLL_INTERVAL + Duration::from_secs(5),
        crate::parent_death(Some(pid)),
    )
    .await
    .expect("watchdog did not resolve after the parent exited");
}

/// No configured parent (server/Docker) -> the watchdog future never resolves.
/// Cross-platform: the watchdog is now wired on Windows too.
#[tokio::test]
async fn parent_death_never_fires_without_a_parent_pid() {
    assert!(
        tokio::time::timeout(Duration::from_millis(200), crate::parent_death(None))
            .await
            .is_err(),
        "watchdog fired with no parent PID configured"
    );
}

/// PIDs of 0 or 1 (and unset/garbage) yield no parent to watch. Cross-platform:
/// the parser backs the watchdog on every OS now.
#[test]
fn parent_pid_to_watch_rejects_init_and_invalid_values() {
    use std::env;
    // Serialize: the helper reads a process-global env var.
    for (value, expected) in [
        (Some("0"), None),
        (Some("1"), None),
        (Some("-5"), None),
        (Some(" not-a-pid "), None),
        (Some(""), None),
        (Some(" 4242 "), Some(4242_i32)),
        (None, None),
    ] {
        match value {
            Some(v) => env::set_var("SCENEWORKS_PARENT_PID", v),
            None => env::remove_var("SCENEWORKS_PARENT_PID"),
        }
        assert_eq!(crate::parent_pid_to_watch(), expected, "value={value:?}");
    }
    env::remove_var("SCENEWORKS_PARENT_PID");
}

// sc-8812 (F-010): the router-wide default body limit is the small JSON cap
// (`MAX_JSON_BODY_BYTES`, 10 MiB). A JSON route must reject a body over that cap with
// 413 *before* buffering/parsing it, so a runaway/malicious caller can't drive a
// multi-GiB memory spike per request.
#[tokio::test]
async fn oversized_json_body_is_rejected_before_parsing() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // 11 MiB of bytes — over the 10 MiB JSON cap. Content need not be valid JSON:
    // the `DefaultBodyLimit` layer trips first, so we never reach the parser.
    let oversized = vec![b'a'; 11 * 1024 * 1024];
    let (status, _headers, _body) = request_raw(
        app,
        "POST",
        "/api/v1/jobs",
        oversized,
        &[("content-type", "application/json")],
    )
    .await;

    assert_eq!(
        status,
        StatusCode::PAYLOAD_TOO_LARGE,
        "JSON route must 413 a body over the small router-wide cap"
    );
}

// sc-8812 (F-010): a JSON body *under* the small cap is still handled normally (proves
// the tightened cap didn't shrink legitimate JSON traffic below a usable size).
#[tokio::test]
async fn under_cap_json_body_is_accepted() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // Well under 10 MiB, but not empty — routed to the handler (400/validation, not 413).
    let (status, _) = request(
        app,
        "POST",
        "/api/v1/jobs",
        json!({ "type": "definitely_not_a_real_job_type" }),
    )
    .await;

    assert_ne!(
        status,
        StatusCode::PAYLOAD_TOO_LARGE,
        "a small JSON body must reach the handler, not be size-rejected"
    );
}

// sc-8812 (F-010): multipart/upload routes re-attach the large per-route limit, so a
// body far larger than the small JSON cap is still accepted. This guards against the
// tightened router-wide default accidentally shrinking an upload route's ceiling.
#[tokio::test]
async fn upload_route_accepts_body_larger_than_json_cap() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Large Upload Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // 12 MiB PNG payload — over the 10 MiB JSON cap, under the 2 GiB upload cap.
    let large_png = vec![0u8; 12 * 1024 * 1024];
    let (status, upload) = request_multipart_upload(
        app,
        &format!("/api/v1/projects/{project_id}/training/uploads"),
        "large.png",
        "image/png",
        &large_png,
    )
    .await;

    assert_ne!(
        status,
        StatusCode::PAYLOAD_TOO_LARGE,
        "upload route must accept a body larger than the JSON cap"
    );
    assert_eq!(
        status,
        StatusCode::CREATED,
        "large upload should be staged successfully (got {status}: {upload})"
    );
}
