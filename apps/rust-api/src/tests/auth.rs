//! rust-api auth tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

#[test]
fn warns_only_on_open_bind_without_token() {
    use std::net::IpAddr;
    let v4 = |s: &str| s.parse::<IpAddr>().unwrap();
    // No token + a wider bind (0.0.0.0 / a concrete LAN IP) → warn.
    assert!(should_warn_open_bind("", v4("0.0.0.0")));
    assert!(should_warn_open_bind("   ", v4("0.0.0.0")));
    assert!(should_warn_open_bind("", v4("192.168.1.5")));
    assert!(should_warn_open_bind("", "::".parse().unwrap()));
    // A token, or a loopback bind, is safe → no warning.
    assert!(!should_warn_open_bind("secret", v4("0.0.0.0")));
    assert!(!should_warn_open_bind("", v4("127.0.0.1")));
    assert!(!should_warn_open_bind("", "::1".parse().unwrap()));
}

#[test]
fn loopback_trusted_only_when_enabled_and_peer_is_local() {
    use std::net::SocketAddr;
    let addr = |s: &str| s.parse::<SocketAddr>().unwrap();
    // Epic 4484: only a loopback peer is trusted, and only when the opt-in is set — so the
    // local desktop UI/worker bypass the password while LAN clients (other IPs) don't.
    assert!(loopback_trusted(true, Some(addr("127.0.0.1:51234"))));
    assert!(loopback_trusted(true, Some(addr("[::1]:51234"))));
    // Enabled but the peer is on the LAN → still gated.
    assert!(!loopback_trusted(true, Some(addr("192.168.1.5:51234"))));
    assert!(!loopback_trusted(true, Some(addr("0.0.0.0:51234"))));
    // A loopback peer alone never trusts without the opt-in (server/Docker default).
    assert!(!loopback_trusted(false, Some(addr("127.0.0.1:51234"))));
    // Unknown peer (e.g. the unit-test oneshot path with no connect info) → not trusted.
    assert!(!loopback_trusted(true, None));
}

#[test]
fn open_bind_override_only_for_explicit_optin() {
    // sc-5720 (API-001): the override that lets an unauthenticated open bind start
    // must require an explicit affirmative value; anything else keeps the refusal.
    for value in ["1", "true", "TRUE", "yes", "YES", " 1 "] {
        assert!(open_bind_override_enabled(value), "{value:?} should opt in");
    }
    for value in ["", "0", "false", "no", "off", "2", "enable"] {
        assert!(
            !open_bind_override_enabled(value),
            "{value:?} must not opt in"
        );
    }
}

#[tokio::test]
async fn access_token_is_enforced_on_protected_routes() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    let (status, access) = request(app.clone(), "GET", "/api/v1/access", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(access["authRequired"], true);

    let (status, error) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(error["detail"], "SceneWorks access token required");

    let (status, jobs) = request_with_headers(
        app,
        "GET",
        "/api/v1/jobs",
        Value::Null,
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(jobs, json!([]));
}

#[tokio::test]
async fn public_health_withholds_host_paths_when_a_token_is_configured() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");

    // No token: single-user/local, directories stay for diagnostics.
    let open_app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, health) = request(open_app, "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["status"], "ok");
    assert_eq!(health["authRequired"], false);
    assert!(health.get("directories").is_some());

    // Token configured but /health is public: don't leak absolute host paths.
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let guarded_app = create_app(settings).expect("app creates");
    let (status, health) = request(guarded_app, "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["status"], "ok");
    assert_eq!(health["authRequired"], true);
    assert!(health.get("directories").is_none());
}

#[tokio::test]
async fn host_capabilities_requires_token_and_reports_host_memory() {
    // epic 4484 story 9: the remote-browser host-memory signal is auth-protected and
    // derives the platform-correct memory from the registered GPU worker's utilization.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    // Protected: no token → 401 (not a public path).
    let (status, _) = request(app.clone(), "GET", "/api/v1/host-capabilities", Value::Null).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // With the token but no workers registered: 200, platform present, memory omitted.
    let (status, caps) = request_with_headers(
        app.clone(),
        "GET",
        "/api/v1/host-capabilities",
        Value::Null,
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(caps.get("platform").is_some());
    assert!(caps.get("unifiedMemoryGb").is_none());

    // Register an MLX worker reporting unified memory (sysctl hw.memsize as total MB);
    // the endpoint surfaces it as GB.
    let (status, _) = request_with_headers(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "mlx-test",
            "gpuId": "mlx",
            "capabilities": [],
            "loadedModels": [],
            "utilization": { "memoryTotalMb": 131072 }
        }),
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, caps) = request_with_headers(
        app,
        "GET",
        "/api/v1/host-capabilities",
        Value::Null,
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(caps["unifiedMemoryGb"], 128.0);
    // No host paths or secrets leak through this endpoint.
    assert!(caps.get("directories").is_none());
}

#[tokio::test]
async fn worker_restart_requires_token() {
    // epic 4484 story 12: the remote-admin worker-restart endpoint is auth-protected.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    let (status, _) = request(app.clone(), "POST", "/api/v1/worker/restart", Value::Null).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let (status, body) = request_with_headers(
        app,
        "POST",
        "/api/v1/worker/restart",
        Value::Null,
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["ok"], true);
}

#[tokio::test]
async fn ui_preferences_get_is_public_but_put_requires_token() {
    // sc-8869 (F-067): with a token configured, the pre-auth theme READ (GET) stays
    // public so the UI can load the theme before auth, but the PUT writes
    // `ui-preferences.json` to disk and must present the token — an unauthenticated
    // LAN caller can no longer overwrite the file (epic 4484: every write authenticated).
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    // GET without a token: still public (theme read loads before auth).
    let (status, prefs) = request(app.clone(), "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert!(prefs.get("theme").is_none());

    // PUT without a token: rejected (disk write is now authenticated).
    let (status, _) = request(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "theme": "dark", "accent": "coral" }),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // The rejected write must not have touched the stored preferences.
    let (status, prefs) = request(app.clone(), "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert!(prefs.get("theme").is_none());

    // PUT with the correct token: accepted, and the write persists.
    let (status, saved) = request_with_headers(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "theme": "dark", "accent": "coral" }),
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(saved["theme"], "dark");
    assert_eq!(saved["accent"], "coral");

    let (status, prefs) = request(app, "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(prefs["theme"], "dark");
    assert_eq!(prefs["accent"], "coral");
}

#[tokio::test]
async fn ui_preferences_put_allowed_via_loopback_trust() {
    // sc-8869 (F-067) / epic 4484: the method-aware gate must not break the
    // loopback-trust bypass. A loopback peer (the embedded desktop UI/worker) with
    // `trust_loopback` on writes preferences without a token, exactly as before, while
    // a LAN peer with no token is still rejected.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    settings.trust_loopback = true;
    let app = create_app(settings).expect("app creates");

    // Loopback peer, no token → allowed (desktop bypass preserved).
    let (status, saved) = request_with_peer(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "theme": "light" }),
        "127.0.0.1:51234",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(saved["theme"], "light");

    // LAN peer, no token → still rejected even with trust_loopback on.
    let (status, _) = request_with_peer(
        app,
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "theme": "dark" }),
        "192.168.1.5:51234",
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn ui_preferences_put_open_when_no_token_configured() {
    // Behavior with NO token configured must be unchanged: everything is open, so an
    // unauthenticated PUT still succeeds (the gate only bites when a token is set).
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    assert!(settings.access_token.is_empty());
    let app = create_app(settings).expect("app creates");

    let (status, saved) = request(
        app,
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "theme": "dark" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(saved["theme"], "dark");
}

#[tokio::test]
async fn ui_preferences_default_generation_quality_round_trips_and_merges() {
    // sc-10728 / epic 10721 R3: the app-wide default generation quality persists like theme/accent so
    // it survives a desktop relaunch — the shell's per-launch 127.0.0.1:<port> origin wipes origin-keyed
    // localStorage, so the durable copy must live server-side. GET resolves an unset value to the `auto`
    // default; PUT accepts auto|bf16|q8|q4 and MERGES, so a theme-only write can't reset a previously-
    // chosen quality, and an invalid value coerces to `auto`.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir); // no token configured ⇒ PUT is open
    assert!(settings.access_token.is_empty());
    let app = create_app(settings).expect("app creates");

    // Fresh install: GET resolves the absent value to the `auto` default so the web can seed it.
    let (status, prefs) = request(app.clone(), "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(prefs["defaultGenerationQuality"], "auto");

    // PUT a valid tier: persisted and echoed back.
    let (status, saved) = request(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "defaultGenerationQuality": "q4" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(saved["defaultGenerationQuality"], "q4");

    // Survives a "relaunch": a fresh GET reads it back from disk.
    let (status, prefs) = request(app.clone(), "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(prefs["defaultGenerationQuality"], "q4");

    // A partial PUT (theme only) MERGES — it must not clobber the stored quality.
    let (status, _) = request(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "theme": "dark" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, prefs) = request(app.clone(), "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(prefs["defaultGenerationQuality"], "q4");
    assert_eq!(prefs["theme"], "dark");

    // A present-but-invalid tier coerces to the `auto` default (defensive; the UI never sends one).
    let (status, saved) = request(
        app,
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "defaultGenerationQuality": "int8-convrot" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(saved["defaultGenerationQuality"], "auto");
}

/// epic 10721 R3: an existing install with a stored `q8` (the old forced default, from before the
/// `auto` option existed) is one-time migrated to `auto` on GET — but a q8 the user picks DELIBERATELY
/// after the upgrade (which sets the migration marker via PUT) is left alone.
#[tokio::test]
async fn ui_preferences_migrates_a_legacy_q8_to_auto_once() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let prefs_path = settings.config_dir.join("ui-preferences.json");
    std::fs::create_dir_all(&settings.config_dir).expect("config dir creates");
    // Simulate a pre-Auto install: a stored q8 with no migration marker (rode along in a theme PUT).
    std::fs::write(
        &prefs_path,
        r#"{"theme":"dark","defaultGenerationQuality":"q8"}"#,
    )
    .expect("seed legacy prefs");
    let app = create_app(settings).expect("app creates");

    // GET migrates the legacy q8 to auto and persists it.
    let (status, prefs) = request(app.clone(), "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(prefs["defaultGenerationQuality"], "auto");
    let on_disk = std::fs::read_to_string(&prefs_path).expect("prefs persisted");
    assert!(
        on_disk.contains("\"auto\""),
        "migration is written to disk: {on_disk}"
    );

    // A DELIBERATE q8 pick after the upgrade sets the marker and sticks across a fresh GET.
    let (status, _) = request(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "defaultGenerationQuality": "q8" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, prefs) = request(app, "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        prefs["defaultGenerationQuality"], "q8",
        "a deliberate post-upgrade q8 is not re-migrated"
    );
}

/// epic 10721 R1: the per-(screen, model) tier sticky persists server-side so a user's studio pick
/// survives a desktop relaunch (localStorage alone is wiped by the shell's per-launch origin). A PUT
/// carrying the map replaces it wholesale (the web sends the full merged map); a PUT without it merges.
#[tokio::test]
async fn ui_preferences_per_model_tier_round_trips_and_merges() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    // PUT a per-model tier map: persisted and echoed back.
    let (status, saved) = request(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "perModelTier": { "image": { "sana_sprint_1600m": "bf16" } } }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(saved["perModelTier"]["image"]["sana_sprint_1600m"], "bf16");

    // Survives a "relaunch": a fresh GET reads it back from disk.
    let (status, prefs) = request(app.clone(), "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(prefs["perModelTier"]["image"]["sana_sprint_1600m"], "bf16");

    // A theme-only PUT MERGES — it must not clobber the stored per-model map.
    let (status, _) = request(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "theme": "light" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, prefs) = request(app.clone(), "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(prefs["perModelTier"]["image"]["sana_sprint_1600m"], "bf16");
    assert_eq!(prefs["theme"], "light");

    // A new full map replaces (the web sends the merged map after adding another model's pick).
    let (status, saved) = request(
        app,
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "perModelTier": { "image": { "sana_sprint_1600m": "q8", "flux_dev": "q4" } } }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(saved["perModelTier"]["image"]["sana_sprint_1600m"], "q8");
    assert_eq!(saved["perModelTier"]["image"]["flux_dev"], "q4");
}

#[test]
fn requires_token_only_gates_api_paths() {
    use axum::http::Method;
    // Non-API paths (embedded UI / SPA fallback) must never require a token,
    // or the browser cannot load the bundle to prompt for one.
    assert!(!requires_token(&Method::GET, "/"));
    assert!(!requires_token(&Method::GET, "/assets/index-abc.js"));
    assert!(!requires_token(&Method::GET, "/projects/some-id"));
    // Public API paths stay open; other API paths stay gated.
    assert!(!requires_token(&Method::GET, "/api/v1/health"));
    assert!(requires_token(&Method::GET, "/api/v1/jobs"));
    assert!(requires_token(&Method::GET, "/api/v1/projects"));
}

#[test]
fn requires_token_ui_preferences_is_method_aware() {
    use axum::http::Method;
    // sc-8869 (F-067): the theme-preferences route shares a GET (pre-auth theme
    // read, public) and a PUT (disk write, must be authenticated). Only the GET is
    // exempt; the PUT — and any other method — is gated when a token is configured.
    assert!(!requires_token(&Method::GET, "/api/v1/ui-preferences"));
    assert!(requires_token(&Method::PUT, "/api/v1/ui-preferences"));
    assert!(requires_token(&Method::POST, "/api/v1/ui-preferences"));
    // Other single-method public paths stay exempt regardless of method (they
    // only wire one route method each), and non-public API paths stay gated.
    assert!(!requires_token(&Method::POST, "/api/v1/auth/verify"));
    assert!(requires_token(&Method::PUT, "/api/v1/credentials"));
}

#[tokio::test]
async fn embedded_ui_root_is_reachable_with_access_token_set() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    // With a token configured and no header, the embedded UI root and assets
    // must not be blocked by auth (404 here under default features since the
    // bundle isn't embedded; the point is it is NOT 401).
    let (status, _) = request(app.clone(), "GET", "/", Value::Null).await;
    assert_ne!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = request(app.clone(), "GET", "/assets/app.js", Value::Null).await;
    assert_ne!(status, StatusCode::UNAUTHORIZED);
    // API routes stay protected.
    let (status, _) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[cfg(feature = "embed-web")]
#[test]
fn embedded_ui_csp_locks_down_scripts_but_allows_app_resources() {
    let csp = crate::web_assets::CONTENT_SECURITY_POLICY;
    // The whole point: scripts only from this origin, no inline/eval escape hatch.
    assert!(csp.contains("script-src 'self'"));
    assert!(!csp.contains("script-src 'self' 'unsafe-inline'"));
    assert!(!csp.contains("unsafe-eval"));
    // Resources the app genuinely needs.
    assert!(csp.contains("default-src 'self'"));
    // Fonts are self-hosted (sc-8956): no third-party font host in the CSP.
    assert!(csp.contains("style-src 'self' 'unsafe-inline'"));
    assert!(csp.contains("font-src 'self'"));
    assert!(csp.contains("img-src 'self' data: blob:"));
    // Guard against a regression that silently re-adds the Google font origins.
    assert!(!csp.contains("fonts.googleapis.com"));
    assert!(!csp.contains("fonts.gstatic.com"));
    // Tauri IPC for the navigated desktop webview.
    assert!(csp.contains("ipc:"));
    // Hardening directives.
    assert!(csp.contains("object-src 'none'"));
    assert!(csp.contains("frame-ancestors 'none'"));
}

#[tokio::test]
async fn bearer_token_is_accepted_for_access_verification() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    let (status, verified) = request_with_headers(
        app,
        "POST",
        "/api/v1/auth/verify",
        Value::Null,
        &[("authorization", "Bearer secret-token")],
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(verified["ok"], true);
}

#[tokio::test]
async fn auth_verify_oracle_throttles_repeated_failures_per_peer() {
    // sc-8870 (F-068): the public `/api/v1/auth/verify` oracle returns `{ok}` for any
    // candidate token, so without a lockout a LAN attacker can brute-force the token
    // (which IS the password in LAN mode) at wire speed. After a burst of wrong-token
    // attempts from one peer IP, further attempts from that IP are refused with 429,
    // while a fresh IP and the valid token are unaffected.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");
    let attacker = "192.168.1.50:40000";
    let bad = [("authorization", "Bearer wrong-token")];

    // The first `AUTH_THROTTLE_MAX_FAILURES` (10) wrong guesses answer normally with
    // `{ok:false}` — the oracle still works, it just counts each miss.
    for _ in 0..10 {
        let (status, verified) = request_with_peer_headers(
            app.clone(),
            "POST",
            "/api/v1/auth/verify",
            Value::Null,
            attacker,
            &bad,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(verified["ok"], false);
    }

    // The next attempt from the same peer is refused before the oracle answers.
    let (status, body) = request_with_peer_headers(
        app.clone(),
        "POST",
        "/api/v1/auth/verify",
        Value::Null,
        attacker,
        &bad,
    )
    .await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        body["detail"], "Too many authentication attempts; try again later",
        "throttled response must not leak validity",
    );

    // Even a *correct* token from the now-blocked peer is refused: the lockout is by
    // IP, so an attacker can't slip a lucky guess past the throttle.
    let (status, _) = request_with_peer_headers(
        app.clone(),
        "POST",
        "/api/v1/auth/verify",
        Value::Null,
        attacker,
        &[("authorization", "Bearer secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);

    // A different peer IP is untouched and the valid token still verifies — a single
    // brute-forcer can't lock out the whole deployment, and legitimate use is fine.
    let (status, verified) = request_with_peer_headers(
        app.clone(),
        "POST",
        "/api/v1/auth/verify",
        Value::Null,
        "10.0.0.9:55555",
        &[("authorization", "Bearer secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(verified["ok"], true);
}

#[tokio::test]
async fn auth_verify_success_clears_peer_throttle_budget() {
    // sc-8870: a legitimate user who mistypes a few times must not get locked out once
    // they authenticate — a valid token clears the peer's failure record.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");
    let peer = "192.168.1.77:41000";

    // A handful of misses (under the cap of 10), then a success that resets the count.
    for _ in 0..5 {
        let (status, _) = request_with_peer_headers(
            app.clone(),
            "POST",
            "/api/v1/auth/verify",
            Value::Null,
            peer,
            &[("authorization", "Bearer wrong-token")],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }
    let (status, verified) = request_with_peer_headers(
        app.clone(),
        "POST",
        "/api/v1/auth/verify",
        Value::Null,
        peer,
        &[("authorization", "Bearer secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(verified["ok"], true);

    // With the budget reset, another full window of misses is tolerated again without
    // an early lockout (would have tripped at 10 total had the success not cleared it).
    for _ in 0..9 {
        let (status, verified) = request_with_peer_headers(
            app.clone(),
            "POST",
            "/api/v1/auth/verify",
            Value::Null,
            peer,
            &[("authorization", "Bearer wrong-token")],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "post-success budget must be reset");
        assert_eq!(verified["ok"], false);
    }
}

#[tokio::test]
async fn loopback_trusted_peer_is_never_throttled() {
    // sc-8870: the epic-4484 loopback-trust bypass must not accrue throttle failures —
    // the desktop UI/worker reach the API over loopback with no token, and that must
    // keep working no matter how many times it happens.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    settings.trust_loopback = true;
    let app = create_app(settings).expect("app creates");

    // Far more than the cap of gated requests with no token, all over loopback → all
    // served (the bypass returns before the throttle ever sees them).
    for _ in 0..20 {
        let (status, _) = request_with_peer(
            app.clone(),
            "GET",
            "/api/v1/jobs",
            Value::Null,
            "127.0.0.1:51234",
        )
        .await;
        assert_ne!(status, StatusCode::UNAUTHORIZED);
        assert_ne!(status, StatusCode::TOO_MANY_REQUESTS);
    }
}

#[tokio::test]
async fn gated_route_brute_force_is_throttled_per_peer() {
    // sc-8870: the throttle also covers the token oracle exposed by every gated route
    // (401-vs-200 reveals a valid token), not just `/auth/verify`. A LAN peer hammering
    // a gated route with a bad token gets locked out with 429 after the failure cap.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");
    let attacker = "192.168.1.60:42000";
    let bad = [("authorization", "Bearer wrong-token")];

    for _ in 0..10 {
        let (status, _) = request_with_peer_headers(
            app.clone(),
            "GET",
            "/api/v1/jobs",
            Value::Null,
            attacker,
            &bad,
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }
    let (status, body) = request_with_peer_headers(
        app.clone(),
        "GET",
        "/api/v1/jobs",
        Value::Null,
        attacker,
        &bad,
    )
    .await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        body["detail"],
        "Too many authentication attempts; try again later"
    );
}

#[tokio::test]
async fn gated_route_success_clears_peer_throttle_budget() {
    // sc-8870 (F-068): a valid token on a *gated route* must clear the peer's failure
    // budget too, not only the `/auth/verify` endpoint. A non-web LAN/API client
    // (epic 4484) hits gated routes directly with the token header; if it occasionally
    // sends a wrong token it would creep toward the cap with no reset path unless a
    // subsequent good request clears it. Accrue several misses, then one successful
    // gated-route auth from the same peer, then confirm a full fresh window of misses
    // is tolerated again (would have tripped at 10 total had success not reset it).
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");
    let peer = "192.168.1.61:43000";
    let bad = [("authorization", "Bearer wrong-token")];
    let good = [("authorization", "Bearer secret-token")];

    // Five misses (under the cap of 10) on a gated route.
    for _ in 0..5 {
        let (status, _) =
            request_with_peer_headers(app.clone(), "GET", "/api/v1/jobs", Value::Null, peer, &bad)
                .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    // A valid token on the same gated route passes auth and resets the budget.
    let (status, _) =
        request_with_peer_headers(app.clone(), "GET", "/api/v1/jobs", Value::Null, peer, &good)
            .await;
    assert_ne!(status, StatusCode::UNAUTHORIZED);
    assert_ne!(status, StatusCode::TOO_MANY_REQUESTS);

    // With the budget reset, nine more misses are tolerated without an early lockout —
    // without the gated-route `record_success` the 5 + 5th earlier miss would already
    // have pushed this past 10 and returned 429 instead of 401.
    for _ in 0..9 {
        let (status, _) =
            request_with_peer_headers(app.clone(), "GET", "/api/v1/jobs", Value::Null, peer, &bad)
                .await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "post-success budget must be reset on the gated-route path"
        );
    }
}

#[tokio::test]
async fn cors_preflight_allows_frontend_origin_and_token_header() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let request = Request::builder()
        .method("OPTIONS")
        .uri("/api/v1/jobs")
        .header("origin", "http://localhost:5173")
        .header("access-control-request-method", "POST")
        .header("access-control-request-headers", "X-SceneWorks-Token")
        .body(Body::empty())
        .expect("request builds");

    let response = app.oneshot(request).await.expect("response returns");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("access-control-allow-origin")
            .and_then(|value| value.to_str().ok()),
        Some("http://localhost:5173")
    );
    assert!(response
        .headers()
        .get("access-control-allow-headers")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("x-sceneworks-token")));
}

#[tokio::test]
async fn credentials_routes_store_redact_and_delete() {
    let temp = tempfile::tempdir().expect("tempdir");
    let settings = test_settings(&temp);

    // Save a credential; PUT returns the updated, redacted listing.
    let (status, body) = request(
        create_app(settings.clone()).expect("app creates"),
        "PUT",
        "/api/v1/credentials",
        json!({ "host": "https://Civitai.com", "label": "Civit.ai", "scheme": "query", "token": "secret-key" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let list = body.as_array().expect("array body");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["host"], "civitai.com"); // normalized
    assert_eq!(list[0]["label"], "Civit.ai");
    assert_eq!(list[0]["scheme"], "query");
    assert_eq!(list[0]["present"], true);
    assert!(
        list[0].get("token").is_none(),
        "listing must not include the token"
    );
    assert!(
        !body.to_string().contains("secret-key"),
        "token leaked in the response"
    );

    // A separate GET is likewise redacted.
    let (status, body) = request(
        create_app(settings.clone()).expect("app creates"),
        "GET",
        "/api/v1/credentials",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body.to_string().contains("secret-key"));

    // An empty token is rejected.
    let (status, _) = request(
        create_app(settings.clone()).expect("app creates"),
        "PUT",
        "/api/v1/credentials",
        json!({ "host": "huggingface.co", "token": "" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Delete returns the now-empty listing.
    let (status, body) = request(
        create_app(settings).expect("app creates"),
        "DELETE",
        "/api/v1/credentials/civitai.com",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.as_array().expect("array body").is_empty());
}

#[tokio::test]
async fn credentials_routes_require_the_access_token() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut settings = test_settings(&temp);
    settings.access_token = "s3cret".to_owned();

    let (status, _) = request(
        create_app(settings.clone()).expect("app creates"),
        "GET",
        "/api/v1/credentials",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let (status, _) = request_with_headers(
        create_app(settings).expect("app creates"),
        "GET",
        "/api/v1/credentials",
        Value::Null,
        &[("x-sceneworks-token", "s3cret")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}
