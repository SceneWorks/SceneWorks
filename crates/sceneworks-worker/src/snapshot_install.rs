//! Pinned-snapshot provisioning helpers and the install-layout smokes (sc-13797, sc-13810).
//!
//! These live in a CROSS-PLATFORM module on purpose. The code here — the download executor wrapper,
//! the transient/permanent error classifier, the pinned hub-layout assertions — is entirely
//! platform-agnostic; only the LIVE ~279 MiB hub download needs lane confinement. Expressing that
//! confinement as `#[cfg(target_os = "macos")]` on the one live TEST (rather than on the whole
//! module, where this used to live inside `voiceclone_smoke`) means every other lane still
//! type-checks and runs this code, so a break surfaces on the PR that caused it instead of only on
//! the macOS worker lane.
//!
//! The network-free counterpart that proves the same layout contract on every platform is
//! `tests::pinned_snapshot_layout_materializes_without_network` (sc-13810 item 3).

// The live macOS smoke is the only consumer of some helpers here; other targets compile the module
// without it (same situation `test_env` documents).
#![allow(dead_code)]

/// Materialize one HF snapshot (`repo` @ `revision`, filtered to `files`) into the hub cache via the
/// worker's REAL model-download executor — `HuggingFaceSnapshot::resolve` + `download_snapshot_into_cache`,
/// the exact seam `run_model_download_job` drives for a Models-screen install. Returns the resolved
/// commit SHA, or the [`crate::WorkerError`] the download seam surfaced (the caller decides whether a
/// transient transport failure should skip vs. hard-fail — see [`is_transient_transport_error`]).
/// Proving the pinned snapshots land here proves the install materializes them (sc-13541).
pub(crate) async fn try_install_snapshot(
    settings: &crate::Settings,
    repo: &str,
    revision: &str,
    files: &[&str],
) -> crate::WorkerResult<String> {
    use crate::downloads::{
        download_snapshot_into_cache, DownloadContext, DownloadProgress, HuggingFaceSnapshot,
    };
    let client = crate::downloads::streaming_download_client();
    let api = crate::ApiClient::new(settings);
    let repo_dir = sceneworks_core::hf_home::huggingface_repo_cache_path(&settings.data_dir, repo)
        .expect("resolve hub cache path");
    let file_patterns: Vec<String> = files.iter().map(|f| (*f).to_owned()).collect();
    let snapshot =
        HuggingFaceSnapshot::resolve(&client, settings, repo, revision, &file_patterns).await?;
    if snapshot.files.is_empty() {
        // A resolve that SUCCEEDED but matched zero files is a CONTRACT/layout break (a wrong pin or a
        // wrong file allow-list), NOT a transport blip — return a non-transport error so the caller
        // hard-fails on it (`is_transient_transport_error` returns `false` for `InvalidPayload`).
        return Err(crate::WorkerError::InvalidPayload(format!(
            "{repo}@{revision} resolved zero files for {file_patterns:?}"
        )));
    }
    let context = DownloadContext {
        api: &api,
        client: &client,
        settings,
        job_id: "sc-13541-offline-parity",
        cancel_message: "canceled",
        fresh_download: false,
    };
    // A report interval longer than any real transfer so the in-loop progress POST never fires: this
    // harness has no job API (the executor's heartbeat/progress are best-effort and only reached via
    // the interval tick), and cancel polls already swallow transport errors. The download itself only
    // talks to the HF hub, which is exactly what we are exercising.
    let mut progress = DownloadProgress::new(
        repo,
        0,
        snapshot.total_bytes(),
        std::time::Duration::from_secs(86_400),
    );
    download_snapshot_into_cache(&context, &repo_dir, revision, &snapshot, &mut progress).await
}

/// Panicking wrapper around [`try_install_snapshot`] for the heavy `#[ignore]d` real-weight smokes,
/// which are run by hand on a stable network and want ANY failure to be a loud panic.
pub(crate) async fn install_snapshot(
    settings: &crate::Settings,
    repo: &str,
    revision: &str,
    files: &[&str],
) -> String {
    try_install_snapshot(settings, repo, revision, files)
        .await
        .unwrap_or_else(|e| panic!("materialize {repo}@{revision}: {e}"))
}

/// Classify a [`crate::WorkerError`] from the model-download seam ([`HuggingFaceSnapshot::resolve`] /
/// `download_snapshot_into_cache`) as a **transient transport failure** — the HF hub was unreachable or
/// returned a transient server error — vs. a layout/integrity/contract failure the smoke MUST catch.
///
/// The set is deliberately NARROW, and by construction can NEVER match a layout regression: the
/// pinned-layout checks in [`whisper_base_fresh_install_pinned_layout_smoke`] are plain `assert!`s in
/// the test body (never `WorkerError`s), and every integrity failure the download seam raises is a
/// `WorkerError::InvalidPayload` (a sha256 or a size mismatch) or the zero-files contract error above —
/// none of which this returns `true` for. Transient means ONLY:
///   * a `reqwest` transport error with no usable HTTP response — connection refused / DNS failure /
///     TLS error / timeout / a reset before or mid-body (`is_connect`/`is_timeout`/`is_body`); OR
///   * an HTTP response carrying a transient server status — 5xx, 429 Too Many Requests, or 408
///     Request Timeout. A 4xx is NOT transient (a wrong pin/file is a 404, auth a 401/403) and stays a
///     hard failure; OR
///   * the download stall watchdog's hung-source signal (a `WorkerError::Io` of kind `TimedOut`).
///
/// Everything else — `InvalidPayload` (integrity/contract), `Json`, `Api`, `Engine`, `Canceled`, and
/// any other `Io` (e.g. a local filesystem error) — is a real regression and returns `false`.
pub(crate) fn is_transient_transport_error(error: &crate::WorkerError) -> bool {
    match error {
        crate::WorkerError::Http(http) => {
            if http.is_timeout() || http.is_connect() || http.is_body() {
                return true;
            }
            matches!(
                http.status(),
                Some(status)
                    if status.is_server_error()
                        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || status == reqwest::StatusCode::REQUEST_TIMEOUT
            )
        }
        crate::WorkerError::Io(io) => io.kind() == std::io::ErrorKind::TimedOut,
        _ => false,
    }
}

// ── sc-13797 (epic 13678): FAST, CI-runnable fresh-install-to-custom-cache layout smoke ──────────
// The pinned-revision hub layout (models--<repo>/snapshots/<rev>/<file> + refs/<sha>) is proven end
// to end by `chatterbox_offline_install_parity_smoke`, but that #[ignore]d smoke pulls the ~3.2 GB
// chatterbox repo — too heavy for routine CI. `whisper_base` is the TINY cataloged pin (three small
// allow-listed files, ~279 MB) that exercises the SAME `downloads.rs` provisioning seam, so the
// smoke below gives the download/layout contract a cheap regression guard on EVERY PR (worker lane)
// instead of only the weekly real-weights lane.
//
// sc-13810 (item 2): the pin is READ from the `whisper_base` builtin-manifest entry
// (`crate::manifest_pins`), not mirrored here as constants. Mirrored constants meant a pin bump or
// allow-list edit in config/manifests/builtin.models.jsonc left this smoke asserting the OLD
// contract — still green, against weights nobody ships.

/// The hub cache this smoke provisions into, keyed by the PINNED revision (sc-13810 item 1).
///
/// A fresh temp `HF_HOME` per run re-downloaded ~279 MiB every time: it widened the window for a
/// transient hub failure (the very thing the transport-skip below exists to absorb) and, because the
/// crate-global `ENV_LOCK` is held across the whole transfer, it parked every other env-touching
/// test behind a network download. Keying the cache by revision keeps BOTH properties that matter:
///   * reruns on a warm runner are near-instant — the executor re-verifies the existing files
///     instead of refetching them;
///   * the fresh-install path is still proven, because a PIN BUMP changes this directory and forces
///     a genuinely cold install (as does a first run on a new runner, or `..._FRESH=1`).
///
/// `SCENEWORKS_WHISPER_SMOKE_CACHE` points it at a durable per-runner volume; `..._FRESH=1` forces a
/// cold run on demand. Both are read before the env lock is taken, which is safe because no other
/// test writes them (unlike `HF_HOME`, which several do).
fn whisper_smoke_hub_cache(revision: &str) -> (std::path::PathBuf, bool) {
    let root = std::env::var_os("SCENEWORKS_WHISPER_SMOKE_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sceneworks-whisper-smoke"));
    let hf_home = root.join(revision);
    let force_fresh = std::env::var_os("SCENEWORKS_WHISPER_SMOKE_FRESH")
        .is_some_and(|value| value != "0" && !value.is_empty());
    if force_fresh {
        let _ = std::fs::remove_dir_all(&hf_home);
    }
    let cold = !hf_home.join("hub").exists();
    (hf_home, cold)
}

/// sc-13797 (epic 13678) — **fast, CI-runnable fresh-install-to-custom-cache layout smoke**. NOT
/// `#[ignore]d`: it runs inside `cargo test -p sceneworks-worker` on EVERY PR (the macOS worker
/// lane), unlike the ~3.2 GB `chatterbox_offline_install_parity_smoke` it COMPLEMENTS. It provisions
/// the TINY `whisper_base` pin (`openai/whisper-base` @ `e37978b…`, three small allow-listed files,
/// ~279 MB) through the SAME real download executor a Models-screen install runs
/// ([`install_snapshot`] → `HuggingFaceSnapshot::resolve` + `download_snapshot_into_cache`) into a
/// FRESH custom `HF_HOME` (temp dir), then asserts the pinned-revision hub layout the runtime
/// resolver reads: `models--openai--whisper-base/snapshots/<commit>/<file>` for EACH allow-listed
/// file, plus the `refs/<rev>` pointer (recording the resolved commit). Network is allowed at test
/// time; only the three allow-listed files download — NOT the whole `whisper-base` repo (which ships
/// many more upstream) — which is what keeps it fast AND is asserted as a discriminating check.
///
/// This is the cheap regression guard for the download/layout contract (epic 13678): a break in how
/// `download_snapshot_into_cache` materializes the pinned layout would turn this red on every PR,
/// rather than surfacing only when someone runs the heavy chatterbox smoke by hand.
///
/// TRANSIENT-FAILURE POLICY (sc-13797 adversarial review, issue 1): the test's PURPOSE is to guard the
/// cache-LAYOUT code, not the hub's uptime. It therefore SKIPS (logs + returns, no failure) when the HF
/// hub is unreachable or returns a transient server error (connection/DNS/timeout/reset, or 5xx/429/408
/// — see [`is_transient_transport_error`]), so a transient outage can't redden UNRELATED PRs on this
/// merge-gating lane. The skip is upstream of EVERY layout/integrity assertion and matches ONLY
/// transport-transient errors, so it can never mask a wrong-snapshot-path / missing-refs /
/// wrong-file-set / corrupt-download regression — all of those still hard-fail.
///
/// LANE CONFINEMENT (sc-13797 adversarial review, issue 2): this test lives in the
/// `#[cfg(all(test, target_os = "macos"))]` module, so it runs ONLY on the stable self-hosted macOS
/// worker lane — deliberately NOT on the hosted Windows worker-test lane or untrusted fork PRs.
/// Confining the LIVE ~279 MiB HF download to a runner with a stable network + warm cache is what keeps
/// this guard reliable; spreading it onto hosted/fork infra we don't control would AMPLIFY the
/// transient flakiness the skip above already has to work around. Network-free cross-platform
/// coverage of the same layout contract is now provided by
/// `tests::pinned_snapshot_layout_materializes_without_network` (sc-13810 item 3), which drives this
/// exact seam against a local stub and so runs on EVERY lane including hosted Windows and forks.
#[test]
fn whisper_base_fresh_install_pinned_layout_smoke() {
    // LANE CONFINEMENT AS A RUNTIME GUARD, NOT A `cfg` (sc-13810). `#[cfg(target_os = "macos")]` on
    // this test would exclude its BODY from compilation everywhere else, so a type error or a stale
    // API call in it could only be discovered on the macOS lane — the exact late-feedback problem
    // moving this module cross-platform was meant to remove. Compiling it on every target and
    // returning early off-macOS keeps the confinement identical (the live ~279 MiB download still
    // happens ONLY on the self-hosted macOS runner) while type-checking the body everywhere.
    //
    // This is a deliberate platform gate on a compile-time constant, not an environment-dependent
    // silent pass: the lanes that skip it are covered for the SAME layout contract by
    // `tests::pinned_snapshot_layout_materializes_without_network`, which needs no network.
    if !cfg!(target_os = "macos") {
        eprintln!(
            "[whisper-layout] SKIP — the LIVE hub download is confined to the self-hosted macOS \
             worker lane. The same pinned-layout contract is asserted network-free on this lane by \
             `pinned_snapshot_layout_materializes_without_network`."
        );
        return;
    }
    // The pin comes from the manifest, so a `revision`/`files` edit is followed here rather than
    // mirrored (sc-13810 item 2).
    let pin = crate::manifest_pins::builtin_model_pin("whisper_base");
    let whisper_files = pin.file_refs();
    // ── Isolate the hub the downloader writes into: HF_HOME at a PER-PIN persistent root (the
    //    desktop's sc-1904 default shape HF_HOME=<root>/.cache/huggingface ⇒ hub at <HF_HOME>/hub,
    //    per huggingface_hub_cache_dir). Clear every other HF cache var so cache-path resolution
    //    can't diverge to another location or the data_dir fallback. EnvVars restores on drop — even
    //    on a panicking assert — so a leaked HF_HOME can't poison sibling tests in the shared
    //    process.
    let data_dir = tempfile::tempdir().expect("temp data dir");
    let (hf_home, cold_install) = whisper_smoke_hub_cache(&pin.revision);
    let hub = hf_home.join("hub");
    let _env = crate::test_env::EnvVars::set(&[
        ("HF_HOME", hf_home.to_str().expect("utf-8 HF_HOME")),
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HUB_OFFLINE", ""),
    ]);
    let repo_slug = pin.repo_slug();
    // On a COLD run (new runner, bumped pin, or SCENEWORKS_WHISPER_SMOKE_FRESH=1) this really is a
    // fresh install and the layout below is built from nothing — the property sc-13797 asserts. On a
    // WARM run the executor re-verifies the cached files instead, which is the point of the per-pin
    // cache; every layout assertion still runs either way.
    if cold_install {
        assert!(
            !hub.join(&repo_slug).exists(),
            "a cold run must start from an EMPTY hub: {}",
            hub.join(&repo_slug).display()
        );
    }

    let settings = crate::Settings {
        api_url: "http://127.0.0.1:1".to_owned(),
        access_token: None,
        data_dir: data_dir.path().to_path_buf(),
        config_dir: data_dir.path().join("config"),
        worker_id: "sc-13797".to_owned(),
        gpu_id: "gpu-0".to_owned(),
        is_child_worker: false,
        poll_seconds: 1,
        heartbeat_seconds: 5,
        shutdown_timeout_seconds: 1,
        huggingface_base_url: crate::DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
        huggingface_token: std::env::var("HF_TOKEN").ok(),
        credentials: Vec::new(),
        max_lora_url_bytes: crate::DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: crate::DEFAULT_MAX_MODEL_URL_BYTES,
        allow_private_lora_urls: false,
        utility_workers: 1,
        backend_mlx_enabled: true,
        backend_candle_enabled: false,
        gpu_memory_limit_bytes: 0,
        external_model_roots: Vec::new(),
    };

    // ── Provision the pinned snapshot through the REAL download executor (online), filtered to the
    //    manifest's allow-listed files — NOT the whole repo. Returns the resolved commit SHA. On a
    //    warm per-pin cache this re-verifies the existing files rather than refetching them.
    let (repo, revision) = (pin.repo.as_str(), pin.revision.as_str());
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let commit = match rt.block_on(try_install_snapshot(
        &settings,
        repo,
        revision,
        &whisper_files,
    )) {
        Ok(commit) => commit,
        // GUARD, not gate (issue 1): a transient HF-hub transport failure (unreachable / DNS / timeout
        // / reset / 5xx / 429) SKIPS rather than reddening this merge-gating lane on an UNRELATED PR.
        // This arm is upstream of every layout/integrity assertion below and matches ONLY
        // transport-transient errors, so it can never mask a wrong-snapshot-path / missing-refs /
        // wrong-file-set / corrupt-download regression — those are non-transient and hard-fail below.
        Err(error) if is_transient_transport_error(&error) => {
            eprintln!(
                "[whisper-layout] SKIP — the HF hub was unreachable or returned a transient error while \
                 provisioning {repo}@{revision}: {error}. The cache-LAYOUT assertions \
                 were NOT exercised (this guards our download code, not the hub's uptime); re-run when \
                 the hub is reachable."
            );
            return;
        }
        // Any NON-transient failure — a wrong pin (404), a corrupt download (sha256/size mismatch), a
        // zero-files contract break, a local filesystem error — is a REAL regression: hard-fail.
        Err(error) => panic!(
            "provisioning {repo}@{revision} failed with a NON-transient error \
             (a real download/layout regression, not a hub outage): {error}"
        ),
    };
    // A pinned commit-SHA revision resolves to itself: the snapshot dir the resolver reads is
    // snapshots/<commit>, and here <commit> == the pinned <rev>.
    assert_eq!(
        commit, revision,
        "a pinned commit revision must resolve to itself (got {commit})"
    );

    // ── Assert the pinned-revision hub layout the resolver reads — exactly what
    //    download_snapshot_into_cache writes: refs/<rev> -> commit, snapshots/<commit>/<file>. ─────
    let repo_dir = hub.join(&repo_slug);
    let snapshot_dir = repo_dir.join("snapshots").join(&commit);
    for file in &whisper_files {
        let path = snapshot_dir.join(file);
        assert!(
            path.exists(),
            "{file} must materialize at the pinned snapshot dir: {}",
            path.display()
        );
    }
    let refs_pointer = repo_dir.join("refs").join(revision);
    assert!(
        refs_pointer.exists(),
        "the refs/<sha> pointer must exist: {}",
        refs_pointer.display()
    );
    let recorded = std::fs::read_to_string(&refs_pointer).expect("read refs/<sha> pointer");
    assert_eq!(
        recorded.trim(),
        commit,
        "refs/<rev> must record the resolved commit SHA (the layout's rev→commit link)"
    );

    // Only the allow-listed files materialized — the smoke stays fast BECAUSE it never pulled the
    // whole repo (whisper-base ships many more files upstream; the manifest allow-list is the lever).
    // An exact match is the discriminating check: a regression that fetched the full repo, or dropped
    // a requested file, would break this.
    let mut installed: Vec<String> = std::fs::read_dir(&snapshot_dir)
        .expect("read pinned snapshot dir")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    installed.sort();
    let mut expected = pin.files.clone();
    expected.sort();
    assert_eq!(
        installed, expected,
        "ONLY the manifest's allow-listed files must materialize (the whole repo is NOT pulled)"
    );

    eprintln!(
        "[whisper-layout] PASS — provisioned {repo}@{revision} ({} files) into the \
         {} per-pin hub {}; pinned layout snapshots/<commit>/<file> + refs/<sha> verified.",
        whisper_files.len(),
        if cold_install { "COLD" } else { "warm" },
        hub.display()
    );
}

/// sc-13797 (issue 1) — lock in the SAFETY-CRITICAL half of [`is_transient_transport_error`]: every
/// integrity/contract error class the download seam raises (and every non-timeout local error) must
/// classify NON-transient, so the whisper smoke's transient-SKIP can NEVER swallow a real
/// download/layout regression. This is a mutation guard: flip the classifier to return `true` for
/// `InvalidPayload` (masking a corrupt/zero-files install) and this test goes red. (The
/// `WorkerError::Http` transport variants have no public `reqwest::Error` constructor to build here, so
/// they can't be exercised without real network; they are covered by the classifier's narrow match on
/// `is_timeout`/`is_connect`/`is_body`/transient-status and the reasoning in its doc comment.)
#[test]
fn transient_classifier_never_skips_integrity_or_contract_failures() {
    use crate::WorkerError;

    // An integrity failure (a sha256/size mismatch surfaces as `InvalidPayload`) MUST hard-fail.
    assert!(
        !is_transient_transport_error(&WorkerError::InvalidPayload(
            "ve.safetensors failed its integrity check (sha256 …)".to_owned()
        )),
        "an integrity mismatch must never be classified transient (it would mask a corrupt install)"
    );
    // The zero-files CONTRACT break (also `InvalidPayload`) MUST hard-fail.
    assert!(
        !is_transient_transport_error(&WorkerError::InvalidPayload(
            "openai/whisper-base@… resolved zero files".to_owned()
        )),
        "a zero-files contract break must never be classified transient"
    );
    // A generic local filesystem error is NOT transient.
    assert!(!is_transient_transport_error(&WorkerError::Io(
        std::io::Error::new(std::io::ErrorKind::PermissionDenied, "fs")
    )));
    // Cancellation and engine failures are NOT transient.
    assert!(!is_transient_transport_error(&WorkerError::Canceled(
        "canceled".to_owned()
    )));
    assert!(!is_transient_transport_error(&WorkerError::Engine(
        "engine".to_owned()
    )));

    // The ONLY non-Http transient signal is the download stall watchdog's hung-source `Io(TimedOut)`.
    assert!(
        is_transient_transport_error(&WorkerError::Io(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "download stalled: no forward progress"
        ))),
        "the stall watchdog's Io(TimedOut) is the one non-Http transient signal"
    );
}
