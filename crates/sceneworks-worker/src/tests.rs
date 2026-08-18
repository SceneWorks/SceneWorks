//! Worker unit-test module (sc-8867, epic 8800).
//!
//! This file was a single 5,400-line flat module mixing 8+ domains (HF-CLI/family validation,
//! snapshot downloads + the axum stub servers, tokenizer overlays, supervisor lifecycle, media
//! planning, credentials, cancel/heartbeat, manifest gating) with the shared setup helpers buried
//! thousands of lines below their first use. It is now split into per-domain files under `tests/`,
//! spliced back in with `include!` — the SAME mechanism `image_jobs.rs` uses to keep a large module
//! in one scope while living in several files.
//!
//! `include!` inlines each file textually at this module's scope, so:
//!   * every `#[test]` / `#[tokio::test]` and its assertions move VERBATIM (nothing added, removed,
//!     skipped, or altered) and keeps its fully-qualified name `tests::<name>` — the exact same test
//!     set runs before and after the split;
//!   * the shared imports above and the shared setup helpers (the axum stub servers, `test_settings`,
//!     the snapshot/JSON builders, `write_safetensors_with_keys`, …) live once and are visible to
//!     every included file without re-declaration or re-import.
//!
//! Grouping is by subsystem; the heavily cross-cutting stub servers + `test_settings` cluster live in
//! `tests/support_stubs.rs`. The `#[cfg(...)]` gates on the imports/tests are preserved as-is.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio as StdStdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode as AxumStatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use sceneworks_core::contracts::{JobSnapshot, WorkerUtilizationSnapshot};
use serde_json::{json, Value};
use tempfile::tempdir;

use super::api_client::ApiClient;
use super::downloads::{
    build_source_url_client, credential_for_host, download_lora_source_url,
    download_progress_payload, download_snapshot_into_cache, download_source_url, normalize_sha256,
    report_download_progress, DownloadContext, DownloadProgress, HuggingFaceSnapshot, SnapshotFile,
};
#[cfg(target_os = "macos")]
use super::gpu::mlx_gpu;
use super::gpu::{
    cached_gpu_utilization_with, cpu_gpu, cpu_worker_id, fallback_gpu, gpu_worker_id,
    parse_max_compute_cap, parse_nvidia_smi_gpus, parse_selected_compute_cap, run_bounded_command,
    visible_gpu_ids, worker_capabilities_with_utility, GpuDiscoveryAttempt, GpuDiscoveryFailure,
};
use super::media_jobs::{
    candidate_people, concat_file_contents, crossfade_duration, output_dimensions, plan_segments,
    run_ffmpeg,
};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::media_jobs::{mask_rollup_state, segment_assembly_frames, SegmentClip, SegmentOutcome};
use super::model_jobs::{
    check_downloaded_model_family, derived_tokenizer_overlay,
    downloaded_model_detection_io_error_is_inconclusive, finalize_converted_dir,
    finalize_converted_dir_with_test_hook, huggingface_receipt_weights_dir,
    overlay_derived_tokenizer, receipt_markers_read, recover_stranded_model_conversions,
    reset_receipt_markers_read, validate_hf_download_inputs, DownloadFamilyCheck,
};
// `terminating_signal` is only exercised by a `#[cfg(unix)]` test (signal-death
// attribution is uncatchable and only observable on Unix), so gate the import to
// match — otherwise it is an unused import on Windows builds.
#[cfg(unix)]
use super::supervisor::terminating_signal;
use super::supervisor::{
    auto_worker_specs, child_died_abnormally, child_environment, choose_auto_worker_specs_with,
    restart_exited_children_at, restart_exited_children_with_spawner, stop_children,
    utility_worker_specs, SupervisedChild, WorkerSpec,
};
use super::{
    allow_pattern_matches, bounded_tail, cancel_requested_peek, cleanup_uploaded_import_source,
    copy_lora_source, fresh_asset_id, heartbeat_while_blocking, import_lora_source_file_as,
    import_lora_source_path, is_database_locked, model_file_identity,
    normalize_app_managed_cache_path, now_rfc3339, parse_credentials_env,
    resolve_model_convert_output, resolve_model_import_target, safe_download_dir,
    safe_project_path, value_f64, wait_for_shutdown_latch, wan_moe_pair_filenames,
    write_model_download_receipt, write_model_install_marker, CredentialScheme, IdleHeartbeat,
    JsonObject, SafetensorsHeaderError, Settings, WorkerCredential, WorkerError,
    DEFAULT_MAX_LORA_URL_BYTES, DEFAULT_MAX_MODEL_URL_BYTES, DEFAULT_TRANSITION_DURATION_SECONDS,
    INSTALL_MARKER, MODEL_CONVERSION_BACKUPS_DIR_NAME,
};

// HF-CLI input validation, downloaded-model family detection, atomic converted-dir finalize, and the
// download-dir / cache-path / repo-slug cross-language contracts.
include!("tests/hf_and_family.rs");
// GPU/capability advertisement (nvidia-smi parsing, MLX vs CPU capability sets) and manifest gating
// (model-table descriptor flags, SD3.5 / SANA manifest entries) + the builtin-manifest builders.
include!("tests/gpu_and_manifest.rs");
// ImageEdit dispatch + child-supervisor lifecycle (restart/backoff, signal-death attribution) and the
// model-install marker write.
include!("tests/supervisor_lifecycle.rs");
// LoRA import/upload semantics + HuggingFace snapshot downloads (digest verify, resume, cache layout)
// and the derived-tokenizer overlay path (with its overlay stub server).
include!("tests/lora_and_downloads.rs");
// Worker-credential parsing/precedence and the source-URL client (redirect/auth-strip/DNS-pin/scheme
// guards) with its cross-host + location redirect stub servers.
include!("tests/credentials_and_source_url.rs");
// Media planning (segment plan, ffmpeg timeline shapes, crossfade defaults, seedvr2 sizing) and the
// path-confinement / import-root guards.
include!("tests/media_and_paths.rs");
// Shared cross-cutting test setup: the axum stub servers (HF, binary, job/progress, cancel-tick,
// cancel-poll, progress-capture), the JSON snapshot builders, `test_settings`, the child spawners,
// and the `huggingface_snapshot_resolve` test that exercises the HF stub directly.
include!("tests/support_stubs.rs");
// Cancel + heartbeat lifecycle: placeholder-job shutdown, begin_*_cancel acks, batched-analysis
// deferral, cancel-join guards, run_blocking_with_heartbeat, keepalive, kill-on-drop, and the
// stub-fallback gate submodule.
include!("tests/cancel_and_heartbeat.rs");
// The `segment_assembly_frames` orchestrator contract tests (fake-segmenter mask assembly/rollup).
include!("tests/segment_assembly.rs");
// Source-level guard keeping test fixtures off the shared temp root (sc-17641 / sc-17707), with the
// production work dirs and deliberate exceptions enumerated and justified.
include!("tests/temp_fixture_guard.rs");

/// sc-19710: the retention driver is opt-in and must never create the managed cache root as a side
/// effect — an opt-out install growing a `models/resolved` tree just because the worker booted
/// would be exactly the unrequested state the policy exists to prevent. A store that already
/// exists is still driven even when the policy is disabled, so entries materialized while it was
/// enabled cannot be stranded by turning the cache off.
#[test]
fn resolved_cache_retention_is_opt_in_and_never_creates_the_store() {
    // `resolved_cache_retention` reads the cache policy from the PROCESS environment, so the
    // "disabled" half of this test only means anything while `SCENEWORKS_RESOLVED_CACHE_ENABLED`
    // is pinned off. Pinning it through the shared `test_env` lock does both jobs at once: it
    // states the precondition instead of inheriting whatever the process happens to have, and it
    // excludes the concurrent env mutators (`with_local_cache` and friends) that would otherwise
    // flip the policy to enabled mid-assertion. `.cargo/config.toml` does NOT set
    // `RUST_TEST_THREADS`, so this suite really does run in parallel and the race is live.
    crate::test_env::temp_env_vars(
        &[(
            sceneworks_core::model_artifacts::resolved_cache::RESOLVED_CACHE_ENABLED_ENV,
            "0",
        )],
        || {
            let temp = tempfile::tempdir().expect("tempdir");
            let data_dir = temp.path().join("data");
            std::fs::create_dir_all(&data_dir).expect("data dir creates");
            let resolved_root = data_dir.join("models").join("resolved");

            assert!(
                crate::resolved_cache_retention(&data_dir).is_none(),
                "a disabled cache with no store has nothing to drive"
            );
            assert!(
                !resolved_root.exists(),
                "probing for retention must not create the managed cache root"
            );

            drop(
                sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheStore::open(
                    &data_dir,
                )
                .expect("store opens"),
            );
            assert!(
                crate::resolved_cache_retention(&data_dir).is_some(),
                "an existing store must still be reconciled after the policy is disabled"
            );
        },
    );
}

/// sc-19710: a retention checkpoint must never sit on the claim path. A sweep that evicts walks
/// every entry and re-hashes each candidate's source, so if the poll loop awaited one, a job
/// submitted just after a sweep began would wait for the whole sweep. This pins the property the
/// loop relies on: the checkpoint is *started* and its handle only polled, so the caller keeps
/// running — and is skipped while a predecessor is still in flight, so sweeps cannot stack.
#[tokio::test]
async fn a_retention_checkpoint_never_blocks_the_caller_that_starts_it() {
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    // A real store, so the checkpoint actually has work to gate on rather than returning early.
    drop(
        sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheStore::open(&data_dir)
            .expect("store opens"),
    );

    let started = std::time::Instant::now();
    let handle = crate::spawn_retention_checkpoint(data_dir.clone(), true);
    // Control returns without running the sweep inline: it is on the blocking pool.
    assert!(
        started.elapsed() < std::time::Duration::from_millis(200),
        "starting a checkpoint must not run it inline"
    );
    // The detached work still completes on its own.
    handle.await.expect("checkpoint task completes");

    // The timing check above cannot fail a fast sweep over a small store, so the real guarantee —
    // that the poll loop never *awaits* a checkpoint — is pinned at the source level, the same way
    // this crate pins its temp-fixture rule. Re-introducing an await on the claim path reddens
    // this immediately, where a behavioural test would silently keep passing on a small cache.
    let source = include_str!("lib.rs");
    let loop_body = source
        .split_once("let mut retention_task =")
        .expect("the poll loop keeps the checkpoint handle in a local")
        .1;
    assert!(
        !loop_body.contains("retention_task.expect")
            && !loop_body.contains("retention_task.unwrap"),
        "the poll loop must not unwrap the checkpoint handle"
    );
    for forbidden in [
        "spawn_retention_checkpoint(settings.data_dir.clone(), false).await",
        "spawn_retention_checkpoint(settings.data_dir.clone(), true).await",
        "run_retention_checkpoint",
    ] {
        assert!(
            !loop_body.contains(forbidden),
            "the poll loop must never await a retention checkpoint (found `{forbidden}`): a sweep \
             re-hashes every eviction candidate's source, so awaiting one delays the next claim"
        );
    }
}
