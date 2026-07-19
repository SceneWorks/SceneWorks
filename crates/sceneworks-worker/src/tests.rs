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

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
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
    download_progress_payload, download_snapshot_into_cache, normalize_sha256,
    report_download_progress, DownloadContext, DownloadProgress, HuggingFaceSnapshot, SnapshotFile,
};
#[cfg(target_os = "macos")]
use super::gpu::mlx_gpu;
use super::gpu::{
    cpu_gpu, cpu_worker_id, fallback_gpu, gpu_worker_id, parse_max_compute_cap,
    parse_nvidia_smi_gpus, visible_gpu_ids, worker_capabilities_with_utility,
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
    overlay_derived_tokenizer, validate_hf_download_inputs, DownloadFamilyCheck,
};
// `terminating_signal` is only exercised by a `#[cfg(unix)]` test (signal-death
// attribution is uncatchable and only observable on Unix), so gate the import to
// match — otherwise it is an unused import on Windows builds.
#[cfg(unix)]
use super::supervisor::terminating_signal;
use super::supervisor::{
    auto_worker_specs, child_died_abnormally, child_environment, restart_exited_children_at,
    restart_exited_children_with_spawner, stop_children, utility_worker_specs, SupervisedChild,
    WorkerSpec,
};
use super::{
    allow_pattern_matches, bounded_tail, cancel_requested_peek, cleanup_uploaded_import_source,
    copy_lora_source, fresh_asset_id, import_lora_source_file_as, import_lora_source_path,
    normalize_app_managed_cache_path, now_rfc3339, parse_credentials_env,
    resolve_model_convert_output, resolve_model_import_target, safe_download_dir,
    safe_project_path, value_f64, wan_moe_pair_filenames, write_model_download_receipt,
    write_model_install_marker, CredentialScheme, IdleHeartbeat, JsonObject,
    SafetensorsHeaderError, Settings, WorkerCredential, WorkerError, DEFAULT_MAX_LORA_URL_BYTES,
    DEFAULT_MAX_MODEL_URL_BYTES, DEFAULT_TRANSITION_DURATION_SECONDS, INSTALL_MARKER,
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
