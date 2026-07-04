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
    cpu_gpu, cpu_worker_id, fallback_gpu, gpu_worker_id, parse_nvidia_smi_gpus, visible_gpu_ids,
    worker_capabilities_with_utility,
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
    hf_cli_encoding_failure, overlay_derived_tokenizer, validate_hf_cli_download_inputs,
    DownloadFamilyCheck, HF_CLI_UTF8_ENV,
};
// `terminating_signal` is only exercised by a `#[cfg(unix)]` test (signal-death
// attribution is uncatchable and only observable on Unix), so gate the import to
// match — otherwise it is an unused import on Windows builds.
#[cfg(unix)]
use super::supervisor::terminating_signal;
use super::supervisor::{
    auto_worker_specs, child_died_abnormally, child_environment, restart_exited_children_at,
    restart_exited_children_with_spawner, utility_worker_specs, SupervisedChild, WorkerSpec,
};
use super::{
    allow_pattern_matches, bounded_tail, cancel_requested_peek, cleanup_uploaded_import_source,
    copy_lora_source, fresh_asset_id, import_lora_source_file_as, import_lora_source_path,
    normalize_app_managed_cache_path, now_rfc3339, parse_credentials_env,
    resolve_model_convert_output, resolve_model_import_target, safe_download_dir,
    safe_project_path, value_f64, wan_moe_pair_filenames, write_model_install_marker,
    CredentialScheme, IdleHeartbeat, JsonObject, SafetensorsHeaderError, Settings,
    WorkerCredential, WorkerError, DEFAULT_MAX_LORA_URL_BYTES, DEFAULT_MAX_MODEL_URL_BYTES,
    DEFAULT_TRANSITION_DURATION_SECONDS, INSTALL_MARKER,
};

fn write_safetensors_with_keys(path: &std::path::Path, keys: &[String]) {
    // Minimal valid safetensors: 8-byte little-endian header length + JSON header.
    // The family detector only reads the header, so empty tensor slices are fine.
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), json!({"format": "pt"}));
    for key in keys {
        header.insert(
            key.clone(),
            json!({"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}),
        );
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
    let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
    buffer.extend_from_slice(&header_bytes);
    std::fs::write(path, buffer).expect("write safetensors");
}

fn wan_video_safetensors_keys() -> Vec<String> {
    // Mirrors the Wan2.2 architecture signature the family detector keys on.
    let mut keys = Vec::new();
    for block in 0..30 {
        for module in ["self_attn.q", "self_attn.k", "cross_attn.q", "ffn.0"] {
            keys.push(format!("transformer.blocks.{block}.{module}.lora_A.weight"));
            keys.push(format!("transformer.blocks.{block}.{module}.lora_B.weight"));
        }
    }
    keys
}

#[test]
fn hf_cli_windows_encoding_failures_are_detected() {
    let stderr = "Fetching 28 files: 100%|##########| 28/28 [00:00<00:00, 14016.05it/s]\n\
                  Error: Invalid value. 'charmap' codec can't encode character '\\u2713' \
                  in position 5: character maps to <undefined>";

    assert!(hf_cli_encoding_failure(stderr));
    assert!(!hf_cli_encoding_failure("Error: Repository not found."));
}

#[test]
fn hf_cli_environment_forces_python_utf8_output() {
    let env: HashMap<_, _> = HF_CLI_UTF8_ENV.into_iter().collect();

    assert_eq!(env.get("PYTHONUTF8"), Some(&"1"));
    assert_eq!(env.get("PYTHONIOENCODING"), Some(&"utf-8"));
    assert_eq!(env.get("HF_HUB_DISABLE_PROGRESS_BARS"), Some(&"1"));
}

#[test]
fn hf_cli_download_inputs_accept_catalog_values() {
    validate_hf_cli_download_inputs(
        "black-forest-labs/FLUX.1-dev",
        "refs/pr/12",
        &[
            "*.safetensors".to_owned(),
            "text_encoder/model-00001-of-00002.safetensors".to_owned(),
            "tokenizer/{config,merges}.json".to_owned(),
        ],
    )
    .expect("catalog HF values are accepted");
}

#[test]
fn hf_cli_download_inputs_reject_option_injection() {
    for repo in ["--local-dir=/Users/me/.ssh", "owner/--local-dir=/tmp/out"] {
        let error = validate_hf_cli_download_inputs(repo, "main", &["*.safetensors".to_owned()])
            .expect_err("malicious repo rejected");
        assert!(matches!(error, WorkerError::InvalidPayload(_)));
    }

    let error = validate_hf_cli_download_inputs(
        "owner/model",
        "--local-dir=/tmp/out",
        &["*.safetensors".to_owned()],
    )
    .expect_err("malicious revision rejected");
    assert!(matches!(error, WorkerError::InvalidPayload(_)));

    let error = validate_hf_cli_download_inputs(
        "owner/model",
        "main",
        &["--local-dir=/tmp/out".to_owned()],
    )
    .expect_err("malicious include pattern rejected");
    assert!(matches!(error, WorkerError::InvalidPayload(_)));
}

#[test]
fn hf_cli_download_inputs_reject_traversal_and_absolute_patterns() {
    for pattern in [
        "../model.safetensors",
        "nested/../../model.safetensors",
        "/tmp/model.bin",
    ] {
        let error = validate_hf_cli_download_inputs("owner/model", "main", &[pattern.to_owned()])
            .expect_err("unsafe include pattern rejected");
        assert!(matches!(error, WorkerError::InvalidPayload(_)));
    }

    for revision in ["../main", "refs/heads/../main", "/refs/main"] {
        let error =
            validate_hf_cli_download_inputs("owner/model", revision, &["*.safetensors".to_owned()])
                .expect_err("unsafe revision rejected");
        assert!(matches!(error, WorkerError::InvalidPayload(_)));
    }
}

#[test]
fn app_managed_cache_path_rejects_escape_and_symlink_escape() {
    let temp = tempdir().expect("temp dir");
    let mut settings = Settings::from_env();
    settings.data_dir = temp.path().join("data");
    let uploads = settings.data_dir.join("cache").join("pose-uploads");
    std::fs::create_dir_all(&uploads).expect("uploads dir");
    let staged = uploads.join("upload.png");
    std::fs::write(&staged, b"image").expect("staged file");

    let accepted = normalize_app_managed_cache_path(
        &settings,
        staged.to_str().unwrap(),
        "pose-uploads",
        "sourcePath",
    )
    .expect("staged path accepted");
    assert_eq!(
        accepted,
        staged.canonicalize().expect("canonical staged path")
    );

    let outside = temp.path().join("outside.png");
    std::fs::write(&outside, b"not staged").expect("outside file");
    let error = normalize_app_managed_cache_path(
        &settings,
        outside.to_str().unwrap(),
        "pose-uploads",
        "sourcePath",
    )
    .expect_err("outside path rejected");
    assert!(matches!(error, WorkerError::InvalidPayload(_)));

    #[cfg(unix)]
    {
        let link = uploads.join("link.png");
        std::os::unix::fs::symlink(&outside, &link).expect("symlink");
        let error = normalize_app_managed_cache_path(
            &settings,
            link.to_str().unwrap(),
            "pose-uploads",
            "sourcePath",
        )
        .expect_err("symlink escape rejected");
        assert!(matches!(error, WorkerError::InvalidPayload(_)));
    }
}

#[test]
fn downloaded_model_windows_untrusted_mount_detection_is_inconclusive() {
    let error = SafetensorsHeaderError::Io(std::io::Error::from_raw_os_error(448));

    assert!(downloaded_model_detection_io_error_is_inconclusive(&error));
    assert!(!downloaded_model_detection_io_error_is_inconclusive(
        &SafetensorsHeaderError::InvalidHeader
    ));
    assert!(!downloaded_model_detection_io_error_is_inconclusive(
        &SafetensorsHeaderError::Io(std::io::Error::from_raw_os_error(5))
    ));
}

#[test]
fn download_family_check_proceeds_when_no_weights_to_detect() {
    // A curated catalog download with no detectable signal (no safetensors yet, or
    // an inconclusive header) is trusted — the guard must never block a legitimate
    // download, whether or not a family was declared.
    let dir = tempdir().expect("tempdir creates");
    assert!(matches!(
        check_downloaded_model_family(Some("z-image".to_owned()), dir.path()),
        DownloadFamilyCheck::Proceed
    ));
    assert!(matches!(
        check_downloaded_model_family(None, dir.path()),
        DownloadFamilyCheck::Proceed
    ));
}

#[test]
fn download_family_check_flags_confident_mismatch() {
    // Weights that confidently detect as one family while the catalog declared
    // another are rejected (parity with model import).
    let dir = tempdir().expect("tempdir creates");
    write_safetensors_with_keys(
        &dir.path().join("model.safetensors"),
        &wan_video_safetensors_keys(),
    );
    match check_downloaded_model_family(Some("z-image".to_owned()), dir.path()) {
        DownloadFamilyCheck::Mismatch(mismatch) => {
            assert_eq!(mismatch.supplied, "z-image");
            assert_eq!(mismatch.detected, "wan-video");
        }
        other => panic!("expected a family mismatch, got {other:?}"),
    }
}

#[test]
fn download_family_check_proceeds_when_detection_matches_catalog() {
    let dir = tempdir().expect("tempdir creates");
    write_safetensors_with_keys(
        &dir.path().join("model.safetensors"),
        &wan_video_safetensors_keys(),
    );
    assert!(matches!(
        check_downloaded_model_family(Some("wan-video".to_owned()), dir.path()),
        DownloadFamilyCheck::Proceed
    ));
}

#[tokio::test]
async fn finalize_converted_dir_promotes_atomically_and_replaces_stale() {
    let temp = tempdir().expect("tempdir creates");
    let root = temp.path();
    let final_dir = root.join("mlx").join("wan_2_2");

    // A completed temp conversion sitting in its sibling staging dir.
    let temp_dir = root.join("mlx").join(".wan_2_2.converting-job1");
    std::fs::create_dir_all(&temp_dir).expect("temp dir");
    std::fs::write(temp_dir.join("config.json"), "{}").expect("config");
    std::fs::write(temp_dir.join("model.safetensors"), b"weights").expect("weights");

    // The canonical dir only appears after finalize, so a partial conversion can
    // never be picked up as a ready model.
    assert!(!final_dir.exists());
    finalize_converted_dir(&temp_dir, &final_dir)
        .await
        .expect("finalize");
    assert!(final_dir.join("config.json").is_file());
    assert!(final_dir.join("model.safetensors").is_file());
    assert!(!temp_dir.exists());

    // A re-conversion replaces a stale final dir wholesale.
    let stale_marker = final_dir.join("stale.txt");
    std::fs::write(&stale_marker, "old").expect("stale");
    let temp_dir2 = root.join("mlx").join(".wan_2_2.converting-job2");
    std::fs::create_dir_all(&temp_dir2).expect("temp dir 2");
    std::fs::write(temp_dir2.join("config.json"), "{}").expect("config 2");
    finalize_converted_dir(&temp_dir2, &final_dir)
        .await
        .expect("finalize 2");
    assert!(final_dir.join("config.json").is_file());
    assert!(!stale_marker.exists());
    assert!(!temp_dir2.exists());
}

/// sc-8837 (F-035): when a previously working install exists and the temp→final
/// promotion fails, the ORIGINAL install must survive — the crash-safe finalize moves
/// the stale install aside rather than destroying it first, then restores it on failure.
/// We inject the rename failure by pointing at a `temp_dir` that does not exist (the aside
/// move of the stale install succeeds, then the temp→final rename fails ENOENT).
#[tokio::test]
async fn finalize_converted_dir_restores_stale_install_on_rename_failure() {
    let temp = tempdir().expect("tempdir");
    let root = temp.path();
    let final_dir = root.join("mlx").join("model");
    std::fs::create_dir_all(&final_dir).expect("final dir");
    std::fs::write(final_dir.join("weights.safetensors"), b"OLD-BUT-WORKING")
        .expect("seed old install");

    // Intentionally never created, so the temp→final rename fails after the stale aside.
    let temp_dir = root.join("mlx").join(".model.converting-jobX");

    let error = finalize_converted_dir(&temp_dir, &final_dir)
        .await
        .expect_err("a missing temp dir must make the promotion fail");
    assert!(
        matches!(error, WorkerError::Io(_)),
        "expected an IO rename error, got {error:?}"
    );

    // The previously working model is left untouched (its contents survived).
    assert!(
        final_dir.join("weights.safetensors").is_file(),
        "the original install must survive a failed finalize"
    );
    assert_eq!(
        std::fs::read(final_dir.join("weights.safetensors")).expect("read restored"),
        b"OLD-BUT-WORKING",
        "restored install must have the original contents"
    );
    // No leftover backup dir next to the install.
    let leftovers: Vec<_> = std::fs::read_dir(final_dir.parent().expect("parent"))
        .expect("read parent")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains("finalize-backup"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "no backup dir should remain after restore, found {leftovers:?}"
    );
}

/// sc-8837: the happy path replaces an existing install with the new contents and
/// leaves no backup dir behind (crash-safe aside is cleaned up on success).
#[tokio::test]
async fn finalize_converted_dir_replaces_existing_install_leaves_no_backup() {
    let temp = tempdir().expect("tempdir");
    let root = temp.path();
    let final_dir = root.join("mlx").join("model");
    std::fs::create_dir_all(&final_dir).expect("final dir");
    std::fs::write(final_dir.join("weights.safetensors"), b"OLD").expect("seed old");
    let temp_dir = root.join("mlx").join(".model.converting-jobY");
    std::fs::create_dir_all(&temp_dir).expect("temp dir");
    std::fs::write(temp_dir.join("weights.safetensors"), b"NEW").expect("seed new");

    finalize_converted_dir(&temp_dir, &final_dir)
        .await
        .expect("finalize should succeed");

    assert_eq!(
        std::fs::read(final_dir.join("weights.safetensors")).expect("read new"),
        b"NEW",
        "final dir must hold the freshly converted contents"
    );
    assert!(
        !temp_dir.exists(),
        "temp dir must have been moved into place"
    );
    let leftovers: Vec<_> = std::fs::read_dir(final_dir.parent().expect("parent"))
        .expect("read parent")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains("finalize-backup"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "no backup dir should remain on success, found {leftovers:?}"
    );
}

/// sc-8837: the first-install case (no prior `final_dir`, nested parent) promotes the
/// temp dir into place without needing anything to move aside.
#[tokio::test]
async fn finalize_converted_dir_first_install_succeeds() {
    let temp = tempdir().expect("tempdir");
    let root = temp.path();
    let final_dir = root.join("mlx").join("installed").join("model");
    let temp_dir = root.join("mlx").join(".model.converting-jobZ");
    std::fs::create_dir_all(&temp_dir).expect("temp dir");
    std::fs::write(temp_dir.join("weights.safetensors"), b"FRESH").expect("seed fresh");

    assert!(!final_dir.exists());
    finalize_converted_dir(&temp_dir, &final_dir)
        .await
        .expect("first install should succeed");

    assert_eq!(
        std::fs::read(final_dir.join("weights.safetensors")).expect("read fresh"),
        b"FRESH",
        "first install must land the converted contents"
    );
    assert!(
        !temp_dir.exists(),
        "temp dir must have been moved into place"
    );
}

#[test]
fn download_progress_payload_matches_python_shape() {
    let payload = download_progress_payload(
        "owner/model",
        512 * 1024 * 1024,
        Some(1024 * 1024 * 1024),
        0,
        Duration::from_secs(2),
    );

    assert_eq!(payload.status.as_str(), "downloading");
    assert_eq!(payload.stage.as_str(), "downloading");
    assert_eq!(payload.progress.as_f64(), Some(0.525));
    assert!(payload.message.contains("512.0 MB of 1.0 GB"));
    assert!(payload.eta_seconds.is_some());
}

#[test]
fn pattern_filtering_and_download_dir_match_python_behavior() {
    assert!(allow_pattern_matches(
        "nested/model.safetensors",
        &["*.safetensors".to_owned()]
    ));
    assert!(!allow_pattern_matches(
        "nested/model.ckpt",
        &["*.safetensors".to_owned()]
    ));
    assert_eq!(safe_download_dir("owner/model name"), "owner__model__name");
    assert_eq!(safe_download_dir("///"), "download");
}

/// F-111 (sc-8913): a hostile job id must never let a person-track / frame-extract work dir
/// escape `temp_dir()`. The handlers build the dir as `temp_dir().join(format!("sw-…-{}",
/// safe_download_dir(&job.id)))`, so the sanitized id must always be a single path component
/// that stays confined under the temp root — no `/`, `\`, `..`, or absolute-root traversal.
#[test]
fn safe_download_dir_confines_hostile_job_ids_to_the_temp_root() {
    let temp_root = std::env::temp_dir();
    for hostile in [
        "../../etc/passwd",
        "/etc/passwd",
        "..\\..\\windows\\system32",
        "a/../../b",
        "....//....//x",
        "job\0id",
        "  ../secret  ",
    ] {
        let sanitized = safe_download_dir(hostile);
        // The sanitized id is a single normal path component (no separators, not `..`/`.`).
        assert!(
            !sanitized.contains('/')
                && !sanitized.contains('\\')
                && sanitized != ".."
                && sanitized != ".",
            "sanitized job id {sanitized:?} (from {hostile:?}) is not a single safe component"
        );
        // Joining it under the temp root yields a path whose only new component sits directly
        // under `temp_dir()` — it can never traverse above the temp root.
        let work_dir = temp_root.join(format!("sw-person-track-{sanitized}"));
        assert_eq!(
            work_dir.parent(),
            Some(temp_root.as_path()),
            "hostile job id {hostile:?} escaped the temp root: {work_dir:?}"
        );
        assert!(
            work_dir.starts_with(&temp_root),
            "hostile job id {hostile:?} produced a path outside the temp root: {work_dir:?}"
        );
    }
}

#[test]
fn huggingface_cache_paths_follow_hub_layout() {
    let root = tempdir().expect("temp dir creates");
    let path = super::huggingface_repo_cache_path(root.path(), "owner/model-name")
        .expect("cache path resolves");

    assert_eq!(
        path.file_name().and_then(|name| name.to_str()),
        Some("models--owner--model-name")
    );
}

#[test]
fn repo_slug_functions_match_cross_language_contract() {
    // story 1667: safe_download_dir is the worker-only repo->dir slug op pinned
    // by the shared repo_slugs.json contract. (safe_repo_dir_name moved to
    // sceneworks-core in sc-4279 and is contract-tested there, so it is no longer
    // re-asserted here.)
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rust_migration_contracts/repo_slugs.json");
    let contract: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read repo_slugs.json"))
            .expect("parse repo_slugs.json");
    let cases = contract["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "repo_slugs fixture has no cases");
    for case in cases {
        let repo = case["repo"].as_str().expect("repo string");
        assert_eq!(
            super::safe_download_dir(repo),
            case["safeDownloadDir"].as_str().expect("safeDownloadDir"),
            "safe_download_dir drift for {repo:?}"
        );
    }
}

#[test]
fn nvidia_smi_parsing_and_visible_device_filtering_match_python_worker() {
    let gpus = parse_nvidia_smi_gpus(
        "0, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887, 4096, 93791, 12\n\
             1, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887, 8192, 89695, 25\n",
    );

    assert_eq!(
        gpus.iter().map(|gpu| gpu.id.as_str()).collect::<Vec<_>>(),
        ["0", "1"]
    );
    assert_eq!(
        gpus[0].name,
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (97887 MB)"
    );
    assert!(gpus[1]
        .capabilities
        .iter()
        .any(|capability| capability.as_str() == "nvidia"));
    assert!(gpus[1]
        .capabilities
        .iter()
        .any(|capability| capability.as_str() == "placeholder"));
    assert_eq!(
        gpus[0].utilization,
        Some(WorkerUtilizationSnapshot {
            memory_total_mb: Some(97887),
            memory_used_mb: Some(4096),
            memory_free_mb: Some(93791),
            gpu_load_percent: Some(12.0),
        })
    );

    assert_eq!(visible_gpu_ids(None), None);
    assert_eq!(visible_gpu_ids(Some("all")), None);
    assert_eq!(visible_gpu_ids(Some("none")), Some(Vec::new()));
    assert_eq!(
        visible_gpu_ids(Some("0, GPU-abcd")),
        Some(vec!["0".to_owned(), "GPU-abcd".to_owned()])
    );
}

#[test]
fn auto_worker_ids_and_child_environment_match_python_supervisor() {
    assert_eq!(gpu_worker_id("worker-gpu-auto-0", "0"), "worker-gpu-auto-0");
    assert_eq!(gpu_worker_id("worker-gpu-auto-0", "1"), "worker-gpu-auto-1");
    assert_eq!(cpu_worker_id("worker-gpu-auto-0"), "worker-gpu-auto-cpu");

    let gpus = vec![fallback_gpu("0"), fallback_gpu("1")];
    let specs = auto_worker_specs("worker-gpu-auto-0", &gpus);
    assert_eq!(
        specs
            .iter()
            .map(|spec| spec.worker_id.as_str())
            .collect::<Vec<_>>(),
        [
            "worker-gpu-auto-0",
            "worker-gpu-auto-1",
            "worker-gpu-auto-cpu"
        ]
    );
    assert_eq!(
        specs
            .iter()
            .map(|spec| spec.gpu_id.as_str())
            .collect::<Vec<_>>(),
        ["0", "1", "cpu"]
    );

    let gpu_env = child_environment(&WorkerSpec {
        worker_id: "worker-gpu-auto-1".to_owned(),
        gpu_id: "1".to_owned(),
    });
    assert_eq!(gpu_env["SCENEWORKS_UTILITY_JOBS"], "0");
    assert_eq!(gpu_env["CUDA_VISIBLE_DEVICES"], "1");

    let cpu_env = child_environment(&WorkerSpec {
        worker_id: "worker-gpu-auto-cpu".to_owned(),
        gpu_id: "cpu".to_owned(),
    });
    assert_eq!(cpu_env["SCENEWORKS_UTILITY_JOBS"], "1");
    assert_eq!(cpu_env["CUDA_VISIBLE_DEVICES"], "");
}

#[test]
fn utility_worker_specs_scale_to_requested_count() {
    let single = utility_worker_specs("rust-utility-worker-0", 1);
    assert_eq!(
        single
            .iter()
            .map(|spec| spec.worker_id.as_str())
            .collect::<Vec<_>>(),
        ["rust-utility-worker-cpu"]
    );

    let pool = utility_worker_specs("rust-utility-worker-0", 4);
    assert_eq!(
        pool.iter()
            .map(|spec| spec.worker_id.as_str())
            .collect::<Vec<_>>(),
        [
            "rust-utility-worker-cpu",
            "rust-utility-worker-cpu-1",
            "rust-utility-worker-cpu-2",
            "rust-utility-worker-cpu-3",
        ]
    );
    assert!(pool.iter().all(|spec| spec.gpu_id == "cpu"));

    // A count of 0 must still yield a single worker rather than an empty pool.
    assert_eq!(utility_worker_specs("rust-utility-worker-0", 0).len(), 1);
}

#[test]
fn rust_cpu_capabilities_do_not_claim_gpu_generation_jobs() {
    let cpu_capabilities = worker_capabilities_with_utility(&cpu_gpu(), true);

    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "model_download"));
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "placeholder"));
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "timeline_export"));
    // The CPU utility worker advertises only the procedural *preview*
    // capabilities; real detection/tracking route to the Python GPU worker.
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_detect_preview"));
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_track_preview"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_detect"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_track"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "image_generate"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "video_generate"));

    let gpu_capabilities = worker_capabilities_with_utility(&fallback_gpu("0"), false);
    assert!(gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "gpu"));
    assert!(gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "placeholder"));
    assert!(!gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "model_download"));
    assert!(!gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "image_generate"));
}

/// The Apple-Silicon MLX GPU worker (epic 3018) advertises `image_generate`,
/// `image_edit` (sc-3513), + `video_generate` so the API routes generation/editing to
/// it, but it must NOT pick up CPU utility jobs (those stay on the CPU worker) — the
/// inverse of the CPU-worker contract above. `video_generate` lands with the video
/// runtime (sc-3033).
#[cfg(target_os = "macos")]
#[test]
fn mlx_gpu_advertises_generation_capabilities_only() {
    let mlx = mlx_gpu(&crate::Settings::from_env());
    assert_eq!(mlx.id, "mlx");
    let capabilities = worker_capabilities_with_utility(&mlx, true);
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "image_generate"));
    // Plain Image Edit (sc-3513): without this the API's worker_supports_job would
    // reject an `image_edit` claim and the job would silently fall back to torch.
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "image_edit"));
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "video_generate"));
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "training_caption"));
    // Real, model-backed person detect/track are ported to the MLX worker (sc-3709): the
    // worker advertises the non-preview capabilities so the API routes real jobs here.
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_detect"));
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_track"));
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "gpu"));
    // No CPU utility capabilities, even with utility jobs enabled — only the CPU
    // worker (which carries `Cpu`) gets those extended onto it.
    for utility in [
        "model_download",
        "model_import",
        "timeline_export",
        "cpu",
        "placeholder",
    ] {
        assert!(
            !capabilities
                .iter()
                .any(|capability| capability.as_str() == utility),
            "MLX GPU worker should not advertise utility capability {utility}"
        );
    }
}

/// sc-3723 acceptance gate: with default settings (mlx enabled) and ALL provider crates
/// linked on macOS, the registry-DERIVED advertisement must equal exactly today's hardcoded
/// MLX capability set (order-independent). This is the invariant that lets the dispatch table
/// move + the flags become descriptor-derived without changing what the worker advertises.
#[cfg(target_os = "macos")]
#[test]
fn mlx_gpu_capability_set_matches_expected_full_set() {
    use sceneworks_core::contracts::WorkerCapability;
    use std::collections::BTreeSet;
    let mlx = mlx_gpu(&crate::Settings::from_env());
    let actual: BTreeSet<_> = mlx.capabilities.iter().cloned().collect();
    let expected: BTreeSet<_> = [
        // seed
        WorkerCapability::Gpu,
        // 7 registry-derived
        WorkerCapability::ImageGenerate,
        WorkerCapability::VideoGenerate,
        WorkerCapability::LoraTrain,
        WorkerCapability::LoraTrainExecute,
        WorkerCapability::TrainingCaption,
        // sc-5552: the native MLX `prompt_refine` TextLlm provider (mlx-gen-prompt-refine) is
        // force-linked on macOS, so the registry now derives PromptRefine from its descriptor.
        WorkerCapability::PromptRefine,
        // sc-6535: mlx-gen-clip registers the CLIP `clip_vit_l14` ImageEmbedder (force-linked in
        // dataset_analysis_jobs.rs), so the registry derives DatasetAnalysis from its descriptor.
        WorkerCapability::DatasetAnalysis,
        // carve-outs
        WorkerCapability::ImageEdit,
        WorkerCapability::ImageDetail,
        WorkerCapability::ImageVqa,
        WorkerCapability::ImageInterleave,
        WorkerCapability::VideoExtend,
        WorkerCapability::VideoBridge,
        WorkerCapability::PersonReplace,
        WorkerCapability::PoseDetect,
        WorkerCapability::KpsExtract,
        // sc-6538: the native SCRFD+ArcFace face stack (mlx-gen-face) is hardcoded in mlx_gpu (no
        // gen-core registry for FaceEmbedder), so DatasetFaceAnalysis is advertised on Mac like
        // KpsExtract.
        WorkerCapability::DatasetFaceAnalysis,
        // sc-4415: on-demand face-likeness compare — same hardcoded-in-mlx_gpu native face stack as
        // DatasetFaceAnalysis (no gen-core registry for FaceEmbedder), advertised on Mac like KpsExtract.
        WorkerCapability::FaceLikenessCompare,
        WorkerCapability::ImageUpscale,
        // sc-6539: Dataset Doctor one-tap upscale — reuses the Real-ESRGAN engine, advertised
        // wherever image_upscale is.
        WorkerCapability::DatasetUpscale,
        // sc-6105: smart-select segmentation (native-MLX SAM3 box-prompt) — Mac-only, advertised
        // only here so an `image_segment` job routes to the MLX worker by construction.
        WorkerCapability::ImageSegment,
        WorkerCapability::VideoUpscale,
        WorkerCapability::PersonDetect,
        WorkerCapability::PersonTrack,
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "registry-derived MLX capability set drifted from the expected full set"
    );
}

/// sc-3723: every MODEL_TABLE row resolves through the registry-joined `mlx_model` lookup
/// (its engine id is registered by a linked provider crate), and the descriptor-derived
/// guidance/negative-prompt flags match the pre-deletion hardcoded values — proving the two
/// removed row flags were faithfully replaced by the engine's own advertised surface.
#[cfg(target_os = "macos")]
#[test]
fn model_table_rows_resolve_and_flags_match_descriptor() {
    use crate::engines::{mlx_model, MODEL_TABLE};
    // (sceneworks_id, supports_guidance, supports_negative_prompt) — the exact pre-sc-3723
    // values that used to live on each MlxModel row.
    let expected: &[(&str, bool, bool)] = &[
        ("z_image_turbo", false, false),
        // Base (non-distilled) Z-Image (sc-8320): the undistilled foundation model is full real CFG —
        // supports_guidance=true + supports_negative_prompt=true (vs Turbo's CFG-free distill).
        ("z_image", true, true),
        // Ideogram 4 (epic 4725): asymmetric-CFG guidance (supports_guidance=true) with no user
        // negative prompt (the "negative" is a fixed unconditional DiT, not a prompt).
        ("ideogram_4", true, false),
        // Ideogram 4 Turbo (mlx-gen #488): CFG-free single-DiT — the descriptor drops guidance
        // (supports_guidance=false), no negative prompt. Requires the mlx-gen pin to include the
        // `ideogram_4_turbo` engine (PR #489) for this row to resolve through the registry.
        ("ideogram_4_turbo", false, false),
        ("z_image_edit", false, false),
        ("flux_schnell", false, false),
        ("flux_dev", true, false),
        ("qwen_image", true, true),
        ("qwen_image_edit", true, true),
        ("qwen_image_edit_2509", true, true),
        ("qwen_image_edit_2511", true, true),
        // sc-3723 finding: the lightning variant shares the `qwen_image_edit` engine id, whose
        // descriptor advertises supports_negative_prompt=true. The old row hardcoded `false`
        // (the CFG-off recipe), but the engine itself drops the negative branch under the
        // `lightning` sampler (model_edit.rs `neg = None` when is_lightning), so the
        // descriptor-derived `true` is behavior-equivalent — the CFG-off recipe is
        // engine-enforced, not a model capability the worker has to suppress.
        ("qwen_image_edit_2511_lightning", true, true),
        ("flux2_klein_9b", true, false),
        ("flux2_klein_9b_kv", true, false),
        ("flux2_klein_9b_true_v2", true, false),
        // FLUX.2-dev (epic 5914 / sc-5921): its own `flux2_dev` engine id, embedded distilled
        // guidance (supports_guidance=true) with no negative prompt / true-CFG.
        ("flux2_dev", true, false),
        ("sdxl", true, true),
        ("realvisxl", true, true),
        // RealVisXL Lightning (sc-6075): shares the `sdxl` engine id, whose descriptor advertises
        // guidance + negative prompt (true, true). The few-step recipe runs CFG-off (guidance 1.0,
        // negative inert) via the worker-forced `lightning` sampler, but that's a recipe default,
        // not a capability the descriptor drops — so the descriptor-derived flags stay (true, true).
        ("realvisxl_lightning", true, true),
        ("kolors", true, true),
        ("chroma1_hd", false, true),
        ("chroma1_base", false, true),
        ("chroma1_flash", false, true),
        ("sensenova_u1_8b", true, false),
        ("sensenova_u1_8b_fast", true, false),
        // Lens / Lens-Turbo (epic 3164 / sc-5105): the `mlx-gen-lens` descriptor advertises the
        // norm-rescaled CFG path (`supports_guidance=true`) + a negative prompt
        // (`supports_negative_prompt=true`) — a standard guidance family (NOT true-CFG), so the worker
        // forwards the CFG scale via `guidance`. Turbo simply defaults guidance to 1.0 (≈ no CFG).
        ("lens", true, true),
        ("lens_turbo", true, true),
        // Bernini still-image companion (epic 4699 / sc-5424): the image-typed `bernini_image` id
        // maps to the SAME `bernini` engine the video id uses (`Modality::Both`). Standard guidance
        // family — `supports_guidance=true` (omega_txt) + `supports_negative_prompt=true`.
        ("bernini_image", true, true),
        // Boogu-Image-0.1 (epic 6387 / sc-6399): Base + Edit are true-CFG (supports_guidance=true)
        // with no user negative prompt (the CFG-negative is a fixed empty/drop instruction). Turbo is
        // the DMD few-step, CFG-free distill (supports_guidance=false). None take a negative prompt.
        ("boogu_image", true, false),
        ("boogu_image_turbo", false, false),
        ("boogu_image_edit", true, false),
        // Krea 2 Turbo (epic 7565 / sc-7572): TDM-distilled few-step, CFG-free
        // (supports_guidance=false) with no user negative prompt (supports_negative_prompt=false) — the
        // z_image_turbo / boogu_image_turbo distilled-turbo pattern.
        ("krea_2_turbo", false, false),
        // SD3.5 Large (epic 7841 / sc-7871): true-CFG MMDiT flagship — supports_guidance=true +
        // supports_negative_prompt=true (the `sd3_5_large` descriptor advertises supports_true_cfg).
        ("sd3_5_large", true, true),
        // SD3.5 Large Turbo (epic 7841 / sc-7871): the ADD-distilled few-step, CFG-off sibling — the
        // `sd3_5_large_turbo` descriptor drops guidance + negative prompt (supports_guidance=false,
        // supports_negative_prompt=false), the distilled-turbo pattern.
        ("sd3_5_large_turbo", false, false),
        // SD3.5 Medium (epic 7841 / sc-7869 M3, wired sc-7871): the MMDiT-X true-CFG variant —
        // supports_guidance=true + supports_negative_prompt=true (the `sd3_5_medium` descriptor advertises
        // supports_true_cfg, same as Large; only the transformer + step/guidance recipe differ).
        ("sd3_5_medium", true, true),
        // SANA 1600M (epic 8485 / sc-8489): true-CFG Linear-DiT — supports_guidance=true +
        // supports_negative_prompt=true (the `sana_1600m` descriptor advertises supports_true_cfg).
        ("sana_1600m", true, true),
        // SANA-Sprint 1.6B (epic 8485 / sc-8490): the CFG-free few-step distillation — the guidance
        // scalar is folded into the trunk via a guidance-embedding (supports_guidance=true) but there is
        // no cond/uncond combine, so the descriptor advertises supports_true_cfg=false +
        // supports_negative_prompt=false (the distilled-turbo "guidance is an embedding, no negative"
        // shape; cf. boogu_image_turbo's CFG-free-without-negative pattern).
        ("sana_sprint_1600m", true, false),
    ];
    // Every row is covered by the expectation table (no row added without a flag pair here).
    assert_eq!(MODEL_TABLE.len(), expected.len());
    for (id, guidance, negative) in expected {
        let m = mlx_model(id).unwrap_or_else(|| panic!("{id} resolves through the registry"));
        assert_eq!(
            m.supports_guidance(),
            *guidance,
            "{id} supports_guidance descriptor drift"
        );
        assert_eq!(
            m.supports_negative_prompt(),
            *negative,
            "{id} supports_negative_prompt descriptor drift"
        );
        assert_eq!(m.backend(), "mlx", "{id} backend");
    }
}

/// sc-7875 (SD3.5 S6, MLX-path validation boundary): the three SD3.5 builtin-manifest entries gate
/// correctly at the catalog layer — `macOnly: false` (cross-platform now that the candle off-Mac lane
/// is wired, sc-7880/epic 7982; availability is driven by the routing tables, not this flag),
/// `capabilities == ["text_to_image"]` only (edit/reference rejected), the family `sd3`, the gated
/// stabilityai/* download with `gated: true` + `credentialHost: huggingface.co`, and the per-tier
/// `mlx.minMemoryGb` (Large/Turbo 64, Medium 56) that drives the memory-eligibility gate. Parses the
/// embedded builtin manifest (the exact bytes shipped) so manifest drift on any of these eligibility
/// levers fails CI without a real download. (The descriptor-derived guidance/negative/backend surface
/// is covered by `model_table_rows_resolve_and_flags_match_descriptor`; the credential-host derivation
/// by the rust-api `gated_credential_tests`; this is the catalog-eligibility counterpart.)
#[test]
fn sd3_5_manifest_entries_gate_correctly() {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(raw)).expect("builtin models parses as JSON");
    let models = manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("models array");

    // (id, expected minMemoryGb) — Large/Turbo flagship-tier 64, Medium light-tier 56
    // (S6 worker-lane footprint ~52 GB Q8 / ~48.6 GB Q4 + headroom, below the 64 flagship tier).
    let expected: &[(&str, u64)] = &[
        ("sd3_5_large", 64),
        ("sd3_5_large_turbo", 64),
        ("sd3_5_medium", 56),
    ];
    for (id, min_mem) in expected {
        let entry = models
            .iter()
            .find(|m| m.get("id").and_then(Value::as_str) == Some(id))
            .unwrap_or_else(|| panic!("{id} present in builtin.models.jsonc"));

        assert_eq!(
            entry.get("family").and_then(Value::as_str),
            Some("sd3"),
            "{id} family"
        );
        // Cross-platform: the candle off-Mac lane is now wired (sc-7880, epic 7982), so `macOnly` is a
        // no-op label flipped to false (mirroring krea/flux2_dev) — availability is driven by the
        // routing tables (`MLX_ROUTED_MODELS` / `CANDLE_ROUTED_MODELS`), not this flag.
        assert_eq!(
            entry.get("macOnly").and_then(Value::as_bool),
            Some(false),
            "{id} macOnly"
        );
        // Capability gate: text_to_image ONLY — edit/reference are rejected (no img2img/inpaint path).
        let caps: Vec<&str> = entry
            .get("capabilities")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert_eq!(caps, vec!["text_to_image"], "{id} capabilities");
        // UN-gated SceneWorks MLX re-host (sc-8513, epic 8506): the pre-built quant-matrix turnkey
        // carries the Stability AI Community License + "Powered by Stability AI" NOTICE, so no HF
        // credential host / license-click. (Was the gated stabilityai/* source + install-time convert.)
        assert_eq!(
            entry.get("gated").and_then(Value::as_bool),
            Some(false),
            "{id} gated"
        );
        assert_eq!(
            entry.get("credentialHost"),
            None,
            "{id} credentialHost (dropped on re-host)"
        );
        // Every tier download is the SceneWorks re-host (q4 default + q8 + bf16), and the default
        // (first) entry is q4.
        let downloads = entry
            .get("downloads")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("{id} downloads array"));
        for dl in downloads {
            let repo = dl.get("repo").and_then(Value::as_str).unwrap_or("");
            assert!(
                repo.starts_with("SceneWorks/sd3.5-"),
                "{id} tier downloads from the SceneWorks re-host, got {repo:?}"
            );
        }
        let first_files: Vec<&str> = downloads
            .first()
            .and_then(|d| d.get("files"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert_eq!(first_files, vec!["q4/*"], "{id} default tier is q4");
        // Per-tier memory-eligibility gate (drives the Studio admit/hide-by-available-memory).
        assert_eq!(
            entry
                .get("mlx")
                .and_then(|m| m.get("minMemoryGb"))
                .and_then(Value::as_u64),
            Some(*min_mem),
            "{id} mlx.minMemoryGb"
        );
        // sd3 LoRA family declared (S5): the picker offers ONLY sd3-family LoRAs (an empty list would
        // match every LoRA, sc-1927).
        let lora_families: Vec<&str> = entry
            .get("loraCompatibility")
            .and_then(|c| c.get("families"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert_eq!(
            lora_families,
            vec!["sd3"],
            "{id} loraCompatibility.families"
        );
    }
}

/// sc-8489 (SANA Phase B2): the SANA builtin-manifest entry gates correctly at the catalog layer —
/// family `sana`, `capabilities == ["text_to_image"]` only (edit/reference rejected), the UN-gated
/// `SceneWorks/Sana_1600M_1024px_mlx` MLX re-host (NOT gated — the mirror carries the NVIDIA
/// non-commercial NOTICE), dense bf16 (NO `mlx.quantize` — the load path rejects a quant), the
/// `mlx.minMemoryGb` memory-eligibility lever, the sana LoRA family, and the NVIDIA non-commercial
/// notice surfaced in the UI description. Parses the embedded builtin manifest (the exact bytes
/// shipped) so manifest drift on any of these levers fails CI without a real download. The
/// descriptor-derived guidance/negative/backend surface is covered by
/// `model_table_rows_resolve_and_flags_match_descriptor`.
#[test]
fn sana_manifest_entry_gates_correctly() {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(raw)).expect("builtin models parses as JSON");
    let models = manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("models array");
    let entry = models
        .iter()
        .find(|m| m.get("id").and_then(Value::as_str) == Some("sana_1600m"))
        .expect("sana_1600m present in builtin.models.jsonc");

    assert_eq!(
        entry.get("family").and_then(Value::as_str),
        Some("sana"),
        "sana family"
    );
    // Capability gate: text_to_image ONLY — edit/reference are rejected (base SANA is plain t2i).
    let caps: Vec<&str> = entry
        .get("capabilities")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(caps, vec!["text_to_image"], "sana capabilities");
    // UN-gated SceneWorks/* MLX re-host (the mirror carries the NVIDIA non-commercial NOTICE; OK to
    // ship un-gated with notice, the Krea/Boogu precedent) — so NO `gated: true`.
    assert_ne!(
        entry.get("gated").and_then(Value::as_bool),
        Some(true),
        "sana is un-gated (re-host carries the notice)"
    );
    let repo = entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|d| d.first())
        .and_then(|d| d.get("repo"))
        .and_then(Value::as_str)
        .unwrap_or("");
    assert_eq!(
        repo, "SceneWorks/Sana_1600M_1024px_mlx",
        "sana downloads from the un-gated SceneWorks/* MLX mirror, got {repo:?}"
    );
    // Dense bf16: NO `mlx.quantize` (the SANA `load` rejects any `spec.quantize`; the worker resolves
    // no quant for it via the `supports_quant()` gate). minMemoryGb drives the admit/hide gate.
    let mlx = entry.get("mlx").expect("sana mlx block");
    assert!(
        mlx.get("quantize").is_none(),
        "sana ships dense bf16 — no mlx.quantize"
    );
    assert!(
        mlx.get("minMemoryGb").and_then(Value::as_u64).is_some(),
        "sana mlx.minMemoryGb present"
    );
    // sana LoRA family declared (reserved; no SANA LoRA wired yet) — an empty list would match every
    // LoRA (sc-1927).
    let lora_families: Vec<&str> = entry
        .get("loraCompatibility")
        .and_then(|c| c.get("families"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        lora_families,
        vec!["sana"],
        "sana loraCompatibility.families"
    );
    // NVIDIA non-commercial notice surfaced in the UI description (the gated-with-notice carrier).
    let desc = entry
        .get("ui")
        .and_then(|u| u.get("description"))
        .and_then(Value::as_str)
        .unwrap_or("");
    assert!(
        desc.contains("NVIDIA") && desc.to_lowercase().contains("non-commercial"),
        "sana UI description carries the NVIDIA non-commercial notice"
    );
}

/// sc-8490: the SANA-Sprint builtin entry gates exactly like base SANA — `sana` family, text_to_image
/// only (CFG-free few-step distillation, no edit/reference surface), un-gated `SceneWorks/*` MLX
/// re-host carrying the NVIDIA non-commercial notice, dense bf16 (no mlx.quantize), and the SANA LoRA
/// family reserved. The few-step default (2 steps) is asserted so a manifest drift to the base 20-step
/// loop fails CI. Descriptor-derived guidance/negative/backend flags are covered by
/// `model_table_rows_resolve_and_flags_match_descriptor`.
#[test]
fn sana_sprint_manifest_entry_gates_correctly() {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(raw)).expect("builtin models parses as JSON");
    let models = manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("models array");
    let entry = models
        .iter()
        .find(|m| m.get("id").and_then(Value::as_str) == Some("sana_sprint_1600m"))
        .expect("sana_sprint_1600m present in builtin.models.jsonc");

    assert_eq!(
        entry.get("family").and_then(Value::as_str),
        Some("sana"),
        "sana-sprint family"
    );
    // Capability gate: text_to_image ONLY — edit/reference are rejected (Sprint is plain few-step t2i).
    let caps: Vec<&str> = entry
        .get("capabilities")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(caps, vec!["text_to_image"], "sana-sprint capabilities");
    // UN-gated SceneWorks/* MLX re-host (the mirror carries the NVIDIA non-commercial NOTICE) — no `gated`.
    assert_ne!(
        entry.get("gated").and_then(Value::as_bool),
        Some(true),
        "sana-sprint is un-gated (re-host carries the notice)"
    );
    let repo = entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|d| d.first())
        .and_then(|d| d.get("repo"))
        .and_then(Value::as_str)
        .unwrap_or("");
    assert_eq!(
        repo, "SceneWorks/Sana_Sprint_1.6B_1024px_mlx",
        "sana-sprint downloads from the un-gated SceneWorks/* MLX mirror, got {repo:?}"
    );
    // Few-step distillation: default steps = 2 (a drift to base SANA's 20-step loop fails here).
    assert_eq!(
        entry
            .get("defaults")
            .and_then(|d| d.get("steps"))
            .and_then(Value::as_u64),
        Some(2),
        "sana-sprint is a few-step (2-step) distillation"
    );
    // Dense bf16: NO `mlx.quantize`.
    let mlx = entry.get("mlx").expect("sana-sprint mlx block");
    assert!(
        mlx.get("quantize").is_none(),
        "sana-sprint ships dense bf16 — no mlx.quantize"
    );
    assert!(
        mlx.get("minMemoryGb").and_then(Value::as_u64).is_some(),
        "sana-sprint mlx.minMemoryGb present"
    );
    // sana LoRA family declared (reserved; no SANA LoRA wired yet).
    let lora_families: Vec<&str> = entry
        .get("loraCompatibility")
        .and_then(|c| c.get("families"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        lora_families,
        vec!["sana"],
        "sana-sprint loraCompatibility.families"
    );
    // NVIDIA non-commercial notice surfaced in the UI description (the gated-with-notice carrier).
    let desc = entry
        .get("ui")
        .and_then(|u| u.get("description"))
        .and_then(Value::as_str)
        .unwrap_or("");
    assert!(
        desc.contains("NVIDIA") && desc.to_lowercase().contains("non-commercial"),
        "sana-sprint UI description carries the NVIDIA non-commercial notice"
    );
}

/// sc-3513: the worker's `JobType::ImageEdit` dispatch arm delegates to
/// `run_image_generate_job` — the engine keys edits on payload model+mode, not job
/// type. Feeding an `image_edit`-typed job into the handler proves it reaches the image
/// pipeline (stopping at the payload's projectId guard) rather than the `run_utility_job`
/// "unsupported job type" default — i.e. plain Image Edit is genuinely handled. The
/// handler never reads `job_type`, so a missing projectId is its first stop (no network).
#[tokio::test]
async fn image_edit_job_dispatches_to_image_generate_handler() {
    let settings = test_settings("http://127.0.0.1".to_owned(), None);
    let api = ApiClient::new(&settings);
    let job: JobSnapshot = serde_json::from_value(json!({
        "id": "job-image-edit-1",
        "type": "image_edit",
        "status": "preparing",
        "projectId": null,
        "projectName": null,
        "payload": { "model": "qwen_image_edit_2511", "mode": "edit_image" },
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": null,
        "progress": 0,
        "stage": "preparing",
        "message": "",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": false,
        "createdAt": "2026-06-07T00:00:00Z",
        "updatedAt": "2026-06-07T00:00:00Z",
        "startedAt": null,
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": null
    }))
    .expect("image_edit job snapshot deserializes");

    let error =
        super::image_jobs::run_image_generate_job(&api, &settings, &reqwest::Client::new(), &job)
            .await
            .expect_err("missing projectId is rejected by the image handler");
    assert!(
        matches!(&error, WorkerError::InvalidPayload(message) if message.contains("projectId")),
        "expected a projectId payload error from the image handler, got {error:?}",
    );
}

#[tokio::test]
async fn supervisor_restarts_exited_children_with_backoff_state() {
    let settings = test_settings("http://127.0.0.1".to_owned(), None);
    let spec = WorkerSpec {
        worker_id: "worker-gpu-auto-0".to_owned(),
        gpu_id: "0".to_owned(),
    };
    let mut exited = spawn_exit_child();
    for _ in 0..20 {
        if exited.try_wait().expect("child status checks").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let mut children = HashMap::from([(
        spec.worker_id.clone(),
        SupervisedChild {
            spec,
            process: exited,
            restart_attempt: 0,
            spawned_at: std::time::Instant::now(),
            next_restart_at: None,
        },
    )]);
    let spawns = std::cell::Cell::new(0_u32);
    let mut spawner = |_settings: &_, _spec: &WorkerSpec| {
        spawns.set(spawns.get() + 1);
        Ok(spawn_sleep_child())
    };

    // Detection tick: the exit is reaped and a backoff deadline is stamped, but the
    // child is not respawned yet because its backoff has not elapsed (sc-8899).
    let t0 = std::time::Instant::now();
    restart_exited_children_at(&settings, &mut children, &mut spawner, t0)
        .await
        .expect("exit is detected");
    assert_eq!(
        spawns.get(),
        0,
        "detection tick does not respawn before backoff"
    );
    {
        let child = children
            .get("worker-gpu-auto-0")
            .expect("exited child stays tracked while backing off");
        assert_eq!(child.restart_attempt, 1);
        assert!(
            child.next_restart_at.is_some(),
            "a backoff deadline is stamped on the exited child"
        );
    }

    // Restart tick past the backoff deadline: the child is respawned exactly once.
    restart_exited_children_at(
        &settings,
        &mut children,
        &mut spawner,
        t0 + Duration::from_secs(30),
    )
    .await
    .expect("child restarts once its backoff elapses");

    assert_eq!(spawns.get(), 1);
    let child = children
        .get_mut("worker-gpu-auto-0")
        .expect("restarted child is tracked");
    assert_eq!(child.restart_attempt, 1);
    assert!(
        child.next_restart_at.is_none(),
        "the backoff deadline clears once the child is respawned"
    );
    assert!(child
        .process
        .try_wait()
        .expect("child status checks")
        .is_none());
    let _ = child.process.start_kill();
    let _ = child.process.wait().await;
}

/// sc-8899 / F-097: one child's restart backoff must not stall the whole
/// supervision tick. A crash-looping child with a long backoff and a healthy
/// sibling that just crashed are handled independently — the sibling restarts on
/// the next eligible tick while the still-backing-off child simply waits its turn.
#[tokio::test]
async fn supervisor_backoff_on_one_child_does_not_stall_another() {
    let settings = test_settings("http://127.0.0.1".to_owned(), None);

    // A crash-looping child mid-backoff: already reaped, with a deadline 30 s out.
    let mut looping = spawn_exit_child();
    for _ in 0..20 {
        if looping.try_wait().expect("child status checks").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let t0 = std::time::Instant::now();
    let looping_child = SupervisedChild {
        spec: WorkerSpec {
            worker_id: "worker-gpu-auto-0".to_owned(),
            gpu_id: "0".to_owned(),
        },
        process: looping,
        restart_attempt: 6,
        spawned_at: t0,
        next_restart_at: Some(t0 + Duration::from_secs(30)),
    };

    // A healthy sibling that already crashed and whose short backoff is now due.
    let mut sibling = spawn_exit_child();
    for _ in 0..20 {
        if sibling.try_wait().expect("child status checks").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let sibling_child = SupervisedChild {
        spec: WorkerSpec {
            worker_id: "worker-gpu-auto-1".to_owned(),
            gpu_id: "1".to_owned(),
        },
        process: sibling,
        restart_attempt: 1,
        spawned_at: t0,
        next_restart_at: Some(t0 + Duration::from_secs(1)),
    };

    let mut children = HashMap::from([
        ("worker-gpu-auto-0".to_owned(), looping_child),
        ("worker-gpu-auto-1".to_owned(), sibling_child),
    ]);
    let mut restarted = Vec::new();
    let mut spawner = |_settings: &_, spec: &WorkerSpec| {
        restarted.push(spec.worker_id.clone());
        Ok(spawn_sleep_child())
    };

    // A single tick at t0 + 5 s: the sibling's 1 s backoff is due while the looping
    // child's 30 s backoff is not. The old inline-sleep design would have blocked the
    // whole tick on whichever child was handled first; the per-child deadline model
    // restarts only the due sibling and leaves the looping child untouched (sc-8899).
    restart_exited_children_at(
        &settings,
        &mut children,
        &mut spawner,
        t0 + Duration::from_secs(5),
    )
    .await
    .expect("tick completes without stalling on the backing-off child");

    assert_eq!(
        restarted,
        vec!["worker-gpu-auto-1".to_owned()],
        "the due sibling restarts while the looping child is still backing off"
    );
    assert!(
        children["worker-gpu-auto-0"].next_restart_at.is_some(),
        "the looping child keeps its unexpired backoff deadline"
    );
    assert!(
        children["worker-gpu-auto-1"].next_restart_at.is_none(),
        "the restarted sibling has its deadline cleared"
    );

    for child in children.values_mut() {
        let _ = child.process.start_kill();
        let _ = child.process.wait().await;
    }
}

/// sc-4282 / F-MLXW-20: a child that ran healthily past the reset threshold
/// before exiting starts its restart backoff fresh, rather than carrying a
/// counter that has saturated upward over many widely-spaced crashes.
#[tokio::test]
async fn supervisor_resets_backoff_after_a_healthy_run() {
    let settings = test_settings("http://127.0.0.1".to_owned(), None);
    let spec = WorkerSpec {
        worker_id: "worker-gpu-auto-0".to_owned(),
        gpu_id: "0".to_owned(),
    };
    let mut exited = spawn_exit_child();
    for _ in 0..20 {
        if exited.try_wait().expect("child status checks").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // The counter has ratcheted up over time, but this run stayed alive well past
    // the healthy-uptime threshold (spawned > 6 minutes ago).
    let mut children = HashMap::from([(
        spec.worker_id.clone(),
        SupervisedChild {
            spec,
            process: exited,
            restart_attempt: 7,
            spawned_at: std::time::Instant::now()
                .checked_sub(Duration::from_secs(360))
                .expect("monotonic clock backdates 6 minutes"),
            next_restart_at: None,
        },
    )]);

    // The backoff reset happens when the exit is detected, so a single detection
    // tick is enough to observe it (the respawn itself waits for the backoff).
    restart_exited_children_with_spawner(&settings, &mut children, |_settings, _spec| {
        Ok(spawn_sleep_child())
    })
    .await
    .expect("exit is detected");

    let child = children
        .get_mut("worker-gpu-auto-0")
        .expect("exited child stays tracked while backing off");
    // Reset to 0 on the healthy run, then advanced once for this restart.
    assert_eq!(
        child.restart_attempt, 1,
        "a healthy run resets the backoff counter"
    );
    let _ = child.process.start_kill();
    let _ = child.process.wait().await;
}

/// sc-4881: a child reaped after an uncatchable signal (here SIGKILL, the OOM
/// killer's weapon) is attributed to that signal, while a clean exit reports none.
/// This is the only layer that can observe the death — it's uncatchable in the
/// dying child itself.
#[cfg(unix)]
#[tokio::test]
async fn terminating_signal_distinguishes_signal_death_from_clean_exit() {
    let mut child = spawn_sleep_child();
    let pid = child.id().expect("child has a pid");
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(pid as i32),
        nix::sys::signal::Signal::SIGKILL,
    )
    .expect("SIGKILL delivered");
    let status = child.wait().await.expect("killed child reaped");
    assert_eq!(terminating_signal(&status), Some(9));

    let mut clean = spawn_exit_child();
    let status = clean.wait().await.expect("clean child reaped");
    assert_eq!(terminating_signal(&status), None);
}

#[test]
fn child_died_abnormally_reports_signals_and_non_zero_exits_not_clean_exits() {
    // sc-6320: the supervisor attributes a real FAILURE for an uncatchable signal
    // death OR a non-zero self-exit (e.g. a Rust panic → 101), but a clean exit-0
    // is graceful and must report nothing (else a normal worker shutdown would
    // spuriously fail its job).
    assert!(child_died_abnormally(Some(9), None), "SIGKILL is abnormal");
    assert!(
        child_died_abnormally(None, Some(101)),
        "a panic exit (101) is abnormal"
    );
    assert!(
        child_died_abnormally(None, Some(1)),
        "any non-zero exit is abnormal"
    );
    assert!(
        !child_died_abnormally(None, Some(0)),
        "a clean exit-0 is graceful, not reported"
    );
}

#[tokio::test]
async fn writes_model_install_marker_with_expected_keys() {
    let temp = tempdir().expect("tempdir creates");
    let mut payload = serde_json::Map::new();
    payload.insert("modelId".to_owned(), json!("base-model"));
    payload.insert("modelName".to_owned(), json!("Base Model"));

    write_model_install_marker(temp.path(), &payload, "owner/model", "job-1")
        .await
        .expect("marker writes");

    let marker_path = temp.path().join(INSTALL_MARKER);
    let marker: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(marker_path).await.unwrap()).unwrap();
    assert_eq!(marker["repo"], "owner/model");
    assert_eq!(marker["modelId"], "base-model");
    assert_eq!(marker["modelName"], "Base Model");
    assert_eq!(marker["jobId"], "job-1");
    assert!(marker["completedAt"].as_str().is_some());
}

#[tokio::test]
async fn lora_file_and_directory_import_preserve_copy_semantics() {
    let temp = tempdir().expect("tempdir creates");
    let source_file = temp.path().join("mira.safetensors");
    tokio::fs::write(&source_file, b"lora").await.unwrap();
    let file_target = temp.path().join("file-target");

    copy_lora_source(&source_file, &file_target).await.unwrap();

    assert_eq!(
        tokio::fs::read(file_target.join("mira.safetensors"))
            .await
            .unwrap(),
        b"lora"
    );

    let source_dir = temp.path().join("source-dir");
    tokio::fs::create_dir_all(source_dir.join("nested"))
        .await
        .unwrap();
    tokio::fs::write(source_dir.join("nested/adapter.safetensors"), b"adapter")
        .await
        .unwrap();
    let dir_target = temp.path().join("dir-target");

    copy_lora_source(&source_dir, &dir_target).await.unwrap();

    assert_eq!(
        tokio::fs::read(dir_target.join("nested/adapter.safetensors"))
            .await
            .unwrap(),
        b"adapter"
    );
}

#[tokio::test]
async fn uploaded_lora_source_cleanup_removes_staged_file_and_parent() {
    let temp = tempdir().expect("tempdir creates");
    let mut settings = test_settings("http://127.0.0.1:9".to_owned(), None);
    settings.data_dir = temp.path().join("data");
    let upload_dir = settings.data_dir.join("cache/lora-uploads/upload-1");
    tokio::fs::create_dir_all(&upload_dir).await.unwrap();
    let source_file = upload_dir.join("detail.safetensors");
    tokio::fs::write(&source_file, b"lora").await.unwrap();
    let mut payload = serde_json::Map::new();
    payload.insert(
        "sourcePath".to_owned(),
        json!(source_file.display().to_string()),
    );
    payload.insert("uploadedSourcePath".to_owned(), json!(true));

    cleanup_uploaded_import_source(&settings, &payload)
        .await
        .unwrap();

    assert!(!source_file.exists());
    assert!(!upload_dir.exists());
}

#[tokio::test]
async fn uploaded_lora_source_cleanup_rejects_paths_outside_upload_cache() {
    let temp = tempdir().expect("tempdir creates");
    let mut settings = test_settings("http://127.0.0.1:9".to_owned(), None);
    settings.data_dir = temp.path().join("data");
    let outside_file = temp.path().join("outside.safetensors");
    tokio::fs::write(&outside_file, b"lora").await.unwrap();
    let mut payload = serde_json::Map::new();
    payload.insert(
        "sourcePath".to_owned(),
        json!(outside_file.display().to_string()),
    );
    payload.insert("uploadedSourcePath".to_owned(), json!(true));

    let error = cleanup_uploaded_import_source(&settings, &payload)
        .await
        .expect_err("outside path is rejected");

    assert!(matches!(error, WorkerError::InvalidPayload(_)));
    assert!(outside_file.exists());
}

#[tokio::test]
async fn uploaded_lora_file_import_prefers_move_over_copy() {
    let temp = tempdir().expect("tempdir creates");
    let source_file = temp.path().join("uploaded.safetensors");
    tokio::fs::write(&source_file, b"lora").await.unwrap();
    let target_dir = temp.path().join("target");

    import_lora_source_path(&source_file, &target_dir, true)
        .await
        .unwrap();

    assert!(!source_file.exists());
    assert_eq!(
        tokio::fs::read(target_dir.join("uploaded.safetensors"))
            .await
            .unwrap(),
        b"lora"
    );
}

#[tokio::test]
async fn paired_moe_upload_writes_high_low_convention_files() {
    // sc-1991: a bring-your-own Wan A14B MoE pair must land in one record under the
    // dot-delimited high/low_noise convention (off-convention upload names are
    // normalized), so the Python resolver detects the pair and resolves the high
    // half as the primary. Both halves are written into the same target dir.
    let temp = tempdir().expect("tempdir creates");
    let high_upload = temp.path().join("upload-hi").join("HighNoise.safetensors");
    let low_upload = temp.path().join("upload-lo").join("low-noise.safetensors");
    tokio::fs::create_dir_all(high_upload.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::create_dir_all(low_upload.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&high_upload, b"high").await.unwrap();
    tokio::fs::write(&low_upload, b"low").await.unwrap();
    let target_dir = temp.path().join("loras").join("my_moe");

    let (high_name, low_name) = wan_moe_pair_filenames("my_moe");
    assert_eq!(high_name, "my_moe.high_noise.safetensors");
    assert_eq!(low_name, "my_moe.low_noise.safetensors");

    import_lora_source_file_as(&high_upload, &target_dir, &high_name, true)
        .await
        .unwrap();
    import_lora_source_file_as(&low_upload, &target_dir, &low_name, true)
        .await
        .unwrap();

    assert_eq!(
        tokio::fs::read(target_dir.join(&high_name)).await.unwrap(),
        b"high"
    );
    assert_eq!(
        tokio::fs::read(target_dir.join(&low_name)).await.unwrap(),
        b"low"
    );
    // The high-noise file sorts first, so directory resolution picks it as primary.
    assert!(high_name < low_name);
    // prefer_move consumed both staged uploads.
    assert!(!high_upload.exists());
    assert!(!low_upload.exists());
}

#[tokio::test]
async fn lora_url_import_downloads_to_named_file() {
    let temp = tempdir().expect("tempdir creates");
    let source_url = spawn_binary_stub(b"url-lora".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = source_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let target_dir = temp.path().join("url-target");

    download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{source_url}/loras/style.safetensors"),
        &target_dir,
    )
    .await
    .expect("url LoRA downloads");

    assert_eq!(
        tokio::fs::read(target_dir.join("style.safetensors"))
            .await
            .unwrap(),
        b"url-lora"
    );
}

#[tokio::test]
async fn lora_url_import_skips_existing_matching_file() {
    let temp = tempdir().expect("tempdir creates");
    let source_url = spawn_binary_stub(b"new-lora".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = source_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let target_dir = temp.path().join("url-target");
    tokio::fs::create_dir_all(&target_dir).await.unwrap();
    tokio::fs::write(target_dir.join("style.safetensors"), b"old-lora")
        .await
        .unwrap();

    download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{source_url}/loras/style.safetensors"),
        &target_dir,
    )
    .await
    .expect("existing LoRA is accepted");

    assert_eq!(
        tokio::fs::read(target_dir.join("style.safetensors"))
            .await
            .unwrap(),
        b"old-lora"
    );
}

#[tokio::test]
async fn download_snapshot_rejects_truncated_file() {
    let temp = tempdir().expect("tempdir creates");
    // The stub serves 4 bytes, but the snapshot claims the shard is 64 —
    // a truncated transfer that must not be accepted as complete.
    let base_url = spawn_binary_stub(b"trun".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let repo_dir = temp.path().join("models--owner--model");

    let snapshot = HuggingFaceSnapshot {
        files: vec![SnapshotFile {
            path: "shard.safetensors".to_owned(),
            size: Some(64),
            download_url: format!("{base_url}/owner/model/resolve/main/shard.safetensors"),
            sha256: None,
        }],
    };
    let mut progress = DownloadProgress::new(
        "owner/model",
        0,
        snapshot.total_bytes(),
        Duration::from_secs(3600),
    );

    let error = download_snapshot_into_cache(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &repo_dir,
        "main",
        &snapshot,
        &mut progress,
    )
    .await
    .expect_err("truncated shard is rejected");

    assert!(error.to_string().contains("expected"));
    // The partial blob is preserved so a retry can resume it, and the snapshot is
    // never materialized over a corrupt blob.
    assert_eq!(
        tokio::fs::read(repo_dir.join("blobs").join("etag-shard.safetensors"))
            .await
            .unwrap(),
        b"trun"
    );
    assert!(!repo_dir.join("snapshots").exists());
}

#[test]
fn normalize_sha256_accepts_only_real_digests() {
    let hex = "a".repeat(64);
    // Bare 64-hex, a `sha256:` prefix, and uppercase all normalize to lowercase hex.
    assert_eq!(normalize_sha256(&hex).as_deref(), Some(hex.as_str()));
    assert_eq!(
        normalize_sha256(&format!("  sha256:{hex}  ")).as_deref(),
        Some(hex.as_str())
    );
    assert_eq!(
        normalize_sha256(&"A".repeat(64)).as_deref(),
        Some(hex.as_str())
    );
    // A git blob SHA-1 (40 hex), a non-hex string, and empty are not content digests.
    assert_eq!(normalize_sha256(&"a".repeat(40)), None);
    assert_eq!(normalize_sha256(&"z".repeat(64)), None);
    assert_eq!(normalize_sha256(""), None);
}

#[tokio::test]
async fn download_snapshot_rejects_digest_mismatch() {
    let temp = tempdir().expect("tempdir creates");
    let base_url = spawn_binary_stub(b"weights!!".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let repo_dir = temp.path().join("models--owner--model");

    // The transfer is complete (size matches) but the source-declared sha256 does
    // not — a corrupted download that must be rejected and discarded (sc-6137).
    let snapshot = HuggingFaceSnapshot {
        files: vec![SnapshotFile {
            path: "model.safetensors".to_owned(),
            size: Some(9),
            download_url: format!("{base_url}/owner/model/resolve/main/model.safetensors"),
            sha256: Some("0".repeat(64)),
        }],
    };
    let mut progress = DownloadProgress::new(
        "owner/model",
        0,
        snapshot.total_bytes(),
        Duration::from_secs(3600),
    );

    let error = download_snapshot_into_cache(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &repo_dir,
        "main",
        &snapshot,
        &mut progress,
    )
    .await
    .expect_err("a digest mismatch is rejected");

    assert!(error
        .to_string()
        .to_ascii_lowercase()
        .contains("integrity check"));
    // The corrupt blob is removed and the snapshot is never materialized.
    assert!(!repo_dir
        .join("blobs")
        .join("etag-model.safetensors")
        .exists());
    assert!(!repo_dir.join("snapshots").exists());
}

#[tokio::test]
async fn download_snapshot_accepts_matching_digest() {
    use sha2::{Digest, Sha256};
    let temp = tempdir().expect("tempdir creates");
    let base_url = spawn_binary_stub(b"weights!!".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let repo_dir = temp.path().join("models--owner--model");

    let digest = format!("{:x}", Sha256::digest(b"weights!!"));
    let snapshot = HuggingFaceSnapshot {
        files: vec![SnapshotFile {
            path: "model.safetensors".to_owned(),
            size: Some(9),
            download_url: format!("{base_url}/owner/model/resolve/main/model.safetensors"),
            sha256: Some(digest),
        }],
    };
    let mut progress = DownloadProgress::new(
        "owner/model",
        0,
        snapshot.total_bytes(),
        Duration::from_secs(3600),
    );

    download_snapshot_into_cache(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &repo_dir,
        "main",
        &snapshot,
        &mut progress,
    )
    .await
    .expect("a matching digest is accepted");

    assert_eq!(
        tokio::fs::read(repo_dir.join("blobs").join("etag-model.safetensors"))
            .await
            .unwrap(),
        b"weights!!"
    );
}

#[tokio::test]
async fn download_snapshot_resumes_existing_partial_blob() {
    let temp = tempdir().expect("tempdir creates");
    let base_url = spawn_binary_stub(b"weights!!".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let repo_dir = temp.path().join("models--owner--model");
    let blob_path = repo_dir.join("blobs").join("etag-model.safetensors");
    tokio::fs::create_dir_all(blob_path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&blob_path, b"weig").await.unwrap();

    let snapshot = HuggingFaceSnapshot {
        files: vec![SnapshotFile {
            path: "model.safetensors".to_owned(),
            size: Some(9),
            download_url: format!("{base_url}/owner/model/resolve/main/model.safetensors"),
            sha256: None,
        }],
    };
    let mut progress = DownloadProgress::new(
        "owner/model",
        4,
        snapshot.total_bytes(),
        Duration::from_secs(3600),
    );

    download_snapshot_into_cache(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &repo_dir,
        "main",
        &snapshot,
        &mut progress,
    )
    .await
    .expect("partial blob resumes");

    assert_eq!(tokio::fs::read(&blob_path).await.unwrap(), b"weights!!");
}

#[tokio::test]
async fn download_snapshot_fresh_retry_discards_partial_blob() {
    let temp = tempdir().expect("tempdir creates");
    let base_url = spawn_binary_stub(b"weights!!".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let repo_dir = temp.path().join("models--owner--model");
    let blob_path = repo_dir.join("blobs").join("etag-model.safetensors");
    tokio::fs::create_dir_all(blob_path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&blob_path, b"bad").await.unwrap();

    let snapshot = HuggingFaceSnapshot {
        files: vec![SnapshotFile {
            path: "model.safetensors".to_owned(),
            size: Some(9),
            download_url: format!("{base_url}/owner/model/resolve/main/model.safetensors"),
            sha256: None,
        }],
    };
    let mut progress = DownloadProgress::new(
        "owner/model",
        3,
        snapshot.total_bytes(),
        Duration::from_secs(3600),
    );

    download_snapshot_into_cache(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: true,
        },
        &repo_dir,
        "main",
        &snapshot,
        &mut progress,
    )
    .await
    .expect("fresh retry redownloads from the beginning");

    assert_eq!(tokio::fs::read(&blob_path).await.unwrap(), b"weights!!");
}

// --- Derived fast-tokenizer overlay (Kolors sc-4764, Qwen-Image sc-6570) --------------------

#[test]
fn derived_tokenizer_overlay_targets_only_known_base_repos() {
    let snap = std::path::Path::new("/snap");
    let want = PathBuf::from("/snap/tokenizer/tokenizer.json");
    // Kolors → its SceneWorks tokenizer repo + the snapshot's tokenizer/tokenizer.json.
    assert_eq!(
        derived_tokenizer_overlay("Kwai-Kolors/Kolors-diffusers", snap),
        Some(("SceneWorks/kolors-chatglm3-tokenizer", want.clone()))
    );
    // Qwen-Image (sc-6570) → its SceneWorks tokenizer repo, same dest.
    assert_eq!(
        derived_tokenizer_overlay("Qwen/Qwen-Image", snap),
        Some(("SceneWorks/qwen-image-tokenizer", want.clone()))
    );
    // Whitespace from a manifest field is tolerated.
    assert_eq!(
        derived_tokenizer_overlay("  Qwen/Qwen-Image  ", snap),
        Some(("SceneWorks/qwen-image-tokenizer", want))
    );
    // Every other model is a no-op — including sibling repos that must NOT match.
    assert_eq!(derived_tokenizer_overlay("owner/model", snap), None);
    assert_eq!(
        derived_tokenizer_overlay("Kwai-Kolors/Kolors-IP-Adapter-Plus", snap),
        None
    );
    // Qwen-Image-Edit-2511 ships its own tokenizer.json upstream → no overlay.
    assert_eq!(
        derived_tokenizer_overlay("Qwen/Qwen-Image-Edit-2511", snap),
        None
    );
}

/// A single stub that serves both the HF tree resolve (for the SceneWorks tokenizer repo) and the
/// file bytes (the catch-all), plus the job/progress routes `check_cancel`/progress need. The tree
/// advertises one `tokenizer.json` file sized to `bytes`.
async fn spawn_overlay_stub(bytes: Vec<u8>) -> String {
    let state = BinaryStubState {
        bytes,
        status: AxumStatusCode::OK,
        cancel_requested: false,
    };
    let app = Router::new()
        .route(
            "/api/models/:owner/:repo/tree/:revision",
            get(overlay_tree_stub),
        )
        .route("/api/v1/jobs/:job_id", get(job_stub))
        .route("/api/v1/jobs/:job_id/progress", post(progress_stub))
        .route("/*path", get(binary_stub).head(binary_head_stub))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn overlay_tree_stub(State(state): State<BinaryStubState>) -> Response {
    Json(json!([
        { "type": "file", "path": "tokenizer.json", "size": state.bytes.len() }
    ]))
    .into_response()
}

#[tokio::test]
async fn overlay_derived_tokenizer_fetches_and_places_the_kolors_json() {
    let temp = tempdir().expect("tempdir creates");
    let bytes = br#"{"version":"1.0","model":{"type":"BPE"}}"#.to_vec();
    let base_url = spawn_overlay_stub(bytes.clone()).await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    // A real snapshot already has a `tokenizer/` dir (with the slow SP files); the overlay adds the
    // fast json next to them.
    let snapshot_dir = temp.path().join("snapshots").join("abc123");
    tokio::fs::create_dir_all(snapshot_dir.join("tokenizer"))
        .await
        .unwrap();

    overlay_derived_tokenizer(
        &api,
        &settings,
        &client,
        "job-1",
        "Kwai-Kolors/Kolors-diffusers",
        &snapshot_dir,
    )
    .await
    .expect("overlay fetches and places the tokenizer");

    let placed = tokio::fs::read(snapshot_dir.join("tokenizer").join("tokenizer.json"))
        .await
        .expect("tokenizer.json was written");
    assert_eq!(placed, bytes);
}

#[tokio::test]
async fn overlay_derived_tokenizer_overlays_qwen_image() {
    // The Qwen-Image base repo (sc-6570) is overlaid exactly like Kolors — same dest, its own repo.
    let temp = tempdir().expect("tempdir creates");
    let bytes = br#"{"version":"1.0","model":{"type":"BPE"},"qwen":true}"#.to_vec();
    let base_url = spawn_overlay_stub(bytes.clone()).await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let snapshot_dir = temp.path().join("snapshots").join("75e0b4b");
    tokio::fs::create_dir_all(snapshot_dir.join("tokenizer"))
        .await
        .unwrap();

    overlay_derived_tokenizer(
        &api,
        &settings,
        &client,
        "job-1",
        "Qwen/Qwen-Image",
        &snapshot_dir,
    )
    .await
    .expect("qwen-image overlay fetches and places the tokenizer");

    let placed = tokio::fs::read(snapshot_dir.join("tokenizer").join("tokenizer.json"))
        .await
        .expect("tokenizer.json was written");
    assert_eq!(placed, bytes);
}

#[tokio::test]
async fn overlay_derived_tokenizer_is_noop_for_other_repos() {
    // An unreachable base URL: if the guard failed to short-circuit, the resolve would error.
    let temp = tempdir().expect("tempdir creates");
    let settings = test_settings("http://127.0.0.1:1".to_owned(), None);
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    overlay_derived_tokenizer(
        &api,
        &settings,
        &client,
        "job-1",
        "owner/model",
        temp.path(),
    )
    .await
    .expect("unlisted repo is a no-op");
}

#[tokio::test]
async fn overlay_derived_tokenizer_skips_when_already_present() {
    // dest exists → return Ok before any network (unreachable URL proves no download is attempted).
    let temp = tempdir().expect("tempdir creates");
    let tokenizer_dir = temp.path().join("tokenizer");
    tokio::fs::create_dir_all(&tokenizer_dir).await.unwrap();
    tokio::fs::write(tokenizer_dir.join("tokenizer.json"), b"existing")
        .await
        .unwrap();
    let settings = test_settings("http://127.0.0.1:1".to_owned(), None);
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    overlay_derived_tokenizer(
        &api,
        &settings,
        &client,
        "job-1",
        "Qwen/Qwen-Image",
        temp.path(),
    )
    .await
    .expect("present tokenizer is left untouched");
    assert_eq!(
        tokio::fs::read(tokenizer_dir.join("tokenizer.json"))
            .await
            .unwrap(),
        b"existing"
    );
}

#[tokio::test]
async fn download_snapshot_writes_hugging_face_cache_layout() {
    let temp = tempdir().expect("tempdir creates");
    let base_url = spawn_binary_stub(b"weights!!".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let repo_dir = temp.path().join("models--owner--model");

    let snapshot = HuggingFaceSnapshot {
        files: vec![SnapshotFile {
            path: "model.safetensors".to_owned(),
            size: Some(9),
            download_url: format!("{base_url}/owner/model/resolve/main/model.safetensors"),
            sha256: None,
        }],
    };
    let mut progress = DownloadProgress::new(
        "owner/model",
        0,
        snapshot.total_bytes(),
        Duration::from_secs(3600),
    );

    download_snapshot_into_cache(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &repo_dir,
        "main",
        &snapshot,
        &mut progress,
    )
    .await
    .expect("snapshot downloads into the hub cache layout");

    // refs/<rev> records the commit reported by the resolve metadata.
    assert_eq!(
        tokio::fs::read_to_string(repo_dir.join("refs").join("main"))
            .await
            .unwrap(),
        "stubcommit"
    );
    // Content lands in a blob named by etag.
    assert_eq!(
        tokio::fs::read(repo_dir.join("blobs").join("etag-model.safetensors"))
            .await
            .unwrap(),
        b"weights!!"
    );
    // The snapshot entry resolves to that content (symlink on unix, copy otherwise).
    assert_eq!(
        tokio::fs::read(
            repo_dir
                .join("snapshots")
                .join("stubcommit")
                .join("model.safetensors")
        )
        .await
        .unwrap(),
        b"weights!!"
    );
}

/// Opt-in: hits the real huggingface.co to confirm the cache layout we write
/// matches what huggingface_hub expects — exercising the live resolve tree, the
/// metadata HEAD (`ETag` for regular files, `X-Linked-Etag` + a CDN redirect for
/// LFS files), and `X-Repo-Commit`. Ignored by default so CI/offline runs never
/// hit the network. Run it with:
///   cargo test -p sceneworks-worker -- --ignored real_huggingface
#[tokio::test]
#[ignore = "network: downloads a tiny public repo from huggingface.co"]
async fn download_snapshot_into_cache_matches_real_huggingface_layout() {
    let temp = tempdir().expect("tempdir creates");
    // Cancel/heartbeat checks go to a benign local stub; the files download from the
    // real huggingface.co set as the HF base URL.
    let api_base = spawn_binary_stub(b"ignored".to_vec()).await;
    let mut settings = test_settings("https://huggingface.co".to_owned(), None);
    settings.api_url = api_base;
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let repo = "hf-internal-testing/tiny-random-bert";
    let repo_dir = temp
        .path()
        .join("models--hf-internal-testing--tiny-random-bert");

    // A small regular file (config.json, ETag path) plus any safetensors weights
    // (LFS path) so both header behaviors are exercised.
    let snapshot = HuggingFaceSnapshot::resolve(
        &client,
        &settings,
        repo,
        "main",
        &["config.json".to_owned(), "*.safetensors".to_owned()],
    )
    .await
    .expect("resolves the live repo tree");
    assert!(
        snapshot.files.iter().any(|file| file.path == "config.json"),
        "expected config.json in the resolved tree"
    );

    let mut progress =
        DownloadProgress::new(repo, 0, snapshot.total_bytes(), Duration::from_secs(3600));
    download_snapshot_into_cache(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "real",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &repo_dir,
        "main",
        &snapshot,
        &mut progress,
    )
    .await
    .expect("downloads the live repo into the hub cache layout");

    // refs/main records the real git commit sha (40 hex chars).
    let commit = tokio::fs::read_to_string(repo_dir.join("refs").join("main"))
        .await
        .expect("refs/main written");
    let commit = commit.trim();
    assert_eq!(
        commit.len(),
        40,
        "commit should be a 40-char git sha: {commit}"
    );

    // Every resolved file materializes under snapshots/<commit>/ with its exact
    // declared size — confirming both the ETag and X-Linked-Etag (LFS) paths.
    let snapshot_dir = repo_dir.join("snapshots").join(commit);
    for file in &snapshot.files {
        let path = snapshot_dir.join(&file.path);
        let bytes = tokio::fs::read(&path)
            .await
            .unwrap_or_else(|_| panic!("{} present in snapshot", file.path));
        assert!(!bytes.is_empty(), "{} is empty", file.path);
        if let Some(size) = file.size {
            assert_eq!(bytes.len() as u64, size, "{} size mismatch", file.path);
        }
    }
    // The blob store is populated (the snapshot entries point into it).
    assert!(
        repo_dir
            .join("blobs")
            .read_dir()
            .map(|mut entries| entries.next().is_some())
            .unwrap_or(false),
        "blobs/ should hold the downloaded content"
    );
}

#[tokio::test]
async fn lora_url_import_rejects_failed_and_oversized_downloads() {
    let temp = tempdir().expect("tempdir creates");
    let missing_url =
        spawn_binary_stub_with_options(b"missing".to_vec(), AxumStatusCode::NOT_FOUND, false).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = missing_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let error = download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{missing_url}/loras/missing.safetensors"),
        &temp.path().join("missing-target"),
    )
    .await
    .expect_err("failed URL returns an error");
    assert!(error.to_string().contains("404"));

    let large_url = spawn_binary_stub(b"too-large".to_vec()).await;
    settings.api_url = large_url.clone();
    settings.max_lora_url_bytes = 4;
    let api = ApiClient::new(&settings);
    let error = download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{large_url}/loras/large.safetensors"),
        &temp.path().join("large-target"),
    )
    .await
    .expect_err("oversized URL returns an error");
    assert!(error.to_string().contains("exceeds"));
}

/// sc-8806 — the source-URL chunk loop no longer polls the cancel endpoint per
/// received HTTP chunk (that issued one `GET /api/v1/jobs/{id}` per chunk on a
/// multi-GB download and serialized the transfer on API round-trips). A user
/// cancel is observed on the progress-report tick instead — exactly like
/// `download_file_inner` — and the decision is read from the `JobSnapshot` the
/// progress POST already returns, never a separate GET. The stub's binary body
/// stalls after the first chunk so ONLY the interval tick can trip this cancel;
/// the GET counter proves the loop never fell back to polling.
#[tokio::test]
async fn lora_url_import_honors_cancel_on_progress_tick() {
    let temp = tempdir().expect("tempdir creates");
    let job_gets = Arc::new(AtomicUsize::new(0));
    let source_url = spawn_cancel_tick_stub(CancelTickStubState {
        job_gets: job_gets.clone(),
        progress_cancel_requested: true,
        stall_after_first_chunk: true,
    })
    .await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = source_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let error = download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "LoRA import canceled by user.",
            fresh_download: false,
        },
        &format!("{source_url}/loras/style.safetensors"),
        &temp.path().join("cancel-target"),
    )
    .await
    .expect_err("cancel request interrupts the URL import on the report tick");

    assert!(matches!(error, WorkerError::Canceled(_)));
    assert_eq!(
        job_gets.load(Ordering::SeqCst),
        0,
        "the chunk loop must not poll the job endpoint; cancel comes from the progress POST snapshot"
    );
}

/// sc-8806 — a successful source-URL download must not touch the job endpoint
/// while streaming: no per-chunk cancel GETs (the regression this story removes).
#[tokio::test]
async fn lora_url_import_streams_chunks_without_cancel_polls() {
    let temp = tempdir().expect("tempdir creates");
    let job_gets = Arc::new(AtomicUsize::new(0));
    let source_url = spawn_cancel_tick_stub(CancelTickStubState {
        job_gets: job_gets.clone(),
        progress_cancel_requested: false,
        stall_after_first_chunk: false,
    })
    .await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = source_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let target_dir = temp.path().join("url-target");

    download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{source_url}/loras/style.safetensors"),
        &target_dir,
    )
    .await
    .expect("url LoRA downloads");

    assert_eq!(
        tokio::fs::read(target_dir.join("style.safetensors"))
            .await
            .unwrap(),
        b"url-lora"
    );
    assert_eq!(
        job_gets.load(Ordering::SeqCst),
        0,
        "a multi-chunk transfer must issue zero cancel GETs"
    );
}

/// sc-8806 (snapshot reuse) — the download report tick reads `cancel_requested`
/// off the `JobSnapshot` the progress POST already returns instead of issuing a
/// third GET per tick. The stub's GET route reports NOT-canceled (and counts
/// hits) while the POST snapshot says canceled: the tick must trip the cancel
/// purely off the POST response, with zero GETs — the pre-fix code (POST result
/// discarded, decision from a fresh GET) would have returned Ok here.
#[tokio::test]
async fn download_progress_tick_cancels_from_progress_post_snapshot() {
    let job_gets = Arc::new(AtomicUsize::new(0));
    let base_url = spawn_cancel_tick_stub(CancelTickStubState {
        job_gets: job_gets.clone(),
        progress_cancel_requested: true,
        stall_after_first_chunk: false,
    })
    .await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let progress = DownloadProgress::new("owner/model", 0, Some(64), Duration::from_secs(5));

    let error = report_download_progress(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &progress,
    )
    .await
    .expect_err("a cancel-requested progress snapshot trips the tick cancel");

    assert!(matches!(error, WorkerError::Canceled(_)));
    assert_eq!(
        job_gets.load(Ordering::SeqCst),
        0,
        "the tick must not GET the job for its cancel decision"
    );
}

#[test]
fn parse_credentials_env_normalizes_and_skips_blanks() {
    let credentials = parse_credentials_env(
        r#"{ "Civitai.com": { "token": " key ", "scheme": "query" },
            "huggingface.co": { "token": "hf" },
            "blank.example": { "token": "" } }"#,
    );
    assert_eq!(credentials.len(), 2);
    let civitai = credentials
        .iter()
        .find(|credential| credential.host == "civitai.com")
        .expect("civitai credential");
    assert_eq!(civitai.token, "key");
    assert_eq!(civitai.scheme, CredentialScheme::Query);
    let hugging_face = credentials
        .iter()
        .find(|credential| credential.host == "huggingface.co")
        .expect("hf credential");
    // An absent scheme defaults to bearer.
    assert_eq!(hugging_face.scheme, CredentialScheme::Bearer);
}

#[test]
fn parse_credentials_env_tolerates_invalid_json() {
    assert!(parse_credentials_env("not json").is_empty());
}

#[test]
fn credential_for_host_matches_case_insensitively() {
    let mut settings = test_settings("https://huggingface.co".to_owned(), None);
    settings.credentials = vec![WorkerCredential {
        host: "civitai.com".to_owned(),
        token: "key".to_owned(),
        scheme: CredentialScheme::Query,
    }];
    assert!(credential_for_host(&settings, "Civitai.com").is_some());
    assert!(credential_for_host(&settings, "example.com").is_none());
    assert!(credential_for_host(&settings, "").is_none());
}

#[test]
fn worker_credentials_env_overrides_file_per_host() {
    // Server reads the config-dir file store; an operator's SCENEWORKS_CREDENTIALS
    // env wins per host, and file-only hosts survive.
    let file = parse_credentials_env(
        r#"{ "civitai.com": { "token": "file-civitai", "scheme": "query" },
            "huggingface.co": { "token": "file-hf" } }"#,
    );
    let env = parse_credentials_env(
        r#"{ "civitai.com": { "token": "env-civitai", "scheme": "bearer" } }"#,
    );
    let merged = super::merge_credentials(file, env);
    assert_eq!(merged.len(), 2);
    let civitai = merged
        .iter()
        .find(|credential| credential.host == "civitai.com")
        .expect("civitai credential");
    assert_eq!(civitai.token, "env-civitai");
    assert_eq!(civitai.scheme, CredentialScheme::Bearer);
    let hugging_face = merged
        .iter()
        .find(|credential| credential.host == "huggingface.co")
        .expect("hf credential");
    assert_eq!(hugging_face.token, "file-hf");
}

#[tokio::test]
async fn source_url_follows_redirect_and_strips_auth_across_hosts() {
    let temp = tempdir().expect("tempdir creates");
    // The download host (127.0.0.1) requires a bearer token, then 302-redirects to
    // a different host (localhost) that rejects any Authorization header — so the
    // download only succeeds if the token is applied on hop 1 and dropped on hop 2.
    let download_base = spawn_cross_host_redirect_stub("testtoken").await;
    let api_base = spawn_binary_stub(b"ignored".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = api_base.clone();
    settings.credentials = vec![WorkerCredential {
        host: "127.0.0.1".to_owned(),
        token: "testtoken".to_owned(),
        scheme: CredentialScheme::Bearer,
    }];
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let target_dir = temp.path().join("redirect-target");

    download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{download_base}/download/style.safetensors"),
        &target_dir,
    )
    .await
    .expect("authenticated redirected download succeeds");

    assert_eq!(
        tokio::fs::read(target_dir.join("style.safetensors"))
            .await
            .unwrap(),
        b"civitai-lora"
    );
}

#[tokio::test]
async fn source_url_client_pins_dns_to_validated_address() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    let state = BinaryStubState {
        bytes: b"weights!!".to_vec(),
        status: AxumStatusCode::OK,
        cancel_requested: false,
    };
    let app = Router::new()
        .route("/file/style.safetensors", get(binary_stub))
        .with_state(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });

    let url = reqwest::Url::parse(&format!(
        "http://rebind.test:{}/file/style.safetensors",
        address.port()
    ))
    .expect("test URL parses");
    let validated = [SocketAddr::new(
        "127.0.0.1".parse().unwrap(),
        address.port(),
    )];
    let client = build_source_url_client(&url, Some(&validated)).expect("client builds");

    let bytes = client
        .get(url)
        .send()
        .await
        .expect("request uses pinned address")
        .error_for_status()
        .expect("stub response is successful")
        .bytes()
        .await
        .expect("response body reads");

    assert_eq!(bytes.as_ref(), b"weights!!");
}

#[tokio::test]
async fn source_url_rejects_redirect_to_non_http_scheme() {
    let temp = tempdir().expect("tempdir creates");
    let download_base = spawn_location_redirect_stub("file:///etc/passwd").await;
    let api_base = spawn_binary_stub(b"ignored".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = api_base;
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let error = download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{download_base}/download/style.safetensors"),
        &temp.path().join("scheme-target"),
    )
    .await
    .expect_err("non-http redirect target is rejected");
    assert!(error.to_string().contains("http or https"));
}

#[tokio::test]
async fn source_url_rejects_excessive_redirects() {
    let temp = tempdir().expect("tempdir creates");
    // Always redirects to a sibling path on the same host — an unterminated loop.
    let download_base = spawn_location_redirect_stub("loop").await;
    let api_base = spawn_binary_stub(b"ignored".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = api_base;
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let error = download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{download_base}/download/style.safetensors"),
        &temp.path().join("loop-target"),
    )
    .await
    .expect_err("a redirect loop is bounded");
    assert!(error.to_string().contains("redirect limit"));
}

#[derive(Clone)]
struct CrossHostRedirectState {
    port: u16,
    token: String,
}

async fn spawn_cross_host_redirect_stub(token: &str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    let state = CrossHostRedirectState {
        port: address.port(),
        token: token.to_owned(),
    };
    let app = Router::new()
        .route(
            "/download/*path",
            get(cross_host_download).head(cross_host_download),
        )
        .route("/file/*path", get(cross_host_file))
        .with_state(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn cross_host_download(
    State(state): State<CrossHostRedirectState>,
    headers: HeaderMap,
) -> Response {
    if !has_bearer(&headers, &state.token) {
        return AxumStatusCode::UNAUTHORIZED.into_response();
    }
    let mut response = Response::new(Body::empty());
    *response.status_mut() = AxumStatusCode::FOUND;
    response.headers_mut().insert(
        axum::http::header::LOCATION,
        axum::http::HeaderValue::from_str(&format!(
            "http://localhost:{}/file/style.safetensors",
            state.port
        ))
        .expect("location header"),
    );
    response
}

async fn cross_host_file(headers: HeaderMap) -> Response {
    // The bearer token must never be carried onto the cross-host CDN hop.
    if headers.contains_key(axum::http::header::AUTHORIZATION) {
        return AxumStatusCode::FORBIDDEN.into_response();
    }
    let bytes = b"civitai-lora".to_vec();
    let length = bytes.len();
    let mut response = bytes.into_response();
    response.headers_mut().insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_str(&length.to_string()).expect("content length header"),
    );
    response
}

#[derive(Clone)]
struct LocationRedirectState {
    location: String,
}

async fn spawn_location_redirect_stub(location: &str) -> String {
    let state = LocationRedirectState {
        location: location.to_owned(),
    };
    let app = Router::new()
        .route(
            "/download/*path",
            get(location_redirect).head(location_redirect),
        )
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn location_redirect(State(state): State<LocationRedirectState>) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = AxumStatusCode::FOUND;
    response.headers_mut().insert(
        axum::http::header::LOCATION,
        axum::http::HeaderValue::from_str(&state.location).expect("location header"),
    );
    response
}

fn has_bearer(headers: &HeaderMap, token: &str) -> bool {
    let expected = format!("Bearer {token}");
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        == Some(expected.as_str())
}

#[test]
fn now_matches_python_second_precision() {
    let value = now_rfc3339();

    assert!(value.ends_with('Z'));
    assert!(!value.trim_end_matches('Z').contains('.'));
}

#[test]
fn ffmpeg_helper_shapes_match_python_timeline_exporter() {
    assert_eq!(output_dimensions("16:9", 720), (1280, 720));
    assert_eq!(output_dimensions("9:16", 720), (720, 1280));
    assert_eq!(output_dimensions("1:1", 721), (722, 722));

    let concat = concat_file_contents(
        [
            PathBuf::from(r"C:\renders\clip one's.mp4"),
            PathBuf::from("nested/two.mp4"),
        ]
        .iter(),
    );
    assert!(concat.contains("C:/renders/clip one'\\''s.mp4"));
    assert!(concat.contains("file 'nested/two.mp4'"));

    let asset_id = fresh_asset_id();
    assert!(asset_id.starts_with("asset_"));
    assert_eq!(asset_id.len(), "asset_".len() + 32);
    assert!(asset_id["asset_".len()..]
        .chars()
        .all(|character| character.is_ascii_hexdigit()));
}

#[test]
fn plan_segments_inserts_gaps_and_totals_duration() {
    let items = vec![
        json!({"assetId": "a", "timelineStart": 1.0, "timelineEnd": 3.0}),
        json!({"assetId": "b", "timelineStart": 3.0, "timelineEnd": 5.0}),
        json!({"assetId": "c", "timelineStart": 6.5, "timelineEnd": 8.0}),
    ];

    let (plan, duration) = plan_segments(&items).expect("plan succeeds");

    assert_eq!(plan.len(), 3);
    // Leading hole before the first item becomes a black gap.
    assert_eq!(plan[0].leading_gap, Some(1.0));
    // Abutting items leave no gap.
    assert_eq!(plan[1].leading_gap, None);
    // Interior hole between items becomes a gap of the missing span.
    assert_eq!(plan[2].leading_gap, Some(1.5));
    // Total duration is the running max of item ends.
    assert_eq!(duration, 8.0);
}

#[test]
fn plan_segments_carries_item_transitions() {
    let items = vec![
        json!({
            "assetId": "a",
            "timelineStart": 0.0,
            "timelineEnd": 2.0,
            "transitionIn": {"type": "crossfade", "duration": 0.8}
        }),
        json!({"assetId": "b", "timelineStart": 2.0, "timelineEnd": 4.0}),
    ];

    let (plan, _) = plan_segments(&items).expect("plan succeeds");

    assert_eq!(plan[0].transition.as_deref(), Some("crossfade"));
    assert_eq!(plan[0].transition_duration, 0.8);
    // Missing transitionIn falls back to the default transition duration.
    assert_eq!(plan[1].transition, None);
    assert_eq!(
        plan[1].transition_duration,
        DEFAULT_TRANSITION_DURATION_SECONDS
    );
}

#[test]
fn plan_segments_rejects_nonpositive_item_span() {
    let items = vec![json!({"assetId": "a", "timelineStart": 2.0, "timelineEnd": 2.0})];

    let error = plan_segments(&items).expect_err("zero-length span rejects");

    assert!(matches!(error, WorkerError::InvalidPayload(_)));
    assert!(error.to_string().contains("timelineEnd must be greater"));
}

#[test]
fn person_detection_jitter_uses_python_sha256_bytes() {
    let detections = candidate_people(1280, 720, "asset_source_clip", 1.25);

    assert_eq!(detections[0]["box"]["x"].as_f64(), Some(0.338));
    assert_eq!(detections[1]["box"]["x"].as_f64(), Some(0.579));
    assert_eq!(detections[2]["box"]["x"].as_f64(), Some(0.134));
}

#[test]
fn missing_crossfade_duration_defaults_to_python_mux_duration() {
    let missing = json!(null);
    assert_eq!(
        value_f64(&missing, DEFAULT_TRANSITION_DURATION_SECONDS),
        0.5
    );
    assert_eq!(crossfade_duration(0.5), 0.5);
    assert_eq!(crossfade_duration(0.0), 0.1);
    assert_eq!(crossfade_duration(2.0), 1.5);
}

#[test]
fn path_and_error_helpers_are_bounded_and_defensive() {
    let temp = tempdir().expect("tempdir creates");
    let error = safe_project_path(temp.path(), "").expect_err("empty relative path rejects");
    assert!(error
        .to_string()
        .contains("Project-relative path is required"));

    // sc-4278 / F-MLXW-14: load_reference_image and resolve_clip_media_path route
    // sidecar-sourced media paths through safe_project_path, so a traversal or
    // absolute path (from a poisoned, user-editable sidecar) must be rejected
    // rather than escaping the project.
    for unsafe_rel in ["../../etc/passwd", "assets/../../secret.png", "/etc/passwd"] {
        let error = safe_project_path(temp.path(), unsafe_rel)
            .expect_err("traversal/absolute path rejects");
        assert!(
            error.to_string().contains("Unsafe project-relative path"),
            "{unsafe_rel} should be rejected as unsafe, got {error}"
        );
    }
    // A normal project-relative media path still resolves under the project root.
    let safe = safe_project_path(temp.path(), "assets/images/x.png").expect("safe path resolves");
    assert!(safe.starts_with(temp.path()));
    assert!(safe.ends_with("assets/images/x.png"));

    let noisy = (0..100)
        .map(|index| format!("line {index} caf\u{e9}"))
        .collect::<Vec<_>>()
        .join("\n");
    let tail = bounded_tail(&noisy, 10, 37);

    assert!(tail.contains("caf\u{e9}"));
    assert!(!tail.contains("line 1 "));
}

#[test]
fn model_destinations_are_constrained_to_data_models() {
    let temp = tempdir().expect("tempdir creates");
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = temp.path().to_path_buf();
    let models_root = super::normalize_absolute_path(&temp.path().join("models"))
        .expect("models root normalizes");
    let fallback = temp.path().join("models").join("fallback");

    // model_download/model_import: a targetDir under data/models is accepted.
    let mut payload = JsonObject::new();
    payload.insert(
        "targetDir".to_owned(),
        Value::String(
            temp.path()
                .join("models")
                .join("z_image_turbo")
                .display()
                .to_string(),
        ),
    );
    let resolved = resolve_model_import_target(&settings, &payload, fallback.clone())
        .expect("destination under data/models is accepted");
    assert!(resolved.starts_with(&models_root));

    // No targetDir falls back to the supplied (contained) default.
    let resolved_fallback =
        resolve_model_import_target(&settings, &JsonObject::new(), fallback.clone())
            .expect("fallback under data/models is accepted");
    assert!(resolved_fallback.starts_with(&models_root));

    // A targetDir outside data/models is rejected (arbitrary write blocked).
    let mut escape = JsonObject::new();
    escape.insert(
        "targetDir".to_owned(),
        Value::String(
            temp.path()
                .join("ssh")
                .join("authorized_keys")
                .display()
                .to_string(),
        ),
    );
    let error = resolve_model_import_target(&settings, &escape, fallback)
        .expect_err("destination outside data/models is rejected");
    assert!(error.to_string().contains("data/models"));

    // model_convert: outputDir under data/models is accepted, traversal is rejected.
    let ok = resolve_model_convert_output(
        &settings,
        &temp
            .path()
            .join("models")
            .join("mlx")
            .join("wan")
            .display()
            .to_string(),
    )
    .expect("convert output under data/models is accepted");
    assert!(ok.starts_with(&models_root));

    let traversal = temp
        .path()
        .join("models")
        .join("..")
        .join("escape")
        .display()
        .to_string();
    let convert_error = resolve_model_convert_output(&settings, &traversal)
        .expect_err("convert output escaping data/models is rejected");
    assert!(convert_error.to_string().contains("data/models"));
}

#[cfg(unix)]
#[test]
fn lora_paths_resolve_symlinks_before_root_check() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    let lora_dir = data_dir.join("loras");
    let outside_dir = temp.path().join("outside");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    std::fs::create_dir_all(&outside_dir).expect("outside dir creates");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir.clone();

    let safe = lora_dir.join("safe.safetensors");
    std::fs::write(&safe, b"safe").expect("safe lora writes");
    let normalized =
        super::normalize_app_managed_lora_path(&settings, &safe).expect("safe lora accepted");
    assert_eq!(
        normalized,
        safe.canonicalize().expect("safe lora canonicalizes")
    );

    let outside = outside_dir.join("escape.safetensors");
    std::fs::write(&outside, b"outside").expect("outside lora writes");
    let link = lora_dir.join("escape-link.safetensors");
    std::os::unix::fs::symlink(&outside, &link).expect("symlink creates");

    let error = super::normalize_app_managed_lora_path(&settings, &link)
        .expect_err("symlink target outside managed roots rejects");
    assert!(error.to_string().contains("LoRA path must be inside"));
}

// sc-8877 / F-075: the write-target confinement helpers must canonicalize before the
// root check so a symlink planted under a managed root can't smuggle a write outside
// via a purely-lexical `starts_with`. Covers all five that used the weaker check.
#[cfg(unix)]
#[test]
fn app_managed_helpers_resolve_symlinks_before_root_check() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    let models_dir = data_dir.join("models");
    let loras_dir = data_dir.join("loras");
    let outside_dir = temp.path().join("outside");
    std::fs::create_dir_all(&models_dir).expect("models dir creates");
    std::fs::create_dir_all(&loras_dir).expect("loras dir creates");
    std::fs::create_dir_all(&outside_dir).expect("outside dir creates");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir.clone();

    // A symlink under a managed root pointing at an outside dir: a lexical
    // `starts_with` would accept it; canonicalization must make each helper reject it.
    let outside_target = outside_dir.join("escape");
    std::fs::create_dir_all(&outside_target).expect("outside target creates");

    // normalize_app_managed_path (write target confined to data_dir)
    let path_link = data_dir.join("path-escape");
    std::os::unix::fs::symlink(&outside_target, &path_link).expect("symlink creates");
    let err =
        super::normalize_app_managed_path(&settings, &path_link.display().to_string(), "Path")
            .expect_err("symlinked path escape rejects");
    assert!(err.to_string().contains("app-managed directory"));

    // normalize_app_managed_model_path (read source confined to data_dir/hf_cache)
    let model_link = models_dir.join("model-escape");
    std::os::unix::fs::symlink(&outside_target, &model_link).expect("symlink creates");
    let err = super::normalize_app_managed_model_path(
        &settings,
        &model_link.display().to_string(),
        "Model",
    )
    .expect_err("symlinked model escape rejects");
    assert!(err.to_string().contains("app-managed directory"));

    // resolve_lora_import_target (write target confined to data/loras)
    let lora_link = loras_dir.join("lora-escape");
    std::os::unix::fs::symlink(&outside_target, &lora_link).expect("symlink creates");
    let mut lora_payload = JsonObject::new();
    lora_payload.insert(
        "targetDir".to_owned(),
        Value::String(lora_link.display().to_string()),
    );
    let err = super::resolve_lora_import_target(&settings, &lora_payload, loras_dir.clone())
        .expect_err("symlinked lora targetDir escape rejects");
    assert!(err.to_string().contains("data/loras"));

    // resolve_model_import_target (write target confined to data/models)
    let import_link = models_dir.join("import-escape");
    std::os::unix::fs::symlink(&outside_target, &import_link).expect("symlink creates");
    let mut model_payload = JsonObject::new();
    model_payload.insert(
        "targetDir".to_owned(),
        Value::String(import_link.display().to_string()),
    );
    let err = super::resolve_model_import_target(&settings, &model_payload, models_dir.clone())
        .expect_err("symlinked model targetDir escape rejects");
    assert!(err.to_string().contains("data/models"));

    // resolve_model_convert_output (write target confined to data/models)
    let convert_link = models_dir.join("convert-escape");
    std::os::unix::fs::symlink(&outside_target, &convert_link).expect("symlink creates");
    let err = super::resolve_model_convert_output(&settings, &convert_link.display().to_string())
        .expect_err("symlinked convert outputDir escape rejects");
    assert!(err.to_string().contains("data/models"));

    // Sanity: a genuine dir under a managed root still resolves (not over-rejected).
    let real = models_dir.join("real_model");
    std::fs::create_dir_all(&real).expect("real model dir creates");
    super::normalize_app_managed_model_path(&settings, &real.display().to_string(), "Model")
        .expect("a real managed dir is still accepted");
}

// sc-8821 / F-019: payload-supplied weight filenames (`advanced.controlWeights.filename`,
// `advanced.pidCheckpoint.filename`) are joined under a resolved HF snapshot / app-cache
// dir, so they must be a single plain path component — traversal, absolute paths, and
// sub-paths are rejected before any join.
#[test]
fn weight_filenames_are_confined_to_plain_components() {
    // Plain filenames pass through (trimmed).
    assert_eq!(
        super::safe_weight_filename("model.safetensors", "advanced.controlWeights.filename")
            .expect("plain filename accepted"),
        "model.safetensors"
    );
    assert_eq!(
        super::safe_weight_filename("  model.safetensors  ", "advanced.controlWeights.filename")
            .expect("surrounding whitespace trimmed"),
        "model.safetensors"
    );

    for unsafe_name in [
        "../../etc/hosts",
        "..",
        ".",
        "",
        "/etc/hosts",
        "sub/dir.safetensors",
        "..\\..\\secrets.safetensors",
        "sub\\dir.safetensors",
        "model.safetensors/",
    ] {
        let error = super::safe_weight_filename(unsafe_name, "advanced.controlWeights.filename")
            .expect_err("non-plain filename rejected");
        assert!(
            error
                .to_string()
                .contains("advanced.controlWeights.filename"),
            "error names the offending field for {unsafe_name:?}: {error}"
        );
    }
}

// sc-8803 / F-002: LoRA/model import *source* paths are client-supplied over the
// unauthenticated jobs API; the worker must confine them before copying (or, for
// uploads, moving) the file into an app-listable directory.
#[test]
fn import_source_paths_are_confined_to_app_managed_roots() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    let staged_dir = data_dir
        .join("cache")
        .join("lora-uploads")
        .join("upload-abc");
    let loras_dir = data_dir.join("loras").join("existing");
    let outside_dir = temp.path().join("outside");
    std::fs::create_dir_all(&staged_dir).expect("staged dir creates");
    std::fs::create_dir_all(&loras_dir).expect("loras dir creates");
    std::fs::create_dir_all(&outside_dir).expect("outside dir creates");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir.clone();

    // Uploaded (move-mode) sources are accepted only from the staged-upload cache.
    let staged_file = staged_dir.join("lora.safetensors");
    std::fs::write(&staged_file, b"staged").expect("staged file writes");
    let mut uploaded = JsonObject::new();
    uploaded.insert("uploadedSourcePath".to_owned(), Value::Bool(true));
    let resolved = super::resolve_import_source_path(
        &settings,
        &uploaded,
        &staged_file.display().to_string(),
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect("staged upload source accepted");
    assert!(resolved.ends_with("lora.safetensors"));

    // Uploaded flag with a source elsewhere in data_dir is rejected: move mode
    // would otherwise delete arbitrary app-managed files.
    let installed_file = loras_dir.join("installed.safetensors");
    std::fs::write(&installed_file, b"installed").expect("installed file writes");
    let error = super::resolve_import_source_path(
        &settings,
        &uploaded,
        &installed_file.display().to_string(),
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect_err("uploaded source outside the staged-upload cache rejects");
    assert!(error
        .to_string()
        .contains("LoRA import sourcePath must be inside"));

    // Copy-mode sources under data_dir are accepted (re-import of an installed file).
    let copy_payload = JsonObject::new();
    super::resolve_import_source_path(
        &settings,
        &copy_payload,
        &installed_file.display().to_string(),
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect("data_dir source accepted for copy-mode import");

    // The model-import upload cache confines the same way.
    let model_staged = data_dir
        .join("cache")
        .join("model-uploads")
        .join("upload-def");
    std::fs::create_dir_all(&model_staged).expect("model staged dir creates");
    let model_file = model_staged.join("model.safetensors");
    std::fs::write(&model_file, b"model").expect("model file writes");
    super::resolve_import_source_path(
        &settings,
        &uploaded,
        &model_file.display().to_string(),
        "model-uploads",
        "Model import sourcePath",
    )
    .expect("staged model upload accepted");

    // An absolute path outside data_dir (the exfiltration primitive) is rejected
    // in both modes.
    let secret = outside_dir.join("id_rsa");
    std::fs::write(&secret, b"secret").expect("secret writes");
    for payload in [&uploaded, &copy_payload] {
        let error = super::resolve_import_source_path(
            &settings,
            payload,
            &secret.display().to_string(),
            "lora-uploads",
            "LoRA import sourcePath",
        )
        .expect_err("host path outside app-managed roots rejects");
        assert!(error
            .to_string()
            .contains("LoRA import sourcePath must be inside"));
    }

    // A `..` traversal that starts inside data_dir but escapes is rejected.
    let traversal = data_dir
        .join("loras")
        .join("..")
        .join("..")
        .join("outside")
        .join("id_rsa");
    super::resolve_import_source_path(
        &settings,
        &copy_payload,
        &traversal.display().to_string(),
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect_err("traversal escape rejects");

    // An empty source path is rejected.
    super::resolve_import_source_path(
        &settings,
        &copy_payload,
        "  ",
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect_err("empty source path rejects");
}

// Symlinks resolve before the root check, so a link planted inside data_dir cannot
// smuggle an outside file through the import copy.
#[cfg(unix)]
#[test]
fn import_source_symlink_escape_is_rejected() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    let loras_dir = data_dir.join("loras");
    let outside_dir = temp.path().join("outside");
    std::fs::create_dir_all(&loras_dir).expect("loras dir creates");
    std::fs::create_dir_all(&outside_dir).expect("outside dir creates");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir.clone();

    let outside_file = outside_dir.join("escape.safetensors");
    std::fs::write(&outside_file, b"outside").expect("outside file writes");
    let link = loras_dir.join("escape-link.safetensors");
    std::os::unix::fs::symlink(&outside_file, &link).expect("symlink creates");

    let error = super::resolve_import_source_path(
        &settings,
        &JsonObject::new(),
        &link.display().to_string(),
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect_err("symlink target outside managed roots rejects");
    assert!(error
        .to_string()
        .contains("LoRA import sourcePath must be inside"));
}

// sc-8898 / F-096: a missing import source now surfaces the friendly "LoRA source
// not found" message. Previously `canonicalize()` failed NotFound first and the
// `!exists()` branch that built this message was dead, so the user only saw the
// raw OS error.
#[tokio::test]
async fn missing_lora_import_source_reports_friendly_not_found() {
    let temp = tempdir().expect("tempdir creates");
    let missing = temp.path().join("does-not-exist.safetensors");
    let target_dir = temp.path().join("target");

    let error = import_lora_source_path(&missing, &target_dir, false)
        .await
        .expect_err("missing source errors");

    match error {
        WorkerError::Io(io_error) => {
            assert_eq!(io_error.kind(), std::io::ErrorKind::NotFound);
            assert!(
                io_error.to_string().contains("LoRA source not found"),
                "unexpected message: {io_error}"
            );
        }
        other => panic!("expected NotFound Io error, got {other:?}"),
    }
    // The target dir is not created for a missing source (the copy never runs).
    assert!(!target_dir.exists());
}

#[tokio::test]
async fn ffmpeg_runner_surfaces_bounded_stderr_from_failing_process() {
    let args = if cfg!(windows) {
        let command = (1..=30)
            .map(|index| format!("echo ffmpeg-line-{index} 1>&2"))
            .collect::<Vec<_>>()
            .join(" & ");
        vec![
            "cmd".to_owned(),
            "/C".to_owned(),
            format!("{command} & exit /B 7"),
        ]
    } else {
        vec![
            "sh".to_owned(),
            "-c".to_owned(),
            "for i in $(seq 1 30); do echo ffmpeg-line-$i >&2; done; exit 7".to_owned(),
        ]
    };

    let error = run_ffmpeg(args, None)
        .await
        .expect_err("non-zero process returns an error");

    match error {
        WorkerError::Engine(message) => {
            assert!(message.contains("ffmpeg-line-30"));
            assert!(!message.contains("ffmpeg-line-1"));
            assert!(message.len() <= 2000);
        }
        other => panic!("expected Engine, got {other:?}"),
    }
}

#[tokio::test]
async fn huggingface_snapshot_resolve_accepts_tree_and_sibling_shapes_with_auth() {
    let array_url = spawn_hf_stub(
        json!([
            { "type": "file", "path": "nested/model.safetensors", "size": 7 },
            { "type": "file", "path": "nested/model.ckpt", "size": 9 },
            { "type": "directory", "path": "nested" }
        ]),
        Some("hf_test"),
    )
    .await;
    let client = reqwest::Client::new();
    let array_settings = test_settings(array_url, Some("hf_test"));

    let snapshot = HuggingFaceSnapshot::resolve(
        &client,
        &array_settings,
        "owner/model",
        "main",
        &["*.safetensors".to_owned()],
    )
    .await
    .expect("tree snapshot resolves");

    assert_eq!(snapshot.files.len(), 1);
    assert_eq!(snapshot.files[0].path, "nested/model.safetensors");
    assert_eq!(snapshot.total_bytes(), Some(7));

    let siblings_url = spawn_hf_stub(
        json!({
            "siblings": [
                { "rfilename": "adapter.safetensors", "size": "5" }
            ]
        }),
        None,
    )
    .await;
    let siblings_settings = test_settings(siblings_url, None);

    let snapshot = HuggingFaceSnapshot::resolve(
        &client,
        &siblings_settings,
        "owner/lora",
        "main",
        &["*.safetensors".to_owned()],
    )
    .await
    .expect("siblings snapshot resolves");

    assert_eq!(snapshot.files.len(), 1);
    assert_eq!(snapshot.files[0].path, "adapter.safetensors");
    assert_eq!(snapshot.total_bytes(), Some(5));
}

#[derive(Clone)]
struct HfStubState {
    payload: serde_json::Value,
    token: Option<String>,
}

async fn spawn_hf_stub(payload: serde_json::Value, token: Option<&str>) -> String {
    let state = HfStubState {
        payload,
        token: token.map(str::to_owned),
    };
    let app = Router::new()
        .route("/api/models/:owner/:repo/tree/:revision", get(hf_stub))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn hf_stub(State(state): State<HfStubState>, headers: HeaderMap) -> Response {
    if let Some(token) = &state.token {
        let expected = format!("Bearer {token}");
        let authorized = headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            == Some(expected.as_str());
        if !authorized {
            return (
                AxumStatusCode::UNAUTHORIZED,
                Json(json!({ "error": "missing token" })),
            )
                .into_response();
        }
    }
    Json(state.payload).into_response()
}

#[derive(Clone)]
struct BinaryStubState {
    bytes: Vec<u8>,
    status: AxumStatusCode,
    cancel_requested: bool,
}

async fn spawn_binary_stub(bytes: Vec<u8>) -> String {
    spawn_binary_stub_with_options(bytes, AxumStatusCode::OK, false).await
}

async fn spawn_binary_stub_with_options(
    bytes: Vec<u8>,
    status: AxumStatusCode,
    cancel_requested: bool,
) -> String {
    let state = BinaryStubState {
        bytes,
        status,
        cancel_requested,
    };
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_stub))
        .route("/api/v1/jobs/:job_id/progress", post(progress_stub))
        .route("/*path", get(binary_stub).head(binary_head_stub))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn binary_stub(State(state): State<BinaryStubState>, headers: HeaderMap) -> Response {
    let length = state.bytes.len();
    if headers
        .get(axum::http::header::RANGE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == format!("bytes={length}-"))
    {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = AxumStatusCode::RANGE_NOT_SATISFIABLE;
        response.headers_mut().insert(
            axum::http::header::CONTENT_RANGE,
            axum::http::HeaderValue::from_str(&format!("bytes */{length}"))
                .expect("content range header"),
        );
        return response;
    }
    if let Some(start) = headers
        .get(axum::http::header::RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("bytes="))
        .and_then(|value| value.strip_suffix('-'))
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|start| *start < length)
    {
        let body = state.bytes[start..].to_vec();
        let mut response = body.into_response();
        *response.status_mut() = AxumStatusCode::PARTIAL_CONTENT;
        response.headers_mut().insert(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from_str(&(length - start).to_string())
                .expect("content length header"),
        );
        response.headers_mut().insert(
            axum::http::header::CONTENT_RANGE,
            axum::http::HeaderValue::from_str(&format!("bytes {start}-{}/{length}", length - 1))
                .expect("content range header"),
        );
        return response;
    }
    let mut response = state.bytes.into_response();
    *response.status_mut() = state.status;
    response.headers_mut().insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_str(&length.to_string()).expect("content length header"),
    );
    response
}

async fn binary_head_stub(
    State(state): State<BinaryStubState>,
    axum::extract::Path(path): axum::extract::Path<String>,
) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = state.status;
    let headers = response.headers_mut();
    headers.insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_str(&state.bytes.len().to_string())
            .expect("content length header"),
    );
    // Mirror Hugging Face's resolve metadata so download_snapshot can name blobs by
    // etag and record the commit (sc-1904).
    let last_segment = path.rsplit('/').next().unwrap_or("blob");
    headers.insert(
        axum::http::header::ETAG,
        axum::http::HeaderValue::from_str(&format!("\"etag-{last_segment}\""))
            .expect("etag header"),
    );
    headers.insert(
        "x-repo-commit",
        axum::http::HeaderValue::from_static("stubcommit"),
    );
    response
}

async fn job_stub(
    State(state): State<BinaryStubState>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Response {
    Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
}

async fn progress_stub(
    State(state): State<BinaryStubState>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Response {
    Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
}

fn job_snapshot_json(job_id: &str, cancel_requested: bool) -> Value {
    json!({
        "id": job_id,
        "type": "lora_import",
        "status": "running",
        "projectId": null,
        "projectName": null,
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": "test-worker",
        "progress": 0.1,
        "stage": "importing",
        "message": "running",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": cancel_requested,
        "createdAt": "2026-05-18T00:00:00Z",
        "updatedAt": "2026-05-18T00:00:00Z",
        "startedAt": null,
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": null
    })
}

fn worker_snapshot_json(worker_id: &str) -> Value {
    json!({
        "id": worker_id,
        "gpuId": "cpu",
        "gpuName": null,
        "status": "busy",
        "currentJobId": "job-1",
        "capabilities": [],
        "loadedModels": [],
        "registeredAt": "2026-07-01T00:00:00Z",
        "lastSeenAt": "2026-07-01T00:00:00Z"
    })
}

/// sc-8806 — stub for the tick-driven download-cancel path. Counts GETs of the
/// job snapshot (the chunk loop must never poll it), serves the progress POST
/// with a configurable `cancelRequested` (the snapshot the tick reuses for its
/// cancel decision), answers worker heartbeats, and serves the binary either as
/// a short multi-chunk body or as a stream that stalls after the first chunk —
/// so only the interval tick can observe a cancel.
#[derive(Clone)]
struct CancelTickStubState {
    job_gets: Arc<AtomicUsize>,
    progress_cancel_requested: bool,
    stall_after_first_chunk: bool,
}

async fn spawn_cancel_tick_stub(state: CancelTickStubState) -> String {
    use futures_util::StreamExt;

    async fn job_route(
        State(state): State<CancelTickStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
    ) -> Response {
        state.job_gets.fetch_add(1, Ordering::SeqCst);
        // Deliberately NOT canceled: only the progress POST snapshot says
        // canceled, so a cancel can only come from reusing that snapshot.
        Json(job_snapshot_json(&job_id, false)).into_response()
    }
    async fn progress_route(
        State(state): State<CancelTickStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
    ) -> Response {
        Json(job_snapshot_json(&job_id, state.progress_cancel_requested)).into_response()
    }
    async fn heartbeat_route(
        axum::extract::Path(worker_id): axum::extract::Path<String>,
    ) -> Response {
        Json(worker_snapshot_json(&worker_id)).into_response()
    }
    async fn binary_route(State(state): State<CancelTickStubState>) -> Response {
        let chunks = futures_util::stream::iter(vec![
            Ok::<_, std::convert::Infallible>(axum::body::Bytes::from_static(b"url-")),
            Ok(axum::body::Bytes::from_static(b"lora")),
        ]);
        if state.stall_after_first_chunk {
            Body::from_stream(chunks.chain(futures_util::stream::pending())).into_response()
        } else {
            Body::from_stream(chunks).into_response()
        }
    }

    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .route("/*path", get(binary_route))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

fn test_settings(huggingface_base_url: String, huggingface_token: Option<&str>) -> Settings {
    Settings {
        api_url: "http://127.0.0.1:8000".to_owned(),
        access_token: None,
        data_dir: PathBuf::from("data"),
        config_dir: PathBuf::from("config"),
        worker_id: "test-worker".to_owned(),
        gpu_id: "cpu".to_owned(),
        is_child_worker: true,
        poll_seconds: 1,
        heartbeat_seconds: 5,
        shutdown_timeout_seconds: 1,
        huggingface_base_url,
        huggingface_token: huggingface_token.map(str::to_owned),
        credentials: Vec::new(),
        max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
        allow_private_lora_urls: true,
        utility_workers: 1,
        backend_mlx_enabled: true,
        backend_candle_enabled: false,
        gpu_memory_limit_bytes: 0,
    }
}

#[test]
fn idle_heartbeat_is_due_immediately_then_waits_for_interval() {
    let mut heartbeat = IdleHeartbeat::new(Duration::from_secs(60));

    assert!(heartbeat.should_send());
    heartbeat.mark_sent();
    assert!(!heartbeat.should_send());
}

#[test]
fn idle_heartbeat_allows_immediate_resend_when_interval_is_zero() {
    let mut heartbeat = IdleHeartbeat::new(Duration::ZERO);

    assert!(heartbeat.should_send());
    heartbeat.mark_sent();
    assert!(heartbeat.should_send());
}

#[test]
fn idle_heartbeat_can_be_forced_due_after_work() {
    let mut heartbeat = IdleHeartbeat::new(Duration::from_secs(60));

    heartbeat.mark_sent();
    assert!(!heartbeat.should_send());
    heartbeat.mark_due();
    assert!(heartbeat.should_send());
}

fn spawn_exit_child() -> tokio::process::Child {
    let mut command = if cfg!(windows) {
        let mut command = tokio::process::Command::new("cmd");
        command.args(["/C", "exit /B 0"]);
        command
    } else {
        let mut command = tokio::process::Command::new("sh");
        command.args(["-c", "exit 0"]);
        command
    };
    command
        .stdout(StdStdio::null())
        .stderr(StdStdio::null())
        .spawn()
        .expect("test child starts")
}

fn spawn_sleep_child() -> tokio::process::Child {
    let mut command = if cfg!(windows) {
        let mut command = tokio::process::Command::new("cmd");
        command.args(["/C", "ping -n 30 127.0.0.1 >NUL"]);
        command
    } else {
        let mut command = tokio::process::Command::new("sh");
        command.args(["-c", "sleep 30"]);
        command
    };
    command
        .stdout(StdStdio::null())
        .stderr(StdStdio::null())
        .spawn()
        .expect("test child starts")
}

/// sc-4174 — the in-band cancel poll for long generations must only cancel on
/// a confirmed user cancel. A transient API failure (on the GET, or on the
/// Canceled-status POST inside check_cancel) is tolerated and retried on the
/// next poll instead of aborting a multi-minute run.
#[derive(Clone)]
struct CancelPollStubState {
    get_status: AxumStatusCode,
    cancel_requested: bool,
    post_status: AxumStatusCode,
}

async fn spawn_cancel_poll_stub(state: CancelPollStubState) -> String {
    async fn job_route(
        State(state): State<CancelPollStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
    ) -> Response {
        if state.get_status != AxumStatusCode::OK {
            return (state.get_status, "stub GET failure").into_response();
        }
        Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
    }
    async fn progress_route(
        State(state): State<CancelPollStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
    ) -> Response {
        if state.post_status != AxumStatusCode::OK {
            return (state.post_status, "stub POST failure").into_response();
        }
        Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
    }
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

/// sc-5515 — the in-loop image cancel poller uses a CHECK-ONLY peek that reads
/// `cancel_requested` without posting any terminal status. The terminal Canceled
/// is posted by `consume_gen_events` only after the blocking generation actually
/// stops, so the worker row isn't freed (and the next queued job isn't misled)
/// while the in-flight image is still rendering. `post_status` is wired to fail
/// here to prove the peek never touches the progress route.
async fn cancel_peek_with(get_status: AxumStatusCode, cancel_requested: bool) -> bool {
    let base_url = spawn_cancel_poll_stub(CancelPollStubState {
        get_status,
        cancel_requested,
        post_status: AxumStatusCode::INTERNAL_SERVER_ERROR,
    })
    .await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    cancel_requested_peek(&api, "job-1").await
}

#[tokio::test]
async fn cancel_peek_reports_confirmed_cancel_without_posting() {
    assert!(
        cancel_peek_with(AxumStatusCode::OK, true).await,
        "a confirmed cancel request must be reported by the check-only peek"
    );
}

#[tokio::test]
async fn cancel_peek_false_when_not_requested() {
    assert!(!cancel_peek_with(AxumStatusCode::OK, false).await);
}

#[tokio::test]
async fn cancel_peek_tolerates_transient_get_errors() {
    assert!(
        !cancel_peek_with(AxumStatusCode::INTERNAL_SERVER_ERROR, true).await,
        "a transient GET failure must not read as a user cancel"
    );
}

// sc-5516 — the in-loop video/training/detail cancel pollers DEFER the terminal `Canceled`
// to actual-stop: at acknowledgement they only trip the engine flag and post a NON-terminal
// "Cancelling…" update (so the worker row isn't freed while the in-flight step is still
// running). This stub captures every progress POST body so a test can assert the
// acknowledgement status is `running`, not the terminal `canceled`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn spawn_progress_capture_stub() -> (String, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
    use std::sync::{Arc, Mutex};
    type Posts = Arc<Mutex<Vec<Value>>>;
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    async fn progress_route(
        State(posts): State<Posts>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
        Json(body): Json<Value>,
    ) -> Response {
        posts.lock().expect("posts lock").push(body);
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    let posts: Posts = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .with_state(posts.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    (format!("http://{address}"), posts)
}

/// sc-8845 (F-043) — capture stub whose job GET reports NO user cancel, so the only cancel that can
/// fire in `run_placeholder_job` is the process-shutdown flag. Records every progress POST body and
/// answers heartbeats.
async fn spawn_no_user_cancel_capture_stub(
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
    use std::sync::{Arc, Mutex};
    type Posts = Arc<Mutex<Vec<Value>>>;
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        // No user cancel — a shutdown-driven cancel must be the ONLY thing that can trip.
        Json(job_snapshot_json(&job_id, false)).into_response()
    }
    async fn progress_route(
        State(posts): State<Posts>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
        Json(body): Json<Value>,
    ) -> Response {
        posts.lock().expect("posts lock").push(body);
        Json(job_snapshot_json(&job_id, false)).into_response()
    }
    async fn heartbeat_route(
        axum::extract::Path(worker_id): axum::extract::Path<String>,
    ) -> Response {
        Json(worker_snapshot_json(&worker_id)).into_response()
    }
    let posts: Posts = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .with_state(posts.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    (format!("http://{address}"), posts)
}

fn placeholder_job_snapshot() -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job-1",
        "type": "placeholder",
        "status": "running",
        "projectId": null,
        "projectName": null,
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": "test-worker",
        "progress": 0.0,
        "stage": "queued",
        "message": "queued",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": false,
        "createdAt": "2026-07-03T00:00:00Z",
        "updatedAt": "2026-07-03T00:00:00Z",
        "startedAt": null,
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": null
    }))
    .expect("placeholder job snapshot deserializes")
}

/// sc-8845 (F-043) — a process shutdown mid-job must NOT drop the in-flight future without a
/// terminal write. When the shared shutdown `CancelFlag` is already tripped, `run_placeholder_job`
/// must post a terminal `Canceled` progress update (status `canceled`) and return
/// `WorkerError::Canceled` — a prompt, specific terminal state — instead of running on or leaving
/// the job `running` for the 90s stale sweep to relabel `interrupted`. It must do so even with the
/// user-cancel GET reporting NOT canceled, proving the shutdown flag (not a user cancel) is the
/// trigger. Discriminator: under the old behavior (no shutdown checkpoint) the job would run its
/// first stage and post a `preparing`/`running` update instead of `canceled`.
#[tokio::test]
async fn run_placeholder_job_posts_terminal_canceled_on_shutdown_flag() {
    let (base_url, posts) = spawn_no_user_cancel_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let job = placeholder_job_snapshot();

    // Process shutdown already observed by the loop supervisor before the first stage runs.
    let shutdown = gen_core::CancelFlag::new();
    shutdown.cancel();

    let result = super::run_placeholder_job(&api, &settings, &job, &shutdown).await;

    assert!(
        matches!(result, Err(WorkerError::Canceled(_))),
        "a tripped shutdown flag must surface as WorkerError::Canceled, not a completed/failed job"
    );
    let posts = posts.lock().expect("posts lock");
    assert_eq!(
        posts.len(),
        1,
        "exactly one progress write is posted — the terminal Canceled — with no work stages first"
    );
    assert_eq!(
        posts[0]["status"], "canceled",
        "the shutdown-during-job write must be the TERMINAL Canceled state (not left running)"
    );
    assert!(
        posts[0]["message"]
            .as_str()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .contains("shut down"),
        "the terminal message should attribute the cancel to worker shutdown, not a user cancel"
    );
}

/// sc-8845 (F-043) — control: an UN-tripped shutdown flag must not spuriously cancel a clean run.
/// The placeholder job should proceed past its first stage (posting a non-terminal `preparing`
/// update) when neither a user cancel nor a shutdown is present, so the new shutdown checkpoint
/// can't false-positive a normal job to `canceled`.
#[tokio::test]
async fn run_placeholder_job_proceeds_when_shutdown_flag_untripped() {
    let (base_url, posts) = spawn_no_user_cancel_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let job = placeholder_job_snapshot();

    let shutdown = gen_core::CancelFlag::new(); // never tripped

    // Drive only the first stage: the placeholder loop sleeps 1.5s per stage, so a short timeout
    // lets us observe the first non-terminal write without waiting out the whole job.
    let run = super::run_placeholder_job(&api, &settings, &job, &shutdown);
    let _ = tokio::time::timeout(Duration::from_millis(400), run).await;

    let posts = posts.lock().expect("posts lock");
    assert!(
        !posts.is_empty(),
        "a job with no cancel and no shutdown must make progress"
    );
    assert_ne!(
        posts[0]["status"], "canceled",
        "an untripped shutdown flag must NOT cancel a clean run — the first write is a work stage"
    );
}

/// sc-5516 — `begin_video_cancel` trips the engine cancel flag and posts the
/// cancel acknowledgement as a NON-terminal `running` "Cancelling…" update. The
/// terminal `Canceled` (which frees the worker row) is posted by `generate_video`
/// only after the blocking generation actually stops, so the next queued job waits
/// until the GPU is genuinely free.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[tokio::test]
async fn begin_video_cancel_trips_flag_and_stays_non_terminal() {
    let (base_url, posts) = spawn_progress_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let cancel = gen_core::CancelFlag::new();
    crate::video_jobs::begin_video_cancel(&api, "job-1", &cancel, "mlx").await;
    assert!(
        cancel.is_cancelled(),
        "begin_video_cancel must trip the engine cancel flag"
    );
    let posts = posts.lock().expect("posts lock");
    assert_eq!(
        posts.len(),
        1,
        "exactly one acknowledgement update is posted"
    );
    assert_eq!(
        posts[0]["status"], "running",
        "the cancel acknowledgement must stay NON-terminal — the terminal Canceled is \
         deferred to actual-stop (sc-5516)"
    );
    assert!(
        posts[0]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("Cancelling"),
        "the acknowledgement message should read as Cancelling…"
    );
}

/// sc-5516 — the training sibling of the above: `begin_training_cancel` trips the
/// flag and acknowledges with a NON-terminal `running` update; the terminal
/// `Canceled` is posted by `consume_training_events` after training stops. Compiled on the macOS MLX
/// path AND the off-Mac candle training lane (sc-7817), where `begin_training_cancel` is also linked.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[tokio::test]
async fn begin_training_cancel_trips_flag_and_stays_non_terminal() {
    let (base_url, posts) = spawn_progress_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let cancel = gen_core::CancelFlag::new();
    crate::training_jobs::begin_training_cancel(&api, "job-1", &cancel, "mlx").await;
    assert!(
        cancel.is_cancelled(),
        "begin_training_cancel must trip the engine cancel flag"
    );
    let posts = posts.lock().expect("posts lock");
    assert_eq!(
        posts.len(),
        1,
        "exactly one acknowledgement update is posted"
    );
    assert_eq!(
        posts[0]["status"], "running",
        "the cancel acknowledgement must stay NON-terminal (sc-5516)"
    );
    assert!(
        posts[0]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("Cancelling"),
        "the acknowledgement message should read as Cancelling…"
    );
}

// sc-9646 (sc-9595 follow-up) — the streaming SeedVR2 upscale writes the whole upscaled PNG sequence
// to a scratch dir before the final encode, so its disk footprint is unbounded after the sc-8829
// host-RAM cap was removed. `check_seedvr2_output_disk` is a fail-loud-before-decode preflight that
// mirrors the removed RAM guard: estimate the output footprint, compare against a generous fraction
// of the scratch volume's free space, and reject a clip that would fill the disk.

#[test]
fn seedvr2_output_bytes_scales_with_frames_and_pixels() {
    // Raw RGB8 upper bound: 1 frame at 100×100 = 100·100·3 = 30_000 bytes.
    assert_eq!(
        crate::video_jobs::seedvr2_estimated_output_bytes(1, 100, 100),
        30_000
    );
    // Linear in frame count and in each dimension.
    assert_eq!(
        crate::video_jobs::seedvr2_estimated_output_bytes(10, 100, 100),
        300_000
    );
    assert_eq!(
        crate::video_jobs::seedvr2_estimated_output_bytes(1, 200, 100),
        60_000
    );
    // A degenerate zero dimension / frame count yields zero (the guard treats that as "unknown").
    assert_eq!(
        crate::video_jobs::seedvr2_estimated_output_bytes(0, 3840, 2160),
        0
    );
    assert_eq!(
        crate::video_jobs::seedvr2_estimated_output_bytes(100, 0, 2160),
        0
    );
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn seedvr2_disk_guard_passes_a_tiny_clip_and_is_noop_when_unknown() {
    let dir = tempdir().expect("tempdir");
    // A few small frames fit any real scratch volume — must pass.
    crate::video_jobs::check_seedvr2_output_disk(dir.path(), 8, 64, 64)
        .expect("a tiny output footprint fits available disk");
    // Unknown frame count / dims short-circuit to Ok (the guard defers to the real count).
    crate::video_jobs::check_seedvr2_output_disk(dir.path(), 0, 3840, 2160)
        .expect("zero frames is a no-op guard");
    crate::video_jobs::check_seedvr2_output_disk(dir.path(), 1_000, 0, 0)
        .expect("zero dims is a no-op guard");
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn seedvr2_disk_guard_rejects_an_impossibly_large_clip() {
    let dir = tempdir().expect("tempdir");
    // 100M frames at 4096×4096 ≈ 4.7 exabytes of PNG — no scratch volume holds that, so the guard
    // must fail loud with an actionable message (unless the free-space probe is unavailable, in which
    // case the guard is a documented no-op and we skip the assertion).
    if crate::video_jobs::available_disk_bytes(dir.path()).is_some() {
        let error =
            crate::video_jobs::check_seedvr2_output_disk(dir.path(), 100_000_000, 4096, 4096)
                .expect_err("an exabyte-scale output must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("Not enough disk space"),
            "the error names the disk-space limit: {message}"
        );
        assert!(
            message.contains("Trim the clip"),
            "the error is actionable (suggests trimming): {message}"
        );
    }
}

// sc-8917 (F-115) — the shared batched-analysis scaffold DEFERS the terminal `Canceled` to
// actual-stop, exactly like the training/video pollers. A cancel observed on the heartbeat-tick
// poll must only trip the engine flag (so the still-running embed loop bails at its next per-item
// check); the terminal `Canceled` is posted only AFTER the blocking task has stopped, so the worker
// row isn't freed — and the scheduler isn't told a busy worker is free — while the GPU is still
// finishing the current embed. Regression: the old scaffold called `check_cancel` on the tick, which
// posted the terminal `Canceled` at acknowledgement time.

/// A capture stub for the analysis scaffold: the job GET/heartbeat report a user cancel (so the
/// interval-tick poll trips), every progress POST body is recorded, and the heartbeat route answers.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn spawn_analysis_cancel_stub() -> (String, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
    use std::sync::{Arc, Mutex};
    type Posts = Arc<Mutex<Vec<Value>>>;
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        // Report the user cancel so `cancel_requested_peek` on the first (immediate) interval tick
        // trips the flag.
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    async fn progress_route(
        State(posts): State<Posts>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
        Json(body): Json<Value>,
    ) -> Response {
        posts.lock().expect("posts lock").push(body);
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    async fn heartbeat_route(
        axum::extract::Path(worker_id): axum::extract::Path<String>,
    ) -> Response {
        Json(worker_snapshot_json(&worker_id)).into_response()
    }
    let posts: Posts = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .with_state(posts.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    (format!("http://{address}"), posts)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[tokio::test]
async fn batched_analysis_defers_terminal_canceled_until_the_task_stops() {
    let (base_url, posts) = spawn_analysis_cancel_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    // A short heartbeat so the interval fires promptly; the first `interval.tick()` resolves
    // immediately regardless, which is what drives the cancel poll here.
    settings.heartbeat_seconds = 5;
    let api = ApiClient::new(&settings);
    let job: JobSnapshot = serde_json::from_value(job_snapshot_json("job-1", true))
        .expect("job snapshot deserializes");

    let cancel = gen_core::CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<usize>(4);
    // The blocking task models a real embed loop: it sends NO item (so `rx.recv()` stays pending and
    // the immediate interval tick wins the `select!`), waits for the flag, and — once tripped —
    // returns `Canceled` exactly as the CLIP/face loops do at their per-item `is_cancelled()` check.
    let task_cancel = cancel.clone();
    let blocking = tokio::task::spawn_blocking(move || -> super::WorkerResult<Vec<u32>> {
        // Keep `tx` alive until we bail so the channel isn't closed early (which would also break the
        // loop, but via the `None` arm rather than the cancel path we want to exercise).
        let _tx = tx;
        for _ in 0..2_000 {
            if task_cancel.is_cancelled() {
                return Err(WorkerError::Canceled(
                    "Analysis canceled by user.".to_owned(),
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(vec![1, 2, 3])
    });

    let cfg = super::AnalysisJobConfig {
        endpoint_suffix: "analysis-embeddings",
        space: "clip-vit-l14",
        cancel_message: "Analysis canceled by user.",
        saving_message: "Saving embeddings.",
        join_error_label: "analysis task join",
        item_message: &|index, total| format!("Analyzed image {} of {}.", index + 1, total),
    };
    let mut records_payload_calls = 0usize;
    let result = super::run_batched_analysis_job(
        &api,
        &settings,
        &job,
        &cfg,
        3,
        "mlx",
        cancel,
        rx,
        blocking,
        |records: &[u32]| {
            records_payload_calls += 1;
            records.iter().map(|r| json!(r)).collect()
        },
        |_records, _stored| unreachable!("a canceled job must not build a completed update"),
    )
    .await;

    assert!(
        matches!(result, Err(WorkerError::Canceled(_))),
        "a canceled analysis job returns WorkerError::Canceled, got {result:?}"
    );
    assert_eq!(
        records_payload_calls, 0,
        "the records → sidecar payload must never be built on cancel (no embeddings are POSTed)"
    );
    let posts = posts.lock().expect("posts lock");
    // Exactly one terminal `canceled`, and it is the FINAL post — never a mid-run acknowledgement.
    let canceled: Vec<&Value> = posts.iter().filter(|p| p["status"] == "canceled").collect();
    assert_eq!(
        canceled.len(),
        1,
        "exactly one terminal canceled is posted, got posts: {posts:?}"
    );
    assert_eq!(
        posts.last().map(|p| &p["status"]),
        Some(&json!("canceled")),
        "the terminal canceled must be the LAST post (deferred to after the task stopped)"
    );
    // The scaffold never reached the saving stage (the records POST) on the cancel path.
    assert!(
        posts.iter().all(|p| p["status"] != "saving"),
        "a canceled job must not post the Saving stage: {posts:?}"
    );
}

// sc-8804 (F-003) — the shared cancel-and-join guard. Every streaming job consumer binds its
// blocking GPU/training task to its `CancelFlag` through `CancelJoinGuard`; on any early return
// (a transient progress/heartbeat POST failure or a 409 stale-sweep reclaim propagating through
// `?`) the guard must trip the flag AND abort the task, so the GPU work stops instead of leaking
// alongside the next claimed job. The happy path reclaims the handle via `into_handle`, disarming
// the guard so a clean completion never aborts.

/// Dropping the guard early (the `?`-return shape) trips the engine cancel flag and aborts the
/// still-running task — the core F-003 defect, tested in isolation.
#[tokio::test]
async fn cancel_join_guard_trips_flag_and_aborts_on_early_drop() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let cancel = gen_core::CancelFlag::new();
    // A task that would run "forever" unless aborted; it flips `ran_to_completion` only if it ever
    // reaches its end. We hold a cancel-flag clone the task polls so a cooperative bail is possible,
    // but the guard's abort is what must stop a task that never checks.
    let completed = Arc::new(AtomicBool::new(false));
    let task_completed = completed.clone();
    let task_cancel = cancel.clone();
    let handle = tokio::task::spawn(async move {
        for _ in 0..1_000 {
            if task_cancel.is_cancelled() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        task_completed.store(true, Ordering::SeqCst);
    });

    {
        let _guard: crate::CancelJoinGuard<gen_core::CancelFlag, ()> =
            crate::CancelJoinGuard::new(cancel.clone(), handle);
        // Simulate the early `?` return: the guard goes out of scope here without `into_handle`.
    }

    assert!(
        cancel.is_cancelled(),
        "dropping the guard early must trip the engine cancel flag"
    );
    // The abort tears the task down; give the runtime a tick to reap it, then confirm it never
    // ran to completion (i.e. it was stopped, not left running).
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !completed.load(Ordering::SeqCst),
        "the guard must abort the task on early drop — it must not run to completion"
    );
}

/// Reclaiming the handle via `into_handle` disarms the guard: a clean completion neither cancels
/// nor aborts, and the task's value is returned intact.
#[tokio::test]
async fn cancel_join_guard_into_handle_disarms_the_guard() {
    let cancel = gen_core::CancelFlag::new();
    let handle = tokio::task::spawn(async { 42_u32 });
    let guard: crate::CancelJoinGuard<gen_core::CancelFlag, u32> =
        crate::CancelJoinGuard::new(cancel.clone(), handle);
    let value = guard.into_handle().await.expect("task joins cleanly");
    assert_eq!(value, 42, "the reclaimed handle yields the task's value");
    assert!(
        !cancel.is_cancelled(),
        "a disarmed guard must never trip the cancel flag on the success path"
    );
}

/// A stub whose worker-heartbeat POST fails (a 409/stale-sweep-class error), while its job GET
/// reports no cancel. `run_blocking_with_heartbeat` posts a `Busy` heartbeat every interval; the
/// failing POST propagates through `?`, which must trip the cancel flag and abort the long task
/// via the guard rather than returning while the task keeps running (sc-8804, F-003).
async fn spawn_failing_heartbeat_stub() -> String {
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        Json(job_snapshot_json(&job_id, false)).into_response()
    }
    async fn progress_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        Json(job_snapshot_json(&job_id, false)).into_response()
    }
    async fn heartbeat_route() -> Response {
        // Mimic a 409 the stale-sweep produces once the job is reclaimed — a non-transport API
        // error that heartbeat() propagates (it swallows only transport errors).
        (AxumStatusCode::CONFLICT, "stub heartbeat conflict").into_response()
    }
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .with_state(());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

#[tokio::test]
async fn run_blocking_with_heartbeat_aborts_task_when_heartbeat_post_fails() {
    let base_url = spawn_failing_heartbeat_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    // Shortest interval the clamp allows, so the failing heartbeat fires quickly.
    settings.heartbeat_seconds = 5;
    let api = ApiClient::new(&settings);

    let cancel = gen_core::CancelFlag::new();
    let task_cancel = cancel.clone();
    // A long "GPU" task that only stops if its cancel flag is tripped — modeling a denoise/training
    // run that would otherwise keep burning the device after the consumer returns.
    let task = tokio::task::spawn(async move {
        for _ in 0..600 {
            if task_cancel.is_cancelled() {
                return Ok::<(), WorkerError>(());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok(())
    });

    let result = crate::run_blocking_with_heartbeat(
        &api,
        &settings,
        "job-1",
        Some(cancel.clone()),
        "canceled",
        "f003 test task",
        crate::no_cancel_ack(),
        task,
    )
    .await;

    assert!(
        result.is_err(),
        "the failing heartbeat POST must propagate as an error"
    );
    assert!(
        cancel.is_cancelled(),
        "on the POST-failure early return the guard must trip the task's cancel flag (sc-8804)"
    );
}

/// sc-8804 (F-003) — the CORE bounded-join assertion, in the real `spawn_blocking` shape the
/// converted consumers use. `JoinHandle::abort()` is INERT on an already-running blocking task, so
/// a drop-and-run teardown (trip flag, abort, return immediately) would return BEFORE the blocking
/// GPU task observes the flag and winds down — and the worker would claim the next job with the
/// device still busy. The fix's `cancel_and_join` must AWAIT the task, so by the time
/// `run_blocking_with_heartbeat` returns the blocking task has ALREADY wound down.
///
/// Discriminator (mutation check): this test asserts `wound_down == true` AT RETURN TIME. Under the
/// old drop-and-run behavior (abort-only, no awaited join) the flag is tripped but the consumer
/// returns before the `spawn_blocking` thread reaches its next cancel poll, so `wound_down` is still
/// `false` at return — the assertion fails. Only an awaited bounded-join makes it pass. The prior
/// test above used `tokio::task::spawn` (where `abort()` DOES stop the task), which is exactly why
/// it could not catch the blocking-task gap.
#[tokio::test]
async fn run_blocking_with_heartbeat_waits_for_blocking_task_winddown_on_post_failure() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let base_url = spawn_failing_heartbeat_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    settings.heartbeat_seconds = 5;
    let api = ApiClient::new(&settings);

    let cancel = gen_core::CancelFlag::new();
    let task_cancel = cancel.clone();
    // Set by the blocking task ONLY when it observes the cancel flag and winds down. The consumer
    // must not return until this is true.
    let wound_down = Arc::new(AtomicBool::new(false));
    let task_wound_down = wound_down.clone();
    // A genuine `spawn_blocking` task (a synchronous busy loop, like a denoise/detect thread) that
    // polls its cancel flag between "steps". `abort()` cannot stop it — only the flag can — so an
    // awaited join is the only way the consumer can observe its wind-down before returning.
    let task = tokio::task::spawn_blocking(move || {
        for _ in 0..2_000 {
            if task_cancel.is_cancelled() {
                task_wound_down.store(true, Ordering::SeqCst);
                return Ok::<(), WorkerError>(());
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        // Reached only if never canceled: still mark wound-down so a hang is distinguishable.
        task_wound_down.store(true, Ordering::SeqCst);
        Ok(())
    });

    let result = crate::run_blocking_with_heartbeat(
        &api,
        &settings,
        "job-1",
        Some(cancel.clone()),
        "canceled",
        "f003 blocking winddown task",
        crate::no_cancel_ack(),
        task,
    )
    .await;

    assert!(
        result.is_err(),
        "the failing heartbeat POST must propagate as an error"
    );
    assert!(
        cancel.is_cancelled(),
        "the teardown must trip the blocking task's cancel flag (sc-8804)"
    );
    // The load-bearing assertion: the consumer AWAITED the bounded join, so the blocking task has
    // already wound down by the time we get here. A drop-and-run teardown fails this.
    assert!(
        wound_down.load(Ordering::SeqCst),
        "run_blocking_with_heartbeat must bounded-join the blocking task — it must NOT return \
         while the task is still running (sc-8804 F-003); the task had not wound down at return"
    );
}

/// sc-8908 (F-106) — the smart-select SAM3 image ops (box/points) pass `None` for `cancel` by
/// explicit per-engine decision: each is one bounded SAM3 forward pass on one image with no
/// per-step loop for a flag to poll, so there is no seam a `Some(flag)` could interrupt. This test
/// locks the `None`-path contract the decision relies on: a bounded single-shot blocking task run
/// with `cancel = None` resolves to its value, and (unlike the `Some(flag)` paths) the consumer
/// never consults the cancel poll — so no "canceled" is ever posted even mid-run. If someone
/// re-adds a `Some(flag)` that the SAM3 engine can't read, this stays green but the flag is a no-op
/// (the F-106 finding); the guard against that is the documented `None`-caller list at the
/// `run_blocking_with_heartbeat` definition, kept in sync with this call site.
#[tokio::test]
async fn run_blocking_with_heartbeat_none_cancel_returns_single_shot_value() {
    let (base_url, posts) = spawn_no_user_cancel_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    settings.heartbeat_seconds = 5;
    let api = ApiClient::new(&settings);

    // A bounded single-shot blocking task, the smart-select SAM3 shape: one forward pass, returns a
    // value, no cancel flag consulted.
    let task = tokio::task::spawn_blocking(|| Ok::<u32, WorkerError>(42));
    let result = crate::run_blocking_with_heartbeat(
        &api,
        &settings,
        "job-1",
        None,
        "",
        "smart-select none-cancel test task",
        crate::no_cancel_ack(),
        task,
    )
    .await;

    assert_eq!(
        result.expect("the None-cancel single-shot task resolves to its value"),
        42
    );
    // No terminal "canceled" progress was posted — the None path never consults the cancel poll.
    let posted = posts.lock().expect("posts lock");
    assert!(
        posted.iter().all(|body| body
            .get("status")
            .and_then(Value::as_str)
            .is_none_or(|s| s != "canceled")),
        "the None-cancel path must never post a terminal canceled status"
    );
}

/// sc-8804 (F-003) — the child-process leg: `run_ffmpeg` (media_jobs) and the `hf` CLI download
/// (model_jobs) build their `tokio::process::Command` with `kill_on_drop(true)` so a
/// heartbeat/cancel `?` early return reaps the child instead of leaving ffmpeg/`hf` writing partial
/// files. A tokio child is NOT reaped on drop by default; this test locks the mechanism the fix
/// relies on — a `kill_on_drop(true)` child is torn down when its handle is dropped.
#[cfg(unix)]
#[tokio::test]
async fn kill_on_drop_reaps_a_dropped_child() {
    use tokio::process::Command;
    // A child that would run for a minute unless killed. Capture its PID, drop the handle, then
    // confirm the OS no longer has it (kill(pid, 0) fails with ESRCH once reaped).
    let child = Command::new("sleep")
        .arg("60")
        .kill_on_drop(true)
        .spawn()
        .expect("spawn sleep");
    let pid = child.id().expect("child has a pid");
    drop(child);
    // Give the runtime a moment to send the kill and reap the zombie.
    tokio::time::sleep(Duration::from_millis(200)).await;
    // `kill -0` probes existence without signaling; a reaped/gone process yields a non-zero status.
    let alive = std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .expect("run kill -0")
        .success();
    assert!(
        !alive,
        "a kill_on_drop(true) child must be reaped when its handle is dropped (pid {pid} still alive)"
    );
}

/// sc-5516 / sc-8805 — the detail sibling: `begin_detail_cancel` trips the engine flag
/// (which the interval-armed detail event loop now reaches within one ~2s poll even while
/// the model is cold-loading or a tile is mid-refine) and acknowledges with a NON-terminal
/// `running` update; the terminal `Canceled` is posted by `run_image_detail_job` only after
/// the blocking refinement actually stops. macOS-only because `image_jobs/detail.rs` is the
/// macOS MLX route (off-Mac `run_image_detail_job` is a hard error stub).
#[cfg(target_os = "macos")]
#[tokio::test]
async fn begin_detail_cancel_trips_flag_and_stays_non_terminal() {
    let (base_url, posts) = spawn_progress_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let cancel = gen_core::CancelFlag::new();
    crate::image_jobs::begin_detail_cancel(&api, "job-1", &cancel, "mlx").await;
    assert!(
        cancel.is_cancelled(),
        "begin_detail_cancel must trip the engine cancel flag"
    );
    let posts = posts.lock().expect("posts lock");
    assert_eq!(
        posts.len(),
        1,
        "exactly one acknowledgement update is posted"
    );
    assert_eq!(
        posts[0]["status"], "running",
        "the cancel acknowledgement must stay NON-terminal (sc-5516)"
    );
    assert!(
        posts[0]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("Cancelling"),
        "the acknowledgement message should read as Cancelling…"
    );
}

/// sc-4176 — on macOS an MLX-routed model whose weights don't resolve must
/// fail loudly with a precise re-download error instead of silently completing
/// with procedural stub output; non-engine model ids keep the stub path.
#[cfg(target_os = "macos")]
mod stub_fallback_gate {
    use super::*;
    use crate::image_jobs::mlx_weights_gap;
    use crate::video_jobs::ensure_video_engine_weights;
    use sceneworks_core::image_request::ImageRequest;
    use sceneworks_core::video_request::VideoRequest;
    use serde_json::Map;

    fn settings_with_empty_data_dir() -> (tempfile::TempDir, Settings) {
        let dir = tempdir().expect("tempdir");
        let mut settings = test_settings("http://127.0.0.1:1".to_owned(), None);
        settings.data_dir = dir.path().to_path_buf();
        (dir, settings)
    }

    fn object(value: serde_json::Value) -> Map<String, serde_json::Value> {
        value.as_object().expect("object").clone()
    }

    #[test]
    fn image_engine_model_without_weights_is_a_precise_error_not_a_stub() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let request = ImageRequest::from_payload(&object(json!({ "model": "z_image_turbo" })));
        let gap = mlx_weights_gap(&request, &settings).expect("missing weights flagged");
        assert!(gap.contains("z_image_turbo"), "{gap}");
        assert!(gap.contains("Re-download"), "{gap}");
    }

    #[test]
    fn image_non_engine_model_keeps_the_stub_path() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let request = ImageRequest::from_payload(&object(json!({ "model": "base-model" })));
        assert_eq!(mlx_weights_gap(&request, &settings), None);
    }

    #[test]
    fn image_explicit_model_path_resolves_and_passes_the_gate() {
        let (dir, settings) = settings_with_empty_data_dir();
        let request = ImageRequest::from_payload(&object(json!({
            "model": "z_image_turbo",
            "advanced": { "modelPath": dir.path().to_string_lossy() },
        })));
        assert_eq!(mlx_weights_gap(&request, &settings), None);
    }

    #[test]
    fn image_explicit_model_path_outside_data_dir_is_rejected() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let outside = tempdir().expect("outside tempdir");
        let request = ImageRequest::from_payload(&object(json!({
            "model": "z_image_turbo",
            "advanced": { "modelPath": outside.path().to_string_lossy() },
        })));
        let gap = mlx_weights_gap(&request, &settings).expect("unsafe modelPath rejected");
        assert!(gap.contains("Image modelPath"), "{gap}");
        assert!(gap.contains("app-managed"), "{gap}");
    }

    #[test]
    fn video_engine_model_without_weights_is_a_precise_error_not_a_stub() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let request = VideoRequest::from_payload(&object(json!({ "model": "wan_2_2_i2v_14b" })));
        let error = ensure_video_engine_weights(&request, &settings)
            .expect_err("missing Wan weights flagged");
        assert!(
            error.to_string().contains("no MLX weights found"),
            "{error}"
        );
    }

    #[test]
    fn video_svd_without_source_asset_is_an_invalid_payload_error() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let request = VideoRequest::from_payload(&object(json!({ "model": "svd" })));
        let error = ensure_video_engine_weights(&request, &settings)
            .expect_err("svd without a source image flagged");
        assert!(error.to_string().contains("source image"), "{error}");
    }

    #[test]
    fn video_non_engine_model_keeps_the_stub_path() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let request = VideoRequest::from_payload(&object(json!({ "model": "stub-model" })));
        ensure_video_engine_weights(&request, &settings).expect("non-engine model passes");
    }
}

// ---------------------------------------------------------------------------
// sc-8807: the shared blocking keepalive (`run_blocking_with_heartbeat`) is the seam the
// SAM2/SAM3 propagate steps now run through. These tests pin the three behaviors the segment
// wiring relies on, with a stand-in blocking task in place of the real (weights-requiring)
// propagate: (a) the worker heartbeat is pinged while the task runs, (b) the API cancel poll
// trips the threaded `CancelFlag`, and (c) the terminal `Canceled` is posted whether the task
// honors the flag itself or completes before the flag is tripped.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct KeepaliveStubState {
    heartbeats: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    progress: std::sync::Arc<std::sync::Mutex<Vec<Value>>>,
}

impl KeepaliveStubState {
    fn new() -> Self {
        Self {
            heartbeats: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            progress: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }
}

/// A stub API that always reports `cancel_requested: true`, counts worker heartbeats, and records
/// every job-progress post (so the terminal `Canceled` write is observable).
async fn spawn_keepalive_stub(state: KeepaliveStubState) -> String {
    async fn heartbeat_route(State(state): State<KeepaliveStubState>) -> Response {
        state
            .heartbeats
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // The body does not parse as a WorkerSnapshot; `heartbeat()` tolerates the decode
        // failure as a transport error, so the ping still counts without modeling the snapshot.
        Json(json!({})).into_response()
    }
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    async fn progress_route(
        State(state): State<KeepaliveStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
        Json(payload): Json<Value>,
    ) -> Response {
        state.progress.lock().unwrap().push(payload);
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    let app = Router::new()
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

/// The MLX lane: the keepalive trips the flag, the engine honors it between propagated frames and
/// surfaces `WorkerError::Canceled` from inside the task — the terminal `Canceled` must be posted
/// and the error propagated (never a dangling non-terminal job).
#[tokio::test]
async fn keepalive_posts_terminal_canceled_when_the_task_honors_the_flag() {
    let state = KeepaliveStubState::new();
    let base = spawn_keepalive_stub(state.clone()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base;
    let api = ApiClient::new(&settings);

    let flag = gen_core::CancelFlag::new();
    let task_flag = flag.clone();
    let task = tokio::task::spawn_blocking(move || -> super::WorkerResult<u32> {
        let start = std::time::Instant::now();
        while !task_flag.is_cancelled() {
            if start.elapsed() > Duration::from_secs(30) {
                return Err(WorkerError::Engine("cancel flag never tripped".to_owned()));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // The engine's per-frame cancel contract (gen-core d8038beb) surfaces Canceled itself.
        Err(WorkerError::Canceled(
            "Person segmentation canceled by user.".to_owned(),
        ))
    });
    let result = super::run_blocking_with_heartbeat(
        &api,
        &settings,
        "job-8807-mlx",
        Some(flag),
        "Person tracking canceled during segmentation.",
        "sam propagate stand-in",
        crate::no_cancel_ack(),
        task,
    )
    .await;

    assert!(
        matches!(result, Err(WorkerError::Canceled(ref m)) if m == "Person segmentation canceled by user."),
        "expected the task's Canceled to propagate, got {result:?}"
    );
    assert!(
        state.heartbeats.load(std::sync::atomic::Ordering::SeqCst) >= 1,
        "Busy heartbeat pinged during the blocking task"
    );
    let posts = state.progress.lock().unwrap();
    assert!(
        posts.iter().any(|p| p["status"] == "canceled"),
        "terminal Canceled posted to the job, got {posts:?}"
    );
}

/// A task that completes before the flag is observed: the compute finishes and returns `Ok`
/// before it sees the tripped flag — the keepalive still posts the terminal `Canceled` with
/// its own cancel copy and returns `WorkerError::Canceled`.
#[tokio::test]
async fn keepalive_cancels_the_job_even_when_the_task_cannot_observe_the_flag() {
    let state = KeepaliveStubState::new();
    let base = spawn_keepalive_stub(state.clone()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base;
    let api = ApiClient::new(&settings);

    let flag = gen_core::CancelFlag::new();
    let wait_flag = flag.clone();
    let task = tokio::task::spawn_blocking(move || -> super::WorkerResult<u32> {
        // Wait until the keepalive has tripped the flag (so `canceled` is set before we return),
        // then finish successfully — the completes-before-observing-the-flag shape.
        let start = std::time::Instant::now();
        while !wait_flag.is_cancelled() && start.elapsed() < Duration::from_secs(30) {
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(7)
    });
    let result = super::run_blocking_with_heartbeat(
        &api,
        &settings,
        "job-8807-candle",
        Some(flag),
        "Person tracking canceled during segmentation.",
        "candle propagate stand-in",
        crate::no_cancel_ack(),
        task,
    )
    .await;

    assert!(
        matches!(result, Err(WorkerError::Canceled(ref m)) if m == "Person tracking canceled during segmentation."),
        "expected the keepalive's Canceled, got {result:?}"
    );
    let posts = state.progress.lock().unwrap();
    assert!(
        posts.iter().any(|p| p["status"] == "canceled"),
        "terminal Canceled posted to the job, got {posts:?}"
    );
}

/// sc-8833 (F-031) — the macOS and off-Mac candle `segment_assembly_frames` twins were collapsed
/// into ONE cfg-free orchestrator parameterized by a segmenter-backend closure. These tests drive
/// that shared orchestrator with a FAKE segmenter (no weights / GPU / network into the model), so
/// the assembly/rollup contract, the OOB bounds guard, and the disabled/missing early-outs are all
/// covered on any lane the orchestrator compiles on. The real model path stays in the `#[ignore]`
/// E2E tests. Compiled only where the orchestrator exists (macOS or `-F backend-candle`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod segment_assembly_frames_tests {
    use super::*;
    use crate::person_track::{NormalizedBox, TrackFrame};
    use std::cell::Cell;

    /// A `TrackFrame` at `timestamp` with the given detection state. Detected frames carry a plausible
    /// box + confidence; gap frames carry `detected=false` (the predictor fills them from memory).
    fn frame(timestamp: f64, detected: bool) -> TrackFrame {
        TrackFrame {
            timestamp,
            box_: NormalizedBox {
                x: 0.25,
                y: 0.10,
                width: 0.30,
                height: 0.70,
            },
            confidence: if detected { 0.9 } else { 0.0 },
            detected,
            flags: Vec::new(),
        }
    }

    /// A `(1280×720)`-sized L8 mask whose first `on_pixels` bytes are 255 (foreground). Zero
    /// `on_pixels` → an all-empty mask (filtered out by the `p > 127` threshold, so no PNG written).
    fn mask(on_pixels: usize) -> Vec<u8> {
        let (w, h) = (1280usize, 720usize);
        let mut buf = vec![0u8; w * h];
        for byte in buf.iter_mut().take(on_pixels) {
            *byte = 255;
        }
        buf
    }

    /// Stub the `check_cancel` GET the orchestrator issues before the segmenter runs (no cancel),
    /// returning an `ApiClient` pointed at it plus a matching `JobSnapshot`.
    async fn api_and_job() -> (ApiClient, JobSnapshot) {
        let base_url = spawn_cancel_poll_stub(CancelPollStubState {
            get_status: AxumStatusCode::OK,
            cancel_requested: false,
            post_status: AxumStatusCode::OK,
        })
        .await;
        let mut settings = test_settings(base_url.clone(), None);
        settings.api_url = base_url;
        let api = ApiClient::new(&settings);
        let job: JobSnapshot = serde_json::from_value(job_snapshot_json("job-1", false))
            .expect("job snapshot deserializes");
        (api, job)
    }

    /// `segment_enabled=false` short-circuits to `missing` before any segmenter or network call —
    /// the fake segmenter must never fire.
    #[tokio::test]
    async fn disabled_segmentation_rolls_up_to_missing_without_running_segmenter() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        let frames = vec![frame(0.0, true), frame(0.5, true)];
        let frame_paths: Vec<PathBuf> = frames
            .iter()
            .enumerate()
            .map(|(i, _)| project.path().join(format!("frame_{i}.png")))
            .collect();
        let mut frames_json = crate::person_track::frames_to_json(&frames);

        let ran = Cell::new(false);
        let state = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_x",
            &frames,
            &frame_paths,
            &mut frames_json,
            false,
            |_clip: SegmentClip| {
                ran.set(true);
                async { SegmentOutcome::Masks(Vec::new()) }
            },
        )
        .await
        .expect("orchestrator ok");

        assert_eq!(
            state, "missing",
            "segmentation disabled → maskState=missing"
        );
        assert!(!ran.get(), "segmenter must not run when disabled");
        assert!(
            frames_json.iter().all(|f| f["mask"].is_null()),
            "no masks written when disabled"
        );
    }

    /// No detected frame → `missing`, again before the segmenter runs.
    #[tokio::test]
    async fn no_detected_frames_rolls_up_to_missing() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        let frames = vec![frame(0.0, false), frame(0.5, false)];
        let frame_paths: Vec<PathBuf> = frames
            .iter()
            .enumerate()
            .map(|(i, _)| project.path().join(format!("frame_{i}.png")))
            .collect();
        let mut frames_json = crate::person_track::frames_to_json(&frames);

        let ran = Cell::new(false);
        let state = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_x",
            &frames,
            &frame_paths,
            &mut frames_json,
            true,
            |_clip: SegmentClip| {
                ran.set(true);
                async { SegmentOutcome::Masks(Vec::new()) }
            },
        )
        .await
        .expect("orchestrator ok");

        assert_eq!(state, "missing");
        assert!(
            !ran.get(),
            "segmenter must not run with zero detected frames"
        );
    }

    /// OOB bounds guard: `frame_paths` shorter than the last detected assembly index would slice
    /// `frame_paths[first..=last]` past the end and panic. The guard must return `degraded` before
    /// the clip is built, and the fake segmenter must never run. Regression for the twin drift where
    /// only one backend carried the guard.
    #[tokio::test]
    async fn short_frame_paths_hits_bounds_guard_and_degrades_without_panicking() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        // Three assembly frames, last detected at index 2 — but only ONE rendered frame path.
        let frames = vec![frame(0.0, false), frame(0.5, false), frame(1.0, true)];
        let frame_paths = vec![project.path().join("frame_0.png")];
        let mut frames_json = crate::person_track::frames_to_json(&frames);

        let ran = Cell::new(false);
        let state = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_x",
            &frames,
            &frame_paths,
            &mut frames_json,
            true,
            |_clip: SegmentClip| {
                ran.set(true);
                async { SegmentOutcome::Masks(Vec::new()) }
            },
        )
        .await
        .expect("orchestrator ok (guard returns Ok(degraded), never panics)");

        assert_eq!(
            state, "degraded",
            "short frame_paths → degraded via the bounds guard"
        );
        assert!(
            !ran.get(),
            "the OOB guard must short-circuit before the segmenter"
        );
    }

    /// Happy path: every detected frame masked → `active`, masks written to disk, and the clip the
    /// backend receives spans `first..=last` with box anchors on detected frames and `None` on gaps.
    #[tokio::test]
    async fn all_detected_masked_writes_pngs_and_rolls_up_active() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        // Detected at 0 and 2, a gap at 1 inside the span → clip is the full first..=last (3 frames).
        let frames = vec![frame(0.0, true), frame(0.5, false), frame(1.0, true)];
        let frame_paths: Vec<PathBuf> = frames
            .iter()
            .enumerate()
            .map(|(i, _)| project.path().join(format!("frame_{i}.png")))
            .collect();
        let mut frames_json = crate::person_track::frames_to_json(&frames);

        let state = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_x",
            &frames,
            &frame_paths,
            &mut frames_json,
            true,
            |clip: SegmentClip| {
                // The orchestrator handed us the whole first..=last span with per-frame anchors.
                assert_eq!(
                    clip.clip_paths.len(),
                    3,
                    "clip covers first..=last inclusive"
                );
                assert_eq!(clip.anchors.len(), 3);
                assert!(clip.anchors[0].is_some(), "detected frame has a box anchor");
                assert!(clip.anchors[1].is_none(), "gap frame anchor is None");
                assert!(clip.anchors[2].is_some());
                // One non-empty mask per clip frame → every frame gets a PNG.
                async { SegmentOutcome::Masks(vec![mask(100), mask(100), mask(100)]) }
            },
        )
        .await
        .expect("orchestrator ok");

        assert_eq!(state, "active", "every detected frame masked → active");
        // Masks written for all three assembly indices (1-based file names).
        for idx in 1..=3 {
            let path = project
                .path()
                .join(format!("person-tracks/track_x/masks/frame_{idx:06}.png"));
            assert!(path.exists(), "mask PNG {idx} written to disk");
        }
        // Each frame's sidecar `mask` is set to its relative path.
        for (i, f) in frames_json.iter().enumerate() {
            let rel = format!("person-tracks/track_x/masks/frame_{:06}.png", i + 1);
            assert_eq!(f["mask"], serde_json::Value::String(rel));
        }
    }

    /// A partial subset masked (some detected frames empty) → `generated`, and empty masks (all
    /// pixels below the `p > 127` threshold) write no PNG.
    #[tokio::test]
    async fn partial_masking_rolls_up_generated_and_empty_masks_skip_pngs() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        // Three detected frames; the middle mask is empty → 2 of 3 masked → generated.
        let frames = vec![frame(0.0, true), frame(0.5, true), frame(1.0, true)];
        let frame_paths: Vec<PathBuf> = frames
            .iter()
            .enumerate()
            .map(|(i, _)| project.path().join(format!("frame_{i}.png")))
            .collect();
        let mut frames_json = crate::person_track::frames_to_json(&frames);

        let state = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_y",
            &frames,
            &frame_paths,
            &mut frames_json,
            true,
            |_clip: SegmentClip| async { SegmentOutcome::Masks(vec![mask(50), mask(0), mask(50)]) },
        )
        .await
        .expect("orchestrator ok");

        assert_eq!(state, "generated", "2 of 3 detected masked → generated");
        // The empty middle mask writes no PNG and leaves its sidecar `mask` null.
        assert!(frames_json[1]["mask"].is_null(), "empty mask writes no PNG");
        assert!(!project
            .path()
            .join("person-tracks/track_y/masks/frame_000002.png")
            .exists());
        assert!(frames_json[0]["mask"].is_string());
        assert!(frames_json[2]["mask"].is_string());
    }

    /// A backend cancel is terminal for the whole job (never a degrade); a backend failure degrades
    /// to box masks. Both mappings live once, in the shared orchestrator.
    #[tokio::test]
    async fn backend_canceled_propagates_and_degraded_maps_to_degraded() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        let frames = vec![frame(0.0, true)];
        let frame_paths = vec![project.path().join("frame_0.png")];

        // Canceled → terminal WorkerError::Canceled.
        let mut json_a = crate::person_track::frames_to_json(&frames);
        let canceled = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_c",
            &frames,
            &frame_paths,
            &mut json_a,
            true,
            |_clip: SegmentClip| async { SegmentOutcome::Canceled("user canceled".to_owned()) },
        )
        .await;
        assert!(
            matches!(canceled, Err(WorkerError::Canceled(ref m)) if m == "user canceled"),
            "backend cancel is terminal, got {canceled:?}"
        );

        // Degraded → Ok("degraded"), box-mask fallback at replacement time.
        let mut json_b = crate::person_track::frames_to_json(&frames);
        let degraded = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_d",
            &frames,
            &frame_paths,
            &mut json_b,
            true,
            |_clip: SegmentClip| async { SegmentOutcome::Degraded },
        )
        .await
        .expect("degraded is Ok");
        assert_eq!(degraded, "degraded");
    }

    /// The rollup helper both backends now share (Python `segment_track` contract).
    #[test]
    fn mask_rollup_state_matches_python_segment_track() {
        assert_eq!(mask_rollup_state(0, 4), "degraded");
        assert_eq!(mask_rollup_state(2, 4), "generated");
        assert_eq!(mask_rollup_state(4, 4), "active");
        assert_eq!(mask_rollup_state(1, 1), "active");
        // generated capped at detected_total → still active.
        assert_eq!(mask_rollup_state(6, 5), "active");
    }
}
