use super::super::ImageRequest;
use super::resolved_mlx_artifact_provenance;
use crate::Path;
use crate::Settings;
use crate::Value;
use crate::DEFAULT_HUGGINGFACE_BASE_URL;
use crate::DEFAULT_MAX_LORA_URL_BYTES;
use crate::DEFAULT_MAX_MODEL_URL_BYTES;
use serde_json::json;

pub(crate) fn settings(data_dir: &Path) -> Settings {
    Settings {
        api_url: "http://127.0.0.1".to_owned(),
        config_dir: data_dir.join("config"),
        worker_id: "test-worker".to_owned(),
        gpu_id: "gpu-0".to_owned(),
        huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
        max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
        backend_mlx_enabled: true,
        ..crate::Settings::for_test(data_dir.to_path_buf())
    }
}

fn request(model_path: &Path, spoofed_tier: &str) -> ImageRequest {
    ImageRequest::from_payload(
            json!({
                "model": "fixture_model",
                "advanced": {
                    "modelPath": model_path,
                    "modelPathProvenance": {
                        "repository": "attacker/spoof",
                        "revision": "attacker",
                        "variant": "attacker",
                        "tier": spoofed_tier,
                        "resolvedPathFingerprint": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
                    }
                }
            })
            .as_object()
            .expect("request object"),
        )
}

#[test]
fn local_resolver_uses_only_worker_receipt_and_requires_its_exact_tier() {
    let data = tempfile::tempdir().expect("data dir");
    let installed = data.path().join("models/mlx/fixture/q4");
    std::fs::create_dir_all(&installed).expect("artifact dir");
    std::fs::write(installed.join("weights.safetensors"), b"trusted").expect("weights");
    let written = crate::model_jobs::write_app_managed_artifact_receipt(
        &installed,
        "SceneWorks/fixture",
        "c0ffee",
        "converted-fixture-q4",
        "q4",
    )
    .expect("worker receipt");
    let settings = settings(data.path());
    let spoofed = request(&installed, "q8");

    assert_eq!(
        resolved_mlx_artifact_provenance(
            &spoofed,
            &settings,
            "ignored/repo",
            &installed,
            Some("q4")
        )
        .expect("resolver"),
        Some(written),
        "request-supplied provenance must not replace the worker receipt"
    );
    assert_eq!(
        resolved_mlx_artifact_provenance(
            &spoofed,
            &settings,
            "ignored/repo",
            &installed,
            Some("q8")
        )
        .expect("resolver"),
        None,
        "the trusted q4 receipt must never be rewritten into a q8 identity"
    );

    let unstamped = data.path().join("models/mlx/unstamped/q8");
    std::fs::create_dir_all(&unstamped).expect("unstamped artifact");
    std::fs::write(unstamped.join("weights.safetensors"), b"untrusted").expect("weights");
    assert_eq!(
        resolved_mlx_artifact_provenance(
            &request(&unstamped, "q8"),
            &settings,
            "ignored/repo",
            &unstamped,
            Some("q8")
        )
        .expect("resolver"),
        None,
        "a spoofed payload cannot manufacture missing worker-owned provenance"
    );
}

/// A model installed before download-time tree stamps existed carries a BACKFILLED receipt
/// (`apps/rust-api/src/models.rs::backfill_current_receipt`), which records `resolvedFiles` read
/// off disk but never an `artifactTreeStamp` — only a download whose job payload carried
/// `memoryCalibrationProvenanceRequired: true` computes one. Such a receipt resolves a fully
/// loadable snapshot and still yields NO provenance, and nothing recomputes the stamp later.
///
/// That is the exact state that made every calibration-opted-in model unusable. Two things had
/// to change, and this pins both: the raw read still reports `None` (nothing is invented), while
/// the MLX provenance resolver ESTABLISHES the missing baseline so the install can reach the
/// evidence path instead of being pinned to the legacy selector for good (sc-16482).
#[test]
fn a_backfilled_receipt_gains_its_missing_tree_stamp_baseline_on_the_mlx_path() {
    let data = tempfile::tempdir().expect("data dir");
    let hub = data.path().join("hub");
    let _env = crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().expect("hub path"))]);
    let repo = "SceneWorks/fixture-turnkey";
    let files = ["q8/model_index.json", "q8/transformer/config.json"];
    let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(data.path(), repo)
        .expect("cache")
        .join("snapshots/rev-backfilled");
    std::fs::create_dir_all(snapshot.join("q8/transformer")).expect("snapshot tree");
    for file in files {
        std::fs::write(snapshot.join(file), b"{}").expect("snapshot file");
    }
    let marker_dir = data
        .path()
        .join("models")
        .join(crate::paths::safe_download_dir(repo));
    std::fs::create_dir_all(&marker_dir).expect("marker dir");
    // Byte-for-byte the backfill writer's shape: no `snapshotRevision`, no `artifactTreeStamp`.
    std::fs::write(
        marker_dir.join(crate::INSTALL_MARKER),
        serde_json::to_vec(&json!({
            "schemaVersion": 2,
            "repo": repo,
            "modelId": "fixture_model",
            "variant": "q8",
            "manifestFiles": ["q8/*"],
            "resolvedFiles": files,
            "backfilled": true,
        }))
        .expect("receipt json"),
    )
    .expect("marker");

    let weights_dir = snapshot.join("q8");
    let marker = marker_dir.join(crate::INSTALL_MARKER);
    let read_only = crate::model_jobs::huggingface_receipt_weights(
        data.path(),
        repo,
        Some("fixture_model"),
        Some("q8"),
        crate::model_jobs::ProvenanceRepair::Skip,
    )
    .expect("a backfilled receipt still resolves loadable weights");
    assert_eq!(
        read_only.path, weights_dir,
        "the artifact itself is found and loadable"
    );
    assert_eq!(
        read_only.provenance, None,
        "a read that may not repair must never invent an identity"
    );

    let request = ImageRequest::from_payload(
        json!({ "model": "fixture_model" })
            .as_object()
            .expect("request object"),
    );
    let repaired = resolved_mlx_artifact_provenance(
        &request,
        &settings(data.path()),
        repo,
        &weights_dir,
        Some("q8"),
    )
    .expect("resolver")
    .expect("the MLX path must establish the missing baseline, not give up on the install");
    assert_eq!(repaired.identity.repository, repo);
    assert_eq!(repaired.identity.revision, "rev-backfilled");
    assert_eq!(repaired.identity.variant, "q8");
    assert_eq!(repaired.fixed_artifact_tier.as_deref(), Some("q8"));

    // The baseline is PERSISTED, so a later mutation is detectable rather than re-blessed, and
    // the resolved revision is recorded so future resolution no longer leans on file-set
    // uniqueness. The repair anchor is recorded and never mistaken for a download-time stamp.
    let stored: Value =
        serde_json::from_slice(&std::fs::read(&marker).expect("marker")).expect("receipt json");
    assert!(
        crate::model_jobs::is_sha256_fingerprint(
            stored["artifactTreeStamp"].as_str().expect("stamp")
        ),
        "repair must write a well-formed stamp"
    );
    assert_eq!(stored["artifactTreeStampSource"], json!("repair"));
    assert_eq!(stored["snapshotRevision"], json!("rev-backfilled"));

    // Now that a baseline exists, a plain read resolves provenance with no further writing...
    assert!(crate::model_jobs::huggingface_receipt_weights(
        data.path(),
        repo,
        Some("fixture_model"),
        Some("q8"),
        crate::model_jobs::ProvenanceRepair::Skip,
    )
    .expect("resolves")
    .provenance
    .is_some());

    // ...and drift against that baseline fails closed instead of being repaired away.
    std::fs::write(snapshot.join(files[0]), b"{\"mutated\":true}").expect("mutate");
    assert_eq!(
        resolved_mlx_artifact_provenance(
            &request,
            &settings(data.path()),
            repo,
            &weights_dir,
            Some("q8"),
        )
        .expect("resolver"),
        None,
        "a receipt whose stamp MISMATCHES is drift; repair must never overwrite it"
    );
}
