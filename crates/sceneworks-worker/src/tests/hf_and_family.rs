
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

fn flux2_safetensors_keys() -> Vec<String> {
    // Mirrors the FLUX.2 architecture signature: the top-level shared-modulation
    // tensors (unique to FLUX.2) plus its double/single stream blocks. klein and dev
    // weights carry no variant-specific signature, so both detect as the base `flux2`.
    let mut keys = vec![
        "double_stream_modulation_img".to_owned(),
        "double_stream_modulation_txt".to_owned(),
        "single_stream_modulation.weight".to_owned(),
    ];
    for block in 0..8 {
        keys.push(format!("double_blocks.{block}.img_attn.qkv.weight"));
        keys.push(format!("double_blocks.{block}.txt_attn.qkv.weight"));
        keys.push(format!("single_blocks.{block}.linear1.weight"));
    }
    keys
}

fn flux1_no_metadata_safetensors_keys() -> Vec<String> {
    // FLUX.1 double + single stream blocks. Chroma is FLUX.1-schnell-derived and shares
    // this exact tensor layout, so a metadata-less Chroma checkpoint key-detects as the
    // base `flux` (the `single_transformer_blocks.` prefix is the Flux discriminator).
    let mut keys = Vec::new();
    for block in 0..8 {
        for module in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
            keys.push(format!("transformer.transformer_blocks.{block}.{module}.weight"));
            keys.push(format!(
                "transformer.single_transformer_blocks.{block}.{module}.weight"
            ));
        }
    }
    keys
}

#[test]
fn hf_download_inputs_accept_catalog_values() {
    validate_hf_download_inputs(
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
fn hf_download_inputs_reject_option_injection() {
    for repo in ["--local-dir=/Users/me/.ssh", "owner/--local-dir=/tmp/out"] {
        let error = validate_hf_download_inputs(repo, "main", &["*.safetensors".to_owned()])
            .expect_err("malicious repo rejected");
        assert!(matches!(error, WorkerError::InvalidPayload(_)));
    }

    let error = validate_hf_download_inputs(
        "owner/model",
        "--local-dir=/tmp/out",
        &["*.safetensors".to_owned()],
    )
    .expect_err("malicious revision rejected");
    assert!(matches!(error, WorkerError::InvalidPayload(_)));

    let error = validate_hf_download_inputs(
        "owner/model",
        "main",
        &["--local-dir=/tmp/out".to_owned()],
    )
    .expect_err("malicious include pattern rejected");
    assert!(matches!(error, WorkerError::InvalidPayload(_)));
}

#[test]
fn hf_download_inputs_reject_traversal_and_absolute_patterns() {
    for pattern in [
        "../model.safetensors",
        "nested/../../model.safetensors",
        "/tmp/model.bin",
    ] {
        let error = validate_hf_download_inputs("owner/model", "main", &[pattern.to_owned()])
            .expect_err("unsafe include pattern rejected");
        assert!(matches!(error, WorkerError::InvalidPayload(_)));
    }

    for revision in ["../main", "refs/heads/../main", "/refs/main"] {
        let error =
            validate_hf_download_inputs("owner/model", revision, &["*.safetensors".to_owned()])
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

#[test]
fn download_family_check_proceeds_for_derived_family_base_architecture() {
    // Regression: a catalog entry declares a model family whose downloaded weights detect
    // only as a compatible *base* architecture. The download guard must treat that as a
    // match, not a false mismatch that fails the install with e.g. "Downloaded model files
    // appear to be flux2, but the catalog declared family flux2-klein".
    //
    // FLUX.2 [klein] / [dev] (`flux2-klein` / `flux2-dev`) detect as `flux2`; a metadata-
    // less Chroma checkpoint (`chroma`, FLUX.1-derived) detects as `flux`.
    let flux2_dir = tempdir().expect("tempdir creates");
    write_safetensors_with_keys(
        &flux2_dir.path().join("model.safetensors"),
        &flux2_safetensors_keys(),
    );
    for declared in ["flux2-klein", "flux2-dev", "flux2"] {
        assert!(
            matches!(
                check_downloaded_model_family(Some(declared.to_owned()), flux2_dir.path()),
                DownloadFamilyCheck::Proceed
            ),
            "declared {declared:?} against flux2 weights should proceed"
        );
    }

    let chroma_dir = tempdir().expect("tempdir creates");
    write_safetensors_with_keys(
        &chroma_dir.path().join("model.safetensors"),
        &flux1_no_metadata_safetensors_keys(),
    );
    for declared in ["chroma", "flux"] {
        assert!(
            matches!(
                check_downloaded_model_family(Some(declared.to_owned()), chroma_dir.path()),
                DownloadFamilyCheck::Proceed
            ),
            "declared {declared:?} against flux weights should proceed"
        );
    }

    // A genuinely wrong declaration is still caught: wan-video weights are not flux.
    let wan_dir = tempdir().expect("tempdir creates");
    write_safetensors_with_keys(
        &wan_dir.path().join("model.safetensors"),
        &wan_video_safetensors_keys(),
    );
    assert!(matches!(
        check_downloaded_model_family(Some("chroma".to_owned()), wan_dir.path()),
        DownloadFamilyCheck::Mismatch(_)
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
fn stale_receipt_resolves_exact_old_quant_tier_after_manifest_rename() {
    let root = tempdir().expect("temp dir creates");
    let hub = root.path().join("hub");
    let _env = crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", hub.to_str().expect("utf-8 hub")),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ]);
    let repo = "owner/model";
    let snapshot = super::huggingface_repo_cache_path(root.path(), repo)
        .expect("cache path")
        .join("snapshots/old-revision");
    for file in ["q4/transformer/old-1.safetensors", "q4/transformer/old-2.safetensors", "q4/config.json"] {
        let path = snapshot.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"old").unwrap();
    }
    let marker = root.path().join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&marker).unwrap();
    std::fs::write(
        marker.join(INSTALL_MARKER),
        serde_json::to_vec(&json!({
            "schemaVersion": 2, "repo": repo, "modelId": "model-id", "variant": "q4",
            "snapshotRevision": "old-revision",
            "manifestFiles": ["q4/transformer/old-*.safetensors", "q4/config.json"],
            "resolvedFiles": ["q4/transformer/old-1.safetensors", "q4/transformer/old-2.safetensors", "q4/config.json"]
        })).unwrap(),
    ).unwrap();

    let resolved = huggingface_receipt_weights_dir(root.path(), repo, Some("model-id"), Some("q4"));
    assert_eq!(resolved, Some(snapshot.join("q4")));
    assert!(!snapshot.join("q4/transformer/new-name.safetensors").exists());
}

#[test]
fn stale_receipt_never_resolves_a_mixed_file_set() {
    let root = tempdir().expect("temp dir creates");
    let hub = root.path().join("hub");
    let _env = crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().unwrap())]);
    let repo = "owner/mixed";
    let repo_cache = super::huggingface_repo_cache_path(root.path(), repo).unwrap();
    let old = repo_cache.join("snapshots/old");
    let current = repo_cache.join("snapshots/current");
    std::fs::create_dir_all(old.join("q8")).unwrap();
    std::fs::create_dir_all(current.join("q8")).unwrap();
    std::fs::write(old.join("q8/a.safetensors"), b"a").unwrap();
    std::fs::write(current.join("q8/b.safetensors"), b"b").unwrap();
    let marker = root.path().join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&marker).unwrap();
    std::fs::write(marker.join(INSTALL_MARKER), serde_json::to_vec(&json!({
        "repo": repo, "modelId": "mixed", "variant": "q8",
        "resolvedFiles": ["q8/a.safetensors", "q8/b.safetensors"]
    })).unwrap()).unwrap();

    assert_eq!(huggingface_receipt_weights_dir(root.path(), repo, Some("mixed"), Some("q8")), None);
}

#[test]
fn stale_single_variant_receipt_resolves_snapshot_root() {
    let root = tempdir().expect("temp dir creates");
    let hub = root.path().join("hub");
    let _env = crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().unwrap())]);
    let repo = "owner/single";
    let snapshot = super::huggingface_repo_cache_path(root.path(), repo)
        .unwrap()
        .join("snapshots/installed");
    std::fs::create_dir_all(snapshot.join("transformer")).unwrap();
    std::fs::write(snapshot.join("transformer/old.safetensors"), b"old").unwrap();
    std::fs::write(snapshot.join("config.json"), b"{}").unwrap();
    let marker = root.path().join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&marker).unwrap();
    std::fs::write(marker.join(INSTALL_MARKER), serde_json::to_vec(&json!({
        "repo": repo, "modelId": "single", "variant": "default",
        "resolvedFiles": ["transformer/old.safetensors", "config.json"]
    })).unwrap()).unwrap();

    assert_eq!(
        huggingface_receipt_weights_dir(root.path(), repo, Some("single"), Some("default")),
        Some(snapshot)
    );
}

#[test]
fn receipt_revision_disambiguates_snapshots_with_identical_filenames() {
    let root = tempdir().unwrap();
    let hub = root.path().join("hub");
    let _env = crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().unwrap())]);
    let repo = "owner/revisions";
    let cache = super::huggingface_repo_cache_path(root.path(), repo).unwrap();
    for (revision, bytes) in [("installed", b"old".as_slice()), ("newer", b"new".as_slice())] {
        let file = cache.join("snapshots").join(revision).join("model.safetensors");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(file, bytes).unwrap();
    }
    let marker = root.path().join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&marker).unwrap();
    std::fs::write(marker.join(INSTALL_MARKER), serde_json::to_vec(&json!({
        "repo": repo, "modelId": "revisions", "variant": "default",
        "snapshotRevision": "installed", "resolvedFiles": ["model.safetensors"]
    })).unwrap()).unwrap();

    let resolved = huggingface_receipt_weights_dir(root.path(), repo, Some("revisions"), Some("default")).unwrap();
    assert_eq!(std::fs::read(resolved.join("model.safetensors")).unwrap(), b"old");
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
