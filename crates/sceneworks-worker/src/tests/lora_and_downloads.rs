
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
async fn paired_moe_upload_rejects_single_digest_before_network_or_file_move() {
    use sha2::{Digest, Sha256};

    let temp = tempdir().expect("tempdir creates");
    let mut settings = test_settings("http://127.0.0.1:1".to_owned(), None);
    settings.data_dir = temp.path().join("data");
    settings.api_url = "http://127.0.0.1:1".to_owned();
    let upload_dir = settings.data_dir.join("cache/lora-uploads/upload-1");
    tokio::fs::create_dir_all(&upload_dir).await.unwrap();
    let high_upload = upload_dir.join("high.safetensors");
    let low_upload = upload_dir.join("low.safetensors");
    tokio::fs::write(&high_upload, b"high").await.unwrap();
    tokio::fs::write(&low_upload, b"low").await.unwrap();
    // This digest matches the lexicographically first/high file, the exact case the old
    // enumeration-first verification incorrectly accepted while leaving the low half unchecked.
    let first_digest = format!("{:x}", Sha256::digest(b"high"));

    let mut job_json = job_snapshot_json("job-1", false);
    job_json["type"] = json!("lora_import");
    job_json["payload"] = json!({
        "loraId": "my_moe",
        "sourcePath": high_upload.display().to_string(),
        "secondarySourcePath": low_upload.display().to_string(),
        "uploadedSourcePath": true,
        "expectedSha256": first_digest,
    });
    let job: JobSnapshot = serde_json::from_value(job_json).expect("job deserializes");
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let error = super::model_jobs::run_lora_import_job(&api, &settings, &client, &job)
        .await
        .expect_err("one digest cannot verify a two-file upload");
    assert!(
        error.to_string().contains("ambiguous with secondarySourcePath"),
        "digest contract rejection must win before the unreachable heartbeat: {error:?}"
    );
    assert!(high_upload.is_file(), "high staged source was not moved");
    assert!(low_upload.is_file(), "low staged source was not moved");
    assert!(
        !settings.data_dir.join("loras/my_moe").exists(),
        "no target directory is created for a rejected contract"
    );
}

#[test]
fn lora_import_digest_contract_allows_unambiguous_single_or_pair_inputs() {
    let mut payload = JsonObject::new();
    payload.insert("expectedSha256".to_owned(), json!("a".repeat(64)));
    super::model_jobs::validate_lora_import_digest_contract(&payload)
        .expect("single-file digest remains supported");

    payload.remove("expectedSha256");
    payload.insert(
        "secondarySourcePath".to_owned(),
        json!("/staged/low.safetensors"),
    );
    super::model_jobs::validate_lora_import_digest_contract(&payload)
        .expect("paired upload without one ambiguous digest remains supported");
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

/// A minimal valid safetensors adapter: 8-byte LE header length + JSON header + tensor bytes.
fn safetensors_adapter_bytes(metadata: Value, keys: &[&str]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), metadata);
    for key in keys {
        header.insert(
            (*key).to_owned(),
            json!({"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}),
        );
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
    let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
    buffer.extend_from_slice(&header_bytes);
    buffer.extend_from_slice(&[0u8; 4]);
    buffer
}

/// 🔴 sc-14057: the URL/repo import route has **no file on disk when the API queues the job**, so
/// the API's pre-flight `read_adapter_metadata` — which fills `networkType` / `rank` / `alpha` for a
/// local or uploaded source — records nothing. Without a post-download read, two identical adapters
/// would be described differently purely by how they were ingested.
///
/// End-to-end through `run_lora_import_job`: the queued `manifestEntry` carries no declared
/// metadata, and the manifest actually written to disk carries exactly what the downloaded file
/// states. Remove the `apply_adapter_metadata_to_manifest_entry` call in `run_lora_import_job` and
/// this goes red.
#[tokio::test]
async fn lora_url_import_records_the_downloaded_adapters_declared_metadata() {
    let temp = tempdir().expect("tempdir creates");
    // Declares a network type and rank but NO alpha — the common third-party shape, and the case
    // that must not be back-filled from `rank`.
    let adapter = safetensors_adapter_bytes(
        json!({ "networkType": "lokr", "rank": "16" }),
        &["transformer_blocks.0.attn.to_q.lokr_w1"],
    );
    let base_url = spawn_binary_stub(adapter).await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url.clone();
    settings.data_dir = temp.path().join("data");
    settings.config_dir = temp.path().join("config");
    let manifest_path = settings
        .config_dir
        .join("manifests")
        .join("user.loras.jsonc");
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let mut job_json = job_snapshot_json("job-1", false);
    job_json["type"] = json!("lora_import");
    // Exactly what `queue_lora_import_job` emits for a `sourceUrl` import: a manifest entry with
    // no `networkType` / `rank` / `alpha`, because there was no file to inspect.
    job_json["payload"] = json!({
        "loraId": "mage_flow_realism",
        "sourceUrl": format!("{base_url}/loras/realism.safetensors"),
        "manifestPath": manifest_path.display().to_string(),
        "manifestEntry": {
            "id": "mage_flow_realism",
            "name": "Realism",
            "scope": "global",
            "source": {
                "provider": "url",
                "url": format!("{base_url}/loras/realism.safetensors"),
                "path": "loras/mage_flow_realism",
            },
            "files": ["realism.safetensors"],
            "triggerWords": [],
        },
    });
    let job: JobSnapshot = serde_json::from_value(job_json).expect("job deserializes");

    super::model_jobs::run_lora_import_job(&api, &settings, &client, &job)
        .await
        .expect("url LoRA import completes");

    let written = tokio::fs::read_to_string(&manifest_path)
        .await
        .expect("manifest written");
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &written,
    ))
        .expect("manifest parses as JSON");
    let entry = manifest["loras"]
        .as_array()
        .and_then(|entries| entries.iter().find(|entry| entry["id"] == "mage_flow_realism"))
        .expect("the imported entry is in the manifest");

    assert_eq!(entry["networkType"], json!("lokr"));
    assert_eq!(entry["rank"], json!(16));
    // 🔴 The file declares no alpha, so none is recorded — `alpha` must NOT inherit `rank`.
    assert!(
        entry.get("alpha").is_none(),
        "alpha must stay absent, not inherit rank: {entry}"
    );
    // The family reconcile this rides alongside still works: a 1-block LoKr proves no architecture,
    // so the entry stays family-less rather than being mislabelled.
    assert!(
        entry.get("family").is_none(),
        "an inconclusive adapter must not gain a family: {entry}"
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
    // Kolors → its SceneWorks tokenizer repo + pinned revision + the snapshot's tokenizer/tokenizer.json.
    assert_eq!(
        derived_tokenizer_overlay("Kwai-Kolors/Kolors-diffusers", snap),
        Some((
            "SceneWorks/kolors-chatglm3-tokenizer",
            "4001e09f10ef05845457b976bbf1d28d54319886",
            want.clone()
        ))
    );
    // Qwen-Image (sc-6570) → its SceneWorks tokenizer repo + pinned revision, same dest.
    assert_eq!(
        derived_tokenizer_overlay("Qwen/Qwen-Image", snap),
        Some((
            "SceneWorks/qwen-image-tokenizer",
            "b0178292f0f4b1be9b5bfdda2b6e97fda0e195c3",
            want.clone()
        ))
    );
    // Qwen-Image-2512 (sc-8271) reuses the same hosted overlay repo → the SAME pinned revision.
    assert_eq!(
        derived_tokenizer_overlay("Qwen/Qwen-Image-2512", snap),
        Some((
            "SceneWorks/qwen-image-tokenizer",
            "b0178292f0f4b1be9b5bfdda2b6e97fda0e195c3",
            want.clone()
        ))
    );
    // Whitespace from a manifest field is tolerated.
    assert_eq!(
        derived_tokenizer_overlay("  Qwen/Qwen-Image  ", snap),
        Some((
            "SceneWorks/qwen-image-tokenizer",
            "b0178292f0f4b1be9b5bfdda2b6e97fda0e195c3",
            want
        ))
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

/// sc-9879 (F-077 follow-up): every derived-tokenizer overlay must pin an exact 40-hex lowercase commit
/// (not the mutable `main` branch) so an upstream re-push can't silently swap the derived `tokenizer.json`
/// the in-process generator/trainer constructs from. Mirrors the `_REVISION` format tests in this PR.
#[test]
fn derived_tokenizer_overlay_revisions_are_pinned_commits_not_main() {
    let snap = std::path::Path::new("/snap");
    for base in [
        "Kwai-Kolors/Kolors-diffusers",
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-2512",
    ] {
        let (_, revision, _) =
            derived_tokenizer_overlay(base, snap).expect("known base has an overlay");
        assert_ne!(revision, "main", "{base} tokenizer must pin a fixed revision");
        assert_eq!(
            revision.len(),
            40,
            "a pinned HF revision is a 40-char commit sha ({base})"
        );
        assert!(
            revision
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "the pinned revision must be lowercase hex ({base})"
        );
    }
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

/// A stub whose HF tree resolve returns an EMPTY file list, plus the job/progress/heartbeat routes a
/// download job touches. Progress POSTs are recorded so the test can assert the job was FAILED.
async fn spawn_empty_tree_stub() -> (String, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
    // No files at this revision under any filter — the unpublished-tier case.
    spawn_tree_stub_with_files(Vec::new()).await
}

/// [`spawn_empty_tree_stub`], but the tree resolve serves `files` (path, size) — so a test can model a
/// repo that satisfies SOME declared patterns and not others (sc-12283, the torn-tier case).
async fn spawn_tree_stub_with_files(
    files: Vec<(&'static str, u64)>,
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
    use std::sync::{Arc, Mutex};
    type Posts = Arc<Mutex<Vec<Value>>>;
    let sizes: Arc<std::collections::HashMap<String, u64>> = Arc::new(
        files
            .iter()
            .map(|(path, size)| ((*path).to_owned(), *size))
            .collect(),
    );
    let tree = Arc::new(Value::Array(
        files
            .into_iter()
            .map(|(path, size)| json!({ "type": "file", "path": path, "size": size }))
            .collect(),
    ));
    let tree_route = {
        let tree = tree.clone();
        move || {
            let tree = tree.clone();
            async move { Json((*tree).clone()).into_response() }
        }
    };
    // Serve exactly the byte count the tree declared for this path, so the transfer isn't rejected as
    // truncated (see `download_snapshot_rejects_truncated_file`).
    let blob_route = {
        let sizes = sizes.clone();
        move |axum::extract::Path((_owner, _repo, _revision, path)): axum::extract::Path<(
            String,
            String,
            String,
            String,
        )>| {
            let sizes = sizes.clone();
            async move {
                match sizes.get(&path) {
                    Some(size) => vec![b'x'; *size as usize].into_response(),
                    None => axum::http::StatusCode::NOT_FOUND.into_response(),
                }
            }
        }
    };
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
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
        .route("/api/models/:owner/:repo/tree/:revision", get(tree_route))
        // Serve the blobs too, so a test that DISABLES the pattern check fails on its own assertions
        // (a torn tier installed + a completion marker written) rather than on an unrelated 404 from a
        // stub that simply can't serve the download. Without this the test would pass for the wrong
        // reason and would not actually guard the regression.
        .route("/:owner/:repo/resolve/:revision/*path", get(blob_route))
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route("/api/v1/workers/:worker_id/heartbeat", post(heartbeat_route))
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

/// Neutralize the ambient HF-cache env so a download job's `ensure_hf_files_cached` writes under the
/// test's tempdir `data_dir`, not a developer's real `HF_HOME` (sc-13834). `huggingface_hub_cache_dir`
/// reads `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `HF_HOME` BEFORE `data_dir`, so `run_model_download_job`
/// / `run_lora_download_job` (which fetch into `snapshots/main` via `ensure_hf_files_cached(…, "main", …)`)
/// otherwise land in the real cache — this is what wrote the krea-2-raw-mlx `snapshots/main` dummy
/// weights. Mirrors the model_jobs / audio_jobs co-requisite seam tests' `isolate_hf_cache`.
fn isolate_hf_cache() -> crate::test_env::EnvVars {
    crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ])
}

#[tokio::test]
async fn model_download_fails_when_the_tier_resolves_no_files() {
    // sc-9909: installing a quant tier whose weights aren't published yet (sc-8513 rollout) resolves
    // ZERO files. That must FAIL with a clear message rather than silently "succeed" and leave an
    // empty cache + a stale completion marker.
    let _env = isolate_hf_cache(); // keep the stubbed download in the tempdir, off the real cache (sc-13834)
    let temp = tempdir().expect("tempdir creates");
    let (base_url, posts) = spawn_empty_tree_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url.clone();
    settings.data_dir = temp.path().to_path_buf();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let mut job_json = job_snapshot_json("job-1", false);
    job_json["type"] = json!("model_download");
    job_json["payload"] = json!({
        "modelId": "sdxl",
        "repo": "SceneWorks/sdxl-base-mlx",
        "files": ["q8/*"],
    });
    let job: JobSnapshot = serde_json::from_value(job_json).expect("job deserializes");

    super::model_jobs::run_model_download_job(&api, &settings, &client, &job)
        .await
        .expect("returns Ok — the failure is reported via a progress post, not an Err");

    // The completion marker must NOT have been written for an empty download.
    let marker = temp
        .path()
        .join("models")
        .join(safe_download_dir("SceneWorks/sdxl-base-mlx"))
        .join(INSTALL_MARKER);
    assert!(!marker.exists(), "no completion marker for a zero-file download");

    // A Failed progress post naming the empty tier was recorded.
    let posts = posts.lock().expect("posts lock");
    let failed = posts.iter().any(|post| {
        post.get("status").and_then(Value::as_str) == Some("failed")
            && post
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("No files to download"))
    });
    assert!(failed, "expected a failed progress post, got {posts:?}");
}

#[tokio::test]
async fn model_download_fails_when_one_declared_pattern_matches_nothing() {
    // sc-12283: `allow_pattern_matches` ORs across the filter, so a tier whose transformer matched but
    // whose `tokenizer/*` matched NOTHING passed the aggregate zero-file check, downloaded, completed
    // and wrote an install marker — an "installed" tier that cannot load. This is the #850 shape.
    let _env = isolate_hf_cache(); // keep the stubbed download in the tempdir, off the real cache (sc-13834)
    let temp = tempdir().expect("tempdir creates");
    // The repo satisfies q8/transformer/* and q8/vae/* but has NO q8/tokenizer/* at all. Sizes are
    // token — the stub serves them verbatim, and this test is about the filter, not the transfer.
    let (base_url, posts) = spawn_tree_stub_with_files(vec![
        ("q8/transformer/diffusion_pytorch_model.safetensors", 8),
        ("q8/vae/diffusion_pytorch_model.safetensors", 4),
    ])
    .await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url.clone();
    settings.data_dir = temp.path().to_path_buf();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let mut job_json = job_snapshot_json("job-1", false);
    job_json["type"] = json!("model_download");
    job_json["payload"] = json!({
        "modelId": "krea_2_raw",
        "repo": "SceneWorks/krea-2-raw-mlx",
        "files": ["q8/transformer/*", "q8/vae/*", "q8/tokenizer/*"],
    });
    let job: JobSnapshot = serde_json::from_value(job_json).expect("job deserializes");

    super::model_jobs::run_model_download_job(&api, &settings, &client, &job)
        .await
        .expect("returns Ok — the failure is reported via a progress post, not an Err");

    // No completion marker: a torn tier must never read as installed.
    let marker = temp
        .path()
        .join("models")
        .join(safe_download_dir("SceneWorks/krea-2-raw-mlx"))
        .join(INSTALL_MARKER);
    assert!(!marker.exists(), "no completion marker for a partial download");

    let posts = posts.lock().expect("posts lock");
    let failed = posts.iter().any(|post| {
        post.get("status").and_then(Value::as_str) == Some("failed")
            && post
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| {
                    // The message must NAME the pattern that matched nothing — "something is missing"
                    // is not actionable, "q8/tokenizer/* matched nothing" is.
                    message.contains("Incomplete download") && message.contains("q8/tokenizer/*")
                })
    });
    assert!(failed, "expected a failed post naming the unmatched pattern, got {posts:?}");
    // The patterns that DID match must not be blamed.
    let blames_matched = posts.iter().any(|post| {
        post.get("message")
            .and_then(Value::as_str)
            .is_some_and(|message| message.contains("q8/transformer/*"))
    });
    assert!(!blames_matched, "only the unmatched pattern should be named");
}

#[tokio::test]
async fn lora_download_fails_when_the_declared_adapter_is_absent() {
    // sc-12288: this path had NO zero-file check. A LoRA whose declared `source.file` is gone from
    // the repo (renamed upstream / typo'd entry / unpublished) resolved zero files, downloaded
    // nothing, and reported the job COMPLETED — while the catalog still read "not installed", because
    // the install probe finds an empty cache. A success post for a fetch of nothing is the bug.
    let _env = isolate_hf_cache(); // keep the stubbed download in the tempdir, off the real cache (sc-13834)
    let temp = tempdir().expect("tempdir creates");
    // The repo exists but holds a DIFFERENT adapter than the one declared.
    let (base_url, posts) =
        spawn_tree_stub_with_files(vec![("some-other-adapter.safetensors", 8)]).await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url.clone();
    settings.data_dir = temp.path().to_path_buf();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let mut job_json = job_snapshot_json("job-1", false);
    job_json["type"] = json!("lora_download");
    job_json["payload"] = json!({
        "loraId": "ltx_2_3_ic_union_control",
        "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        "files": ["ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"],
    });
    let job: JobSnapshot = serde_json::from_value(job_json).expect("job deserializes");

    super::model_jobs::run_lora_download_job(&api, &settings, &client, &job)
        .await
        .expect("returns Ok — the failure is reported via a progress post, not an Err");

    let posts = posts.lock().expect("posts lock");
    let failed = posts.iter().any(|post| {
        post.get("status").and_then(Value::as_str) == Some("failed")
            && post
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("No files to download for LoRA"))
    });
    assert!(failed, "expected a failed progress post, got {posts:?}");
    // The job must NOT also report success — that is the whole defect.
    let completed = posts
        .iter()
        .any(|post| post.get("status").and_then(Value::as_str) == Some("completed"));
    assert!(!completed, "a fetch of nothing must never report completed");
}

#[tokio::test]
async fn lora_download_fails_when_one_of_several_declared_files_is_absent() {
    // sc-12288: `create_lora_download_job` accepts an explicit `files` LIST and the filter ORs across
    // it, so a multi-file adapter that landed one half and not the other passes the aggregate
    // zero-file check and completes as a torn install. Same gap as sc-12283, LoRA side.
    let _env = isolate_hf_cache(); // keep the stubbed download in the tempdir, off the real cache (sc-13834)
    let temp = tempdir().expect("tempdir creates");
    let (base_url, posts) = spawn_tree_stub_with_files(vec![("high_noise.safetensors", 8)]).await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url.clone();
    settings.data_dir = temp.path().to_path_buf();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let mut job_json = job_snapshot_json("job-1", false);
    job_json["type"] = json!("lora_download");
    job_json["payload"] = json!({
        "loraId": "pair",
        "repo": "owner/pair",
        "files": ["high_noise.safetensors", "low_noise.safetensors"],
    });
    let job: JobSnapshot = serde_json::from_value(job_json).expect("job deserializes");

    super::model_jobs::run_lora_download_job(&api, &settings, &client, &job)
        .await
        .expect("returns Ok — the failure is reported via a progress post, not an Err");

    let posts = posts.lock().expect("posts lock");
    let failed = posts.iter().any(|post| {
        post.get("status").and_then(Value::as_str) == Some("failed")
            && post
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| {
                    message.contains("Incomplete download for LoRA")
                        && message.contains("low_noise.safetensors")
                })
    });
    assert!(failed, "expected a failed post naming the absent half, got {posts:?}");
    // The half that DID resolve must not be blamed.
    let blames_present = posts.iter().any(|post| {
        post.get("message")
            .and_then(Value::as_str)
            .is_some_and(|message| message.contains("nothing matches") && message.contains("high_noise"))
    });
    assert!(!blames_present, "only the absent file should be named");
}

// ---------------------------------------------------------------------------
// sc-13583 / F-002: HF repo/revision/pattern validation on the user-facing jobs
// ---------------------------------------------------------------------------
//
// `validate_hf_download_inputs` was called from ONLY `ensure_hf_files_cached` (the on-demand tier
// fetch). The four user-facing entry points — model/LoRA download and model/LoRA import — passed a
// payload `repo`/`revision`/`files` straight to the HF URL builder and, for the downloads, to the
// `refs/<rev>` / `snapshots/<rev>` cache joins with no validation, giving a LAN-API caller (epic
// 4484) an arbitrary-path write/read primitive. Each entry point must now reject a path-escaping
// revision UP FRONT. Every assertion below pins the validator's own "traversal components are not
// allowed" message, so a test stays RED if ONLY the entry-point call is removed — the downstream
// `safe_join` join-guard would otherwise still error, but with a distinct "Unsafe snapshot path"
// message, masking the entry-point regression (the false-green trap).

#[tokio::test]
async fn model_download_rejects_a_path_escaping_revision_before_any_network() {
    let _env = isolate_hf_cache();
    let temp = tempdir().expect("tempdir creates");
    // An unreachable API proves validation fires BEFORE any heartbeat / tree-resolve I/O: with the
    // guard the job errors on the revision; without it, the first thing that happens is a heartbeat
    // that can't connect — never the traversal message.
    let mut settings = test_settings("http://127.0.0.1:1".to_owned(), None);
    settings.data_dir = temp.path().to_path_buf();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let mut job_json = job_snapshot_json("job-1", false);
    job_json["type"] = json!("model_download");
    job_json["payload"] = json!({
        "modelId": "sdxl",
        "repo": "SceneWorks/sdxl-base-mlx",
        "revision": "../../evil",
        "files": ["*.safetensors"],
    });
    let job: JobSnapshot = serde_json::from_value(job_json).expect("job deserializes");

    let error = super::model_jobs::run_model_download_job(&api, &settings, &client, &job)
        .await
        .expect_err("a path-escaping revision is rejected at the entry point");
    assert!(
        error
            .to_string()
            .contains("traversal components are not allowed"),
        "expected the revision validator to fire, got {error:?}"
    );
}

#[tokio::test]
async fn lora_download_rejects_a_path_escaping_revision_before_any_network() {
    let _env = isolate_hf_cache();
    let temp = tempdir().expect("tempdir creates");
    let mut settings = test_settings("http://127.0.0.1:1".to_owned(), None);
    settings.data_dir = temp.path().to_path_buf();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let mut job_json = job_snapshot_json("job-1", false);
    job_json["type"] = json!("lora_download");
    job_json["payload"] = json!({
        "loraId": "probe",
        "repo": "owner/adapter",
        "revision": "../../evil",
        "files": ["*.safetensors"],
    });
    let job: JobSnapshot = serde_json::from_value(job_json).expect("job deserializes");

    let error = super::model_jobs::run_lora_download_job(&api, &settings, &client, &job)
        .await
        .expect_err("a path-escaping revision is rejected at the entry point");
    assert!(
        error
            .to_string()
            .contains("traversal components are not allowed"),
        "expected the revision validator to fire, got {error:?}"
    );
}

#[tokio::test]
async fn lora_import_rejects_a_path_escaping_revision() {
    let _env = isolate_hf_cache();
    let temp = tempdir().expect("tempdir creates");
    // Imports heartbeat / post progress BEFORE the HF branch validates, so give them a live stub; the
    // rejection must still come from the revision validator, not a transport error.
    let (base_url, _posts) = spawn_tree_stub_with_files(vec![("model.safetensors", 8)]).await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url.clone();
    settings.data_dir = temp.path().to_path_buf();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let mut job_json = job_snapshot_json("job-1", false);
    job_json["type"] = json!("lora_import");
    job_json["payload"] = json!({
        "loraId": "probe",
        "repo": "owner/adapter",
        "revision": "../../evil",
        "files": ["*.safetensors"],
    });
    let job: JobSnapshot = serde_json::from_value(job_json).expect("job deserializes");

    let error = super::model_jobs::run_lora_import_job(&api, &settings, &client, &job)
        .await
        .expect_err("a path-escaping revision is rejected in the HF import branch");
    assert!(
        error
            .to_string()
            .contains("traversal components are not allowed"),
        "expected the revision validator to fire, got {error:?}"
    );
}

#[tokio::test]
async fn model_import_rejects_a_path_escaping_revision() {
    let _env = isolate_hf_cache();
    let temp = tempdir().expect("tempdir creates");
    let (base_url, _posts) = spawn_tree_stub_with_files(vec![("model.safetensors", 8)]).await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url.clone();
    settings.data_dir = temp.path().to_path_buf();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let mut job_json = job_snapshot_json("job-1", false);
    job_json["type"] = json!("model_import");
    job_json["payload"] = json!({
        "modelId": "probe",
        "repo": "owner/model",
        "revision": "../../evil",
        "files": ["*.safetensors"],
    });
    let job: JobSnapshot = serde_json::from_value(job_json).expect("job deserializes");

    let error = super::model_jobs::run_model_import_job(&api, &settings, &client, &job)
        .await
        .expect_err("a path-escaping revision is rejected in the HF import branch");
    assert!(
        error
            .to_string()
            .contains("traversal components are not allowed"),
        "expected the revision validator to fire, got {error:?}"
    );
}

/// The join-level twin of the entry-point guards: `download_snapshot_into_cache` joins `revision`
/// into `refs/<rev>` (and, via the `commit` fallback, `snapshots/<rev>`). A `..` revision must be
/// confined by `safe_join`, so the receipt pointer can never be written ABOVE the repo cache dir —
/// defense in depth for any caller that reaches this without the entry validator.
#[tokio::test]
async fn download_snapshot_into_cache_confines_a_path_escaping_revision() {
    let temp = tempdir().expect("tempdir creates");
    let base_url = spawn_binary_stub(b"weights!!".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    // Nest the repo dir two levels deep so an un-guarded `../../` revision would land at
    // `temp/cache/pwned` — outside the repo cache dir.
    let repo_dir = temp.path().join("cache").join("models--owner--model");

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
        "../../pwned",
        &snapshot,
        &mut progress,
    )
    .await
    .expect_err("a path-escaping revision is rejected");
    assert!(
        error.to_string().contains("Unsafe snapshot path"),
        "expected a safe_join rejection, got {error:?}"
    );
    // The escaped `refs/<rev>` pointer must never have been written above the cache dir.
    assert!(
        !temp.path().join("cache").join("pwned").exists(),
        "the refs/<rev> write must stay confined to the repo cache dir"
    );
}

/// sc-13810 (item 3) — **network-free, cross-platform** proof of the pinned-revision hub layout.
///
/// The live `whisper_base_fresh_install_pinned_layout_smoke` (sc-13797) is confined to the
/// self-hosted macOS lane: it pulls ~279 MiB from the real hub, so spreading it onto hosted Windows
/// and fork PRs would amplify exactly the transient flakiness its transport-skip already works
/// around. The consequence was that `downloads.rs`'s layout contract — which is entirely
/// platform-agnostic code — had NO coverage on the platforms where its filesystem behavior actually
/// differs (Windows has no symlinks for the blob→snapshot placement and falls back to a copy).
///
/// This test closes that hole with zero internet: a local stub serves the HF tree API and the file
/// bytes, and the REAL seam (`HuggingFaceSnapshot::resolve` + `download_snapshot_into_cache`) runs
/// against it. It therefore runs on EVERY lane — hosted Windows, Linux, fork PRs — and cannot flake
/// on hub availability.
///
/// The pin comes from the builtin manifest (sc-13810 item 2), so an allow-list or revision edit is
/// carried into the assertions rather than mirrored beside them.
#[tokio::test]
async fn pinned_snapshot_layout_materializes_without_network() {
    let pin = crate::manifest_pins::builtin_model_pin("whisper_base");
    let temp = tempdir().expect("tempdir creates");
    // A file the manifest does NOT allow-list. whisper-base really does ship many more files
    // upstream, so serving one here makes "only the allow-listed files materialize" a
    // DISCRIMINATING assertion rather than a tautology over a stub that offered nothing else.
    let decoy = "pytorch_model.bin";
    let (base_url, contents) = spawn_pinned_repo_stub(&pin, decoy).await;

    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let repo_dir = temp.path().join(pin.repo_slug());

    let snapshot = HuggingFaceSnapshot::resolve(
        &client,
        &settings,
        &pin.repo,
        &pin.revision,
        &pin.files,
    )
    .await
    .expect("resolves the stubbed tree");
    assert_eq!(
        snapshot.files.len(),
        pin.files.len(),
        "the allow-list must filter the tree down to exactly the manifest's files, got {:?}",
        snapshot
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>()
    );

    let mut progress = DownloadProgress::new(
        &pin.repo,
        0,
        snapshot.total_bytes(),
        Duration::from_secs(3600),
    );
    let commit = download_snapshot_into_cache(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "sc-13810",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &repo_dir,
        &pin.revision,
        &snapshot,
        &mut progress,
    )
    .await
    .expect("materializes the pinned layout");

    // A pinned commit-SHA revision resolves to itself, so the snapshot dir the OFFLINE resolver
    // reads (`snapshots/<sha>/`) is the one written here.
    assert_eq!(
        commit, pin.revision,
        "a pinned commit revision must resolve to itself"
    );
    let refs_pointer = repo_dir.join("refs").join(&pin.revision);
    assert_eq!(
        tokio::fs::read_to_string(&refs_pointer)
            .await
            .expect("refs/<sha> pointer written")
            .trim(),
        pin.revision,
        "refs/<rev> must record the resolved commit (the layout's rev→commit link)"
    );

    // Every allow-listed file resolves to its exact content through the snapshot placement —
    // symlink on unix, copy on Windows. Reading the bytes back (rather than checking existence)
    // is what makes this meaningful on the copy path.
    let snapshot_dir = repo_dir.join("snapshots").join(&commit);
    for file in &pin.files {
        let placed = tokio::fs::read(snapshot_dir.join(file))
            .await
            .unwrap_or_else(|error| panic!("{file} must materialize in the snapshot: {error}"));
        assert_eq!(
            placed,
            contents[file.as_str()],
            "{file} must resolve to its downloaded content"
        );
    }

    // ONLY the allow-listed files materialized — the manifest allow-list, not the repo, decides
    // what is fetched. A regression that ignored the filter would drag the decoy in.
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
        "only the manifest's allow-listed files may materialize (the whole repo is NOT pulled)"
    );
    assert!(
        !snapshot_dir.join(decoy).exists(),
        "the non-allow-listed {decoy} must never be fetched"
    );
}

/// Serve one pinned HF repo — the tree listing plus the file bytes — from a local socket, so the
/// real download seam can run with no internet. Returns the base URL and the exact bytes served for
/// each allow-listed file, so the caller can assert content round-trips through the placement.
///
/// The HEAD mirrors the metadata `download_snapshot_into_cache` reads: `x-repo-commit` (set to the
/// PINNED sha, so the layout is asserted against the pin) and an `ETag` that names the blob.
async fn spawn_pinned_repo_stub(
    pin: &crate::manifest_pins::ManifestPin,
    decoy: &str,
) -> (String, HashMap<String, Vec<u8>>) {
    // Deterministic per-file content, distinct per name so a mixed-up placement is detectable.
    let mut contents: HashMap<String, Vec<u8>> = pin
        .files
        .iter()
        .map(|file| (file.clone(), format!("content of {file}").into_bytes()))
        .collect();
    contents.insert(
        decoy.to_owned(),
        format!("content of {decoy}").into_bytes(),
    );

    #[derive(Clone)]
    struct PinnedRepoState {
        contents: HashMap<String, Vec<u8>>,
        revision: String,
    }

    // The tree API lists the allow-listed files AND the decoy; `resolve` applies the filter.
    async fn tree(State(state): State<PinnedRepoState>) -> Response {
        let mut entries: Vec<Value> = state
            .contents
            .iter()
            .map(|(path, bytes)| json!({ "type": "file", "path": path, "size": bytes.len() }))
            .collect();
        // Stable order so a failure message reads the same way twice.
        entries.sort_by_key(|entry| {
            entry
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        });
        Json(entries).into_response()
    }

    // `resolve` builds `<base>/<repo>/resolve/<revision>/<path>`, so the trailing segment after the
    // revision is the repo-relative file path.
    fn requested_file(state: &PinnedRepoState, path: &str) -> Option<Vec<u8>> {
        let marker = format!("/resolve/{}/", state.revision);
        let relative = path.split_once(&marker).map(|(_, tail)| tail)?;
        state.contents.get(relative).cloned()
    }

    async fn file_get(
        State(state): State<PinnedRepoState>,
        axum::extract::Path(path): axum::extract::Path<String>,
    ) -> Response {
        match requested_file(&state, &format!("/{path}")) {
            Some(bytes) => bytes.into_response(),
            None => (AxumStatusCode::NOT_FOUND, "no such file").into_response(),
        }
    }

    async fn file_head(
        State(state): State<PinnedRepoState>,
        axum::extract::Path(path): axum::extract::Path<String>,
    ) -> Response {
        let Some(bytes) = requested_file(&state, &format!("/{path}")) else {
            return (AxumStatusCode::NOT_FOUND, "no such file").into_response();
        };
        let mut response = Response::new(Body::empty());
        let headers = response.headers_mut();
        headers.insert(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from_str(&bytes.len().to_string())
                .expect("content length header"),
        );
        let last_segment = path.rsplit('/').next().unwrap_or("blob");
        headers.insert(
            axum::http::header::ETAG,
            axum::http::HeaderValue::from_str(&format!("\"etag-{last_segment}\""))
                .expect("etag header"),
        );
        // The pinned sha, so `refs/<rev>` and `snapshots/<commit>` are asserted against the pin.
        headers.insert(
            "x-repo-commit",
            axum::http::HeaderValue::from_str(&state.revision).expect("commit header"),
        );
        response
    }

    let state = PinnedRepoState {
        contents: contents.clone(),
        revision: pin.revision.clone(),
    };
    let app = Router::new()
        .route("/api/models/:owner/:repo/tree/:revision", get(tree))
        .route("/*path", get(file_get).head(file_head))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    (format!("http://{address}"), contents)
}

/// A tree stub that paginates: page 1 (no cursor) returns bf16/q4 files plus a `Link: rel="next"`
/// header pointing at page 2; page 2 (cursor=next) returns the q8 file and no further link.
async fn spawn_paginated_tree_stub() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    let base = format!("http://{address}");
    let base_for_handler = base.clone();
    async fn page(
        base: String,
        axum::extract::RawQuery(query): axum::extract::RawQuery,
    ) -> Response {
        let on_page_two = query.as_deref().is_some_and(|q| q.contains("cursor=next"));
        if on_page_two {
            // The later tier — only visible if pagination is followed.
            Json(json!([
                { "type": "file", "path": "q8/unet/model.safetensors", "size": 3_000_000_000u64 }
            ]))
            .into_response()
        } else {
            let mut response = Json(json!([
                { "type": "file", "path": "bf16/model_index.json", "size": 600 },
                { "type": "file", "path": "q4/model_index.json", "size": 600 }
            ]))
            .into_response();
            let link = format!("<{base}/api/models/owner/repo/tree/main?recursive=true&expand=true&limit=50&cursor=next>; rel=\"next\"");
            response.headers_mut().insert(
                reqwest::header::LINK,
                axum::http::HeaderValue::from_str(&link).expect("valid link header"),
            );
            response
        }
    }
    let app = Router::new().route(
        "/api/models/:owner/:repo/tree/:revision",
        get(move |raw| page(base_for_handler.clone(), raw)),
    );
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    base
}

#[tokio::test]
async fn resolve_follows_tree_pagination_across_pages() {
    // sc-9909: the HF tree `expand=1` view is paginated (limit 50). A later tier's files land on a
    // subsequent page, so resolve MUST follow `Link: rel="next"` — otherwise a `q8/*` filter matches
    // nothing (the q8 files are on page 2) and the tier silently "downloads" empty.
    let base_url = spawn_paginated_tree_stub().await;
    let settings = test_settings(base_url, None);
    let client = reqwest::Client::new();

    let snapshot =
        HuggingFaceSnapshot::resolve(&client, &settings, "owner/repo", "main", &["q8/*".to_owned()])
            .await
            .expect("resolves across pages");

    let paths: Vec<&str> = snapshot.files.iter().map(|file| file.path.as_str()).collect();
    assert_eq!(
        paths,
        vec!["q8/unet/model.safetensors"],
        "the q8 file from page 2 must be resolved (pagination followed)"
    );
}

// ---- sc-20636: managed model import is transactional end to end -----------------------------

/// An API stub that always reports the job as cancelled, so `check_cancel` trips on the import
/// job's first poll.
async fn spawn_cancelling_job_stub() -> String {
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    async fn progress_route(
        axum::extract::Path(job_id): axum::extract::Path<String>,
        Json(_body): Json<Value>,
    ) -> Response {
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    async fn heartbeat_route(
        axum::extract::Path(worker_id): axum::extract::Path<String>,
    ) -> Response {
        Json(worker_snapshot_json(&worker_id)).into_response()
    }
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route("/api/v1/workers/:worker_id/heartbeat", post(heartbeat_route));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

/// A minimal single-file Krea 2 native DiT the base-weight detector and the checkpoint inspector
/// both recognise, with deterministic bytes.
fn write_krea_native_safetensors(path: &std::path::Path, fill: u8) {
    let entries = [
        ("model.diffusion_model.txtfusion.projector.weight", "BF16"),
        ("model.diffusion_model.blocks.0.attn.wq.weight", "BF16"),
        ("model.diffusion_model.first.weight", "BF16"),
    ];
    let mut header = serde_json::Map::new();
    let mut offset = 0_u64;
    for (name, dtype) in entries {
        header.insert(
            name.to_owned(),
            json!({"dtype": dtype, "shape": [1], "data_offsets": [offset, offset + 2]}),
        );
        offset += 2;
    }
    let encoded = serde_json::to_vec(&Value::Object(header)).expect("header serializes");
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.resize(bytes.len() + offset as usize, fill);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("dir creates");
    std::fs::write(path, bytes).expect("fixture writes");
}

fn sha256_hex(path: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(std::fs::read(path).expect("fixture reads"));
    format!("{:x}", hasher.finalize())
}

struct ManagedImportFixture {
    _temp: tempfile::TempDir,
    settings: Settings,
    source: PathBuf,
    install_dir: PathBuf,
    manifest_path: PathBuf,
}

async fn managed_import_fixture(base_url: String) -> ManagedImportFixture {
    let temp = tempdir().expect("tempdir creates");
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    settings.data_dir = temp.path().join("data");
    settings.config_dir = temp.path().join("config");
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).expect("manifest dir creates");
    let manifest_path = manifests.join("user.models.jsonc");
    std::fs::write(&manifest_path, r#"{ "schemaVersion": 1, "models": [] }"#)
        .expect("manifest writes");
    let source = settings.data_dir.join("models/incoming/kreamania.safetensors");
    write_krea_native_safetensors(&source, 0x5a);
    let install_dir = settings.data_dir.join("models/imports/managed_krea");
    ManagedImportFixture {
        _temp: temp,
        settings,
        source,
        install_dir,
        manifest_path,
    }
}

impl ManagedImportFixture {
    fn job(&self, expected_sha256: Option<&str>) -> JobSnapshot {
        let mut job_json = job_snapshot_json("job-1", false);
        job_json["type"] = json!("model_import");
        job_json["payload"] = json!({
            "modelId": "managed_krea",
            "modelName": "Managed Krea",
            "sourcePath": self.source.to_str().expect("utf-8 source"),
            "targetDir": self.install_dir.to_str().expect("utf-8 target"),
            "manifestPath": self.manifest_path.to_str().expect("utf-8 manifest"),
            "expectedSha256": expected_sha256,
            "ownershipMode": "managed",
            "importProvenance": {
                "source": "civitai",
                "reference": "modelVersion/9931",
                "url": "https://civitai.com/api/download/models/9931?type=Model",
                "versionId": "9931",
                "fileId": "40277",
                "credentialHost": "civitai.com"
            },
            "manifestEntry": {
                "id": "managed_krea",
                "name": "Managed Krea",
                "type": "image",
                "family": "krea_2",
                "source": { "provider": "civitai", "path": "models/imports/managed_krea" },
            },
        });
        serde_json::from_value(job_json).expect("job deserializes")
    }

    fn manifest_entry(&self) -> Option<Value> {
        let raw = std::fs::read_to_string(&self.manifest_path).expect("manifest reads");
        let parsed: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&raw))
                .expect("manifest parses");
        parsed
            .get("models")?
            .as_array()?
            .iter()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some("managed_krea"))
            .cloned()
    }

    /// The invariant a refused import must satisfy: nothing installed, nothing in the manifest,
    /// nothing in the checkpoint store, and no staging left behind.
    fn assert_nothing_installed(&self, context: &str) {
        assert!(
            !self.install_dir.exists(),
            "{context}: a partial install survived at {}",
            self.install_dir.display()
        );
        assert!(
            self.manifest_entry().is_none(),
            "{context}: a manifest entry was written for a refused import"
        );
        let store = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(
            &self.settings.data_dir,
        );
        assert!(store.record("managed/managed_krea").is_err(), "{context}");
        assert_eq!(
            store.inventory().map(|inventory| inventory.records.len()),
            Ok(0),
            "{context}"
        );
        let staging = store.staging_root().join("managed_krea");
        assert!(!staging.exists(), "{context}: staging survived");
        assert!(
            self.source.is_file(),
            "{context}: the user's own source file must never be touched"
        );
    }
}

/// AC1 + AC2 + AC3: a legacy-shaped local-copy import runs through the transactional managed path,
/// commits atomically, publishes a resolvable plan, and stamps the manifest entry so the install is
/// selectable through the plan route.
#[tokio::test]
async fn managed_model_import_commits_atomically_and_stamps_the_plan_binding() {
    let _env = isolate_hf_cache();
    let (base_url, posts) = spawn_tree_stub_with_files(vec![("model.safetensors", 8)]).await;
    let fixture = managed_import_fixture(base_url).await;
    let expected = sha256_hex(&fixture.source);
    let api = ApiClient::new(&fixture.settings);
    let client = reqwest::Client::new();

    // A LINKED twin of the very bytes about to be imported, compiled before the import runs. The
    // compile reports it as a duplicate; neither copy is ever deleted, because keeping both is a
    // legitimate choice. Which makes it the user's decision — so it has to reach the user, and a
    // `tracing::info!` never does (sc-20636 review).
    let library = fixture.source.parent().expect("source has a parent");
    let twin_store = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(
        &fixture.settings.data_dir,
    );
    let root = twin_store
        .approve_root(library)
        .expect("the incoming dir approves as a library root");
    let twin = twin_store
        .compile_linked(&root.root_id, "kreamania.safetensors")
        .expect("a linked twin compiles");

    super::model_jobs::run_model_import_job(
        &api,
        &fixture.settings,
        &client,
        &fixture.job(Some(&expected)),
    )
    .await
    .expect("a valid managed import succeeds");

    // The duplicate is on the JOB RESULT, naming the checkpoint the user already has.
    let completed = posts
        .lock()
        .expect("posts lock")
        .iter()
        .rev()
        .find(|post| post.get("status").and_then(Value::as_str) == Some("completed"))
        .cloned()
        .expect("the import posts a completed update");
    assert_eq!(
        completed["result"]["duplicateCheckpointIds"],
        json!([twin.checkpoint_id]),
        "the import must report the checkpoint the user already has: {completed}"
    );
    // ...and both copies survive: reporting is never acting.
    assert!(fixture.install_dir.join("kreamania.safetensors").is_file());
    assert!(twin_store.resolve(&twin.checkpoint_id).is_ok());

    // Committed: the install is at the path imported models have always occupied.
    assert!(fixture.install_dir.join("kreamania.safetensors").is_file());
    assert_eq!(sha256_hex(&fixture.install_dir.join("kreamania.safetensors")), expected);
    assert!(fixture.source.is_file(), "a local copy never moves the user's file");

    // Published: the plan resolves and re-verifies, and its locator is MANAGED with provenance.
    let store =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&fixture.settings.data_dir);
    let resolved = store
        .resolve("managed/managed_krea")
        .expect("the published plan resolves");
    assert_eq!(resolved.family(), "krea_2");
    let sceneworks_core::checkpoint_import::SourceLocatorV1::Managed {
        install_id,
        sha256,
        provenance,
        ..
    } = &resolved.plan.layers[0].source
    else {
        panic!("a managed import must compile managed locators");
    };
    assert_eq!(install_id, "managed_krea");
    assert_eq!(sha256, &expected);
    assert_eq!(provenance.source, "civitai");
    assert_eq!(provenance.version_id.as_deref(), Some("9931"));
    assert_eq!(provenance.credential_host.as_deref(), Some("civitai.com"));

    // Bound: the manifest entry carries the checkpoint identity the plan route resolves through,
    // beside the installed path its family's bespoke lane still uses.
    let entry = fixture.manifest_entry().expect("the import writes a manifest entry");
    assert_eq!(entry["importPlan"]["checkpointId"], json!("managed/managed_krea"));
    assert_eq!(entry["importSourceShape"], json!("transformer_file"));
    assert_eq!(
        entry["paths"]["model"],
        json!(std::fs::canonicalize(&fixture.install_dir)
            .expect("install dir canonicalizes")
            .display()
            .to_string()),
        "the entry keeps the installed path its family's bespoke lane loads from"
    );
    assert_eq!(
        sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(
            entry.as_object().expect("entry is an object")
        ),
        Some("managed/managed_krea")
    );

    // Nothing is left staged.
    assert!(!store.staging_root().join("managed_krea").exists());
}

/// AC1: a hash-mismatched ingest leaves no runnable partial install — no install directory, no
/// manifest entry, no plan, no catalog record — AND the terminal failure it reports NAMES the
/// digest mismatch. "Nothing installed" alone is satisfied by any failure whatsoever (a 404, a
/// panic in the compiler, a disk error), so without the message assertions this test would pass
/// for a refusal that never checked a digest at all.
#[tokio::test]
async fn a_hash_mismatched_managed_import_leaves_no_partial_install() {
    let _env = isolate_hf_cache();
    let (base_url, posts) = spawn_tree_stub_with_files(vec![("model.safetensors", 8)]).await;
    let fixture = managed_import_fixture(base_url).await;
    // The digest the ingest will actually compute over the bytes it staged, derived here from the
    // source file itself rather than restated — the declared digest below is deliberately not it.
    let actual = sha256_hex(&fixture.source);
    let declared = "a".repeat(64);
    assert_ne!(actual, declared, "the fixture must really mismatch");
    let api = ApiClient::new(&fixture.settings);
    let client = reqwest::Client::new();

    super::model_jobs::run_model_import_job(&api, &fixture.settings, &client, &fixture.job(Some(&declared)))
        .await
        .expect("a hash mismatch fails the job rather than erroring the worker");

    fixture.assert_nothing_installed("hash mismatch");

    let posts = posts.lock().expect("posts lock");
    let failure = posts
        .iter()
        .find(|post| post.get("status").and_then(Value::as_str) == Some("failed"))
        .unwrap_or_else(|| panic!("expected a failed progress post, got {posts:?}"));
    let detail = failure
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("a failed import must carry an error detail: {failure:?}"));
    assert!(
        detail.contains("integrity check"),
        "the failure must name the integrity check, got {detail:?}"
    );
    // Both digests, so the operator can see WHICH bytes arrived and what was expected.
    assert!(
        detail.contains(&actual),
        "the failure must name the digest actually computed ({actual}), got {detail:?}"
    );
    assert!(
        detail.contains(&declared),
        "the failure must name the declared digest ({declared}), got {detail:?}"
    );
}

/// AC1: a cancelled ingest leaves no runnable partial install. The cancel arrives through the same
/// `check_cancel` the transfer already polls, and the staging session's destructor is what makes it
/// leave nothing behind.
#[tokio::test]
async fn a_cancelled_managed_import_leaves_no_partial_install() {
    let _env = isolate_hf_cache();
    let base_url = spawn_cancelling_job_stub().await;
    let fixture = managed_import_fixture(base_url).await;
    let api = ApiClient::new(&fixture.settings);
    let client = reqwest::Client::new();

    let error = super::model_jobs::run_model_import_job(
        &api,
        &fixture.settings,
        &client,
        &fixture.job(None),
    )
    .await
    .expect_err("a cancelled import stops the job");
    assert!(
        matches!(error, WorkerError::Canceled(_)),
        "expected a cancellation, got {error:?}"
    );

    fixture.assert_nothing_installed("cancel");
}
