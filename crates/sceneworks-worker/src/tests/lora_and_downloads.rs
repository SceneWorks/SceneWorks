
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
