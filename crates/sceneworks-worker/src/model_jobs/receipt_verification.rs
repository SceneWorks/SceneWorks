//! Cold recovery of stale install stamps against the immutable upstream content.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use super::*;

/// Runs before MLX admission, never once per ladder candidate. Healthy receipts remain offline.
/// A changed mount ID is indistinguishable from file drift in old metadata-only receipts, so
/// revalidate every recorded file against its pinned upstream revision before replacing the stamp.
pub(crate) async fn ensure_huggingface_receipt_provenance(
    settings: &Settings,
    repo: &str,
    model_id: &str,
    variant: Option<&str>,
    weights_dir: &Path,
) -> WorkerResult<()> {
    let Some(resolved) = huggingface_receipt_weights(
        &settings.data_dir,
        repo,
        Some(model_id),
        variant,
        ProvenanceRepair::Skip,
    ) else {
        return Ok(());
    };
    if !weights_dir.starts_with(&resolved.path) {
        return Ok(());
    }
    if resolved.provenance.is_some() {
        let files = resolved
            .receipt
            .get("resolvedFiles")
            .and_then(Value::as_array)
            .and_then(|files| files.iter().map(Value::as_str).collect::<Option<Vec<_>>>())
            .ok_or_else(|| WorkerError::InvalidPayload("invalid receipt file list".to_owned()))?;
        let stable = resolved_files_tree_stamp(&resolved.snapshot, &files)?;
        if resolved.receipt["artifactTreeStamp"].as_str() != Some(&stable) {
            // The old baseline still matches in this boot. Upgrade it now while that proof is
            // available, so the next mount does not need a network/content verification pass.
            establish_receipt_tree_stamp(
                &resolved.marker,
                &resolved.receipt,
                &resolved.snapshot,
                &files,
                "verified-legacy-metadata",
                Some(&stable),
            )?;
        }
        return Ok(());
    }
    // Unstamped legacy installs retain the existing local baseline-establishment path.
    if resolved
        .receipt
        .get("artifactTreeStamp")
        .is_none_or(Value::is_null)
    {
        return Ok(());
    }
    revalidate_receipt(settings, repo, &resolved).await.map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "Cannot verify the installed {model_id} artifact after its file metadata changed: {error}. Check the connection or repair the model download."
        ))
    })
}

async fn revalidate_receipt(
    settings: &Settings,
    repo: &str,
    resolved: &ResolvedWeights,
) -> WorkerResult<()> {
    let invalid = |message: &str| WorkerError::InvalidPayload(message.to_owned());
    let revision = resolved
        .snapshot
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("missing snapshot revision"))?;
    sceneworks_core::model_artifacts::validate_immutable_revision(revision)
        .map_err(|error| invalid(&error.to_string()))?;
    let files = resolved
        .receipt
        .get("resolvedFiles")
        .and_then(Value::as_array)
        .and_then(|files| files.iter().map(Value::as_str).collect::<Option<Vec<_>>>())
        .filter(|files| !files.is_empty())
        .ok_or_else(|| invalid("invalid receipt file list"))?;
    // Also validates that every recorded path is confined and readable.
    let before = resolved_files_tree_stamp(&resolved.snapshot, &files)?;
    let client = crate::downloads::streaming_download_client();
    let remote = HuggingFaceSnapshot::resolve(
        &client,
        settings,
        repo,
        revision,
        &files
            .iter()
            .map(|file| (*file).to_owned())
            .collect::<Vec<_>>(),
    )
    .await?;
    for name in &files {
        let entries = remote
            .files
            .iter()
            .filter(|file| file.path == *name)
            .collect::<Vec<_>>();
        let [file] = entries.as_slice() else {
            return Err(invalid(&format!(
                "upstream revision does not uniquely identify {name}"
            )));
        };
        let path = resolved.snapshot.join(name);
        let size = file
            .size
            .ok_or_else(|| invalid("upstream file size missing"))?;
        if std::fs::metadata(&path)?.len() != size {
            return Err(invalid(&format!("installed file size differs: {name}")));
        }
        let expected =
            crate::downloads::huggingface_file_content_sha256(&client, settings, file).await?;
        let actual = tokio::task::spawn_blocking(move || -> std::io::Result<String> {
            use std::io::Read;
            let mut file = std::fs::File::open(path)?;
            let mut digest = Sha256::new();
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
            }
            Ok(format!("{:x}", digest.finalize()))
        })
        .await
        .map_err(|error| invalid(&error.to_string()))??;
        if actual != expected {
            return Err(invalid(&format!(
                "installed content checksum differs: {name}"
            )));
        }
    }
    establish_receipt_tree_stamp(
        &resolved.marker,
        &resolved.receipt,
        &resolved.snapshot,
        &files,
        "verified-content",
        Some(&before),
    )?;
    tracing::info!(event = "artifact_tree_stamp_content_verified", repo, revision,
        "restored installed artifact identity after verifying every file against its pinned revision");
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use axum::{routing::get, Json, Router};
    use sceneworks_core::image_request::ImageRequest;
    use serde_json::json;

    fn stamp_from_previous_mount(root: &Path, files: &[&str]) -> String {
        use std::os::unix::fs::MetadataExt;
        let mut names = files.to_vec();
        names.sort_unstable();
        let mut digest = Sha256::new();
        for name in names {
            let m = std::fs::metadata(root.join(name)).unwrap();
            let modified = m
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            digest.update(name.as_bytes());
            digest.update([0]);
            digest.update(m.len().to_le_bytes());
            digest.update(modified.as_secs().to_le_bytes());
            digest.update(modified.subsec_nanos().to_le_bytes());
            digest.update((m.dev() + 1).to_le_bytes());
            digest.update(m.ino().to_le_bytes());
            digest.update(m.ctime().to_le_bytes());
            digest.update(m.ctime_nsec().to_le_bytes());
            digest.update([0xff]);
        }
        format!("sha256:{:x}", digest.finalize())
    }

    #[tokio::test]
    async fn stale_receipt_recovers_sdxl_decode_identity_only_for_verified_content() {
        let data = tempfile::tempdir().unwrap();
        let hub = data.path().join("hub");
        let _env = crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().unwrap())]);
        let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
            include_str!("../../../../config/manifests/builtin.models.jsonc"),
        ))
        .unwrap();
        let model = manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == "sdxl")
            .unwrap()
            .as_object()
            .unwrap();
        let download = model["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["variant"] == "q4")
            .unwrap();
        let repo = download["repo"].as_str().unwrap();
        let revision = download["revision"].as_str().unwrap();
        let snapshot = hub
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(revision);
        let weights = snapshot.join("q4");
        std::fs::create_dir_all(&weights).unwrap();
        std::fs::write(weights.join("model.safetensors"), b"original").unwrap();
        std::fs::write(weights.join("config.json"), b"{}").unwrap();
        let files = ["q4/model.safetensors", "q4/config.json"];
        let marker = data
            .path()
            .join("models")
            .join(safe_download_dir(repo))
            .join(INSTALL_MARKER);
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        let receipt = json!({"repo":repo,"modelId":"sdxl","variant":"q4",
            "snapshotRevision":revision,"resolvedFiles":files,
            "artifactTreeStamp":stamp_from_previous_mount(&snapshot, &files)});
        let q8 = snapshot.join("q8");
        std::fs::create_dir_all(&q8).unwrap();
        std::fs::write(q8.join("model.safetensors"), b"q8 weights").unwrap();
        let q8_receipt = json!({"repo":repo,"modelId":"sdxl","variant":"q8",
            "snapshotRevision":revision,"resolvedFiles":["q8/model.safetensors"],
            "artifactTreeStamp":resolved_files_tree_stamp(&snapshot, &["q8/model.safetensors"]).unwrap()});
        let mirrored = json!({"repo":repo,"modelId":"sdxl","variant":"q4",
            "snapshotRevision":revision,"resolvedFiles":files,
            "artifactTreeStamp":receipt["artifactTreeStamp"],"receipts":[receipt,q8_receipt]});
        std::fs::write(&marker, serde_json::to_vec(&mirrored).unwrap()).unwrap();
        let request = ImageRequest::from_payload(
            json!({"model":"sdxl","advanced":{"mlxQuantize":8},
            "modelManifestEntry":model})
            .as_object()
            .unwrap(),
        );
        let mut settings =
            crate::image_jobs::resolved_artifact_provenance_tests::settings(data.path());
        let resolved = crate::image_jobs::resolve_weights_dir(&request, &settings)
            .unwrap()
            .unwrap();
        assert_eq!(resolved, q8);
        assert!(crate::image_jobs::resolved_mlx_artifact_provenance(
            &request,
            &settings,
            repo,
            &weights,
            Some("q4")
        )
        .unwrap()
        .is_none());
        let hash = format!("{:x}", Sha256::digest(b"original"));
        let listing = json!([
            {"type":"file","path":files[0],"size":8,"lfs":{"oid":hash}},
            {"type":"file","path":files[1],"size":2}
        ]);
        let app = Router::new()
            .route(
                &format!("/api/models/{repo}/tree/{revision}"),
                get(move || {
                    let listing = listing.clone();
                    async move { Json(listing) }
                }),
            )
            .route(
                &format!("/{repo}/resolve/{revision}/q4/config.json"),
                get(|| async { "{}" }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        settings.huggingface_base_url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        // A fitting q8 must never consult the stale q4 receipt (or require its network recovery).
        let before = std::fs::read(&marker).unwrap();
        crate::image_jobs::choose_mlx_request_tier(
            vec![resolved.clone(), weights.clone()],
            |dir| {
                let request = &request;
                let settings = &settings;
                let q8 = &q8;
                async move {
                    crate::image_jobs::ensure_mlx_candidate_provenance(
                        request, settings, repo, &dir,
                    )
                    .await?;
                    assert_eq!(&dir, q8, "unused q4 must not be verified");
                    Ok(crate::image_jobs::MlxTierFit::Fits(()))
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read(&marker).unwrap(), before);
        // When q8 does not fit, the same production selector must recover q4 before scoring it.
        crate::image_jobs::choose_mlx_request_tier(
            vec![resolved.clone(), weights.clone()],
            |dir| {
                let request = &request;
                let settings = &settings;
                let q8 = &q8;
                async move {
                    crate::image_jobs::ensure_mlx_candidate_provenance(
                        request, settings, repo, &dir,
                    )
                    .await?;
                    Ok(if &dir == q8 {
                        crate::image_jobs::MlxTierFit::TooBig(WorkerError::InvalidPayload(
                            "q8 exceeds budget".to_owned(),
                        ))
                    } else {
                        crate::image_jobs::MlxTierFit::Fits(())
                    })
                }
            },
        )
        .await
        .unwrap();
        let provenance = crate::image_jobs::resolved_mlx_artifact_provenance(
            &request,
            &settings,
            repo,
            &weights,
            Some("q4"),
        )
        .unwrap()
        .unwrap();
        let binding = crate::mlx_fit_gate::bind_decode_quality_policies_from_manifest(
            model,
            "sdxl",
            Some(&provenance),
        )
        .unwrap();
        assert!(
            !binding.policies.is_empty(),
            "the production SDXL decode policy must bind"
        );
        let repaired: Value = serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
        assert_eq!(repaired["artifactTreeStampSource"], "verified-content");
        assert_eq!(
            repaired["artifactTreeStamp"],
            repaired["receipts"][0]["artifactTreeStamp"]
        );
        // A content change with the same size must never inherit the upstream identity.
        std::fs::write(weights.join("model.safetensors"), b"modified").unwrap();
        let before = std::fs::read(&marker).unwrap();
        let error =
            ensure_huggingface_receipt_provenance(&settings, repo, "sdxl", Some("q4"), &weights)
                .await
                .unwrap_err();
        assert!(error.to_string().contains("checksum differs"));
        assert_eq!(std::fs::read(&marker).unwrap(), before);
        std::fs::write(weights.join("model.safetensors"), b"original").unwrap();
        ensure_huggingface_receipt_provenance(&settings, repo, "sdxl", Some("q4"), &weights)
            .await
            .unwrap();
        server.abort();
        // Once verified, admission works offline and does not rewrite the receipt.
        let before = std::fs::read(&marker).unwrap();
        ensure_huggingface_receipt_provenance(&settings, repo, "sdxl", Some("q4"), &weights)
            .await
            .unwrap();
        assert_eq!(std::fs::read(&marker).unwrap(), before);
    }
}
