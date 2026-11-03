const ARTIFACT_PROVENANCE_MARKER: &str = ".sceneworks-artifact-provenance.json";
const ARTIFACT_PROVENANCE_SCHEMA_VERSION: u32 = 1;
const ARTIFACT_PROVENANCE_LOCK: &str = ".sceneworks-artifact-provenance.lock";

fn lock_app_managed_receipt(root: &Path) -> std::io::Result<FileLock> {
    FileLock::open_exclusive(&root.join(ARTIFACT_PROVENANCE_LOCK))
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AppManagedArtifactReceipt {
    schema_version: u32,
    repository: String,
    revision: String,
    variant: String,
    tier: String,
    resolved_path_fingerprint: String,
    tree_stamp: String,
}

pub(crate) fn resolved_artifact_fingerprint(
    repository: &str,
    revision: &str,
    variant: &str,
) -> String {
    format!("{repository}@{revision}:{variant}")
}

fn supported_artifact_tier(tier: &str) -> Option<String> {
    matches!(tier, "q4" | "q8" | "nvfp4" | "bf16" | "fp32").then(|| tier.to_owned())
}

pub(crate) fn is_sha256_fingerprint(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn receipt_tier(variant: &str) -> Option<String> {
    supported_artifact_tier(variant.rsplit('-').next().unwrap_or(variant))
}

pub(crate) fn download_payload_resolved_tier(payload: &JsonObject) -> Option<String> {
    payload
        .get("variant")
        .and_then(Value::as_str)
        .and_then(receipt_tier)
}

fn artifact_files(root: &Path) -> WorkerResult<Vec<PathBuf>> {
    fn visit(
        root: &Path,
        dir: &Path,
        active_dirs: &mut Vec<PathBuf>,
        files: &mut Vec<PathBuf>,
    ) -> WorkerResult<()> {
        let canonical_dir = std::fs::canonicalize(dir)?;
        if active_dirs.contains(&canonical_dir) {
            return Ok(());
        }
        active_dirs.push(canonical_dir);
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let relative = path.strip_prefix(root).map_err(|_| {
                WorkerError::InvalidPayload(format!(
                    "artifact entry {} escaped {}",
                    path.display(),
                    root.display()
                ))
            })?;
            if relative == Path::new(ARTIFACT_PROVENANCE_MARKER)
                || relative == Path::new(ARTIFACT_PROVENANCE_LOCK)
            {
                continue;
            }
            let metadata = std::fs::symlink_metadata(&path)?;
            if metadata.file_type().is_dir() {
                visit(root, &path, active_dirs, files)?;
            } else {
                files.push(path.clone());
                if metadata.file_type().is_symlink() && std::fs::metadata(&path)?.is_dir() {
                    // Converted layouts legitimately expose component directories through links
                    // into a shared cache. Walk the loader-visible alias while retaining the link
                    // itself in the digest. `active_dirs` breaks ancestor cycles without suppressing
                    // a second independent alias of the same component.
                    visit(root, &path, active_dirs, files)?;
                }
            }
        }
        active_dirs.pop();
        Ok(())
    }

    let mut files = Vec::new();
    visit(root, root, &mut Vec::new(), &mut files)?;
    files.sort_by(|left, right| {
        left.strip_prefix(root)
            .unwrap_or(left)
            .cmp(right.strip_prefix(root).unwrap_or(right))
    });
    Ok(files)
}

use sceneworks_core::download_receipt::update_metadata_stamp;

fn artifact_tree_stamp(root: &Path) -> WorkerResult<String> {
    artifact_tree_stamp_format(root, false)
}

fn artifact_tree_stamp_format(root: &Path, legacy: bool) -> WorkerResult<String> {
    let update_metadata_stamp = if legacy {
        sceneworks_core::download_receipt::update_legacy_metadata_stamp
    } else {
        update_metadata_stamp
    };
    let mut digest = Sha256::new();
    for path in artifact_files(root)? {
        let relative = path.strip_prefix(root).map_err(|_| {
            WorkerError::InvalidPayload(format!(
                "artifact entry {} escaped {}",
                path.display(),
                root.display()
            ))
        })?;
        let metadata = std::fs::symlink_metadata(&path)?;
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update([0]);
        update_metadata_stamp(&mut digest, &metadata);
        if metadata.file_type().is_symlink() {
            digest.update(std::fs::read_link(&path)?.to_string_lossy().as_bytes());
            digest.update(b"followed-target");
            update_metadata_stamp(&mut digest, &std::fs::metadata(&path)?);
        }
        digest.update([0xff]);
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

pub(crate) fn resolved_files_tree_stamp(
    root: &Path,
    files: &[impl AsRef<str>],
) -> WorkerResult<String> {
    Ok(sceneworks_core::download_receipt::resolved_files_tree_stamp(root, files)?)
}

fn artifact_content_fingerprint(root: &Path) -> WorkerResult<String> {
    let mut digest = Sha256::new();
    for path in artifact_files(root)? {
        let relative = path.strip_prefix(root).map_err(|_| {
            WorkerError::InvalidPayload(format!(
                "artifact entry {} escaped {}",
                path.display(),
                root.display()
            ))
        })?;
        let metadata = std::fs::symlink_metadata(&path)?;
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update([0]);
        if metadata.file_type().is_symlink() {
            digest.update(std::fs::read_link(&path)?.to_string_lossy().as_bytes());
            if std::fs::metadata(&path)?.is_file() {
                sceneworks_core::hashing::update_sha256(std::fs::File::open(&path)?, &mut digest)?;
            }
        } else if metadata.is_file() {
            sceneworks_core::hashing::update_sha256(std::fs::File::open(&path)?, &mut digest)?;
        }
        digest.update([0xff]);
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

/// Write worker-owned provenance while an app-managed artifact is being installed or converted.
/// The strong content digest is paid once on that cold path. Generation validates only the
/// metadata tree stamp, keeping the admission hot path O(file-count), not O(model-bytes).
pub(crate) fn write_app_managed_artifact_receipt(
    root: &Path,
    repository: &str,
    revision: &str,
    variant: &str,
    tier: &str,
) -> WorkerResult<ResolvedArtifactProvenance> {
    if [repository, revision, variant]
        .iter()
        .any(|value| value.trim().is_empty())
    {
        return Err(WorkerError::InvalidPayload(
            "app-managed artifact provenance requires nonempty repository, revision, and variant"
                .to_owned(),
        ));
    }
    let tier = supported_artifact_tier(tier).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("unsupported app-managed artifact tier {tier:?}"))
    })?;
    let _lock = lock_app_managed_receipt(root)?;
    let fingerprint = artifact_content_fingerprint(root)?;
    let tree_stamp = artifact_tree_stamp(root)?;
    let receipt = AppManagedArtifactReceipt {
        schema_version: ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        repository: repository.to_owned(),
        revision: revision.to_owned(),
        variant: variant.to_owned(),
        tier: tier.clone(),
        resolved_path_fingerprint: fingerprint.clone(),
        tree_stamp,
    };
    let marker = root.join(ARTIFACT_PROVENANCE_MARKER);
    sceneworks_core::fsutil::write_atomic(&marker, &serde_json::to_vec_pretty(&receipt)?)?;
    Ok(ResolvedArtifactProvenance {
        identity: ResolvedArtifactIdentity {
            repository: repository.to_owned(),
            revision: revision.to_owned(),
            variant: variant.to_owned(),
            fingerprint,
        },
        fixed_artifact_tier: Some(tier),
    })
}

/// Read only worker-owned app-managed provenance and prove that the actual file tree is still the
/// one stamped at install/convert time. Request and manifest payload provenance are never consulted.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn app_managed_artifact_provenance(
    root: &Path,
) -> WorkerResult<Option<ResolvedArtifactProvenance>> {
    let marker = root.join(ARTIFACT_PROVENANCE_MARKER);
    if !marker.is_file() {
        return Ok(None);
    }
    let _lock = lock_app_managed_receipt(root)?;
    let bytes = match std::fs::read(&marker) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut receipt: AppManagedArtifactReceipt =
        serde_json::from_slice(&bytes).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "invalid app-managed artifact provenance at {}: {error}",
                marker.display()
            ))
        })?;
    if receipt.schema_version != ARTIFACT_PROVENANCE_SCHEMA_VERSION {
        return Ok(None);
    }
    if [&receipt.repository, &receipt.revision, &receipt.variant]
        .iter()
        .any(|value| value.trim().is_empty())
        || !is_sha256_fingerprint(&receipt.resolved_path_fingerprint)
    {
        return Ok(None);
    }
    let Some(tier) = supported_artifact_tier(&receipt.tier) else {
        return Ok(None);
    };
    let stable_stamp = artifact_tree_stamp(root)?;
    if stable_stamp != receipt.tree_stamp
        && artifact_tree_stamp_format(root, true)? != receipt.tree_stamp
    {
        // Converted artifacts already carry an independent content digest. Metadata drift can
        // regain identity only when that original digest still matches, including all components.
        if artifact_content_fingerprint(root)? != receipt.resolved_path_fingerprint
            || artifact_tree_stamp(root)? != stable_stamp
        {
            return Ok(None);
        }
    }
    if stable_stamp != receipt.tree_stamp {
        receipt.tree_stamp = stable_stamp;
        sceneworks_core::download_receipt::write(&marker, &serde_json::to_value(&receipt)?)?;
    }
    Ok(Some(ResolvedArtifactProvenance {
        identity: ResolvedArtifactIdentity {
            repository: receipt.repository,
            revision: receipt.revision,
            variant: receipt.variant,
            fingerprint: receipt.resolved_path_fingerprint,
        },
        fixed_artifact_tier: Some(tier),
    }))
}

#[cfg(test)]
mod artifact_provenance_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn receipt_migration_rechecks_legacy_baseline_and_rejects_malformed_file_lists() {
        let data = tempfile::tempdir().unwrap();
        let hub = data.path().join("hub");
        let _env = crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().unwrap())]);
        let repo = "owner/model";
        let revision = "1111111111111111111111111111111111111111";
        let snapshot = hub.join("models--owner--model/snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        let weight = snapshot.join("model.safetensors");
        std::fs::write(&weight, b"original").unwrap();
        let files = ["model.safetensors"];
        let old_stamp = artifact_tree_stamp_format(&snapshot, true).unwrap();
        let receipt = json!({"repo":repo,"modelId":"fixture","variant":"q4",
            "snapshotRevision":revision,"resolvedFiles":files,"artifactTreeStamp":old_stamp});
        let marker = data.path().join("models/owner__model").join(INSTALL_MARKER);
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(&marker, serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert!(huggingface_receipt_weights(
            data.path(),
            repo,
            Some("fixture"),
            Some("q4"),
            ProvenanceRepair::Skip
        )
        .unwrap()
        .provenance
        .is_some());
        let stable = resolved_files_tree_stamp(&snapshot, &files).unwrap();
        establish_receipt_tree_stamp(
            &marker,
            &receipt,
            &snapshot,
            &files,
            "verified-legacy-metadata",
            Some(&stable),
        )
        .unwrap();
        let migrated: Value = serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
        assert_eq!(migrated["artifactTreeStamp"], stable);
        // Reproduce the interval between resolver verification and migration's fresh stat.
        std::fs::write(&marker, serde_json::to_vec(&receipt).unwrap()).unwrap();
        std::fs::write(&weight, b"modified").unwrap();
        let changed = resolved_files_tree_stamp(&snapshot, &files).unwrap();
        let before = std::fs::read(&marker).unwrap();
        assert!(establish_receipt_tree_stamp(
            &marker,
            &receipt,
            &snapshot,
            &files,
            "verified-legacy-metadata",
            Some(&changed)
        )
        .is_err());
        assert_eq!(std::fs::read(&marker).unwrap(), before);
        // A valid stamp for a subset cannot authorize a malformed recorded closure.
        let mut malformed = receipt;
        malformed["artifactTreeStamp"] = json!(changed);
        malformed["resolvedFiles"] = json!(["model.safetensors", null]);
        std::fs::write(&marker, serde_json::to_vec(&malformed).unwrap()).unwrap();
        assert!(huggingface_receipt_weights(
            data.path(),
            repo,
            Some("fixture"),
            Some("q4"),
            ProvenanceRepair::Skip
        )
        .is_none());
    }

    #[test]
    fn app_managed_receipt_migrates_stamps_only_when_original_content_survives() {
        let data = tempfile::tempdir().unwrap();
        let weight = data.path().join("model.safetensors");
        std::fs::write(&weight, b"original").unwrap();
        let identity =
            write_app_managed_artifact_receipt(data.path(), "owner/model", "revision", "q4", "q4")
                .unwrap();
        let marker = data.path().join(ARTIFACT_PROVENANCE_MARKER);
        let mut receipt: Value = serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
        receipt["treeStamp"] = json!(artifact_tree_stamp_format(data.path(), true).unwrap());
        std::fs::write(&marker, serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert_eq!(
            app_managed_artifact_provenance(data.path()).unwrap(),
            Some(identity.clone())
        );
        // Simulate an old metadata baseline that no longer matches after a remount.
        receipt["treeStamp"] = json!(format!("sha256:{}", "0".repeat(64)));
        std::fs::write(&marker, serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert_eq!(
            app_managed_artifact_provenance(data.path()).unwrap(),
            Some(identity)
        );
        let migrated = std::fs::read(&marker).unwrap();
        let parsed: Value = serde_json::from_slice(&migrated).unwrap();
        assert_eq!(
            parsed["treeStamp"],
            artifact_tree_stamp(data.path()).unwrap()
        );
        app_managed_artifact_provenance(data.path()).unwrap();
        assert_eq!(std::fs::read(&marker).unwrap(), migrated);
        std::fs::write(weight, b"modified").unwrap();
        assert!(app_managed_artifact_provenance(data.path())
            .unwrap()
            .is_none());
        assert_eq!(std::fs::read(&marker).unwrap(), migrated);
    }

    #[test]
    fn tree_stamp_repair_fills_null_revision_and_refuses_a_replaced_receipt() {
        let temp = tempfile::tempdir().unwrap();
        let rev = "1111111111111111111111111111111111111111";
        let snapshot = temp.path().join(rev);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"weights").unwrap();
        let marker = temp.path().join(".sceneworks-download-complete.json");
        let receipt = json!({"repo":"owner/model","variant":"q4","snapshotRevision":null,"resolvedFiles":["model.safetensors"]});
        let mut top = receipt.clone();
        top["receipts"] = json!([receipt]);
        std::fs::write(&marker, serde_json::to_vec(&top).unwrap()).unwrap();
        establish_receipt_tree_stamp(
            &marker,
            &receipt,
            &snapshot,
            &["model.safetensors"],
            "repair",
            None,
        )
        .unwrap();
        let after: Value = serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
        assert_eq!(after["snapshotRevision"], rev);
        assert_eq!(after["receipts"][0]["snapshotRevision"], rev);
        let before = std::fs::read(&marker).unwrap();
        assert!(establish_receipt_tree_stamp(
            &marker,
            &receipt,
            &snapshot,
            &["model.safetensors"],
            "repair",
            None
        )
        .is_err());
        assert_eq!(std::fs::read(&marker).unwrap(), before);
    }

    #[test]
    fn app_managed_receipt_detects_same_path_mutation_and_rejects_bad_identity() {
        let data = tempfile::tempdir().expect("artifact dir");
        let weights = data.path().join("weights.safetensors");
        std::fs::write(&weights, b"before").expect("weights");
        let written = write_app_managed_artifact_receipt(
            data.path(),
            "SceneWorks/local",
            "immutable-revision",
            "converted-q4",
            "q4",
        )
        .expect("receipt");
        assert_eq!(
            app_managed_artifact_provenance(data.path()).expect("read"),
            Some(written)
        );

        std::fs::write(&weights, b"after-with-different-size").expect("mutate same path");
        assert_eq!(
            app_managed_artifact_provenance(data.path()).expect("read after mutation"),
            None,
            "metadata tree-stamp drift must invalidate same-path artifact replacement"
        );

        let no_revision = tempfile::tempdir().expect("invalid artifact dir");
        std::fs::write(no_revision.path().join("weights.safetensors"), b"weights")
            .expect("weights");
        assert!(write_app_managed_artifact_receipt(
            no_revision.path(),
            "SceneWorks/local",
            "",
            "converted-q4",
            "q4",
        )
        .expect_err("an unresolved source revision cannot publish provenance")
        .to_string()
        .contains("nonempty"));
        assert!(
            !no_revision.path().join(ARTIFACT_PROVENANCE_MARKER).exists(),
            "failed receipt creation must not publish a marker"
        );
    }

    #[test]
    fn app_managed_receipt_reader_rejects_malformed_strong_fingerprint() {
        let data = tempfile::tempdir().expect("artifact dir");
        std::fs::write(data.path().join("weights.safetensors"), b"weights").expect("weights");
        write_app_managed_artifact_receipt(
            data.path(),
            "SceneWorks/local",
            "revision",
            "converted-q8",
            "q8",
        )
        .expect("receipt");
        let marker = data.path().join(ARTIFACT_PROVENANCE_MARKER);
        let mut receipt: Value =
            serde_json::from_slice(&std::fs::read(&marker).expect("marker")).expect("json");
        receipt["resolvedPathFingerprint"] = json!("sha256:NOT-A-DIGEST");
        std::fs::write(&marker, serde_json::to_vec(&receipt).expect("json")).expect("marker");
        assert_eq!(
            app_managed_artifact_provenance(data.path()).expect("read"),
            None
        );
    }

    #[cfg(unix)]
    #[test]
    fn app_managed_symlink_receipt_hashes_and_stamps_the_followed_target() {
        use std::os::unix::fs::symlink;

        let data = tempfile::tempdir().expect("data dir");
        let artifact = data.path().join("artifact");
        let blob = data.path().join("blob.safetensors");
        std::fs::create_dir_all(&artifact).expect("artifact");
        std::fs::write(&blob, b"original-blob").expect("blob");
        symlink(&blob, artifact.join("weights.safetensors")).expect("symlink");
        let written = write_app_managed_artifact_receipt(
            &artifact,
            "SceneWorks/local",
            "revision",
            "converted-q4",
            "q4",
        )
        .expect("receipt");
        assert_eq!(
            app_managed_artifact_provenance(&artifact).expect("read"),
            Some(written)
        );
        std::fs::write(&blob, b"mutated-underlying-blob").expect("mutate target");
        assert_eq!(
            app_managed_artifact_provenance(&artifact).expect("read after mutation"),
            None,
            "unchanged link metadata/path must not hide followed-target mutation"
        );
    }

    #[cfg(unix)]
    #[test]
    fn app_managed_receipt_walks_symlinked_directories_and_terminates_cycles() {
        use std::os::unix::fs::symlink;

        let data = tempfile::tempdir().expect("data dir");
        let artifact = data.path().join("artifact");
        let shared_component = data.path().join("shared-component");
        std::fs::create_dir_all(&artifact).expect("artifact");
        std::fs::create_dir_all(shared_component.join("nested")).expect("component");
        let weights = shared_component.join("nested/weights.safetensors");
        std::fs::write(&weights, b"original-component").expect("weights");
        symlink(&shared_component, artifact.join("text_encoder")).expect("component symlink");
        symlink(
            &shared_component,
            shared_component.join("cycle-to-component"),
        )
        .expect("cycle symlink");

        let written = write_app_managed_artifact_receipt(
            &artifact,
            "SceneWorks/local",
            "revision",
            "converted-q8",
            "q8",
        )
        .expect("cycle-safe receipt");
        assert_eq!(
            app_managed_artifact_provenance(&artifact).expect("read"),
            Some(written),
            "a shipped-style symlinked component directory must be provenance-capable"
        );

        std::fs::write(&weights, b"mutated-component-beneath-the-same-link")
            .expect("mutate component");
        assert_eq!(
            app_managed_artifact_provenance(&artifact).expect("read after mutation"),
            None,
            "mutation beneath an unchanged directory symlink must invalidate provenance"
        );
    }

    fn write_hf_receipt(
        data_dir: &Path,
        repo: &str,
        model_id: &str,
        revision: &str,
        variant: &str,
        resolved_tier: Option<&str>,
        files: &[&str],
    ) {
        let marker = data_dir.join("models").join(safe_download_dir(model_id));
        std::fs::create_dir_all(&marker).expect("marker dir");
        let snapshot = huggingface_repo_cache_path(data_dir, repo)
            .expect("cache")
            .join("snapshots")
            .join(revision);
        let artifact_tree_stamp =
            resolved_files_tree_stamp(&snapshot, files).expect("artifact stamp");
        std::fs::write(
            marker.join(INSTALL_MARKER),
            serde_json::to_vec(&json!({
                "schemaVersion": 2,
                "repo": repo,
                "modelId": model_id,
                "variant": variant,
                "resolvedTier": resolved_tier,
                "snapshotRevision": revision,
                "resolvedFiles": files,
                "artifactTreeStamp": artifact_tree_stamp,
            }))
            .expect("receipt json"),
        )
        .expect("receipt");
    }

    /// The API convert pre-flight must agree with what `resolve_convert_plan` will find: a missing
    /// snapshot, a snapshot missing THIS variant's file (the shared-repo case — the three Anima
    /// variants live in one repo and download one at a time), and the ready case are distinguished so
    /// the request boundary can explain itself instead of leaving the worker to fail the job.
    #[test]
    fn convert_source_state_separates_uncached_repo_from_a_variant_still_downloading() {
        let data = tempfile::tempdir().expect("data dir");
        let hub = data.path().join("hub");
        let _env =
            crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().expect("hub path"))]);

        let repo = "owner/shared";
        let alpha = "split_files/diffusion_models/alpha.safetensors";
        let beta = "split_files/diffusion_models/beta.safetensors";
        assert_eq!(
            convert_source_state(data.path(), repo, alpha),
            ConvertSourceState::RepoNotCached
        );

        let revision = "0123456789abcdef0123456789abcdef01234567";
        let snapshot = huggingface_repo_cache_path(data.path(), repo)
            .expect("cache")
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(snapshot.join("split_files/diffusion_models")).expect("dit dir");
        std::fs::write(snapshot.join(alpha), b"weights").expect("alpha weights");
        assert_eq!(
            convert_source_state(data.path(), repo, alpha),
            ConvertSourceState::Ready
        );
        assert_eq!(
            convert_source_state(data.path(), repo, beta),
            ConvertSourceState::FileMissing,
            "a sibling variant sharing the repo is NOT ready just because the snapshot exists"
        );
        // A directory is not a convertible source file, and a traversal-shaped path is refused rather
        // than probed outside the snapshot.
        assert_eq!(
            convert_source_state(data.path(), repo, "split_files/diffusion_models"),
            ConvertSourceState::FileMissing
        );
        assert_eq!(
            convert_source_state(data.path(), repo, "../../../etc/passwd"),
            ConvertSourceState::FileMissing
        );
    }

    #[test]
    fn hf_receipt_handoff_covers_tiered_and_flat_default_snapshots() {
        let data = tempfile::tempdir().expect("data dir");
        let hub = data.path().join("hub");
        let _env =
            crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().expect("hub path"))]);

        let tiered_repo = "owner/tiered";
        let tiered = huggingface_repo_cache_path(data.path(), tiered_repo)
            .expect("cache")
            .join("snapshots/rev-tiered/q4");
        std::fs::create_dir_all(&tiered).expect("tier dir");
        std::fs::write(tiered.join("weights.safetensors"), b"q4").expect("weights");
        write_hf_receipt(
            data.path(),
            tiered_repo,
            "tiered-model",
            "rev-tiered",
            "q4",
            Some("q4"),
            &["q4/weights.safetensors"],
        );
        let tiered_resolved = huggingface_receipt_weights(
            data.path(),
            tiered_repo,
            Some("tiered-model"),
            Some("q4"),
            ProvenanceRepair::Skip,
        )
        .expect("tiered receipt");
        assert_eq!(tiered_resolved.path, tiered);
        let tiered_provenance = tiered_resolved.provenance.expect("tiered provenance");
        assert_eq!(tiered_provenance.fixed_artifact_tier.as_deref(), Some("q4"));
        assert_eq!(tiered_provenance.identity.variant, "q4");
        std::fs::write(
            tiered.join("weights.safetensors"),
            b"q4-mutated-at-same-path",
        )
        .expect("mutate tiered weights");
        assert_eq!(
            huggingface_receipt_weights(
                data.path(),
                tiered_repo,
                Some("tiered-model"),
                Some("q4"),
                ProvenanceRepair::Skip
            )
            .expect("path remains usable")
            .provenance,
            None,
            "same-snapshot content-state drift must invalidate HF calibration provenance"
        );

        let flat_repo = "owner/flat";
        let flat = huggingface_repo_cache_path(data.path(), flat_repo)
            .expect("cache")
            .join("snapshots/rev-flat");
        std::fs::create_dir_all(&flat).expect("flat snapshot");
        std::fs::write(flat.join("weights.safetensors"), b"flat").expect("weights");
        write_hf_receipt(
            data.path(),
            flat_repo,
            "flat-model",
            "rev-flat",
            "default",
            Some("q8"),
            &["weights.safetensors"],
        );
        let flat_resolved = huggingface_receipt_weights(
            data.path(),
            flat_repo,
            Some("flat-model"),
            Some("default"),
            ProvenanceRepair::Skip,
        )
        .expect("flat receipt");
        assert_eq!(flat_resolved.path, flat);
        let flat_provenance = flat_resolved.provenance.expect("flat provenance");
        assert_eq!(flat_provenance.fixed_artifact_tier.as_deref(), Some("q8"));
        assert_eq!(flat_provenance.identity.variant, "default");
        assert_eq!(flat_provenance.identity.revision, "rev-flat");

        let marker = data
            .path()
            .join("models")
            .join(safe_download_dir("flat-model"))
            .join(INSTALL_MARKER);
        let mut unknown_tier: Value =
            serde_json::from_slice(&std::fs::read(&marker).expect("marker")).expect("json");
        unknown_tier
            .as_object_mut()
            .expect("receipt object")
            .remove("resolvedTier");
        std::fs::write(&marker, serde_json::to_vec(&unknown_tier).expect("json")).expect("marker");
        assert_eq!(
            huggingface_receipt_weights(
                data.path(),
                flat_repo,
                Some("flat-model"),
                Some("default"),
                ProvenanceRepair::Skip
            )
            .expect("flat path remains usable")
            .provenance
            .and_then(|provenance| provenance.fixed_artifact_tier),
            None,
            "an old flat/default receipt leaves artifact tier unconstrained"
        );

        let mut mismatched = unknown_tier;
        mismatched["snapshotRevision"] = json!("different-revision");
        std::fs::write(&marker, serde_json::to_vec(&mismatched).expect("json")).expect("marker");
        assert!(
            huggingface_receipt_weights(
                data.path(),
                flat_repo,
                Some("flat-model"),
                Some("default"),
                ProvenanceRepair::Skip
            )
            .is_none(),
            "receipt/snapshot revision mismatch must not manufacture provenance"
        );
    }

    #[test]
    fn exact_revision_handoff_rejects_crossed_or_unproven_receipts() {
        let data = tempfile::tempdir().expect("data dir");
        let hub = data.path().join("hub");
        let _env =
            crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().expect("hub path"))]);
        let repo = "starvector/exact";
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let snapshot = huggingface_repo_cache_path(data.path(), repo)
            .expect("cache")
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot");
        std::fs::write(snapshot.join("model.safetensors"), b"exact weights").expect("weights");
        write_hf_receipt(
            data.path(),
            repo,
            "starvector-exact",
            revision,
            "default",
            None,
            &["model.safetensors"],
        );

        assert_eq!(
            huggingface_receipt_weights_dir_at_revision(
                data.path(),
                repo,
                revision,
                Some("starvector-exact"),
                None,
            ),
            Some(snapshot.clone())
        );
        assert_eq!(
            huggingface_receipt_weights_dir_at_revision(
                data.path(),
                repo,
                "different-revision",
                Some("starvector-exact"),
                None,
            ),
            None,
            "a receipt for another immutable revision must never cross the native load boundary"
        );

        std::fs::write(snapshot.join("model.safetensors"), b"mutated weights")
            .expect("mutate weights");
        assert_eq!(
            huggingface_receipt_weights_dir_at_revision(
                data.path(),
                repo,
                revision,
                Some("starvector-exact"),
                None,
            ),
            None,
            "an exact path without valid receipt provenance is not loadable"
        );
    }

    #[test]
    fn exact_revision_handoff_repairs_an_unstamped_mirrored_v2_receipt_once() {
        let data = tempfile::tempdir().expect("data dir");
        let hub = data.path().join("hub");
        let _env =
            crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().expect("hub path"))]);
        let repo = "starvector/terminal-copy";
        let model_id = "starvector_1b";
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let snapshot = huggingface_repo_cache_path(data.path(), repo)
            .expect("cache")
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot");
        std::fs::write(snapshot.join("model.safetensors"), b"exact weights").expect("weights");
        let managed = data.path().join("models").join(safe_download_dir(model_id));
        std::fs::create_dir_all(&managed).expect("managed receipt dir");
        let entry = json!({
            "schemaVersion": 2,
            "repo": repo,
            "modelId": model_id,
            "variant": "default",
            "snapshotRevision": revision,
            "resolvedFiles": ["model.safetensors"],
            "artifactTreeStamp": null,
        });
        let mut envelope = entry.clone();
        envelope["receipts"] = json!([entry]);
        let marker = managed.join(INSTALL_MARKER);
        std::fs::write(
            &marker,
            serde_json::to_vec(&envelope).expect("receipt json"),
        )
        .expect("receipt");

        assert_eq!(
            huggingface_receipt_weights_dir_at_revision(
                data.path(),
                repo,
                revision,
                Some(model_id),
                None,
            ),
            Some(snapshot.clone()),
            "the strict native handoff must establish proof for an exact unstamped v2 receipt"
        );
        let repaired: Value =
            serde_json::from_slice(&std::fs::read(&marker).expect("repaired receipt"))
                .expect("receipt json");
        let stamp = repaired["artifactTreeStamp"]
            .as_str()
            .expect("top-level stamp");
        assert!(is_sha256_fingerprint(stamp));
        assert_eq!(repaired["artifactTreeStampSource"], "repair");
        assert_eq!(repaired["receipts"][0]["artifactTreeStamp"], stamp);
        assert_eq!(repaired["receipts"][0]["artifactTreeStampSource"], "repair");

        std::fs::write(snapshot.join("model.safetensors"), b"mutated after repair")
            .expect("mutate weights");
        let before_rejected_read = std::fs::read(&marker).expect("receipt before rejected read");
        assert_eq!(
            huggingface_receipt_weights_dir_at_revision(
                data.path(),
                repo,
                revision,
                Some(model_id),
                None,
            ),
            None,
            "an existing mismatched stamp must remain fail-closed"
        );
        assert_eq!(
            std::fs::read(&marker).expect("receipt after rejected read"),
            before_rejected_read,
            "the native handoff must never repair over recorded integrity drift"
        );
    }

    #[test]
    fn exact_revision_handoff_never_repairs_an_inexact_receipt_identity() {
        let data = tempfile::tempdir().expect("data dir");
        let hub = data.path().join("hub");
        let _env =
            crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().expect("hub path"))]);
        let repo = "starvector/terminal-copy";
        let model_id = "starvector_1b";
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let snapshot = huggingface_repo_cache_path(data.path(), repo)
            .expect("cache")
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot");
        std::fs::write(snapshot.join("model.safetensors"), b"exact weights").expect("weights");
        let managed = data.path().join("models").join(safe_download_dir(model_id));
        std::fs::create_dir_all(&managed).expect("managed receipt dir");
        let marker = managed.join(INSTALL_MARKER);
        let valid = json!({
            "schemaVersion": 2,
            "repo": repo,
            "modelId": model_id,
            "variant": "default",
            "snapshotRevision": revision,
            "resolvedFiles": ["model.safetensors"],
            "artifactTreeStamp": null,
        });
        let mutations = [
            (
                "wrong revision",
                "snapshotRevision",
                json!("different-revision"),
            ),
            ("missing model", "modelId", Value::Null),
            ("wrong schema", "schemaVersion", json!(1)),
            ("wrong variant", "variant", json!("q4")),
        ];
        for (label, key, value) in mutations {
            let mut entry = valid.clone();
            if value.is_null() {
                entry.as_object_mut().expect("entry").remove(key);
            } else {
                entry[key] = value;
            }
            let mut envelope = entry.clone();
            envelope["receipts"] = json!([entry]);
            let before = serde_json::to_vec(&envelope).expect("receipt json");
            std::fs::write(&marker, &before).expect("receipt");
            assert_eq!(
                huggingface_receipt_weights_dir_at_revision(
                    data.path(),
                    repo,
                    revision,
                    Some(model_id),
                    None,
                ),
                None,
                "{label} must be rejected before repair"
            );
            assert_eq!(
                std::fs::read(&marker).expect("receipt after rejected read"),
                before,
                "{label} must leave marker bytes unchanged"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn hf_tree_stamp_detects_in_place_blob_mutation_beneath_unchanged_snapshot_symlink() {
        use std::os::unix::fs::symlink;

        let data = tempfile::tempdir().expect("data dir");
        let hub = data.path().join("hub");
        let _env =
            crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().expect("hub path"))]);
        let repo = "owner/symlinked";
        let cache = huggingface_repo_cache_path(data.path(), repo).expect("cache");
        let blob = cache.join("blobs/blob-id");
        let tier = cache.join("snapshots/revision/q4");
        std::fs::create_dir_all(blob.parent().expect("blob parent")).expect("blobs");
        std::fs::create_dir_all(&tier).expect("tier");
        std::fs::write(&blob, b"original-blob").expect("blob");
        symlink("../../../blobs/blob-id", tier.join("weights.safetensors")).expect("snapshot link");
        write_hf_receipt(
            data.path(),
            repo,
            "symlinked-model",
            "revision",
            "q4",
            Some("q4"),
            &["q4/weights.safetensors"],
        );
        assert!(huggingface_receipt_weights(
            data.path(),
            repo,
            Some("symlinked-model"),
            Some("q4"),
            ProvenanceRepair::Skip
        )
        .expect("receipt path")
        .provenance
        .is_some());
        std::fs::write(&blob, b"mutated-underlying-blob").expect("mutate blob in place");
        assert_eq!(
            huggingface_receipt_weights(
                data.path(),
                repo,
                Some("symlinked-model"),
                Some("q4"),
                ProvenanceRepair::Skip
            )
            .expect("path remains usable")
            .provenance,
            None
        );
    }
}

#[cfg(any(test, native_lane))]
pub(crate) fn huggingface_receipt_weights_dir(
    data_dir: &Path,
    repo: &str,
    model_id: Option<&str>,
    variant: Option<&str>,
) -> Option<PathBuf> {
    // Path-only callers never read provenance, so they must not trigger a receipt write as a side
    // effect of locating weights. Repair belongs to the provenance consumer.
    let resolved =
        huggingface_receipt_weights(data_dir, repo, model_id, variant, ProvenanceRepair::Skip)?;
    // sc-19707: the receipt resolves and PROVES its install against the authoritative source
    // (its tree stamp is a source-tree fact and must stay one); only the path handed to the loader
    // is then served from a leased app-owned bundle covering that exact snapshot. Inert without an
    // active lease.
    let library = sceneworks_core::hf_home::model_source_library(data_dir);
    Some(
        sceneworks_core::model_artifacts::local_preference::redirect_source_library_path(
            library.root(),
            &resolved.path,
        )
        .unwrap_or(resolved.path),
    )
}

/// Resolve a receipt-backed snapshot only when its worker-owned provenance proves the exact
/// immutable revision the caller requested. This is stricter than
/// [`huggingface_receipt_weights_dir`], whose legacy compatibility path may return an unproven
/// receipt: native providers whose architecture contract is revision-pinned must never load that
/// compatibility fallback or infer identity from a directory name.
pub(crate) fn huggingface_receipt_weights_dir_at_revision(
    data_dir: &Path,
    repo: &str,
    revision: &str,
    model_id: Option<&str>,
    variant: Option<&str>,
) -> Option<PathBuf> {
    // An exact-revision native provider is itself a provenance consumer. A source-produced v2
    // receipt may predate download-time tree stamping (or have crossed a verified offline-library
    // materialization boundary), so establish the canonical locked/atomic baseline when the stamp
    // is absent. Before any repair write, require the v2 receipt to name the complete identity the
    // native provider requested. Existing stamp drift is never repaired.
    let resolved = huggingface_receipt_weights_with_revision(
        data_dir,
        repo,
        model_id,
        variant,
        Some(revision),
        ProvenanceRepair::Allow,
    )?;
    let identity = &resolved.provenance.as_ref()?.identity;
    if identity.repository != repo || identity.revision != revision {
        return None;
    }
    let library = sceneworks_core::hf_home::model_source_library(data_dir);
    Some(
        sceneworks_core::model_artifacts::local_preference::redirect_source_library_path(
            library.root(),
            &resolved.path,
        )
        .unwrap_or(resolved.path),
    )
}

/// Exact PUP package handoff: every file the native package can consume must belong to the
/// verified download inventory, including an optional shipped system-prompt override. A compatible
/// Qwen directory or an extra unreceipted sidecar is not an installation receipt.
#[cfg(any(test, native_lane))]
pub(crate) fn flux3_prompt_upsampler_snapshot(data_dir: &Path) -> Option<PathBuf> {
    use gen_core_flux3::pup::{
        SYSTEM_PROMPTS_FILE, UPSAMPLER_REPO, UPSAMPLER_REQUIRED_FILES, UPSAMPLER_REVISION,
    };
    let resolved = huggingface_receipt_weights_with_revision(
        data_dir,
        UPSAMPLER_REPO,
        Some("flux3_image_klein_pup"),
        None,
        Some(UPSAMPLER_REVISION),
        ProvenanceRepair::Allow,
    )?;
    let identity = &resolved.provenance.as_ref()?.identity;
    if identity.repository != UPSAMPLER_REPO || identity.revision != UPSAMPLER_REVISION {
        return None;
    }
    let root = &resolved.snapshot;
    let repo = huggingface_repo_cache_path(data_dir, UPSAMPLER_REPO)?;
    let canonical_repo = std::fs::canonicalize(&repo).ok()?;
    if !canonical_repo.starts_with(std::fs::canonicalize(huggingface_hub_cache_dir(data_dir)).ok()?)
        || !std::fs::canonicalize(root)
            .ok()?
            .starts_with(canonical_repo.join("snapshots"))
    {
        return None;
    }
    let files = resolved.receipt.get("resolvedFiles")?.as_array()?;
    for file in files {
        let file = file.as_str()?;
        let path = safe_join(root, file).ok()?;
        crate::flux3_snapshot_member(root, &path, false).ok()?;
    }
    for name in UPSAMPLER_REQUIRED_FILES.into_iter().chain(
        root.join(SYSTEM_PROMPTS_FILE)
            .exists()
            .then_some(SYSTEM_PROMPTS_FILE),
    ) {
        if !files.iter().any(|file| file.as_str() == Some(name)) {
            return None;
        }
        crate::flux3_snapshot_member(root, &root.join(name), false).ok()?;
    }
    // Both native LLM loaders discover *all* root safetensors, and optional sidecars such as
    // generation_config.json change inference behavior. Bind the complete discovered tree, not
    // an allowlist of required names. Download receipts and HF metadata live outside snapshots;
    // no hidden file or other snapshot member is exempt from this inventory comparison.
    fn consume_inventory(
        root: &Path,
        dir: &Path,
        inventory: &mut std::collections::HashSet<String>,
        visited: &mut std::collections::HashSet<PathBuf>,
    ) -> Option<()> {
        crate::flux3_snapshot_member(root, dir, true).ok()?;
        if !visited.insert(std::fs::canonicalize(dir).ok()?) {
            return None;
        }
        for entry in std::fs::read_dir(dir).ok()? {
            let path = entry.ok()?.path();
            if path.is_dir() {
                consume_inventory(root, &path, inventory, visited)?;
            } else {
                crate::flux3_snapshot_member(root, &path, false).ok()?;
                let relative = path.strip_prefix(root).ok()?.to_str()?.replace('\\', "/");
                if !path.is_file() || !inventory.remove(&relative) {
                    return None;
                }
            }
        }
        Some(())
    }
    let mut inventory = files
        .iter()
        .map(|file| file.as_str().map(str::to_owned))
        .collect::<Option<std::collections::HashSet<_>>>()?;
    if inventory.len() != files.len() {
        return None;
    }
    consume_inventory(
        root,
        root,
        &mut inventory,
        &mut std::collections::HashSet::new(),
    )?;
    inventory.is_empty().then(|| root.clone())
}

#[cfg(any(test, native_lane))]
pub(crate) fn huggingface_receipt_weights(
    data_dir: &Path,
    repo: &str,
    model_id: Option<&str>,
    variant: Option<&str>,
    repair: ProvenanceRepair,
) -> Option<ResolvedWeights> {
    huggingface_receipt_weights_with_revision(data_dir, repo, model_id, variant, None, repair)
}

fn huggingface_receipt_weights_with_revision(
    data_dir: &Path,
    repo: &str,
    model_id: Option<&str>,
    variant: Option<&str>,
    required_revision: Option<&str>,
    repair: ProvenanceRepair,
) -> Option<ResolvedWeights> {
    let models_dir = data_dir.join("models");
    let repo_marker = models_dir
        .join(safe_download_dir(repo))
        .join(INSTALL_MARKER);
    let mut targeted = vec![repo_marker];
    if let Some(model_id) = model_id {
        targeted.push(
            models_dir
                .join(safe_download_dir(model_id))
                .join(INSTALL_MARKER),
        );
    }
    targeted.dedup();

    // The repo/model marker is the overwhelmingly common path. Resolve it before walking every
    // installed model directory; a targeted hit makes this lookup O(1) in the catalog size.
    for marker in &targeted {
        if let Some(weights_dir) = receipt_weights_dir_from_marker(
            data_dir,
            marker,
            repo,
            model_id,
            variant,
            required_revision,
            repair,
        ) {
            return Some(weights_dir);
        }
    }

    let entries = std::fs::read_dir(&models_dir).ok()?;
    for marker in entries
        .flatten()
        .map(|entry| entry.path().join(INSTALL_MARKER))
        .filter(|marker| !targeted.contains(marker))
    {
        if let Some(weights_dir) = receipt_weights_dir_from_marker(
            data_dir,
            &marker,
            repo,
            model_id,
            variant,
            required_revision,
            repair,
        ) {
            return Some(weights_dir);
        }
    }
    None
}

/// Whether a resolver call may establish a tree-stamp baseline for an install that never got one.
///
/// A receipt written before download-time stamping — every `backfilled` receipt, plus any download
/// whose payload lacked `memoryCalibrationProvenanceRequired` — resolves a loadable artifact and
/// yields NO provenance, and nothing else ever recomputes the stamp. That permanently pins the
/// install to the legacy admission selector, which is not a decision anyone made (sc-16482).
///
/// `Allow` stamps the artifact AS IT STANDS and persists that baseline, so mutation is detected from
/// then on. It cannot prove the artifact was untouched between download and repair — no
/// after-the-fact check can — but it is the same guarantee shape the download path itself gives
/// (it stamps what the hub client just wrote, without re-reading content), just anchored later. The
/// receipt records `artifactTreeStampSource` so the weaker anchor is never mistaken for the original.
///
/// A present but mismatched stamp stays unproven here. It may reflect file drift or a changed
/// mount number in an old stamp; only independent content verification may replace that baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProvenanceRepair {
    Allow,
    #[cfg(any(test, native_lane))]
    Skip,
}

/// Stamp an unstamped receipt in place and return the baseline that was established.
fn establish_receipt_tree_stamp(
    marker: &Path,
    receipt: &Value,
    snapshot: &Path,
    files: &[&str],
    source: &str,
    verified_stamp: Option<&str>,
) -> WorkerResult<String> {
    let _receipt_lock = sceneworks_core::download_receipt::lock(
        marker
            .parent()
            .ok_or_else(|| WorkerError::InvalidPayload("receipt has no parent".to_owned()))?,
    )?;
    let stamp = resolved_files_tree_stamp(snapshot, files)?;
    if verified_stamp.is_some_and(|verified| verified != stamp) {
        return Err(WorkerError::InvalidPayload(
            "artifact changed during content verification".to_owned(),
        ));
    }
    if source == "verified-legacy-metadata" {
        let matches = receipt
            .get("artifactTreeStamp")
            .and_then(Value::as_str)
            .is_some_and(|expected| {
                sceneworks_core::download_receipt::resolved_files_tree_stamp_matches(
                    snapshot, files, expected,
                )
                .unwrap_or(false)
            });
        if !matches || resolved_files_tree_stamp(snapshot, files)? != stamp {
            return Err(WorkerError::InvalidPayload(
                "artifact changed since its legacy baseline was verified".to_owned(),
            ));
        }
    }
    let revision = snapshot
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("unreadable snapshot name {}", snapshot.display()))
        })?;
    let identity = |candidate: &Value| {
        candidate.get("repo") == receipt.get("repo")
            && candidate.get("modelId") == receipt.get("modelId")
            && candidate.get("variant") == receipt.get("variant")
    };
    let patch = |entry: &mut Value| {
        let Some(object) = entry.as_object_mut() else {
            return;
        };
        object.insert("artifactTreeStamp".to_owned(), Value::String(stamp.clone()));
        object.insert(
            "artifactTreeStampSource".to_owned(),
            Value::String(source.to_owned()),
        );
        // A backfilled receipt carries no revision, so resolution leans on the exact file set
        // identifying exactly one snapshot. Record what we just resolved to remove that ambiguity.
        if object.get("snapshotRevision").is_none_or(|value| {
            value.is_null()
                || value
                    .as_str()
                    .is_some_and(|revision| revision.trim().is_empty())
        }) {
            object.insert(
                "snapshotRevision".to_owned(),
                Value::String(revision.to_owned()),
            );
        }
    };

    let mut top = serde_json::from_slice::<Value>(&std::fs::read(marker)?).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "invalid install receipt at {}: {error}",
            marker.display()
        ))
    })?;
    // The caller read before acquiring the write lock. A concurrent download or discovery
    // repair may have replaced that identity; never stamp the replacement with the old tree.
    let current = top
        .get("receipts")
        .and_then(Value::as_array)
        .map(|entries| entries.iter().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![&top]);
    if !current.iter().any(|entry| identity(entry))
        || current.iter().filter(|entry| identity(entry)).any(|entry| {
            [
                "schemaVersion",
                "resolvedFiles",
                "snapshotRevision",
                "artifactTreeStamp",
            ]
            .iter()
            .any(|key| entry.get(*key) != receipt.get(*key))
        })
    {
        return Err(WorkerError::InvalidPayload(
            "install receipt changed during provenance repair".to_owned(),
        ));
    }
    // `write_model_download_receipt` mirrors the newest receipt at the top level AND in `receipts`.
    // Patch both when they name the same artifact, so neither copy contradicts the other.
    if identity(&top) {
        patch(&mut top);
    }
    if let Some(entries) = top.get_mut("receipts").and_then(Value::as_array_mut) {
        for entry in entries.iter_mut().filter(|entry| identity(entry)) {
            patch(entry);
        }
    }
    sceneworks_core::download_receipt::write(marker, &top)?;
    Ok(stamp)
}

fn receipt_weights_dir_from_marker(
    data_dir: &Path,
    marker: &Path,
    repo: &str,
    model_id: Option<&str>,
    variant: Option<&str>,
    required_revision: Option<&str>,
    repair: ProvenanceRepair,
) -> Option<ResolvedWeights> {
    #[cfg(test)]
    RECEIPT_MARKERS_READ.with(|count| count.set(count.get() + 1));
    let bytes = std::fs::read(marker).ok()?;
    let top = serde_json::from_slice::<Value>(&bytes).ok()?;
    let receipts = top
        .get("receipts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![top]);
    for receipt in receipts {
        if receipt.get("repo").and_then(Value::as_str) != Some(repo) {
            continue;
        }
        // Exact native handoffs may establish a missing tree stamp, but only after the receipt
        // already proves the complete requested identity. This keeps an A lookup from stamping an
        // unstamped B receipt and excludes permissive legacy receipts from the repair path.
        if let Some(revision) = required_revision {
            if receipt.get("schemaVersion").and_then(Value::as_u64) != Some(2)
                || receipt.get("modelId").and_then(Value::as_str) != model_id
                || receipt.get("variant").and_then(Value::as_str)
                    != Some(variant.unwrap_or("default"))
                || receipt.get("snapshotRevision").and_then(Value::as_str) != Some(revision)
            {
                continue;
            }
        }
        if let Some(model_id) = model_id {
            if receipt
                .get("modelId")
                .and_then(Value::as_str)
                .is_some_and(|id| id != model_id)
            {
                continue;
            }
        }
        if let Some(variant) = variant {
            if receipt.get("variant").and_then(Value::as_str) != Some(variant) {
                continue;
            }
        }
        let Some(files) = receipt
            .get("resolvedFiles")
            .and_then(Value::as_array)
            .and_then(|files| files.iter().map(Value::as_str).collect::<Option<Vec<_>>>())
        else {
            continue;
        };
        if files.is_empty() {
            continue;
        }
        let repo_dir = huggingface_repo_cache_path(data_dir, repo)?;
        let snapshots = repo_dir.join("snapshots");
        let snapshot_entries = std::fs::read_dir(&snapshots).ok()?;
        let recorded_revision = crate::payload::trimmed_string(receipt.get("snapshotRevision"));
        let matching = snapshot_entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|snapshot| {
                snapshot.is_dir() && files.iter().all(|file| snapshot.join(file).is_file())
            })
            .filter(|snapshot| {
                recorded_revision.is_none_or(|revision| {
                    snapshot.file_name().and_then(|name| name.to_str()) == Some(revision)
                })
            })
            .collect::<Vec<_>>();
        // Schema-v2 receipts written before snapshotRevision are accepted only when their exact
        // file set identifies one revision unambiguously. Never guess by filesystem order.
        if recorded_revision.is_none() && matching.len() != 1 {
            continue;
        }
        if let Some(snapshot) = matching.into_iter().next() {
            let receipt_tree_stamp = receipt
                .get("artifactTreeStamp")
                .and_then(Value::as_str)
                .filter(|stamp| is_sha256_fingerprint(stamp));
            let tree_stamp_matches = match receipt_tree_stamp {
                // A stamp exists: retain its baseline. Cold content verification owns recovery
                // when metadata changes; this synchronous lookup never repairs over a mismatch.
                Some(expected) => {
                    sceneworks_core::download_receipt::resolved_files_tree_stamp_matches(
                        &snapshot, &files, expected,
                    )
                    .unwrap_or(false)
                }
                // No stamp was ever recorded (sc-16482). Establish one now when the caller allows
                // it, so this install can reach the evidence path instead of being pinned to legacy
                // forever. A repair failure is never fatal — it just leaves provenance unproven.
                None if repair == ProvenanceRepair::Allow => {
                    match establish_receipt_tree_stamp(
                        marker, &receipt, &snapshot, &files, "repair", None,
                    ) {
                        Ok(stamp) => {
                            tracing::info!(
                                event = "artifact_tree_stamp_repaired",
                                repo,
                                marker = %marker.display(),
                                snapshot = %snapshot.display(),
                                stamp,
                                "established a tree-stamp baseline for a previously unstamped install"
                            );
                            true
                        }
                        Err(error) => {
                            tracing::warn!(
                                event = "artifact_tree_stamp_repair_failed",
                                repo,
                                marker = %marker.display(),
                                error = %error,
                                "left the install unproven; it stays on the legacy selector"
                            );
                            false
                        }
                    }
                }
                None => false,
            };
            // Tier receipts are self-contained below a tier directory. Only descend when every
            // file belongs to that same recorded tier; otherwise return the snapshot root.
            if let Some(variant) = receipt.get("variant").and_then(Value::as_str) {
                let tier = snapshot.join(variant);
                let prefix = format!("{variant}/");
                if files.iter().all(|file| file.starts_with(&prefix)) && tier.is_dir() {
                    let revision = snapshot.file_name()?.to_str()?.to_owned();
                    let variant = variant.to_owned();
                    return Some(ResolvedWeights {
                        receipt: receipt.clone(),
                        marker: marker.to_path_buf(),
                        snapshot: snapshot.clone(),
                        path: tier,
                        provenance: tree_stamp_matches.then(|| ResolvedArtifactProvenance {
                            identity: ResolvedArtifactIdentity {
                                repository: repo.to_owned(),
                                revision: revision.clone(),
                                variant: variant.clone(),
                                fingerprint: resolved_artifact_fingerprint(
                                    repo, &revision, &variant,
                                ),
                            },
                            fixed_artifact_tier: receipt_tier(&variant).or_else(|| {
                                receipt
                                    .get("resolvedTier")
                                    .and_then(Value::as_str)
                                    .and_then(supported_artifact_tier)
                            }),
                        }),
                    });
                }
            }
            let revision = snapshot.file_name()?.to_str()?.to_owned();
            let variant = receipt
                .get("variant")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_owned();
            return Some(ResolvedWeights {
                receipt: receipt.clone(),
                marker: marker.to_path_buf(),
                snapshot: snapshot.clone(),
                path: snapshot,
                provenance: tree_stamp_matches.then(|| ResolvedArtifactProvenance {
                    identity: ResolvedArtifactIdentity {
                        repository: repo.to_owned(),
                        revision: revision.clone(),
                        variant: variant.clone(),
                        fingerprint: resolved_artifact_fingerprint(repo, &revision, &variant),
                    },
                    fixed_artifact_tier: receipt
                        .get("resolvedTier")
                        .and_then(Value::as_str)
                        .and_then(supported_artifact_tier)
                        .or_else(|| receipt_tier(&variant)),
                }),
            });
        }
    }
    None
}

#[cfg(test)]
thread_local! {
    static RECEIPT_MARKERS_READ: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_receipt_markers_read() {
    RECEIPT_MARKERS_READ.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn receipt_markers_read() -> usize {
    RECEIPT_MARKERS_READ.with(std::cell::Cell::get)
}

fn resolve_huggingface_snapshot_dir(data_dir: &Path, repo: &str) -> Option<PathBuf> {
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
        sceneworks_core::hf_home::model_source_library(data_dir),
    );
    let repo_dir = resolver.source_library().repository_root(repo).ok()?;
    let snapshots = repo_dir.join("snapshots");
    // Prefer `refs/main`, but ONLY when the snapshot it names actually holds files. A polluted
    // `refs/main` — e.g. a test that clobbered it to an empty placeholder snapshot (sc-13834) —
    // must not strand the model on an empty dir. This is best-effort defense in depth, not a
    // substitute for a correct `refs/main`: it distinguishes a materialized snapshot from an
    // empty/torn one, not dummy fixture weights from real ones.
    if let Ok(rev) = std::fs::read_to_string(repo_dir.join("refs").join("main")) {
        let revision = rev.trim();
        let candidate = snapshots.join(revision);
        if snapshot_has_any_file(&candidate) {
            // The name is read off disk, so it is resolved as an installed snapshot rather than
            // admitted as a caller-supplied revision: an install materialized under a mutable name
            // (the endpoint omitted `X-Repo-Commit`) is still installed. Demanding an immutable
            // name here returned `None` for it — and, because this arm returns, without even
            // trying the fallback scan below.
            return resolver
                .source_library()
                .discover_installed_snapshot_path(repo, revision)
                .ok();
        }
    }
    // Fallback: the cached snapshot with the most files, so an empty/partial one never wins over a
    // fully materialized sibling (the old "first dir wins" scan happily returned an empty snapshot).
    let from_source_library = std::fs::read_dir(&snapshots)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .map(|path| (snapshot_file_count(&path), path))
        .filter(|(count, _)| *count > 0)
        .max_by_key(|(count, _)| *count)
        .and_then(|(_, path)| {
            let revision = path.file_name()?.to_str()?;
            resolver
                .source_library()
                .discover_installed_snapshot_path(repo, revision)
                .ok()
        });
    if from_source_library.is_some() {
        return from_source_library;
    }
    // Last resort (sc-19707): the configured library cannot name a revision at all — it is
    // disconnected, or was never installed here. The typed resolver answers from a leased
    // app-owned bundle when exactly one revision of this repository is being served, which is what
    // keeps a complete local model loadable while the drive is unplugged. With no active lease it
    // still returns nothing, so an uninstalled model keeps its established behavior.
    resolver
        .discover_source_snapshot(repo, None)
        .ok()
        .map(|(_, path)| path)
}

/// Short-circuiting presence probe for the hot `refs/main` path. Unlike
/// [`snapshot_file_count`], it stops at the first file instead of walking a large snapshot tree on
/// every model resolve.
fn snapshot_has_any_file(dir: &Path) -> bool {
    snapshot_tree_contains(dir, &|_| true)
}

/// Recursively count regular files under `dir` (symlinks counted as files — HF snapshots link into
/// `blobs/`), returning 0 when it is absent, unreadable, or holds only empty subdirectories. Lets
/// [`resolve_huggingface_snapshot_dir`] tell a fully materialized snapshot from an empty placeholder.
fn snapshot_file_count(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut count = 0;
    for entry in entries.flatten() {
        match entry.file_type() {
            Ok(file_type) if file_type.is_dir() => count += snapshot_file_count(&entry.path()),
            Ok(_) => count += 1,
            Err(_) => {}
        }
    }
    count
}

/// Replace the relative `snapshots/<rev>/… -> ../blobs/<etag>` symlinks under a resolved
/// HF snapshot dir with hardlinks to the same blob. **Windows-only** (a no-op
/// elsewhere): the candle worker process cannot traverse those reparse points — the
/// open fails with `ERROR_UNTRUSTED_MOUNT_POINT` (os error 448, see
/// [`downloaded_model_detection_io_error_is_inconclusive`]) — but a hardlink is a plain
/// directory entry to the blob data (no reparse, same volume, still deduped) and reads
/// fine. Caches our downloader writes now hardlink from the start
/// ([`crate::downloads`]); this repairs caches already materialized as symlinks (e.g. by
/// `huggingface_hub`). Best-effort and idempotent: once converted there is nothing left
/// to do, and any per-file failure leaves that entry untouched for the loader to
/// surface. Compiled on `test` too so the conversion is unit-testable off-Windows.
#[cfg(any(windows, test))]
fn materialize_snapshot_hardlinks(dir: &Path) {
    fn repair(dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_symlink() {
                relink_to_blob(&path);
            } else if file_type.is_dir() {
                repair(&path);
            }
        }
    }
    repair(dir);
}

/// Replace a single snapshot symlink with a hardlink (or, failing that, a copy) to its
/// blob, so the entry is always present afterward. The symlink target is read **without
/// traversing it** (`read_link`), so resolving the blob never hits the os-448 path; the
/// symlink is removed only once its blob is confirmed to exist.
#[cfg(any(windows, test))]
fn relink_to_blob(link: &Path) {
    let Ok(target) = std::fs::read_link(link) else {
        return;
    };
    let Some(parent) = link.parent() else {
        return;
    };
    // `canonicalize` resolves the `..` segments and confirms the blob exists; it opens
    // the blob (a plain file, not a reparse point) so it does not hit the os-448 path.
    let Ok(blob) = std::fs::canonicalize(parent.join(target)) else {
        return;
    };
    if std::fs::remove_file(link).is_err() {
        return;
    }
    if std::fs::hard_link(&blob, link).is_err() {
        // Cross-volume or a filesystem without hardlinks: fall back to a real copy so
        // the entry is never left missing.
        let _ = std::fs::copy(&blob, link);
    }
}
