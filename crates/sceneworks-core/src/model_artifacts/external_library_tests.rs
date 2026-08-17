use super::*;
use crate::model_artifacts::{
    ArtifactAvailability, ArtifactCompleteness, ArtifactFile, ArtifactIdentity, ArtifactLocation,
    ArtifactMemberRole, ArtifactProvenance, ResolvedBundleClosure, ResolvedBundleMember,
    ResolvedModelArtifact, MODEL_ARTIFACT_CONTRACT_VERSION,
};
use tempfile::TempDir;

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

#[cfg(target_os = "macos")]
#[test]
fn macos_volume_uuid_parser_is_exact_and_fail_closed() {
    let valid = br#"<?xml version="1.0"?><plist><dict><key>VolumeUUID</key>
        <string>01234567-89AB-CDEF-0123-456789ABCDEF</string></dict></plist>"#;
    assert_eq!(
        macos_volume_uuid_from_diskutil(true, valid, b"").unwrap(),
        "0123456789abcdef0123456789abcdef"
    );
    for invalid in [
        br#"<plist><dict></dict></plist>"#.as_slice(),
        br#"<key>VolumeUUID</key><string>short</string>"#.as_slice(),
        br#"<key>VolumeUUID</key><string>zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz</string>"#
            .as_slice(),
        br#"<key>VolumeUUID</key><key>Other</key><string>01234567-89ab-cdef-0123-456789abcdef</string>"#
            .as_slice(),
        &[0xff, 0xfe],
    ] {
        assert!(macos_volume_uuid_from_diskutil(true, invalid, b"").is_err());
    }
    assert!(macos_volume_uuid_from_diskutil(false, valid, b"not available").is_err());
    let unavailable = macos_command_output(
        "diskutil",
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "fixture command missing",
        )),
    )
    .unwrap_err();
    assert!(unavailable.to_string().contains("diskutil is unavailable"));
}

fn requirement(repo: &str, file: &str) -> ExternalArtifactRequirement {
    ExternalArtifactRequirement {
        repository: repo.to_owned(),
        revision: Some(REVISION.to_owned()),
        variant: "default".to_owned(),
        files: vec![PathBuf::from(file)],
        is_primary: true,
    }
}

fn seed_snapshot(root: &Path, repo: &str, file: &str) -> PathBuf {
    let repo = root.join(format!("models--{}", safe_repo_dir_name(repo).unwrap()));
    let snapshot = repo.join("snapshots").join(REVISION);
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join(file), b"weights").unwrap();
    snapshot
}

#[test]
fn disconnect_reconnect_preserves_binding_and_receipt_identity() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let (binding, probe) = store
        .bind_or_probe_validated(&library, &requirements)
        .unwrap();
    assert_eq!(probe.status, ExternalLibraryProbeStatus::Available);

    let disconnected = temp.path().join("detached");
    std::fs::rename(&library, &disconnected).unwrap();
    assert_eq!(
        store.probe_bound(&library, &binding).status,
        ExternalLibraryProbeStatus::Unavailable
    );
    assert_eq!(store.load().unwrap(), Some(binding.clone()));

    std::fs::rename(&disconnected, &library).unwrap();
    assert_eq!(
        store.probe_bound(&library, &binding).status,
        ExternalLibraryProbeStatus::Available
    );
}

#[test]
fn binding_and_disconnect_probes_never_mutate_install_receipts() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    let receipt_dir = data.join("models").join("owner--model");
    std::fs::create_dir_all(&receipt_dir).unwrap();
    let receipt_path = receipt_dir.join(".sceneworks-download-complete.json");
    let receipt = br#"{"repo":"owner/model","snapshotRevision":"0123456789abcdef0123456789abcdef01234567","resolvedFiles":["model.safetensors"]}"#;
    std::fs::write(&receipt_path, receipt).unwrap();
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let (binding, _) = store
        .bind_or_probe_validated(&library, &requirements)
        .unwrap();
    std::fs::rename(&library, temp.path().join("detached")).unwrap();
    assert_eq!(
        store.probe_bound(&library, &binding).status,
        ExternalLibraryProbeStatus::Unavailable
    );
    assert_eq!(std::fs::read(receipt_path).unwrap(), receipt);
}

#[test]
fn stamped_resolution_is_versioned_and_serde_round_trips_exactly() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let (binding, _) = store
        .bind_or_probe_validated(&library, &requirements)
        .unwrap();
    let resolution = ModelResolution::external_ready(library, binding, requirements).unwrap();
    let serialized = serde_json::to_vec(&resolution).unwrap();
    let decoded: ModelResolution = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(decoded, resolution);
    assert_eq!(decoded.schema_version, EXTERNAL_LIBRARY_CONTRACT_VERSION);
}

#[test]
fn advisory_resolution_must_match_the_durable_binding_ledger_exactly() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let (binding, _) = store
        .bind_or_probe_validated(&library, &requirements)
        .unwrap();
    let mut forged = ModelResolution::external_ready(library, binding, requirements).unwrap();
    forged
        .expected_library
        .as_mut()
        .unwrap()
        .physical_identity
        .directory_id ^= 1;

    let probe = store.probe_resolution(&forged).unwrap();
    assert_eq!(probe.status, ExternalLibraryProbeStatus::IdentityMismatch);
}

#[test]
fn local_ready_remains_available_without_any_source_library() {
    let temp = TempDir::new().unwrap();
    let local = temp.path().join("resolved");
    std::fs::create_dir_all(&local).unwrap();
    std::fs::write(local.join("model.safetensors"), b"local").unwrap();
    let identity = ArtifactIdentity::pinned("owner/model", REVISION, "default").unwrap();
    let artifact = ResolvedModelArtifact {
        schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
        identity: identity.clone(),
        location: ArtifactLocation::ResolvedLocal {
            root: local.clone(),
        },
        closure: ResolvedBundleClosure::new(vec![ResolvedBundleMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: identity.clone(),
            tier: None,
            source_subpath: PathBuf::new(),
            destination: PathBuf::new(),
            files: vec![ArtifactFile::new("model.safetensors").unwrap()],
        }])
        .unwrap(),
        provenance: ArtifactProvenance {
            identity,
            fixed_artifact_tier: None,
        },
        completeness: ArtifactCompleteness::Complete,
        availability: ArtifactAvailability::Available,
    };
    let mut source_tier = artifact.clone();
    source_tier.location = ArtifactLocation::SourceLibrary {
        root: local.clone(),
    };
    assert!(ModelResolution::local_ready(source_tier).is_err());

    let resolution = ModelResolution::local_ready(artifact).unwrap();
    let mut forged = resolution.clone();
    forged.configured_library_path = temp.path().join("external");
    assert!(forged.validate().is_err());
    let store = ExternalLibraryBindingStore::new(&temp.path().join("data")).unwrap();
    let probe = store.probe_resolution(&resolution).unwrap();
    assert_eq!(probe.status, ExternalLibraryProbeStatus::Available);
    assert_eq!(probe.observed_path, Some(local));
}

#[test]
fn same_path_reuse_by_an_unrelated_directory_fails_closed() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let (binding, _) = store
        .bind_or_probe_validated(&library, &requirements)
        .unwrap();

    std::fs::rename(&library, temp.path().join("old-library")).unwrap();
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    assert_eq!(
        store.probe_bound(&library, &binding).status,
        ExternalLibraryProbeStatus::IdentityMismatch
    );
}

#[test]
fn unavailable_corequisite_prevents_legacy_binding() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/primary", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let requirements = vec![
        requirement("owner/primary", "model.safetensors"),
        ExternalArtifactRequirement {
            is_primary: false,
            ..requirement("owner/encoder", "encoder.safetensors")
        },
    ];
    assert!(store
        .bind_or_probe_validated(&library, &requirements)
        .is_err());
    assert!(store.load().unwrap().is_none());
}

#[test]
fn exact_requirement_rejects_traversal_and_snapshot_ambiguity() {
    let temp = TempDir::new().unwrap();
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    let traversal = ExternalArtifactRequirement {
        files: vec![PathBuf::from("../outside")],
        ..requirement("owner/model", "model.safetensors")
    };
    assert!(validate_requirements_at_root(&library, &[traversal]).is_err());

    let legacy = ExternalArtifactRequirement {
        revision: None,
        ..requirement("owner/model", "model.safetensors")
    };
    let second = library
        .join("models--owner--model")
        .join("snapshots")
        .join("fedcba9876543210fedcba9876543210fedcba98");
    std::fs::create_dir_all(&second).unwrap();
    std::fs::write(second.join("model.safetensors"), b"other").unwrap();
    assert!(validate_requirements_at_root(&library, &[legacy]).is_err());
}

#[test]
fn source_session_cleanup_is_exact_and_never_touches_external_bytes() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    let snapshot = seed_snapshot(&library, "owner/model", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let (binding, _) = store
        .bind_or_probe_validated(&library, &requirements)
        .unwrap();
    let resolution =
        ModelResolution::external_ready(library.clone(), binding, requirements).unwrap();

    let session = ExternalSourceSession::begin(&data, &resolution).unwrap();
    let staging = session.staging_root().to_path_buf();
    let unrelated = data.join("models").join("unrelated.partial");
    std::fs::write(staging.join("owned.partial"), b"partial").unwrap();
    std::fs::write(&unrelated, b"keep").unwrap();
    drop(session);

    assert!(!staging.exists());
    assert_eq!(std::fs::read(unrelated).unwrap(), b"keep");
    assert_eq!(
        std::fs::read(snapshot.join("model.safetensors")).unwrap(),
        b"weights"
    );
}

#[cfg(unix)]
#[test]
fn source_session_cleanup_never_follows_a_replaced_staging_symlink() {
    use std::os::unix::fs::symlink;
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let (binding, _) = store
        .bind_or_probe_validated(&library, &requirements)
        .unwrap();
    let resolution = ModelResolution::external_ready(library, binding, requirements).unwrap();
    let session = ExternalSourceSession::begin(&data, &resolution).unwrap();
    let staging = session.staging_root().to_path_buf();
    let outside = temp.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(outside.join("sentinel"), b"keep").unwrap();
    std::fs::remove_dir(&staging).unwrap();
    symlink(&outside, &staging).unwrap();

    drop(session);

    assert_eq!(std::fs::read(outside.join("sentinel")).unwrap(), b"keep");
}

#[cfg(windows)]
#[test]
fn windows_source_session_cleanup_never_follows_a_replaced_staging_junction() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let (binding, _) = store
        .bind_or_probe_validated(&library, &requirements)
        .unwrap();
    let resolution = ModelResolution::external_ready(library, binding, requirements).unwrap();
    let session = ExternalSourceSession::begin(&data, &resolution).unwrap();
    let staging = session.staging_root().to_path_buf();
    let outside = temp.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(outside.join("sentinel"), b"keep").unwrap();
    std::fs::remove_dir(&staging).unwrap();
    let output = std::process::Command::new("cmd")
        .args(["/D", "/C", "mklink", "/J"])
        .arg(&staging)
        .arg(&outside)
        .output()
        .expect("create staging junction");
    assert!(
        output.status.success(),
        "mklink /J failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    drop(session);

    assert_eq!(std::fs::read(outside.join("sentinel")).unwrap(), b"keep");
    if staging.exists() {
        std::fs::remove_dir(staging).unwrap();
    }
}

#[cfg(unix)]
#[test]
fn normal_hugging_face_snapshot_blob_symlinks_remain_supported() {
    use std::os::unix::fs::symlink;
    let temp = TempDir::new().unwrap();
    let library = temp.path().join("external-hf");
    let repo = library.join("models--owner--model");
    let snapshot = repo.join("snapshots").join(REVISION);
    let blobs = repo.join("blobs");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::write(blobs.join("abc123"), b"weights").unwrap();
    symlink("../../blobs/abc123", snapshot.join("model.safetensors")).unwrap();

    validate_requirements_at_root(&library, &[requirement("owner/model", "model.safetensors")])
        .unwrap();
}

/// The full typed lifecycle through the ONE resolver: ready → disconnected (typed, installed
/// state and receipts intact) → reconnected (ready again) → component removed while connected
/// (incomplete, not unavailable). Every catalog row, preflight, and the worker guard read these
/// states from this single function.
#[test]
fn resolver_walks_ready_unavailable_reconnected_and_incomplete_states() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    let snapshot = seed_snapshot(&library, "owner/model", "model.safetensors");
    let requirements = vec![requirement("owner/model", "model.safetensors")];

    let ready = resolve_model_availability(&data, &library, &requirements, true, &[]);
    assert_eq!(ready.availability, ModelAvailability::ExternalReady);
    ready.validate().unwrap();
    let binding = ready.expected_library.clone().unwrap();

    // Disconnect: installed identity (requirements + binding) is preserved, availability is the
    // typed unavailable condition, and nothing rewrote the binding ledger.
    let detached = temp.path().join("detached");
    std::fs::rename(&library, &detached).unwrap();
    let unavailable = resolve_model_availability(&data, &library, &requirements, true, &[]);
    assert_eq!(
        unavailable.availability,
        ModelAvailability::InstalledExternalUnavailable
    );
    assert_eq!(unavailable.expected_library.as_ref(), Some(&binding));
    assert_eq!(unavailable.requirements, requirements);
    unavailable.validate().unwrap();

    // Reconnect the SAME volume/path: availability returns to ready under the original binding.
    std::fs::rename(&detached, &library).unwrap();
    let reconnected = resolve_model_availability(&data, &library, &requirements, true, &[]);
    assert_eq!(reconnected.availability, ModelAvailability::ExternalReady);
    assert_eq!(reconnected.expected_library.as_ref(), Some(&binding));

    // Remove one required component while the library stays connected: this is a stale/incomplete
    // install, NOT a library disconnect — the two conditions must never blur.
    std::fs::remove_file(snapshot.join("model.safetensors")).unwrap();
    let incomplete = resolve_model_availability(&data, &library, &requirements, true, &[]);
    assert_eq!(incomplete.availability, ModelAvailability::Incomplete);
    assert!(incomplete.expected_library.is_none());
}

/// Install evidence gates the typed disconnect state when no binding exists yet. A declared-only
/// closure (manifest identity, no receipts) with no library present was never installed: Missing,
/// so the established download path is preserved. A receipt-backed closure in the same situation
/// is the pre-binding upgrade path — the receipts prove the install, so it is the typed
/// unavailable condition, never a silent re-download.
#[test]
fn without_a_binding_only_receipt_evidence_can_prove_installed_but_unavailable() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("never-mounted");
    let requirements = vec![requirement("owner/model", "model.safetensors")];

    let declared_only = resolve_model_availability(&data, &library, &requirements, false, &[]);
    assert_eq!(declared_only.availability, ModelAvailability::Missing);
    assert!(declared_only.expected_library.is_none());

    let receipt_backed = resolve_model_availability(&data, &library, &requirements, true, &[]);
    assert_eq!(
        receipt_backed.availability,
        ModelAvailability::InstalledExternalUnavailable
    );
}

/// With a durable binding present and the volume disconnected, the typed unavailable state still
/// requires install evidence for THE MODEL, not merely for the library: a receipt-less legacy
/// install whose exact closure was validated while connected (the validated-closures ledger) stays
/// installed-but-unavailable, while a declared-exact model that was never installed resolves
/// Missing — the binding of an unrelated model must not misrepresent it as installed, 503 its
/// submissions, or typed-fail its established install path.
#[test]
fn with_a_binding_only_receipted_or_ledger_validated_closures_read_unavailable() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    // Legacy install: present in the library, NO download receipt anywhere. Validating it while
    // connected binds the library and records the closure in the ledger.
    seed_snapshot(&library, "owner/legacy", "legacy.safetensors");
    let legacy = vec![requirement("owner/legacy", "legacy.safetensors")];
    let ready = resolve_model_availability(&data, &library, &legacy, false, &[]);
    assert_eq!(ready.availability, ModelAvailability::ExternalReady);
    // Never-installed sibling: declared-exact manifest identity only, no files, no receipt.
    let never_installed = vec![requirement("owner/never-installed", "missing.safetensors")];

    std::fs::rename(&library, temp.path().join("detached")).unwrap();

    let legacy_offline = resolve_model_availability(&data, &library, &legacy, false, &[]);
    assert_eq!(
        legacy_offline.availability,
        ModelAvailability::InstalledExternalUnavailable,
        "a ledger-validated receipt-less install must never degrade to Missing on disconnect"
    );
    let never_installed_offline =
        resolve_model_availability(&data, &library, &never_installed, false, &[]);
    assert_eq!(
        never_installed_offline.availability,
        ModelAvailability::Missing,
        "an unrelated model's binding must not manufacture installed-but-unavailable"
    );

    // Reconnect: the ledger-validated install returns to ready under the original binding.
    std::fs::rename(temp.path().join("detached"), &library).unwrap();
    let reconnected = resolve_model_availability(&data, &library, &legacy, false, &[]);
    assert_eq!(reconnected.availability, ModelAvailability::ExternalReady);
    assert_eq!(
        reconnected.expected_library, ready.expected_library,
        "reconnecting the same volume preserves the original binding"
    );
}

/// An empty requirement closure means no durable install identity: the resolver reports Missing
/// (preserving established download behavior) and never invents a binding.
#[test]
fn resolver_reports_missing_for_an_empty_closure_without_binding() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    let missing = resolve_model_availability(&data, &library, &[], false, &[]);
    assert_eq!(missing.availability, ModelAvailability::Missing);
    assert!(missing.expected_library.is_none());
    missing.validate().unwrap();
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    assert!(store.load().unwrap().is_none());
}

fn covering_local_artifact(root: &Path) -> ResolvedModelArtifact {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(root.join("model.safetensors"), b"local").unwrap();
    let identity = ArtifactIdentity::pinned("owner/model", REVISION, "default").unwrap();
    ResolvedModelArtifact {
        schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
        identity: identity.clone(),
        location: ArtifactLocation::ResolvedLocal {
            root: root.to_path_buf(),
        },
        closure: ResolvedBundleClosure::new(vec![ResolvedBundleMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: identity.clone(),
            tier: None,
            source_subpath: PathBuf::new(),
            destination: PathBuf::new(),
            files: vec![ArtifactFile::new("model.safetensors").unwrap()],
        }])
        .unwrap(),
        provenance: ArtifactProvenance {
            identity,
            fixed_artifact_tier: None,
        },
        completeness: ArtifactCompleteness::Complete,
        availability: ArtifactAvailability::Available,
    }
}

/// A covering app-owned resolved-local artifact wins over any external state — including a
/// disconnected library — and an artifact for a DIFFERENT variant never covers the selection.
#[test]
fn resolver_prefers_covering_local_artifact_and_rejects_variant_mismatch() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let artifact = covering_local_artifact(&temp.path().join("resolved"));

    // Library never existed / is unavailable: local artifact still resolves LocalReady.
    let local = resolve_model_availability(
        &data,
        &library,
        &requirements,
        true,
        std::slice::from_ref(&artifact),
    );
    assert_eq!(local.availability, ModelAvailability::LocalReady);
    local.validate().unwrap();

    // Same repository but a different selected variant: the artifact must NOT cover it —
    // completeness is judged against the exact selected variant closure.
    let mut q8 = requirements.clone();
    q8[0].variant = "q8".to_owned();
    let not_covered = resolve_model_availability(&data, &library, &q8, true, &[artifact]);
    assert_ne!(not_covered.availability, ModelAvailability::LocalReady);
}

#[cfg(unix)]
#[test]
fn source_file_symlink_must_remain_inside_the_same_repository() {
    use std::os::unix::fs::symlink;
    let temp = TempDir::new().unwrap();
    let library = temp.path().join("external-hf");
    let snapshot = seed_snapshot(&library, "owner/model", "placeholder");
    let outside = temp.path().join("outside.safetensors");
    std::fs::write(&outside, b"outside").unwrap();
    symlink(&outside, snapshot.join("model.safetensors")).unwrap();
    assert!(validate_requirements_at_root(
        &library,
        &[requirement("owner/model", "model.safetensors")]
    )
    .is_err());
}
