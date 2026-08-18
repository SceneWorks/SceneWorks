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

// ---------------------------------------------------------------------------------------------
// Relocation (sc-19709). `bind_or_probe_validated` never replaces a binding, which is what makes a
// different volume at the same path fail closed; `relocate_binding` is the one deliberate escape
// hatch, and every test below exists to keep it from becoming a way to bind the WRONG volume.
// ---------------------------------------------------------------------------------------------

/// A durable download receipt for `repo` — install evidence that exists WITHOUT any binding or
/// validated-closure record, which is the state every relocation check has to survive.
fn write_download_receipt(data_dir: &Path, repo: &str, file: &str) {
    let managed = data_dir
        .join("models")
        .join(crate::model_artifacts::artifact_selection::safe_download_dir(repo));
    std::fs::create_dir_all(&managed).unwrap();
    std::fs::write(
        managed.join(".sceneworks-download-complete.json"),
        serde_json::to_vec(&serde_json::json!({
            "receipts": [{
                "repo": repo,
                "variant": "default",
                "resolvedFiles": [file],
                "snapshotRevision": REVISION,
            }]
        }))
        .unwrap(),
    )
    .unwrap();
}

/// Nothing installed means nothing to relocate. Binding whatever Hugging-Face-shaped directory the
/// operator happened to pick would be a guess, and it would silently become the identity every
/// later install is judged against.
#[test]
fn relocation_refuses_outright_when_there_is_no_install_evidence() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let candidate = temp.path().join("someones-cache").join("hub");
    std::fs::create_dir_all(&candidate).unwrap();
    seed_snapshot(&candidate, "someone/unrelated", "model.safetensors");

    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    assert_eq!(
        store.relocate_binding(&candidate).unwrap_err(),
        LibraryRelocationError::Rejected(LibraryRelocationRejection::NoInstalledModels)
    );
    assert!(store.load().unwrap().is_none());
}

/// THE fail-open this exists to prevent. Evidence can be RECEIPTS ONLY — an install that predates
/// the validated-closure ledger, or one never validated on a bound library. Judging the candidate
/// against the ledger alone would find nothing to check, so any directory with a single `models--*`
/// child would validate vacuously and capture the binding.
#[test]
fn relocation_checks_receipt_evidence_even_with_an_empty_ledger() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    write_download_receipt(&data, "owner/model", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    assert!(
        store.validated_closures().unwrap().is_empty(),
        "this test is only meaningful while the ledger is empty"
    );

    let decoy = temp.path().join("decoy").join("hub");
    std::fs::create_dir_all(&decoy).unwrap();
    seed_snapshot(&decoy, "someone/unrelated", "model.safetensors");
    assert_eq!(
        store.relocate_binding(&decoy).unwrap_err(),
        LibraryRelocationError::Rejected(LibraryRelocationRejection::MissingInstalledModels {
            repositories: vec!["owner/model".to_owned()],
        })
    );
    assert!(store.load().unwrap().is_none(), "a refusal writes nothing");

    // The library that actually holds the receipted install is adopted.
    let moved = temp.path().join("moved").join("hub");
    std::fs::create_dir_all(&moved).unwrap();
    seed_snapshot(&moved, "owner/model", "model.safetensors");
    let binding = store.relocate_binding(&moved).unwrap();
    assert_eq!(
        binding.canonical_path,
        std::fs::canonicalize(&moved).unwrap()
    );
}

/// A legacy receipt with no recorded variant is the `default` variant, exactly as
/// `receipt_requirements_for_model` reads it — otherwise a pre-variant install would contribute no
/// evidence and reopen the same fail-open.
#[test]
fn a_variant_less_legacy_receipt_still_counts_as_evidence() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let managed = data
        .join("models")
        .join(crate::model_artifacts::artifact_selection::safe_download_dir("owner/legacy"));
    std::fs::create_dir_all(&managed).unwrap();
    std::fs::write(
        managed.join(".sceneworks-download-complete.json"),
        serde_json::to_vec(&serde_json::json!({
            "receipts": [{
                "repo": "owner/legacy",
                "resolvedFiles": ["legacy.safetensors"],
                "snapshotRevision": REVISION,
            }]
        }))
        .unwrap(),
    )
    .unwrap();

    let decoy = temp.path().join("decoy").join("hub");
    std::fs::create_dir_all(&decoy).unwrap();
    seed_snapshot(&decoy, "someone/unrelated", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    assert_eq!(
        store.relocate_binding(&decoy).unwrap_err(),
        LibraryRelocationError::Rejected(LibraryRelocationRejection::MissingInstalledModels {
            repositories: vec!["owner/legacy".to_owned()],
        })
    );
}

/// A library that holds SOME of what is installed is still the wrong library: accepting it would
/// silently orphan the rest. Every missing repository is named, so the guidance is actionable.
#[test]
fn relocation_refuses_a_partial_library_and_names_what_is_missing() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let original = temp.path().join("original").join("hub");
    std::fs::create_dir_all(&original).unwrap();
    seed_snapshot(&original, "owner/one", "model.safetensors");
    seed_snapshot(&original, "owner/two", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    // Both closures validated on the bound library, so both are in the ledger.
    store
        .bind_or_probe_validated(&original, &[requirement("owner/one", "model.safetensors")])
        .unwrap();
    store
        .bind_or_probe_validated(&original, &[requirement("owner/two", "model.safetensors")])
        .unwrap();
    // A receipt for a third install with no ledger record at all.
    write_download_receipt(&data, "owner/three", "model.safetensors");

    let partial = temp.path().join("partial").join("hub");
    std::fs::create_dir_all(&partial).unwrap();
    seed_snapshot(&partial, "owner/one", "model.safetensors");
    match store.relocate_binding(&partial).unwrap_err() {
        LibraryRelocationError::Rejected(LibraryRelocationRejection::MissingInstalledModels {
            repositories,
        }) => assert_eq!(repositories, ["owner/three", "owner/two"]),
        other => panic!("expected a missing-models refusal, got {other:?}"),
    }
}

/// The dry run answers identically and writes nothing — neither a binding nor a ledger byte — which
/// is what lets the client validate before it persists its own copy of the location.
#[test]
fn validate_relocation_answers_like_the_real_call_and_writes_nothing() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let original = temp.path().join("original").join("hub");
    std::fs::create_dir_all(&original).unwrap();
    seed_snapshot(&original, "owner/model", "model.safetensors");
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    store
        .bind_or_probe_validated(
            &original,
            &[requirement("owner/model", "model.safetensors")],
        )
        .unwrap();
    let binding_before = store.load().unwrap();
    let closures_before = store.validated_closures().unwrap();

    let moved = temp.path().join("moved").join("hub");
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    std::fs::rename(&original, &moved).unwrap();
    store.validate_relocation(&moved).unwrap();
    assert_eq!(store.load().unwrap(), binding_before);
    assert_eq!(store.validated_closures().unwrap(), closures_before);

    let unrelated = temp.path().join("holiday-photos");
    std::fs::create_dir_all(&unrelated).unwrap();
    assert_eq!(
        store.validate_relocation(&unrelated).unwrap_err(),
        LibraryRelocationError::Rejected(LibraryRelocationRejection::NotAModelLibrary)
    );
    assert_eq!(store.load().unwrap(), binding_before);
}

/// The TOCTOU re-check: the closure walk touches many files and can race an unmount/remount, so the
/// exact canonical path AND physical identity must still hold after validation. Asserted on the
/// pure comparison, because the race itself cannot be scheduled deterministically in a test.
#[test]
fn relocation_requires_the_identity_to_hold_across_the_closure_walk() {
    let path = PathBuf::from("/Volumes/Models/hf/hub");
    let other = PathBuf::from("/Volumes/Models 1/hf/hub");
    let identity = ExternalLibraryPhysicalIdentity {
        volume_id: "volume-a".to_owned(),
        directory_id: 7,
    };
    let remounted = ExternalLibraryPhysicalIdentity {
        volume_id: "volume-b".to_owned(),
        directory_id: 7,
    };
    assert!(relocation_identity_held(&path, &path, &identity, &identity));
    assert!(!relocation_identity_held(
        &path, &other, &identity, &identity
    ));
    assert!(!relocation_identity_held(
        &path, &path, &identity, &remounted
    ));
}

/// Relocation replaces the ONE durable binding, so a resolution stamped against the old library —
/// the copy a worker is still carrying with an in-flight job — must stop validating rather than be
/// silently accepted against the new volume. It fails closed as an identity mismatch.
#[test]
fn a_resolution_stamped_before_relocation_no_longer_matches_the_binding() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let original = temp.path().join("original").join("hub");
    std::fs::create_dir_all(&original).unwrap();
    seed_snapshot(&original, "owner/model", "model.safetensors");
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let (old_binding, _) = store
        .bind_or_probe_validated(&original, &requirements)
        .unwrap();
    let stamped =
        ModelResolution::external_ready(original.clone(), old_binding, requirements.clone())
            .unwrap();

    let moved = temp.path().join("moved").join("hub");
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    std::fs::rename(&original, &moved).unwrap();
    store.relocate_binding(&moved).unwrap();

    assert_eq!(
        store.probe_resolution(&stamped).unwrap().status,
        ExternalLibraryProbeStatus::IdentityMismatch,
    );
    // Install evidence is untouched by relocation: nothing is redownloaded and nothing is lost.
    assert_eq!(
        store.validated_closures().unwrap(),
        vec![canonical_requirement_closure(&requirements)]
    );
}

/// The full identity matrix behind `probe_binding` (sc-19709). CANONICAL PATH + VOLUME IDENTITY are
/// the authority; the lexical name the library is configured as is not.
///
/// The case that forced this: `~/.cache/huggingface/hub` is a symlink to a real external drive, so
/// an operator who configures the drive directly — the intended way to use an external library —
/// was handing the app a second NAME for the library it had already bound. Comparing names first
/// made that an identity mismatch, which pinned every receipt-backed model to
/// `installed_external_unavailable` and pushed every other one back onto the download path, for a
/// library that was sitting right there.
#[cfg(unix)]
#[test]
fn an_alias_of_the_bound_library_probes_available_while_real_changes_still_fail_closed() {
    use std::os::unix::fs::symlink;
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let real = temp.path().join("Volumes").join("Models").join("hub");
    std::fs::create_dir_all(&real).unwrap();
    seed_snapshot(&real, "owner/model", "model.safetensors");
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let store = ExternalLibraryBindingStore::new(&data).unwrap();
    let (binding, _) = store.bind_or_probe_validated(&real, &requirements).unwrap();

    // 1. Same lexical name → available (unchanged behavior).
    assert_eq!(
        probe_binding(&real, &binding).status,
        ExternalLibraryProbeStatus::Available
    );

    // 2. DIFFERENT lexical name, same canonical path, same volume: the symlink an HF cache home
    //    normally is. Provably the same directory, so it must be available.
    let alias_home = temp.path().join("home-cache");
    std::fs::create_dir_all(&alias_home).unwrap();
    let alias = alias_home.join("hub");
    symlink(&real, &alias).unwrap();
    assert_eq!(
        probe_binding(&alias, &binding).status,
        ExternalLibraryProbeStatus::Available,
        "a symlink to the bound library is the SAME library"
    );

    // 3. Different lexical name AND different canonical path → still a mismatch.
    let other = temp.path().join("other").join("hub");
    std::fs::create_dir_all(&other).unwrap();
    assert_eq!(
        probe_binding(&other, &binding).status,
        ExternalLibraryProbeStatus::IdentityMismatch
    );

    // 4. Same canonical path, DIFFERENT recorded volume identity — the decoy remounted where the
    //    real drive was. The identity half of the comparison is what catches this.
    let decoy_binding = ExternalLibraryBinding {
        physical_identity: ExternalLibraryPhysicalIdentity {
            volume_id: "some-other-volume".to_owned(),
            directory_id: binding.physical_identity.directory_id,
        },
        ..binding.clone()
    };
    assert_eq!(
        probe_binding(&real, &decoy_binding).status,
        ExternalLibraryProbeStatus::IdentityMismatch,
        "the same path on a different volume must never read available"
    );

    // 5. A missing configured root is unavailable (disconnected), not a mismatch.
    assert_eq!(
        probe_binding(&temp.path().join("never-mounted"), &binding).status,
        ExternalLibraryProbeStatus::Unavailable
    );
}

/// End to end on the resolver: switching the configured library from the symlink to the drive it
/// points at keeps every installed model ready, with no user action, no re-download, and no typed
/// disconnect. The ledger quietly records the name now in use so the rest of the seam — stamped
/// resolutions, the worker's pre-loader guard — keeps its `configured_path` invariant.
#[cfg(unix)]
#[test]
fn configuring_the_drive_directly_instead_of_its_symlink_stays_ready() {
    use std::os::unix::fs::symlink;
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let real = temp.path().join("Volumes").join("Models").join("hub");
    std::fs::create_dir_all(&real).unwrap();
    seed_snapshot(&real, "owner/model", "model.safetensors");
    let alias_home = temp.path().join("home-cache");
    std::fs::create_dir_all(&alias_home).unwrap();
    let alias = alias_home.join("hub");
    symlink(&real, &alias).unwrap();
    let requirements = vec![requirement("owner/model", "model.safetensors")];

    // Bound while configured through the symlink, as an existing install would be.
    let through_symlink = resolve_model_availability(&data, &alias, &requirements, true, &[]);
    assert_eq!(
        through_symlink.availability,
        ModelAvailability::ExternalReady
    );

    // The operator now configures the drive directly. Same library, so it stays ready.
    let direct = resolve_model_availability(&data, &real, &requirements, true, &[]);
    assert_eq!(
        direct.availability,
        ModelAvailability::ExternalReady,
        "the same library under its real path must not read as a different one"
    );
    direct.validate().unwrap();
    let binding = direct.expected_library.as_ref().unwrap();
    assert_eq!(
        binding.configured_path, real,
        "the ledger records the name now in use, so stamped resolutions stay valid"
    );
    assert_eq!(
        binding.canonical_path,
        std::fs::canonicalize(&real).unwrap()
    );

    // And back again: neither name is privileged.
    let back = resolve_model_availability(&data, &alias, &requirements, true, &[]);
    assert_eq!(back.availability, ModelAvailability::ExternalReady);
}

/// A library that is PRESENT but whose identity disagrees is a different problem from a
/// disconnected one, and it is not fixed by reconnecting anything. The resolution says so, which is
/// what lets the prompt lead with "choose the library" instead of "reconnect the drive" — the state
/// was otherwise a dead end with no path out the user could find.
#[test]
fn a_present_but_mismatched_library_is_flagged_separately_from_a_disconnected_one() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let library = temp.path().join("external-hf");
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    let requirements = vec![requirement("owner/model", "model.safetensors")];
    let ready = resolve_model_availability(&data, &library, &requirements, true, &[]);
    assert_eq!(ready.availability, ModelAvailability::ExternalReady);

    // Disconnected: the configured root is gone.
    let detached = temp.path().join("detached");
    std::fs::rename(&library, &detached).unwrap();
    let disconnected = resolve_model_availability(&data, &library, &requirements, true, &[]);
    assert_eq!(
        disconnected.availability,
        ModelAvailability::InstalledExternalUnavailable
    );
    assert!(
        !disconnected.library_present,
        "a disconnected drive is not present"
    );

    // Present but different: an unrelated directory now occupies the configured path.
    std::fs::create_dir_all(&library).unwrap();
    seed_snapshot(&library, "owner/model", "model.safetensors");
    let mismatched = resolve_model_availability(&data, &library, &requirements, true, &[]);
    assert_eq!(
        mismatched.availability,
        ModelAvailability::InstalledExternalUnavailable
    );
    assert!(
        mismatched.library_present,
        "a browsable library with the wrong identity must be distinguishable from a missing one"
    );
    assert!(
        ExternalLibraryUnavailableContext::from_resolution("m", None, &mismatched).library_present,
        "the typed context the prompt reads must carry the same distinction"
    );
}
