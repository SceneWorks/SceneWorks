use super::*;
use crate::model_artifacts::{
    ActiveArtifactLeaseRegistry, ArtifactAvailability, ArtifactCompleteness, ArtifactFile,
    ArtifactIdentity, ArtifactMemberRole, ArtifactProvenance, ArtifactSourceLibrary,
    ResolvedBundleClosure, ResolvedBundleMember, MODEL_ARTIFACT_CONTRACT_VERSION,
};
use std::collections::BTreeMap;
use std::process::Command;
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

const REVISION_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const REVISION_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn policy_from(values: &[(&str, &str)]) -> Result<ResolvedCachePolicy, ResolvedCacheError> {
    let values = values.iter().copied().collect::<BTreeMap<_, _>>();
    ResolvedCachePolicy::from_env_values(|name| values.get(name).map(|value| (*value).to_owned()))
}

fn source_candidate(root: &Path, revision: &str) -> PromotionCandidate {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(root.join("weights.bin"), b"model-weights").unwrap();
    let identity = ArtifactIdentity::pinned("owner/model", revision, "default").unwrap();
    let closure = ResolvedBundleClosure::new(vec![ResolvedBundleMember {
        role: ArtifactMemberRole::Primary,
        component_id: None,
        source: identity.clone(),
        tier: Some("fp16".to_owned()),
        source_subpath: PathBuf::new(),
        destination: PathBuf::new(),
        files: vec![ArtifactFile::new("weights.bin").unwrap()],
    }])
    .unwrap();
    let artifact = ResolvedModelArtifact {
        schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
        identity: identity.clone(),
        location: ArtifactLocation::SourceLibrary {
            root: root.to_path_buf(),
        },
        closure,
        provenance: ArtifactProvenance {
            identity,
            fixed_artifact_tier: Some("fp16".to_owned()),
        },
        completeness: ArtifactCompleteness::Complete,
        availability: ArtifactAvailability::Available,
    };
    artifact.validate().unwrap();
    PromotionCandidate {
        cache_key: artifact.cache_key().unwrap(),
        artifact,
    }
}

fn make_complete(
    store: &ResolvedCacheStore,
    candidate: &PromotionCandidate,
    source: &Path,
) -> ResolvedCacheMetadata {
    let reservation = match store.reserve(candidate, source, "fixture:model").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        ReservationOutcome::AlreadyComplete(_) => panic!("unexpected complete entry"),
        ReservationOutcome::Contended => panic!("unexpected reservation contention"),
    };
    let bundle = reservation.bundle_path().unwrap();
    std::fs::create_dir_all(&bundle).unwrap();
    std::fs::copy(source.join("weights.bin"), bundle.join("weights.bin")).unwrap();
    let mut local = candidate.artifact.clone();
    local.location = ArtifactLocation::ResolvedLocal { root: bundle };
    let metadata = reservation.record_complete(local).unwrap();
    assert!(!store
        .entry_path(&candidate.cache_key)
        .unwrap()
        .join(".sceneworks-download-complete.json")
        .exists());
    metadata
}

fn resolver(root: &Path, registry: ActiveArtifactLeaseRegistry) -> ModelArtifactResolver {
    ModelArtifactResolver::with_lease_registry(ArtifactSourceLibrary::new(root).unwrap(), registry)
}

#[cfg(windows)]
struct WindowsDirectoryJunction {
    path: PathBuf,
    removed: bool,
}

#[cfg(windows)]
impl WindowsDirectoryJunction {
    fn create(path: &Path, target: &Path) -> Self {
        let output = Command::new("cmd")
            .args(["/D", "/C", "mklink", "/J"])
            .arg(path)
            .arg(target)
            .output()
            .expect("launch cmd.exe to create directory junction");
        assert!(
            output.status.success(),
            "mklink /J failed with {}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata = std::fs::symlink_metadata(path)
            .expect("read the created directory junction without following it");
        assert!(
            metadata_is_reparse_point(&metadata),
            "mklink /J fixture must carry FILE_ATTRIBUTE_REPARSE_POINT"
        );
        Self {
            path: path.to_path_buf(),
            removed: false,
        }
    }

    fn remove(mut self) {
        std::fs::remove_dir(&self.path)
            .expect("remove the junction itself without traversing its target");
        self.removed = true;
    }
}

#[cfg(windows)]
impl Drop for WindowsDirectoryJunction {
    fn drop(&mut self) {
        if !self.removed {
            let _ = std::fs::remove_dir(&self.path);
        }
    }
}

#[test]
fn reservation_outcome_keeps_the_lock_holding_reservation_behind_indirection() {
    assert!(
        std::mem::size_of::<ReservationOutcome>() <= 4 * std::mem::size_of::<usize>(),
        "ReservationOutcome must not inline the lock-holding reservation"
    );
}

#[test]
fn policy_defaults_are_finite_disabled_and_serde_defaults_upgrade() {
    let policy = ResolvedCachePolicy::default();
    assert!(!policy.enabled);
    assert_eq!(policy.max_bytes, 68_719_476_736);
    assert_eq!(policy.inactivity_seconds, 1_209_600);
    policy.validate().unwrap();

    #[derive(Default, Deserialize)]
    #[serde(default)]
    struct LegacySettings {
        resolved_cache: ResolvedCachePolicy,
    }
    let upgraded: LegacySettings = serde_json::from_str("{}").unwrap();
    assert_eq!(upgraded.resolved_cache, policy);
    let future: ResolvedCachePolicy = serde_json::from_str(
        r#"{"enabled":false,"maxBytes":123,"inactivitySeconds":456,"future":true}"#,
    )
    .unwrap();
    assert_eq!(future.max_bytes, 123);
}

#[test]
fn env_policy_is_exact_and_invalid_values_fail_closed() {
    assert_eq!(policy_from(&[]).unwrap(), ResolvedCachePolicy::default());
    let enabled = policy_from(&[
        (RESOLVED_CACHE_ENABLED_ENV, "true"),
        (RESOLVED_CACHE_MAX_BYTES_ENV, "4096"),
        (RESOLVED_CACHE_INACTIVITY_SECONDS_ENV, "60"),
    ])
    .unwrap();
    assert!(enabled.enabled);
    assert_eq!(
        enabled.env_pairs().unwrap(),
        [
            (RESOLVED_CACHE_ENABLED_ENV, "true".to_owned()),
            (RESOLVED_CACHE_MAX_BYTES_ENV, "4096".to_owned()),
            (RESOLVED_CACHE_INACTIVITY_SECONDS_ENV, "60".to_owned()),
        ]
    );
    for (name, value) in [
        (RESOLVED_CACHE_ENABLED_ENV, "sometimes"),
        (RESOLVED_CACHE_MAX_BYTES_ENV, "0"),
        (RESOLVED_CACHE_MAX_BYTES_ENV, "unlimited"),
        (RESOLVED_CACHE_INACTIVITY_SECONDS_ENV, "0"),
    ] {
        assert!(policy_from(&[(name, value)]).is_err(), "{name}={value}");
    }
}

#[test]
fn store_refuses_unmanaged_roots_and_hashes_all_entry_paths() {
    let scratch = TempDir::new().unwrap();
    let data = scratch.path().join("data with spaces 雪");
    let unmanaged = data.join("models/resolved");
    std::fs::create_dir_all(&unmanaged).unwrap();
    std::fs::write(unmanaged.join("keep.bin"), b"keep").unwrap();
    let error = ResolvedCacheStore::open(&data).unwrap_err();
    assert!(error.to_string().contains("refusing to adopt unmanaged"));
    assert_eq!(std::fs::read(unmanaged.join("keep.bin")).unwrap(), b"keep");

    let managed_data = scratch.path().join("managed 雪");
    let store = ResolvedCacheStore::open(&managed_data).unwrap();
    let key = format!("sha256:{}", "a".repeat(64));
    let entry = store.entry_path(&key).unwrap();
    assert!(entry.starts_with(store.root()));
    assert_eq!(entry.file_name().unwrap(), "a".repeat(64).as_str());
    for hostile in ["../escape", "sha256:../escape", "sha256:ABC", "C:\\weights"] {
        assert!(store.entry_path(hostile).is_err());
    }
    for neighbor in [
        "receipts",
        "mlx",
        "converted",
        "imported",
        "trained",
        "user",
    ] {
        assert!(!managed_data.join("models").join(neighbor).exists());
    }
}

#[cfg(unix)]
#[test]
fn store_refuses_symlink_root_without_mutating_target() {
    use std::os::unix::fs::symlink;
    let scratch = TempDir::new().unwrap();
    let target = scratch.path().join("outside");
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("keep.bin"), b"keep").unwrap();
    let data = scratch.path().join("data");
    std::fs::create_dir_all(data.join("models")).unwrap();
    symlink(&target, data.join("models/resolved")).unwrap();
    assert!(ResolvedCacheStore::open(&data).is_err());
    assert_eq!(std::fs::read(target.join("keep.bin")).unwrap(), b"keep");

    let data_with_linked_models = scratch.path().join("data-linked-models");
    std::fs::create_dir(&data_with_linked_models).unwrap();
    symlink(&target, data_with_linked_models.join("models")).unwrap();
    assert!(ResolvedCacheStore::open(&data_with_linked_models).is_err());
    assert!(!target.join("resolved").exists());
}

#[test]
fn reservation_is_exclusive_unrelated_keys_progress_and_drop_interrupts() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source_a = scratch.path().join("source-a");
    let source_b = scratch.path().join("source-b");
    let candidate_a = source_candidate(&source_a, REVISION_A);
    let candidate_b = source_candidate(&source_b, REVISION_B);
    let mut first = match store
        .reserve(&candidate_a, &source_a, "image:model-a")
        .unwrap()
    {
        ReservationOutcome::Acquired(value) => value,
        ReservationOutcome::AlreadyComplete(_) => panic!("first cannot already be complete"),
        ReservationOutcome::Contended => panic!("first must win"),
    };
    let active = store
        .enumerate()
        .unwrap()
        .into_iter()
        .find(|entry| entry.cache_key == candidate_a.cache_key)
        .unwrap()
        .metadata
        .unwrap();
    assert_eq!(active.reservation_owner.as_deref(), Some("image:model-a"));
    let mut traversing = active.clone();
    traversing.session_id = Some("../../outside".to_owned());
    assert!(validate_metadata_shape(
        &traversing,
        &cache_key_digest(&candidate_a.cache_key).unwrap()
    )
    .is_err());
    assert!(matches!(
        store
            .reserve(&candidate_a, &source_a, "image:model-a")
            .unwrap(),
        ReservationOutcome::Contended
    ));
    first.artifact_lock.take();
    assert!(matches!(
        store
            .reserve(&candidate_a, &source_a, "image:model-a")
            .unwrap(),
        ReservationOutcome::Contended
    ));
    let unrelated = store
        .reserve(&candidate_b, &source_b, "video:model-b")
        .unwrap();
    assert!(matches!(unrelated, ReservationOutcome::Acquired(_)));
    drop(unrelated);
    drop(first);
    let metadata = store
        .enumerate()
        .unwrap()
        .into_iter()
        .find(|entry| entry.cache_key == candidate_a.cache_key)
        .unwrap()
        .metadata
        .unwrap();
    assert_eq!(metadata.state, ResolvedCacheEntryState::Interrupted);
    assert!(store
        .lookup_complete(&candidate_a.cache_key)
        .unwrap()
        .is_none());
    assert!(store.reserve(&candidate_a, &source_a, "\n").is_err());
}

#[test]
fn complete_entries_are_preserved_and_reservations_require_exact_ownership() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    let candidate = source_candidate(&source, REVISION_A);
    let complete = make_complete(&store, &candidate, &source);
    let bundle_file = store
        .bundle_path(&candidate.cache_key)
        .unwrap()
        .join("weights.bin");
    let bytes = std::fs::read(&bundle_file).unwrap();
    match store.reserve(&candidate, &source, "image:model-a").unwrap() {
        ReservationOutcome::AlreadyComplete(metadata) => assert_eq!(*metadata, complete),
        ReservationOutcome::Acquired(_) => panic!("complete entry was overwritten"),
        ReservationOutcome::Contended => panic!("complete entry was treated as materializing"),
    }
    assert_eq!(std::fs::read(&bundle_file).unwrap(), bytes);

    let source_b = scratch.path().join("source-b");
    let candidate_b = source_candidate(&source_b, REVISION_B);
    let mut reservation = match store
        .reserve(&candidate_b, &source_b, "video:model-b")
        .unwrap()
    {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("new entry must reserve"),
    };
    let genuine_id = reservation.reservation_id.clone();
    reservation.reservation_id = "cccccccccccccccccccccccccccccccc".to_owned();
    assert!(reservation.mark_interrupted().is_err());
    assert_eq!(
        store
            .enumerate()
            .unwrap()
            .into_iter()
            .find(|entry| entry.cache_key == candidate_b.cache_key)
            .unwrap()
            .state,
        ResolvedCacheEntryState::Materializing
    );
    reservation.reservation_id = genuine_id;
    assert_eq!(
        reservation.mark_interrupted().unwrap().state,
        ResolvedCacheEntryState::Interrupted
    );
}

#[test]
fn one_reservation_cannot_interrupt_or_complete_another_owner() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    let candidate = source_candidate(&source, REVISION_A);
    let mut reservation = match store.reserve(&candidate, &source, "image:model-a").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("new entry must reserve"),
    };
    let original_id = reservation.reservation_id.clone();
    let original_owner = reservation.reservation_owner.clone();
    store
        .update_metadata(&candidate.cache_key, |metadata| {
            metadata.reservation_id = Some("dddddddddddddddddddddddddddddddd".to_owned());
            metadata.reservation_owner = Some("other:model".to_owned());
            Ok(())
        })
        .unwrap();
    assert!(reservation.mark_interrupted().is_err());
    let current = store.enumerate().unwrap()[0].metadata.clone().unwrap();
    assert_eq!(
        current.reservation_id.as_deref(),
        Some("dddddddddddddddddddddddddddddddd")
    );
    assert_eq!(current.reservation_owner.as_deref(), Some("other:model"));

    store
        .update_metadata(&candidate.cache_key, |metadata| {
            metadata.reservation_id = Some(original_id.clone());
            metadata.reservation_owner = Some(original_owner.clone());
            Ok(())
        })
        .unwrap();
    reservation.mark_interrupted().unwrap();

    let source_b = scratch.path().join("source-b");
    let candidate_b = source_candidate(&source_b, REVISION_B);
    let reservation_b = match store
        .reserve(&candidate_b, &source_b, "video:model-b")
        .unwrap()
    {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("second entry must reserve"),
    };
    let bundle = reservation_b.bundle_path().unwrap();
    std::fs::create_dir(&bundle).unwrap();
    std::fs::copy(source_b.join("weights.bin"), bundle.join("weights.bin")).unwrap();
    store
        .update_metadata(&candidate_b.cache_key, |metadata| {
            metadata.reservation_owner = Some("other:model".to_owned());
            Ok(())
        })
        .unwrap();
    let mut local = candidate_b.artifact.clone();
    local.location = ArtifactLocation::ResolvedLocal { root: bundle };
    assert!(reservation_b.record_complete(local).is_err());
    let current_b = store
        .enumerate()
        .unwrap()
        .into_iter()
        .find(|entry| entry.cache_key == candidate_b.cache_key)
        .unwrap()
        .metadata
        .unwrap();
    assert_eq!(current_b.state, ResolvedCacheEntryState::Materializing);
    assert_eq!(current_b.reservation_owner.as_deref(), Some("other:model"));
}

#[cfg(unix)]
#[test]
fn completion_rejects_a_bundle_symlink_without_touching_external_bytes() {
    use std::os::unix::fs::symlink;
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    let candidate = source_candidate(&source, REVISION_A);
    let reservation = match store.reserve(&candidate, &source, "image:model-a").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("new entry must reserve"),
    };
    let external = scratch.path().join("external");
    std::fs::create_dir(&external).unwrap();
    std::fs::write(external.join("weights.bin"), b"external-model-bytes").unwrap();
    let bundle = reservation.bundle_path().unwrap();
    symlink(&external, &bundle).unwrap();
    let mut local = candidate.artifact.clone();
    local.location = ArtifactLocation::ResolvedLocal { root: bundle };
    assert!(reservation.record_complete(local).is_err());
    assert_eq!(
        std::fs::read(external.join("weights.bin")).unwrap(),
        b"external-model-bytes"
    );
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_none());
}

#[cfg(unix)]
#[test]
fn complete_validation_rejects_a_post_publication_bundle_swap_everywhere() {
    use std::os::unix::fs::symlink;
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    let candidate = source_candidate(&source, REVISION_A);
    make_complete(&store, &candidate, &source);
    let bundle = store.bundle_path(&candidate.cache_key).unwrap();
    std::fs::remove_dir_all(&bundle).unwrap();
    let external = scratch.path().join("external");
    std::fs::create_dir(&external).unwrap();
    std::fs::write(external.join("weights.bin"), b"model-weights").unwrap();
    std::fs::write(external.join("sentinel"), b"external-must-survive").unwrap();
    symlink(&external, &bundle).unwrap();

    assert!(store.lookup_complete(&candidate.cache_key).is_err());
    let resolver = resolver(&source, ActiveArtifactLeaseRegistry::default());
    assert!(store
        .acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
        .is_err());
    assert!(store.enumerate().is_err());
    let digest = cache_key_digest(&candidate.cache_key).unwrap();
    let _lock = store.lock_metadata(&digest).unwrap();
    assert_eq!(
        store.read_metadata_locked(&digest).unwrap().last_used_at,
        None
    );
    drop(_lock);

    let entry = store.entry_path(&candidate.cache_key).unwrap();
    for slot in 0..=1 {
        std::fs::write(entry.join(format!("metadata.{slot}.json")), b"corrupt").unwrap();
    }
    let recovered = store.recover().unwrap();
    assert_eq!(recovered[0].state, ResolvedCacheEntryState::Corrupt);
    assert!(recovered[0].metadata.is_none());
    assert_eq!(
        std::fs::read(external.join("weights.bin")).unwrap(),
        b"model-weights"
    );
    assert_eq!(
        std::fs::read(external.join("sentinel")).unwrap(),
        b"external-must-survive"
    );
}

#[cfg(windows)]
#[test]
fn completion_rejects_a_directory_junction_without_touching_external_bytes() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    let candidate = source_candidate(&source, REVISION_A);
    let reservation = match store.reserve(&candidate, &source, "image:model-a").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("new entry must reserve"),
    };
    let external = scratch.path().join("external");
    std::fs::create_dir(&external).unwrap();
    std::fs::write(external.join("weights.bin"), b"external-model-bytes").unwrap();
    std::fs::write(external.join("sentinel"), b"external-must-survive").unwrap();
    let bundle = reservation.bundle_path().unwrap();
    let junction = WindowsDirectoryJunction::create(&bundle, &external);
    let mut local = candidate.artifact.clone();
    local.location = ArtifactLocation::ResolvedLocal { root: bundle };
    assert!(reservation.record_complete(local).is_err());
    assert_eq!(
        std::fs::read(external.join("weights.bin")).unwrap(),
        b"external-model-bytes"
    );
    assert_eq!(
        std::fs::read(external.join("sentinel")).unwrap(),
        b"external-must-survive"
    );
    junction.remove();
    assert!(external.is_dir());
}

#[cfg(windows)]
#[test]
fn complete_validation_rejects_a_post_publication_directory_junction_everywhere() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    let candidate = source_candidate(&source, REVISION_A);
    make_complete(&store, &candidate, &source);
    let bundle = store.bundle_path(&candidate.cache_key).unwrap();
    std::fs::remove_dir_all(&bundle).unwrap();
    let external = scratch.path().join("external");
    std::fs::create_dir(&external).unwrap();
    std::fs::write(external.join("weights.bin"), b"model-weights").unwrap();
    std::fs::write(external.join("sentinel"), b"external-must-survive").unwrap();
    let junction = WindowsDirectoryJunction::create(&bundle, &external);
    assert!(store.lookup_complete(&candidate.cache_key).is_err());
    let resolver = resolver(&source, ActiveArtifactLeaseRegistry::default());
    assert!(store
        .acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
        .is_err());
    assert!(store.enumerate().is_err());
    let digest = cache_key_digest(&candidate.cache_key).unwrap();
    let metadata_lock = store.lock_metadata(&digest).unwrap();
    assert_eq!(
        store.read_metadata_locked(&digest).unwrap().last_used_at,
        None
    );
    drop(metadata_lock);

    let entry = store.entry_path(&candidate.cache_key).unwrap();
    for slot in 0..=1 {
        std::fs::write(entry.join(format!("metadata.{slot}.json")), b"corrupt").unwrap();
    }
    let recovered = store.recover().unwrap();
    assert_eq!(recovered[0].state, ResolvedCacheEntryState::Corrupt);
    assert!(recovered[0].metadata.is_none());
    assert_eq!(
        std::fs::read(external.join("weights.bin")).unwrap(),
        b"model-weights"
    );
    assert_eq!(
        std::fs::read(external.join("sentinel")).unwrap(),
        b"external-must-survive"
    );
    junction.remove();
    assert!(external.is_dir());
}

#[test]
fn catalog_reads_do_not_stamp_usage_but_runtime_acquisition_does() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    let candidate = source_candidate(&source, REVISION_A);
    make_complete(&store, &candidate, &source);
    assert_eq!(
        store
            .lookup_complete(&candidate.cache_key)
            .unwrap()
            .unwrap()
            .last_used_at,
        None
    );
    assert_eq!(
        store.enumerate().unwrap()[0]
            .metadata
            .as_ref()
            .unwrap()
            .last_used_at,
        None
    );
    let registry = ActiveArtifactLeaseRegistry::default();
    let resolver = resolver(&source, registry.clone());
    let lease = store
        .acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
        .unwrap()
        .unwrap();
    assert_eq!(registry.active_lease_count(&candidate.cache_key), 1);
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .unwrap()
        .last_used_at
        .is_some());
    assert_eq!(lease.mark_success().cache_key, candidate.cache_key);
    assert_eq!(registry.active_lease_count(&candidate.cache_key), 0);
}

#[test]
fn pin_owners_aggregate_exactly() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    let candidate = source_candidate(&source, REVISION_A);
    make_complete(&store, &candidate, &source);
    assert!(!store.effective_pin(&candidate.cache_key).unwrap());
    store
        .set_model_pin(&candidate.cache_key, "image:model-a", true)
        .unwrap();
    store
        .set_model_pin(&candidate.cache_key, "video:model-b", true)
        .unwrap();
    let metadata = store
        .set_model_pin(&candidate.cache_key, "image:model-a", false)
        .unwrap();
    assert_eq!(metadata.model_pin_owners.len(), 1);
    store
        .set_model_pin(&candidate.cache_key, "video:model-b", false)
        .unwrap();
    assert!(!store.effective_pin(&candidate.cache_key).unwrap());
    store.set_artifact_pin(&candidate.cache_key, true).unwrap();
    assert!(store.effective_pin(&candidate.cache_key).unwrap());
    assert!(store
        .set_model_pin(&candidate.cache_key, "\n", true)
        .is_err());
}

#[test]
fn newest_corruption_recovers_older_and_both_slots_require_valid_receipt() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    let candidate = source_candidate(&source, REVISION_A);
    make_complete(&store, &candidate, &source);
    store.set_artifact_pin(&candidate.cache_key, false).unwrap();
    let entry = store.entry_path(&candidate.cache_key).unwrap();
    let newest = [entry.join("metadata.0.json"), entry.join("metadata.1.json")]
        .into_iter()
        .max_by_key(|path| {
            read_journal(path)
                .map(|value| value.generation)
                .unwrap_or(0)
        })
        .unwrap();
    std::fs::write(newest, b"corrupt").unwrap();
    let recovered = store.recover().unwrap();
    let metadata = recovered[0].metadata.as_ref().unwrap();
    assert_eq!(
        metadata.recovery_status,
        RecoveryStatus::RecoveredFromOlderSlot
    );
    assert!(metadata.effective_pin);

    for slot in 0..=1 {
        std::fs::write(entry.join(format!("metadata.{slot}.json")), b"corrupt").unwrap();
    }
    let recovered = store.recover().unwrap();
    assert_eq!(
        recovered[0].metadata.as_ref().unwrap().recovery_status,
        RecoveryStatus::ReconstructedFromCompleteReceipt
    );
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_some());
    assert!(store
        .bundle_path(&candidate.cache_key)
        .unwrap()
        .join("weights.bin")
        .is_file());

    for slot in 0..=1 {
        std::fs::write(
            entry.join(format!("metadata.{slot}.json")),
            b"corrupt-again",
        )
        .unwrap();
    }
    std::fs::write(entry.join("complete.receipt.json"), b"bad-receipt").unwrap();
    let recovered = store.recover().unwrap();
    assert_eq!(recovered[0].state, ResolvedCacheEntryState::Corrupt);
    assert!(recovered[0].metadata.is_none());
    assert!(entry.join("corrupt.marker.json").is_file());
    assert!(store
        .bundle_path(&candidate.cache_key)
        .unwrap()
        .join("weights.bin")
        .is_file());
}

#[test]
fn incomplete_complete_entry_is_never_available_or_reconstructed() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    let candidate = source_candidate(&source, REVISION_A);
    make_complete(&store, &candidate, &source);
    let bundle_file = store
        .bundle_path(&candidate.cache_key)
        .unwrap()
        .join("weights.bin");
    std::fs::write(&bundle_file, b"short").unwrap();
    assert!(store.lookup_complete(&candidate.cache_key).is_err());

    let entry = store.entry_path(&candidate.cache_key).unwrap();
    for slot in 0..=1 {
        std::fs::write(entry.join(format!("metadata.{slot}.json")), b"corrupt").unwrap();
    }
    let recovered = store.recover().unwrap();
    assert_eq!(recovered[0].state, ResolvedCacheEntryState::Corrupt);
    assert!(recovered[0].metadata.is_none());
    assert_eq!(std::fs::read(bundle_file).unwrap(), b"short");
}

#[test]
fn checked_enumeration_rejects_overflow() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source_a = scratch.path().join("source-a");
    let source_b = scratch.path().join("source-b");
    let candidate_a = source_candidate(&source_a, REVISION_A);
    let candidate_b = source_candidate(&source_b, REVISION_B);
    make_complete(&store, &candidate_a, &source_a);
    make_complete(&store, &candidate_b, &source_b);
    for (candidate, bytes) in [(&candidate_a, u64::MAX), (&candidate_b, 1)] {
        let digest = cache_key_digest(&candidate.cache_key).unwrap();
        let _lock = store.lock_metadata(&digest).unwrap();
        let mut metadata = store.read_metadata_locked(&digest).unwrap();
        metadata.verified_bytes = bytes;
        store.write_metadata_unlocked(&digest, &metadata).unwrap();
    }
    assert!(store.checked_verified_bytes().is_err());
}

#[test]
fn volume_probe_is_deterministic_and_missing_source_is_unavailable() {
    assert_eq!(
        relation_for_volume_identities(Some(7), Some(7), true),
        SourceVolumeRelation::Same
    );
    assert_eq!(
        relation_for_volume_identities(Some(7), Some(8), true),
        SourceVolumeRelation::Different
    );
    assert_eq!(
        relation_for_volume_identities(None, Some(8), true),
        SourceVolumeRelation::Unknown
    );
    assert_eq!(
        relation_for_volume_identities(Some(7), Some(7), false),
        SourceVolumeRelation::Unavailable
    );
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    assert_eq!(
        store
            .compare_source_volume(scratch.path())
            .unwrap()
            .relation,
        SourceVolumeRelation::Same
    );
    let missing = store
        .compare_source_volume(&scratch.path().join("missing/source"))
        .unwrap();
    assert_eq!(missing.relation, SourceVolumeRelation::Unavailable);
    assert!(missing.source_identity.is_none());
}

#[test]
fn stale_sessions_are_removed_only_after_their_lock_is_acquirable() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let own_records = store.root().join("sessions").join(store.session_id());
    std::fs::write(own_records.join("keep.json"), b"live").unwrap();
    let unrelated_entry = store.root().join("keep-unrelated");
    std::fs::write(&unrelated_entry, b"unrelated").unwrap();
    for malformed in [
        ".lock",
        "..lock",
        "...lock",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.lock",
        "short.lock",
        "deadbeefdeadbeefdeadbeefdeadbeef.lock.extra",
    ] {
        std::fs::write(store.root().join("sessions").join(malformed), b"").unwrap();
    }
    assert!(!is_valid_session_id(""));
    assert!(!is_valid_session_id(".."));
    assert!(!is_valid_session_id("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"));
    assert!(is_valid_session_id("deadbeefdeadbeefdeadbeefdeadbeef"));
    let stale = "deadbeefdeadbeefdeadbeefdeadbeef";
    let lock = store.root().join("sessions").join(format!("{stale}.lock"));
    std::fs::write(&lock, b"").unwrap();
    let records = store.root().join("sessions").join(stale);
    std::fs::create_dir(&records).unwrap();
    std::fs::write(records.join("malformed.json"), b"garbage").unwrap();
    store.recover().unwrap();
    assert!(!records.exists());
    assert!(!lock.exists());

    let live = "feedfacefeedfacefeedfacefeedface";
    let live_lock_path = store.root().join("sessions").join(format!("{live}.lock"));
    let live_lock = open_lock_file(&live_lock_path).unwrap();
    FileExt::lock_exclusive(&live_lock).unwrap();
    let live_records = store.root().join("sessions").join(live);
    std::fs::create_dir(&live_records).unwrap();
    std::fs::write(live_records.join("keep.json"), b"garbage").unwrap();
    store.recover().unwrap();
    assert!(live_records.exists());
    assert!(live_lock_path.exists());
    assert_eq!(
        std::fs::read(own_records.join("keep.json")).unwrap(),
        b"live"
    );
    assert_eq!(std::fs::read(unrelated_entry).unwrap(), b"unrelated");
    for malformed in [
        ".lock",
        "..lock",
        "...lock",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.lock",
        "short.lock",
        "deadbeefdeadbeefdeadbeefdeadbeef.lock.extra",
    ] {
        assert!(store.root().join("sessions").join(malformed).exists());
    }
}

#[test]
fn lock_contention_classifier_covers_fs2_and_would_block_without_masking_other_errors() {
    let fs2_error = fs2::lock_contended_error();
    assert!(is_lock_contended(&fs2_error));
    assert!(is_lock_contended(&std::io::Error::from(
        std::io::ErrorKind::WouldBlock
    )));
    assert!(!is_lock_contended(&std::io::Error::from(
        std::io::ErrorKind::PermissionDenied
    )));
}

#[cfg(windows)]
#[test]
fn windows_runtime_file_lock_contention_uses_the_portable_classifier() {
    let scratch = TempDir::new().unwrap();
    let path = scratch.path().join("contention.lock");
    let first = open_lock_file(&path).unwrap();
    let second = open_lock_file(&path).unwrap();
    FileExt::lock_exclusive(&first).unwrap();
    let error = FileExt::try_lock_exclusive(&second).unwrap_err();
    assert!(is_lock_contended(&error));
}

#[cfg(windows)]
#[test]
fn windows_directory_volume_probe_returns_a_native_same_volume_identity() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let observation = store.compare_source_volume(scratch.path()).unwrap();
    assert_eq!(observation.relation, SourceVolumeRelation::Same);
    assert!(observation.source_identity.is_some());
    assert_eq!(
        observation.source_identity, observation.resolved_identity,
        "source and resolved directory handles must report the same volume serial"
    );
}

#[test]
fn cross_process_same_key_reservation_shared_lease_and_kill_recovery() {
    let scratch = TempDir::new().unwrap();
    let data = scratch.path().join("data");
    let source_a = scratch.path().join("source-a");
    let source_b = scratch.path().join("source-b");
    let store = ResolvedCacheStore::open(&data).unwrap();
    let candidate_a = source_candidate(&source_a, REVISION_A);
    let candidate_b = source_candidate(&source_b, REVISION_B);
    let mut child = spawn_child(&data, &source_a, scratch.path(), "reserve");
    wait_for(&scratch.path().join("reserve-ready"));
    assert!(matches!(
        store
            .reserve(&candidate_a, &source_a, "image:model-a")
            .unwrap(),
        ReservationOutcome::Contended
    ));
    assert!(matches!(
        store
            .reserve(&candidate_b, &source_b, "video:model-b")
            .unwrap(),
        ReservationOutcome::Acquired(_)
    ));
    child.kill().unwrap();
    child.wait().unwrap();
    store.recover().unwrap();
    assert_eq!(
        store
            .enumerate()
            .unwrap()
            .into_iter()
            .find(|entry| entry.cache_key == candidate_a.cache_key)
            .unwrap()
            .state,
        ResolvedCacheEntryState::Interrupted
    );

    make_complete(&store, &candidate_a, &source_a);
    let mut child = spawn_child(&data, &source_a, scratch.path(), "lease");
    wait_for(&scratch.path().join("lease-ready"));
    assert!(matches!(
        store
            .reserve(&candidate_a, &source_a, "image:model-a")
            .unwrap(),
        ReservationOutcome::Contended
    ));
    child.kill().unwrap();
    child.wait().unwrap();
    store.recover().unwrap();
    assert!(store
        .lookup_complete(&candidate_a.cache_key)
        .unwrap()
        .is_some());
}

fn spawn_child(data: &Path, source: &Path, scratch: &Path, mode: &str) -> std::process::Child {
    Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("model_artifacts::resolved_cache::tests::cross_process_store_child")
        .arg("--nocapture")
        .env("SCENEWORKS_CACHE_CHILD_MODE", mode)
        .env("SCENEWORKS_CACHE_CHILD_DATA", data)
        .env("SCENEWORKS_CACHE_CHILD_SOURCE", source)
        .env(
            "SCENEWORKS_CACHE_CHILD_READY",
            scratch.join(format!("{mode}-ready")),
        )
        .env(
            "SCENEWORKS_CACHE_CHILD_RELEASE",
            scratch.join(format!("{mode}-release")),
        )
        .spawn()
        .unwrap()
}

fn wait_for(path: &Path) {
    for _ in 0..200 {
        if path.exists() {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("child did not create {}", path.display());
}

#[test]
fn cross_process_store_child() {
    let Ok(mode) = std::env::var("SCENEWORKS_CACHE_CHILD_MODE") else {
        return;
    };
    let data = PathBuf::from(std::env::var_os("SCENEWORKS_CACHE_CHILD_DATA").unwrap());
    let source = PathBuf::from(std::env::var_os("SCENEWORKS_CACHE_CHILD_SOURCE").unwrap());
    let ready = PathBuf::from(std::env::var_os("SCENEWORKS_CACHE_CHILD_READY").unwrap());
    let release = PathBuf::from(std::env::var_os("SCENEWORKS_CACHE_CHILD_RELEASE").unwrap());
    let store = ResolvedCacheStore::open(&data).unwrap();
    let candidate = source_candidate(&source, REVISION_A);
    let _held: Box<dyn std::any::Any> = if mode == "reserve" {
        match store.reserve(&candidate, &source, "child:model").unwrap() {
            ReservationOutcome::Acquired(reservation) => reservation,
            ReservationOutcome::AlreadyComplete(_) => panic!("child entry is already complete"),
            ReservationOutcome::Contended => panic!("child reservation contended"),
        }
    } else {
        let resolver = resolver(&source, ActiveArtifactLeaseRegistry::default());
        Box::new(
            store
                .acquire_complete(&candidate.cache_key, &resolver, "child:model")
                .unwrap()
                .expect("complete child artifact"),
        )
    };
    std::fs::write(ready, b"ready").unwrap();
    while !release.exists() {
        thread::sleep(Duration::from_millis(20));
    }
}
