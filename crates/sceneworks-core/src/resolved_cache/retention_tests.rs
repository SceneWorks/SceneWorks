use super::*;
use crate::model_artifacts::{
    ActiveArtifactLeaseRegistry, ArtifactAvailability, ArtifactCompleteness, ArtifactFile,
    ArtifactIdentity, ArtifactMemberRole, ArtifactProvenance, ArtifactSourceLibrary,
    ModelArtifactResolver, ResolvedBundleClosure, ResolvedBundleMember,
    MODEL_ARTIFACT_CONTRACT_VERSION,
};
use tempfile::TempDir;

const REV_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const REV_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const REV_C: &str = "cccccccccccccccccccccccccccccccccccccccc";

/// A fake-clock instant far beyond any real timestamp the fixtures can produce, so leases that
/// stamp real usage times still look expired against `FAR_FUTURE`.
const FAR_FUTURE: u64 = 5_000_000_000;

fn identity(repository: &str, revision: &str) -> ArtifactIdentity {
    ArtifactIdentity::pinned(repository, revision, "default").unwrap()
}

fn snapshot(library: &Path, repository: &str, revision: &str) -> PathBuf {
    let snapshot = ArtifactSourceLibrary::new(library)
        .unwrap()
        .repository_root(repository)
        .unwrap()
        .join("snapshots")
        .join(revision);
    std::fs::create_dir_all(&snapshot).unwrap();
    snapshot
}

fn member(
    role: ArtifactMemberRole,
    component_id: Option<&str>,
    source: ArtifactIdentity,
    tier: Option<&str>,
    destination: &str,
    files: &[&str],
) -> ResolvedBundleMember {
    ResolvedBundleMember {
        role,
        component_id: component_id.map(str::to_owned),
        source,
        tier: tier.map(str::to_owned),
        source_subpath: PathBuf::new(),
        destination: PathBuf::from(destination),
        files: files
            .iter()
            .map(|file| ArtifactFile::new(file).unwrap())
            .collect(),
    }
}

fn candidate(
    primary_snapshot: PathBuf,
    primary_identity: ArtifactIdentity,
    tier: &str,
    members: Vec<ResolvedBundleMember>,
) -> PromotionCandidate {
    let closure = ResolvedBundleClosure::new(members).unwrap();
    let artifact = ResolvedModelArtifact {
        schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
        identity: primary_identity.clone(),
        location: ArtifactLocation::SourceLibrary {
            root: primary_snapshot,
        },
        closure,
        provenance: ArtifactProvenance {
            identity: primary_identity,
            fixed_artifact_tier: Some(tier.to_owned()),
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

/// Single-member candidate whose primary weights file carries `bytes`.
fn flat_candidate(
    library: &Path,
    repository: &str,
    revision: &str,
    tier: &str,
    bytes: &[u8],
) -> PromotionCandidate {
    let primary_identity = identity(repository, revision);
    let primary_snapshot = snapshot(library, repository, revision);
    std::fs::write(primary_snapshot.join("model.safetensors"), bytes).unwrap();
    candidate(
        primary_snapshot,
        primary_identity.clone(),
        tier,
        vec![member(
            ArtifactMemberRole::Primary,
            None,
            primary_identity,
            Some(tier),
            "",
            &["model.safetensors"],
        )],
    )
}

/// Candidate whose closure also carries a shared component from another repository.
fn shared_component_candidate(
    library: &Path,
    repository: &str,
    revision: &str,
    tier: &str,
    bytes: &[u8],
) -> PromotionCandidate {
    let primary_identity = identity(repository, revision);
    let primary_snapshot = snapshot(library, repository, revision);
    std::fs::write(primary_snapshot.join("model.safetensors"), bytes).unwrap();
    let component_identity = identity("SceneWorks/component-c", REV_C);
    let component_snapshot = snapshot(library, "SceneWorks/component-c", REV_C);
    let component_file = component_snapshot.join("te.safetensors");
    if !component_file.exists() {
        std::fs::write(&component_file, b"shared-text-encoder").unwrap();
    }
    candidate(
        primary_snapshot,
        primary_identity.clone(),
        tier,
        vec![
            member(
                ArtifactMemberRole::Primary,
                None,
                primary_identity,
                Some(tier),
                "",
                &["model.safetensors"],
            ),
            member(
                ArtifactMemberRole::RequiredComponent,
                Some("te"),
                component_identity,
                Some(tier),
                "te",
                &["te.safetensors"],
            ),
        ],
    )
}

/// Materializes through the real copy/verify/publish pipeline so the stored closure carries
/// enriched sizes and sha256 hashes — the pre-eviction source verification depends on them.
fn materialize_complete(
    store: &ResolvedCacheStore,
    library: &Path,
    candidate: &PromotionCandidate,
) -> ResolvedCacheMetadata {
    let materializer = ResolvedCacheMaterializer::new(store.clone());
    match materializer
        .materialize(
            candidate,
            library,
            "fixture:model",
            &MaterializationCancellation::default(),
        )
        .unwrap()
    {
        MaterializationOutcome::Published(metadata) => *metadata,
        other => panic!("fixture materialization must publish, got {other:?}"),
    }
}

fn stamp_activity(
    store: &ResolvedCacheStore,
    cache_key: &str,
    last_used_at: Option<u64>,
    created_at: u64,
) {
    store
        .update_metadata(cache_key, |metadata| {
            metadata.last_used_at = last_used_at;
            metadata.created_at = created_at;
            Ok(())
        })
        .unwrap();
}

fn policy(max_bytes: u64, inactivity_seconds: u64) -> ResolvedCachePolicy {
    ResolvedCachePolicy {
        enabled: true,
        max_bytes,
        inactivity_seconds,
    }
}

fn resolver(library: &Path, registry: ActiveArtifactLeaseRegistry) -> ModelArtifactResolver {
    ModelArtifactResolver::with_lease_registry(
        ArtifactSourceLibrary::new(library).unwrap(),
        registry,
    )
}

fn entry_dir(store: &ResolvedCacheStore, cache_key: &str) -> PathBuf {
    store.entry_path(cache_key).unwrap()
}

fn audit_records(store: &ResolvedCacheStore) -> Vec<PathBuf> {
    let audit = store.root().join(AUDIT_DIR);
    if !audit.exists() {
        return Vec::new();
    }
    let mut records = std::fs::read_dir(audit)
        .unwrap()
        .map(|item| item.unwrap().path())
        .collect::<Vec<_>>();
    records.sort();
    records
}

fn retained_hold(report: &RetentionReport, cache_key: &str) -> RetentionHold {
    report
        .retained
        .iter()
        .find(|record| record.cache_key == cache_key)
        .unwrap_or_else(|| panic!("expected a retained record for {cache_key}"))
        .hold
        .clone()
}

#[test]
fn lru_size_eviction_and_ttl_are_deterministic_under_a_fake_clock() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate_a = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    let candidate_b = flat_candidate(&library, "SceneWorks/m-b", REV_A, "q8", b"0123456789");
    let candidate_c = flat_candidate(&library, "SceneWorks/m-c", REV_A, "q8", b"0123456789");
    for candidate in [&candidate_a, &candidate_b, &candidate_c] {
        materialize_complete(&store, &library, candidate);
    }
    stamp_activity(&store, &candidate_a.cache_key, Some(1_000), 500);
    stamp_activity(&store, &candidate_b.cache_key, Some(2_000), 500);
    stamp_activity(&store, &candidate_c.cache_key, Some(3_000), 500);

    // Size pressure: 30 bytes of complete entries against a 25-byte limit evicts exactly the
    // least recently used entry.
    let report = store
        .enforce_retention(&policy(25, 1_000_000), 10_000)
        .unwrap();
    assert_eq!(report.complete_bytes_before, 30);
    assert_eq!(report.complete_bytes_after, 20);
    assert!(report.limit_satisfied);
    assert!(report.failed.is_empty());
    assert_eq!(
        report
            .evicted
            .iter()
            .map(|record| (
                record.cache_key.as_str(),
                record.bytes,
                record.cause.clone()
            ))
            .collect::<Vec<_>>(),
        vec![(
            candidate_a.cache_key.as_str(),
            10,
            EvictionCause::SizePressure
        )]
    );
    assert_eq!(
        retained_hold(&report, &candidate_b.cache_key),
        RetentionHold::Fresh
    );
    assert!(!entry_dir(&store, &candidate_a.cache_key).exists());
    assert!(store
        .lookup_complete(&candidate_a.cache_key)
        .unwrap()
        .is_none());
    assert!(store
        .lookup_complete(&candidate_b.cache_key)
        .unwrap()
        .is_some());
    assert!(store
        .lookup_complete(&candidate_c.cache_key)
        .unwrap()
        .is_some());
    assert_eq!(audit_records(&store).len(), 1);

    // TTL: with the fake clock at 2_000 + ttl, entry b (activity 2_000) is expired and entry c
    // (activity 3_000) is not — deterministically.
    let report = store
        .enforce_retention(&policy(1_000_000, 1_000_000), 1_002_000)
        .unwrap();
    assert_eq!(
        report
            .evicted
            .iter()
            .map(|record| (record.cache_key.as_str(), record.cause.clone()))
            .collect::<Vec<_>>(),
        vec![(candidate_b.cache_key.as_str(), EvictionCause::TtlExpired)]
    );
    assert_eq!(
        retained_hold(&report, &candidate_c.cache_key),
        RetentionHold::Fresh
    );
    assert!(store
        .lookup_complete(&candidate_c.cache_key)
        .unwrap()
        .is_some());
    assert_eq!(audit_records(&store).len(), 2);
}

#[test]
fn equal_activity_lru_ties_break_deterministically_by_cache_key() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate_a = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    let candidate_b = flat_candidate(&library, "SceneWorks/m-b", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate_a);
    materialize_complete(&store, &library, &candidate_b);
    stamp_activity(&store, &candidate_a.cache_key, Some(1_000), 500);
    stamp_activity(&store, &candidate_b.cache_key, Some(1_000), 500);
    let report = store
        .enforce_retention(&policy(10, 1_000_000), 10_000)
        .unwrap();
    let expected = std::cmp::min(&candidate_a.cache_key, &candidate_b.cache_key);
    assert_eq!(report.evicted.len(), 1);
    assert_eq!(&report.evicted[0].cache_key, expected);
}

#[test]
fn active_lease_survives_automatic_cleanup_and_eviction_resumes_after_release() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    let resolver = resolver(&library, ActiveArtifactLeaseRegistry::default());
    let lease = store
        .acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
        .unwrap()
        .unwrap();

    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert!(report.evicted.is_empty());
    assert_eq!(
        retained_hold(&report, &candidate.cache_key),
        RetentionHold::ActiveUse
    );
    assert!(!report.limit_satisfied);
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_some());

    drop(lease);
    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert_eq!(report.evicted.len(), 1);
    assert_eq!(report.evicted[0].cause, EvictionCause::TtlExpired);
    assert!(!entry_dir(&store, &candidate.cache_key).exists());
}

#[test]
fn pinned_entries_survive_automatic_cleanup_until_unpinned() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    stamp_activity(&store, &candidate.cache_key, Some(1_000), 500);

    let pins: [fn(&ResolvedCacheStore, &str); 2] = [
        |store, key| {
            store.set_artifact_pin(key, true).unwrap();
        },
        |store, key| {
            store.set_model_pin(key, "image:model-a", true).unwrap();
        },
    ];
    for pin in pins {
        pin(&store, &candidate.cache_key);
        let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
        assert!(report.evicted.is_empty());
        assert_eq!(
            retained_hold(&report, &candidate.cache_key),
            RetentionHold::Pinned
        );
        assert!(!report.limit_satisfied);
        assert!(store
            .lookup_complete(&candidate.cache_key)
            .unwrap()
            .is_some());
        store.set_artifact_pin(&candidate.cache_key, false).unwrap();
        store
            .set_model_pin(&candidate.cache_key, "image:model-a", false)
            .unwrap();
        stamp_activity(&store, &candidate.cache_key, Some(1_000), 500);
    }

    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert_eq!(report.evicted.len(), 1);
    assert!(!entry_dir(&store, &candidate.cache_key).exists());
}

#[test]
fn live_materialization_and_interrupted_partials_are_never_evicted() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    let reservation = match store
        .reserve(&candidate, &library, "image:model-a")
        .unwrap()
    {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("fixture reservation must acquire"),
    };

    // A live in-progress materialization is protected.
    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert!(report.evicted.is_empty());
    assert_eq!(
        retained_hold(&report, &candidate.cache_key),
        RetentionHold::MaterializationInProgress
    );
    let digest = cache_key_digest(&candidate.cache_key).unwrap();
    {
        let _lock = store.lock_metadata(&digest).unwrap();
        assert_eq!(
            store.read_metadata_locked(&digest).unwrap().state,
            ResolvedCacheEntryState::Materializing
        );
    }

    // An interrupted partial is a recovery candidate, not an eviction target.
    drop(reservation);
    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert!(report.evicted.is_empty());
    assert_eq!(
        retained_hold(&report, &candidate.cache_key),
        RetentionHold::RecoveryCandidate
    );
    {
        let _lock = store.lock_metadata(&digest).unwrap();
        assert_eq!(
            store.read_metadata_locked(&digest).unwrap().state,
            ResolvedCacheEntryState::Interrupted
        );
    }
}

#[test]
fn source_unverifiable_entries_are_retained_automatically_including_content_divergence() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    stamp_activity(&store, &candidate.cache_key, Some(1_000), 500);
    let source_file = snapshot(&library, "SceneWorks/m-a", REV_A).join("model.safetensors");
    let bundle_file = store
        .bundle_path(&candidate.cache_key)
        .unwrap()
        .join("model.safetensors");

    // Source file missing: the resolved bundle may be the sole copy.
    std::fs::remove_file(&source_file).unwrap();
    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert!(report.evicted.is_empty());
    assert_eq!(
        retained_hold(&report, &candidate.cache_key),
        RetentionHold::SourceUnverified
    );
    assert_eq!(std::fs::read(&bundle_file).unwrap(), b"0123456789");

    // Same-size different content: only the sha256 re-verification can catch this, so this also
    // proves eviction hash-verifies the source rather than trusting existence and size.
    std::fs::write(&source_file, b"9876543210").unwrap();
    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert!(report.evicted.is_empty());
    assert_eq!(
        retained_hold(&report, &candidate.cache_key),
        RetentionHold::SourceUnverified
    );
    assert_eq!(std::fs::read(&bundle_file).unwrap(), b"0123456789");

    // Source restored bit-for-bit: eviction proceeds.
    std::fs::write(&source_file, b"0123456789").unwrap();
    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert_eq!(report.evicted.len(), 1);
    assert!(!entry_dir(&store, &candidate.cache_key).exists());
    assert_eq!(std::fs::read(&source_file).unwrap(), b"0123456789");
}

#[test]
fn corrupt_entries_are_retained_for_recovery_never_auto_evicted() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    let entry = entry_dir(&store, &candidate.cache_key);
    for slot in 0..=1 {
        std::fs::write(entry.join(format!("metadata.{slot}.json")), b"corrupt").unwrap();
    }
    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert!(report.evicted.is_empty());
    assert_eq!(
        retained_hold(&report, &candidate.cache_key),
        RetentionHold::RecoveryCandidate
    );
    assert!(entry.join("bundle").join("model.safetensors").is_file());
}

#[test]
fn size_enforcement_explains_why_protected_entries_prevent_reaching_the_limit() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate_a = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    let candidate_b = flat_candidate(&library, "SceneWorks/m-b", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate_a);
    materialize_complete(&store, &library, &candidate_b);
    store
        .set_artifact_pin(&candidate_a.cache_key, true)
        .unwrap();
    store
        .set_artifact_pin(&candidate_b.cache_key, true)
        .unwrap();
    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert!(report.evicted.is_empty());
    assert!(!report.limit_satisfied);
    assert_eq!(report.complete_bytes_before, 20);
    assert_eq!(report.complete_bytes_after, 20);
    assert_eq!(report.retained.len(), 2);
    assert!(report
        .retained
        .iter()
        .all(|record| record.hold == RetentionHold::Pinned));
    assert!(store
        .lookup_complete(&candidate_a.cache_key)
        .unwrap()
        .is_some());
    assert!(store
        .lookup_complete(&candidate_b.cache_key)
        .unwrap()
        .is_some());
}

#[test]
fn manual_remove_reports_reclaimable_bytes_and_respects_pins_and_leases() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let data = scratch.path().join("data");
    let store = ResolvedCacheStore::open(&data).unwrap();
    let candidate_a = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    let candidate_b = flat_candidate(&library, "SceneWorks/m-b", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate_a);
    materialize_complete(&store, &library, &candidate_b);
    let sibling = data.join("models").join("converted");
    std::fs::create_dir_all(&sibling).unwrap();
    std::fs::write(sibling.join("sentinel.bin"), b"converted-artifact").unwrap();

    // Pinned entries refuse manual removal until unpinned; the preview says so.
    store
        .set_artifact_pin(&candidate_a.cache_key, true)
        .unwrap();
    let preview = store
        .manual_removal_preview(&candidate_a.cache_key)
        .unwrap();
    assert!(preview.blocked.as_deref().unwrap().contains("pinned"));
    assert!(preview.artifact_pinned);
    assert!(preview.source_unavailable_warning.is_none());
    assert!(preview.reclaimable_bytes >= 10);
    let error = store
        .remove_entry(&candidate_a.cache_key, 1_000)
        .unwrap_err();
    assert!(error.to_string().contains("pinned"));
    store
        .set_artifact_pin(&candidate_a.cache_key, false)
        .unwrap();

    // An active lease refuses manual removal.
    let resolver = resolver(&library, ActiveArtifactLeaseRegistry::default());
    let lease = store
        .acquire_complete(&candidate_a.cache_key, &resolver, "runtime:image:model")
        .unwrap()
        .unwrap();
    let preview = store
        .manual_removal_preview(&candidate_a.cache_key)
        .unwrap();
    assert!(preview.blocked.as_deref().unwrap().contains("active"));
    let error = store
        .remove_entry(&candidate_a.cache_key, 1_000)
        .unwrap_err();
    assert!(error.to_string().contains("active lease"));
    drop(lease);

    // Removal of an entry whose source vanished succeeds but warns that the model is unavailable
    // until the source returns; the reclaimable preview matches the actual reclaim.
    std::fs::remove_dir_all(
        ArtifactSourceLibrary::new(&library)
            .unwrap()
            .repository_root("SceneWorks/m-a")
            .unwrap(),
    )
    .unwrap();
    let preview = store
        .manual_removal_preview(&candidate_a.cache_key)
        .unwrap();
    assert!(preview.blocked.is_none());
    assert!(preview.source_unavailable_warning.is_some());
    let outcome = store.remove_entry(&candidate_a.cache_key, 1_000).unwrap();
    assert_eq!(outcome.reclaimed_bytes, preview.reclaimable_bytes);
    assert!(outcome.source_unavailable_warning.is_some());
    assert!(!entry_dir(&store, &candidate_a.cache_key).exists());
    assert_eq!(audit_records(&store).len(), 1);

    // The sibling entry, sibling model directories, and the remaining source are untouched.
    assert!(store
        .lookup_complete(&candidate_b.cache_key)
        .unwrap()
        .is_some());
    assert_eq!(
        std::fs::read(sibling.join("sentinel.bin")).unwrap(),
        b"converted-artifact"
    );
    assert!(store
        .manual_removal_preview(&candidate_a.cache_key)
        .is_err());
}

#[test]
fn reconciliation_covers_tier_deletion_revision_replacement_and_full_uninstall() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let data = scratch.path().join("data");
    let store = ResolvedCacheStore::open(&data).unwrap();
    let x_a_q8 = shared_component_candidate(&library, "SceneWorks/model-x", REV_A, "q8", b"x-a-q8");
    let x_a_q4 = flat_candidate(&library, "SceneWorks/model-x", REV_A, "q4", b"x-a-q4");
    let x_b_q8 = flat_candidate(&library, "SceneWorks/model-x", REV_B, "q8", b"x-b-q8");
    let y_a_q8 = shared_component_candidate(&library, "SceneWorks/model-y", REV_A, "q8", b"y-a-q8");
    for candidate in [&x_a_q8, &x_a_q4, &x_b_q8, &y_a_q8] {
        materialize_complete(&store, &library, candidate);
    }
    let sibling = data.join("models").join("imported");
    std::fs::create_dir_all(&sibling).unwrap();
    std::fs::write(sibling.join("sentinel.bin"), b"imported-asset").unwrap();
    let component_source =
        snapshot(&library, "SceneWorks/component-c", REV_C).join("te.safetensors");
    let y_receipt = entry_dir(&store, &y_a_q8.cache_key).join("complete.receipt.json");
    let y_receipt_bytes = std::fs::read(&y_receipt).unwrap();

    // Single-tier deletion removes exactly the matching tier.
    let report = store
        .reconcile_removed_source(
            &SourceLifecycleSelector {
                repository: "SceneWorks/model-x".to_owned(),
                revision: Some(REV_A.to_owned()),
                tier: Some("q4".to_owned()),
            },
            1_000,
        )
        .unwrap();
    assert_eq!(
        report
            .removed
            .iter()
            .map(|record| record.cache_key.as_str())
            .collect::<Vec<_>>(),
        vec![x_a_q4.cache_key.as_str()]
    );
    assert!(report.deferred.is_empty());
    assert!(store.lookup_complete(&x_a_q8.cache_key).unwrap().is_some());

    // Revision replacement removes the remaining old-revision entry; the new revision survives.
    let report = store
        .reconcile_removed_source(
            &SourceLifecycleSelector {
                repository: "SceneWorks/model-x".to_owned(),
                revision: Some(REV_A.to_owned()),
                tier: None,
            },
            2_000,
        )
        .unwrap();
    assert_eq!(
        report
            .removed
            .iter()
            .map(|record| record.cache_key.as_str())
            .collect::<Vec<_>>(),
        vec![x_a_q8.cache_key.as_str()]
    );
    assert!(store.lookup_complete(&x_b_q8.cache_key).unwrap().is_some());

    // Full uninstall removes every remaining entry of the model, pins included: the pin protected
    // cache retention while the model existed, not an explicitly uninstalled model.
    store.set_artifact_pin(&x_b_q8.cache_key, true).unwrap();
    let report = store
        .reconcile_removed_source(
            &SourceLifecycleSelector {
                repository: "SceneWorks/model-x".to_owned(),
                revision: None,
                tier: None,
            },
            3_000,
        )
        .unwrap();
    assert_eq!(
        report
            .removed
            .iter()
            .map(|record| (record.cache_key.as_str(), record.cause.clone()))
            .collect::<Vec<_>>(),
        vec![(x_b_q8.cache_key.as_str(), EvictionCause::SourceRemoved)]
    );

    // The sibling model that shares component C stays fully valid: its self-contained bundle,
    // receipt, shared-component source file, and sibling model directories are all intact.
    assert!(store.lookup_complete(&y_a_q8.cache_key).unwrap().is_some());
    assert_eq!(std::fs::read(&y_receipt).unwrap(), y_receipt_bytes);
    assert_eq!(
        std::fs::read(&component_source).unwrap(),
        b"shared-text-encoder"
    );
    assert_eq!(
        std::fs::read(sibling.join("sentinel.bin")).unwrap(),
        b"imported-asset"
    );
    assert_eq!(audit_records(&store).len(), 3);
}

#[test]
fn reconciliation_defers_active_leases_and_surfaces_unreadable_entries() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate_y = flat_candidate(&library, "SceneWorks/model-y", REV_A, "q8", b"y-bytes");
    let candidate_z = flat_candidate(&library, "SceneWorks/model-z", REV_A, "q8", b"z-bytes");
    materialize_complete(&store, &library, &candidate_y);
    materialize_complete(&store, &library, &candidate_z);
    let entry_z = entry_dir(&store, &candidate_z.cache_key);
    for slot in 0..=1 {
        std::fs::write(entry_z.join(format!("metadata.{slot}.json")), b"corrupt").unwrap();
    }
    let resolver = resolver(&library, ActiveArtifactLeaseRegistry::default());
    let lease = store
        .acquire_complete(&candidate_y.cache_key, &resolver, "runtime:image:model")
        .unwrap()
        .unwrap();
    let selector = SourceLifecycleSelector {
        repository: "SceneWorks/model-y".to_owned(),
        revision: None,
        tier: None,
    };

    let report = store.reconcile_removed_source(&selector, 1_000).unwrap();
    assert!(report.removed.is_empty());
    assert_eq!(report.deferred.len(), 1);
    assert_eq!(report.deferred[0].hold, RetentionHold::ActiveUse);
    assert_eq!(report.unmatched_unreadable, 1);
    assert!(store
        .lookup_complete(&candidate_y.cache_key)
        .unwrap()
        .is_some());

    drop(lease);
    let report = store.reconcile_removed_source(&selector, 2_000).unwrap();
    assert_eq!(report.removed.len(), 1);
    assert!(report.deferred.is_empty());
    assert!(!entry_dir(&store, &candidate_y.cache_key).exists());
    // The unreadable entry is surfaced, never silently removed.
    assert!(entry_z.join("bundle").join("model.safetensors").is_file());
}

#[test]
fn interrupted_eviction_converges_on_recovery_lookup_and_reservation() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    let digest = cache_key_digest(&candidate.cache_key).unwrap();

    // Simulate a crash immediately after the tombstone became durable.
    store
        .write_eviction_marker(
            &digest,
            &EvictionMarker {
                schema_version: RESOLVED_CACHE_STORE_VERSION,
                cache_key: candidate.cache_key.clone(),
                cause: EvictionCause::TtlExpired,
                reclaimable_bytes: 10,
                requested_at: 1_000,
                session_id: store.session_id().to_owned(),
            },
        )
        .unwrap();

    // Every reader treats a valid tombstone as already gone.
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_none());
    let resolver = resolver(&library, ActiveArtifactLeaseRegistry::default());
    assert!(store
        .acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
        .unwrap()
        .is_none());
    let summaries = store.enumerate().unwrap();
    assert_eq!(summaries[0].state, ResolvedCacheEntryState::Evicting);

    // Recovery finishes the interrupted removal and records the audit trail.
    store.recover().unwrap();
    assert!(!entry_dir(&store, &candidate.cache_key).exists());
    assert_eq!(audit_records(&store).len(), 1);

    // A new reservation for the same key can also converge a pending tombstone on its own.
    materialize_complete(&store, &library, &candidate);
    store
        .write_eviction_marker(
            &digest,
            &EvictionMarker {
                schema_version: RESOLVED_CACHE_STORE_VERSION,
                cache_key: candidate.cache_key.clone(),
                cause: EvictionCause::SizePressure,
                reclaimable_bytes: 10,
                requested_at: 2_000,
                session_id: store.session_id().to_owned(),
            },
        )
        .unwrap();
    match store
        .reserve(&candidate, &library, "image:model-a")
        .unwrap()
    {
        ReservationOutcome::Acquired(reservation) => drop(reservation),
        other => panic!("reservation over a tombstone must converge and acquire, got {other:?}"),
    }
    assert_eq!(audit_records(&store).len(), 2);
    let republished = materialize_complete(&store, &library, &candidate);
    assert_eq!(republished.state, ResolvedCacheEntryState::Complete);
}

#[test]
fn invalid_tombstone_fails_safe_and_recovery_resurrects_the_receipt_backed_entry() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    let entry = entry_dir(&store, &candidate.cache_key);
    std::fs::write(entry.join(EVICTED_MARKER_FILE), b"garbage-tombstone").unwrap();

    // An invalid tombstone is never a sanction to delete: reads fail closed instead.
    assert!(store.lookup_complete(&candidate.cache_key).is_err());

    // Recovery drops the garbage tombstone and resurrects the receipt-validated entry pinned.
    store.recover().unwrap();
    assert!(!entry.join(EVICTED_MARKER_FILE).exists());
    let metadata = store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .unwrap();
    assert_eq!(
        metadata.recovery_status,
        RecoveryStatus::ReconstructedFromCompleteReceipt
    );
    assert!(metadata.effective_pin);
    assert_eq!(
        std::fs::read(entry.join("bundle").join("model.safetensors")).unwrap(),
        b"0123456789"
    );
}

/// The tombstone probe runs on every metadata read, so it is confinement-checked like every other
/// managed path in this module: a linked tombstone is refused rather than followed, and refusing it
/// fails the read closed instead of silently treating the entry as evictable.
#[cfg(unix)]
#[test]
fn eviction_tombstone_probe_refuses_a_link_without_reading_through_it() {
    use std::os::unix::fs::symlink;

    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    let external = scratch.path().join("external-tombstone.json");
    std::fs::write(&external, b"external-must-not-be-read").unwrap();
    symlink(
        &external,
        entry_dir(&store, &candidate.cache_key).join(EVICTED_MARKER_FILE),
    )
    .unwrap();

    let error = store
        .lookup_complete(&candidate.cache_key)
        .expect_err("a linked eviction tombstone must fail the read closed");
    assert!(error.to_string().contains("link"));
    assert_eq!(
        std::fs::read(&external).unwrap(),
        b"external-must-not-be-read"
    );
    assert!(entry_dir(&store, &candidate.cache_key)
        .join("bundle")
        .join("model.safetensors")
        .is_file());
}

/// A reservation whose interruption cannot be recorded must leave the entry conservatively
/// `Materializing` (never silently "clean"), so a later pass still treats it as this session's
/// live work rather than as an eviction candidate.
#[test]
fn a_reservation_that_cannot_record_its_interruption_leaves_materializing_state() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    let reservation = match store
        .reserve(&candidate, &library, "image:model-a")
        .unwrap()
    {
        ReservationOutcome::Acquired(reservation) => reservation,
        other => panic!("fixture reservation must acquire, got {other:?}"),
    };
    let digest = cache_key_digest(&candidate.cache_key).unwrap();
    std::fs::write(
        entry_dir(&store, &candidate.cache_key).join(EVICTED_MARKER_FILE),
        b"garbage-tombstone",
    )
    .unwrap();
    drop(reservation);

    let _lock = store.lock_metadata(&digest).unwrap();
    assert!(store.read_metadata_locked(&digest).is_err());
    let entry = store.inner.root.join("entries").join(&digest);
    let envelope = [entry.join("metadata.0.json"), entry.join("metadata.1.json")]
        .into_iter()
        .filter_map(|path| read_journal(&path).ok())
        .max_by_key(|envelope| envelope.generation)
        .expect("the journal still holds the reservation state");
    assert_eq!(
        envelope.metadata.state,
        ResolvedCacheEntryState::Materializing
    );
}

#[cfg(unix)]
#[test]
fn eviction_removal_never_follows_a_swapped_symlink_outside_the_root() {
    use std::os::unix::fs::symlink;

    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    stamp_activity(&store, &candidate.cache_key, Some(1_000), 500);
    let entry = entry_dir(&store, &candidate.cache_key);
    let external = scratch.path().join("external");
    std::fs::create_dir(&external).unwrap();
    let sentinel = external.join("sentinel");
    std::fs::write(&sentinel, b"untouched").unwrap();
    let swapped_entry = entry.clone();
    let external_target = external.clone();
    set_remove_entry_after_stat_hook(move || {
        std::fs::remove_dir_all(&swapped_entry).unwrap();
        symlink(&external_target, &swapped_entry).unwrap();
    });

    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert!(report.evicted.is_empty());
    assert_eq!(report.failed.len(), 1);
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"untouched");
    assert!(external.is_dir());
    // The removal failed, so nothing may claim it completed: the audit trail records completed
    // removals only, and the tombstone remains as the durable record of intent.
    assert!(audit_records(&store).is_empty());
    std::fs::remove_file(&entry).unwrap();
}

#[test]
fn retention_checkpoints_gate_on_policy_and_idleness() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    stamp_activity(&store, &candidate.cache_key, Some(1_000), 500);

    let disabled =
        ResolvedCacheRetention::new(store.clone(), ResolvedCachePolicy::default()).unwrap();
    assert_eq!(
        disabled.run_if_idle(true, FAR_FUTURE).unwrap(),
        RetentionCheckpointOutcome::Disabled
    );
    assert_eq!(
        disabled.run_after_recovery(FAR_FUTURE).unwrap(),
        RetentionCheckpointOutcome::Disabled
    );
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_some());

    let enabled = ResolvedCacheRetention::new(store.clone(), policy(1_000_000, 1_000)).unwrap();
    assert_eq!(
        enabled.run_if_idle(false, FAR_FUTURE).unwrap(),
        RetentionCheckpointOutcome::NotIdle
    );
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_some());
    match enabled.run_if_idle(true, FAR_FUTURE).unwrap() {
        RetentionCheckpointOutcome::Ran(report) => {
            assert_eq!(report.evicted.len(), 1);
            assert_eq!(report.evicted[0].cause, EvictionCause::TtlExpired);
        }
        other => panic!("idle checkpoint must run retention, got {other:?}"),
    }
    assert!(!entry_dir(&store, &candidate.cache_key).exists());
}

/// An incidental open handle is NOT an eviction protection, and must not be mistaken for one.
/// Windows deletes through an ordinary shared reader (POSIX-semantics delete), so a reader that
/// holds no lease cannot keep a policy-expired entry alive; the artifact lease is the protection,
/// and `active_lease_survives_automatic_cleanup_and_eviction_resumes_after_release` proves that
/// one on this platform too. This test pins the real behavior so a future reader-blocks-delete
/// assumption cannot creep back in.
#[cfg(windows)]
#[test]
fn windows_shared_reader_without_a_lease_does_not_protect_an_expired_entry() {
    use std::io::Read;

    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    stamp_activity(&store, &candidate.cache_key, Some(1_000), 500);
    let bundle_file = store
        .bundle_path(&candidate.cache_key)
        .unwrap()
        .join("model.safetensors");

    let mut reader = File::open(&bundle_file).unwrap();
    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert!(report.failed.is_empty());
    assert_eq!(report.evicted.len(), 1);
    assert!(!entry_dir(&store, &candidate.cache_key).exists());
    assert_eq!(audit_records(&store).len(), 1);

    // The orphaned handle keeps serving the bytes it already opened; nothing is corrupted.
    let mut contents = Vec::new();
    reader.read_to_end(&mut contents).unwrap();
    assert_eq!(contents, b"0123456789");
}

/// A genuine Windows sharing violation must fail closed and stay convergent: the durable tombstone
/// survives the failed deletion, the entry reads as `Evicting` rather than as a usable bundle, no
/// audit record claims a removal that did not happen, and the next checkpoint after the handle
/// closes finishes the removal exactly once.
///
/// The handle deliberately shares READ and WRITE but withholds DELETE. Sharing reads is what makes
/// this test exercise the *removal* stage: every pre-eviction check reopens the bundle file (the
/// enriched closure is size- and sha256-verified), so a fully exclusive `share_mode(0)` handle
/// would instead fail that verification and retain the entry as unverifiable — correct fail-safe
/// behavior, but a different code path than the one under test here.
#[cfg(windows)]
#[test]
fn windows_sharing_violation_keeps_the_eviction_pending_until_it_converges() {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;

    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("library");
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let candidate = flat_candidate(&library, "SceneWorks/m-a", REV_A, "q8", b"0123456789");
    materialize_complete(&store, &library, &candidate);
    stamp_activity(&store, &candidate.cache_key, Some(1_000), 500);
    let bundle_file = store
        .bundle_path(&candidate.cache_key)
        .unwrap()
        .join("model.safetensors");

    let delete_blocking = OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .open(&bundle_file)
        .unwrap();
    let report = store.enforce_retention(&policy(1, 1), FAR_FUTURE).unwrap();
    assert!(report.evicted.is_empty());
    assert_eq!(report.failed.len(), 1);
    assert!(audit_records(&store).is_empty());
    assert_eq!(
        store.enumerate().unwrap()[0].state,
        ResolvedCacheEntryState::Evicting
    );
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_none());

    drop(delete_blocking);
    store.recover().unwrap();
    assert!(!entry_dir(&store, &candidate.cache_key).exists());
    assert_eq!(audit_records(&store).len(), 1);
}
