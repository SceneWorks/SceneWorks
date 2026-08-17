use super::*;
use crate::model_artifacts::{
    ArtifactAvailability, ArtifactCompleteness, ArtifactIdentity, ArtifactMemberRole,
    ArtifactProvenance, ArtifactSourceLibrary, ModelArtifactResolver, ResolvedBundleMember,
    MODEL_ARTIFACT_CONTRACT_VERSION,
};
use std::sync::mpsc;
use std::thread;
use tempfile::TempDir;

const REV_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const REV_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn identity(repository: &str, revision: &str, variant: &str) -> ArtifactIdentity {
    ArtifactIdentity::pinned(repository, revision, variant).unwrap()
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
    source_subpath: &str,
    destination: &str,
    files: &[&str],
) -> ResolvedBundleMember {
    ResolvedBundleMember {
        role,
        component_id: component_id.map(str::to_owned),
        source,
        tier: tier.map(str::to_owned),
        source_subpath: PathBuf::from(source_subpath),
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
            fixed_artifact_tier: Some("q8".to_owned()),
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

fn flat_fixture(scratch: &TempDir) -> (PathBuf, PromotionCandidate) {
    flat_fixture_at_revision(scratch, REV_A)
}

fn flat_fixture_at_revision(scratch: &TempDir, revision: &str) -> (PathBuf, PromotionCandidate) {
    let library = scratch.path().join("source");
    let primary_identity = identity("SceneWorks/model", revision, "q8");
    let primary_snapshot = snapshot(&library, "SceneWorks/model", revision);
    std::fs::write(primary_snapshot.join("model.safetensors"), b"model-weights").unwrap();
    let candidate = candidate(
        primary_snapshot,
        primary_identity.clone(),
        vec![member(
            ArtifactMemberRole::Primary,
            None,
            primary_identity,
            Some("q8"),
            "",
            "",
            &["model.safetensors"],
        )],
    );
    (library, candidate)
}

fn enabled_policy() -> ResolvedCachePolicy {
    ResolvedCachePolicy {
        enabled: true,
        ..ResolvedCachePolicy::default()
    }
}

#[cfg(windows)]
struct WindowsDirectoryJunction(PathBuf);

#[cfg(windows)]
impl WindowsDirectoryJunction {
    fn create(path: &Path, target: &Path) -> Self {
        let output = std::process::Command::new("cmd")
            .args(["/D", "/C", "mklink", "/J"])
            .arg(path)
            .arg(target)
            .output()
            .expect("launch cmd.exe to create source directory junction");
        assert!(
            output.status.success(),
            "mklink /J failed with {}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata = std::fs::symlink_metadata(path).unwrap();
        assert!(metadata_is_reparse_point(&metadata));
        Self(path.to_owned())
    }

    fn remove(self) {
        std::fs::remove_dir(&self.0).unwrap();
        std::mem::forget(self);
    }
}

#[cfg(windows)]
impl Drop for WindowsDirectoryJunction {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.0);
    }
}

#[test]
fn flat_bundle_streams_verifies_enriches_and_publishes_atomically() {
    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let key_before = candidate.cache_key.clone();
    let outcome = ResolvedCacheMaterializer::new(store.clone())
        .materialize(
            &candidate,
            &source,
            "image:model",
            &MaterializationCancellation::default(),
        )
        .unwrap();
    let metadata = match outcome {
        MaterializationOutcome::Published(metadata) => metadata,
        other => panic!("unexpected materialization outcome: {other:?}"),
    };
    assert_eq!(metadata.cache_key, key_before);
    assert_eq!(metadata.state, ResolvedCacheEntryState::Complete);
    assert_eq!(metadata.verified_bytes, b"model-weights".len() as u64);
    let file = &metadata.artifact.closure.members[0].files[0];
    assert_eq!(file.size_bytes, Some(b"model-weights".len() as u64));
    assert!(file
        .sha256
        .as_deref()
        .is_some_and(|hash| hash.starts_with("sha256:")));
    assert_eq!(metadata.artifact.cache_key().unwrap(), key_before);
    assert_eq!(
        std::fs::read(
            store
                .bundle_path(&key_before)
                .unwrap()
                .join("model.safetensors")
        )
        .unwrap(),
        b"model-weights"
    );
    assert!(std::fs::read_dir(store.root().join("staging"))
        .unwrap()
        .next()
        .is_none());
    assert!(store.lookup_complete(&key_before).unwrap().is_some());
    assert!(matches!(
        ResolvedCacheMaterializer::new(store)
            .materialize(
                &candidate,
                &source,
                "image:model",
                &MaterializationCancellation::default(),
            )
            .unwrap(),
        MaterializationOutcome::AlreadyComplete(_)
    ));
}

#[test]
fn tiered_multi_repo_optional_derived_and_shared_members_are_self_contained() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("source");
    let primary_identity = identity("SceneWorks/model", REV_A, "q8");
    let tokenizer_identity = identity("SceneWorks/tokenizer", REV_B, "default");
    let primary_snapshot = snapshot(&library, "SceneWorks/model", REV_A);
    let tokenizer_snapshot = snapshot(&library, "SceneWorks/tokenizer", REV_B);
    for directory in ["q8", "optional", "derived", "shared"] {
        std::fs::create_dir_all(primary_snapshot.join(directory)).unwrap();
    }
    std::fs::create_dir_all(tokenizer_snapshot.join("tokenizer")).unwrap();
    std::fs::write(primary_snapshot.join("q8/model.safetensors"), b"primary").unwrap();
    std::fs::write(
        primary_snapshot.join("optional/style.safetensors"),
        b"style",
    )
    .unwrap();
    std::fs::write(
        primary_snapshot.join("derived/tokenizer_config.json"),
        b"derived",
    )
    .unwrap();
    std::fs::write(primary_snapshot.join("shared/config.json"), b"shared").unwrap();
    std::fs::write(
        tokenizer_snapshot.join("tokenizer/tokenizer.json"),
        b"tokenizer",
    )
    .unwrap();
    let candidate = candidate(
        primary_snapshot,
        primary_identity.clone(),
        vec![
            member(
                ArtifactMemberRole::Primary,
                None,
                primary_identity.clone(),
                Some("q8"),
                "q8",
                "",
                &["model.safetensors"],
            ),
            member(
                ArtifactMemberRole::OptionalComponent,
                Some("style"),
                primary_identity.clone(),
                Some("q8"),
                "optional",
                "adapters",
                &["style.safetensors"],
            ),
            member(
                ArtifactMemberRole::CoRequisite,
                Some("tokenizer"),
                tokenizer_identity,
                None,
                "tokenizer",
                "tokenizer",
                &["tokenizer.json"],
            ),
            member(
                ArtifactMemberRole::DerivedOverlay,
                Some("tokenizer-config"),
                primary_identity.clone(),
                Some("q8"),
                "derived",
                "tokenizer",
                &["tokenizer_config.json"],
            ),
            member(
                ArtifactMemberRole::RequiredComponent,
                Some("shared-a"),
                primary_identity.clone(),
                Some("q8"),
                "shared",
                "component-a",
                &["config.json"],
            ),
            member(
                ArtifactMemberRole::RequiredComponent,
                Some("shared-b"),
                primary_identity,
                Some("q8"),
                "shared",
                "component-b",
                &["config.json"],
            ),
        ],
    );
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let outcome = ResolvedCacheMaterializer::new(store.clone())
        .materialize(
            &candidate,
            &library,
            "video:model",
            &MaterializationCancellation::default(),
        )
        .unwrap();
    assert!(matches!(outcome, MaterializationOutcome::Published(_)));
    let bundle = store.bundle_path(&candidate.cache_key).unwrap();
    for file in [
        "model.safetensors",
        "adapters/style.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "component-a/config.json",
        "component-b/config.json",
    ] {
        assert!(bundle.join(file).is_file(), "missing materialized {file}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            std::fs::metadata(bundle.join("component-a/config.json"))
                .unwrap()
                .ino(),
            std::fs::metadata(bundle.join("component-b/config.json"))
                .unwrap()
                .ino(),
            "identical immutable source members should deduplicate inside staging"
        );
    }
}

#[test]
fn cancellation_and_io_failures_leave_only_interrupted_metadata() {
    #[cfg(unix)]
    let failures = vec![
        std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        std::io::Error::from_raw_os_error(libc::ENOSPC),
    ];
    #[cfg(not(unix))]
    let failures = vec![std::io::Error::from(std::io::ErrorKind::PermissionDenied)];
    for failure in failures {
        let scratch = TempDir::new().unwrap();
        let (source, candidate) = flat_fixture(&scratch);
        let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
        let kind = failure.kind();
        let raw = failure.raw_os_error();
        let materializer =
            ResolvedCacheMaterializer::new(store.clone()).with_test_hook(move |_, _| {
                Err(raw
                    .map(std::io::Error::from_raw_os_error)
                    .unwrap_or_else(|| std::io::Error::from(kind)))
            });
        assert!(materializer
            .materialize(
                &candidate,
                &source,
                "image:model",
                &MaterializationCancellation::default(),
            )
            .is_err());
        assert!(store
            .lookup_complete(&candidate.cache_key)
            .unwrap()
            .is_none());
        assert_eq!(
            store.enumerate().unwrap()[0].state,
            ResolvedCacheEntryState::Interrupted
        );
        assert!(std::fs::read_dir(store.root().join("staging"))
            .unwrap()
            .next()
            .is_none());
        assert_eq!(
            std::fs::read(
                match &candidate.artifact.location {
                    ArtifactLocation::SourceLibrary { root } => root,
                    _ => unreachable!(),
                }
                .join("model.safetensors")
            )
            .unwrap(),
            b"model-weights"
        );
    }

    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let cancellation = MaterializationCancellation::default();
    cancellation.cancel();
    assert_eq!(
        ResolvedCacheMaterializer::new(store.clone())
            .materialize(&candidate, &source, "image:model", &cancellation)
            .unwrap(),
        MaterializationOutcome::Cancelled
    );
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_none());
}

#[test]
fn source_disconnect_or_provenance_mutation_rejects_publication() {
    for mutate in [false, true] {
        let scratch = TempDir::new().unwrap();
        let (source, mut candidate) = flat_fixture(&scratch);
        let source_file = match &candidate.artifact.location {
            ArtifactLocation::SourceLibrary { root } => root.join("model.safetensors"),
            _ => unreachable!(),
        };
        if mutate {
            candidate.artifact.closure.members[0].files[0].size_bytes =
                Some(b"model-weights".len() as u64);
            candidate.artifact.closure.members[0].files[0].sha256 =
                Some(format!("sha256:{:x}", Sha256::digest(b"model-weights")));
            candidate.cache_key = candidate.artifact.cache_key().unwrap();
        }
        let hook_path = source_file.clone();
        let materializer = ResolvedCacheMaterializer::new(
            ResolvedCacheStore::open(&scratch.path().join("data")).unwrap(),
        )
        .with_test_hook(move |_, _| {
            if mutate {
                std::fs::write(&hook_path, b"changed-bytes").unwrap();
            } else {
                std::fs::remove_file(&hook_path).unwrap();
            }
            Ok(())
        });
        assert!(materializer
            .materialize(
                &candidate,
                &source,
                "image:model",
                &MaterializationCancellation::default(),
            )
            .is_err());
        assert!(!matches!(
            materializer.store().lookup_complete(&candidate.cache_key),
            Ok(Some(_))
        ));
    }
}

#[test]
fn same_size_no_hash_source_replacement_cannot_change_the_planned_file() {
    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let source_file = match &candidate.artifact.location {
        ArtifactLocation::SourceLibrary { root } => root.join("model.safetensors"),
        _ => unreachable!(),
    };
    let replacement = scratch.path().join("replacement");
    std::fs::write(&replacement, b"other-weights").unwrap();
    assert_eq!(
        std::fs::metadata(&source_file).unwrap().len(),
        std::fs::metadata(&replacement).unwrap().len()
    );
    let hook_source = source_file.clone();
    let materializer = ResolvedCacheMaterializer::new(
        ResolvedCacheStore::open(&scratch.path().join("data")).unwrap(),
    )
    .with_test_hook(move |_, _| {
        std::fs::remove_file(&hook_source).unwrap();
        std::fs::rename(&replacement, &hook_source).unwrap();
        Ok(())
    });
    assert!(materializer
        .materialize(
            &candidate,
            &source,
            "image:model",
            &MaterializationCancellation::default(),
        )
        .is_err());
    assert!(!matches!(
        materializer.store().lookup_complete(&candidate.cache_key),
        Ok(Some(_))
    ));
}

#[cfg(unix)]
#[test]
fn post_plan_source_symlink_swap_cannot_publish_external_bytes() {
    use std::os::unix::fs::symlink;

    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let source_file = match &candidate.artifact.location {
        ArtifactLocation::SourceLibrary { root } => root.join("model.safetensors"),
        _ => unreachable!(),
    };
    let external = scratch.path().join("external");
    std::fs::write(&external, b"other-weights").unwrap();
    let hook_source = source_file.clone();
    let hook_external = external.clone();
    let materializer = ResolvedCacheMaterializer::new(
        ResolvedCacheStore::open(&scratch.path().join("data")).unwrap(),
    )
    .with_test_hook(move |_, _| {
        std::fs::remove_file(&hook_source).unwrap();
        symlink(&hook_external, &hook_source).unwrap();
        Ok(())
    });
    assert!(materializer
        .materialize(
            &candidate,
            &source,
            "image:model",
            &MaterializationCancellation::default(),
        )
        .is_err());
    assert!(!matches!(
        materializer.store().lookup_complete(&candidate.cache_key),
        Ok(Some(_))
    ));
    assert_eq!(std::fs::read(external).unwrap(), b"other-weights");
}

#[cfg(unix)]
#[test]
fn staging_root_swap_cannot_write_into_an_external_directory() {
    use std::os::unix::fs::symlink;

    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let external = scratch.path().join("external-staging-target");
    std::fs::create_dir(&external).unwrap();
    let sentinel = external.join("sentinel");
    std::fs::write(&sentinel, b"untouched").unwrap();
    let parked = scratch.path().join("parked-staging");
    let staging_root = store.root().join("staging");
    let materializer = ResolvedCacheMaterializer::new(store.clone()).with_test_hook({
        let external = external.clone();
        let parked = parked.clone();
        move |_, _| {
            let staging = std::fs::read_dir(&staging_root)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            std::fs::rename(&staging, &parked).unwrap();
            symlink(&external, &staging).unwrap();
            Ok(())
        }
    });

    assert!(materializer
        .materialize(
            &candidate,
            &source,
            "image:model",
            &MaterializationCancellation::default(),
        )
        .is_err());
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"untouched");
    assert!(!external.join("model.safetensors").exists());
    let linked_staging = std::fs::read_dir(store.root().join("staging"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::remove_file(linked_staging).unwrap();
    std::fs::remove_dir_all(parked).unwrap();
}

#[cfg(windows)]
#[test]
fn staging_root_junction_swap_cannot_write_into_an_external_directory() {
    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let external = scratch.path().join("external-staging-target");
    std::fs::create_dir(&external).unwrap();
    let sentinel = external.join("sentinel");
    std::fs::write(&sentinel, b"untouched").unwrap();
    let staging_root = store.root().join("staging");
    let external_target = external.clone();
    set_windows_directory_after_validation_hook(move || {
        let staging = std::fs::read_dir(&staging_root)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        std::fs::remove_dir_all(&staging).unwrap();
        let junction = WindowsDirectoryJunction::create(&staging, &external_target);
        std::mem::forget(junction);
    });

    assert!(ResolvedCacheMaterializer::new(store.clone())
        .materialize(
            &candidate,
            &source,
            "image:model",
            &MaterializationCancellation::default(),
        )
        .is_err());
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"untouched");
    assert!(!external.join("model.safetensors").exists());
    let linked_staging = std::fs::read_dir(store.root().join("staging"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::remove_dir(linked_staging).unwrap();
}

#[test]
fn same_key_concurrency_exposes_no_partial_and_publishes_one_bundle() {
    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let first_materializer = ResolvedCacheMaterializer::new(store.clone()).with_test_hook({
        let release_rx = Arc::clone(&release_rx);
        move |_, _| {
            entered_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
            Ok(())
        }
    });
    let first_candidate = candidate.clone();
    let first_source = source.clone();
    let first = thread::spawn(move || {
        first_materializer.materialize(
            &first_candidate,
            &first_source,
            "image:model",
            &MaterializationCancellation::default(),
        )
    });
    entered_rx.recv().unwrap();
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_none());
    assert!(!store.bundle_path(&candidate.cache_key).unwrap().exists());
    assert_eq!(
        ResolvedCacheMaterializer::new(store.clone())
            .materialize(
                &candidate,
                &source,
                "image:model",
                &MaterializationCancellation::default(),
            )
            .unwrap(),
        MaterializationOutcome::Contended
    );
    release_tx.send(()).unwrap();
    assert!(matches!(
        first.join().unwrap().unwrap(),
        MaterializationOutcome::Published(_)
    ));
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_some());
}

#[test]
fn stale_staging_cleanup_requires_artifact_and_session_authority() {
    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let reservation = match store.reserve(&candidate, &source, "image:model").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("fixture reservation must acquire"),
    };
    let staging = reservation.staging_path().to_owned();
    std::fs::write(staging.join("partial"), b"partial").unwrap();
    assert_eq!(store.cleanup_stale_staging().unwrap(), 0);
    assert!(staging.exists());
    drop(reservation);
    assert_eq!(store.cleanup_stale_staging().unwrap(), 1);
    assert!(!staging.exists());

    let malformed = store.root().join("staging/not-a-cache-key");
    std::fs::create_dir(&malformed).unwrap();
    let external = scratch.path().join("external-sentinel");
    std::fs::write(&external, b"untouched").unwrap();
    assert_eq!(store.cleanup_stale_staging().unwrap(), 0);
    assert!(malformed.exists());
    assert_eq!(std::fs::read(external).unwrap(), b"untouched");
}

#[test]
fn crash_after_atomic_rename_stays_unavailable_and_next_reservation_republishes() {
    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let reservation = match store.reserve(&candidate, &source, "image:model").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("fixture reservation must acquire"),
    };
    reservation.prepare_for_materialization().unwrap();
    std::fs::write(
        reservation.staging_path().join("model.safetensors"),
        b"model-weights",
    )
    .unwrap();
    let orphan_bundle = reservation.bundle_path().unwrap();
    std::fs::rename(reservation.staging_path(), &orphan_bundle).unwrap();
    drop(reservation);

    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_none());
    assert!(orphan_bundle.join("model.safetensors").is_file());
    let recovered = store.recover().unwrap();
    assert_eq!(recovered[0].state, ResolvedCacheEntryState::Interrupted);
    assert!(orphan_bundle.join("model.safetensors").is_file());

    assert!(matches!(
        ResolvedCacheMaterializer::new(store.clone())
            .materialize(
                &candidate,
                &source,
                "image:model",
                &MaterializationCancellation::default(),
            )
            .unwrap(),
        MaterializationOutcome::Published(_)
    ));
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_some());
}

#[test]
fn success_candidate_schedules_nonblocking_and_materializes_only_when_idle() {
    let scratch = TempDir::new().unwrap();
    let (source, direct_candidate) = flat_fixture(&scratch);
    let library = ArtifactSourceLibrary::new(&source).unwrap();
    let resolver = ModelArtifactResolver::new(library);
    let lease = resolver
        .acquire_runtime_lease(&Arc::new(direct_candidate.artifact.clone()))
        .unwrap();
    let candidate = lease.mark_success();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let scheduler = ResolvedCachePromotionScheduler::new(
        enabled_policy(),
        ResolvedCacheMaterializer::new(store.clone()),
        2,
    )
    .unwrap();
    assert_eq!(
        scheduler
            .schedule(candidate.clone(), source.clone(), "image:model".to_owned())
            .unwrap(),
        PromotionScheduleOutcome::Enqueued
    );
    assert_eq!(scheduler.queued_len(), 1);
    assert!(!store.bundle_path(&candidate.cache_key).unwrap().exists());
    assert_eq!(
        scheduler.run_next_if_idle(false).unwrap(),
        IdlePromotionOutcome::NotIdle
    );
    assert!(!store.bundle_path(&candidate.cache_key).unwrap().exists());
    assert!(matches!(
        scheduler.run_next_if_idle(true).unwrap(),
        IdlePromotionOutcome::Materialized(MaterializationOutcome::Published(_))
    ));
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_some());
}

#[test]
fn active_scheduler_cancellation_interrupts_and_cleans_the_reservation() {
    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let materializer = ResolvedCacheMaterializer::new(store.clone()).with_test_hook({
        let release_rx = Arc::clone(&release_rx);
        move |_, _| {
            entered_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
            Ok(())
        }
    });
    let scheduler =
        ResolvedCachePromotionScheduler::new(enabled_policy(), materializer, 1).unwrap();
    assert_eq!(
        scheduler
            .schedule(candidate.clone(), source, "image:model".to_owned())
            .unwrap(),
        PromotionScheduleOutcome::Enqueued
    );
    let worker = {
        let scheduler = scheduler.clone();
        thread::spawn(move || scheduler.run_next_if_idle(true))
    };
    entered_rx.recv().unwrap();
    assert!(scheduler.cancel_active());
    release_tx.send(()).unwrap();
    assert_eq!(
        worker.join().unwrap().unwrap(),
        IdlePromotionOutcome::Materialized(MaterializationOutcome::Cancelled)
    );
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_none());
    assert!(std::fs::read_dir(store.root().join("staging"))
        .unwrap()
        .next()
        .is_none());
}

#[test]
fn concurrent_idle_runners_never_overwrite_the_active_promotion() {
    let scratch = TempDir::new().unwrap();
    let (source_a, candidate_a) = flat_fixture(&scratch);
    let second = TempDir::new().unwrap();
    let (source_b, candidate_b) = flat_fixture_at_revision(&second, REV_B);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let materializer = ResolvedCacheMaterializer::new(store).with_test_hook({
        let release_rx = Arc::clone(&release_rx);
        move |_, _| {
            entered_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
            Ok(())
        }
    });
    let scheduler =
        ResolvedCachePromotionScheduler::new(enabled_policy(), materializer, 2).unwrap();
    assert_eq!(
        scheduler
            .schedule(candidate_a, source_a, "image:model-a".to_owned())
            .unwrap(),
        PromotionScheduleOutcome::Enqueued
    );
    assert_eq!(
        scheduler
            .schedule(candidate_b, source_b, "video:model-b".to_owned())
            .unwrap(),
        PromotionScheduleOutcome::Enqueued
    );
    let first = {
        let scheduler = scheduler.clone();
        thread::spawn(move || scheduler.run_next_if_idle(true))
    };
    entered_rx.recv().unwrap();
    assert_eq!(
        scheduler.run_next_if_idle(true).unwrap(),
        IdlePromotionOutcome::NotIdle
    );
    assert_eq!(scheduler.queued_len(), 1);
    assert!(scheduler.cancel_active());
    release_tx.send(()).unwrap();
    assert_eq!(
        first.join().unwrap().unwrap(),
        IdlePromotionOutcome::Materialized(MaterializationOutcome::Cancelled)
    );
    assert_eq!(scheduler.queued_len(), 1);
    scheduler.shutdown();
}

#[test]
fn scheduler_is_bounded_coalescing_disabled_and_stoppable() {
    let scratch = TempDir::new().unwrap();
    let (source, candidate_a) = flat_fixture(&scratch);
    let second = TempDir::new().unwrap();
    let (source_b, candidate_b) = flat_fixture_at_revision(&second, REV_B);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let disabled = ResolvedCachePromotionScheduler::new(
        ResolvedCachePolicy::default(),
        ResolvedCacheMaterializer::new(store.clone()),
        1,
    )
    .unwrap();
    assert_eq!(
        disabled
            .schedule(
                candidate_a.clone(),
                source.clone(),
                "image:model".to_owned()
            )
            .unwrap(),
        PromotionScheduleOutcome::Disabled
    );
    let scheduler = ResolvedCachePromotionScheduler::new(
        enabled_policy(),
        ResolvedCacheMaterializer::new(store),
        1,
    )
    .unwrap();
    assert_eq!(
        scheduler
            .schedule(
                candidate_a.clone(),
                source.clone(),
                "image:model".to_owned()
            )
            .unwrap(),
        PromotionScheduleOutcome::Enqueued
    );
    assert_eq!(
        scheduler
            .schedule(candidate_a, source, "image:model".to_owned())
            .unwrap(),
        PromotionScheduleOutcome::Coalesced
    );
    assert_eq!(
        scheduler
            .schedule(candidate_b, source_b, "video:model".to_owned())
            .unwrap(),
        PromotionScheduleOutcome::Full
    );
    scheduler.shutdown();
    assert_eq!(scheduler.queued_len(), 0);
    assert_eq!(
        scheduler.run_next_if_idle(true).unwrap(),
        IdlePromotionOutcome::Stopped
    );
}

#[cfg(unix)]
#[test]
fn materializer_copies_a_same_repository_hf_blob_symlink() {
    use std::os::unix::fs::symlink;

    let scratch = TempDir::new().unwrap();
    let (source, candidate) = flat_fixture(&scratch);
    let source_file = match &candidate.artifact.location {
        ArtifactLocation::SourceLibrary { root } => root.join("model.safetensors"),
        _ => unreachable!(),
    };
    let repository = ArtifactSourceLibrary::new(&source)
        .unwrap()
        .repository_root("SceneWorks/model")
        .unwrap();
    std::fs::remove_file(&source_file).unwrap();
    std::fs::create_dir(repository.join("blobs")).unwrap();
    std::fs::write(repository.join("blobs/model-blob"), b"model-weights").unwrap();
    symlink("../../blobs/model-blob", &source_file).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    assert!(matches!(
        ResolvedCacheMaterializer::new(store.clone())
            .materialize(
                &candidate,
                &source,
                "image:model",
                &MaterializationCancellation::default(),
            )
            .unwrap(),
        MaterializationOutcome::Published(_)
    ));
    assert_eq!(
        std::fs::read(
            store
                .bundle_path(&candidate.cache_key)
                .unwrap()
                .join("model.safetensors")
        )
        .unwrap(),
        b"model-weights"
    );
}

#[cfg(windows)]
#[test]
fn materializer_rejects_a_source_subpath_junction_without_touching_external_bytes() {
    let scratch = TempDir::new().unwrap();
    let library = scratch.path().join("source");
    let primary_identity = identity("SceneWorks/model", REV_A, "q8");
    let primary_snapshot = snapshot(&library, "SceneWorks/model", REV_A);
    let selected = primary_snapshot.join("selected");
    std::fs::create_dir(&selected).unwrap();
    std::fs::write(selected.join("model.safetensors"), b"model-weights").unwrap();
    let candidate = candidate(
        primary_snapshot,
        primary_identity.clone(),
        vec![member(
            ArtifactMemberRole::Primary,
            None,
            primary_identity,
            Some("q8"),
            "selected",
            "",
            &["model.safetensors"],
        )],
    );
    std::fs::remove_dir_all(&selected).unwrap();
    let external = scratch.path().join("external");
    std::fs::create_dir(&external).unwrap();
    std::fs::write(external.join("model.safetensors"), b"model-weights").unwrap();
    std::fs::write(external.join("sentinel"), b"must-survive").unwrap();
    let junction = WindowsDirectoryJunction::create(&selected, &external);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    assert!(ResolvedCacheMaterializer::new(store.clone())
        .materialize(
            &candidate,
            &library,
            "image:model",
            &MaterializationCancellation::default(),
        )
        .is_err());
    assert!(store
        .lookup_complete(&candidate.cache_key)
        .unwrap()
        .is_none());
    assert_eq!(
        std::fs::read(external.join("model.safetensors")).unwrap(),
        b"model-weights"
    );
    assert_eq!(
        std::fs::read(external.join("sentinel")).unwrap(),
        b"must-survive"
    );
    junction.remove();
}
