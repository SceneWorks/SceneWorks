use super::*;
use crate::model_artifacts::{
    ActiveArtifactLeaseRegistry, ArtifactAvailability, ArtifactCompleteness, ArtifactFile,
    ArtifactIdentity, ArtifactMemberRole, ArtifactProvenance, ArtifactSourceLibrary,
    ResolvedBundleClosure, ResolvedBundleMember, MODEL_ARTIFACT_CONTRACT_VERSION,
};
use fs2::FileExt;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
    let library = ArtifactSourceLibrary::new(root).unwrap();
    let snapshot = library
        .repository_root("owner/model")
        .unwrap()
        .join("snapshots")
        .join(revision);
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("weights.bin"), b"model-weights").unwrap();
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
        location: ArtifactLocation::SourceLibrary { root: snapshot },
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

fn source_snapshot(candidate: &PromotionCandidate) -> &Path {
    match &candidate.artifact.location {
        ArtifactLocation::SourceLibrary { root } => root,
        ArtifactLocation::ResolvedLocal { .. } => panic!("fixture candidate is source-backed"),
    }
}

/// Park a fixture entry in the non-live state `cleanup_stale_staging` requires before it will
/// touch a staging directory at all.
///
/// This is a PRECONDITION for the two directory-swap tests below, never their subject.
/// `ResolvedCacheReservation::drop` records the interruption on a best-effort basis: it *warns*
/// and leaves the entry `Materializing`, still owned by this live session, whenever its metadata
/// write cannot complete — a transient open/read/write failure under the descriptor and IO
/// pressure of a loaded hosted runner is the documented case (see the note on
/// `ResolvedCacheStore::read_eviction_marker`, which was made descriptor-cheap for exactly this
/// reason). `cleanup_stale_staging` then correctly *skips* the entry fail-closed and returns
/// `Ok(0)`, the swap hook never runs, and a test that only asserts "the sweep returned an error"
/// reads that skip as "the symlink was followed". That is what reddened
/// `stale_cleanup_directory_swap_never_follows_an_external_symlink` on the hosted macOS lane.
///
/// Forcing the transition here costs no coverage: the drop-marks-interrupted contract is owned by
/// `materialization::tests::stale_staging_cleanup_requires_artifact_and_session_authority` and by
/// `reservation_is_exclusive_unrelated_keys_progress_and_drop_interrupts`, which assert it
/// directly instead of inheriting it as setup.
fn force_interrupted_reservation(store: &ResolvedCacheStore, cache_key: &str) {
    store
        .update_metadata(cache_key, |metadata| {
            metadata.state = ResolvedCacheEntryState::Interrupted;
            metadata.reservation_id = None;
            metadata.reservation_owner = None;
            metadata.session_id = None;
            metadata.recovery_status = RecoveryStatus::InterruptedReservation;
            Ok(())
        })
        .expect("the fixture entry must be parkable as an interrupted reservation");
}

#[cfg(unix)]
#[test]
fn stale_cleanup_directory_swap_never_follows_an_external_symlink() {
    use std::os::unix::fs::symlink;

    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let candidate = source_candidate(&source, REVISION_A);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let reservation = match store.reserve(&candidate, &source, "fixture:model").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("fixture reservation must acquire"),
    };
    let staging = reservation.staging_path().to_owned();
    std::fs::write(staging.join("partial"), b"partial").unwrap();
    drop(reservation);
    force_interrupted_reservation(&store, &candidate.cache_key);

    let external = scratch.path().join("external");
    std::fs::create_dir(&external).unwrap();
    let sentinel = external.join("sentinel");
    std::fs::write(&sentinel, b"untouched").unwrap();
    let swapped_staging = staging.clone();
    let external_target = external.clone();
    set_remove_entry_after_stat_hook(move || {
        std::fs::remove_dir_all(&swapped_staging).unwrap();
        symlink(&external_target, &swapped_staging).unwrap();
    });

    assert!(store.cleanup_stale_staging().is_err());
    // The swap has to be what the sweep refused. A sweep that never reached the removal at all
    // (it skipped the entry, or failed earlier) also returns without deleting the sentinel, so
    // without this the assertions below pass for the wrong reason: the hook only runs inside
    // `remove_entry_at`, so a staging path that is now a symlink is the witness that the sweep
    // stood on the swapped path and still refused to follow it.
    assert!(
        std::fs::symlink_metadata(&staging)
            .unwrap()
            .file_type()
            .is_symlink(),
        "the removal must have reached the swapped staging path"
    );
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"untouched");
    assert!(external.is_dir());
    std::fs::remove_file(staging).unwrap();
}

/// The local-tier scan validates on paths and sizes; the LEASE still re-hashes (sc-19712 F-3).
///
/// The scan is read work on two hot surfaces — the API's per-submission preflight and its catalog
/// build, and the worker's pre-loader guard. Re-hashing there made the cost of *asking* what the
/// cache holds proportional to the whole cache, so populating the cache made every job submission
/// slower than the load the cache exists to save (measured: 929.6 s for one 5.57 GB bundle).
///
/// A same-length byte alteration is the one thing only a content hash can see, so it separates the
/// two modes exactly: the scan must still offer the entry, and `acquire_complete` must still refuse
/// it. Both halves are load-bearing — the first pins the scan onto the cheap mode, the second pins
/// the boundary that makes the cheap mode safe.
#[test]
fn the_local_tier_scan_skips_content_hashing_while_the_lease_boundary_still_refuses_altered_bytes()
{
    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    let data = scratch.path().join("data");
    std::fs::create_dir(&source).unwrap();
    let store = ResolvedCacheStore::open(&data).unwrap();
    let candidate = hub_layout_candidate(&source, REVISION_A);
    let materializer = ResolvedCacheMaterializer::new(store.clone());
    let published = match materializer
        .materialize(
            &candidate,
            &source,
            "fixture:model",
            &MaterializationCancellation::default(),
        )
        .unwrap()
    {
        MaterializationOutcome::Published(metadata) => *metadata,
        other => panic!("fixture bundle was not published: {other:?}"),
    };
    assert_eq!(
        ResolvedCacheStore::valid_local_artifacts(&data).artifacts,
        vec![published.artifact.clone()],
        "the published entry must be offered before it is tampered with"
    );

    // Same length, different bytes. Sizes and paths still match the recorded closure, so only a
    // content re-hash can tell the difference.
    let bundle_file = store
        .bundle_path(&candidate.cache_key)
        .unwrap()
        .join(&published.artifact.closure.members[0].destination)
        .join("weights.bin");
    assert_eq!(std::fs::read(&bundle_file).unwrap(), b"model-weights");
    std::fs::write(&bundle_file, b"MODEL-WEIGHTS").unwrap();

    // The SCAN does not notice, and must not: this is the cheap mode, and paying for the hash here
    // is what F-3 measured.
    let scanned = ResolvedCacheStore::valid_local_artifacts(&data);
    assert_eq!(
        scanned.artifacts,
        vec![published.artifact],
        "the local-tier scan must judge on paths and sizes, not by re-hashing every bundle"
    );
    assert!(scanned.rejections.is_empty());

    // The LEASE does notice. This is what makes the cheap scan safe: the altered bundle is refused
    // before any bytes reach a runtime, and the caller falls back to the source tier.
    let registry = ActiveArtifactLeaseRegistry::default();
    let resolver = resolver(&source, registry.clone());
    assert!(
        store
            .acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
            .is_err(),
        "the load boundary must re-hash and refuse altered bytes"
    );
    assert_eq!(registry.active_lease_count(&candidate.cache_key), 0);
    // The refusal has to come from the check BEFORE the lease is issued, not from the usage stamp
    // that follows it. The stamp also validates at full strength, so it would mask a downgraded
    // boundary while leaking the session record the lease had already written — this asserts the
    // side-effect-free refusal that isolates the boundary check itself.
    let session_records = std::fs::read_dir(store.root().join("sessions").join(store.session_id()))
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
                .count()
        })
        .unwrap_or(0);
    assert_eq!(
        session_records, 0,
        "the lease boundary must refuse altered bytes before it writes a session record"
    );
}

/// sc-21534 — one `acquire_complete` call re-reads each closure file exactly ONCE.
///
/// The load boundary is the single full-strength check that keeps every cheap paths-and-sizes
/// read safe, but before this pin it hashed the closure FOUR times per load: twice inside
/// `validate_complete_metadata` (the shape check duplicated the content check), once more in
/// `acquire_runtime_lease`, and once more in the usage-stamp journal write — all under the same
/// locks in the same call, so the extras closed no window. On a 33 GB bundle that multiple turned
/// one ~70 s verify into the ~4.5-minute pass that outlasted the API's 90 s stale-worker sweep.
/// Stated as an operation count, like
/// `the_availability_decision_never_re_reads_the_artifact_it_admits`: a duration cannot see this
/// on a small fixture.
#[test]
fn the_load_boundary_hashes_the_closure_exactly_once() {
    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    let data = scratch.path().join("data");
    std::fs::create_dir(&source).unwrap();
    let store = ResolvedCacheStore::open(&data).unwrap();
    let candidate = hub_layout_candidate(&source, REVISION_A);
    let materializer = ResolvedCacheMaterializer::new(store.clone());
    let published = match materializer
        .materialize(
            &candidate,
            &source,
            "fixture:model",
            &MaterializationCancellation::default(),
        )
        .unwrap()
    {
        MaterializationOutcome::Published(metadata) => *metadata,
        other => panic!("fixture bundle was not published: {other:?}"),
    };
    let hashed_files = published
        .artifact
        .closure
        .members
        .iter()
        .flat_map(|member| member.files.iter())
        .filter(|file| file.sha256.is_some())
        .count() as u64;
    assert!(
        hashed_files >= 1,
        "the fixture must record content digests or the count below is vacuous"
    );

    let registry = ActiveArtifactLeaseRegistry::default();
    let resolver = resolver(&source, registry.clone());
    let (lease, hashes) = crate::model_artifacts::observe_content_hashes(|| {
        store.acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
    });
    assert!(
        lease.unwrap().is_some(),
        "the boundary must still issue the lease it is charging one hash pass for"
    );
    assert_eq!(
        hashes, hashed_files,
        "acquiring a complete entry must hash each closure file exactly once — the shape check, \
         the runtime lease, and the usage stamp must all reuse the one full-strength verdict"
    );
}

/// The load boundary hashes once per CHANGE, not once per load — and still refuses a bundle that
/// changed.
///
/// Hashing exactly once per acquisition (pinned above) still charged every job the full read: a
/// 19 GB bundle measured ~61 s of SHA-256 between the claim and "Preparing N image(s)", re-proving
/// bytes that had not moved since the previous job proved them. The verdict is now recorded
/// alongside the closure's stat identity and reused while that identity holds.
///
/// All three properties are asserted together on purpose. "Reuses the verdict" and "still refuses
/// altered bytes" are the two halves of the same tradeoff, and a test that pinned only the first
/// would stay green if the receipt were reused unconditionally — which is precisely the defect
/// worth guarding against.
#[test]
fn a_recorded_verdict_is_reused_while_the_closure_is_untouched_and_dropped_when_it_changes() {
    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    let data = scratch.path().join("data");
    std::fs::create_dir(&source).unwrap();
    let store = ResolvedCacheStore::open(&data).unwrap();
    let candidate = hub_layout_candidate(&source, REVISION_A);
    let materializer = ResolvedCacheMaterializer::new(store.clone());
    let published = match materializer
        .materialize(
            &candidate,
            &source,
            "fixture:model",
            &MaterializationCancellation::default(),
        )
        .unwrap()
    {
        MaterializationOutcome::Published(metadata) => *metadata,
        other => panic!("fixture bundle was not published: {other:?}"),
    };
    let hashed_files = published
        .artifact
        .closure
        .members
        .iter()
        .flat_map(|member| member.files.iter())
        .filter(|file| file.sha256.is_some())
        .count() as u64;
    assert!(
        hashed_files >= 1,
        "the fixture must record content digests or the counts below are vacuous"
    );

    let registry = ActiveArtifactLeaseRegistry::default();
    let resolver = resolver(&source, registry.clone());

    // A freshly published entry carries no verdict, so the first acquisition pays the full read.
    let (lease, first) = crate::model_artifacts::observe_content_hashes(|| {
        store.acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
    });
    assert_eq!(
        first, hashed_files,
        "the first acquisition of a bundle with no recorded verdict must read every file"
    );
    drop(
        lease
            .unwrap()
            .expect("the first acquisition must issue a lease"),
    );

    // Nothing has touched the bundle, so the recorded verdict still describes it.
    let (lease, second) = crate::model_artifacts::observe_content_hashes(|| {
        store.acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
    });
    assert!(
        lease.unwrap().is_some(),
        "reusing the recorded verdict must still issue the lease"
    );
    assert_eq!(
        second, 0,
        "a bundle whose stat identity is unchanged must not be re-read — this is the whole cost \
         the receipt exists to remove"
    );

    // Same length, different bytes: paths-and-sizes alone cannot see this, so only the receipt
    // being invalidated by the changed mtime routes it back to the full read that refuses it.
    let bundle_file = store
        .bundle_path(&candidate.cache_key)
        .unwrap()
        .join(&candidate.artifact.closure.members[0].destination)
        .join("weights.bin");
    assert_eq!(std::fs::read(&bundle_file).unwrap(), b"model-weights");
    std::fs::write(&bundle_file, b"MODEL-WEIGHTS").unwrap();
    // Stamped explicitly rather than relying on the write: a filesystem with coarse timestamp
    // granularity could otherwise reproduce the previous mtime and make this assertion a clock
    // race rather than a contract.
    std::fs::File::options()
        .write(true)
        .open(&bundle_file)
        .unwrap()
        .set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(60))
        .unwrap();

    let (refused, third) = crate::model_artifacts::observe_content_hashes(|| {
        store.acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
    });
    assert!(
        refused.is_err(),
        "a bundle altered after its verdict was recorded must still be refused at the load boundary"
    );
    assert_eq!(
        third, hashed_files,
        "a stat identity that moved must send the closure back through the full read, not reuse \
         the verdict recorded for the bytes it replaced"
    );
}

/// A standing verdict expires, so decay with no filesystem write behind it cannot hide forever.
///
/// Stat identity detects a WRITE. Bit rot, a failing enclosure, a drive rewriting a sector — none
/// of those move size or mtime, so the receipt would keep matching and the bytes would never be
/// read again. Before receipts every job re-read the bundle, which caught that class by accident;
/// the max-age is what keeps catching it on purpose. This matters because the cache fronts a model
/// library that can live on external USB media, where silent decay is the plausible failure and a
/// timestamp-restoring writer is not.
///
/// The receipt is aged by rewriting `verifiedAt` rather than by waiting, so this asserts the
/// contract rather than sleeping through it.
#[test]
fn a_standing_verdict_expires_and_sends_the_closure_back_through_the_full_read() {
    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    let data = scratch.path().join("data");
    std::fs::create_dir(&source).unwrap();
    let store = ResolvedCacheStore::open(&data).unwrap();
    let candidate = hub_layout_candidate(&source, REVISION_A);
    let materializer = ResolvedCacheMaterializer::new(store.clone());
    let published = match materializer
        .materialize(
            &candidate,
            &source,
            "fixture:model",
            &MaterializationCancellation::default(),
        )
        .unwrap()
    {
        MaterializationOutcome::Published(metadata) => *metadata,
        other => panic!("fixture bundle was not published: {other:?}"),
    };
    let hashed_files = published
        .artifact
        .closure
        .members
        .iter()
        .flat_map(|member| member.files.iter())
        .filter(|file| file.sha256.is_some())
        .count() as u64;

    let resolver = resolver(&source, ActiveArtifactLeaseRegistry::default());
    drop(
        store
            .acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
            .unwrap()
            .expect("the fixture entry must be acquirable"),
    );

    // Backdate the standing verdict past the bound. Nothing about the bundle changes, so only the
    // age can send this back through the read.
    let sidecar = store
        .entry_path(&candidate.cache_key)
        .unwrap()
        .join("content.verification.json");
    let mut receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
    let verified_at = receipt["verifiedAt"].as_u64().expect("a recorded verdict");
    receipt["verifiedAt"] =
        serde_json::json!(verified_at.saturating_sub(CONTENT_RECEIPT_MAX_AGE_SECONDS + 1));
    std::fs::write(&sidecar, serde_json::to_vec(&receipt).unwrap()).unwrap();

    let (lease, hashes) = crate::model_artifacts::observe_content_hashes(|| {
        store.acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
    });
    assert!(
        lease.unwrap().is_some(),
        "an expired verdict re-proves the bundle; it does not refuse an intact one"
    );
    assert_eq!(
        hashes, hashed_files,
        "a verdict older than the max age must not be redeemed — the bytes are read again"
    );

    // And the re-proof stands on its own: the refreshed verdict is redeemable immediately.
    let (lease, after_refresh) = crate::model_artifacts::observe_content_hashes(|| {
        store.acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
    });
    assert!(lease.unwrap().is_some());
    assert_eq!(
        after_refresh, 0,
        "re-proving must stamp a fresh verdict, not leave the entry hashing on every acquisition"
    );
}

/// The recorded verdict must stay OUT of the metadata journal, so a downgrade still reads the store.
///
/// `ResolvedCacheMetadata` is `deny_unknown_fields` and the journal keeps two slots. A receipt field
/// on it would therefore be invisible for one acquisition and fatal after two: both slots would
/// carry a key the previous build rejects, every entry would read as `BOTH_METADATA_SLOTS_CORRUPT`,
/// and corrupt is the classification manual removal is allowed to clear — so rolling a release back
/// would invite deleting and re-copying a cache that was never damaged.
///
/// Asserted on the raw journal bytes rather than through a round-trip, because a round-trip in this
/// process uses this build's struct and so cannot see the field the OTHER build would choke on.
#[test]
fn the_recorded_verdict_lives_outside_the_metadata_journal() {
    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    let data = scratch.path().join("data");
    std::fs::create_dir(&source).unwrap();
    let store = ResolvedCacheStore::open(&data).unwrap();
    let candidate = hub_layout_candidate(&source, REVISION_A);
    let materializer = ResolvedCacheMaterializer::new(store.clone());
    match materializer
        .materialize(
            &candidate,
            &source,
            "fixture:model",
            &MaterializationCancellation::default(),
        )
        .unwrap()
    {
        MaterializationOutcome::Published(_) => {}
        other => panic!("fixture bundle was not published: {other:?}"),
    }

    let resolver = resolver(&source, ActiveArtifactLeaseRegistry::default());
    // Twice: one acquisition could leave the older slot innocent, which is exactly the shape that
    // would let this regression ship looking harmless.
    for _ in 0..2 {
        drop(
            store
                .acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
                .unwrap()
                .expect("the fixture entry must be acquirable"),
        );
    }

    let entry = store.entry_path(&candidate.cache_key).unwrap();
    let sidecar = entry.join("content.verification.json");
    let recorded = std::fs::read_to_string(&sidecar)
        .expect("a successful acquisition must record its verdict somewhere");
    assert!(
        recorded.contains("verifiedAt"),
        "the sidecar must be where the verdict actually lands"
    );

    for slot in 0..=1 {
        let path = entry.join(format!("metadata.{slot}.json"));
        let Ok(journal) = std::fs::read_to_string(&path) else {
            continue;
        };
        assert!(
            !journal.contains("verifiedAt") && !journal.contains("contentVerification"),
            "metadata.{slot}.json carries the content verdict; a build without that field \
             rejects the whole journal and reports a healthy entry as corrupt"
        );
    }
}

/// A submission's availability read must not park behind the sweep's expensive verification
/// (sc-19712, the coupled residue).
///
/// `recover()` — the startup half of every maintenance checkpoint — begins with `enumerate()`,
/// which validates at FULL strength. While that judgement ran under each entry's exclusive
/// metadata lock, `valid_local_artifacts` (the provider behind `preflight_payload_model_sources`,
/// and so behind every job submission) blocked on the same lock: one submission measured 33.2 s
/// and the next had not returned after ~11 minutes, API at 0 % CPU on `flock`, worker at 99 % CPU
/// hashing. F-2 gave the status endpoint this treatment; the submission path did not get it.
///
/// Observed from inside the listing's own validation window, so it pins the ordering rather than a
/// duration. The probe is non-blocking on purpose: a `try_lock` reports a held lock, where the
/// blocking acquire the real reader uses would simply hang the suite.
#[test]
fn a_full_strength_listing_never_parks_the_availability_read_behind_its_verification() {
    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    let data = scratch.path().join("data");
    std::fs::create_dir(&source).unwrap();
    let store = ResolvedCacheStore::open(&data).unwrap();
    let materializer = ResolvedCacheMaterializer::new(store.clone());
    for revision in [REVISION_A, REVISION_B] {
        let candidate = hub_layout_candidate(&source, revision);
        match materializer
            .materialize(
                &candidate,
                &source,
                "fixture:model",
                &MaterializationCancellation::default(),
            )
            .unwrap()
        {
            MaterializationOutcome::Published(_) => {}
            other => panic!("fixture bundle was not published: {other:?}"),
        }
    }
    let lock_paths = std::fs::read_dir(store.root().join("entries"))
        .unwrap()
        .flatten()
        .map(|entry| {
            let digest = entry.file_name().to_str().unwrap().to_owned();
            store
                .root()
                .join("locks")
                .join(format!("{digest}.metadata.lock"))
        })
        .collect::<Vec<_>>();
    assert_eq!(lock_paths.len(), 2, "the fixture cache holds two entries");

    let observed = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(AtomicBool::new(false));
    let probe = {
        let observed = Arc::clone(&observed);
        let seen = Arc::clone(&seen);
        let lock_paths = lock_paths.clone();
        move || {
            seen.store(true, Ordering::SeqCst);
            observed.store(
                lock_paths.iter().all(|path| {
                    let handle = open_lock_file(path).unwrap();
                    FileExt::try_lock_exclusive(&handle).is_ok()
                }),
                Ordering::SeqCst,
            );
        }
    };
    set_listing_validation_observer(probe);

    // The startup checkpoint's own entry point, at full strength.
    let recovered = store.recover().unwrap();
    assert_eq!(recovered.len(), 2, "recovery still judges every entry");
    assert!(seen.load(Ordering::SeqCst), "the observer must have run");
    assert!(
        observed.load(Ordering::SeqCst),
        "no entry's metadata lock may be held while a listing verifies bundle contents, or the \
         availability read behind every job submission blocks for the whole checkpoint"
    );

    // And the read the submission path actually performs still answers, with both entries intact.
    assert_eq!(
        ResolvedCacheStore::valid_local_artifacts(&data)
            .artifacts
            .len(),
        2,
        "the availability read still offers both published entries"
    );
}

#[cfg(windows)]
#[test]
fn stale_cleanup_directory_swap_never_follows_an_external_junction() {
    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let candidate = source_candidate(&source, REVISION_A);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let reservation = match store.reserve(&candidate, &source, "fixture:model").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("fixture reservation must acquire"),
    };
    let staging = reservation.staging_path().to_owned();
    std::fs::write(staging.join("partial"), b"partial").unwrap();
    drop(reservation);
    force_interrupted_reservation(&store, &candidate.cache_key);

    let external = scratch.path().join("external");
    std::fs::create_dir(&external).unwrap();
    let sentinel = external.join("sentinel");
    std::fs::write(&sentinel, b"untouched").unwrap();
    let swapped_staging = staging.clone();
    let external_target = external.clone();
    set_windows_directory_after_validation_hook(move || {
        std::fs::remove_dir_all(&swapped_staging).unwrap();
        let junction = WindowsDirectoryJunction::create(&swapped_staging, &external_target);
        std::mem::forget(junction);
    });

    assert!(store.cleanup_stale_staging().is_err());
    // The swap has to be what the sweep refused. A sweep that never reached the removal at all
    // (it skipped the entry, or failed earlier) also returns without deleting the sentinel, so
    // without this the assertions below pass for the wrong reason: the hook only runs inside
    // `windows_confined_directory`, so a staging path that is now a reparse point is the witness
    // that the sweep stood on the swapped path and still refused to follow it.
    assert!(
        metadata_is_reparse_point(&std::fs::symlink_metadata(&staging).unwrap()),
        "the removal must have reached the swapped staging path"
    );
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"untouched");
    assert!(external.is_dir());
    std::fs::remove_dir(staging).unwrap();
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
    std::fs::copy(
        source_snapshot(candidate).join("weights.bin"),
        bundle.join("weights.bin"),
    )
    .unwrap();
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

/// A source candidate laid out the way a promotion writes it: the bundle mirrors the source
/// library, which is what makes a published entry loadable by every runtime (sc-19707).
fn hub_layout_candidate(root: &Path, revision: &str) -> PromotionCandidate {
    let mut candidate = source_candidate(root, revision);
    let mut members = candidate.artifact.closure.members.clone();
    members[0].destination =
        crate::model_artifacts::local_preference::hub_cache_member_destination(
            &members[0].source.repository,
            &members[0].source.revision,
            Path::new(""),
        )
        .unwrap();
    candidate.artifact.closure = ResolvedBundleClosure::new(members).unwrap();
    candidate.cache_key = candidate.artifact.cache_key().unwrap();
    candidate
}

/// The one provider both the API preflight and the worker's pre-loader guard read. It must offer
/// ONLY entries a runtime can actually load: published, verifiable, and stored in the source
/// library layout. Everything else stays on the source tier instead of poisoning the answer.
#[test]
fn valid_local_artifacts_offers_only_loadable_published_entries() {
    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    let data = scratch.path().join("data");
    std::fs::create_dir(&source).unwrap();
    assert!(ResolvedCacheStore::valid_local_artifacts(&data)
        .artifacts
        .is_empty());

    let store = ResolvedCacheStore::open(&data).unwrap();
    // An uninitialized-but-open cache offers nothing.
    assert!(ResolvedCacheStore::valid_local_artifacts(&data)
        .artifacts
        .is_empty());

    // An entry still materializing is never a candidate: its bytes are not published yet.
    let in_flight = hub_layout_candidate(&source, REVISION_B);
    let reservation = match store.reserve(&in_flight, &source, "fixture:model").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("fixture reservation must acquire"),
    };
    assert!(ResolvedCacheStore::valid_local_artifacts(&data)
        .artifacts
        .is_empty());
    drop(reservation);
    assert!(ResolvedCacheStore::valid_local_artifacts(&data)
        .artifacts
        .is_empty());

    // A published hub-layout bundle IS a candidate.
    let candidate = hub_layout_candidate(&source, REVISION_A);
    let materializer = ResolvedCacheMaterializer::new(store.clone());
    let published = match materializer
        .materialize(
            &candidate,
            &source,
            "fixture:model",
            &MaterializationCancellation::default(),
        )
        .unwrap()
    {
        MaterializationOutcome::Published(metadata) => *metadata,
        other => panic!("fixture bundle was not published: {other:?}"),
    };
    let offered = ResolvedCacheStore::valid_local_artifacts(&data);
    assert_eq!(offered.artifacts, vec![published.artifact.clone()]);
    assert!(offered.rejections.is_empty());

    // A published bundle stored in any other layout cannot be handed to the shared snapshot
    // resolvers, so it is not offered even though the store considers it complete — and it is
    // REPORTED as an unsupported shape rather than dropped, so the guard can name it. Dropping it
    // here is what made the local-tier-unsupported class unreachable for the case it names.
    let legacy = source_candidate(&scratch.path().join("legacy-source"), REVISION_B);
    match materializer
        .materialize(
            &legacy,
            &scratch.path().join("legacy-source"),
            "fixture:legacy",
            &MaterializationCancellation::default(),
        )
        .unwrap()
    {
        MaterializationOutcome::Published(_) => {}
        other => panic!("legacy fixture was not published: {other:?}"),
    }
    let scanned = ResolvedCacheStore::valid_local_artifacts(&data);
    assert_eq!(scanned.artifacts, vec![published.artifact]);
    assert_eq!(scanned.rejections.len(), 1, "{:?}", scanned.rejections);
    assert_eq!(scanned.rejections[0].revision, REVISION_B);
    assert!(
        scanned.rejections[0]
            .reason
            .contains("source-library layout"),
        "{}",
        scanned.rejections[0].reason
    );
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
    assert!(validate_metadata_shape_with(
        &traversing,
        &cache_key_digest(&candidate_a.cache_key).unwrap(),
        ContentVerification::PathsAndSizesOnly
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
    std::fs::copy(
        source_snapshot(&candidate_b).join("weights.bin"),
        bundle.join("weights.bin"),
    )
    .unwrap();
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
    assert_eq!(
        store.enumerate().unwrap()[0].state,
        ResolvedCacheEntryState::Corrupt
    );
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
    assert_eq!(
        store.enumerate().unwrap()[0].state,
        ResolvedCacheEntryState::Corrupt
    );
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

/// sc-21534 — the older-slot recovery must not PIN a content-altered bundle.
///
/// The listings (including `recover()`'s opening `enumerate()`) validate on paths and sizes only,
/// so this branch is the last reader before an entry recovered from its older journal slot gets
/// re-pinned. A same-size alteration pinned here would be refused at every load yet never
/// evicted — a permanent disk leak — so the branch must prove the bytes at full strength first
/// and, on failure, leave the entry UNPINNED: still refused at the load boundary, but evictable,
/// so retention can reclaim it and the next use re-materializes clean.
#[test]
fn older_slot_recovery_refuses_to_pin_altered_bytes() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    std::fs::create_dir(&source).unwrap();
    // Materialize through the real promotion path: it enriches the recorded closure with content
    // digests post-copy, which is what lets the full-strength check see the alteration below (the
    // `source_candidate` shortcut records no digests, so it cannot exercise this branch).
    let candidate = hub_layout_candidate(&source, REVISION_A);
    let materializer = ResolvedCacheMaterializer::new(store.clone());
    match materializer
        .materialize(
            &candidate,
            &source,
            "fixture:model",
            &MaterializationCancellation::default(),
        )
        .unwrap()
    {
        MaterializationOutcome::Published(_) => {}
        other => panic!("fixture bundle was not published: {other:?}"),
    }
    // Unpin twice so BOTH journal slots record `artifact_pinned: false` — otherwise the older
    // slot the recovery falls back to still carries the materialization-era pin, and the assert
    // below could not tell an inherited pin from the recovery-added one under test.
    store.set_artifact_pin(&candidate.cache_key, false).unwrap();
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
    // Same length, different bytes: paths-and-sizes reads cannot see it, only the full-strength
    // check this branch must run before pinning can.
    let bundle_file = store
        .bundle_path(&candidate.cache_key)
        .unwrap()
        .join(&candidate.artifact.closure.members[0].destination)
        .join("weights.bin");
    assert_eq!(std::fs::read(&bundle_file).unwrap(), b"model-weights");
    std::fs::write(&bundle_file, b"MODEL-WEIGHTS").unwrap();

    let recovered = store.recover().unwrap();
    assert_eq!(recovered.len(), 1);
    let metadata = recovered[0].metadata.as_ref().unwrap();
    assert!(
        !metadata.effective_pin,
        "an older-slot recovery of altered bytes must never pin the entry — pinned it would be \
         refused at every load yet blocked from eviction forever"
    );
    assert_ne!(
        metadata.recovery_status,
        RecoveryStatus::RecoveredFromOlderSlot,
        "the recovery must not claim it recovered content it could not verify"
    );

    // The unpinned entry is still refused where it matters: the load boundary.
    let registry = ActiveArtifactLeaseRegistry::default();
    let resolver = resolver(&source, registry.clone());
    assert!(
        store
            .acquire_complete(&candidate.cache_key, &resolver, "runtime:image:model")
            .is_err(),
        "altered bytes stay refused at the load boundary"
    );
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

/// A managed cache may outlive manual filesystem changes. Foreign residue must not make the
/// cache unusable, while a damaged managed entry must remain visible for recovery without
/// withholding valid entries from cache reads or retention.
#[test]
fn foreign_entries_and_damaged_complete_entries_are_isolated_per_entry() {
    let scratch = TempDir::new().unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let source = scratch.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let damaged = source_candidate(&source, REVISION_A);
    let valid = source_candidate(&source, REVISION_B);
    make_complete(&store, &damaged, &source);
    make_complete(&store, &valid, &source);

    // Not an app-owned digest directory. Enumeration and retention must leave it untouched.
    let foreign = store.root().join("entries").join("foreign-import.bin");
    std::fs::write(&foreign, b"not a resolved-cache entry").unwrap();

    // This is a real app-owned entry whose journal says Complete, but its bundle is no longer
    // structurally valid. It must be shown as corrupt rather than poisoning the valid entry.
    let damaged_file = store
        .bundle_path(&damaged.cache_key)
        .unwrap()
        .join("weights.bin");
    std::fs::write(&damaged_file, b"short").unwrap();

    let summaries = store.enumerate().unwrap();
    assert_eq!(summaries.len(), 2, "foreign residue is not a cache entry");
    assert_eq!(
        summaries
            .iter()
            .find(|summary| summary.cache_key == damaged.cache_key)
            .unwrap()
            .state,
        ResolvedCacheEntryState::Corrupt
    );
    assert_eq!(
        summaries
            .iter()
            .find(|summary| summary.cache_key == valid.cache_key)
            .unwrap()
            .state,
        ResolvedCacheEntryState::Complete
    );
    assert!(
        store.lookup_complete(&valid.cache_key).unwrap().is_some(),
        "the damaged entry does not disable valid cache reads"
    );

    let recovered = store.recover().unwrap();
    assert_eq!(
        recovered
            .iter()
            .find(|summary| summary.cache_key == damaged.cache_key)
            .unwrap()
            .state,
        ResolvedCacheEntryState::Corrupt
    );
    assert!(
        store
            .entry_path(&damaged.cache_key)
            .unwrap()
            .join("corrupt.marker.json")
            .is_file(),
        "recovery persists the damaged entry's isolated corruption state"
    );

    let report = store
        .enforce_retention(
            &ResolvedCachePolicy {
                enabled: true,
                max_bytes: 1,
                inactivity_seconds: u64::MAX,
            },
            0,
        )
        .unwrap();
    assert!(
        report.failed.is_empty(),
        "foreign and damaged entries do not abort retention"
    );
    assert!(report
        .evicted
        .iter()
        .any(|entry| entry.cache_key == valid.cache_key));
    assert!(report.retained.iter().any(|entry| {
        entry.cache_key == damaged.cache_key && entry.hold == RetentionHold::RecoveryCandidate
    }));
    assert!(
        foreign.is_file(),
        "the cache never adopts or deletes foreign residue"
    );
}

/// Publication's pre-rename walk must never reopen staged files by path: the reopen is the
/// TOCTOU window (a parent junction swap between validation and reopen flushes the wrong file)
/// and a write-capable reopen cannot even be assumed possible (read-only staged content,
/// `FlushFileBuffers` denials on Windows). Content durability is bound to the copy-time write
/// handle in `copy_and_verify`; the walk only validates entries and syncs directories. A
/// read-only staged file therefore must not fail the walk — under both earlier reopen-flush
/// revisions this failed on Windows with `ERROR_ACCESS_DENIED`.
#[test]
fn publish_walk_accepts_read_only_staged_files_without_reopening_them() {
    let scratch = TempDir::new().unwrap();
    let staged = scratch.path().join("staged");
    std::fs::create_dir_all(staged.join("nested")).unwrap();
    std::fs::write(staged.join("model.safetensors"), b"weights").unwrap();
    std::fs::write(staged.join("nested/config.json"), b"{}").unwrap();
    for file in ["model.safetensors", "nested/config.json"] {
        let path = staged.join(file);
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&path, permissions).unwrap();
    }
    sync_tree(&staged).unwrap();
    for file in ["model.safetensors", "nested/config.json"] {
        let path = staged.join(file);
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        permissions.set_readonly(false);
        std::fs::set_permissions(&path, permissions).unwrap();
    }
}

#[test]
fn checked_enumeration_ignores_corrupt_entries() {
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
    assert_eq!(store.checked_verified_bytes().unwrap(), 0);
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
    assert!(std::fs::read_dir(store.root().join("staging"))
        .unwrap()
        .next()
        .is_none());

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
            ReservationOutcome::Acquired(reservation) => {
                std::fs::write(reservation.staging_path().join("partial"), b"partial").unwrap();
                reservation
            }
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

// ---------------------------------------------------------------------------------------------
// produced (derived) bundles — publication refusals (sc-20635)
// ---------------------------------------------------------------------------------------------

/// A derived reservation's staging directory is a WRITE surface with a declared shape, and
/// publication proves that shape rather than trusting the producer.
///
/// `CheckpointDerivativeOutputs::create` refuses an undeclared name and confines every create to
/// the staging root, so nothing reachable through the public derivative API can produce either of
/// the trees below — which is exactly why they are exercised here, at the boundary that is the
/// last line of defence if a producer ever writes to the staging path by some other means.
fn derived_plan(outputs: &[&str]) -> DerivedArtifactPlan {
    let identity = ArtifactIdentity::pinned(
        "sceneworks-checkpoint-derivative/checkpoint",
        REVISION_A,
        "derived",
    )
    .unwrap();
    let member = ResolvedBundleMember {
        role: ArtifactMemberRole::Primary,
        component_id: Some("derived-index".to_owned()),
        source: identity.clone(),
        tier: None,
        source_subpath: PathBuf::new(),
        destination: PathBuf::new(),
        files: outputs
            .iter()
            .map(|output| ArtifactFile::new(output).unwrap())
            .collect(),
    };
    DerivedArtifactPlan::new(identity, vec![member]).unwrap()
}

fn derived_reservation(
    store: &ResolvedCacheStore,
    input: &Path,
    plan: &DerivedArtifactPlan,
) -> ResolvedCacheReservation {
    let reservation = match store
        .reserve_derived(
            plan,
            input,
            "linked/root-fixture/dit.safetensors",
            "fixture:model",
        )
        .unwrap()
    {
        ReservationOutcome::Acquired(reservation) => *reservation,
        _ => panic!("fixture derived reservation must acquire"),
    };
    reservation.prepare_for_materialization().unwrap();
    reservation
}

fn journal_slots(store: &ResolvedCacheStore, cache_key: &str) -> Vec<PathBuf> {
    let digest = cache_key_digest(cache_key).unwrap();
    let entry = store.root().join("entries").join(digest);
    (0..=1)
        .map(|slot| entry.join(format!("metadata.{slot}.json")))
        .filter(|path| path.exists())
        .collect()
}

/// The two fields a derived entry added (`production`, `derivedFrom`) widen the journal document,
/// and `ResolvedCacheMetadata` is `deny_unknown_fields`, so a binary that predates them cannot
/// decode one. sc-20635 therefore writes a derived journal at
/// [`RESOLVED_CACHE_DERIVED_STORE_VERSION`] and leaves a source copy at
/// [`RESOLVED_CACHE_STORE_VERSION`] — the version is a property of the DOCUMENT, not of the store.
///
/// That split is what makes the bump safe to land on a warm cache: an existing v1 source copy is
/// unchanged on disk and still readable. Bumping the store-wide constant instead would have
/// invalidated every entry already there, and retention cannot reclaim an entry it cannot read, so
/// the whole cache would have been stranded as unreclaimable recovery candidates.
///
/// Failing mutations: make `ResolvedCacheProduction::store_version` return
/// `RESOLVED_CACHE_STORE_VERSION` for `Derived` (the derived assertion goes red) or
/// `RESOLVED_CACHE_DERIVED_STORE_VERSION` for `SourceCopy` (the byte-shape assertion goes red).
#[test]
fn only_a_derived_journal_declares_the_widened_document_version() {
    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();

    let candidate = source_candidate(&source, REVISION_A);
    let copy = match store.reserve(&candidate, &source, "fixture:model").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("fixture reservation must acquire"),
    };
    let copy_key = candidate.cache_key.clone();
    drop(copy);
    for path in journal_slots(&store, &copy_key) {
        let body: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            body["schemaVersion"],
            json!(RESOLVED_CACHE_STORE_VERSION),
            "a source copy stays at the version every older build already reads: {path:?}"
        );
        // Byte-shape, not just version: both new keys are absent, so the document a pre-sc-20635
        // build wrote and the one this build writes are the same document.
        assert!(
            body["metadata"].get("production").is_none()
                && body["metadata"].get("derivedFrom").is_none(),
            "a source copy must not widen the journal: {body}"
        );
    }

    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let plan = derived_plan(&["index.json"]);
    let derived = derived_reservation(&store, &input, &plan);
    let derived_key = plan.cache_key().unwrap();
    drop(derived);
    let derived_slots = journal_slots(&store, &derived_key);
    assert!(!derived_slots.is_empty(), "the derived entry has a journal");
    for path in &derived_slots {
        let body: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(
            body["schemaVersion"],
            json!(RESOLVED_CACHE_DERIVED_STORE_VERSION),
            "a derived journal declares the version its extra fields need: {path:?}"
        );
        assert_eq!(body["metadata"]["production"], json!("derived"));
    }

    // And a journal from a NEWER writer is a version refusal, not "your cache is corrupt" — the
    // classification that manual removal is allowed to clear.
    for path in &derived_slots {
        let mut body: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        body["schemaVersion"] = json!(RESOLVED_CACHE_DERIVED_STORE_VERSION + 1);
        std::fs::write(path, serde_json::to_vec(&body).unwrap()).unwrap();
    }
    let error = store.lookup_complete(&derived_key).unwrap_err();
    assert!(
        error.to_string().contains(UNSUPPORTED_JOURNAL_VERSION),
        "{error}"
    );
    assert!(
        !error.is_unrecoverable_metadata(),
        "a forward-version journal is not proven-corrupt residue: {error}"
    );
}

/// Late-delta review finding 2: a CONTENDED lock wait must observe cancellation, not block until
/// the holder finishes. The holder below keeps the metadata lock across a window far longer than
/// the whole test; the checked lookup must surface its own cancellation while the lock is still
/// held — a blocking acquisition would instead queue behind the holder and return the entry.
#[test]
fn a_canceled_lookup_waiting_on_a_contended_metadata_lock_returns_before_the_holder_releases() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&["index.json"]);
    let reservation = derived_reservation(&store, &input, &plan);
    let staging = reservation.staging_path().to_owned();
    std::fs::write(staging.join("index.json"), b"{}").unwrap();
    reservation
        .publish_produced(&plan)
        .expect("fixture publishes a complete entry");
    let key = plan.cache_key().unwrap();
    let digest = cache_key_digest(&key).unwrap();

    // Another operation holds the metadata lock across a long content verification window.
    let holder = store.lock_metadata(&digest).unwrap();
    let released = Arc::new(AtomicBool::new(false));
    let flag = released.clone();
    let holder_thread = thread::spawn(move || {
        thread::sleep(Duration::from_millis(1200));
        drop(holder);
        flag.store(true, Ordering::SeqCst);
    });

    // The pre-lock checkpoint passes; the CONTENDED wait must observe the flip and return the
    // cancellation itself, never queue silently behind the holder.
    let calls = std::cell::Cell::new(0u32);
    let started = std::time::Instant::now();
    let error = store
        .lookup_complete_with_check(&key, &|| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "[h3-preparation-canceled] checkpoint preparation canceled",
                ))
            }
        })
        .unwrap_err();
    assert!(
        error.to_string().contains("h3-preparation-canceled"),
        "the contended wait must surface the caller's own cancellation: {error}"
    );
    assert!(
        !released.load(Ordering::SeqCst),
        "the canceled wait returned while the lock was still held (after {:?})",
        started.elapsed()
    );
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "the wait must not queue behind the holder's full window"
    );
    holder_thread.join().unwrap();
    // The entry is untouched by the canceled read.
    assert!(store.lookup_complete(&key).unwrap().is_some());
}

/// Late-delta review finding 2, second half: a cancellation observed while WAITING for the FINAL
/// publication lock must not be followed by a complete receipt or `Complete` metadata. Driven at
/// the materializer-style completion, where the transaction's bundle is already in place at its
/// published path and the ONLY metadata lock left to take is the final one — so the flip (which
/// lands while this transaction is contending for it, before the holder releases) can be seen by
/// the cancellation-aware wait or the under-lock re-check alone. The still-owned unpublished
/// bundle is rolled back and nothing completes. Under the pre-lock-only shape, the blocking
/// acquisition outlives the flip and publishes — this regression is what goes red then.
#[test]
fn a_cancellation_observed_while_waiting_for_the_publication_lock_publishes_nothing() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&["index.json"]);
    let reservation = derived_reservation(&store, &input, &plan);
    let key = plan.cache_key().unwrap();
    let bundle = store.bundle_path(&key).unwrap();
    let digest = cache_key_digest(&key).unwrap();

    // This transaction's bundle already sits at its published path, exactly as a completed
    // rename would have left it: only the final publication lock separates it from Complete.
    std::fs::create_dir_all(&bundle).unwrap();
    std::fs::write(bundle.join("index.json"), b"{}").unwrap();
    let artifact = plan.artifact_at(&bundle, ArtifactState::Published);

    // The holder keeps the final publication lock contended; the flip lands well after the
    // in-place validation (a one-file fixture finishes in microseconds) and while this
    // transaction is still waiting, then the lock is released so a blocking acquisition would
    // proceed straight to publishing.
    let flip = Arc::new(AtomicBool::new(false));
    let released = Arc::new(AtomicBool::new(false));
    let holder = store.lock_metadata(&digest).unwrap();
    let trip = flip.clone();
    let released_by_holder = released.clone();
    let holder_thread = thread::spawn(move || {
        thread::sleep(Duration::from_millis(400));
        trip.store(true, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(300));
        drop(holder);
        released_by_holder.store(true, Ordering::SeqCst);
    });

    let error = reservation
        .record_complete_checked(
            artifact,
            &|| {
                if flip.load(Ordering::SeqCst) {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "[h3-preparation-canceled] checkpoint preparation canceled",
                    ))
                } else {
                    Ok(())
                }
            },
            true,
            None,
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("h3-preparation-canceled"),
        "the cancellation observed while waiting for the publication lock must surface, \
         not publish: {error}"
    );
    assert!(
        !released.load(Ordering::SeqCst),
        "the canceled publication wait must return before the lock holder finishes"
    );
    let artifact_probe =
        FileLock::try_shared(open_lock_file(&store.artifact_lock_path(&digest)).unwrap());
    assert!(
        artifact_probe.is_err(),
        "owned rollback retains its exclusive artifact lease until cleanup ends"
    );
    assert!(
        !store
            .entry_path(&key)
            .unwrap()
            .join("complete.receipt.json")
            .exists(),
        "a canceled waiter cannot publish while cleanup is pending"
    );
    let pending = store.canceled_cleanup_receipt(&key).unwrap().unwrap();
    assert_eq!(pending.state, CanceledCleanupState::Pending);
    assert!(
        pending.process_exit_before_cleanup_possible,
        "Pending explicitly records that process exit still requires recovery"
    );
    // Make the source unavailable before cleanup runs. Success must require NO source reads,
    // while the unrelated source bytes remain unchanged and outside the deletion boundary.
    std::fs::write(input.join("outside.bin"), b"unowned source bytes").unwrap();
    let hidden_source = scratch.path().join("source-still-owned-by-user");
    std::fs::rename(&input, &hidden_source).unwrap();
    holder_thread.join().unwrap();
    store.join_canceled_cleanups().unwrap();
    assert_eq!(
        store.canceled_cleanup_receipt(&key).unwrap().unwrap().state,
        CanceledCleanupState::Cleaned
    );
    assert_eq!(
        std::fs::read(hidden_source.join("outside.bin")).unwrap(),
        b"unowned source bytes"
    );
    std::fs::rename(&hidden_source, &input).unwrap();
    assert!(
        !bundle.exists(),
        "the still-owned unpublished bundle was rolled back"
    );
    assert_eq!(
        store.lookup_complete(&key).unwrap(),
        None,
        "no complete receipt and no Complete metadata may follow the canceled publication"
    );
    assert!(
        !store
            .entry_path(&key)
            .unwrap()
            .join("complete.receipt.json")
            .exists(),
        "cancellation must not leave a complete receipt either"
    );
    assert_eq!(
        store.read_metadata_locked(&digest).unwrap().state,
        ResolvedCacheEntryState::Interrupted,
        "the retained reservation records interruption after owned cleanup"
    );

    // The interruption is retryable, not a poisoned entry: a fresh reservation publishes.
    let reservation = derived_reservation(&store, &input, &plan);
    let staging = reservation.staging_path().to_owned();
    std::fs::write(staging.join("index.json"), b"{}").unwrap();
    let published = reservation
        .publish_produced_with_check(&plan, &|| Ok(()))
        .expect("retry publishes after the canceled publication");
    assert_eq!(published.state, ResolvedCacheEntryState::Complete);
}

#[test]
fn canceled_cleanup_deadline_is_recorded_pinned_and_recoverable() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&["index.json"]);
    let reservation = derived_reservation(&store, &input, &plan);
    let key = plan.cache_key().unwrap();
    let digest = cache_key_digest(&key).unwrap();
    let bundle = store.bundle_path(&key).unwrap();
    std::fs::create_dir_all(&bundle).unwrap();
    std::fs::write(bundle.join("index.json"), b"{}").unwrap();
    let holder = store.lock_metadata(&digest).unwrap();
    let started = std::time::Instant::now();
    reservation
        .schedule_canceled_cleanup(true, Duration::from_millis(40))
        .unwrap();
    store.join_canceled_cleanups().unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "tracked cleanup never waits forever for the holder"
    );
    let receipt = store.canceled_cleanup_receipt(&key).unwrap().unwrap();
    assert_eq!(receipt.state, CanceledCleanupState::NeedsRetry);
    assert!(receipt.detail.contains("deadline"));
    assert!(receipt.process_exit_before_cleanup_possible);
    assert!(
        FileLock::try_shared(open_lock_file(&store.artifact_lock_path(&digest)).unwrap()).is_err(),
        "the timed-out reservation remains pinned instead of becoming an ownerless bundle"
    );
    assert_eq!(
        store.read_metadata_locked(&digest).unwrap().state,
        ResolvedCacheEntryState::Materializing,
        "cleanup outstanding must not read as settled Interrupted"
    );
    assert!(!store
        .entry_path(&key)
        .unwrap()
        .join("complete.receipt.json")
        .exists());
    drop(holder);
    store.recover().unwrap();
    assert!(
        !bundle.exists(),
        "later recovery reclaims the verified owned unpublished bundle"
    );
    assert_eq!(
        store.canceled_cleanup_receipt(&key).unwrap().unwrap().state,
        CanceledCleanupState::Cleaned
    );
    assert_eq!(
        store.read_metadata_locked(&digest).unwrap().state,
        ResolvedCacheEntryState::Interrupted
    );
}

#[test]
fn closing_cleanup_owner_leaves_durable_recovery_evidence_for_a_new_session() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    let data = scratch.path().join("data");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&data).unwrap();
    let old_session = store.session_id().to_owned();
    let plan = derived_plan(&["index.json"]);
    let reservation = derived_reservation(&store, &input, &plan);
    let key = plan.cache_key().unwrap();
    let digest = cache_key_digest(&key).unwrap();
    let bundle = store.bundle_path(&key).unwrap();
    std::fs::create_dir_all(&bundle).unwrap();
    std::fs::write(bundle.join("index.json"), b"{}").unwrap();
    let holder = store.lock_metadata(&digest).unwrap();
    reservation
        .schedule_canceled_cleanup(true, Duration::from_millis(40))
        .unwrap();
    drop(store);
    let restarted = ResolvedCacheStore::open(&data).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !restarted.session_lock_is_acquirable(&old_session).unwrap()
        && std::time::Instant::now() < deadline
    {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        restarted.session_lock_is_acquirable(&old_session).unwrap(),
        "the bounded cleanup owner exits rather than pinning an orphan forever"
    );
    let receipt = restarted.canceled_cleanup_receipt(&key).unwrap().unwrap();
    assert_eq!(receipt.state, CanceledCleanupState::NeedsRetry);
    assert!(receipt.process_exit_before_cleanup_possible);
    drop(holder);
    restarted.recover().unwrap();
    assert!(!bundle.exists());
    assert_eq!(
        restarted
            .canceled_cleanup_receipt(&key)
            .unwrap()
            .unwrap()
            .state,
        CanceledCleanupState::Cleaned
    );
    assert!(restarted.lookup_complete(&key).unwrap().is_none());
}

#[test]
fn canceled_cleanup_does_not_remove_a_new_reservations_bundle() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&["index.json"]);
    let reservation = derived_reservation(&store, &input, &plan);
    let key = plan.cache_key().unwrap();
    let digest = cache_key_digest(&key).unwrap();
    let bundle = store.bundle_path(&key).unwrap();
    std::fs::create_dir_all(&bundle).unwrap();
    std::fs::write(bundle.join("index.json"), b"new owner's bytes").unwrap();
    let holder = store.lock_metadata(&digest).unwrap();
    reservation
        .schedule_canceled_cleanup(true, Duration::from_secs(1))
        .unwrap();
    // Deterministic ownership drift under the real lock; the cleanup must check this before any
    // deletion, even though it still holds the artifact pin from the former reservation.
    let mut metadata = store.read_metadata_locked(&digest).unwrap();
    metadata.reservation_id = Some(random_id().unwrap());
    store.write_metadata_unlocked(&digest, &metadata).unwrap();
    drop(holder);
    store.join_canceled_cleanups().unwrap();
    assert_eq!(
        std::fs::read(bundle.join("index.json")).unwrap(),
        b"new owner's bytes"
    );
    assert_eq!(
        store.canceled_cleanup_receipt(&key).unwrap().unwrap().state,
        CanceledCleanupState::OwnershipChanged
    );
    assert_eq!(
        store.read_metadata_locked(&digest).unwrap().reservation_id,
        metadata.reservation_id
    );
    assert!(!store
        .entry_path(&key)
        .unwrap()
        .join("complete.receipt.json")
        .exists());
}

/// The last cancellation checkpoint is UNDER the publication lock, not merely before it. This
/// real-lock interleaving trips the flag after a contended acquisition succeeds, immediately
/// before that checkpoint; a pre-lock-only mutant publishes and this assertion goes red.
#[test]
fn cancellation_at_the_publication_fence_never_writes_complete_metadata_or_receipt() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&["index.json"]);
    let reservation = derived_reservation(&store, &input, &plan);
    let key = plan.cache_key().unwrap();
    let digest = cache_key_digest(&key).unwrap();
    let bundle = store.bundle_path(&key).unwrap();
    std::fs::create_dir_all(&bundle).unwrap();
    std::fs::write(bundle.join("index.json"), b"{}").unwrap();
    let artifact = plan.artifact_at(&bundle, ArtifactState::Published);
    let canceled = Arc::new(AtomicBool::new(false));
    let trip = canceled.clone();
    FINAL_PUBLICATION_LOCK_ACQUIRED_HOOK.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(move || trip.store(true, Ordering::SeqCst)));
    });
    let holder = store.lock_metadata(&digest).unwrap();
    let holder_thread = thread::spawn(move || {
        thread::sleep(Duration::from_millis(80));
        drop(holder);
    });
    let error = reservation
        .record_complete_checked(
            artifact,
            &|| {
                if canceled.load(Ordering::SeqCst) {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "canceled at publication fence",
                    ))
                } else {
                    Ok(())
                }
            },
            true,
            None,
        )
        .unwrap_err();
    holder_thread.join().unwrap();
    assert!(
        error.to_string().contains("canceled at publication fence"),
        "{error}"
    );
    assert!(
        !bundle.exists(),
        "the under-lock rollback removes only its own unpublished bundle"
    );
    assert!(!store
        .entry_path(&key)
        .unwrap()
        .join("complete.receipt.json")
        .exists());
    assert_eq!(
        store.read_metadata_locked(&digest).unwrap().state,
        ResolvedCacheEntryState::Interrupted
    );
    assert!(store.lookup_complete(&key).unwrap().is_none());
}

#[test]
fn a_canceled_derivative_publication_removes_staging_and_publishes_nothing() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&["index.json"]);
    let reservation = derived_reservation(&store, &input, &plan);
    let staging = reservation.staging_path().to_owned();
    std::fs::write(staging.join("index.json"), b"{}").unwrap();

    // Cancel between the bounded measure/read passes, before any rename can publish.
    let calls = std::cell::Cell::new(0u32);
    let error = reservation
        .publish_produced_with_check(&plan, &|| {
            calls.set(calls.get() + 1);
            if calls.get() == 2 {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "cancel during measured publication",
                ))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cancel during measured publication"),
        "the cancellation itself surfaces, not a generic publication failure: {error}"
    );
    assert_eq!(calls.get(), 2, "stopped inside the first bounded file pass");
    assert!(
        !staging.exists(),
        "a canceled publication removes its own staging tree"
    );
    assert_eq!(
        store.lookup_complete(&plan.cache_key().unwrap()).unwrap(),
        None,
        "nothing partial or complete is published for a canceled derivative"
    );

    // The same reservation slot can be re-acquired and published to completion afterwards —
    // cancellation is a retryable interruption, not a poisoned entry.
    let reservation = derived_reservation(&store, &input, &plan);
    let staging = reservation.staging_path().to_owned();
    std::fs::write(staging.join("index.json"), b"{}").unwrap();
    let published = reservation
        .publish_produced_with_check(&plan, &|| Ok(()))
        .expect("retry publishes the complete derivative");
    assert_eq!(published.state, ResolvedCacheEntryState::Complete);
}

#[test]
fn a_produced_bundle_holding_an_undeclared_file_is_never_published() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&["index.json"]);
    let reservation = derived_reservation(&store, &input, &plan);
    let staging = reservation.staging_path().to_owned();

    std::fs::write(staging.join("index.json"), b"{}").unwrap();
    // Bytes the entry's own accounting would never count, so retention would under-measure the
    // entry and over-reclaim against it.
    std::fs::create_dir(staging.join("nested")).unwrap();
    std::fs::write(staging.join("nested/stowaway.bin"), b"extra").unwrap();

    let error = reservation.publish_produced(&plan).unwrap_err().to_string();
    assert!(
        error.contains("staged an undeclared file") && error.contains("stowaway.bin"),
        "{error}"
    );
    assert_eq!(
        store.lookup_complete(&plan.cache_key().unwrap()).unwrap(),
        None,
        "a refused publication leaves nothing complete behind"
    );
}

#[cfg(unix)]
#[test]
fn a_produced_bundle_whose_declared_output_is_a_symlink_is_never_published() {
    use std::os::unix::fs::symlink;

    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let outside = scratch.path().join("outside.bin");
    std::fs::write(&outside, b"bytes that live somewhere else").unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&["index.json"]);
    let reservation = derived_reservation(&store, &input, &plan);
    let staging = reservation.staging_path().to_owned();

    // A declared output that is a LINK publishes an entry whose bytes are not IN the entry: the
    // target can change or vanish under a lease, and removal would reclaim nothing.
    symlink(&outside, staging.join("index.json")).unwrap();

    let error = reservation.publish_produced(&plan).unwrap_err().to_string();
    assert!(
        error.contains("is linked, reparsed, or not a file") && error.contains("index.json"),
        "{error}"
    );
    assert_eq!(
        store.lookup_complete(&plan.cache_key().unwrap()).unwrap(),
        None,
        "a refused publication leaves nothing complete behind"
    );
    assert!(
        outside.is_file(),
        "the refusal never touches what the link pointed at"
    );
    assert!(
        staging.join("index.json").is_symlink(),
        "refusal teardown leaves the linked staging entry itself in place instead of following it"
    );
    assert_eq!(
        std::fs::read(&outside).unwrap(),
        b"bytes that live somewhere else",
        "nothing outside the staging tree was modified by the refusal"
    );
}

/// Released cache locks must be free IMMEDIATELY, even while a descriptor this process handed to a
/// child still references the same open file description.
///
/// `flock(2)` locks live on the open file description, and `fork(2)` gives the child a reference to
/// it, so releasing by `close(2)` alone only takes effect once every such reference is gone.
/// `inherited_descriptor` reproduces that sharing without a child process, so the interleaving is
/// injected rather than waited for. Before sc-22738 a dropped reservation's artifact lock, and a
/// released metadata lock, both kept reading as HELD for as long as an unrelated `Command` sat
/// between `fork` and `exec` — costing every contending reservation, lease, recovery and retention
/// pass a spurious "in active use" verdict or a block.
#[test]
fn released_cache_locks_are_free_even_while_an_inherited_descriptor_survives() {
    let scratch = TempDir::new().unwrap();
    let source = scratch.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let candidate = source_candidate(&source, REVISION_A);
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let digest = cache_key_digest(&candidate.cache_key).unwrap();

    let reservation = match store.reserve(&candidate, &source, "fixture:model").unwrap() {
        ReservationOutcome::Acquired(reservation) => reservation,
        _ => panic!("fixture reservation must acquire"),
    };
    let inherited = reservation
        .artifact_lock
        .as_ref()
        .expect("an acquired reservation holds the artifact lock")
        .inherited_descriptor()
        .expect("descriptor duplicates");
    drop(reservation);
    let probe =
        FileLock::try_exclusive(open_lock_file(&store.artifact_lock_path(&digest)).unwrap());
    assert!(
        probe.is_ok(),
        "an artifact lock released with the reservation must not stay held by an inherited \
         descriptor: {:?}",
        probe.err()
    );
    drop(probe);
    drop(inherited);

    let metadata_lock = store.lock_metadata(&digest).unwrap();
    let inherited = metadata_lock
        .inherited_descriptor()
        .expect("descriptor duplicates");
    drop(metadata_lock);
    let probe =
        FileLock::try_exclusive(open_lock_file(&store.metadata_lock_path(&digest)).unwrap());
    assert!(
        probe.is_ok(),
        "a released metadata lock must not stay held by an inherited descriptor: {:?}",
        probe.err()
    );
    drop(inherited);
}

// ---------------------------------------------------------------------------------------------
// sc-26691 — a produced derivative is hashed once, as it is written.
// ---------------------------------------------------------------------------------------------

const PRODUCED_WEIGHTS: &str = "diffusion_pytorch_model-00001-of-00001.safetensors";
const PRODUCED_INDEX: &str = "diffusion_pytorch_model.safetensors.index.json";
/// Large enough that the write crosses many buffer boundaries, small enough for a unit test.
const PRODUCED_WEIGHTS_BYTES: usize = 8 * 1024 * 1024;

fn produced_weights_payload() -> Vec<u8> {
    (0..PRODUCED_WEIGHTS_BYTES)
        .map(|index| (index % 251) as u8)
        .collect()
}

/// Produce the two-file fixture bundle through hashing writers, the way
/// `CheckpointDerivativeOutputs::create` hands them to a producer.
fn produce_attested(staging: &Path) -> ProducedAttestations {
    let attested = ProducedAttestations::default();
    let payload = produced_weights_payload();
    let mut weights = attested
        .create(staging, Path::new(PRODUCED_WEIGHTS))
        .unwrap();
    for chunk in payload.chunks(1024 * 1024 - 7) {
        weights.write_all(chunk).unwrap();
    }
    weights.finish().unwrap();
    // Dropped rather than finished: the serde_json::to_writer shape every H3 sidecar uses.
    serde_json::to_writer(
        attested.create(staging, Path::new(PRODUCED_INDEX)).unwrap(),
        &json!({"weight_map": {"a": PRODUCED_WEIGHTS}}),
    )
    .unwrap();
    assert_eq!(attested.len(), 2, "both streams must record an attestation");
    attested
}

/// Durable before attested: every produced file is `sync_all`-ed before its mtime is backdated
/// and its closing stamp is taken, so no attestation (and no content receipt redeemed from one) can
/// ever describe bytes that were still only in the OS cache. Both close shapes are covered: the
/// explicit `finish` (weights) and the drop (the JSON sidecar).
#[test]
fn a_produced_file_is_synced_before_its_attestation_is_stamped() {
    let scratch = TempDir::new().unwrap();
    let (attested, steps) = produced::observe_close_steps(|| produce_attested(scratch.path()));
    assert_eq!(attested.len(), 2);
    assert_eq!(
        steps,
        vec![
            "sync_all",
            "set_modified",
            "stamp",
            "sync_all",
            "set_modified",
            "stamp"
        ],
        "each close syncs, then backdates, then stamps"
    );
}

fn sha256_of(path: &Path) -> String {
    format!("sha256:{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

/// The H3 import defect (CUDA run 40643143319): a 34 GB derivative was copied once and then
/// SHA-256'd four more times by this store alone — measurement, publication validation, completion
/// validation, and the first `acquire_complete` — before the native load seal read it again.
///
/// Stated as operation counts across the whole prepare flow on a multi-MB file: ONE hash pass per
/// produced file (taken from the write stream) and ZERO re-reads through measure → publish →
/// record_complete → acquire → native load seal, which redeems the stream's digest through
/// `gen_core::PinnedWeightsFile::pin_attested` and reports zero hashed bytes.
#[test]
fn a_produced_derivative_is_hashed_once_as_it_is_written_through_its_first_load() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&[PRODUCED_WEIGHTS, PRODUCED_INDEX]);
    let key = plan.cache_key().unwrap();
    let resolver = resolver(&input, ActiveArtifactLeaseRegistry::default());

    let (((published, lease, attested), rereads), streamed) =
        produced::observe_stream_hashes(|| {
            crate::model_artifacts::observe_content_hashes(|| {
                let reservation = derived_reservation(&store, &input, &plan);
                let attested = produce_attested(reservation.staging_path());
                let published = reservation
                    .publish_produced_attested_with_check(&plan, &attested, &|| Ok(()))
                    .expect("the attested bundle publishes");
                let lease = store
                    .acquire_complete(&key, &resolver, "runtime:video:model")
                    .unwrap()
                    .expect("the first load leases the published bundle");
                (published, lease, attested)
            })
        });
    assert_eq!(
        streamed, 2,
        "each produced file is hashed exactly once, by its write stream"
    );
    assert_eq!(
        rereads, 0,
        "measure, publish, record_complete and the first acquire must all reuse the write-stream \
         digest instead of re-reading the bundle"
    );

    // The recorded digests are the real ones, and the load seal binds to them.
    let bundle = store.bundle_path(&key).unwrap();
    for file in published
        .artifact
        .closure
        .members
        .iter()
        .flat_map(|member| member.files.iter())
    {
        let path = bundle.join(&file.relative_path);
        assert_eq!(file.sha256.as_deref(), Some(sha256_of(&path).as_str()));
        // The load seal redeems the write-stream digest (no content read) through the gen-core
        // attestation captured when the stream closed, across the staging → bundle rename.
        let attestation = attested
            .get(&file.relative_path)
            .and_then(|attested| attested.seal_attestation().cloned())
            .expect("the stream captured a gen-core attestation");
        let mut acquisitions = Vec::new();
        let seal = gen_core::PinnedWeightsFile::pin_attested_with_progress(
            &path,
            &attestation,
            || Ok(()),
            |progress| acquisitions.push((progress.acquisition, progress.hashed_bytes)),
        )
        .unwrap();
        assert_eq!(
            acquisitions,
            vec![(gen_core::runtime::ArtifactSealAcquisition::Attested, 0)],
            "the native load seal must not re-read a file its writer already hashed"
        );
        let sealed = format!(
            "sha256:{}",
            seal.content_sha256()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        assert_eq!(file.sha256.as_deref(), Some(sealed.as_str()));
    }
    assert_eq!(
        published.artifact.closure.members[0].files[0].size_bytes,
        Some(PRODUCED_WEIGHTS_BYTES as u64)
    );
    drop(lease);
}

/// The other half of the tradeoff: trusting the write stream must not weaken drift detection.
/// A same-length rewrite AFTER the stream closed (in staging, before publication) moves the
/// identity stamp, so measurement re-reads the file — and refuses it, because the bytes are no
/// longer what the producer wrote.
#[test]
fn a_produced_file_rewritten_after_its_stream_closed_is_re_hashed_and_refused() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&[PRODUCED_WEIGHTS, PRODUCED_INDEX]);
    let key = plan.cache_key().unwrap();

    let reservation = derived_reservation(&store, &input, &plan);
    let staging = reservation.staging_path().to_owned();
    let attested = produce_attested(&staging);
    let mut tampered = produced_weights_payload();
    tampered[PRODUCED_WEIGHTS_BYTES / 2] ^= 0xff;
    std::fs::write(staging.join(PRODUCED_WEIGHTS), &tampered).unwrap();

    let (refused, rereads) = crate::model_artifacts::observe_content_hashes(|| {
        reservation.publish_produced_attested_with_check(&plan, &attested, &|| Ok(()))
    });
    let error = refused.expect_err("bytes the producer did not write must never publish");
    assert!(
        error
            .to_string()
            .contains("changed after its producer wrote it"),
        "{error}"
    );
    assert_eq!(
        rereads, 1,
        "only the file whose stamp moved is re-read; the untouched one keeps its stream digest"
    );
    assert_eq!(store.lookup_complete(&key).unwrap(), None);
}

/// And after publication: the content receipt the produced path records is keyed on the
/// closure's stat identity, so a same-length rewrite of a published file sends the first load
/// back through the full read, which refuses the bundle.
#[test]
fn a_published_produced_bundle_altered_before_its_first_load_is_re_hashed_and_refused() {
    let scratch = TempDir::new().unwrap();
    let input = scratch.path().join("library");
    std::fs::create_dir(&input).unwrap();
    let store = ResolvedCacheStore::open(&scratch.path().join("data")).unwrap();
    let plan = derived_plan(&[PRODUCED_WEIGHTS, PRODUCED_INDEX]);
    let key = plan.cache_key().unwrap();
    let resolver = resolver(&input, ActiveArtifactLeaseRegistry::default());

    let reservation = derived_reservation(&store, &input, &plan);
    let attested = produce_attested(reservation.staging_path());
    reservation
        .publish_produced_attested_with_check(&plan, &attested, &|| Ok(()))
        .unwrap();
    let digest = cache_key_digest(&key).unwrap();
    assert!(
        store.read_content_receipt(&digest, &key).is_some(),
        "the produced path records the verdict its completion pass earned"
    );

    let published = store.bundle_path(&key).unwrap().join(PRODUCED_WEIGHTS);
    let mut tampered = produced_weights_payload();
    tampered[7] ^= 0x01;
    std::fs::write(&published, &tampered).unwrap();

    let (refused, rereads) = crate::model_artifacts::observe_content_hashes(|| {
        store.acquire_complete(&key, &resolver, "runtime:video:model")
    });
    assert!(
        refused.is_err(),
        "a produced bundle altered after publication must be refused at the load boundary"
    );
    assert!(
        rereads >= 1,
        "a moved stat identity must invalidate the receipt and re-read the closure, not reuse it"
    );
    assert!(
        store.read_content_receipt(&digest, &key).is_none(),
        "the stale verdict is retracted by the pass that refused the bundle"
    );
}
