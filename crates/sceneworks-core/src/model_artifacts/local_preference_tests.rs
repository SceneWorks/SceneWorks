use super::*;
use crate::model_artifacts::{
    ArtifactAvailability, ArtifactCompleteness, ArtifactFile, ArtifactIdentity, ArtifactProvenance,
    ArtifactSourceLibrary, ResolvedBundleClosure, ResolvedBundleMember,
};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

const PRIMARY_REV: &str = "0123456789abcdef0123456789abcdef01234567";
const COMPONENT_REV: &str = "89abcdef0123456789abcdef0123456789abcdef";
const OTHER_REV: &str = "1111111111111111111111111111111111111111";

/// The preference scope is process-wide by design (model paths resolve on engine threads far
/// below the job task), so these tests serialize against each other.
fn overlay_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

fn member(
    role: ArtifactMemberRole,
    component_id: Option<&str>,
    repository: &str,
    revision: &str,
    subpath: &str,
    file: &str,
) -> ResolvedBundleMember {
    let source_subpath = PathBuf::from(subpath);
    ResolvedBundleMember {
        role,
        component_id: component_id.map(str::to_owned),
        source: ArtifactIdentity::pinned(repository, revision, "q4").unwrap(),
        tier: Some("q4".to_owned()),
        destination: hub_cache_member_destination(repository, revision, &source_subpath).unwrap(),
        source_subpath,
        files: vec![ArtifactFile::new(file).unwrap()],
    }
}

/// A published bundle laid out exactly as materialization writes it: one primary plus one
/// co-requisite from a different repository and revision.
fn bundle(root: &Path) -> ResolvedModelArtifact {
    let identity = ArtifactIdentity::pinned("owner/model", PRIMARY_REV, "q4").unwrap();
    let closure = ResolvedBundleClosure::new(vec![
        member(
            ArtifactMemberRole::Primary,
            None,
            "owner/model",
            PRIMARY_REV,
            "q4",
            "model.safetensors",
        ),
        member(
            ArtifactMemberRole::CoRequisite,
            Some("text_encoder"),
            "owner/encoder",
            COMPONENT_REV,
            "",
            "encoder.safetensors",
        ),
    ])
    .unwrap();
    for path in closure.rebased_paths(root).unwrap() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"weights").unwrap();
    }
    let artifact = ResolvedModelArtifact {
        schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
        provenance: ArtifactProvenance {
            identity: identity.clone(),
            fixed_artifact_tier: Some("q4".to_owned()),
        },
        identity,
        location: ArtifactLocation::ResolvedLocal {
            root: root.to_path_buf(),
        },
        closure,
        completeness: ArtifactCompleteness::Complete,
        availability: ArtifactAvailability::Available,
    };
    artifact.validate().unwrap();
    artifact
}

fn seed_source(library: &Path, repository: &str, revision: &str, file: &str) -> PathBuf {
    let snapshot = ArtifactSourceLibrary::new(library)
        .unwrap()
        .repository_root(repository)
        .unwrap()
        .join("snapshots")
        .join(revision);
    std::fs::create_dir_all(snapshot.join(file).parent().unwrap()).unwrap();
    std::fs::write(snapshot.join(file), b"weights").unwrap();
    snapshot
}

fn write_refs_main(library: &Path, repository: &str, revision: &str) {
    let repo_root = ArtifactSourceLibrary::new(library)
        .unwrap()
        .repository_root(repository)
        .unwrap();
    std::fs::create_dir_all(repo_root.join("refs")).unwrap();
    std::fs::write(repo_root.join("refs").join("main"), revision).unwrap();
}

/// The bundle layout IS the source-library layout. This is the invariant every runtime relies on.
#[test]
fn member_destination_mirrors_the_source_library_layout() {
    assert_eq!(
        hub_cache_member_destination("owner/model", PRIMARY_REV, Path::new("")).unwrap(),
        PathBuf::from(format!("models--owner--model/snapshots/{PRIMARY_REV}"))
    );
    assert_eq!(
        hub_cache_member_destination("owner/model", PRIMARY_REV, Path::new("q4")).unwrap(),
        PathBuf::from(format!("models--owner--model/snapshots/{PRIMARY_REV}/q4"))
    );
    assert!(hub_cache_member_destination("owner/model", "main", Path::new("")).is_err());
    assert!(
        hub_cache_member_destination("owner/model", PRIMARY_REV, Path::new("../escape")).is_err()
    );
}

/// A bundle stored in any other layout cannot be handed to the shared snapshot resolvers, so it is
/// reported as an unsupported shape instead of being half-used.
#[test]
fn a_bundle_outside_the_source_library_layout_is_an_unsupported_shape() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("bundle");
    let mut artifact = bundle(&root);
    assert_eq!(overlay_entries_for_artifact(&artifact).unwrap().len(), 2);

    let mut members = artifact.closure.members.clone();
    members[0].destination = PathBuf::from("primary");
    artifact.closure = ResolvedBundleClosure::new(members).unwrap();
    let error = overlay_entries_for_artifact(&artifact).unwrap_err();
    assert!(
        error.to_string().contains("source-library layout"),
        "{error}"
    );

    // A source-tier artifact is never a local-tier candidate.
    let mut source_tier = bundle(&root);
    source_tier.location = ArtifactLocation::SourceLibrary { root };
    assert!(overlay_entries_for_artifact(&source_tier).is_err());
}

/// The walking skeleton at the contract level: with a scope active, the CONFIGURED library serves
/// the leased bundle for every member's exact pair, and the scope's drop restores the source tier.
#[test]
fn an_active_scope_serves_every_member_and_drop_restores_the_source_tier() {
    let _serialized = overlay_guard();
    let temp = tempfile::tempdir().unwrap();
    let library_root = temp.path().join("external-hf");
    let primary_source = seed_source(
        &library_root,
        "owner/model",
        PRIMARY_REV,
        "q4/model.safetensors",
    );
    let encoder_source = seed_source(
        &library_root,
        "owner/encoder",
        COMPONENT_REV,
        "encoder.safetensors",
    );
    let artifact = bundle(&temp.path().join("bundle"));
    let configured = ArtifactSourceLibrary::new_preferring_local(&library_root).unwrap();

    assert_eq!(
        configured
            .discover_snapshot("owner/model", Some(PRIMARY_REV))
            .unwrap()
            .1,
        primary_source
    );

    let scope = prefer_local_artifacts(std::slice::from_ref(&artifact)).unwrap();
    let (_, primary) = configured
        .discover_snapshot("owner/model", Some(PRIMARY_REV))
        .unwrap();
    assert_eq!(
        primary,
        artifact
            .location
            .root()
            .join(format!("models--owner--model/snapshots/{PRIMARY_REV}"))
    );
    assert!(primary.join("q4/model.safetensors").is_file());
    let (_, encoder) = configured
        .discover_snapshot("owner/encoder", Some(COMPONENT_REV))
        .unwrap();
    assert!(encoder.join("encoder.safetensors").is_file());
    assert_ne!(encoder, encoder_source);

    // A library built for maintenance/materialization keeps reading the authoritative source, so
    // promotion can never copy a bundle out of a bundle.
    assert_eq!(
        ArtifactSourceLibrary::new(&library_root)
            .unwrap()
            .discover_snapshot("owner/model", Some(PRIMARY_REV))
            .unwrap()
            .1,
        primary_source
    );

    drop(scope);
    assert_eq!(
        configured
            .discover_snapshot("owner/model", Some(PRIMARY_REV))
            .unwrap()
            .1,
        primary_source
    );
}

/// A local bundle never wins for a revision it does not hold: the authoritative library still
/// chooses the revision whenever it can be read.
#[test]
fn a_superseded_revision_is_served_from_the_source_tier() {
    let _serialized = overlay_guard();
    let temp = tempfile::tempdir().unwrap();
    let library_root = temp.path().join("external-hf");
    let newer = seed_source(
        &library_root,
        "owner/model",
        OTHER_REV,
        "q4/model.safetensors",
    );
    write_refs_main(&library_root, "owner/model", OTHER_REV);
    let artifact = bundle(&temp.path().join("bundle"));
    let configured = ArtifactSourceLibrary::new_preferring_local(&library_root).unwrap();

    let _scope = prefer_local_artifacts(std::slice::from_ref(&artifact)).unwrap();
    // Explicit pin at the newer revision: the bundle holds a different commit and must not answer.
    assert_eq!(
        configured
            .discover_snapshot("owner/model", Some(OTHER_REV))
            .unwrap()
            .1,
        newer
    );
    // Unpinned resolution reads the library's own `refs/main`, which names the newer commit.
    assert_eq!(
        configured.discover_snapshot("owner/model", None).unwrap().1,
        newer
    );
}

/// With the configured library disconnected, an unpinned resolve is served by the leased bundle —
/// this is what keeps a complete local model usable while the drive is unplugged.
#[test]
fn a_disconnected_library_falls_back_to_the_unique_leased_revision() {
    let _serialized = overlay_guard();
    let temp = tempfile::tempdir().unwrap();
    let library_root = temp.path().join("never-mounted");
    let artifact = bundle(&temp.path().join("bundle"));
    let configured = ArtifactSourceLibrary::new_preferring_local(&library_root).unwrap();

    assert!(configured.discover_snapshot("owner/model", None).is_err());
    let scope = prefer_local_artifacts(std::slice::from_ref(&artifact)).unwrap();
    let (identity, snapshot) = configured.discover_snapshot("owner/model", None).unwrap();
    assert_eq!(identity.revision, PRIMARY_REV);
    assert!(snapshot.join("q4/model.safetensors").is_file());
    drop(scope);
    assert!(configured.discover_snapshot("owner/model", None).is_err());
}

/// Two bundles claiming one immutable snapshot is ambiguous, and ambiguity must never be resolved
/// by picking one: the installation fails and the caller keeps the load on the source tier.
#[test]
fn conflicting_bundles_refuse_to_install_a_scope() {
    let _serialized = overlay_guard();
    let temp = tempfile::tempdir().unwrap();
    let first = bundle(&temp.path().join("bundle-a"));
    let second = bundle(&temp.path().join("bundle-b"));
    let error = prefer_local_artifacts(&[first.clone(), second]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("two resolved-local bundles claim"),
        "{error}"
    );
    // The failed installation left nothing behind.
    assert!(local_snapshot("owner/model", PRIMARY_REV).is_none());

    let scope = prefer_local_artifacts(std::slice::from_ref(&first)).unwrap();
    assert!(prefer_local_artifacts(std::slice::from_ref(&first)).is_err());
    drop(scope);
    assert!(prefer_local_artifacts(std::slice::from_ref(&first)).is_ok());
}

/// A model directory that was resolved to a concrete source path before the load (training base
/// models, captioners, analyzers) is redirected into the leased bundle — and only when the bundle
/// really holds that file.
#[test]
fn a_resolved_source_path_is_redirected_only_when_the_bundle_holds_it() {
    let _serialized = overlay_guard();
    let temp = tempfile::tempdir().unwrap();
    let library_root = temp.path().join("external-hf");
    let source = seed_source(
        &library_root,
        "owner/model",
        PRIMARY_REV,
        "q4/model.safetensors",
    );
    let artifact = bundle(&temp.path().join("bundle"));
    let _scope = prefer_local_artifacts(std::slice::from_ref(&artifact)).unwrap();

    let redirected = redirect_source_library_path(&library_root, &source.join("q4")).unwrap();
    assert!(redirected.join("model.safetensors").is_file());
    assert!(redirected.starts_with(artifact.location.root()));

    // A file the bundle does not hold keeps its authoritative source path.
    assert!(redirect_source_library_path(&library_root, &source.join("q8")).is_none());
    // A path outside the configured library is never rewritten.
    assert!(redirect_source_library_path(&library_root, Path::new("/etc/hosts")).is_none());
}
