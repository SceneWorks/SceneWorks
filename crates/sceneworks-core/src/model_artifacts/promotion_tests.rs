use super::*;
use crate::model_artifacts::artifact_selection::{
    safe_download_dir, selected_requirements_for_model,
};
use crate::model_artifacts::external_library::local_artifact_for_requirements;
use crate::model_artifacts::resolved_cache::{
    MaterializationCancellation, MaterializationOutcome, ResolvedCacheMaterializer,
    ResolvedCachePolicy, ResolvedCacheStore,
};
use crate::model_artifacts::ArtifactLocation;
use serde_json::{json, Value};
use std::path::Path;

const PRIMARY_REVISION: &str = "1111111111111111111111111111111111111111";
const OTHER_REVISION: &str = "2222222222222222222222222222222222222222";
const TEXT_ENCODER_REVISION: &str = "3333333333333333333333333333333333333333";

/// The exact `models--<safe>/snapshots/<revision>/<file>` layout a real Hugging Face library has.
fn install_snapshot_file(
    library: &Path,
    repository: &str,
    revision: &str,
    file: &str,
    body: &[u8],
) {
    let path = library
        .join(format!("models--{}", repository.replace('/', "--")))
        .join("snapshots")
        .join(revision)
        .join(file);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, body).unwrap();
}

fn write_receipts(data_dir: &Path, repository: &str, receipts: Value) {
    let managed = data_dir.join("models").join(safe_download_dir(repository));
    std::fs::create_dir_all(&managed).unwrap();
    std::fs::write(
        managed.join(".sceneworks-download-complete.json"),
        serde_json::to_vec(&json!({ "receipts": receipts })).unwrap(),
    )
    .unwrap();
}

fn resolver_for(library: &Path) -> ModelArtifactResolver {
    ModelArtifactResolver::new(ArtifactSourceLibrary::new(library).unwrap())
}

fn requirement(
    repository: &str,
    revision: Option<&str>,
    variant: &str,
    files: &[&str],
    is_primary: bool,
) -> ExternalArtifactRequirement {
    ExternalArtifactRequirement {
        repository: repository.to_owned(),
        revision: revision.map(str::to_owned),
        variant: variant.to_owned(),
        files: files.iter().map(PathBuf::from).collect(),
        is_primary,
    }
}

/// The whole selected closure — primary tier plus a multi-repository co-requisite — becomes ONE
/// bundle laid out as a miniature source library, with every member pinned to its exact immutable
/// revision. This is the shape the local tier reads back, so it is asserted member by member.
#[test]
fn a_multi_repository_closure_becomes_one_source_library_shaped_bundle() {
    let temp = tempfile::tempdir().unwrap();
    let library = temp.path().join("library");
    install_snapshot_file(
        &library,
        "owner/model",
        PRIMARY_REVISION,
        "q4/model.safetensors",
        b"primary-weights",
    );
    install_snapshot_file(
        &library,
        "owner/text-encoder",
        TEXT_ENCODER_REVISION,
        "encoder.safetensors",
        b"encoder-weights",
    );

    let candidate = promotion_candidate_for_requirements(
        &resolver_for(&library),
        &[
            requirement(
                "owner/model",
                Some(PRIMARY_REVISION),
                "q4",
                &["q4/model.safetensors"],
                true,
            ),
            requirement(
                "owner/text-encoder",
                Some(TEXT_ENCODER_REVISION),
                "default",
                &["encoder.safetensors"],
                false,
            ),
        ],
    )
    .expect("an installed closure promotes");

    assert_eq!(candidate.artifact.identity.repository, "owner/model");
    assert_eq!(candidate.artifact.identity.revision, PRIMARY_REVISION);
    assert_eq!(candidate.artifact.identity.variant, "q4");
    assert_eq!(
        candidate.artifact.provenance.fixed_artifact_tier.as_deref(),
        Some("q4"),
        "the selected tier is the promoted artifact's fixed tier"
    );
    assert_eq!(candidate.artifact.closure.members.len(), 2);

    let primary = candidate
        .artifact
        .closure
        .members
        .iter()
        .find(|member| member.role == ArtifactMemberRole::Primary)
        .expect("the closure has a primary");
    assert_eq!(primary.component_id, None);
    assert_eq!(primary.source_subpath, PathBuf::new());
    assert_eq!(
        primary.destination,
        PathBuf::from(format!("models--owner--model/snapshots/{PRIMARY_REVISION}"))
    );

    let co_requisite = candidate
        .artifact
        .closure
        .members
        .iter()
        .find(|member| member.role == ArtifactMemberRole::CoRequisite)
        .expect("the closure keeps its co-requisite");
    assert_eq!(
        co_requisite.component_id.as_deref(),
        Some("owner/text-encoder@default")
    );
    assert_eq!(co_requisite.tier, None, "an untiered row declares no tier");
    assert_eq!(
        co_requisite.destination,
        PathBuf::from(format!(
            "models--owner--text-encoder/snapshots/{TEXT_ENCODER_REVISION}"
        ))
    );

    // Every member's destination is exactly the shared layout rule, so a published bundle root is
    // itself a source library. Asserted through the rule rather than a literal so the two cannot
    // drift apart silently.
    for member in &candidate.artifact.closure.members {
        assert_eq!(
            member.destination,
            hub_cache_member_destination(
                &member.source.repository,
                &member.source.revision,
                &member.source_subpath,
            )
            .unwrap()
        );
    }
}

/// Two tiers of ONE repository stay two distinct members. Their destinations legitimately coincide
/// (same repository, same revision — that IS the source layout), so distinctness has to come from
/// the component id, and their file sets must not collide.
#[test]
fn sibling_tiers_of_one_repository_stay_distinct_members() {
    let temp = tempfile::tempdir().unwrap();
    let library = temp.path().join("library");
    install_snapshot_file(
        &library,
        "owner/model",
        PRIMARY_REVISION,
        "q4/model.safetensors",
        b"q4",
    );
    install_snapshot_file(
        &library,
        "owner/model",
        PRIMARY_REVISION,
        "shared/vae.safetensors",
        b"vae",
    );

    let candidate = promotion_candidate_for_requirements(
        &resolver_for(&library),
        &[
            requirement(
                "owner/model",
                Some(PRIMARY_REVISION),
                "q4",
                &["q4/model.safetensors"],
                true,
            ),
            requirement(
                "owner/model",
                Some(PRIMARY_REVISION),
                "shared",
                &["shared/vae.safetensors"],
                false,
            ),
        ],
    )
    .expect("a same-repository co-requisite promotes");

    assert_eq!(candidate.artifact.closure.members.len(), 2);
    assert_eq!(
        candidate.artifact.closure.members[0].destination,
        candidate.artifact.closure.members[1].destination,
        "one repository at one revision is one source-library directory"
    );
}

/// A legacy receipt that never recorded its snapshot revision adopts the only installed snapshot
/// that holds its exact recorded file set.
#[test]
fn a_revisionless_receipt_adopts_the_only_matching_snapshot() {
    let temp = tempfile::tempdir().unwrap();
    let library = temp.path().join("library");
    install_snapshot_file(
        &library,
        "owner/model",
        PRIMARY_REVISION,
        "model.safetensors",
        b"weights",
    );
    // A second snapshot of the same repository that does NOT hold the recorded file.
    install_snapshot_file(
        &library,
        "owner/model",
        OTHER_REVISION,
        "other.safetensors",
        b"other",
    );

    let candidate = promotion_candidate_for_requirements(
        &resolver_for(&library),
        &[requirement(
            "owner/model",
            None,
            "default",
            &["model.safetensors"],
            true,
        )],
    )
    .expect("an unambiguous revision-less receipt promotes");
    assert_eq!(candidate.artifact.identity.revision, PRIMARY_REVISION);
}

/// Ambiguity declines. Promoting an arbitrary snapshot would give the bundle an identity the
/// install never had.
#[test]
fn an_ambiguous_revisionless_receipt_declines() {
    let temp = tempfile::tempdir().unwrap();
    let library = temp.path().join("library");
    for revision in [PRIMARY_REVISION, OTHER_REVISION] {
        install_snapshot_file(&library, "owner/model", revision, "model.safetensors", b"w");
    }

    let error = promotion_candidate_for_requirements(
        &resolver_for(&library),
        &[requirement(
            "owner/model",
            None,
            "default",
            &["model.safetensors"],
            true,
        )],
    )
    .expect_err("two candidate snapshots must decline");
    assert!(
        error.to_string().contains("exactly one is required"),
        "unexpected error: {error}"
    );
}

/// The source library is the ONLY place promotion reads from. A closure whose files are not
/// installed produces no candidate and writes nothing — promotion can never turn into a download.
#[test]
fn an_uninstalled_closure_declines_without_creating_anything() {
    let temp = tempfile::tempdir().unwrap();
    let library = temp.path().join("library");
    install_snapshot_file(
        &library,
        "owner/model",
        PRIMARY_REVISION,
        "model.safetensors",
        b"weights",
    );

    let error = promotion_candidate_for_requirements(
        &resolver_for(&library),
        &[
            requirement(
                "owner/model",
                Some(PRIMARY_REVISION),
                "default",
                &["model.safetensors"],
                true,
            ),
            requirement(
                "owner/missing",
                Some(OTHER_REVISION),
                "default",
                &["encoder.safetensors"],
                false,
            ),
        ],
    )
    .expect_err("an uninstalled co-requisite must decline");
    assert!(!error.to_string().is_empty());
    assert!(
        !library.join("models--owner--missing").exists(),
        "promotion must never create, fetch, or reserve anything for an absent source"
    );
}

/// A closure without exactly one primary is not a runtime closure and cannot become a bundle.
#[test]
fn a_closure_without_exactly_one_primary_declines() {
    let temp = tempfile::tempdir().unwrap();
    let library = temp.path().join("library");
    install_snapshot_file(
        &library,
        "owner/model",
        PRIMARY_REVISION,
        "model.safetensors",
        b"weights",
    );
    let resolver = resolver_for(&library);
    assert!(promotion_candidate_for_requirements(&resolver, &[]).is_err());
    assert!(promotion_candidate_for_requirements(
        &resolver,
        &[
            requirement(
                "owner/model",
                Some(PRIMARY_REVISION),
                "default",
                &["model.safetensors"],
                true
            ),
            requirement(
                "owner/model",
                Some(PRIMARY_REVISION),
                "alt",
                &["model.safetensors"],
                true
            ),
        ]
    )
    .is_err());
}

/// The full data-driven round trip inside core: a real manifest entry and its install receipts go
/// through the ONE selection module, become a candidate, materialize, and the published bundle is
/// then recognized by the availability resolver's local-tier matcher for that same closure.
#[test]
fn a_manifest_entry_promotes_and_is_then_recognized_as_the_local_tier() {
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path().join("data");
    let library = temp.path().join("library");
    std::fs::create_dir_all(&data_dir).unwrap();
    install_snapshot_file(
        &library,
        "owner/model",
        PRIMARY_REVISION,
        "q4/model.safetensors",
        b"primary-weights",
    );
    install_snapshot_file(
        &library,
        "owner/text-encoder",
        TEXT_ENCODER_REVISION,
        "encoder.safetensors",
        b"encoder-weights",
    );
    write_receipts(
        &data_dir,
        "owner/model",
        json!([{
            "repo": "owner/model",
            "modelId": "demo",
            "variant": "q4",
            "resolvedFiles": ["q4/model.safetensors"],
            "snapshotRevision": PRIMARY_REVISION,
        }]),
    );
    write_receipts(
        &data_dir,
        "owner/text-encoder",
        json!([{
            "repo": "owner/text-encoder",
            "resolvedFiles": ["encoder.safetensors"],
            "snapshotRevision": TEXT_ENCODER_REVISION,
        }]),
    );
    let model = json!({
        "id": "demo",
        "downloads": [
            {"provider": "huggingface", "repo": "owner/model", "variant": "q4", "default": true,
             "files": ["q4/*"]},
            {"provider": "huggingface", "repo": "owner/text-encoder", "coRequisite": true,
             "files": ["encoder.safetensors"]}
        ]
    });

    let selected =
        selected_requirements_for_model(&model, std::env::consts::OS, Some("q4"), &data_dir);
    assert!(selected.receipt_backed);
    assert_eq!(selected.requirements.len(), 2);

    let resolver = resolver_for(&library);
    let candidate =
        promotion_candidate_for_requirements(&resolver, &selected.requirements).unwrap();

    let store = ResolvedCacheStore::open(&data_dir).unwrap();
    let outcome = ResolvedCacheMaterializer::new(store.clone())
        .materialize(
            &candidate,
            &library,
            "demo",
            &MaterializationCancellation::default(),
        )
        .unwrap();
    let metadata = match outcome {
        MaterializationOutcome::Published(metadata) => *metadata,
        other => panic!("expected a published bundle, got {other:?}"),
    };
    assert!(metadata.verified_bytes > 0);

    // The published artifact is what the shared availability resolver would prefer for the very
    // same selected closure — the identity/coverage rule, not a path guess.
    let published = metadata.artifact;
    assert!(matches!(
        published.location,
        ArtifactLocation::ResolvedLocal { .. }
    ));
    assert_eq!(
        local_artifact_for_requirements(std::slice::from_ref(&published), &selected.requirements)
            .map(|artifact| artifact.identity),
        Some(published.identity.clone()),
        "the promoted bundle must cover the exact closure that produced it"
    );

    // Every promoted byte landed in the source-library layout, so the bundle root resolves as a
    // library in its own right.
    let ArtifactLocation::ResolvedLocal { root } = &published.location else {
        unreachable!()
    };
    let bundle_library = ArtifactSourceLibrary::new(root).unwrap();
    for (repository, revision) in [
        ("owner/model", PRIMARY_REVISION),
        ("owner/text-encoder", TEXT_ENCODER_REVISION),
    ] {
        let (identity, snapshot) = bundle_library
            .discover_snapshot(repository, Some(revision))
            .expect("the bundle root is a valid source library");
        assert_eq!(identity.revision, revision);
        assert!(snapshot.is_dir());
    }
    assert_eq!(
        std::fs::read(
            root.join(format!("models--owner--model/snapshots/{PRIMARY_REVISION}"))
                .join("q4/model.safetensors")
        )
        .unwrap(),
        b"primary-weights"
    );

    // Nothing about the promotion touched the policy defaults: the cache is opt-in and the store
    // was opened explicitly by this test, never as a side effect of resolving a candidate.
    assert!(!ResolvedCachePolicy::default().enabled);
}
