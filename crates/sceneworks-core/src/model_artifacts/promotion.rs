//! The one closure producer: how a real manifest entry becomes a [`PromotionCandidate`].
//!
//! sc-19704 froze the typed two-tier contract, sc-19705 the store, sc-19706 the materializer and
//! its scheduler, sc-19708 the *one* selection module that reduces a manifest entry plus this
//! host's platform plus the request's selected tier to an exact
//! [`ExternalArtifactRequirement`] closure. Nothing joined them: no production code ever built a
//! [`ResolvedBundleClosure`] for a real model, so `models/resolved` was only ever populated by
//! tests. This module is that join.
//!
//! It deliberately computes **no** closure of its own. Its input is the requirement closure the
//! selection module already produced, so a new model stays a manifest-data change: there is no
//! per-model, per-route or per-job-type branch anywhere below.
//!
//! Two invariants make the produced candidate safe to hand to the materializer:
//!
//! - **Every member is pinned to an immutable snapshot.** A requirement that records its revision
//!   is resolved to exactly that snapshot; a legacy receipt without one must be satisfied by
//!   exactly one installed snapshot, the same uniqueness rule the availability resolver applies.
//! - **The bundle mirrors the source library.** Each member's destination is
//!   `models--<safe repository>/snapshots/<revision>/<source subpath>`, so a published bundle root
//!   is itself a valid [`ArtifactSourceLibrary`] holding precisely the closure's repositories and
//!   revisions. That is what lets the local tier be preferred without one line of per-model code.
//!
//! Nothing here downloads. The candidate is built from files that are **already present** in the
//! configured source library — [`ModelArtifactResolver::resolve_source`] validates the whole
//! closure against that library before a candidate exists at all, so a disconnected or incomplete
//! source produces an error and therefore no promotion, never a fetch.

use super::external_library::ExternalArtifactRequirement;
use super::{
    ArtifactAvailability, ArtifactContractError, ArtifactFile, ArtifactIdentity, ArtifactMemberRole,
    ArtifactSourceLibrary, ModelArtifactResolver, PromotionCandidate, ResolvedBundleClosure,
    ResolvedBundleMember,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The variant recorded by a manifest download row that declares no tier. It names no artifact
/// tier, so it must not become a `fixedArtifactTier` in the promoted provenance.
const UNTIERED_VARIANT: &str = "default";

/// The canonical bundle-relative destination of one closure member: the source library's own
/// layout for that member's repository, immutable revision and subpath.
///
/// Materialization writes members here and the local tier reads them back through this identical
/// rule, so the two can never drift into a layout only one of them understands.
pub fn hub_cache_member_destination(
    repository: &str,
    revision: &str,
    source_subpath: &Path,
) -> Result<PathBuf, ArtifactContractError> {
    super::validate_immutable_revision(revision)?;
    super::validate_repository(repository)?;
    let safe = repository.replace('/', "--");
    let mut destination = PathBuf::from(format!("models--{safe}"))
        .join("snapshots")
        .join(revision);
    if source_subpath.as_os_str().is_empty() {
        return Ok(destination);
    }
    super::validate_relative_path(source_subpath, "artifact source subpath")?;
    destination.push(source_subpath);
    Ok(destination)
}

/// Build the promotion candidate for one exact selected requirement closure that a runtime just
/// loaded successfully from the configured source library.
///
/// `requirements` must be the closure produced by
/// [`crate::model_artifacts::artifact_selection::selected_requirements_for_model`] — the primary
/// tier this host actually selected plus its hard co-requisites, and nothing else. Every member,
/// including multi-repository co-requisites and sibling-repo overlays, is carried through; the
/// bundle is either the whole closure or no bundle at all.
///
/// The candidate is minted through the sc-19704 lease boundary: an artifact is resolved against
/// the source library, leased, and only a lease that is marked successful becomes a candidate.
pub fn promotion_candidate_for_requirements(
    resolver: &ModelArtifactResolver,
    requirements: &[ExternalArtifactRequirement],
) -> Result<PromotionCandidate, ArtifactContractError> {
    if requirements.is_empty() {
        return Err(ArtifactContractError(
            "promotion requires a nonempty selected requirement closure".to_owned(),
        ));
    }
    if requirements
        .iter()
        .filter(|requirement| requirement.is_primary)
        .count()
        != 1
    {
        return Err(ArtifactContractError(
            "promotion requires exactly one primary requirement".to_owned(),
        ));
    }
    let library = resolver.source_library();
    let mut members = Vec::with_capacity(requirements.len());
    let mut primary_identity = None;
    for requirement in requirements {
        let revision = resolve_requirement_revision(library, requirement)?;
        let source = ArtifactIdentity::pinned(
            &requirement.repository,
            &revision,
            requirement.variant.trim(),
        )?;
        if requirement.is_primary {
            primary_identity = Some(source.clone());
        }
        members.push(ResolvedBundleMember {
            role: if requirement.is_primary {
                ArtifactMemberRole::Primary
            } else {
                ArtifactMemberRole::CoRequisite
            },
            // The primary carries no component id by contract; every co-requisite needs a stable
            // one, and its repository plus selected variant is the only identity the requirement
            // closure actually has. Two rows of one repository at different tiers stay distinct.
            component_id: (!requirement.is_primary)
                .then(|| format!("{}@{}", requirement.repository, requirement.variant.trim())),
            tier: (requirement.variant.trim() != UNTIERED_VARIANT)
                .then(|| requirement.variant.trim().to_owned()),
            // Receipt and manifest file lists are snapshot-relative, so the member root is the
            // snapshot itself.
            source_subpath: PathBuf::new(),
            destination: hub_cache_member_destination(
                &requirement.repository,
                &revision,
                Path::new(""),
            )?,
            files: requirement
                .files
                .iter()
                .map(ArtifactFile::new)
                .collect::<Result<Vec<_>, _>>()?,
            source,
        });
    }
    let closure = ResolvedBundleClosure::new(members)?;
    let identity = primary_identity.expect("a closure with exactly one primary has its identity");
    // Resolves against — and fully validates the closure inside — the configured source library.
    // A disconnected, incomplete, or tampered source fails here, so no candidate exists to promote.
    let artifact = resolver.resolve_source(identity, closure, ArtifactAvailability::Available)?;
    // sc-19704's documented promotion boundary. `mark_success` is the only way to mint a
    // candidate, so an artifact that never held a successful runtime lease can never be promoted.
    Ok(resolver
        .acquire_runtime_lease(&Arc::new(artifact))?
        .mark_success())
}

/// The immutable revision that satisfies `requirement` in `library`.
///
/// A recorded revision is authoritative and is only checked for installation. A legacy receipt
/// without one must be satisfied by **exactly one** installed snapshot — the same uniqueness rule
/// [`crate::model_artifacts::external_library::validate_requirements_at_root`] applies, so an
/// ambiguous repository declines rather than promoting an arbitrary snapshot.
fn resolve_requirement_revision(
    library: &ArtifactSourceLibrary,
    requirement: &ExternalArtifactRequirement,
) -> Result<String, ArtifactContractError> {
    if let Some(revision) = requirement.revision.as_deref() {
        library.discover_snapshot(&requirement.repository, Some(revision))?;
        return Ok(revision.to_owned());
    }
    let snapshots = library
        .repository_root(&requirement.repository)?
        .join("snapshots");
    let mut matching = Vec::new();
    for entry in std::fs::read_dir(&snapshots).map_err(|error| {
        ArtifactContractError(format!(
            "source repository {} is unavailable: {error}",
            requirement.repository
        ))
    })? {
        let entry =
            entry.map_err(|error| ArtifactContractError(format!("read source snapshots: {error}")))?;
        let Some(revision) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if super::validate_immutable_revision(&revision).is_err()
            || !entry.file_type().is_ok_and(|kind| kind.is_dir())
        {
            continue;
        }
        let path = entry.path();
        if requirement
            .files
            .iter()
            .all(|file| path.join(file).is_file())
        {
            matching.push(revision);
        }
    }
    match matching.as_slice() {
        [revision] => Ok(revision.clone()),
        _ => Err(ArtifactContractError(format!(
            "source repository {} has {} snapshots holding the recorded file set; exactly one is \
             required to promote a revision-less install receipt",
            requirement.repository,
            matching.len()
        ))),
    }
}

#[cfg(test)]
#[path = "promotion_tests.rs"]
mod tests;
