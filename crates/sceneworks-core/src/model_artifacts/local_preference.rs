//! The one local-tier preference seam (sc-19707).
//!
//! A resolved-local bundle is materialized as a **miniature source library**: every member's
//! destination is exactly `models--<safe repository>/snapshots/<revision>/<source subpath>`, so a
//! published bundle root is a valid [`crate::model_artifacts::ArtifactSourceLibrary`] holding
//! precisely the closure's repositories and immutable revisions. That single layout invariant is
//! what lets every model-consuming runtime prefer the local tier without one line of per-model,
//! per-route, or per-job-type code: the shared snapshot resolvers already ask the configured
//! source library for `(repository, revision)`, and this module answers with the leased bundle's
//! snapshot when one covers that exact pair.
//!
//! Scope and safety:
//!
//! - Only the worker's pre-loader guard installs a preference scope, and only for artifacts it
//!   holds a runtime lease on, so an artifact can never be evicted while a load reads it.
//! - The scope is process-wide because model paths are resolved on engine threads and blocking
//!   pools far below the async job task. Which `(repository, revision)` pairs may be served at all
//!   is decided by the GUARD — the only layer that can see every model a job will load — and handed
//!   here already decided (see [`prefer_local_snapshots`]); a directory-level seam cannot re-ask
//!   "does this bundle hold what THIS caller needs" once it has answered with a path.
//! - **The bound on "process-wide", stated honestly.** The worker claim loop runs one job at a
//!   time, so no two JOBS overlap. That is not the whole story: in the desktop deployment the
//!   utility worker runs INSIDE the API process (`spawn_inprocess_utility_worker`), so an API read
//!   path resolving through the same prefer-local
//!   [`crate::hf_home::model_source_library`] can observe a scope a job installed. For a file
//!   OUTSIDE that job's union the leased bundle does not hold it, and such a reader sees it as
//!   ABSENT while the scope is live. The bound is therefore: a concurrent same-process reader may
//!   get a transient **false absent** that self-heals when the job ends — and never WRONG BYTES,
//!   because a served path only ever names an already-verified bundle and an out-of-union name
//!   simply does not exist inside it. Pinned by
//!   `an_out_of_union_reader_sees_absent_never_the_wrong_bytes`. Narrowing this further means
//!   giving API read paths a non-preferring library, which is a change across every
//!   `model_source_library` caller rather than a local one.
//! - Preference is keyed on the exact `(repository, immutable revision)` pair. A superseded
//!   revision therefore never matches, and a bundle is never consulted for a revision it does not
//!   contain — the source tier keeps serving those.
//! - Nothing here downloads, writes, or mutates anything. Redirection only ever answers with a
//!   path inside an already-published, already-verified bundle.

use super::{
    ArtifactContractError, ArtifactLocation, ArtifactMemberRole, ResolvedModelArtifact,
    MODEL_ARTIFACT_CONTRACT_VERSION,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// One `(repository, revision)` pair served from a leased local bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalArtifactOverlayEntry {
    pub repository: String,
    pub revision: String,
    /// `<bundle>/models--<safe>/snapshots/<revision>/` — what a snapshot resolver returns.
    pub snapshot_root: PathBuf,
    /// Snapshot-relative files this bundle holds for the pair, sorted. Two bundles may both cover
    /// one shared snapshot (a text encoder used by two models); this is what decides whether an
    /// already-serving bundle covers everything a second one would have served.
    pub files: Vec<PathBuf>,
}

/// The canonical bundle-relative destination of one closure member: the source library's own
/// layout for that member's repository, immutable revision and subpath. Materialization writes
/// members here, and local preference reads them back through the identical rule, so the two can
/// never drift into a layout only one of them understands.
pub fn hub_cache_member_destination(
    repository: &str,
    revision: &str,
    source_subpath: &Path,
) -> Result<PathBuf, ArtifactContractError> {
    super::validate_immutable_revision(revision)?;
    let safe = safe_repository_dir(repository)?;
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

/// The `models--<X>` directory name component for `repository`. Byte-identical to
/// [`crate::model_artifacts::ArtifactSourceLibrary::repository_root`], which is the layout a
/// bundle mirrors.
fn safe_repository_dir(repository: &str) -> Result<String, ArtifactContractError> {
    super::validate_repository(repository)?;
    Ok(repository.replace('/', "--"))
}

/// Every `(repository, revision)` a published bundle can serve, or an error naming the exact way
/// its shape is unsupported.
///
/// This is deliberately strict. An artifact whose members do not mirror the source library layout
/// cannot be handed to the shared snapshot resolvers at all, so it must be reported as an
/// unsupported shape rather than silently half-used with the rest of the load reading the external
/// path.
pub fn overlay_entries_for_artifact(
    artifact: &ResolvedModelArtifact,
) -> Result<Vec<LocalArtifactOverlayEntry>, ArtifactContractError> {
    if artifact.schema_version != MODEL_ARTIFACT_CONTRACT_VERSION {
        return Err(ArtifactContractError(
            "unsupported model artifact contract version".to_owned(),
        ));
    }
    let ArtifactLocation::ResolvedLocal { root } = &artifact.location else {
        return Err(ArtifactContractError(
            "only an app-owned resolved-local artifact can serve the local tier".to_owned(),
        ));
    };
    if !artifact
        .closure
        .members
        .iter()
        .any(|member| member.role == ArtifactMemberRole::Primary)
    {
        return Err(ArtifactContractError(
            "resolved-local artifact has no primary member".to_owned(),
        ));
    }
    let mut entries: Vec<LocalArtifactOverlayEntry> = Vec::new();
    for member in &artifact.closure.members {
        member.validate()?;
        let expected = hub_cache_member_destination(
            &member.source.repository,
            &member.source.revision,
            &member.source_subpath,
        )?;
        if member.destination != expected {
            return Err(ArtifactContractError(format!(
                "resolved-local member {}@{} is not stored in the source-library layout \
                 (expected {}, found {})",
                member.source.repository,
                member.source.revision,
                expected.display(),
                member.destination.display()
            )));
        }
        let safe = safe_repository_dir(&member.source.repository)?;
        let snapshot_root = root
            .join(format!("models--{safe}"))
            .join("snapshots")
            .join(&member.source.revision);
        // Snapshot-relative, so two bundles holding the same repository/revision can be compared
        // for coverage regardless of which bundle they live in.
        let files = member
            .files
            .iter()
            .map(|file| member.source_subpath.join(&file.relative_path))
            .collect::<Vec<_>>();
        match entries.iter_mut().find(|held| {
            held.repository == member.source.repository && held.revision == member.source.revision
        }) {
            Some(held) => held.files.extend(files),
            None => entries.push(LocalArtifactOverlayEntry {
                repository: member.source.repository.clone(),
                revision: member.source.revision.clone(),
                snapshot_root,
                files,
            }),
        }
    }
    for entry in &mut entries {
        entry.files.sort();
        entry.files.dedup();
    }
    Ok(entries)
}

#[derive(Debug)]
struct HeldEntry {
    entry: LocalArtifactOverlayEntry,
    /// How many live scopes are serving this exact entry. Ownership is REFCOUNTED rather than
    /// single-owner: two scopes may legitimately serve one shared component snapshot, and a
    /// single-owner model would silently un-serve the other one the moment the first scope drops.
    holders: usize,
}

#[derive(Debug, Default)]
struct ActiveOverlay {
    entries: Vec<HeldEntry>,
}

fn active() -> &'static Mutex<ActiveOverlay> {
    static ACTIVE: std::sync::OnceLock<Mutex<ActiveOverlay>> = std::sync::OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(ActiveOverlay::default()))
}

fn lock_active() -> MutexGuard<'static, ActiveOverlay> {
    active().lock().unwrap_or_else(PoisonError::into_inner)
}

/// RAII preference scope. Dropping it restores source-tier resolution exactly; it holds no locks
/// and performs no I/O, so a panicking load still leaves the process on the source tier.
#[derive(Debug)]
pub struct ActiveLocalArtifacts {
    entries: Arc<Vec<LocalArtifactOverlayEntry>>,
}

impl ActiveLocalArtifacts {
    pub fn entries(&self) -> &[LocalArtifactOverlayEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Drop for ActiveLocalArtifacts {
    fn drop(&mut self) {
        let mut active = lock_active();
        for entry in self.entries.iter() {
            let Some(index) = active.entries.iter().position(|held| &held.entry == entry) else {
                continue;
            };
            if active.entries[index].holders > 1 {
                active.entries[index].holders -= 1;
            } else {
                active.entries.remove(index);
            }
        }
    }
}

/// Install a local-tier preference scope over EXACTLY the snapshots the caller selected.
///
/// The caller — the worker's pre-loader guard — is the only layer that can see every model a job
/// will load, so it is the layer that decides which `(repository, revision)` pairs may be served
/// locally at all. This function therefore takes DECIDED entries rather than artifacts: a
/// directory-level seam cannot ask "does this bundle hold what THIS caller needs" at lookup time,
/// so a pair is either serveable for every consumer in the process or served to none of them.
///
/// A pair already served by a live scope is refcounted, and only when the serving entry is
/// IDENTICAL. A different bundle claiming the same snapshot is refused outright: silently picking
/// one is exactly how a load gets handed a snapshot verified for a different selection.
pub fn prefer_local_snapshots(
    entries: Vec<LocalArtifactOverlayEntry>,
) -> Result<ActiveLocalArtifacts, ArtifactContractError> {
    let mut requested: Vec<LocalArtifactOverlayEntry> = Vec::new();
    for entry in entries {
        match requested
            .iter()
            .find(|held| held.repository == entry.repository && held.revision == entry.revision)
        {
            Some(existing) if existing != &entry => return Err(conflicting_claim(&entry)),
            Some(_) => continue,
            None => requested.push(entry),
        }
    }
    let mut active = lock_active();
    for entry in &requested {
        if let Some(held) = active.entries.iter().find(|held| {
            held.entry.repository == entry.repository && held.entry.revision == entry.revision
        }) {
            if &held.entry != entry {
                return Err(conflicting_claim(entry));
            }
        }
    }
    for entry in &requested {
        match active.entries.iter_mut().find(|held| &held.entry == entry) {
            Some(held) => held.holders += 1,
            None => active.entries.push(HeldEntry {
                entry: entry.clone(),
                holders: 1,
            }),
        }
    }
    drop(active);
    Ok(ActiveLocalArtifacts {
        entries: Arc::new(requested),
    })
}

fn conflicting_claim(entry: &LocalArtifactOverlayEntry) -> ArtifactContractError {
    ArtifactContractError(format!(
        "another resolved-local bundle already claims {}@{}",
        entry.repository, entry.revision
    ))
}

/// Whether this bundle holds every one of `files` (snapshot-relative) for its pair. A snapshot may
/// only be served to consumers whose ENTIRE file requirement lives inside the bundle — the
/// directory-level seam cannot re-ask that question once a path has been handed out.
pub fn entry_covers(entry: &LocalArtifactOverlayEntry, files: &[PathBuf]) -> bool {
    files
        .iter()
        .all(|file| entry.files.binary_search(file).is_ok())
}

/// The leased local snapshot for this exact immutable pair, if one is active AND still on disk.
/// The lease is what keeps a published bundle from being evicted mid-load, so the presence check
/// is belt-and-braces: an answer this function gives must be loadable, never a path the caller
/// discovers is gone.
pub fn local_snapshot(repository: &str, revision: &str) -> Option<PathBuf> {
    lock_active()
        .entries
        .iter()
        .find(|held| {
            held.entry.repository == repository
                && held.entry.revision == revision
                && held.entry.snapshot_root.is_dir()
        })
        .map(|held| held.entry.snapshot_root.clone())
}

/// The leased local snapshot for `repository` when EXACTLY ONE revision of it is active. Used only
/// where the source library cannot answer which revision a load wants (a disconnected library has
/// no `refs/main` to read): with two revisions active the question is genuinely ambiguous and the
/// caller must fail rather than guess.
pub fn unique_local_snapshot(repository: &str) -> Option<(String, PathBuf)> {
    let active = lock_active();
    let mut matches = active
        .entries
        .iter()
        .filter(|held| held.entry.repository == repository && held.entry.snapshot_root.is_dir());
    let first = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some((
        first.entry.revision.clone(),
        first.entry.snapshot_root.clone(),
    ))
}

/// Rewrite an already-confined path that points into the configured source library so it reads
/// from the leased local bundle instead.
///
/// This serves the runtimes whose model directory was resolved to a concrete source path before
/// the load (training base models, captioners, analyzers). Only a path that lies under
/// `<source_library_root>/models--<safe>/snapshots/<revision>/` for a leased pair is rewritten,
/// and only when the rewritten path actually exists in the bundle — otherwise the caller keeps the
/// authoritative source path.
pub fn redirect_source_library_path(source_library_root: &Path, path: &Path) -> Option<PathBuf> {
    let active = lock_active();
    for held in &active.entries {
        let entry = &held.entry;
        let Ok(safe) = safe_repository_dir(&entry.repository) else {
            continue;
        };
        let source_snapshot = source_library_root
            .join(format!("models--{safe}"))
            .join("snapshots")
            .join(&entry.revision);
        // Both the lexical and the canonical form of the source snapshot: callers hand over paths
        // that have already been confined (canonicalized), while the configured library root is
        // whatever the environment named — on macOS those differ for any `/var`-style symlink.
        let mut prefixes = vec![source_snapshot.clone()];
        if let Ok(canonical) = std::fs::canonicalize(&source_snapshot) {
            if canonical != source_snapshot {
                prefixes.push(canonical);
            }
        }
        for prefix in prefixes {
            let Ok(relative) = path.strip_prefix(&prefix) else {
                continue;
            };
            let redirected = entry.snapshot_root.join(relative);
            if redirected.exists() {
                return Some(redirected);
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "local_preference_tests.rs"]
mod tests;
