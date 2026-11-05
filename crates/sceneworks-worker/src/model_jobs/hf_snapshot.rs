/// Resolve the local Hugging Face snapshot directory for a cached repo (the dir that
/// actually holds the checkpoint files). Prefers the commit referenced by
/// `refs/main`, else the first snapshot directory. Returns `None` when the repo is
/// not present in the cache.
///
/// On Windows the resolved snapshot is repaired in place
/// ([`materialize_snapshot_hardlinks`]): the candle worker process cannot traverse
/// the relative `snapshots/<rev>/… -> ../blobs/<etag>` symlinks that `huggingface_hub`
/// (and, before this fix, our own downloader) create — the open fails with
/// `ERROR_UNTRUSTED_MOUNT_POINT` (os error 448), so every model load died at the first
/// file read. The repair replaces those symlinks with hardlinks to the same blob (no
/// reparse point, same volume, still deduped); it is idempotent and best-effort, and a
/// no-op on every other platform.
pub(crate) fn huggingface_snapshot_dir(data_dir: &Path, repo: &str) -> Option<PathBuf> {
    let dir = resolve_huggingface_snapshot_dir(data_dir, repo)?;
    // Windows-only in production (the symlink-traversal defect is Windows-specific); the
    // function itself is still compiled under `test` so the conversion is unit-testable
    // off-Windows, but it must not run against a developer's real HF cache during an
    // unrelated `cargo test` on Unix — hence the call, not just the body, is gated.
    #[cfg(windows)]
    materialize_snapshot_hardlinks(&dir);
    Some(dir)
}

/// Whether `revision` is an immutable pin (F-029) rather than a mutable branch name like `main`.
///
/// A pinned revision is exactly one lowercase 40-hex commit component. This is BOTH a shape check
/// and a confinement precondition: `revision` reaches the snapshot resolvers straight from the job
/// payload's `modelManifestEntry`, which is unvalidated over the LAN jobs API (epic 4484), and a
/// `..`/absolute revision would otherwise join OUTSIDE `snapshots/` — a dir that passes `is_dir()`
/// and, on Windows, gets its symlinks rewritten to hardlinks by `materialize_snapshot_hardlinks`.
/// The manifest schema is not authoritative for the copy embedded in an untrusted job payload
/// (sc-13842 / sc-13583), so the runtime contract is enforced here.
///
/// Extracted (sc-17626) so [`huggingface_pinned_snapshot_dir`] and
/// [`crate::downloads::resolve_hf_component_file`] share one definition of "pinned": the latter
/// must decide, BEFORE calling a resolver, whether to demand the exact snapshot or accept
/// `refs/main`, and a second hand-rolled hex check could drift into accepting a mutable branch
/// where the other demands the pin.
///
/// Note this is not the only pinned/unpinned branch in the crate: [`resolve_co_requisites`] and
/// [`resolve_optional_component`] key off whether the manifest row *declares* a `revision` at all,
/// which is a different question — a row declaring `"main"` is unpinned here but `Some` there.
pub(crate) fn is_pinned_hf_revision(revision: &str) -> bool {
    sceneworks_core::model_artifacts::validate_immutable_revision(revision).is_ok()
}

/// Whether a convert job's declared source file is already materialized in the HF cache — the
/// API-side pre-flight for `POST /models/:id/convert` (see `create_model_convert_job`).
///
/// The worker resolves `checkpoint_dir` the same way ([`huggingface_snapshot_dir`]) and fails the job
/// when the source weights are absent; without this the API happily queued a convert against a
/// half-downloaded snapshot, so a shared-repo family whose sibling cards had already flipped to
/// *downloaded* (the three Anima variants share `circlestone-labs/Anima`, and the per-variant
/// downloads are serialized) surfaced a bare "source DiT is missing" failure while the variant's own
/// 4 GB DiT was still streaming. HF only links `snapshots/<rev>/…` into `blobs/` once a blob
/// completes, so file presence is the authoritative "ready to convert" signal.
///
/// `source_file` is the catalog-declared `mlx.convertSourceFile` (repo-relative); an unsafe path is
/// reported [`ConvertSourceState::FileMissing`] rather than joined.
pub fn convert_source_state(data_dir: &Path, repo: &str, source_file: &str) -> ConvertSourceState {
    let Some(snapshot) = huggingface_snapshot_dir(data_dir, repo) else {
        return ConvertSourceState::RepoNotCached;
    };
    match safe_join(&snapshot, source_file) {
        Ok(path) if path.is_file() => ConvertSourceState::Ready,
        _ => ConvertSourceState::FileMissing,
    }
}

/// The outcome of [`convert_source_state`]. `RepoNotCached` and `FileMissing` are distinguished so
/// the API can say "download it first" vs "this variant's weights have not landed yet" — on a shared
/// repo the second is the common case and the first would be misleading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertSourceState {
    /// The declared source file is on disk; the worker will find it.
    Ready,
    /// The repo has no materialized snapshot in the HF cache at all.
    RepoNotCached,
    /// The snapshot exists but this variant's source file has not landed yet.
    FileMissing,
}

/// Resolve the local HF snapshot dir for a repo PINNED to an exact commit (`snapshots/<revision>/`) —
/// the cache-only twin of [`huggingface_snapshot_dir`] for a download that carries an immutable
/// `revision` (F-029). Unlike [`huggingface_snapshot_dir`] (which prefers `refs/main`), this reads the
/// EXACT pinned snapshot, so a co-requisite pinned to a non-main commit in a SHARED repo (e.g.
/// chatterbox's `ve.safetensors` @ 5bb1f6ee living alongside the primary `refs/main` snapshot) resolves
/// to the right snapshot — mirroring the inference-side `hf_get_pinned`. Returns `None` when the
/// revision is not a pin ([`is_pinned_hf_revision`]) or that pinned snapshot is not cached.
pub(crate) fn huggingface_pinned_snapshot_dir(
    data_dir: &Path,
    repo: &str,
    revision: &str,
) -> Option<PathBuf> {
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
        sceneworks_core::hf_home::model_source_library(data_dir),
    );
    let (_, dir) = resolver
        .discover_source_snapshot(repo, Some(revision))
        .ok()?;
    #[cfg(windows)]
    materialize_snapshot_hardlinks(&dir);
    Some(dir)
}

/// Resolve the one immutable file whose Hugging Face repository was renamed upstream.
///
/// HF's cache namespace includes the repository id, so correcting a manifest from a dead alias to
/// its canonical id would otherwise strand already-downloaded multi-GB blobs under the old cache
/// directory. The shared core declaration is deliberately exact (old id, new id, revision, and
/// files): this is a cache-only migration, never a network fallback and never permission to borrow
/// bytes from a different revision. The shared validator returns the confined plain blob path behind
/// a normal HF snapshot symlink. Returning that file path, rather than the snapshot directory, is
/// important on Windows: opening the snapshot reparse point can fail with
/// `ERROR_UNTRUSTED_MOUNT_POINT` even after the link target was independently validated.
///
/// The preferred result is a hard-linked canonical entry with no duplicate storage. If that
/// cache-only materialization cannot be completed safely (for example, the filesystem does not
/// support hardlinks or a pre-existing destination escapes the canonical cache namespace), return
/// the already-validated plain legacy source directly. Never repair or copy the legacy snapshot:
/// this exact single-file seam must not duplicate the ~3.95 GB blob on a no-hardlink filesystem.
fn resolve_renamed_hf_single_file(
    data_dir: &Path,
    repo: &str,
    revision: &str,
    files: &[String],
) -> Option<PathBuf> {
    resolve_renamed_hf_single_file_using(data_dir, repo, revision, files, |source, destination| {
        std::fs::hard_link(source, destination)
    })
}

/// Create or resolve a relative directory below an already-canonical cache root without following
/// a pre-existing component symlink outside that root. Components are handled one at a time so no
/// `create_dir_all` call can write through an unvalidated parent.
fn ensure_confined_subdirectory(canonical_root: &Path, relative: &Path) -> Option<PathBuf> {
    let mut current = canonical_root.to_path_buf();
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            return None;
        };
        let next = current.join(component);
        match std::fs::symlink_metadata(&next) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match std::fs::create_dir(&next) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(_) => return None,
                }
            }
            Err(_) => return None,
        }
        let canonical_next = std::fs::canonicalize(&next).ok()?;
        if !canonical_next.is_dir() || !canonical_next.starts_with(canonical_root) {
            return None;
        }
        current = canonical_next;
    }
    Some(current)
}

fn resolve_renamed_hf_single_file_using(
    data_dir: &Path,
    repo: &str,
    revision: &str,
    files: &[String],
    hard_link: impl Fn(&Path, &Path) -> std::io::Result<()>,
) -> Option<PathBuf> {
    let rename =
        sceneworks_core::hf_repo_renames::exact_huggingface_repo_rename(repo, revision, files)?;
    let [file] = rename.files else {
        return None;
    };
    let hub_cache_root = huggingface_hub_cache_dir(data_dir);

    // Prefer an already-complete canonical cache, but use the validator-returned plain source path
    // rather than joining through its snapshot entry. This is both the fast path and the Windows
    // reparse-safe path.
    if let Some(source) = confined_renamed_hf_single_file(data_dir, rename.current_repo, rename) {
        return Some(source);
    }

    let legacy_repo_dir = huggingface_repo_cache_path(data_dir, rename.legacy_repo)?;
    let legacy_snapshot = safe_join(&legacy_repo_dir.join("snapshots"), revision).ok()?;
    let (_legacy_snapshot, sources) =
        sceneworks_core::hf_repo_renames::validate_exact_huggingface_repo_rename_snapshot(
            &hub_cache_root,
            &legacy_repo_dir,
            &legacy_snapshot,
            rename,
        )?;
    let [legacy_source] = sources.as_slice() else {
        return None;
    };
    let legacy_source = legacy_source.to_path_buf();

    // Build the canonical destination only after proving both its repository root and snapshot
    // remain inside the configured Hub cache. `safe_join` prevents lexical traversal, while the
    // canonical checks also reject pre-existing directory symlinks that redirect writes outside
    // the cache.
    let current_source = (|| {
        std::fs::create_dir_all(&hub_cache_root).ok()?;
        let canonical_hub_root = std::fs::canonicalize(&hub_cache_root).ok()?;
        let current_repo_dir = huggingface_repo_cache_path(data_dir, repo)?;
        std::fs::create_dir_all(&current_repo_dir).ok()?;
        let canonical_current_root = std::fs::canonicalize(&current_repo_dir).ok()?;
        if !canonical_current_root.starts_with(&canonical_hub_root) {
            return None;
        }
        let current_snapshot = ensure_confined_subdirectory(
            &canonical_current_root,
            &Path::new("snapshots").join(revision),
        )?;

        let relative = Path::new(file);
        let relative_parent = relative.parent()?;
        let canonical_parent = ensure_confined_subdirectory(&current_snapshot, relative_parent)?;
        let destination = canonical_parent.join(relative.file_name()?);
        match std::fs::symlink_metadata(&destination) {
            Ok(_) => {
                // A racing writer or pre-existing entry is accepted only through the same exact
                // confinement validator used by the API. Do not traverse it here.
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match hard_link(&legacy_source, &destination) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(_) => return None,
                }
            }
            Err(_) => return None,
        }
        confined_renamed_hf_single_file(data_dir, rename.current_repo, rename)
    })();

    if let Some(current_source) = current_source {
        tracing::info!(
            event = "hf_renamed_repo_cache_relinked",
            legacy_repo = rename.legacy_repo,
            repo,
            revision,
            source = %current_source.display(),
            "re-linked a renamed Hugging Face repository file from the existing cache"
        );
        Some(current_source)
    } else {
        tracing::info!(
            event = "hf_renamed_repo_cache_legacy_fallback",
            legacy_repo = rename.legacy_repo,
            repo,
            revision,
            source = %legacy_source.display(),
            "using the validated plain legacy Hugging Face file because canonical cache materialization is unavailable"
        );
        Some(legacy_source)
    }
}

/// Resolve the validator-proven plain file behind one side of an exact repository rename. The
/// returned path is confined to that repository cache and never requires the caller to traverse the
/// snapshot's final symlink/reparse point.
fn confined_renamed_hf_single_file(
    data_dir: &Path,
    repo: &str,
    rename: &sceneworks_core::hf_repo_renames::HuggingFaceRepoRename,
) -> Option<PathBuf> {
    let [_file] = rename.files else {
        return None;
    };
    let repo_root = huggingface_repo_cache_path(data_dir, repo)?;
    let snapshot = safe_join(&repo_root.join("snapshots"), rename.revision).ok()?;
    sceneworks_core::hf_repo_renames::validate_exact_huggingface_repo_rename_snapshot(
        &huggingface_hub_cache_dir(data_dir),
        &repo_root,
        &snapshot,
        rename,
    )
    .and_then(|(_, sources)| match sources.as_slice() {
        [source] => Some(source.to_path_buf()),
        _ => None,
    })
}

/// Resolve the exact pinned snapshot for a tier row, first materializing it cache-only from an
/// exact SUPERSEDED revision when the shared core declares one for this row
/// ([`sceneworks_core::hf_revision_supersessions`]).
///
/// The tier loaders read ONLY `snapshots/<pinned sha>/`, so bumping a re-host pin to a commit that
/// merely adds a small file (sc-25909: Qwen-Image 2.1's q8/q4 republished to add each tier's
/// NOTICE) would otherwise strand every install made at the previous commit — the Model Manager
/// still sees the tier on disk while a render fails "not installed". Rows without a declaration
/// resolve exactly as [`huggingface_pinned_snapshot_dir`] always has.
#[cfg_attr(not(native_lane), allow(dead_code))]
pub(crate) fn huggingface_pinned_tier_snapshot_dir(
    data_dir: &Path,
    repo: &str,
    revision: &str,
    files: &[String],
) -> Option<PathBuf> {
    if let Some(supersession) =
        sceneworks_core::hf_revision_supersessions::exact_huggingface_revision_supersession(
            repo, revision, files,
        )
    {
        // The result is advisory: on success the pinned snapshot below now holds the tier; on a
        // refusal the cache is untouched and the ordinary pinned lookup decides, as before.
        let _ = materialize_superseded_hf_revision(data_dir, supersession);
    }
    huggingface_pinned_snapshot_dir(data_dir, repo, revision)
}

/// Build the superseding revision's tier inside the HF cache from the exact legacy snapshot, the
/// cache-only sibling of [`resolve_renamed_hf_single_file`]. Every carried entry is linked to the
/// legacy snapshot's own validated blob (a relative `../../blobs/<etag>` symlink — the layout HF
/// and our downloader write — on Unix, a hardlink on Windows or for a non-blob plain entry), so no
/// weight byte is copied or fetched. Each added file's declared content is written once as its
/// own blob, named by its Hub git oid, and linked the same way. Nothing is written unless the
/// legacy snapshot first proves every carried file at its exact size; a partial legacy tier is
/// refused and left alone. Idempotent: a complete superseding tier returns at once, and an
/// interrupted run only fills the entries that are still missing. Returns the canonical current
/// snapshot when it holds the complete tier.
fn materialize_superseded_hf_revision(
    data_dir: &Path,
    supersession: &sceneworks_core::hf_revision_supersessions::HuggingFaceRevisionSupersession,
) -> Option<PathBuf> {
    use sceneworks_core::hf_revision_supersessions::{
        validate_superseded_legacy_snapshot, validate_superseding_current_snapshot,
    };
    let hub_cache_root = huggingface_hub_cache_dir(data_dir);
    let repo_root = huggingface_repo_cache_path(data_dir, supersession.repo)?;
    let snapshots = repo_root.join("snapshots");
    let current_snapshot = safe_join(&snapshots, supersession.current_revision).ok()?;
    if let Some(snapshot) = validate_superseding_current_snapshot(
        &hub_cache_root,
        &repo_root,
        &current_snapshot,
        supersession,
    ) {
        return Some(snapshot);
    }
    let legacy_snapshot = safe_join(&snapshots, supersession.legacy_revision).ok()?;
    if !legacy_snapshot.is_dir() {
        return None;
    }
    let Some((_, sources)) = validate_superseded_legacy_snapshot(
        &hub_cache_root,
        &repo_root,
        &legacy_snapshot,
        supersession,
    ) else {
        tracing::info!(
            event = "hf_superseded_revision_cache_refused",
            repo = supersession.repo,
            legacy_revision = supersession.legacy_revision,
            revision = supersession.current_revision,
            files = ?supersession.files,
            "the superseded snapshot does not hold the complete carried tier; leaving the cache untouched"
        );
        return None;
    };

    let materialized = (|| {
        let canonical_hub_root = std::fs::canonicalize(&hub_cache_root).ok()?;
        let canonical_root = std::fs::canonicalize(&repo_root).ok()?;
        if !canonical_root.starts_with(&canonical_hub_root) {
            return None;
        }
        let current = ensure_confined_subdirectory(
            &canonical_root,
            &Path::new("snapshots").join(supersession.current_revision),
        )?;
        for (file, source) in supersession.carried.iter().zip(&sources) {
            link_superseded_snapshot_entry(&canonical_root, &current, file.path, source)?;
        }
        for added in supersession.added {
            let blobs = ensure_confined_subdirectory(&canonical_root, Path::new("blobs"))?;
            let blob = write_declared_blob(&blobs, added)?;
            link_superseded_snapshot_entry(&canonical_root, &current, added.path, &blob)?;
        }
        validate_superseding_current_snapshot(
            &hub_cache_root,
            &repo_root,
            &current_snapshot,
            supersession,
        )
    })();
    match &materialized {
        Some(snapshot) => tracing::info!(
            event = "hf_superseded_revision_cache_relinked",
            repo = supersession.repo,
            legacy_revision = supersession.legacy_revision,
            revision = supersession.current_revision,
            files = ?supersession.files,
            snapshot = %snapshot.display(),
            "materialized a superseding Hugging Face revision from the existing cache"
        ),
        None => tracing::info!(
            event = "hf_superseded_revision_cache_unavailable",
            repo = supersession.repo,
            legacy_revision = supersession.legacy_revision,
            revision = supersession.current_revision,
            files = ?supersession.files,
            "could not materialize the superseding Hugging Face revision from the existing cache"
        ),
    }
    materialized
}

/// Link one superseding snapshot entry to an already-validated, repo-confined source. A
/// pre-existing entry is left for the final whole-tier validation to accept or reject; it is never
/// traversed or replaced here.
fn link_superseded_snapshot_entry(
    canonical_root: &Path,
    snapshot: &Path,
    relative: &str,
    source: &Path,
) -> Option<()> {
    let relative = Path::new(relative);
    let parent = ensure_confined_subdirectory(snapshot, relative.parent()?)?;
    let destination = parent.join(relative.file_name()?);
    match std::fs::symlink_metadata(&destination) {
        Ok(_) => return Some(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return None,
    }
    let linked = match source.strip_prefix(canonical_root.join("blobs")) {
        #[cfg(unix)]
        Ok(blob) => {
            // `snapshots/<rev>/q8/processor/x` → `../../../../blobs/<etag>`: one `..` per component
            // between the repository root and the entry's parent, exactly as the downloader writes.
            let depth = parent
                .strip_prefix(canonical_root)
                .ok()?
                .components()
                .count();
            let mut target = PathBuf::new();
            for _ in 0..depth {
                target.push("..");
            }
            target.push("blobs");
            target.push(blob);
            std::os::unix::fs::symlink(&target, &destination)
        }
        _ => std::fs::hard_link(source, &destination),
    };
    match linked {
        Ok(()) => Some(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Some(()),
        Err(_) => None,
    }
}

/// Write a declared added file's content as its repo-local blob (once), returning the blob path.
/// An existing blob of that name is accepted only when it holds exactly the declared bytes.
fn write_declared_blob(
    blobs: &Path,
    added: &sceneworks_core::hf_revision_supersessions::AddedRevisionFile,
) -> Option<PathBuf> {
    if format!("{:x}", Sha256::digest(added.contents)) != added.sha256 {
        return None;
    }
    let blob = safe_join(blobs, added.blob).ok()?;
    match std::fs::symlink_metadata(&blob) {
        Ok(metadata) => {
            return (metadata.is_file() && std::fs::read(&blob).ok()? == added.contents)
                .then_some(blob);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return None,
    }
    let temporary = blobs.join(format!(
        "{}.sceneworks-{}.incomplete",
        added.blob,
        std::process::id()
    ));
    std::fs::write(&temporary, added.contents).ok()?;
    if std::fs::rename(&temporary, &blob).is_err() {
        let _ = std::fs::remove_file(&temporary);
        return (std::fs::read(&blob).ok()? == added.contents).then_some(blob);
    }
    Some(blob)
}

/// Stage a superseded legacy tier in the Hugging Face cache layout: every carried file at its
/// exact declared size (sparse, so multi-GB weights cost no disk), either as `../blobs/<id>`
/// symlinks (`symlinked`) or as plain snapshot files. `contents` overrides the leading bytes of
/// named files (padded with spaces to the declared size) so a loader probe can read them.
#[cfg(all(test, unix))]
pub(crate) fn stage_superseded_legacy_tier_for_test(
    data_dir: &Path,
    supersession: &sceneworks_core::hf_revision_supersessions::HuggingFaceRevisionSupersession,
    symlinked: bool,
    contents: &[(&str, &[u8])],
) -> PathBuf {
    let repo_root = huggingface_repo_cache_path(data_dir, supersession.repo).expect("repo cache");
    let snapshot = repo_root
        .join("snapshots")
        .join(supersession.legacy_revision);
    let blobs = repo_root.join("blobs");
    std::fs::create_dir_all(&blobs).expect("blobs dir");
    for (index, file) in supersession.carried.iter().enumerate() {
        let entry = snapshot.join(file.path);
        let parent = entry.parent().expect("entry parent");
        std::fs::create_dir_all(parent).expect("snapshot dirs");
        let blob_name = format!("legacy-blob-{index:02}-{}", file.path.replace('/', "-"));
        let target = if symlinked {
            blobs.join(&blob_name)
        } else {
            entry.clone()
        };
        let mut bytes = contents
            .iter()
            .find(|(path, _)| *path == file.path)
            .map(|(_, bytes)| bytes.to_vec())
            .unwrap_or_default();
        assert!(
            bytes.len() as u64 <= file.size,
            "{} override too large",
            file.path
        );
        if file.size <= 1 << 20 {
            bytes.resize(file.size as usize, b' ');
        }
        std::fs::write(&target, &bytes).expect("write staged blob");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&target)
            .and_then(|handle| handle.set_len(file.size))
            .expect("size staged blob");
        if symlinked {
            let depth = parent
                .strip_prefix(&repo_root)
                .expect("under repo")
                .components()
                .count();
            let mut relative = PathBuf::new();
            for _ in 0..depth {
                relative.push("..");
            }
            std::os::unix::fs::symlink(relative.join("blobs").join(&blob_name), &entry)
                .expect("snapshot symlink");
        }
    }
    snapshot
}

/// Resolve a model's **caller-provisioned components** (epic 13657, sc-13679) — the generic worker
/// seam that turns a provider's [`gen_core::ModelDescriptor::required_components`] advertisement into
/// the resolved local paths a caller stages in [`gen_core::LoadSpec::components`] (via
/// `LoadSpec::with_component`). It is the SceneWorks-side complement of the engine's own load-time
/// [`gen_core::require_component`] gate: the descriptor advertises which component ids a model needs,
/// the manifest declares WHERE each comes from (a `coRequisite` download tagged with an explicit
/// `componentId`), and this resolves each to a local [`gen_core::WeightsSource`].
///
/// Driven PURELY by `descriptor.required_components`: for each declared id it finds the manifest
/// hard `coRequisite` download whose `componentId` matches (never inferred from repo names,
/// sc-13679), then resolves that repo's cached snapshot ENV-CORRECTLY, cache-only. A soft component
/// is request-specific and must use [`resolve_optional_component`] or a stricter dedicated route;
/// it can never satisfy a provider's unconditional `required_components` advertisement. A
/// co-requisite that pins an immutable `revision` (F-029) reads that exact `snapshots/<sha>/` via
/// [`huggingface_pinned_snapshot_dir`]; an unpinned one falls back to the `refs/main`-preferring
/// [`huggingface_snapshot_dir`] — the same seams the PiD-gemma / Wan-Lightning co-requisites use, so
/// a custom `HF_HOME`/hub is honored and nothing is fetched here.
///
/// All-or-nothing: a declared component whose co-requisite is not installed (snapshot absent, or the
/// named file missing) returns a caller-actionable [`WorkerError::InvalidPayload`] naming the
/// component id + repo, so the JOB fails BEFORE the engine load — never a mid-render hub fetch. An
/// empty `required_components` (every image/video model and every single-file audio model except
/// chatterbox today) returns an empty map with no filesystem access, so every existing lane is a
/// no-op. A single-file co-requisite (`files: ["x.safetensors"]`, e.g. chatterbox
/// `perth`/`voice_embedding`) resolves to `WeightsSource::File(<snapshot>/x.safetensors)`; a
/// multi-file / whole-repo co-requisite resolves to `WeightsSource::Dir(<snapshot>)`.
pub(crate) fn resolve_co_requisites(
    descriptor: &gen_core::ModelDescriptor,
    manifest_entry: &Value,
    settings: &Settings,
) -> WorkerResult<BTreeMap<String, gen_core::WeightsSource>> {
    resolve_co_requisites_for_tier(descriptor, manifest_entry, settings, None)
}

/// [`resolve_co_requisites`], for a model whose co-requisites are themselves **per-tier** (sc-14980).
///
/// Mage-Flow hosts its shared Qwen3-VL text encoder and Mage-VAE once, as `q4/`+`q8/`+`bf16/`
/// subtrees of one components mirror, so a component id maps to THREE `coRequisite` rows that differ
/// only by `variant`. `tier` selects among them, and each row's `subdir` addresses the component dir
/// inside the shared snapshot (`resolve_co_requisites` otherwise resolves whole-snapshot dirs, which
/// cannot express "the `q4/text_encoder/` subtree").
///
/// Behavior is unchanged wherever a component id maps to exactly ONE row — every audio co-requisite
/// today — so this is additive. When a component id maps to several rows, a `tier` is REQUIRED: a
/// first-wins pick would silently serve a bf16 text encoder to a q4 request, which is precisely the
/// class of silent tier mismatch physical per-tier artifacts exist to eliminate.
pub(crate) fn resolve_co_requisites_for_tier(
    descriptor: &gen_core::ModelDescriptor,
    manifest_entry: &Value,
    settings: &Settings,
    tier: Option<&str>,
) -> WorkerResult<BTreeMap<String, gen_core::WeightsSource>> {
    let mut components = BTreeMap::new();
    if descriptor.required_components.is_empty() {
        return Ok(components);
    }
    let model_id = descriptor.id;
    let downloads: &[Value] = manifest_entry
        .get("downloads")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    for &component_id in descriptor.required_components {
        // The `coRequisite` download(s) that PROVISION this component id — matched on the explicit
        // `componentId` field the manifest sets (never inferred from repo names, sc-13679).
        let candidates: Vec<&Value> = downloads
            .iter()
            .filter(|download| {
                download.get("coRequisite").and_then(Value::as_bool) == Some(true)
                    && download.get("componentId").and_then(Value::as_str) == Some(component_id)
                    && download.get("required").and_then(Value::as_str) != Some("soft")
            })
            .collect();
        let download = match candidates.as_slice() {
            [] => {
                if downloads.iter().any(|download| {
                    download.get("coRequisite").and_then(Value::as_bool) == Some(true)
                        && download.get("componentId").and_then(Value::as_str) == Some(component_id)
                        && download.get("required").and_then(Value::as_str) == Some("soft")
                }) {
                    return Err(WorkerError::InvalidPayload(format!(
                        "{model_id} requires the model component '{component_id}', but its catalog \
                         row is `required: soft` — optional components cannot satisfy \
                         descriptor.required_components and must be resolved only by the \
                         request-specific route"
                    )));
                }
                return Err(WorkerError::InvalidPayload(format!(
                "{model_id} requires the model component '{component_id}', but its catalog entry \
                     declares no `coRequisite` download tagged `componentId: \"{component_id}\"` — \
                     the model manifest entry is misconfigured"
            )));
            }
            [only] => *only,
            many => {
                let tier = tier.ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "{model_id}: the '{component_id}' component declares {} per-tier \
                         `coRequisite` rows but no tier was resolved for this request — refusing to \
                         guess, because picking the wrong one would silently serve another tier's \
                         weights",
                        many.len()
                    ))
                })?;
                many.iter()
                    .copied()
                    .find(|download| {
                        download
                            .get("variant")
                            .and_then(Value::as_str)
                            .is_some_and(|variant| variant.eq_ignore_ascii_case(tier))
                    })
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "{model_id}: the '{component_id}' component declares no `coRequisite` row \
                             for the resolved {tier} tier — install {model_id}'s {tier} tier, or the \
                             model manifest entry is misconfigured"
                        ))
                    })?
            }
        };
        let repo = download
            .get("repo")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{model_id}: the '{component_id}' co-requisite download declares no `repo`"
                ))
            })?;
        let files: Vec<String> = download
            .get("files")
            .and_then(Value::as_array)
            .map(|files| {
                files
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        // Resolve the co-requisite's cache location cache-only. A co-requisite pins an immutable `revision`
        // (F-029, required by sc-13659) — read THAT exact `snapshots/<sha>/` (like the inference-side
        // `hf_get_pinned`), never `refs/main`, so a companion weight pinned to a non-main commit in a
        // SHARED repo (chatterbox's `ve.safetensors` @ 5bb1f6ee, alongside the primary `refs/main`
        // snapshot) resolves correctly. An unpinned co-requisite falls back to the `refs/main`-preferring
        // resolver.
        enum ComponentCacheLocation {
            ExactRenamedFile(PathBuf),
            Snapshot(PathBuf),
        }
        let location = match download.get("revision").and_then(Value::as_str) {
            Some(revision)
                if sceneworks_core::hf_repo_renames::exact_huggingface_repo_rename(
                    repo, revision, &files,
                )
                .is_some() =>
            {
                resolve_renamed_hf_single_file(&settings.data_dir, repo, revision, &files)
                    .map(ComponentCacheLocation::ExactRenamedFile)
            }
            Some(revision) => huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision)
                .map(ComponentCacheLocation::Snapshot),
            None => huggingface_snapshot_dir(&settings.data_dir, repo)
                .map(ComponentCacheLocation::Snapshot),
        }
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{model_id}: the '{component_id}' component ({repo}) is not installed — install \
                 {model_id} from the Model Manager (installing it pulls this co-requisite), or \
                 predownload {repo} for an offline install"
            ))
        })?;
        let source = match location {
            // The exact rename validator already proved this plain path is the one declared file,
            // readable and confined to the current or legacy repository cache. Do not join back
            // through the snapshot symlink: Windows may reject that reparse point at open time.
            ComponentCacheLocation::ExactRenamedFile(path) => gen_core::WeightsSource::File(path),
            ComponentCacheLocation::Snapshot(snapshot) => match files.as_slice() {
                // A single LITERAL filename resolves to that file (chatterbox's `ve.safetensors`). A
                // single GLOB does not: it names a set, so it must take the directory arm below like any
                // other multi-entry predicate (sc-14980 — Mage's `["q4/text_encoder/*"]`). Without this
                // guard the pattern is `safe_join`ed verbatim, `is_file()` is false for a path with a
                // literal `*` in it, and a correctly-installed component reports as missing.
                [file] if !file.contains(['*', '?', '[']) => {
                    // `file` comes from the payload `modelManifestEntry`; confine it under the snapshot so
                    // a `..`/absolute entry can't stage an arbitrary host file as this component's weights
                    // (on Windows `join` with an absolute path replaces the base entirely) (sc-13583).
                    let path = safe_join(&snapshot, file)?;
                    if !path.is_file() {
                        return Err(WorkerError::InvalidPayload(format!(
                        "{model_id}: the '{component_id}' component file {repo}/{file} is missing \
                         from the cached snapshot — reinstall {model_id} to repair it"
                    )));
                    }
                    gen_core::WeightsSource::File(path)
                }
                [] => gen_core::WeightsSource::Dir(component_dir(&snapshot, download, model_id)?),
                _ => {
                    // Multi-file components are staged as directories because providers join their own
                    // internal paths, but the pre-load all-or-nothing contract still requires every
                    // manifest-declared file or pattern to be present. The declared patterns are
                    // SNAPSHOT-relative (they are what the downloader fetched), while `subdir` narrows
                    // what the provider is handed — so check against the snapshot, stage the subdir.
                    for file in &files {
                        if !snapshot_contains_declared_file(&snapshot, file)? {
                            return Err(WorkerError::InvalidPayload(format!(
                            "{model_id}: the '{component_id}' component file {repo}/{file} is missing \
                             from the cached snapshot — reinstall {model_id} to repair it"
                        )));
                        }
                    }
                    gen_core::WeightsSource::Dir(component_dir(&snapshot, download, model_id)?)
                }
            },
        };
        components.insert(component_id.to_owned(), source);
    }
    Ok(components)
}

/// The directory a co-requisite download stages: the cached snapshot, narrowed by an optional
/// `subdir` (sc-14980).
///
/// A shared components mirror hosts several components AND several tiers in one repo
/// (`<tier>/text_encoder/`, `<tier>/vae/`), so "the snapshot" is not the thing a provider should be
/// handed. `subdir` names the exact component dir. It arrives from the payload `modelManifestEntry`
/// — unvalidated over the LAN jobs API — so it is confined under the snapshot with [`safe_join`] for
/// the same reason `files` and `revision` are (sc-13583 / sc-13842): a `..`/absolute entry would
/// otherwise stage an arbitrary host directory as this component's weights.
fn component_dir(snapshot: &Path, download: &Value, model_id: &str) -> WorkerResult<PathBuf> {
    let Some(subdir) = crate::payload::trimmed_string(download.get("subdir")) else {
        return Ok(snapshot.to_path_buf());
    };
    let dir = safe_join(snapshot, subdir)?;
    if !dir.is_dir() {
        return Err(WorkerError::InvalidPayload(format!(
            "{model_id}: the co-requisite subdirectory `{subdir}` is missing from the cached \
             snapshot — reinstall {model_id} to repair it"
        )));
    }
    Ok(dir)
}

fn snapshot_contains_declared_file(snapshot: &Path, declared: &str) -> WorkerResult<bool> {
    if !declared.contains(['*', '?', '[']) {
        return Ok(safe_join(snapshot, declared)?.is_file());
    }
    let relative = Path::new(declared);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(WorkerError::InvalidPayload(format!(
            "model component file pattern `{declared}` must stay within its cached snapshot"
        )));
    }
    let pattern = glob::Pattern::new(declared).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "invalid model component file pattern `{declared}`: {error}"
        ))
    })?;
    Ok(snapshot_tree_contains(snapshot, &|path| {
        path.strip_prefix(snapshot)
            .is_ok_and(|relative| pattern.matches_path(relative))
    }))
}

fn snapshot_tree_contains(dir: &Path, predicate: &impl Fn(&Path) -> bool) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(file_type) if file_type.is_dir() => {
                if snapshot_tree_contains(&path, predicate) {
                    return true;
                }
            }
            Ok(_) if predicate(&path) => return true,
            Ok(_) | Err(_) => {}
        }
    }
    false
}

/// Resolve an OPTIONAL, on-demand model component the caller stages only for specific requests — the
/// counterpart to [`resolve_co_requisites`] for a component deliberately kept OUT of the descriptor's
/// `required_components` so the generic seam never stages it. The motivating case is acestep's
/// `sft_cover` Cover snapshot (sc-13821): a ~8.76 GB checkpoint read only for a Cover restyle, Cover-only
/// and lazy, mirroring LTX's `uncensored_enhancer` / sensenova's `distill_lora`.
///
/// Returns `Some(source)` when the manifest declares a `coRequisite` download tagged
/// `componentId: <component_id>` AND its pinned snapshot is installed on disk; `None` when the manifest
/// declares no such component (an older inference pin, or a different model) OR its snapshot is not
/// installed. `None` is a deliberate no-op: an optional component is never install-gating (a `soft`
/// co-requisite leaves text2music + the region edit modes usable), and the engine surfaces its own
/// actionable "component not staged" error on the one request that actually needs the component — so
/// the worker does not pre-empt that with a hard error here (which on the pre-adoption inference pin,
/// whose Cover path still self-fetches, would be a regression). Snapshot resolution mirrors
/// [`resolve_co_requisites`]: a pinned `revision` reads the exact `snapshots/<sha>/` (never `refs/main`).
pub(crate) fn resolve_optional_component(
    manifest_entry: &Value,
    component_id: &str,
    settings: &Settings,
) -> Option<gen_core::WeightsSource> {
    let downloads = manifest_entry.get("downloads").and_then(Value::as_array)?;
    let download = downloads.iter().find(|download| {
        download.get("coRequisite").and_then(Value::as_bool) == Some(true)
            && download.get("componentId").and_then(Value::as_str) == Some(component_id)
    })?;
    let repo = download.get("repo").and_then(Value::as_str)?;
    let snapshot = match download.get("revision").and_then(Value::as_str) {
        Some(revision) => huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision),
        None => huggingface_snapshot_dir(&settings.data_dir, repo),
    }?;
    let files: Vec<&str> = download
        .get("files")
        .and_then(Value::as_array)
        .map(|files| files.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    // A single concrete file resolves to that file; a multi-file / glob component (sft_cover's
    // `transformer/*` + the two FSQ dirs) resolves to the snapshot ROOT — the provider joins its own
    // internal subpaths (candle-audio-acestep reads `transformer/`, `audio_tokenizer/`,
    // `audio_token_detokenizer/` under the staged dir).
    match files.as_slice() {
        [file] if !file.contains('*') => {
            // Confine the payload-supplied file under the snapshot (sc-13583) — the optional-component
            // twin of the `resolve_co_requisites` guard above; a `..`/absolute entry yields `None`
            // (component simply not staged), never an out-of-snapshot host file.
            let path = safe_join(&snapshot, file).ok()?;
            match sceneworks_core::managed_file::ManagedFileIdentity::from_download(download) {
                Ok(Some(identity)) if identity.verify(&path).is_err() => return None,
                Err(_) => return None,
                _ => {}
            }
            path.is_file()
                .then_some(gen_core::WeightsSource::File(path))
        }
        _ => Some(gen_core::WeightsSource::Dir(snapshot)),
    }
}

/// Resolve the exact snapshot/tier recorded by a completed model-download receipt.
///
/// A manifest may rename files after an install.  In that case the catalog deliberately keeps the
/// install usable (sc-13076), and generation must use the receipt's *entire* resolved file set rather
/// than combining files from the receipt with current-manifest names.  A receipt is accepted only
/// when every recorded file exists in one snapshot directory; a torn set returns `None` atomically.
/// `model_id` narrows primary-model receipts, while the repo-wide fallback also covers co-requisite
/// downloads whose marker directory is not known to the loader.
pub(crate) type ResolvedArtifactIdentity = sceneworks_core::model_artifacts::ArtifactIdentity;
pub(crate) type ResolvedArtifactProvenance = sceneworks_core::model_artifacts::ArtifactProvenance;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedWeights {
    pub(crate) path: PathBuf,
    pub(crate) provenance: Option<ResolvedArtifactProvenance>,
    receipt: Value,
    marker: PathBuf,
    snapshot: PathBuf,
}
