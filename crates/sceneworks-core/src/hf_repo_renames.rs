//! Exact cache-compatibility declarations for Hugging Face repositories renamed upstream.
//!
//! A Hugging Face cache namespace includes the repository id, so correcting a manifest can strand
//! an otherwise valid immutable snapshot under the former id. These declarations are intentionally
//! exact across the old id, current id, pinned revision, and complete file list. They permit only a
//! cache-local compatibility lookup; they do not authorize network redirects or revision drift.

use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct HuggingFaceRepoRename {
    pub legacy_repo: &'static str,
    pub current_repo: &'static str,
    pub revision: &'static str,
    pub files: &'static [&'static str],
}

pub const MMAUDIO_DFN5B_CLIP_RENAME: HuggingFaceRepoRename = HuggingFaceRepoRename {
    legacy_repo: "apple/DFN5B-CLIP-ViT-H-14-384",
    current_repo: "apple/DFN5B-CLIP-ViT-H-14-378",
    revision: "01b771ed0d1395ca5ffdd279897d665ebe00dfd2",
    files: &["open_clip_pytorch_model.bin"],
};

const RENAMES: &[HuggingFaceRepoRename] = &[MMAUDIO_DFN5B_CLIP_RENAME];

pub fn exact_huggingface_repo_rename(
    current_repo: &str,
    revision: &str,
    files: &[String],
) -> Option<&'static HuggingFaceRepoRename> {
    RENAMES.iter().find(|rename| {
        rename.current_repo == current_repo
            && rename.revision == revision
            && files.len() == rename.files.len()
            && files
                .iter()
                .zip(rename.files)
                .all(|(actual, expected)| actual == expected)
    })
}

/// Resolve every file in an exact legacy snapshot and prove that its materialized bytes remain
/// inside that repository's Hugging Face cache directory.
///
/// Snapshot entries are normally symlinks into the repository-local `blobs/` directory, so files
/// cannot be confined to the lexical snapshot directory itself. Canonicalizing both the snapshot
/// and each file against the canonical repository root accepts that normal layout while rejecting
/// an escaping snapshot or file symlink. The repository root is independently confined to the
/// configured Hub cache so an escaping `models--...` symlink is rejected too.
pub fn validate_exact_huggingface_repo_rename_snapshot(
    hub_cache_root: &Path,
    legacy_repo_root: &Path,
    legacy_snapshot: &Path,
    rename: &HuggingFaceRepoRename,
) -> Option<(PathBuf, Vec<PathBuf>)> {
    validate_huggingface_snapshot_files(
        hub_cache_root,
        legacy_repo_root,
        legacy_snapshot,
        rename.files,
    )
}

/// Generic implementation of [`validate_exact_huggingface_repo_rename_snapshot`], also used to
/// validate the canonical side of a rename before API readiness or worker resolution accepts it.
pub fn validate_huggingface_snapshot_files(
    hub_cache_root: &Path,
    repo_root: &Path,
    snapshot: &Path,
    files: &[&str],
) -> Option<(PathBuf, Vec<PathBuf>)> {
    let canonical_hub_root = std::fs::canonicalize(hub_cache_root).ok()?;
    let canonical_repo_root = std::fs::canonicalize(repo_root).ok()?;
    if !canonical_repo_root.starts_with(&canonical_hub_root) {
        return None;
    }
    let canonical_snapshot = std::fs::canonicalize(snapshot).ok()?;
    if !canonical_snapshot.is_dir() || !canonical_snapshot.starts_with(&canonical_repo_root) {
        return None;
    }

    let sources = files
        .iter()
        .map(|file| {
            let relative = Path::new(file);
            if relative.is_absolute()
                || relative
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
            {
                return None;
            }
            let snapshot_entry = canonical_snapshot.join(relative);
            let canonical_parent = std::fs::canonicalize(snapshot_entry.parent()?).ok()?;
            if !canonical_parent.starts_with(&canonical_repo_root) {
                return None;
            }
            let snapshot_entry = canonical_parent.join(relative.file_name()?);
            // Windows may refuse to open/canonicalize a Hugging Face snapshot symlink with
            // ERROR_UNTRUSTED_MOUNT_POINT. Inspect the final entry without traversing it, then
            // canonicalize its plain repository-local blob target instead. Normal files keep the
            // ordinary canonicalization path. Parent directories were already confined above.
            let metadata = std::fs::symlink_metadata(&snapshot_entry).ok()?;
            let source = if metadata.file_type().is_symlink() {
                let target = std::fs::read_link(&snapshot_entry).ok()?;
                let target = if target.is_absolute() {
                    target
                } else {
                    canonical_parent.join(target)
                };
                std::fs::canonicalize(target).ok()?
            } else {
                std::fs::canonicalize(snapshot_entry).ok()?
            };
            (source.is_file() && source.starts_with(&canonical_repo_root)).then_some(source)
        })
        .collect::<Option<Vec<_>>>()?;
    Some((canonical_snapshot, sources))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rename_lookup_requires_the_exact_repo_revision_and_file_set() {
        let files = vec!["open_clip_pytorch_model.bin".to_owned()];
        assert_eq!(
            exact_huggingface_repo_rename(
                MMAUDIO_DFN5B_CLIP_RENAME.current_repo,
                MMAUDIO_DFN5B_CLIP_RENAME.revision,
                &files,
            ),
            Some(&MMAUDIO_DFN5B_CLIP_RENAME)
        );
        assert!(exact_huggingface_repo_rename(
            MMAUDIO_DFN5B_CLIP_RENAME.current_repo,
            "wrong-revision",
            &files,
        )
        .is_none());
        assert!(exact_huggingface_repo_rename(
            MMAUDIO_DFN5B_CLIP_RENAME.current_repo,
            MMAUDIO_DFN5B_CLIP_RENAME.revision,
            &["different.bin".to_owned()],
        )
        .is_none());
    }

    #[cfg(unix)]
    #[test]
    fn exact_snapshot_validation_accepts_repo_local_blobs_and_rejects_escape() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("temporary cache");
        let hub = temp.path().join("hub");
        let repo = hub.join("models--apple--DFN5B-CLIP-ViT-H-14-384");
        let snapshot = repo
            .join("snapshots")
            .join(MMAUDIO_DFN5B_CLIP_RENAME.revision);
        let blobs = repo.join("blobs");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        let blob = blobs.join("clip");
        std::fs::write(&blob, b"weights").unwrap();
        let snapshot_file = snapshot.join(MMAUDIO_DFN5B_CLIP_RENAME.files[0]);
        symlink(&blob, &snapshot_file).unwrap();

        let (validated_snapshot, sources) = validate_exact_huggingface_repo_rename_snapshot(
            &hub,
            &repo,
            &snapshot,
            &MMAUDIO_DFN5B_CLIP_RENAME,
        )
        .expect("a normal snapshot-to-local-blob link is valid");
        assert_eq!(
            validated_snapshot,
            std::fs::canonicalize(&snapshot).unwrap()
        );
        assert_eq!(sources, vec![std::fs::canonicalize(&blob).unwrap()]);

        std::fs::remove_file(&snapshot_file).unwrap();
        let outside = temp.path().join("outside.bin");
        std::fs::write(&outside, b"outside").unwrap();
        symlink(&outside, &snapshot_file).unwrap();
        assert!(validate_exact_huggingface_repo_rename_snapshot(
            &hub,
            &repo,
            &snapshot,
            &MMAUDIO_DFN5B_CLIP_RENAME,
        )
        .is_none());
    }
}
