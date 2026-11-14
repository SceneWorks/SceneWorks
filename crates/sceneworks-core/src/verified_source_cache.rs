//! Hash-once-per-host verification of immutable source files read in place.
//!
//! A pinned Hugging Face snapshot is content-addressed: its file bytes cannot change while their
//! filesystem identity is unchanged. A job that binds such a file verifies its SHA-256 against the
//! frozen ledger once, records the verdict keyed by the file's real path and the same
//! filesystem stamp the checkpoint-plan store trusts ([`SourceStampV1`]: size, mtime and, on Unix,
//! device, inode and ctime), and later attempts reuse that verdict while the stamp is unchanged.
//! Any stamp change, a racy stamp (modified within one timestamp tick of its observation), a
//! different expected digest or an unreadable record re-hashes the file. A mismatch is never
//! recorded. A file with no ledger in advance (a local model captured by its bytes) records the
//! digest it hashed to, under the same key, so a later ledger check of those bytes reuses it.
use crate::checkpoint_plan_store::{source_file_sha256_with_check, SourceStampV1, StoreClockV1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io;
use std::path::{Path, PathBuf};

const RECORD_SCHEMA_VERSION: u32 = 1;

/// Whether a filesystem stamp may stand in for a hash of a mutable local file (a user model,
/// adapter, control or produced tree, as opposed to a content-addressed hub blob).
///
/// Not on Windows: there a same-length in-place rewrite that restores the last-write time leaves
/// every field std exposes unchanged (size, creation time, last-write time). The NTFS change time,
/// which no ordinary timestamp API lets a caller restore, is reachable only through
/// `GetFileInformationByHandleEx` (std's `MetadataExt::change_time` is unstable at the pinned
/// toolchain), and that FFI call needs `unsafe`, which this workspace forbids. So Windows hashes
/// mutable local files fresh on every check.
const STAMPS_IDENTIFY_MUTABLE_FILES: bool = cfg!(not(windows));

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct VerifiedSourceRecordV1 {
    schema_version: u32,
    real_path: PathBuf,
    sha256: String,
    stamp: SourceStampV1,
}

/// How a file was verified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceVerification {
    /// A recorded verdict for the unchanged file was reused; nothing was read.
    Reused,
    /// The file was read and hashed, and the verdict recorded.
    Hashed,
}

/// The verdict store for one host (one data directory).
#[derive(Clone, Debug)]
pub struct VerifiedSourceCache {
    root: PathBuf,
}

/// Where a host keeps its verdicts, relative to its data directory. The API (job admission) and
/// the worker (job preparation) both open the store here, so whichever checks a file first on a
/// host records the verdict the other reuses.
pub const VERIFIED_SOURCES_RELATIVE_DIR: &str = "models/portable-attempts/.verified-sources";

impl VerifiedSourceCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// This host's store under `data_dir` ([`VERIFIED_SOURCES_RELATIVE_DIR`]).
    pub fn for_data_dir(data_dir: &Path) -> Self {
        Self::new(data_dir.join(VERIFIED_SOURCES_RELATIVE_DIR))
    }

    fn record_path(&self, real_path: &Path) -> PathBuf {
        let digest = Sha256::digest(real_path.as_os_str().as_encoded_bytes());
        self.root.join(format!("{digest:x}.json"))
    }

    /// The digest recorded for `real_path` while its stamp is still `current` and not racy.
    fn recorded_sha256(&self, real_path: &Path, current: &SourceStampV1) -> Option<String> {
        let body = std::fs::read(self.record_path(real_path)).ok()?;
        let record = serde_json::from_slice::<VerifiedSourceRecordV1>(&body).ok()?;
        (record.schema_version == RECORD_SCHEMA_VERSION
            && record.real_path == real_path
            && !record.stamp.is_racy()
            && record.stamp.identifies_same_entry(current))
        .then_some(record.sha256)
    }

    /// Hash `real_path` (`sha256:<hex>`), failing if its stamp moved from `before` meanwhile.
    fn hash(
        real_path: &Path,
        before: &SourceStampV1,
        check: &dyn Fn() -> io::Result<()>,
    ) -> io::Result<(String, SourceStampV1)> {
        let digest = source_file_sha256_with_check(real_path, check)?;
        let after = SourceStampV1::of(real_path, StoreClockV1::SYSTEM)?;
        if !before.identifies_same_entry(&after) {
            return Err(io::Error::other(format!(
                "{} changed while it was verified",
                real_path.display()
            )));
        }
        Ok((format!("sha256:{digest}"), after))
    }

    fn record(&self, real_path: &Path, sha256: &str, stamp: SourceStampV1) -> io::Result<()> {
        let record = VerifiedSourceRecordV1 {
            schema_version: RECORD_SCHEMA_VERSION,
            real_path: real_path.to_path_buf(),
            sha256: sha256.to_owned(),
            stamp,
        };
        std::fs::create_dir_all(&self.root)?;
        let body = serde_json::to_vec(&record).map_err(io::Error::other)?;
        crate::fsutil::write_atomic(&self.record_path(real_path), &body)
    }

    /// Verify that `real_path` (already canonical) holds exactly `expected_sha256`
    /// (`sha256:<hex>`). A digest mismatch is an `InvalidData` error; a file that changed while it
    /// was hashed is an error too. `check` is polled between reads (cancellation).
    pub fn verify(
        &self,
        real_path: &Path,
        expected_sha256: &str,
        check: &dyn Fn() -> io::Result<()>,
    ) -> io::Result<SourceVerification> {
        let before = SourceStampV1::of(real_path, StoreClockV1::SYSTEM)?;
        if self.recorded_sha256(real_path, &before).as_deref() == Some(expected_sha256) {
            return Ok(SourceVerification::Reused);
        }
        let (digest, after) = Self::hash(real_path, &before, check)?;
        if digest != expected_sha256 {
            let _ = std::fs::remove_file(self.record_path(real_path));
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} does not match its pinned SHA-256 ledger",
                    real_path.display()
                ),
            ));
        }
        self.record(real_path, expected_sha256, after)?;
        Ok(SourceVerification::Hashed)
    }

    /// The SHA-256 (`sha256:<hex>`) of `real_path` (already canonical) when no ledger names it in
    /// advance: the recorded digest while the file's stamp is unchanged, otherwise a fresh hash
    /// that is then recorded. A file that changed while it was hashed is an error. On Windows
    /// such a mutable file is always hashed fresh and never recorded
    /// ([`STAMPS_IDENTIFY_MUTABLE_FILES`]).
    pub fn sha256(
        &self,
        real_path: &Path,
        check: &dyn Fn() -> io::Result<()>,
    ) -> io::Result<(String, SourceVerification)> {
        let before = SourceStampV1::of(real_path, StoreClockV1::SYSTEM)?;
        if !STAMPS_IDENTIFY_MUTABLE_FILES {
            let (digest, _) = Self::hash(real_path, &before, check)?;
            return Ok((digest, SourceVerification::Hashed));
        }
        if let Some(sha256) = self.recorded_sha256(real_path, &before) {
            return Ok((sha256, SourceVerification::Reused));
        }
        let (digest, after) = Self::hash(real_path, &before, check)?;
        self.record(real_path, &digest, after)?;
        Ok((digest, SourceVerification::Hashed))
    }

    /// Stamp `real_path` (already canonical) before a caller's own hash of it, so the verdict of
    /// that hash can be recorded afterwards with [`Self::record_hashed`].
    pub fn observe(&self, real_path: &Path) -> io::Result<ObservedSource> {
        Ok(ObservedSource {
            real_path: real_path.to_path_buf(),
            before: SourceStampV1::of(real_path, StoreClockV1::SYSTEM)?,
        })
    }

    /// Record that the file `observed` before a caller's hash holds `sha256` (`sha256:<hex>`),
    /// only when its stamp is unchanged since that observation. Returns whether it was recorded.
    pub fn record_hashed(&self, observed: ObservedSource, sha256: &str) -> io::Result<bool> {
        let after = SourceStampV1::of(&observed.real_path, StoreClockV1::SYSTEM)?;
        if !observed.before.identifies_same_entry(&after) {
            return Ok(false);
        }
        self.record(&observed.real_path, sha256, after)?;
        Ok(true)
    }

    fn tree_record_path(&self, real_root: &Path) -> PathBuf {
        let digest = Sha256::digest(real_root.as_os_str().as_encoded_bytes());
        self.root.join(format!("tree-{digest:x}.json"))
    }

    /// A whole-tree content fingerprint of `root`, computed by `fingerprint` at most once per host
    /// while every file `list_files` names keeps its stamp. The record is keyed by the root's real
    /// path and a digest of every listed entry's relative path, link target and stamp (size, mtime
    /// and, on Unix, device, inode and ctime), so a file added, removed, renamed or changed
    /// anywhere re-runs `fingerprint`. A racy stamp, or a tree that changed while it was
    /// fingerprinted, is never recorded. On Windows a produced tree is mutable local content and
    /// is always fingerprinted fresh ([`STAMPS_IDENTIFY_MUTABLE_FILES`]).
    pub fn tree_fingerprint<E: From<io::Error>>(
        &self,
        root: &Path,
        list_files: impl Fn() -> Result<Vec<PathBuf>, E>,
        fingerprint: impl FnOnce() -> Result<String, E>,
    ) -> Result<(String, SourceVerification), E> {
        if !STAMPS_IDENTIFY_MUTABLE_FILES {
            return Ok((fingerprint()?, SourceVerification::Hashed));
        }
        let real_root = std::fs::canonicalize(root)?;
        let record_path = self.tree_record_path(&real_root);
        let (before, _) = tree_stamp(root, &list_files()?)?;
        if let Some(record) = std::fs::read(&record_path)
            .ok()
            .and_then(|body| serde_json::from_slice::<VerifiedTreeRecordV1>(&body).ok())
        {
            if record.schema_version == RECORD_SCHEMA_VERSION
                && record.real_root == real_root
                && record.stamp_sha256 == before
            {
                return Ok((record.fingerprint, SourceVerification::Reused));
            }
        }
        let fingerprint = fingerprint()?;
        let (after, racy) = tree_stamp(root, &list_files()?)?;
        if after == before && !racy {
            let record = VerifiedTreeRecordV1 {
                schema_version: RECORD_SCHEMA_VERSION,
                real_root,
                stamp_sha256: after,
                fingerprint: fingerprint.clone(),
            };
            std::fs::create_dir_all(&self.root)?;
            let body = serde_json::to_vec(&record).map_err(io::Error::other)?;
            crate::fsutil::write_atomic(&record_path, &body)?;
        }
        Ok((fingerprint, SourceVerification::Hashed))
    }
}

/// A file's stamp taken before a caller hashes it ([`VerifiedSourceCache::observe`]).
#[derive(Debug)]
pub struct ObservedSource {
    real_path: PathBuf,
    before: SourceStampV1,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct VerifiedTreeRecordV1 {
    schema_version: u32,
    real_root: PathBuf,
    stamp_sha256: String,
    fingerprint: String,
}

/// The digest of every listed entry's relative path, link target and followed stamp, and whether
/// any of those stamps is racy.
fn tree_stamp(root: &Path, files: &[PathBuf]) -> io::Result<(String, bool)> {
    let mut entries = files
        .iter()
        .map(|path| {
            let relative = path.strip_prefix(root).unwrap_or(path).to_path_buf();
            let link = std::fs::symlink_metadata(path)?
                .file_type()
                .is_symlink()
                .then(|| std::fs::read_link(path))
                .transpose()?;
            let mut stamp = SourceStampV1::of(path, StoreClockV1::SYSTEM)?;
            let racy = stamp.is_racy();
            stamp.observed_nanos = None;
            Ok((relative, link, stamp, racy))
        })
        .collect::<io::Result<Vec<_>>>()?;
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut digest = Sha256::new();
    let mut racy = false;
    for (relative, link, stamp, entry_racy) in entries {
        racy |= entry_racy;
        digest.update(relative.as_os_str().as_encoded_bytes());
        digest.update([0]);
        if let Some(link) = link {
            digest.update([1]);
            digest.update(link.as_os_str().as_encoded_bytes());
        }
        digest.update([0]);
        digest.update(serde_json::to_vec(&stamp).map_err(io::Error::other)?);
        digest.update([0xff]);
    }
    Ok((format!("{:x}", digest.finalize()), racy))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(bytes: &[u8]) -> String {
        format!("sha256:{}", crate::hashing::sha256_bytes(bytes))
    }

    /// Older than the stamp's racy window, so the recorded verdict may stand in for a hash.
    fn settle(path: &Path) {
        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(10))
            .unwrap();
    }

    #[test]
    fn an_unchanged_file_is_hashed_once_and_a_changed_one_again() {
        let temp = tempfile::tempdir().unwrap();
        let cache = VerifiedSourceCache::new(temp.path().join("verified"));
        let path = temp.path().join("blob");
        std::fs::write(&path, b"weights").unwrap();
        settle(&path);
        let path = std::fs::canonicalize(&path).unwrap();
        let ok = || Ok(());
        assert_eq!(
            cache.verify(&path, &sha(b"weights"), &ok).unwrap(),
            SourceVerification::Hashed
        );
        assert_eq!(
            cache.verify(&path, &sha(b"weights"), &ok).unwrap(),
            SourceVerification::Reused
        );
        // A same-size rewrite changes the stamp: re-hashed, and refused.
        std::fs::write(&path, b"WEIGHTS").unwrap();
        settle(&path);
        let error = cache.verify(&path, &sha(b"weights"), &ok).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        // The mismatch is not recorded: restoring the bytes is verified by a fresh hash.
        std::fs::write(&path, b"weights").unwrap();
        settle(&path);
        assert_eq!(
            cache.verify(&path, &sha(b"weights"), &ok).unwrap(),
            SourceVerification::Hashed
        );
        // A different expected digest never reuses another ledger's verdict.
        assert!(cache.verify(&path, &sha(b"other"), &ok).is_err());
    }

    #[test]
    fn a_racy_stamp_is_not_trusted() {
        let temp = tempfile::tempdir().unwrap();
        let cache = VerifiedSourceCache::new(temp.path().join("verified"));
        let path = temp.path().join("blob");
        std::fs::write(&path, b"weights").unwrap();
        let path = std::fs::canonicalize(&path).unwrap();
        let ok = || Ok(());
        // Freshly written: the recorded observation is within the racy window of the mtime.
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(60))
            .unwrap();
        assert_eq!(
            cache.verify(&path, &sha(b"weights"), &ok).unwrap(),
            SourceVerification::Hashed
        );
        assert_eq!(
            cache.verify(&path, &sha(b"weights"), &ok).unwrap(),
            SourceVerification::Hashed
        );
    }

    /// Windows: a same-length rewrite that restores the last-write time is invisible to the stamp,
    /// so a mutable file's digest and a produced tree's fingerprint are never reused there.
    #[cfg(windows)]
    #[test]
    fn windows_hashes_mutable_files_fresh_after_a_timestamp_preserving_rewrite() {
        let temp = tempfile::tempdir().unwrap();
        let cache = VerifiedSourceCache::for_data_dir(temp.path());
        let path = temp.path().join("blob");
        std::fs::write(&path, b"weights").unwrap();
        settle(&path);
        let path = std::fs::canonicalize(&path).unwrap();
        let ok = || Ok(());
        assert_eq!(
            cache.sha256(&path, &ok).unwrap(),
            (sha(b"weights"), SourceVerification::Hashed)
        );
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::fs::write(&path, b"WEIGHTS").unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(mtime)
            .unwrap();
        assert_eq!(
            cache.sha256(&path, &ok).unwrap(),
            (sha(b"WEIGHTS"), SourceVerification::Hashed)
        );
        let root = temp.path().join("tree");
        std::fs::create_dir_all(&root).unwrap();
        for _ in 0..2 {
            let (_, verification) = cache
                .tree_fingerprint(&root, || Ok::<_, io::Error>(Vec::new()), || Ok("fp".into()))
                .unwrap();
            assert_eq!(verification, SourceVerification::Hashed);
        }
    }

    #[cfg(not(windows))]
    #[test]
    fn an_unledgered_digest_is_recorded_once_and_shared_with_verify() {
        let temp = tempfile::tempdir().unwrap();
        let cache = VerifiedSourceCache::for_data_dir(temp.path());
        let path = temp.path().join("blob");
        std::fs::write(&path, b"weights").unwrap();
        settle(&path);
        let path = std::fs::canonicalize(&path).unwrap();
        let ok = || Ok(());
        assert_eq!(
            cache.sha256(&path, &ok).unwrap(),
            (sha(b"weights"), SourceVerification::Hashed)
        );
        assert_eq!(
            cache.sha256(&path, &ok).unwrap(),
            (sha(b"weights"), SourceVerification::Reused)
        );
        // A digest recorded without a ledger serves a later ledger check of the same bytes.
        assert_eq!(
            cache.verify(&path, &sha(b"weights"), &ok).unwrap(),
            SourceVerification::Reused
        );
        // A same-size rewrite that restores the old mtime still moves the stamp (ctime). Windows
        // stamps carry no ctime, so this case is Unix-only.
        if cfg!(not(unix)) {
            return;
        }
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::fs::write(&path, b"WEIGHTS").unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(mtime)
            .unwrap();
        assert_eq!(
            cache.sha256(&path, &ok).unwrap(),
            (sha(b"WEIGHTS"), SourceVerification::Hashed)
        );
    }

    #[test]
    fn a_hash_taken_elsewhere_is_recorded_only_while_the_stamp_is_unchanged() {
        let temp = tempfile::tempdir().unwrap();
        let cache = VerifiedSourceCache::for_data_dir(temp.path());
        let path = temp.path().join("blob");
        std::fs::write(&path, b"weights").unwrap();
        settle(&path);
        let path = std::fs::canonicalize(&path).unwrap();
        let ok = || Ok(());
        let observed = cache.observe(&path).unwrap();
        assert!(cache.record_hashed(observed, &sha(b"weights")).unwrap());
        assert_eq!(
            cache.verify(&path, &sha(b"weights"), &ok).unwrap(),
            SourceVerification::Reused
        );
        // Changed between the observation and the record: nothing is recorded.
        let observed = cache.observe(&path).unwrap();
        std::fs::write(&path, b"WEIGHTS").unwrap();
        settle(&path);
        assert!(!cache.record_hashed(observed, &sha(b"weights")).unwrap());
        assert_eq!(
            cache
                .verify(&path, &sha(b"weights"), &ok)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }

    /// Lists every regular file under `root`, like a produced-artifact walk.
    #[cfg(not(windows))]
    fn list(root: &Path) -> io::Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(root)? {
            files.push(entry?.path());
        }
        Ok(files)
    }

    #[cfg(not(windows))]
    #[test]
    fn a_tree_is_fingerprinted_once_until_any_file_is_added_removed_or_changed() {
        let temp = tempfile::tempdir().unwrap();
        let cache = VerifiedSourceCache::new(temp.path().join("verified"));
        let root = temp.path().join("model");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.safetensors"), b"weights").unwrap();
        settle(&root.join("a.safetensors"));
        let calls = std::cell::Cell::new(0);
        let run = || {
            cache.tree_fingerprint(
                &root,
                || list(&root),
                || {
                    calls.set(calls.get() + 1);
                    Ok::<_, io::Error>(format!("fp-{}", calls.get()))
                },
            )
        };
        assert_eq!(run().unwrap(), ("fp-1".into(), SourceVerification::Hashed));
        assert_eq!(run().unwrap(), ("fp-1".into(), SourceVerification::Reused));
        assert_eq!(calls.get(), 1, "an unchanged tree is not read again");

        std::fs::write(root.join("b.json"), b"{}").unwrap();
        settle(&root.join("b.json"));
        assert_eq!(run().unwrap().1, SourceVerification::Hashed, "added");
        assert_eq!(run().unwrap().1, SourceVerification::Reused);

        std::fs::remove_file(root.join("b.json")).unwrap();
        assert_eq!(run().unwrap().1, SourceVerification::Hashed, "removed");
        assert_eq!(run().unwrap().1, SourceVerification::Reused);

        if cfg!(unix) {
            // Same size, the old mtime restored: ctime still moves.
            let path = root.join("a.safetensors");
            let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
            std::fs::write(&path, b"WEIGHTS").unwrap();
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(mtime)
                .unwrap();
            assert_eq!(run().unwrap().1, SourceVerification::Hashed, "substituted");
        }

        // A freshly written file is racy: fingerprinted, never recorded.
        std::fs::write(root.join("c.json"), b"{}").unwrap();
        assert_eq!(run().unwrap().1, SourceVerification::Hashed);
        assert_eq!(run().unwrap().1, SourceVerification::Hashed);
    }
}
