//! Hash-once-per-host verification of source files read in place.
//!
//! A job that binds a file verifies its SHA-256 (against a frozen ledger, or records the digest it
//! hashed to when no ledger names it in advance), records the verdict keyed by the file's real path
//! and the filesystem stamp the checkpoint-plan store trusts ([`SourceStampV1`]: size, mtime and,
//! on Unix, device, inode and ctime; on Windows, creation and last-write time, volume serial number,
//! file index and the NTFS change time), and later checks on the same host — API admission and the
//! worker share the store — reuse that verdict while the stamp is unchanged.
//!
//! Any stamp change, a racy stamp (the file was modified within one timestamp tick of the stamp
//! taken BEFORE it was hashed), a different expected digest or an unreadable record re-hashes the
//! file. A mismatch is never recorded. Recording is best-effort: a store that cannot be written is
//! logged once and the next check simply hashes again; it never refuses correct bytes.
//!
//! Where the guarantee holds: on APFS, ext4 and other native Unix filesystems, and on NTFS, a
//! same-size replacement that restores the modified time still moves the change key (`ctime`, or
//! the NTFS change time) and is re-hashed. On filesystems that synthesize inode numbers or `ctime`
//! (exFAT, FAT, some SMB mounts) such a replacement is NOT detected on Unix. On Windows a zero
//! change time marks those volumes; their files are never recorded and are hashed on every check.
//! A file whose modified time is in the future stays racy and is hashed on every check (logged once
//! per file at info level).
use crate::checkpoint_plan_store::{source_file_sha256_with_check, SourceStampV1, StoreClockV1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Windows records are at v2: their stamps gained the NTFS change time, volume serial number and
/// file index, so a v1 Windows record (which could not see a same-length rewrite that restores the
/// last-write time) is never trusted and its file is hashed once more. Unix records stay at v1.
const RECORD_SCHEMA_VERSION: u32 = if cfg!(windows) { 2 } else { 1 };

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
    /// The file was read and hashed (and its verdict recorded when the store allowed it).
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

/// Per-path read counts, so tests can prove that a check read no bytes no matter which thread or
/// blocking task ran it.
#[cfg(any(test, feature = "test-support"))]
pub mod read_counts {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    static COUNTS: Mutex<BTreeMap<PathBuf, usize>> = Mutex::new(BTreeMap::new());

    /// How many times this process has read `path` (a real file path, or a tree root) to hash it.
    pub fn of(path: &Path) -> usize {
        COUNTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(path)
            .copied()
            .unwrap_or(0)
    }

    pub(super) fn note(path: &Path) {
        *COUNTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(path.to_path_buf())
            .or_default() += 1;
    }
}

fn note_read(path: &Path) {
    #[cfg(any(test, feature = "test-support"))]
    read_counts::note(path);
    #[cfg(not(any(test, feature = "test-support")))]
    let _ = path;
}

/// True the first time `key` is seen in `seen` by this process.
fn first_time(seen: &'static OnceLock<Mutex<BTreeSet<PathBuf>>>, key: &Path) -> bool {
    seen.get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(key.to_path_buf())
}

static UNWRITABLE_RECORDS: OnceLock<Mutex<BTreeSet<PathBuf>>> = OnceLock::new();
static FUTURE_DATED: OnceLock<Mutex<BTreeSet<PathBuf>>> = OnceLock::new();

/// Whether `stamp` (taken before `path` was hashed) may later stand in for that hash.
fn trustworthy(path: &Path, stamp: &SourceStampV1) -> bool {
    if stamp.is_weak() {
        return false;
    }
    if stamp.is_racy() {
        if stamp.is_future_dated() && first_time(&FUTURE_DATED, path) {
            tracing::info!(
                path = %path.display(),
                "source file's modified time is in the future; it is hashed on every check until that time passes"
            );
        }
        return false;
    }
    true
}

/// Write `body` to `record_path`, best-effort: a failure is logged once per record and the next
/// check hashes again.
fn write_record(root: &Path, record_path: &Path, body: io::Result<Vec<u8>>) -> bool {
    let written = body.and_then(|body| {
        std::fs::create_dir_all(root)?;
        crate::fsutil::write_atomic(record_path, &body)
    });
    match written {
        Ok(()) => true,
        Err(error) => {
            if first_time(&UNWRITABLE_RECORDS, record_path) {
                tracing::warn!(
                    record = %record_path.display(),
                    %error,
                    "verified-source record could not be written; the file will be hashed again on its next check"
                );
            }
            false
        }
    }
}

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

    fn read_record(&self, real_path: &Path) -> Option<VerifiedSourceRecordV1> {
        let body = std::fs::read(self.record_path(real_path)).ok()?;
        let record = serde_json::from_slice::<VerifiedSourceRecordV1>(&body).ok()?;
        (record.schema_version == RECORD_SCHEMA_VERSION
            && record.real_path == real_path
            && !record.stamp.is_racy()
            && !record.stamp.is_weak())
        .then_some(record)
    }

    /// The digest recorded for `real_path` while its stamp is still `current`.
    fn recorded_sha256(&self, real_path: &Path, current: &SourceStampV1) -> Option<String> {
        self.read_record(real_path)
            .filter(|record| record.stamp.identifies_same_entry(current))
            .map(|record| record.sha256)
    }

    /// Hash `real_path` (`sha256:<hex>`), failing if its stamp moved from `before` meanwhile.
    fn hash(
        real_path: &Path,
        before: &SourceStampV1,
        check: &dyn Fn() -> io::Result<()>,
    ) -> io::Result<String> {
        note_read(real_path);
        let digest = source_file_sha256_with_check(real_path, check)?;
        let after = SourceStampV1::of(real_path, StoreClockV1::SYSTEM)?;
        if !before.identifies_same_entry(&after) {
            return Err(io::Error::other(format!(
                "{} changed while it was verified",
                real_path.display()
            )));
        }
        Ok(format!("sha256:{digest}"))
    }

    /// Record `sha256` for `real_path` under `stamp`, the stamp taken BEFORE the hash, so the
    /// racy-stamp rule is measured from the moment verification began. Best-effort.
    fn record(&self, real_path: &Path, sha256: &str, stamp: &SourceStampV1) -> bool {
        if !trustworthy(real_path, stamp) {
            return false;
        }
        let record = VerifiedSourceRecordV1 {
            schema_version: RECORD_SCHEMA_VERSION,
            real_path: real_path.to_path_buf(),
            sha256: sha256.to_owned(),
            stamp: stamp.clone(),
        };
        let record_path = self.record_path(real_path);
        write_record(
            &self.root,
            &record_path,
            serde_json::to_vec(&record).map_err(io::Error::other),
        )
    }

    fn mismatch(real_path: &Path) -> io::Error {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} does not match its pinned SHA-256 ledger",
                real_path.display()
            ),
        )
    }

    /// Verify that `real_path` (already canonical) holds exactly `expected_sha256`
    /// (`sha256:<hex>`). A digest mismatch is an `InvalidData` error — answered from the record
    /// without reading the file when a trusted record for the unchanged file names other bytes; a
    /// file that changed while it was hashed is an error too. `check` is polled between reads.
    pub fn verify(
        &self,
        real_path: &Path,
        expected_sha256: &str,
        check: &dyn Fn() -> io::Result<()>,
    ) -> io::Result<SourceVerification> {
        let before = SourceStampV1::of(real_path, StoreClockV1::SYSTEM)?;
        match self.recorded_sha256(real_path, &before) {
            Some(recorded) if recorded == expected_sha256 => return Ok(SourceVerification::Reused),
            Some(_) => return Err(Self::mismatch(real_path)),
            None => {}
        }
        let digest = Self::hash(real_path, &before, check)?;
        if digest != expected_sha256 {
            let _ = std::fs::remove_file(self.record_path(real_path));
            return Err(Self::mismatch(real_path));
        }
        self.record(real_path, expected_sha256, &before);
        Ok(SourceVerification::Hashed)
    }

    /// The SHA-256 (`sha256:<hex>`) of `real_path` (already canonical) when no ledger names it in
    /// advance: the recorded digest while the file's stamp is unchanged, otherwise a fresh hash
    /// that is then recorded. A file that changed while it was hashed is an error.
    pub fn sha256(
        &self,
        real_path: &Path,
        check: &dyn Fn() -> io::Result<()>,
    ) -> io::Result<(String, SourceVerification)> {
        let before = SourceStampV1::of(real_path, StoreClockV1::SYSTEM)?;
        if let Some(sha256) = self.recorded_sha256(real_path, &before) {
            return Ok((sha256, SourceVerification::Reused));
        }
        let digest = Self::hash(real_path, &before, check)?;
        self.record(real_path, &digest, &before);
        Ok((digest, SourceVerification::Hashed))
    }

    /// The digest recorded for `real_path` (already canonical) while its stamp is unchanged, so a
    /// caller that hashes on its own (an async, heartbeating hash) can skip that hash. `None` when
    /// there is no trustworthy record.
    pub fn recorded(&self, real_path: &Path) -> Option<String> {
        let current = SourceStampV1::of(real_path, StoreClockV1::SYSTEM).ok()?;
        self.recorded_sha256(real_path, &current)
    }

    /// Stamp `real_path` (already canonical) before a caller's own hash of it, so the verdict of
    /// that hash can be recorded afterwards with [`Self::record_hashed`].
    pub fn observe(&self, real_path: &Path) -> io::Result<ObservedSource> {
        Ok(ObservedSource {
            real_path: real_path.to_path_buf(),
            before: SourceStampV1::of(real_path, StoreClockV1::SYSTEM)?,
        })
    }

    /// Record that the file `observed` before a caller's hash of that same path holds `sha256`
    /// (`sha256:<hex>`), under the observed stamp, only when the stamp is unchanged since. Returns
    /// whether it was recorded (best-effort).
    pub fn record_hashed(&self, observed: ObservedSource, sha256: &str) -> io::Result<bool> {
        let after = SourceStampV1::of(&observed.real_path, StoreClockV1::SYSTEM)?;
        if !observed.before.identifies_same_entry(&after) {
            return Ok(false);
        }
        Ok(self.record(&observed.real_path, sha256, &observed.before))
    }

    fn tree_record_path(&self, real_root: &Path) -> PathBuf {
        let digest = Sha256::digest(real_root.as_os_str().as_encoded_bytes());
        self.root.join(format!("tree-{digest:x}.json"))
    }

    /// A whole-tree content fingerprint of `root`, computed by `fingerprint` at most once per host
    /// while the tree keeps its stamps. The record is keyed by the root's real path and a digest
    /// of every listed file's relative path, link target and stamp plus the stamp of the root and
    /// every directory above a listed file, so a file added, removed, renamed or changed anywhere
    /// re-runs `fingerprint`. A fingerprint is recorded only when the tree's stamps are the same
    /// after it as before, no file stamp is racy, and `accept` approves it (the caller's expected
    /// value), so a fingerprint taken while a transient file came and went is never reused.
    pub fn tree_fingerprint<E: From<io::Error>>(
        &self,
        root: &Path,
        list_files: impl Fn() -> Result<Vec<PathBuf>, E>,
        fingerprint: impl FnOnce() -> Result<String, E>,
        accept: impl FnOnce(&str) -> bool,
    ) -> Result<(String, SourceVerification), E> {
        let real_root = std::fs::canonicalize(root)?;
        let record_path = self.tree_record_path(&real_root);
        let (before, racy) = tree_stamp(root, &list_files()?)?;
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
        note_read(&real_root);
        let fingerprint = fingerprint()?;
        let (after, _) = tree_stamp(root, &list_files()?)?;
        if after == before && !racy && accept(&fingerprint) {
            let record = VerifiedTreeRecordV1 {
                schema_version: RECORD_SCHEMA_VERSION,
                real_root,
                stamp_sha256: before,
                fingerprint: fingerprint.clone(),
            };
            write_record(
                &self.root,
                &record_path,
                serde_json::to_vec(&record).map_err(io::Error::other),
            );
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

impl ObservedSource {
    /// The real path that was stamped; hash exactly this path.
    pub fn path(&self) -> &Path {
        &self.real_path
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct VerifiedTreeRecordV1 {
    schema_version: u32,
    real_root: PathBuf,
    stamp_sha256: String,
    fingerprint: String,
}

/// The digest of every listed file's relative path, link target and followed stamp, and of the
/// stamp of the root and of every directory between it and a listed file; and whether any FILE
/// stamp cannot stand in for a hash (racy or weak). Directory stamps detect an entry that came
/// and went; they are not held to the racy rule, since entries change them routinely.
fn tree_stamp(root: &Path, files: &[PathBuf]) -> io::Result<(String, bool)> {
    let mut untrusted = false;
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
            untrusted |= !trustworthy(path, &stamp);
            stamp.observed_nanos = None;
            Ok((relative, link, stamp))
        })
        .collect::<io::Result<Vec<_>>>()?;
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut directories = BTreeSet::from([PathBuf::new()]);
    for (relative, _, _) in &entries {
        let mut parent = relative.parent();
        while let Some(directory) = parent.filter(|directory| !directory.as_os_str().is_empty()) {
            directories.insert(directory.to_path_buf());
            parent = directory.parent();
        }
    }
    let mut digest = Sha256::new();
    for (relative, link, stamp) in entries {
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
    for directory in directories {
        let mut stamp = SourceStampV1::of(&root.join(&directory), StoreClockV1::SYSTEM)?;
        stamp.observed_nanos = None;
        digest.update([2]);
        digest.update(directory.as_os_str().as_encoded_bytes());
        digest.update([0]);
        digest.update(serde_json::to_vec(&stamp).map_err(io::Error::other)?);
        digest.update([0xff]);
    }
    Ok((format!("{:x}", digest.finalize()), untrusted))
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
        // A same-size rewrite that restores the old mtime still moves the stamp (ctime on Unix, the
        // NTFS change time on Windows): re-hashed by both `sha256` and `verify`.
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::fs::write(&path, b"WEIGHTS").unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(mtime)
            .unwrap();
        // The record still names the old bytes; a stamp blind to the rewrite would reuse it.
        assert_eq!(
            cache
                .verify(&path, &sha(b"weights"), &ok)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
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
    fn list(root: &Path) -> io::Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(root)? {
            files.push(entry?.path());
        }
        Ok(files)
    }

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
                |_| true,
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

        // Same size, the old mtime restored: ctime (Unix) or the NTFS change time (Windows) moves.
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

        // A freshly written file is racy: fingerprinted, never recorded.
        std::fs::write(root.join("c.json"), b"{}").unwrap();
        assert_eq!(run().unwrap().1, SourceVerification::Hashed);
        assert_eq!(run().unwrap().1, SourceVerification::Hashed);
    }

    /// A store that cannot be written never refuses correct bytes: the check succeeds, reports
    /// `Hashed`, and the next check hashes again.
    #[test]
    fn an_unwritable_store_still_verifies_correct_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let blocker = temp.path().join("not-a-directory");
        std::fs::write(&blocker, b"").unwrap();
        let cache = VerifiedSourceCache::new(blocker.join("verified"));
        let path = temp.path().join("blob");
        std::fs::write(&path, b"weights").unwrap();
        settle(&path);
        let path = std::fs::canonicalize(&path).unwrap();
        let ok = || Ok(());
        for _ in 0..2 {
            assert_eq!(
                cache.verify(&path, &sha(b"weights"), &ok).unwrap(),
                SourceVerification::Hashed
            );
            assert_eq!(
                cache.sha256(&path, &ok).unwrap(),
                (sha(b"weights"), SourceVerification::Hashed)
            );
        }
        let observed = cache.observe(&path).unwrap();
        assert!(!cache.record_hashed(observed, &sha(b"weights")).unwrap());
        let root = temp.path().join("tree");
        std::fs::create_dir_all(&root).unwrap();
        let (_, verification) = cache
            .tree_fingerprint(
                &root,
                || Ok::<_, io::Error>(Vec::new()),
                || Ok("fp".into()),
                |_| true,
            )
            .unwrap();
        assert_eq!(verification, SourceVerification::Hashed);
    }

    /// The racy rule is measured from the stamp taken BEFORE the hash: a file modified just before
    /// verification began is not recorded even when the hash itself takes longer than the tick.
    #[test]
    fn the_racy_rule_uses_the_stamp_taken_before_the_hash() {
        let temp = tempfile::tempdir().unwrap();
        let cache = VerifiedSourceCache::new(temp.path().join("verified"));
        let path = temp.path().join("blob");
        std::fs::write(&path, b"weights").unwrap();
        // Modified 40 ms "from now": racy at the stamp before the hash, settled by the time a
        // deliberately slow hash ends.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(std::time::SystemTime::now() + std::time::Duration::from_millis(40))
            .unwrap();
        let path = std::fs::canonicalize(&path).unwrap();
        let slow = || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            Ok(())
        };
        assert_eq!(
            cache.verify(&path, &sha(b"weights"), &slow).unwrap(),
            SourceVerification::Hashed
        );
        assert_eq!(
            cache.verify(&path, &sha(b"weights"), &|| Ok(())).unwrap(),
            SourceVerification::Hashed,
            "a verdict whose before-stamp was racy is never reused"
        );
    }

    /// A trusted record for the unchanged file that names other bytes answers a mismatch without
    /// reading the file, and the valid record survives.
    #[test]
    fn a_known_mismatch_is_answered_without_reading() {
        let temp = tempfile::tempdir().unwrap();
        let cache = VerifiedSourceCache::new(temp.path().join("verified"));
        let path = temp.path().join("blob");
        std::fs::write(&path, b"weights").unwrap();
        settle(&path);
        let path = std::fs::canonicalize(&path).unwrap();
        let ok = || Ok(());
        cache.sha256(&path, &ok).unwrap();
        let reads = read_counts::of(&path);
        let error = cache.verify(&path, &sha(b"other"), &ok).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(read_counts::of(&path), reads, "the mismatch read no bytes");
        assert_eq!(
            cache.verify(&path, &sha(b"weights"), &ok).unwrap(),
            SourceVerification::Reused,
            "the valid record was kept"
        );
    }

    /// A file that appears in the tree and vanishes again while it is fingerprinted, or a
    /// fingerprint the caller rejects, is never reused.
    #[test]
    fn a_transient_file_or_rejected_fingerprint_does_not_poison_the_tree_record() {
        let temp = tempfile::tempdir().unwrap();
        let cache = VerifiedSourceCache::new(temp.path().join("verified"));
        let root = temp.path().join("model");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.safetensors"), b"weights").unwrap();
        settle(&root.join("a.safetensors"));
        let transient = root.join(".provenance.tmp");
        let (_, first) = cache
            .tree_fingerprint(
                &root,
                || list(&root),
                || {
                    std::fs::write(&transient, b"x").unwrap();
                    std::fs::remove_file(&transient).unwrap();
                    Ok::<_, io::Error>("poisoned".to_owned())
                },
                |_| true,
            )
            .unwrap();
        assert_eq!(first, SourceVerification::Hashed);
        let run = |fingerprint: &'static str, accept: bool| {
            cache
                .tree_fingerprint(
                    &root,
                    || list(&root),
                    || Ok::<_, io::Error>(fingerprint.to_owned()),
                    |_| accept,
                )
                .unwrap()
        };
        assert_eq!(
            run("rejected", false),
            ("rejected".to_owned(), SourceVerification::Hashed),
            "the transient fingerprint was not reused"
        );
        assert_eq!(
            run("good", true),
            ("good".to_owned(), SourceVerification::Hashed),
            "the rejected fingerprint was not recorded"
        );
        assert_eq!(
            run("unused", true),
            ("good".to_owned(), SourceVerification::Reused)
        );
    }
}
