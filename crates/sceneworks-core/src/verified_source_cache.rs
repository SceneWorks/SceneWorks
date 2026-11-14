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
        let (digest, after) = Self::hash(real_path, &before, check)?;
        self.record(real_path, &digest, after)?;
        Ok((digest, SourceVerification::Hashed))
    }
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
}
