//! Hash-once-per-host verification of immutable source files read in place.
//!
//! A pinned Hugging Face snapshot is content-addressed: its file bytes cannot change while their
//! filesystem identity is unchanged. A job that binds such a file verifies its SHA-256 against the
//! frozen ledger once, records the verdict keyed by the file's real path and the same
//! filesystem stamp the checkpoint-plan store trusts ([`SourceStampV1`]: size, mtime and, on Unix,
//! device, inode and ctime), and later attempts reuse that verdict while the stamp is unchanged.
//! Any stamp change, a racy stamp (modified within one timestamp tick of its observation), a
//! different expected digest or an unreadable record re-hashes the file. A mismatch is never
//! recorded.
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

impl VerifiedSourceCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn record_path(&self, real_path: &Path) -> PathBuf {
        let digest = Sha256::digest(real_path.as_os_str().as_encoded_bytes());
        self.root.join(format!("{digest:x}.json"))
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
        let record_path = self.record_path(real_path);
        if let Some(record) = std::fs::read(&record_path)
            .ok()
            .and_then(|body| serde_json::from_slice::<VerifiedSourceRecordV1>(&body).ok())
        {
            if record.schema_version == RECORD_SCHEMA_VERSION
                && record.real_path == real_path
                && record.sha256 == expected_sha256
                && !record.stamp.is_racy()
                && record.stamp.identifies_same_entry(&before)
            {
                return Ok(SourceVerification::Reused);
            }
        }
        let digest = source_file_sha256_with_check(real_path, check)?;
        if format!("sha256:{digest}") != expected_sha256 {
            let _ = std::fs::remove_file(&record_path);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} does not match its pinned SHA-256 ledger",
                    real_path.display()
                ),
            ));
        }
        let after = SourceStampV1::of(real_path, StoreClockV1::SYSTEM)?;
        if !before.identifies_same_entry(&after) {
            return Err(io::Error::other(format!(
                "{} changed while it was verified",
                real_path.display()
            )));
        }
        let record = VerifiedSourceRecordV1 {
            schema_version: RECORD_SCHEMA_VERSION,
            real_path: real_path.to_path_buf(),
            sha256: expected_sha256.to_owned(),
            stamp: after,
        };
        std::fs::create_dir_all(&self.root)?;
        let body = serde_json::to_vec(&record).map_err(io::Error::other)?;
        crate::fsutil::write_atomic(&record_path, &body)?;
        Ok(SourceVerification::Hashed)
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
}
