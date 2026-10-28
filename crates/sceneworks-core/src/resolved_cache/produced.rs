//! Hash-while-write attestations for produced derivative files (sc-26691).
//!
//! A derived bundle's bytes do not exist until a producer writes them, and the producer already
//! streams every one of those bytes through memory on the way to disk. Before this module, the
//! store then re-read and re-hashed the finished file once per stage that wanted its digest —
//! `measure_produced_file`, the publication validation, the completion validation and the first
//! `acquire_complete` — four full reads of a 34 GB MiniMax-H3 derivative straight after the copy
//! that wrote it, on top of the native load seal.
//!
//! [`ProducedFile`] hashes the bytes as they are written instead. When the producer closes it, the
//! file's identity is captured as a [`SourceStampV1`] — the same stamp checkpoint source-drift
//! detection uses (size, mtime, inode/device/ctime on Unix, creation/last-write time on Windows) —
//! and recorded beside the digest as a [`ProducedFileAttestation`]. Every later stage redeems the
//! attestation only while a fresh stamp of the same path still identifies the same entry; any
//! difference, or anything that makes the stamp untrustworthy, falls back to the real re-hash
//! exactly as before. Nothing is skipped: either the bytes were hashed as they were written and
//! provably not touched since, or they are hashed again.
//!
//! **Why the mtime is backdated.** A stamp taken immediately after the last write is "racy" in
//! git's sense ([`SourceStampV1::is_racy`]): on a filesystem whose timestamp tick is coarser than
//! the gap between our stamp and a later same-size rewrite, that rewrite would leave every field of
//! the stamp unchanged. So on close the writer sets the file's mtime to [`BACKDATE`] before the
//! moment the file was opened. Any write after that — through any handle — stamps the file with
//! the then-current time, which is necessarily newer than the backdated one by more than any
//! filesystem's granularity, so it can never reproduce the attested stamp. A stamp that still looks
//! racy (a filesystem that refused the backdate, a clock that moved) is simply never redeemed.

use crate::checkpoint_plan_store::{SourceStampV1, StoreClockV1};
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// How far before the file was opened its mtime is set when the writer closes it. Comfortably
/// coarser than every timestamp granularity a model library can sit on (FAT's 2 s included).
const BACKDATE: Duration = Duration::from_secs(10);

/// The digest a producer's write stream yielded for one file, and the identity that file had the
/// moment the stream closed.
#[derive(Clone, Debug)]
pub struct ProducedFileAttestation {
    size_bytes: u64,
    sha256: String,
    stamp: SourceStampV1,
    /// The same digest bound to gen-core's own target fingerprint, so the native load seal can
    /// redeem it instead of re-reading the file (`ArtifactSeal::pin_attested`). `None` when the
    /// fingerprint could not be captured; the seal then reads the file as before.
    seal: Option<gen_core::ArtifactDigestAttestation>,
}

impl ProducedFileAttestation {
    /// The attested `(size, "sha256:<hex>")`, but only while `path` is still the entry the write
    /// stream produced. `expected` (a digest the caller already holds) must also agree.
    pub(crate) fn redeem(&self, path: &Path, expected: Option<&str>) -> Option<(u64, String)> {
        let now = SourceStampV1::of(path, StoreClockV1::SYSTEM).ok()?;
        let unchanged = self.stamp.identifies_same_entry(&now)
            && now.size_bytes == self.size_bytes
            && !self.stamp.is_racy()
            && !now.is_racy()
            && expected.is_none_or(|expected| expected == self.sha256);
        unchanged.then(|| (self.size_bytes, self.sha256.clone()))
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }

    pub(crate) fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// The gen-core attestation for the native load seal, if one was captured.
    pub fn seal_attestation(&self) -> Option<&gen_core::ArtifactDigestAttestation> {
        self.seal.as_ref()
    }
}

/// The attestations one production recorded, keyed by bundle-relative path. Cheap to clone; every
/// clone shares the same record.
#[derive(Clone, Debug, Default)]
pub struct ProducedAttestations(Arc<Mutex<BTreeMap<PathBuf, ProducedFileAttestation>>>);

impl ProducedAttestations {
    /// Create `relative` under `root` (`create_new`, never an existing file) as a hashing writer
    /// whose attestation lands here when it closes. On Windows the handle denies other writers for
    /// as long as it is open, so the stream is the only thing writing the file it hashes.
    pub fn create(&self, root: &Path, relative: &Path) -> std::io::Result<ProducedFile> {
        let path = root.join(relative);
        let opened_at = SystemTime::now();
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt as _;
            // FILE_SHARE_READ only: no concurrent writer or deleter while the stream is open.
            options.share_mode(0x0000_0001);
        }
        let file = options.open(&path)?;
        // A re-created path must never inherit an attestation from an earlier stream.
        self.0.lock().remove(relative);
        Ok(ProducedFile {
            file: Some(file),
            hasher: Sha256::new(),
            written: 0,
            failed: false,
            path,
            relative: relative.to_path_buf(),
            opened_at,
            sink: self.clone(),
        })
    }

    pub fn get(&self, relative: &Path) -> Option<ProducedFileAttestation> {
        self.0.lock().get(relative).cloned()
    }

    pub fn len(&self) -> usize {
        self.0.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A produced output file that hashes every byte it writes. Implements only [`Write`] — no seek —
/// so the digest is of the file exactly as written, front to back.
#[derive(Debug)]
pub struct ProducedFile {
    file: Option<File>,
    hasher: Sha256,
    written: u64,
    failed: bool,
    path: PathBuf,
    relative: PathBuf,
    opened_at: SystemTime,
    sink: ProducedAttestations,
}

impl Write for ProducedFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| std::io::Error::other("produced file is already closed"))?;
        match file.write(buf) {
            Ok(written) => {
                self.hasher.update(&buf[..written]);
                self.written += written as u64;
                Ok(written)
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self.file.as_mut() {
            Some(file) => file.flush().inspect_err(|_| self.failed = true),
            None => Ok(()),
        }
    }
}

impl ProducedFile {
    /// Close the stream and record its attestation. Dropping the writer does the same; this form
    /// only surfaces the (never authorizing) reason an attestation could not be recorded.
    pub fn finish(mut self) -> std::io::Result<()> {
        self.close()
    }

    fn close(&mut self) -> std::io::Result<()> {
        let Some(mut file) = self.file.take() else {
            return Ok(());
        };
        let hasher = std::mem::take(&mut self.hasher);
        if self.failed {
            return Err(std::io::Error::other(
                "produced file had a failed write; no attestation recorded",
            ));
        }
        file.flush()?;
        let backdated = self
            .opened_at
            .checked_sub(BACKDATE)
            .ok_or_else(|| std::io::Error::other("system clock is before the backdate window"))?;
        file.set_modified(backdated)?;
        drop(file);
        let stamp = SourceStampV1::of(&self.path, StoreClockV1::SYSTEM)?;
        if stamp.size_bytes != self.written || stamp.is_racy() {
            return Err(std::io::Error::other(
                "produced file's closing stamp does not describe the hashed stream",
            ));
        }
        let digest: [u8; 32] = hasher.finalize().into();
        let seal = gen_core::ArtifactDigestAttestation::capture(&self.path, digest).ok();
        #[cfg(test)]
        count_stream_hash();
        self.sink.0.lock().insert(
            self.relative.clone(),
            ProducedFileAttestation {
                size_bytes: self.written,
                sha256: format!(
                    "sha256:{}",
                    digest
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>()
                ),
                stamp,
                seal,
            },
        );
        Ok(())
    }
}

impl Drop for ProducedFile {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::debug!(path = %self.path.display(), %error,
                "produced file closed without an attestation; publication will re-hash it");
        }
    }
}

// Counts files whose digest was taken from the write stream, so a test can state the whole
// prepare flow as "one pass per produced file" rather than as a duration.
#[cfg(test)]
thread_local! {
    static STREAM_HASH_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn count_stream_hash() {
    STREAM_HASH_COUNT.with(|count| count.set(count.get().saturating_add(1)));
}

/// Runs `body` and reports how many produced files it hashed while writing.
#[cfg(test)]
pub(crate) fn observe_stream_hashes<T>(body: impl FnOnce() -> T) -> (T, u64) {
    STREAM_HASH_COUNT.with(|count| count.set(0));
    let value = body();
    (value, STREAM_HASH_COUNT.with(std::cell::Cell::get))
}
