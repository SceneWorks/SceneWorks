//! The guard every `fs2` advisory lock holder in this workspace releases through, so no holder
//! releases by `close(2)` alone.
//!
//! (`catalog_store::CatalogProcessingLease` keeps its own equivalent `Drop`, written first when
//! sc-22738 was diagnosed there; it holds the same `LOCK_UN` invariant and its own regression
//! test.)
//!
//! `flock(2)` locks belong to the OPEN FILE DESCRIPTION, not to the descriptor and not to the
//! process. `fork(2)` hands a child a reference to that same description, so closing the owner's
//! descriptor only drops ONE reference: while any concurrently forked child still holds the
//! inherited one — the window between `fork` and `exec` for every `Command` the process spawns
//! (ffmpeg, sips, the worker binaries) — the lock keeps reading as HELD to everyone else. Rust
//! opens with `O_CLOEXEC`, so the child sheds it at `exec`, but on a loaded runner that window is
//! long enough to straddle an unrelated caller's release.
//!
//! sc-22738 measured that on the catalog processing lease, where it surfaced as a user-visible 409
//! (`catalog_processing_conflict`) from a lease its owner had already dropped. Every other holder
//! in the workspace released the same close-only way; on blocking or retrying call sites the cost
//! is a delay rather than a refusal, but it is the same defect.
//!
//! [`FileLock`] releases with `LOCK_UN`, which acts on the open file description itself and so
//! takes effect the moment the owner is done, no matter who else still references it.

use std::fs::File;
use std::io;

use fs2::FileExt;

/// An acquired `fs2` advisory lock. Dropping it releases the lock explicitly.
#[derive(Debug)]
pub struct FileLock {
    file: File,
}

impl FileLock {
    /// Block until the exclusive lock is held.
    pub fn exclusive(file: File) -> io::Result<Self> {
        FileExt::lock_exclusive(&file)?;
        Ok(Self { file })
    }

    /// Block until the shared lock is held.
    pub fn shared(file: File) -> io::Result<Self> {
        FileExt::lock_shared(&file)?;
        Ok(Self { file })
    }

    /// Take the exclusive lock or fail; contention is reported as [`fs2::lock_contended_error`].
    pub fn try_exclusive(file: File) -> io::Result<Self> {
        FileExt::try_lock_exclusive(&file)?;
        Ok(Self { file })
    }

    /// Retry-loop variant of [`Self::try_exclusive`]: hands the handle back on failure so a
    /// spin-waiting caller (`fs2` has no timed blocking-lock API) can attempt again without
    /// reopening the lock file.
    #[allow(clippy::result_large_err)]
    pub fn try_exclusive_retryable(file: File) -> Result<Self, (File, io::Error)> {
        match FileExt::try_lock_exclusive(&file) {
            Ok(()) => Ok(Self { file }),
            Err(error) => Err((file, error)),
        }
    }

    /// Retry-loop variant of [`Self::try_shared`]; see [`Self::try_exclusive_retryable`].
    #[allow(clippy::result_large_err)]
    pub fn try_shared_retryable(file: File) -> Result<Self, (File, io::Error)> {
        match FileExt::try_lock_shared(&file) {
            Ok(()) => Ok(Self { file }),
            Err(error) => Err((file, error)),
        }
    }

    /// Take the shared lock or fail; contention is reported as [`fs2::lock_contended_error`].
    pub fn try_shared(file: File) -> io::Result<Self> {
        FileExt::try_lock_shared(&file)?;
        Ok(Self { file })
    }

    /// The locked handle, for holders that also read or write the lock file.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// A second descriptor over the SAME open file description — exactly what `fork(2)` hands a
    /// child, and therefore over the same `flock` lock. Lets a test inject the interleaving that
    /// makes a close-only release visible, instead of waiting for a real spawn to straddle it.
    pub fn inherited_descriptor(&self) -> io::Result<File> {
        self.file.duplicate()
    }
}

impl Drop for FileLock {
    /// Releases the lock EXPLICITLY rather than letting `close(2)` do it. See the module docs:
    /// `close(2)` only drops one reference to the open file description that owns the lock.
    fn drop(&mut self) {
        // Released through the same crate that took the lock, not through std's inherent
        // `File::unlock`, so acquisition and release cannot drift apart.
        let _ = FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock_file(path: &std::path::Path) -> File {
        std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .expect("lock file opens")
    }

    /// A released lock must be free IMMEDIATELY, even while a descriptor this process handed to a
    /// child still references the same open file description. Removing the `LOCK_UN` in
    /// [`FileLock::drop`] fails this: the inherited descriptor keeps the lock held.
    #[test]
    fn a_released_lock_is_free_even_while_an_inherited_descriptor_survives() {
        let directory = tempfile::tempdir().expect("temp directory");
        let path = directory.path().join("guard.lock");

        let guard = FileLock::try_exclusive(lock_file(&path)).expect("lock acquires");
        let inherited = guard.inherited_descriptor().expect("descriptor duplicates");
        drop(guard);

        let reacquired = FileLock::try_exclusive(lock_file(&path));
        assert!(
            reacquired.is_ok(),
            "a lock released by its owner must not stay held by an inherited descriptor: {:?}",
            reacquired.err()
        );
        drop(inherited);
    }

    /// The guard is still a real lock: a second acquisition is refused while it is alive.
    #[test]
    fn a_held_lock_still_refuses_a_second_acquisition() {
        let directory = tempfile::tempdir().expect("temp directory");
        let path = directory.path().join("guard.lock");

        let held = FileLock::try_exclusive(lock_file(&path)).expect("lock acquires");
        assert!(
            FileLock::try_exclusive(lock_file(&path)).is_err(),
            "a held exclusive lock must refuse a second acquisition"
        );
        drop(held);
    }
}
