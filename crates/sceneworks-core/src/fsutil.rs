//! Durable file publication shared by stores, API, workers and desktop.
//!
//! This owns byte-file publication only. Callers retain confinement checks, locks and
//! read/modify/write transactions; staged media and directory trees have separate contracts.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Publication policy. Temporary files use exclusive creation and random names.
#[derive(Clone, Copy, Debug)]
pub struct AtomicWriteOptions<'a> {
    /// Replace an existing destination atomically. False uses an atomic no-clobber link.
    pub replace: bool,
    /// On Unix, create the temporary file with mode 0600 before writing any bytes.
    /// Other platforms retain their normal inherited ACL policy.
    pub secret: bool,
    /// Optional caller-owned staging directory on the destination filesystem.
    /// The caller must validate its confinement before invoking this API.
    pub temp_directory: Option<&'a Path>,
}

impl Default for AtomicWriteOptions<'_> {
    fn default() -> Self {
        Self {
            replace: true,
            secret: false,
            temp_directory: None,
        }
    }
}

/// Write all bytes, sync the file, atomically replace the destination and sync its
/// parent. A failure before publication preserves the old destination and cleans up
/// the temporary file. A directory-sync failure after publication is reported even
/// though the new file is already visible. Windows has no POSIX directory fsync.
pub fn write_atomic(path: &Path, contents: &[u8]) -> io::Result<()> {
    write_atomic_with(path, contents, AtomicWriteOptions::default(), || Ok(()))
}

/// As [`write_atomic`], with publication policy and a last-moment authorization check.
/// The hook runs after temporary-file I/O, immediately before rename/link. Its error
/// type is preserved so a caller can retain typed claim-loss/cancellation errors.
/// Callers must keep any publication lock held across this call.
pub fn write_atomic_with<E: From<io::Error>>(
    path: &Path,
    contents: &[u8],
    options: AtomicWriteOptions<'_>,
    before_publish: impl FnOnce() -> Result<(), E>,
) -> Result<(), E> {
    let parent = parent_directory(path);
    fs::create_dir_all(parent)?;
    let temporary_directory = options.temp_directory.unwrap_or(parent);
    fs::create_dir_all(temporary_directory)?;
    let (temporary, mut file) = create_temporary(temporary_directory, options.secret)?;
    file.write_all(contents)?;
    file.sync_all()?;
    drop(file); // Windows cannot rename an open staging handle.
    sync_parent_directory(&temporary.0)?;
    before_publish()?;
    if options.replace {
        fs::rename(&temporary.0, path)?;
    } else {
        // exists()+rename races on Unix and can silently overwrite another publisher.
        // Linking publishes the complete inode only if the name is still absent.
        fs::hard_link(&temporary.0, path)?;
        fs::remove_file(&temporary.0)?;
    }
    sync_parent_directory(path)?;
    if temporary_directory != parent {
        sync_parent_directory(&temporary.0)?;
    }
    Ok(())
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

struct TemporaryFile(PathBuf);
impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn create_temporary(directory: &Path, secret: bool) -> io::Result<(TemporaryFile, fs::File)> {
    // Retry exclusive-create collisions without ever opening/truncating a stale temp.
    for _ in 0..16 {
        let mut token = [0_u8; 16];
        getrandom::fill(&mut token).map_err(io::Error::other)?;
        let suffix: String = token.iter().map(|byte| format!("{byte:02x}")).collect();
        let path = directory.join(format!(".sceneworks-{suffix}.tmp"));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        if secret {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        #[cfg(not(unix))]
        let _ = secret;
        match options.open(&path) {
            Ok(file) => return Ok((TemporaryFile(path), file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "cannot allocate unique atomic-write temporary file",
    ))
}

/// Persist a changed directory entry. Errors propagate on Unix; Windows has no
/// POSIX directory fsync, so file content sync remains the durability boundary there.
pub fn sync_parent_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    fs::File::open(parent_directory(path))?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn names(directory: &Path) -> Vec<PathBuf> {
        fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect()
    }

    #[test]
    fn replacement_and_failed_publication_leave_no_temporary_files() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("nested/record.json");
        write_atomic(&path, b"old").unwrap();
        let error = write_atomic_with(&path, b"new", Default::default(), || {
            assert_eq!(fs::read(&path).unwrap(), b"old");
            assert_eq!(names(path.parent().unwrap()).len(), 2);
            Err::<(), _>(io::Error::other("injected claim loss"))
        })
        .unwrap_err();
        assert!(error.to_string().contains("injected claim loss"));
        assert_eq!(fs::read(&path).unwrap(), b"old");
        assert_eq!(names(path.parent().unwrap()), vec![path.clone()]);
        write_atomic(&path, b"new").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(write_atomic(&path, b"cannot replace directory").is_err());
        assert_eq!(names(path.parent().unwrap()), vec![path]);
    }

    #[test]
    fn panic_in_hook_cleans_up_and_preserves_old_file() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("record");
        write_atomic(&path, b"old").unwrap();
        assert!(std::panic::catch_unwind(|| {
            let _: io::Result<()> =
                write_atomic_with(&path, b"new", Default::default(), || panic!("stop"));
        })
        .is_err());
        assert_eq!(fs::read(&path).unwrap(), b"old");
        assert_eq!(names(root.path()), vec![path]);
    }

    #[test]
    fn concurrent_writers_publish_complete_files_without_colliding() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("record");
        let barrier = Arc::new(Barrier::new(8));
        std::thread::scope(|scope| {
            for byte in 0_u8..8 {
                let path = &path;
                let barrier = barrier.clone();
                scope.spawn(move || {
                    let body = vec![byte; 32 * 1024];
                    write_atomic_with(path, &body, Default::default(), || {
                        barrier.wait();
                        Ok::<(), io::Error>(())
                    })
                    .unwrap();
                });
            }
        });
        let contents = fs::read(&path).unwrap();
        assert_eq!(contents.len(), 32 * 1024);
        assert!(contents.iter().all(|byte| *byte == contents[0]));
        assert_eq!(names(root.path()), vec![path]);
    }

    #[test]
    fn no_replace_has_exactly_one_concurrent_winner_and_cleans_staging() {
        let root = tempfile::tempdir().unwrap();
        let staging = root.path().join("staging");
        let path = root.path().join("record");
        let barrier = Arc::new(Barrier::new(8));
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for byte in 0_u8..8 {
                let path = &path;
                let staging = &staging;
                let barrier = barrier.clone();
                handles.push(scope.spawn(move || {
                    write_atomic_with(
                        path,
                        &[byte; 1024],
                        AtomicWriteOptions {
                            replace: false,
                            temp_directory: Some(staging),
                            ..Default::default()
                        },
                        || {
                            barrier.wait();
                            Ok::<(), io::Error>(())
                        },
                    )
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert!(results
            .iter()
            .filter_map(|result| result.as_ref().err())
            .all(|error| error.kind() == io::ErrorKind::AlreadyExists));
        let contents = fs::read(path).unwrap();
        assert_eq!(contents.len(), 1024);
        assert!(contents.iter().all(|byte| *byte == contents[0]));
        assert!(names(&staging).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn secrets_are_private_before_publication_and_replace_loose_files() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("credentials.json");
        fs::write(&path, b"old").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o666)).unwrap();
        write_atomic_with(
            &path,
            b"secret",
            AtomicWriteOptions {
                secret: true,
                ..Default::default()
            },
            || {
                let temporary = names(root.path())
                    .into_iter()
                    .find(|candidate| candidate != &path)
                    .unwrap();
                assert_eq!(
                    fs::metadata(temporary).unwrap().permissions().mode() & 0o777,
                    0o600
                );
                Ok::<(), io::Error>(())
            },
        )
        .unwrap();
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
