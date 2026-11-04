//! Durable file publication shared by stores, API, workers and desktop.
//!
//! This owns byte-file publication only. Callers retain confinement checks, locks and
//! read/modify/write transactions; staged media and directory trees have separate contracts.

use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// Publication policy. Temporary files use exclusive creation and random names.
#[derive(Clone, Copy, Debug)]
pub struct AtomicWriteOptions<'a> {
    /// Replace an existing destination atomically. False uses platform-native atomic no-clobber publication.
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
/// though the new file is already visible. Unsupported directory fsync is best-effort;
/// Windows has no POSIX directory fsync.
pub fn write_atomic(path: &Path, contents: &[u8]) -> io::Result<()> {
    write_atomic_with(path, contents, AtomicWriteOptions::default(), || Ok(()))
}

/// As [`write_atomic`], with publication policy and a last-moment authorization check.
/// The hook runs after temporary-file I/O, immediately before publication. Its error
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
        publish_noclobber(&temporary.0, path)?;
    }
    sync_parent_directory(path)?;
    if temporary_directory != parent {
        sync_parent_directory(&temporary.0)?;
    }
    Ok(())
}

/// Probe the destination filesystem before a caller consumes a publication fence.
/// Probe files remain private and are removed. False means neither exclusive rename
/// nor hard-link publication is supported; [`write_atomic_with`] still fails closed.
pub fn atomic_create_supported(directory: &Path) -> io::Result<bool> {
    fs::create_dir_all(directory)?;
    let (source, file) = create_temporary(directory, false)?;
    drop(file);
    let (destination, file) = create_temporary(directory, false)?;
    drop(file);
    fs::remove_file(&destination.0)?;
    match publish_noclobber(&source.0, &destination.0) {
        Ok(()) => Ok(true),
        Err(error) if unsupported_operation(&error) => Ok(false),
        Err(error) => Err(error),
    }
}

fn publish_noclobber(source: &Path, destination: &Path) -> io::Result<()> {
    // tempfile uses exclusive rename on Apple/Linux and MoveFileExW without
    // replacement/copy on Windows. Its older-Unix fallback is link+unlink.
    if let Err(mut error) =
        tempfile::TempPath::try_from_path(source.to_path_buf())?.persist_noclobber(destination)
    {
        // The caller's guard owns cleanup, including when native rename is not
        // supported and we try the other strictly no-clobber primitive.
        error.path.disable_cleanup(true);
        if !unsupported_operation(&error.error) {
            return Err(error.error);
        }
        fs::hard_link(source, destination)?;
    }
    // The dependency's link fallback ignores unlink failure. Complete cleanup
    // before syncing directory entries and reporting success.
    match fs::remove_file(source) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Exclusively create a recoverable, UNPUBLISHED non-secret file. Unlike
/// [`write_atomic`], the pathname exists while bytes are being written. The caller
/// must own a product-level lease and keep every reader behind its journal/pointer
/// publication gate. Never use this for an already referenced/public pathname.
/// A crash can leave partial bytes; the caller's recovery must reject/reclaim them.
pub fn write_new_unpublished_with(
    path: &Path,
    contents: &[u8],
    before_publication_gate: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    write_new_unpublished_using(
        path,
        |file| file.write_all(contents),
        before_publication_gate,
    )
}

fn write_new_unpublished_using(
    path: &Path,
    write: impl FnOnce(&mut fs::File) -> io::Result<()>,
    before_publication_gate: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    fs::create_dir_all(parent_directory(path))?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut owned = TemporaryFile(path.to_path_buf());
    let result = write(&mut file).and_then(|()| file.sync_all());
    drop(file);
    result?;
    sync_parent_directory(path)?;
    // All extra I/O precedes the caller's current-authority check. The caller's
    // existing journal/record transaction is the publication operation.
    before_publication_gate()?;
    owned.0.clear();
    Ok(())
}

fn unsupported_operation(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::Unsupported {
        return true;
    }
    #[cfg(unix)]
    if error.raw_os_error().is_some_and(|code| {
        code == libc::EINVAL
            || code == libc::ENOTSUP
            || code == libc::EOPNOTSUPP
            || code == libc::ENOSYS
    }) {
        return true;
    }
    #[cfg(windows)]
    if matches!(error.raw_os_error(), Some(1 | 50)) {
        return true;
    }
    false
}

/// Metadata for a payload entry, excluding real macOS AppleDouble bookkeeping.
/// ExFAT/SMB can materialize `._name` companions automatically and remove them
/// together with `name`; a directory iterator can consequently yield a vanished
/// companion. Ordinary `._` data without the AppleDouble magic/version still counts.
pub fn payload_entry_metadata(path: &Path) -> io::Result<Option<fs::Metadata>> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if !metadata.is_file()
        || !path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("._"))
    {
        return Ok(Some(metadata));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_OPEN_REPARSE_POINT: inspect the entry, never its link target.
        options.custom_flags(0x0020_0000);
    }
    let mut file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        // If a similarly named payload cannot be inspected, conservatively count
        // its bytes rather than require new read permission for quota accounting.
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => return Ok(Some(metadata)),
        Err(error) => return Err(error),
    };
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if file.metadata()?.file_attributes() & 0x400 != 0 {
            return Ok(Some(metadata));
        }
    }
    let mut header = [0_u8; 8];
    match file.read_exact(&mut header) {
        // AppleDouble magic + version 2. This is OS metadata, not artifact data.
        Ok(()) if header == [0, 5, 22, 7, 0, 2, 0, 0] => Ok(None),
        Ok(()) => Ok(Some(metadata)),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(Some(metadata)),
        Err(error) => Err(error),
    }
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

struct TemporaryFile(PathBuf);
impl Drop for TemporaryFile {
    fn drop(&mut self) {
        if !self.0.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.0);
        }
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

/// Persist a changed directory entry when the filesystem supports it. File-content
/// sync remains mandatory. Unsupported directory sync retains the previous stores'
/// best-effort policy; open failures and actual I/O errors still propagate. Windows
/// has no POSIX directory fsync.
pub fn sync_parent_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    directory_sync_result(fs::File::open(parent_directory(path))?.sync_all())?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(unix)]
fn directory_sync_result(result: io::Result<()>) -> io::Result<()> {
    match result {
        // A valid directory handle can reject fsync even though regular file I/O
        // and rename work. Do not make directory-fsync support a catalog prerequisite.
        Err(error) if unsupported_operation(&error) => Ok(()),
        result => result,
    }
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
    fn payload_metadata_skips_appledouble_and_vanished_entries_but_keeps_normal_dot_data() {
        let root = tempfile::tempdir().unwrap();
        let metadata = root.path().join("._image.png");
        fs::write(&metadata, [0, 5, 22, 7, 0, 2, 0, 0, 1]).unwrap();
        assert!(payload_entry_metadata(&metadata).unwrap().is_none());
        let data = root.path().join("._ordinary-data");
        fs::write(&data, b"payload").unwrap();
        assert_eq!(payload_entry_metadata(&data).unwrap().unwrap().len(), 7);
        fs::remove_file(&metadata).unwrap();
        assert!(payload_entry_metadata(&metadata).unwrap().is_none());
        #[cfg(unix)]
        {
            let link = root.path().join("._link");
            std::os::unix::fs::symlink(&data, &link).unwrap();
            assert!(payload_entry_metadata(&link)
                .unwrap()
                .unwrap()
                .file_type()
                .is_symlink());
        }
    }

    #[cfg(unix)]
    #[test]
    fn unsupported_directory_sync_is_best_effort_but_io_failures_propagate() {
        for code in [libc::EINVAL, libc::ENOTSUP, libc::EOPNOTSUPP, libc::ENOSYS] {
            directory_sync_result(Err(io::Error::from_raw_os_error(code))).unwrap();
        }
        for code in [libc::EIO, libc::ENOSPC, libc::EACCES] {
            assert_eq!(
                directory_sync_result(Err(io::Error::from_raw_os_error(code)))
                    .unwrap_err()
                    .raw_os_error(),
                Some(code)
            );
        }
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
        assert_one_noclobber_winner(root.path());
    }

    fn assert_one_noclobber_winner(root: &Path) {
        let staging = root.join("staging");
        let path = root.join("record");
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
        assert_eq!(
            results.iter().filter(|result| result.is_ok()).count(),
            1,
            "publication results: {results:?}"
        );
        assert!(results
            .iter()
            .filter_map(|result| result.as_ref().err())
            .all(|error| error.kind() == io::ErrorKind::AlreadyExists));
        let contents = fs::read(path).unwrap();
        assert_eq!(contents.len(), 1024);
        assert!(contents.iter().all(|byte| *byte == contents[0]));
        assert!(names(&staging).is_empty());
    }

    #[test]
    fn unpublished_create_cleans_partial_io_and_checks_fence_after_sync() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("private");
        let error = write_new_unpublished_using(
            &path,
            |file| {
                file.write_all(b"partial")?;
                Err(io::Error::other("injected interrupted copy"))
            },
            || panic!("fence must not run after failed I/O"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("interrupted copy"));
        assert!(!path.exists());
        write_new_unpublished_with(&path, b"complete", || {
            assert_eq!(fs::read(&path)?, b"complete");
            Err(io::Error::other("claim lost after private write"))
        })
        .unwrap_err();
        assert!(!path.exists());
        write_new_unpublished_with(&path, b"complete", || Ok(())).unwrap();
        assert_eq!(
            write_new_unpublished_with(&path, b"clobber", || Ok(()))
                .unwrap_err()
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(fs::read(&path).unwrap(), b"complete");
    }

    #[test]
    fn unpublished_exclusive_create_has_one_concurrent_winner() {
        let root = tempfile::tempdir().unwrap();
        assert_unpublished_noclobber_winner(root.path());
    }

    fn assert_unpublished_noclobber_winner(root: &Path) {
        let path = root.join("private");
        let barrier = Arc::new(Barrier::new(8));
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for byte in 0_u8..8 {
                let path = &path;
                let barrier = barrier.clone();
                handles.push(scope.spawn(move || {
                    barrier.wait();
                    write_new_unpublished_with(path, &[byte; 1024], || Ok(()))
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
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 1024);
        assert!(bytes.iter().all(|byte| *byte == bytes[0]));
    }

    /// Explicit mounted-filesystem witness, separate from ordinary CI. The test
    /// first proves links are unsupported, so a hard-link-only regression fails.
    #[test]
    #[ignore = "requires SCENEWORKS_ATOMIC_NO_LINK_TEST_ROOT on a mounted filesystem without hard links"]
    fn no_replace_on_filesystem_without_hard_links() {
        let mount = std::env::var_os("SCENEWORKS_ATOMIC_NO_LINK_TEST_ROOT")
            .expect("provide the mounted no-hard-link filesystem root");
        let root = tempfile::tempdir_in(mount).unwrap();
        let probe = root.path().join("link-probe");
        fs::write(&probe, b"probe").unwrap();
        let error = fs::hard_link(&probe, root.path().join("unsupported-link"))
            .expect_err("this fixture must actually reject hard links");
        eprintln!("fixture hard-link rejection: {error:?}");
        fs::remove_file(probe).unwrap();
        if atomic_create_supported(root.path()).unwrap() {
            assert_one_noclobber_winner(root.path());
        } else {
            let path = root.path().join("record");
            let error = write_atomic_with(
                &path,
                b"complete",
                AtomicWriteOptions {
                    replace: false,
                    ..Default::default()
                },
                || Ok::<(), io::Error>(()),
            )
            .unwrap_err();
            assert!(unsupported_operation(&error), "{error}");
            assert!(
                names(root.path()).is_empty(),
                "unsupported atomic publication stays absent"
            );
            assert_unpublished_noclobber_winner(root.path());
        }
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
