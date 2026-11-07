use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};

/// Return a process-scoped sibling used to build a shared cache before publishing it.
pub(crate) fn staging_path(final_path: &Path) -> PathBuf {
    let file_name = final_path
        .file_name()
        .expect("cache path must have a final component");
    let staging_name =
        if let (Some(stem), Some(extension)) = (final_path.file_stem(), final_path.extension()) {
            let mut name = OsString::from(stem);
            name.push(format!(".tmp.{}.", std::process::id()));
            name.push(extension);
            name
        } else {
            let mut name = OsString::from(file_name);
            name.push(format!(".tmp.{}", std::process::id()));
            name
        };
    final_path.with_file_name(staging_name)
}

fn path_exists(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok()
}

fn remove_path(path: &Path) -> io::Result<()> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    if metadata.file_type().is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

/// Clear a stale staging artifact from an interrupted run and ensure its parent exists.
pub(crate) fn prepare_staging(final_path: &Path) -> io::Result<PathBuf> {
    let staging = staging_path(final_path);
    if let Some(parent) = staging.parent() {
        std::fs::create_dir_all(parent)?;
    }
    remove_path(&staging)?;
    Ok(staging)
}

/// Atomically publish a complete staging artifact at its stable shared name.
///
/// Directory rename cannot replace a non-empty directory on POSIX. If another process wins that
/// race, discard this process's complete staging artifact and reuse the winner instead.
pub(crate) fn publish(staging: &Path, final_path: &Path) -> io::Result<()> {
    match std::fs::rename(staging, final_path) {
        Ok(()) => Ok(()),
        Err(_) if path_exists(final_path) => remove_path(staging),
        Err(error) => Err(error),
    }
}

/// Create a symlink at a process-scoped sibling and publish it atomically.
pub(crate) fn symlink_or_reuse(source: &Path, final_path: &Path) -> io::Result<()> {
    if path_exists(final_path) {
        return Ok(());
    }
    let staging = prepare_staging(final_path)?;
    std::os::unix::fs::symlink(source, &staging)?;
    publish(&staging, final_path)
}
