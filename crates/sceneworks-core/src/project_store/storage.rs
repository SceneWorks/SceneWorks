//! ProjectStore storage aggregate.

use super::*;

/// True when `error` means the storage location rejected a write (as opposed to a
/// logic/validation error). Covers SQLite `SQLITE_READONLY` ("attempt to write a
/// readonly database"), `SQLITE_CANTOPEN` (read-only directory), `SQLITE_PERM`,
/// and IO `PermissionDenied`/`ReadOnlyFilesystem` — the whole family a
/// non-writable workspace folder produces (issue #1435 / sc-11855).
pub(super) fn is_write_denied(error: &ProjectStoreError) -> bool {
    match error {
        ProjectStoreError::Sqlite(rusqlite::Error::SqliteFailure(err, _)) => matches!(
            err.code,
            rusqlite::ErrorCode::ReadOnly
                | rusqlite::ErrorCode::CannotOpen
                | rusqlite::ErrorCode::PermissionDenied
        ),
        ProjectStoreError::Io(err) => matches!(
            err.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem
        ),
        _ => false,
    }
}

/// Actionable, path-naming message for a non-writable workspace folder. Replaces
/// the opaque raw SQLite/IO text the user would otherwise see (issue #1435).
pub(super) fn storage_not_writable_message(dir: &Path) -> String {
    format!(
        "SceneWorks can't write to your workspace folder ({}). This is almost always a \
         permissions problem with that location — pick a different workspace folder in \
         Settings, or fix the folder's permissions, then try again.",
        dir.display()
    )
}

/// Exercise the exact write path a `project.db` needs, so a non-writable
/// workspace folder fails fast with an actionable error instead of surfacing an
/// opaque `SQLITE_READONLY` from deep in project provisioning (issue #1435 /
/// sc-11855). A plain "can I create a file here" check is NOT enough: a granular
/// write denial (a macOS inherited ACL, or an endpoint-security/backup agent) can
/// permit creating and appending to files while blocking the in-place rewrites a
/// rollback-journal SQLite db performs — which is exactly why the WAL-mode
/// `jobs.db` keeps working while `project.db` writes are refused. So we open a
/// throwaway rollback-mode db in `dir`, write to it, and clean up.
///
/// Public so the rust-api startup path can run the same faithful probe against
/// the projects tree and log the result for diagnosis (sc-11855 fix C).
pub fn probe_storage_writable(dir: &Path) -> ProjectStoreResult<()> {
    fs::create_dir_all(dir)?;
    let probe = dir.join(format!(".sceneworks-write-test-{}.db", random_hex(8)?));
    let outcome = (|| -> rusqlite::Result<()> {
        let connection = Connection::open(&probe)?;
        connection.execute_batch("create table t (x); insert into t values (1); drop table t;")
    })();
    // Best-effort cleanup of the probe db and any rollback-journal sidecar,
    // regardless of outcome — never mask the real error with a cleanup failure.
    let _ = fs::remove_file(&probe);
    let mut journal = probe.clone().into_os_string();
    journal.push("-journal");
    let _ = fs::remove_file(PathBuf::from(journal));
    outcome.map_err(ProjectStoreError::from)
}

pub(super) fn read_dir_paths(path: &Path) -> ProjectStoreResult<Vec<PathBuf>> {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::PathStat);
    if !path.exists() {
        return Ok(Vec::new());
    }
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::DirectoryScan);
    fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()).map_err(Into::into))
        .collect()
}

pub(super) fn move_or_copy_file(source: &Path, destination: &Path) -> ProjectStoreResult<()> {
    match fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(source, destination)?;
            fs::remove_file(source)?;
            Ok(())
        }
    }
}
