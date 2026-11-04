/// Atomically promote a freshly converted temp directory to its final location,
/// replacing any stale directory there. On error the final location is left
/// untouched (the caller removes the temp dir), so a complete `final_dir` only ever
/// appears after a fully successful conversion.
///
/// sc-8837 (F-035): the previous implementation `remove_dir_all(final_dir)` *before*
/// renaming the temp dir in, so a rename failure — or a crash in the window between
/// the delete and the rename — left the user with no model at all after minutes of
/// conversion work. Instead we move any existing `final_dir` ASIDE to a sibling backup
/// first, rename temp → final, and only then delete the backup. If anything fails after
/// the aside move, we RESTORE the backup back to `final_dir`, honoring the doc's promise
/// that on error the previously working model is left untouched.
pub(crate) async fn finalize_converted_dir(
    temp_dir: &Path,
    final_dir: &Path,
    data_dir: &Path,
) -> WorkerResult<()> {
    finalize_converted_dir_with_hook(temp_dir, final_dir, data_dir, |_| {}).await
}

/// Backup path used to move a stale converted dir aside during finalize (sc-8837).
/// Kept in a dedicated child of the conversion root so recovery cannot confuse a
/// legitimate model name with a backup, while aside/restore remain same-filesystem
/// atomic renames.
fn converted_dir_backup_path(final_dir: &Path, backup_root: &Path) -> PathBuf {
    let name = final_dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    let suffix = format!(
        ".{name}.finalize-backup-{}-{}",
        std::process::id(),
        Uuid::new_v4()
    );
    backup_root.join(suffix)
}

const CONVERSION_MODELS_DIR: &str = "models/mlx";
const CONVERSION_LOCKS_DIR: &str = "cache/worker-locks/model-conversion";
const CONVERSION_LIFECYCLE_LOCK: &str = "lifecycle.lock";
const CONVERSION_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const CONVERSION_LOCK_POLL: std::time::Duration = std::time::Duration::from_millis(25);

/// Run one final promotion under a cross-process shared lifecycle lock plus an exclusive per-target
/// lock. The shared lock excludes the startup sweep; the target lock serializes the four utility
/// worker processes when they race to replace the same converted model.
async fn finalize_converted_dir_with_hook<F>(
    temp_dir: &Path,
    final_dir: &Path,
    data_dir: &Path,
    after_aside: F,
) -> WorkerResult<()>
where
    F: FnOnce(&Path) + Send + 'static,
{
    let temp_dir = temp_dir.to_path_buf();
    let final_dir = final_dir.to_path_buf();
    let data_dir = data_dir.to_path_buf();
    crate::load_progress::spawn_blocking(move || {
        finalize_converted_dir_blocking(&temp_dir, &final_dir, &data_dir, after_aside)
    })
    .await
    .map_err(|error| task_join_error("converted model finalize", error))?
}

fn finalize_converted_dir_blocking<F>(
    temp_dir: &Path,
    final_dir: &Path,
    data_dir: &Path,
    after_aside: F,
) -> WorkerResult<()>
where
    F: FnOnce(&Path),
{
    let ConversionManagedRoots {
        conversion_root,
        backup_root,
        lock_root,
    } = resolve_conversion_managed_roots(data_dir)?;
    let final_parent = final_dir.parent().ok_or_else(|| {
        WorkerError::InvalidPayload("Converted model final directory has no parent.".to_owned())
    })?;
    let final_parent = std::fs::canonicalize(final_parent)?;
    if final_parent != conversion_root {
        return Err(WorkerError::InvalidPayload(format!(
            "Converted model final directory {} must be a direct child of {}.",
            final_dir.display(),
            conversion_root.display()
        )));
    }
    let target_name = final_dir.file_name().ok_or_else(|| {
        WorkerError::InvalidPayload("Converted model final directory has no name.".to_owned())
    })?;
    if target_name == std::ffi::OsStr::new(MODEL_CONVERSION_BACKUPS_DIR_NAME) {
        return Err(WorkerError::InvalidPayload(format!(
            "Converted model final directory name {MODEL_CONVERSION_BACKUPS_DIR_NAME} is reserved for conversion recovery."
        )));
    }
    let canonical_target = conversion_root.join(target_name);
    let _locks = ConversionFinalizeLocks::acquire(&lock_root, &canonical_target)?;

    // Move any stale install aside instead of destroying it, so a later failure can restore it. No
    // sweep runs here: another process may have a live recovery source. Stranded backups are handled
    // only by the exclusive startup lifecycle pass.
    let backup_dir = if final_dir.exists() {
        let backup = converted_dir_backup_path(final_dir, &backup_root);
        std::fs::rename(final_dir, &backup)?;
        after_aside(&backup);
        Some(backup)
    } else {
        None
    };

    if let Err(error) = std::fs::rename(temp_dir, final_dir) {
        if let Some(backup) = backup_dir {
            let _ = std::fs::remove_dir_all(final_dir);
            let _ = std::fs::rename(&backup, final_dir);
        }
        return Err(error.into());
    }

    if let Some(backup) = backup_dir {
        let _ = std::fs::remove_dir_all(backup);
    }
    Ok(())
}

/// Cross-platform: the conversion locks are taken on every backend, so this module is NOT gated on
/// macOS the way the converter tests below are.
#[cfg(test)]
mod conversion_lock_tests {
    use super::*;
    use fs2::FileExt as _;

    /// sc-22738: released conversion locks must be free IMMEDIATELY, even while a descriptor this
    /// process handed to a child still references the same open file description. `flock(2)` locks
    /// live on the open file description, so a close-only release only takes effect once every such
    /// reference is gone — and conversion forks converter processes throughout, which would park
    /// the next finalize (and the startup sweep) behind an already-released lock for up to the
    /// 30-second timeout.
    #[test]
    fn released_conversion_locks_are_free_even_while_an_inherited_descriptor_survives() {
        let temporary = tempfile::tempdir().expect("temp directory");
        let lock_root = temporary.path().join("locks");
        std::fs::create_dir_all(&lock_root).expect("lock root");
        let target = temporary.path().join("models").join("converted");

        let locks = ConversionFinalizeLocks::acquire(&lock_root, &target).expect("locks acquire");
        let inherited = [
            locks
                ._lifecycle
                .inherited_descriptor()
                .expect("descriptor duplicates"),
            locks
                ._target
                .inherited_descriptor()
                .expect("descriptor duplicates"),
        ];
        drop(locks);

        // The lifecycle lock is taken SHARED by finalizers and EXCLUSIVE by the startup sweep, so
        // the sweep's acquisition is what a stuck release blocks; probe it that way.
        let lifecycle = open_conversion_lock(&lock_root.join(CONVERSION_LIFECYCLE_LOCK))
            .expect("lifecycle lock opens");
        assert!(
            lifecycle.try_lock_exclusive().is_ok(),
            "a released lifecycle lock must not stay held by an inherited descriptor"
        );
        drop(inherited);
    }
}

struct ConversionFinalizeLocks {
    /// [`FileLock`]s, not bare handles: each releases with an explicit `LOCK_UN`, because
    /// `close(2)` alone leaves the lock held while any forked child still references the same open
    /// file description (sc-22738), and this process forks converters throughout a conversion.
    _lifecycle: FileLock,
    _target: FileLock,
}

impl ConversionFinalizeLocks {
    fn acquire(lock_root: &Path, final_dir: &Path) -> WorkerResult<Self> {
        let lifecycle_path = lock_root.join(CONVERSION_LIFECYCLE_LOCK);
        let lifecycle = lock_conversion_file(
            open_conversion_lock(&lifecycle_path)?,
            &lifecycle_path,
            true,
        )?;

        // Hash the canonical target path so arbitrary/custom model ids can never alias the
        // lifecycle lock or another model's lock-file pathname.
        let target_digest = format!(
            "{:x}",
            Sha256::digest(final_dir.to_string_lossy().as_bytes())
        );
        let target_path = lock_root.join(format!("target-{target_digest}.lock"));
        let target =
            lock_conversion_file(open_conversion_lock(&target_path)?, &target_path, false)?;
        Ok(Self {
            _lifecycle: lifecycle,
            _target: target,
        })
    }
}

fn open_conversion_lock(path: &Path) -> WorkerResult<std::fs::File> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            return Err(WorkerError::InvalidPayload(format!(
                "Conversion lock path {} must be a regular file.",
                path.display()
            )));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)?)
}

/// Resolve one worker-owned directory and prove its canonical location remains inside the
/// canonical configured data root. This permits a symlinked `data_dir` itself (the macOS
/// `/var` -> `/private/var` case) while rejecting a nested `models`, `mlx`, or lock-root symlink
/// that redirects cleanup/locking outside app-managed storage.
fn canonical_managed_worker_dir(
    data_dir: &Path,
    relative: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    std::fs::create_dir_all(data_dir)?;
    let canonical_data_dir = std::fs::canonicalize(data_dir)?;
    let requested = data_dir.join(relative);
    // Resolve through the deepest existing ancestor before creating anything, so a nested symlink
    // cannot cause even the directory creation itself to escape the managed root.
    let normalized = normalize_existing_or_absolute(&requested)?;
    if normalized == canonical_data_dir || !normalized.starts_with(&canonical_data_dir) {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} {} escapes managed data root {}.",
            requested.display(),
            canonical_data_dir.display()
        )));
    }
    std::fs::create_dir_all(&normalized)?;
    let canonical = std::fs::canonicalize(&normalized)?;
    if canonical == canonical_data_dir || !canonical.starts_with(&canonical_data_dir) {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} {} escapes managed data root {}.",
            requested.display(),
            canonical_data_dir.display()
        )));
    }
    Ok(canonical)
}

struct ConversionManagedRoots {
    conversion_root: PathBuf,
    backup_root: PathBuf,
    lock_root: PathBuf,
}

fn canonical_conversion_backup_root(conversion_root: &Path) -> WorkerResult<PathBuf> {
    let backup_root = conversion_root.join(MODEL_CONVERSION_BACKUPS_DIR_NAME);
    match std::fs::symlink_metadata(&backup_root) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(WorkerError::InvalidPayload(format!(
                "Conversion backup root {} must be a real directory.",
                backup_root.display()
            )));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir(&backup_root)?;
        }
        Err(error) => return Err(error.into()),
    }
    let canonical = std::fs::canonicalize(&backup_root)?;
    if canonical.parent() != Some(conversion_root) {
        return Err(WorkerError::InvalidPayload(format!(
            "Conversion backup root {} must be a direct child of conversion model root {}.",
            canonical.display(),
            conversion_root.display()
        )));
    }
    Ok(canonical)
}

/// Resolve the model-conversion data and coordination roots and prove their canonical namespaces are
/// disjoint. Checking each root only against `data_dir` is insufficient: an internal symlink can
/// legally remain under app-managed storage while aliasing `cache/worker-locks/model-conversion` to
/// `models/mlx` (or one root below the other). That would place `lifecycle.lock` / `target-*.lock`
/// among model targets and let a reserved-looking model id rename or replace a live coordination
/// file. Reject equality and either-direction ancestor overlap before opening a lock or touching an
/// install.
fn resolve_conversion_managed_roots(data_dir: &Path) -> WorkerResult<ConversionManagedRoots> {
    let conversion_root =
        canonical_managed_worker_dir(data_dir, CONVERSION_MODELS_DIR, "conversion model root")?;
    let lock_root =
        canonical_managed_worker_dir(data_dir, CONVERSION_LOCKS_DIR, "conversion lock root")?;
    if conversion_root == lock_root
        || conversion_root.starts_with(&lock_root)
        || lock_root.starts_with(&conversion_root)
    {
        return Err(WorkerError::InvalidPayload(format!(
            "Conversion model root {} and lock root {} must be disjoint managed directories.",
            conversion_root.display(),
            lock_root.display()
        )));
    }
    let backup_root = canonical_conversion_backup_root(&conversion_root)?;
    Ok(ConversionManagedRoots {
        conversion_root,
        backup_root,
        lock_root,
    })
}

fn lock_conversion_file(file: std::fs::File, path: &Path, shared: bool) -> WorkerResult<FileLock> {
    let deadline = std::time::Instant::now() + CONVERSION_LOCK_TIMEOUT;
    let contended = fs2::lock_contended_error().raw_os_error();
    let mut file = file;
    loop {
        let result = if shared {
            FileLock::try_shared_retryable(file)
        } else {
            FileLock::try_exclusive_retryable(file)
        };
        match result {
            Ok(lock) => return Ok(lock),
            Err((handle, error)) if error.raw_os_error() == contended => {
                file = handle;
                if std::time::Instant::now() >= deadline {
                    return Err(WorkerError::Io(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!(
                            "timed out after {:?} waiting for conversion lock {}",
                            CONVERSION_LOCK_TIMEOUT,
                            path.display()
                        ),
                    )));
                }
                std::thread::sleep(CONVERSION_LOCK_POLL);
            }
            Err((_handle, error)) => return Err(error.into()),
        }
    }
}

/// Recover or remove crash-stranded converted-model backups at top-level worker startup.
///
/// The exclusive lifecycle lock guarantees no finalizer is active in any utility process. The scan
/// is bounded to the worker-owned backup namespace under `<data>/models/mlx`, so a legitimate model
/// directory whose name resembles the old sibling-backup grammar can never be touched. Only real
/// directories matching the exact backup naming convention are considered. If the canonical target
/// is missing, the newest backup is restored before older copies are removed.
pub async fn recover_stranded_model_conversions(data_dir: &Path) -> WorkerResult<()> {
    let data_dir = data_dir.to_path_buf();
    crate::load_progress::spawn_blocking(move || {
        sweep_stranded_conversion_backups_blocking(&data_dir)
    })
    .await
    .map_err(|error| task_join_error("converted model startup sweep", error))?
}

fn sweep_stranded_conversion_backups_blocking(data_dir: &Path) -> WorkerResult<()> {
    let ConversionManagedRoots {
        conversion_root,
        backup_root,
        lock_root,
    } = resolve_conversion_managed_roots(data_dir)?;
    let lifecycle_path = lock_root.join(CONVERSION_LIFECYCLE_LOCK);
    let _lifecycle = lock_conversion_file(
        open_conversion_lock(&lifecycle_path)?,
        &lifecycle_path,
        false,
    )?;

    let mut by_target: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for entry in std::fs::read_dir(&backup_root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_dir() || file_type.is_symlink() {
            continue;
        }
        let Some(target) = converted_backup_target_name(&entry.file_name()) else {
            continue;
        };
        by_target.entry(target).or_default().push(entry.path());
    }

    for (target, mut backups) in by_target {
        let final_dir = conversion_root.join(&target);
        match std::fs::symlink_metadata(&final_dir) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
                for backup in backups {
                    std::fs::remove_dir_all(backup)?;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                backups.sort_by(|left, right| {
                    backup_modified_time(left)
                        .cmp(&backup_modified_time(right))
                        .then_with(|| left.file_name().cmp(&right.file_name()))
                });
                let restore = backups.pop().expect("backup group is non-empty");
                std::fs::rename(&restore, &final_dir)?;
                for backup in backups {
                    std::fs::remove_dir_all(backup)?;
                }
            }
            Ok(_) => {
                tracing::warn!(
                    event = "conversion_backup_sweep_skipped_target",
                    target = %final_dir.display(),
                    "converted-model startup sweep found a non-directory target; preserving backups"
                );
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn converted_backup_target_name(name: &std::ffi::OsStr) -> Option<String> {
    let name = name.to_str()?.strip_prefix('.')?;
    let (target, suffix) = name.rsplit_once(".finalize-backup-")?;
    let mut suffix_parts = suffix.splitn(2, '-');
    let process_id = suffix_parts.next()?;
    if target.is_empty()
        || target.contains('/')
        || target.contains('\\')
        || process_id.parse::<u32>().is_err()
        || suffix_parts
            .next()
            .is_some_and(|value| Uuid::parse_str(value).is_err())
    {
        return None;
    }
    Some(target.to_owned())
}

fn backup_modified_time(path: &Path) -> std::time::SystemTime {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
}

#[cfg(test)]
pub(crate) async fn finalize_converted_dir_with_test_hook<F>(
    temp_dir: &Path,
    final_dir: &Path,
    data_dir: &Path,
    after_aside: F,
) -> WorkerResult<()>
where
    F: FnOnce(&Path) + Send + 'static,
{
    finalize_converted_dir_with_hook(temp_dir, final_dir, data_dir, after_aside).await
}
