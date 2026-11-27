//! Size-bounded, append-only log files for the desktop shell.
//!
//! The desktop owns every file in `%LOCALAPPDATA%\SceneWorks\logs`: it pipes the API and GPU
//! worker sidecars' stdout/stderr into `api.log` / `<label>-worker.log` and writes its own
//! tracing output to `desktop.log`. Those files used to be opened `append(true)` and never
//! trimmed, so a long-lived install grew `api.log` past 1 GB. [`RotatingLogFile`] bounds each
//! file to [`RotationPolicy::max_bytes`] and keeps at most [`RotationPolicy::keep`] older
//! generations (`api.1.log` newest .. `api.N.log` oldest), so one log's disk use is at most
//! `(keep + 1) * max_bytes`.
//!
//! Windows constraints shape the rotation:
//!
//! * The writer closes its own handle before renaming the live file, so the only handles that
//!   can block the rename belong to other processes (a log viewer or `Get-Content -Wait` opened
//!   without `FILE_SHARE_DELETE`). When the rename is refused, the recent tail is COPIED into
//!   the first backup and the live file is truncated in place, which only needs the far more
//!   commonly granted `FILE_SHARE_WRITE`. If that also fails, the writer keeps appending and
//!   retries after a further quarter of the budget instead of on every write.
//! * Every writer for one path in this process must share ONE handle ([`shared_log_file`]):
//!   a second long-lived handle would keep appending into the renamed backup after another
//!   writer rotated, re-growing it without bound.
//!
//! A pre-existing file already over budget (one written before rotation existed) is rotated
//! the first time it is opened, and only its last `max_bytes` (cut at a line boundary) are kept
//! in the backup — the most recent lines survive while the legacy bulk is released at once.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// How large one log may grow and how many rotated generations are kept beside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotationPolicy {
    /// Rotate before a write would take the live file past this many bytes.
    pub max_bytes: u64,
    /// Rotated generations kept (`name.1.ext` .. `name.<keep>.ext`). `0` truncates in place.
    pub keep: usize,
}

impl RotationPolicy {
    /// The desktop default: 50 MiB per file, three older generations (≤ 200 MiB per log).
    pub const DESKTOP: Self = Self {
        max_bytes: 50 * 1024 * 1024,
        keep: 3,
    };
}

/// Wait this long before re-trying to open a log that could not be opened, so a persistently
/// unwritable logs folder doesn't cost an `open` syscall per sidecar output chunk.
const REOPEN_BACKOFF: Duration = Duration::from_secs(5);

/// An append-only log file that rotates itself by size. See the module docs.
#[derive(Debug)]
pub struct RotatingLogFile {
    path: PathBuf,
    policy: RotationPolicy,
    file: Option<File>,
    len: u64,
    /// Rotate once a write would pass this size. Normally `max_bytes`; pushed further out
    /// after a failed rotation so a locked file isn't re-tried on every write.
    rotate_at: u64,
    last_open_failure: Option<Instant>,
}

impl RotatingLogFile {
    /// Open (creating it and its directory if needed) `path` for appending. A file already at
    /// or over budget is rotated immediately. Never fails: an unopenable path drops writes and
    /// re-tries the open later.
    pub fn open(path: impl Into<PathBuf>, policy: RotationPolicy) -> Self {
        let mut log = Self {
            path: path.into(),
            policy: RotationPolicy {
                max_bytes: policy.max_bytes.max(1),
                keep: policy.keep,
            },
            file: None,
            len: 0,
            rotate_at: policy.max_bytes.max(1),
            last_open_failure: None,
        };
        log.reopen();
        if log.file.is_some() && log.len >= log.rotate_at {
            log.rotate();
        }
        log
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append `bytes` (rotating first if they would overflow the budget) and flush.
    pub fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if self.file.is_none() {
            let backed_off = self
                .last_open_failure
                .is_some_and(|failed| failed.elapsed() < REOPEN_BACKOFF);
            if backed_off {
                return Err(io::Error::other("log file is not open"));
            }
            self.reopen();
        }
        if self.len > 0 && self.len.saturating_add(bytes.len() as u64) > self.rotate_at {
            self.rotate();
        }
        let Some(file) = self.file.as_mut() else {
            return Err(io::Error::other("log file is not open"));
        };
        file.write_all(bytes)?;
        file.flush()?;
        self.len = self.len.saturating_add(bytes.len() as u64);
        Ok(())
    }

    fn reopen(&mut self) {
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            Ok(file) => {
                self.len = file.metadata().map(|meta| meta.len()).unwrap_or(0);
                self.file = Some(file);
                self.last_open_failure = None;
            }
            Err(_) => {
                self.file = None;
                self.last_open_failure = Some(Instant::now());
            }
        }
    }

    fn rotate(&mut self) {
        // Release our own handle first: on Windows it is the one handle we control that could
        // otherwise block the rename.
        self.file = None;
        let outcome = rotate_files(&self.path, self.policy);
        self.reopen();
        match outcome {
            Ok(()) => self.rotate_at = self.policy.max_bytes,
            Err(error) => {
                self.rotate_at = self.len.saturating_add(self.policy.max_bytes / 4 + 1);
                if let Some(file) = self.file.as_mut() {
                    let note = format!(
                        "[sceneworks] log rotation of {} failed: {error}; retrying later\n",
                        self.path.display()
                    );
                    if file.write_all(note.as_bytes()).is_ok() {
                        self.len = self.len.saturating_add(note.len() as u64);
                    }
                }
            }
        }
    }
}

impl Write for RotatingLogFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.append(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.file.as_mut() {
            Some(file) => file.flush(),
            None => Ok(()),
        }
    }
}

/// `name.ext` -> `name.<generation>.ext` (`api.log` -> `api.1.log`); extension-less names get
/// a plain `.N` suffix.
pub fn rotated_path(path: &Path, generation: usize) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let name = match path.extension() {
        Some(ext) => format!("{stem}.{generation}.{}", ext.to_string_lossy()),
        None => format!("{stem}.{generation}"),
    };
    path.with_file_name(name)
}

/// Shift the backups down one generation and move the live file's content into generation 1.
/// The caller must have closed its own handle to `path`.
fn rotate_files(path: &Path, policy: RotationPolicy) -> io::Result<()> {
    if policy.keep == 0 {
        return truncate_in_place(path);
    }
    // Oldest first, so each rename lands on a freed slot. Failing to shift an old backup only
    // costs that backup, never the live file, so it is not an error.
    let _ = std::fs::remove_file(rotated_path(path, policy.keep));
    for generation in (1..policy.keep).rev() {
        let from = rotated_path(path, generation);
        if from.exists() {
            let _ = std::fs::rename(&from, rotated_path(path, generation + 1));
        }
    }
    let first = rotated_path(path, 1);
    let len = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    // A file within budget is moved whole. An oversized legacy file — or one another process
    // holds without delete sharing, so it cannot be renamed — keeps only its recent tail.
    if len <= policy.max_bytes && std::fs::rename(path, &first).is_ok() {
        return Ok(());
    }
    copy_tail(path, &first, policy.max_bytes)?;
    truncate_in_place(path)
}

/// Copy the last `max_bytes` of `source` (starting at a line boundary) into `target`, via a
/// temporary sibling so a failed copy never leaves a half-written backup.
fn copy_tail(source: &Path, target: &Path, max_bytes: u64) -> io::Result<()> {
    let mut input = File::open(source)?;
    let len = input.metadata()?.len();
    let start = len.saturating_sub(max_bytes);
    input.seek(SeekFrom::Start(start))?;
    let mut reader = BufReader::new(input);
    if start > 0 {
        // Drop the partial first line so the backup starts on a whole entry.
        let mut partial = Vec::new();
        reader.read_until(b'\n', &mut partial)?;
    }
    let temporary = target.with_extension("rotating");
    let copied = (|| {
        let mut output = File::create(&temporary)?;
        io::copy(&mut reader, &mut output)?;
        output.flush()
    })();
    if let Err(error) = copied {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    std::fs::rename(&temporary, target).inspect_err(|_| {
        let _ = std::fs::remove_file(&temporary);
    })
}

fn truncate_in_place(path: &Path) -> io::Result<()> {
    OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .map(drop)
}

/// One process-wide [`RotatingLogFile`] per path, shared by every writer of that log.
#[derive(Debug)]
pub struct SharedLogFile {
    inner: Mutex<RotatingLogFile>,
}

impl SharedLogFile {
    pub fn append(&self, bytes: &[u8]) -> io::Result<()> {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .append(bytes)
    }
}

/// `&SharedLogFile` is a writer, so `Arc<SharedLogFile>` is a `tracing_subscriber`
/// `MakeWriter` and the desktop's tracing file layer rotates through the same handle.
impl Write for &SharedLogFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.append(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .flush()
    }
}

/// The shared, rotating writer for `path` under the desktop [`RotationPolicy::DESKTOP`],
/// opened (and, if oversized, rotated) on first use.
pub fn shared_log_file(path: &Path) -> Arc<SharedLogFile> {
    shared_log_file_with_policy(path, RotationPolicy::DESKTOP)
}

/// [`shared_log_file`] with an explicit policy. The policy of the FIRST caller for a path wins.
pub fn shared_log_file_with_policy(path: &Path, policy: RotationPolicy) -> Arc<SharedLogFile> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Arc<SharedLogFile>>>> = OnceLock::new();
    let mut registry = REGISTRY
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    registry
        .entry(path.to_path_buf())
        .or_insert_with(|| {
            Arc::new(SharedLogFile {
                inner: Mutex::new(RotatingLogFile::open(path, policy)),
            })
        })
        .clone()
}

#[cfg(test)]
#[path = "log_rotation_tests.rs"]
mod tests;
