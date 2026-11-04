//! Bounded child-process lifecycle shared by media imports and worker jobs.
//!
//! The caller owns media arguments and job controls; this module owns pipe drainage,
//! deadlines, kill/reap and teardown. No decoder or worker/API dependency belongs here.
use std::borrow::Cow;
use std::future::Future;
use std::io;
#[cfg(any(windows, test))]
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};

/// Exports retain the existing twelve-hour default. Uploads impose a shorter cap.
const FFMPEG_EXECUTION_TIMEOUT: Duration = Duration::from_secs(12 * 60 * 60);
/// Operator overrides remain finite and representable, even for a mistyped u64::MAX.
pub const FFMPEG_EXECUTION_TIMEOUT_MAX: Duration = Duration::from_secs(7 * 24 * 60 * 60);
const FFMPEG_TEARDOWN_TIMEOUT: Duration = Duration::from_secs(5);
/// Ordinary diagnostics retain a tail; structured consumers use their own streaming reader.
pub const STDERR_LIMIT: usize = 64 * 1024;

pub fn ffmpeg_execution_timeout_from(raw: Option<&str>) -> Duration {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(FFMPEG_EXECUTION_TIMEOUT)
        .min(FFMPEG_EXECUTION_TIMEOUT_MAX)
}

pub fn ffmpeg_execution_timeout() -> Duration {
    ffmpeg_execution_timeout_from(
        std::env::var("SCENEWORKS_FFMPEG_TIMEOUT_SECONDS")
            .ok()
            .as_deref(),
    )
}

pub fn ffmpeg_timeout_io_error(timeout: Duration) -> io::Error {
    io::Error::new(
        io::ErrorKind::TimedOut,
        format!("FFmpeg execution exceeded its {timeout:?} deadline."),
    )
}

fn checked_deadline_from(now: tokio::time::Instant, timeout: Duration) -> tokio::time::Instant {
    now.checked_add(timeout).unwrap_or(now)
}

pub fn ffmpeg_execution_deadline(requested: Duration) -> (tokio::time::Instant, Duration) {
    let effective = if requested.is_zero() {
        FFMPEG_EXECUTION_TIMEOUT
    } else {
        requested.min(FFMPEG_EXECUTION_TIMEOUT_MAX)
    };
    (
        checked_deadline_from(tokio::time::Instant::now(), effective),
        effective,
    )
}

/// Apply the desktop override only to the default name; explicit caller programs are preserved.
pub fn ffmpeg_program(program: &str) -> String {
    let configured = match std::env::var("SCENEWORKS_FFMPEG") {
        Ok(path) if program == "ffmpeg" && !path.trim().is_empty() => path,
        _ => program.to_owned(),
    };
    resolve_ffmpeg_program(&configured).into_owned()
}

/// Resolve an ffmpeg program string to a spawn-ready path, working around a Windows launch
/// failure. An executable reached through a symlink/reparse point — e.g. WinGet's
/// `…\Microsoft\WinGet\Links\ffmpeg.exe` shim, a symbolic link to the real Gyan.FFmpeg binary —
/// cannot be started under redirection-trust enforcement: `CreateProcess` returns
/// `ERROR_UNTRUSTED_MOUNT_POINT` (os error 448), the very wall the HF-cache symlinks hit in the
/// model loaders (see `sceneworks_worker::model_jobs`). We locate the program (searching `PATH`
/// for a bare name) and, if it is a reparse point, resolve it to its real target — read **without
/// traversing it** (`read_link`, chasing chained links) — and return that plain-file path. A plain
/// executable, a name we cannot locate, or any non-Windows host returns the input unchanged so
/// normal resolution (PATH search, the host's own ordering) is preserved.
///
/// This is the single resolver shared by the worker's `media_jobs::run_ffmpeg` and the transcoder
/// below, so every ffmpeg spawn — timeline export, video encode, audio mux, frame extract, image
/// transcode — is reparse-safe.
pub fn resolve_ffmpeg_program(program: &str) -> Cow<'_, str> {
    #[cfg(windows)]
    {
        if let Some(real) = resolve_ffmpeg_program_reparse(program) {
            return Cow::Owned(real);
        }
    }
    Cow::Borrowed(program)
}

/// Windows reparse-resolution core, factored out so it is unit-testable off-Windows
/// (`#[cfg(any(windows, test))]`); the public resolver only invokes it under `#[cfg(windows)]`.
/// Returns `Some(real_path)` only when the program resolves to a reparse point that must be
/// rewritten; `None` (keep the original string) for a plain file or a name we cannot locate.
#[cfg(any(windows, test))]
fn resolve_ffmpeg_program_reparse(program: &str) -> Option<String> {
    let located = locate_executable(program)?;
    // A plain file launches fine — leave it to normal resolution.
    if !is_symlink(&located) {
        return None;
    }
    let real = resolve_reparse_chain(&located)?;
    Some(real.to_string_lossy().into_owned())
}

/// `symlink_metadata` does not traverse the link, so a 448-prone reparse point is still detected.
#[cfg(any(windows, test))]
fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_symlink())
        .unwrap_or(false)
}

/// Find the file `CreateProcess` would launch. An explicit path is used as-is; a bare name is
/// searched on `PATH` (trying the name and a `.exe` suffix). A candidate is accepted if it is a
/// file *or* a symlink — the latter via `symlink_metadata`, which does not traverse and so still
/// sees a link whose target cannot be opened.
#[cfg(any(windows, test))]
fn locate_executable(program: &str) -> Option<PathBuf> {
    let direct = Path::new(program);
    if direct.is_absolute() || direct.components().count() > 1 {
        return (direct.exists() || is_symlink(direct)).then(|| direct.to_path_buf());
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        for suffix in ["", ".exe"] {
            let candidate = dir.join(format!("{program}{suffix}"));
            if candidate.is_file() || is_symlink(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// Follow a symlink to the first non-symlink path, resolving relative targets against the link's
/// parent and bounding the walk so a cyclic link cannot loop forever. The caller only invokes this
/// once `start` is known to be a symlink, so `read_link` runs without ever traversing the reparse
/// point that would otherwise hit os error 448.
#[cfg(any(windows, test))]
fn resolve_reparse_chain(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    for _ in 0..16 {
        if !is_symlink(&current) {
            return Some(current);
        }
        let target = std::fs::read_link(&current).ok()?;
        current = if target.is_absolute() {
            target
        } else {
            current.parent()?.join(target)
        };
    }
    None
}

#[derive(Debug)]
pub enum ProcessError<E> {
    Spawn(io::Error),
    Io(io::Error),
    /// Cleanup is complete before returning the caller's original control error.
    Control(E, Option<String>),
}

impl<E: std::fmt::Display> std::fmt::Display for ProcessError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(e) => write!(
                f,
                "Failed to start FFmpeg. Ensure ffmpeg is installed and on PATH: {e}"
            ),
            Self::Io(e) => e.fmt(f),
            Self::Control(e, cleanup) => {
                e.fmt(f)?;
                if let Some(cleanup) = cleanup {
                    write!(f, " FFmpeg cleanup also failed: {cleanup}")?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug)]
pub struct ProcessOutput<T, S = Vec<u8>> {
    pub status: ExitStatus,
    pub stdout: T,
    pub stderr: S,
}

/// Last lines/characters for a bounded, Unicode-safe process diagnostic.
pub fn bounded_tail(value: &str, max_lines: usize, max_chars: usize) -> String {
    let mut lines = value.lines().rev().take(max_lines).collect::<Vec<_>>();
    lines.reverse();
    let mut output = lines.join("\n");
    let char_count = output.chars().count();
    if char_count > max_chars {
        if max_chars == 0 {
            output.clear();
        } else {
            let start = output
                .char_indices()
                .nth(char_count - max_chars)
                .map_or(output.len(), |(index, _)| index);
            output = output[start..].to_owned();
        }
    }
    output
}

impl<T, S: AsRef<[u8]>> ProcessOutput<T, S> {
    /// Shared failure reporting for imports and jobs. Probes may intentionally
    /// consume nonzero status, so the runner preserves it instead of rejecting it.
    pub fn failure_detail(&self) -> Option<String> {
        if self.status.success() {
            return None;
        }
        let stderr = String::from_utf8_lossy(self.stderr.as_ref());
        let tail = bounded_tail(&stderr, 10, 2000);
        Some(if tail.trim().is_empty() {
            "FFmpeg command failed without stderr output.".to_owned()
        } else {
            tail
        })
    }
}

pub async fn discard_stdout(mut stdout: ChildStdout) -> io::Result<()> {
    tokio::io::copy(&mut stdout, &mut tokio::io::sink()).await?;
    Ok(())
}

pub async fn read_stderr_tail(mut pipe: ChildStderr) -> io::Result<Vec<u8>> {
    let mut tail = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        let count = pipe.read(&mut buffer).await?;
        if count == 0 {
            return Ok(tail);
        }
        let drop = (tail.len() + count).saturating_sub(STDERR_LIMIT);
        tail.drain(..drop);
        tail.extend_from_slice(&buffer[..count]);
    }
}

/// Run with one deadline for child execution AND all pipe joins. `control` may run a
/// caller-owned heartbeat/cancellation loop; its first error terminates/reaps the child.
/// The stdout consumer must stream or impose its own allocation bound.
pub async fn run_command<E, C, F, R, T>(
    command: Command,
    stdin_chunks: Option<Vec<Vec<u8>>>,
    timeout: Duration,
    control: C,
    read_stdout: F,
) -> Result<ProcessOutput<T>, ProcessError<E>>
where
    C: Future<Output = E>,
    F: FnOnce(ChildStdout) -> R,
    R: Future<Output = io::Result<T>> + Send + 'static,
    T: Send + 'static,
{
    run_command_with_stderr(
        command,
        stdin_chunks,
        timeout,
        control,
        read_stdout,
        read_stderr_tail,
    )
    .await
}

/// Run the same bounded lifecycle with a caller-owned streaming stderr consumer.
/// Structured readers must bound diagnostics and retain only the data their contract needs.
pub async fn run_command_with_stderr<E, C, F, R, T, G, H, S>(
    mut command: Command,
    stdin_chunks: Option<Vec<Vec<u8>>>,
    timeout: Duration,
    control: C,
    read_stdout: F,
    read_stderr: G,
) -> Result<ProcessOutput<T, S>, ProcessError<E>>
where
    C: Future<Output = E>,
    F: FnOnce(ChildStdout) -> R,
    R: Future<Output = io::Result<T>> + Send + 'static,
    T: Send + 'static,
    G: FnOnce(ChildStderr) -> H,
    H: Future<Output = io::Result<S>> + Send + 'static,
    S: Send + 'static,
{
    let (deadline, timeout) = ffmpeg_execution_deadline(timeout);
    command
        .stdin(if stdin_chunks.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn().map_err(ProcessError::Spawn)?;
    let mut stdin_task = stdin_chunks.map(|chunks| {
        let mut stdin = child.stdin.take().expect("piped stdin");
        tokio::spawn(async move {
            for chunk in chunks {
                stdin.write_all(&chunk).await?;
            }
            stdin.shutdown().await
        })
    });
    let mut stderr_task = Some(tokio::spawn(read_stderr(
        child.stderr.take().expect("piped stderr"),
    )));
    let mut stdout_task = Some(tokio::spawn(read_stdout(
        child.stdout.take().expect("piped stdout"),
    )));
    tokio::pin!(control);
    let result = async {
        let mut stdout_value = None;
        let mut stderr_value = None;
        let status = loop {
            tokio::select! {
                status = child.wait() => break status.map_err(ProcessError::Io)?,
                _ = tokio::time::sleep_until(deadline) => return Err(ProcessError::Io(ffmpeg_timeout_io_error(timeout))),
                error = &mut control => return Err(ProcessError::Control(error, None)),
                output = async { stdout_task.as_mut().expect("stdout task").await }, if stdout_task.is_some() => {
                    let _ = stdout_task.take();
                    stdout_value = Some(output.map_err(|e| process_task_error(FfmpegTaskDeadlineError::Join(e), timeout))?.map_err(ProcessError::Io)?);
                }
                output = async { stderr_task.as_mut().expect("stderr task").await }, if stderr_task.is_some() => {
                    let _ = stderr_task.take();
                    stderr_value = Some(output.map_err(|e| process_task_error(FfmpegTaskDeadlineError::Join(e), timeout))?.map_err(ProcessError::Io)?);
                }
            }
        };
        let stderr = match stderr_value {
            Some(value) => value,
            None => join_ffmpeg_task_until(&mut stderr_task, deadline)
                .await.map_err(|e| process_task_error(e, timeout))?.map_err(ProcessError::Io)?,
        };
        let stdout = match stdout_value {
            Some(value) => value,
            None => join_ffmpeg_task_until(&mut stdout_task, deadline)
                .await.map_err(|e| process_task_error(e, timeout))?.map_err(ProcessError::Io)?,
        };
        if status.success() {
            if stdin_task.is_some() {
                join_ffmpeg_task_until(&mut stdin_task, deadline)
                    .await.map_err(|e| process_task_error(e, timeout))?.map_err(ProcessError::Io)?;
            }
        } else {
            abort_ffmpeg_task_until(&mut stdin_task, deadline)
                .await.map_err(|e| process_task_error(e, timeout))?;
        }
        Ok(ProcessOutput { status, stdout, stderr })
    }.await;
    match result {
        Ok(output) => Ok(output),
        Err(error) => {
            let cleanup = stop_ffmpeg_child(
                &mut child,
                &mut stdin_task,
                &mut stderr_task,
                &mut stdout_task,
            )
            .await;
            Err(match error {
                ProcessError::Io(error) => ProcessError::Io(io_error_with_cleanup(error, cleanup)),
                ProcessError::Control(error, _) => ProcessError::Control(error, cleanup),
                other => other,
            })
        }
    }
}

fn process_task_error<E>(error: FfmpegTaskDeadlineError, timeout: Duration) -> ProcessError<E> {
    ProcessError::Io(match error {
        FfmpegTaskDeadlineError::TimedOut => ffmpeg_timeout_io_error(timeout),
        FfmpegTaskDeadlineError::Join(error) => {
            io::Error::other(format!("FFmpeg pipe task failed: {error}"))
        }
    })
}

fn io_error_with_cleanup(error: io::Error, cleanup: Option<String>) -> io::Error {
    match cleanup {
        None => error,
        Some(cleanup) => io::Error::new(
            error.kind(),
            format!("{error} FFmpeg cleanup also failed: {cleanup}"),
        ),
    }
}

enum FfmpegTaskDeadlineError {
    TimedOut,
    Join(tokio::task::JoinError),
}

async fn join_ffmpeg_task_until<T>(
    task: &mut Option<tokio::task::JoinHandle<T>>,
    deadline: tokio::time::Instant,
) -> Result<T, FfmpegTaskDeadlineError> {
    let mut handle = task
        .take()
        .expect("FFmpeg pipe task exists until joined or aborted");
    match tokio::time::timeout_at(deadline, &mut handle).await {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(error)) => Err(FfmpegTaskDeadlineError::Join(error)),
        Err(_) => {
            *task = Some(handle);
            Err(FfmpegTaskDeadlineError::TimedOut)
        }
    }
}

async fn abort_ffmpeg_task_until<T>(
    task: &mut Option<tokio::task::JoinHandle<T>>,
    deadline: tokio::time::Instant,
) -> Result<(), FfmpegTaskDeadlineError> {
    let Some(mut handle) = task.take() else {
        return Ok(());
    };
    handle.abort();
    match tokio::time::timeout_at(deadline, &mut handle).await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) if error.is_cancelled() => Ok(()),
        Ok(Err(error)) => Err(FfmpegTaskDeadlineError::Join(error)),
        Err(_) => {
            *task = Some(handle);
            Err(FfmpegTaskDeadlineError::TimedOut)
        }
    }
}

fn task_cleanup_detail(label: &str, error: FfmpegTaskDeadlineError) -> String {
    match error {
        FfmpegTaskDeadlineError::TimedOut => {
            format!("{label} did not stop within {FFMPEG_TEARDOWN_TIMEOUT:?}")
        }
        FfmpegTaskDeadlineError::Join(error) => format!("{label} failed to join: {error}"),
    }
}

/// Explicitly terminate and reap the child, then stand down both pipe tasks. `kill_on_drop` remains
/// a panic/unwind backstop, but ordinary timeout/heartbeat/cancel paths never rely on drop semantics
/// and never leave a detached stdin or stderr task behind.
async fn stop_ffmpeg_child<T, S>(
    child: &mut Child,
    stdin_task: &mut Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    stderr_task: &mut Option<tokio::task::JoinHandle<std::io::Result<S>>>,
    stdout_task: &mut Option<tokio::task::JoinHandle<std::io::Result<T>>>,
) -> Option<String> {
    let deadline = checked_deadline_from(tokio::time::Instant::now(), FFMPEG_TEARDOWN_TIMEOUT);
    let mut details = Vec::new();
    if let Err(error) = abort_ffmpeg_task_until(stdin_task, deadline).await {
        details.push(task_cleanup_detail("FFmpeg stdin writer", error));
    }
    if let Err(error) = abort_ffmpeg_task_until(stderr_task, deadline).await {
        details.push(task_cleanup_detail("FFmpeg stderr reader", error));
    }

    if let Err(error) = abort_ffmpeg_task_until(stdout_task, deadline).await {
        details.push(task_cleanup_detail("FFmpeg stdout reader", error));
    }

    let mut child_reaped = false;
    match child.try_wait() {
        Ok(Some(_)) => child_reaped = true,
        Ok(None) => {}
        Err(error) => details.push(format!("could not inspect FFmpeg child status: {error}")),
    }
    if !child_reaped {
        let kill_error = child.start_kill().err();
        match tokio::time::timeout_at(deadline, child.wait()).await {
            // `start_kill` can lose a benign child-exited race. A successful wait proves that race
            // was safe and that the child was reaped, so the transient kill error is not reported.
            Ok(Ok(_)) => {}
            Ok(Err(wait_error)) => {
                if let Some(kill_error) = kill_error {
                    details.push(format!("could not kill FFmpeg child: {kill_error}"));
                }
                details.push(format!("could not reap FFmpeg child: {wait_error}"));
            }
            Err(_) => {
                if let Some(kill_error) = kill_error {
                    details.push(format!("could not kill FFmpeg child: {kill_error}"));
                }
                details.push(format!(
                    "FFmpeg child was not reaped within {FFMPEG_TEARDOWN_TIMEOUT:?}"
                ));
            }
        }
    }

    (!details.is_empty()).then(|| details.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    /// The ffmpeg reparse-resolution fix (os error 448): a program reached through a symlink — e.g.
    /// WinGet's `…\WinGet\Links\ffmpeg.exe` shim — resolves to its real, non-reparse target so
    /// `CreateProcess` never has to traverse the reparse point. Skips when the platform won't let
    /// the test create a symlink (e.g. Windows without Developer Mode).
    #[test]
    fn resolve_ffmpeg_program_follows_reparse_to_real_binary() {
        let dir = tempfile::tempdir().expect("temp dir");
        let real = dir.path().join("ffmpeg-real.exe");
        std::fs::write(&real, b"MZ-stub").expect("write real binary");
        let link = dir.path().join("ffmpeg.exe");

        // Absolute target, like WinGet's Links shim.
        #[cfg(unix)]
        let made = std::os::unix::fs::symlink(&real, &link).is_ok();
        #[cfg(windows)]
        let made = std::os::windows::fs::symlink_file(&real, &link).is_ok();
        #[cfg(not(any(unix, windows)))]
        let made = false;
        if !made {
            return; // no privilege to create a symlink — nothing to resolve
        }
        assert!(is_symlink(&link), "precondition: the shim is a symlink");

        let resolved = resolve_ffmpeg_program_reparse(&link.to_string_lossy())
            .expect("a symlinked program resolves to its target");
        assert_eq!(
            std::fs::canonicalize(&resolved).expect("canonicalize resolved"),
            std::fs::canonicalize(&real).expect("canonicalize real"),
            "resolved path must point at the real binary"
        );
        assert!(
            !is_symlink(Path::new(&resolved)),
            "resolved path must not itself be a reparse point"
        );

        // A plain binary is left untouched (None → caller keeps the original program).
        assert!(
            resolve_ffmpeg_program_reparse(&real.to_string_lossy()).is_none(),
            "a plain file needs no rewrite"
        );
    }

    fn fixture(mode: &str) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "media_process::tests::process_fixture",
                "--ignored",
                "--nocapture",
            ])
            .env("SW_MEDIA_PROCESS_FIXTURE", mode);
        command
    }

    #[test]
    #[ignore = "subprocess fixture for lifecycle tests"]
    fn process_fixture() {
        use std::io::Write;
        if let Some(path) = std::env::var_os("SW_MEDIA_PIDFILE") {
            std::fs::write(path, std::process::id().to_string()).unwrap();
        }
        match std::env::var("SW_MEDIA_PROCESS_FIXTURE").unwrap().as_str() {
            "sleep" => std::thread::sleep(Duration::from_secs(3)),
            "flood" => {
                for _ in 0..64 {
                    std::io::stdout().write_all(&[b'o'; 8192]).unwrap();
                    std::io::stderr().write_all(&[b'e'; 8192]).unwrap();
                }
                eprintln!("final-stderr-marker");
            }
            "inherited" => {
                let child = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "media_process::tests::process_fixture",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env("SW_MEDIA_PROCESS_FIXTURE", "sleep")
                    .stdin(Stdio::null())
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .unwrap();
                drop(child);
            }
            other => panic!("unknown fixture {other}"),
        }
    }

    #[tokio::test]
    async fn drains_both_output_pipes_without_retaining_unbounded_output() {
        let output = run_command(
            fixture("flood"),
            None,
            Duration::from_secs(3),
            std::future::pending::<Infallible>(),
            discard_stdout,
        )
        .await
        .unwrap();
        assert!(output.status.success());
        assert!(output.stderr.len() <= STDERR_LIMIT);
        assert!(output.stderr.ends_with(b"final-stderr-marker\n"));
    }

    #[tokio::test]
    async fn finite_timeout_aborts_blocked_stdin_and_kills_and_reaps_child() {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("pid");
        let mut command = fixture("sleep");
        command.env("SW_MEDIA_PIDFILE", &pidfile);
        let started = std::time::Instant::now();
        let error = run_command(
            command,
            Some(vec![vec![0; 2 * 1024 * 1024]]),
            Duration::from_millis(500),
            std::future::pending::<Infallible>(),
            discard_stdout,
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, ProcessError::Io(ref error) if error.kind() == io::ErrorKind::TimedOut)
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        #[cfg(unix)]
        {
            let pid = std::fs::read_to_string(pidfile).unwrap();
            assert!(!std::process::Command::new("kill")
                .args(["-0", &pid])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap()
                .success());
        }
    }

    #[tokio::test]
    async fn inherited_open_output_pipes_share_execution_deadline() {
        let started = std::time::Instant::now();
        let error = run_command(
            fixture("inherited"),
            None,
            Duration::from_millis(500),
            std::future::pending::<Infallible>(),
            discard_stdout,
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, ProcessError::Io(ref error) if error.kind() == io::ErrorKind::TimedOut)
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        // Let the intentionally inherited holder finish before this fixture returns.
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    #[tokio::test]
    async fn stdout_consumer_failure_terminates_child_immediately() {
        let started = std::time::Instant::now();
        let error = run_command(
            fixture("sleep"),
            None,
            Duration::from_secs(10),
            std::future::pending::<Infallible>(),
            |_| async { Err::<(), _>(io::Error::other("invalid frame")) },
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, ProcessError::Io(ref error) if error.to_string().contains("invalid frame"))
        );
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn control_errors_remain_typed_after_teardown() {
        let error = run_command(
            fixture("sleep"),
            None,
            Duration::from_secs(10),
            async { 42u32 },
            discard_stdout,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, ProcessError::Control(42, None)));
    }

    #[test]
    fn timeout_configuration_is_finite_and_fail_closed() {
        let default = ffmpeg_execution_timeout_from(None);
        assert!(default > Duration::ZERO);
        assert_eq!(
            ffmpeg_execution_timeout_from(Some("37")),
            Duration::from_secs(37)
        );
        assert_eq!(ffmpeg_execution_timeout_from(Some("0")), default);
        assert_eq!(ffmpeg_execution_timeout_from(Some("invalid")), default);
        assert_eq!(
            ffmpeg_execution_timeout_from(Some(&u64::MAX.to_string())),
            FFMPEG_EXECUTION_TIMEOUT_MAX
        );
        let (_, effective) = ffmpeg_execution_deadline(Duration::MAX);
        assert_eq!(effective, FFMPEG_EXECUTION_TIMEOUT_MAX);
    }
}
