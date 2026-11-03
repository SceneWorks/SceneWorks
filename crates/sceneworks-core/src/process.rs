//! Metadata-only process liveness shared by API parent watching and desktop orphan recovery.

#[cfg(unix)]
fn unix_probe_alive(result: Result<(), nix::errno::Errno>) -> bool {
    result.is_ok() || result == Err(nix::errno::Errno::EPERM)
}

/// Permission denial means the process exists. Invalid/nonpositive PIDs never name a parent.
#[cfg(unix)]
pub fn pid_alive(pid: i32) -> bool {
    pid > 0
        && unix_probe_alive(nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid),
            None,
        ))
}

#[cfg(any(windows, test))]
fn tasklist_stdout_alive(pid: i32, stdout: Option<&[u8]>) -> bool {
    // A launch failure must not make a child self-terminate; CSV quotes delimit the exact PID.
    stdout.is_none_or(|bytes| String::from_utf8_lossy(bytes).contains(&format!("\"{pid}\"")))
}

#[cfg(windows)]
pub fn pid_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    let output = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output();
    tasklist_stdout_alive(
        pid,
        output.as_ref().ok().map(|output| output.stdout.as_slice()),
    )
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    #[test]
    fn permission_denial_is_alive_and_absence_is_not() {
        use nix::errno::Errno;
        assert!(super::unix_probe_alive(Ok(())));
        assert!(super::unix_probe_alive(Err(Errno::EPERM)));
        assert!(!super::unix_probe_alive(Err(Errno::ESRCH)));
        assert!(!super::pid_alive(0));
        assert!(!super::pid_alive(-1));
    }

    #[test]
    fn tasklist_uses_exact_csv_pid_and_keeps_launch_failure_alive() {
        assert!(super::tasklist_stdout_alive(
            42,
            Some(b"\"app.exe\",\"42\",\"Console\"")
        ));
        assert!(!super::tasklist_stdout_alive(
            42,
            Some(b"\"app42.exe\",\"142\"")
        ));
        assert!(!super::tasklist_stdout_alive(42, Some(b"INFO: no tasks")));
        assert!(super::tasklist_stdout_alive(42, None));
    }
}
