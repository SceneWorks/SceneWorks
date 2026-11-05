//! Metadata-only process liveness shared by API parent watching and desktop orphan recovery.

#[cfg(unix)]
fn unix_probe_alive(result: Result<(), nix::errno::Errno>) -> bool {
    result.is_ok() || result == Err(nix::errno::Errno::EPERM)
}

#[cfg(any(windows, test))]
fn tasklist_stdout_alive(pid: i32, stdout: Option<&[u8]>) -> bool {
    // A launch failure must not make a child self-terminate; CSV quotes delimit the exact PID.
    stdout.is_none_or(|bytes| String::from_utf8_lossy(bytes).contains(&format!("\"{pid}\"")))
}

/// Permission denial means the process exists. Invalid/nonpositive PIDs never name a parent.
#[cfg(any(unix, windows))]
pub fn pid_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
        unix_probe_alive(nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid),
            None,
        ))
    }
    #[cfg(windows)]
    {
        let output = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output();
        tasklist_stdout_alive(
            pid,
            output.as_ref().ok().map(|output| output.stdout.as_slice()),
        )
    }
}

/// Desktop child handles carry unsigned PIDs; reject values the signed platform probe cannot use.
#[cfg(any(unix, windows))]
pub fn pid_alive_u32(pid: u32) -> bool {
    i32::try_from(pid).is_ok_and(pid_alive)
}

#[cfg(test)]
mod tests {
    #[test]
    fn unsigned_process_ids_preserve_zero_and_unrepresentable_refusal() {
        assert!(!super::pid_alive_u32(0));
        assert!(!super::pid_alive_u32(u32::MAX));
    }

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
