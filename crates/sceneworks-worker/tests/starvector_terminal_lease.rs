use std::{
    io::{BufRead, BufReader},
    process::{Command, Stdio},
};

#[test]
fn compiled_helper_holds_an_os_advisory_lock_and_rejects_a_second_holder() {
    let root = tempfile::tempdir().expect("temporary lease root");
    let lock = root.path().join("shared.lock");
    let helper = env!("CARGO_BIN_EXE_starvector_terminal_lease");

    let mut first = Command::new(helper)
        .args(["hold", lock.to_str().unwrap(), r#"{"owner":"first"}"#])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start first lease holder");
    let mut readiness = String::new();
    BufReader::new(first.stdout.take().expect("first stdout"))
        .read_line(&mut readiness)
        .expect("read first readiness");
    assert_eq!(readiness.trim(), "locked");

    let second = Command::new(helper)
        .args(["hold", lock.to_str().unwrap(), r#"{"owner":"second"}"#])
        .stdin(Stdio::null())
        .output()
        .expect("start second lease holder");
    assert_eq!(second.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&second.stderr)
        .contains("advisory lease is held; never auto-break"));

    drop(first.stdin.take());
    assert!(first.wait().expect("release first holder").success());
}
