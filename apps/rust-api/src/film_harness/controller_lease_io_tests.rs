use super::{ControllerLease, CONTROLLER_LOCK_FILE};
use std::io::{Read as _, Seek as _};

#[test]
fn interrupted_takeover_reads_owned_handle_and_rewinds_replacement_metadata() {
    let temporary = tempfile::Builder::new()
        .prefix(&format!("controller-lease-io-{}-", std::process::id()))
        .tempdir()
        .unwrap();
    let path = temporary.path().join(CONTROLLER_LOCK_FILE);
    let old_owner = "dead-marker-long-interrupted-owner-".repeat(8);
    std::fs::write(&path, format!("owner={old_owner}\npid=999999\n")).unwrap();

    let lease = ControllerLease::acquire_interrupted(temporary.path(), "recovered")
        .unwrap()
        .expect("OS-released crash metadata is recoverable on Windows and POSIX");
    let mut file = lease._lock.file();
    file.rewind().unwrap();
    let mut metadata = String::new();
    file.read_to_string(&mut metadata).unwrap();
    assert!(metadata.starts_with("owner=recovered\n"), "{metadata:?}");
    assert!(
        !metadata.contains('\0'),
        "the old read cursor must not leave NUL padding"
    );
    assert!(
        !metadata.contains("dead-marker"),
        "the previous owner must be truncated"
    );
    assert!(metadata
        .lines()
        .any(|line| line == format!("pid={}", std::process::id())));
    assert!(
        ControllerLease::acquire(temporary.path(), "competing").is_err(),
        "inspection must retain exclusive ownership"
    );
    drop(lease);
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "",
        "a clean release still clears recovery metadata"
    );
}
