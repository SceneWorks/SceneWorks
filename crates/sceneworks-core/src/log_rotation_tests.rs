use super::*;

fn small_policy(max_bytes: u64, keep: usize) -> RotationPolicy {
    RotationPolicy { max_bytes, keep }
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

fn log_files(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("dir lists")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

#[test]
fn rotated_names_keep_the_log_extension() {
    let path = Path::new("logs").join("api.log");
    assert_eq!(rotated_path(&path, 1), Path::new("logs").join("api.1.log"));
    assert_eq!(
        rotated_path(&Path::new("logs").join("candle-worker.log"), 3),
        Path::new("logs").join("candle-worker.3.log")
    );
    assert_eq!(
        rotated_path(&Path::new("logs").join("plain"), 2),
        Path::new("logs").join("plain.2")
    );
}

#[test]
fn rotates_at_threshold_and_keeps_a_bounded_number_of_generations() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("api.log");
    let mut log = RotatingLogFile::open(&path, small_policy(100, 2));
    // 40 lines of 10 bytes = 400 bytes against a 100-byte budget.
    for index in 0..40 {
        log.append(format!("line {index:04}\n").as_bytes())
            .expect("append");
    }
    drop(log);

    assert_eq!(
        log_files(dir.path()),
        ["api.1.log", "api.2.log", "api.log"],
        "exactly the live file plus `keep` generations remain"
    );
    for name in ["api.log", "api.1.log", "api.2.log"] {
        let len = std::fs::metadata(dir.path().join(name))
            .expect("meta")
            .len();
        assert!(
            len <= 100,
            "{name} is {len} bytes, over the 100-byte budget"
        );
    }
    // Nothing recent was lost: the last 30 lines are contiguous across the three files.
    let combined = format!(
        "{}{}{}",
        read(&dir.path().join("api.2.log")),
        read(&dir.path().join("api.1.log")),
        read(&path)
    );
    let expected: String = (10..40).map(|index| format!("line {index:04}\n")).collect();
    assert_eq!(combined, expected);
}

#[test]
fn an_oversized_file_found_at_open_is_rotated_out_keeping_only_its_recent_tail() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("api.log");
    let legacy: String = (0..1000).map(|index| format!("old {index:05}\n")).collect();
    std::fs::write(&path, &legacy).expect("seed legacy log");

    let mut log = RotatingLogFile::open(&path, small_policy(100, 3));
    log.append(b"new session\n").expect("append");
    drop(log);

    assert_eq!(read(&path), "new session\n", "the live file starts fresh");
    let backup = read(&dir.path().join("api.1.log"));
    assert!(
        backup.len() <= 100,
        "backup trimmed to budget: {}",
        backup.len()
    );
    assert!(
        backup.ends_with("old 00999\n"),
        "the most recent legacy lines survive: {backup:?}"
    );
    assert!(
        backup.starts_with("old "),
        "the backup starts on a whole line: {backup:?}"
    );
    assert!(!dir.path().join("api.1.rotating").exists());
}

#[test]
fn a_file_under_budget_is_appended_to_without_rotation() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("desktop.log");
    std::fs::write(&path, "earlier\n").expect("seed");
    let mut log = RotatingLogFile::open(&path, small_policy(1024, 3));
    log.append(b"later\n").expect("append");
    drop(log);
    assert_eq!(read(&path), "earlier\nlater\n");
    assert_eq!(log_files(dir.path()), ["desktop.log"]);
}

#[test]
fn keep_zero_truncates_in_place() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("api.log");
    let mut log = RotatingLogFile::open(&path, small_policy(20, 0));
    log.append(b"0123456789\n").expect("append");
    log.append(b"abcdefghij\n").expect("append");
    drop(log);
    assert_eq!(read(&path), "abcdefghij\n");
    assert_eq!(log_files(dir.path()), ["api.log"]);
}

#[test]
fn shared_writers_for_one_path_use_one_handle_across_rotation() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("candle-worker.log");
    let first = shared_log_file_with_policy(&path, small_policy(30, 1));
    let second = shared_log_file_with_policy(&path, small_policy(9999, 9));
    assert!(Arc::ptr_eq(&first, &second), "one writer per path");
    for index in 0..6 {
        let writer = if index % 2 == 0 { &first } else { &second };
        writer
            .append(format!("entry {index}\n").as_bytes())
            .expect("append");
    }
    // Both writers rotated through the same handle: the backup did not keep growing.
    assert!(read(&dir.path().join("candle-worker.1.log")).len() <= 30);
    assert!(read(&path).ends_with("entry 5\n"));
    assert_eq!(log_files(dir.path()).len(), 2);
}

/// Windows: a reader that opened the live log WITHOUT `FILE_SHARE_DELETE` (as some log viewers
/// do) makes the rename fail. Rotation must still bound the file by copying the tail and
/// truncating in place, without losing the recent lines.
#[cfg(windows)]
#[test]
fn rotation_falls_back_to_copy_and_truncate_when_another_handle_blocks_rename() {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_SHARE_READ: u32 = 0x1;
    const FILE_SHARE_WRITE: u32 = 0x2;

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("api.log");
    let mut log = RotatingLogFile::open(&path, small_policy(50, 2));
    log.append(b"first 0001\nfirst 0002\nfirst 0003\n")
        .expect("append");
    let viewer = OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .open(&path)
        .expect("viewer opens without delete sharing");
    assert!(
        std::fs::rename(&path, dir.path().join("probe.log")).is_err(),
        "precondition: the viewer handle blocks renaming"
    );

    log.append(b"second 0001\nsecond 0002\n").expect("append");
    drop(log);
    drop(viewer);

    assert_eq!(read(&path), "second 0001\nsecond 0002\n");
    assert_eq!(
        read(&dir.path().join("api.1.log")),
        "first 0001\nfirst 0002\nfirst 0003\n"
    );
}
