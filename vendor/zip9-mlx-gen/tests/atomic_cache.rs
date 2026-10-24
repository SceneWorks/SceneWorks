#[path = "support/atomic_cache.rs"]
#[allow(clippy::module_inception)] // the test file and the support module share the name
mod atomic_cache;

#[test]
fn staging_is_process_scoped_and_publish_keeps_the_shared_name_stable() {
    let root_tmp = tempfile::tempdir().unwrap();
    let root = root_tmp.path().to_path_buf();
    let final_path = root.join("shared-cache");
    let staging = atomic_cache::prepare_staging(&final_path).expect("prepare staging");

    assert_eq!(final_path.file_name().unwrap(), "shared-cache");
    assert_eq!(
        staging.file_name().unwrap(),
        format!("shared-cache.tmp.{}", std::process::id()).as_str()
    );
    assert!(
        !final_path.exists(),
        "staging must not expose the final cache"
    );

    std::fs::create_dir_all(&staging).expect("create staged tree");
    std::fs::write(staging.join("complete"), b"complete cache").expect("write staged tree");
    atomic_cache::publish(&staging, &final_path).expect("publish cache");

    assert_eq!(
        std::fs::read(final_path.join("complete")).expect("read published cache"),
        b"complete cache"
    );
    assert!(!staging.exists(), "rename must consume the staging path");
}

#[test]
fn staging_preserves_a_file_extension_required_by_the_writer() {
    let final_path_tmp = tempfile::tempdir().unwrap();
    let final_path = final_path_tmp.path().join("shared-cache.safetensors");

    assert_eq!(
        atomic_cache::staging_path(&final_path).file_name().unwrap(),
        format!("shared-cache.tmp.{}.safetensors", std::process::id()).as_str()
    );
}

#[test]
fn losing_directory_publisher_reuses_the_complete_winner() {
    let root_tmp = tempfile::tempdir().unwrap();
    let root = root_tmp.path().to_path_buf();
    let final_path = root.join("shared-cache");
    let staging = atomic_cache::prepare_staging(&final_path).expect("prepare staging");
    std::fs::create_dir_all(&final_path).expect("create winning cache");
    std::fs::write(final_path.join("winner"), b"winner").expect("write winning cache");
    std::fs::create_dir_all(&staging).expect("create losing cache");
    std::fs::write(staging.join("loser"), b"loser").expect("write losing cache");

    atomic_cache::publish(&staging, &final_path).expect("reuse winner");

    assert_eq!(
        std::fs::read(final_path.join("winner")).expect("read winner"),
        b"winner"
    );
    assert!(!staging.exists(), "losing staging tree must be removed");
}

#[test]
fn symlink_is_not_visible_until_publish() {
    let root_tmp = tempfile::tempdir().unwrap();
    let root = root_tmp.path().to_path_buf();
    let source = root.join("source");
    let final_path = root.join("shared-link");
    std::fs::create_dir_all(&source).expect("create symlink source");

    atomic_cache::symlink_or_reuse(&source, &final_path).expect("publish symlink");

    assert_eq!(
        std::fs::read_link(&final_path).expect("read published symlink"),
        source
    );
    assert!(!atomic_cache::staging_path(&final_path).exists());
}
