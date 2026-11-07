use super::*;
#[tokio::test]
async fn prune_keeps_snapshot_and_blob_payloads_but_removes_empty_skeletons() {
    for kind in ["snapshot", "blob", "empty"] {
        let temporary = tempfile::tempdir().unwrap();
        let repository = temporary.path().join("models--fixture--repo");
        std::fs::create_dir_all(repository.join("snapshots/revision/empty")).unwrap();
        std::fs::create_dir_all(repository.join("blobs")).unwrap();
        std::fs::create_dir_all(repository.join("refs")).unwrap();
        std::fs::write(repository.join("refs/main"), b"revision").unwrap();
        let payload = match kind {
            "snapshot" => repository.join("snapshots/revision/payload.bin"),
            _ => repository.join("blobs/payload.bin"),
        };
        if kind != "empty" {
            std::fs::write(&payload, b"fixture-payload").unwrap();
        }
        prune_empty_repo_cache(&repository).await;
        if kind == "empty" {
            assert!(!repository.exists());
        } else {
            assert_eq!(std::fs::read(payload).unwrap(), b"fixture-payload");
        }
    }
}
#[tokio::test]
async fn prune_refuses_a_path_outside_the_repository_namespace() {
    let temporary = tempfile::tempdir().unwrap();
    let directory = temporary.path().join("user-data");
    std::fs::create_dir_all(directory.join("snapshots/empty")).unwrap();
    std::fs::write(directory.join("keep.txt"), b"keep").unwrap();
    prune_empty_repo_cache(&directory).await;
    assert_eq!(std::fs::read(directory.join("keep.txt")).unwrap(), b"keep");
    assert!(directory.join("snapshots/empty").is_dir());
}
