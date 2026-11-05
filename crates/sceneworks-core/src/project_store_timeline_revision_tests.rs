use super::*;
#[test]
fn timeline_cas_rejects_stale_saves_and_has_one_concurrent_winner() {
    let temp = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store.create_project("Concurrent cut").unwrap();
    let cut = store
        .create_timeline(&project.id, "Cut", "16:9", 24)
        .unwrap();
    assert_eq!(cut["revision"], 1);
    let barrier = std::sync::Barrier::new(2);
    let outcomes = std::thread::scope(|scope| {
        let run = || {
            barrier.wait();
            store.save_timeline(&project.id, cut.clone())
        };
        let first = scope.spawn(run);
        let second = scope.spawn(run);
        [first.join().unwrap(), second.join().unwrap()]
    });
    assert_eq!(outcomes.iter().filter(|r| r.is_ok()).count(), 1);
    assert!(outcomes.iter().any(|r| matches!(
        r,
        Err(ProjectStoreError::TimelineConflict {
            code: "timeline_revision_conflict",
            ..
        })
    )));
    let id = cut["id"].as_str().unwrap();
    let file = store.timeline_file(&project.id, id).unwrap();
    let bytes = fs::read(&file.path).unwrap();
    assert!(store.save_timeline(&project.id, cut).is_err());
    assert_eq!(bytes, fs::read(file.path).unwrap());
}
#[test]
fn legacy_revision_zero_migrates_once_and_export_snapshot_is_immutable() {
    let temp = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store.create_project("Legacy cut").unwrap();
    let mut cut = store
        .create_timeline(&project.id, "Cut", "16:9", 24)
        .unwrap();
    let id = cut["id"].as_str().unwrap().to_owned();
    cut.as_object_mut().unwrap().remove("revision");
    let file = store.timeline_file(&project.id, &id).unwrap();
    write_json(&file.path, &cut).unwrap();
    let saved = store.save_timeline(&project.id, cut.clone()).unwrap();
    assert_eq!(saved["revision"], 1);
    assert!(store.save_timeline(&project.id, cut).is_err());
    let snapshot = store.snapshot_timeline_export(&project.id, &id).unwrap();
    let mut changed = saved;
    changed["name"] = json!("Edited");
    store.save_timeline(&project.id, changed).unwrap();
    assert_eq!(read_json(&snapshot.file.path).unwrap()["name"], "Cut");
    assert_eq!(snapshot.document["revision"], 1);
}
