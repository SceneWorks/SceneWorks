use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn materialization_record(expected_hash: Option<&str>) -> CatalogRecord {
    CatalogRecord {
        id: "record".to_owned(),
        image_path: "images/record.png".to_owned(),
        thumbnail_path: None,
        embedding_path: None,
        artifact_path: None,
        metadata: serde_json::json!({
            "acquisition": {"contentHash": expected_hash}
        }),
        created_at: "2026-07-27T00:00:00Z".to_owned(),
        updated_at: "2026-07-27T00:00:00Z".to_owned(),
    }
}

#[test]
fn remote_fallback_must_match_the_catalog_acquisition_hash() {
    assert!(
        verify_record_content_hash(&materialization_record(Some("expected")), "changed").is_err()
    );
    assert!(
        verify_record_content_hash(&materialization_record(Some("expected")), "expected").is_ok()
    );
    assert!(verify_record_content_hash(&materialization_record(None), "downloaded").is_ok());
}

#[tokio::test(flavor = "current_thread")]
async fn global_scan_admission_bounds_distinct_catalog_work_and_preserves_db_capacity() {
    let slots = Arc::new(Semaphore::new(2));
    let release = Arc::new(AtomicBool::new(false));
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let cancellation = CancellationToken::new();
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let slots = slots.clone();
        let release = release.clone();
        let active = active.clone();
        let peak = peak.clone();
        let cancellation = cancellation.clone();
        tasks.push(tokio::spawn(async move {
            catalog_scan_blocking(slots, &cancellation, move || {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                while !release.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                active.fetch_sub(1, Ordering::SeqCst);
            })
            .await
            .expect("blocking task joins")
        }));
    }
    tokio::time::timeout(Duration::from_secs(1), async {
        while active.load(Ordering::SeqCst) != 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("two scan slots fill");
    assert_eq!(peak.load(Ordering::SeqCst), 2);
    assert_eq!(slots.available_permits(), 0);

    let status_result = tokio::time::timeout(
        Duration::from_millis(100),
        tokio::task::spawn_blocking(|| 42_u64),
    )
    .await
    .expect("unrelated catalog DB work retains blocking-pool capacity")
    .expect("status task joins");
    assert_eq!(status_result, 42);

    release.store(true, Ordering::SeqCst);
    for task in tasks {
        assert!(task.await.expect("admission task joins").is_some());
    }
    assert_eq!(peak.load(Ordering::SeqCst), 2);
}
