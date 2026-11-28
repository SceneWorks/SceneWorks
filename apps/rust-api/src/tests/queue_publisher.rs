//! Coalesced `queue.updated` publication (`crate::queue_publisher`). These tests turn
//! off the route-test flush (`TestHooks::defer_queue_publishes`) to observe the
//! production behavior, and never sleep: the publisher is held off by holding
//! `queue_snapshot_lock`, and observed through `flush_queue_publishes`.
use super::support::*;
use crate::{flush_queue_publishes, request_queue_publish};
use futures_util::FutureExt;
use std::sync::atomic::Ordering;
use tokio_stream::wrappers::ReceiverStream;

/// Every event already delivered to `events`, without waiting for more.
fn buffered(events: &mut ReceiverStream<EventMessage>) -> Vec<EventMessage> {
    let mut out = Vec::new();
    while let Some(Some(message)) = events.next().now_or_never() {
        out.push(message);
    }
    out
}

fn queue_updates(events: &[EventMessage]) -> Vec<Value> {
    events
        .iter()
        .filter(|message| message.event == "queue.updated")
        .map(|message| serde_json::from_str(&message.data).expect("queue event parses"))
        .collect()
}

async fn create_detail_job(app: axum::Router, prompt: &str) -> String {
    let (status, created) = request(
        app,
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": prompt },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    created["id"].as_str().expect("job id").to_owned()
}

#[tokio::test]
async fn mutations_respond_before_the_rebuild_and_coalesce_into_the_final_state() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");
    state
        .test_hooks
        .defer_queue_publishes
        .store(true, Ordering::SeqCst);
    let mut events = state.events.subscribe();

    // Hold the snapshot lock the publisher rebuilds under. Before coalescing every
    // mutation rebuilt inline under this lock, so these requests would never return.
    let snapshot_guard = state.queue_snapshot_lock.lock().await;
    let mut job_ids = Vec::new();
    for index in 0..6 {
        let job_id = tokio::time::timeout(
            Duration::from_secs(60),
            create_detail_job(app.clone(), &format!("prompt {index}")),
        )
        .await
        .expect("a mutation responds without waiting for the queue rebuild");
        job_ids.push(job_id);
    }
    let (status, canceled) = tokio::time::timeout(
        Duration::from_secs(60),
        request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{}/cancel", job_ids[0]),
            json!({}),
        ),
    )
    .await
    .expect("cancel responds without waiting for the queue rebuild");
    assert_eq!(status, StatusCode::OK, "{canceled}");
    let early = buffered(&mut events);
    assert_eq!(
        early
            .iter()
            .filter(|message| message.event == "job.updated")
            .count(),
        7,
        "per-job events still go out with each request"
    );
    assert!(
        queue_updates(&early).is_empty(),
        "no queue rebuild ran while the snapshot lock was held: {:?}",
        early.iter().map(|m| &m.event).collect::<Vec<_>>()
    );

    drop(snapshot_guard);
    flush_queue_publishes(&state)
        .await
        .expect("coalesced publish succeeds");
    let updates = queue_updates(&buffered(&mut events));
    // The round blocked on the lock had read the request count at some point during
    // the seven requests; at most one more round covers the rest. Inline publishing
    // sent seven.
    assert!(
        (1..=2).contains(&updates.len()),
        "seven mutations coalesce into at most two publishes, got {}",
        updates.len()
    );
    let last = updates.last().expect("a queue update");
    assert_eq!(last["counts"]["queued"], 5, "{last}");
    assert_eq!(last["counts"]["canceled"], 1, "{last}");
    let active = last["activeJobs"]
        .as_array()
        .expect("active jobs")
        .iter()
        .map(|job| job["id"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    assert_eq!(active.len(), 5);
    assert!(!active.contains(&job_ids[0]));

    // Nothing pending: a flush returns at once and publishes nothing more.
    flush_queue_publishes(&state).await.expect("idle flush");
    assert!(queue_updates(&buffered(&mut events)).is_empty());
}

#[tokio::test]
async fn flush_publishes_a_pending_request_without_waiting_out_the_interval() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");
    state
        .test_hooks
        .defer_queue_publishes
        .store(true, Ordering::SeqCst);
    let job_id = create_detail_job(app.clone(), "first").await;
    flush_queue_publishes(&state).await.expect("first flush");
    let mut events = state.events.subscribe();

    // The previous round just started, so this request falls inside the coalescing
    // interval; the flush skips the rest of it.
    state
        .jobs_store
        .cancel_job(&job_id)
        .expect("out-of-band transition commits");
    request_queue_publish(&state);
    tokio::time::timeout(Duration::from_secs(60), flush_queue_publishes(&state))
        .await
        .expect("flush completes")
        .expect("publish succeeds");
    let updates = queue_updates(&buffered(&mut events));
    assert_eq!(updates.len(), 1, "{updates:?}");
    assert_eq!(updates[0]["counts"]["queued"], 0);
    assert_eq!(updates[0]["counts"]["canceled"], 1);
}

#[tokio::test]
async fn route_tests_see_the_publish_as_part_of_the_request() {
    // The default test harness (no defer hook) flushes after each request, so existing
    // route tests keep observing `queue.updated` before the response returns.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");
    let mut events = state.events.subscribe();
    create_detail_job(app, "flushed").await;
    let updates = queue_updates(&buffered(&mut events));
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0]["counts"]["queued"], 1);
}
