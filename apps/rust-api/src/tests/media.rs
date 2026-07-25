//! rust-api media tests (split from tests.rs, sc-11217 F-030).
use super::support::*;
use crate::publish_queue;
use sceneworks_core::{contracts::JobType, jobs_store::CreateJob};
use std::sync::Arc;

#[tokio::test]
async fn project_file_route_serves_files_and_rejects_traversal() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Files" }),
    )
    .await;
    let project_id = created["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(created["path"].as_str().unwrap());
    let media_path = project_path.join("assets/images/image.png");
    std::fs::write(&media_path, b"image-bytes").expect("media writes");
    let outside_path = temp_dir.path().join("data").join("outside.txt");
    std::fs::write(outside_path, b"nope").expect("outside writes");

    let (status, headers, bytes) = request_raw(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/files/assets/images/image.png"),
        Body::empty(),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(bytes, b"image-bytes");
    assert_eq!(
        headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("image/png")
    );
    // sc-9674 (sc-8872 follow-up): the serve response forbids MIME sniffing so a
    // user-controlled project file can't be reinterpreted as active content.
    assert_eq!(
        headers
            .get("x-content-type-options")
            .and_then(|value| value.to_str().ok()),
        Some("nosniff")
    );

    let (status, _, bytes) = request_raw(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/files/%2E%2E%2F%2E%2E%2Foutside.txt"),
        Body::empty(),
        &[],
    )
    .await;
    let error: Value = serde_json::from_slice(&bytes).expect("json error parses");
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "Invalid project file path");

    let (status, _, bytes) = request_raw(
        app,
        "GET",
        &format!("/api/v1/projects/{project_id}/files/%2E%2E%5C%2E%2E%5Coutside.txt"),
        Body::empty(),
        &[],
    )
    .await;
    let error: Value = serde_json::from_slice(&bytes).expect("json error parses");
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "Invalid project file path");
}

#[tokio::test]
async fn project_file_route_serves_byte_ranges() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Ranges" }),
    )
    .await;
    let project_id = created["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(created["path"].as_str().unwrap());
    let media_path = project_path.join("assets/videos/clip.mp4");
    std::fs::write(&media_path, b"0123456789").expect("media writes");
    let uri = format!("/api/v1/projects/{project_id}/files/assets/videos/clip.mp4");

    // A full request advertises range support so WebKit knows it can seek.
    let (status, headers, bytes) = request_raw(app.clone(), "GET", &uri, Body::empty(), &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(bytes, b"0123456789");
    assert_eq!(
        headers.get("accept-ranges").and_then(|v| v.to_str().ok()),
        Some("bytes")
    );

    // A bounded range yields 206 with the exact slice and Content-Range.
    let (status, headers, bytes) = request_raw(
        app.clone(),
        "GET",
        &uri,
        Body::empty(),
        &[("range", "bytes=2-5")],
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(bytes, b"2345");
    assert_eq!(
        headers.get("content-range").and_then(|v| v.to_str().ok()),
        Some("bytes 2-5/10")
    );
    assert_eq!(
        headers.get("accept-ranges").and_then(|v| v.to_str().ok()),
        Some("bytes")
    );
    // sc-9674: the 206 partial-content branch also carries nosniff.
    assert_eq!(
        headers
            .get("x-content-type-options")
            .and_then(|v| v.to_str().ok()),
        Some("nosniff")
    );

    // An open-ended range serves to EOF (this is how WebKit fetches the
    // trailing moov atom on a non-faststart MP4).
    let (status, _, bytes) = request_raw(
        app.clone(),
        "GET",
        &uri,
        Body::empty(),
        &[("range", "bytes=7-")],
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(bytes, b"789");

    // An unsatisfiable range is rejected with 416.
    let (status, _, _) =
        request_raw(app, "GET", &uri, Body::empty(), &[("range", "bytes=99-")]).await;
    assert_eq!(status, StatusCode::RANGE_NOT_SATISFIABLE);
}

#[tokio::test]
async fn event_tickets_are_protected_and_match_contract_shape() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    let (status, error) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(error["detail"], "SceneWorks access token required");

    let (status, ticket) = request_with_headers(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        json!({ "activeJobIds": ["job-a", "job-a", "job-b"] }),
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(ticket["ticket"]
        .as_str()
        .is_some_and(|value| value.len() == 32 && value.chars().all(|c| c.is_ascii_hexdigit())));
    assert_eq!(ticket["expiresInSeconds"], 30);

    let (status, error) = request_with_headers(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        json!({ "activeJobIds": [""] }),
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        error["detail"],
        "activeJobIds must contain non-empty job ids"
    );

    let (status, error) = request(
        app,
        "GET",
        "/api/v1/jobs/events?ticket=missing",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(error["detail"], "Invalid or expired event stream ticket");
}

#[tokio::test]
async fn event_ticket_mint_preserves_the_complete_reconnect_context() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");
    let active_job_ids = (0..600)
        .map(|index| format!("00000000-0000-4000-8000-{index:012}"))
        .collect::<Vec<_>>();
    let terminal_job_ids = (0..200)
        .map(|index| format!("10000000-0000-4000-8000-{index:012}"))
        .collect::<Vec<_>>();

    let (status, response) = request(
        app,
        "POST",
        "/api/v1/jobs/events/ticket",
        json!({
            "activeJobIds": active_job_ids.clone(),
            "knownTerminalJobIds": terminal_job_ids.clone(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let ticket = response["ticket"].as_str().expect("ticket");
    assert_eq!(
        state
            .event_tickets
            .consume_event(ticket)
            .expect("ticket redeems"),
        crate::tickets::EventTicketContext {
            active_job_ids,
            known_terminal_job_ids: terminal_job_ids,
        },
        "ticket mint must preserve every bounded reconnect id without putting it in the GET URL"
    );
}

#[tokio::test]
async fn event_ticket_context_rejects_each_route_specific_memory_bound() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let too_many_active = (0..=crate::MAX_EVENT_TICKET_ACTIVE_JOB_IDS)
        .map(|index| format!("active-{index}"))
        .collect::<Vec<_>>();
    let (status, error) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        json!({ "activeJobIds": too_many_active }),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        error["detail"],
        format!(
            "activeJobIds may contain at most {} job ids",
            crate::MAX_EVENT_TICKET_ACTIVE_JOB_IDS
        )
    );

    let too_many_terminal = (0..=crate::MAX_EVENT_TICKET_TERMINAL_JOB_IDS)
        .map(|index| format!("terminal-{index}"))
        .collect::<Vec<_>>();
    let (status, error) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        json!({ "knownTerminalJobIds": too_many_terminal }),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        error["detail"],
        format!(
            "knownTerminalJobIds may contain at most {} job ids",
            crate::MAX_EVENT_TICKET_TERMINAL_JOB_IDS
        )
    );

    let (status, error) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        json!({ "activeJobIds": ["x".repeat(crate::MAX_EVENT_TICKET_JOB_ID_BYTES + 1)] }),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        error["detail"],
        format!(
            "activeJobIds job ids may contain at most {} bytes",
            crate::MAX_EVENT_TICKET_JOB_ID_BYTES
        )
    );

    let aggregate = (0..crate::MAX_EVENT_TICKET_ACTIVE_JOB_IDS)
        .map(|index| format!("{index:04}-{}", "x".repeat(44)))
        .collect::<Vec<_>>();
    assert!(
        aggregate.iter().map(String::len).sum::<usize>() > crate::MAX_EVENT_TICKET_CONTEXT_BYTES
    );
    let (status, error) = request(
        app,
        "POST",
        "/api/v1/jobs/events/ticket",
        json!({ "activeJobIds": aggregate }),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        error["detail"],
        format!(
            "event ticket job-id context exceeds {} bytes",
            crate::MAX_EVENT_TICKET_CONTEXT_BYTES
        )
    );
}

#[tokio::test]
async fn event_ticket_route_has_a_small_body_limit_and_outstanding_backpressure() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let oversized = vec![b'a'; crate::MAX_EVENT_TICKET_BODY_BYTES + 1];
    let (status, _, _) = request_raw(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        oversized,
        &[("content-type", "application/json")],
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);

    for _ in 0..crate::MAX_OUTSTANDING_EVENT_TICKETS {
        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs/events/ticket",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }
    let (status, error) = request(app, "POST", "/api/v1/jobs/events/ticket", Value::Null).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        error["detail"],
        format!(
            "Too many outstanding event tickets; retry after at most {} seconds",
            crate::EVENT_TICKET_TTL_SECONDS
        )
    );
}

#[tokio::test]
async fn sse_event_ticket_is_single_use_at_the_endpoint() {
    // sc-8947 (F-146): the SSE ticket rides in the `?ticket=` query string because
    // EventSource can't set headers. The accepted control that bounds a leaked URL is
    // that the ticket is single-use (and short-TTL): the first `GET /jobs/events`
    // redeems it, a replay of the same ticket is rejected. This pins that invariant at
    // the HTTP layer (not just the ticket store) so nobody loosens the SSE gate.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    let (status, ticket) = request_with_headers(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        Value::Null,
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let ticket_value = ticket["ticket"].as_str().expect("ticket value").to_owned();

    // First redemption connects the stream (200 OK, then the SSE body streams — we
    // only read the status so the never-ending body doesn't hang the test).
    let status = request_status_only(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/events?ticket={ticket_value}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Replaying the same ticket is rejected — a leaked URL can't be reused.
    let (status, error) = request(
        app,
        "GET",
        &format!("/api/v1/jobs/events?ticket={ticket_value}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(error["detail"], "Invalid or expired event stream ticket");
}

#[tokio::test]
async fn sse_connection_starts_with_authoritative_queue_snapshot() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, events) = request_sse_prefix(app, "/api/v1/jobs/events", 3).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        events[0],
        ("ready".to_owned(), json!({ "status": "connected" }))
    );
    assert_eq!(events[1].0, "jobs.snapshot");
    assert!(
        events[1].1["jobs"]
            .as_array()
            .expect("jobs snapshot array")
            .iter()
            .any(|job| job["id"] == created["id"]),
        "the reconnect snapshot must include the authoritative job row"
    );
    assert_eq!(events[2].0, "queue.updated");
    assert_eq!(events[2].1["counts"]["queued"], 1);
    assert!(
        events[2].1["activeJobs"]
            .as_array()
            .expect("active jobs array")
            .iter()
            .any(|job| job["id"] == created["id"]),
        "the initial stream snapshot must include jobs created before this connection"
    );
}

#[tokio::test]
async fn sse_snapshot_precedes_a_concurrent_queue_publication() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = created["id"].as_str().expect("job id").to_owned();

    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    *state.sse_snapshot_before_subscribe_once.lock() = Some(barrier.clone());
    let stream = tokio::spawn(request_sse_prefix(app, "/api/v1/jobs/events", 4));
    barrier.wait().await;

    state
        .jobs_store
        .cancel_job(&job_id)
        .expect("concurrent terminal transition commits");
    let publish_state = state.clone();
    let publish = tokio::spawn(async move { publish_queue(&publish_state).await });
    barrier.wait().await;

    publish
        .await
        .expect("queue publisher joins")
        .expect("queue publication succeeds");
    let (status, events) = stream.await.expect("SSE request joins");
    assert_eq!(status, StatusCode::OK);
    assert_eq!(events[0].0, "ready");
    assert_eq!(events[1].0, "jobs.snapshot");
    assert_eq!(events[2].0, "queue.updated");
    assert_eq!(events[2].1["counts"]["queued"], 1);
    assert_eq!(events[3].0, "queue.updated");
    assert_eq!(events[3].1["counts"]["queued"], 0);
    assert_eq!(events[3].1["counts"]["canceled"], 1);
    assert!(
        events[3].1["activeJobs"]
            .as_array()
            .expect("active jobs array")
            .iter()
            .all(|job| job["id"] != json!(job_id)),
        "the concurrent live event must follow and supersede the older snapshot"
    );
}

#[tokio::test]
async fn sse_revision_barrier_orders_a_delayed_equal_timestamp_job_event_behind_snapshot() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let job_id = created["id"].as_str().expect("job id").to_owned();
    let mut old_job = state.jobs_store.get_job(&job_id).expect("old job reads");

    // Commit first, but retain the pre-commit row for a deliberately delayed
    // publication with an equal-second timestamp.
    let terminal = state
        .jobs_store
        .cancel_job(&job_id)
        .expect("terminal transition commits");
    old_job.updated_at = terminal.updated_at.clone();

    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    *state.sse_snapshot_before_subscribe_once.lock() = Some(barrier.clone());
    let stream = tokio::spawn(request_sse_prefix_with_ids(app, "/api/v1/jobs/events", 4));
    barrier.wait().await;

    // Publish only after the terminal snapshot and stream barrier were built.
    // The stale row is globally newer but durably older.
    state.events.publish(EventMessage {
        event: "job.updated".to_owned(),
        data: serde_json::to_string(&old_job).expect("old job serializes"),
        revision: 0,
    });
    barrier.wait().await;

    let (status, events) = stream.await.expect("SSE request joins");
    assert_eq!(status, StatusCode::OK);
    let snapshot_revision = events[1].2["revision"].as_u64().expect("snapshot revision");
    let snapshot_job = events[1].2["jobs"]
        .as_array()
        .expect("snapshot jobs")
        .iter()
        .find(|job| job["id"] == json!(job_id))
        .expect("snapshot contains job");
    assert_eq!(snapshot_job["status"], "canceled");
    assert_eq!(snapshot_job["updatedAt"], old_job.updated_at);
    assert!(
        snapshot_job["revision"]
            .as_i64()
            .expect("snapshot job revision")
            > old_job.extra["revision"]
                .as_i64()
                .expect("old job revision")
    );
    assert_eq!(events[3].0, "job.updated");
    assert_eq!(events[3].2["status"], "queued");
    assert_eq!(events[3].2["updatedAt"], old_job.updated_at);
    assert!(
        events[3].1.expect("buffered event revision") > snapshot_revision,
        "the delayed stale publication must reproduce a globally newer event"
    );
}

#[tokio::test]
async fn sse_snapshot_reconciles_only_requested_rows_beyond_the_recent_history_window() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.jobs_retention_days = 0;
    let (app, state) = create_app_with_state(settings).expect("app creates");
    let create = || CreateJob {
        job_type: JobType::ImageDetail,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: serde_json::Map::new(),
        requested_gpu: "auto".to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    };
    let target = state
        .jobs_store
        .create_job(create())
        .expect("target creates");
    let terminal = state
        .jobs_store
        .cancel_job(&target.id)
        .expect("target completes before disconnect churn");
    let cleared_active = state
        .jobs_store
        .create_job(create())
        .expect("active-then-cleared target creates");
    state
        .jobs_store
        .cancel_job(&cleared_active.id)
        .expect("active-then-cleared target completes");
    state
        .jobs_store
        .clear_job(&cleared_active.id)
        .expect("active-then-cleared target clears");
    let cleared_terminal = state
        .jobs_store
        .create_job(create())
        .expect("known-terminal target creates");
    state
        .jobs_store
        .cancel_job(&cleared_terminal.id)
        .expect("known-terminal target completes");
    state
        .jobs_store
        .clear_job(&cleared_terminal.id)
        .expect("known-terminal target clears");
    let unrelated_cleared = state
        .jobs_store
        .create_job(create())
        .expect("unrelated cleared target creates");
    state
        .jobs_store
        .cancel_job(&unrelated_cleared.id)
        .expect("unrelated cleared target completes");
    state
        .jobs_store
        .clear_job(&unrelated_cleared.id)
        .expect("unrelated cleared target clears");

    // updated_at is second-granularity. Cross a boundary before producing more
    // than the bounded history window so the target is deterministically absent
    // from that generic window.
    tokio::time::sleep(Duration::from_secs(1)).await;
    for _ in 0..501 {
        state
            .jobs_store
            .create_job(create())
            .expect("newer filler creates");
    }
    assert!(
        state
            .jobs_store
            .list_jobs_recently_updated(500)
            .expect("bounded history reads")
            .iter()
            .all(|job| job.id != target.id),
        "the locally active target must reproduce the high-volume reconnect gap"
    );

    let (status, ticket) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        json!({
            "activeJobIds": [target.id, cleared_active.id],
            "knownTerminalJobIds": [cleared_terminal.id],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let ticket = ticket["ticket"].as_str().expect("ticket").to_owned();
    let uri = format!("/api/v1/jobs/events?ticket={ticket}");
    let (status, events) = request_sse_prefix(app, &uri, 3).await;
    assert_eq!(status, StatusCode::OK);
    let reconciled = events[1].1["jobs"]
        .as_array()
        .expect("snapshot jobs")
        .iter()
        .find(|job| job["id"] == json!(target.id))
        .expect("requested pre-disconnect active job is reconciled");
    assert_eq!(reconciled["status"], terminal.status.as_str());
    assert_eq!(reconciled["revision"], terminal.extra["revision"]);
    assert_eq!(
        events[1].1["clearedJobIds"],
        json!([cleared_active.id, cleared_terminal.id]),
        "reconnect returns only tombstones intersecting the bounded client context"
    );
    assert!(
        events[1].1["clearedJobIds"]
            .as_array()
            .expect("cleared ids")
            .iter()
            .all(|id| id != &json!(unrelated_cleared.id)),
        "full retained clear history is never exported"
    );
}

#[tokio::test]
async fn media_tickets_authenticate_project_file_urls() {
    // sc-8810: element-driven media requests (<img>/<video>/<a download>) cannot
    // attach the token header, so the files route honors a short-lived query-param
    // ticket minted by an authenticated client — mirroring the SSE ticket.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");
    let auth = [("x-sceneworks-token", "secret-token")];

    let (_, created) = request_with_headers(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Ticketed media" }),
        &auth,
    )
    .await;
    let project_id = created["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(created["path"].as_str().unwrap());
    std::fs::write(project_path.join("assets/images/image.png"), b"image-bytes")
        .expect("media writes");
    let file_uri = format!("/api/v1/projects/{project_id}/files/assets/images/image.png");

    // Minting a media ticket itself requires authentication.
    let (status, _) = request(app.clone(), "POST", "/api/v1/files/ticket", Value::Null).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, ticket) = request_with_headers(
        app.clone(),
        "POST",
        "/api/v1/files/ticket",
        Value::Null,
        &auth,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let ticket_value = ticket["ticket"].as_str().expect("ticket string").to_owned();
    assert!(ticket_value.len() == 32 && ticket_value.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(ticket["expiresInSeconds"], 300);

    // Re-minting while the ticket is alive returns the SAME sliding ticket, so
    // already-rendered media URLs stay stable across client refreshes.
    let (_, reissued) = request_with_headers(
        app.clone(),
        "POST",
        "/api/v1/files/ticket",
        Value::Null,
        &auth,
    )
    .await;
    assert_eq!(reissued["ticket"], ticket_value.as_str());

    // Bare media URL (what an <img src> sends): still 401 without a ticket.
    let (status, _) = request(app.clone(), "GET", &file_uri, Value::Null).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // With the ticket: authorized, and multi-use (a page renders many thumbnails
    // and <video> issues multiple Range requests against one URL).
    for _ in 0..2 {
        let (status, _, bytes) = request_raw(
            app.clone(),
            "GET",
            &format!("{file_uri}?ticket={ticket_value}"),
            Body::empty(),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(bytes, b"image-bytes");
    }

    // A garbage ticket stays locked out.
    let (status, _) = request(
        app.clone(),
        "GET",
        &format!("{file_uri}?ticket=not-a-ticket"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Scope isolation: an SSE event ticket is NOT valid on the files route…
    let (_, event_ticket) = request_with_headers(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        Value::Null,
        &auth,
    )
    .await;
    let event_ticket_value = event_ticket["ticket"].as_str().expect("event ticket");
    let (status, _) = request(
        app.clone(),
        "GET",
        &format!("{file_uri}?ticket={event_ticket_value}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // …and a media ticket is NOT valid on the SSE stream…
    let (status, _) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/events?ticket={ticket_value}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // …and a media ticket never unlocks any non-media route.
    let (status, _) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs?ticket={ticket_value}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects?ticket={ticket_value}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    // A files-shaped path with a non-GET method stays locked too.
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("{file_uri}?ticket={ticket_value}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Pose previews are the other element-driven media family: the ticket clears
    // auth (the 404 is the handler's own missing-file answer, not a 401).
    let (status, _) = request(
        app.clone(),
        "GET",
        "/api/v1/poses/preview/job_missing/preview.png",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = request(
        app,
        "GET",
        &format!("/api/v1/poses/preview/job_missing/preview.png?ticket={ticket_value}"),
        Value::Null,
    )
    .await;
    assert_ne!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn pose_preview_route_sets_nosniff() {
    // sc-9674 (sc-8872 follow-up): the pose-preview serve endpoint is a sibling
    // media route on the API origin, so it must also forbid MIME sniffing. Served
    // inline for <img> preview, so no attachment disposition.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");

    // The handler reads the rendered skeleton from the pose-detect cache; write one.
    let preview_dir = data_dir.join("cache").join("pose_detect").join("job_ok");
    std::fs::create_dir_all(&preview_dir).expect("preview dir creates");
    std::fs::write(preview_dir.join("preview.png"), PNG_32X32).expect("preview writes");

    let (status, headers, bytes) = request_raw(
        app,
        "GET",
        "/api/v1/poses/preview/job_ok/preview.png",
        Body::empty(),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(bytes, PNG_32X32);
    assert_eq!(
        headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("image/png")
    );
    assert_eq!(
        headers
            .get("x-content-type-options")
            .and_then(|value| value.to_str().ok()),
        Some("nosniff")
    );
}

#[test]
fn media_ticket_paths_cover_exactly_the_media_routes() {
    use crate::auth::is_ticketed_media_path;
    // Project files + pose previews are ticketed…
    assert!(is_ticketed_media_path(
        "/api/v1/projects/p1/files/assets/images/one.png"
    ));
    assert!(is_ticketed_media_path("/api/v1/projects/p1/files/a"));
    assert!(is_ticketed_media_path("/api/v1/poses/preview/job_1/p.png"));
    // …nothing else is.
    assert!(!is_ticketed_media_path("/api/v1/projects"));
    assert!(!is_ticketed_media_path("/api/v1/projects/p1"));
    assert!(!is_ticketed_media_path("/api/v1/projects/p1/files"));
    assert!(!is_ticketed_media_path("/api/v1/projects/p1/files/"));
    assert!(!is_ticketed_media_path("/api/v1/projects/p1/assets"));
    assert!(!is_ticketed_media_path("/api/v1/projects//files/a"));
    assert!(!is_ticketed_media_path("/api/v1/poses/preview/"));
    assert!(!is_ticketed_media_path("/api/v1/jobs"));
    assert!(!is_ticketed_media_path("/api/v1/credentials"));
}

#[test]
fn ticket_store_sliding_reuse_and_expiry() {
    use crate::tickets::{EventTicketContext, TicketStore};
    // Sliding (media) tickets: reusable, stable across re-issue, non-consuming.
    let store = TicketStore::new(300);
    let first = store.issue_sliding();
    let second = store.issue_sliding();
    assert_eq!(first.ticket, second.ticket, "live sliding ticket is reused");
    assert!(store.validate(&first.ticket));
    assert!(store.validate(&first.ticket), "validate must not consume");
    assert!(!store.validate("bogus"));
    assert!(!store.validate(""));

    // Single-use (SSE) tickets: consume removes them.
    let sse = store.issue();
    assert_eq!(
        store.consume_event(&sse.ticket),
        Some(EventTicketContext::default())
    );
    assert_eq!(
        store.consume_event(&sse.ticket),
        None,
        "consume is single-use"
    );

    let active_ids = (0..600)
        .map(|index| format!("job-{index}"))
        .collect::<Vec<_>>();
    let context = EventTicketContext {
        active_job_ids: active_ids,
        known_terminal_job_ids: vec!["terminal-1".to_owned()],
    };
    let contextual = store
        .try_issue_event(context.clone())
        .expect("unbounded store issues");
    assert_eq!(
        store.consume_event(&contextual.ticket),
        Some(context),
        "the single-use ticket must preserve the complete reconnect set without a cap"
    );
    assert_eq!(
        store.consume_event(&contextual.ticket),
        None,
        "context redemption remains single-use"
    );

    // TTL 0: expired as soon as any time passes (the sleep guards against two
    // Instant::now() calls landing on the same tick).
    let expired = TicketStore::new(0);
    let sliding = expired.issue_sliding();
    let single = expired.issue();
    std::thread::sleep(Duration::from_millis(5));
    assert!(!expired.validate(&sliding.ticket));
    assert_eq!(expired.consume_event(&single.ticket), None);

    // Expired unredeemed event tickets are pruned before the outstanding cap is
    // enforced, so backpressure never becomes a permanent lockout.
    let bounded = TicketStore::with_max_outstanding(0, 1);
    let first = bounded
        .try_issue_event(EventTicketContext::default())
        .expect("first bounded ticket issues");
    std::thread::sleep(Duration::from_millis(5));
    let second = bounded
        .try_issue_event(EventTicketContext::default())
        .expect("expired ticket frees bounded capacity");
    assert_ne!(first.ticket, second.ticket);
    assert_eq!(bounded.consume_event(&first.ticket), None);
    assert_eq!(bounded.consume_event(&second.ticket), None);
}

#[tokio::test]
async fn lagged_event_subscribers_are_disconnected() {
    let hub = EventHub::default();
    let mut stream = hub.subscribe();

    for index in 0..EVENT_BUFFER_SIZE {
        hub.publish(EventMessage {
            event: "job.updated".to_owned(),
            data: json!({ "index": index }).to_string(),
            revision: 0,
        });
    }
    hub.publish(EventMessage {
        event: "job.updated".to_owned(),
        data: json!({ "index": EVENT_BUFFER_SIZE }).to_string(),
        revision: 0,
    });

    for _ in 0..EVENT_BUFFER_SIZE {
        assert!(stream.next().await.is_some());
    }
    assert!(stream.next().await.is_none());
}

#[test]
fn heartbeat_event_matches_contract_wire_shape() {
    assert_eq!(HEARTBEAT_SSE_DATA, "{}");
    assert_eq!(HEARTBEAT_SSE_WIRE, "event: heartbeat\ndata: {}\n\n");
}

/// sc-6539: the synchronous smart-crop + EXIF-strip endpoints rewrite an item's pixels and re-point
/// it in one round-trip — the response carries the updated dataset (immediate UI refresh).
#[tokio::test]
async fn smart_crop_and_strip_exif_rewrite_and_repoint_items() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Crop Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // A 64×16 PNG: crop-loss (64-16)/64 = 0.75, well over the 0.35 flag.
    let mut wide = image::RgbImage::new(64, 16);
    for (x, _, pixel) in wide.enumerate_pixels_mut() {
        *pixel = image::Rgb([(x * 4) as u8, 80, 160]);
    }
    let mut buffer = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(wide)
        .write_to(&mut buffer, image::ImageFormat::Png)
        .expect("encode png");
    let png = buffer.into_inner();

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "wide.png",
        "image/png",
        &png,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "wide set", "items": [{ "assetId": asset_id }] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    assert_eq!(dataset["items"][0]["width"], 64);
    assert_eq!(dataset["items"][0]["height"], 16);

    // Smart-crop the wide item: short edge kept, long edge trimmed below the flag, version bumped.
    let (status, cropped) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/smart-crop"),
        json!({ "itemIds": ["item_0001"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cropped["applied"], 1);
    let item = &cropped["dataset"]["items"][0];
    assert_eq!(item["height"], 16, "short edge kept in full");
    let new_w = item["width"].as_u64().expect("width");
    assert!(new_w < 64, "long edge trimmed (was 64, now {new_w})");
    let after = (new_w as f64 - 16.0) / new_w as f64;
    assert!(after < 0.35, "crop-loss cleared the flag (now {after})");
    assert_eq!(cropped["dataset"]["version"], 2, "version bumped");

    // Strip EXIF from all items (none named) — re-encodes, version bumps again.
    let (status, stripped) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/strip-exif"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(stripped["applied"], 1);
    assert_eq!(stripped["dataset"]["version"], 3);
}
