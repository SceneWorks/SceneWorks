//! Shared real HTTP routes for worker contract tests. No model or learned computation.
use axum::{
    extract::{Path, State},
    handler::Handler,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};

/// Shared routes; caller state owns snapshot/cancel policy and progress observations.
pub(crate) fn routes<S: Clone + Send + Sync + 'static>(
    snapshot: fn(&S, &str, bool) -> Value,
    progress: fn(&S, &str, Value),
    heartbeat: fn(&S, &str) -> Value,
) -> Router<S> {
    job_progress_heartbeat_routes(
        move |State(state): State<S>, Path(id): Path<String>| async move {
            Json(snapshot(&state, &id, false))
        },
        move |State(state): State<S>, Path(id): Path<String>, Json(body): Json<Value>| async move {
            progress(&state, &id, body);
            Json(snapshot(&state, &id, true))
        },
        move |State(state): State<S>, Path(id): Path<String>| async move {
            Json(heartbeat(&state, &id))
        },
    )
    .route(
        "/api/v1/jobs/:job_id/metrics",
        post(|Json(body): Json<Value>| async move { Json(body) }),
    )
}

pub(crate) async fn serve(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("stub listener binds");
    let address = listener.local_addr().expect("stub has address");
    crate::load_progress::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

#[cfg(native_lane)]
pub(crate) type ProgressPosts = std::sync::Arc<std::sync::Mutex<Vec<(String, Value)>>>;

#[cfg(native_lane)]
pub(crate) fn smoke_job(
    job_id: &str,
    job_type: &str,
    project_id: Option<&str>,
    project_name: Option<&str>,
    worker_id: &str,
    stamp: &str,
) -> Value {
    json!({"id":job_id,"type":job_type,"status":"running","projectId":project_id,"projectName":project_name,"payload":{},"result":{},"requestedGpu":"auto","assignedGpu":"0","workerId":worker_id,"progress":0.1,"stage":"running","message":"running","error":null,"etaSeconds":null,"elapsedSeconds":null,"attempts":1,"sourceJobId":null,"duplicateOfJobId":null,"cancelRequested":false,"createdAt":stamp,"updatedAt":stamp,"startedAt":stamp,"completedAt":null,"canceledAt":null,"lastHeartbeatAt":stamp})
}

#[cfg(native_lane)]
pub(crate) fn smoke_worker(worker_id: &str, job_id: &str, stamp: &str) -> Value {
    json!({"id":worker_id,"gpuId":"0","gpuName":null,"status":"busy","currentJobId":job_id,"capabilities":[],"loadedModels":[],"registeredAt":stamp,"lastSeenAt":stamp})
}

#[cfg(native_lane)]
pub(crate) async fn smoke_api(
    job: Value,
    worker: Value,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> (String, ProgressPosts) {
    #[derive(Clone)]
    struct SmokeState {
        job: Value,
        worker: Value,
        posts: ProgressPosts,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    }
    let posts = ProgressPosts::default();
    let state = SmokeState {
        job,
        worker,
        posts: posts.clone(),
        cancel,
    };
    let app = routes(
        |state: &SmokeState, id, _| {
            let mut job = state.job.clone();
            job["id"] = json!(id);
            if let Some(cancel) = &state.cancel {
                job["cancelRequested"] = json!(cancel.load(std::sync::atomic::Ordering::SeqCst));
            }
            job
        },
        |state, id, body| {
            state
                .posts
                .lock()
                .expect("posts lock")
                .push((id.to_owned(), body));
        },
        |state, id| {
            let mut worker = state.worker.clone();
            worker["id"] = json!(id);
            worker
        },
    )
    .with_state(state);
    (serve(app).await, posts)
}

#[tokio::test]
async fn shared_routes_preserve_snapshot_progress_metrics_heartbeat_and_cancel() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    };
    type Stub = (Arc<AtomicBool>, Arc<Mutex<Vec<(String, Value)>>>);
    let state: Stub = (Arc::new(AtomicBool::new(false)), Arc::default());
    let url = serve(routes(|state: &Stub, id, posted| json!({"id":id,"cancelRequested":state.0.load(Ordering::SeqCst),"posted":posted}), |state,id,body|state.1.lock().unwrap().push((id.to_owned(),body)), |_,id|json!({"id":id})).with_state(state.clone())).await;
    let client = reqwest::Client::new();
    let get: Value = client
        .get(format!("{url}/api/v1/jobs/a"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        get,
        json!({"id":"a","cancelRequested":false,"posted":false})
    );
    state.0.store(true, Ordering::SeqCst);
    let posted: Value = client
        .post(format!("{url}/api/v1/jobs/b/progress"))
        .json(&json!({"progress":0.5}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        posted,
        json!({"id":"b","cancelRequested":true,"posted":true})
    );
    assert_eq!(
        *state.1.lock().unwrap(),
        vec![("b".to_owned(), json!({"progress":0.5}))]
    );
    let metrics: Value = client
        .post(format!("{url}/api/v1/jobs/b/metrics"))
        .json(&json!({"m":1}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(metrics, json!({"m":1}));
    let heartbeat: Value = client
        .post(format!("{url}/api/v1/workers/w/heartbeat"))
        .json(&json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(heartbeat, json!({"id":"w"}));
}

/// The same HTTP topology for fixtures with specialized response/status handlers.
pub(crate) fn job_progress_routes<S, J: 'static, P: 'static>(
    job: impl Handler<J, S>,
    progress: impl Handler<P, S>,
) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/api/v1/jobs/:job_id", get(job))
        .route("/api/v1/jobs/:job_id/progress", post(progress))
}

pub(crate) fn job_progress_heartbeat_routes<S, J: 'static, P: 'static, H: 'static>(
    job: impl Handler<J, S>,
    progress: impl Handler<P, S>,
    heartbeat: impl Handler<H, S>,
) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    job_progress_routes(job, progress)
        .route("/api/v1/workers/:worker_id/heartbeat", post(heartbeat))
}
