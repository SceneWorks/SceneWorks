use super::*;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/workers", get(list_workers))
        .route(
            "/api/v1/capabilities/person",
            get(person_capability_readiness),
        )
        .route("/api/v1/capabilities/mac", get(mac_capability_support))
        // Host memory for remote-browser model gating (epic 4484 story 9).
        .route("/api/v1/host-capabilities", get(host_capabilities))
        // Remote-admin GPU worker restart (epic 4484 story 12).
        .route("/api/v1/worker/restart", post(request_worker_restart))
        .route("/api/v1/workers/register", post(register_worker))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_worker),
        )
        .route(
            "/api/v1/workers/:worker_id/portable-source-challenges",
            post(portable_source_challenges),
        )
        .route(
            "/api/v1/workers/:worker_id/portable-source-proof",
            post(publish_portable_source_proof),
        )
        .route(
            "/api/v1/workers/:worker_id/terminated",
            post(worker_terminated),
        )
}
