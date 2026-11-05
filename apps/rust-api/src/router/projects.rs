use super::*;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/projects", get(list_projects).post(create_project))
        .route("/api/v1/projects/:project_id", get(get_project))
        .route(
            "/api/v1/projects/:project_id/reindex",
            post(reindex_project_endpoint),
        )
}
