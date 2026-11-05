use super::*;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/access", get(access))
        .route("/api/v1/auth/verify", post(verify_access))
        .route(
            "/api/v1/credentials",
            get(list_credentials).put(set_credential),
        )
        .route("/api/v1/credentials/:host", delete(delete_credential))
        .route(
            UI_PREFERENCES_PATH,
            get(get_ui_preferences).put(set_ui_preferences),
        )
}
