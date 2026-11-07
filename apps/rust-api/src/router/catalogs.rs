use super::*;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/catalogs", get(list_catalogs).post(create_catalog))
        .route("/api/v1/catalogs/attach", post(attach_catalog))
        .route(
            "/api/v1/catalogs/:catalog_id",
            get(get_catalog).delete(detach_catalog),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/status",
            get(get_catalog_status),
        )
        .route("/api/v1/catalogs/:catalog_id/query", post(query_catalog))
        .route(
            "/api/v1/catalogs/:catalog_id/curation/query",
            post(curate_catalog),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/curation/facets",
            post(catalog_curation_facets),
        )
        .route("/api/v1/catalogs/:catalog_id/facets", post(catalog_facets))
        .route(
            "/api/v1/catalogs/:catalog_id/saved-views",
            get(list_catalog_saved_views).post(create_catalog_saved_view),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/saved-views/:view_id",
            put(update_catalog_saved_view).delete(delete_catalog_saved_view),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/records/:record_id/review",
            put(review_catalog_record),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/records/:record_id/thumbnail",
            get(catalog_record_thumbnail),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/analyzer-config",
            put(update_catalog_analyzer_config),
        )
        .route("/api/v1/catalogs/:catalog_id/pause", post(pause_catalog))
        .route("/api/v1/catalogs/:catalog_id/resume", post(resume_catalog))
        .route(
            "/api/v1/catalogs/:catalog_id/analyze",
            post(run_catalog_analysis),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/on-disk",
            delete(delete_catalog_on_disk),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/materialize",
            post(materialize_catalog_results),
        )
}
