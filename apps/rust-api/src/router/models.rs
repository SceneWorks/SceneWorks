use super::*;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/api/v1/models/:model_id/license-acknowledgment",
            get(get_license_acknowledgment)
                .put(put_license_acknowledgment)
                .delete(delete_license_acknowledgment),
        )
        // The model source library's live status + relocation seam (sc-19709). Its own path
        // prefix, not `/models/...`, so a library operation can never collide with a model id.
        .route(
            "/api/v1/model-library",
            get(model_library::get_model_library),
        )
        .route(
            "/api/v1/model-library/relocate",
            post(model_library::relocate_model_library),
        )
        .route("/api/v1/models", get(list_models))
        // The curated managed NVFP4 variants (sc-11043, epic 11037). A STATIC sibling of
        // `/api/v1/models/:model_id` for the same reason the library routes are.
        .route(
            "/api/v1/models/managed-variants",
            get(checkpoint_library::list_managed_variants),
        )
        // Linked checkpoint libraries (epic 20398, sc-20635). A STATIC sibling of
        // `/api/v1/models/:model_id`, exactly like `/api/v1/models/import`, so a library
        // operation is never mistaken for a model id.
        .route(
            "/api/v1/models/library-roots",
            get(checkpoint_library::list_library_roots)
                .post(checkpoint_library::approve_library_root),
        )
        .route(
            "/api/v1/models/library-roots/:root_id",
            patch(checkpoint_library::update_library_root)
                .delete(checkpoint_library::remove_library_root),
        )
        .route(
            "/api/v1/models/library-roots/:root_id/scan",
            get(checkpoint_library::scan_library_root),
        )
        .route(
            "/api/v1/models/library-roots/:root_id/rescan",
            post(checkpoint_library::rescan_library_checkpoint),
        )
        .route("/api/v1/models/:model_id", delete(delete_model))
        .route(
            "/api/v1/models/:model_id/variants/:variant",
            delete(delete_model_variant),
        )
        .route(
            "/api/v1/models/:model_id/download",
            post(create_model_download_job),
        )
        .route(
            "/api/v1/models/:model_id/conditional-components/:purpose/download",
            post(create_conditional_components_download_jobs),
        )
        .route(
            "/api/v1/models/:model_id/convert",
            post(create_model_convert_job),
        )
        .route(
            "/api/v1/models/import",
            post(create_model_import_job)
                .layer(DefaultBodyLimit::max(MAX_MODEL_MULTIPART_BODY_BYTES)),
        )
        .route(
            "/api/v1/models/:model_id/repair",
            post(repair_model_import_job)
                .layer(DefaultBodyLimit::max(MAX_MODEL_MULTIPART_BODY_BYTES)),
        )
        // Resolved-model hot cache (sc-19711). The GET is UI-polled, so it is deliberately one
        // cheap write-free listing; the three POSTs are single deliberate user actions.
        .route("/api/v1/model-cache", get(get_model_cache))
        .route(
            "/api/v1/model-cache/removal-preview",
            post(preview_model_cache_removal),
        )
        .route("/api/v1/model-cache/remove", post(remove_model_cache_entry))
        .route("/api/v1/model-cache/pin", post(set_model_cache_pin))
}
