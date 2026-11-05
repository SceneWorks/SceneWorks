use super::*;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/api/v1/projects/:project_id/assets",
            get(list_assets)
                .post(import_asset)
                // sc-8812 (F-010): streaming multipart asset upload needs the large
                // limit; re-attach it per-route since the router default is now the
                // small JSON cap. GET has no body, so this is harmless for listing.
                .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        // sc-15950: read a shared image's embedded workflow WITHOUT creating an asset. Registered
        // beside the asset routes because it shares their multipart shape and staging area, but it
        // is deliberately not project-scoped — it mutates nothing, so there is nothing to scope.
        // Same auth posture as the upload route above; a larger-than-JSON per-route limit is
        // re-attached for the same reason (the router default is the small JSON cap), but it is
        // this route's OWN cap plus framing headroom rather than the 2 GiB asset ceiling — see
        // `MAX_WORKFLOW_INSPECT_BYTES` for why both numbers are what they are.
        .route(
            "/api/v1/workflows/inspect",
            post(inspect_workflow).layer(DefaultBodyLimit::max(max_inspect_multipart_body_bytes())),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id",
            get(get_asset).delete(delete_asset),
        )
        // sc-15952: the resolution report for the envelope an IMPORTED asset is already carrying
        // at `extra.importedWorkflow`. The envelope needs no route — it rides on the asset — but
        // the report is computed against catalogs that live behind this API, and it goes stale the
        // moment a download finishes, so re-resolving after an install is a plain refetch of this.
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/workflow",
            get(get_asset_workflow),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/poster/:poster_sha256",
            get(get_asset_poster),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/purge",
            delete(purge_asset),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/move-to-library",
            post(move_asset_to_library),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/move-to-character",
            post(move_asset_to_character),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/status",
            patch(update_asset_status),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/tags",
            patch(update_asset_tags),
        )
        .route(
            "/api/v1/projects/:project_id/files/*relative_path",
            get(get_project_file),
        )
        .route(
            "/api/v1/projects/:project_id/yue2/transcriptions/:transcription_id/files/*file",
            get(get_yue2_transcription_file),
        )
        // Media ticket (sc-8810): auth-protected mint endpoint; the ticket is honored
        // as a query param by the project-files and pose-preview GETs (see auth.rs).
        .route("/api/v1/files/ticket", post(create_media_ticket))
}
