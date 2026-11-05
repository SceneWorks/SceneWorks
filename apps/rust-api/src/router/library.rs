use super::*;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/poses", post(create_poses))
        .route(
            "/api/v1/poses/sources",
            // sc-8812 (F-010): multipart pose-source image upload; large per-route limit.
            post(create_pose_sources).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        .route(
            "/api/v1/poses/preview/:job_id/:file_name",
            get(get_pose_preview),
        )
        .route("/api/v1/keypoints", post(create_keypoint))
        .route(
            "/api/v1/keypoints/sources",
            // sc-8812 (F-010): multipart keypoint-source image upload; large per-route limit.
            post(create_keypoint_sources).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        .route("/api/v1/keypoints/presets", get(list_keypoint_presets))
        .route(
            "/api/v1/keypoints/collections",
            get(list_keypoint_collections).post(upsert_keypoint_collection),
        )
        .route(
            "/api/v1/keypoints/collections/:collection_id",
            delete(delete_keypoint_collection),
        )
        .route(
            "/api/v1/keypoints/collections/:collection_id/default",
            put(set_default_keypoint_collection),
        )
        .route("/api/v1/control-overlays", get(list_control_overlays))
        .route("/api/v1/styles", get(list_styles))
        .route("/api/v1/loras", get(list_loras))
        .route(
            "/api/v1/loras/:lora_id",
            delete(delete_lora).patch(update_lora),
        )
        .route(
            "/api/v1/loras/:lora_id/embedded-tags",
            get(lora_embedded_tags),
        )
        .route(
            "/api/v1/loras/:lora_id/download",
            post(create_lora_download_job),
        )
        .route(
            "/api/v1/loras/import",
            post(create_lora_import_job)
                .layer(DefaultBodyLimit::max(MAX_LORA_MULTIPART_BODY_BYTES)),
        )
        .route(
            "/api/v1/recipe-presets",
            get(list_recipe_presets).post(create_recipe_preset),
        )
        .route(
            "/api/v1/recipe-presets/:preset_id",
            get(get_recipe_preset)
                .patch(update_recipe_preset)
                .delete(delete_recipe_preset),
        )
        .route(
            "/api/v1/recipe-presets/:preset_id/duplicate",
            post(duplicate_recipe_preset),
        )
        .route(
            "/api/v1/prompt-batches",
            get(list_prompt_batches).post(create_prompt_batch),
        )
        .route(
            "/api/v1/prompt-batches/:batch_id",
            get(get_prompt_batch)
                .patch(update_prompt_batch)
                .delete(delete_prompt_batch),
        )
        .route(
            "/api/v1/prompt-batches/:batch_id/duplicate",
            post(duplicate_prompt_batch),
        )
}
