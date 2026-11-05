use super::*;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/image/jobs", post(create_image_job))
        .route("/api/v1/image/vectorize/jobs", post(create_vector_job))
        .route(
            "/api/v1/image/vectorize/prompt/jobs",
            post(create_vector_prompt_workflow),
        )
        .route("/api/v1/image/depth/jobs", post(create_image_depth_job))
        .route("/api/v1/image/vqa/jobs", post(create_vqa_job))
        .route("/api/v1/image/interleave/jobs", post(create_interleave_job))
        .route("/api/v1/video/jobs", post(create_video_job))
        .route("/api/v1/audio/jobs", post(create_audio_job))
        .route("/api/v1/prompts/refine", post(create_prompt_refine_job))
        .route(
            "/api/v1/face-likeness/compare",
            post(create_face_likeness_compare_job),
        )
        .route(
            "/api/v1/image-restoration/plan",
            post(plan_image_restoration),
        )
}
