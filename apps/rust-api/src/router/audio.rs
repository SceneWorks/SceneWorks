use super::*;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/api/v1/projects/:project_id/voices",
            get(list_saved_voices).post(create_saved_voice),
        )
        .route(
            "/api/v1/projects/:project_id/voices/:voice_id",
            delete(delete_saved_voice),
        )
        // YuE2 score inspection, bounded edits, renders and listening comparisons (sc-22997).
        .route("/api/v1/yue2/score/inspect", post(inspect_yue2_score))
        .route(
            "/api/v1/projects/:project_id/yue2/score-versions",
            get(list_yue2_score_versions).post(create_yue2_score_version),
        )
        .route(
            "/api/v1/projects/:project_id/yue2/score-versions/:version_id",
            get(get_yue2_score_version),
        )
        .route(
            "/api/v1/projects/:project_id/yue2/score-versions/:version_id/inspection",
            get(inspect_yue2_score_version),
        )
        .route(
            "/api/v1/projects/:project_id/yue2/score-versions/:version_id/edits",
            post(edit_yue2_score_version),
        )
        .route(
            "/api/v1/projects/:project_id/yue2/score-versions/:version_id/renders",
            get(list_yue2_score_renders).post(record_yue2_score_render),
        )
        .route(
            "/api/v1/projects/:project_id/yue2/comparisons",
            get(list_yue2_comparisons).post(create_yue2_comparison),
        )
        // Recording transcriptions (sc-23002): the review artifact's record, scores and exports.
        .route(
            "/api/v1/projects/:project_id/yue2/transcriptions",
            get(list_yue2_transcriptions),
        )
        .route(
            "/api/v1/projects/:project_id/yue2/transcriptions/:transcription_id",
            get(get_yue2_transcription),
        )
        // YuE2 song jobs (sc-22999 / sc-23002): create / plan / fromPlan / cover / renderVersion /
        // decode / transcribe.
        .route(
            "/api/v1/projects/:project_id/yue2/jobs",
            post(create_yue2_jobs),
        )
        .route(
            "/api/v1/projects/:project_id/yue2/comparisons/:comparison_id",
            get(get_yue2_comparison),
        )
}
