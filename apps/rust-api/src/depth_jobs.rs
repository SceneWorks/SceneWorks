use super::*;
use sceneworks_core::depth_map::{ImageDepthJobPayload, DEPTH_ASSET_TYPE};

/// The typed `image_depth` request (Iris-3B monocular depth, sc-25682): the shared
/// [`ImageDepthJobPayload`] contract plus the queue's GPU preference.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ImageDepthJobRequest {
    #[serde(flatten)]
    pub(crate) request: ImageDepthJobPayload,
    #[serde(default = "crate::defaults::default_requested_gpu")]
    pub(crate) requested_gpu: String,
}

/// `POST /api/v1/image/depth/jobs` — enqueue one `image_depth` job for one source image. The Image
/// Editor action posts one; the Library batch action posts one per selected image.
///
/// The source is resolved up front so a missing, non-image or depth-map source is a clean 400 rather
/// than a doomed job; the worker re-resolves and re-confines the path independently.
pub(crate) async fn create_image_depth_job(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<ImageDepthJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let ImageDepthJobRequest {
        request,
        requested_gpu,
    } = body;
    request.validate().map_err(ApiError::bad_request)?;
    // A trained depth model (sc-25687) must be a registered depth model, never another task's.
    if let Some(model) = request.trained_model() {
        crate::iris_training::validate_trained_task_model(
            &state,
            model,
            sceneworks_core::training::iris::IrisObjective::Depth,
        )
        .await?;
    }
    ensure_depth_source_is_an_image(&state, &request.project_id, &request.source_asset_id).await?;
    let payload = to_json_object(&request)?;
    let job = create_generation_job(
        state,
        JobType::ImageDepth,
        Some(request.project_id.clone()),
        request.project_name.clone(),
        payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

/// Validate an `image_depth` payload arriving on the raw queue route (and on retry/duplicate, which
/// re-validate a stored payload through the same seam): the same typed contract as the typed route.
pub(crate) fn validate_raw_image_depth_payload(payload: &JsonObject) -> Result<(), ApiError> {
    ImageDepthJobPayload::from_job_payload(payload)
        .map(|_| ())
        .map_err(ApiError::bad_request)
}

async fn ensure_depth_source_is_an_image(
    state: &AppState,
    project_id: &str,
    source_asset_id: &str,
) -> Result<(), ApiError> {
    let project_id = project_id.to_owned();
    let asset_id = source_asset_id.to_owned();
    let asset = project_call(state.clone(), move |store| {
        store.get_asset(&project_id, &asset_id)
    })
    .await?;
    let asset_type = asset.get("type").and_then(Value::as_str).unwrap_or("");
    let mime = asset
        .pointer("/file/mimeType")
        .and_then(Value::as_str)
        .unwrap_or("");
    if asset_type == DEPTH_ASSET_TYPE {
        return Err(ApiError::bad_request(
            "The source is already a depth map; depth is estimated from an image.",
        ));
    }
    let is_image = (asset_type == "image" || mime.starts_with("image/"))
        && mime != "image/svg+xml"
        && asset_type != "vector";
    if !is_image {
        return Err(ApiError::bad_request(format!(
            "Depth is estimated from an image; source asset {source_asset_id} is a {}.",
            if asset_type.is_empty() {
                "non-image asset"
            } else {
                asset_type
            }
        )));
    }
    Ok(())
}
