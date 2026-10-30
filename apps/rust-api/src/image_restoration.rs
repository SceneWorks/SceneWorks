//! Iris-3B restoration pre-submission plan preview (sc-25683, epic 25678).
//!
//! `POST /api/v1/image-restoration/plan` runs the engine's own pure planner
//! (`gen_core_iris::restoration::plan`, through `sceneworks_worker::iris_restore`) with no weights,
//! so the Image Studio post-processing panel and the Image Editor / Library restore actions can show
//! the ACTUAL processing and output dimensions before anything is submitted — and label a
//! budget-downscaled result honestly ("budgeted: 512x384 input → 2048x1536", never "4× of the
//! original"). The worker plans the job with the same function, so the preview cannot drift from
//! the run.
//!
//! Three source forms:
//! * `{ width, height }` — a not-yet-rendered image (the Studio post-processing pass plans against
//!   the generation geometry);
//! * `{ sizes: [{ width, height }, ...] }` — several not-yet-rendered sizes (a Studio prompt batch
//!   whose prompts carry their own `[WxH]` directives plans every distinct size, so the Run gate
//!   covers each render rather than only the Studio default);
//! * `{ projectId, assetIds: [...] }` — existing assets; each is measured from its file header with
//!   its EXIF orientation applied (the job bakes orientation in before restoring, so a rotated phone
//!   photo plans against its upright size).
//!
//! `irisModel` (a trained restoration model) plans with THAT export's tile geometry, exactly as the
//! job does; a model that cannot run comes back as each item's refusal.
//!
//! The Iris controls (`scale`, `inputSizing`, `colorFix`) parse through the one shared
//! `sceneworks_core::image_restoration` parse the job validation and the worker use. A plan the
//! worker would refuse (e.g. an output over the per-side bound) comes back as that item's `error`
//! text, so the UI can block submission with the same words the job would fail with.

use super::*;
use sceneworks_core::image_restoration::{parse_iris_restore_model, parse_iris_restore_params};
use sceneworks_worker::iris_restore::{
    oriented_image_dimensions, preview_restore_plan_with, restore_plan_geometry, PlanGeometry,
};

/// The most assets one preview call measures (a Library batch selection).
pub(crate) const MAX_PLAN_ASSETS: usize = 200;

pub(crate) async fn plan_image_restoration(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<JsonObject>,
) -> Result<Json<Value>, ApiError> {
    let params = parse_iris_restore_params(&body).map_err(ApiError::bad_request)?;
    let trained_model = parse_iris_restore_model(&body).map_err(ApiError::bad_request)?;
    let geometry = {
        let data_dir = state.settings.data_dir.clone();
        tokio::task::spawn_blocking(move || {
            restore_plan_geometry(&data_dir, trained_model.as_deref())
        })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
    };
    let dimension = |key: &str| -> Result<Option<u32>, ApiError> {
        match body.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value > 0)
                .map(Some)
                .ok_or_else(|| ApiError::bad_request(format!("{key} must be a positive integer"))),
        }
    };
    let (width, height) = (dimension("width")?, dimension("height")?);
    let asset_ids: Vec<String> = match body.get("assetIds") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(ids)) => ids
            .iter()
            .map(|id| {
                id.as_str()
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| ApiError::bad_request("assetIds must be non-empty strings"))
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(ApiError::bad_request("assetIds must be an array")),
    };
    let sizes: Vec<(u32, u32)> = match body.get("sizes") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(sizes)) => sizes
            .iter()
            .map(|size| {
                let side = |key: &str| {
                    size.get(key)
                        .and_then(Value::as_u64)
                        .and_then(|value| u32::try_from(value).ok())
                        .filter(|value| *value > 0)
                };
                side("width").zip(side("height")).ok_or_else(|| {
                    ApiError::bad_request("sizes must be { width, height } positive integers")
                })
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(ApiError::bad_request("sizes must be an array")),
    };

    let items = match (width, height, asset_ids.is_empty(), sizes.is_empty()) {
        (Some(width), Some(height), true, true) => {
            vec![plan_item(None, Some((width, height)), &params, &geometry)]
        }
        (None, None, true, false) => {
            if sizes.len() > MAX_PLAN_ASSETS {
                return Err(ApiError::bad_request(format!(
                    "at most {MAX_PLAN_ASSETS} sizes per plan request"
                )));
            }
            sizes
                .into_iter()
                .map(|size| plan_item(None, Some(size), &params, &geometry))
                .collect()
        }
        (None, None, false, true) => {
            if asset_ids.len() > MAX_PLAN_ASSETS {
                return Err(ApiError::bad_request(format!(
                    "at most {MAX_PLAN_ASSETS} assetIds per plan request"
                )));
            }
            let project_id = body
                .get("projectId")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| ApiError::bad_request("projectId is required with assetIds"))?
                .to_owned();
            let project_path = project_path_for_id(state.clone(), &project_id).await?;
            let mut items = Vec::with_capacity(asset_ids.len());
            for asset_id in asset_ids {
                let path = resolve_project_confined_asset_path(
                    state.clone(),
                    &project_id,
                    &asset_id,
                    &project_path,
                )
                .await?;
                let dims = tokio::task::spawn_blocking(move || oriented_image_dimensions(&path))
                    .await
                    .map_err(|error| ApiError::internal(error.to_string()))?;
                items.push(plan_item(Some(asset_id), dims, &params, &geometry));
            }
            items
        }
        _ => {
            return Err(ApiError::bad_request(
                "give exactly one of width + height (a not-yet-rendered image), sizes, or \
                 projectId + assetIds",
            ))
        }
    };
    Ok(Json(json!({ "items": items })))
}

fn plan_item(
    asset_id: Option<String>,
    dims: Option<(u32, u32)>,
    params: &sceneworks_core::image_restoration::IrisRestoreParams,
    geometry: &Result<PlanGeometry, String>,
) -> Value {
    let mut item = json!({ "assetId": asset_id, "plan": null, "error": null });
    match (dims, geometry) {
        (None, _) => item["error"] = json!("The image's dimensions could not be read."),
        (Some(_), Err(error)) => item["error"] = json!(error),
        (Some((width, height)), Ok(geometry)) => {
            match preview_restore_plan_with(width, height, params, *geometry) {
                Ok(plan) => item["plan"] = json!(plan),
                Err(error) => item["error"] = json!(error),
            }
        }
    }
    item
}
