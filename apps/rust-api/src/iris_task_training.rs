//! The API half of Iris-3B DEPTH (S9 sc-25687) and RESTORATION (S10 sc-25688) training datasets:
//!
//! * **sidecars** — `POST|DELETE /api/v1/projects/:project/training/datasets/:dataset/items/:item/
//!   iris-sidecar/:kind` attaches (or removes) an item's aligned depth map (`depth`: `.npy` or 8/16-bit
//!   grayscale `.png`, with optional `scale` / `minDepth` / `maxDepth`), its validity mask (`mask`) or
//!   its aligned degraded input (`degraded`). The bytes are checked by the ENGINE's readers (format,
//!   pixel alignment with the item's image, integer-scale pairing) before the store copies them beside
//!   the dataset — a misaligned or unreadable sidecar is a 400, never a mid-run failure;
//! * **the dataset check** — `POST /api/v1/projects/:project/training/datasets/:dataset/iris-check`
//!   `{targetId, config}` runs the trainer's own per-item dataset pass (depth: decode, alignment,
//!   covering size, validity under the run's percentiles / `min_valid_pixels`; restoration: the pair
//!   rule and the crop) and reports each item under the run's explicit invalid-example policy, so the
//!   Studio shows what `reject` would refuse or `skip` would drop before Start.

use super::*;
use sceneworks_core::training::iris::IrisObjective;
use sceneworks_core::training_store::{IrisItemSidecar, IrisSidecarKind};

fn sidecar_kind(kind: &str) -> Result<IrisSidecarKind, ApiError> {
    IrisSidecarKind::parse(kind).ok_or_else(|| {
        ApiError::bad_request(format!(
            "Unknown Iris sidecar kind {kind:?} (depth | mask | degraded)."
        ))
    })
}

fn optional_number(text: &str, key: &str) -> Result<Option<f64>, ApiError> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    text.parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
        .map(Some)
        .ok_or_else(|| ApiError::bad_request(format!("{key} must be a finite number.")))
}

/// The item's image and the dataset root, for the content checks.
async fn item_image(
    state: &AppState,
    project_id: &str,
    dataset_id: &str,
    item_id: &str,
) -> Result<PathBuf, ApiError> {
    let (dataset, root, _) = project_call(state.clone(), {
        let project_id = project_id.to_owned();
        let dataset_id = dataset_id.to_owned();
        move |store| store.training_dataset_for_plan(&project_id, &dataset_id)
    })
    .await?;
    let item = dataset
        .items
        .iter()
        .find(|item| item.id == item_id)
        .ok_or_else(|| ApiError::bad_request("Training dataset item not found"))?;
    let mut path = root;
    for component in FsPath::new(&item.path).components() {
        path.push(component);
    }
    Ok(path)
}

pub(crate) async fn upload_iris_item_sidecar(
    State(state): State<AppState>,
    Path((project_id, dataset_id, item_id, kind)): Path<(String, String, String, String)>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<TrainingDataset>), ApiError> {
    let kind = sidecar_kind(&kind)?;
    let mut staged: Option<(PathBuf, String)> = None;
    let (mut scale, mut min_depth, mut max_depth) = (None, None, None);
    let result = async {
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            match field.name() {
                Some("file") => {
                    let extension = field
                        .file_name()
                        .and_then(|name| FsPath::new(name).extension())
                        .and_then(|extension| extension.to_str())
                        .map(str::to_ascii_lowercase)
                        .ok_or_else(|| {
                            ApiError::bad_request("The sidecar upload needs a file extension.")
                        })?;
                    let temp = write_upload_field_to_temp_file(&state, field).await?;
                    // The engine readers dispatch on the extension.
                    let named = temp.with_extension(&extension);
                    std::fs::rename(&temp, &named)
                        .map_err(|error| ApiError::internal(error.to_string()))?;
                    staged = Some((named, extension));
                }
                Some(name @ ("scale" | "minDepth" | "maxDepth")) => {
                    let name = name.to_owned();
                    let text = field
                        .text()
                        .await
                        .map_err(|error| ApiError::bad_request(error.to_string()))?;
                    let value = optional_number(&text, &name)?;
                    match name.as_str() {
                        "scale" => scale = value,
                        "minDepth" => min_depth = value,
                        _ => max_depth = value,
                    }
                }
                _ => {}
            }
        }
        let (source, extension) = staged
            .clone()
            .ok_or_else(|| ApiError::bad_request("Upload file field is required"))?;
        if kind != IrisSidecarKind::Depth && (scale.is_some() || min_depth.is_some() || max_depth.is_some())
        {
            return Err(ApiError::bad_request(
                "scale / minDepth / maxDepth describe a depth map's values; send them with the depth \
                 sidecar.",
            ));
        }
        let image = item_image(&state, &project_id, &dataset_id, &item_id).await?;
        let check_source = source.clone();
        tokio::task::spawn_blocking(move || match kind {
            IrisSidecarKind::Depth | IrisSidecarKind::DepthMask => {
                sceneworks_worker::check_depth_sidecar(&check_source, &image).map(drop)
            }
            IrisSidecarKind::Degraded => {
                sceneworks_worker::check_degraded_sidecar(&check_source, &image).map(drop)
            }
        })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(|error| ApiError::bad_request(format!("The {} sidecar: {error}", kind.as_str())))?;
        let sidecar = IrisItemSidecar {
            kind,
            source: Some((source, extension)),
            scale,
            min_depth,
            max_depth,
        };
        project_call(state.clone(), move |store| {
            store.set_iris_item_sidecar(&project_id, &dataset_id, &item_id, &sidecar)
        })
        .await
    }
    .await;
    if let Some((path, _)) = staged {
        let _ = std::fs::remove_file(path);
    }
    Ok((StatusCode::CREATED, Json(result?)))
}

pub(crate) async fn delete_iris_item_sidecar(
    State(state): State<AppState>,
    Path((project_id, dataset_id, item_id, kind)): Path<(String, String, String, String)>,
) -> Result<Json<TrainingDataset>, ApiError> {
    let sidecar = IrisItemSidecar {
        kind: sidecar_kind(&kind)?,
        source: None,
        scale: None,
        min_depth: None,
        max_depth: None,
    };
    Ok(Json(
        project_call(state, move |store| {
            store.set_iris_item_sidecar(&project_id, &dataset_id, &item_id, &sidecar)
        })
        .await?,
    ))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct IrisDatasetCheckRequest {
    target_id: String,
    config: sceneworks_core::training::TrainingConfig,
}

/// The pre-launch dataset report of a depth / restoration run.
pub(crate) async fn check_iris_task_dataset(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    ApiJson(body): ApiJson<IrisDatasetCheckRequest>,
) -> Result<Json<Value>, ApiError> {
    let registry = sceneworks_core::training::builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == body.target_id)
        .ok_or_else(|| {
            ApiError::bad_request(format!("Unknown training target: {}", body.target_id))
        })?;
    let objective: IrisObjective =
        sceneworks_worker::iris_task_objective(&target.kernel, &target.base_model).ok_or_else(
            || ApiError::bad_request("The dataset check applies to Iris depth and restoration."),
        )?;
    sceneworks_core::training::validate_training_config_for_target(target, &body.config)
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let (dataset, root, _) = project_call(state.clone(), {
        let project_id = project_id.clone();
        let dataset_id = dataset_id.clone();
        move |store| store.training_dataset_for_plan(&project_id, &dataset_id)
    })
    .await?;
    let data_dir = state.settings.data_dir.clone();
    let config = body.config;
    let report = tokio::task::spawn_blocking(move || {
        sceneworks_worker::iris_task_dataset_report(&data_dir, objective, &config, &dataset, &root)
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(Json(report))
}
