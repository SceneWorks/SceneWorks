use super::*;

/// Query params for `GET /api/v1/control-overlays` (sc-10165, epic 10159 B4). Optional project scope +
/// `baseModel` filter so the Studio lists only overlays applicable to the selected model (the frozen
/// inference base the overlay applies on, e.g. `krea_2_turbo`).
#[derive(Debug, Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ControlOverlaysQuery {
    pub project_id: Option<String>,
    pub base_model: Option<String>,
}

/// List installed + registered control overlays. Studio-trained overlays register here on completion
/// (`register_trained_control_overlay`); a hosted catalog + user import arrive with sc-8466 / sc-10979,
/// which extend this same registry. Mirrors [`list_loras`]: assemble user-global + project manifests,
/// normalize install state, optionally filter by `baseModel`.
pub(crate) async fn list_control_overlays(
    State(state): State<AppState>,
    Query(query): Query<ControlOverlaysQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let mut items = control_overlay_catalog(&state, query.project_id.as_deref()).await?;
    if let Some(base_model) = query.base_model {
        let base_model = base_model.trim().to_owned();
        items.retain(|item| {
            item.get("baseModel").and_then(Value::as_str) == Some(base_model.as_str())
        });
    }
    Ok(Json(items))
}

/// Assemble the control-overlay catalog: user-global (`user.control_overlays.jsonc`) + the project
/// manifest (`<project>/control-overlays/manifest.jsonc`) when a project id is given. Each entry is
/// normalized (installedPath / installState / manifestPath) via the shared [`normalize_lora_entry`] — the
/// resolution is identical (a relative `source.path` under the scope root; the training provider carries
/// no HF repo, so the local-path branch resolves it). No builtin control overlays yet — a hosted catalog
/// (`builtin.control_overlays.jsonc`) lands with sc-8466.
pub(crate) async fn control_overlay_catalog(
    state: &AppState,
    project_id: Option<&str>,
) -> Result<Vec<Value>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let user_manifest = manifest_dir.join("user.control_overlays.jsonc");
    let user = load_manifest_entries(state, &user_manifest, "controlOverlays").await?;
    let data_dir = state.settings.data_dir.clone();

    let mut overlays = {
        let data_dir = data_dir.clone();
        let user_manifest = user_manifest.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Value>, ApiError> {
            let mut out = Vec::new();
            for entry in user {
                out.push(crate::loras::normalize_lora_entry(
                    entry,
                    "global",
                    &user_manifest,
                    &data_dir,
                    &data_dir,
                )?);
            }
            Ok(out)
        })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))??
    };

    if let Some(project_id) = project_id {
        let project_path = project_path_for_id(state.clone(), project_id).await?;
        let project_manifest = project_path.join("control-overlays").join("manifest.jsonc");
        let entries = load_manifest_entries(state, &project_manifest, "controlOverlays").await?;
        let data_dir = data_dir.clone();
        let project_overlays =
            tokio::task::spawn_blocking(move || -> Result<Vec<Value>, ApiError> {
                let mut out = Vec::new();
                for entry in entries {
                    out.push(crate::loras::normalize_lora_entry(
                        entry,
                        "project",
                        &project_manifest,
                        &project_path,
                        &data_dir,
                    )?);
                }
                Ok(out)
            })
            .await
            .map_err(|error| ApiError::internal(error.to_string()))??;
        overlays.extend(project_overlays);
    }

    Ok(overlays)
}

/// Resolve a selected control overlay id (`advanced.controlWeights.overlayId`, set by the Studio's
/// ControlPanel picker) to its installed `.safetensors` path, written back as
/// `advanced.controlWeights.path` — the field the worker strict-control lane
/// (`ensure_krea_control_weights`) reads (sc-10165, epic 10159 B4). A no-op when no overlay is selected.
/// A selected-but-unknown/uninstalled overlay is a clean 400 rather than a deep worker error. Mirrors the
/// LoRA id→path resolution (`validate_job_lora_compatibility_with`): one catalog snapshot, in-place
/// `advanced` mutation.
pub(crate) async fn resolve_control_overlay_selection(
    state: &AppState,
    project_id: Option<&str>,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(overlay_id) = job_payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("controlWeights"))
        .and_then(Value::as_object)
        .and_then(|cw| cw.get("overlayId"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
    else {
        return Ok(());
    };

    let overlay = control_overlay_catalog(state, project_id)
        .await?
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(overlay_id.as_str()))
        .ok_or_else(|| ApiError::bad_request(format!("Control overlay not found: {overlay_id}")))?;
    if overlay.get("installState").and_then(Value::as_str) != Some("installed") {
        return Err(ApiError::bad_request(format!(
            "Control overlay '{overlay_id}' is not installed"
        )));
    }
    let installed_path = overlay
        .get("installedPath")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "Control overlay '{overlay_id}' has no installed weights"
            ))
        })?;
    let weights_path = resolve_overlay_weights_file(installed_path, overlay.get("files"))
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "Control overlay '{overlay_id}' weights (.safetensors) not found under {installed_path}"
            ))
        })?;

    let advanced = job_payload
        .entry("advanced".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    if let Some(advanced) = advanced.as_object_mut() {
        let control_weights = advanced
            .entry("controlWeights".to_owned())
            .or_insert_with(|| Value::Object(JsonObject::new()));
        if let Some(control_weights) = control_weights.as_object_mut() {
            control_weights.insert("path".to_owned(), Value::String(weights_path));
        }
    }
    Ok(())
}

/// The overlay's registered `installedPath` may be the overlay dir (the normalized `source.path`) or the
/// `.safetensors` file itself; the worker wants the file. Resolve to the file: the path directly if it is
/// a file, else `<dir>/<files[0]>` when that exists on disk.
fn resolve_overlay_weights_file(installed_path: &str, files: Option<&Value>) -> Option<String> {
    let base = PathBuf::from(installed_path);
    if base.is_file() {
        return Some(installed_path.to_owned());
    }
    let file = files
        .and_then(Value::as_array)
        .and_then(|entries| entries.first())
        .and_then(Value::as_str)?;
    let candidate = base.join(file);
    candidate.is_file().then(|| candidate.display().to_string())
}
