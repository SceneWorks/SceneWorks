//! API-owned readiness for MOSS edit artifacts. Core storage never depends on JobsStore.
use super::*;
use std::collections::{HashMap, HashSet};

pub(crate) fn canonical_batch(job: &JobSnapshot) -> Option<&Vec<Value>> {
    if !jobs::moss_edit_output_batch(job)
        || job.status != JobStatus::Completed
        || job.result.contains_key("assetWrites")
    {
        return None;
    }
    let set = job.result.get("generationSetId")?.as_str()?;
    let assets = job.result.get("assets")?.as_array()?;
    let ids = job.result.get("assetIds")?.as_array()?;
    let expected = job.result.get("expectedCount")?.as_u64()? as usize;
    if expected
        != sceneworks_core::image_request::ImageRequest::from_payload(&job.payload).image_count()
            as usize
    {
        return None;
    }
    let output_count = if Value::Object(job.payload.clone())
        .pointer("/upscale/enabled")
        .and_then(Value::as_bool)
        == Some(true)
        && assets
            .iter()
            .any(|asset| asset.get("type").and_then(Value::as_str) == Some("image"))
    {
        expected.checked_mul(2)?
    } else {
        expected
    };
    if expected == 0 || expected > 8 || assets.len() != output_count || ids.len() != assets.len() {
        return None;
    }
    let advanced = job
        .payload
        .get("advanced")
        .and_then(|value| value.get(sceneworks_core::moss_image_advanced::KEY));
    let tensor_output = advanced
        .and_then(|value| value.get("output"))
        .and_then(Value::as_str)
        == Some("latents");
    let explicit_noise = advanced
        .and_then(|value| value.get("noise"))
        .is_some_and(|value| !value.is_null());
    let mut unique = HashSet::new();
    for (asset, id) in assets.iter().zip(ids) {
        let asset_id = id.as_str()?;
        if !unique.insert(asset_id)
            || asset.get("id") != Some(id)
            || asset.get("projectId").and_then(Value::as_str) != job.project_id.as_deref()
            || asset.get("generationSetId").and_then(Value::as_str) != Some(set)
            || asset.pointer("/lineage/jobId").and_then(Value::as_str) != Some(job.id.as_str())
        {
            return None;
        }
        if asset.get("type").and_then(Value::as_str)
            != Some(if tensor_output { "tensor" } else { "image" })
            || (explicit_noise && asset.pointer("/recipe/seed") != Some(&Value::Null))
        {
            return None;
        }
        let path = asset.pointer("/file/path")?.as_str()?;
        let (candidate, recognized) = path_generation(path)?;
        if candidate != set {
            return None;
        }
        if recognized || asset.pointer("/extra/mossPublication").is_some() {
            if asset
                .pointer("/extra/mossPublication/jobId")
                .and_then(Value::as_str)
                != Some(job.id.as_str())
                || asset
                    .pointer("/extra/mossPublication/attempt")
                    .and_then(Value::as_u64)
                    != Some(job.attempts as u64)
                || asset
                    .pointer("/extra/mossPublication/generationSetId")
                    .and_then(Value::as_str)
                    != Some(set)
            {
                return None;
            }
        }
    }
    Some(assets)
}

fn path_generation(path: &str) -> Option<(&str, bool)> {
    let parts = std::path::Path::new(path)
        .components()
        .map(|part| match part {
            std::path::Component::Normal(part) => part.to_str(),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    if parts.len() >= 4 && parts[0] == "assets" && matches!(parts[1], "images" | "tensors") {
        if parts[2] == "moss-edit" {
            return parts.get(3).map(|id| (*id, true));
        }
        return Some((parts[2], false));
    }
    if parts.len() == 2 && parts[0] == "generation-sets" {
        return parts[1].strip_suffix(".json").map(|id| (id, false));
    }
    None
}

fn path_in_batch(assets: &[Value], path: &str) -> bool {
    assets.iter().any(|asset| {
        asset.pointer("/file/path").and_then(Value::as_str) == Some(path)
            || asset.pointer("/preview/path").and_then(Value::as_str) == Some(path)
            || asset
                .pointer("/extra/workflowCompanion/path")
                .and_then(Value::as_str)
                == Some(path)
    })
}

pub(crate) async fn require_path(
    state: &AppState,
    project: &str,
    path: &str,
) -> Result<(), ApiError> {
    let normalized = std::path::Path::new(path)
        .components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    let raw_recognized = path_generation(path).is_some_and(|(_, recognized)| recognized);
    if raw_recognized && path != normalized {
        return Err(ApiError::bad_request(
            "MOSS output paths must use their canonical artifact spelling",
        ));
    }
    // ProjectStore permits project-local symlinks. Gate their actual target too, so an
    // ordinary-looking alias cannot expose MOSS staging before publication.
    let canonical = project_call(state.clone(), {
        let project = project.to_owned();
        let path = path.to_owned();
        move |store| {
            let file = match store.project_file(&project, &path) {
                Ok(file) => file,
                Err(ProjectStoreError::NotFound(_)) => return Ok(None),
                Err(error) => return Err(error),
            };
            let root = std::fs::canonicalize(store.get_project(&project)?.path)?;
            let relative = file.path.strip_prefix(root).map_err(|_| {
                ProjectStoreError::BadRequest("Invalid canonical project file".into())
            })?;
            Ok(Some(
                relative
                    .components()
                    .map(|part| part.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/"),
            ))
        }
    })
    .await?;
    if raw_recognized
        && canonical
            .as_deref()
            .is_some_and(|canonical| canonical != normalized)
    {
        return Err(ApiError::bad_request(
            "MOSS output paths cannot redirect to another artifact",
        ));
    }
    let path = canonical.as_deref().unwrap_or(path);
    let Some((generation, recognized)) = path_generation(path) else {
        return Ok(());
    };
    let found = store_call(state.clone(), {
        let project = project.to_owned();
        let generation = generation.to_owned();
        move |store, _| store.jobs_for_generation_set(&project, &generation)
    })
    .await?;
    let moss = found
        .iter()
        .filter(|job| jobs::moss_edit_output_batch(job))
        .collect::<Vec<_>>();
    if !recognized && moss.is_empty() {
        return Ok(());
    }
    if found.len() != 1
        || moss.len() != 1
        || !canonical_batch(moss[0]).is_some_and(|assets| path_in_batch(assets, path))
    {
        return Err(ApiError::conflict("MOSS output publication is pending or this path is not a canonical output. Poll its job for ready assets."));
    }
    Ok(())
}

fn same_asset_binding(canonical: &Value, asset: &Value) -> bool {
    canonical.get("id") == asset.get("id")
        && canonical.pointer("/file/path") == asset.pointer("/file/path")
        && canonical.get("generationSetId") == asset.get("generationSetId")
        && canonical.pointer("/lineage/jobId") == asset.pointer("/lineage/jobId")
        && canonical.pointer("/extra/mossPublication") == asset.pointer("/extra/mossPublication")
}

pub(crate) async fn require_asset(
    state: &AppState,
    project: &str,
    asset: &Value,
) -> Result<(), ApiError> {
    if let Some(path) = asset.pointer("/file/path").and_then(Value::as_str) {
        require_path(state, project, path).await?;
        if let Some((generation, recognized)) = path_generation(path) {
            let found = store_call(state.clone(), {
                let project = project.to_owned();
                let generation = generation.to_owned();
                move |store, _| store.jobs_for_generation_set(&project, &generation)
            })
            .await?;
            let moss = found
                .iter()
                .filter(|job| jobs::moss_edit_output_batch(job))
                .collect::<Vec<_>>();
            if (recognized || !moss.is_empty())
                && (found.len() != 1
                    || moss.len() != 1
                    || !canonical_batch(moss[0]).is_some_and(|assets| {
                        assets
                            .iter()
                            .any(|canonical| same_asset_binding(canonical, asset))
                    }))
            {
                return Err(ApiError::conflict(
                    "MOSS asset is not part of the canonical completed output batch",
                ));
            }
        }
    }
    Ok(())
}

pub(crate) async fn visible_assets(
    state: &AppState,
    project: &str,
    assets: Vec<Value>,
) -> Result<Vec<Value>, ApiError> {
    // Reuse each exact generation lookup within this response only. Recovery may change it later.
    let mut readiness: HashMap<String, Option<Vec<Value>>> = HashMap::new();
    let mut visible = Vec::with_capacity(assets.len());
    for asset in assets {
        let Some(path) = asset.pointer("/file/path").and_then(Value::as_str) else {
            visible.push(asset);
            continue;
        };
        let Some((generation, recognized)) = path_generation(path) else {
            visible.push(asset);
            continue;
        };
        if !readiness.contains_key(generation) {
            let jobs = store_call(state.clone(), {
                let project = project.to_owned();
                let generation = generation.to_owned();
                move |store, _| store.jobs_for_generation_set(&project, &generation)
            })
            .await?;
            let moss = jobs
                .iter()
                .filter(|job| jobs::moss_edit_output_batch(job))
                .collect::<Vec<_>>();
            let batch = if jobs.len() == 1 && moss.len() == 1 {
                canonical_batch(moss[0]).cloned()
            } else {
                None
            };
            if !recognized && moss.is_empty() {
                visible.push(asset);
                continue;
            }
            readiness.insert(generation.to_owned(), batch);
        }
        if readiness
            .get(generation)
            .and_then(Option::as_ref)
            .is_some_and(|batch| {
                batch.iter().any(|canonical| {
                    same_asset_binding(canonical, &asset)
                        && path_in_batch(std::slice::from_ref(canonical), path)
                })
            })
        {
            visible.push(asset);
        }
    }
    Ok(visible)
}
