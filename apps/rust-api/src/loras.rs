use super::*;

pub(crate) async fn list_loras(
    State(state): State<AppState>,
    Query(query): Query<LorasQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let mut items = lora_catalog(&state, query.project_id.as_deref()).await?;
    // Tell the client which rows sit behind a licence acknowledgment (sc-17227). The download route
    // below refuses an unacknowledged fetch of a licence-gated repo; without this the refusal has no
    // remedy in any shipped surface, because nothing on the row says an acknowledgment is needed and
    // no LoRA card carries licence copy of its own. The annotation names the MODEL whose card does.
    crate::models::annotate_license_acknowledgment_sources(&state, &mut items, |item| {
        item.get("source")
            .and_then(Value::as_object)
            .and_then(|source| source.get("repo"))
            .or_else(|| item.get("repo"))
            .and_then(Value::as_str)
            .map(str::to_owned)
    })
    .await?;
    if let Some(model_family) = query.model_family {
        // `lora_families` returns canonical tokens, so canonicalize the raw query
        // param too — otherwise a `?model_family=krea-2` filter would miss a stored
        // `krea_2` LoRA.
        let model_family = normalize_lora_family(&model_family);
        items.retain(|item| {
            lora_families(item)
                .iter()
                .any(|family| family == &model_family)
        });
    }
    Ok(Json(items))
}

/// Explicitly download a catalog LoRA (built-in or user-global) whose source is a
/// Hugging Face repo into the shared HF cache (sc-5944). Mirrors the model-download
/// endpoint: the worker fetches the adapter file(s) into the cache the install-state
/// probe (`lora_huggingface_cached_file`) reads, so the catalog row flips to
/// "installed". Built-in LoRAs previously had no way to be fetched from the UI — they
/// were only pulled on-demand at first generation.
pub(crate) async fn create_lora_download_job(
    State(state): State<AppState>,
    Path(lora_id): Path<String>,
    ApiJson(payload): ApiJson<ModelDownloadRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let lora = lora_catalog(&state, None)
        .await?
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(lora_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "LoRA not found".to_owned(),
            context: None,
            code: None,
        })?;
    if lora.get("installState").and_then(Value::as_str) == Some("installed")
        && lora.get("updateAvailable").and_then(Value::as_bool) != Some(true)
    {
        // `installState` is probed live from the HF cache on every catalog read, but a
        // client's rendered badge is a snapshot from its last fetch. Anything that fills
        // the cache out-of-band — the on-demand pull at first generation, another client,
        // a manual `hf download` — flips the server to "installed" while an open catalog
        // view still reads "Not Installed". Tag the rejection so the client can tell this
        // benign disagreement apart from a real failure and resync instead of surfacing a
        // message that contradicts the badge it is showing.
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            detail: "LoRA is already installed".to_owned(),
            context: None,
            code: Some("lora_already_installed"),
        });
    }
    let source = lora.get("source").and_then(Value::as_object);
    let provider = source
        .and_then(|source| source.get("provider"))
        .or_else(|| lora.get("provider"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if provider != Some("huggingface") {
        return Err(ApiError::bad_request(
            "LoRA does not define a Hugging Face download source",
        ));
    }
    let repo = source
        .and_then(|source| source.get("repo"))
        .or_else(|| lora.get("repo"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ApiError::bad_request("LoRA download source is missing a repo"))?
        .to_owned();
    // Licence acknowledgment, keyed on the repo this download will FETCH (sc-17227) — the same
    // predicate `POST /api/v1/jobs`, `/models/import` and `/loras/import` apply.
    //
    // This route had no licence check of any kind, which made it asymmetric with
    // `create_model_download_job`: that route gates on its catalog entry, this one did not gate at
    // all. The reason previously recorded for leaving it alone — that the repo comes from the
    // catalog entry the path id names, so "a caller cannot supply a repo" — answers a different
    // question. Who CHOOSES the repo is not who is bound by its licence: a catalog LoRA whose
    // `source.repo` names a repo a `requiresLicenseAcknowledgment` model declares would have been
    // fetched here with no acknowledgment, while the identical `lora_download` job posted to
    // `/api/v1/jobs` was answered 403. Unreachable in the shipped LoRA catalog today; the route
    // comment above notes the on-demand pull at first generation, which is the path that would
    // surface it.
    crate::models::ensure_license_acknowledged_for_source(
        &state,
        &[Some(repo.as_str())],
        None,
        payload.license_acknowledged,
    )
    .await?;
    // A single `file` or an explicit `files` list narrows the snapshot to the adapter
    // weights; an empty list lets the worker fetch the (small) repo.
    let mut files: Vec<String> = Vec::new();
    if let Some(file) = source
        .and_then(|source| source.get("file"))
        .or_else(|| lora.get("file"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        files.push(file.to_owned());
    }
    if files.is_empty() {
        if let Some(list) = source
            .and_then(|source| source.get("files"))
            .or_else(|| lora.get("files"))
            .and_then(Value::as_array)
        {
            files.extend(
                list.iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned),
            );
        }
    }

    let mut job_payload = JsonObject::new();
    job_payload.insert("loraId".to_owned(), Value::String(lora_id.clone()));
    job_payload.insert(
        "loraName".to_owned(),
        Value::String(
            lora.get("name")
                .and_then(Value::as_str)
                .unwrap_or(&lora_id)
                .to_owned(),
        ),
    );
    job_payload.insert(
        "provider".to_owned(),
        Value::String("huggingface".to_owned()),
    );
    job_payload.insert("repo".to_owned(), Value::String(repo));
    job_payload.insert("files".to_owned(), json!(files));
    // Record the acknowledgment ON the job (sc-17227), for the reason `create_model_download_job`
    // does: RETRY and DUPLICATE re-run `validate_raw_job_payload` over the STORED payload, and the
    // repo-keyed gate there would otherwise refuse a download this route had already authorized.
    if payload.license_acknowledged {
        job_payload.insert(
            crate::models::LICENSE_ACKNOWLEDGED_PAYLOAD_KEY.to_owned(),
            Value::Bool(true),
        );
    }
    if let Some(revision) = source
        .and_then(|source| source.get("revision"))
        .or_else(|| lora.get("revision"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        job_payload.insert("revision".to_owned(), Value::String(revision.to_owned()));
    }
    if let Some(family) = lora.get("family").and_then(Value::as_str) {
        if !family.trim().is_empty() {
            job_payload.insert("family".to_owned(), Value::String(family.to_owned()));
        }
    }

    let job = create_generation_job(
        state,
        JobType::LoraDownload,
        None,
        None,
        job_payload,
        requested_gpu_or_auto(payload.requested_gpu),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

pub(crate) async fn delete_lora(
    State(state): State<AppState>,
    Path(lora_id): Path<String>,
    Query(query): Query<CatalogDeleteQuery>,
) -> Result<Json<Value>, ApiError> {
    let catalog = lora_catalog(&state, query.project_id.as_deref()).await?;
    let lora = catalog
        .into_iter()
        .find(|item| {
            item.get("id").and_then(Value::as_str) == Some(lora_id.as_str())
                && query
                    .scope
                    .as_deref()
                    .is_none_or(|scope| item.get("scope").and_then(Value::as_str) == Some(scope))
        })
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "LoRA not found".to_owned(),
            context: None,
            code: None,
        })?;
    let scope = query
        .scope
        .as_deref()
        .or_else(|| lora.get("scope").and_then(Value::as_str))
        .unwrap_or("global");
    let (manifest_path, allowed_roots, default_root) = match scope {
        "global" => (
            Some(
                state
                    .settings
                    .config_dir
                    .join("manifests")
                    .join("user.loras.jsonc"),
            ),
            vec![state.settings.data_dir.join("loras")],
            state.settings.data_dir.clone(),
        ),
        "project" => {
            let Some(project_id) = query.project_id.as_deref() else {
                return Err(ApiError::bad_request(
                    "Project LoRA deletion requires projectId",
                ));
            };
            let project_path = project_path_for_id(state.clone(), project_id).await?;
            (
                Some(project_path.join("loras").join("manifest.jsonc")),
                vec![
                    state.settings.data_dir.join("loras"),
                    project_path.join("loras"),
                ],
                project_path,
            )
        }
        "builtin" => (
            None,
            vec![state.settings.data_dir.join("loras")],
            state.settings.data_dir.clone(),
        ),
        // sc-10452: the file lives in the operator's own external tree (e.g. ComfyUI).
        // We read it in place and never took ownership, so deletion is not ours to do.
        crate::external_loras::EXTERNAL_SCOPE => {
            return Err(ApiError::bad_request(
                "This LoRA lives in an external model folder and is read-only. \
                 Delete it from that folder directly.",
            ));
        }
        _ => return Err(ApiError::bad_request("Unsupported LoRA scope")),
    };
    let permanent = query.permanent.unwrap_or(false);
    // Peek (not remove) the manifest entry so a failed move-to-trash leaves the
    // catalog intact for the client's permanent-delete confirmation prompt.
    let manifest_entry = if let Some(manifest_path) = manifest_path.as_deref() {
        load_manifest_entries(&state, manifest_path, "loras")
            .await?
            .into_iter()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(lora_id.as_str()))
    } else {
        None
    };
    let cleanup_source = manifest_entry.as_ref().unwrap_or(&lora);
    let mut removal = remove_owned_artifacts(
        lora_artifact_paths(cleanup_source, &default_root),
        &allowed_roots,
        permanent,
    )
    .await?;
    // Sweep any preview-sample directory a training run parked under the owning project (see
    // `lora_preview_sample_paths`). Run for every scope, not just `global`: it costs one readdir,
    // and keying it off the scope string would leak previews the moment that metadata drifts. These
    // paths get their OWN narrow allowed root — `<data>/projects` must not widen what a
    // manifest-supplied artifact path is allowed to remove.
    let preview_paths = lora_preview_sample_paths(&state.settings.data_dir, &lora_id).await;
    if !preview_paths.is_empty() {
        let preview_removal = remove_owned_artifacts(
            preview_paths,
            &[state.settings.data_dir.join("projects")],
            permanent,
        )
        .await?;
        removal.removed_paths.extend(preview_removal.removed_paths);
        removal
            .retained_paths
            .extend(preview_removal.retained_paths);
        removal
            .trash_failed_paths
            .extend(preview_removal.trash_failed_paths);
    }
    if !permanent && !removal.trash_failed_paths.is_empty() {
        return Ok(Json(json!({
            "id": lora_id,
            "kind": "lora",
            "scope": scope,
            "trashUnavailable": true,
            "trashFailedPaths": removal.trash_failed_paths,
            "removedManifestEntry": false,
            "removedLocalArtifacts": !removal.removed_paths.is_empty(),
            "removedPaths": removal.removed_paths,
            "retainedPaths": removal.retained_paths,
        })));
    }
    let removed_entry = if let Some(manifest_path) = manifest_path.as_deref() {
        remove_catalog_manifest_entry(&state, manifest_path, "loras", &lora_id).await?
    } else {
        None
    };
    if removed_entry.is_none() && removal.removed_paths.is_empty() {
        return Err(ApiError::bad_request(
            "Built-in LoRA catalog entries are read-only unless local files are installed",
        ));
    }
    let warnings =
        catalog_delete_warnings(&state, "lora", &lora_id, query.project_id.as_deref(), None)
            .await?;
    let policy = if removed_entry.is_some() {
        "Removed the LoRA registry entry and SceneWorks-owned local LoRA files."
    } else {
        "Built-in LoRA catalog entries are retained; SceneWorks-owned local LoRA files were removed."
    };
    Ok(Json(json!({
        "id": lora_id,
        "kind": "lora",
        "scope": scope,
        "trashed": !permanent,
        "removedManifestEntry": removed_entry.is_some(),
        "removedLocalArtifacts": !removal.removed_paths.is_empty(),
        "removedPaths": removal.removed_paths,
        "retainedPaths": removal.retained_paths,
        "warnings": warnings,
        "policy": policy,
    })))
}

/// Edit a catalog LoRA's user-facing metadata (trigger keywords and usage notes)
/// after import — the capability scoped under epic 1092 / story 1168 but never
/// shipped. Only the fields present in the request change. Built-in entries are
/// read-only (their manifest is compiled in); import a copy to annotate them.
pub(crate) async fn update_lora(
    State(state): State<AppState>,
    Path(lora_id): Path<String>,
    Query(query): Query<LoraCatalogItemQuery>,
    ApiJson(body): ApiJson<LoraUpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    let lora = lora_catalog(&state, query.project_id.as_deref())
        .await?
        .into_iter()
        .find(|item| {
            item.get("id").and_then(Value::as_str) == Some(lora_id.as_str())
                && query
                    .scope
                    .as_deref()
                    .is_none_or(|scope| item.get("scope").and_then(Value::as_str) == Some(scope))
        })
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "LoRA not found".to_owned(),
            context: None,
            code: None,
        })?;
    let scope = query
        .scope
        .as_deref()
        .or_else(|| lora.get("scope").and_then(Value::as_str))
        .unwrap_or("global");
    // Mirror delete_lora's scope routing to find the writable manifest.
    let manifest_path = match scope {
        "global" => state
            .settings
            .config_dir
            .join("manifests")
            .join("user.loras.jsonc"),
        "project" => {
            let Some(project_id) = query.project_id.as_deref() else {
                return Err(ApiError::bad_request(
                    "Editing a project LoRA requires projectId",
                ));
            };
            project_path_for_id(state.clone(), project_id)
                .await?
                .join("loras")
                .join("manifest.jsonc")
        }
        "builtin" => {
            return Err(ApiError::bad_request(
                "Built-in LoRA metadata is read-only. Import a copy to add trigger keywords or notes.",
            ));
        }
        // sc-10452: no manifest backs an external row, so there is nothing to write to.
        crate::external_loras::EXTERNAL_SCOPE => {
            return Err(ApiError::bad_request(
                "LoRAs discovered in an external model folder are read-only. \
                 Import a copy to add trigger keywords or notes.",
            ));
        }
        _ => return Err(ApiError::bad_request("Unsupported LoRA scope")),
    };
    let trigger_words = body.trigger_words.clone();
    let notes = body.notes.clone();
    let updated = mutate_manifest_entries(&state, &manifest_path, "loras", move |entries| {
        let mut updated = None;
        let entries = entries
            .into_iter()
            .map(|mut entry| {
                if entry.get("id").and_then(Value::as_str) == Some(lora_id.as_str()) {
                    if let Some(object) = entry.as_object_mut() {
                        if let Some(trigger_words) = trigger_words.as_ref() {
                            object.insert("triggerWords".to_owned(), json!(trigger_words));
                        }
                        if let Some(notes) = notes.as_ref() {
                            object.insert("notes".to_owned(), Value::String(notes.clone()));
                        }
                        object.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
                    }
                    updated = Some(entry.clone());
                }
                entry
            })
            .collect::<Vec<_>>();
        Ok((entries, updated))
    })
    .await?;
    updated.map(Json).ok_or_else(|| ApiError {
        status: StatusCode::NOT_FOUND,
        detail: "LoRA has no editable manifest entry in this scope".to_owned(),
        context: None,
        code: None,
    })
}

/// Best-effort trigger-keyword suggestions for a LoRA, read live from the
/// installed adapter's embedded `ss_tag_frequency` metadata. Powers the tag
/// editor's click-to-add typeahead. Returns an empty list (never an error) when
/// the LoRA isn't installed or its file carries no such metadata.
pub(crate) async fn lora_embedded_tags(
    State(state): State<AppState>,
    Path(lora_id): Path<String>,
    Query(query): Query<LoraCatalogItemQuery>,
) -> Result<Json<Value>, ApiError> {
    let lora = lora_catalog(&state, query.project_id.as_deref())
        .await?
        .into_iter()
        .find(|item| {
            item.get("id").and_then(Value::as_str) == Some(lora_id.as_str())
                && query
                    .scope
                    .as_deref()
                    .is_none_or(|scope| item.get("scope").and_then(Value::as_str) == Some(scope))
        })
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "LoRA not found".to_owned(),
            context: None,
            code: None,
        })?;
    let Some(installed_path) = lora
        .get("installedPath")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return Ok(Json(json!({ "tags": [] })));
    };
    let tags = tokio::task::spawn_blocking(move || read_embedded_trigger_tags(&installed_path))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(Json(json!({ "tags": tags })))
}

/// Split a comma-delimited trigger-keyword string into a trimmed, de-duplicated
/// list. Multipart form fields arrive as one string (the web tag editor joins the
/// keyword chips with commas); the JSON import path sends a `triggerWords` array
/// and bypasses this.
fn parse_comma_keywords(value: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .filter(|token| seen.insert(token.to_owned()))
        .map(str::to_owned)
        .collect()
}

/// `ss_tag_frequency` can list hundreds of caption tokens; cap suggestions to the
/// most frequent so the editor stays usable.
const MAX_EMBEDDED_TAGS: usize = 40;

/// Resolve a LoRA's installed file and extract ranked candidate trigger tags from
/// its embedded metadata. Any failure yields an empty list, since this only feeds
/// a suggestion UI.
fn read_embedded_trigger_tags(installed_path: &str) -> Vec<String> {
    let Some(safetensors_path) = first_safetensors_path(FsPath::new(installed_path)) else {
        return Vec::new();
    };
    match read_safetensors_header(&safetensors_path) {
        Ok(header) => parse_tag_frequency(&header),
        Err(_) => Vec::new(),
    }
}

/// Parse `ss_tag_frequency` from a safetensors `__metadata__` header into a
/// frequency-ranked, de-duplicated tag list. The value is usually a JSON-encoded
/// string of shape `{ "<dataset_dir>": { "tag": count, .. }, .. }` (occasionally
/// the flat `{ "tag": count }`). Tolerant of a missing key, a non-string value,
/// invalid JSON, and either shape.
fn parse_tag_frequency(header: &Value) -> Vec<String> {
    let Some(raw) = header
        .get("__metadata__")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("ss_tag_frequency"))
    else {
        return Vec::new();
    };
    let decoded = match raw {
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(value) => value,
            Err(_) => return Vec::new(),
        },
        other => other.clone(),
    };
    let Some(outer) = decoded.as_object() else {
        return Vec::new();
    };
    let mut counts: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (key, value) in outer {
        match value {
            // Nested shape: dataset dir -> { tag: count }.
            Value::Object(inner) => {
                for (tag, count) in inner {
                    let tag = tag.trim();
                    if !tag.is_empty() {
                        *counts.entry(tag.to_owned()).or_default() += count.as_f64().unwrap_or(0.0);
                    }
                }
            }
            // Flat shape: tag -> count.
            Value::Number(number) => {
                let tag = key.trim();
                if !tag.is_empty() {
                    *counts.entry(tag.to_owned()).or_default() += number.as_f64().unwrap_or(0.0);
                }
            }
            _ => {}
        }
    }
    let mut ranked: Vec<(String, f64)> = counts.into_iter().collect();
    // Most frequent first; ties broken alphabetically for deterministic output.
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
        .into_iter()
        .map(|(tag, _)| tag)
        .take(MAX_EMBEDDED_TAGS)
        .collect()
}

#[cfg(test)]
mod embedded_tag_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_nested_tag_frequency_ranked_by_count() {
        let header = json!({
            "__metadata__": {
                "ss_tag_frequency": "{\"1_concept\": {\"rare_token\": 2, \"common_token\": 10}}"
            }
        });
        assert_eq!(
            parse_tag_frequency(&header),
            vec!["common_token", "rare_token"]
        );
    }

    #[test]
    fn parses_flat_tag_frequency() {
        let header = json!({
            "__metadata__": { "ss_tag_frequency": "{\"solo\": 5, \"1girl\": 9}" }
        });
        assert_eq!(parse_tag_frequency(&header), vec!["1girl", "solo"]);
    }

    #[test]
    fn merges_counts_across_dirs() {
        let header = json!({
            "__metadata__": {
                "ss_tag_frequency": "{\"a\": {\"shared\": 1}, \"b\": {\"shared\": 4, \"other\": 2}}"
            }
        });
        assert_eq!(parse_tag_frequency(&header), vec!["shared", "other"]);
    }

    #[test]
    fn tolerates_missing_and_malformed() {
        assert!(parse_tag_frequency(&json!({ "__metadata__": {} })).is_empty());
        assert!(parse_tag_frequency(&json!({})).is_empty());
        assert!(parse_tag_frequency(
            &json!({ "__metadata__": { "ss_tag_frequency": "not json" } })
        )
        .is_empty());
        assert!(
            parse_tag_frequency(&json!({ "__metadata__": { "ss_tag_frequency": 42 } })).is_empty()
        );
    }

    #[test]
    fn parse_comma_keywords_trims_and_dedupes() {
        assert_eq!(parse_comma_keywords(" a , b ,a,, c "), vec!["a", "b", "c"]);
    }
}

pub(crate) async fn lora_catalog(
    state: &AppState,
    project_id: Option<&str>,
) -> Result<Vec<Value>, ApiError> {
    // sc-8819 (F-017): observe full-catalog assembly (the per-LoRA FS install-state probe
    // sweep) so a test can assert it runs once per job-create.
    #[cfg(test)]
    crate::test_note_lora_catalog_build();
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.loras.jsonc"), "loras").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.loras.jsonc"), "loras").await?;
    let data_dir = state.settings.data_dir.clone();
    let builtin_manifest = manifest_dir.join("builtin.loras.jsonc");
    let user_manifest = manifest_dir.join("user.loras.jsonc");
    // sc-4202 (F-API-3): normalize_lora_entry probes the filesystem for installed
    // artifact paths; run the builtin+user normalize off the async executor.
    // Epic 10451 / sc-10452: adapters an operator already has on disk (a ComfyUI
    // `models/loras` tree). Empty unless `SCENEWORKS_EXTERNAL_MODEL_ROOTS` is set, so
    // the default catalog is unchanged. Scanned inside the same blocking task as the
    // manifest normalize sweep — both are filesystem-bound.
    let external_roots = state.settings.external_model_roots.clone();
    let external_cache = state.external_lora_cache.clone();
    let mut loras = {
        let data_dir = data_dir.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Value>, ApiError> {
            let mut loras = Vec::new();
            for lora in builtin {
                loras.push(normalize_lora_entry(
                    lora,
                    "builtin",
                    &builtin_manifest,
                    &data_dir,
                    &data_dir,
                )?);
            }
            let user = user
                .into_iter()
                .map(|lora| {
                    normalize_lora_entry(lora, "global", &user_manifest, &data_dir, &data_dir)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let loras = merge_entries_by_id(loras, user);
            // External rows carry their own `installedPath`/`installState` (the scan
            // proved the file exists), so they skip `normalize_lora_entry` — there is no
            // manifest behind them to resolve a relative path against. Merged last of the
            // global sources; their `external_…` ids cannot collide with a manifest entry.
            let external = {
                let mut cache = external_cache.lock();
                crate::external_loras::scan_external_loras(&external_roots, &mut cache)
            };
            Ok(merge_entries_by_id(loras, external))
        })
        .await
        .map_err(|err| ApiError::internal(format!("LoRA catalog normalize task failed: {err}")))??
    };
    if let Some(project_id) = project_id {
        let project_path = project_path_for_id(state.clone(), project_id).await?;
        let project_manifest = project_path.join("loras").join("manifest.jsonc");
        let entries = load_manifest_entries(state, &project_manifest, "loras").await?;
        let data_dir = data_dir.clone();
        let project_loras = tokio::task::spawn_blocking(move || -> Result<Vec<Value>, ApiError> {
            entries
                .into_iter()
                .map(|lora| {
                    normalize_lora_entry(
                        lora,
                        "project",
                        &project_manifest,
                        &project_path,
                        &data_dir,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .await
        .map_err(|err| {
            ApiError::internal(format!("LoRA catalog project normalize task failed: {err}"))
        })??;
        loras = merge_entries_by_id(loras, project_loras);
    }
    for lora in &mut loras {
        let object = lora
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("LoRA manifest entry must be an object"))?;
        let scope = object
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or("builtin");
        let installed = object
            .get("installState")
            .and_then(Value::as_str)
            .is_some_and(|state| state == "installed");
        let removable = match scope {
            // An external adapter lives in the user's own ComfyUI tree; we borrowed it,
            // we do not own it. `delete_lora` refuses the scope, so a Delete button here
            // could only ever 400 — and offering to remove someone else's file at all is
            // the wrong promise (epic 10451 / sc-10452).
            crate::external_loras::EXTERNAL_SCOPE => false,
            // A built-in is only removable once its weights are actually on disk.
            "builtin" => installed,
            _ => true,
        };
        object.insert("removable".to_owned(), Value::Bool(removable));
    }
    loras.sort_by(|left, right| {
        let left_key = (
            left.get("scope")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            left.get("family")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            left.get("name").and_then(Value::as_str).unwrap_or_default(),
        );
        let right_key = (
            right
                .get("scope")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            right
                .get("family")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            right
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        left_key.cmp(&right_key)
    });
    Ok(loras)
}

/// Return a worker-cached exact digest only while the marker still describes the selected file.
/// Missing markers (including external/shared model roots) simply leave the catalog hashless.
fn cached_lora_sha256(entry: &Value, installed_path: &FsPath, data_dir: &FsPath) -> Option<String> {
    let declared = entry
        .get("files")
        .and_then(Value::as_array)
        .and_then(|files| files.first())
        .or_else(|| entry.get("source").and_then(|source| source.get("file")))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let file = if installed_path.is_dir() {
        sceneworks_core::lora_family::resolve_adapter_in_dir(installed_path, declared)?
    } else {
        installed_path.to_path_buf()
    };
    let source = entry.get("source").and_then(Value::as_object);
    let hf_repo = source
        .and_then(|source| source.get("provider"))
        .or_else(|| entry.get("provider"))
        .and_then(Value::as_str)
        .filter(|provider| *provider == "huggingface")
        .and_then(|_| {
            source
                .and_then(|source| source.get("repo"))
                .or_else(|| entry.get("repo"))
                .and_then(Value::as_str)
        });
    let receipt_marker = entry
        .get("id")
        .and_then(Value::as_str)
        .zip(hf_repo)
        .map(|(id, repo)| {
            (
                data_dir
                    .join("loras")
                    .join(safe_download_dir(id))
                    .join(".sceneworks-download-complete.json"),
                repo,
            )
        })
        .filter(|(candidate, repo)| {
            std::fs::read(candidate)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .and_then(|marker| {
                    marker
                        .get("repo")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .as_deref()
                == Some(*repo)
        })
        .map(|(candidate, _)| candidate);
    let marker_path = receipt_marker.or_else(|| {
        file.ancestors()
            .take(8)
            .map(|directory| directory.join(".sceneworks-download-complete.json"))
            .find(|candidate| candidate.is_file())
    })?;
    let marker: Value = serde_json::from_slice(&std::fs::read(marker_path).ok()?).ok()?;
    let metadata = std::fs::metadata(&file).ok()?;
    let name = file.file_name()?.to_string_lossy();
    let modified_nanos = metadata
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos()
        .to_string();
    if marker.get("loraFileName").and_then(Value::as_str) != Some(name.as_ref())
        || marker.get("loraFileBytes").and_then(Value::as_u64) != Some(metadata.len())
        || marker.get("loraFileModifiedNanos").and_then(Value::as_str)
            != Some(modified_nanos.as_str())
    {
        return None;
    }
    let hash = marker.get("loraFileSha256").and_then(Value::as_str)?.trim();
    (hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| hash.to_ascii_lowercase())
}

pub(crate) fn normalize_lora_entry(
    mut lora: Value,
    scope: &str,
    manifest_path: &FsPath,
    default_root: &FsPath,
    data_dir: &FsPath,
) -> Result<Value, ApiError> {
    let object = lora
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("LoRA manifest entry must be an object"))?;
    object
        .entry("scope".to_owned())
        .or_insert_with(|| Value::String(scope.to_owned()));
    let source_path = object
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .or_else(|| object.get("path").and_then(Value::as_str));
    let local_path = source_path.map(|source_path| {
        let path = PathBuf::from(source_path);
        if path.is_absolute() {
            path
        } else {
            default_root.join(path)
        }
    });
    let lora_snapshot = Value::Object(object.clone());
    let installed_path = match local_path.as_ref() {
        Some(path) if lora_is_installed(path) => Some(path.clone()),
        _ => match lora_huggingface_cached_file(&lora_snapshot, data_dir) {
            Some(path) if lora_is_installed(&path) => Some(path),
            _ => local_path.clone(),
        },
    };
    let install_state = match installed_path.as_ref() {
        Some(path) if lora_is_installed(path) => "installed",
        _ => "missing",
    };
    let installed_hash = installed_path
        .as_deref()
        .and_then(|path| cached_lora_sha256(&lora_snapshot, path, data_dir));
    object.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    object.insert(
        "installedPath".to_owned(),
        installed_path
            .as_ref()
            .map(|path| Value::String(path.display().to_string()))
            .unwrap_or(Value::Null),
    );
    object.insert(
        "installState".to_owned(),
        Value::String(install_state.to_owned()),
    );
    if let Some(hash) = installed_hash {
        object.insert("sha256".to_owned(), Value::String(hash));
    } else {
        object.remove("sha256");
    }
    let requested_file_present =
        lora_huggingface_requested_file(&lora_snapshot, data_dir).is_some();
    object.insert(
        "updateAvailable".to_owned(),
        Value::Bool(
            install_state == "installed"
                && !requested_file_present
                && lora_is_huggingface(&lora_snapshot),
        ),
    );
    Ok(lora)
}

pub(crate) async fn create_lora_import_job(
    State(state): State<AppState>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    let is_multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("multipart/form-data"));
    if is_multipart {
        let multipart = Multipart::from_request(request, &state)
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()).into_response())?;
        let (payload, staged_paths) = lora_import_request_from_multipart(&state, multipart)
            .await
            .map_err(IntoResponse::into_response)?;
        let result = queue_lora_import_job(state, payload).await;
        if result.is_err() {
            for path in &staged_paths {
                cleanup_staged_lora_upload(path).await;
            }
        }
        return result.map_err(IntoResponse::into_response);
    }

    let payload = Json::<LoraImportRequest>::from_request(request, &state)
        .await
        .map(|Json(payload)| payload)
        .map_err(json_rejection_response)?;
    queue_lora_import_job(state, payload)
        .await
        .map_err(IntoResponse::into_response)
}

pub(crate) async fn queue_lora_import_job(
    state: AppState,
    mut payload: LoraImportRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    if option_str_is_empty(payload.repo.as_deref())
        && option_str_is_empty(payload.source_url.as_deref())
        && option_str_is_empty(payload.source_path.as_deref())
    {
        return Err(ApiError::bad_request(
            "Provide a Hugging Face repo, source URL, or source path",
        ));
    }
    if let Some(source_url) = payload.source_url.as_deref() {
        validate_source_url(source_url)?;
    }
    // Licence acknowledgment, keyed on the repo this import will FETCH (sc-17227) — the same
    // predicate `POST /api/v1/jobs` and `POST /api/v1/models/import` apply.
    //
    // This route had no licence logic of any kind, and the reason first given for leaving it out —
    // that no LoRA in the catalog declares the flag or names an H3 repo — was wrong on its own
    // terms: nothing here ever consults the LoRA catalog for the repo. The worker
    // (`run_lora_import_job`) takes the payload's `repo` verbatim, and with an empty `files` list
    // `HuggingFaceSnapshot::resolve` + `download_snapshot` pull the WHOLE repo. So
    // `{"repo": "MiniMaxAI/MiniMax-H3"}` fetched the restricted weights here while the identical
    // `lora_import` job posted to `/api/v1/jobs` was answered 403. What the LoRA catalog contains
    // has no bearing on what this route can reach; the repo the caller names is the whole of it.
    crate::models::ensure_license_acknowledged_for_source(
        &state,
        &[payload.repo.as_deref()],
        payload.source_url.as_deref(),
        payload.license_acknowledged,
    )
    .await?;
    if !matches!(payload.scope.as_str(), "global" | "project") {
        return Err(ApiError::bad_request(
            "LoRA scope must be global or project",
        ));
    }
    if let Some(family) = payload.family.take() {
        let models = model_catalog(&state).await?;
        payload.family = Some(validate_lora_family(&models, &family)?);
    }
    let name = payload
        .name
        .clone()
        .or_else(|| payload.repo.clone())
        .or_else(|| {
            payload
                .source_url
                .as_deref()
                .and_then(|value| lora_source_url_file_stem(value).ok())
        })
        .or_else(|| {
            payload.source_path.as_deref().and_then(|path| {
                FsPath::new(path)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(str::to_owned)
            })
        })
        .unwrap_or_else(|| "Imported LoRA".to_owned());
    // Scope-derived roots and the manifest path do not depend on the LoRA id, so
    // resolve them first. The id is minted *after* family resolution (below) so the
    // target folder can carry the canonical family token (sc-10214): a krea_2 and a
    // flux2 LoRA that share a display name (e.g. two "Realism Engine" variants) then
    // land in separate `<family>_<slug>` folders instead of co-mingling their
    // safetensors in one dir, where the header inspector picks a file non-deterministically.
    let (
        loras_base_dir,
        manifest_path,
        path_prefix,
        project_id,
        project_name,
        allowed_source_roots,
    ) = if payload.scope == "project" {
        let Some(project_id) = payload.project_id.clone() else {
            return Err(ApiError::bad_request(
                "Project LoRA imports require projectId",
            ));
        };
        let project_path = project_path_for_id(state.clone(), &project_id).await?;
        (
            project_path.join("loras").join("imports"),
            project_path.join("loras").join("manifest.jsonc"),
            "loras/imports".to_owned(),
            Some(project_id),
            None,
            vec![
                state.settings.data_dir.join("loras"),
                project_path.join("loras"),
            ],
        )
    } else {
        (
            state.settings.data_dir.join("loras"),
            state
                .settings
                .config_dir
                .join("manifests")
                .join("user.loras.jsonc"),
            "loras".to_owned(),
            None,
            None,
            vec![state.settings.data_dir.join("loras")],
        )
    };
    // Resolve the family before the id: a local file's architecture is detected from
    // its safetensors header and reconciled against any user-declared family so the id
    // below can carry the canonical family token (sc-10214).
    let source_roots = if payload.uploaded_source_path {
        vec![state.settings.data_dir.join("cache").join("lora-uploads")]
    } else {
        allowed_source_roots
    };
    let mut adapter_metadata = AdapterFileMetadata::default();
    if let Some(local_source) = payload.source_path.clone() {
        let secondary_source = payload.secondary_source_path.clone();
        let h3_expected = payload
            .family
            .as_deref()
            .is_some_and(|family| canonical_lora_family(family) == "minimax-h3");
        let (local_source, detected, declared) = tokio::task::spawn_blocking(move || {
            validate_lora_import_source_path(&local_source, &source_roots)?;
            // A paired Wan A14B MoE upload (sc-1991) carries a second low-noise
            // file; validate it against the same upload root.
            if let Some(secondary_source) = secondary_source.as_deref() {
                validate_lora_import_source_path(secondary_source, &source_roots)?;
            }
            inspect_lora_source_for_family(&local_source, h3_expected)
                .map(|(detected, declared)| (local_source, detected, declared))
        })
        .await
        .map_err(|error| {
            ApiError::internal(format!("LoRA import inspection task failed: {error}"))
        })??;
        adapter_metadata = declared;
        payload.family = reconcile_lora_family(
            payload.family.take(),
            detected,
            &format!("source_path={local_source}"),
        )?;
    }
    // Mint the id (see `derive_lora_id`): explicit caller id wins, else a
    // family-scoped `<family>_<slug>` so folders never collide across families.
    let lora_id = derive_lora_id(payload.lora_id.as_deref(), &name, payload.family.as_deref());
    let target_name = safe_download_dir(&lora_id);
    let target_dir = loras_base_dir.join(&target_name);
    let source_path = format!("{path_prefix}/{target_name}");
    // Record the paired Wan MoE halves now that the family-scoped target name exists.
    if payload.source_path.is_some() && payload.secondary_source_path.is_some() {
        let (high_name, low_name) = wan_moe_pair_filenames(&target_name);
        payload.files = vec![high_name, low_name];
    }
    // Belt-and-suspenders against co-mingling different families in one record folder
    // (sc-10214): reject when the resolved folder already holds a safetensors of a
    // *different* family, with an honest message instead of stacking two adapters the
    // header inspector would arbitrate non-deterministically. Same-family re-imports
    // (the folder's existing file matches) are still allowed.
    if let Some(family) = payload.family.clone() {
        let target_dir_for_inspection = target_dir.clone();
        let family_for_inspection = family.clone();
        let existing = tokio::task::spawn_blocking(move || {
            conflicting_folder_family(&target_dir_for_inspection, &family_for_inspection)
        })
        .await
        .map_err(|error| {
            ApiError::internal(format!("LoRA target inspection task failed: {error}"))
        })??;
        if let Some(existing) = existing {
            return Err(ApiError::bad_request(format!(
                "The folder for LoRA '{lora_id}' already contains a {existing} LoRA. \
                 Import this {family} LoRA under a different name so each family keeps its own folder."
            )));
        }
    }
    let timestamp = now_rfc3339();
    let mut manifest_entry = json!({
        "id": lora_id,
        "name": name,
        "scope": payload.scope.clone(),
        "source": {
            "provider": lora_source_provider(&payload),
            "repo": payload.repo.clone(),
            "path": source_path,
        },
        "files": payload.files.clone(),
        // Always recorded (empty when unset) so imported entries match the built-in
        // catalog shape and the required `triggerWords` contract field.
        "triggerWords": payload.trigger_words.clone(),
        "createdAt": timestamp,
        "updatedAt": timestamp,
    });
    if let Some(source_url) = payload.source_url.clone() {
        if let Some(source) = manifest_entry
            .get_mut("source")
            .and_then(Value::as_object_mut)
        {
            source.insert("url".to_owned(), Value::String(source_url));
        }
    }
    if let Some(family) = payload.family.clone() {
        if let Some(object) = manifest_entry.as_object_mut() {
            object.insert("family".to_owned(), Value::String(family));
        }
    }
    if let Some(base_model) = payload.base_model.clone() {
        if let Some(object) = manifest_entry.as_object_mut() {
            object.insert("baseModel".to_owned(), Value::String(base_model));
        }
    }
    // What the adapter file itself declares (sc-14057): the network type the worker's
    // `classify_adapter` keys the engine adapter `kind` off, plus rank/alpha. Recorded so the
    // imported entry describes the file rather than losing facts the header already carried.
    //
    // `adapter_metadata` is empty for a repo/URL import — there is no file on disk at queue time —
    // so the worker fills the same fields in from the downloaded adapter through this same writer
    // once the transfer lands. Shared deliberately: an adapter must be described identically
    // whichever route ingested it.
    if let Some(object) = manifest_entry.as_object_mut() {
        apply_adapter_metadata_to_manifest_entry(object, &adapter_metadata);
    }
    let trimmed_notes = payload.notes.trim();
    if !trimmed_notes.is_empty() {
        if let Some(object) = manifest_entry.as_object_mut() {
            object.insert("notes".to_owned(), Value::String(trimmed_notes.to_owned()));
        }
    }
    let mut payload = to_json_object(&payload)?;
    payload.insert("loraId".to_owned(), manifest_entry["id"].clone());
    payload.insert("name".to_owned(), manifest_entry["name"].clone());
    payload.insert(
        "targetDir".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    payload.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    payload.insert("manifestEntry".to_owned(), manifest_entry);
    let job = create_generation_job(
        state,
        JobType::LoraImport,
        project_id,
        project_name,
        payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

pub(crate) async fn lora_import_request_from_multipart(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<(LoraImportRequest, Vec<PathBuf>), ApiError> {
    let mut payload = LoraImportRequest {
        lora_id: None,
        name: None,
        repo: None,
        source_url: None,
        source_path: None,
        files: Vec::new(),
        trigger_words: Vec::new(),
        notes: String::new(),
        family: None,
        base_model: None,
        expected_sha256: None,
        license_acknowledged: false,
        scope: default_lora_scope(),
        project_id: None,
        uploaded_source_path: false,
        secondary_source_path: None,
    };
    let mut staged_path = None;
    // Wan A14B MoE imports (sc-1991) carry a second `secondaryFile` part for the
    // low-noise expert half. Staged separately so a failed queue cleans up both.
    let mut secondary_staged_path = None;

    let parse_result = async {
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            let field_name = field.name().unwrap_or("").to_owned();
            if field_name == "file" {
                if staged_path.is_some() {
                    return Err(ApiError::bad_request("Only one LoRA file can be uploaded"));
                }
                let upload_name =
                    sanitized_upload_filename(field.file_name().unwrap_or("lora.safetensors"));
                let path =
                    write_lora_upload_field_to_staged_file(state, field, &upload_name).await?;
                payload.source_path = Some(path.display().to_string());
                payload.files = vec![upload_name];
                payload.uploaded_source_path = true;
                staged_path = Some(path);
                continue;
            }
            if field_name == "secondaryFile" {
                if secondary_staged_path.is_some() {
                    return Err(ApiError::bad_request(
                        "Only one low-noise expert file can be uploaded",
                    ));
                }
                let upload_name =
                    sanitized_upload_filename(field.file_name().unwrap_or("low_noise.safetensors"));
                let path =
                    write_lora_upload_field_to_staged_file(state, field, &upload_name).await?;
                payload.secondary_source_path = Some(path.display().to_string());
                secondary_staged_path = Some(path);
                continue;
            }

            let value = field
                .text()
                .await
                .map_err(|error| ApiError::bad_request(error.to_string()))?;
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            match field_name.as_str() {
                "loraId" => payload.lora_id = Some(value.to_owned()),
                "name" => payload.name = Some(value.to_owned()),
                "family" => payload.family = Some(value.to_owned()),
                "baseModel" => payload.base_model = Some(value.to_owned()),
                // The web tag editor joins keyword chips with commas; the JSON import
                // path instead sends a `triggerWords` array and bypasses this parser.
                "triggerWords" => payload.trigger_words = parse_comma_keywords(value),
                "notes" => payload.notes = value.to_owned(),
                "scope" => payload.scope = value.to_owned(),
                "projectId" => payload.project_id = Some(value.to_owned()),
                // Accepted for parity with the model-import parser (sc-17227). Unlike that one,
                // this parser accepts no `repo`/`sourceUrl` and rejects a request without an
                // upload `file` below, so the licence gate never has a candidate here and this
                // assertion cannot currently be needed. Parsed anyway so the field is not the
                // missing half if a remote source is ever added to this form — the same reason
                // `_ => {}` silently dropping it would be the wrong default.
                "licenseAcknowledged" => {
                    payload.license_acknowledged = value.eq_ignore_ascii_case("true")
                }
                _ => {}
            }
        }
        Ok(())
    }
    .await;
    let staged_paths: Vec<PathBuf> = staged_path
        .iter()
        .chain(secondary_staged_path.iter())
        .cloned()
        .collect();
    if let Err(error) = parse_result {
        for path in &staged_paths {
            cleanup_staged_lora_upload(path).await;
        }
        return Err(error);
    }

    if staged_path.is_none() {
        for path in &staged_paths {
            cleanup_staged_lora_upload(path).await;
        }
        return Err(ApiError::bad_request("Upload file field is required"));
    }
    Ok((payload, staged_paths))
}

pub(crate) async fn write_lora_upload_field_to_staged_file(
    state: &AppState,
    field: axum::extract::multipart::Field<'_>,
    filename: &str,
) -> Result<PathBuf, ApiError> {
    let upload_dir = state
        .settings
        .data_dir
        .join("cache")
        .join("lora-uploads")
        .join(format!("upload-{}", Uuid::new_v4().simple()));
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let temp_path = upload_dir.join(filename);
    // sc-8886 (F-084): shared streaming writer. Cleanup removes the staged file AND its
    // per-upload parent directory, so an aborted upload leaves no `upload-<uuid>/` dir.
    stream_multipart_field_to_file(
        field,
        &temp_path,
        max_lora_upload_bytes(),
        || "Uploaded LoRA file exceeds the 2GB limit".to_owned(),
        || cleanup_staged_lora_upload(&temp_path),
    )
    .await?;
    Ok(temp_path)
}

pub(crate) async fn cleanup_staged_lora_upload(path: &FsPath) {
    let _ = tokio::fs::remove_file(path).await;
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::remove_dir(parent).await;
    }
}

pub(crate) fn max_lora_upload_bytes() -> usize {
    #[cfg(test)]
    {
        let limit = TEST_MAX_LORA_UPLOAD_BYTES.with(std::cell::Cell::get);
        if limit > 0 {
            return limit;
        }
    }
    MAX_UPLOAD_BYTES
}

pub(crate) async fn validate_job_lora_compatibility(
    state: &AppState,
    project_id: Option<&str>,
    job_payload: &mut JsonObject,
    allow_inline_loras: bool,
) -> Result<(), ApiError> {
    validate_job_lora_compatibility_with(state, project_id, job_payload, allow_inline_loras, None)
        .await
}

/// `validate_job_lora_compatibility` with an optional caller-supplied catalog snapshot
/// (sc-8819). When `snapshot` is `Some`, the model and LoRA catalogs it exposes are reused
/// instead of re-running the per-model/per-LoRA filesystem install-state probes; the
/// validation result is identical to the `None` path.
pub(crate) async fn validate_job_lora_compatibility_with(
    state: &AppState,
    project_id: Option<&str>,
    job_payload: &mut JsonObject,
    allow_inline_loras: bool,
    snapshot: Option<&JobCatalogSnapshot>,
) -> Result<(), ApiError> {
    let Some(loras) = job_payload
        .get("loras")
        .and_then(Value::as_array)
        .filter(|loras| !loras.is_empty())
        .cloned()
    else {
        return Ok(());
    };
    let model_id = job_payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("Model is required for LoRA compatibility"))?;
    let model_id = model_id.to_owned();
    let models = match snapshot {
        Some(snapshot) => Arc::new(snapshot.models(state).await?.to_vec()),
        None => Arc::new(model_catalog(state).await?),
    };
    let catalog_loras = match snapshot {
        Some(snapshot) => snapshot.loras(state, project_id).await?,
        None => Arc::new(lora_catalog(state, project_id).await?),
    };
    // sc-4202 (F-API-3): validate_lora_specs_for_model reads safetensors headers off
    // disk (validate_lora_safetensors_header) inline. Run it on the blocking pool so a
    // slow/network volume can't stall a tokio worker thread on the job-creation path.
    let normalized = tokio::task::spawn_blocking(move || {
        validate_lora_specs_for_model(
            models.as_slice(),
            catalog_loras.as_slice(),
            &model_id,
            &loras,
            allow_inline_loras,
            "LoRA",
        )
    })
    .await
    .map_err(|err| {
        ApiError::internal(format!("LoRA compatibility validation task failed: {err}"))
    })??;
    job_payload.insert("loras".to_owned(), Value::Array(normalized));
    Ok(())
}

pub(crate) fn validate_lora_specs_for_model(
    models: &[Value],
    catalog_loras: &[Value],
    model_id: &str,
    attached_loras: &[Value],
    allow_inline_loras: bool,
    lora_label: &str,
) -> Result<Vec<Value>, ApiError> {
    if attached_loras.is_empty() {
        return Ok(Vec::new());
    }
    let Some(model) = models
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
    else {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} not found; cannot verify LoRA compatibility"
        )));
    };
    // The families this model can LOAD — declared plus extra-compatible (sc-15017). Using the
    // declared set here made the gate stricter than the registry the worker and engine honor.
    let model_families = crate::accepted_model_lora_families(model);
    if model_families.is_empty() {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} has no declared LoRA families"
        )));
    }
    // The model's OWN declared family drives the base-model gate below (an extra-compatible family
    // is not the model's identity), so keep it separate from the accepted set.
    let model_own_family = model
        .get("family")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| model_lora_families(model).into_iter().next())
        .unwrap_or_default();
    let mut normalized_loras = Vec::new();
    for attached_lora in attached_loras {
        let Some((lora_id, lora, normalized_lora, catalog_backed)) =
            hydrate_lora_spec(catalog_loras, attached_lora, allow_inline_loras, lora_label)?
        else {
            continue;
        };
        let install_state = lora.get("installState").and_then(Value::as_str);
        if install_state.is_some_and(|state| state != "installed")
            || (catalog_backed && install_state.is_none())
        {
            return Err(ApiError::bad_request(format!(
                "{lora_label} is not installed: {lora_id}"
            )));
        }
        let header = validate_lora_safetensors_header(lora_id, lora)?;
        if let Some(header) = &header {
            let h3_expected = model_families.iter().any(|family| family == "minimax-h3");
            validate_minimax_h3_trainer_header(header, h3_expected).map_err(|error| {
                ApiError::bad_request(format!(
                    "LoRA {lora_id} has an unsupported MiniMax-H3 adapter layout: {error}"
                ))
            })?;
        }
        if let Some(detected_family) = header.as_ref().and_then(detect_lora_family) {
            // `model_families` are normalized (via `model_lora_families` →
            // `normalize_lora_family`, `_`→`-`), but `detect_lora_family` returns the catalog/
            // trainer-verbatim token (e.g. `krea_2`, underscore — chosen so import-time
            // reconciliation matches the catalog string raw). Normalize the detected family to the
            // same canonical form before the membership test, else a krea_2 LoRA is falsely
            // rejected against a `krea-2`-normalized model surface (sc-8185).
            let detected_family = normalize_lora_family(&detected_family);
            // The detected family is the LoRA file's *base architecture*, which is not always the
            // model's own family token. A model whose weights ARE some base architecture loads that
            // architecture's LoRAs — Krea Realtime is Wan 2.1 T2V 14B weight-for-weight, SCAIL-2 is
            // Wan2.1-I2V-derived — and that one-directional relation is registered in
            // `extra_compatible_lora_families`, deliberately kept OUT of the manifest's
            // `loraCompatibility.families` so it does not leak into the other family-keyed gates.
            // This test previously read the manifest list alone, so it was blind to that registry
            // and rejected exactly the cross-architecture LoRAs the engines are built to load
            // (sc-18200). Widen it to the model's full accepted set: its declared families plus
            // whatever each of those may additionally load.
            // Strictly ADDITIVE: keep `model_families` verbatim and append only the registry
            // extras. Replacing the list instead would silently re-spell it — `model_families` is
            // in `normalize_lora_family` space (which canonicalizes `krea-2` -> `krea_2`) while
            // `accepted_lora_families` normalizes to model space (`krea-2`), so round-tripping the
            // declared families through it turns a passing `krea_2` match into a rejection
            // (sc-8185's regression). The extras come back through `normalize_lora_family` for the
            // same reason, so both sides of the comparison stay in one canonical space.
            let accepted_families = model_families
                .iter()
                .cloned()
                .chain(
                    model_families
                        .iter()
                        .flat_map(|family| accepted_lora_families(family))
                        .map(|family| normalize_lora_family(&family)),
                )
                .collect::<Vec<_>>();
            if !accepted_families
                .iter()
                .any(|model_family| model_family == &detected_family)
            {
                let model_family_list = model_families.join(", ");
                return Err(ApiError::bad_request(format!(
                    "LoRA {lora_id} appears to be a {detected_family} LoRA, which is not compatible with model {model_id} ({model_family_list})"
                )));
            }
        }
        let families = lora_families(lora);
        if families.is_empty() {
            return Err(ApiError::bad_request(format!(
                "LoRA {lora_id} has no declared family; cannot verify compatibility with model {model_id}"
            )));
        }
        if !families.iter().any(|family| {
            model_families
                .iter()
                .any(|model_family| model_family == family)
        }) {
            return Err(ApiError::bad_request(format!(
                "LoRA {lora_id} is not compatible with model {model_id}"
            )));
        }
        // ── Declared-partition gating (sc-19563), the FAMILY-AGNOSTIC arm ──────────────────────
        //
        // The gate below this one is hardcoded to `wan-video`. This one deliberately is not, and
        // that is the whole point of the story: it fires on the LoRA's own `modelIds` declaration,
        // whatever family it belongs to, so closing the next family's version of this gap is a
        // manifest edit rather than a third hardcoded arm here.
        //
        // What it closes: `family` cannot express a partition. MiniMax-H3 publishes `minimax_h3`
        // (t2va/fl2va) and `minimax_h3_ref` (ref2va) as ONE DiT architecture with ONE geometry, so
        // both declare `family: minimax-h3` and family detection cannot separate them — correctly,
        // because they really are the same architecture. But lightx2v distils the fl2v and ref2v
        // turbo adapters for one partition each, and cross-selecting one folds CLEANLY: no shape
        // error, no refusal, just a quality mismatch. That is what makes it easy to ship and hard
        // to notice.
        //
        // Absent `modelIds` means family gating alone, so no existing entry is tightened.
        let declared_model_ids = lora_model_ids(lora);
        if !declared_model_ids.is_empty() && !declared_model_ids.iter().any(|id| id == model_id) {
            return Err(ApiError::bad_request(format!(
                "LoRA {lora_id} is declared for model {}, not {model_id}. These are separate \
                 partitions of the same model family, so the adapter would attach and fold \
                 cleanly at the wrong quality rather than fail — which is why the pairing is \
                 enforced here rather than left to the family check.",
                declared_model_ids.join(" or ")
            )));
        }
        // Base-model gating: for families where a matching family is insufficient
        // (Wan 5B vs 14B both declare `wan-video` but have incompatible
        // architectures — 48 vs 16 latent channels), a LoRA that records its
        // trained base model only loads on that exact model. LoRAs without a
        // recorded base model fall back to family gating (legacy/imported), so this
        // never tightens behavior for existing LoRAs.
        //
        // Kept alongside the declared-partition gate above rather than merged into it: this one
        // reads what an adapter RECORDS about its own training, that one reads what a catalog
        // author DECLARES. An imported Wan LoRA has the former and no catalog entry at all.
        if families.iter().any(|family| family == "wan-video") {
            if let Some(base_model) = lora_base_model(lora) {
                // Shared with the worker's own gate (sc-15017): exact-id equality, plus the
                // extra-compatible arm that keeps the 5B-vs-14B split while letting a Wan-14B
                // LoRA onto a 14B-class model that accepts `wan-video` through the registry —
                // its recorded base can never equal that model's id.
                if !sceneworks_core::lora_family::base_model_satisfies_gate(
                    &model_own_family,
                    model_id,
                    &base_model,
                ) {
                    return Err(ApiError::bad_request(format!(
                        "LoRA {lora_id} was trained for base model {base_model}, not {model_id}; \
                         Wan 5B and 14B LoRAs are not interchangeable"
                    )));
                }
            }
        }
        normalized_loras.push(normalized_lora);
    }
    Ok(normalized_loras)
}

pub(crate) fn hydrate_lora_spec<'a>(
    catalog_loras: &'a [Value],
    attached_lora: &'a Value,
    allow_inline_loras: bool,
    lora_label: &str,
) -> Result<Option<(&'a str, &'a Value, Value, bool)>, ApiError> {
    let Some(lora_id) = job_lora_id(attached_lora) else {
        return Ok(None);
    };
    let catalog_lora = if allow_inline_loras {
        None
    } else {
        catalog_loras
            .iter()
            .find(|lora| lora.get("id").and_then(Value::as_str) == Some(lora_id))
    };
    if catalog_lora.is_none() && !allow_inline_loras {
        return Err(ApiError::bad_request(format!(
            "{lora_label} not found: {lora_id}"
        )));
    }
    let source_lora = catalog_lora.unwrap_or(attached_lora);
    let normalized_lora = match catalog_lora {
        Some(catalog_lora) => serialize_job_lora(catalog_lora, attached_lora, lora_id),
        None => normalize_inline_job_lora(attached_lora, lora_id),
    };
    Ok(Some((
        lora_id,
        source_lora,
        normalized_lora,
        catalog_lora.is_some(),
    )))
}

/// Returns the parsed safetensors header for `lora` when one is available
/// on disk. Returns `Ok(None)` when the manifest entry has no installed
/// path or no `.safetensors` file is present under it (the same "skip"
/// semantics this helper has always had). Returns an error if the file
/// exists but the header is malformed.
pub(crate) fn validate_lora_safetensors_header(
    lora_id: &str,
    lora: &Value,
) -> Result<Option<Value>, ApiError> {
    let Some(path) = lora
        .get("installedPath")
        .or_else(|| lora.get("sourcePath"))
        .or_else(|| lora.get("path"))
        .and_then(Value::as_str)
    else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let Some(safetensors_path) = first_safetensors_path(&path) else {
        return Ok(None);
    };
    read_safetensors_header_for_api(lora_id, &safetensors_path).map(Some)
}

pub(crate) fn read_safetensors_header_for_api(
    lora_id: &str,
    path: &FsPath,
) -> Result<Value, ApiError> {
    read_safetensors_header(path).map_err(|error| match error {
        SafetensorsHeaderError::Io(io_error) => {
            ApiError::bad_request(format!("Unable to inspect LoRA {lora_id}: {io_error}"))
        }
        SafetensorsHeaderError::InvalidHeader => {
            ApiError::bad_request(format!("LoRA {lora_id} has an invalid safetensors header"))
        }
        SafetensorsHeaderError::IncompleteData { declared, actual } => {
            ApiError::bad_request(format!(
                "LoRA {lora_id} is incomplete or corrupt ({actual} bytes on disk, but its header \
             declares at least {declared} bytes of tensor data); the file was likely truncated \
             during download. Re-import the complete file."
            ))
        }
    })
}

pub(crate) fn sweep_stale_lora_uploads(data_dir: &FsPath) -> std::io::Result<usize> {
    sweep_stale_lora_uploads_before(
        data_dir,
        SystemTime::now() - Duration::from_secs(STALE_UPLOAD_SECONDS),
    )
}

pub(crate) fn sweep_stale_lora_uploads_before(
    data_dir: &FsPath,
    cutoff: SystemTime,
) -> std::io::Result<usize> {
    // sc-8885 (F-083): LoRA uploads only ever stage per-upload directories, so the
    // shared sweeper's file+dir handling is a strict superset of the old dir-only loop.
    sweep_stale_uploads(data_dir, "lora-uploads", cutoff)
}

pub(crate) fn lora_source_provider(payload: &LoraImportRequest) -> &'static str {
    if payload.repo.is_some() {
        "huggingface"
    } else if payload.source_url.is_some() {
        "url"
    } else {
        "local"
    }
}

/// The `<stem>.high_noise.safetensors` / `<stem>.low_noise.safetensors` filenames
/// for a paired Wan A14B MoE LoRA stored under one record (sc-1991). The high-noise
/// file sorts first, so it resolves as the primary (transformer) and the low-noise
/// file as the `transformer_2` sibling. Must match the worker's identical
/// convention so the manifest `files` agree with the on-disk layout.
pub(crate) fn wan_moe_pair_filenames(stem: &str) -> (String, String) {
    (
        format!("{stem}.high_noise.safetensors"),
        format!("{stem}.low_noise.safetensors"),
    )
}

pub(crate) fn lora_url_error_message(error: LoraUrlError) -> &'static str {
    error.message()
}

/// Parses the safetensors header at `source_path` (or the first
/// `.safetensors` file under it) and runs the architecture detector.
/// Returns `Ok(None)` when no header is available or the signature is
/// inconclusive. Returns `Err` only when the file exists but its header
/// is malformed — that mirrors the pre-existing validation behaviour and
/// gives the user a clear "the file is broken" message instead of a
/// silent acceptance.
pub(crate) fn detect_family_from_local_path(source_path: &str) -> Result<Option<String>, ApiError> {
    Ok(inspect_lora_source(source_path)?.0)
}

/// One header read, two answers: the detected architecture family **and** what the file declares
/// about itself in `__metadata__` (network type / rank / alpha, sc-14057).
///
/// Kept together deliberately — the import path needs both, and the safetensors header is the same
/// (up to 16 MiB) parse for each. Either half may legitimately be absent: a file with no
/// recognizable architecture returns `None`, and one with a bare `{"format": "pt"}` metadata block
/// returns an empty [`AdapterFileMetadata`] rather than invented defaults.
pub(crate) fn inspect_lora_source(
    source_path: &str,
) -> Result<(Option<String>, AdapterFileMetadata), ApiError> {
    inspect_lora_source_for_family(source_path, false)
}

fn inspect_lora_source_for_family(
    source_path: &str,
    h3_expected: bool,
) -> Result<(Option<String>, AdapterFileMetadata), ApiError> {
    let path = FsPath::new(source_path);
    let Some(safetensors_path) = first_safetensors_path(path) else {
        return Ok((None, AdapterFileMetadata::default()));
    };
    let header = read_safetensors_header(&safetensors_path).map_err(|error| match error {
        SafetensorsHeaderError::Io(io_error) => {
            ApiError::bad_request(format!("Unable to inspect LoRA file: {io_error}"))
        }
        SafetensorsHeaderError::InvalidHeader => {
            ApiError::bad_request("LoRA file has an invalid safetensors header".to_owned())
        }
        SafetensorsHeaderError::IncompleteData { declared, actual } => {
            ApiError::bad_request(format!(
            "LoRA file is incomplete or corrupt ({actual} bytes on disk, but its header declares \
             at least {declared} bytes of tensor data); the file was likely truncated during \
             download. Re-import the complete file."
        ))
        }
    })?;
    validate_minimax_h3_trainer_header(&header, h3_expected).map_err(|error| {
        ApiError::bad_request(format!(
            "Unsupported MiniMax-H3 adapter namespace or layout: {error}"
        ))
    })?;
    Ok((detect_lora_family(&header), read_adapter_metadata(&header)))
}

/// Applies the import-time family policy: confident detection rejects a
/// mismatched user-supplied family; an unsupplied family is filled in from
/// the detection; an inconclusive detection logs a warning and accepts the
/// supplied family unchanged.
pub(crate) fn reconcile_lora_family(
    supplied: Option<String>,
    detected: Option<String>,
    context: &str,
) -> Result<Option<String>, ApiError> {
    // Log the inconclusive case (a supplied family with no confident detection)
    // before delegating, so the operational signal is preserved.
    if let (Some(supplied), None) = (&supplied, &detected) {
        tracing::info!(
            event = "lora_import_architecture_inconclusive",
            context = %context,
            family = %supplied,
            "LoRA import: architecture detection inconclusive; accepting supplied family"
        );
    }
    // Shared policy + canonicalization: spelling variants of one family (Krea 2's
    // `krea2`/`krea-2`/`krea_2`) reconcile instead of being rejected, and the result
    // is the canonical stored token.
    reconcile_detected_family(supplied, detected).map_err(|mismatch| {
        ApiError::bad_request(format!(
            "LoRA file appears to be a {} model, but family was declared as {}. Re-import with family {} or pick a different file.",
            mismatch.detected, mismatch.supplied, mismatch.detected
        ))
    })
}

/// Derives the stored LoRA id (which also names its on-disk folder via
/// `safe_download_dir`) for an import (sc-10214).
///
/// An explicit caller-supplied id wins — programmatic callers own their id. Otherwise
/// the display name is slugified and, when the family is known, prefixed with the
/// canonical family token (itself slugified so a hyphenated/dotted token like `z-image`
/// or `sd1.5` yields a clean all-underscore id) so `<family>_<slug>` folders never
/// collide across families: a krea_2 and a flux2 LoRA that share a display name (e.g.
/// two "Realism Engine" variants) resolve to `krea_2_realism_engine` and
/// `flux2_realism_engine` rather than one shared folder, and a z-image LoRA to
/// `z_image_<slug>`. An unresolved family (HF/URL imports) falls back to the bare slug.
pub(crate) fn derive_lora_id(
    explicit_id: Option<&str>,
    name: &str,
    family: Option<&str>,
) -> String {
    if let Some(explicit) = explicit_id {
        return explicit.to_owned();
    }
    let slug = slugify_lora_id(name);
    match family {
        Some(family) => format!(
            "{}_{}",
            slugify_lora_id(&canonical_lora_family(family)),
            slug
        ),
        None => slug,
    }
}

/// Returns the family already occupying `target_dir` when it differs from
/// `incoming_family` (sc-10214), so the caller can reject the import instead of letting
/// two families share one record folder — where the header inspector (`first_safetensors_path`)
/// would pick a file non-deterministically. Returns `Ok(None)` for an empty folder or a
/// same-family folder (a legitimate re-import/update).
fn conflicting_folder_family(
    target_dir: &FsPath,
    incoming_family: &str,
) -> Result<Option<String>, ApiError> {
    let Some(existing) = detect_family_from_local_path(target_dir.to_string_lossy().as_ref())?
    else {
        return Ok(None);
    };
    if canonical_lora_family(&existing) != canonical_lora_family(incoming_family) {
        Ok(Some(existing))
    } else {
        Ok(None)
    }
}

pub(crate) fn lora_is_installed(path: &FsPath) -> bool {
    first_safetensors_path(path).is_some()
}

pub(crate) fn lora_artifact_paths(lora: &Value, default_root: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let is_huggingface_source = lora
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("provider"))
        .or_else(|| lora.get("provider"))
        .and_then(Value::as_str)
        == Some("huggingface");
    if !is_huggingface_source {
        if let Some(installed_path) = lora
            .get("installedPath")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty() && !value.contains("${"))
        {
            paths.push(PathBuf::from(installed_path));
        }
    }
    if let Some(source_path) = lora
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .or_else(|| lora.get("path").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.contains("${"))
    {
        let path = PathBuf::from(source_path);
        paths.push(if path.is_absolute() {
            path
        } else {
            default_root.join(path)
        });
    }
    unique_paths(paths)
}

/// Preview-sample directories a training run may have parked OUTSIDE the LoRA's own output dir.
///
/// A global-scope training run writes its adapter to `<data>/loras/<lora_id>` but its step previews
/// to `<project>/training/samples/<lora_id>` — they have to live inside a project tree or the
/// Training Studio cannot render them (media is only served from
/// `GET /api/v1/projects/:project_id/files/*`; see `resolve_sample_root` in
/// crates/sceneworks-worker/src/training_jobs.rs). Nothing in the global LoRA manifest entry records
/// which project ran the training, so deleting the LoRA would otherwise orphan those PNGs forever.
/// `lora_id` is a unique generated id, so scanning the projects tree for that exact directory name
/// is unambiguous — at most one project owns it, and a stale match is impossible.
///
/// Returns only paths this function built itself from `data_dir` + a real directory entry + a fixed
/// tail, never a caller-supplied path. A `lora_id` that is not a single plain path component is
/// refused outright (the id reaches us from the request URL), and the caller still confines every
/// returned path to `<data>/projects` before removing it.
async fn lora_preview_sample_paths(data_dir: &FsPath, lora_id: &str) -> Vec<PathBuf> {
    let mut components = FsPath::new(lora_id).components();
    let single_plain = matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none();
    if !single_plain || lora_id.contains('/') || lora_id.contains('\\') {
        return Vec::new();
    }
    let Ok(mut entries) = tokio::fs::read_dir(data_dir.join("projects")).await else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let candidate = entry.path().join("training").join("samples").join(lora_id);
        if tokio::fs::symlink_metadata(&candidate).await.is_ok() {
            paths.push(candidate);
        }
    }
    paths
}

pub(crate) fn lora_huggingface_cached_file(lora: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    lora_huggingface_requested_file(lora, data_dir)
        .or_else(|| lora_huggingface_receipted_file(lora, data_dir))
}

fn lora_is_huggingface(lora: &Value) -> bool {
    let source = lora.get("source").and_then(Value::as_object);
    source
        .and_then(|source| source.get("provider"))
        .or_else(|| lora.get("provider"))
        .and_then(Value::as_str)
        == Some("huggingface")
}

fn lora_huggingface_source(lora: &Value) -> Option<(&str, Option<&str>, &str)> {
    let source = lora.get("source").and_then(Value::as_object);
    let provider = source
        .and_then(|source| source.get("provider"))
        .or_else(|| lora.get("provider"))
        .and_then(Value::as_str)?;
    if provider != "huggingface" {
        return None;
    }
    let repo = source
        .and_then(|source| source.get("repo"))
        .or_else(|| lora.get("repo"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let file_name = source
        .and_then(|source| source.get("file"))
        .or_else(|| lora.get("file"))
        .and_then(Value::as_str)
        .or_else(|| {
            source
                .and_then(|source| source.get("files"))
                .or_else(|| lora.get("files"))
                .and_then(Value::as_array)
                .and_then(|files| files.first())
                .and_then(Value::as_str)
        });
    let revision = source
        .and_then(|source| source.get("revision"))
        .or_else(|| lora.get("revision"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("main");
    Some((repo, file_name, revision))
}

fn lora_huggingface_requested_file(lora: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let (repo, file_name, revision) = lora_huggingface_source(lora)?;
    if let Some(file_name) = file_name {
        sceneworks_core::model_artifacts::ArtifactFile::new(file_name).ok()?;
        let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
            sceneworks_core::hf_home::model_source_library(data_dir),
        );
        let (_, snapshot) = resolver.discover_source_reference(repo, revision).ok()?;
        let candidate = snapshot.join(file_name);
        return candidate.is_file().then_some(candidate);
    }
    None
}

fn lora_huggingface_receipted_file(lora: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let (repo, _, _) = lora_huggingface_source(lora)?;
    let id = lora.get("id").and_then(Value::as_str)?;
    let marker = data_dir
        .join("loras")
        .join(safe_download_dir(id))
        .join(".sceneworks-download-complete.json");
    let value: Value = serde_json::from_slice(&std::fs::read(marker).ok()?).ok()?;
    let entries = value
        .get("receipts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![value]);
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
        sceneworks_core::hf_home::model_source_library(data_dir),
    );
    for receipt in entries.iter().rev() {
        if receipt.get("repo").and_then(Value::as_str) != Some(repo) {
            continue;
        }
        let Some(revision) = receipt.get("snapshotRevision").and_then(Value::as_str) else {
            continue;
        };
        let Ok((_, snapshot)) = resolver.discover_source_snapshot(repo, Some(revision)) else {
            continue;
        };
        let Some(files) = receipt.get("resolvedFiles").and_then(Value::as_array) else {
            continue;
        };
        for file in files.iter().filter_map(Value::as_str) {
            if sceneworks_core::model_artifacts::ArtifactFile::new(file).is_err() {
                continue;
            }
            let candidate = snapshot.join(file);
            if candidate.is_file()
                && candidate.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
            {
                return Some(candidate);
            }
        }
    }
    None
}

pub(crate) fn lora_families(lora: &Value) -> Vec<String> {
    families_from_value_chain(
        lora,
        &["families", "compatibleFamilies", "modelFamilies"],
        Some("compatibility"),
    )
}

/// The exact model ids a catalog entry **declares** this adapter is for (sc-19563), or empty when
/// it declares none.
///
/// The generalisation of the base-model gate below. Two things distinguish it from
/// [`lora_base_model`], and both are why it is a separate reader rather than another spelling of
/// the same one:
///
/// * **Direction.** `baseModel` is a value a trained or imported adapter *records about itself*;
///   `modelIds` is a catalog author *declaring* which partitions an adapter may attach to.
/// * **Arity.** A recorded base model is one id. A declaration can legitimately name several, so
///   this is a list.
///
/// Absent means family gating alone, exactly as before — so adding the key tightens nothing for any
/// entry that does not declare it. Not normalized: model ids are exact strings, like
/// [`lora_base_model`]'s. `model_ids` is accepted alongside `modelIds` for the same reason
/// `lora_base_model` accepts `base_model` — an inline spec may arrive in either casing.
pub(crate) fn lora_model_ids(lora: &Value) -> Vec<String> {
    for key in ["modelIds", "model_ids"] {
        if let Some(items) = lora.get(key).and_then(Value::as_array) {
            let ids: Vec<String> = items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_owned)
                .collect();
            if !ids.is_empty() {
                return ids;
            }
        }
    }
    Vec::new()
}

/// The specific base model a LoRA records it was trained for (e.g. `wan_2_2`,
/// `wan_2_2_t2v_14b`), or None. Used to gate families where a matching family is
/// not sufficient (Wan 5B and 14B both declare `wan-video` but have incompatible
/// architectures). Not normalized like families — model ids are exact strings.
pub(crate) fn lora_base_model(lora: &Value) -> Option<String> {
    for key in ["baseModel", "base_model"] {
        if let Some(value) = lora.get(key).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod huggingface_receipt_tests {
    use super::*;
    use crate::tests::support::isolate_hf_cache;

    #[test]
    fn normalized_catalog_exposes_only_a_file_identity_validated_lora_hash() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let install = data_dir.join("loras").join("author-style");
        std::fs::create_dir_all(&install).unwrap();
        let file = install.join("author-style.safetensors");
        std::fs::write(&file, b"exact adapter bytes").unwrap();
        let metadata = std::fs::metadata(&file).unwrap();
        let modified_nanos = metadata
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string();
        let hash = "1111111111111111111111111111111111111111111111111111111111111111";
        std::fs::write(
            install.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({
                "loraFileName": "author-style.safetensors",
                "loraFileBytes": metadata.len(),
                "loraFileModifiedNanos": modified_nanos,
                "loraFileSha256": hash
            }))
            .unwrap(),
        )
        .unwrap();
        let entry = json!({
            "id": "author-style",
            "name": "Author Style",
            "source": { "provider": "local", "path": install },
            "files": ["author-style.safetensors"]
        });
        let normalized = normalize_lora_entry(
            entry.clone(),
            "global",
            FsPath::new("user.loras.jsonc"),
            data_dir,
            data_dir,
        )
        .unwrap();
        assert_eq!(normalized["sha256"], hash);

        std::fs::write(&file, b"changed adapter bytes with another size").unwrap();
        let changed = normalize_lora_entry(
            entry,
            "global",
            FsPath::new("user.loras.jsonc"),
            data_dir,
            data_dir,
        )
        .unwrap();
        assert!(changed.get("sha256").is_none());
    }

    #[test]
    fn hf_catalog_reads_the_app_owned_receipt_for_a_hub_cache_adapter() {
        let _env = isolate_hf_cache();
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "author/style";
        let repo_root = huggingface_repo_cache_path(data_dir, repo).unwrap();
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let snapshot = repo_root.join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(repo_root.join("refs")).unwrap();
        std::fs::write(repo_root.join("refs").join("main"), revision).unwrap();
        let file = snapshot.join("style.safetensors");
        std::fs::write(&file, b"hub adapter bytes").unwrap();
        let metadata = std::fs::metadata(&file).unwrap();
        let modified_nanos = metadata
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string();
        let receipt_dir = data_dir
            .join("loras")
            .join(safe_download_dir("author-style"));
        std::fs::create_dir_all(&receipt_dir).unwrap();
        let hash = "2222222222222222222222222222222222222222222222222222222222222222";
        std::fs::write(
            receipt_dir.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({
                "repo": repo,
                "loraFileName": "style.safetensors",
                "loraFileBytes": metadata.len(),
                "loraFileModifiedNanos": modified_nanos,
                "loraFileSha256": hash
            }))
            .unwrap(),
        )
        .unwrap();
        let entry = json!({
            "id": "author-style",
            "name": "Author Style",
            "source": {
                "provider": "huggingface",
                "repo": repo,
                "file": "style.safetensors"
            }
        });

        let normalized = normalize_lora_entry(
            entry,
            "builtin",
            FsPath::new("builtin.loras.jsonc"),
            data_dir,
            data_dir,
        )
        .unwrap();
        assert_eq!(normalized["installedPath"], file.display().to_string());
        assert_eq!(normalized["sha256"], hash);
    }

    #[test]
    fn old_receipted_adapter_is_usable_stale_but_arbitrary_safetensors_is_not() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "SceneWorks/krea-edit";
        let repo_root = huggingface_repo_cache_path(data_dir, repo).unwrap();
        let revision = "1111111111111111111111111111111111111111";
        let snapshot = repo_root.join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("v1.1.safetensors"), b"old").unwrap();
        let lora = json!({
            "id": "krea2_identity_edit",
            "source": { "provider": "huggingface", "repo": repo, "file": "v1.2.safetensors" }
        });

        assert_eq!(lora_huggingface_cached_file(&lora, data_dir), None);

        let marker_dir = data_dir
            .join("loras")
            .join(safe_download_dir("krea2_identity_edit"));
        std::fs::create_dir_all(&marker_dir).unwrap();
        std::fs::write(
            marker_dir.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({
                "schemaVersion": 2, "repo": repo, "snapshotRevision": revision,
                "resolvedFiles": ["v1.1.safetensors"]
            }))
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            lora_huggingface_cached_file(&lora, data_dir),
            Some(snapshot.join("v1.1.safetensors"))
        );
        let normalized = normalize_lora_entry(
            lora,
            "builtin",
            FsPath::new("builtin.loras.jsonc"),
            data_dir,
            data_dir,
        )
        .unwrap();
        assert_eq!(normalized["installState"], "installed");
        assert_eq!(normalized["updateAvailable"], true);
        assert_eq!(
            normalized["installedPath"],
            snapshot.join("v1.1.safetensors").display().to_string()
        );
    }

    #[test]
    fn current_adapter_clears_update_available() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "SceneWorks/krea-edit";
        let revision = "2222222222222222222222222222222222222222";
        let snapshot = huggingface_repo_cache_path(data_dir, repo)
            .unwrap()
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("v1.2.safetensors"), b"new").unwrap();
        std::fs::create_dir_all(snapshot.parent().unwrap().parent().unwrap().join("refs")).unwrap();
        std::fs::write(
            snapshot
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("refs")
                .join("main"),
            revision,
        )
        .unwrap();
        let normalized = normalize_lora_entry(
            json!({
                "id": "krea2_identity_edit",
                "source": { "provider": "huggingface", "repo": repo, "file": "v1.2.safetensors" }
            }),
            "builtin",
            FsPath::new("builtin.loras.jsonc"),
            data_dir,
            data_dir,
        )
        .unwrap();
        assert_eq!(normalized["installState"], "installed");
        assert_eq!(normalized["updateAvailable"], false);
        assert_eq!(
            normalized["installedPath"],
            snapshot.join("v1.2.safetensors").display().to_string()
        );
    }
}

#[cfg(test)]
mod base_model_gating_tests {
    use super::*;

    fn wan_models() -> Vec<Value> {
        vec![
            json!({ "id": "wan_2_2", "loraCompatibility": { "families": ["wan-video"] } }),
            json!({ "id": "wan_2_2_t2v_14b", "loraCompatibility": { "families": ["wan-video"] } }),
        ]
    }

    #[test]
    fn rejects_wan_5b_lora_on_14b_model() {
        let models = wan_models();
        let lora = json!({ "id": "char", "families": ["wan-video"], "baseModel": "wan_2_2" });
        let err =
            validate_lora_specs_for_model(&models, &[], "wan_2_2_t2v_14b", &[lora], true, "LoRA")
                .expect_err("5B LoRA must be rejected on the 14B model");
        assert!(
            format!("{err:?}").contains("not interchangeable"),
            "got: {err:?}"
        );
    }

    #[test]
    fn accepts_wan_lora_on_matching_base_model() {
        let models = wan_models();
        let lora = json!({ "id": "char", "families": ["wan-video"], "baseModel": "wan_2_2" });
        validate_lora_specs_for_model(&models, &[], "wan_2_2", &[lora], true, "LoRA")
            .expect("exact base-model match must pass");
    }

    #[test]
    fn lora_without_base_model_falls_back_to_family_gating() {
        let models = wan_models();
        // No recorded baseModel (legacy/imported) -> family gating only, no rejection.
        let lora = json!({ "id": "legacy", "families": ["wan-video"] });
        validate_lora_specs_for_model(&models, &[], "wan_2_2_t2v_14b", &[lora], true, "LoRA")
            .expect("family-only LoRA must still pass");
    }

    /// Minimal valid safetensors whose tensor keys make `detect_lora_family` report `wan-video`
    /// — the header check is the gate that actually fired in the app, so a fixture without a real
    /// file would not reproduce the bug.
    fn write_wan_lora(dir: &std::path::Path) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        for block in 0..30 {
            for module in ["self_attn.q", "self_attn.k", "cross_attn.q", "ffn.0"] {
                for side in ["lora_A", "lora_B"] {
                    header.insert(
                        format!("transformer.blocks.{block}.{module}.{side}.weight"),
                        json!({"dtype": "F16", "shape": [16, 1024], "data_offsets": [0, 0]}),
                    );
                }
            }
        }
        let bytes_header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut bytes = (bytes_header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&bytes_header);
        std::fs::File::create(dir.join("wan_style.safetensors"))
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }

    fn krea_realtime_models() -> Vec<Value> {
        vec![
            json!({
                "id": "krea_realtime_14b",
                "family": "krea-realtime",
                "loraCompatibility": { "families": ["krea-realtime"] }
            }),
            json!({
                "id": "wan_2_2_t2v_14b",
                "family": "wan-video",
                "loraCompatibility": { "families": ["wan-video"] }
            }),
        ]
    }

    /// 🔴 sc-15017, caught by running the real app rather than by a test: this generate-time gate
    /// keyed on the model's DECLARED families, so it was stricter than the
    /// `extra_compatible_lora_families` registry the worker and engine honor. A Wan style LoRA the
    /// engine installs happily was refused at submit with "appears to be a wan-video LoRA, which
    /// is not compatible with model krea_realtime_14b (krea-realtime)".
    ///
    /// Both rejection points are exercised: the DETECTED family (from the file header — the one
    /// the app actually hit) and the DECLARED family.
    #[test]
    fn accepts_a_wan_lora_on_krea_realtime_and_stays_one_directional() {
        let tmp = tempfile::tempdir().unwrap();
        write_wan_lora(tmp.path());
        let models = krea_realtime_models();
        let wan_lora = json!({
            "id": "origami_wan_style",
            "installState": "installed",
            "installedPath": tmp.path().to_str().unwrap(),
            "families": ["wan-video"],
        });
        validate_lora_specs_for_model(
            &models,
            &[],
            "krea_realtime_14b",
            std::slice::from_ref(&wan_lora),
            true,
            "LoRA",
        )
        .expect("a Wan-family LoRA must pass the generate gate on Krea Realtime 14B");

        // Control 1 — one-directional. A krea-realtime LoRA is NOT thereby accepted on a Wan
        // model. (No header here: the declared-family check is the one under test.)
        let krea_lora = json!({
            "id": "krea_native", "installState": "installed", "families": ["krea-realtime"],
        });
        validate_lora_specs_for_model(&models, &[], "wan_2_2_t2v_14b", &[krea_lora], true, "LoRA")
            .expect_err("the extra-compatible relation must not run backwards");

        // Control 2 — the accepted set is an ADDITION, not a replacement: an unrelated family is
        // still refused on Krea, so the pass above is not "everything now passes".
        let ltx_lora = json!({
            "id": "ltx_style", "installState": "installed", "families": ["ltx-video"],
        });
        validate_lora_specs_for_model(&models, &[], "krea_realtime_14b", &[ltx_lora], true, "LoRA")
            .expect_err("an LTX LoRA must still be refused on Krea Realtime");
    }

    /// The base-model half, on the same route: SceneWorks' own importer stamps `baseModel`, and a
    /// Wan LoRA's stamp can never equal `krea_realtime_14b`. The 5B/14B split it exists for stays.
    #[test]
    fn krea_realtime_admits_a_base_model_stamped_wan_14b_lora_but_not_a_5b_one() {
        let tmp = tempfile::tempdir().unwrap();
        write_wan_lora(tmp.path());
        let models = krea_realtime_models();
        let stamped = |base: &str| {
            json!({
                "id": "stamped",
                "installState": "installed",
                "installedPath": tmp.path().to_str().unwrap(),
                "families": ["wan-video"],
                "baseModel": base,
            })
        };
        validate_lora_specs_for_model(
            &models,
            &[],
            "krea_realtime_14b",
            &[stamped("wan_2_2_t2v_14b")],
            true,
            "LoRA",
        )
        .expect("a 14B-stamped Wan LoRA must run on Krea Realtime");
        let err = validate_lora_specs_for_model(
            &models,
            &[],
            "krea_realtime_14b",
            &[stamped("wan_2_2")],
            true,
            "LoRA",
        )
        .expect_err("the 5B TI2V base must still be refused");
        assert!(
            format!("{err:?}").contains("not interchangeable"),
            "got: {err:?}"
        );

        // 🔴 An I2V-stamped base is the same 14B SIZE class but the wrong CONDITIONING class:
        // Krea Realtime is text-to-video, and an I2V LoRA targets `cross_attn.k_img`/`v_img`, which
        // it does not have. Refusing it here is a 400 at submit; admitting it would mean a hard
        // engine error after the tier fetch. The sibling T2V model already refuses this stamp.
        let i2v_err = validate_lora_specs_for_model(
            &models,
            &[],
            "krea_realtime_14b",
            &[stamped("wan_2_2_i2v_14b")],
            true,
            "LoRA",
        )
        .expect_err("an I2V-stamped base must be refused on a T2V backbone");
        assert!(
            format!("{i2v_err:?}").contains("not interchangeable"),
            "got: {i2v_err:?}"
        );
        // Control: the same I2V-stamped LoRA is fine on the I2V model itself, so the refusal above
        // is the T2V/I2V mismatch and not the stamp merely being present.
        let mut i2v_models = krea_realtime_models();
        i2v_models.push(json!({
            "id": "wan_2_2_i2v_14b",
            "family": "wan-video",
            "loraCompatibility": { "families": ["wan-video"] }
        }));
        validate_lora_specs_for_model(
            &i2v_models,
            &[],
            "wan_2_2_i2v_14b",
            &[stamped("wan_2_2_i2v_14b")],
            true,
            "LoRA",
        )
        .expect("an I2V LoRA still runs on the I2V model");
    }

    /// Minimal valid safetensors with one krea-identifying tensor key (`text_fusion`), so
    /// `detect_lora_family` returns the catalog/trainer-verbatim `krea_2` (underscore).
    fn write_krea_lora(dir: &std::path::Path) {
        use std::io::Write;
        let header = r#"{"diffusion_model.text_fusion.projector.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        std::fs::File::create(dir.join("krea.safetensors"))
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }

    /// sc-8185: a Krea LoRA's header detects as `krea_2` (underscore — the verbatim catalog/
    /// trainer token), but the model surface declares `krea_2` which `model_lora_families`
    /// normalizes to `krea-2`. The detected family must be normalized before the membership test,
    /// else the LoRA is falsely rejected as "appears to be a krea_2 LoRA ... not compatible".
    #[test]
    fn krea_lora_passes_detected_family_check_despite_underscore() {
        let tmp = tempfile::tempdir().unwrap();
        write_krea_lora(tmp.path());
        let models = vec![json!({
            "id": "krea_2_turbo",
            "loraCompatibility": { "families": ["krea_2"] }
        })];
        let lora = json!({
            "id": "krea_2_mysticxxx",
            "installState": "installed",
            "installedPath": tmp.path().to_str().unwrap(),
            "families": ["krea_2"],
        });
        validate_lora_specs_for_model(&models, &[], "krea_2_turbo", &[lora], true, "LoRA")
            .expect("krea_2 detected family must pass against a krea_2 (→krea-2) model surface");
    }

    /// A lightx2v Wan2.1-I2V step-distill ("lightning") adapter, keyed exactly like the real
    /// `Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors`: `diffusion_model.`
    /// namespace, per-block `lora_down`/`lora_up` factors, full-rank `.diff_b` bias deltas, and the
    /// I2V-only `k_img`/`v_img` cross-attention targets. It carries NO metadata blob, so detection
    /// is purely key-based — the same path the real file takes.
    fn write_wan_i2v_lightning_lora(dir: &std::path::Path) {
        use std::io::Write;
        let mut entries = Vec::new();
        let mut offset = 0_usize;
        let push = |name: String, len: usize, entries: &mut Vec<String>, offset: &mut usize| {
            entries.push(format!(
                r#""{name}":{{"dtype":"F32","shape":[1],"data_offsets":[{},{}]}}"#,
                *offset,
                *offset + len
            ));
            *offset += len;
        };
        for block in 0..2 {
            for target in [
                "self_attn.q",
                "cross_attn.k_img",
                "cross_attn.v_img",
                "ffn.0",
            ] {
                for suffix in ["lora_down.weight", "lora_up.weight", "diff_b"] {
                    push(
                        format!("diffusion_model.blocks.{block}.{target}.{suffix}"),
                        4,
                        &mut entries,
                        &mut offset,
                    );
                }
            }
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&vec![0u8; offset]);
        std::fs::File::create(dir.join("wan_lightning.safetensors"))
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }

    /// Writes a MiniMax-H3 turbo LoRA fixture in the published DIFFUSERS key space (sc-18725):
    /// `transformer_blocks.<n>.` + the `token_refiner.refiner_blocks.<n>.` marker, with PEFT's
    /// `.default` adapter-name infix. Block counts are tiny — detection keys on the refiner segment,
    /// not on a block census.
    fn write_minimax_h3_turbo_lora(dir: &std::path::Path, file_name: &str) {
        use std::io::Write;
        let mut entries = Vec::new();
        let mut offset = 0_usize;
        let push = |name: String, len: usize, entries: &mut Vec<String>, offset: &mut usize| {
            entries.push(format!(
                r#""{name}":{{"dtype":"BF16","shape":[1],"data_offsets":[{},{}]}}"#,
                *offset,
                *offset + len
            ));
            *offset += len;
        };
        for target in ["attn.to_q", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"] {
            for suffix in ["lora_A.default.weight", "lora_B.default.weight"] {
                push(
                    format!("transformer_blocks.0.{target}.{suffix}"),
                    2,
                    &mut entries,
                    &mut offset,
                );
                push(
                    format!("token_refiner.refiner_blocks.0.{target}.{suffix}"),
                    2,
                    &mut entries,
                    &mut offset,
                );
            }
        }
        let header = format!(
            r#"{{"__metadata__":{{"alpha":"8"}},{}}}"#,
            entries.join(",")
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&vec![0u8; offset]);
        std::fs::File::create(dir.join(file_name))
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }

    fn write_minimax_h3_trainer_lora(
        dir: &std::path::Path,
        file_name: &str,
        mutate: impl FnOnce(&mut serde_json::Map<String, Value>),
    ) {
        use std::io::Write;
        let rank = 16;
        let mut header = serde_json::Map::new();
        header.insert(
            "__metadata__".to_owned(),
            json!({
                "ss_network_module": "networks.lora_minimax_h3",
                "ss_h3_lora_token_refiner": "False",
                "ss_network_dim": "16",
                "ss_network_alpha": "16",
            }),
        );
        for block in 0..50 {
            for (leaf, input, output) in [
                ("attn_qkv_proj", 5_376, 21_504),
                ("attn_out_proj", 7_168, 5_376),
                ("mlp_fc1", 5_376, 28_672),
                ("mlp_fc2", 14_336, 5_376),
            ] {
                let target = format!("lora_unet_blocks_{block}_{leaf}");
                header.insert(
                    format!("{target}.lora_down.weight"),
                    json!({ "dtype": "F16", "shape": [rank, input], "data_offsets": [0, 0] }),
                );
                header.insert(
                    format!("{target}.lora_up.weight"),
                    json!({ "dtype": "F16", "shape": [output, rank], "data_offsets": [0, 0] }),
                );
                header.insert(
                    format!("{target}.alpha"),
                    json!({ "dtype": "F32", "shape": [], "data_offsets": [0, 0] }),
                );
            }
        }
        mutate(&mut header);
        let bytes_header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut bytes = (bytes_header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&bytes_header);
        std::fs::File::create(dir.join(file_name))
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }

    fn minimax_h3_model_fixture() -> Vec<Value> {
        vec![json!({
            "id": "minimax_h3",
            "family": "minimax-h3",
            "loraCompatibility": { "families": ["minimax-h3"] }
        })]
    }

    #[test]
    fn trainer_namespace_is_classified_at_import_and_accepted_by_h3_preflight() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimax_h3_trainer_lora(tmp.path(), "community.safetensors", |_| {});
        let path = tmp.path().join("community.safetensors");

        let (family, metadata) = inspect_lora_source(path.to_str().unwrap())
            .expect("the exact trainer namespace imports");
        assert_eq!(family.as_deref(), Some("minimax-h3"));
        assert_eq!((metadata.rank, metadata.alpha), (Some(16), Some(16.0)));

        let lora = json!({
            "id": "minimax_h3_community",
            "installState": "installed",
            "installedPath": path,
            "families": ["minimax-h3"],
        });
        validate_lora_specs_for_model(
            &minimax_h3_model_fixture(),
            &[],
            "minimax_h3",
            &[lora],
            true,
            "LoRA",
        )
        .expect("the intentional trunk-only trainer export passes before generation");
    }

    #[test]
    fn malformed_or_unsupported_h3_trainer_namespaces_fail_actionably_before_generation() {
        let partial = tempfile::tempdir().unwrap();
        write_minimax_h3_trainer_lora(partial.path(), "partial.safetensors", |header| {
            header.remove("lora_unet_blocks_49_mlp_fc2.alpha");
        });
        let import_error =
            inspect_lora_source(partial.path().join("partial.safetensors").to_str().unwrap())
                .expect_err("local import must reject a partial trainer export");
        assert!(
            format!("{import_error:?}").contains("missing alpha"),
            "{import_error:?}"
        );

        let unsupported = tempfile::tempdir().unwrap();
        use std::io::Write;
        let header = serde_json::to_vec(&json!({
            "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": {
                "dtype": "F16", "shape": [16, 5376], "data_offsets": [0, 0]
            }
        }))
        .unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        let path = unsupported.path().join("unknown.safetensors");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let local_error = inspect_lora_source_for_family(path.to_str().unwrap(), true)
            .expect_err("a local import declared as H3 must reject an unknown namespace");
        assert!(
            format!("{local_error:?}").contains("unsupported MiniMax-H3 adapter namespace"),
            "{local_error:?}"
        );
        let lora = json!({
            "id": "minimax_h3_unknown_namespace",
            "installState": "installed",
            "installedPath": path,
            "families": ["minimax-h3"],
        });
        let preflight_error = validate_lora_specs_for_model(
            &minimax_h3_model_fixture(),
            &[],
            "minimax_h3",
            &[lora],
            true,
            "LoRA",
        )
        .expect_err("an unknown H3 namespace must not reach generation");
        assert!(
            format!("{preflight_error:?}").contains("unsupported MiniMax-H3 adapter namespace"),
            "{preflight_error:?}"
        );
    }

    /// sc-18725: the end-to-end submit gate for the turbo accelerators, on BOTH H3 partitions.
    ///
    /// The model surfaces and the LoRA below are `json!` fixtures that MIRROR what
    /// `builtin.models.jsonc` and `builtin.loras.jsonc` declare — this test reads neither manifest,
    /// so a withdrawn or mistyped `families` list does NOT turn it red. What it proves is that
    /// `validate_lora_specs_for_model` accepts that shape on both partitions. The manifest side is
    /// held in `sceneworks-core`, by `both_minimax_h3_partitions_advertise_the_minimax_h3_lora_family`
    /// (the model declarations) and `minimax_h3_turbo_loras_are_registered_and_sha_pinned` (the LoRA
    /// entries), which do read the embedded manifests. Keep these fixtures in step with them by hand.
    ///
    /// It also pins that detection reports `minimax-h3` — before sc-18725 it reported `None`, which
    /// let the file through this gate by ACCIDENT (the detected-family check is skipped on `None`)
    /// while leaving a user-imported copy family-less and hidden by the web picker's fail-closed
    /// rule.
    #[test]
    fn minimax_h3_turbo_lora_passes_the_submit_gate_on_both_partitions() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimax_h3_turbo_lora(
            tmp.path(),
            "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
        );
        // The fixture is only meaningful if it really detects as minimax-h3 — pin that, so a change
        // in the detector turns into a clear failure here rather than a vacuous pass.
        let header = read_safetensors_header(
            &tmp.path()
                .join("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"),
        )
        .unwrap();
        assert_eq!(detect_lora_family(&header).as_deref(), Some("minimax-h3"));

        let models = vec![
            json!({
                "id": "minimax_h3",
                "family": "minimax-h3",
                "loraCompatibility": { "families": ["minimax-h3"] }
            }),
            json!({
                "id": "minimax_h3_ref",
                "family": "minimax-h3",
                "loraCompatibility": { "families": ["minimax-h3"] }
            }),
        ];
        let lora = json!({
            "id": "minimax_h3_turbo_8step",
            "installState": "installed",
            "installedPath": tmp.path().to_str().unwrap(),
            "families": ["minimax-h3"],
        });
        for model_id in ["minimax_h3", "minimax_h3_ref"] {
            validate_lora_specs_for_model(
                &models,
                &[],
                model_id,
                std::slice::from_ref(&lora),
                true,
                "LoRA",
            )
            .unwrap_or_else(|error| {
                panic!("the turbo accelerator must be accepted on {model_id}: {error:?}")
            });
        }
    }

    /// sc-19563 — **the declared-partition gate**, both directions, with the control that makes it
    /// attributable.
    ///
    /// The family check cannot see this: `minimax_h3` and `minimax_h3_ref` are one architecture and
    /// one family, so before this gate the ref2v adapter attached to `minimax_h3` and **folded
    /// cleanly** — no shape error, no refusal, just a quality mismatch, which is what made it easy
    /// to ship and hard to notice.
    ///
    /// Four arms, and the two `Ok` ones are load-bearing: without them a gate that refused
    /// *everything* would pass the two `Err` arms.
    #[test]
    fn a_declared_partition_gates_the_lora_to_that_model_id() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimax_h3_turbo_lora(
            tmp.path(),
            "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
        );
        let models = vec![
            json!({
                "id": "minimax_h3",
                "family": "minimax-h3",
                "loraCompatibility": { "families": ["minimax-h3"] }
            }),
            json!({
                "id": "minimax_h3_ref",
                "family": "minimax-h3",
                "loraCompatibility": { "families": ["minimax-h3"] }
            }),
        ];
        let ref2v = json!({
            "id": "minimax_h3_ref2v_turbo_4step",
            "installState": "installed",
            "installedPath": tmp.path().to_str().unwrap(),
            "families": ["minimax-h3"],
            "modelIds": ["minimax_h3_ref"],
        });
        let fl2v = json!({
            "id": "minimax_h3_turbo_8step",
            "installState": "installed",
            "installedPath": tmp.path().to_str().unwrap(),
            "families": ["minimax-h3"],
            "modelIds": ["minimax_h3"],
        });

        // ── the two refusals, each naming BOTH the LoRA and the partition.
        for (lora, model_id, declared) in [
            (&ref2v, "minimax_h3", "minimax_h3_ref"),
            (&fl2v, "minimax_h3_ref", "minimax_h3"),
        ] {
            let error = validate_lora_specs_for_model(
                &models,
                &[],
                model_id,
                std::slice::from_ref(lora),
                true,
                "LoRA",
            )
            .expect_err("a cross-selected partition adapter must be refused");
            let message = format!("{error:?}");
            let lora_id = lora["id"].as_str().unwrap();
            assert!(
                message.contains(lora_id),
                "the refusal must name the LoRA; got {message}"
            );
            assert!(
                message.contains(model_id),
                "the refusal must name the model it was refused ON; got {message}"
            );
            // **The exact phrase, not a substring search.** `minimax_h3_ref2v_turbo_4step`
            // CONTAINS `minimax_h3_ref`, so a bare `message.contains(declared)` is satisfied by
            // the LoRA id already in the message and asserts nothing — mutation testing caught
            // exactly that: blanking the declared-partition interpolation left this green.
            assert!(
                message.contains(&format!("is declared for model {declared}")),
                "the refusal must name the partition it IS for; got {message}"
            );
        }

        // ── the controls: each adapter is accepted on its OWN partition. Without these the test
        //    would pass against a gate that refused every H3 LoRA outright.
        for (lora, model_id) in [(&ref2v, "minimax_h3_ref"), (&fl2v, "minimax_h3")] {
            validate_lora_specs_for_model(
                &models,
                &[],
                model_id,
                std::slice::from_ref(lora),
                true,
                "LoRA",
            )
            .unwrap_or_else(|error| panic!("must be accepted on its own partition: {error:?}"));
        }
    }

    /// The gate is **not** hardcoded to a family — that was the whole point of sc-19563, whose
    /// predecessor gate reads `families.iter().any(|f| f == "wan-video")`.
    ///
    /// A `modelIds` declaration on an invented family with no in-tree special-casing must gate just
    /// as hard. If someone later re-hardcodes this to `minimax-h3`, this arm reds and the H3 arms
    /// above do not.
    #[test]
    fn the_declared_partition_gate_is_family_agnostic() {
        let models = vec![
            json!({
                "id": "acme_alpha",
                "family": "acme",
                "loraCompatibility": { "families": ["acme"] }
            }),
            json!({
                "id": "acme_beta",
                "family": "acme",
                "loraCompatibility": { "families": ["acme"] }
            }),
        ];
        // No `installedPath`, so no header is read and no family detection runs — this isolates the
        // declared-partition gate from the detected-family one.
        let lora = json!({
            "id": "acme_style",
            "installState": "installed",
            "families": ["acme"],
            "modelIds": ["acme_beta"],
        });
        validate_lora_specs_for_model(
            &models,
            &[],
            "acme_beta",
            std::slice::from_ref(&lora),
            true,
            "LoRA",
        )
        .expect("accepted on the partition it declares");
        let error = validate_lora_specs_for_model(
            &models,
            &[],
            "acme_alpha",
            std::slice::from_ref(&lora),
            true,
            "LoRA",
        )
        .expect_err("a family with no in-tree special-casing must still be gated");
        assert!(format!("{error:?}").contains("acme_beta"), "{error:?}");
    }

    /// **A LoRA that declares NO `modelIds` is untouched.** The key is optional, so the gate must
    /// not tighten a single existing catalog entry — every LoRA in the tree today declares none.
    #[test]
    fn a_lora_without_declared_model_ids_is_not_gated() {
        let models = vec![json!({
            "id": "minimax_h3_ref",
            "family": "minimax-h3",
            "loraCompatibility": { "families": ["minimax-h3"] }
        })];
        let lora = json!({
            "id": "legacy_h3_style",
            "installState": "installed",
            "families": ["minimax-h3"],
        });
        validate_lora_specs_for_model(&models, &[], "minimax_h3_ref", &[lora], true, "LoRA")
            .expect("an undeclared LoRA keeps family gating alone");
        // ...and the reader itself agrees there is nothing to gate on.
        assert!(lora_model_ids(&json!({ "id": "x" })).is_empty());
        assert!(lora_model_ids(&json!({ "modelIds": [] })).is_empty());
        assert!(lora_model_ids(&json!({ "modelIds": ["  ", ""] })).is_empty());
        assert_eq!(
            lora_model_ids(&json!({ "modelIds": [" minimax_h3_ref "] })),
            vec!["minimax_h3_ref".to_string()],
            "ids are trimmed, matching lora_base_model"
        );
        assert_eq!(
            lora_model_ids(&json!({ "model_ids": ["minimax_h3"] })),
            vec!["minimax_h3".to_string()],
            "the snake_case alias is read too, like base_model"
        );
    }

    /// The detected-family gate is a REAL gate for this family, not a formality: a LoRA from another
    /// architecture is still refused on H3. Without this, `minimax_h3_turbo_lora_passes_...` above
    /// would pass just as well against a model surface that accepted everything.
    #[test]
    fn a_foreign_lora_is_still_rejected_on_minimax_h3() {
        let tmp = tempfile::tempdir().unwrap();
        write_wan_i2v_lightning_lora(tmp.path());
        let models = vec![json!({
            "id": "minimax_h3",
            "family": "minimax-h3",
            "loraCompatibility": { "families": ["minimax-h3"] }
        })];
        let lora = json!({
            "id": "some_wan_lora",
            "installState": "installed",
            "installedPath": tmp.path().to_str().unwrap(),
            "families": ["minimax-h3"],
        });
        let error =
            validate_lora_specs_for_model(&models, &[], "minimax_h3", &[lora], true, "LoRA")
                .expect_err("a wan-video LoRA must not be accepted on minimax_h3");
        assert!(
            format!("{error:?}").contains("wan-video"),
            "the rejection must name the detected family; got {error:?}"
        );
    }

    /// sc-18200: the bundled `scail2_lightning` speed toggle IS a lightx2v Wan2.1-I2V LoRA, applied
    /// to SCAIL-2 cross-architecture on purpose. `detect_lora_family` therefore reports `wan-video`
    /// — correctly — while the model surface declares `scail2`. The gate used to compare the
    /// detected family against the manifest list ALONE, so it rejected the transplant the engine
    /// exists to perform ("appears to be a wan-video LoRA, which is not compatible with model
    /// scail2_14b"). It must consult the model's full accepted set (`extra_compatible_lora_families`).
    #[test]
    fn wan_lightning_lora_passes_detected_family_check_on_scail2() {
        let tmp = tempfile::tempdir().unwrap();
        write_wan_i2v_lightning_lora(tmp.path());
        // The fixture is only meaningful if it really detects as wan-video — pin that, so a change
        // in the detector turns into a clear failure here rather than a vacuous pass.
        let header =
            read_safetensors_header(&tmp.path().join("wan_lightning.safetensors")).unwrap();
        assert_eq!(detect_lora_family(&header).as_deref(), Some("wan-video"));

        let models = vec![json!({
            "id": "scail2_14b",
            "family": "scail2",
            "loraCompatibility": { "families": ["scail2"] }
        })];
        let lora = json!({
            "id": "scail2_lightning",
            "installState": "installed",
            "installedPath": tmp.path().to_str().unwrap(),
            "families": ["scail2"],
        });
        validate_lora_specs_for_model(&models, &[], "scail2_14b", &[lora], true, "LoRA")
            .expect("a wan-video-detected lightning LoRA must be accepted on a scail2 model");
    }

    /// The same defect, second instance: Krea Realtime's DiT is Wan 2.1 T2V 14B weight-for-weight,
    /// and its manifest deliberately keeps `wan-video` OUT of `loraCompatibility.families` so the
    /// token does not leak into the other family-keyed gates — the relation lives in
    /// `extra_compatible_lora_families` instead. That left this gate blind to it too.
    #[test]
    fn wan_lora_passes_detected_family_check_on_krea_realtime() {
        let tmp = tempfile::tempdir().unwrap();
        write_wan_i2v_lightning_lora(tmp.path());
        let models = vec![json!({
            "id": "krea_realtime_14b",
            "family": "krea-realtime",
            "loraCompatibility": { "families": ["krea-realtime"] }
        })];
        let lora = json!({
            "id": "some_wan_motion_lora",
            "installState": "installed",
            "installedPath": tmp.path().to_str().unwrap(),
            "families": ["krea-realtime"],
        });
        validate_lora_specs_for_model(&models, &[], "krea_realtime_14b", &[lora], true, "LoRA")
            .expect("a wan-video LoRA must be accepted on krea-realtime");
    }

    /// The widening must stay narrow: a model with no extra-compatible relation still rejects a
    /// foreign detected architecture. Without this, "accept the registry" could silently become
    /// "accept anything" and the gate would stop catching genuinely wrong files.
    #[test]
    fn wan_lora_still_rejected_on_an_unrelated_model_family() {
        let tmp = tempfile::tempdir().unwrap();
        write_wan_i2v_lightning_lora(tmp.path());
        let models = vec![json!({
            "id": "z_image_turbo",
            "family": "z-image",
            "loraCompatibility": { "families": ["z-image"] }
        })];
        let lora = json!({
            "id": "mislabelled",
            "installState": "installed",
            "installedPath": tmp.path().to_str().unwrap(),
            "families": ["z-image"],
        });
        let error =
            validate_lora_specs_for_model(&models, &[], "z_image_turbo", &[lora], true, "LoRA")
                .expect_err("a wan-video LoRA must still be rejected on an unrelated family");
        assert!(
            error.detail.contains("wan-video"),
            "unexpected rejection detail: {}",
            error.detail
        );
    }

    // sc-10214: the id (and thus the on-disk folder) is family-scoped so two variants of
    // one LoRA that share a display name never resolve to the same folder.
    #[test]
    fn derive_lora_id_prefixes_canonical_family() {
        assert_eq!(
            derive_lora_id(None, "Realism Engine", Some("flux2")),
            "flux2_realism_engine"
        );
        assert_eq!(
            derive_lora_id(None, "Realism Engine", Some("krea_2")),
            "krea_2_realism_engine"
        );
        // A hyphenated family token slugifies to a clean all-underscore id.
        assert_eq!(
            derive_lora_id(None, "Detail LoRA", Some("z-image")),
            "z_image_detail_lora"
        );
        // A krea_2 and a flux2 variant of the same-named LoRA land in different folders —
        // the exact collision that mis-detected a flux2 import as krea_2.
        assert_ne!(
            derive_lora_id(None, "Realism Engine", Some("flux2")),
            derive_lora_id(None, "Realism Engine", Some("krea_2"))
        );
    }

    #[test]
    fn derive_lora_id_canonicalizes_family_spelling() {
        // ai-toolkit's separator-less `krea2` and the hyphen form both canonicalize to
        // the stored `krea_2` token, so the folder is stable regardless of source spelling.
        assert_eq!(
            derive_lora_id(None, "Realism Engine", Some("krea2")),
            "krea_2_realism_engine"
        );
        assert_eq!(
            derive_lora_id(None, "Realism Engine", Some("krea-2")),
            "krea_2_realism_engine"
        );
    }

    #[test]
    fn derive_lora_id_falls_back_and_respects_explicit_id() {
        // Unresolved family (HF/URL import) -> bare slug, unchanged from prior behaviour.
        assert_eq!(
            derive_lora_id(None, "Realism Engine", None),
            "realism_engine"
        );
        // An explicit caller-supplied id is used verbatim (never re-prefixed).
        assert_eq!(
            derive_lora_id(Some("my_custom_id"), "Realism Engine", Some("flux2")),
            "my_custom_id"
        );
    }

    #[test]
    fn conflicting_folder_family_flags_cross_family_folder() {
        let tmp = tempfile::tempdir().unwrap();
        write_krea_lora(tmp.path());
        // A flux2 import targeting a folder that already holds a krea_2 adapter conflicts.
        assert_eq!(
            conflicting_folder_family(tmp.path(), "flux2")
                .unwrap()
                .as_deref(),
            Some("krea_2")
        );
        // Same-family re-import is allowed (no conflict), incl. spelling variants.
        assert_eq!(
            conflicting_folder_family(tmp.path(), "krea_2").unwrap(),
            None
        );
        assert_eq!(
            conflicting_folder_family(tmp.path(), "krea2").unwrap(),
            None
        );
    }

    #[test]
    fn conflicting_folder_family_ignores_empty_folder() {
        let tmp = tempfile::tempdir().unwrap();
        // A fresh (or never-created) folder never conflicts — the common new-import path.
        assert_eq!(
            conflicting_folder_family(tmp.path(), "flux2").unwrap(),
            None
        );
        assert_eq!(
            conflicting_folder_family(&tmp.path().join("does_not_exist"), "flux2").unwrap(),
            None
        );
    }

    // ---- Mage-Flow import (sc-14057) -----------------------------------------------------

    /// Write a minimal valid safetensors adapter with `keys` and the given `__metadata__`.
    fn write_adapter(path: &std::path::Path, metadata: &str, keys: &[String]) {
        use std::io::Write;
        let tensors: Vec<String> = keys
            .iter()
            .map(|key| format!(r#""{key}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#))
            .collect();
        let header = format!(r#"{{"__metadata__":{metadata},{}}}"#, tensors.join(","));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        std::fs::File::create(path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }

    /// A 12-block dual-stream Mage-Flow adapter, no family metadata (the community shape).
    fn mage_lora_keys() -> Vec<String> {
        let mut keys = Vec::new();
        for block in 0..12 {
            for module in ["attn.to_q", "attn.add_q_proj", "img_mlp.net.0.proj"] {
                for role in ["lora_A.weight", "lora_B.weight"] {
                    keys.push(format!(
                        "transformer.transformer_blocks.{block}.{module}.{role}"
                    ));
                }
            }
        }
        keys
    }

    /// sc-14057 + sc-10214: a Mage adapter gets its own family-scoped folder, so a Mage and a
    /// Qwen-Image LoRA sharing one display name can never co-mingle in a single record dir (the
    /// collision that made the header inspector arbitrate non-deterministically).
    #[test]
    fn mage_flow_imports_into_a_family_scoped_folder() {
        assert_eq!(
            derive_lora_id(None, "Realism Engine", Some("mage-flow")),
            "mage_flow_realism_engine"
        );
        // The inference trainer's own `mage_flow` spelling canonicalizes to the same folder.
        assert_eq!(
            derive_lora_id(None, "Realism Engine", Some("mage_flow")),
            "mage_flow_realism_engine"
        );
        for sibling in ["qwen-image", "z-image", "krea_2"] {
            assert_ne!(
                derive_lora_id(None, "Realism Engine", Some("mage-flow")),
                derive_lora_id(None, "Realism Engine", Some(sibling)),
                "mage-flow must not share a folder with {sibling}"
            );
        }
    }

    /// The cross-family guard, exercised on the pair that actually looks alike: a Mage folder
    /// rejects an incoming Qwen-Image import (and vice versa), while a same-family re-import is
    /// still allowed.
    #[test]
    fn a_mage_folder_rejects_a_foreign_family_reimport() {
        let tmp = tempfile::tempdir().unwrap();
        write_adapter(
            &tmp.path().join("mage.safetensors"),
            r#"{"format":"pt"}"#,
            &mage_lora_keys(),
        );
        assert_eq!(
            conflicting_folder_family(tmp.path(), "qwen-image")
                .unwrap()
                .as_deref(),
            Some("mage-flow")
        );
        assert_eq!(
            conflicting_folder_family(tmp.path(), "mage-flow").unwrap(),
            None
        );
        assert_eq!(
            conflicting_folder_family(tmp.path(), "mage_flow").unwrap(),
            None
        );
    }

    /// The import inspector returns the detected family AND what the file declares about itself,
    /// from one header read — and declares nothing it was not told (sc-14057 trap: never default
    /// alpha to rank).
    #[test]
    fn inspect_lora_source_reports_family_and_declared_rank_alpha() {
        let tmp = tempfile::tempdir().unwrap();
        let stamped = tmp.path().join("stamped");
        std::fs::create_dir_all(&stamped).unwrap();
        write_adapter(
            &stamped.join("adapter.safetensors"),
            r#"{"family":"mage_flow","networkType":"lokr","rank":"16","alpha":"32"}"#,
            &["transformer_blocks.0.attn.to_q.lokr_w1".to_owned()],
        );
        let (family, meta) = inspect_lora_source(stamped.to_str().unwrap()).unwrap();
        assert_eq!(family.as_deref(), Some("mage-flow"));
        assert_eq!(meta.network_type.as_deref(), Some("lokr"));
        assert_eq!(meta.rank, Some(16));
        assert_eq!(meta.alpha, Some(32.0));

        // A community file with neither a family stamp nor rank/alpha: the family still resolves
        // from the 12-block geometry, and nothing is invented for the rest.
        let bare = tmp.path().join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        write_adapter(
            &bare.join("adapter.safetensors"),
            r#"{"format":"pt"}"#,
            &mage_lora_keys(),
        );
        let (family, meta) = inspect_lora_source(bare.to_str().unwrap()).unwrap();
        assert_eq!(family.as_deref(), Some("mage-flow"));
        assert!(
            meta.is_empty(),
            "an adapter that declares no rank/alpha/type must record none: {meta:?}"
        );
    }
}
