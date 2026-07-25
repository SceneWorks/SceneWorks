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

/// Assemble the control-overlay catalog: builtin/hosted (`builtin.control_overlays.jsonc`, sc-8466) +
/// user-global (`user.control_overlays.jsonc`) + the project manifest
/// (`<project>/control-overlays/manifest.jsonc`) when a project id is given, merged by id (a later scope
/// overrides an earlier one — the `lora_catalog` pattern). Each entry is normalized (installedPath /
/// installState / manifestPath) via the shared [`normalize_lora_entry`]: a hosted entry
/// (`source.provider="huggingface"` + `source.repo`) resolves its install state from the HF cache, while a
/// studio-trained entry (a relative `source.path` under the scope root) resolves the local path.
pub(crate) async fn control_overlay_catalog(
    state: &AppState,
    project_id: Option<&str>,
) -> Result<Vec<Value>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin_manifest = manifest_dir.join("builtin.control_overlays.jsonc");
    let builtin = load_manifest_entries(state, &builtin_manifest, "controlOverlays").await?;
    let user_manifest = manifest_dir.join("user.control_overlays.jsonc");
    let user = load_manifest_entries(state, &user_manifest, "controlOverlays").await?;
    let data_dir = state.settings.data_dir.clone();

    // Normalize builtin (hosted) + user (global) install-state off the async executor, then merge by id so
    // a user override replaces a builtin of the same id.
    let mut overlays = {
        let data_dir = data_dir.clone();
        let builtin_manifest = builtin_manifest.clone();
        let user_manifest = user_manifest.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Value>, ApiError> {
            let mut builtin_out = Vec::new();
            for entry in builtin {
                builtin_out.push(crate::loras::normalize_lora_entry(
                    entry,
                    "builtin",
                    &builtin_manifest,
                    &data_dir,
                    &data_dir,
                )?);
            }
            let mut user_out = Vec::new();
            for entry in user {
                user_out.push(crate::loras::normalize_lora_entry(
                    entry,
                    "global",
                    &user_manifest,
                    &data_dir,
                    &data_dir,
                )?);
            }
            Ok(crate::manifest::merge_entries_by_id(builtin_out, user_out))
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
        overlays = crate::manifest::merge_entries_by_id(overlays, project_overlays);
    }

    Ok(overlays)
}

/// Resolve a selected control overlay id (`advanced.controlWeights.overlayId`, set by the Studio's
/// ControlPanel picker) to what the worker strict-control lane (`ensure_krea_control_weights`) reads
/// (sc-10165 B4 + sc-8466). An INSTALLED overlay resolves to its `.safetensors` path
/// (`advanced.controlWeights.path`, studio-trained/registered or an already-cached hosted overlay); a
/// hosted overlay not yet on disk (the built-in beta or any un-cached hosted entry) resolves to
/// `advanced.controlWeights.{repo,filename}` so the worker lazy-downloads it on first use (the FLUX.2
/// control precedent). A no-op when no overlay is selected. A selected-but-unknown overlay, or a
/// training/local overlay that is not installed, is a clean 400 rather than a deep worker error. Mirrors
/// the LoRA id→path resolution: one catalog snapshot, in-place `advanced` mutation.
pub(crate) async fn resolve_control_overlay_selection(
    state: &AppState,
    project_id: Option<&str>,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    validate_raw_control_weights(job_payload)?;
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
    if let (Some(model), Some(base_model)) = (
        job_payload.get("model").and_then(Value::as_str),
        overlay.get("baseModel").and_then(Value::as_str),
    ) {
        if model != base_model {
            return Err(ApiError::bad_request(format!(
                "Control overlay '{overlay_id}' is registered for {base_model}, not {model}"
            )));
        }
    }

    // What the worker resolves the overlay from: an installed local `.safetensors` (`controlWeights.path`
    // — studio-trained/registered, or an already-cached hosted overlay), else a hosted repo/filename it
    // lazy-downloads on first use (`controlWeights.{repo,filename}` — the built-in beta or any not-yet-
    // cached hosted overlay). A training/local overlay that is not installed is a real 400.
    let installed = overlay.get("installState").and_then(Value::as_str) == Some("installed");
    let weights: Vec<(&str, Value)> = if installed {
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
        vec![("path", Value::String(weights_path))]
    } else if overlay
        .pointer("/source/provider")
        .or_else(|| overlay.get("provider"))
        .and_then(Value::as_str)
        == Some("huggingface")
    {
        let (repo, filename, revision) =
            hosted_overlay_repo_file(&overlay).ok_or_else(|| {
                ApiError::bad_request(format!(
                    "Control overlay '{overlay_id}' must register one weights file and a pinned 40-hex source.revision"
                ))
            })?;
        vec![
            ("repo", Value::String(repo)),
            ("filename", Value::String(filename)),
            ("revision", Value::String(revision)),
        ]
    } else {
        return Err(ApiError::bad_request(format!(
            "Control overlay '{overlay_id}' is not installed"
        )));
    };

    let advanced = job_payload
        .entry("advanced".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    if let Some(advanced) = advanced.as_object_mut() {
        let control_weights = advanced
            .entry("controlWeights".to_owned())
            .or_insert_with(|| Value::Object(JsonObject::new()));
        if let Some(control_weights) = control_weights.as_object_mut() {
            // The catalog selection is authoritative. Remove every caller value
            // before writing the registered location so a mixed payload cannot
            // retain an injected repo, filename, path, revision, or internal bit.
            control_weights.clear();
            control_weights.insert("overlayId".to_owned(), Value::String(overlay_id));
            control_weights.insert("_catalogAuthorized".to_owned(), Value::Bool(true));
            for (key, value) in weights {
                control_weights.insert(key.to_owned(), value);
            }
        }
    }
    Ok(())
}

/// The (repo, filename) of a hosted overlay entry (`source.provider="huggingface"` + `source.repo` +
/// `files[0]`/`source.file`) — the worker lazy-downloads these when the overlay is not yet cached,
/// mirroring the built-in default in `ensure_krea_control_weights`. `None` for a training/local overlay.
fn hosted_overlay_repo_file(overlay: &Value) -> Option<(String, String, String)> {
    let source = overlay.get("source").and_then(Value::as_object);
    let provider = source
        .and_then(|s| s.get("provider"))
        .or_else(|| overlay.get("provider"))
        .and_then(Value::as_str)?;
    if provider != "huggingface" {
        return None;
    }
    let repo = source
        .and_then(|s| s.get("repo"))
        .or_else(|| overlay.get("repo"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|v| !v.is_empty())?
        .to_owned();
    let filename = overlay
        .get("files")
        .and_then(Value::as_array)
        .and_then(|files| files.first())
        .and_then(Value::as_str)
        .or_else(|| source.and_then(|s| s.get("file")).and_then(Value::as_str))
        .map(str::trim)
        .filter(|v| !v.is_empty())?
        .to_owned();
    let revision = source
        .and_then(|s| s.get("revision"))
        .or_else(|| overlay.get("revision"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|revision| {
            revision.len() == 40
                && revision
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        })?
        .to_owned();
    Some((repo, filename, revision))
}

/// Reject raw remote/local control-weight locations before a job is queued.
///
/// A caller may omit `controlWeights` and use the shipped engine default, name
/// an exact shipped repo/file tuple, or select a registered `overlayId`. It may
/// not manufacture a catalog authorization bit or choose an arbitrary remote
/// repository/path. The worker repeats the engine-specific exact-tuple check.
fn validate_raw_control_weights(job_payload: &mut JsonObject) -> Result<(), ApiError> {
    let Some(control_weights) = job_payload
        .get_mut("advanced")
        .and_then(Value::as_object_mut)
        .and_then(|advanced| advanced.get_mut("controlWeights"))
        .and_then(Value::as_object_mut)
    else {
        return Ok(());
    };

    // Internal fields are derived only after a successful catalog lookup.
    control_weights.remove("_catalogAuthorized");
    control_weights.remove("revision");

    let overlay_id = control_weights
        .get("overlayId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if overlay_id.is_some() {
        return Ok(());
    }
    if control_weights.contains_key("path") {
        return Err(ApiError::bad_request(
            "advanced.controlWeights.path requires a registered overlayId",
        ));
    }

    let repo = control_weights
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let file = control_weights
        .get("filename")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match (repo, file) {
        (None, None) => Ok(()),
        (Some(repo), Some(file))
            if sceneworks_core::control_weights::shipped_control_weight_by_repo_file(repo, file)
                .is_some() =>
        {
            Ok(())
        }
        _ => Err(ApiError::bad_request(
            "advanced.controlWeights repo/filename must be an exact shipped artifact or come from a registered overlayId",
        )),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hosted_overlay_repo_file_reads_huggingface_source() {
        // A built-in/hosted overlay: the picker-selected id resolves to (repo, files[0]) for the worker to
        // lazy-download — `files[0]` wins over `source.file`.
        let overlay = json!({
            "id": "krea_2_pose_beta",
            "source": {
                "provider": "huggingface",
                "repo": "SceneWorks/krea2-pose-controlnet-beta",
                "file": "should-not-win.safetensors",
                "revision": "cb3a0ac7590f5ec594a4eeb43b95ee1da0b5a0ac"
            },
            "files": ["control_step5000.safetensors"]
        });
        assert_eq!(
            hosted_overlay_repo_file(&overlay),
            Some((
                "SceneWorks/krea2-pose-controlnet-beta".to_owned(),
                "control_step5000.safetensors".to_owned(),
                "cb3a0ac7590f5ec594a4eeb43b95ee1da0b5a0ac".to_owned()
            ))
        );
    }

    #[test]
    fn hosted_overlay_repo_file_falls_back_to_source_file() {
        let overlay = json!({
            "source": {
                "provider": "huggingface",
                "repo": "acme/x",
                "file": "w.safetensors",
                "revision": "0123456789abcdef0123456789abcdef01234567"
            }
        });
        assert_eq!(
            hosted_overlay_repo_file(&overlay),
            Some((
                "acme/x".to_owned(),
                "w.safetensors".to_owned(),
                "0123456789abcdef0123456789abcdef01234567".to_owned()
            ))
        );
    }

    #[test]
    fn hosted_overlay_requires_a_pinned_revision() {
        let overlay = json!({
            "source": { "provider": "huggingface", "repo": "acme/x", "file": "w.safetensors" }
        });
        assert_eq!(hosted_overlay_repo_file(&overlay), None);
    }

    #[test]
    fn hosted_overlay_repo_file_none_for_training_overlay() {
        // A studio-trained/registered overlay carries a local `source.path`, not an HF repo — so resolve
        // must NOT treat "not installed" as a lazy-download (it is a real 400 instead).
        let overlay = json!({
            "source": { "provider": "training", "path": "control-overlays/my-overlay" },
            "files": ["overlay.safetensors"]
        });
        assert_eq!(hosted_overlay_repo_file(&overlay), None);
    }

    #[test]
    fn raw_control_weights_accept_only_exact_shipped_artifacts() {
        let mut shipped = json!({
            "advanced": {
                "controlWeights": {
                    "repo": "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
                    "filename": "FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors"
                }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        assert!(validate_raw_control_weights(&mut shipped).is_ok());

        for weights in [
            json!({"repo": "attacker/payload", "filename": "model.safetensors"}),
            json!({
                "repo": "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
                "filename": "other.safetensors"
            }),
            json!({"path": "/tmp/payload.safetensors"}),
        ] {
            let mut payload = json!({"advanced": {"controlWeights": weights}})
                .as_object()
                .unwrap()
                .clone();
            assert!(validate_raw_control_weights(&mut payload).is_err());
        }
    }

    #[test]
    fn raw_catalog_stamp_is_removed_before_overlay_lookup() {
        let mut payload = json!({
            "advanced": {
                "controlWeights": {
                    "overlayId": "registered",
                    "repo": "attacker/payload",
                    "filename": "payload.safetensors",
                    "revision": "0123456789abcdef0123456789abcdef01234567",
                    "_catalogAuthorized": true
                }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        validate_raw_control_weights(&mut payload).unwrap();
        let weights = payload["advanced"]["controlWeights"].as_object().unwrap();
        assert!(!weights.contains_key("_catalogAuthorized"));
        assert!(!weights.contains_key("revision"));
    }
}
