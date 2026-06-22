//! LoRA/model import + copy helpers, install-marker writers, and download-dir glob matching.
use super::*;

pub async fn copy_lora_source(source: &Path, target_dir: &Path) -> WorkerResult<()> {
    import_lora_source_path(source, target_dir, false).await
}

pub(crate) async fn import_lora_source_path(
    source: &Path,
    target_dir: &Path,
    prefer_move: bool,
) -> WorkerResult<()> {
    let source = source.canonicalize()?;
    if !source.exists() {
        return Err(WorkerError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("LoRA source not found: {}", source.display()),
        )));
    }
    tokio::fs::create_dir_all(target_dir).await?;
    if source.is_dir() {
        copy_dir_recursive(&source, target_dir).await?;
    } else {
        let target = target_dir.join(source.file_name().ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA source has no filename".to_owned())
        })?);
        if prefer_move {
            match tokio::fs::rename(&source, &target).await {
                Ok(()) => return Ok(()),
                Err(error) if is_cross_device_rename_error(&error) => {}
                Err(error) => return Err(error.into()),
            }
        }
        tokio::fs::copy(source, target).await?;
    }
    Ok(())
}

/// Write one staged LoRA upload into `target_dir` under an explicit
/// `target_filename`, renaming it from its uploaded name. Used for paired Wan A14B
/// MoE imports (sc-1991): the two staged uploads must land as
/// `<stem>.high_noise.safetensors` / `<stem>.low_noise.safetensors` so the Python
/// worker's filename-convention split detects them as one two-expert pair.
pub(crate) async fn import_lora_source_file_as(
    source: &Path,
    target_dir: &Path,
    target_filename: &str,
    prefer_move: bool,
) -> WorkerResult<()> {
    let source = source.canonicalize()?;
    tokio::fs::create_dir_all(target_dir).await?;
    let target = target_dir.join(target_filename);
    if prefer_move {
        match tokio::fs::rename(&source, &target).await {
            Ok(()) => return Ok(()),
            Err(error) if is_cross_device_rename_error(&error) => {}
            Err(error) => return Err(error.into()),
        }
    }
    tokio::fs::copy(source, target).await?;
    Ok(())
}

/// The `<stem>.high_noise.safetensors` / `<stem>.low_noise.safetensors` filenames
/// for a Wan A14B MoE LoRA pair stored under one record. The high-noise file sorts
/// first alphabetically, so it resolves as the primary (transformer) and the
/// low-noise file as the `transformer_2` sibling.
pub(crate) fn wan_moe_pair_filenames(stem: &str) -> (String, String) {
    (
        format!("{stem}.high_noise.safetensors"),
        format!("{stem}.low_noise.safetensors"),
    )
}

pub(crate) fn is_cross_device_rename_error(error: &std::io::Error) -> bool {
    matches!(error.raw_os_error(), Some(17 | 18))
}

pub(crate) async fn copy_dir_recursive(source: &Path, target: &Path) -> WorkerResult<()> {
    let mut stack = vec![(source.to_path_buf(), target.to_path_buf())];
    while let Some((source_dir, target_dir)) = stack.pop() {
        tokio::fs::create_dir_all(&target_dir).await?;
        let mut entries = tokio::fs::read_dir(&source_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let file_type = entry.file_type().await?;
            let destination = target_dir.join(entry.file_name());
            if file_type.is_dir() {
                stack.push((entry.path(), destination));
            } else if file_type.is_file() {
                tokio::fs::copy(entry.path(), destination).await?;
            }
        }
    }
    Ok(())
}

pub(crate) async fn write_model_install_marker(
    target_dir: &Path,
    payload: &JsonObject,
    repo: &str,
    job_id: &str,
) -> WorkerResult<()> {
    tokio::fs::create_dir_all(target_dir).await?;
    let marker = json!({
        "repo": repo,
        "modelId": payload.get("modelId").cloned().unwrap_or(Value::Null),
        "modelName": payload.get("modelName").cloned().unwrap_or(Value::Null),
        "jobId": job_id,
        "completedAt": now_rfc3339(),
    });
    let bytes = serde_json::to_vec_pretty(&marker)?;
    tokio::fs::write(target_dir.join(INSTALL_MARKER), bytes).await?;
    Ok(())
}

pub(crate) async fn write_lora_install_marker(
    target_dir: &Path,
    payload: &JsonObject,
    job_id: &str,
) -> WorkerResult<()> {
    tokio::fs::create_dir_all(target_dir).await?;
    let marker = json!({
        "loraId": payload.get("loraId").cloned().unwrap_or(Value::Null),
        "loraName": payload.get("name").cloned().unwrap_or(Value::Null),
        "repo": payload.get("repo").cloned().unwrap_or(Value::Null),
        "sourceUrl": payload.get("sourceUrl").cloned().unwrap_or(Value::Null),
        "sourcePath": payload.get("sourcePath").cloned().unwrap_or(Value::Null),
        "jobId": job_id,
        "completedAt": now_rfc3339(),
    });
    let bytes = serde_json::to_vec_pretty(&marker)?;
    tokio::fs::write(target_dir.join(INSTALL_MARKER), bytes).await?;
    Ok(())
}

pub fn allow_pattern_matches(path: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    patterns
        .iter()
        .any(|pattern| pattern_matches(pattern, path))
}

pub(crate) fn pattern_matches(pattern: &str, value: &str) -> bool {
    let (pattern, value) = if cfg!(windows) {
        (pattern.to_ascii_lowercase(), value.to_ascii_lowercase())
    } else {
        (pattern.to_owned(), value.to_owned())
    };
    glob::Pattern::new(&pattern).is_ok_and(|pattern| pattern.matches(&value))
}

pub(crate) async fn cleanup_uploaded_import_source(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<()> {
    if !payload_bool(payload, "uploadedSourcePath") {
        return Ok(());
    }
    let Some(source_path) = optional_payload_string(payload, "sourcePath") else {
        return Ok(());
    };
    let source_path = normalize_absolute_path(Path::new(source_path))?;
    let allowed_roots = [
        normalize_absolute_path(&settings.data_dir.join("cache").join("lora-uploads"))?,
        normalize_absolute_path(&settings.data_dir.join("cache").join("model-uploads"))?,
    ];
    let source_path = ensure_path_under(source_path, &allowed_roots, "Uploaded sourcePath")?;
    let _ = tokio::fs::remove_file(&source_path).await;
    if let Some(parent) = source_path.parent() {
        if allowed_roots
            .iter()
            .any(|root| parent.starts_with(root) && parent != root)
        {
            let _ = tokio::fs::remove_dir(parent).await;
        }
    }
    Ok(())
}
