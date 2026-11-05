//! ProjectStore media aggregate.

use super::*;

/// Public file downloads expose product media and explicit authoring/export documents,
/// not the project storage tree. Keep the allowlist here so token and media-ticket
/// callers share it, and check both the requested and canonical relative paths.
fn is_public_project_file(path: &Path) -> bool {
    let Some(path) = path.to_str() else {
        return false;
    };
    let normalized = path.replace('\\', "/");
    let parts = normalized.split('/').collect::<Vec<_>>();
    if parts
        .iter()
        .any(|part| part.starts_with('.') || part.is_empty())
    {
        return false;
    }
    let name = parts
        .last()
        .copied()
        .unwrap_or_default()
        .to_ascii_lowercase();
    // Durable sidecars are API state, even when next to a public image or document.
    if name.contains(".sceneworks.") {
        return false;
    }
    let extension = name
        .rsplit_once('.')
        .map(|(_, ext)| ext)
        .unwrap_or_default();
    let media = SAFE_UPLOAD_EXTENSIONS.contains(&extension)
        || matches!(
            extension,
            "svg" | "mp3" | "flac" | "ogg" | "opus" | "m4a" | "aac" | "aiff" | "aif"
        );
    let tensor = matches!(extension, "safetensors" | "npy" | "npz");
    let document = matches!(
        extension,
        "json" | "jsonc" | "txt" | "md" | "csv" | "tsv" | "srt" | "vtt" | "pdf"
    );
    match parts.as_slice() {
        ["assets", "documents", ..] => media || document,
        // MOSS writes one portable authoring envelope alongside each numbered
        // tensor output. The UI downloads it and the asset-workflow reader verifies
        // its recorded digest. This is not a general JSON/state-file allowance.
        ["assets", "tensors", _, file]
            if file
                .strip_suffix(".sceneworks-workflow.json")
                .is_some_and(|sample| {
                    sample.len() >= 4 && sample.bytes().all(|byte| byte.is_ascii_digit())
                }) =>
        {
            true
        }
        ["assets", "tensors", ..] => media || tensor,
        ["assets", "depth", ..] => media || extension == "npy",
        // Uploaded NPY assets and the importer's inert fallback .bin are public;
        // training latent/optimizer caches with these suffixes are not media.
        ["assets", "uploads", ..] => media || tensor || extension == "bin",
        ["training", "uploads", ..] => media || extension == "bin",
        // These numerical files are authored inputs with source-owned layouts,
        // distinct from the trainer's private latent/optimizer caches.
        ["training", "datasets", _, "prepared", _] => extension == "safetensors",
        ["training", "datasets", _, "sidecars", file] if extension == "npy" => {
            file.ends_with(".depth.npy") || file.ends_with(".mask.npy")
        }
        ["assets", ..]
        | ["training", "datasets" | "samples", ..]
        | ["films", "draft-assets", ..] => media,
        // Project-scoped training stores previews beside the adapter, while
        // global adapters keep theirs under training/samples. Do not expose
        // adapter manifests, receipts or optimizer/checkpoint state.
        ["loras", _, "samples", ..] | ["training", "loras", _, "samples", ..] => media,
        ["films", "drafts", file] => file.ends_with(".json"),
        ["films", "runs", _, file]
            if matches!(
                *file,
                "authoring.json"
                    | "plan.json"
                    | "references.json"
                    | "compiled.json"
                    | "review.jsonc"
            ) =>
        {
            true
        }
        ["films", "runs", _, ..] => media,
        ["timeline-exports" | "exports", ..] => {
            media
                || tensor
                || document
                || matches!(
                    extension,
                    "bin" | "zip" | "otio" | "otioz" | "fcpxml" | "xml" | "edl" | "aaf" | "mlt"
                )
        }
        _ => false,
    }
}

/// Resolve a project-relative deletion target without following linked parents.
///
/// Purge paths come from durable sidecars and the derived index, both of which can
/// be modified outside the application. A lexical `..` check is therefore not
/// enough: an otherwise-normal component can be a symlink or Windows reparse
/// point that redirects the later trash/remove operation outside the project.
pub(super) fn confined_deletion_target(
    project_path: &Path,
    relative_path: &str,
    label: &str,
) -> ProjectStoreResult<PathBuf> {
    if !is_safe_relative_path(relative_path) {
        return Err(ProjectStoreError::BadRequest(format!(
            "{label} path must be project-relative"
        )));
    }

    let root = fs::canonicalize(project_path)?;
    let components = Path::new(relative_path).components().collect::<Vec<_>>();
    let mut target = root.clone();
    for (index, component) in components.iter().enumerate() {
        let std::path::Component::Normal(component) = component else {
            return Err(ProjectStoreError::BadRequest(format!(
                "{label} path must be project-relative"
            )));
        };
        target.push(component);
        let metadata = match fs::symlink_metadata(&target) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                for remainder in components.iter().skip(index + 1) {
                    if let std::path::Component::Normal(remainder) = remainder {
                        target.push(remainder);
                    }
                }
                return Ok(target);
            }
            Err(error) => return Err(ProjectStoreError::Io(error)),
        };
        if metadata.file_type().is_symlink() || metadata_is_reparse_point(&metadata) {
            return Err(ProjectStoreError::BadRequest(format!(
                "{label} path contains a symlink or reparse point"
            )));
        }
        if index + 1 < components.len() && !metadata.is_dir() {
            return Err(ProjectStoreError::BadRequest(format!(
                "{label} path contains a non-directory parent"
            )));
        }
        if !fs::canonicalize(&target)?.starts_with(&root) {
            return Err(ProjectStoreError::BadRequest(format!(
                "{label} path escapes the project"
            )));
        }
    }
    Ok(target)
}

#[cfg(windows)]
fn metadata_is_reparse_point(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn metadata_is_reparse_point(_metadata: &fs::Metadata) -> bool {
    false
}

/// Remove a source sibling poster without ever traversing a symlinked parent.
///
/// Poster bytes are already durable in `project.db`; this lifecycle cleanup
/// prevents a purged/orphaned id reused at the same media path from inheriting
/// its predecessor's source poster.
pub(super) fn remove_sibling_poster_no_symlinks(
    project_path: &Path,
    media_rel: &str,
) -> ProjectStoreResult<()> {
    if !is_safe_relative_path(media_rel) {
        return Ok(());
    }
    let poster_rel = Path::new(media_rel).with_extension("poster.jpg");
    let root = fs::canonicalize(project_path)?;
    let mut target = root.clone();
    let components = poster_rel.components().collect::<Vec<_>>();
    for component in components.iter().take(components.len().saturating_sub(1)) {
        let std::path::Component::Normal(component) = component else {
            return Ok(());
        };
        target.push(component);
        let Ok(metadata) = fs::symlink_metadata(&target) else {
            return Ok(());
        };
        if metadata.file_type().is_symlink() {
            return Ok(());
        }
    }
    let Some(std::path::Component::Normal(file_name)) = components.last() else {
        return Ok(());
    };
    target.push(file_name);
    let Ok(metadata) = fs::symlink_metadata(&target) else {
        return Ok(());
    };
    if metadata.is_dir() {
        return Ok(());
    }
    if !metadata.file_type().is_symlink() {
        let Ok(canonical_target) = fs::canonicalize(&target) else {
            return Ok(());
        };
        if !canonical_target.starts_with(&root) {
            return Ok(());
        }
    }
    fs::remove_file(target)?;
    Ok(())
}

impl ProjectStore {
    pub fn get_asset_poster(
        &self,
        project_id: &str,
        asset_id: &str,
        poster_sha256: &str,
    ) -> ProjectStoreResult<AssetPoster> {
        if !is_safe_id(asset_id)
            || poster_sha256.len() != 64
            || !poster_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(ProjectStoreError::BadRequest(
                "Invalid indexed poster id".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let _project_guard = lock_project_files(&project_path);
        let connection = connect_project_db_ready_locked(&project_path)?;
        let poster = connection
            .query_row(
                "
                select poster_bytes, poster_sha256
                  from assets
                 where id = ?1
                   and poster_bytes is not null
                ",
                params![asset_id],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .optional()?
            .ok_or_else(|| ProjectStoreError::NotFound("Poster not found".to_owned()))?;
        let (bytes, Some(stored_sha256)) = poster else {
            return Err(ProjectStoreError::NotFound("Poster not found".to_owned()));
        };
        let recomputed_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let stored_matches_bytes: bool = stored_sha256
            .as_bytes()
            .ct_eq(recomputed_sha256.as_bytes())
            .into();
        let request_matches_bytes: bool = poster_sha256
            .as_bytes()
            .ct_eq(recomputed_sha256.as_bytes())
            .into();
        if !stored_matches_bytes || !request_matches_bytes {
            return Err(ProjectStoreError::NotFound("Poster not found".to_owned()));
        }
        Ok(AssetPoster {
            bytes,
            sha256: stored_sha256,
        })
    }

    pub fn project_file(
        &self,
        project_id: &str,
        relative_path: &str,
    ) -> ProjectStoreResult<ProjectFile> {
        if !is_safe_relative_path(relative_path) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid project file path".to_owned(),
            ));
        }
        if !is_public_project_file(Path::new(relative_path)) {
            return Err(ProjectStoreError::NotFound("File not found".to_owned()));
        }
        let project_path = self.find_project_path(project_id)?;
        let root = fs::canonicalize(&project_path)?;
        let target = crate::safe_path::resolve_local_file(&project_path, relative_path).map_err(
            |error| match error.kind() {
                std::io::ErrorKind::NotFound => {
                    ProjectStoreError::NotFound("File not found".to_owned())
                }
                std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::InvalidInput => {
                    ProjectStoreError::BadRequest("Invalid project file path".to_owned())
                }
                _ => ProjectStoreError::Io(error),
            },
        )?;
        // Recheck the canonical relative path too: a public-looking symlink must
        // never turn this read-only media boundary into access to internal state.
        if !target.is_file() || !target.strip_prefix(&root).is_ok_and(is_public_project_file) {
            return Err(ProjectStoreError::NotFound("File not found".to_owned()));
        }
        let content_type = guess_mime_from_filename(&target.display().to_string())
            .unwrap_or_else(|| "application/octet-stream".to_owned());
        Ok(ProjectFile {
            path: target,
            content_type,
        })
    }
}
