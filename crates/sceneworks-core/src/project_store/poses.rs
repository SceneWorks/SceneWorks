//! ProjectStore poses aggregate.

use super::*;

/// Reserved project that holds the GLOBAL pose library (epic 2282). User-created
/// poses live here as ordinary `type:"pose"` assets so the existing asset store +
/// Trashcan work unchanged; it is hidden from `list_projects` so it never appears in
/// the project switcher. Built-in poses stay bundled in the web app (read-only).
pub const GLOBAL_POSES_PROJECT_ID: &str = "project_global_poses";

pub const GLOBAL_POSES_PROJECT_NAME: &str = "Pose Library";

/// Staging cache dir for Pose Library source uploads (the `pose_detect` lane's sibling of
/// [`KEYPOINT_UPLOADS_CACHE_DIR`]): API staging route, startup sweep, and the worker's
/// `pose sourcePath` confinement + post-detect cleanup.
pub const POSE_UPLOADS_CACHE_DIR: &str = "pose-uploads";

impl ProjectStore {
    /// Ensure the reserved global pose library project exists (idempotent), returning
    /// its summary. Created lazily on first pose write/list (epic 2282, sc-2284).
    pub fn ensure_global_poses_project(&self) -> ProjectStoreResult<ProjectSummary> {
        let mut registry_cache = self.lock.lock();
        self.ensure_data_dirs()?;
        let project_path = self.projects_dir().join("global-poses.sceneworks");
        if project_path.exists() {
            return read_project_summary(&project_path);
        }
        self.provision_project_locked(
            GLOBAL_POSES_PROJECT_ID,
            GLOBAL_POSES_PROJECT_NAME,
            &project_path,
            &mut registry_cache,
        )
    }

    /// Resolve + guard a worker pose-detect cache preview
    /// (`<data_dir>/cache/pose_detect/<jobId>/<file>`). The job id must be a safe
    /// id and the filename a single normal component; the result is canonicalized
    /// and asserted to live under the pose-detect cache root, so a caller can never
    /// reach a file outside it. Shared by the preview endpoint and pose save.
    pub fn pose_preview_path(&self, job_id: &str, file_name: &str) -> ProjectStoreResult<PathBuf> {
        if !is_safe_id(job_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid pose jobId".to_owned(),
            ));
        }
        if !is_safe_relative_path(file_name) || Path::new(file_name).components().count() != 1 {
            return Err(ProjectStoreError::BadRequest(
                "Invalid pose preview filename".to_owned(),
            ));
        }
        let cache_root = self.data_dir.join("cache").join("pose_detect");
        let candidate = cache_root.join(job_id).join(file_name);
        // canonicalize resolves any symlink/`..` so the prefix check below is sound.
        let canonical_root = fs::canonicalize(&cache_root)
            .map_err(|_| ProjectStoreError::NotFound("Pose cache is unavailable".to_owned()))?;
        let canonical = fs::canonicalize(&candidate)
            .map_err(|_| ProjectStoreError::NotFound("Pose preview not found".to_owned()))?;
        if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
            return Err(ProjectStoreError::BadRequest(
                "Pose preview path is invalid".to_owned(),
            ));
        }
        Ok(canonical)
    }

    /// Persist a curated DWPose skeleton as a `type:"pose"` asset in the reserved
    /// global pose library (epic 2282, sc-2287). The skeleton PNG was already
    /// rendered by the worker's `pose_detect` job into
    /// `<data_dir>/cache/pose_detect/<jobId>/<file>`; here we copy it into
    /// `assets/poses/`, fold the chosen single `category` into the keypoint
    /// metadata, attach free `tags`, and write the sidecar + index (mirrors
    /// `import_asset`/`persist_generated_asset`). The source path is rebuilt from
    /// `data_dir` + a validated job id/filename and canonicalized under the
    /// pose-detect cache root, so a client can't copy a file from outside it.
    /// Returns the normalized asset. The reserved project is created lazily here.
    pub fn create_pose_asset(&self, spec: &Value) -> ProjectStoreResult<Value> {
        let job_id = spec
            .get("jobId")
            .and_then(Value::as_str)
            .ok_or_else(|| ProjectStoreError::BadRequest("pose spec missing jobId".to_owned()))?;
        let skeleton_file = spec
            .get("skeletonFile")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("pose spec missing skeletonFile".to_owned())
            })?;
        // Resolve + guard the cached skeleton path under the pose-detect cache root.
        let canonical_source = self.pose_preview_path(job_id, skeleton_file)?;

        // Reserved global pose project (created lazily on first save).
        self.ensure_global_poses_project()?;
        let (project_path, _project_guard) = self.lock_project(GLOBAL_POSES_PROJECT_ID)?;

        let asset_id = format!("asset_{}", random_hex(16)?);
        let created_at = utc_now();
        let poses_dir = project_path.join("assets").join("poses");
        fs::create_dir_all(&poses_dir)?;
        let media_path = poses_dir.join(format!("{asset_id}.png"));
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        // Copy (not move): the same cached preview may back other persons/candidates.
        fs::copy(&canonical_source, &media_path)?;
        let media_rel = relative_string(&project_path, &media_path)?;

        // Category is single-valued; fold it into the pose metadata so the screen can
        // group by `pose.category`. Free tags stay top-level (like the rest of the app).
        let category = spec
            .get("category")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let mut pose_meta = spec.get("pose").cloned().unwrap_or_else(|| json!({}));
        if let Some(object) = pose_meta.as_object_mut() {
            object.insert(
                "category".to_owned(),
                category.clone().map(Value::String).unwrap_or(Value::Null),
            );
        }
        let tags = match spec.get("tags").and_then(Value::as_array) {
            Some(values) => normalize_asset_tags(
                &values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect::<Vec<_>>(),
            )?,
            None => Vec::new(),
        };
        // displayName is required (schema minLength 1) — fall back to category, then a default.
        let display_name = spec
            .get("displayName")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .or_else(|| category.clone())
            .unwrap_or_else(|| "Pose".to_owned());
        let source_asset_id = pose_meta
            .get("sourceAssetId")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                spec.get("sourceAssetId")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            });
        let parents = source_asset_id
            .clone()
            .map(|id| vec![Value::String(id)])
            .unwrap_or_default();

        let asset = json!({
            "schemaVersion": 1,
            "id": asset_id,
            "projectId": GLOBAL_POSES_PROJECT_ID,
            "generationSetId": Value::Null,
            "type": "pose",
            "displayName": display_name,
            "createdAt": created_at,
            "origin": "pose_library",
            "tags": tags,
            "file": {
                "path": media_rel,
                "mimeType": "image/png",
                "width": spec.get("width").cloned().unwrap_or(Value::Null),
                "height": spec.get("height").cloned().unwrap_or(Value::Null),
                "duration": Value::Null,
                "fps": Value::Null
            },
            "status": { "favorite": false, "rating": 0, "rejected": false, "trashed": false },
            "recipe": {
                "mode": "upload",
                "model": "dwpose",
                "adapter": "api-upload",
                "prompt": display_name,
                "negativePrompt": "",
                "seed": 0,
                "loras": [],
                "stylePreset": "none",
                "normalizedSettings": {},
                "rawAdapterSettings": { "detector": "dwpose", "jobId": job_id }
            },
            "lineage": {
                "parents": parents,
                "sourceAssetId": source_asset_id,
                "sourceTimestamp": Value::Null,
                "jobId": job_id
            },
            "pose": pose_meta
        });
        let sidecar_path = media_path.with_extension("sceneworks.json");
        write_json(&sidecar_path, &asset)?;
        index_asset(
            GLOBAL_POSES_PROJECT_ID,
            &project_path,
            &asset,
            Some(&sidecar_path),
        )?;
        index_mutation.commit();
        normalize_asset(GLOBAL_POSES_PROJECT_ID, &project_path, &sidecar_path)
    }
}
