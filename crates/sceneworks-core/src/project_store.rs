use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use parking_lot::{Mutex, ReentrantMutexGuard};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::asset_index::{
    asset_origin, asset_sidecars, find_asset_sidecar_path_on_connection, index_asset_on_connection,
    normalize_asset, normalize_asset_cached, GenerationSetCache,
};
use crate::character_store::{
    apply_character_migrations, clear_character_tables, reindex_characters_on_connection,
    CharacterStore,
};
pub use crate::character_store::{
    CharacterCreateInput, CharacterLookInput, CharacterLookUpdateInput, CharacterLoraInput,
    CharacterLoraUpdateInput, CharacterMutationResult, CharacterReferenceInput,
    CharacterReferenceUpdateInput, CharacterUpdateInput, CHARACTER_SIDECAR_PATTERN,
};
use crate::dataset_quality::{
    CachedTier0Scalars, DatasetEmbeddings, DatasetFaceRecords, QualityAck, QualityCheck,
};
use crate::slug::slugify;
use crate::store_util::{
    ensure_column, is_safe_id, is_safe_relative_path, lock_project_files, optional_f64,
    optional_str, optional_u64, random_hex, read_json, relative_string, write_json,
};
use crate::time::utc_now;
use crate::training::TrainingDataset;
use crate::training_store::{
    apply_training_dataset_migrations, DatasetItemRepoint, TrainingCaptionSidecarsResult,
    TrainingDatasetBatchRenameInput, TrainingDatasetCaptionSidecarsInput,
    TrainingDatasetCreateInput, TrainingDatasetMutationResult, TrainingDatasetStore,
    TrainingDatasetSummary, TrainingDatasetUpdateInput,
};

pub const PROJECT_FOLDERS: &[&str] = &[
    "assets/images",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "assets/documents",
    "assets/poses",
    "characters",
    "generation-sets",
    "loras",
    "person-tracks",
    "recipes",
    "timelines",
    "training/datasets",
    "training/uploads",
    "trash",
    "cache",
];

/// Reserved project that holds the GLOBAL pose library (epic 2282). User-created
/// poses live here as ordinary `type:"pose"` assets so the existing asset store +
/// Trashcan work unchanged; it is hidden from `list_projects` so it never appears in
/// the project switcher. Built-in poses stay bundled in the web app (read-only).
pub const GLOBAL_POSES_PROJECT_ID: &str = "project_global_poses";
pub const GLOBAL_POSES_PROJECT_NAME: &str = "Pose Library";

/// Reserved project that holds the GLOBAL Key Point Library (epic 4422, sc-4434). User-created
/// face-angle presets live here as ordinary `type:"keypoint"` assets (kps + the retained source
/// image), and the user's angle-set collections live in a `keypoint-collections.json` sidecar.
/// Hidden from `list_projects` like the pose library. The built-in 11 angle presets stay in
/// [`crate::angle_kps`] (virtual, always-present, protected — never stored here).
pub const GLOBAL_KEYPOINTS_PROJECT_ID: &str = "project_global_keypoints";
pub const GLOBAL_KEYPOINTS_PROJECT_NAME: &str = "Key Point Library";
/// The user angle-set collections store, relative to the global keypoints project.
const KEYPOINT_COLLECTIONS_FILE: &str = "keypoint-collections.json";

/// Staging cache dir (under `<data>/cache`) for Key Point Library source uploads. One
/// constant shared by the API staging route (`create_keypoint_sources`), the store's
/// save-preset guard ([`ProjectStore`]'s keypoint upload check), and the worker's
/// `kps sourcePath` confinement — sc-13713 was these drifting apart (the worker confined
/// to the pose lane's dir, rejecting every capture).
pub const KEYPOINT_UPLOADS_CACHE_DIR: &str = "keypoint-uploads";
/// Staging cache dir for Pose Library source uploads (the `pose_detect` lane's sibling of
/// [`KEYPOINT_UPLOADS_CACHE_DIR`]): API staging route, startup sweep, and the worker's
/// `pose sourcePath` confinement + post-detect cleanup.
pub const POSE_UPLOADS_CACHE_DIR: &str = "pose-uploads";

/// One resolved angle preset for generation (sc-4450): the kps to draw + a display name + (for
/// built-ins) the canonical angle name that drives prompt augmentation. User presets have
/// `angle: None` (no canonical prompt clause). Produced by [`ProjectStore::resolve_angle_collection`].
#[derive(Debug, Clone)]
pub struct ResolvedAnglePreset {
    pub preset_id: String,
    pub name: String,
    pub angle: Option<String>,
    pub kps: [(f32, f32); 5],
}

pub type ProjectStoreResult<T> = Result<T, ProjectStoreError>;

#[derive(Debug)]
pub enum ProjectStoreError {
    Io(std::io::Error),
    Sqlite(rusqlite::Error),
    Json(serde_json::Error),
    BadRequest(String),
    NotFound(String),
    /// The workspace storage location rejected the writes a project needs — the
    /// raw SQLite/IO error ("attempt to write a readonly database", permission
    /// denied) is opaque, so this carries an actionable, path-naming message
    /// instead (issue #1435 / sc-11855). Surfaced when the projects tree can be
    /// read/created but not modified in place (e.g. a macOS inherited ACL or an
    /// endpoint-security/backup agent that blocks in-place file rewrites).
    StorageNotWritable(String),
}

impl std::fmt::Display for ProjectStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Sqlite(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::BadRequest(detail) => write!(formatter, "{detail}"),
            Self::NotFound(detail) => write!(formatter, "{detail}"),
            Self::StorageNotWritable(detail) => write!(formatter, "{detail}"),
        }
    }
}

impl std::error::Error for ProjectStoreError {}

impl From<std::io::Error> for ProjectStoreError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<rusqlite::Error> for ProjectStoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Sqlite(error)
    }
}

impl From<serde_json::Error> for ProjectStoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub id: String,
    pub name: String,
    pub path: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReindexResult {
    pub project_id: String,
    pub assets: u32,
    pub generation_sets: u32,
    pub timelines: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineSummary {
    pub id: String,
    pub name: String,
    pub file_path: String,
    pub aspect_ratio: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration: f64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineFile {
    pub path: PathBuf,
    pub relative_path: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimelineFileDocument {
    pub file: TimelineFile,
    pub document: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetMutationResult {
    pub id: String,
    pub status: String,
}

/// Which assets a listing should include (sc-2024). `All` is the historical
/// behaviour used by Image/Video studios, the Editor, and the per-character
/// gallery. `Library` is the Asset Library view, which excludes Character Studio
/// test outputs (`origin = character_studio`) so they live only under the
/// character. Dataset images are already excluded — they are stored outside the
/// indexed asset folders and never enter the assets table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssetScope {
    #[default]
    All,
    Library,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetStatusPatch {
    pub favorite: Option<bool>,
    pub rating: Option<u8>,
    pub rejected: Option<bool>,
    pub trashed: Option<bool>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetTagsPatch {
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct UploadAsset {
    pub filename: String,
    pub content_type: Option<String>,
    pub source_path: PathBuf,
    /// Optional source asset this upload was derived from (Image Editor Save,
    /// sc-2434). Sets `lineage.parents`/`sourceAssetId` so the saved edit links
    /// back to the asset it was opened from; the source is never modified.
    pub source_asset_id: Option<String>,
    /// Optional free-form provenance (e.g. the Image Editor edit chain) stored
    /// under the asset's top-level `extra`, mirroring generated-asset extras.
    pub provenance: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct ProjectFile {
    pub path: PathBuf,
    pub content_type: String,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct RegistryItem {
    id: Option<String>,
    name: Option<String>,
    path: Option<String>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Debug)]
pub struct ProjectStore {
    data_dir: PathBuf,
    app_version: String,
    /// Guards recent-project registry reads/writes only; project DB and project-file mutations
    /// rely on SQLite transactions and filesystem operations for their own serialization.
    lock: Mutex<()>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingDatasetUpload {
    pub filename: String,
    pub content_type: Option<String>,
    pub source_path: PathBuf,
}

impl ProjectStore {
    pub fn new(data_dir: impl Into<PathBuf>, app_version: impl Into<String>) -> Self {
        Self {
            data_dir: data_dir.into(),
            app_version: app_version.into(),
            lock: Mutex::new(()),
        }
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.data_dir.join("projects")
    }

    pub fn registry_path(&self) -> PathBuf {
        self.data_dir.join("recent-projects.json")
    }

    pub fn list_projects(&self) -> ProjectStoreResult<Vec<ProjectSummary>> {
        let _guard = self.lock.lock();
        self.ensure_data_dirs()?;
        let mut projects = Vec::new();
        for item in self.load_registry()? {
            // The reserved global pose + keypoint libraries are addressable by id but
            // hidden from the project switcher (epic 2282 / epic 4422).
            if matches!(
                item.id.as_deref(),
                Some(GLOBAL_POSES_PROJECT_ID) | Some(GLOBAL_KEYPOINTS_PROJECT_ID)
            ) {
                continue;
            }
            let Some(path) = item.path else {
                continue;
            };
            let project_path = PathBuf::from(path);
            if project_path.exists() {
                // Fail open per entry: one project whose `project.json` is missing,
                // unreadable, or short a required field must not take down the whole
                // listing and leave every healthy project unreachable. Skip the bad
                // entry with a structured warning and keep the rest.
                match read_project_summary(&project_path) {
                    Ok(summary) => projects.push(summary),
                    Err(error) => tracing::warn!(
                        event = "list_projects_summary_failed",
                        project = %project_path.display(),
                        error = %error,
                        "skipping project with unreadable project.json"
                    ),
                }
            }
        }
        Ok(projects)
    }

    pub fn create_project(&self, name: &str) -> ProjectStoreResult<ProjectSummary> {
        let name = name.trim();
        if name.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Project name is required".to_owned(),
            ));
        }
        if name.chars().count() > 120 {
            return Err(ProjectStoreError::BadRequest(
                "Project name must be at most 120 characters".to_owned(),
            ));
        }

        let _guard = self.lock.lock();
        self.ensure_data_dirs()?;
        // Fail fast with an actionable error if the workspace folder rejects writes,
        // rather than surfacing an opaque SQLITE_READONLY from deep in provisioning
        // after a half-built project dir already exists (issue #1435 / sc-11855).
        if let Err(error) = probe_storage_writable(&self.projects_dir()) {
            if is_write_denied(&error) {
                return Err(ProjectStoreError::StorageNotWritable(
                    storage_not_writable_message(&self.projects_dir()),
                ));
            }
            return Err(error);
        }
        let project_id = format!("project_{}", random_hex(16)?);
        let slug = slugify(name, "project", None);
        let mut project_path = self.projects_dir().join(format!("{slug}.sceneworks"));
        if project_path.exists() {
            let suffix = &project_id[project_id.len().saturating_sub(8)..];
            project_path = self
                .projects_dir()
                .join(format!("{slug}-{suffix}.sceneworks"));
        }

        self.provision_project_locked(&project_id, name, &project_path)
    }

    /// Provision a project directory (folders + project file + db + registry entry)
    /// for a known id/path. Caller must hold `self.lock`. Shared by `create_project`
    /// (random id) and `ensure_global_poses_project` (fixed id).
    fn provision_project_locked(
        &self,
        project_id: &str,
        name: &str,
        project_path: &Path,
    ) -> ProjectStoreResult<ProjectSummary> {
        fs::create_dir_all(project_path)?;
        for folder in PROJECT_FOLDERS {
            fs::create_dir_all(project_path.join(folder))?;
        }
        write_project_file(&self.app_version, project_path, project_id, name)?;
        // Backstop for the create_project preflight (and the only guard on the
        // lazily-provisioned global pose/keypoint projects): translate a write
        // denial on the project.db migration into the actionable, path-naming
        // error instead of a raw SQLITE_READONLY (issue #1435 / sc-11855).
        if let Err(error) =
            connect_project_db(project_path).and_then(|conn| apply_project_migrations(&conn))
        {
            return Err(if is_write_denied(&error) {
                ProjectStoreError::StorageNotWritable(storage_not_writable_message(
                    project_path.parent().unwrap_or(project_path),
                ))
            } else {
                error
            });
        }

        let mut registry = self
            .load_registry()?
            .into_iter()
            .filter(|item| item.id.as_deref() != Some(project_id))
            .collect::<Vec<_>>();
        registry.insert(
            0,
            RegistryItem {
                id: Some(project_id.to_owned()),
                name: Some(name.to_owned()),
                path: Some(project_path.display().to_string()),
                extra: Map::new(),
            },
        );
        self.save_registry(&registry)?;

        read_project_summary(project_path)
    }

    /// Ensure the reserved global pose library project exists (idempotent), returning
    /// its summary. Created lazily on first pose write/list (epic 2282, sc-2284).
    pub fn ensure_global_poses_project(&self) -> ProjectStoreResult<ProjectSummary> {
        let _guard = self.lock.lock();
        self.ensure_data_dirs()?;
        let project_path = self.projects_dir().join("global-poses.sceneworks");
        if project_path.exists() {
            return read_project_summary(&project_path);
        }
        self.provision_project_locked(
            GLOBAL_POSES_PROJECT_ID,
            GLOBAL_POSES_PROJECT_NAME,
            &project_path,
        )
    }

    /// Ensure the reserved global Key Point Library project exists (idempotent), returning its
    /// summary. Created lazily on first preset/collection write (epic 4422, sc-4434).
    pub fn ensure_global_keypoints_project(&self) -> ProjectStoreResult<ProjectSummary> {
        let _guard = self.lock.lock();
        self.ensure_data_dirs()?;
        let project_path = self.projects_dir().join("global-keypoints.sceneworks");
        if project_path.exists() {
            return read_project_summary(&project_path);
        }
        self.provision_project_locked(
            GLOBAL_KEYPOINTS_PROJECT_ID,
            GLOBAL_KEYPOINTS_PROJECT_NAME,
            &project_path,
        )
    }

    pub fn get_project(&self, project_id: &str) -> ProjectStoreResult<ProjectSummary> {
        let project_path = self.find_project_path(project_id)?;
        read_project_summary(&project_path)
    }

    pub fn list_training_datasets(
        &self,
        project_id: &str,
    ) -> ProjectStoreResult<Vec<TrainingDatasetSummary>> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path).list_datasets(project_id)
    }

    pub fn create_training_dataset(
        &self,
        project_id: &str,
        input: TrainingDatasetCreateInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).create_dataset(project_id, input)
    }

    pub fn upload_training_dataset_item(
        &self,
        project_id: &str,
        upload: TrainingDatasetUpload,
    ) -> ProjectStoreResult<Value> {
        if fs::metadata(&upload.source_path)?.len() == 0 {
            return Err(ProjectStoreError::BadRequest(
                "Uploaded file is empty".to_owned(),
            ));
        }
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let upload_dir = project_path.join("training").join("uploads");
        fs::create_dir_all(&upload_dir)?;

        let guessed_mime = guess_mime_from_filename(&upload.filename);
        let content_type = upload
            .content_type
            .as_deref()
            .filter(|value| !value.is_empty() && *value != "application/octet-stream")
            .map(str::to_owned)
            .or(guessed_mime)
            .unwrap_or_else(|| "application/octet-stream".to_owned());
        if !content_type.starts_with("image/") {
            return Err(ProjectStoreError::BadRequest(
                "Only image dataset uploads are supported".to_owned(),
            ));
        }

        // sc-6143: training reads dataset images straight through the engine (no worker decode
        // backstop on this path), so normalize a valid-but-unsupported format to lossless PNG here
        // — the only lever that makes AVIF/HEIC training images decode.
        let normalized = normalize_image_upload(
            &upload.source_path,
            &content_type,
            &upload.filename,
            &upload_dir,
        )?;
        let content_type = normalized.content_type.clone();

        let upload_id = format!("dataset_upload_{}", random_hex(16)?);
        let extension = normalized.extension.clone();
        let suffix = &upload_id[upload_id.len().saturating_sub(8)..];
        let filename = format!(
            "{}-{suffix}{extension}",
            safe_filename(&upload.filename, &upload_id)
        );
        let media_path = upload_dir.join(filename);
        move_normalized_upload(&normalized, &upload.source_path, &media_path)?;
        let media_rel = relative_string(&project_path, &media_path)?;
        let display_name = Path::new(&upload.filename)
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("dataset-upload")
            .to_owned();

        // sc-6531 (Dataset Doctor): emit the stored image's real pixel dimensions + a content
        // hash instead of the historical `Null`s, so the asset layer / pre-train readout
        // (`datasetHelpers.js`) and Tier-0 checks have them up front. Read from the normalized
        // stored file — header-only dims + a streamed hash, no full decode (core stays codec-free).
        let (width, height) = crate::media_convert::image_dimensions(&media_path)
            .map_or((Value::Null, Value::Null), |(w, h)| (json!(w), json!(h)));
        let content_hash = crate::media_convert::file_content_hash(&media_path)
            .ok()
            .map_or(Value::Null, Value::String);

        Ok(json!({
            "id": upload_id,
            "projectId": project_id,
            "datasetOnly": true,
            "type": "image",
            "displayName": display_name,
            "file": {
                "path": media_rel,
                "mimeType": content_type,
                "width": width,
                "height": height,
                "contentHash": content_hash
            },
            "url": format!("/api/v1/projects/{project_id}/files/{media_rel}")
        }))
    }

    pub fn get_training_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<TrainingDataset> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path).get_dataset(project_id, dataset_id)
    }

    /// Loads a training dataset together with its absolute on-disk root and the
    /// project stem — the inputs the Rust API needs to resolve a
    /// [`crate::training::TrainingPlan`] and label the queued job. Resolves the
    /// project path once rather than re-reading the registry per value.
    pub fn training_dataset_for_plan(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<(TrainingDataset, PathBuf, String)> {
        let project_path = self.find_project_path(project_id)?;
        let dataset =
            TrainingDatasetStore::new(project_path.clone()).get_dataset(project_id, dataset_id)?;
        let root = crate::training_store::dataset_root(&project_path, dataset_id);
        let project_stem = project_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_owned();
        Ok((dataset, root, project_stem))
    }

    pub fn update_training_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetUpdateInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).update_dataset(project_id, dataset_id, input)
    }

    /// Persist freshly-extracted Tier-0 scalars onto dataset items as the readiness cache (sc-6533).
    /// Locked like any dataset mutation; a metadata-only write (no version/`updated_at` bump).
    pub fn cache_dataset_tier0_scalars(
        &self,
        project_id: &str,
        dataset_id: &str,
        updates: &[(String, CachedTier0Scalars)],
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).cache_tier0_scalars(project_id, dataset_id, updates)
    }

    /// Persist (or clear) a per-image quality override (sc-6534). Locked wrapper over
    /// [`TrainingDatasetStore::set_item_quality_ack`].
    pub fn set_dataset_item_quality_ack(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_id: &str,
        checks: &[QualityCheck],
        expected_content_hash: Option<&str>,
        expected_caption_hash: Option<&str>,
    ) -> ProjectStoreResult<Option<QualityAck>> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).set_item_quality_ack(
            project_id,
            dataset_id,
            item_id,
            checks,
            expected_content_hash,
            expected_caption_hash,
        )
    }

    /// Persist the dataset's CLIP embeddings sidecar (sc-6535). Locked wrapper over
    /// [`TrainingDatasetStore::write_dataset_embeddings`].
    pub fn write_dataset_embeddings(
        &self,
        project_id: &str,
        dataset_id: &str,
        embeddings: &DatasetEmbeddings,
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .write_dataset_embeddings(project_id, dataset_id, embeddings)
    }

    /// Re-point dataset items at derived (e.g. upscaled) child assets (sc-6539). Locked wrapper over
    /// [`TrainingDatasetStore::repoint_dataset_items`].
    pub fn repoint_dataset_items(
        &self,
        project_id: &str,
        dataset_id: &str,
        repoints: &[DatasetItemRepoint],
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .repoint_dataset_items(project_id, dataset_id, repoints)
    }

    /// Backfill a training dataset's valid-but-unsupported images (AVIF/HEIC/…) to PNG in place
    /// (sc-8225) — the one-time fix for datasets built before normalization, run pre-train so the
    /// engine (which reads dataset images with no decode backstop) never sees an undecodable format.
    /// Locked wrapper over [`TrainingDatasetStore::normalize_unsupported_images`].
    pub fn normalize_training_dataset_unsupported_images(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).normalize_unsupported_images(project_id, dataset_id)
    }

    /// Read the dataset's CLIP embeddings sidecar (`None` if the analysis job hasn't run). Locked
    /// wrapper over [`TrainingDatasetStore::read_dataset_embeddings`].
    pub fn read_dataset_embeddings(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<Option<DatasetEmbeddings>> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).read_dataset_embeddings(project_id, dataset_id)
    }

    /// Persist the dataset's face sidecar (sc-6538). Locked wrapper over
    /// [`TrainingDatasetStore::write_dataset_faces`].
    pub fn write_dataset_faces(
        &self,
        project_id: &str,
        dataset_id: &str,
        faces: &DatasetFaceRecords,
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).write_dataset_faces(project_id, dataset_id, faces)
    }

    /// Read the dataset's face sidecar (`None` if the worker face pass hasn't run). Locked wrapper
    /// over [`TrainingDatasetStore::read_dataset_faces`].
    pub fn read_dataset_faces(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<Option<DatasetFaceRecords>> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).read_dataset_faces(project_id, dataset_id)
    }

    pub fn batch_rename_training_dataset_items(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetBatchRenameInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .batch_rename_dataset_items(project_id, dataset_id, input)
    }

    pub fn write_training_dataset_caption_sidecars(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetCaptionSidecarsInput,
    ) -> ProjectStoreResult<TrainingCaptionSidecarsResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .write_caption_sidecars(project_id, dataset_id, input)
    }

    pub fn delete_training_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<TrainingDatasetMutationResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).delete_dataset(project_id, dataset_id)
    }

    pub fn project_stem(&self, project_id: &str) -> ProjectStoreResult<String> {
        let project_path = self.find_project_path(project_id)?;
        Ok(project_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_owned())
    }

    pub fn reindex_project(&self, project_id: &str) -> ProjectStoreResult<ReindexResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let counts = reindex_project_path(&project_path)?;
        // Rebuilding re-adds a row for every sidecar on disk, including any whose
        // media was purged. Prune those straight back out so an explicit reindex
        // leaves the registry as clean as the startup sweep does. The auto-reindex
        // in `list_assets` calls `reindex_project_path` directly (not this method),
        // so it is unaffected — it still recovers on-disk sidecars as before.
        let pruned = prune_orphaned_asset_rows(&project_path)?;
        Ok(ReindexResult {
            project_id: project_id.to_owned(),
            assets: counts.assets.saturating_sub(pruned as u32),
            generation_sets: counts.generation_sets,
            timelines: counts.timelines,
        })
    }

    /// Remove index rows for "purged-but-referenced" assets — rows whose recorded
    /// media file no longer exists on disk. These arise when an asset's media is
    /// deleted out-of-band, or a purge is interrupted after the file is unlinked
    /// but before the row is removed: the row (and its sidecar) linger, so
    /// `list_assets` keeps serving an asset whose `/files/` URL 404s on every open.
    /// The web UI degrades each to a placeholder (`assetMedia.jsx`), so the only
    /// symptom is server-log noise — one 404 per missing asset, every startup.
    ///
    /// Deletes the index row and moves the orphaned `*.sceneworks.json` sidecar out
    /// of the scanned asset tree into the project's `.orphaned-sidecars/` quarantine
    /// (see [`prune_orphaned_asset_rows`]). Quarantining rather than deleting keeps
    /// the recipe metadata recoverable, and moving it out of the folders
    /// `asset_sidecars` scans stops a rebuild (`reindex` / the empty-table
    /// auto-reindex in `list_assets`) from resurrecting the row. Media lives beside
    /// the sidecar under the project directory, so in the normal case a file missing
    /// from a readable project is a genuine purge. Idempotent; returns the number of
    /// rows pruned.
    pub fn prune_orphaned_assets(&self, project_id: &str) -> ProjectStoreResult<usize> {
        let project_path = self.find_project_path(project_id)?;
        prune_orphaned_asset_rows(&project_path)
    }

    /// Run [`prune_orphaned_assets`](Self::prune_orphaned_assets) across every
    /// registered project (including the reserved global libraries) that already
    /// has an initialized `project.db`. Best-effort per project: one project's
    /// failure is logged and does not abort the sweep. Intended as a startup
    /// data-integrity pass so the Library never fetches a purged asset. Returns the
    /// total number of rows pruned across all projects.
    pub fn prune_all_orphaned_assets(&self) -> ProjectStoreResult<usize> {
        let project_paths: Vec<PathBuf> = {
            let _guard = self.lock.lock();
            self.load_registry()?
                .into_iter()
                .filter_map(|item| item.path.map(PathBuf::from))
                // Only touch projects that have been opened at least once (a DB
                // exists); never materialize a `project.db` for one that has not.
                .filter(|path| path.join("project.db").exists())
                .collect()
        };
        let mut total = 0;
        for project_path in project_paths {
            match prune_orphaned_asset_rows(&project_path) {
                Ok(count) => total += count,
                Err(error) => tracing::warn!(
                    event = "orphaned_asset_prune_project_failed",
                    project = %project_path.display(),
                    error = %error,
                    "could not prune orphaned assets for project"
                ),
            }
        }
        Ok(total)
    }

    pub fn list_timelines(&self, project_id: &str) -> ProjectStoreResult<Vec<TimelineSummary>> {
        let project_path = self.find_project_path(project_id)?;
        ensure_project_db_ready(&project_path)?;
        let connection = connect_project_db(&project_path)?;
        let mut statement = connection.prepare(
            "
            select id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
              from timelines
             order by updated_at desc
            ",
        )?;
        let timelines = statement
            .query_map([], |row| {
                Ok(TimelineSummary {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    file_path: row.get(2)?,
                    aspect_ratio: row.get(3)?,
                    width: row.get(4)?,
                    height: row.get(5)?,
                    fps: row.get(6)?,
                    duration: row.get(7)?,
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProjectStoreError::from)?;
        Ok(timelines)
    }

    pub fn create_timeline(
        &self,
        project_id: &str,
        name: &str,
        aspect_ratio: &str,
        fps: u32,
    ) -> ProjectStoreResult<Value> {
        if name.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Timeline name is required".to_owned(),
            ));
        }
        if name.chars().count() > 120 {
            return Err(ProjectStoreError::BadRequest(
                "Timeline name must be at most 120 characters".to_owned(),
            ));
        }
        if !(1..=60).contains(&fps) {
            return Err(ProjectStoreError::BadRequest(
                "FPS must be between 1 and 60".to_owned(),
            ));
        }
        let (width, height) = timeline_dimensions(aspect_ratio)?;
        let timeline = json!({
            "schemaVersion": 1,
            "id": format!("timeline_{}", random_hex(16)?),
            "projectId": project_id,
            "name": name,
            "aspectRatio": aspect_ratio,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": 0.0,
            "tracks": default_timeline_tracks(),
            "transitions": [],
            "createdAt": Value::Null,
            "updatedAt": Value::Null,
        });
        self.save_timeline(project_id, timeline)
    }

    pub fn get_timeline(&self, project_id: &str, timeline_id: &str) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        let timeline_file = find_timeline_file(&project_path, timeline_id)?;
        read_json(&timeline_file.path)
    }

    pub fn save_timeline(
        &self,
        project_id: &str,
        mut timeline: Value,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let timeline_id = required_str(&timeline, "id")?.to_owned();
        // The client supplies `id`, and `timeline_file_path` embeds its last 8
        // chars into the write path while `relative_string`'s lexical strip lets
        // an unnormalized `..`-bearing id through — charset-check it (matching the
        // asset/character/track/dataset guards) BEFORE any path is derived so the
        // write can never escape the project dir (sc-8871 / F-069).
        if !is_safe_id(&timeline_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid timeline id".to_owned(),
            ));
        }
        let timeline_project_id = required_str(&timeline, "projectId")?;
        if timeline_project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Project ID mismatch".to_owned(),
            ));
        }
        let name = required_str(&timeline, "name")?.to_owned();
        if name.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Timeline name is required".to_owned(),
            ));
        }
        let now = utc_now();
        validate_and_default_timeline(&mut timeline)?;
        let duration = compute_timeline_duration(&timeline);
        let object = timeline.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Timeline must be an object".to_owned())
        })?;
        if !object.contains_key("createdAt") || object.get("createdAt") == Some(&Value::Null) {
            object.insert("createdAt".to_owned(), Value::String(now.clone()));
        }
        object.insert("updatedAt".to_owned(), Value::String(now));
        object.insert("duration".to_owned(), json!(duration));
        if !object
            .get("tracks")
            .and_then(Value::as_array)
            .is_some_and(|tracks| !tracks.is_empty())
        {
            object.insert("tracks".to_owned(), default_timeline_tracks());
        }
        object
            .entry("transitions".to_owned())
            .or_insert_with(|| json!([]));
        normalize_timeline_items(&mut timeline)?;

        let path = timeline_file_path(&project_path, &timeline_id, &name);
        let rel_path = relative_string(&project_path, &path)?;
        write_json(&path, &timeline)?;
        index_timeline(&project_path, &timeline, &rel_path)?;
        Ok(timeline)
    }

    pub fn save_existing_timeline(
        &self,
        project_id: &str,
        timeline_id: &str,
        timeline: Value,
    ) -> ProjectStoreResult<Value> {
        if required_str(&timeline, "id")? != timeline_id {
            return Err(ProjectStoreError::BadRequest(
                "Timeline ID mismatch".to_owned(),
            ));
        }
        self.save_timeline(project_id, timeline)
    }

    pub fn timeline_file(
        &self,
        project_id: &str,
        timeline_id: &str,
    ) -> ProjectStoreResult<TimelineFile> {
        let project_path = self.find_project_path(project_id)?;
        find_timeline_file(&project_path, timeline_id)
    }

    pub fn timeline_file_and_document(
        &self,
        project_id: &str,
        timeline_id: &str,
    ) -> ProjectStoreResult<TimelineFileDocument> {
        let project_path = self.find_project_path(project_id)?;
        let file = find_timeline_file(&project_path, timeline_id)?;
        let document = read_json(&file.path)?;
        Ok(TimelineFileDocument { file, document })
    }

    pub fn list_assets(
        &self,
        project_id: &str,
        include_rejected: bool,
        include_trashed: bool,
        scope: AssetScope,
    ) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        ensure_project_db_ready(&project_path)?;
        let total = {
            let connection = connect_project_db(&project_path)?;
            connection.query_row("select count(*) from assets", [], |row| {
                row.get::<_, i64>(0)
            })?
        };
        // Pre-migration projects (created before the `sidecar_path` column /
        // asset index existed) surface as EMPTY libraries even though their
        // assets are still on disk as `.sceneworks.json` sidecars (V-4). The
        // first open auto-reindexes from those sidecars so users never see a
        // silently-empty library. Idempotent: only fires when the table is
        // empty AND sidecars exist on disk, so a genuinely-empty project does
        // no work and there is no reindex loop. A failed reindex must not 500
        // the listing — fail open and return whatever the table currently
        // holds (possibly empty), which is strictly better than an error.
        if total == 0 && project_has_sidecars(&project_path) {
            let _ = reindex_project_path(&project_path);
        }

        let connection = connect_project_db(&project_path)?;
        // The Library view hides Character Studio test outputs (sc-2024). Rows
        // with a null origin (should not occur after the schema-bump reindex)
        // fail open and stay visible.
        let origin_filter = match scope {
            AssetScope::All => "",
            AssetScope::Library => " and (origin is null or origin != 'character_studio')",
        };
        let mut statement = connection.prepare(&format!(
            "
            select sidecar_path, file_path
              from assets
             where (?1 or rejected = 0)
               and (?2 or trashed = 0)
               {origin_filter}
             order by created_at desc
            "
        ))?;
        let rows = statement.query_map(params![include_rejected, include_trashed], |row| {
            Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, Option<String>>(1)?,
            ))
        })?;
        // HashSet dedup (was an O(n²) Vec linear scan) and a per-call
        // generation-set cache so a set's JSON is read once, not once per asset
        // in it (sc-4270 / F-CORE-10).
        let mut seen_asset_ids = std::collections::HashSet::new();
        let mut generation_sets = GenerationSetCache::default();
        let mut assets = Vec::new();
        for row in rows {
            let (sidecar_rel, file_rel) = row?;
            let mut candidates = Vec::new();
            if let Some(sidecar_rel) = sidecar_rel {
                candidates.push(project_path.join(sidecar_rel));
            }
            if let Some(file_rel) = file_rel {
                candidates.push(
                    project_path
                        .join(file_rel)
                        .with_extension("sceneworks.json"),
                );
            }
            for sidecar_path in candidates {
                if !sidecar_path.exists() {
                    continue;
                }
                let Ok(asset) = normalize_asset_cached(
                    project_id,
                    &project_path,
                    &sidecar_path,
                    &mut generation_sets,
                ) else {
                    continue;
                };
                let asset_id = asset
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                if !seen_asset_ids.insert(asset_id) {
                    break;
                }
                assets.push(asset);
                break;
            }
        }
        Ok(assets)
    }

    /// List the project's saved voices (Voice Clone registry, sc-13517).
    pub fn list_saved_voices(&self, project_id: &str) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        crate::voice_store::SavedVoiceStore::new(project_path).list_saved_voices(project_id)
    }

    /// Register a saved voice; returns the hydrated voice plus any near-duplicate hit (sc-13517).
    pub fn create_saved_voice(
        &self,
        project_id: &str,
        input: crate::voice_store::SavedVoiceCreateInput,
        dedup_threshold: f32,
    ) -> ProjectStoreResult<(Value, Option<crate::voice_store::SavedVoiceDuplicate>)> {
        let project_path = self.find_project_path(project_id)?;
        crate::voice_store::SavedVoiceStore::new(project_path).create_saved_voice(
            project_id,
            input,
            dedup_threshold,
        )
    }

    /// Delete a saved voice (sc-13517).
    pub fn delete_saved_voice(
        &self,
        project_id: &str,
        voice_id: &str,
    ) -> ProjectStoreResult<crate::voice_store::SavedVoiceMutationResult> {
        let project_path = self.find_project_path(project_id)?;
        crate::voice_store::SavedVoiceStore::new(project_path)
            .delete_saved_voice(project_id, voice_id)
    }

    /// Resolve a library asset id to its absolute media file path, guarded against path traversal.
    /// Used by the saved-voice register flow to hand the reference clip to the worker embed path
    /// (sc-13517). Mirrors `get_asset`'s resolution but returns the on-disk path rather than the JSON.
    pub fn resolve_asset_media_path(
        &self,
        project_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<PathBuf> {
        let project_path = self.find_project_path(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let asset = normalize_asset(project_id, &project_path, &sidecar_path)?;
        let media_rel = asset
            .pointer("/file/path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if media_rel.is_empty() || !is_safe_relative_path(media_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Asset media path must be project-relative".to_owned(),
            ));
        }
        Ok(project_path.join(media_rel))
    }

    pub fn list_characters(
        &self,
        project_id: &str,
        include_archived: bool,
    ) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path)
            .list_characters(project_id, include_archived)
    }

    pub fn create_character(
        &self,
        project_id: &str,
        input: CharacterCreateInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).create_character(project_id, input)
    }

    pub fn get_character(&self, project_id: &str, character_id: &str) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).get_character(project_id, character_id)
    }

    pub fn update_character(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_character(
            project_id,
            character_id,
            input,
        )
    }

    pub fn archive_character(
        &self,
        project_id: &str,
        character_id: &str,
    ) -> ProjectStoreResult<CharacterMutationResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).archive_character(character_id)
    }

    pub fn purge_character(
        &self,
        project_id: &str,
        character_id: &str,
    ) -> ProjectStoreResult<CharacterMutationResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).purge_character(character_id)
    }

    pub fn add_character_reference(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterReferenceInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).add_reference(
            project_id,
            character_id,
            input,
        )
    }

    pub fn update_character_reference(
        &self,
        project_id: &str,
        character_id: &str,
        asset_id: &str,
        input: CharacterReferenceUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_reference(
            project_id,
            character_id,
            asset_id,
            input,
        )
    }

    pub fn remove_character_reference(
        &self,
        project_id: &str,
        character_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).remove_reference(
            project_id,
            character_id,
            asset_id,
        )
    }

    pub fn create_character_look(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterLookInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).create_look(
            project_id,
            character_id,
            input,
        )
    }

    pub fn update_character_look(
        &self,
        project_id: &str,
        character_id: &str,
        look_id: &str,
        input: CharacterLookUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_look(
            project_id,
            character_id,
            look_id,
            input,
        )
    }

    pub fn delete_character_look(
        &self,
        project_id: &str,
        character_id: &str,
        look_id: &str,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).delete_look(
            project_id,
            character_id,
            look_id,
        )
    }

    pub fn attach_character_lora(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterLoraInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).attach_lora(
            project_id,
            character_id,
            input,
        )
    }

    pub fn update_character_lora(
        &self,
        project_id: &str,
        character_id: &str,
        link_id: &str,
        input: CharacterLoraUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_lora_link(
            project_id,
            character_id,
            link_id,
            input,
        )
    }

    pub fn detach_character_lora(
        &self,
        project_id: &str,
        character_id: &str,
        link_id: &str,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).detach_lora(
            project_id,
            character_id,
            link_id,
        )
    }

    pub fn list_person_tracks(&self, project_id: &str) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        let folder = project_path.join("person-tracks");
        if !folder.exists() {
            return Ok(Vec::new());
        }

        let mut tracks = Vec::new();
        for entry in read_dir_paths(&folder)? {
            if !entry
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.ends_with(".sceneworks.person-track.json"))
            {
                continue;
            }
            if let Ok(track) = normalize_person_track(&project_path, &entry) {
                tracks.push(track);
            }
        }
        tracks.sort_by(|left, right| {
            right
                .get("createdAt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .cmp(
                    left.get("createdAt")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                )
        });
        Ok(tracks)
    }

    pub fn get_person_track(&self, project_id: &str, track_id: &str) -> ProjectStoreResult<Value> {
        if !is_safe_track_id(track_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let tracks_dir = project_path.join("person-tracks");
        let track_path = project_path
            .join("person-tracks")
            .join(format!("{track_id}.sceneworks.person-track.json"));
        if !track_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Person track not found".to_owned(),
            ));
        }
        let tracks_root = fs::canonicalize(&tracks_dir)?;
        let resolved_path = fs::canonicalize(&track_path)?;
        if !resolved_path.starts_with(&tracks_root) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }
        normalize_person_track(&project_path, &track_path)
    }

    /// Replace the correction set on a person track (sc-1485). Each incoming
    /// correction targets a frame by index and either adjusts its box or rejects
    /// the frame; the server stamps author/createdAt/source so the sidecar always
    /// records who corrected it and when. The corrections array is the UI's full
    /// view, so it is written wholesale (a frame omitted from the set is treated
    /// as having no correction). Returns the normalized, updated track.
    pub fn set_person_track_corrections(
        &self,
        project_id: &str,
        track_id: &str,
        corrections: Vec<Value>,
    ) -> ProjectStoreResult<Value> {
        if !is_safe_track_id(track_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let tracks_dir = project_path.join("person-tracks");
        let track_path = tracks_dir.join(format!("{track_id}.sceneworks.person-track.json"));
        if !track_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Person track not found".to_owned(),
            ));
        }
        let tracks_root = fs::canonicalize(&tracks_dir)?;
        let resolved_path = fs::canonicalize(&track_path)?;
        if !resolved_path.starts_with(&tracks_root) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }

        let mut track = read_json(&track_path)?;
        let frame_count = track
            .get("frames")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        let mut normalized = normalize_person_track_corrections(corrections, frame_count)?;
        let has_corrections = !normalized.is_empty();
        normalized.sort_by_key(|correction| {
            correction
                .get("frameIndex")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        });

        let object = track.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Person track sidecar must be an object".to_owned())
        })?;
        object.insert("corrections".to_owned(), Value::Array(normalized));
        let status = object
            .entry("status")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Person track status must be an object".to_owned())
            })?;
        status.insert(
            "correctionState".to_owned(),
            Value::String(
                if has_corrections {
                    "box_corrections_applied"
                } else {
                    "ready_for_box_corrections"
                }
                .to_owned(),
            ),
        );

        write_json(&track_path, &track)?;
        normalize_person_track(&project_path, &track_path)
    }

    pub fn import_asset(&self, project_id: &str, upload: UploadAsset) -> ProjectStoreResult<Value> {
        if fs::metadata(&upload.source_path)?.len() == 0 {
            return Err(ProjectStoreError::BadRequest(
                "Uploaded file is empty".to_owned(),
            ));
        }
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let upload_dir = project_path.join("assets").join("uploads");
        fs::create_dir_all(&upload_dir)?;

        let guessed_mime = guess_mime_from_filename(&upload.filename);
        let content_type = upload
            .content_type
            .as_deref()
            .filter(|value| !value.is_empty() && *value != "application/octet-stream")
            .map(str::to_owned)
            .or(guessed_mime)
            .unwrap_or_else(|| "application/octet-stream".to_owned());
        if !content_type.starts_with("image/") && !content_type.starts_with("video/") {
            return Err(ProjectStoreError::BadRequest(
                "Only image and video uploads are supported".to_owned(),
            ));
        }

        // sc-6143: transcode a valid-but-unsupported image (AVIF/HEIC/HEIF/TIFF/BMP/GIF) to lossless
        // PNG before storing, so every downstream decode site, thumbnail, and preview reads a format
        // it supports. Supported formats and videos pass through unchanged.
        let normalized = normalize_image_upload(
            &upload.source_path,
            &content_type,
            &upload.filename,
            &upload_dir,
        )?;
        let content_type = normalized.content_type.clone();

        let asset_id = format!("asset_{}", random_hex(16)?);
        let created_at = utc_now();
        let extension = normalized.extension.clone();
        let suffix = &asset_id[asset_id.len().saturating_sub(8)..];
        let filename = format!(
            "{}-{suffix}{extension}",
            safe_filename(&upload.filename, &asset_id)
        );
        let media_path = upload_dir.join(filename);
        move_normalized_upload(&normalized, &upload.source_path, &media_path)?;
        let media_rel = relative_string(&project_path, &media_path)?;
        let display_name = Path::new(&upload.filename)
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| {
                media_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("upload")
            })
            .to_owned();

        let mut asset = json!({
            "schemaVersion": 1,
            "id": asset_id,
            "projectId": project_id,
            "generationSetId": Value::Null,
            "type": media_type_for_mime(&content_type)?,
            "displayName": display_name,
            "createdAt": created_at,
            // Manual imports are Library media in their own right (sc-2024).
            "origin": "upload",
            "file": {
                "path": media_rel,
                "mimeType": content_type,
                "width": Value::Null,
                "height": Value::Null,
                "duration": Value::Null,
                "fps": Value::Null
            },
            "status": {
                "favorite": false,
                "rating": 0,
                "rejected": false,
                "trashed": false
            },
            "recipe": {
                "mode": "upload",
                "model": "manual-import",
                "adapter": "api-upload",
                "prompt": upload.filename,
                "negativePrompt": "",
                "seed": 0,
                "loras": [],
                "stylePreset": "none",
                "normalizedSettings": {},
                "rawAdapterSettings": { "contentType": content_type }
            },
            "lineage": {
                "parents": upload
                    .source_asset_id
                    .clone()
                    .map(|id| vec![Value::String(id)])
                    .unwrap_or_default(),
                "sourceAssetId": upload
                    .source_asset_id
                    .clone()
                    .map(Value::String)
                    .unwrap_or(Value::Null),
                "sourceTimestamp": Value::Null,
                "jobId": Value::Null
            }
        });
        // Provenance (e.g. the Image Editor edit chain) rides the same top-level
        // `extra` slot generated assets use, so the Library/lineage UI can read it.
        if let Some(provenance) = &upload.provenance {
            if let Some(object) = asset.as_object_mut() {
                object.insert("extra".to_owned(), provenance.clone());
            }
        }
        let sidecar_path = media_path.with_extension("sceneworks.json");
        write_json(&sidecar_path, &asset)?;
        index_asset(&project_path, &asset, Some(&sidecar_path))?;
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    /// Persist a generated asset the worker reported as flat facts: build the
    /// sidecar (Rust owns this schema now — story 1656), write it next to the
    /// media the worker already saved, write the recipe, and index project.db.
    /// Idempotent — re-applied progress updates upsert the same row/files.
    /// Returns the built sidecar so the API can re-inject it into the job result.
    pub fn persist_generated_asset(
        &self,
        project_id: &str,
        job_id: &str,
        generation_set_id: &str,
        fact: &Value,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let media_rel = fact
            .get("mediaPath")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("generated asset fact missing mediaPath".to_owned())
            })?;
        // mediaPath comes from the worker result fact and is joined into the project
        // path below for sidecar/dir writes; reject traversal/absolute paths before any
        // create_dir_all/write so the worker->API boundary can't write outside the
        // project root (sc-5721 / CORE-002->F). Mirrors the is_safe_relative_path guard
        // used at every other path-from-outside site in this store.
        if !is_safe_relative_path(media_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid media path".to_owned(),
            ));
        }
        let asset_id = fact.get("assetId").and_then(Value::as_str).ok_or_else(|| {
            ProjectStoreError::BadRequest("generated asset fact missing assetId".to_owned())
        })?;
        // The asset id comes from the worker's result fact and is joined into the
        // recipe path below, so charset-check it before path use (sc-4211 / F-CORE-7).
        if !is_safe_id(asset_id) {
            return Err(ProjectStoreError::BadRequest("Invalid asset id".to_owned()));
        }
        let asset = build_generated_asset_sidecar(project_id, job_id, generation_set_id, fact);
        let media_path = project_path.join(media_rel);
        let sidecar_path = media_path.with_extension("sceneworks.json");
        if let Some(parent) = sidecar_path.parent() {
            fs::create_dir_all(parent)?;
        }
        write_json(&sidecar_path, &asset)?;
        let recipes_dir = project_path.join("recipes");
        fs::create_dir_all(&recipes_dir)?;
        write_json(
            &recipes_dir.join(format!("{asset_id}.recipe.json")),
            &asset["recipe"],
        )?;
        index_asset(&project_path, &asset, Some(&sidecar_path))?;
        Ok(asset)
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
        index_asset(&project_path, &asset, Some(&sidecar_path))?;
        normalize_asset(GLOBAL_POSES_PROJECT_ID, &project_path, &sidecar_path)
    }

    // ---- Key Point Library (epic 4422, sc-4434) -----------------------------------------

    /// Guard a staged source-image path under the keypoint-uploads cache (mirrors
    /// [`Self::pose_preview_path`]) so a save can't copy an arbitrary file into the library.
    fn keypoint_upload_path(&self, path: &str) -> ProjectStoreResult<PathBuf> {
        let cache_root = self.data_dir.join("cache").join(KEYPOINT_UPLOADS_CACHE_DIR);
        let canonical_root = fs::canonicalize(&cache_root).map_err(|_| {
            ProjectStoreError::NotFound("Key Point upload cache is unavailable".to_owned())
        })?;
        let canonical = fs::canonicalize(path).map_err(|_| {
            ProjectStoreError::NotFound("Key Point source image not found".to_owned())
        })?;
        if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
            return Err(ProjectStoreError::BadRequest(
                "Key Point source path is invalid".to_owned(),
            ));
        }
        Ok(canonical)
    }

    /// Persist a user keypoint preset: the 5-point kps + the retained source image (copied from
    /// a staged upload into the library). Built-in presets are NOT stored here — they live in
    /// [`crate::angle_kps`] (virtual, protected). Spec:
    /// `{ name, kps[[x,y]×5], sourceUploadPath, sourceAssetId? }`.
    pub fn create_keypoint_asset(&self, spec: &Value) -> ProjectStoreResult<Value> {
        let name = spec
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ProjectStoreError::BadRequest("keypoint spec missing name".to_owned()))?
            .to_owned();
        let kps = parse_normalized_kps(spec.get("kps"))?;
        let upload_path = spec
            .get("sourceUploadPath")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("keypoint spec missing sourceUploadPath".to_owned())
            })?;
        let canonical_source = self.keypoint_upload_path(upload_path)?;

        self.ensure_global_keypoints_project()?;
        let (project_path, _project_guard) = self.lock_project(GLOBAL_KEYPOINTS_PROJECT_ID)?;

        let asset_id = format!("asset_{}", random_hex(16)?);
        let created_at = utc_now();
        let dir = project_path.join("assets").join("keypoints");
        fs::create_dir_all(&dir)?;
        // Determine the format from the file CONTENT, not the staged path's extension: uploads
        // arrive as `upload-<uuid>.tmp` (the generic temp writer keeps no extension), so keying
        // off the extension would mislabel every capture as PNG. A valid-but-unsupported format
        // (AVIF/HEIC/HEIF/TIFF/BMP/GIF) is transcoded to PNG so the retained source image is always
        // decodable and never mislabeled (sc-6143) — without this an AVIF capture would be copied
        // verbatim under a `.png` name. A sniffable supported format keeps its bytes; when the magic
        // bytes aren't recognized at all (e.g. SVG) we fall back to the extension, then PNG.
        let source_kind = crate::media_convert::sniff_image_kind_at(&canonical_source);
        let needs_transcode = source_kind.is_some_and(|kind| !kind.is_natively_supported());
        let (ext, mime) = if needs_transcode {
            ("png", "image/png")
        } else {
            sniff_image_format(&canonical_source)
                .or_else(|| {
                    canonical_source
                        .extension()
                        .and_then(|value| value.to_str())
                        .and_then(|value| match value.to_ascii_lowercase().as_str() {
                            "jpg" | "jpeg" => Some(("jpg", "image/jpeg")),
                            "webp" => Some(("webp", "image/webp")),
                            "png" => Some(("png", "image/png")),
                            _ => None,
                        })
                })
                .unwrap_or(("png", "image/png"))
        };
        let media_path = dir.join(format!("{asset_id}.{ext}"));
        if needs_transcode {
            let kind = source_kind.expect("needs_transcode implies a sniffed kind");
            crate::media_convert::transcode_to_png(&canonical_source, &media_path).map_err(
                |error| {
                    ProjectStoreError::BadRequest(format!(
                        "Could not convert {} image to a supported format: {error}",
                        kind.label()
                    ))
                },
            )?;
        } else {
            fs::copy(&canonical_source, &media_path)?;
        }
        let media_rel = relative_string(&project_path, &media_path)?;

        let source_asset_id = spec
            .get("sourceAssetId")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let parents = source_asset_id
            .clone()
            .map(|id| vec![Value::String(id)])
            .unwrap_or_default();

        let asset = json!({
            "schemaVersion": 1,
            "id": asset_id,
            "projectId": GLOBAL_KEYPOINTS_PROJECT_ID,
            "generationSetId": Value::Null,
            "type": "keypoint",
            "displayName": name,
            "createdAt": created_at,
            "origin": "keypoint_library",
            "tags": Vec::<Value>::new(),
            "file": {
                "path": media_rel,
                "mimeType": mime,
                "width": spec.get("sourceWidth").cloned().unwrap_or(Value::Null),
                "height": spec.get("sourceHeight").cloned().unwrap_or(Value::Null),
                "duration": Value::Null,
                "fps": Value::Null
            },
            "status": { "favorite": false, "rating": 0, "rejected": false, "trashed": false },
            "recipe": {
                "mode": "upload",
                "model": "scrfd",
                "adapter": "api-upload",
                "prompt": name,
                "negativePrompt": "",
                "seed": 0,
                "loras": [],
                "stylePreset": "none",
                "normalizedSettings": {},
                "rawAdapterSettings": { "detector": "scrfd" }
            },
            "lineage": {
                "parents": parents,
                "sourceAssetId": source_asset_id,
                "sourceTimestamp": Value::Null,
                "jobId": Value::Null
            },
            "keypoint": {
                "kps": kps,
                "builtin": false
            }
        });
        let sidecar_path = media_path.with_extension("sceneworks.json");
        write_json(&sidecar_path, &asset)?;
        index_asset(&project_path, &asset, Some(&sidecar_path))?;
        normalize_asset(GLOBAL_KEYPOINTS_PROJECT_ID, &project_path, &sidecar_path)
    }

    /// All keypoint presets: the built-in 11 (from [`crate::angle_kps`]) followed by the user's
    /// stored `type:"keypoint"` assets, each as a `{ id, name, kps, builtin, sourceImageRef }`
    /// record. The single read the Key Point Library renders from.
    pub fn list_keypoint_presets(&self) -> ProjectStoreResult<Vec<Value>> {
        let mut presets = crate::angle_kps::builtin_preset_records();
        presets.extend(self.list_user_keypoint_presets()?);
        Ok(presets)
    }

    fn list_user_keypoint_presets(&self) -> ProjectStoreResult<Vec<Value>> {
        self.ensure_global_keypoints_project()?;
        let assets =
            self.list_assets(GLOBAL_KEYPOINTS_PROJECT_ID, false, false, AssetScope::All)?;
        Ok(assets
            .iter()
            .filter(|asset| asset.get("type").and_then(Value::as_str) == Some("keypoint"))
            .map(keypoint_asset_to_preset)
            .collect())
    }

    /// The set of preset ids a collection may reference: the built-in ids + the user's stored
    /// keypoint asset ids.
    fn known_preset_ids(&self) -> ProjectStoreResult<std::collections::HashSet<String>> {
        let mut ids: std::collections::HashSet<String> = crate::angle_kps::BUILTIN_ANGLE_SET_ORDER
            .iter()
            .map(|angle| crate::angle_kps::builtin_preset_id(angle))
            .collect();
        for preset in self.list_user_keypoint_presets()? {
            if let Some(id) = preset.get("id").and_then(Value::as_str) {
                ids.insert(id.to_owned());
            }
        }
        Ok(ids)
    }

    /// All angle-set collections: the virtual built-in default (the 11 in order) followed by the
    /// user's collections. The built-in's `isDefault` is true unless a user collection claims it.
    pub fn list_keypoint_collections(&self) -> ProjectStoreResult<Vec<Value>> {
        self.ensure_global_keypoints_project()?;
        let project_path = self.find_project_path(GLOBAL_KEYPOINTS_PROJECT_ID)?;
        let user = read_user_collections(&project_path)?;
        let any_user_default = user
            .iter()
            .any(|collection| collection.get("isDefault").and_then(Value::as_bool) == Some(true));
        let mut builtin = crate::angle_kps::builtin_default_collection();
        builtin["isDefault"] = Value::Bool(!any_user_default);
        let mut all = vec![builtin];
        all.extend(user);
        Ok(all)
    }

    /// Create or update a user angle-set collection: `{ id?, name, orderedPresetIds[], isDefault? }`.
    /// Validates every referenced preset exists; marking it default clears the flag on the others
    /// (and on the built-in, which then reports `isDefault:false`).
    pub fn upsert_keypoint_collection(&self, spec: &Value) -> ProjectStoreResult<Value> {
        let name = spec
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ProjectStoreError::BadRequest("collection missing name".to_owned()))?
            .to_owned();
        let ordered = spec
            .get("orderedPresetIds")
            .and_then(Value::as_array)
            .filter(|values| !values.is_empty())
            .ok_or_else(|| {
                ProjectStoreError::BadRequest(
                    "collection needs a non-empty orderedPresetIds".to_owned(),
                )
            })?;
        let ordered_ids: Vec<String> = ordered
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        if ordered_ids.len() != ordered.len() {
            return Err(ProjectStoreError::BadRequest(
                "orderedPresetIds must be strings".to_owned(),
            ));
        }
        let known = self.known_preset_ids()?;
        for id in &ordered_ids {
            if !known.contains(id) {
                return Err(ProjectStoreError::BadRequest(format!(
                    "collection references unknown preset id {id}"
                )));
            }
        }
        let requested_id = spec
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if requested_id == Some(crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID) {
            return Err(ProjectStoreError::BadRequest(
                "the built-in default collection is read-only".to_owned(),
            ));
        }
        let id = match requested_id {
            Some(value) if is_safe_id(value) => value.to_owned(),
            Some(_) => {
                return Err(ProjectStoreError::BadRequest(
                    "invalid collection id".to_owned(),
                ))
            }
            None => format!("kpc_{}", random_hex(8)?),
        };
        let make_default = spec.get("isDefault").and_then(Value::as_bool) == Some(true);

        self.ensure_global_keypoints_project()?;
        let (project_path, _project_guard) = self.lock_project(GLOBAL_KEYPOINTS_PROJECT_ID)?;
        let mut collections = read_user_collections(&project_path)?;
        let record = json!({
            "id": id,
            "name": name,
            "orderedPresetIds": ordered_ids,
            "isDefault": make_default,
            "builtin": false,
        });
        match collections
            .iter_mut()
            .find(|collection| collection.get("id").and_then(Value::as_str) == Some(id.as_str()))
        {
            Some(existing) => *existing = record.clone(),
            None => collections.push(record.clone()),
        }
        if make_default {
            for collection in collections.iter_mut() {
                if collection.get("id").and_then(Value::as_str) != Some(id.as_str()) {
                    collection["isDefault"] = Value::Bool(false);
                }
            }
        }
        write_user_collections(&project_path, &collections)?;
        Ok(record)
    }

    /// Mark one collection the default (clearing the others). Passing the built-in id clears all
    /// user defaults so the built-in default becomes effective again.
    pub fn set_default_keypoint_collection(&self, id: &str) -> ProjectStoreResult<Vec<Value>> {
        self.ensure_global_keypoints_project()?;
        {
            let (project_path, _project_guard) = self.lock_project(GLOBAL_KEYPOINTS_PROJECT_ID)?;
            let mut collections = read_user_collections(&project_path)?;
            if id == crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID {
                for collection in collections.iter_mut() {
                    collection["isDefault"] = Value::Bool(false);
                }
            } else {
                if !collections
                    .iter()
                    .any(|collection| collection.get("id").and_then(Value::as_str) == Some(id))
                {
                    return Err(ProjectStoreError::NotFound(format!(
                        "collection {id} not found"
                    )));
                }
                for collection in collections.iter_mut() {
                    let is_target = collection.get("id").and_then(Value::as_str) == Some(id);
                    collection["isDefault"] = Value::Bool(is_target);
                }
            }
            write_user_collections(&project_path, &collections)?;
        }
        self.list_keypoint_collections()
    }

    /// Delete a user collection. The built-in default collection cannot be deleted.
    pub fn delete_keypoint_collection(&self, id: &str) -> ProjectStoreResult<()> {
        if id == crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID {
            return Err(ProjectStoreError::BadRequest(
                "the built-in default collection cannot be deleted".to_owned(),
            ));
        }
        self.ensure_global_keypoints_project()?;
        let (project_path, _project_guard) = self.lock_project(GLOBAL_KEYPOINTS_PROJECT_ID)?;
        let mut collections = read_user_collections(&project_path)?;
        let before = collections.len();
        collections.retain(|collection| collection.get("id").and_then(Value::as_str) != Some(id));
        if collections.len() == before {
            return Err(ProjectStoreError::NotFound(format!(
                "collection {id} not found"
            )));
        }
        write_user_collections(&project_path, &collections)
    }

    /// Resolve the active angle-set collection to its ordered presets (kps + name + optional
    /// canonical angle) for generation (sc-4450). Selection: an explicit `collection_id` override
    /// → the user's default → the built-in default (the 11). Referenced presets that no longer
    /// resolve (e.g. a custom preset deleted after being added) are skipped; if nothing resolves,
    /// falls back to the built-in 11. Returns `(collection_id_used, presets)`.
    pub fn resolve_angle_collection(
        &self,
        collection_id: Option<&str>,
    ) -> ProjectStoreResult<(String, Vec<ResolvedAnglePreset>)> {
        let collections = self.list_keypoint_collections()?;
        let target = match collection_id {
            Some(id) => collections
                .iter()
                .find(|collection| collection.get("id").and_then(Value::as_str) == Some(id)),
            None => collections.iter().find(|collection| {
                collection.get("isDefault").and_then(Value::as_bool) == Some(true)
            }),
        };
        let used_id = target
            .and_then(|collection| collection.get("id").and_then(Value::as_str))
            .unwrap_or(crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID)
            .to_owned();
        let ordered: Vec<String> = target
            .and_then(|collection| collection.get("orderedPresetIds"))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();

        let user_presets = self.list_user_keypoint_presets()?;
        let mut resolved = Vec::with_capacity(ordered.len());
        for id in &ordered {
            if let Some(angle) = id.strip_prefix("builtin_") {
                if let Some(kps) = crate::angle_kps::angle_kps(angle) {
                    resolved.push(ResolvedAnglePreset {
                        preset_id: id.clone(),
                        name: crate::angle_kps::builtin_angle_display_name(angle),
                        angle: Some(angle.to_owned()),
                        kps,
                    });
                }
            } else if let Some(record) = user_presets
                .iter()
                .find(|preset| preset.get("id").and_then(Value::as_str) == Some(id.as_str()))
            {
                if let Some(kps) = parse_kps_tuple(record.get("kps")) {
                    resolved.push(ResolvedAnglePreset {
                        preset_id: id.clone(),
                        name: record
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or(id)
                            .to_owned(),
                        angle: None,
                        kps,
                    });
                }
            }
        }
        if resolved.is_empty() {
            return Ok((
                crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID.to_owned(),
                builtin_resolved_presets(),
            ));
        }
        Ok((used_id, resolved))
    }

    /// Write the generation-set JSON for a job from the worker-reported facts.
    /// Idempotent — overwrites the same `<id>.json` on re-applied updates.
    pub fn write_generation_set(
        &self,
        project_id: &str,
        job_id: &str,
        generation_set: &Value,
        recipe_fact: Option<&Value>,
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let id = generation_set
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("generation set fact missing id".to_owned())
            })?;
        let get = |key: &str| generation_set.get(key).cloned().unwrap_or(Value::Null);
        let record = json!({
            "schemaVersion": 1,
            "id": id,
            "projectId": project_id,
            "jobId": job_id,
            "mode": get("mode"),
            "model": get("model"),
            "prompt": get("prompt"),
            "negativePrompt": get("negativePrompt"),
            "count": get("count"),
            "createdAt": get("createdAt"),
        });
        let embedded_recipe = generation_set.get("recipe").cloned().or_else(|| {
            recipe_fact.and_then(|fact| {
                build_generated_asset_sidecar(project_id, job_id, id, fact)
                    .get("recipe")
                    .cloned()
            })
        });
        let mut record = record;
        if let Some(recipe) = embedded_recipe {
            if let Some(object) = record.as_object_mut() {
                object.insert("recipe".to_owned(), recipe);
            }
        }
        let dir = project_path.join("generation-sets");
        fs::create_dir_all(&dir)?;
        write_json(&dir.join(format!("{id}.json")), &record)?;
        Ok(())
    }

    pub fn update_asset_status(
        &self,
        project_id: &str,
        asset_id: &str,
        patch: AssetStatusPatch,
    ) -> ProjectStoreResult<Value> {
        if patch.rating.is_some_and(|rating| rating > 5) {
            return Err(ProjectStoreError::BadRequest(
                "Rating must be between 0 and 5".to_owned(),
            ));
        }
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let mut asset = read_json(&sidecar_path)?;
        let status = asset
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
            })?
            .entry("status")
            .or_insert_with(|| json!({}));
        let status = status.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Asset status must be an object".to_owned())
        })?;
        if let Some(value) = patch.favorite {
            status.insert("favorite".to_owned(), Value::Bool(value));
        }
        if let Some(value) = patch.rating {
            status.insert("rating".to_owned(), json!(value));
        }
        if let Some(value) = patch.rejected {
            status.insert("rejected".to_owned(), Value::Bool(value));
        }
        if let Some(value) = patch.trashed {
            status.insert("trashed".to_owned(), Value::Bool(value));
        }
        write_json(&sidecar_path, &asset)?;
        index_asset(&project_path, &asset, Some(&sidecar_path))?;
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    pub fn update_asset_tags(
        &self,
        project_id: &str,
        asset_id: &str,
        patch: AssetTagsPatch,
    ) -> ProjectStoreResult<Value> {
        let tags = normalize_asset_tags(&patch.tags)?;
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let mut asset = read_json(&sidecar_path)?;
        let object = asset.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
        })?;
        object.insert(
            "tags".to_owned(),
            Value::Array(tags.into_iter().map(Value::String).collect()),
        );
        write_json(&sidecar_path, &asset)?;
        index_asset(&project_path, &asset, Some(&sidecar_path))?;
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    /// Promote a character asset into the Main Asset Library (sc-8341). This is a
    /// true *move*: the asset leaves every character and surfaces in the Library.
    /// Because the Library excludes `origin == "character_studio"` and the Character
    /// Assets grid keys off `recipe.normalizedSettings.characterId` +
    /// `metadata.characterReferences`, all three must change: flip `origin` to a
    /// library-visible studio value, drop the recipe's `characterId`, and strip the
    /// asset's `characterReferences`, then unlink it from every character sidecar.
    ///
    /// The move carries the asset's whole upscale-fold group (sc-10205), so a
    /// folded original/upscaled pair never gets split across collections. Returns
    /// the array of updated normalized assets, the requested one first.
    pub fn move_asset_to_library(
        &self,
        project_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let character_store = CharacterStore::new(&self.data_dir, project_path.clone());
        let mut moved = Vec::new();
        for member_id in upscale_lineage_group(&project_path, asset_id) {
            let sidecar_path = self.find_asset_sidecar(&project_path, &member_id)?;
            let mut asset = read_json(&sidecar_path)?;
            {
                let object = asset.as_object_mut().ok_or_else(|| {
                    ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
                })?;
                // Promote to a library-visible origin by media type so the
                // `character_studio` exclusion (LibraryScreen) no longer applies.
                let asset_type = object
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let origin = match asset_type {
                    "video" => "video_studio",
                    "document" => "document_studio",
                    _ => "image_studio",
                };
                object.insert("origin".to_owned(), Value::String(origin.to_owned()));
                // Detach the character association the grid filters on.
                if let Some(settings) = object
                    .get_mut("recipe")
                    .and_then(Value::as_object_mut)
                    .and_then(|recipe| recipe.get_mut("normalizedSettings"))
                    .and_then(Value::as_object_mut)
                {
                    settings.remove("characterId");
                }
                if let Some(metadata) = object.get_mut("metadata").and_then(Value::as_object_mut) {
                    metadata.remove("characterReferences");
                }
            }
            write_json(&sidecar_path, &asset)?;
            index_asset(&project_path, &asset, Some(&sidecar_path))?;
            // Drop the asset from every character's references[] (character sidecars + index).
            character_store.remove_asset_references(&member_id)?;
            moved.push(normalize_asset(project_id, &project_path, &sidecar_path)?);
        }
        Ok(Value::Array(moved))
    }

    /// Move a library asset into a character's assets (sc-10200): the true-move
    /// twin of [`Self::move_asset_to_library`]. A character-sidecar `references[]`
    /// entry is NOT the right vehicle for a move — it leaves the asset in the Main
    /// Asset Library (its `origin` stays library-visible) and surfaces it in the
    /// curated "Approved set" panel (which renders every `references[]` entry). So
    /// a move must: flip `origin` to `character_studio` (the Library's allow-list
    /// then excludes it), anchor the target association in
    /// `metadata.characterReferences` only (the Character assets grid and the
    /// `?character=` scope both match on it), and detach the asset from everything
    /// else — the recipe's `characterId` and every character's curated
    /// `references[]` — without adding a curated reference to the target.
    ///
    /// The move carries the asset's whole upscale-fold group (sc-10205): the
    /// Library renders a linked original/upscaled pair as ONE folded tile, so a
    /// move of the visible tile must take the hidden fold-mate along or it stays
    /// stranded in the Library. Returns the array of updated normalized assets,
    /// the requested one first.
    pub fn move_asset_to_character(
        &self,
        project_id: &str,
        asset_id: &str,
        character_id: &str,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let character_store = CharacterStore::new(&self.data_dir, project_path.clone());
        // Reject an unknown/deleted target before mutating anything.
        character_store.get_character(project_id, character_id)?;
        let mut moved = Vec::new();
        for member_id in upscale_lineage_group(&project_path, asset_id) {
            let sidecar_path = self.find_asset_sidecar(&project_path, &member_id)?;
            let mut asset = read_json(&sidecar_path)?;
            {
                let object = asset.as_object_mut().ok_or_else(|| {
                    ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
                })?;
                object.insert(
                    "origin".to_owned(),
                    Value::String("character_studio".to_owned()),
                );
                // The recipe's characterId is a competing association vector (it would
                // keep the asset in the previous character's grid), so clear it and let
                // the metadata anchor own the membership.
                if let Some(settings) = object
                    .get_mut("recipe")
                    .and_then(Value::as_object_mut)
                    .and_then(|recipe| recipe.get_mut("normalizedSettings"))
                    .and_then(Value::as_object_mut)
                {
                    settings.remove("characterId");
                }
                let metadata = object
                    .entry("metadata".to_owned())
                    .or_insert_with(|| json!({}));
                let metadata = metadata.as_object_mut().ok_or_else(|| {
                    ProjectStoreError::BadRequest("Asset metadata must be an object".to_owned())
                })?;
                // `source` distinguishes this move anchor from the "character-sidecar"
                // mirror entries `update_asset_character_link` manages — that rebuild
                // must not drop the anchor when a curated reference is added/removed.
                metadata.insert(
                    "characterReferences".to_owned(),
                    json!([{
                        "characterId": character_id,
                        "source": "library-move",
                        "approved": false,
                        "role": "asset",
                        "linkedAt": utc_now(),
                    }]),
                );
            }
            write_json(&sidecar_path, &asset)?;
            index_asset(&project_path, &asset, Some(&sidecar_path))?;
            // Leave no curated membership behind: the move detaches the asset from every
            // character's references[] (including the target's — the Approved set is
            // hand-curated and a bulk move must not populate it).
            character_store.remove_asset_references(&member_id)?;
            moved.push(normalize_asset(project_id, &project_path, &sidecar_path)?);
        }
        Ok(Value::Array(moved))
    }

    pub fn get_asset(&self, project_id: &str, asset_id: &str) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    pub fn index_asset_sidecar(
        &self,
        project_id: &str,
        sidecar_path: &Path,
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let asset = read_json(sidecar_path)?;
        index_asset(&project_path, &asset, Some(sidecar_path))
    }

    pub fn delete_asset(
        &self,
        project_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<AssetMutationResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let mut asset = read_json(&sidecar_path)?;
        let media_rel = asset
            .pointer("/file/path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if !media_rel.is_empty() && !is_safe_relative_path(&media_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Asset media path must be project-relative".to_owned(),
            ));
        }
        let media_path = project_path.join(&media_rel);
        let status = asset
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
            })?
            .entry("status")
            .or_insert_with(|| json!({}));
        let status = status.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Asset status must be an object".to_owned())
        })?;
        status.insert("trashed".to_owned(), Value::Bool(true));

        let trash_dir = project_path.join("trash").join(asset_id);
        fs::create_dir_all(&trash_dir)?;
        if media_path.exists() && media_path.is_file() {
            let trashed_media_path = trash_dir.join(media_path.file_name().ok_or_else(|| {
                ProjectStoreError::BadRequest("Invalid asset media path".to_owned())
            })?);
            fs::rename(&media_path, &trashed_media_path)?;
            if let Some(file) = asset.get_mut("file").and_then(Value::as_object_mut) {
                file.insert(
                    "path".to_owned(),
                    Value::String(relative_string(&project_path, &trashed_media_path)?),
                );
            }
        }
        let trashed_sidecar_path = trash_dir.join(sidecar_path.file_name().ok_or_else(|| {
            ProjectStoreError::BadRequest("Invalid asset sidecar path".to_owned())
        })?);
        write_json(&trashed_sidecar_path, &asset)?;
        if sidecar_path != trashed_sidecar_path {
            fs::remove_file(&sidecar_path).ok();
        }
        index_asset(&project_path, &asset, Some(&trashed_sidecar_path))?;
        Ok(AssetMutationResult {
            id: asset_id.to_owned(),
            status: "trashed".to_owned(),
        })
    }

    /// Permanently remove a (usually already-trashed) asset. With `permanent = false`
    /// the on-disk media/sidecar are moved to the OS trash (recoverable); if that move
    /// fails nothing is deleted and the result carries `status = "trash_unavailable"`
    /// so the caller can prompt the user to confirm a permanent delete (`permanent = true`).
    pub fn purge_asset(
        &self,
        project_id: &str,
        asset_id: &str,
        permanent: bool,
    ) -> ProjectStoreResult<AssetMutationResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let asset = read_json(&sidecar_path)?;
        let media_path = project_path.join(
            asset
                .pointer("/file/path")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        let trash_dir = project_path.join("trash");
        let asset_trash_dir = sidecar_path
            .parent()
            .filter(|parent| {
                parent.file_name().and_then(|name| name.to_str()) == Some(asset_id)
                    && parent.parent() == Some(trash_dir.as_path())
            })
            .map(Path::to_path_buf);
        if !permanent {
            // Move the asset's footprint to the OS trash rather than unlinking it. When
            // the asset already lives in `trash/<id>/`, trash that directory in one shot;
            // otherwise trash the media file and sidecar directly.
            let targets: Vec<PathBuf> = if let Some(dir) = asset_trash_dir.clone() {
                vec![dir]
            } else {
                let mut targets = Vec::new();
                if media_path.exists() && media_path.is_file() {
                    targets.push(media_path.clone());
                }
                if sidecar_path.exists() {
                    targets.push(sidecar_path.clone());
                }
                targets
            };
            if !targets.is_empty() && trash::delete_all(&targets).is_err() {
                // Nothing was removed; let the caller confirm a permanent delete.
                return Ok(AssetMutationResult {
                    id: asset_id.to_owned(),
                    status: "trash_unavailable".to_owned(),
                });
            }
            CharacterStore::new(&self.data_dir, project_path.clone())
                .remove_asset_references(asset_id)?;
            purge_asset_record(&project_path, asset_id)?;
            return Ok(AssetMutationResult {
                id: asset_id.to_owned(),
                status: "purged".to_owned(),
            });
        }
        CharacterStore::new(&self.data_dir, project_path.clone())
            .remove_asset_references(asset_id)?;
        if media_path.exists() && media_path.is_file() {
            fs::remove_file(media_path)?;
        }
        fs::remove_file(&sidecar_path).ok();
        if let Some(parent) = asset_trash_dir {
            fs::remove_dir_all(parent).ok();
        }
        purge_asset_record(&project_path, asset_id)?;
        Ok(AssetMutationResult {
            id: asset_id.to_owned(),
            status: "purged".to_owned(),
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
        let project_path = self.find_project_path(project_id)?;
        let root = fs::canonicalize(&project_path)?;
        let target = project_path.join(relative_path);
        let target = fs::canonicalize(&target).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ProjectStoreError::NotFound("File not found".to_owned())
            } else {
                ProjectStoreError::Io(error)
            }
        })?;
        if !target.starts_with(&root) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid project file path".to_owned(),
            ));
        }
        if !target.is_file() {
            return Err(ProjectStoreError::NotFound("File not found".to_owned()));
        }
        let content_type = guess_mime_from_filename(&target.display().to_string())
            .unwrap_or_else(|| "application/octet-stream".to_owned());
        Ok(ProjectFile {
            path: target,
            content_type,
        })
    }

    fn ensure_data_dirs(&self) -> ProjectStoreResult<()> {
        fs::create_dir_all(self.projects_dir())?;
        for folder in ["models", "loras", "cache"] {
            fs::create_dir_all(self.data_dir.join(folder))?;
        }
        Ok(())
    }

    fn load_registry(&self) -> ProjectStoreResult<Vec<RegistryItem>> {
        let registry_path = self.registry_path();
        if !registry_path.exists() {
            return Ok(Vec::new());
        }
        let payload = fs::read_to_string(registry_path)?;
        Ok(serde_json::from_str(&payload)?)
    }

    fn save_registry(&self, projects: &[RegistryItem]) -> ProjectStoreResult<()> {
        fs::create_dir_all(&self.data_dir)?;
        // `write_json` stages to a unique temp and renames atomically (sc-1633),
        // so the registry write no longer needs its own intermediate temp file.
        write_json(&self.registry_path(), &serde_json::to_value(projects)?)
    }

    /// Resolves the project path and acquires the per-project file lock, so the
    /// caller's read-modify-write of that project's JSON/sidecar files is
    /// serialized against other threads mutating the same project (sc-1633).
    /// Read-only paths keep using [`find_project_path`] and don't take the lock.
    fn lock_project(
        &self,
        project_id: &str,
    ) -> ProjectStoreResult<(PathBuf, ReentrantMutexGuard<'static, ()>)> {
        let project_path = self.find_project_path(project_id)?;
        let guard = lock_project_files(&project_path);
        Ok((project_path, guard))
    }

    fn find_project_path(&self, project_id: &str) -> ProjectStoreResult<PathBuf> {
        let _guard = self.lock.lock();
        for item in self.load_registry()? {
            if item.id.as_deref() == Some(project_id) {
                let Some(path) = item.path else {
                    break;
                };
                let project_path = PathBuf::from(path);
                if project_path.exists() {
                    return Ok(project_path);
                }
                break;
            }
        }
        Err(ProjectStoreError::NotFound("Project not found".to_owned()))
    }

    fn find_asset_sidecar(
        &self,
        project_path: &Path,
        asset_id: &str,
    ) -> ProjectStoreResult<PathBuf> {
        // Charset-check before any path use. This is the chokepoint every
        // asset-mutating method (get/update/delete/purge) hits first, so a
        // crafted-sidecar id like `../../x` is rejected before `delete_asset`/
        // `purge_asset` join it into a `trash/<asset_id>` dir or `remove_dir_all`
        // it (sc-4211 / F-CORE-7).
        if !is_safe_id(asset_id) {
            return Err(ProjectStoreError::BadRequest("Invalid asset id".to_owned()));
        }
        find_asset_sidecar_path(project_path, asset_id)?
            .ok_or_else(|| ProjectStoreError::NotFound("Asset not found".to_owned()))
    }
}

#[derive(Default)]
struct ReindexCounts {
    assets: u32,
    generation_sets: u32,
    timelines: u32,
}

/// Bump whenever the project.db schema changes (new table, column, or index).
/// `apply_project_migrations` only re-runs its DDL when `PRAGMA user_version`
/// is behind this; forgetting to bump means an existing DB never gets the change.
///
/// v3: sc-2022 added the `training_datasets.character_id` column to the migration
/// but left this at 2, so any project.db already stamped at v2 hit the early-return
/// gate and never got the column — the dataset list query then failed with
/// "no such column: character_id". Bumping forces the idempotent migration to
/// replay and add the column on existing databases.
///
/// v4: sc-10117 — no schema change, but the bump forces the one-time reindex in
/// `ensure_project_db_ready`, which now heals inline-upscaled asset sidecars that
/// were missing their fold lineage (`extra.upscaledFromAssetId` / `lineage`), so
/// existing upscale pairs collapse in the Library on next open.
///
/// v5: sc-13517 — added the `saved_voices` table (Voice Clone "register a voice"
/// registry). The bump lets existing DBs pick up the new table through the gated
/// migration; the table is DB-authoritative (no sidecar), so the reindex the bump
/// triggers — which only clears the sidecar-rebuilt tables — leaves it intact.
const PROJECT_SCHEMA_VERSION: i64 = 5;

fn project_schema_version(connection: &Connection) -> ProjectStoreResult<i64> {
    Ok(connection.query_row("pragma user_version", [], |row| row.get(0))?)
}

pub fn apply_project_migrations(connection: &Connection) -> ProjectStoreResult<()> {
    if project_schema_version(connection)? >= PROJECT_SCHEMA_VERSION {
        return Ok(());
    }
    connection.execute_batch(
        "
        create table if not exists project_metadata (
          key text primary key,
          value text not null
        );
        insert or ignore into project_metadata (key, value) values ('schemaVersion', '1');
        create table if not exists assets (
          id text primary key,
          type text not null,
          display_name text not null,
          file_path text not null,
          generation_set_id text,
          created_at text not null,
          favorite integer not null default 0,
          rating integer not null default 0,
          rejected integer not null default 0,
          trashed integer not null default 0
        );
        create table if not exists generation_sets (
          id text primary key,
          mode text not null,
          model text not null,
          prompt text not null,
          created_at text not null,
          job_id text
        );
        create table if not exists timelines (
          id text primary key,
          name text not null,
          file_path text not null,
          aspect_ratio text not null,
          width integer not null,
          height integer not null,
          fps integer not null,
          duration real not null default 0,
          created_at text not null,
          updated_at text not null
        );
        ",
    )?;
    ensure_column(connection, "assets", "sidecar_path", "text")?;
    // sc-2024: the originating studio (image_studio / video_studio /
    // document_studio / character_studio / upload). Existing rows are backfilled
    // by the reindex that `ensure_project_db_ready` runs on this version bump.
    ensure_column(connection, "assets", "origin", "text")?;
    apply_character_migrations(connection)?;
    apply_training_dataset_migrations(connection)?;
    crate::voice_store::apply_voice_migrations(connection)?;
    // Pragma assignment cannot be parameterized; the version is a trusted const.
    connection.execute_batch(&format!("pragma user_version = {PROJECT_SCHEMA_VERSION}"))?;
    Ok(())
}

pub fn ensure_project_db_ready(project_path: &Path) -> ProjectStoreResult<()> {
    let connection = connect_project_db(project_path)?;
    let version_before = project_schema_version(&connection)?;
    apply_project_migrations(&connection)?;
    drop(connection);
    // A schema bump can add derived columns (e.g. `origin`, sc-2024) that
    // existing rows lack. Rebuild the index from sidecars once so those columns
    // are backfilled; subsequent calls are no-ops (version already current).
    if version_before < PROJECT_SCHEMA_VERSION {
        reindex_project_path(project_path)?;
    }
    Ok(())
}

/// sc-10117: heal upscale-variant lineage on inline-upscaled asset sidecars.
///
/// The Image Studio inline "Upscale" post-pass historically linked each `(Nx upscaled)` variant to
/// its base image with a bare `upscaledFrom` field that nothing read — and that was dropped at
/// sidecar-build time — so those variants never carried `extra.upscaledFromAssetId` /
/// `lineage.sourceAssetId` and never folded with their originals in the Library / Recent Batches.
/// The link was never persisted, so it is reconstructed from generation-set structure: within a
/// generation set an upscaled file `…_{index}_up{N}x.<ext>` is the upscaled variant of the base
/// `…_{index}.<ext>`. Rewrites only upscaled sidecars still missing the link (idempotent + additive)
/// and returns how many were healed. A reconstruction with no matching base is left untouched rather
/// than inventing a bogus parent.
fn backfill_upscale_variant_lineage(project_path: &Path) -> ProjectStoreResult<usize> {
    let sidecars = asset_sidecars(project_path)?;
    // (generationSetId, relative media path) -> asset id, for every asset in the project.
    let mut base_ids: std::collections::HashMap<(String, String), String> =
        std::collections::HashMap::new();
    let mut parsed: Vec<(PathBuf, Value)> = Vec::with_capacity(sidecars.len());
    for sidecar_path in sidecars {
        let Ok(asset) = read_json(&sidecar_path) else {
            continue;
        };
        if let (Some(id), Some(genset), Some(path)) = (
            asset.get("id").and_then(Value::as_str),
            asset.get("generationSetId").and_then(Value::as_str),
            asset.pointer("/file/path").and_then(Value::as_str),
        ) {
            base_ids.insert((genset.to_owned(), path.to_owned()), id.to_owned());
        }
        parsed.push((sidecar_path, asset));
    }

    let mut healed = 0usize;
    for (sidecar_path, mut asset) in parsed {
        // Already linked (a standalone image_upscale asset, or a prior heal) — skip.
        if asset
            .pointer("/extra/upscaledFromAssetId")
            .and_then(Value::as_str)
            .is_some()
        {
            continue;
        }
        // Only touch assets the inline upscale post-pass produced (they carry an `upscale` record).
        if asset
            .pointer("/recipe/rawAdapterSettings/upscale")
            .is_none()
        {
            continue;
        }
        let (Some(genset), Some(path)) = (
            asset
                .get("generationSetId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            asset
                .pointer("/file/path")
                .and_then(Value::as_str)
                .map(str::to_owned),
        ) else {
            continue;
        };
        let Some((base_path, factor)) = strip_upscale_suffix(&path) else {
            continue;
        };
        // The reconstructed base must actually exist in the same generation set.
        let Some(base_id) = base_ids.get(&(genset, base_path)).cloned() else {
            continue;
        };
        let engine = asset
            .pointer("/recipe/rawAdapterSettings/upscale/engine")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        apply_upscale_variant_link(&mut asset, &base_id, factor, &engine);
        // A single failed rewrite must not abort the whole heal (or the reindex that calls it).
        if write_json(&sidecar_path, &asset).is_ok() {
            healed += 1;
        }
    }
    Ok(healed)
}

/// Split an upscaled media path `…_up{N}x.<ext>` into its base path `….<ext>` and the factor `N`.
/// Returns `None` when the filename doesn't carry the inline-upscale suffix.
fn strip_upscale_suffix(path: &str) -> Option<(String, u8)> {
    let (stem, ext) = path.rsplit_once('.')?;
    let marker = stem.rfind("_up")?;
    let factor: u8 = stem[marker + 3..].strip_suffix('x')?.parse().ok()?;
    Some((format!("{}.{ext}", &stem[..marker]), factor))
}

/// Stamp the fold/lineage keys the Library reads onto an upscaled asset in place, mirroring the
/// worker's standalone `image_upscale` fact (`sourceAssetId` / `parents` / `extra` markers).
fn apply_upscale_variant_link(asset: &mut Value, base_id: &str, factor: u8, engine: &str) {
    let Some(object) = asset.as_object_mut() else {
        return;
    };
    if let Some(lineage) = object
        .entry("lineage")
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        lineage.insert("sourceAssetId".to_owned(), json!(base_id));
        lineage.insert("parents".to_owned(), json!([base_id]));
    }
    if let Some(extra) = object
        .entry("extra")
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        extra.insert("isUpscaled".to_owned(), json!(true));
        extra.insert("upscaledFromAssetId".to_owned(), json!(base_id));
        extra.insert("factor".to_owned(), json!(factor));
        if !engine.is_empty() {
            extra.insert("engine".to_owned(), json!(engine));
        }
    }
}

/// Collect the upscale-fold lineage group around `asset_id` (sc-10205): the asset
/// itself, the original it was upscaled from, and transitively every upscaled
/// variant pointing back into the group. Mirrors the web's fold keys
/// (`extra.upscaledFromAssetId`, falling back to `lineage.sourceAssetId` when
/// `extra.isUpscaled` — assetVariants.js): the Library renders such a pair as ONE
/// tile, so a move of the visible asset must carry the whole group. The requested
/// id is always first. Unreadable sidecars are skipped — worst case the group
/// degrades to the single asset, never to an error.
fn upscale_lineage_group(project_path: &Path, asset_id: &str) -> Vec<String> {
    let mut group = vec![asset_id.to_owned()];
    let Ok(sidecars) = asset_sidecars(project_path) else {
        return group;
    };
    // Every asset's fold parent (the original it was upscaled from), if any.
    let mut parent_of: Vec<(String, String)> = Vec::new();
    for sidecar_path in sidecars {
        let Ok(asset) = read_json(&sidecar_path) else {
            continue;
        };
        let Some(id) = asset.get("id").and_then(Value::as_str) else {
            continue;
        };
        let parent = asset
            .pointer("/extra/upscaledFromAssetId")
            .and_then(Value::as_str)
            .or_else(|| {
                asset
                    .pointer("/extra/isUpscaled")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                    .then(|| {
                        asset
                            .pointer("/lineage/sourceAssetId")
                            .and_then(Value::as_str)
                    })
                    .flatten()
            });
        if let Some(parent) = parent {
            parent_of.push((id.to_owned(), parent.to_owned()));
        }
    }
    // Fixed-point closure over the parent links in both directions (covers
    // upscale-of-upscale chains and multiple variants of one original).
    loop {
        let before = group.len();
        for (child, parent) in &parent_of {
            let child_in = group.iter().any(|id| id == child);
            let parent_in = group.iter().any(|id| id == parent);
            if child_in && !parent_in {
                group.push(parent.clone());
            } else if parent_in && !child_in {
                group.push(child.clone());
            }
        }
        if group.len() == before {
            break;
        }
    }
    group
}

fn reindex_project_path(project_path: &Path) -> ProjectStoreResult<ReindexCounts> {
    // sc-10117: rewrite inline-upscaled sidecars missing their fold link BEFORE indexing, so the
    // rebuilt index — and the Library, which reads the sidecars directly — sees the healed lineage.
    // Additive + idempotent, so it is safe to run on every reindex.
    backfill_upscale_variant_lineage(project_path)?;

    let mut connection = connect_project_db(project_path)?;
    let transaction = connection.transaction()?;
    apply_project_migrations(&transaction)?;
    transaction.execute("delete from assets", [])?;
    transaction.execute("delete from generation_sets", [])?;
    transaction.execute("delete from timelines", [])?;
    clear_character_tables(&transaction)?;

    let mut counts = ReindexCounts::default();
    for sidecar_path in asset_sidecars(project_path)? {
        let Ok(asset) = read_json(&sidecar_path) else {
            continue;
        };
        if asset.get("id").is_none() || asset.pointer("/file/path").is_none() {
            continue;
        }
        index_asset_on_connection(&transaction, project_path, &asset, Some(&sidecar_path))?;
        counts.assets += 1;
    }

    let generation_set_dir = project_path.join("generation-sets");
    for entry in read_dir_paths(&generation_set_dir)? {
        if entry.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(generation_set) = read_json(&entry) else {
            continue;
        };
        if generation_set.get("id").is_none() {
            continue;
        }
        transaction.execute(
            "
            insert or replace into generation_sets (id, mode, model, prompt, created_at, job_id)
            values (?1, ?2, ?3, ?4, ?5, ?6)
            ",
            params![
                required_str(&generation_set, "id")?,
                optional_str(&generation_set, "mode").unwrap_or("unknown"),
                optional_str(&generation_set, "model").unwrap_or("unknown"),
                optional_str(&generation_set, "prompt").unwrap_or(""),
                optional_str(&generation_set, "createdAt").unwrap_or(""),
                optional_str(&generation_set, "jobId"),
            ],
        )?;
        counts.generation_sets += 1;
    }

    let timeline_dir = project_path.join("timelines");
    for entry in read_dir_paths(&timeline_dir)? {
        if !entry
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
        {
            continue;
        }
        let Ok(timeline) = read_json(&entry) else {
            continue;
        };
        if timeline.get("id").is_none() {
            continue;
        }
        let rel_path = relative_string(project_path, &entry)?;
        transaction.execute(
            "
            insert or replace into timelines (
              id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ",
            params![
                required_str(&timeline, "id")?,
                optional_str(&timeline, "name").unwrap_or("Timeline"),
                rel_path,
                optional_str(&timeline, "aspectRatio").unwrap_or("16:9"),
                optional_u64(&timeline, "width").unwrap_or(1280),
                optional_u64(&timeline, "height").unwrap_or(720),
                optional_u64(&timeline, "fps").unwrap_or(30),
                optional_f64(&timeline, "duration").unwrap_or(0.0),
                optional_str(&timeline, "createdAt").unwrap_or(""),
                optional_str(&timeline, "updatedAt")
                    .or_else(|| optional_str(&timeline, "createdAt"))
                    .unwrap_or(""),
            ],
        )?;
        counts.timelines += 1;
    }

    reindex_characters_on_connection(&transaction, project_path)?;

    transaction.commit()?;
    Ok(counts)
}

fn timeline_dimensions(aspect_ratio: &str) -> ProjectStoreResult<(u32, u32)> {
    match aspect_ratio {
        "16:9" => Ok((1280, 720)),
        "9:16" => Ok((720, 1280)),
        "1:1" => Ok((1024, 1024)),
        _ => Err(ProjectStoreError::BadRequest(
            "Aspect ratio must be one of 16:9, 9:16, or 1:1".to_owned(),
        )),
    }
}

fn default_timeline_tracks() -> Value {
    json!([
        {"id": "track_main", "name": "Main", "kind": "video", "locked": false, "muted": false, "items": []},
        {"id": "track_overlay", "name": "Overlay", "kind": "overlay", "locked": false, "muted": false, "items": []},
        {"id": "track_audio", "name": "Audio", "kind": "audio", "locked": false, "muted": false, "items": []}
    ])
}

fn compute_timeline_duration(timeline: &Value) -> f64 {
    timeline
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|track| track.get("items").and_then(Value::as_array))
        .flatten()
        .filter_map(|item| item.get("timelineEnd").and_then(Value::as_f64))
        .fold(0.0, f64::max)
}

fn validate_and_default_timeline(timeline: &mut Value) -> ProjectStoreResult<()> {
    let object = timeline
        .as_object_mut()
        .ok_or_else(|| ProjectStoreError::BadRequest("Timeline must be an object".to_owned()))?;
    object
        .entry("schemaVersion".to_owned())
        .or_insert_with(|| json!(1));
    object
        .entry("aspectRatio".to_owned())
        .or_insert_with(|| Value::String("16:9".to_owned()));
    object
        .entry("width".to_owned())
        .or_insert_with(|| json!(1280));
    object
        .entry("height".to_owned())
        .or_insert_with(|| json!(720));
    object.entry("fps".to_owned()).or_insert_with(|| json!(30));
    object
        .entry("duration".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("tracks".to_owned())
        .or_insert_with(|| json!([]));
    object
        .entry("transitions".to_owned())
        .or_insert_with(|| json!([]));

    validate_u64_range(timeline, "schemaVersion", 0, u32::MAX as u64)?;
    validate_enum(timeline, "aspectRatio", &["16:9", "9:16", "1:1"])?;
    validate_u64_range(timeline, "width", 256, 3840)?;
    validate_u64_range(timeline, "height", 256, 3840)?;
    validate_u64_range(timeline, "fps", 1, 60)?;
    validate_f64_range(timeline, "duration", 0.0, f64::INFINITY)?;

    let tracks = timeline
        .get_mut("tracks")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProjectStoreError::BadRequest("tracks must be an array".to_owned()))?;
    for track in tracks {
        validate_timeline_track(track)?;
    }

    let transitions = timeline
        .get_mut("transitions")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProjectStoreError::BadRequest("transitions must be an array".to_owned()))?;
    for transition in transitions {
        validate_timeline_transition(transition)?;
    }
    Ok(())
}

fn validate_timeline_track(track: &mut Value) -> ProjectStoreResult<()> {
    required_str(track, "id")?;
    required_str(track, "name")?;
    validate_enum(track, "kind", &["video", "overlay", "audio"])?;
    let object = track.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline track must be an object".to_owned())
    })?;
    object
        .entry("locked".to_owned())
        .or_insert_with(|| Value::Bool(false));
    object
        .entry("muted".to_owned())
        .or_insert_with(|| Value::Bool(false));
    object
        .entry("items".to_owned())
        .or_insert_with(|| json!([]));
    validate_bool(track, "locked")?;
    validate_bool(track, "muted")?;
    let items = track
        .get_mut("items")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProjectStoreError::BadRequest("track items must be an array".to_owned()))?;
    for item in items {
        validate_timeline_item(item)?;
    }
    Ok(())
}

fn validate_timeline_item(item: &mut Value) -> ProjectStoreResult<()> {
    required_str(item, "trackId")?;
    validate_required_string(item, "assetId", Some(1), None)?;
    validate_required_string(item, "displayName", Some(1), Some(160))?;
    let object = item.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline item must be an object".to_owned())
    })?;
    if !object.contains_key("id") || object.get("id") == Some(&Value::Null) {
        object.insert(
            "id".to_owned(),
            Value::String(format!("item_{}", random_hex(16)?)),
        );
    }
    object
        .entry("type".to_owned())
        .or_insert_with(|| Value::String("video".to_owned()));
    object
        .entry("sourceIn".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("sourceOut".to_owned())
        .or_insert_with(|| json!(4.0));
    object
        .entry("timelineStart".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("timelineEnd".to_owned())
        .or_insert_with(|| json!(4.0));
    object
        .entry("speed".to_owned())
        .or_insert_with(|| json!(1.0));
    object
        .entry("fit".to_owned())
        .or_insert_with(|| Value::String("fit".to_owned()));
    object
        .entry("volume".to_owned())
        .or_insert_with(|| json!(1.0));
    object
        .entry("versionAssetIds".to_owned())
        .or_insert_with(|| json!([]));
    object
        .entry("versionHistory".to_owned())
        .or_insert_with(|| json!([]));

    validate_required_string(item, "id", Some(1), None)?;
    validate_enum(item, "type", &["video", "image", "audio"])?;
    let source_in = validate_f64_range(item, "sourceIn", 0.0, f64::INFINITY)?;
    let source_out = validate_f64_range(item, "sourceOut", 0.0, f64::INFINITY)?;
    let timeline_start = validate_f64_range(item, "timelineStart", 0.0, f64::INFINITY)?;
    let timeline_end = validate_f64_range(item, "timelineEnd", 0.0, f64::INFINITY)?;
    let speed = validate_f64_range(item, "speed", 0.1, 8.0)?;
    validate_enum(item, "fit", &["fit", "fill", "stretch"])?;
    let volume = validate_f64_range(item, "volume", 0.0, 2.0)?;
    let object = item.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline item must be an object".to_owned())
    })?;
    object.insert("sourceIn".to_owned(), json!(source_in));
    object.insert("sourceOut".to_owned(), json!(source_out));
    object.insert("timelineStart".to_owned(), json!(timeline_start));
    object.insert("timelineEnd".to_owned(), json!(timeline_end));
    object.insert("speed".to_owned(), json!(speed));
    object.insert("volume".to_owned(), json!(volume));
    object
        .entry("transitionIn".to_owned())
        .or_insert(Value::Null);
    object
        .entry("transitionOut".to_owned())
        .or_insert(Value::Null);
    if source_out <= source_in {
        return Err(ProjectStoreError::BadRequest(
            "sourceOut must be greater than sourceIn.".to_owned(),
        ));
    }
    if timeline_end <= timeline_start {
        return Err(ProjectStoreError::BadRequest(
            "timelineEnd must be greater than timelineStart.".to_owned(),
        ));
    }

    item.get("versionAssetIds")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProjectStoreError::BadRequest("versionAssetIds must be an array".to_owned())
        })?;
    let versions = item
        .get_mut("versionHistory")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| {
            ProjectStoreError::BadRequest("versionHistory must be an array".to_owned())
        })?;
    for version in versions {
        validate_timeline_item_version(version)?;
    }
    if let Some(transition) = item
        .get_mut("transitionIn")
        .filter(|value| !value.is_null())
    {
        validate_timeline_transition(transition)?;
    }
    if let Some(transition) = item
        .get_mut("transitionOut")
        .filter(|value| !value.is_null())
    {
        validate_timeline_transition(transition)?;
    }
    Ok(())
}

fn validate_timeline_item_version(version: &mut Value) -> ProjectStoreResult<()> {
    validate_required_string(version, "assetId", Some(1), None)?;
    let object = version.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline item version must be an object".to_owned())
    })?;
    object
        .entry("source".to_owned())
        .or_insert_with(|| Value::String("manual".to_owned()));
    validate_enum(
        version,
        "source",
        &[
            "original",
            "replacement",
            "extension",
            "bridge",
            "restore",
            "manual",
        ],
    )
}

fn validate_timeline_transition(transition: &mut Value) -> ProjectStoreResult<()> {
    let object = transition.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline transition must be an object".to_owned())
    })?;
    if !object.contains_key("id") || object.get("id") == Some(&Value::Null) {
        object.insert(
            "id".to_owned(),
            Value::String(format!("transition_{}", random_hex(16)?)),
        );
    }
    object
        .entry("type".to_owned())
        .or_insert_with(|| Value::String("cut".to_owned()));
    object
        .entry("duration".to_owned())
        .or_insert_with(|| json!(0.0));
    validate_required_string(transition, "id", Some(1), None)?;
    validate_enum(
        transition,
        "type",
        &["cut", "crossfade", "fade_from_black", "fade_to_black"],
    )?;
    validate_f64_range(transition, "duration", 0.0, 10.0)?;
    Ok(())
}

fn normalize_timeline_items(timeline: &mut Value) -> ProjectStoreResult<()> {
    let Some(tracks) = timeline.get_mut("tracks").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for track in tracks {
        let Some(items) = track.get_mut("items").and_then(Value::as_array_mut) else {
            continue;
        };
        for item in items {
            let Some(object) = item.as_object_mut() else {
                continue;
            };
            let Some(asset_id) = object
                .get("assetId")
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                continue;
            };
            object
                .entry("currentVersionAssetId".to_owned())
                .or_insert_with(|| Value::String(asset_id.clone()));
            let version_ids = object
                .entry("versionAssetIds".to_owned())
                .or_insert_with(|| json!([]));
            if let Some(version_ids) = version_ids.as_array_mut() {
                let has_asset = version_ids
                    .iter()
                    .any(|value| value.as_str() == Some(&asset_id));
                if !has_asset {
                    version_ids.push(Value::String(asset_id.clone()));
                }
            }
            let needs_history = object
                .get("versionHistory")
                .and_then(Value::as_array)
                .map_or(true, Vec::is_empty);
            if needs_history {
                object.insert(
                    "versionHistory".to_owned(),
                    json!([{
                        "assetId": asset_id,
                        "createdAt": Value::Null,
                        "source": "original",
                        "jobId": Value::Null,
                        "note": Value::Null
                    }]),
                );
            }
        }
    }
    Ok(())
}

fn validate_required_string(
    payload: &Value,
    key: &str,
    min_length: Option<usize>,
    max_length: Option<usize>,
) -> ProjectStoreResult<()> {
    let value = required_str(payload, key)?;
    let length = value.chars().count();
    if let Some(min) = min_length.filter(|min| length < *min) {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be at least {min} characters"
        )));
    }
    if let Some(max) = max_length.filter(|max| length > *max) {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be at most {max} characters"
        )));
    }
    Ok(())
}

fn validate_enum(payload: &Value, key: &str, allowed: &[&str]) -> ProjectStoreResult<()> {
    let value = required_str(payload, key)?;
    if !allowed.contains(&value) {
        return Err(ProjectStoreError::BadRequest(format!(
            "Unsupported value for {key}: {value}"
        )));
    }
    Ok(())
}

fn validate_bool(payload: &Value, key: &str) -> ProjectStoreResult<()> {
    if payload.get(key).and_then(Value::as_bool).is_none() {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be a boolean"
        )));
    }
    Ok(())
}

fn validate_u64_range(payload: &Value, key: &str, min: u64, max: u64) -> ProjectStoreResult<u64> {
    let value = payload
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("{key} must be an integer")))?;
    if value < min || value > max {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be between {min} and {max}"
        )));
    }
    Ok(value)
}

fn validate_f64_range(payload: &Value, key: &str, min: f64, max: f64) -> ProjectStoreResult<f64> {
    let value = payload
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("{key} must be a number")))?;
    if !value.is_finite() || value < min || value > max {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be between {min} and {max}"
        )));
    }
    Ok(value)
}

fn timeline_file_path(project_path: &Path, timeline_id: &str, name: &str) -> PathBuf {
    let suffix = timeline_id
        .chars()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    project_path.join("timelines").join(format!(
        "{}-{suffix}.sceneworks.timeline.json",
        slugify(name, "timeline", Some(48))
    ))
}

fn index_timeline(project_path: &Path, timeline: &Value, rel_path: &str) -> ProjectStoreResult<()> {
    let mut connection = connect_project_db(project_path)?;
    let transaction = connection.transaction()?;
    apply_project_migrations(&transaction)?;
    transaction.execute(
        "
        insert or replace into timelines (
          id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
        ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        ",
        params![
            required_str(timeline, "id")?,
            optional_str(timeline, "name").unwrap_or("Timeline"),
            rel_path,
            optional_str(timeline, "aspectRatio").unwrap_or("16:9"),
            optional_u64(timeline, "width").unwrap_or(1280),
            optional_u64(timeline, "height").unwrap_or(720),
            optional_u64(timeline, "fps").unwrap_or(30),
            optional_f64(timeline, "duration").unwrap_or(0.0),
            optional_str(timeline, "createdAt").unwrap_or(""),
            optional_str(timeline, "updatedAt")
                .or_else(|| optional_str(timeline, "createdAt"))
                .unwrap_or(""),
        ],
    )?;
    transaction.commit()?;
    Ok(())
}

fn find_timeline_file(project_path: &Path, timeline_id: &str) -> ProjectStoreResult<TimelineFile> {
    ensure_project_db_ready(project_path)?;
    let connection = connect_project_db(project_path)?;
    let indexed_path = connection
        .query_row(
            "select file_path from timelines where id = ?1",
            params![timeline_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if let Some(indexed_path) = indexed_path.as_deref() {
        let path = if is_safe_relative_path(indexed_path) {
            Some(project_path.join(indexed_path))
        } else {
            None
        };
        if let Some(path) = path.filter(|path| path.exists()) {
            let project_root = project_path.canonicalize()?;
            let canonical = path.canonicalize()?;
            if !canonical.starts_with(&project_root) {
                return Err(ProjectStoreError::BadRequest(
                    "Timeline file path must stay inside the project".to_owned(),
                ));
            }
            return Ok(TimelineFile {
                path: canonical,
                relative_path: indexed_path.to_owned(),
            });
        }
    }

    let timeline_dir = project_path.join("timelines");
    for candidate in read_dir_paths(&timeline_dir)? {
        if !candidate
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
        {
            continue;
        }
        let Ok(timeline) = read_json(&candidate) else {
            continue;
        };
        if timeline.get("id").and_then(Value::as_str) == Some(timeline_id) {
            let rel_path = relative_string(project_path, &candidate)?;
            // Single-statement autocommit keeps the stale-index repair atomic; if this grows
            // beyond one write, wrap the whole repair in an explicit transaction.
            connection.execute(
                "update timelines set file_path = ?1 where id = ?2",
                params![rel_path, timeline_id],
            )?;
            return Ok(TimelineFile {
                path: candidate,
                relative_path: rel_path,
            });
        }
    }

    if let Some(indexed_path) = indexed_path {
        return Err(ProjectStoreError::NotFound(format!(
            "Timeline file not found at indexed path {indexed_path}; reindex required"
        )));
    }
    Err(ProjectStoreError::NotFound("Timeline not found".to_owned()))
}

fn read_project_summary(project_path: &Path) -> ProjectStoreResult<ProjectSummary> {
    let project_file = project_path.join("project.json");
    if !project_file.exists() {
        return Err(ProjectStoreError::NotFound(
            "Project file not found".to_owned(),
        ));
    }
    let payload = read_json(&project_file)?;
    Ok(ProjectSummary {
        id: required_str(&payload, "id")?.to_owned(),
        name: required_str(&payload, "name")?.to_owned(),
        path: project_path.display().to_string(),
        created_at: required_str(&payload, "createdAt")?.to_owned(),
    })
}

fn write_project_file(
    app_version: &str,
    project_path: &Path,
    project_id: &str,
    name: &str,
) -> ProjectStoreResult<()> {
    let folders = PROJECT_FOLDERS
        .iter()
        .map(|folder| {
            (
                folder.replace('/', "_"),
                Value::String((*folder).to_owned()),
            )
        })
        .collect::<Map<_, _>>();
    write_json(
        &project_path.join("project.json"),
        &json!({
            "schemaVersion": 1,
            "appVersion": app_version,
            "id": project_id,
            "name": name,
            "createdAt": utc_now(),
            "folders": folders
        }),
    )
}

fn connect_project_db(project_path: &Path) -> ProjectStoreResult<Connection> {
    fs::create_dir_all(project_path)?;
    let connection = Connection::open(project_path.join("project.db"))?;
    connection.busy_timeout(Duration::from_millis(5000))?;
    Ok(connection)
}

/// True when `error` means the storage location rejected a write (as opposed to a
/// logic/validation error). Covers SQLite `SQLITE_READONLY` ("attempt to write a
/// readonly database"), `SQLITE_CANTOPEN` (read-only directory), `SQLITE_PERM`,
/// and IO `PermissionDenied`/`ReadOnlyFilesystem` — the whole family a
/// non-writable workspace folder produces (issue #1435 / sc-11855).
fn is_write_denied(error: &ProjectStoreError) -> bool {
    match error {
        ProjectStoreError::Sqlite(rusqlite::Error::SqliteFailure(err, _)) => matches!(
            err.code,
            rusqlite::ErrorCode::ReadOnly
                | rusqlite::ErrorCode::CannotOpen
                | rusqlite::ErrorCode::PermissionDenied
        ),
        ProjectStoreError::Io(err) => matches!(
            err.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem
        ),
        _ => false,
    }
}

/// Actionable, path-naming message for a non-writable workspace folder. Replaces
/// the opaque raw SQLite/IO text the user would otherwise see (issue #1435).
fn storage_not_writable_message(dir: &Path) -> String {
    format!(
        "SceneWorks can't write to your workspace folder ({}). This is almost always a \
         permissions problem with that location — pick a different workspace folder in \
         Settings, or fix the folder's permissions, then try again.",
        dir.display()
    )
}

/// Exercise the exact write path a `project.db` needs, so a non-writable
/// workspace folder fails fast with an actionable error instead of surfacing an
/// opaque `SQLITE_READONLY` from deep in project provisioning (issue #1435 /
/// sc-11855). A plain "can I create a file here" check is NOT enough: a granular
/// write denial (a macOS inherited ACL, or an endpoint-security/backup agent) can
/// permit creating and appending to files while blocking the in-place rewrites a
/// rollback-journal SQLite db performs — which is exactly why the WAL-mode
/// `jobs.db` keeps working while `project.db` writes are refused. So we open a
/// throwaway rollback-mode db in `dir`, write to it, and clean up.
///
/// Public so the rust-api startup path can run the same faithful probe against
/// the projects tree and log the result for diagnosis (sc-11855 fix C).
pub fn probe_storage_writable(dir: &Path) -> ProjectStoreResult<()> {
    fs::create_dir_all(dir)?;
    let probe = dir.join(format!(".sceneworks-write-test-{}.db", random_hex(8)?));
    let outcome = (|| -> rusqlite::Result<()> {
        let connection = Connection::open(&probe)?;
        connection.execute_batch("create table t (x); insert into t values (1); drop table t;")
    })();
    // Best-effort cleanup of the probe db and any rollback-journal sidecar,
    // regardless of outcome — never mask the real error with a cleanup failure.
    let _ = fs::remove_file(&probe);
    let mut journal = probe.clone().into_os_string();
    journal.push("-journal");
    let _ = fs::remove_file(PathBuf::from(journal));
    outcome.map_err(ProjectStoreError::from)
}

/// Open `project.db` with the shared `busy_timeout` AND run the version-gated
/// migrations, then hand back the ready connection. This is the house helper
/// external store modules (e.g. `training_store`) should use instead of a raw
/// `Connection::open(project_path.join("project.db"))`, which sets no
/// `busy_timeout` (so concurrent worker+API access hits an immediate
/// `SQLITE_BUSY` instead of waiting) and can miss migrations (sc-11202 / F-026).
/// `apply_project_migrations` is `PRAGMA user_version`-gated, so the DDL only
/// runs when the schema is actually behind — cheap on the common already-migrated
/// path.
pub(crate) fn connect_project_db_migrated(project_path: &Path) -> ProjectStoreResult<Connection> {
    let connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    Ok(connection)
}

/// Assemble the on-disk asset sidecar from the worker-reported flat facts. Rust
/// is the single owner of this envelope schema now (story 1656): the worker
/// ships values (paths, dimensions, seed, recipe inputs) and Rust builds the
/// `file`/`status`/`recipe`/`lineage` structure, matching the shape pinned by
/// `resource_sidecars.json` and the worker's former Python `build_asset_sidecar`.
/// `type` is derived from the mime so the video slice can reuse this.
fn build_generated_asset_sidecar(
    project_id: &str,
    job_id: &str,
    generation_set_id: &str,
    fact: &Value,
) -> Value {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    // The worker sends an explicit `type` (a procedural video preview is a .webp,
    // so mime alone can't classify it); fall back to mime for image facts that
    // predate it.
    let mime = fact
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or("image/png");
    let media_type = fact
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if mime.starts_with("video/") {
                "video".to_owned()
            } else {
                "image".to_owned()
            }
        });
    let (file, recipe, lineage) = match media_type.as_str() {
        "video" => build_video_sidecar_parts(job_id, fact),
        "audio" => build_audio_sidecar_parts(job_id, fact),
        "document" => build_document_sidecar_parts(job_id, fact),
        _ => build_image_sidecar_parts(job_id, fact),
    };
    let mut asset = json!({
        "schemaVersion": 1,
        "id": get("assetId"),
        "projectId": project_id,
        "generationSetId": generation_set_id,
        "type": media_type,
        "displayName": get("displayName"),
        "createdAt": get("createdAt"),
        "file": file,
        "status": { "favorite": false, "rating": 0, "rejected": false, "trashed": false },
        "recipe": recipe,
        "lineage": lineage,
    });
    // Record the originating studio so the Asset Library can exclude Character
    // Studio test outputs (sc-2024). Derived from recipe mode + media type.
    let origin = asset_origin(&asset);
    if let Some(object) = asset.as_object_mut() {
        object.insert("origin".to_owned(), Value::String(origin));
    }
    if let Some(extra) = fact.get("extra") {
        if let Some(object) = asset.as_object_mut() {
            object.insert("extra".to_owned(), extra.clone());
        }
    }
    asset
}

/// `(file, recipe, lineage)` for a generated image asset, matching the worker's
/// former `build_asset_sidecar` (story 1656).
fn build_image_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    let parents = fact
        .get("parents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| match fact.get("sourceAssetId").and_then(Value::as_str) {
            Some(source) => vec![Value::String(source.to_owned())],
            None => Vec::new(),
        });
    let file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": get("width"),
        "height": get("height"),
        "duration": Value::Null,
        "fps": Value::Null,
    });
    let recipe = json!({
        "mode": get("mode"),
        "model": get("model"),
        "adapter": get("adapter"),
        "prompt": get("prompt"),
        "negativePrompt": get("negativePrompt"),
        "seed": get("seed"),
        "loras": fact.get("loras").cloned().unwrap_or_else(|| json!([])),
        "stylePreset": get("stylePreset"),
        "normalizedSettings": {
            "width": get("normalizedWidth"),
            "height": get("normalizedHeight"),
            "count": get("count"),
            "family": get("family"),
            "characterId": get("characterId"),
            "characterLookId": get("characterLookId"),
            "characterConditioningActive": false,
            "characterConditioningNote": "Character metadata is recorded, but adapter-level character conditioning is not active in this build.",
        },
        "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
    });
    let lineage = json!({
        "parents": parents,
        "sourceAssetId": get("sourceAssetId"),
        "sourceTimestamp": Value::Null,
        "jobId": job_id,
    });
    (file, recipe, lineage)
}

/// `(file, recipe, lineage)` for a generated video asset, matching the worker's
/// former `build_video_asset_sidecar` (story 1656): video file carries
/// duration/fps, the recipe has no `stylePreset`, normalizedSettings are
/// video-shaped (and fold in the honest `replacement_status` for replace_person),
/// and lineage tracks the four source clip/frame ids + the timeline context.
fn build_video_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    // List-valued ids keep an array shape even on facts written before sc-12345, so the replay
    // reader never has to distinguish "absent" from "empty".
    let list_or_empty = |key: &str| match fact.get(key) {
        Some(value @ Value::Array(_)) => value.clone(),
        _ => json!([]),
    };
    let source_keys = [
        "sourceAssetId",
        "lastFrameAssetId",
        "sourceClipAssetId",
        "bridgeRightClipAssetId",
    ];
    // The list-valued source ids are parents too — mv2v's clips, the reference-driven modes'
    // subject images, and ads2v's reference video. Without them those modes' provenance names
    // only a subset of what the clip was actually derived from (sc-12345).
    let list_source_keys = ["sourceClipAssetIds", "referenceAssetIds"];
    let parents: Vec<Value> = source_keys
        .iter()
        .chain(std::iter::once(&"referenceClipAssetId"))
        .filter_map(|key| fact.get(*key).and_then(Value::as_str))
        .map(|id| Value::String(id.to_owned()))
        .chain(list_source_keys.iter().flat_map(|key| {
            fact.get(*key)
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(Value::as_str)
                        .map(|id| Value::String(id.to_owned()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        }))
        .collect();
    // sc-12371: `file` describes the mp4 on disk, so it takes the worker's MEASURED clip facts —
    // counted off the frames that were encoded, not the duration/fps the user asked for. Those
    // differ whenever the model's temporal stride snaps the frame count, which is usually: a 6 s x
    // 25 fps ask on a Wan engine renders 149 frames = 5.96 s, and `file.duration` claimed 6.0.
    //
    // Falls back to the requested values for facts written before the worker measured (assets
    // already on disk, and the non-video callers of this shape) — a slightly-wrong duration on an
    // old asset beats a missing one, and it is what those assets have always reported.
    let measured = |encoded: &str, requested: &str| match fact.get(encoded) {
        Some(value) if !value.is_null() => value.clone(),
        _ => get(requested),
    };
    let file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": get("width"),
        "height": get("height"),
        "duration": measured("encodedDuration", "duration"),
        "fps": measured("encodedFps", "fps"),
        "frameCount": fact.get("encodedFrameCount").cloned().unwrap_or(Value::Null),
    });
    // ...whereas `normalizedSettings` is the REPLAY record: the knobs the user picked off the
    // model's `limits.durations` / `limits.fps` menus, which "re-run this generation" (sc-12324 /
    // sc-12345) rebuilds the payload from. It must round-trip the ask, not the measurement.
    let mut normalized = json!({
        "duration": get("duration"),
        "fps": get("fps"),
        "width": get("width"),
        "height": get("height"),
        "quality": get("quality"),
        "family": get("family"),
        "sourceAssetId": get("sourceAssetId"),
        "lastFrameAssetId": get("lastFrameAssetId"),
        "sourceClipAssetId": get("sourceClipAssetId"),
        "bridgeRightClipAssetId": get("bridgeRightClipAssetId"),
        // sc-12345: the fit and the list-valued source ids the replay path needs. Older facts
        // predate them — `get` yields null and the studio falls back to its own defaults.
        "fitMode": get("fitMode"),
        "sourceClipAssetIds": list_or_empty("sourceClipAssetIds"),
        "referenceAssetIds": list_or_empty("referenceAssetIds"),
        "referenceClipAssetId": get("referenceClipAssetId"),
        "characterId": get("characterId"),
        "characterLookId": get("characterLookId"),
        "personTrackId": get("personTrackId"),
        "replacementMode": get("replacementMode"),
        "timelineContextRef": "lineage.timeline",
    });
    if fact.get("mode").and_then(Value::as_str) == Some("replace_person") {
        if let Some(object) = normalized.as_object_mut() {
            object.insert("personDetectionActive".to_owned(), Value::Bool(false));
            object.insert("personTrackingActive".to_owned(), Value::Bool(false));
            object.insert("replacementActive".to_owned(), Value::Bool(false));
            if let Some(status) = fact.get("replacementStatus").and_then(Value::as_object) {
                for (key, value) in status {
                    object.insert(key.clone(), value.clone());
                }
            }
        }
    }
    let recipe = json!({
        "mode": get("mode"),
        "model": get("model"),
        "adapter": get("adapter"),
        "prompt": get("prompt"),
        "negativePrompt": get("negativePrompt"),
        "seed": get("seed"),
        "loras": fact.get("loras").cloned().unwrap_or_else(|| json!([])),
        "normalizedSettings": normalized,
        "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
    });
    let lineage = json!({
        "parents": parents,
        "sourceAssetId": get("sourceAssetId"),
        "lastFrameAssetId": get("lastFrameAssetId"),
        "sourceClipAssetId": get("sourceClipAssetId"),
        "bridgeRightClipAssetId": get("bridgeRightClipAssetId"),
        "personTrackId": get("personTrackId"),
        "characterId": get("characterId"),
        "characterLookId": get("characterLookId"),
        "replacementMode": get("replacementMode"),
        "sourceTimestamp": Value::Null,
        "timeline": fact.get("timelineContext").cloned().unwrap_or_else(|| json!({})),
        "jobId": job_id,
    });
    (file, recipe, lineage)
}

/// `(file, recipe, lineage)` for a generated audio asset (SceneWorks Audio Studio, epic 13400 /
/// sc-13404) — the audio twin of [`build_video_sidecar_parts`]. The worker writes a WAV (via
/// `write_wav_pcm16`) and reports the measured `duration`/`sampleRate`/`channels` off the produced
/// `AudioTrack`, so the `file` block is honest about what is on disk; the recipe's
/// `normalizedSettings` round-trips the ASK (voice / language / targetDuration) so "re-run this
/// generation" can rebuild the payload. A pure-audio asset has no width/height (unlike an image or
/// video), so those are null.
fn build_audio_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    let parents = fact
        .get("parents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| match fact.get("sourceAssetId").and_then(Value::as_str) {
            Some(source) => vec![Value::String(source.to_owned())],
            None => Vec::new(),
        });
    let file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": Value::Null,
        "height": Value::Null,
        // MEASURED off the produced AudioTrack (the WAV on disk), mirroring the video file block's
        // encoded facts. `duration` is the clip's real running time in seconds; `sampleRate` and
        // `channels` describe the PCM the worker wrote.
        "duration": get("duration"),
        "fps": Value::Null,
        "sampleRate": get("sampleRate"),
        "channels": get("channels"),
    });
    let recipe = json!({
        "mode": get("mode"),
        "model": get("model"),
        "adapter": get("adapter"),
        "prompt": get("prompt"),
        "negativePrompt": get("negativePrompt"),
        "seed": get("seed"),
        "loras": fact.get("loras").cloned().unwrap_or_else(|| json!([])),
        // The REPLAY record: the audio knobs the user picked (voice / language / requested
        // duration) so a re-run rebuilds the same request. `sampleRate` records the rate actually
        // produced (Kokoro's native 24 kHz today).
        "normalizedSettings": {
            "voice": get("voice"),
            "language": get("language"),
            "targetDurationSecs": get("targetDurationSecs"),
            "sampleRate": get("sampleRate"),
            "family": get("family"),
        },
        "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
    });
    let lineage = json!({
        "parents": parents,
        "sourceAssetId": get("sourceAssetId"),
        "sourceTimestamp": Value::Null,
        "jobId": job_id,
    });
    (file, recipe, lineage)
}

/// `(file, recipe, lineage)` for an interleaved document asset (story 1656,
/// slice 4). The worker writes the document body JSON (the "media"); Rust builds
/// this sidecar from the document fact. The recipe mode is `interleave` and
/// lineage parents are the embedded image asset ids.
fn build_document_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    let file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": Value::Null,
        "height": Value::Null,
        "duration": Value::Null,
        "fps": Value::Null,
    });
    let recipe = json!({
        "mode": get("mode"),
        "model": get("model"),
        "adapter": get("adapter"),
        "prompt": get("prompt"),
        "negativePrompt": get("negativePrompt"),
        "seed": get("seed"),
        "loras": fact.get("loras").cloned().unwrap_or_else(|| json!([])),
        "normalizedSettings": {
            "maxImages": get("maxImages"),
            "resolution": get("resolution"),
            "imageCount": get("imageCount"),
        },
        "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
    });
    let lineage = json!({
        "parents": fact.get("parents").cloned().unwrap_or_else(|| json!([])),
        "sourceAssetId": Value::Null,
        "sourceTimestamp": Value::Null,
        "jobId": job_id,
    });
    (file, recipe, lineage)
}

/// Parse + validate the 5-point normalized kps from a spec into a JSON `[[x,y]×5]` array.
/// Sniff a still-image format from its magic bytes → `(extension, mime)`, or `None` when the
/// header isn't a format the library retains. Used to label a captured keypoint source correctly
/// regardless of the staged temp file's (extension-less) name.
/// Recognize a natively-decodable image (png/jpeg/webp) from its magic bytes, returning the
/// `(extension, mime)` to store it under. Returns `None` for any other format — the keypoint path's
/// callers then fall back to the extension/PNG. Delegates to the shared [`crate::media_convert`]
/// sniffer so there is one source of truth for image magic bytes (sc-6143).
fn sniff_image_format(path: &Path) -> Option<(&'static str, &'static str)> {
    let kind = crate::media_convert::sniff_image_kind_at(path)?;
    kind.is_natively_supported().then(|| kind.canonical())
}

/// The outcome of import-time image normalization (sc-6143): the path whose bytes should be moved
/// into the asset store, plus the mime type and file extension to record. When `transcoded_temp` is
/// set, the move consumes that temp PNG (not the original upload), so the caller must remove the
/// original source itself.
struct NormalizedUpload {
    source_path: PathBuf,
    content_type: String,
    extension: String,
    transcoded_temp: Option<PathBuf>,
}

/// Normalize a valid-but-unsupported image upload (AVIF/HEIC/HEIF/TIFF/BMP/GIF) to lossless PNG
/// before it is stored, so every downstream decode site, thumbnail, and preview stays on a format
/// it can read (sc-6143). The format is sniffed by content, never the extension, so a `.png` that is
/// really AVIF is still handled. Already-supported formats (png/jpeg/webp) and non-image uploads
/// pass through untouched — no temp file, no re-encode, no quality loss.
fn normalize_image_upload(
    source_path: &Path,
    content_type: &str,
    filename: &str,
    work_dir: &Path,
) -> ProjectStoreResult<NormalizedUpload> {
    let passthrough = || NormalizedUpload {
        source_path: source_path.to_path_buf(),
        content_type: content_type.to_owned(),
        extension: upload_extension(filename, content_type),
        transcoded_temp: None,
    };
    if !content_type.starts_with("image/") {
        return Ok(passthrough());
    }
    let Some(kind) = crate::media_convert::sniff_image_kind_at(source_path) else {
        // Unrecognized magic (e.g. SVG) — leave it as-is; we can't transcode what we can't sniff.
        return Ok(passthrough());
    };
    if kind.is_natively_supported() {
        // Already decodable: keep the bytes (no re-encode) but record the format we actually
        // detected, so a mislabeled extension (a `.avif` that is really PNG) is corrected.
        let (extension, mime) = kind.canonical();
        return Ok(NormalizedUpload {
            source_path: source_path.to_path_buf(),
            content_type: mime.to_owned(),
            extension: format!(".{extension}"),
            transcoded_temp: None,
        });
    }
    let temp_png = work_dir.join(format!("upload-transcode-{}.png", random_hex(8)?));
    crate::media_convert::transcode_to_png(source_path, &temp_png).map_err(|error| {
        let _ = fs::remove_file(&temp_png);
        ProjectStoreError::BadRequest(format!(
            "Could not convert {} image to a supported format: {error}",
            kind.label()
        ))
    })?;
    Ok(NormalizedUpload {
        source_path: temp_png.clone(),
        content_type: "image/png".to_owned(),
        extension: ".png".to_owned(),
        transcoded_temp: Some(temp_png),
    })
}

/// Move a (possibly transcoded) upload into place, cleaning up the transcode temp on a move failure
/// and the now-orphaned original on success. Shared by the asset + training-dataset import paths.
fn move_normalized_upload(
    normalized: &NormalizedUpload,
    original_source: &Path,
    media_path: &Path,
) -> ProjectStoreResult<()> {
    if let Err(error) = move_or_copy_file(&normalized.source_path, media_path) {
        if let Some(temp) = &normalized.transcoded_temp {
            let _ = fs::remove_file(temp);
        }
        return Err(error);
    }
    // A transcode moved the temp PNG into place, so the original upload was never consumed — drop it.
    if normalized.transcoded_temp.is_some() {
        let _ = fs::remove_file(original_source);
    }
    Ok(())
}

fn parse_normalized_kps(value: Option<&Value>) -> ProjectStoreResult<Vec<Value>> {
    let points = value.and_then(Value::as_array).ok_or_else(|| {
        ProjectStoreError::BadRequest("keypoint spec missing kps array".to_owned())
    })?;
    if points.len() != 5 {
        return Err(ProjectStoreError::BadRequest(format!(
            "keypoint kps must have 5 points, got {}",
            points.len()
        )));
    }
    let mut out = Vec::with_capacity(5);
    for point in points {
        let pair = point
            .as_array()
            .filter(|values| values.len() == 2)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("each kps point must be [x, y]".to_owned())
            })?;
        let x = pair[0]
            .as_f64()
            .ok_or_else(|| ProjectStoreError::BadRequest("kps x must be a number".to_owned()))?;
        let y = pair[1]
            .as_f64()
            .ok_or_else(|| ProjectStoreError::BadRequest("kps y must be a number".to_owned()))?;
        if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
            return Err(ProjectStoreError::BadRequest(
                "kps must be normalized to [0,1]".to_owned(),
            ));
        }
        out.push(json!([x, y]));
    }
    Ok(out)
}

/// Parse a stored/served kps record (`[[x,y]×5]`) into a fixed `[(f32,f32);5]`, or `None` if it
/// isn't exactly 5 numeric pairs. Used to resolve a collection's presets for generation (sc-4450).
fn parse_kps_tuple(value: Option<&Value>) -> Option<[(f32, f32); 5]> {
    let points = value.and_then(Value::as_array)?;
    if points.len() != 5 {
        return None;
    }
    let mut out = [(0.0f32, 0.0f32); 5];
    for (slot, point) in out.iter_mut().zip(points) {
        let pair = point.as_array().filter(|values| values.len() == 2)?;
        *slot = (pair[0].as_f64()? as f32, pair[1].as_f64()? as f32);
    }
    Some(out)
}

/// The built-in 11 as resolved angle presets (the generation fallback / default set, sc-4450).
fn builtin_resolved_presets() -> Vec<ResolvedAnglePreset> {
    crate::angle_kps::BUILTIN_ANGLE_KPS
        .iter()
        .map(|(angle, kps)| ResolvedAnglePreset {
            preset_id: crate::angle_kps::builtin_preset_id(angle),
            name: crate::angle_kps::builtin_angle_display_name(angle),
            angle: Some((*angle).to_owned()),
            kps: *kps,
        })
        .collect()
}

/// Map a stored `type:"keypoint"` asset → a Key Point Library preset record.
fn keypoint_asset_to_preset(asset: &Value) -> Value {
    json!({
        "id": asset.get("id").cloned().unwrap_or(Value::Null),
        "name": asset.get("displayName").cloned().unwrap_or(Value::Null),
        "kps": asset
            .get("keypoint")
            .and_then(|keypoint| keypoint.get("kps"))
            .cloned()
            .unwrap_or(Value::Null),
        "builtin": false,
        "sourceImageRef": asset
            .get("file")
            .and_then(|file| file.get("path"))
            .cloned()
            .unwrap_or(Value::Null),
        "sourceAssetId": asset
            .get("lineage")
            .and_then(|lineage| lineage.get("sourceAssetId"))
            .cloned()
            .unwrap_or(Value::Null),
    })
}

/// Read the user angle-set collections (`keypoint-collections.json`) → `[]` when absent.
fn read_user_collections(project_path: &Path) -> ProjectStoreResult<Vec<Value>> {
    let path = project_path.join(KEYPOINT_COLLECTIONS_FILE);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let document = read_json(&path)?;
    Ok(document
        .get("collections")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default())
}

/// Persist the user angle-set collections sidecar.
fn write_user_collections(project_path: &Path, collections: &[Value]) -> ProjectStoreResult<()> {
    let path = project_path.join(KEYPOINT_COLLECTIONS_FILE);
    write_json(
        &path,
        &json!({ "schemaVersion": 1, "collections": collections }),
    )
}

fn index_asset(
    project_path: &Path,
    asset: &Value,
    sidecar_path: Option<&Path>,
) -> ProjectStoreResult<()> {
    let sidecar_path = match sidecar_path {
        Some(path) => path.to_path_buf(),
        None => project_path
            .join(required_str(
                asset.get("file").ok_or_else(|| {
                    ProjectStoreError::BadRequest("Asset file is required".to_owned())
                })?,
                "path",
            )?)
            .with_extension("sceneworks.json"),
    };
    let mut connection = connect_project_db(project_path)?;
    let transaction = connection.transaction()?;
    apply_project_migrations(&transaction)?;
    index_asset_on_connection(&transaction, project_path, asset, Some(&sidecar_path))?;
    transaction.commit()?;
    Ok(())
}

/// Thin DB-prep wrapper over the shared resolver in `asset_index` (sc-4272).
/// `ensure_project_db_ready` backfills derived columns after a schema bump
/// before the lookup, unlike the character-store wrapper's lighter migrate.
fn find_asset_sidecar_path(
    project_path: &Path,
    asset_id: &str,
) -> ProjectStoreResult<Option<PathBuf>> {
    ensure_project_db_ready(project_path)?;
    let connection = connect_project_db(project_path)?;
    find_asset_sidecar_path_on_connection(&connection, project_path, asset_id)
}

fn purge_asset_record(project_path: &Path, asset_id: &str) -> ProjectStoreResult<()> {
    let mut connection = connect_project_db(project_path)?;
    let transaction = connection.transaction()?;
    apply_project_migrations(&transaction)?;
    transaction.execute("delete from assets where id = ?1", params![asset_id])?;
    transaction.commit()?;
    Ok(())
}

/// Sidecars of pruned "purged-but-referenced" assets are moved here, out of the
/// folders `asset_sidecars` scans, so a rebuild can't resurrect them. A dotdir so
/// it reads as internal state, not user content.
const ORPHANED_SIDECAR_DIR: &str = ".orphaned-sidecars";

/// Delete index rows whose recorded media file is gone from disk, and quarantine
/// each orphaned sidecar out of the scanned asset tree. Backs
/// [`ProjectStore::prune_orphaned_assets`]; see that method for the rationale.
/// Takes the per-project file lock so the read-then-delete is serialized against
/// concurrent indexing/mutation.
fn prune_orphaned_asset_rows(project_path: &Path) -> ProjectStoreResult<usize> {
    let _guard = lock_project_files(project_path);
    let mut connection = connect_project_db(project_path)?;
    let transaction = connection.transaction()?;
    apply_project_migrations(&transaction)?;
    // (asset id, orphaned sidecar to quarantine) for every row whose media is gone.
    let orphans: Vec<(String, Option<PathBuf>)> = {
        let mut statement =
            transaction.prepare("select id, file_path, sidecar_path from assets")?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        let mut orphans = Vec::new();
        for row in rows {
            let (asset_id, file_rel, sidecar_rel) = row?;
            let Some(file_rel) = file_rel else {
                continue;
            };
            // Normalize separators before joining — matching `normalize_asset_cached`'s
            // URL builder — so a Windows-authored sidecar (backslash paths) isn't
            // misread as missing on a POSIX host and wrongly pruned.
            let file_rel = file_rel.replace('\\', "/");
            if project_path.join(&file_rel).exists() {
                continue;
            }
            // Resolve the orphan's sidecar (the indexed path first, else derived from
            // the media path — mirroring `list_assets`) so it can be quarantined too.
            let sidecar = sidecar_rel
                .map(|rel| project_path.join(rel.replace('\\', "/")))
                .filter(|path| path.exists())
                .or_else(|| {
                    let derived = project_path
                        .join(&file_rel)
                        .with_extension("sceneworks.json");
                    derived.exists().then_some(derived)
                });
            orphans.push((asset_id, sidecar));
        }
        orphans
    };
    for (asset_id, _) in &orphans {
        transaction.execute("delete from assets where id = ?1", params![asset_id])?;
    }
    transaction.commit()?;
    // The row(s) are gone from the registry; now move each orphaned sidecar OUT of
    // the scanned asset folders into `.orphaned-sidecars/` (never scanned by
    // `asset_sidecars`). This declutters the project and closes the resurrection
    // edge — a fully-orphaned project would otherwise re-add these rows via the
    // empty-table auto-reindex in `list_assets`. The sidecar is retained (moved, not
    // deleted) so the recipe metadata stays recoverable. Best-effort: a move failure
    // just leaves the sidecar in place (the row is already pruned).
    for (asset_id, sidecar) in &orphans {
        let Some(sidecar) = sidecar else {
            continue;
        };
        if let Err(error) = quarantine_orphaned_sidecar(project_path, asset_id, sidecar) {
            tracing::debug!(
                event = "orphaned_sidecar_quarantine_failed",
                asset_id = %asset_id,
                sidecar = %sidecar.display(),
                error = %error,
                "could not move orphaned sidecar out of the asset tree; leaving it in place"
            );
        }
    }
    Ok(orphans.len())
}

/// Move an orphaned asset's sidecar into `<project>/.orphaned-sidecars/`, keyed by
/// the (charset-validated) asset id. Retained rather than deleted so the metadata
/// is recoverable; a same-key file from a prior prune is overwritten.
fn quarantine_orphaned_sidecar(
    project_path: &Path,
    asset_id: &str,
    sidecar: &Path,
) -> std::io::Result<()> {
    let dir = project_path.join(ORPHANED_SIDECAR_DIR);
    fs::create_dir_all(&dir)?;
    let dest = dir.join(format!("{asset_id}.sceneworks.json"));
    if dest.exists() {
        fs::remove_file(&dest)?;
    }
    fs::rename(sidecar, &dest)
}

fn project_has_sidecars(project_path: &Path) -> bool {
    asset_sidecars(project_path).is_ok_and(|paths| !paths.is_empty())
}

fn normalize_person_track(project_path: &Path, track_path: &Path) -> ProjectStoreResult<Value> {
    let mut track = read_json(track_path)?;
    let track_rel = relative_string(project_path, track_path)?;
    if let Some(object) = track.as_object_mut() {
        object.insert("path".to_owned(), Value::String(track_rel));
    }
    Ok(track)
}

/// Validate and canonicalize the incoming correction set for a person track.
/// Each entry targets an in-range frame and carries a box adjustment, a
/// rejection, or both; the server stamps author/createdAt/source. Entries that
/// neither adjust a box nor reject the frame are dropped (a reset frame).
/// Duplicate frame indices keep the last occurrence so the set is one entry per
/// frame.
fn normalize_person_track_corrections(
    corrections: Vec<Value>,
    frame_count: usize,
) -> ProjectStoreResult<Vec<Value>> {
    let created_at = utc_now();
    let mut normalized: Vec<(u64, Value)> = Vec::with_capacity(corrections.len());
    for correction in corrections {
        let object = correction.as_object().ok_or_else(|| {
            ProjectStoreError::BadRequest("Each correction must be an object".to_owned())
        })?;
        let frame_index = object
            .get("frameIndex")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest(
                    "Correction frameIndex must be a non-negative integer".to_owned(),
                )
            })?;
        if frame_count == 0 || frame_index as usize >= frame_count {
            return Err(ProjectStoreError::BadRequest(format!(
                "Correction frameIndex {frame_index} is outside the track's {frame_count} frames"
            )));
        }
        let rejected = object
            .get("rejected")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let box_value = match object.get("box") {
            Some(Value::Null) | None => None,
            Some(value) => Some(validate_normalized_box(value)?),
        };
        if box_value.is_none() && !rejected {
            continue;
        }
        let author = object
            .get("author")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("ui")
            .to_owned();
        let source = object
            .get("source")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("manual")
            .to_owned();
        let mut entry = serde_json::Map::new();
        entry.insert("frameIndex".to_owned(), json!(frame_index));
        if let Some(box_value) = box_value {
            entry.insert("box".to_owned(), box_value);
        }
        entry.insert("rejected".to_owned(), Value::Bool(rejected));
        entry.insert("author".to_owned(), Value::String(author));
        entry.insert("source".to_owned(), Value::String(source));
        entry.insert("createdAt".to_owned(), Value::String(created_at.clone()));
        if let Some(slot) = normalized
            .iter_mut()
            .find(|(index, _)| *index == frame_index)
        {
            slot.1 = Value::Object(entry);
        } else {
            normalized.push((frame_index, Value::Object(entry)));
        }
    }
    Ok(normalized.into_iter().map(|(_, value)| value).collect())
}

/// Validate a normalized 0..1 bounding box and return a canonical copy holding
/// only x/y/width/height as finite numbers in range with positive extent.
fn validate_normalized_box(value: &Value) -> ProjectStoreResult<Value> {
    let object = value.as_object().ok_or_else(|| {
        ProjectStoreError::BadRequest("Correction box must be an object".to_owned())
    })?;
    let mut box_out = serde_json::Map::new();
    for key in ["x", "y", "width", "height"] {
        let component = object.get(key).and_then(Value::as_f64).ok_or_else(|| {
            ProjectStoreError::BadRequest(format!("Correction box.{key} must be a number"))
        })?;
        if !component.is_finite() || !(0.0..=1.0).contains(&component) {
            return Err(ProjectStoreError::BadRequest(format!(
                "Correction box.{key} must be between 0 and 1"
            )));
        }
        box_out.insert(key.to_owned(), json!(component));
    }
    let width = box_out.get("width").and_then(Value::as_f64).unwrap_or(0.0);
    let height = box_out.get("height").and_then(Value::as_f64).unwrap_or(0.0);
    if width <= 0.0 || height <= 0.0 {
        return Err(ProjectStoreError::BadRequest(
            "Correction box must have positive width and height".to_owned(),
        ));
    }
    Ok(Value::Object(box_out))
}

fn read_dir_paths(path: &Path) -> ProjectStoreResult<Vec<PathBuf>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()).map_err(Into::into))
        .collect()
}

fn is_safe_track_id(track_id: &str) -> bool {
    !track_id.trim().is_empty()
        && track_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

fn required_str<'a>(payload: &'a Value, key: &str) -> ProjectStoreResult<&'a str> {
    optional_str(payload, key)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("Missing required field: {key}")))
}

fn normalize_asset_tags(tags: &[String]) -> ProjectStoreResult<Vec<String>> {
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::new();
    for tag in tags {
        let value = tag.trim().to_ascii_lowercase();
        if value.is_empty() {
            continue;
        }
        if value.len() > 40 {
            return Err(ProjectStoreError::BadRequest(
                "Asset tags must be 40 characters or fewer".to_owned(),
            ));
        }
        if seen.insert(value.clone()) {
            normalized.push(value);
        }
    }
    if normalized.len() > 24 {
        return Err(ProjectStoreError::BadRequest(
            "Assets can have at most 24 tags".to_owned(),
        ));
    }
    Ok(normalized)
}

fn safe_filename(value: &str, fallback: &str) -> String {
    let name = value.replace('\\', "/");
    let stem = Path::new(name.rsplit('/').next().unwrap_or_default())
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let slug = slugify(stem, fallback, Some(64));
    if slug.is_empty() {
        fallback.to_owned()
    } else {
        slug
    }
}

fn media_type_for_mime(mime_type: &str) -> ProjectStoreResult<&'static str> {
    if mime_type.starts_with("image/") {
        return Ok("image");
    }
    if mime_type.starts_with("video/") {
        return Ok("video");
    }
    Err(ProjectStoreError::BadRequest(
        "Only image and video uploads are supported".to_owned(),
    ))
}

fn guess_mime_from_filename(filename: &str) -> Option<String> {
    mime_guess::from_path(filename)
        .first_raw()
        .map(str::to_owned)
        .or_else(|| {
            match Path::new(filename)
                .extension()
                .and_then(|value| value.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("heic") => Some("image/heic".to_owned()),
                Some("heif") => Some("image/heif".to_owned()),
                Some("avif") => Some("image/avif".to_owned()),
                Some("tif" | "tiff") => Some("image/tiff".to_owned()),
                _ => None,
            }
        })
}

/// Extensions we allow a stored upload to carry, keyed by the media type they belong to. A stored
/// filename's extension re-derives the serve mime in [`ProjectStore::project_file`] via
/// [`guess_mime_from_filename`], and that mime is echoed verbatim into the file-serving endpoint's
/// `Content-Type` header, so an attacker who controls the stored extension controls the served
/// content type on the API origin. This allow-list is the gate: only extensions that map to an
/// inert media type (image/video) are ever stored, so a `video/mp4` upload named `evil.html` can
/// never be stored `.html` and served `text/html` → no content-type confusion / stored XSS
/// (sc-8872). Notably it excludes `.svg` (→ `image/svg+xml`, script-capable) and every
/// document/script extension.
///
/// Every extension here re-derives, through `guess_mime_from_filename`, to a mime that starts with
/// `image/` or `video/` and is NOT `image/svg+xml` — the property the tests pin.
const SAFE_UPLOAD_EXTENSIONS: &[&str] = &[
    // Raster images (SVG deliberately omitted — it is script-capable).
    "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff", "avif", "heic",
    "heif", // Video.
    "mp4", "m4v", "mov", "webm", "mkv", "avi", "ogv", "mpeg", "mpg", "wmv", "flv", "3gp", "3g2",
];

/// True when `extension` (no leading dot, already lowercased) is an allow-listed media extension
/// whose serve mime is an inert `image/`/`video/` type (never `text/html`, `image/svg+xml`, etc.).
fn is_safe_upload_extension(extension: &str) -> bool {
    SAFE_UPLOAD_EXTENSIONS.contains(&extension)
}

/// Pick the stored extension for an upload **without trusting the client filename** (sc-8872).
///
/// The stored extension is what re-derives the serve mime later, so it must reflect the actual
/// media type, not whatever the client named the file. Precedence:
/// 1. The declared/sniffed `mime_type`'s canonical safe extension, when we recognize the mime.
/// 2. Otherwise the client filename's extension, but only if it is on the media allow-list.
/// 3. Otherwise `.bin` (serves `application/octet-stream` — inert), never an attacker-chosen
///    `.html`/`.svg`/`.js`.
///
/// A `video/mp4` upload named `evil.html` therefore stores `.mp4` (rule 1); a genuinely unknown
/// mime with a dangerous extension neutralizes to `.bin` (rule 3).
fn upload_extension(filename: &str, mime_type: &str) -> String {
    // 1. Prefer the extension the declared/sniffed mime canonically maps to, so the stored file's
    //    serve mime round-trips back to the type we actually accepted. mime_guess lists candidates;
    //    take the first one that is on our media allow-list.
    if let Some(extension) = mime_guess::get_mime_extensions_str(mime_type)
        .into_iter()
        .flatten()
        .map(|value| value.to_ascii_lowercase())
        .find(|value| is_safe_upload_extension(value))
    {
        return format!(".{extension}");
    }

    // 2. Fall back to the client extension only when it is itself an allow-listed media extension —
    //    never a document/script extension the client renamed the file to.
    if let Some(extension) = Path::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .filter(|value| is_safe_upload_extension(value))
    {
        return format!(".{extension}");
    }

    // 3. Anything else neutralizes to an inert, non-renderable extension.
    ".bin".to_owned()
}

fn move_or_copy_file(source: &Path, destination: &Path) -> ProjectStoreResult<()> {
    match fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(source, destination)?;
            fs::remove_file(source)?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_project_migrations, backfill_upscale_variant_lineage, build_generated_asset_sidecar,
        connect_project_db, find_timeline_file, guess_mime_from_filename, index_timeline,
        is_safe_relative_path, is_safe_upload_extension, normalize_asset_tags, read_json,
        sniff_image_format, upload_extension, AssetScope, CharacterCreateInput, CharacterLookInput,
        CharacterReferenceInput, ProjectStore, ProjectStoreError, UploadAsset,
        GLOBAL_KEYPOINTS_PROJECT_ID, GLOBAL_POSES_PROJECT_ID, PROJECT_FOLDERS,
        PROJECT_SCHEMA_VERSION, SAFE_UPLOAD_EXTENSIONS,
    };
    use rusqlite::Connection;
    use serde_json::{json, Value};
    use std::sync::Arc;

    /// sc-2022 added `training_datasets.character_id` to the migration without
    /// bumping `PROJECT_SCHEMA_VERSION`, so DBs already stamped at the prior
    /// version (2) hit the early-return gate and never got the column — the
    /// dataset list query failed with "no such column: character_id". This
    /// pins that an old-version DB lacking the column receives it on the next
    /// `apply_project_migrations` and can be queried.
    #[test]
    fn migrations_backfill_training_dataset_character_id_on_old_db() {
        let connection = Connection::open_in_memory().expect("in-memory db");
        // Reproduce the pre-character_id table shape and stamp the DB at the
        // version it would have reached before sc-2022 landed.
        connection
            .execute_batch(
                "
                create table training_datasets (
                  id text primary key,
                  project_id text not null,
                  name text not null,
                  modality text not null,
                  status text not null,
                  version integer not null,
                  item_count integer not null default 0,
                  file_path text not null,
                  created_at text not null,
                  updated_at text not null
                );
                pragma user_version = 2;
                ",
            )
            .expect("seed old schema");

        let has_character_id = |conn: &Connection| {
            let mut statement = conn
                .prepare("pragma table_info(training_datasets)")
                .expect("table_info");
            let columns: Vec<String> = statement
                .query_map([], |row| row.get::<_, String>("name"))
                .expect("columns")
                .filter_map(Result::ok)
                .collect();
            columns.iter().any(|name| name == "character_id")
        };
        assert!(
            !has_character_id(&connection),
            "precondition: old DB lacks character_id"
        );

        apply_project_migrations(&connection).expect("migration runs");

        assert!(
            has_character_id(&connection),
            "migration must add character_id to a DB stamped at the prior version"
        );
        let stamped: i64 = connection
            .query_row("pragma user_version", [], |row| row.get(0))
            .expect("user_version");
        assert_eq!(stamped, PROJECT_SCHEMA_VERSION);
        // The exact list query that surfaced the bug now succeeds.
        connection
            .prepare(
                "select id, project_id, name, modality, status, version, item_count, \
                 created_at, updated_at, file_path, character_id \
                 from training_datasets where project_id = ?1 \
                 order by updated_at desc, name asc",
            )
            .expect("dataset list query prepares against migrated schema");
    }

    /// Deterministic fingerprint of the full project.db schema a fresh
    /// `apply_project_migrations` produces: the stamped `user_version` plus every
    /// user table's columns (name/type/notnull/default/pk in declaration order)
    /// and every user-defined index. Used by the schema-drift guard below.
    fn project_db_schema_fingerprint(connection: &Connection) -> String {
        let mut lines = Vec::new();
        let version: i64 = connection
            .query_row("pragma user_version", [], |row| row.get(0))
            .expect("user_version");
        lines.push(format!("user_version={version}"));

        let mut table_names = connection
            .prepare(
                "select name from sqlite_master where type = 'table' \
                 and name not like 'sqlite_%' order by name",
            )
            .expect("table query");
        let tables: Vec<String> = table_names
            .query_map([], |row| row.get::<_, String>(0))
            .expect("tables")
            .filter_map(Result::ok)
            .collect();
        for table in &tables {
            let mut columns = connection
                .prepare(&format!("pragma table_info({table})"))
                .expect("table_info");
            let column_specs: Vec<String> = columns
                .query_map([], |row| {
                    let name: String = row.get("name")?;
                    let ctype: String = row.get("type")?;
                    let notnull: i64 = row.get("notnull")?;
                    let default: Option<String> = row.get("dflt_value")?;
                    let pk: i64 = row.get("pk")?;
                    Ok(format!(
                        "{name} {ctype} notnull={notnull} default={} pk={pk}",
                        default.as_deref().unwrap_or("NULL")
                    ))
                })
                .expect("columns")
                .filter_map(Result::ok)
                .collect();
            lines.push(format!("table {table}: {}", column_specs.join(", ")));
        }

        let mut index_query = connection
            .prepare(
                "select name, tbl_name from sqlite_master where type = 'index' \
                 and sql is not null and name not like 'sqlite_%' order by name",
            )
            .expect("index query");
        let indexes: Vec<String> = index_query
            .query_map([], |row| {
                Ok(format!(
                    "index {} on {}",
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?
                ))
            })
            .expect("indexes")
            .filter_map(Result::ok)
            .collect();
        lines.extend(indexes);

        lines.join("\n")
    }

    /// The expected project.db schema, version included. Regenerate ONLY
    /// alongside a deliberate schema change — and when you do, you MUST bump
    /// `PROJECT_SCHEMA_VERSION` so the `user_version=` line below changes too.
    const EXPECTED_PROJECT_DB_SCHEMA: &str = concat!(
        "user_version=5\n",
        "table assets: id TEXT notnull=0 default=NULL pk=1, type TEXT notnull=1 default=NULL pk=0, display_name TEXT notnull=1 default=NULL pk=0, file_path TEXT notnull=1 default=NULL pk=0, generation_set_id TEXT notnull=0 default=NULL pk=0, created_at TEXT notnull=1 default=NULL pk=0, favorite INTEGER notnull=1 default=0 pk=0, rating INTEGER notnull=1 default=0 pk=0, rejected INTEGER notnull=1 default=0 pk=0, trashed INTEGER notnull=1 default=0 pk=0, sidecar_path TEXT notnull=0 default=NULL pk=0, origin TEXT notnull=0 default=NULL pk=0\n",
        "table character_looks: id TEXT notnull=0 default=NULL pk=1, character_id TEXT notnull=1 default=NULL pk=0, name TEXT notnull=1 default=NULL pk=0, description TEXT notnull=1 default='' pk=0, approved_reference_ids TEXT notnull=1 default='[]' pk=0, recipe_settings TEXT notnull=1 default='{}' pk=0, created_at TEXT notnull=1 default=NULL pk=0, updated_at TEXT notnull=1 default=NULL pk=0\n",
        "table character_loras: id TEXT notnull=0 default=NULL pk=1, character_id TEXT notnull=1 default=NULL pk=0, lora_id TEXT notnull=0 default=NULL pk=0, name TEXT notnull=1 default=NULL pk=0, source_path TEXT notnull=0 default=NULL pk=0, project_path TEXT notnull=0 default=NULL pk=0, copied_into_project INTEGER notnull=1 default=0 pk=0, category TEXT notnull=1 default='character' pk=0, scope TEXT notnull=1 default='project' pk=0, trigger_words TEXT notnull=1 default='[]' pk=0, default_weight REAL notnull=1 default=1.0 pk=0, compatibility TEXT notnull=1 default='{}' pk=0, created_at TEXT notnull=1 default=NULL pk=0, updated_at TEXT notnull=1 default=NULL pk=0\n",
        "table character_references: character_id TEXT notnull=1 default=NULL pk=1, asset_id TEXT notnull=1 default=NULL pk=2, approved INTEGER notnull=1 default=0 pk=0, role TEXT notnull=1 default='reference' pk=0, notes TEXT notnull=1 default='' pk=0, added_at TEXT notnull=1 default=NULL pk=0, approved_at TEXT notnull=0 default=NULL pk=0\n",
        "table characters: id TEXT notnull=0 default=NULL pk=1, project_id TEXT notnull=1 default=NULL pk=0, name TEXT notnull=1 default=NULL pk=0, type TEXT notnull=1 default=NULL pk=0, description TEXT notnull=1 default='' pk=0, sidecar_path TEXT notnull=1 default=NULL pk=0, created_at TEXT notnull=1 default=NULL pk=0, updated_at TEXT notnull=1 default=NULL pk=0, archived INTEGER notnull=1 default=0 pk=0\n",
        "table generation_sets: id TEXT notnull=0 default=NULL pk=1, mode TEXT notnull=1 default=NULL pk=0, model TEXT notnull=1 default=NULL pk=0, prompt TEXT notnull=1 default=NULL pk=0, created_at TEXT notnull=1 default=NULL pk=0, job_id TEXT notnull=0 default=NULL pk=0\n",
        "table project_metadata: key TEXT notnull=0 default=NULL pk=1, value TEXT notnull=1 default=NULL pk=0\n",
        "table saved_voices: id TEXT notnull=0 default=NULL pk=1, project_id TEXT notnull=1 default=NULL pk=0, name TEXT notnull=1 default=NULL pk=0, reference_audio_asset_id TEXT notnull=1 default=NULL pk=0, embedding TEXT notnull=1 default=NULL pk=0, created_at TEXT notnull=1 default=NULL pk=0\n",
        "table timelines: id TEXT notnull=0 default=NULL pk=1, name TEXT notnull=1 default=NULL pk=0, file_path TEXT notnull=1 default=NULL pk=0, aspect_ratio TEXT notnull=1 default=NULL pk=0, width INTEGER notnull=1 default=NULL pk=0, height INTEGER notnull=1 default=NULL pk=0, fps INTEGER notnull=1 default=NULL pk=0, duration REAL notnull=1 default=0 pk=0, created_at TEXT notnull=1 default=NULL pk=0, updated_at TEXT notnull=1 default=NULL pk=0\n",
        "table training_datasets: id TEXT notnull=0 default=NULL pk=1, project_id TEXT notnull=1 default=NULL pk=0, name TEXT notnull=1 default=NULL pk=0, modality TEXT notnull=1 default=NULL pk=0, status TEXT notnull=1 default=NULL pk=0, version INTEGER notnull=1 default=NULL pk=0, item_count INTEGER notnull=1 default=0 pk=0, character_id TEXT notnull=0 default=NULL pk=0, file_path TEXT notnull=1 default=NULL pk=0, created_at TEXT notnull=1 default=NULL pk=0, updated_at TEXT notnull=1 default=NULL pk=0\n",
        "index idx_training_datasets_project_updated on training_datasets",
    );

    /// Guards the failure mode behind sc-2537: adding a column/table/index to the
    /// migration without bumping `PROJECT_SCHEMA_VERSION` leaves existing DBs
    /// (stamped at the old version) stuck behind the early-return gate, so they
    /// never receive the change and crash on the next query. This pins the full
    /// schema — including the stamped `user_version` — so any schema change makes
    /// the test fail loudly and demand a matching version bump before it can pass.
    #[test]
    fn project_db_schema_matches_snapshot_and_forces_version_bump() {
        let connection = Connection::open_in_memory().expect("in-memory db");
        apply_project_migrations(&connection).expect("migration runs");
        let actual = project_db_schema_fingerprint(&connection);
        assert_eq!(
            actual, EXPECTED_PROJECT_DB_SCHEMA,
            "\n\nproject.db schema drift detected.\n\
             If this change is intentional you MUST:\n  \
             1. bump PROJECT_SCHEMA_VERSION (so existing DBs re-run the migration), and\n  \
             2. replace EXPECTED_PROJECT_DB_SCHEMA with the actual schema below.\n\
             Forgetting step 1 is exactly the sc-2537 bug.\n\n\
             ----- actual schema -----\n{actual}\n-------------------------\n"
        );
    }

    #[test]
    fn normalize_asset_tags_trims_lowercases_and_deduplicates() {
        let tags = normalize_asset_tags(&[
            " Portrait ".to_owned(),
            "portrait".to_owned(),
            "Reference".to_owned(),
            "".to_owned(),
        ])
        .expect("normalized tags");

        assert_eq!(tags, vec!["portrait", "reference"]);
    }

    #[test]
    fn build_generated_asset_sidecar_assembles_the_contract_schema() {
        // sc-1656: Rust owns the generated-asset sidecar schema now. Pin the
        // envelope it assembles from worker facts to the shared contract
        // (resource_sidecars.json imageAsset + recipe required keys) and the
        // load-bearing field mappings.
        let fact = json!({
            "assetId": "asset_abc",
            "mediaPath": "assets/images/genset_x/2026-05-25_z_image_turbo_city_0001.png",
            "mimeType": "image/png",
            "width": 1280,
            "height": 768,
            "normalizedWidth": 1024,
            "normalizedHeight": 768,
            "count": 1,
            "family": "z-image",
            "seed": 42,
            "index": 0,
            "displayName": "city #1",
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "city",
            "negativePrompt": "",
            "loras": [],
            "stylePreset": "none",
            "characterId": "character-1",
            "characterLookId": "look-1",
            "sourceAssetId": Value::Null,
            "rawAdapterSettings": {"steps": 8},
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_x", &fact);

        for key in [
            "schemaVersion",
            "id",
            "projectId",
            "generationSetId",
            "type",
            "displayName",
            "createdAt",
            "file",
            "status",
            "recipe",
            "lineage",
        ] {
            assert!(asset.get(key).is_some(), "missing top-level key {key}");
        }
        for key in [
            "mode",
            "model",
            "adapter",
            "prompt",
            "negativePrompt",
            "seed",
            "loras",
            "normalizedSettings",
            "rawAdapterSettings",
        ] {
            assert!(
                asset["recipe"].get(key).is_some(),
                "missing recipe key {key}"
            );
        }
        assert_eq!(asset["type"], json!("image"));
        assert_eq!(asset["id"], json!("asset_abc"));
        assert_eq!(asset["projectId"], json!("project-1"));
        assert_eq!(asset["generationSetId"], json!("genset_x"));
        assert_eq!(asset["file"]["path"], fact["mediaPath"]);
        assert_eq!(asset["file"]["width"], json!(1280));
        assert_eq!(asset["recipe"]["adapter"], json!("z_image_diffusers"));
        assert_eq!(asset["recipe"]["seed"], json!(42));
        assert_eq!(asset["recipe"]["normalizedSettings"]["width"], json!(1024));
        assert_eq!(
            asset["recipe"]["normalizedSettings"]["family"],
            json!("z-image")
        );
        // Character metadata propagates into normalizedSettings but conditioning
        // stays inactive (formerly pinned Python-side by
        // test_character_image_recipe_marks_conditioning_inactive).
        assert_eq!(
            asset["recipe"]["normalizedSettings"]["characterId"],
            json!("character-1")
        );
        assert_eq!(
            asset["recipe"]["normalizedSettings"]["characterLookId"],
            json!("look-1")
        );
        assert_eq!(
            asset["recipe"]["normalizedSettings"]["characterConditioningActive"],
            json!(false)
        );
        assert_eq!(asset["lineage"]["jobId"], json!("job-1"));
    }

    /// sc-4408: a scored generation's `rawAdapterSettings.faceLikeness` block (attached worker-side by
    /// `attach_face_likeness`) survives verbatim through `build_image_sidecar_parts` into
    /// `recipe.rawAdapterSettings` — the persistence pathway the scoring surfaces (4409/4410/4411) rely
    /// on. The omit-when-absent contract is covered too: no `faceLikeness` key in ⇒ none out.
    #[test]
    fn build_image_sidecar_carries_face_likeness_when_present_and_omits_when_absent() {
        let base_fact = |raw: Value| {
            json!({
                "assetId": "asset_fl",
                "mediaPath": "assets/images/genset_x/img_0001.png",
                "mimeType": "image/png",
                "width": 1024,
                "height": 1024,
                "mode": "character_image",
                "model": "z_image_turbo",
                "adapter": "z_image_diffusers",
                "prompt": "portrait",
                "loras": [],
                "rawAdapterSettings": raw,
            })
        };

        // Present: the detected:true block (and a detected:false N/A block) passes through verbatim.
        let scored = json!({
            "faceLikeness": {
                "score": 0.873,
                "detected": true,
                "method": "arcface_antelopev2",
                "sourceAssetId": "asset_source_fixture",
            }
        });
        let asset = build_generated_asset_sidecar(
            "project-1",
            "job-1",
            "genset_x",
            &base_fact(scored.clone()),
        );
        assert_eq!(asset["recipe"]["rawAdapterSettings"], scored);
        assert_eq!(
            asset["recipe"]["rawAdapterSettings"]["faceLikeness"]["detected"],
            json!(true)
        );

        let na = json!({
            "faceLikeness": {
                "score": Value::Null,
                "detected": false,
                "method": "arcface_antelopev2",
                "sourceAssetId": "asset_source_fixture",
                "reason": "no_face",
            }
        });
        let asset_na =
            build_generated_asset_sidecar("project-1", "job-1", "genset_x", &base_fact(na.clone()));
        assert_eq!(asset_na["recipe"]["rawAdapterSettings"], na);

        // Absent: no faceLikeness key in rawAdapterSettings ⇒ none in the sidecar (clean, no crash).
        let asset_absent =
            build_generated_asset_sidecar("project-1", "job-1", "genset_x", &base_fact(json!({})));
        assert_eq!(asset_absent["recipe"]["rawAdapterSettings"], json!({}));
        assert!(
            asset_absent["recipe"]["rawAdapterSettings"]
                .get("faceLikeness")
                .is_none(),
            "absent score ⇒ faceLikeness omitted from sidecar"
        );
    }

    /// sc-5721 (CORE-002→F): `persist_generated_asset` joins the worker-supplied
    /// `mediaPath` into the project tree and `create_dir_all`/`write`s a sidecar there,
    /// so a traversal or absolute path must be rejected before any filesystem write —
    /// the worker→API boundary must not write outside the project root. A normal
    /// relative path is still accepted.
    #[test]
    fn persist_generated_asset_rejects_unsafe_media_path() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Boundary").expect("project creates");

        let fact_with = |media_path: &str| {
            json!({
                "assetId": "asset_safe1",
                "mediaPath": media_path,
                "mimeType": "image/png",
                "width": 64,
                "height": 64,
                "model": "z_image_turbo",
                "adapter": "z_image_diffusers",
                "prompt": "x",
                "loras": [],
            })
        };

        for unsafe_path in [
            "../../../../tmp/escape.png",
            "/etc/passwd",
            "assets/../../escape.png",
        ] {
            let result = store.persist_generated_asset(
                &project.id,
                "job-1",
                "genset_x",
                &fact_with(unsafe_path),
            );
            assert!(
                matches!(result, Err(ProjectStoreError::BadRequest(_))),
                "expected {unsafe_path:?} to be rejected, got {result:?}"
            );
        }

        // A normal project-relative path is still accepted and written under the project.
        let safe_fact = json!({
            "assetId": "asset_safe1",
            "mediaPath": "assets/images/genset_x/asset_safe1.png",
            "mimeType": "image/png",
            "width": 64,
            "height": 64,
            "normalizedWidth": 64,
            "normalizedHeight": 64,
            "count": 1,
            "family": "z-image",
            "seed": 1,
            "index": 0,
            "displayName": "safe #1",
            "createdAt": "2026-06-15T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "x",
            "negativePrompt": "",
            "loras": [],
            "stylePreset": "none",
            "rawAdapterSettings": {"steps": 8},
        });
        let safe = store
            .persist_generated_asset(&project.id, "job-1", "genset_x", &safe_fact)
            .expect("safe media path persists");
        assert_eq!(safe["id"], json!("asset_safe1"));
    }

    /// sc-8871 (F-069): `save_timeline` derives the write path from the
    /// client-supplied `id` (last 8 chars slugged into the filename) and
    /// `relative_string`'s lexical strip lets an unnormalized `..`-bearing id
    /// through, so — matching the asset/character/track/dataset guards — the id
    /// must be charset-checked before any path is derived. A traversal or
    /// separator-bearing id is rejected with `BadRequest` and writes nothing;
    /// a normal id still saves under the project's `timelines/` dir.
    #[test]
    fn save_timeline_rejects_unsafe_id_before_deriving_path() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Boundary").expect("project creates");
        let timelines_dir = std::path::Path::new(&project.path).join("timelines");

        let timeline_with = |id: &str| {
            json!({
                "id": id,
                "projectId": project.id,
                "name": "My Timeline",
            })
        };

        for unsafe_id in [
            "../../../../tmp/escape",
            "timeline/../../escape",
            "a/b/c",
            "id.with.dots",
            "  ",
            "",
        ] {
            let result = store.save_timeline(&project.id, timeline_with(unsafe_id));
            assert!(
                matches!(result, Err(ProjectStoreError::BadRequest(_))),
                "expected id {unsafe_id:?} to be rejected, got {result:?}"
            );
        }

        // Nothing should have been written for any rejected id — no file can land
        // inside (or escape) the timelines dir.
        let timeline_files: Vec<_> = std::fs::read_dir(&timelines_dir)
            .map(|entries| entries.filter_map(Result::ok).collect())
            .unwrap_or_default();
        assert!(
            timeline_files.is_empty(),
            "rejected ids must not write any timeline file, found {timeline_files:?}"
        );
        // And the traversal target above the project root must not exist either.
        assert!(
            !temp_dir
                .path()
                .join("escape.sceneworks.timeline.json")
                .exists(),
            "traversal id must not write outside the project dir"
        );

        // A normal safe id still saves correctly under timelines/.
        let saved = store
            .save_timeline(&project.id, timeline_with("timeline_abc123"))
            .expect("safe timeline id persists");
        assert_eq!(saved["id"], json!("timeline_abc123"));
        let written: Vec<_> = std::fs::read_dir(&timelines_dir)
            .expect("timelines dir exists after save")
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
            })
            .collect();
        assert_eq!(
            written.len(),
            1,
            "exactly one timeline file should be written for the safe id, found {written:?}"
        );
    }

    #[test]
    fn write_generation_set_embeds_replayable_recipe_from_first_asset_fact() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Fixture").expect("project creates");
        let generation_set = json!({
            "id": "genset_recipe",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "city",
            "negativePrompt": "",
            "count": 1,
            "createdAt": "2026-05-25T00:00:00Z",
        });
        let fact = json!({
            "assetId": "asset_abc",
            "mediaPath": "assets/images/genset_recipe/city.png",
            "mimeType": "image/png",
            "width": 1280,
            "height": 768,
            "normalizedWidth": 1024,
            "normalizedHeight": 768,
            "count": 1,
            "family": "z-image",
            "seed": 42,
            "displayName": "city #1",
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "city",
            "negativePrompt": "",
            "loras": [{"id": "style_lora", "weight": 0.8}],
            "stylePreset": "none",
            "rawAdapterSettings": {"steps": 8},
        });

        store
            .write_generation_set(&project.id, "job-1", &generation_set, Some(&fact))
            .expect("generation set writes");

        let written: Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::path::Path::new(&project.path).join("generation-sets/genset_recipe.json"),
            )
            .expect("generation set file exists"),
        )
        .expect("generation set json parses");
        assert_eq!(written["recipe"]["adapter"], json!("z_image_diffusers"));
        assert_eq!(written["recipe"]["seed"], json!(42));
        assert_eq!(written["recipe"]["loras"][0]["id"], json!("style_lora"));
        assert_eq!(
            written["recipe"]["normalizedSettings"]["width"],
            json!(1024)
        );
        assert_eq!(written["recipe"]["rawAdapterSettings"]["steps"], json!(8));
    }

    /// sc-4270 / F-CORE-10: list_assets resolves the embedded `generationSet`
    /// through a per-call cache (read once per set, not once per asset). Two
    /// assets sharing a set must each still carry the full, correct set — proving
    /// the cache doesn't drop or corrupt the shared value, and that dedup keeps
    /// both distinct assets.
    #[test]
    fn list_assets_embeds_shared_generation_set_for_each_asset() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Gallery").expect("project creates");

        let generation_set = json!({
            "id": "genset_shared",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "city",
            "count": 2,
            "createdAt": "2026-05-25T00:00:00Z",
        });
        let make_fact = |asset_id: &str, name: &str| {
            json!({
                "assetId": asset_id,
                "mediaPath": format!("assets/images/genset_shared/{asset_id}.png"),
                "mimeType": "image/png",
                "displayName": name,
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "adapter": "z_image_diffusers",
                "prompt": "city",
            })
        };

        store
            .write_generation_set(
                &project.id,
                "job-1",
                &generation_set,
                Some(&make_fact("asset_one", "city #1")),
            )
            .expect("generation set writes");
        store
            .persist_generated_asset(
                &project.id,
                "job-1",
                "genset_shared",
                &make_fact("asset_one", "city #1"),
            )
            .expect("asset one persists");
        store
            .persist_generated_asset(
                &project.id,
                "job-1",
                "genset_shared",
                &make_fact("asset_two", "city #2"),
            )
            .expect("asset two persists");

        let assets = store
            .list_assets(&project.id, false, false, AssetScope::All)
            .expect("list");
        assert_eq!(assets.len(), 2, "both distinct assets are listed (deduped)");
        for asset in &assets {
            assert_eq!(
                asset["generationSet"]["id"],
                json!("genset_shared"),
                "each asset embeds the shared generation set"
            );
        }
    }

    #[test]
    fn move_asset_to_library_promotes_origin_and_detaches_character() {
        use crate::store_util::{read_json, write_json};
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Gallery").expect("project creates");

        // A Character Studio output: persisted, then marked up to look like a
        // character-scoped asset (origin character_studio + recipe characterId +
        // characterReferences) — the state the Library excludes and the Character
        // Assets grid keys on.
        let fact = json!({
            "assetId": "char_shot",
            "mediaPath": "assets/images/genset_char/char_shot.png",
            "mimeType": "image/png",
            "displayName": "Mira test #1",
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "character_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "portrait",
        });
        store
            .persist_generated_asset(&project.id, "job-1", "genset_char", &fact)
            .expect("asset persists");

        let project_path = store.find_project_path(&project.id).expect("project path");
        let sidecar = project_path.join("assets/images/genset_char/char_shot.sceneworks.json");
        {
            let mut asset = read_json(&sidecar).expect("read sidecar");
            let object = asset.as_object_mut().expect("asset object");
            object.insert("origin".to_owned(), json!("character_studio"));
            let recipe = object
                .entry("recipe")
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .expect("recipe object");
            recipe
                .entry("normalizedSettings")
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .expect("normalizedSettings object")
                .insert("characterId".to_owned(), json!("char-1"));
            object
                .entry("metadata")
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .expect("metadata object")
                .insert(
                    "characterReferences".to_owned(),
                    json!([{ "characterId": "char-1" }]),
                );
            write_json(&sidecar, &asset).expect("write sidecar");
        }
        store
            .index_asset_sidecar(&project.id, &sidecar)
            .expect("reindex");

        // Precondition: excluded from the Library by origin.
        let before = store
            .get_asset(&project.id, "char_shot")
            .expect("get before");
        assert_eq!(before["origin"], json!("character_studio"));

        // sc-10205: the endpoint returns the moved group (requested asset first).
        let moved_group = store
            .move_asset_to_library(&project.id, "char_shot")
            .expect("move to library");
        let moved = &moved_group[0];

        // Origin promoted (image media -> image_studio), so the Library no longer excludes it.
        assert_eq!(moved["origin"], json!("image_studio"));
        // Character association dropped on both vectors the grid filters on.
        assert!(
            moved["recipe"]["normalizedSettings"]
                .get("characterId")
                .is_none(),
            "recipe characterId is cleared"
        );
        assert!(
            moved
                .get("metadata")
                .and_then(|metadata| metadata.get("characterReferences"))
                .map(Value::is_null)
                .unwrap_or(true),
            "characterReferences is cleared"
        );

        // Persisted: a fresh read reflects the move.
        let after = store
            .get_asset(&project.id, "char_shot")
            .expect("get after");
        assert_eq!(after["origin"], json!("image_studio"));
    }

    /// sc-10200: the true-move twin of `move_asset_to_library`. A Library asset
    /// moved into a character must leave the Library (origin flip), anchor on the
    /// target via `metadata.characterReferences` ONLY (no curated `references[]`
    /// entry — the "Approved set" is hand-curated), detach from every other
    /// character, and survive a curated reference add/remove cycle on the target.
    #[test]
    fn move_asset_to_character_flips_origin_and_anchors_without_curated_reference() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Gallery").expect("project creates");

        // A plain Image Studio output: origin image_studio, library-visible.
        let fact = json!({
            "assetId": "lib_shot",
            "mediaPath": "assets/images/genset_lib/lib_shot.png",
            "mimeType": "image/png",
            "displayName": "Plate #1",
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "plate",
        });
        store
            .persist_generated_asset(&project.id, "job-1", "genset_lib", &fact)
            .expect("asset persists");

        let mira = store
            .create_character(
                &project.id,
                CharacterCreateInput {
                    name: "Mira".to_owned(),
                    character_type: "person".to_owned(),
                    description: String::new(),
                },
            )
            .expect("Mira creates");
        let dax = store
            .create_character(
                &project.id,
                CharacterCreateInput {
                    name: "Dax".to_owned(),
                    character_type: "person".to_owned(),
                    description: String::new(),
                },
            )
            .expect("Dax creates");
        let mira_id = mira["id"].as_str().expect("Mira id").to_owned();
        let dax_id = dax["id"].as_str().expect("Dax id").to_owned();

        // Pre-existing curated link on Mira (the legacy "Move" behavior): the move
        // must clean this up rather than leave a dual membership behind.
        store
            .add_character_reference(
                &project.id,
                &mira_id,
                CharacterReferenceInput {
                    asset_id: "lib_shot".to_owned(),
                    approved: false,
                    role: "asset".to_owned(),
                    notes: String::new(),
                },
            )
            .expect("Mira link");

        // An unknown target must reject before mutating anything.
        assert!(store
            .move_asset_to_character(&project.id, "lib_shot", "char_missing")
            .is_err());

        // sc-10205: the endpoint returns the moved group (requested asset first).
        let moved_group = store
            .move_asset_to_character(&project.id, "lib_shot", &dax_id)
            .expect("move to Dax");
        let moved = &moved_group[0];

        // Origin flipped, so the Library allow-list now excludes the asset.
        assert_eq!(moved["origin"], json!("character_studio"));
        // Anchored on Dax via metadata only, tagged as a move (not a sidecar mirror).
        let links = moved["metadata"]["characterReferences"]
            .as_array()
            .expect("anchor links");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0]["characterId"], json!(dax_id));
        assert_eq!(links[0]["source"], json!("library-move"));
        // Neither character carries a curated references[] entry: Mira's stale link
        // is gone and Dax's Approved set stays hand-curated.
        let mira_after = store
            .get_character(&project.id, &mira_id)
            .expect("Mira after");
        assert_eq!(mira_after["references"].as_array().map(Vec::len), Some(0));
        let dax_after = store
            .get_character(&project.id, &dax_id)
            .expect("Dax after");
        assert_eq!(dax_after["references"].as_array().map(Vec::len), Some(0));

        // A curated reference add/remove cycle on the target must not orphan the
        // asset: the sidecar mirror rebuild only manages its own entries.
        store
            .add_character_reference(
                &project.id,
                &dax_id,
                CharacterReferenceInput {
                    asset_id: "lib_shot".to_owned(),
                    approved: true,
                    role: "reference".to_owned(),
                    notes: String::new(),
                },
            )
            .expect("Dax curated link");
        store
            .remove_character_reference(&project.id, &dax_id, "lib_shot")
            .expect("Dax curated unlink");
        let cycled = store
            .get_asset(&project.id, "lib_shot")
            .expect("get cycled");
        let cycled_links = cycled["metadata"]["characterReferences"]
            .as_array()
            .expect("anchor survives");
        assert_eq!(cycled_links.len(), 1);
        assert_eq!(cycled_links[0]["source"], json!("library-move"));

        // Reversible: move-to-library restores a library origin and drops the anchor.
        let back_group = store
            .move_asset_to_library(&project.id, "lib_shot")
            .expect("move back");
        let back = &back_group[0];
        assert_eq!(back["origin"], json!("image_studio"));
        assert!(back["metadata"]["characterReferences"].is_null());
    }

    /// sc-10205: a linked original/upscaled pair renders as ONE folded Library
    /// tile (the upscaled child is the visible asset), so moving the visible
    /// asset must carry the whole fold group — in both directions. Splitting the
    /// pair strands the hidden mate in the source collection.
    #[test]
    fn move_asset_carries_the_upscale_fold_group_both_directions() {
        use crate::store_util::{read_json, write_json};
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Gallery").expect("project creates");

        for (asset_id, path) in [
            ("base_shot", "assets/images/genset_pair/base_shot.png"),
            ("up_shot", "assets/images/genset_pair/base_shot_up2x.png"),
        ] {
            let fact = json!({
                "assetId": asset_id,
                "mediaPath": path,
                "mimeType": "image/png",
                "displayName": asset_id,
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "adapter": "z_image_diffusers",
                "prompt": "plate",
            });
            store
                .persist_generated_asset(&project.id, "job-1", "genset_pair", &fact)
                .expect("asset persists");
        }

        // Stamp the fold lineage the Library reads (sc-10117 keys) on the child.
        let project_path = store.find_project_path(&project.id).expect("project path");
        let child_sidecar =
            project_path.join("assets/images/genset_pair/base_shot_up2x.sceneworks.json");
        {
            let mut asset = read_json(&child_sidecar).expect("read child");
            super::apply_upscale_variant_link(&mut asset, "base_shot", 2, "seedvr2");
            write_json(&child_sidecar, &asset).expect("write child");
        }
        store
            .index_asset_sidecar(&project.id, &child_sidecar)
            .expect("reindex child");

        let character = store
            .create_character(
                &project.id,
                CharacterCreateInput {
                    name: "Mira".to_owned(),
                    character_type: "person".to_owned(),
                    description: String::new(),
                },
            )
            .expect("character creates");
        let character_id = character["id"].as_str().expect("character id").to_owned();

        // Move the VISIBLE folded asset (the upscaled child): both must move.
        let moved = store
            .move_asset_to_character(&project.id, "up_shot", &character_id)
            .expect("move pair to character");
        let moved = moved.as_array().expect("moved group");
        assert_eq!(moved.len(), 2, "the whole fold group moves");
        assert_eq!(moved[0]["id"], json!("up_shot"), "requested asset first");
        for asset in moved {
            assert_eq!(asset["origin"], json!("character_studio"));
            assert_eq!(
                asset["metadata"]["characterReferences"][0]["characterId"],
                json!(character_id)
            );
        }
        let base_after = store
            .get_asset(&project.id, "base_shot")
            .expect("base after");
        assert_eq!(
            base_after["origin"],
            json!("character_studio"),
            "the hidden fold-mate left the Library too"
        );

        // And back: moving the child to the Library brings the original along.
        let back = store
            .move_asset_to_library(&project.id, "up_shot")
            .expect("move pair back");
        let back = back.as_array().expect("back group");
        assert_eq!(back.len(), 2);
        for asset in back {
            assert_eq!(asset["origin"], json!("image_studio"));
        }
    }

    /// V-4: a pre-migration project surfaces an EMPTY `assets` table even though
    /// its assets are still on disk as `.sceneworks.json` sidecars (these DBs
    /// predate the asset index / `sidecar_path` column and were never reindexed).
    /// `list_assets` must auto-reindex from the on-disk sidecars on first open so
    /// the library is not silently empty. This reproduces that state by wiping the
    /// index rows (leaving the sidecar on disk) and asserts the asset comes back.
    #[test]
    fn list_assets_auto_reindexes_pre_migration_project_with_sidecars_on_disk() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store
            .create_project("Pre-migration")
            .expect("project creates");

        let fact = json!({
            "assetId": "asset_legacy",
            "mediaPath": "assets/images/genset_legacy/asset_legacy.png",
            "mimeType": "image/png",
            "displayName": "legacy #1",
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "old",
        });
        store
            .persist_generated_asset(&project.id, "job-1", "genset_legacy", &fact)
            .expect("asset persists (media + sidecar + index row)");

        // Simulate the pre-migration state: the sidecar is on disk but the index
        // table is empty (as it would be for a DB created before the asset index
        // existed and never reindexed).
        let project_path = store.find_project_path(&project.id).expect("project path");
        {
            let connection = connect_project_db(&project_path).expect("open db");
            connection
                .execute("delete from assets", [])
                .expect("clear index");
            let remaining: i64 = connection
                .query_row("select count(*) from assets", [], |row| row.get(0))
                .expect("count");
            assert_eq!(remaining, 0, "precondition: index table is empty");
        }
        assert!(
            project_path
                .join("assets/images/genset_legacy/asset_legacy.sceneworks.json")
                .exists(),
            "precondition: the asset sidecar is still on disk"
        );

        // First open: list_assets must rebuild the index from the sidecar.
        let assets = store
            .list_assets(&project.id, false, false, AssetScope::All)
            .expect("list auto-reindexes instead of returning empty");
        assert_eq!(
            assets.len(),
            1,
            "the on-disk asset is recovered via auto-reindex, not silently dropped"
        );
        assert_eq!(assets[0]["id"], json!("asset_legacy"));

        // Idempotent: the index is now populated, so a second open does not depend
        // on (or re-run) the reindex and still returns the asset.
        let again = store
            .list_assets(&project.id, false, false, AssetScope::All)
            .expect("second list");
        assert_eq!(again.len(), 1, "result is stable on subsequent opens");
    }

    /// V-4 guard: a genuinely empty project (no sidecars on disk) must NOT trigger
    /// a reindex and must return an empty list cleanly rather than erroring.
    #[test]
    fn list_assets_returns_empty_for_truly_empty_project_without_reindex_loop() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Empty").expect("project creates");

        let assets = store
            .list_assets(&project.id, false, false, AssetScope::All)
            .expect("empty project lists cleanly");
        assert!(
            assets.is_empty(),
            "no sidecars on disk => empty library, no spurious assets"
        );
    }

    /// Fact for a generated image asset in `genset`. `persist_generated_asset`
    /// writes the sidecar + index row but NOT the media file (the worker writes
    /// media), so a test controls "media present" by writing the `.png` itself.
    fn orphan_test_fact(asset_id: &str) -> Value {
        json!({
            "assetId": asset_id,
            "mediaPath": format!("assets/images/genset/{asset_id}.png"),
            "mimeType": "image/png",
            "displayName": asset_id,
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "x",
        })
    }

    #[test]
    fn prune_orphaned_assets_removes_only_rows_whose_media_is_missing() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Prune").expect("project creates");
        let project_path = store.find_project_path(&project.id).expect("project path");

        store
            .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("keep"))
            .expect("keep persists");
        store
            .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("gone"))
            .expect("gone persists");
        // `keep` has its media on disk; `gone` does not (its media was purged).
        std::fs::write(
            project_path.join("assets/images/genset/keep.png"),
            b"png-bytes",
        )
        .expect("write keep media");

        // Precondition: both rows are indexed and listed (list_assets checks the
        // sidecar, not the media, so it still serves the media-less `gone`).
        let before = store
            .list_assets(&project.id, true, true, AssetScope::All)
            .expect("list before prune");
        assert_eq!(before.len(), 2, "both assets listed before prune");

        let pruned = store
            .prune_orphaned_assets(&project.id)
            .expect("prune succeeds");
        assert_eq!(pruned, 1, "only the media-less row is pruned");

        let after = store
            .list_assets(&project.id, true, true, AssetScope::All)
            .expect("list after prune");
        assert_eq!(after.len(), 1, "the purged asset is no longer served");
        assert_eq!(after[0]["id"], json!("keep"));

        // Idempotent: a second pass finds nothing to prune.
        assert_eq!(
            store
                .prune_orphaned_assets(&project.id)
                .expect("second prune"),
            0
        );

        // The orphaned sidecar is moved OUT of the scanned asset tree into the
        // `.orphaned-sidecars/` quarantine (declutter + no resurrection), but retained
        // there (moved, not deleted) so the recipe metadata stays recoverable.
        assert!(
            !project_path
                .join("assets/images/genset/gone.sceneworks.json")
                .exists(),
            "orphaned sidecar is removed from the scanned asset folder"
        );
        assert!(
            project_path
                .join(".orphaned-sidecars/gone.sceneworks.json")
                .exists(),
            "orphaned sidecar is retained in the quarantine"
        );
    }

    #[test]
    fn prune_all_orphaned_assets_sweeps_every_registered_project() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        // Two projects, each with one asset whose media was never written (purged).
        for name in ["Alpha", "Beta"] {
            let project = store.create_project(name).expect("project creates");
            store
                .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("gone"))
                .expect("gone persists");
        }

        let total = store
            .prune_all_orphaned_assets()
            .expect("sweep succeeds across projects");
        assert_eq!(total, 2, "one orphan pruned from each project");

        // A second sweep is a no-op.
        assert_eq!(store.prune_all_orphaned_assets().expect("second sweep"), 0);
    }

    #[test]
    fn prune_quarantines_sidecar_so_empty_table_reindex_does_not_resurrect() {
        // A project whose ONLY asset is orphaned: after pruning, the assets table is
        // empty, so `list_assets`' auto-reindex (fires only when the table is empty)
        // would rebuild the row from the sidecar — UNLESS the sidecar has been moved
        // out of the scanned tree. This is the resurrection edge sc-10467 closes.
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("AllOrphans").expect("project creates");
        store
            .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("gone"))
            .expect("gone persists");

        assert_eq!(
            store.prune_orphaned_assets(&project.id).expect("prune"),
            1,
            "the sole (media-less) asset is pruned, emptying the table"
        );

        // The empty table would trigger auto-reindex here; the quarantined sidecar is
        // no longer scannable, so nothing is rebuilt.
        let assets = store
            .list_assets(&project.id, true, true, AssetScope::All)
            .expect("list after prune");
        assert!(
            assets.is_empty(),
            "orphan is not resurrected via the empty-table auto-reindex"
        );
    }

    #[test]
    fn reindex_project_does_not_resurrect_orphaned_assets() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Reindex").expect("project creates");
        let project_path = store.find_project_path(&project.id).expect("project path");

        store
            .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("present"))
            .expect("present persists");
        store
            .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("orphan"))
            .expect("orphan persists");
        // Only `present` has media on disk; both sidecars exist (so a rebuild would
        // re-add both rows without the post-reindex prune).
        std::fs::write(
            project_path.join("assets/images/genset/present.png"),
            b"png-bytes",
        )
        .expect("write present media");

        let counts = store.reindex_project(&project.id).expect("reindex works");
        assert_eq!(
            counts.assets, 1,
            "reindex re-adds both sidecars then prunes the media-less one"
        );

        let assets = store
            .list_assets(&project.id, true, true, AssetScope::All)
            .expect("list after reindex");
        assert_eq!(assets.len(), 1, "the orphan is not resurrected by reindex");
        assert_eq!(assets[0]["id"], json!("present"));
    }

    #[test]
    fn list_assets_advertises_poster_url_only_when_the_poster_exists() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Videos").expect("project creates");
        let project_path = store.find_project_path(&project.id).expect("project path");

        let video_fact = |id: &str| {
            json!({
                "assetId": id,
                "mediaPath": format!("assets/videos/genset/{id}.mp4"),
                "mimeType": "video/mp4",
                "type": "video",
                "displayName": id,
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "image_to_video",
                "model": "wan",
                "adapter": "wan",
                "prompt": "x",
            })
        };
        store
            .persist_generated_asset(&project.id, "job-1", "genset", &video_fact("withposter"))
            .expect("withposter persists");
        store
            .persist_generated_asset(&project.id, "job-1", "genset", &video_fact("noposter"))
            .expect("noposter persists");
        // Only `withposter` has its `<name>.poster.jpg` sibling on disk.
        std::fs::write(
            project_path.join("assets/videos/genset/withposter.poster.jpg"),
            b"jpg-bytes",
        )
        .expect("write poster");

        let assets = store
            .list_assets(&project.id, true, true, AssetScope::All)
            .expect("list");
        let by_id = |id: &str| {
            assets
                .iter()
                .find(|asset| asset["id"] == id)
                .unwrap_or_else(|| panic!("asset {id} present"))
        };
        assert_eq!(
            by_id("withposter")["posterUrl"],
            json!(format!(
                "/api/v1/projects/{}/files/assets/videos/genset/withposter.poster.jpg",
                project.id
            )),
            "posterUrl is advertised when the poster file exists"
        );
        assert!(
            by_id("noposter").get("posterUrl").is_none(),
            "no posterUrl is advertised when the poster file is absent"
        );
    }

    #[test]
    fn build_generated_asset_sidecar_derives_video_type_from_mime() {
        let fact = json!({
            "assetId": "asset_v",
            "mediaPath": "assets/videos/genset_y/clip.mp4",
            "mimeType": "video/mp4",
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_y", &fact);
        assert_eq!(asset["type"], json!("video"));
        assert_eq!(asset["file"]["mimeType"], json!("video/mp4"));
    }

    /// sc-13404: an audio fact assembles a `type: "audio"` asset whose `file` block carries the
    /// measured duration/sampleRate/channels (no width/height/fps) and whose recipe round-trips the
    /// requested voice/language. Mirrors the video-type derivation test.
    #[test]
    fn build_generated_asset_sidecar_assembles_audio() {
        let fact = json!({
            "type": "audio",
            "assetId": "asset_a",
            "mediaPath": "assets/audios/genset_z/hello.wav",
            "mimeType": "audio/wav",
            "duration": 3.25,
            "sampleRate": 24_000,
            "channels": 1,
            "family": "kokoro",
            "displayName": "Hello from SceneWorks audio.",
            "model": "kokoro_82m",
            "adapter": "kokoro",
            "prompt": "Hello from SceneWorks audio.",
            "voice": "af_heart",
            "language": "en-US",
            "targetDurationSecs": 3.0,
            "seed": 42,
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_z", &fact);
        assert_eq!(asset["type"], json!("audio"));
        // Origin is derived to the Audio Studio, not the image_studio catch-all.
        assert_eq!(asset["origin"], json!("audio_studio"));
        // File block: measured audio facts, no width/height/fps.
        assert_eq!(asset["file"]["mimeType"], json!("audio/wav"));
        assert_eq!(asset["file"]["duration"], json!(3.25));
        assert_eq!(asset["file"]["sampleRate"], json!(24_000));
        assert_eq!(asset["file"]["channels"], json!(1));
        assert_eq!(asset["file"]["width"], Value::Null);
        assert_eq!(asset["file"]["fps"], Value::Null);
        // Recipe replays the requested knobs.
        assert_eq!(asset["recipe"]["model"], json!("kokoro_82m"));
        assert_eq!(
            asset["recipe"]["normalizedSettings"]["voice"],
            json!("af_heart")
        );
        assert_eq!(
            asset["recipe"]["normalizedSettings"]["language"],
            json!("en-US")
        );
        assert_eq!(
            asset["recipe"]["normalizedSettings"]["targetDurationSecs"],
            json!(3.0)
        );
        assert_eq!(asset["lineage"]["jobId"], json!("job-1"));
    }

    #[test]
    fn build_generated_asset_sidecar_preserves_variant_lineage_and_extra() {
        let fact = json!({
            "assetId": "asset_upscaled",
            "mediaPath": "assets/images/genset_x/city_0001_upscaled_x2.png",
            "mimeType": "image/png",
            "sourceAssetId": "asset_original",
            "parents": ["asset_original"],
            "extra": {
                "isUpscaled": true,
                "upscaledFromAssetId": "asset_original",
                "factor": 2,
                "engine": "real-esrgan",
            },
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_x", &fact);

        assert_eq!(asset["lineage"]["sourceAssetId"], json!("asset_original"));
        assert_eq!(asset["lineage"]["parents"], json!(["asset_original"]));
        assert_eq!(asset["extra"]["isUpscaled"], json!(true));
        assert_eq!(
            asset["extra"]["upscaledFromAssetId"],
            json!("asset_original")
        );
        assert_eq!(asset["extra"]["factor"], json!(2));
        assert_eq!(asset["extra"]["engine"], json!("real-esrgan"));
    }

    /// sc-10117: the inline Image Studio upscale post-pass used to link its `(Nx upscaled)` variant
    /// to the base only via a dead `upscaledFrom` field, so existing upscaled assets have no fold
    /// lineage. The backfill reconstructs it from the `_up{N}x` filename within the generation set so
    /// the pair collapses in the Library. Additive + idempotent.
    #[test]
    fn backfill_reconstructs_inline_upscale_variant_lineage() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Upscales").expect("project creates");

        // A base image and its inline-upscaled variant in one generation set. The upscaled fact
        // mirrors the OLD write_upscaled_asset output: an `upscale` recipe record but NO
        // sourceAssetId/extra link (the dead `upscaledFrom` field never reached the sidecar).
        let base = json!({
            "assetId": "base_1",
            "mediaPath": "assets/images/genset_up/2026-07-01_z_image_turbo_scene_0001.png",
            "mimeType": "image/png",
            "displayName": "Scene #1",
            "createdAt": "2026-07-01T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
        });
        let upscaled = json!({
            "assetId": "up_1",
            "mediaPath": "assets/images/genset_up/2026-07-01_z_image_turbo_scene_0001_up2x.png",
            "mimeType": "image/png",
            "displayName": "Scene #1 (2x upscaled)",
            "createdAt": "2026-07-01T00:00:01Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "rawAdapterSettings": { "upscale": { "enabled": true, "engine": "seedvr2", "factor": 2 } },
        });
        store
            .persist_generated_asset(&project.id, "job-1", "genset_up", &base)
            .expect("base persists");
        store
            .persist_generated_asset(&project.id, "job-1", "genset_up", &upscaled)
            .expect("upscaled persists");

        let project_path = store.find_project_path(&project.id).expect("project path");
        let up_sidecar = project_path.join(
            "assets/images/genset_up/2026-07-01_z_image_turbo_scene_0001_up2x.sceneworks.json",
        );

        // Precondition: the upscaled sidecar has no fold link yet.
        let before = read_json(&up_sidecar).expect("read upscaled sidecar");
        assert!(
            before.pointer("/extra/upscaledFromAssetId").is_none(),
            "precondition: the buggy inline upscale left no fold link"
        );
        assert!(before
            .pointer("/lineage/sourceAssetId")
            .and_then(Value::as_str)
            .is_none());

        let healed = backfill_upscale_variant_lineage(&project_path).expect("backfill runs");
        assert_eq!(healed, 1, "exactly the one upscaled variant is healed");

        let after = read_json(&up_sidecar).expect("re-read healed sidecar");
        assert_eq!(after["extra"]["upscaledFromAssetId"], json!("base_1"));
        assert_eq!(after["extra"]["isUpscaled"], json!(true));
        assert_eq!(after["extra"]["factor"], json!(2));
        assert_eq!(after["extra"]["engine"], json!("seedvr2"));
        assert_eq!(after["lineage"]["sourceAssetId"], json!("base_1"));
        assert_eq!(after["lineage"]["parents"], json!(["base_1"]));

        // The base sidecar is untouched — it must NOT be marked as an upscale.
        let base_sidecar = project_path
            .join("assets/images/genset_up/2026-07-01_z_image_turbo_scene_0001.sceneworks.json");
        let base_after = read_json(&base_sidecar).expect("read base sidecar");
        assert!(
            base_after.pointer("/extra/isUpscaled").is_none(),
            "the base image must not be marked upscaled"
        );

        // Idempotent: a second pass heals nothing (the link is now present).
        let again = backfill_upscale_variant_lineage(&project_path).expect("second backfill");
        assert_eq!(again, 0, "already-linked variants are skipped");
    }

    #[test]
    fn build_generated_asset_sidecar_assembles_video_replace_person() {
        // sc-1656 slice 3: the video branch carries duration/fps, has no
        // stylePreset, builds video-shaped normalizedSettings + lineage, and fills
        // the honest replace_person defaults when no replacementStatus is reported.
        let fact = json!({
            "type": "video",
            "assetId": "asset-output",
            "mediaPath": "assets/videos/replacement.mp4",
            "mimeType": "video/mp4",
            "width": 1280, "height": 720, "duration": 6.0, "fps": 25,
            "quality": "balanced", "family": "ltx-video",
            "seed": 44, "displayName": "Replace the hero",
            "createdAt": "2026-05-17T00:00:00Z",
            "mode": "replace_person", "model": "wan_2_2", "adapter": "wan_video",
            "prompt": "Replace the hero", "negativePrompt": "", "loras": [],
            "rawAdapterSettings": {},
            "sourceClipAssetId": "asset-video",
            "personTrackId": "track-1",
            "replacementMode": "full_person_keep_outfit",
            "characterId": "character-1", "characterLookId": "look-1",
            "timelineContext": {},
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);
        assert_eq!(asset["type"], json!("video"));
        assert_eq!(asset["file"]["duration"], json!(6.0));
        assert_eq!(asset["file"]["fps"], json!(25));
        assert!(asset["recipe"].get("stylePreset").is_none());
        let normalized = &asset["recipe"]["normalizedSettings"];
        assert_eq!(normalized["personTrackId"], json!("track-1"));
        assert_eq!(
            normalized["replacementMode"],
            json!("full_person_keep_outfit")
        );
        assert_eq!(normalized["timelineContextRef"], json!("lineage.timeline"));
        assert_eq!(normalized["personDetectionActive"], json!(false));
        assert_eq!(normalized["replacementActive"], json!(false));
        assert_eq!(asset["lineage"]["sourceClipAssetId"], json!("asset-video"));
        assert_eq!(asset["lineage"]["characterId"], json!("character-1"));
        assert_eq!(asset["lineage"]["parents"], json!(["asset-video"]));
    }

    /// sc-12371: `asset.file` MEASURES the clip; `recipe.normalizedSettings` REPLAYS the ask. The
    /// same fact feeds both and they must not be given the same number.
    ///
    /// The trap this pins is a real one that nearly shipped: making `file.duration` honest by
    /// changing the fact's `duration` key ALSO rewrites `normalizedSettings.duration`, which
    /// sc-12324's "re-run this generation" rebuilds the payload from. A 6 s ask would replay as
    /// 5.96 s — off the model's `limits.durations` menu, which sc-12347 now enforces server-side,
    /// so the re-run could be refused outright.
    ///
    /// DISCRIMINATING: the measured pair (149 frames / 5.96 s) differs from the requested pair
    /// (6.0 s), so collapsing the two in EITHER direction is RED here.
    #[test]
    fn video_sidecar_file_measures_the_clip_while_normalized_settings_replay_the_ask() {
        let fact = json!({
            "type": "video",
            "assetId": "asset-1",
            "mediaPath": "assets/videos/hero.mp4",
            "mimeType": "video/mp4",
            "width": 1280, "height": 720,
            // What the user PICKED off the model's menus.
            "duration": 6.0, "fps": 25,
            // What a Wan engine actually rendered for that ask: the 4k+1 stride's 149 frames.
            "encodedFrameCount": 149, "encodedDuration": 5.96, "encodedFps": 25,
            "quality": "balanced", "family": "wan-video",
            "seed": 7, "displayName": "Hero", "createdAt": "2026-07-16T00:00:00Z",
            "mode": "text_to_video", "model": "bernini", "adapter": "mlx_bernini",
            "prompt": "hero", "negativePrompt": "", "loras": [],
            "rawAdapterSettings": { "frameCount": 149 },
            "timelineContext": {},
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);

        // The file block describes the mp4 that exists.
        assert_eq!(asset["file"]["duration"], json!(5.96));
        assert_eq!(asset["file"]["frameCount"], json!(149));
        assert_eq!(asset["file"]["fps"], json!(25));
        assert_ne!(
            asset["file"]["duration"],
            json!(6.0),
            "file.duration must not echo the request — that is the sc-12371 lie"
        );

        // The recipe replays what was asked for, so the studio form's duration select still has an
        // option to land on.
        let normalized = &asset["recipe"]["normalizedSettings"];
        assert_eq!(normalized["duration"], json!(6.0));
        assert_eq!(normalized["fps"], json!(25));
        assert_ne!(
            normalized["duration"],
            json!(5.96),
            "normalizedSettings must round-trip the user's ask — a measured duration is off the \
             model's limits.durations menu and sc-12347 enforces that menu"
        );
    }

    /// sc-12371: a fact written before the worker measured its clip (every asset already on disk)
    /// still gets a `file.duration` — the requested value it has always reported. A missing
    /// duration would be worse than a slightly optimistic one.
    #[test]
    fn video_sidecar_file_falls_back_to_the_requested_duration_for_older_facts() {
        let fact = json!({
            "type": "video",
            "assetId": "asset-old",
            "mediaPath": "assets/videos/old.mp4",
            "mimeType": "video/mp4",
            "width": 1280, "height": 720, "duration": 6.0, "fps": 25,
            "quality": "balanced", "family": "wan-video",
            "seed": 7, "displayName": "Old", "createdAt": "2026-05-17T00:00:00Z",
            "mode": "text_to_video", "model": "wan_2_2", "adapter": "mlx_wan",
            "prompt": "old", "negativePrompt": "", "loras": [],
            "rawAdapterSettings": {},
            "timelineContext": {},
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);
        assert_eq!(asset["file"]["duration"], json!(6.0));
        assert_eq!(asset["file"]["fps"], json!(25));
        // No measurement was recorded, so the file block reports no frame count rather than
        // inventing one.
        assert_eq!(asset["file"]["frameCount"], Value::Null);
    }

    /// sc-12345: the fit and the list-valued source ids survive onto `recipe.normalizedSettings`
    /// so the sc-12324 replay can rebuild the form, and every source asset lands in
    /// `lineage.parents` — not just the singular ones. ads2v is the densest mode.
    #[test]
    fn build_generated_asset_sidecar_carries_video_fit_and_multi_source_ids() {
        let fact = json!({
            "type": "video",
            "assetId": "asset-output",
            "mediaPath": "assets/videos/ad.mp4",
            "mimeType": "video/mp4",
            "width": 1280, "height": 720, "duration": 6.0, "fps": 25,
            "quality": "balanced", "family": "bernini",
            "seed": 44, "displayName": "The hero drives past",
            "createdAt": "2026-07-16T00:00:00Z",
            "mode": "ads2v", "model": "bernini_2", "adapter": "mlx_bernini",
            "prompt": "the hero drives past", "negativePrompt": "", "loras": [],
            "rawAdapterSettings": {},
            "fitMode": "pad",
            "sourceClipAssetId": "clip-main",
            "referenceClipAssetId": "clip-ref",
            "referenceAssetIds": ["ref-1", "ref-2"],
            "sourceClipAssetIds": [],
            "timelineContext": {},
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);
        let normalized = &asset["recipe"]["normalizedSettings"];
        assert_eq!(normalized["fitMode"], json!("pad"));
        assert_eq!(normalized["referenceClipAssetId"], json!("clip-ref"));
        assert_eq!(normalized["referenceAssetIds"], json!(["ref-1", "ref-2"]));
        // Every source the clip was derived from — the reference video and the subject
        // references are parents just as much as the main clip.
        assert_eq!(
            asset["lineage"]["parents"],
            json!(["clip-main", "clip-ref", "ref-1", "ref-2"])
        );

        // mv2v's clip array is a parent set too.
        let mv2v = json!({
            "type": "video",
            "assetId": "asset-mv2v",
            "mediaPath": "assets/videos/stitch.mp4",
            "mimeType": "video/mp4",
            "mode": "multi_video_to_video", "model": "bernini_2",
            "sourceClipAssetIds": ["clip-a", "clip-b"],
        });
        let stitched = build_generated_asset_sidecar("project-1", "job-2", "genset-1", &mv2v);
        assert_eq!(
            stitched["recipe"]["normalizedSettings"]["sourceClipAssetIds"],
            json!(["clip-a", "clip-b"])
        );
        assert_eq!(stitched["lineage"]["parents"], json!(["clip-a", "clip-b"]));

        // A fact written before sc-12345 has no such keys: the lists stay array-shaped so the
        // replay reader never distinguishes "absent" from "empty", and the fit reads null.
        let legacy = json!({
            "type": "video",
            "assetId": "asset-old",
            "mediaPath": "assets/videos/old.mp4",
            "mimeType": "video/mp4",
            "mode": "text_to_video", "model": "ltx_2_3",
        });
        let old = build_generated_asset_sidecar("project-1", "job-3", "genset-1", &legacy);
        let old_normalized = &old["recipe"]["normalizedSettings"];
        assert_eq!(old_normalized["referenceAssetIds"], json!([]));
        assert_eq!(old_normalized["sourceClipAssetIds"], json!([]));
        assert_eq!(old_normalized["fitMode"], Value::Null);
        assert_eq!(old["lineage"]["parents"], json!([]));
    }

    #[test]
    fn build_generated_asset_sidecar_merges_replacement_status() {
        let fact = json!({
            "type": "video",
            "assetId": "a",
            "mediaPath": "assets/videos/x.mp4",
            "mimeType": "video/mp4",
            "mode": "replace_person",
            "replacementStatus": {"replacementActive": true, "maskMode": "segmentation"},
        });
        let asset = build_generated_asset_sidecar("p", "j", "g", &fact);
        let normalized = &asset["recipe"]["normalizedSettings"];
        assert_eq!(normalized["replacementActive"], json!(true));
        assert_eq!(normalized["maskMode"], json!("segmentation"));
        assert_eq!(normalized["personDetectionActive"], json!(false));
    }

    #[test]
    fn build_generated_asset_sidecar_assembles_document() {
        // sc-1656 slice 4: interleaved document asset. The worker writes the body
        // JSON; Rust builds this sidecar — type document, mimeType
        // application/json, recipe.mode interleave, lineage.parents = image ids.
        let fact = json!({
            "type": "document",
            "assetId": "asset-doc",
            "mediaPath": "assets/documents/doc_1.json",
            "mimeType": "application/json",
            "displayName": "An illustrated guide",
            "createdAt": "2026-05-17T00:00:00Z",
            "mode": "interleave",
            "model": "sensenova_u1_8b",
            "adapter": "sensenova_u1",
            "prompt": "An illustrated guide",
            "negativePrompt": "",
            "seed": 7,
            "loras": [],
            "rawAdapterSettings": {"maxImages": 6},
            "maxImages": 6,
            "resolution": "2048x1152",
            "imageCount": 2,
            "parents": ["asset-img-1", "asset-img-2"],
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);
        assert_eq!(asset["type"], json!("document"));
        assert_eq!(asset["file"]["mimeType"], json!("application/json"));
        assert_eq!(asset["file"]["path"], json!("assets/documents/doc_1.json"));
        assert_eq!(asset["recipe"]["mode"], json!("interleave"));
        assert_eq!(
            asset["recipe"]["normalizedSettings"]["imageCount"],
            json!(2)
        );
        assert_eq!(
            asset["recipe"]["normalizedSettings"]["resolution"],
            json!("2048x1152")
        );
        assert_eq!(
            asset["lineage"]["parents"],
            json!(["asset-img-1", "asset-img-2"])
        );
        assert_eq!(asset["generationSetId"], json!("genset-1"));
    }

    #[test]
    fn project_create_writes_python_compatible_files_and_registry() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");

        let project = store.create_project("My Project").expect("project creates");

        assert!(project.id.starts_with("project_"));
        assert!(project.path.ends_with("my-project.sceneworks"));
        for folder in PROJECT_FOLDERS {
            assert!(std::path::Path::new(&project.path).join(folder).exists());
        }
        let registry = std::fs::read_to_string(temp_dir.path().join("data/recent-projects.json"))
            .expect("registry reads");
        assert!(registry.contains(&project.id));
        assert!(std::path::Path::new(&project.path)
            .join("project.db")
            .exists());
    }

    #[test]
    fn ensure_global_poses_project_is_idempotent_and_hidden() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");

        store
            .create_project("Visible Project")
            .expect("project creates");

        let poses = store
            .ensure_global_poses_project()
            .expect("poses project ensures");
        assert_eq!(poses.id, GLOBAL_POSES_PROJECT_ID);
        assert!(std::path::Path::new(&poses.path)
            .join("assets/poses")
            .exists());

        // Idempotent: a second ensure returns the same project path, no duplicate.
        let again = store
            .ensure_global_poses_project()
            .expect("ensure idempotent");
        assert_eq!(again.path, poses.path);

        // Hidden from the project switcher, but addressable directly by id.
        let listed = store.list_projects().expect("list");
        assert!(listed.iter().all(|p| p.id != GLOBAL_POSES_PROJECT_ID));
        assert_eq!(
            store
                .get_project(GLOBAL_POSES_PROJECT_ID)
                .expect("get reserved")
                .id,
            GLOBAL_POSES_PROJECT_ID
        );
    }

    #[test]
    fn list_projects_skips_a_corrupt_entry_and_returns_the_healthy_ones() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");

        let good = store.create_project("Healthy One").expect("first project");
        let broken = store.create_project("Corrupt One").expect("second project");
        let also_good = store.create_project("Healthy Two").expect("third project");

        // Corrupt one project's `project.json` after the fact: truncate it to an
        // object missing every required field (id/name/createdAt) so
        // `read_project_summary` fails for that entry only.
        let broken_file = std::path::Path::new(&broken.path).join("project.json");
        std::fs::write(&broken_file, b"{}").expect("clobber project.json");

        let listed = store.list_projects().expect("list survives a bad entry");

        let ids: Vec<&str> = listed.iter().map(|p| p.id.as_str()).collect();
        assert!(
            ids.contains(&good.id.as_str()),
            "healthy project before the bad entry is still listed"
        );
        assert!(
            ids.contains(&also_good.id.as_str()),
            "healthy project after the bad entry is still listed"
        );
        assert!(
            !ids.contains(&broken.id.as_str()),
            "the corrupt entry is skipped, not surfaced"
        );
        assert_eq!(listed.len(), 2, "exactly the two healthy projects remain");
    }

    #[test]
    fn create_pose_asset_copies_skeleton_and_writes_pose_sidecar() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let data_dir = temp_dir.path().join("data");
        let store = ProjectStore::new(&data_dir, "test-version");

        // Simulate the worker's pose_detect output in the shared cache.
        let job_id = "job_pose_1";
        let cache_dir = data_dir.join("cache").join("pose_detect").join(job_id);
        std::fs::create_dir_all(&cache_dir).expect("cache dir");
        let skeleton = cache_dir.join("photo_p0_skel.png");
        std::fs::write(&skeleton, b"\x89PNG fake skeleton bytes").expect("skeleton writes");

        let spec = json!({
            "jobId": job_id,
            "skeletonFile": "photo_p0_skel.png",
            "displayName": "Arm Raised",
            "category": "Dance",
            "tags": ["Dynamic", "dynamic", "  hero  "],
            "width": 768,
            "height": 1280,
            "pose": {
                "personIndex": 0,
                "facing": "front",
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "keypoints": [[0.5, 0.1, 0.9]],
                "hands": [[], []],
                "face": [],
                "sourceAspect": 0.6,
                "sourceAssetId": "asset_src_1"
            }
        });

        let asset = store.create_pose_asset(&spec).expect("pose asset creates");

        assert!(asset["id"].as_str().unwrap().starts_with("asset_"));
        assert_eq!(asset["type"], json!("pose"));
        assert_eq!(asset["projectId"], json!(GLOBAL_POSES_PROJECT_ID));
        assert_eq!(asset["displayName"], json!("Arm Raised"));
        // Category folded into pose metadata for grouping; tags normalized + deduped.
        assert_eq!(asset["pose"]["category"], json!("Dance"));
        assert_eq!(asset["pose"]["facing"], json!("front"));
        assert_eq!(asset["tags"], json!(["dynamic", "hero"]));
        assert_eq!(asset["lineage"]["jobId"], json!(job_id));
        assert_eq!(asset["lineage"]["sourceAssetId"], json!("asset_src_1"));
        assert_eq!(asset["lineage"]["parents"], json!(["asset_src_1"]));
        assert_eq!(asset["file"]["mimeType"], json!("image/png"));
        assert_eq!(asset["file"]["width"], json!(768));

        // The PNG was copied into the reserved project's assets/poses folder.
        let project_path =
            std::path::PathBuf::from(store.get_project(GLOBAL_POSES_PROJECT_ID).unwrap().path);
        let rel = asset["file"]["path"].as_str().unwrap();
        assert!(rel.starts_with("assets/poses/"));
        assert!(project_path.join(rel).exists());

        // Indexed: it comes back from list_assets on the reserved project, pose intact.
        let listed = store
            .list_assets(GLOBAL_POSES_PROJECT_ID, true, true, AssetScope::All)
            .expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["id"], asset["id"]);
        assert_eq!(listed[0]["pose"]["category"], json!("Dance"));
    }

    #[test]
    fn import_asset_records_source_lineage_and_provenance() {
        // Image Editor Save (sc-2434): a rasterized edit is uploaded with the id
        // of the asset it was opened from + an edit-chain provenance blob. The new
        // asset must link back via lineage and carry the provenance in `extra`,
        // while the original source asset is left untouched.
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Edits").expect("project creates");

        // The original source asset (a plain import, no lineage).
        let src_file = temp_dir.path().join("source.png");
        std::fs::write(&src_file, b"\x89PNG source bytes").expect("source writes");
        let source = store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: "source.png".to_owned(),
                    content_type: Some("image/png".to_owned()),
                    source_path: src_file,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .expect("source imports");
        let source_id = source["id"].as_str().expect("source id").to_owned();
        assert_eq!(source["lineage"]["sourceAssetId"], Value::Null);

        // The edited bitmap saved from the editor, derived from the source.
        let edited_file = temp_dir.path().join("source-edited.png");
        std::fs::write(&edited_file, b"\x89PNG edited bytes").expect("edited writes");
        let provenance = json!({
            "editor": "image_editor",
            "edits": [{ "op": "crop" }, { "op": "upscale", "engine": "real-esrgan", "factor": 4 }],
        });
        let edited = store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: "source-edited.png".to_owned(),
                    content_type: Some("image/png".to_owned()),
                    source_path: edited_file,
                    source_asset_id: Some(source_id.clone()),
                    provenance: Some(provenance.clone()),
                },
            )
            .expect("edited imports");

        // New, distinct asset that links back to the source and is Library-visible.
        assert_ne!(edited["id"], source["id"]);
        assert_eq!(edited["origin"], json!("upload"));
        assert_eq!(edited["lineage"]["sourceAssetId"], json!(source_id));
        assert_eq!(edited["lineage"]["parents"], json!([source_id]));
        // Provenance survives normalization in the top-level `extra` slot.
        assert_eq!(edited["extra"], provenance);

        // The source asset itself is untouched: still no lineage.
        let source_after = store
            .get_asset(&project.id, &source_id)
            .expect("source still present");
        assert_eq!(source_after["lineage"]["sourceAssetId"], Value::Null);
        assert!(source_after.get("extra").is_none());
    }

    /// sc-13517: the saved-voice id→clip hop end-to-end, cheaply. A saved voice's stored
    /// `referenceAudioAssetId` resolves back to the exact library clip on disk through
    /// `resolve_asset_media_path` — the link the register + generate flow relies on, proven directly
    /// rather than only by composition in the release DoD render. `resolve_asset_media_path` is
    /// media-type agnostic (it resolves the sidecar's `file.path`), so an imported image asset
    /// exercises the identical id→path resolution the audio reference clip uses.
    #[test]
    fn saved_voice_reference_id_resolves_to_the_clip_path() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Voices").expect("project creates");

        let src = temp_dir.path().join("clip.png");
        std::fs::write(&src, b"\x89PNG reference clip bytes").expect("source writes");
        let asset = store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: "clip.png".to_owned(),
                    content_type: Some("image/png".to_owned()),
                    source_path: src,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .expect("asset imports");
        let asset_id = asset["id"].as_str().expect("asset id").to_owned();

        let (voice, _dup) = store
            .create_saved_voice(
                &project.id,
                crate::voice_store::SavedVoiceCreateInput {
                    name: "Narrator".to_owned(),
                    reference_audio_asset_id: asset_id.clone(),
                    embedding: vec![0.1, 0.2, 0.3, 0.4],
                },
                crate::voice_store::DEFAULT_VOICE_DEDUP_THRESHOLD,
            )
            .expect("register saved voice");
        // The store persisted the reference pointer verbatim.
        let stored_ref = voice["referenceAudioAssetId"].as_str().expect("ref id");
        assert_eq!(stored_ref, asset_id);

        // The stored referenceAudioAssetId resolves to the real clip file on disk.
        let resolved = store
            .resolve_asset_media_path(&project.id, stored_ref)
            .expect("resolve clip");
        assert!(
            resolved.is_file(),
            "resolved clip path must exist: {}",
            resolved.display()
        );
        let expected_rel = store
            .get_asset(&project.id, &asset_id)
            .expect("asset")
            .pointer("/file/path")
            .and_then(Value::as_str)
            .expect("file path")
            .to_owned();
        assert!(
            resolved.ends_with(&expected_rel),
            "resolved {} must end with the asset's relative path {expected_rel}",
            resolved.display()
        );
    }

    /// sc-13517: `resolve_asset_media_path` resolves a real asset AND rejects a path-traversal id
    /// (the user-controlled `../` escape vector) before joining it into any filesystem path — the
    /// direct unit test the review flagged as missing.
    #[test]
    fn resolve_asset_media_path_resolves_and_rejects_traversal() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Guard").expect("project creates");

        let src = temp_dir.path().join("clip.png");
        std::fs::write(&src, b"\x89PNG bytes").expect("source writes");
        let asset = store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: "clip.png".to_owned(),
                    content_type: Some("image/png".to_owned()),
                    source_path: src,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .expect("asset imports");
        let asset_id = asset["id"].as_str().expect("asset id").to_owned();

        // Safe id → a real file under the project directory.
        let resolved = store
            .resolve_asset_media_path(&project.id, &asset_id)
            .expect("resolve safe id");
        assert!(resolved.is_file());
        assert!(resolved.starts_with(temp_dir.path()));

        // A `../` escape id is rejected outright (never joined into a path).
        for escape in ["../../etc/passwd", "..", "a/../../b"] {
            let result = store.resolve_asset_media_path(&project.id, escape);
            assert!(
                matches!(result, Err(ProjectStoreError::BadRequest(_))),
                "traversal id {escape:?} must be rejected, got {result:?}"
            );
        }
    }

    /// sc-13517: `saved_voices` rows are DB-authoritative and MUST survive a reindex. The reindex
    /// rebuilds the sidecar-backed tables (assets / generation_sets / timelines / characters) but must
    /// leave `saved_voices` untouched — this guards that claim against a future schema bump that runs
    /// the reindex on every existing project.
    #[test]
    fn saved_voices_survive_reindex() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Reindex").expect("project creates");

        let (voice, _dup) = store
            .create_saved_voice(
                &project.id,
                crate::voice_store::SavedVoiceCreateInput {
                    name: "Narrator".to_owned(),
                    reference_audio_asset_id: "asset_ref".to_owned(),
                    embedding: vec![0.5, 0.5, 0.5],
                },
                crate::voice_store::DEFAULT_VOICE_DEDUP_THRESHOLD,
            )
            .expect("register saved voice");
        let voice_id = voice["id"].as_str().expect("voice id").to_owned();
        assert_eq!(store.list_saved_voices(&project.id).expect("list").len(), 1);

        // An explicit reindex rebuilds the sidecar tables from disk.
        store.reindex_project(&project.id).expect("reindex runs");

        let after = store
            .list_saved_voices(&project.id)
            .expect("list after reindex");
        assert_eq!(after.len(), 1, "the saved voice must survive reindex");
        assert_eq!(after[0]["id"].as_str(), Some(voice_id.as_str()));
        assert_eq!(after[0]["name"].as_str(), Some("Narrator"));
    }

    /// sc-6143: a natively-supported image is stored byte-for-byte (no transcode, no re-encode),
    /// even when its declared content type is wrong — classification is by content.
    #[test]
    fn import_asset_leaves_supported_format_unchanged() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Keep").expect("project creates");

        // Valid PNG signature + arbitrary tail; declared (wrongly) as AVIF.
        let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        bytes.extend_from_slice(b"the rest of a png");
        let src = temp_dir.path().join("misnamed.avif");
        std::fs::write(&src, &bytes).expect("source writes");

        let asset = store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: "misnamed.avif".to_owned(),
                    content_type: Some("image/avif".to_owned()),
                    source_path: src,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .expect("imports");

        // Recorded as PNG (its real format), and the stored bytes are untouched.
        assert_eq!(asset["file"]["mimeType"], json!("image/png"));
        let project_path = store.find_project_path(&project.id).expect("project path");
        let stored = project_path.join(asset["file"]["path"].as_str().expect("path"));
        assert_eq!(std::fs::read(&stored).expect("read stored"), bytes);
    }

    /// sc-8872 (F-070): the stored extension is derived from the declared/sniffed mime, never
    /// trusted from the client filename. A `video/mp4` upload named `evil.html` must NOT be stored
    /// `.html` (which `project_file` would then serve as `text/html`, enabling stored XSS on the
    /// API origin). `.svg`/`.js`/`.html` neutralize; a genuine mime always wins.
    #[test]
    fn upload_extension_derives_from_mime_not_client_filename() {
        // Declared video mime wins over a dangerous client extension.
        assert_eq!(upload_extension("evil.html", "video/mp4"), ".mp4");
        assert_eq!(upload_extension("evil.svg", "video/mp4"), ".mp4");
        assert_eq!(upload_extension("evil.js", "video/webm"), ".webm");
        // A legit media extension on a matching mime is preserved.
        assert_eq!(upload_extension("clip.mp4", "video/mp4"), ".mp4");
        assert_eq!(upload_extension("clip.webm", "video/webm"), ".webm");
        // Unknown mime: a safe client media extension is honored, a dangerous one neutralizes.
        assert_eq!(
            upload_extension("clip.mp4", "application/octet-stream"),
            ".mp4"
        );
        assert_eq!(
            upload_extension("evil.html", "application/octet-stream"),
            ".bin"
        );
        // SVG is script-capable and deliberately excluded, so it never survives as `.svg`.
        assert_eq!(upload_extension("logo.svg", "image/svg+xml"), ".bin");
        // Every allow-listed extension re-derives to an inert image/video serve mime — never
        // `text/html` or `image/svg+xml` — which is the property `project_file` relies on.
        for extension in SAFE_UPLOAD_EXTENSIONS {
            let mime = guess_mime_from_filename(&format!("stored.{extension}"))
                .unwrap_or_else(|| format!("no mime for .{extension}"));
            assert!(
                mime.starts_with("image/") || mime.starts_with("video/"),
                ".{extension} serves inert media mime, got {mime}"
            );
            assert_ne!(mime, "image/svg+xml", ".{extension} must not serve svg");
        }
        assert!(is_safe_upload_extension("mp4"));
        assert!(!is_safe_upload_extension("svg"));
        assert!(!is_safe_upload_extension("html"));
    }

    /// sc-8872 (F-070): end-to-end — a `video/mp4` upload named `evil.html` is stored with a `.mp4`
    /// extension and `project_file` serves it as `video/mp4`, never `text/html`. A legitimate
    /// `.webm` upload still round-trips. This pins the content-type-confusion fix through the exact
    /// serve path the file endpoint uses.
    #[test]
    fn import_asset_video_stores_and_serves_safe_mime_regardless_of_client_filename() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Serve").expect("project creates");

        // A `video/mp4` upload whose client filename lies about its type.
        let evil_src = temp_dir.path().join("upload-evil");
        std::fs::write(&evil_src, b"<script>alert(1)</script> not really html")
            .expect("source writes");
        let evil = store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: "evil.html".to_owned(),
                    content_type: Some("video/mp4".to_owned()),
                    source_path: evil_src,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .expect("imports");

        let evil_rel = evil["file"]["path"].as_str().expect("path");
        assert!(
            evil_rel.ends_with(".mp4"),
            "video stored as .mp4, not the client .html, got {evil_rel}"
        );
        assert_eq!(evil["type"], json!("video"));
        // The user's original filename is still shown, only the stored name is neutralized.
        assert_eq!(evil["displayName"], json!("evil.html"));

        // The file-serving endpoint derives its Content-Type from the stored name — it must now be
        // an inert video type, never `text/html`.
        let served = store.project_file(&project.id, evil_rel).expect("serves");
        assert_eq!(served.content_type, "video/mp4");
        assert_ne!(served.content_type, "text/html");

        // A legitimate webm upload still stores and serves correctly.
        let webm_src = temp_dir.path().join("upload-webm");
        std::fs::write(&webm_src, b"fake but non-empty webm bytes").expect("source writes");
        let webm = store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: "clip.webm".to_owned(),
                    content_type: Some("video/webm".to_owned()),
                    source_path: webm_src,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .expect("imports");
        let webm_rel = webm["file"]["path"].as_str().expect("path");
        assert!(webm_rel.ends_with(".webm"), "webm kept, got {webm_rel}");
        let webm_served = store.project_file(&project.id, webm_rel).expect("serves");
        assert_eq!(webm_served.content_type, "video/webm");
    }

    /// sc-6143: a valid-but-unsupported image (BMP here) is transcoded to lossless PNG at import,
    /// the original upload temp is cleaned up, and the user-facing display name is retained.
    /// macOS-only (real `sips`); the ffmpeg path is covered by the worker tests.
    #[cfg(target_os = "macos")]
    #[test]
    fn import_asset_transcodes_unsupported_bmp_to_png() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Convert").expect("project creates");

        // A valid 1×1 24-bit BMP — not in the worker's png/jpeg/webp `image` build.
        let mut bmp = Vec::new();
        bmp.extend_from_slice(b"BM");
        bmp.extend_from_slice(&58u32.to_le_bytes());
        bmp.extend_from_slice(&0u16.to_le_bytes());
        bmp.extend_from_slice(&0u16.to_le_bytes());
        bmp.extend_from_slice(&54u32.to_le_bytes());
        bmp.extend_from_slice(&40u32.to_le_bytes());
        bmp.extend_from_slice(&1i32.to_le_bytes());
        bmp.extend_from_slice(&1i32.to_le_bytes());
        bmp.extend_from_slice(&1u16.to_le_bytes());
        bmp.extend_from_slice(&24u16.to_le_bytes());
        bmp.extend_from_slice(&0u32.to_le_bytes());
        bmp.extend_from_slice(&0u32.to_le_bytes());
        bmp.extend_from_slice(&2835i32.to_le_bytes());
        bmp.extend_from_slice(&2835i32.to_le_bytes());
        bmp.extend_from_slice(&0u32.to_le_bytes());
        bmp.extend_from_slice(&0u32.to_le_bytes());
        bmp.extend_from_slice(&[0x20, 0x40, 0x80, 0x00]);

        let src = temp_dir.path().join("photo.bmp");
        std::fs::write(&src, &bmp).expect("source writes");

        let asset = store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: "photo.bmp".to_owned(),
                    content_type: Some("image/bmp".to_owned()),
                    source_path: src.clone(),
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .expect("imports");

        // Stored as PNG: mime, extension, and actual on-disk magic bytes.
        assert_eq!(asset["file"]["mimeType"], json!("image/png"));
        let rel = asset["file"]["path"].as_str().expect("path");
        assert!(rel.ends_with(".png"), "stored as png, got {rel}");
        assert_eq!(asset["type"], json!("image"));
        // Display name keeps the user's original filename.
        assert_eq!(asset["displayName"], json!("photo.bmp"));

        let project_path = store.find_project_path(&project.id).expect("project path");
        let stored = std::fs::read(project_path.join(rel)).expect("read stored");
        assert_eq!(
            crate::media_convert::sniff_image_kind(&stored),
            Some(crate::media_convert::ImageKind::Png)
        );
        // The original upload temp was consumed (transcode moved the PNG, not the BMP).
        assert!(!src.exists(), "original upload temp removed");
    }

    #[test]
    fn create_pose_asset_rejects_path_traversal() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let data_dir = temp_dir.path().join("data");
        let store = ProjectStore::new(&data_dir, "test-version");

        // A real file outside the pose cache that traversal would try to reach.
        std::fs::create_dir_all(&data_dir).expect("data dir");
        std::fs::write(data_dir.join("secret.png"), b"top secret").expect("secret writes");
        let job_id = "job_pose_evil";
        std::fs::create_dir_all(data_dir.join("cache").join("pose_detect").join(job_id))
            .expect("cache dir");

        let spec = json!({
            "jobId": job_id,
            "skeletonFile": "../../secret.png",
            "category": "x"
        });
        let err = store
            .create_pose_asset(&spec)
            .expect_err("traversal rejected");
        assert!(matches!(err, ProjectStoreError::BadRequest(_)));
    }

    /// sc-4211 / F-CORE-7: asset ids are joined into `trash/<id>` and recipe
    /// paths, so the asset-sidecar chokepoint must reject a traversal id before
    /// any path use. Without it, `delete_asset`/`purge_asset` could create or
    /// `remove_dir_all` directories outside the project's trash folder.
    #[test]
    fn asset_methods_reject_path_traversal_ids() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Assets").expect("project creates");

        // A real sidecar outside the project that traversal would try to reach.
        let outside = temp_dir.path().join("outside");
        std::fs::create_dir_all(&outside).expect("outside dir");
        std::fs::write(outside.join("victim.txt"), b"do not touch").expect("victim writes");

        let evil = "../../../outside/victim";
        assert!(matches!(
            store.get_asset(&project.id, evil),
            Err(ProjectStoreError::BadRequest(_))
        ));
        assert!(matches!(
            store.delete_asset(&project.id, evil),
            Err(ProjectStoreError::BadRequest(_))
        ));
        assert!(matches!(
            store.purge_asset(&project.id, evil, true),
            Err(ProjectStoreError::BadRequest(_))
        ));
        // The traversal target is untouched.
        assert!(outside.join("victim.txt").exists());
    }

    #[test]
    fn delete_asset_rejects_unsafe_sidecar_media_path() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Assets").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        let image_dir = project_path.join("assets/images");
        let outside = temp_dir.path().join("outside.png");
        std::fs::write(&outside, b"outside").expect("outside media writes");
        std::fs::write(
            image_dir.join("unsafe.sceneworks.json"),
            serde_json::to_string_pretty(&json!({
                "id": "asset-unsafe",
                "type": "image",
                "displayName": "Unsafe",
                "createdAt": "2026-06-15T00:00:00Z",
                "file": {"path": outside.to_string_lossy()},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false}
            }))
            .expect("json"),
        )
        .expect("sidecar writes");

        let error = store
            .delete_asset(&project.id, "asset-unsafe")
            .expect_err("unsafe media path rejected");
        assert!(matches!(error, ProjectStoreError::BadRequest(_)));
        assert!(outside.exists(), "delete must not move outside media");
    }

    #[test]
    fn find_timeline_file_ignores_unsafe_indexed_path() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Timeline").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        let timeline = json!({
            "id": "timeline-1",
            "name": "Main",
            "aspectRatio": "16:9",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "duration": 0.0,
            "createdAt": "2026-06-15T00:00:00Z",
            "updatedAt": "2026-06-15T00:00:00Z"
        });
        let safe_path = project_path
            .join("timelines")
            .join("main.sceneworks.timeline.json");
        std::fs::write(
            &safe_path,
            serde_json::to_string_pretty(&timeline).expect("json"),
        )
        .expect("timeline writes");
        index_timeline(&project_path, &timeline, "../outside.timeline.json").expect("index writes");

        let found = find_timeline_file(&project_path, "timeline-1").expect("timeline found");
        assert_eq!(
            found.path.canonicalize().expect("canonical found path"),
            safe_path.canonicalize().expect("canonical safe path")
        );
        assert_eq!(
            found.relative_path,
            "timelines/main.sceneworks.timeline.json"
        );
    }

    // ---- Key Point Library (sc-4434) ---------------------------------------------------

    fn stage_kps_upload(data_dir: &std::path::Path, name: &str) -> String {
        let dir = data_dir.join("cache").join("keypoint-uploads");
        std::fs::create_dir_all(&dir).expect("uploads dir");
        let path = dir.join(name);
        std::fs::write(&path, b"\x89PNG\r\n\x1a\n fake png bytes").expect("staged image");
        path.to_string_lossy().into_owned()
    }

    fn front_kps() -> serde_json::Value {
        json!([
            [0.40, 0.34],
            [0.59, 0.34],
            [0.50, 0.43],
            [0.43, 0.53],
            [0.58, 0.53]
        ])
    }

    #[test]
    fn create_keypoint_asset_persists_with_source_image() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let data_dir = temp_dir.path().join("data");
        let store = ProjectStore::new(&data_dir, "test-version");
        let upload = stage_kps_upload(&data_dir, "upload-abc.png");

        let asset = store
            .create_keypoint_asset(&json!({
                "name": "My Front",
                "kps": front_kps(),
                "sourceUploadPath": upload,
            }))
            .expect("preset persists");
        assert_eq!(asset["type"], "keypoint");
        assert_eq!(asset["displayName"], "My Front");
        assert_eq!(asset["keypoint"]["builtin"], false);
        assert_eq!(asset["keypoint"]["kps"].as_array().unwrap().len(), 5);
        // The source image was copied into the library.
        let media_rel = asset["file"]["path"].as_str().expect("media path");
        let project_path = store
            .find_project_path(GLOBAL_KEYPOINTS_PROJECT_ID)
            .expect("project");
        assert!(
            project_path.join(media_rel).exists(),
            "source image retained"
        );

        // list_keypoint_presets = the 11 built-ins + this user preset.
        let presets = store.list_keypoint_presets().expect("list");
        assert_eq!(presets.len(), 12);
        assert_eq!(presets[0]["id"], "builtin_front");
        assert!(presets[0]["builtin"].as_bool().unwrap());
        let user = presets.last().unwrap();
        assert_eq!(user["name"], "My Front");
        assert_eq!(user["builtin"], false);
        assert!(user["sourceImageRef"].as_str().is_some());
    }

    #[test]
    fn create_keypoint_asset_labels_extensionless_upload_by_content() {
        // Regression: uploads stage as `upload-<uuid>.tmp` (no real extension). The saved
        // library asset must take its extension + mime from the file CONTENT, not the `.tmp`
        // name, or a JPEG capture gets mislabeled image/png.
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let data_dir = temp_dir.path().join("data");
        let store = ProjectStore::new(&data_dir, "test-version");
        let dir = data_dir.join("cache").join("keypoint-uploads");
        std::fs::create_dir_all(&dir).expect("uploads dir");
        let upload = dir.join("upload-deadbeef.tmp");
        // Minimal JPEG SOI + APP0 marker bytes — enough for the magic-byte sniff.
        std::fs::write(&upload, [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]).expect("staged jpeg");

        let asset = store
            .create_keypoint_asset(&json!({
                "name": "From Photo",
                "kps": front_kps(),
                "sourceUploadPath": upload.to_string_lossy(),
            }))
            .expect("preset persists");
        let media_rel = asset["file"]["path"].as_str().expect("media path");
        assert!(
            media_rel.ends_with(".jpg"),
            "expected .jpg, got {media_rel}"
        );
        assert_eq!(asset["file"]["mimeType"], "image/jpeg");
    }

    /// sc-6143: a keypoint capture in a valid-but-unsupported format (here BMP — the same decode gap
    /// AVIF hits) is transcoded to a real PNG as it lands in the library, never copied verbatim under
    /// a mislabeled `.png` name. macOS-only (relies on `sips`); the ffmpeg path off macOS is identical.
    #[cfg(target_os = "macos")]
    #[test]
    fn create_keypoint_asset_transcodes_an_unsupported_capture_to_png() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let data_dir = temp_dir.path().join("data");
        let store = ProjectStore::new(&data_dir, "test-version");
        let dir = data_dir.join("cache").join("keypoint-uploads");
        std::fs::create_dir_all(&dir).expect("uploads dir");
        // Staged with no real extension (`.tmp`), and the bytes are BMP — unsupported by the worker's
        // image build, the exact case that previously slipped through mislabeled as PNG.
        let upload = dir.join("upload-cafef00d.tmp");
        std::fs::write(&upload, sniff_bmp_bytes()).expect("staged bmp");

        let asset = store
            .create_keypoint_asset(&json!({
                "name": "From BMP",
                "kps": front_kps(),
                "sourceUploadPath": upload.to_string_lossy(),
            }))
            .expect("preset persists");

        let media_rel = asset["file"]["path"].as_str().expect("media path");
        assert!(
            media_rel.ends_with(".png"),
            "expected .png, got {media_rel}"
        );
        assert_eq!(asset["file"]["mimeType"], "image/png");
        // The stored bytes are genuinely PNG (sniffed by content), not the original BMP renamed.
        let project_path = store
            .find_project_path(GLOBAL_KEYPOINTS_PROJECT_ID)
            .expect("project");
        let stored = std::fs::read(project_path.join(media_rel)).expect("read stored");
        assert_eq!(
            crate::media_convert::sniff_image_kind(&stored),
            Some(crate::media_convert::ImageKind::Png)
        );
    }

    /// A valid 1×1 24-bit BMP (no Rust image dep needed to build one).
    #[cfg(target_os = "macos")]
    fn sniff_bmp_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"BM");
        bytes.extend_from_slice(&58u32.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&54u32.to_le_bytes());
        bytes.extend_from_slice(&40u32.to_le_bytes());
        bytes.extend_from_slice(&1i32.to_le_bytes());
        bytes.extend_from_slice(&1i32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&24u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&2835i32.to_le_bytes());
        bytes.extend_from_slice(&2835i32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&[0x20, 0x40, 0x80, 0x00]);
        bytes
    }

    #[test]
    fn sniff_image_format_reads_magic_bytes() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let png = temp_dir.path().join("a.bin");
        std::fs::write(&png, b"\x89PNG\r\n\x1a\n....").unwrap();
        assert_eq!(sniff_image_format(&png), Some(("png", "image/png")));
        let jpg = temp_dir.path().join("b.bin");
        std::fs::write(&jpg, [0xFF, 0xD8, 0xFF, 0xE0]).unwrap();
        assert_eq!(sniff_image_format(&jpg), Some(("jpg", "image/jpeg")));
        let webp = temp_dir.path().join("c.bin");
        std::fs::write(&webp, b"RIFF\x00\x00\x00\x00WEBPVP8 ").unwrap();
        assert_eq!(sniff_image_format(&webp), Some(("webp", "image/webp")));
        let other = temp_dir.path().join("d.bin");
        std::fs::write(&other, b"not an image").unwrap();
        assert_eq!(sniff_image_format(&other), None);
    }

    #[test]
    fn create_keypoint_asset_rejects_bad_inputs() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let data_dir = temp_dir.path().join("data");
        let store = ProjectStore::new(&data_dir, "test-version");
        let upload = stage_kps_upload(&data_dir, "upload-xyz.png");

        // Wrong kps arity.
        assert!(matches!(
            store.create_keypoint_asset(&json!({
                "name": "Bad", "kps": [[0.1, 0.1]], "sourceUploadPath": &upload
            })),
            Err(ProjectStoreError::BadRequest(_))
        ));
        // Out-of-range kps.
        assert!(matches!(
            store.create_keypoint_asset(&json!({
                "name": "Bad",
                "kps": [[1.4, 0.1], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                "sourceUploadPath": &upload
            })),
            Err(ProjectStoreError::BadRequest(_))
        ));
        // Path traversal outside the uploads cache.
        std::fs::write(data_dir.join("secret.png"), b"secret").expect("secret");
        assert!(store
            .create_keypoint_asset(&json!({
                "name": "Evil", "kps": front_kps(),
                "sourceUploadPath": data_dir.join("cache/keypoint-uploads/../../secret.png").to_string_lossy()
            }))
            .is_err());
    }

    #[test]
    fn keypoint_collections_default_to_builtin_then_user() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let data_dir = temp_dir.path().join("data");
        let store = ProjectStore::new(&data_dir, "test-version");

        // Fresh: only the built-in default collection, and it is the default.
        let collections = store.list_keypoint_collections().expect("list");
        assert_eq!(collections.len(), 1);
        assert_eq!(
            collections[0]["id"],
            crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID
        );
        assert_eq!(collections[0]["isDefault"], true);
        assert_eq!(
            collections[0]["orderedPresetIds"].as_array().unwrap().len(),
            11
        );

        // Create a user collection from a subset of built-ins, mark it default.
        let created = store
            .upsert_keypoint_collection(&json!({
                "name": "Just profiles",
                "orderedPresetIds": ["builtin_left_profile", "builtin_right_profile"],
                "isDefault": true,
            }))
            .expect("upsert");
        let collection_id = created["id"].as_str().unwrap().to_owned();

        let collections = store.list_keypoint_collections().expect("list");
        assert_eq!(collections.len(), 2);
        // Built-in default now yields to the user's default.
        assert_eq!(collections[0]["isDefault"], false);
        assert_eq!(collections[1]["isDefault"], true);
        assert_eq!(collections[1]["name"], "Just profiles");

        // Referencing an unknown preset is rejected.
        assert!(matches!(
            store.upsert_keypoint_collection(&json!({
                "name": "Bad", "orderedPresetIds": ["builtin_nope"]
            })),
            Err(ProjectStoreError::BadRequest(_))
        ));

        // Reset default back to the built-in.
        let collections = store
            .set_default_keypoint_collection(crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID)
            .expect("set default");
        assert_eq!(collections[0]["isDefault"], true);
        assert_eq!(collections[1]["isDefault"], false);

        // Delete the user collection; the built-in cannot be deleted.
        assert!(store
            .delete_keypoint_collection(crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID)
            .is_err());
        store
            .delete_keypoint_collection(&collection_id)
            .expect("delete user collection");
        assert_eq!(store.list_keypoint_collections().expect("list").len(), 1);
    }

    #[test]
    fn resolve_angle_collection_default_override_and_custom_preset() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let data_dir = temp_dir.path().join("data");
        let store = ProjectStore::new(&data_dir, "test-version");

        // No user collection → the built-in 11 in order.
        let (id, presets) = store.resolve_angle_collection(None).expect("resolve");
        assert_eq!(id, crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID);
        assert_eq!(presets.len(), 11);
        assert_eq!(presets[0].preset_id, "builtin_front");
        assert_eq!(presets[0].angle.as_deref(), Some("front"));

        // A user preset (real source image) + a collection mixing it with a built-in.
        let upload = stage_kps_upload(&data_dir, "upload-mix.png");
        let preset = store
            .create_keypoint_asset(&json!({
                "name": "My Angle", "kps": front_kps(), "sourceUploadPath": upload
            }))
            .expect("preset");
        let preset_id = preset["id"].as_str().unwrap().to_owned();
        let collection = store
            .upsert_keypoint_collection(&json!({
                "name": "Mix",
                "orderedPresetIds": [preset_id, "builtin_left_profile"],
                "isDefault": true,
            }))
            .expect("collection");
        let collection_id = collection["id"].as_str().unwrap().to_owned();

        // Default now resolves to the user's collection (2 presets, in order).
        let (id, presets) = store.resolve_angle_collection(None).expect("resolve");
        assert_eq!(id, collection_id);
        assert_eq!(presets.len(), 2);
        assert_eq!(presets[0].name, "My Angle");
        assert!(
            presets[0].angle.is_none(),
            "custom preset has no canonical angle"
        );
        assert_eq!(presets[1].preset_id, "builtin_left_profile");
        assert_eq!(presets[1].angle.as_deref(), Some("left_profile"));

        // Explicit override to the built-in default beats the user default.
        let (id, presets) = store
            .resolve_angle_collection(Some(crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID))
            .expect("resolve override");
        assert_eq!(id, crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID);
        assert_eq!(presets.len(), 11);
    }

    #[test]
    fn keypoint_library_hidden_from_project_switcher() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let data_dir = temp_dir.path().join("data");
        let store = ProjectStore::new(&data_dir, "test-version");
        store.ensure_global_keypoints_project().expect("ensure");
        let visible = store.list_projects().expect("list projects");
        assert!(visible
            .iter()
            .all(|project| project.id != GLOBAL_KEYPOINTS_PROJECT_ID));
    }

    #[test]
    fn reindex_rebuilds_python_project_tables() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Fixture").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        let image_dir = project_path.join("assets/images");
        std::fs::write(image_dir.join("image.png"), b"image").expect("image writes");
        std::fs::write(
            image_dir.join("image.sceneworks.json"),
            serde_json::to_string_pretty(&json!({
                "id": "asset-1",
                "type": "image",
                "displayName": "Image",
                "createdAt": "2026-05-17T00:00:00Z",
                "generationSetId": "genset-1",
                "file": {"path": "assets/images/image.png"},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false}
            }))
            .expect("json"),
        )
        .expect("sidecar writes");
        std::fs::write(
            project_path.join("generation-sets/genset-1.json"),
            serde_json::to_string_pretty(&json!({
                "id": "genset-1",
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "prompt": "test",
                "createdAt": "2026-05-17T00:00:00Z",
                "jobId": "job-1"
            }))
            .expect("json"),
        )
        .expect("genset writes");
        std::fs::write(
            project_path.join("timelines/main.sceneworks.timeline.json"),
            serde_json::to_string_pretty(&json!({
                "id": "timeline-1",
                "name": "Main",
                "aspectRatio": "16:9",
                "width": 1280,
                "height": 720,
                "fps": 30,
                "duration": 3.5,
                "createdAt": "2026-05-17T00:00:00Z",
                "updatedAt": "2026-05-17T00:00:00Z"
            }))
            .expect("json"),
        )
        .expect("timeline writes");

        let counts = store.reindex_project(&project.id).expect("reindex works");

        assert_eq!(counts.assets, 1);
        assert_eq!(counts.generation_sets, 1);
        assert_eq!(counts.timelines, 1);
    }

    #[test]
    fn document_assets_are_indexed_and_listed() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Docs").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);

        // assets/documents is created at project init (PROJECT_FOLDERS).
        assert!(project_path.join("assets/documents").exists());

        let document_dir = project_path.join("assets/documents");
        std::fs::write(
            document_dir.join("doc_1.json"),
            serde_json::to_string_pretty(&json!({
                "schemaVersion": 1,
                "id": "doc_1",
                "projectId": project.id,
                "jobId": "job-1",
                "model": "sensenova_u1_8b",
                "prompt": "illustrated guide",
                "createdAt": "2026-05-23T00:00:00Z",
                "segments": [
                    {"type": "text", "text": "Step one."},
                    {"type": "image", "assetId": "asset-img-1", "path": "assets/images/a.png"}
                ]
            }))
            .expect("json"),
        )
        .expect("document writes");
        std::fs::write(
            document_dir.join("doc_1.sceneworks.json"),
            serde_json::to_string_pretty(&json!({
                "id": "doc_1",
                "type": "document",
                "displayName": "Illustrated guide",
                "createdAt": "2026-05-23T00:00:00Z",
                "file": {"path": "assets/documents/doc_1.json"},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false}
            }))
            .expect("json"),
        )
        .expect("sidecar writes");

        let counts = store.reindex_project(&project.id).expect("reindex works");
        assert_eq!(counts.assets, 1);

        let assets = store
            .list_assets(&project.id, false, false, AssetScope::All)
            .expect("assets list");
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0]["id"], "doc_1");
        assert_eq!(assets[0]["type"], "document");
        // The document asset carries a derived origin (sc-2024).
        assert_eq!(assets[0]["origin"], "document_studio");
    }

    #[test]
    fn asset_reads_include_generation_set_recipe_when_available() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Recipes").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        let image_dir = project_path.join("assets/images");
        std::fs::write(image_dir.join("image.png"), b"image").expect("image writes");
        std::fs::write(
            image_dir.join("image.sceneworks.json"),
            serde_json::to_string_pretty(&json!({
                "id": "asset-1",
                "type": "image",
                "displayName": "Image",
                "createdAt": "2026-06-07T00:00:00Z",
                "generationSetId": "genset_recipe",
                "file": {"path": "assets/images/image.png"},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false},
                "recipe": {
                    "mode": "text_to_image",
                    "model": "z_image_turbo",
                    "adapter": "z_image_diffusers",
                    "prompt": "single asset",
                    "negativePrompt": "",
                    "seed": 7,
                    "loras": [],
                    "normalizedSettings": {},
                    "rawAdapterSettings": {}
                },
                "lineage": {"parents": [], "sourceAssetId": Value::Null, "sourceTimestamp": Value::Null, "jobId": "job-1"}
            }))
            .expect("json"),
        )
        .expect("sidecar writes");
        std::fs::write(
            project_path.join("generation-sets/genset_recipe.json"),
            serde_json::to_string_pretty(&json!({
                "schemaVersion": 1,
                "id": "genset_recipe",
                "projectId": project.id,
                "jobId": "job-1",
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "prompt": "batch prompt",
                "negativePrompt": "",
                "count": 4,
                "createdAt": "2026-06-07T00:00:00Z",
                "recipe": {
                    "mode": "text_to_image",
                    "model": "z_image_turbo",
                    "adapter": "z_image_diffusers",
                    "prompt": "batch prompt",
                    "negativePrompt": "",
                    "seed": 42,
                    "loras": [{"id": "style_lora", "weight": 0.75}],
                    "normalizedSettings": {"count": 4, "width": 1024, "height": 1024},
                    "rawAdapterSettings": {"steps": 8}
                }
            }))
            .expect("json"),
        )
        .expect("generation set writes");

        store.reindex_project(&project.id).expect("reindex works");

        let assets = store
            .list_assets(&project.id, false, false, AssetScope::All)
            .expect("assets list");
        assert_eq!(
            assets[0]["generationSet"]["recipe"]["prompt"],
            "batch prompt"
        );
        assert_eq!(assets[0]["generationSet"]["recipe"]["seed"], json!(42));
        assert_eq!(
            assets[0]["generationSet"]["recipe"]["normalizedSettings"]["count"],
            json!(4)
        );

        let detail = store
            .get_asset(&project.id, "asset-1")
            .expect("asset detail reads");
        assert_eq!(
            detail["generationSet"]["recipe"]["loras"][0]["id"],
            "style_lora"
        );
    }

    #[test]
    fn library_scope_excludes_character_studio_outputs() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Scoped").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        let image_dir = project_path.join("assets/images");

        // Two image asset sidecars written WITHOUT an explicit `origin` field, to
        // exercise the derive-on-reindex backfill: a normal Image Studio output
        // and a Character Studio test output (recipe.mode == "character_image").
        let write_sidecar = |id: &str, mode: &str| {
            std::fs::write(
                image_dir.join(format!("{id}.png")),
                b"not-a-real-png".as_slice(),
            )
            .expect("media writes");
            std::fs::write(
                image_dir.join(format!("{id}.sceneworks.json")),
                serde_json::to_string_pretty(&json!({
                    "id": id,
                    "type": "image",
                    "displayName": id,
                    "createdAt": "2026-05-23T00:00:00Z",
                    "file": {"path": format!("assets/images/{id}.png")},
                    "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false},
                    "recipe": {"mode": mode},
                }))
                .expect("json"),
            )
            .expect("sidecar writes");
        };
        write_sidecar("img_studio_1", "text_to_image");
        write_sidecar("char_test_1", "character_image");

        store.reindex_project(&project.id).expect("reindex works");

        // All scope returns both, each with a derived origin.
        let all = store
            .list_assets(&project.id, false, false, AssetScope::All)
            .expect("all list");
        assert_eq!(all.len(), 2);
        let origin_of = |id: &str| {
            all.iter()
                .find(|asset| asset["id"] == id)
                .map(|asset| asset["origin"].as_str().unwrap_or_default().to_owned())
                .unwrap_or_default()
        };
        assert_eq!(origin_of("img_studio_1"), "image_studio");
        assert_eq!(origin_of("char_test_1"), "character_studio");

        // Library scope drops the Character Studio output, keeps the studio image.
        let library = store
            .list_assets(&project.id, false, false, AssetScope::Library)
            .expect("library list");
        assert_eq!(library.len(), 1);
        assert_eq!(library[0]["id"], "img_studio_1");
    }

    #[test]
    fn person_tracks_list_and_detail_match_python_sidecar_shape() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Fixture").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        std::fs::write(
            project_path.join("person-tracks/track_1.sceneworks.person-track.json"),
            serde_json::to_string_pretty(&json!({
                "schemaVersion": 1,
                "id": "track_1",
                "projectId": project.id,
                "name": "Hero",
                "createdAt": "2026-05-17T00:00:00Z",
                "sourceAssetId": "asset-video",
                "representativeFrameAssetId": "asset-frame",
                "frames": [],
                "status": {}
            }))
            .expect("json"),
        )
        .expect("track sidecar writes");

        let tracks = store.list_person_tracks(&project.id).expect("tracks list");
        assert_eq!(tracks[0]["id"], "track_1");
        assert_eq!(
            tracks[0]["path"],
            "person-tracks/track_1.sceneworks.person-track.json"
        );

        let track = store
            .get_person_track(&project.id, "track_1")
            .expect("track detail");
        assert_eq!(track["name"], "Hero");
        assert!(store.get_person_track(&project.id, "../track_1").is_err());
        assert!(store.get_person_track(&project.id, "track~1").is_err());
    }

    #[test]
    fn person_track_corrections_persist_validate_and_clear() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Fixture").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        std::fs::write(
            project_path.join("person-tracks/track_1.sceneworks.person-track.json"),
            serde_json::to_string_pretty(&json!({
                "schemaVersion": 1,
                "id": "track_1",
                "projectId": project.id,
                "name": "Hero",
                "createdAt": "2026-05-17T00:00:00Z",
                "sourceAssetId": "asset-video",
                "representativeFrameAssetId": "asset-frame",
                "frames": [
                    {"timestamp": 0.0, "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.5}},
                    {"timestamp": 0.5, "box": {"x": 0.3, "y": 0.1, "width": 0.2, "height": 0.5}}
                ],
                "corrections": [],
                "status": {"correctionState": "ready_for_box_corrections"}
            }))
            .expect("json"),
        )
        .expect("track sidecar writes");

        // A box adjustment and a rejection persist; the server stamps metadata and
        // drops the trailing no-op entry (no box, not rejected) for frame 0.
        let updated = store
            .set_person_track_corrections(
                &project.id,
                "track_1",
                vec![
                    json!({"frameIndex": 0, "box": {"x": 0.5, "y": 0.2, "width": 0.25, "height": 0.4}, "author": "editor"}),
                    json!({"frameIndex": 1, "rejected": true}),
                    json!({"frameIndex": 0, "rejected": false}),
                ],
            )
            .expect("corrections persist");
        let corrections = updated["corrections"]
            .as_array()
            .expect("corrections array");
        assert_eq!(corrections.len(), 2);
        assert_eq!(corrections[0]["frameIndex"], 0);
        assert_eq!(corrections[0]["box"]["x"], 0.5);
        assert_eq!(corrections[0]["author"], "editor");
        assert_eq!(corrections[0]["source"], "manual");
        assert!(corrections[0]["createdAt"].is_string());
        assert_eq!(corrections[1]["frameIndex"], 1);
        assert_eq!(corrections[1]["rejected"], true);
        assert_eq!(
            updated["status"]["correctionState"],
            "box_corrections_applied"
        );

        // Out-of-range frame indices and out-of-bounds boxes are rejected.
        assert!(store
            .set_person_track_corrections(
                &project.id,
                "track_1",
                vec![json!({"frameIndex": 9, "rejected": true})]
            )
            .is_err());
        assert!(store
            .set_person_track_corrections(
                &project.id,
                "track_1",
                vec![json!({"frameIndex": 0, "box": {"x": 1.5, "y": 0.0, "width": 0.2, "height": 0.2}})]
            )
            .is_err());

        // Clearing corrections returns the track to the ready state.
        let cleared = store
            .set_person_track_corrections(&project.id, "track_1", vec![])
            .expect("corrections clear");
        assert!(cleared["corrections"].as_array().expect("array").is_empty());
        assert_eq!(
            cleared["status"]["correctionState"],
            "ready_for_box_corrections"
        );
    }

    #[test]
    fn project_file_paths_reject_traversal_and_backslashes() {
        assert!(is_safe_relative_path("assets/images/image.png"));
        assert!(!is_safe_relative_path("../outside.txt"));
        assert!(!is_safe_relative_path("assets\\..\\outside.txt"));
        assert!(!is_safe_relative_path("/absolute/path.png"));
    }

    #[test]
    fn mime_guess_covers_modern_image_uploads() {
        assert_eq!(
            guess_mime_from_filename("reference.heic").as_deref(),
            Some("image/heic")
        );
        assert_eq!(
            guess_mime_from_filename("reference.avif").as_deref(),
            Some("image/avif")
        );
    }

    #[test]
    fn concurrent_look_adds_do_not_lose_updates() {
        // sc-1633: create_character_look does a read-modify-write of the character
        // sidecar (read looks -> prepend -> write). The per-project file lock makes
        // overlapping calls serialize, so concurrent threads can't clobber each
        // other's appended look. Without the lock this race drops updates and the
        // final count comes up short. Asserts the lock is wired into the mutation.
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = Arc::new(ProjectStore::new(
            temp_dir.path().join("data"),
            "test-version",
        ));
        let project = store.create_project("Race").expect("project creates");
        let character = store
            .create_character(
                &project.id,
                CharacterCreateInput {
                    name: "Hero".to_owned(),
                    character_type: "person".to_owned(),
                    description: String::new(),
                },
            )
            .expect("character creates");
        let character_id = character
            .get("id")
            .and_then(Value::as_str)
            .expect("character id")
            .to_owned();

        let threads = 4;
        let per_thread = 12;
        std::thread::scope(|scope| {
            for worker in 0..threads {
                let store = Arc::clone(&store);
                let project_id = project.id.clone();
                let character_id = character_id.clone();
                scope.spawn(move || {
                    for index in 0..per_thread {
                        store
                            .create_character_look(
                                &project_id,
                                &character_id,
                                CharacterLookInput {
                                    name: format!("look-{worker}-{index}"),
                                    description: String::new(),
                                    approved_reference_ids: Vec::new(),
                                    recipe_settings: serde_json::Map::new(),
                                },
                            )
                            .expect("look adds");
                    }
                });
            }
        });

        let character = store
            .get_character(&project.id, &character_id)
            .expect("character reads");
        let looks = character
            .get("looks")
            .and_then(Value::as_array)
            .expect("looks array");
        assert_eq!(looks.len(), threads * per_thread);
    }

    /// issue #1435 / sc-11855: a workspace folder that rejects writes must fail
    /// with the actionable `StorageNotWritable` error (naming the folder), not a
    /// raw `SQLITE_READONLY`/`unable to open database file` that the UI shows
    /// verbatim. A read-only `projects` dir is the portable stand-in for the
    /// user's granular write denial.
    #[cfg(unix)]
    #[test]
    fn create_project_on_unwritable_projects_dir_reports_storage_not_writable() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp.path().join("data"), "test-app");
        let projects = store.projects_dir();
        std::fs::create_dir_all(&projects).expect("seed projects dir");
        std::fs::set_permissions(&projects, std::fs::Permissions::from_mode(0o555))
            .expect("lock projects dir read-only");

        let result = store.create_project("Test");

        // Restore perms so the tempdir teardown can remove the tree.
        let _ = std::fs::set_permissions(&projects, std::fs::Permissions::from_mode(0o755));

        match result {
            Err(ProjectStoreError::StorageNotWritable(detail)) => {
                assert!(
                    detail.contains("workspace folder") && detail.contains("Settings"),
                    "message should be actionable and name the workspace folder: {detail}"
                );
            }
            other => panic!("expected StorageNotWritable, got {other:?}"),
        }
    }
}
