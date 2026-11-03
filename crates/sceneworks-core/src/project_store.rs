#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use parking_lot::{Mutex, ReentrantMutexGuard};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use subtle::ConstantTimeEq;

use crate::asset_index::{
    asset_origin, asset_sidecars, find_asset_sidecar_path_on_connection,
    hydrate_indexed_asset_cached, index_asset_on_connection, normalize_asset, GenerationSetCache,
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
use crate::contracts::ExtraFields;
use crate::dataset_quality::{
    CachedTier0Scalars, DatasetEmbeddings, DatasetFaceRecords, QualityAck, QualityCheck,
};
use crate::film_compile::{production_plan_sha256, CompiledPlan, COMPILED_PACK_STALENESS_FIELD};
use crate::film_plan::{
    validate_reference_pack, ReferenceEntry, ReferencePack, PLAN_SCHEMA_VERSION, REFERENCE_KINDS,
    REFERENCE_PACK_SCHEMA_VERSION,
};
use crate::film_workspace::{FilmDraft, FilmRunLocator};
use crate::slug::slugify;
use crate::store_util::{
    ensure_column, is_safe_id, is_safe_relative_path, lock_project_files, optional_f64,
    optional_str, optional_u64, random_hex, read_json, relative_string, required_str, write_json,
};
use crate::time::utc_now;
use crate::training::TrainingDataset;
use crate::training_store::{
    apply_training_dataset_migrations, DatasetItemRepoint, ExternalTrainingDatasetItemInput,
    TrainingCaptionSidecarsResult, TrainingDatasetBatchRenameInput,
    TrainingDatasetCaptionSidecarsInput, TrainingDatasetCreateInput, TrainingDatasetMutationResult,
    TrainingDatasetStore, TrainingDatasetSummary, TrainingDatasetUpdateInput,
};
use crate::workflow_share::WorkflowShare;

/// Key inside an imported asset's top-level `extra` that carries the sanitized workflow envelope
/// the uploaded file had embedded (sc-15949).
///
/// Named `imported` on purpose. It says where the recipe came from — a file someone else's install
/// wrote — and keeps it distinguishable from this asset's own `recipe`, which for an upload is the
/// `mode: "upload"` stub and stays that way. sc-15952 reads this to offer "Use this recipe".
pub const IMPORTED_WORKFLOW_KEY: &str = "importedWorkflow";

pub const PROJECT_FOLDERS: &[&str] = &[
    "assets/images",
    "assets/tensors",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "assets/documents",
    "assets/poses",
    "assets/keypoints",
    "assets/depth",
    "characters",
    "generation-sets",
    "loras",
    "person-tracks",
    "recipes",
    "timelines",
    "films/drafts",
    "films/draft-assets",
    "films/runs",
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

/// `recent-projects.json` contains only one small metadata record per project.
/// Eight MiB accommodates many thousands of projects even with long paths while
/// bounding every cache fingerprint/read if the file is externally corrupted
/// or replaced with an unexpectedly large payload.
const MAX_REGISTRY_BYTES: u64 = 8 * 1024 * 1024;

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
    /// A valid mutation rejected because its observed store revision is stale.
    Conflict(String),
    NotFound(String),
    TimelineConflict {
        code: &'static str,
        context: Value,
    },
    /// The workspace storage location rejected the writes a project needs — the
    /// raw SQLite/IO error ("attempt to write a readonly database", permission
    /// denied) is opaque, so this carries an actionable, path-naming message
    /// instead (issue #1435 / sc-11855). Surfaced when the projects tree can be
    /// read/created but not modified in place (e.g. a macOS inherited ACL or an
    /// endpoint-security/backup agent that blocks in-place file rewrites).
    StorageNotWritable(String),
    /// A request value that failed validation, attributed to the named input field so a client can
    /// render the error beside that field instead of parsing `detail` (epic 2123 E6). `code` is a
    /// stable machine-readable reason (e.g. `subject_mask_too_large`).
    FieldInvalid {
        field: &'static str,
        code: &'static str,
        detail: String,
    },
}

impl std::fmt::Display for ProjectStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Sqlite(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::BadRequest(detail) => write!(formatter, "{detail}"),
            Self::Conflict(detail) => write!(formatter, "{detail}"),
            Self::NotFound(detail) => write!(formatter, "{detail}"),
            Self::TimelineConflict { code, context } => write!(formatter, "{code}: {context}"),
            Self::StorageNotWritable(detail) => write!(formatter, "{detail}"),
            Self::FieldInvalid { detail, .. } => write!(formatter, "{detail}"),
        }
    }
}

impl std::error::Error for ProjectStoreError {}

impl From<crate::media_convert::IngestImageError> for ProjectStoreError {
    fn from(error: crate::media_convert::IngestImageError) -> Self {
        match error {
            crate::media_convert::IngestImageError::Io(error) => Self::Io(error),
            error => Self::BadRequest(error.to_string()),
        }
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmRunFiles {
    pub directory: PathBuf,
    pub authoring_snapshot: PathBuf,
    pub plan: PathBuf,
    pub reference_pack: PathBuf,
    pub compiled: PathBuf,
    pub review_plan: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetMutationResult {
    pub id: String,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetPoster {
    pub bytes: Vec<u8>,
    pub sha256: String,
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

/// Filesystem operations performed while serving one asset-list request.
///
/// This is intentionally one aggregate surface rather than a sidecar-only
/// counter: removing sidecar hashing must not hide the generation-set reads,
/// poster probes, or SQLite opens that remain on the request path (SC-14799).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AssetListFilesystemOperations {
    pub registry_opens: u64,
    pub registry_metadata_reads: u64,
    pub registry_content_reads: u64,
    pub path_stats: u64,
    pub directory_scans: u64,
    pub sidecar_reads: u64,
    pub generation_set_reads: u64,
    pub timeline_reads: u64,
    pub character_reads: u64,
    pub poster_stats: u64,
    pub poster_reads: u64,
    pub index_marker_reads: u64,
    pub index_marker_writes: u64,
    pub index_marker_removes: u64,
    pub directory_create_calls: u64,
    pub db_opens: u64,
}

impl AssetListFilesystemOperations {
    pub const fn total(self) -> u64 {
        self.registry_opens
            + self.registry_metadata_reads
            + self.registry_content_reads
            + self.path_stats
            + self.directory_scans
            + self.sidecar_reads
            + self.generation_set_reads
            + self.timeline_reads
            + self.character_reads
            + self.poster_stats
            + self.poster_reads
            + self.index_marker_reads
            + self.index_marker_writes
            + self.index_marker_removes
            + self.directory_create_calls
            + self.db_opens
    }
}

#[derive(Clone, Copy)]
pub(crate) enum AssetListFilesystemOperation {
    RegistryOpen,
    RegistryMetadataRead,
    RegistryContentRead,
    PathStat,
    DirectoryScan,
    SidecarRead,
    GenerationSetRead,
    TimelineRead,
    CharacterRead,
    PosterStat,
    PosterRead,
    IndexMarkerRead,
    IndexMarkerWrite,
    IndexMarkerRemove,
    DirectoryCreateCall,
    DbOpen,
}

thread_local! {
    static ASSET_LIST_FILESYSTEM_OPERATIONS: RefCell<Vec<AssetListFilesystemOperations>> =
        const { RefCell::new(Vec::new()) };
}

pub(crate) fn record_asset_list_filesystem_operation(operation: AssetListFilesystemOperation) {
    ASSET_LIST_FILESYSTEM_OPERATIONS.with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(operations) = stack.last_mut() else {
            return;
        };
        match operation {
            AssetListFilesystemOperation::RegistryOpen => operations.registry_opens += 1,
            AssetListFilesystemOperation::RegistryMetadataRead => {
                operations.registry_metadata_reads += 1;
            }
            AssetListFilesystemOperation::RegistryContentRead => {
                operations.registry_content_reads += 1;
            }
            AssetListFilesystemOperation::PathStat => operations.path_stats += 1,
            AssetListFilesystemOperation::DirectoryScan => operations.directory_scans += 1,
            AssetListFilesystemOperation::SidecarRead => operations.sidecar_reads += 1,
            AssetListFilesystemOperation::GenerationSetRead => {
                operations.generation_set_reads += 1;
            }
            AssetListFilesystemOperation::TimelineRead => operations.timeline_reads += 1,
            AssetListFilesystemOperation::CharacterRead => operations.character_reads += 1,
            AssetListFilesystemOperation::PosterStat => operations.poster_stats += 1,
            AssetListFilesystemOperation::PosterRead => operations.poster_reads += 1,
            AssetListFilesystemOperation::IndexMarkerRead => operations.index_marker_reads += 1,
            AssetListFilesystemOperation::IndexMarkerWrite => operations.index_marker_writes += 1,
            AssetListFilesystemOperation::IndexMarkerRemove => {
                operations.index_marker_removes += 1;
            }
            AssetListFilesystemOperation::DirectoryCreateCall => {
                operations.directory_create_calls += 1;
            }
            AssetListFilesystemOperation::DbOpen => operations.db_opens += 1,
        }
    });
}

struct AssetListFilesystemOperationScope {
    active: bool,
}

impl AssetListFilesystemOperationScope {
    fn begin() -> Self {
        ASSET_LIST_FILESYSTEM_OPERATIONS.with(|stack| {
            stack
                .borrow_mut()
                .push(AssetListFilesystemOperations::default());
        });
        Self { active: true }
    }

    fn finish(mut self) -> AssetListFilesystemOperations {
        self.active = false;
        ASSET_LIST_FILESYSTEM_OPERATIONS.with(|stack| stack.borrow_mut().pop().unwrap_or_default())
    }
}

impl Drop for AssetListFilesystemOperationScope {
    fn drop(&mut self) {
        if self.active {
            ASSET_LIST_FILESYSTEM_OPERATIONS.with(|stack| {
                stack.borrow_mut().pop();
            });
        }
    }
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

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FilmReferenceInput {
    pub draft_revision: u32,
    /// The project-library image this role is backed by, or `None` for a DESCRIBED-ONLY role
    /// (sc-24025): one the pack states in words because no photograph of it exists. A fileless
    /// role is how the Film workspace authors the subjects the compiler holds steady by repeating
    /// their `description`, so the panel that adds them posts to this same route with no
    /// `assetId` rather than through a second path with a second validator.
    #[serde(default)]
    pub asset_id: Option<String>,
    pub role: String,
    pub kind: String,
    #[serde(default)]
    pub description: String,
    /// Which subject in the image this role names, when the same library image backs several roles
    /// (sc-24024). Adding one asset twice under two roles gives both the same
    /// `references/<assetId>.<ext>` file, which is exactly the shared-file case the pack now
    /// supports — and which [`validate_reference_pack`] refuses without a locator on each.
    #[serde(default)]
    pub locator: Option<String>,
    #[serde(default)]
    pub approved: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FilmSoundInput {
    pub draft_revision: u32,
    pub asset_id: String,
    pub role: String,
    pub kind: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct ProjectFile {
    pub path: PathBuf,
    pub content_type: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct RegistryItem {
    id: Option<String>,
    name: Option<String>,
    path: Option<String>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegistryFingerprint {
    modified: SystemTime,
    len: u64,
    // `modified + len` misses in-place same-length rewrites when a tool
    // preserves mtime. Unix ctime is the filesystem change revision; dev+ino
    // also distinguish atomic replacement by another process.
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    // Stable Rust does not expose the filesystem ChangeTime on Windows, and
    // this workspace forbids unsafe Win32 calls. File identity covers atomic
    // replacement; a fixed-size digest from the same open handle covers every
    // in-place rewrite even when length and last-write time are restored.
    #[cfg(windows)]
    volume_serial_number: u64,
    #[cfg(windows)]
    file_index: u64,
    #[cfg(windows)]
    content_sha256: [u8; 32],
}

#[derive(Debug, Default)]
struct RegistryCache {
    fingerprint: Option<RegistryFingerprint>,
    items: Vec<RegistryItem>,
    loaded: bool,
    #[cfg(test)]
    disk_reads: usize,
}

fn registry_fingerprint(path: &Path) -> ProjectStoreResult<Option<RegistryFingerprint>> {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::RegistryOpen);
    match fs::File::open(path) {
        Ok(file) => {
            #[cfg(windows)]
            let mut file = file;
            record_asset_list_filesystem_operation(
                AssetListFilesystemOperation::RegistryMetadataRead,
            );
            let metadata = file.metadata()?;
            ensure_registry_size(metadata.len())?;
            #[cfg(windows)]
            let file_information = winapi_util::file::information(&file)?;
            #[cfg(windows)]
            let content_sha256 = registry_content_sha256(&mut file)?;
            Ok(Some(RegistryFingerprint {
                modified: metadata.modified()?,
                len: metadata.len(),
                #[cfg(unix)]
                device: metadata.dev(),
                #[cfg(unix)]
                inode: metadata.ino(),
                #[cfg(unix)]
                changed_seconds: metadata.ctime(),
                #[cfg(unix)]
                changed_nanoseconds: metadata.ctime_nsec(),
                #[cfg(windows)]
                volume_serial_number: file_information.volume_serial_number(),
                #[cfg(windows)]
                file_index: file_information.file_index(),
                #[cfg(windows)]
                content_sha256,
            }))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn ensure_registry_size(len: u64) -> ProjectStoreResult<()> {
    if len > MAX_REGISTRY_BYTES {
        return Err(ProjectStoreError::Io(registry_size_error(len)));
    }
    Ok(())
}

fn registry_size_error(len: u64) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!(
            "recent-projects.json is {len} bytes; the maximum supported size is \
             {MAX_REGISTRY_BYTES} bytes"
        ),
    )
}

#[derive(Default)]
struct RegistrySizeCounter {
    len: u64,
}

impl Write for RegistrySizeCounter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let next_len = self
            .len
            .saturating_add(u64::try_from(buffer.len()).unwrap_or(u64::MAX));
        if next_len > MAX_REGISTRY_BYTES {
            return Err(registry_size_error(next_len));
        }
        self.len = next_len;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn ensure_registry_serialized_size(value: &Value) -> ProjectStoreResult<()> {
    let mut counter = RegistrySizeCounter::default();
    serde_json::to_writer_pretty(&mut counter, value)?;
    counter.write_all(b"\n")?;
    Ok(())
}

#[cfg(windows)]
fn registry_content_sha256(file: &mut fs::File) -> ProjectStoreResult<[u8; 32]> {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::RegistryContentRead);
    let mut hasher = Sha256::new();
    let mut limited = file.take(MAX_REGISTRY_BYTES + 1);
    crate::hashing::update_sha256(&mut limited, &mut hasher)?;
    ensure_registry_size(MAX_REGISTRY_BYTES + 1 - limited.limit())?;
    Ok(hasher.finalize().into())
}

fn read_registry_payload(path: &Path) -> ProjectStoreResult<String> {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::RegistryOpen);
    let file = fs::File::open(path)?;
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::RegistryMetadataRead);
    let len = file.metadata()?.len();
    ensure_registry_size(len)?;
    let mut payload = String::with_capacity(
        usize::try_from(len).expect("registry byte ceiling fits every supported target"),
    );
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::RegistryContentRead);
    file.take(MAX_REGISTRY_BYTES + 1)
        .read_to_string(&mut payload)?;
    ensure_registry_size(u64::try_from(payload.len()).unwrap_or(u64::MAX))?;
    Ok(payload)
}

#[derive(Debug)]
pub struct ProjectStore {
    data_dir: PathBuf,
    app_version: String,
    /// Guards recent-project registry reads/writes only; project DB and project-file mutations
    /// rely on SQLite transactions and filesystem operations for their own serialization.
    lock: Mutex<RegistryCache>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingDatasetUpload {
    pub filename: String,
    pub content_type: Option<String>,
    pub source_path: PathBuf,
}

fn copy_film_reference_files(
    project_path: &Path,
    draft_id: &str,
    pack: &ReferencePack,
    destination_root: &Path,
    include_sound: bool,
) -> ProjectStoreResult<()> {
    // Nothing to copy when no entry names a file on disk. A DESCRIBED-ONLY reference role is words
    // and nothing else (sc-24025), exactly as a synthesized dialogue line is, so a pack made only
    // of them stages no images — and must not be sent through the staging directory lookup below,
    // which refuses a draft whose immutable input directory was never created.
    if pack
        .references
        .iter()
        .all(|reference| reference.file().is_none())
        && (!include_sound || pack.sound.iter().all(|sound| sound.file.is_none()))
    {
        return Ok(());
    }
    let destination_relative = destination_root.strip_prefix(project_path).map_err(|_| {
        ProjectStoreError::BadRequest(
            "Film reference destination must remain inside its project".to_owned(),
        )
    })?;
    if destination_relative
        .components()
        .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(ProjectStoreError::BadRequest(
            "Film reference destination must remain inside its project".to_owned(),
        ));
    }

    let canonical_project = fs::canonicalize(project_path)?;
    fs::create_dir_all(destination_root)?;
    let canonical_destination = fs::canonicalize(destination_root)?;
    if !canonical_destination.starts_with(&canonical_project) {
        return Err(ProjectStoreError::BadRequest(
            "Film reference destination must remain inside its project".to_owned(),
        ));
    }

    let draft_assets = project_path.join("films/draft-assets").join(draft_id);
    let canonical_assets = fs::canonicalize(&draft_assets).map_err(|_| {
        ProjectStoreError::BadRequest(format!(
            "Film draft {draft_id:?} is missing its immutable reference directory"
        ))
    })?;
    if !canonical_assets.starts_with(&canonical_project) {
        return Err(ProjectStoreError::BadRequest(
            "Film reference source must remain inside its project".to_owned(),
        ));
    }
    for reference in &pack.references {
        // A described-only role stages no image, so there is nothing to find or copy (sc-24025).
        let Some(file) = reference.file() else {
            continue;
        };
        stage_draft_file(
            &canonical_assets,
            &canonical_destination,
            file,
            &reference.role,
            "reference",
            "image",
            false,
        )?;
    }
    if include_sound {
        for sound in &pack.sound {
            let Some(file) = sound.file.as_deref() else {
                continue;
            };
            // Generated dialogue may reserve a filename that the harness creates later.
            stage_draft_file(
                &canonical_assets,
                &canonical_destination,
                file,
                &sound.role,
                "sound",
                "audio",
                sound.text.is_some(),
            )?;
        }
    }
    Ok(())
}

/// Copy one immutable draft input; described-only references are skipped by the caller.
fn stage_draft_file(
    draft_assets: &Path,
    destination_root: &Path,
    file: &str,
    role: &str,
    kind: &str,
    media: &str,
    allow_missing: bool,
) -> ProjectStoreResult<()> {
    if !is_safe_relative_path(file) {
        return Err(ProjectStoreError::BadRequest(format!(
            "Film {kind} {role:?} has an unsafe file path"
        )));
    }
    let source = draft_assets.join(file);
    if allow_missing && !source.is_file() {
        return Ok(());
    }
    let canonical_source = fs::canonicalize(&source).map_err(|_| {
        ProjectStoreError::BadRequest(format!(
            "Film {kind} {role:?} is missing its staged {media} {file}"
        ))
    })?;
    if !canonical_source.starts_with(draft_assets) || !canonical_source.is_file() {
        return Err(ProjectStoreError::BadRequest(format!(
            "Film {kind} {role:?} resolves outside its immutable draft input"
        )));
    }
    let destination = destination_root.join(file);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(canonical_source, destination)?;
    Ok(())
}

impl ProjectStore {
    pub fn new(data_dir: impl Into<PathBuf>, app_version: impl Into<String>) -> Self {
        Self {
            data_dir: data_dir.into(),
            app_version: app_version.into(),
            lock: Mutex::new(RegistryCache::default()),
        }
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.data_dir.join("projects")
    }

    pub fn registry_path(&self) -> PathBuf {
        self.data_dir.join("recent-projects.json")
    }

    pub fn list_projects(&self) -> ProjectStoreResult<Vec<ProjectSummary>> {
        let mut registry_cache = self.lock.lock();
        self.ensure_data_dirs()?;
        let mut projects = Vec::new();
        for item in self.load_registry(&mut registry_cache)? {
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

        let mut registry_cache = self.lock.lock();
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

        self.provision_project_locked(&project_id, name, &project_path, &mut registry_cache)
    }

    pub fn list_film_drafts(&self, project_id: &str) -> ProjectStoreResult<Vec<FilmDraft>> {
        let project_path = self.find_project_path(project_id)?;
        let drafts_dir = project_path.join("films/drafts");
        if !drafts_dir.exists() {
            return Ok(Vec::new());
        }
        let mut drafts = Vec::new();
        for path in read_dir_paths(&drafts_dir)? {
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let draft = Self::carry_film_draft_forward(read_json(&path)?)?;
            if draft.project_id != project_id {
                return Err(ProjectStoreError::BadRequest(format!(
                    "film draft {} belongs to a different project",
                    draft.id
                )));
            }
            drafts.push(draft);
        }
        drafts.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        Ok(drafts)
    }

    pub fn create_film_draft(
        &self,
        project_id: &str,
        draft_id: &str,
        title: &str,
    ) -> ProjectStoreResult<FilmDraft> {
        self.create_film_draft_document(
            project_id,
            FilmDraft::manual_one_shot(project_id, draft_id, title),
        )
    }

    /// Persist a freshly constructed draft without a revision bump. The API uses this after it has
    /// resolved host-dependent render defaults; ordinary edits still go through
    /// [`Self::save_film_draft`] and optimistic revision checks.
    pub fn create_film_draft_document(
        &self,
        project_id: &str,
        draft: FilmDraft,
    ) -> ProjectStoreResult<FilmDraft> {
        let draft_id = draft.id.as_str();
        if !is_safe_id(draft_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film draft ID".to_owned(),
            ));
        }
        if draft.project_id != project_id || draft.revision != 1 {
            return Err(ProjectStoreError::BadRequest(
                "New film draft identity or revision does not match its project".to_owned(),
            ));
        }
        let (project_path, _guard) = self.lock_project(project_id)?;
        let path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if path.exists() {
            return Err(ProjectStoreError::BadRequest(
                "Film draft already exists".to_owned(),
            ));
        }
        write_json(&path, &draft)?;
        Ok(draft)
    }

    /// Carry a persisted film draft forward to the schema versions this build reads, then decode
    /// it (sc-24024, sc-24026).
    ///
    /// THE one place a draft read from `films/drafts/<id>.json` is brought up to date, called from
    /// every read path so they cannot drift, and the place a later schema bump extends rather than
    /// adding a fourth copy of the same idea.
    ///
    /// It takes the raw [`Value`] rather than a decoded [`FilmDraft`] because part of the
    /// carry-forward has to happen BEFORE serde sees it. Plan schema version 3 renamed the optional
    /// `shots[].sound` to a required `shots[].audio`, and [`crate::film_plan::Shot`] is
    /// `deny_unknown_fields`: a draft written by an earlier build has both faults at once and each
    /// is a hard decode error, so no helper that accepts a `FilmDraft` can ever run on one. The
    /// reference-pack stamp is applied after the decode, on the typed value, because a version 1
    /// pack decodes perfectly well.
    ///
    /// A draft is project-store STATE, not a document a person authors: it is deserialized verbatim
    /// and the workspace offers no way to edit a `schemaVersion` key inside it. Left alone, a
    /// pre-existing draft breaks in two ways at once. The plan faults above make a single stale
    /// draft 500 its own GET — and, because `list_film_drafts` propagates, take the whole Films
    /// list for that project down with it, so the user cannot even reach the draft to repair it.
    /// And when the reference pack went to version 2 (sc-24024), every draft on disk still said
    /// version 1, while `validate_reference_pack` — which `add_film_reference`, `add_film_sound`
    /// and the pack PUT all run — refuses a pack by version, so each of those would fail on every
    /// pre-existing draft with a refusal telling the user to make an edit they cannot make.
    ///
    /// Carrying forward is honest here and only here, and for a different reason per field. Pack
    /// version 2 only ADDS the optional `locator`, so a version 1 draft pack IS a structurally
    /// valid version 2 pack. The shot's old `sound` prose moves into `audio` when it said something
    /// and blanks when it did not, which is the truth in both cases: the version 2 field meant the
    /// same thing, and a draft that never had one has an author who has not yet said what the shot
    /// sounds like. Blank is not silently accepted — it surfaces as the shot-named `audio` finding,
    /// in the Audio field the workspace now shows, which is exactly where it gets fixed.
    ///
    /// DOCUMENTS on disk stay refused by version, both kinds: a pack document under E6, and a plan
    /// document in [`crate::film_plan::parse_plan_document`]. Those have an author who can edit
    /// them.
    fn carry_film_draft_forward(mut stored: Value) -> ProjectStoreResult<FilmDraft> {
        Self::carry_film_draft_plan_forward(&mut stored);
        let mut draft: FilmDraft = serde_json::from_value(stored)?;
        draft.reference_pack.schema_version = REFERENCE_PACK_SCHEMA_VERSION;
        Ok(draft)
    }

    /// The JSON-level half of [`Self::carry_film_draft_forward`]: the production-plan fixups that
    /// must land before the typed decode, because each of them is a hard serde error (sc-24026).
    fn carry_film_draft_plan_forward(stored: &mut Value) {
        let Some(plan) = stored
            .get_mut("productionPlan")
            .and_then(Value::as_object_mut)
        else {
            return;
        };
        if let Some(shots) = plan.get_mut("shots").and_then(Value::as_array_mut) {
            for shot in shots {
                let Some(shot) = shot.as_object_mut() else {
                    continue;
                };
                // Removed either way: version 3 has no `sound` on a shot, and leaving the key would
                // be an unknown field whichever way `audio` resolved.
                let carried = shot.remove("sound");
                if !shot.contains_key("audio") {
                    let audio = carried
                        .as_ref()
                        .and_then(Value::as_str)
                        .filter(|value| !value.trim().is_empty())
                        .unwrap_or_default()
                        .to_owned();
                    shot.insert("audio".to_owned(), Value::String(audio));
                }
            }
        }
        // Stamped last, and only from a version this carry-forward actually handles: an unknown
        // FUTURE version must stay unstamped so it is still reported rather than claimed.
        if plan
            .get("schemaVersion")
            .and_then(Value::as_u64)
            .is_some_and(|version| version == 1 || version == 2)
        {
            plan.insert("schemaVersion".to_owned(), Value::from(PLAN_SCHEMA_VERSION));
        }
    }

    pub fn get_film_draft(
        &self,
        project_id: &str,
        draft_id: &str,
    ) -> ProjectStoreResult<FilmDraft> {
        if !is_safe_id(draft_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film draft ID".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let draft = Self::carry_film_draft_forward(read_json(&path)?)?;
        if draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity does not match its project path".to_owned(),
            ));
        }
        Ok(draft)
    }

    pub fn save_film_draft(
        &self,
        project_id: &str,
        draft_id: &str,
        mut draft: FilmDraft,
    ) -> ProjectStoreResult<FilmDraft> {
        if !is_safe_id(draft_id) || draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity cannot be changed".to_owned(),
            ));
        }
        let (project_path, _guard) = self.lock_project(project_id)?;
        let path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let current = Self::carry_film_draft_forward(read_json(&path)?)?;
        if draft.revision != current.revision {
            return Err(ProjectStoreError::Conflict(format!(
                "Film draft revision conflict: expected {}, got {}",
                current.revision, draft.revision
            )));
        }
        // A production+compiled import is one document transaction. If the supplied compile
        // exactly matches the supplied plan, carry it across the store-owned revision bump by
        // updating only its identity hash. If the plan was edited independently, preserve the old
        // identity so preflight reports it stale and rendering cannot silently use old prompts.
        //
        // The question asked here is about the PLAN only (sc-24029). The bump this carries across
        // rewrites the plan's identity — `version`, and therefore its hash — and nothing else, so
        // a pack that no longer matches must survive the save as exactly that: one finding naming
        // the pack. Counting the pack here instead would withhold the plan carry-across, and a
        // pack edit would then surface as a STALE PLAN as well, putting the wrong cause first in
        // the refusal the operator reads.
        let supplied_plan_sha = production_plan_sha256(&draft.production_plan)?;
        let compiled_was_current = draft.compiled_plan.as_ref().is_some_and(|compiled| {
            compiled
                .staleness_findings(
                    &draft.production_plan,
                    &supplied_plan_sha,
                    &draft.reference_pack,
                )
                .iter()
                // Everything BUT the pack: a pack finding is not an answer about the plan, and
                // it is deliberately carried through the save rather than resolved by it.
                .all(|finding| finding.field == COMPILED_PACK_STALENESS_FIELD)
        });
        draft.reconcile_review_plan();
        draft.revision = current.revision.saturating_add(1);
        draft.created_at = current.created_at;
        draft.updated_at = utc_now();
        draft.production_plan.id = draft.id.clone();
        draft.production_plan.version = draft.revision;
        draft.production_plan.title = draft.title.clone();
        if compiled_was_current {
            let normalized_sha = production_plan_sha256(&draft.production_plan)?;
            if let Some(compiled) = draft.compiled_plan.as_mut() {
                compiled.plan_id = draft.production_plan.id.clone();
                compiled.plan_version = draft.production_plan.version;
                compiled.plan_sha256 = normalized_sha;
            }
        }
        write_json(&path, &draft)?;
        Ok(draft)
    }

    /// Copy a project-library image into a draft-owned reference pack and persist the new draft
    /// revision. Runs later copy from this immutable draft input rather than reading mutable asset
    /// library state.
    ///
    /// With no [`FilmReferenceInput::asset_id`] this authors a DESCRIBED-ONLY role instead
    /// (sc-24025): no image is resolved, none is copied, and the entry names neither a `file` nor a
    /// `sourceAssetId`. Both shapes are checked by the one [`validate_reference_pack`] below, so a
    /// described-only role with no `description` is refused in the core's own words rather than by
    /// a rule this route states a second time.
    pub fn add_film_reference(
        &self,
        project_id: &str,
        draft_id: &str,
        input: FilmReferenceInput,
    ) -> ProjectStoreResult<FilmDraft> {
        if !is_safe_id(draft_id)
            || input
                .asset_id
                .as_deref()
                .is_some_and(|asset_id| !is_safe_id(asset_id))
        {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film draft or source asset ID".to_owned(),
            ));
        }
        if !REFERENCE_KINDS.contains(&input.kind.as_str()) {
            return Err(ProjectStoreError::BadRequest(format!(
                "Reference kind must be one of {}",
                REFERENCE_KINDS.join(", ")
            )));
        }
        // Resolved BEFORE the project lock exactly as it always was, and only for a role that
        // names an image. A described-only role has no bytes to find, stat or copy.
        let image = match input.asset_id.as_deref() {
            Some(asset_id) => {
                let source = self.resolve_asset_media_path(project_id, asset_id)?;
                let extension = source
                    .extension()
                    .and_then(|value| value.to_str())
                    .map(str::to_ascii_lowercase)
                    .filter(|value| matches!(value.as_str(), "png" | "jpg" | "jpeg" | "webp"))
                    .ok_or_else(|| {
                        ProjectStoreError::BadRequest(
                            "Film references must use a PNG, JPEG, or WebP image asset".to_owned(),
                        )
                    })?;
                Some((source, format!("references/{asset_id}.{extension}")))
            }
            None => None,
        };

        let (project_path, _guard) = self.lock_project(project_id)?;
        let draft_path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !draft_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let mut draft = Self::carry_film_draft_forward(read_json(&draft_path)?)?;
        if draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity does not match its project path".to_owned(),
            ));
        }
        if input.draft_revision != draft.revision {
            return Err(ProjectStoreError::Conflict(format!(
                "Film draft revision conflict: expected {}, got {}",
                draft.revision, input.draft_revision
            )));
        }

        draft.reference_pack.references.push(ReferenceEntry {
            role: input.role,
            kind: input.kind,
            // BOTH or NEITHER (sc-24025). An entry naming a `sourceAssetId` with no `file` is
            // refused by `validate_reference_pack` as an image-backed role that forgot where its
            // picture is, so the described-only shape has to drop the asset id too — which it
            // does by construction here, because both are read from the one `image` above.
            file: image.as_ref().map(|(_, file)| file.clone()),
            source_asset_id: input.asset_id,
            description: input.description,
            locator: input.locator,
            approved: input.approved,
            generated: false,
            generation: None,
        });
        draft.reference_pack.version = draft.reference_pack.version.saturating_add(1);
        let findings = validate_reference_pack(&draft.reference_pack);
        if !findings.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                findings
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "),
            ));
        }

        if let Some((source, relative_file)) = image {
            let destination = project_path
                .join("films/draft-assets")
                .join(draft_id)
                .join(&relative_file);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&source, &destination)?;
        }
        draft.revision = draft.revision.saturating_add(1);
        draft.production_plan.version = draft.revision;
        draft.updated_at = utc_now();
        write_json(&draft_path, &draft)?;
        Ok(draft)
    }

    /// Pin a draft revision's immutable reference bytes beside one planning operation. The caller
    /// writes the matching `references.json`; this method guarantees every file named by that
    /// document is copied from the draft-owned store into the contained operation directory.
    pub fn stage_film_planning_references(
        &self,
        project_id: &str,
        draft_id: &str,
        draft_revision: u32,
        operation_id: &str,
    ) -> ProjectStoreResult<()> {
        if !is_safe_id(draft_id) || !is_safe_id(operation_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film draft or planning operation ID".to_owned(),
            ));
        }
        let (project_path, _guard) = self.lock_project(project_id)?;
        let draft_path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !draft_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let draft = Self::carry_film_draft_forward(read_json(&draft_path)?)?;
        if draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity does not match its project path".to_owned(),
            ));
        }
        if draft.revision != draft_revision {
            return Err(ProjectStoreError::Conflict(format!(
                "Film draft revision conflict: expected {}, got {draft_revision}",
                draft.revision
            )));
        }
        let operation_dir = project_path
            .join("films/planning")
            .join(draft_id)
            .join("operations")
            .join(operation_id);
        fs::create_dir_all(&operation_dir)?;
        copy_film_reference_files(
            &project_path,
            draft_id,
            &draft.reference_pack,
            &operation_dir,
            false,
        )
    }

    /// Copy a project-library audio asset into a draft-owned sound pack and persist the new draft
    /// revision. The immutable staged file is what a later run pins, even if the library asset is
    /// renamed, replaced, or removed in the meantime.
    pub fn add_film_sound(
        &self,
        project_id: &str,
        draft_id: &str,
        input: FilmSoundInput,
    ) -> ProjectStoreResult<FilmDraft> {
        if !is_safe_id(draft_id) || !is_safe_id(&input.asset_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film draft or source asset ID".to_owned(),
            ));
        }
        if !crate::film_plan::SOUND_KINDS.contains(&input.kind.as_str()) {
            return Err(ProjectStoreError::BadRequest(format!(
                "Sound kind must be one of {}",
                crate::film_plan::SOUND_KINDS.join(", ")
            )));
        }
        let source = self.resolve_asset_media_path(project_id, &input.asset_id)?;
        let extension = source
            .extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .filter(|value| {
                matches!(
                    value.as_str(),
                    "wav" | "mp3" | "m4a" | "aac" | "flac" | "ogg" | "opus"
                )
            })
            .ok_or_else(|| {
                ProjectStoreError::BadRequest(
                    "Film sound must use a WAV, MP3, M4A, AAC, FLAC, OGG, or Opus audio asset"
                        .to_owned(),
                )
            })?;

        let (project_path, _guard) = self.lock_project(project_id)?;
        let draft_path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !draft_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let mut draft = Self::carry_film_draft_forward(read_json(&draft_path)?)?;
        if draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity does not match its project path".to_owned(),
            ));
        }
        if input.draft_revision != draft.revision {
            return Err(ProjectStoreError::Conflict(format!(
                "Film draft revision conflict: expected {}, got {}",
                draft.revision, input.draft_revision
            )));
        }

        let relative_file = format!("sound/{}.{}", input.asset_id, extension);
        draft
            .reference_pack
            .sound
            .push(crate::film_plan::SoundEntry {
                role: input.role,
                kind: input.kind,
                file: Some(relative_file.clone()),
                description: input.description,
                text: None,
                voice: None,
                model: None,
            });
        draft.reference_pack.version = draft.reference_pack.version.saturating_add(1);
        let findings = validate_reference_pack(&draft.reference_pack);
        if !findings.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                findings
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "),
            ));
        }

        let destination = project_path
            .join("films/draft-assets")
            .join(draft_id)
            .join(&relative_file);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&source, &destination)?;
        draft.revision = draft.revision.saturating_add(1);
        draft.production_plan.version = draft.revision;
        draft.updated_at = utc_now();
        write_json(&draft_path, &draft)?;
        Ok(draft)
    }

    pub fn create_film_run(
        &self,
        project_id: &str,
        run_locator_id: &str,
        draft_id: &str,
        selected_shot_ids: Vec<String>,
        compiled: Option<CompiledPlan>,
    ) -> ProjectStoreResult<FilmRunLocator> {
        self.pin_film_run(
            project_id,
            run_locator_id,
            draft_id,
            None,
            selected_shot_ids,
            compiled,
        )
    }

    /// Pin exactly the draft the caller validated; the comparison and all documents share a lock.
    pub fn create_film_run_at_revision(
        &self,
        project_id: &str,
        run_locator_id: &str,
        draft_id: &str,
        expected_revision: u32,
        selected_shot_ids: Vec<String>,
    ) -> ProjectStoreResult<FilmRunLocator> {
        self.pin_film_run(
            project_id,
            run_locator_id,
            draft_id,
            Some(expected_revision),
            selected_shot_ids,
            None,
        )
    }

    fn pin_film_run(
        &self,
        project_id: &str,
        run_locator_id: &str,
        draft_id: &str,
        expected_revision: Option<u32>,
        selected_shot_ids: Vec<String>,
        compiled: Option<CompiledPlan>,
    ) -> ProjectStoreResult<FilmRunLocator> {
        if !is_safe_id(run_locator_id) || !is_safe_id(draft_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film run ID".to_owned(),
            ));
        }
        let (project_path, _guard) = self.lock_project(project_id)?;
        let draft_path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !draft_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let draft = Self::carry_film_draft_forward(read_json(&draft_path)?)?;
        if draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity does not match its project path".to_owned(),
            ));
        }
        if expected_revision.is_some_and(|expected| expected != draft.revision) {
            return Err(ProjectStoreError::Conflict(format!(
                "Film draft revision conflict: expected {}, found {}; reload and preflight the draft before rendering",
                expected_revision.expect("checked"), draft.revision)));
        }
        let selected_shot_ids = if selected_shot_ids.is_empty() {
            draft
                .production_plan
                .shots
                .iter()
                .map(|shot| shot.id.clone())
                .collect()
        } else {
            selected_shot_ids
        };
        if selected_shot_ids.iter().any(|id| {
            !draft
                .production_plan
                .shots
                .iter()
                .any(|shot| &shot.id == id)
        }) {
            return Err(ProjectStoreError::BadRequest(
                "Film run selection contains an unknown shot".to_owned(),
            ));
        }
        let compiled = compiled.or_else(|| draft.compiled_plan.clone());
        if let Some(compiled) = &compiled {
            let sha = production_plan_sha256(&draft.production_plan)?;
            let findings =
                compiled.staleness_findings(&draft.production_plan, &sha, &draft.reference_pack);
            if !findings.is_empty() {
                // `Display`, not `{:?}` — this string reaches an operator through the API, and a
                // `Vec<PlanDiagnostic>` struct dump is the store's private notation (sc-24029).
                return Err(ProjectStoreError::BadRequest(format!(
                    "Film compiled plan does not match the pinned draft: {}",
                    findings
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("; ")
                )));
            }
        }
        let mut draft = draft;
        draft.reconcile_review_plan();
        let relative_dir = format!("films/runs/{run_locator_id}");
        let run_dir = project_path.join(&relative_dir);
        if run_dir.exists() {
            return Err(ProjectStoreError::BadRequest(
                "Film run already exists".to_owned(),
            ));
        }
        fs::create_dir_all(&run_dir)?;
        let pin_result = (|| -> ProjectStoreResult<()> {
            copy_film_reference_files(
                &project_path,
                draft_id,
                &draft.reference_pack,
                &run_dir,
                true,
            )?;
            write_json(&run_dir.join("authoring.json"), &draft)?;
            write_json(&run_dir.join("plan.json"), &draft.production_plan)?;
            write_json(&run_dir.join("references.json"), &draft.reference_pack)?;
            write_json(&run_dir.join("review.jsonc"), &draft.review_plan_for_run())?;
            if let Some(compiled) = compiled.as_ref() {
                write_json(&run_dir.join("compiled.json"), compiled)?;
            }
            Ok(())
        })();
        if let Err(error) = pin_result {
            let _ = fs::remove_dir_all(&run_dir);
            return Err(error);
        }
        let locator = FilmRunLocator {
            schema_version: crate::film_workspace::FILM_RUN_LOCATOR_SCHEMA_VERSION,
            id: run_locator_id.to_owned(),
            project_id: project_id.to_owned(),
            draft_id: draft_id.to_owned(),
            draft_revision: draft.revision,
            selected_shot_ids,
            record_directory: relative_dir,
            authoring_snapshot_path: Some(format!("films/runs/{run_locator_id}/authoring.json")),
            created_at: utc_now(),
        };
        write_json(&run_dir.join("locator.json"), &locator)?;
        Ok(locator)
    }

    pub fn get_film_run(
        &self,
        project_id: &str,
        run_locator_id: &str,
    ) -> ProjectStoreResult<FilmRunLocator> {
        let files = self.film_run_files(project_id, run_locator_id)?;
        let path = files.directory.join("locator.json");
        if !path.exists() {
            return Err(ProjectStoreError::NotFound("Film run not found".to_owned()));
        }
        let locator: FilmRunLocator = serde_json::from_value(read_json(&path)?)?;
        if locator.id != run_locator_id || locator.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film run identity does not match its project path".to_owned(),
            ));
        }
        Ok(locator)
    }

    /// Read the complete authoring document retained for a run. Locators created before authoring
    /// snapshots were introduced return `None`; derived render documents are never used to invent
    /// a historical draft.
    pub fn get_film_run_authoring_snapshot(
        &self,
        project_id: &str,
        run_locator_id: &str,
    ) -> ProjectStoreResult<Option<FilmDraft>> {
        let locator = self.get_film_run(project_id, run_locator_id)?;
        let Some(relative_path) = locator.authoring_snapshot_path.as_deref() else {
            return Ok(None);
        };
        if !is_safe_relative_path(relative_path) {
            return Err(ProjectStoreError::BadRequest(
                "Film run authoring snapshot path must be project-relative".to_owned(),
            ));
        }
        let files = self.film_run_files(project_id, run_locator_id)?;
        let expected_path = format!("{}/authoring.json", locator.record_directory);
        if relative_path != expected_path {
            return Err(ProjectStoreError::BadRequest(
                "Film run authoring snapshot path does not match its run directory".to_owned(),
            ));
        }
        let snapshot = Self::carry_film_draft_forward(read_json(&files.authoring_snapshot)?)?;
        if snapshot.id != locator.draft_id
            || snapshot.project_id != locator.project_id
            || snapshot.revision != locator.draft_revision
        {
            return Err(ProjectStoreError::BadRequest(
                "Film run authoring snapshot identity does not match its locator".to_owned(),
            ));
        }
        Ok(Some(snapshot))
    }

    pub fn list_film_runs(&self, project_id: &str) -> ProjectStoreResult<Vec<FilmRunLocator>> {
        let project_path = self.find_project_path(project_id)?;
        let runs_dir = project_path.join("films/runs");
        if !runs_dir.exists() {
            return Ok(Vec::new());
        }
        let mut runs = Vec::new();
        for directory in read_dir_paths(&runs_dir)? {
            let locator_path = directory.join("locator.json");
            if !locator_path.is_file() {
                continue;
            }
            let locator: FilmRunLocator = serde_json::from_value(read_json(&locator_path)?)?;
            if locator.project_id != project_id
                || directory.file_name().and_then(|value| value.to_str())
                    != Some(locator.id.as_str())
            {
                return Err(ProjectStoreError::BadRequest(format!(
                    "Film run identity does not match {}",
                    locator_path.display()
                )));
            }
            runs.push(locator);
        }
        runs.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        Ok(runs)
    }

    pub fn film_run_files(
        &self,
        project_id: &str,
        run_locator_id: &str,
    ) -> ProjectStoreResult<FilmRunFiles> {
        if !is_safe_id(run_locator_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film run ID".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let directory = project_path.join("films/runs").join(run_locator_id);
        Ok(FilmRunFiles {
            authoring_snapshot: directory.join("authoring.json"),
            plan: directory.join("plan.json"),
            reference_pack: directory.join("references.json"),
            compiled: directory.join("compiled.json"),
            review_plan: directory.join("review.jsonc"),
            directory,
        })
    }

    /// Provision a project directory (folders + project file + db + registry entry)
    /// for a known id/path. Caller must hold `self.lock`. Shared by `create_project`
    /// (random id) and `ensure_global_poses_project` (fixed id).
    fn provision_project_locked(
        &self,
        project_id: &str,
        name: &str,
        project_path: &Path,
        registry_cache: &mut RegistryCache,
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
        if let Err(error) = connect_project_db_migrated(project_path).and_then(|mut connection| {
            let conn = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            // A newly provisioned project has no legacy sidecars to backfill;
            // mark its empty index complete so the first indexed write is not
            // immediately erased by an unnecessary rebuild.
            conn.execute(
                "insert or replace into project_metadata (key, value) values (?1, ?2)",
                params![ASSET_INDEX_VERSION_KEY, PROJECT_SCHEMA_VERSION.to_string()],
            )?;
            conn.commit()?;
            Ok(())
        }) {
            return Err(if is_write_denied(&error) {
                ProjectStoreError::StorageNotWritable(storage_not_writable_message(
                    project_path.parent().unwrap_or(project_path),
                ))
            } else {
                error
            });
        }

        let mut registry = self
            .load_registry(registry_cache)?
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
        self.save_registry(&registry, registry_cache)?;

        read_project_summary(project_path)
    }

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

    /// Ensure the reserved global Key Point Library project exists (idempotent), returning its
    /// summary. Created lazily on first preset/collection write (epic 4422, sc-4434).
    pub fn ensure_global_keypoints_project(&self) -> ProjectStoreResult<ProjectSummary> {
        let mut registry_cache = self.lock.lock();
        self.ensure_data_dirs()?;
        let project_path = self.projects_dir().join("global-keypoints.sceneworks");
        if project_path.exists() {
            return read_project_summary(&project_path);
        }
        self.provision_project_locked(
            GLOBAL_KEYPOINTS_PROJECT_ID,
            GLOBAL_KEYPOINTS_PROJECT_NAME,
            &project_path,
            &mut registry_cache,
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

    pub fn create_training_dataset_from_external(
        &self,
        project_id: &str,
        input: TrainingDatasetCreateInput,
        items: Vec<ExternalTrainingDatasetItemInput>,
        extra: ExtraFields,
        idempotency_key: &str,
    ) -> ProjectStoreResult<(TrainingDataset, bool)> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).create_dataset_from_external(
            project_id,
            input,
            items,
            extra,
            idempotency_key,
        )
    }

    pub fn get_catalog_materialization(
        &self,
        project_id: &str,
        idempotency_key: &str,
    ) -> ProjectStoreResult<Option<TrainingDataset>> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path)
            .get_catalog_materialization(project_id, idempotency_key)
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
        // SVG is active document syntax, not a generic upload format. Vector assets may enter
        // only through the worker-owned `vector_generate` sanitizer/rasterizer boundary, which
        // writes a canonical SVG plus a bounded PNG preview together (sc-22251).
        if content_type.eq_ignore_ascii_case("image/svg+xml")
            || Path::new(&upload.filename)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("svg"))
        {
            return Err(ProjectStoreError::BadRequest(
                "SVG uploads are not supported; use Vector Studio".to_owned(),
            ));
        }
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
            // A dataset item is not an asset: it has no `extra` slot, and no reader offers its
            // recipe back. So this lane opts out of the sc-15949 chunk scan rather than walking the
            // metadata of every image in a bulk dataset import for an answer nobody reads.
            WorkflowScan::Skip,
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

    /// Finalize an empty import against its observed revision under the same
    /// project lock as materialization and the dataset write.
    pub fn finalize_training_dataset_import(
        &self,
        project_id: &str,
        dataset_id: &str,
        expected_version: u32,
        items: Vec<crate::training_store::TrainingDatasetItemInput>,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let store = TrainingDatasetStore::new(project_path);
        let current = store.get_dataset(project_id, dataset_id)?;
        if current.version != expected_version {
            return Err(ProjectStoreError::Conflict(format!(
                "Dataset changed during Parquet import (expected version {}, found {}).",
                expected_version, current.version
            )));
        }
        if !current.items.is_empty() {
            return Err(ProjectStoreError::Conflict(
                "Dataset is no longer empty; Parquet import was not applied.".to_owned(),
            ));
        }
        store.update_dataset(
            project_id,
            dataset_id,
            TrainingDatasetUpdateInput {
                name: None,
                status: None,
                character_id: None,
                items: Some(items),
                preserve_caption_item_ids: Vec::new(),
            },
        )
    }

    pub fn install_ltx_prepared_bundle(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_id: &str,
        source_path: &Path,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).install_ltx_prepared_bundle(
            project_id,
            dataset_id,
            item_id,
            source_path,
        )
    }

    /// Install or remove an Iris training sidecar of a dataset item (sc-25687 / sc-25688).
    pub fn set_iris_item_sidecar(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_id: &str,
        sidecar: &crate::training_store::IrisItemSidecar,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .set_iris_item_sidecar(project_id, dataset_id, item_id, sidecar)
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

    /// The images a `dataset_subject_mask` job segments (sc-2126). Locked wrapper over
    /// [`TrainingDatasetStore::subject_mask_targets`] (it may backfill legacy content hashes).
    pub fn training_dataset_subject_mask_targets(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_ids: Option<&[String]>,
    ) -> ProjectStoreResult<(
        Vec<crate::training_subject_masks::SubjectMaskTarget>,
        PathBuf,
        String,
    )> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let targets = TrainingDatasetStore::new(project_path.clone())
            .subject_mask_targets(project_id, dataset_id, item_ids)?;
        let root = crate::training_store::dataset_root(&project_path, dataset_id);
        let project_stem = project_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_owned();
        Ok((targets, root, project_stem))
    }

    /// Persist generated or uploaded subject masks (sc-2126). Locked wrapper over
    /// [`TrainingDatasetStore::write_subject_masks`].
    pub fn write_training_dataset_subject_masks(
        &self,
        project_id: &str,
        dataset_id: &str,
        writes: Vec<crate::training_subject_masks::SubjectMaskWrite>,
        source: crate::training_subject_masks::SubjectMaskSource,
    ) -> ProjectStoreResult<crate::training_subject_masks::SubjectMaskReport> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .write_subject_masks(project_id, dataset_id, writes, source)
    }

    /// Install a user-uploaded replacement subject mask (sc-2126). Locked wrapper over
    /// [`TrainingDatasetStore::upload_subject_mask`].
    pub fn upload_training_dataset_subject_mask(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_id: &str,
        source_path: &Path,
    ) -> ProjectStoreResult<crate::training_subject_masks::SubjectMaskReport> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).upload_subject_mask(
            project_id,
            dataset_id,
            item_id,
            source_path,
        )
    }

    /// A dataset's subject-mask coverage (sc-2126). Locked wrapper over
    /// [`TrainingDatasetStore::subject_mask_report`].
    pub fn training_dataset_subject_mask_report(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<crate::training_subject_masks::SubjectMaskReport> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).subject_mask_report(project_id, dataset_id)
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
        let counts = reindex_project_path(project_id, &project_path, true)?;
        Ok(ReindexResult {
            project_id: project_id.to_owned(),
            assets: counts.assets,
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
            let mut registry_cache = self.lock.lock();
            self.load_registry(&mut registry_cache)?
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
        let existing = match find_timeline_file(&project_path, &timeline_id) {
            Ok(file) => Some(read_json(&file.path)?),
            Err(ProjectStoreError::NotFound(_)) => None,
            Err(error) => return Err(error),
        };
        let current_revision = existing
            .as_ref()
            .map(crate::film_timeline::revision)
            .unwrap_or(0);
        if timeline
            .get("revision")
            .is_some_and(|value| value.as_u64().is_none())
        {
            return Err(ProjectStoreError::BadRequest(
                "revision must be a non-negative integer".into(),
            ));
        }
        let expected_revision = crate::film_timeline::revision(&timeline);
        if expected_revision != current_revision {
            return Err(ProjectStoreError::TimelineConflict {
                code: "timeline_revision_conflict",
                context: json!({"timelineId": timeline_id, "expectedRevision": expected_revision, "currentRevision": current_revision}),
            });
        }
        crate::film_timeline::validate_metadata(&timeline)?;
        if let Some(previous) = &existing {
            crate::film_timeline::validate_metadata(previous)?;
            crate::film_timeline::reconcile_cut(previous, &mut timeline, current_revision + 1);
        }
        timeline["revision"] = json!(current_revision + 1);
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
        // A rename changes the slug portion of the timeline file name while keeping
        // its id. Remove every prior document for this id after the replacement has
        // reached disk; otherwise a later reindex sees duplicate ids and can
        // nondeterministically revive the pre-rename content.
        remove_stale_timeline_files(&project_path, &timeline_id, &path)?;
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
        let (_path, _guard) = self.lock_project(project_id)?;
        self.get_timeline(project_id, timeline_id)?;
        self.save_timeline(project_id, timeline)
    }

    /// The same project lock covers delivery read, merge, validation and CAS write.
    pub fn deliver_film_timeline(
        &self,
        project_id: &str,
        timeline_id: &str,
        delivery: Value,
    ) -> ProjectStoreResult<Value> {
        let (_path, _guard) = self.lock_project(project_id)?;
        let mut timeline = self.get_timeline(project_id, timeline_id)?;
        if let Some(expected) = delivery.get("expectedRevision").and_then(Value::as_u64) {
            let current = crate::film_timeline::revision(&timeline);
            if expected != current {
                return Err(ProjectStoreError::TimelineConflict {
                    code: "timeline_revision_conflict",
                    context: json!({"timelineId": timeline_id, "expectedRevision": expected, "currentRevision": current}),
                });
            }
        }
        let before = timeline.clone();
        crate::film_timeline::deliver(&mut timeline, delivery, |asset_id| {
            let asset = self.get_asset(project_id, asset_id)?;
            asset
                .get("file")
                .and_then(|file| file.get("duration"))
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| {
                    ProjectStoreError::BadRequest(format!(
                        "Asset {asset_id} has no measured duration"
                    ))
                })
        })?;
        if timeline == before {
            return Ok(before);
        }
        self.save_existing_timeline(project_id, timeline_id, timeline)
    }

    /// Freeze the document the export worker will actually read. A later edit cannot change it.
    pub fn snapshot_timeline_export(
        &self,
        project_id: &str,
        timeline_id: &str,
    ) -> ProjectStoreResult<TimelineFileDocument> {
        let (project_path, _guard) = self.lock_project(project_id)?;
        let mut result = self.timeline_file_and_document(project_id, timeline_id)?;
        let dir = project_path.join("timeline-exports");
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}_{}.json", timeline_id, random_hex(16)?));
        write_json(&path, &result.document)?;
        result.file.relative_path = relative_string(&project_path, &path)?;
        result.file.path = path;
        Ok(result)
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
        self.list_assets_with_filesystem_operations(
            project_id,
            include_rejected,
            include_trashed,
            scope,
        )
        .map(|(assets, _)| assets)
    }

    /// List assets and return the complete request-scoped filesystem-operation
    /// inventory used by performance tests, benchmarks, and API observability.
    ///
    /// The indexed `asset_json` envelope is authoritative on this clean path,
    /// including its embedded generation set and DB-backed poster URL.
    /// SceneWorks-owned mutations update the sidecar and index before returning.
    /// Tools that edit
    /// sidecars or generation-set JSON out of band must call
    /// [`Self::reindex_project`] (the REST equivalent is
    /// `POST /api/v1/projects/:project_id/reindex`) before expecting those edits
    /// in listings. This explicit contract includes deletion, same-size rewrites,
    /// and edits that preserve timestamps. Advertised posters are served from
    /// the same indexed DB row, so list requests do no poster filesystem work.
    pub fn list_assets_with_filesystem_operations(
        &self,
        project_id: &str,
        include_rejected: bool,
        include_trashed: bool,
        scope: AssetScope,
    ) -> ProjectStoreResult<(Vec<Value>, AssetListFilesystemOperations)> {
        let operation_scope = AssetListFilesystemOperationScope::begin();
        let result =
            self.list_assets_filtered(project_id, include_rejected, include_trashed, scope, None);
        let operations = operation_scope.finish();
        result.map(|assets| (assets, operations))
    }

    fn list_assets_filtered(
        &self,
        project_id: &str,
        include_rejected: bool,
        include_trashed: bool,
        scope: AssetScope,
        asset_type: Option<&str>,
    ) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        #[cfg(test)]
        run_ensure_ready_before_lock_hook(&project_path);
        let _project_guard = lock_project_files(&project_path);
        let mut connection = connect_project_db_ready_locked(&project_path)?;
        let total = connection.query_row("select count(*) from assets", [], |row| {
            row.get::<_, i64>(0)
        })?;
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
            if let Err(error) = reindex_project_path(project_id, &project_path, false) {
                tracing::warn!(
                    event = "asset_list_auto_reindex_failed",
                    project_id,
                    project = %project_path.display(),
                    error = %error,
                    "could not rebuild the asset index while listing; returning indexed rows"
                );
            }
            connection = connect_project_db(&project_path)?;
        }

        // The Library view hides Character Studio test outputs (sc-2024). Rows
        // with a null origin (should not occur after the schema-bump reindex)
        // fail open and stay visible.
        let origin_filter = match scope {
            AssetScope::All => " and (origin is null or origin != 'vector_workflow_intermediate')",
            AssetScope::Library => " and (origin is null or origin not in ('character_studio', 'vector_workflow_intermediate'))",
        };
        let mut statement = connection.prepare(&format!(
            "
            select id, sidecar_path, asset_json
              from assets
             where (?1 or rejected = 0)
               and (?2 or trashed = 0)
               and (?3 is null or type = ?3)
               {origin_filter}
             order by created_at desc
            "
        ))?;
        let rows = statement
            .query_map(
                params![include_rejected, include_trashed, asset_type],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;
        drop(statement);
        // HashSet dedup (was an O(n²) Vec linear scan) and a per-call
        // generation-set cache so a set's JSON is read once, not once per asset
        // in it (sc-4270 / F-CORE-10).
        let mut seen_asset_ids = std::collections::HashSet::new();
        let mut generation_sets = GenerationSetCache::default();
        let mut assets = Vec::new();
        for (row_id, sidecar_rel, asset_json) in rows {
            let Some(asset_json) = asset_json else {
                tracing::warn!(event = "asset_list_indexed_envelope_missing", asset_id = %row_id, "skipping asset row with no indexed envelope");
                continue;
            };
            let mut asset = match serde_json::from_str::<Value>(&asset_json) {
                Ok(asset) => asset,
                Err(error) => {
                    tracing::warn!(event = "asset_list_indexed_envelope_invalid", asset_id = %row_id, error = %error, "skipping malformed indexed asset envelope");
                    continue;
                }
            };
            if !indexed_asset_envelope_is_valid(&row_id, sidecar_rel.as_deref(), &asset) {
                tracing::warn!(event = "asset_list_indexed_envelope_invalid", asset_id = %row_id, "skipping invalid indexed asset envelope");
                continue;
            }
            if let Some(sidecar_rel) = sidecar_rel {
                if let Some(object) = asset.as_object_mut() {
                    object.insert("sidecarPath".to_owned(), Value::String(sidecar_rel));
                }
            }
            let asset = match hydrate_indexed_asset_cached(
                project_id,
                &project_path,
                asset,
                &mut generation_sets,
            ) {
                Ok(asset) => asset,
                Err(error) => {
                    tracing::warn!(event = "asset_list_indexed_envelope_hydration_failed", asset_id = %row_id, error = %error, "skipping asset whose indexed envelope could not be hydrated");
                    continue;
                }
            };
            if seen_asset_ids.insert(row_id) {
                assets.push(asset);
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

    /// The project's YuE2 score version/render/comparison records (sc-22997).
    pub fn yue2_score_store(
        &self,
        project_id: &str,
    ) -> ProjectStoreResult<crate::yue2_score::store::Yue2ScoreStore> {
        let project_path = self.find_project_path(project_id)?;
        Ok(crate::yue2_score::store::Yue2ScoreStore::new(
            project_path,
            project_id,
        ))
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
        crate::safe_path::resolve_local_file(&project_path, media_rel).map_err(|error| {
            ProjectStoreError::BadRequest(format!(
                "Asset media path is not a confined project file: {error}"
            ))
        })
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
        if !is_safe_id(track_id) {
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
        if !is_safe_id(track_id) {
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
        // `extra` is an object in the sidecar contract, and `provenance` is written into it, so a
        // provenance that is not an object is refused here rather than reshaped or ignored.
        //
        // It used to be written through verbatim, which made `extra` a scalar — and once sc-15949
        // had a second writer of that slot, it became a silent-loss path: a `provenance` of `42`,
        // `"editor"`, `[1,2]` or `null` would take the whole slot and throw away a workflow the
        // reader had already found in the file. Rejecting is the only answer that neither loses the
        // envelope nor quietly rewrites what the caller sent. `apps/rust-api/src/assets.rs` refuses
        // the same shape at the multipart boundary, so this is the belt to that braces; the API is
        // the only production caller.
        if upload
            .provenance
            .as_ref()
            .is_some_and(|provenance| !provenance.is_object())
        {
            return Err(ProjectStoreError::BadRequest(
                "Upload provenance must be a JSON object".to_owned(),
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
        let content_type = if Path::new(&upload.filename)
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("npy"))
        {
            crate::tensor_artifact::MIME.to_owned()
        } else {
            content_type
        };
        let tensor = if content_type == crate::tensor_artifact::MIME {
            Some(
                crate::tensor_artifact::inspect(&upload.source_path)
                    .map_err(ProjectStoreError::BadRequest)?,
            )
        } else {
            None
        };
        // SVG is active document syntax, not a generic upload format. Vector assets may enter
        // only through the worker-owned `vector_generate` sanitizer/rasterizer boundary, which
        // writes a canonical SVG plus a bounded PNG preview together (sc-22251).
        if content_type.eq_ignore_ascii_case("image/svg+xml")
            || Path::new(&upload.filename)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("svg"))
        {
            return Err(ProjectStoreError::BadRequest(
                "SVG uploads are not supported; use Vector Studio".to_owned(),
            ));
        }
        if !content_type.starts_with("image/")
            && !content_type.starts_with("video/")
            && !content_type.starts_with("audio/")
            && content_type != crate::tensor_artifact::MIME
        {
            return Err(ProjectStoreError::BadRequest(
                "Only image, video and audio uploads are supported".to_owned(),
            ));
        }

        // sc-18650: audio is normalized by a DIFFERENT rule than image/video, so it is dispatched
        // here rather than folded into `normalize_image_upload` (which passes every non-image
        // through untouched). See `normalize_audio_upload` for why an uploaded clip is always
        // converted rather than stored as sent.
        //
        // sc-6143: transcode a valid-but-unsupported image (AVIF/HEIC/HEIF/TIFF/BMP/GIF) to lossless
        // PNG before storing, so every downstream decode site, thumbnail, and preview reads a format
        // it supports. Supported formats and videos pass through unchanged.
        let normalized = if content_type.starts_with("audio/") {
            normalize_audio_upload(&upload.source_path, &upload_dir)?
        } else {
            normalize_image_upload(
                &upload.source_path,
                &content_type,
                &upload.filename,
                &upload_dir,
                // sc-15949: a shared image may carry the sanitized recipe that made it. Read it from
                // the bytes the user handed us, before any transcode can strip it.
                WorkflowScan::Read,
            )?
        };
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
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
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

        let media_type = media_type_for_mime(&content_type)?;
        let mut file = json!({
            "path": media_rel,
            "mimeType": content_type,
            "width": Value::Null,
            "height": Value::Null,
            "duration": Value::Null,
            "fps": Value::Null
        });
        // An audio upload's `file` block additionally carries the clip facts the audio tile, the
        // transport and the waveform all read — the SAME three fields `build_audio_sidecar_parts`
        // writes for an Audio Studio render, so the two doors an audio asset can come through
        // produce one shape (sc-18650). Without them the tile renders `0:00` and the waveform has
        // nothing to size against.
        //
        // MEASURED off the stored WAV, never trusted from the upload: the transcode above is what
        // decided them. They are added only for audio, so no image/video sidecar grows a key.
        if media_type == "audio" {
            if let Some(facts) = wav_facts(&media_path) {
                file["duration"] = json!(facts.duration_secs);
                file["sampleRate"] = json!(facts.sample_rate);
                file["channels"] = json!(facts.channels);
            }
        }
        let mut asset = json!({
            "schemaVersion": 1,
            "id": asset_id,
            "projectId": project_id,
            "generationSetId": Value::Null,
            "type": media_type,
            "displayName": display_name,
            "createdAt": created_at,
            // Manual imports are Library media in their own right (sc-2024).
            "origin": "upload",
            "file": file,
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
        if let Some(tensor) = tensor {
            asset["tensor"] = serde_json::to_value(&tensor)?;
            asset["file"]["contentHash"] = json!(format!("sha256:{}", tensor.sha256));
        }
        // Provenance (e.g. the Image Editor edit chain) rides the same top-level
        // `extra` slot generated assets use, so the Library/lineage UI can read it.
        //
        // sc-15949 shares that slot: an imported image that carried a workflow chunk records the
        // parsed, sanitized envelope under `extra.importedWorkflow`. The RECORD stays honest — the
        // stub recipe above is untouched and `recipe.mode` is still `"upload"`, because this image
        // WAS uploaded and was not generated here. sc-15952 offers the envelope as "Use this
        // recipe"; nothing here fabricates a `text_to_image` mode, a seed we did not run, or a
        // generation set that never existed.
        if let Some(object) = asset.as_object_mut() {
            // A non-object provenance was already refused at the top of this function, so this is
            // the only shape left.
            let provenance = upload.provenance.as_ref().and_then(Value::as_object);
            let mut extra = provenance.cloned().unwrap_or_default();
            // Cleared UNCONDITIONALLY, before the insert below and whether or not there is anything
            // to insert. `extra.importedWorkflow` is defined as the envelope THIS import read and
            // sanitized, so the key must be either ours or absent — never free JSON a caller put
            // under it in `provenance`.
            //
            // Inserting ours last is not enough on its own, and that was the bug: when the file
            // carries no chunk there is nothing to insert, and a caller-supplied
            // `provenance.importedWorkflow` was written through untouched — an unsanitized `model`,
            // an `advanced` map the allow-list would have emptied, and a `loras[].repo` of
            // `../../../../etc/passwd` in the one field sc-15950/sc-15952 turn into a cache path.
            // The guard has to be a removal, not an ordering.
            extra.remove(IMPORTED_WORKFLOW_KEY);
            // Upload/edit metadata is not authority for model identity or inherited obligations.
            extra.remove(crate::model_artifacts::provenance::MODEL_PROVENANCE_KEY);
            if let Some(workflow) = normalized
                .workflow
                .as_ref()
                .and_then(|share| serde_json::to_value(share).ok())
            {
                extra.insert(IMPORTED_WORKFLOW_KEY.to_owned(), workflow);
            }
            // An explicit `provenance` keeps its slot even when it reduces to nothing, because
            // `provenance: {}` wrote `extra: {}` before this story and "no behaviour change" means
            // the empty object too. With no provenance and no workflow, `extra` stays absent.
            if provenance.is_some() || !extra.is_empty() {
                object.insert("extra".to_owned(), Value::Object(extra));
            }
        }
        self.inherit_model_provenance(&project_path, &mut asset)?;
        let sidecar_path = media_path.with_extension("sceneworks.json");
        write_json(&sidecar_path, &asset)?;
        index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
        index_mutation.commit();
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
        // A preview path (vector / depth) is joined into the project path by every reader, so it
        // gets the same traversal guard as mediaPath (sc-25682).
        if let Some(preview_rel) = fact.pointer("/preview/path") {
            if !preview_rel.as_str().is_some_and(is_safe_relative_path) {
                return Err(ProjectStoreError::BadRequest(
                    "Invalid preview path".to_owned(),
                ));
            }
        }
        let mut asset = build_generated_asset_sidecar(project_id, job_id, generation_set_id, fact);
        self.inherit_model_provenance(&project_path, &mut asset)?;
        let media_path = project_path.join(media_rel);
        let sidecar_path = media_path.with_extension("sceneworks.json");
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
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
        index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
        index_mutation.commit();
        Ok(asset)
    }

    /// Keep immutable model ancestry through edits, media conversion, and multi-input generation,
    /// including producer types that do not themselves load a checkpoint. Flatten and deduplicate
    /// commitments rather than recursively copying entire parent assets or mutable recipes.
    fn inherit_model_provenance(
        &self,
        project_path: &Path,
        asset: &mut Value,
    ) -> ProjectStoreResult<()> {
        let parents = asset
            .pointer("/lineage/parents")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut inherited = std::collections::BTreeMap::new();
        for provenance in asset
            .pointer("/lineage/modelProvenance")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            inherited.insert(serde_json::to_string(provenance)?, provenance.clone());
        }
        for parent in parents.iter().filter_map(Value::as_str) {
            let path = match self.find_asset_sidecar(project_path, parent) {
                Ok(path) => path,
                Err(ProjectStoreError::NotFound(_)) => continue,
                Err(error) => return Err(error),
            };
            let source: Value = read_json(&path)?;
            if let Some(provenance) = source.get("modelProvenance") {
                inherited.insert(serde_json::to_string(provenance)?, provenance.clone());
            }
            for provenance in source
                .pointer("/lineage/modelProvenance")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                inherited.insert(serde_json::to_string(provenance)?, provenance.clone());
            }
        }
        if !inherited.is_empty() {
            asset["lineage"]["modelProvenance"] = Value::Array(inherited.into_values().collect());
        }
        Ok(())
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
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        let ingested = crate::media_convert::ingest_image_with_naming(
            &canonical_source,
            &dir.join(&asset_id),
            crate::media_convert::ImageCopyNaming::Canonical,
        )?;
        let media_path = ingested.path;
        let mime = ingested.mime;
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
        index_asset(
            GLOBAL_KEYPOINTS_PROJECT_ID,
            &project_path,
            &asset,
            Some(&sidecar_path),
        )?;
        index_mutation.commit();
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
        let assets = self.list_assets_filtered(
            GLOBAL_KEYPOINTS_PROJECT_ID,
            false,
            false,
            AssetScope::All,
            Some("keypoint"),
        )?;
        Ok(assets.iter().map(keypoint_asset_to_preset).collect())
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
        // The id comes from the worker-reported fact and is joined into the
        // generation-sets path below, so charset-check it before path use — the
        // worker->API boundary must not write outside the project root
        // (sc-13582; mirrors the persist_generated_asset guards, sc-5721/sc-4211).
        if !is_safe_id(id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid generation set id".to_owned(),
            ));
        }
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
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        let dir = project_path.join("generation-sets");
        fs::create_dir_all(&dir)?;
        write_json(&dir.join(format!("{id}.json")), &record)?;
        maybe_fail_asset_index_db_mutation()?;
        let mut connection = connect_project_db(&project_path)?;
        apply_project_migrations(&connection)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "
            insert or replace into generation_sets (id, mode, model, prompt, created_at, job_id)
            values (?1, ?2, ?3, ?4, ?5, ?6)
            ",
            params![
                id,
                optional_str(&record, "mode").unwrap_or("unknown"),
                optional_str(&record, "model").unwrap_or("unknown"),
                optional_str(&record, "prompt").unwrap_or(""),
                optional_str(&record, "createdAt").unwrap_or(""),
                optional_str(&record, "jobId"),
            ],
        )?;
        let indexed_assets = {
            let mut statement = transaction
                .prepare("select id, asset_json from assets where generation_set_id = ?1")?;
            let rows = statement
                .query_map(params![id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        for (asset_id, asset_json) in indexed_assets {
            let Some(asset_json) = asset_json else {
                continue;
            };
            let Ok(mut asset) = serde_json::from_str::<Value>(&asset_json) else {
                continue;
            };
            let Some(object) = asset.as_object_mut() else {
                continue;
            };
            object.insert("generationSet".to_owned(), record.clone());
            transaction.execute(
                "update assets set asset_json = ?1 where id = ?2",
                params![serde_json::to_string(&asset)?, asset_id],
            )?;
        }
        transaction.commit()?;
        index_mutation.commit();
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
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        write_json(&sidecar_path, &asset)?;
        index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
        index_mutation.commit();
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
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        write_json(&sidecar_path, &asset)?;
        index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
        index_mutation.commit();
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
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        let mut moved = Vec::new();
        for member_id in upscale_lineage_group(&project_path, asset_id)? {
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
            index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
            // Drop the asset from every character's references[] (character sidecars + index).
            character_store.remove_asset_references(&member_id)?;
            moved.push(normalize_asset(project_id, &project_path, &sidecar_path)?);
        }
        index_mutation.commit();
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
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        let mut moved = Vec::new();
        for member_id in upscale_lineage_group(&project_path, asset_id)? {
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
            index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
            // Leave no curated membership behind: the move detaches the asset from every
            // character's references[] (including the target's — the Approved set is
            // hand-curated and a bulk move must not populate it).
            character_store.remove_asset_references(&member_id)?;
            moved.push(normalize_asset(project_id, &project_path, &sidecar_path)?);
        }
        index_mutation.commit();
        Ok(Value::Array(moved))
    }

    pub fn get_asset(&self, project_id: &str, asset_id: &str) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    /// Mark the raster produced by a disclosed prompt-to-vector workflow as hidden but retained.
    /// The exact workflow and parent ids are the deletion authority used by cleanup; callers can
    /// never turn an arbitrary project asset into a cleanup target by naming only its id.
    pub fn mark_vector_workflow_intermediate(
        &self,
        project_id: &str,
        asset_id: &str,
        workflow_id: &str,
        parent_job_id: &str,
        child_job_id: &str,
    ) -> ProjectStoreResult<Value> {
        if !is_safe_id(workflow_id) || !is_safe_id(parent_job_id) || !is_safe_id(child_job_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid vector workflow ownership id".to_owned(),
            ));
        }
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let mut asset = read_json(&sidecar_path)?;
        let object = asset.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
        })?;
        if object.get("type").and_then(Value::as_str) != Some("image") {
            return Err(ProjectStoreError::BadRequest(
                "Vector workflow intermediate must be a raster image".to_owned(),
            ));
        }
        object.insert(
            "origin".to_owned(),
            Value::String("vector_workflow_intermediate".to_owned()),
        );
        let extra = object
            .entry("extra")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Asset extra must be an object".to_owned())
            })?;
        extra.insert(
            "vectorWorkflow".to_owned(),
            json!({
                "role": "retained_intermediate",
                "publication": "unpublished",
                "workflowId": workflow_id,
                "parentJobId": parent_job_id,
                "childJobId": child_job_id,
                "hidden": true,
            }),
        );
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        write_json(&sidecar_path, &asset)?;
        index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
        index_mutation.commit();
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    /// Permanently purge a workflow-owned intermediate only when every ownership fact matches.
    /// Returns false for a user asset, another workflow's asset, or an already-cleaned id.
    pub fn purge_vector_workflow_intermediate(
        &self,
        project_id: &str,
        asset_id: &str,
        workflow_id: &str,
        parent_job_id: &str,
    ) -> ProjectStoreResult<bool> {
        let asset = match self.get_asset(project_id, asset_id) {
            Ok(asset) => asset,
            Err(ProjectStoreError::NotFound(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        let owned = asset.get("origin").and_then(Value::as_str)
            == Some("vector_workflow_intermediate")
            && asset
                .pointer("/extra/vectorWorkflow/workflowId")
                .and_then(Value::as_str)
                == Some(workflow_id)
            && asset
                .pointer("/extra/vectorWorkflow/parentJobId")
                .and_then(Value::as_str)
                == Some(parent_job_id)
            && asset
                .pointer("/extra/vectorWorkflow/role")
                .and_then(Value::as_str)
                == Some("retained_intermediate")
            && asset
                .pointer("/extra/vectorWorkflow/publication")
                .and_then(Value::as_str)
                == Some("unpublished");
        if !owned {
            return Ok(false);
        }
        self.purge_asset(project_id, asset_id, true)?;
        Ok(true)
    }

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

    pub fn index_asset_sidecar(
        &self,
        project_id: &str,
        sidecar_path: &Path,
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let asset = read_json(sidecar_path)?;
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        index_asset(project_id, &project_path, &asset, Some(sidecar_path))?;
        index_mutation.commit();
        Ok(())
    }

    /// Persist a native worker-produced asset sidecar and recipe, then index the
    /// envelope as one repair-guarded project mutation.
    ///
    /// Native workers must use this instead of writing the sidecar and later
    /// calling [`Self::index_asset_sidecar`]: the durable marker and reentrant
    /// project lock begin before either JSON write and remain owned until the
    /// separate SQLite transaction commits.
    pub fn persist_native_asset_sidecar(
        &self,
        project_id: &str,
        sidecar_path: &Path,
        mut asset: Value,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_rel = relative_string(&project_path, sidecar_path)?;
        if !is_safe_relative_path(&sidecar_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Asset sidecar path must be project-relative".to_owned(),
            ));
        }
        self.inherit_model_provenance(&project_path, &mut asset)?;
        let asset_id = required_str(&asset, "id")?;
        if !is_safe_id(asset_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid native asset id".to_owned(),
            ));
        }
        let recipe = asset.get("recipe").ok_or_else(|| {
            ProjectStoreError::BadRequest("Native asset recipe is required".to_owned())
        })?;
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        write_json(sidecar_path, &asset)?;
        write_json(
            &project_path
                .join("recipes")
                .join(format!("{asset_id}.recipe.json")),
            recipe,
        )?;
        index_asset(project_id, &project_path, &asset, Some(sidecar_path))?;
        index_mutation.commit();
        Ok(asset)
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

        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        let trash_dir = project_path.join("trash").join(asset_id);
        fs::create_dir_all(&trash_dir)?;
        remove_sibling_poster_no_symlinks(&project_path, &media_rel)?;
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
        index_asset(
            project_id,
            &project_path,
            &asset,
            Some(&trashed_sidecar_path),
        )?;
        index_mutation.commit();
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
        let sidecar_rel = relative_string(&project_path, &sidecar_path)?;
        if !is_safe_relative_path(&sidecar_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Asset sidecar path must be project-relative".to_owned(),
            ));
        }
        let sidecar_path = confined_deletion_target(&project_path, &sidecar_rel, "Asset sidecar")?;
        let asset = read_json(&sidecar_path)?;
        let media_rel = asset
            .pointer("/file/path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        // The sidecar is durable, externally writable project state. Validate its
        // media path before constructing any deletion or trash target from it.
        if !media_rel.is_empty() && !is_safe_relative_path(media_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Asset media path must be project-relative".to_owned(),
            ));
        }
        let media_path = if media_rel.is_empty() {
            None
        } else {
            Some(confined_deletion_target(
                &project_path,
                media_rel,
                "Asset media",
            )?)
        };
        // A separately stored derived preview (vector PNG, depth preview — sc-25682) is part of the
        // asset's footprint: soft delete leaves it in place for the trashed sidecar to keep
        // rendering, and purge removes it with the media instead of orphaning it.
        let preview_rel = asset
            .pointer("/preview/path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let preview_path = if preview_rel.is_empty() || !is_safe_relative_path(preview_rel) {
            None
        } else {
            Some(confined_deletion_target(
                &project_path,
                preview_rel,
                "Asset preview",
            )?)
        };
        let project_root = fs::canonicalize(&project_path)?;
        let trash_dir = project_root.join("trash");
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
            let mut targets: Vec<PathBuf> = if let Some(dir) = asset_trash_dir.clone() {
                vec![dir]
            } else {
                let mut targets = Vec::new();
                if let Some(media_path) = media_path
                    .as_ref()
                    .filter(|path| path.exists() && path.is_file())
                {
                    targets.push(media_path.clone());
                }
                if sidecar_path.exists() {
                    targets.push(sidecar_path.clone());
                }
                targets
            };
            if let Some(preview_path) = preview_path
                .as_ref()
                .filter(|path| path.exists() && path.is_file())
            {
                targets.push(preview_path.clone());
            }
            let index_mutation = AssetIndexMutation::begin(&project_path)?;
            remove_sibling_poster_no_symlinks(&project_path, media_rel)?;
            if !targets.is_empty() && trash::delete_all(&targets).is_err() {
                // The trash API does not guarantee an atomic multi-target operation.
                // Keep the marker so the next read repairs any partial deletion before
                // serving indexed rows.
                return Ok(AssetMutationResult {
                    id: asset_id.to_owned(),
                    status: "trash_unavailable".to_owned(),
                });
            }
            CharacterStore::new(&self.data_dir, project_path.clone())
                .remove_asset_references(asset_id)?;
            purge_asset_record(&project_path, asset_id)?;
            index_mutation.commit();
            return Ok(AssetMutationResult {
                id: asset_id.to_owned(),
                status: "purged".to_owned(),
            });
        }
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        remove_sibling_poster_no_symlinks(&project_path, media_rel)?;
        CharacterStore::new(&self.data_dir, project_path.clone())
            .remove_asset_references(asset_id)?;
        if let Some(media_path) = media_path.filter(|path| path.exists() && path.is_file()) {
            fs::remove_file(media_path)?;
        }
        if let Some(preview_path) = preview_path.filter(|path| path.exists() && path.is_file()) {
            fs::remove_file(preview_path)?;
        }
        remove_asset_sidecar(&sidecar_path)?;
        if let Some(parent) = asset_trash_dir {
            fs::remove_dir_all(parent).ok();
        }
        purge_asset_record(&project_path, asset_id)?;
        index_mutation.commit();
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

    fn ensure_data_dirs(&self) -> ProjectStoreResult<()> {
        fs::create_dir_all(self.projects_dir())?;
        for folder in ["models", "loras", "cache"] {
            fs::create_dir_all(self.data_dir.join(folder))?;
        }
        Ok(())
    }

    fn load_registry(
        &self,
        registry_cache: &mut RegistryCache,
    ) -> ProjectStoreResult<Vec<RegistryItem>> {
        let registry_path = self.registry_path();
        let fingerprint = registry_fingerprint(&registry_path)?;
        if registry_cache.loaded && registry_cache.fingerprint == fingerprint {
            return Ok(registry_cache.items.clone());
        }
        let items = if fingerprint.is_some() {
            #[cfg(test)]
            {
                registry_cache.disk_reads += 1;
            }
            let payload = read_registry_payload(&registry_path)?;
            serde_json::from_str(&payload)?
        } else {
            Vec::new()
        };
        registry_cache.fingerprint = fingerprint;
        registry_cache.items = items.clone();
        registry_cache.loaded = true;
        Ok(items)
    }

    fn save_registry(
        &self,
        projects: &[RegistryItem],
        registry_cache: &mut RegistryCache,
    ) -> ProjectStoreResult<()> {
        fs::create_dir_all(&self.data_dir)?;
        // `write_json` stages to a unique temp and renames atomically (sc-1633),
        // so the registry write no longer needs its own intermediate temp file.
        let registry_path = self.registry_path();
        let registry_value = serde_json::to_value(projects)?;
        ensure_registry_serialized_size(&registry_value)?;
        write_json(&registry_path, &registry_value)?;
        registry_cache.fingerprint = registry_fingerprint(&registry_path)?;
        registry_cache.items = projects.to_vec();
        registry_cache.loaded = true;
        Ok(())
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
        let mut registry_cache = self.lock.lock();
        for item in self.load_registry(&mut registry_cache)? {
            if item.id.as_deref() == Some(project_id) {
                let Some(path) = item.path else {
                    break;
                };
                let project_path = PathBuf::from(path);
                record_asset_list_filesystem_operation(AssetListFilesystemOperation::PathStat);
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
///
/// v6: sc-14787 — rebuilt indexed asset envelopes for the authoritative clean
/// path and its durable dirty-marker repair contract.
///
/// v7: sc-14799 — materializes generation-set and poster hydration in each
/// indexed asset envelope, removing per-row filesystem reads from clean lists.
///
/// v8: sc-14799 review convergence stores poster bytes and their digest in
/// `project.db`, so advertised posters need no clean-list filesystem scan.
const PROJECT_SCHEMA_VERSION: i64 = 8;
const ASSET_INDEX_VERSION_KEY: &str = "assetIndexVersion";
const ASSET_INDEX_DIRTY_MARKER: &str = ".asset-index-dirty";

thread_local! {
    static ACTIVE_ASSET_INDEX_MUTATIONS: RefCell<Vec<PathBuf>> = const { RefCell::new(Vec::new()) };
}

fn current_thread_owns_asset_index_mutation(project_path: &Path) -> bool {
    ACTIVE_ASSET_INDEX_MUTATIONS.with(|active| {
        active
            .borrow()
            .iter()
            .any(|active_path| active_path == project_path)
    })
}

/// Durable guard for a filesystem mutation whose indexed envelope is committed
/// in a separate SQLite transaction. The marker is written before the first
/// media/sidecar change and removed only after the DB transaction commits.
/// Any error or process interruption leaves it behind, so the next indexed read
/// rebuilds from disk instead of serving a stale envelope indefinitely.
pub(crate) struct AssetIndexMutation {
    project_path: PathBuf,
    marker_path: PathBuf,
    _project_guard: ReentrantMutexGuard<'static, ()>,
}

impl AssetIndexMutation {
    pub(crate) fn begin(project_path: &Path) -> ProjectStoreResult<Self> {
        let project_guard = lock_project_files(project_path);
        // Never let a new mutation overwrite and later clear a marker left by a
        // prior failed mutation. Repair that earlier filesystem/DB split first.
        ensure_project_db_ready(project_path)?;
        Self::write_marker(project_path, project_guard)
    }

    fn begin_reindex(project_path: &Path) -> ProjectStoreResult<Self> {
        let project_guard = lock_project_files(project_path);
        Self::write_marker(project_path, project_guard)
    }

    fn write_marker(
        project_path: &Path,
        project_guard: ReentrantMutexGuard<'static, ()>,
    ) -> ProjectStoreResult<Self> {
        let marker_path = project_path.join(ASSET_INDEX_DIRTY_MARKER);
        record_asset_list_filesystem_operation(AssetListFilesystemOperation::IndexMarkerWrite);
        write_json(
            &marker_path,
            &json!({
                "schemaVersion": 1,
                "reason": "asset-filesystem-mutation"
            }),
        )?;
        ACTIVE_ASSET_INDEX_MUTATIONS.with(|active| {
            active.borrow_mut().push(project_path.to_path_buf());
        });
        Ok(Self {
            project_path: project_path.to_path_buf(),
            marker_path,
            _project_guard: project_guard,
        })
    }

    pub(crate) fn commit(self) {
        clear_asset_index_dirty_marker(&self.marker_path);
    }
}

impl Drop for AssetIndexMutation {
    fn drop(&mut self) {
        ACTIVE_ASSET_INDEX_MUTATIONS.with(|active| {
            let mut active = active.borrow_mut();
            if let Some(index) = active
                .iter()
                .rposition(|active_path| active_path == &self.project_path)
            {
                active.remove(index);
            }
        });
    }
}

fn clear_asset_index_dirty_marker(marker_path: &Path) {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::IndexMarkerRemove);
    if let Err(error) = fs::remove_file(marker_path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::debug!(
                event = "asset_index_dirty_marker_remove_failed",
                marker = %marker_path.display(),
                error = %error,
                "asset index is committed but its repair marker remains; next read will reindex"
            );
        }
    }
}

fn asset_index_is_dirty(project_path: &Path) -> bool {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::IndexMarkerRead);
    project_path.join(ASSET_INDEX_DIRTY_MARKER).exists()
}

#[cfg(test)]
type EnsureReadyBeforeLockHook = Option<(PathBuf, std::sync::Arc<std::sync::Barrier>)>;

#[cfg(test)]
fn ensure_ready_before_lock_hooks() -> &'static Mutex<EnsureReadyBeforeLockHook> {
    static HOOK: std::sync::OnceLock<Mutex<EnsureReadyBeforeLockHook>> = std::sync::OnceLock::new();
    HOOK.get_or_init(|| Mutex::new(None))
}

#[cfg(test)]
fn install_ensure_ready_before_lock_hook(
    project_path: PathBuf,
    barrier: std::sync::Arc<std::sync::Barrier>,
) {
    *ensure_ready_before_lock_hooks().lock() = Some((project_path, barrier));
}

#[cfg(test)]
fn run_ensure_ready_before_lock_hook(project_path: &Path) {
    let barrier = {
        let mut hook = ensure_ready_before_lock_hooks().lock();
        match hook.as_ref() {
            Some((hook_path, _)) if hook_path == project_path => {
                hook.take().map(|(_, barrier)| barrier)
            }
            _ => None,
        }
    };
    if let Some(barrier) = barrier {
        barrier.wait();
    }
}

#[cfg(test)]
thread_local! {
    static FAIL_NEXT_ASSET_INDEX_DB_MUTATION: Cell<bool> = const { Cell::new(false) };
    static FAIL_NEXT_ASSET_SIDECAR_REMOVE: Cell<bool> = const { Cell::new(false) };
    static FAIL_NEXT_UPSCALE_LINEAGE_DB_READ: Cell<bool> = const { Cell::new(false) };
    static FAIL_NEXT_ASSET_POSTER_DB_READ: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
fn fail_next_asset_index_db_mutation() {
    FAIL_NEXT_ASSET_INDEX_DB_MUTATION.with(|fail| fail.set(true));
}

#[cfg(test)]
fn fail_next_asset_sidecar_remove() {
    FAIL_NEXT_ASSET_SIDECAR_REMOVE.with(|fail| fail.set(true));
}

#[cfg(test)]
fn fail_next_upscale_lineage_db_read() {
    FAIL_NEXT_UPSCALE_LINEAGE_DB_READ.with(|fail| fail.set(true));
}

#[cfg(test)]
pub(crate) fn fail_next_asset_poster_db_read() {
    FAIL_NEXT_ASSET_POSTER_DB_READ.with(|fail| fail.set(true));
}

#[cfg(test)]
fn maybe_fail_upscale_lineage_db_read() -> ProjectStoreResult<()> {
    if FAIL_NEXT_UPSCALE_LINEAGE_DB_READ.with(|fail| fail.replace(false)) {
        return Err(ProjectStoreError::Sqlite(rusqlite::Error::InvalidQuery));
    }
    Ok(())
}

#[cfg(not(test))]
fn maybe_fail_upscale_lineage_db_read() -> ProjectStoreResult<()> {
    Ok(())
}

#[cfg(test)]
pub(crate) fn maybe_fail_asset_poster_db_read() -> ProjectStoreResult<()> {
    if FAIL_NEXT_ASSET_POSTER_DB_READ.with(|fail| fail.replace(false)) {
        return Err(ProjectStoreError::Sqlite(rusqlite::Error::InvalidQuery));
    }
    Ok(())
}

#[cfg(not(test))]
pub(crate) fn maybe_fail_asset_poster_db_read() -> ProjectStoreResult<()> {
    Ok(())
}

#[cfg(test)]
fn maybe_fail_asset_index_db_mutation() -> ProjectStoreResult<()> {
    let should_fail = FAIL_NEXT_ASSET_INDEX_DB_MUTATION.with(|fail| fail.replace(false));
    if should_fail {
        return Err(ProjectStoreError::Io(std::io::Error::other(
            "injected asset index DB mutation failure",
        )));
    }
    Ok(())
}

#[cfg(test)]
fn remove_asset_sidecar(path: &Path) -> ProjectStoreResult<()> {
    let should_fail = FAIL_NEXT_ASSET_SIDECAR_REMOVE.with(|fail| fail.replace(false));
    if should_fail {
        return Err(ProjectStoreError::Io(std::io::Error::other(
            "injected asset sidecar removal failure",
        )));
    }
    fs::remove_file(path)?;
    Ok(())
}

#[cfg(not(test))]
fn remove_asset_sidecar(path: &Path) -> ProjectStoreResult<()> {
    fs::remove_file(path)?;
    Ok(())
}

#[cfg(not(test))]
fn maybe_fail_asset_index_db_mutation() -> ProjectStoreResult<()> {
    Ok(())
}

fn project_schema_version(connection: &Connection) -> ProjectStoreResult<i64> {
    Ok(connection.query_row("pragma user_version", [], |row| row.get(0))?)
}

pub fn apply_project_migrations(connection: &Connection) -> ProjectStoreResult<()> {
    if project_schema_version(connection)? >= PROJECT_SCHEMA_VERSION {
        return Ok(());
    }
    // Serialize migrations across processes, then re-read: another opener may
    // have finished the entire migration while this connection waited for the lock.
    let transaction = Transaction::new_unchecked(connection, TransactionBehavior::Immediate)?;
    if project_schema_version(&transaction)? >= PROJECT_SCHEMA_VERSION {
        return Ok(());
    }
    migrate_project_schema(&transaction)?;
    transaction.commit()?;
    Ok(())
}

fn migrate_project_schema(connection: &Connection) -> ProjectStoreResult<()> {
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
    // Normalized sidecar envelope + lineage keys make list hydration and
    // upscale-fold traversal indexed operations. Existing projects are
    // backfilled by the version-bump reindex below.
    ensure_column(connection, "assets", "asset_json", "text")?;
    ensure_column(connection, "assets", "source_asset_id", "text")?;
    ensure_column(connection, "assets", "upscaled_from_asset_id", "text")?;
    ensure_column(connection, "assets", "sidecar_size", "integer")?;
    ensure_column(connection, "assets", "sidecar_modified_ns", "text")?;
    ensure_column(connection, "assets", "sidecar_sha256", "text")?;
    ensure_column(connection, "assets", "poster_bytes", "blob")?;
    ensure_column(connection, "assets", "poster_sha256", "text")?;
    connection.execute_batch(
        "
        create index if not exists idx_assets_source_asset
          on assets(source_asset_id);
        create index if not exists idx_assets_upscaled_from
          on assets(upscaled_from_asset_id);
        ",
    )?;
    apply_character_migrations(connection)?;
    apply_training_dataset_migrations(connection)?;
    crate::voice_store::apply_voice_migrations(connection)?;
    // Pragma assignment cannot be parameterized; the version is a trusted const.
    connection.execute_batch(&format!("pragma user_version = {PROJECT_SCHEMA_VERSION}"))?;
    Ok(())
}

pub fn ensure_project_db_ready(project_path: &Path) -> ProjectStoreResult<()> {
    #[cfg(test)]
    run_ensure_ready_before_lock_hook(project_path);
    // Marker inspection, rebuild, orphan pruning, and marker clear must be one
    // serialized project operation. Mutation guards use the same reentrant lock,
    // so a list cannot repair and clear the marker owned by an in-flight write.
    let _project_guard = lock_project_files(project_path);
    connect_project_db_ready_locked(project_path).map(drop)
}

/// Return one ready connection while the caller holds the project lock.
///
/// The clean path keeps and reuses the connection opened for migrations/index
/// readiness, so an asset list performs one `project.db` open. A version bump or
/// dirty-marker repair rebuilds transactionally and then opens the rebuilt DB;
/// repair is intentionally outside the clean-path operation-count contract.
fn connect_project_db_ready_locked(project_path: &Path) -> ProjectStoreResult<Connection> {
    let connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    let indexed_version = connection
        .query_row(
            "select value from project_metadata where key = ?1",
            params![ASSET_INDEX_VERSION_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    // The schema stamp and the derived-data rebuild are separate operations.
    // Record completion only in the rebuild transaction so a failed reindex is
    // retried even though PRAGMA user_version already reached this version.
    let asset_index_dirty = asset_index_is_dirty(project_path)
        && !current_thread_owns_asset_index_mutation(project_path);
    if indexed_version.as_deref() != Some(&PROJECT_SCHEMA_VERSION.to_string()) || asset_index_dirty
    {
        drop(connection);
        // Legacy/version-bump rebuilds continue to recover sidecars even when
        // their media is temporarily unavailable. Only dirty-mutation repair
        // prunes missing-media rows before clearing the marker, closing the
        // partial-purge resurrection window.
        let project_id = read_project_summary(project_path)?.id;
        reindex_project_path(&project_id, project_path, asset_index_dirty)?;
        return connect_project_db(project_path);
    }
    Ok(connection)
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
        record_asset_list_filesystem_operation(AssetListFilesystemOperation::SidecarRead);
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
const UPSCALE_LINEAGE_QUERY: &str = "
    with recursive connected(id) as (
      values (?1)
      union
      select parent.upscaled_from_asset_id
        from assets parent join connected c on parent.id = c.id
       where parent.upscaled_from_asset_id is not null
      union
      select child.id
        from assets child join connected c on child.upscaled_from_asset_id = c.id
    )
    select id from connected
";

fn upscale_lineage_group(project_path: &Path, asset_id: &str) -> ProjectStoreResult<Vec<String>> {
    let mut group = vec![asset_id.to_owned()];
    let connection = connect_project_db(project_path)?;
    maybe_fail_upscale_lineage_db_read()?;
    let mut statement = connection.prepare(UPSCALE_LINEAGE_QUERY)?;
    let rows = statement.query_map(params![asset_id], |row| row.get::<_, String>(0))?;
    for id in rows {
        let id = id?;
        if id != asset_id && !group.contains(&id) {
            group.push(id);
        }
    }
    Ok(group)
}

fn reindex_project_path(
    project_id: &str,
    project_path: &Path,
    prune_orphans: bool,
) -> ProjectStoreResult<ReindexCounts> {
    let index_mutation = AssetIndexMutation::begin_reindex(project_path)?;
    // sc-10117: rewrite inline-upscaled sidecars missing their fold link BEFORE indexing, so the
    // rebuilt index — and the Library, which reads the sidecars directly — sees the healed lineage.
    // Additive + idempotent, so it is safe to run on every reindex.
    backfill_upscale_variant_lineage(project_path)?;

    let mut connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute("delete from assets", [])?;
    transaction.execute("delete from generation_sets", [])?;
    transaction.execute("delete from timelines", [])?;
    clear_character_tables(&transaction)?;

    let mut counts = ReindexCounts::default();
    for sidecar_path in asset_sidecars(project_path)? {
        record_asset_list_filesystem_operation(AssetListFilesystemOperation::SidecarRead);
        let asset = match read_json(&sidecar_path) {
            Ok(asset) => asset,
            Err(error) => {
                tracing::warn!(event = "asset_reindex_sidecar_invalid", sidecar = %sidecar_path.display(), error = %error, "skipping malformed asset sidecar during reindex");
                continue;
            }
        };
        if let Err(error) = index_asset_on_connection(
            &transaction,
            project_id,
            project_path,
            &asset,
            Some(&sidecar_path),
        ) {
            match error {
                // The caller's rebuild transaction is the SQLite invariant: never
                // turn a database failure into a superficially successful index.
                ProjectStoreError::Sqlite(_) => return Err(error),
                error => {
                    tracing::warn!(event = "asset_reindex_sidecar_invalid", sidecar = %sidecar_path.display(), error = %error, "skipping malformed asset sidecar during reindex");
                    continue;
                }
            }
        }
        counts.assets += 1;
    }

    let generation_set_dir = project_path.join("generation-sets");
    for entry in read_dir_paths(&generation_set_dir)? {
        if entry.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        record_asset_list_filesystem_operation(AssetListFilesystemOperation::GenerationSetRead);
        let generation_set = match read_json(&entry) {
            Ok(generation_set) => generation_set,
            Err(error) => {
                tracing::warn!(event = "generation_set_reindex_sidecar_invalid", sidecar = %entry.display(), error = %error, "skipping malformed generation set sidecar during reindex");
                continue;
            }
        };
        let id = match required_str(&generation_set, "id") {
            Ok(id) => id,
            Err(error) => {
                tracing::warn!(event = "generation_set_reindex_sidecar_invalid", sidecar = %entry.display(), error = %error, "skipping malformed generation set sidecar during reindex");
                continue;
            }
        };
        transaction.execute(
            "
            insert or replace into generation_sets (id, mode, model, prompt, created_at, job_id)
            values (?1, ?2, ?3, ?4, ?5, ?6)
            ",
            params![
                id,
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
        record_asset_list_filesystem_operation(AssetListFilesystemOperation::TimelineRead);
        let timeline = match read_json(&entry) {
            Ok(timeline) => timeline,
            Err(error) => {
                tracing::warn!(event = "timeline_reindex_sidecar_invalid", sidecar = %entry.display(), error = %error, "skipping malformed timeline sidecar during reindex");
                continue;
            }
        };
        let id = match required_str(&timeline, "id") {
            Ok(id) => id,
            Err(error) => {
                tracing::warn!(event = "timeline_reindex_sidecar_invalid", sidecar = %entry.display(), error = %error, "skipping malformed timeline sidecar during reindex");
                continue;
            }
        };
        let rel_path = relative_string(project_path, &entry)?;
        transaction.execute(
            "
            insert or replace into timelines (
              id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ",
            params![
                id,
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
    transaction.execute(
        "insert or replace into project_metadata (key, value) values (?1, ?2)",
        params![ASSET_INDEX_VERSION_KEY, PROJECT_SCHEMA_VERSION.to_string()],
    )?;

    transaction.commit()?;
    if prune_orphans {
        // Keep the dirty marker and project lock until missing-media rows have
        // been removed and their surviving sidecars quarantined. Clearing it
        // after only the rebuild could resurrect a partially purged asset after
        // a crash.
        let pruned = prune_orphaned_asset_rows(project_path)?;
        counts.assets = counts.assets.saturating_sub(pruned as u32);
    }
    index_mutation.commit();
    Ok(counts)
}

/// Every aspect ratio a timeline may declare, with the frame it renders at. One declaration
/// (sc-22710): a caller that must CHOOSE a ratio for footage of some other shape — the film
/// harness picks one before it creates a timeline — reads this list rather than hand-copying it
/// and drifting into a 400 here.
pub const TIMELINE_ASPECT_RATIOS: &[(&str, u32, u32)] = &[
    ("16:9", 1280, 720),
    ("9:16", 720, 1280),
    ("1:1", 1024, 1024),
];

fn timeline_dimensions(aspect_ratio: &str) -> ProjectStoreResult<(u32, u32)> {
    TIMELINE_ASPECT_RATIOS
        .iter()
        .find(|(name, _, _)| *name == aspect_ratio)
        .map(|(_, width, height)| (*width, *height))
        .ok_or_else(|| {
            ProjectStoreError::BadRequest(
                "Aspect ratio must be one of 16:9, 9:16, or 1:1".to_owned(),
            )
        })
}

/// Roles a timeline track may declare. `kind` says how a track is rendered (picture, overlay,
/// mixed audio); `role` says what it CARRIES, which is what lets dialogue, ambience and music be
/// three independently gained and muted buses rather than one undifferentiated "Audio" lane
/// (sc-22712). The exporter mixes every non-muted `kind: "audio"` track regardless of role — role
/// is descriptive, never a routing switch — so an unrecognised bed still reaches the MP4.
pub const TIMELINE_TRACK_ROLES: &[&str] = &[
    "picture", "overlay", "dialogue", "ambience", "music", "sfx", "sound",
];

/// What a picture item's own (generated) audio does in the mix. Defaults to `mute`, which is the
/// only default that cannot silently double a placed dialogue clip (sc-22712): a generated take
/// whose model spoke the line and a recorded dialogue clip for the same beat would otherwise both
/// land in the mix, and nothing in the timeline would say which one the editor meant.
pub const TIMELINE_GENERATED_AUDIO: &[&str] = &["include", "mute"];

/// The role a track's `kind` implies when a document does not declare one. Keeps every timeline
/// written before sc-22712 loading unchanged: its three default tracks become picture / overlay /
/// sound, and `sound` is mixed exactly like a named bed.
fn default_track_role(kind: &str) -> &'static str {
    match kind {
        "overlay" => "overlay",
        "audio" => "sound",
        _ => "picture",
    }
}

fn default_timeline_tracks() -> Value {
    json!([
        {"id": "track_main", "name": "Main", "kind": "video", "role": "picture", "locked": false, "muted": false, "gain": 1.0, "items": []},
        {"id": "track_overlay", "name": "Overlay", "kind": "overlay", "role": "overlay", "locked": false, "muted": false, "gain": 1.0, "items": []},
        {"id": "track_audio", "name": "Audio", "kind": "audio", "role": "sound", "locked": false, "muted": false, "gain": 1.0, "items": []}
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
    let kind = required_str(track, "kind")?.to_owned();
    let object = track.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline track must be an object".to_owned())
    })?;
    object
        .entry("locked".to_owned())
        .or_insert_with(|| Value::Bool(false));
    object
        .entry("muted".to_owned())
        .or_insert_with(|| Value::Bool(false));
    // sc-22712: `gain` is the track's bus fader, and `role` names the bus. Both are defaulted from
    // what the document already says, so a timeline saved before this existed round-trips to the
    // same audible result (unity gain, role implied by kind) rather than being rejected or muted.
    object
        .entry("gain".to_owned())
        .or_insert_with(|| json!(1.0));
    if !object.contains_key("role") || object.get("role") == Some(&Value::Null) {
        let role = default_track_role(&kind);
        object.insert("role".to_owned(), Value::String(role.to_owned()));
    }
    object
        .entry("items".to_owned())
        .or_insert_with(|| json!([]));
    validate_bool(track, "locked")?;
    validate_bool(track, "muted")?;
    validate_enum(track, "role", TIMELINE_TRACK_ROLES)?;
    let gain = validate_f64_range(track, "gain", 0.0, 4.0)?;
    track
        .as_object_mut()
        .expect("checked above")
        .insert("gain".to_owned(), json!(gain));
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
    // sc-22712. `fadeInSeconds`/`fadeOutSeconds` are AUDIO fades, measured from the item's own
    // ends, and are what lets a continuous bed come up under the first cut and away under the last
    // without being chopped into per-shot pieces. They are deliberately separate from
    // `transitionIn`/`transitionOut`, which are picture transitions with their own vocabulary.
    object
        .entry("fadeInSeconds".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("fadeOutSeconds".to_owned())
        .or_insert_with(|| json!(0.0));
    // A picture item's own audio track is MUTED unless the document says otherwise. See
    // `TIMELINE_GENERATED_AUDIO`: this default is the doubling guard, so it must stay `mute`.
    object
        .entry("generatedAudio".to_owned())
        .or_insert_with(|| Value::String("mute".to_owned()));
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
    let fade_in = validate_f64_range(item, "fadeInSeconds", 0.0, 60.0)?;
    let fade_out = validate_f64_range(item, "fadeOutSeconds", 0.0, 60.0)?;
    validate_enum(item, "generatedAudio", TIMELINE_GENERATED_AUDIO)?;
    let object = item.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline item must be an object".to_owned())
    })?;
    object.insert("sourceIn".to_owned(), json!(source_in));
    object.insert("sourceOut".to_owned(), json!(source_out));
    object.insert("timelineStart".to_owned(), json!(timeline_start));
    object.insert("timelineEnd".to_owned(), json!(timeline_end));
    object.insert("speed".to_owned(), json!(speed));
    object.insert("volume".to_owned(), json!(volume));
    object.insert("fadeInSeconds".to_owned(), json!(fade_in));
    object.insert("fadeOutSeconds".to_owned(), json!(fade_out));
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
                .is_none_or(Vec::is_empty);
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
    apply_project_migrations(&connection)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
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

/// Remove every other readable timeline document with this id. Used by
/// [`ProjectStore::save_timeline`] to make the replacement file authoritative
/// without trusting the current database path alone.
fn remove_stale_timeline_files(
    project_path: &Path,
    timeline_id: &str,
    authoritative_path: &Path,
) -> ProjectStoreResult<()> {
    for candidate in read_dir_paths(&project_path.join("timelines"))? {
        if candidate == authoritative_path
            || !candidate
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
        {
            continue;
        }
        let Ok(candidate_timeline) = read_json(&candidate) else {
            continue;
        };
        if candidate_timeline.get("id").and_then(Value::as_str) == Some(timeline_id) {
            fs::remove_file(candidate)?;
        }
    }
    Ok(())
}

fn find_timeline_file(project_path: &Path, timeline_id: &str) -> ProjectStoreResult<TimelineFile> {
    ensure_project_db_ready(project_path)?;
    let mut db = connect_project_db_migrated(project_path)?;
    let connection = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
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
            connection.execute(
                "update timelines set file_path = ?1 where id = ?2",
                params![rel_path, timeline_id],
            )?;
            connection.commit()?;
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
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::DirectoryCreateCall);
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::DbOpen);
    fs::create_dir_all(project_path)?;
    let connection = Connection::open(project_path.join("project.db"))?;
    crate::store_util::configure_connection(&connection, crate::store_util::Profile::Project)?;
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
        "vector" => build_vector_sidecar_parts(job_id, fact),
        "depth" => build_depth_sidecar_parts(job_id, fact),
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
    // The API replaces this fact from its immutable job/plan binding before any producer reaches
    // this common sidecar boundary. Keep the same projection in the saved recipe and asset row.
    if let Some(provenance) = fact.get(crate::model_artifacts::provenance::MODEL_PROVENANCE_KEY) {
        asset["modelProvenance"] = provenance.clone();
        asset["recipe"]["modelProvenance"] = provenance.clone();
    }
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
    // The API strips this field from every worker report and restores it only from the persisted
    // server-owned child relationship. Applying it during sidecar construction closes the window
    // in which a composed workflow's unpublished raster could appear in the Asset Library.
    if media_type == "image" {
        if let Some(ownership) = fact
            .get("vectorWorkflowOwnership")
            .and_then(Value::as_object)
        {
            let workflow_id = ownership.get("workflowId").and_then(Value::as_str);
            let parent_job_id = ownership.get("parentJobId").and_then(Value::as_str);
            let child_job_id = ownership.get("childJobId").and_then(Value::as_str);
            let valid = [workflow_id, parent_job_id, child_job_id]
                .into_iter()
                .all(|id| id.is_some_and(is_safe_id))
                && ownership.get("role").and_then(Value::as_str) == Some("retained_intermediate")
                && ownership.get("publication").and_then(Value::as_str) == Some("unpublished")
                && ownership.get("hidden").and_then(Value::as_bool) == Some(true);
            if valid {
                let object = asset.as_object_mut().expect("generated asset is an object");
                object.insert(
                    "origin".to_owned(),
                    Value::String("vector_workflow_intermediate".to_owned()),
                );
                let extra = object.entry("extra").or_insert_with(|| json!({}));
                if !extra.is_object() {
                    *extra = json!({});
                }
                extra
                    .as_object_mut()
                    .expect("workflow asset extra is an object")
                    .insert(
                        "vectorWorkflow".to_owned(),
                        Value::Object(ownership.clone()),
                    );
            }
        }
    }
    // SVG is never a browser-previewable media type. The worker supplies a separately stored,
    // bounded PNG fact after sanitizing/rasterizing the SVG; keep it on the vector sidecar rather
    // than asking clients to infer a sibling filename or to fetch the active source document.
    //
    // A depth map (sc-25682) has the same split: its `file` is the raw float32 `.npy`, never
    // rewritten, and the `preview` is a separately stored PNG derived from it. Its `depth` block is
    // the reload contract — value convention, layout, preprocessing and model/config identity — so a
    // consumer can read the map without knowing which job or model produced it.
    if media_type == "vector" || media_type == "depth" {
        if let Some(preview) = fact.get("preview") {
            if let Some(object) = asset.as_object_mut() {
                object.insert("preview".to_owned(), preview.clone());
            }
        }
    }
    if media_type == "tensor" {
        if let Some(tensor) = fact.get("tensor") {
            asset["tensor"] = tensor.clone();
            if let Some(hash) = tensor.get("sha256").and_then(Value::as_str) {
                asset["file"]["contentHash"] = json!(format!("sha256:{hash}"));
            }
        }
        if let Some(index) = fact.get("batchIndex") {
            asset["batchIndex"] = index.clone();
        }
    }
    if media_type == "depth" {
        if let Some(depth) = fact.get("depth") {
            if let Some(object) = asset.as_object_mut() {
                object.insert("depth".to_owned(), depth.clone());
            }
        }
    }
    asset
}

/// `(file, recipe, lineage)` for a depth-map asset (Iris-3B, sc-25682). The file is the raw `.npy`
/// at the EXIF-corrected source size; the recipe records the job's own settings
/// (`rawAdapterSettings.depth`) so the run is reopenable; lineage names the source image.
fn build_depth_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
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
        "prompt": "",
        "negativePrompt": "",
        "seed": Value::Null,
        "loras": [],
        "stylePreset": Value::Null,
        "normalizedSettings": {
            "width": get("width"),
            "height": get("height"),
            "count": 1,
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

/// A list-valued id key as an array, even on facts written before the key existed, so a replay
/// reader never has to distinguish "absent" from "empty" (sc-12345).
fn fact_id_list(fact: &Value, key: &str) -> Value {
    match fact.get(key) {
        Some(value @ Value::Array(_)) => value.clone(),
        _ => json!([]),
    }
}

/// The imported checkpoint a generation selected (sc-26687), or `None` when it ran from a builtin.
///
/// `recipe.model` names the selection; this records WHICH import answered it, so a replay or an API
/// consumer re-resolves the same checkpoint rather than a same-named builtin. Every field is read
/// off the worker's own fact — the plan identity the imported route stamped into
/// `rawAdapterSettings.importPlan` at load time, and the assembly partition beside it. Nothing is
/// synthesized from a catalog entry: a fact carrying no plan identity is not an imported
/// generation and gets no `importedModelRef` at all, rather than an object claiming "an import,
/// unnamed".
fn imported_model_ref(fact: &Value) -> Option<Map<String, Value>> {
    let model = fact
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())?;
    let plan = fact
        .pointer("/rawAdapterSettings/importPlan")
        .and_then(Value::as_object);
    let checkpoint_id = plan
        .and_then(|plan| plan.get("checkpointId"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())?;
    let mut reference = Map::new();
    reference.insert("modelId".to_owned(), Value::String(model.to_owned()));
    if let Some(family) = fact
        .get("family")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        reference.insert("family".to_owned(), Value::String(family.to_owned()));
    }
    // The declared task partition, taken from the bound assembly the route recorded — never from
    // a filename or a directory name, which prove nothing about task compatibility.
    if let Some(partition) = fact
        .pointer("/rawAdapterSettings/importPlan/assembly/partition")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        reference.insert(
            "declaredPartition".to_owned(),
            Value::String(partition.to_owned()),
        );
    }
    if let Some(base_model) = fact
        .get("baseModel")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        reference.insert("baseModel".to_owned(), Value::String(base_model.to_owned()));
    }
    reference.insert(
        "checkpointId".to_owned(),
        Value::String(checkpoint_id.to_owned()),
    );
    if let Some(plan_id) = plan
        .and_then(|plan| plan.get("planId"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        reference.insert("planId".to_owned(), Value::String(plan_id.to_owned()));
    }
    if let Some(semantic_digest) = plan
        .and_then(|plan| plan.get("semanticDigest"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        reference.insert(
            "semanticDigest".to_owned(),
            Value::String(semantic_digest.to_owned()),
        );
    }
    Some(reference)
}

/// Attach the imported-checkpoint identity to a recipe when the generation carried one.
fn stamp_imported_model_ref(recipe: &mut Value, fact: &Value) {
    if let (Some(reference), Some(object)) = (imported_model_ref(fact), recipe.as_object_mut()) {
        object.insert("importedModelRef".to_owned(), Value::Object(reference));
    }
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
    let mut recipe = json!({
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
            // sc-26687: the ORDERED image conditioning the replay path restores. `referenceAssetIds`
            // is a list whose element order is semantic — the engine labels the supplied images
            // positionally — so it is recorded verbatim, never sorted or de-duplicated. The image
            // recipe recorded neither the edit source nor the reference list before this, so
            // "re-run this generation" silently rebuilt a job with different conditioning.
            "sourceAssetId": get("sourceAssetId"),
            "referenceAssetId": get("referenceAssetId"),
            "referenceAssetIds": fact_id_list(fact, "referenceAssetIds"),
        },
        "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
    });
    stamp_imported_model_ref(&mut recipe, fact);
    let lineage = json!({
        "parents": parents,
        "sourceAssetId": get("sourceAssetId"),
        "sourceTimestamp": Value::Null,
        "jobId": job_id,
    });
    (file, recipe, lineage)
}

/// Vector Studio replay is intentionally explicit rather than inheriting the image recipe's
/// width/height/LoRA shape. These are the complete generation inputs and the two versioned
/// post-processing boundaries required to reproduce or explain a stored SVG.
fn build_vector_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    let source_asset_id = get("sourceAssetId");
    let parents = source_asset_id
        .as_str()
        .map(|source| vec![Value::String(source.to_owned())])
        .unwrap_or_default();
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
        "sampling": get("sampling"),
        "detailBudget": get("detailBudget"),
        "sanitizerVersion": get("sanitizerVersion"),
        "rendererVersion": get("rendererVersion"),
        "workflow": get("workflow"),
    });
    let workflow_id = fact.pointer("/workflow/id").cloned().unwrap_or(Value::Null);
    let lineage = json!({
        "parents": parents,
        "sourceAssetId": source_asset_id,
        "sourceTimestamp": Value::Null,
        "jobId": job_id,
        "workflowId": workflow_id,
        "intermediateAssetId": get("sourceAssetId"),
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
    let list_or_empty = |key: &str| fact_id_list(fact, key);
    let source_keys = [
        "sourceAssetId",
        "lastFrameAssetId",
        "sourceClipAssetId",
        "bridgeRightClipAssetId",
    ];
    // The list-valued source ids are parents too — mv2v's clips, the reference-driven modes'
    // subject images, and ads2v's reference video. Without them those modes' provenance names
    // only a subset of what the clip was actually derived from (sc-12345).
    // sc-17160 adds the audio references: media the clip was genuinely derived from, so leaving
    // them out would name only a subset of its provenance — the same gap sc-12345 closed for the
    // clip and image lists.
    let list_source_keys = [
        "sourceClipAssetIds",
        "referenceAssetIds",
        "referenceAudioAssetIds",
    ];
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
    let mut file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": get("width"),
        "height": get("height"),
        "duration": measured("encodedDuration", "duration"),
        "fps": measured("encodedFps", "fps"),
        "frameCount": fact.get("encodedFrameCount").cloned().unwrap_or(Value::Null),
    });
    // sc-19577: whether the mp4 carries a soundtrack, measured off what the worker actually MUXED.
    //
    // Written ONLY when the fact carries it, and deliberately not defaulted to `false`. Every asset
    // generated before sc-19577 has no such fact, and a `false` there would be a fresh claim that
    // those clips are silent — which for the LTX renders among them is untrue. Absent means UNKNOWN
    // and the UI shows nothing; `false` means measured-and-silent. Same shape `frameCount` would
    // have needed had a wrong default been possible for it.
    if let (Some(has_audio), Some(object)) = (
        fact.get("hasAudio").filter(|value| value.is_boolean()),
        file.as_object_mut(),
    ) {
        object.insert("hasAudio".to_owned(), has_audio.clone());
    }
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
        // sc-17160. `list_or_empty` keeps the array shape on facts written before this key
        // existed, so the replay reader never distinguishes "absent" from "empty".
        "referenceAudioAssetIds": list_or_empty("referenceAudioAssetIds"),
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
    let mut recipe = json!({
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
    stamp_imported_model_ref(&mut recipe, fact);
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

/// The outcome of import-time image normalization (sc-6143): the path whose bytes should be moved
/// into the asset store, plus the mime type and file extension to record. When `transcoded_temp` is
/// set, the move consumes that temp PNG (not the original upload), so the caller must remove the
/// original source itself.
struct NormalizedUpload {
    source_path: PathBuf,
    content_type: String,
    extension: String,
    transcoded_temp: Option<PathBuf>,
    /// The sanitized workflow envelope the upload carried in a PNG text chunk, when a
    /// [`WorkflowScan::Read`] caller asked and one was found (sc-15949).
    ///
    /// Read from the ORIGINAL upload bytes, so it survives a transcode that re-encodes them.
    workflow: Option<WorkflowShare>,
}

/// Whether an upload's metadata is searched for an embedded workflow envelope (sc-15949).
///
/// A parameter rather than an unconditional scan, so the cost is paid only where the answer is
/// used. `import_asset` records what it finds under `extra.importedWorkflow`; the training-dataset
/// lane has no recipe slot and no reader for one, so it opts out and pays nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkflowScan {
    /// Look for a `sceneworks:workflow` chunk, before any transcode.
    Read,
    /// Do not look.
    Skip,
}

/// Normalize a valid-but-unsupported image upload (AVIF/HEIC/HEIF/TIFF/BMP/GIF) to lossless PNG
/// before it is stored, so every downstream decode site, thumbnail, and preview stays on a format
/// it can read (sc-6143). The format is sniffed by content, never the extension, so a `.png` that is
/// really AVIF is still handled. Already-supported formats (png/jpeg/webp) and non-image uploads
/// pass through untouched — no temp file, no re-encode, no quality loss.
///
/// `scan` also makes this the one place an incoming image is searched for an embedded workflow
/// (sc-15949), because this is the one place that knows what is about to happen to the bytes; see
/// the comment at the scan itself for why the ordering is load-bearing in both directions.
fn normalize_image_upload(
    source_path: &Path,
    content_type: &str,
    filename: &str,
    work_dir: &Path,
    scan: WorkflowScan,
) -> ProjectStoreResult<NormalizedUpload> {
    let passthrough = || NormalizedUpload {
        source_path: source_path.to_path_buf(),
        content_type: content_type.to_owned(),
        extension: upload_extension(filename, content_type),
        transcoded_temp: None,
        workflow: None,
    };
    if !content_type.starts_with("image/") {
        return Ok(passthrough());
    }
    let Some(kind) = crate::media_convert::sniff_image_kind_at(source_path) else {
        // Unrecognized magic (e.g. SVG) — leave it as-is; we can't transcode what we can't sniff.
        return Ok(passthrough());
    };
    // sc-15949: the chunk scan goes HERE, and the position is the decision.
    //
    // BEFORE the natively-supported branch below, because that branch is a `return`: a PNG — the
    // one format that carries our chunk — would otherwise never be looked at.
    //
    // BEFORE the transcode further down, because `sips`/`ffmpeg` re-encode from pixels and drop
    // every ancillary chunk by definition. A scan afterwards could only ever report an absence, so
    // a workflow found here is carried forward on `NormalizedUpload` even though the stored bytes
    // end up re-encoded.
    //
    // And it scans CHUNKS, not pixels. `read_workflow_chunk_file` stops at the first IDAT, and when
    // that finds nothing it walks the remaining framing with no output buffer, which `png` answers
    // by consuming the IDAT payload instead of inflating it. So this function's whole point — a
    // natively-decodable upload is stored byte-for-byte, with no decode anywhere on the path —
    // still holds.
    let workflow = match scan {
        WorkflowScan::Read => read_upload_workflow(source_path),
        WorkflowScan::Skip => None,
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
            workflow,
        });
    }
    if kind.png_transcode_is_lossy() {
        // Scene-linear HDR (sc-18790). Storing the PNG and discarding the source is right for every
        // other transcoded format — their pixels survive the conversion — but here it would throw
        // away the float latitude the file exists for, leaving a tone-mapped proxy and no way back.
        // So the original bytes are stored; the browser-renderable PNG is produced on demand by the
        // `?thumbnail=` derivative route, which already falls back to `transcode_to_png` when the
        // `image` crate cannot decode a source. Download therefore serves the original.
        let (extension, mime) = kind.canonical();
        return Ok(NormalizedUpload {
            source_path: source_path.to_path_buf(),
            content_type: mime.to_owned(),
            extension: format!(".{extension}"),
            transcoded_temp: None,
            workflow,
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
        workflow,
    })
}

/// Normalize an audio upload to the canonical PCM-16 RIFF/WAVE the product can actually read back
/// (sc-18650), ALWAYS — there is no pass-through branch HERE.
///
/// The rule is deliberately unlike [`normalize_image_upload`]'s, where an already-decodable PNG or
/// JPEG is stored byte-for-byte. Audio has exactly one reader,
/// `sceneworks_worker::audio_jobs::read_wav_pcm16`, and it accepts exactly one encoding, so
/// "already supported" is a much narrower set than "already audio": a 24-bit WAV, a float WAV, or a
/// WAVE_FORMAT_EXTENSIBLE header are all `.wav` files it refuses. Every upload therefore goes
/// through `transcode_to_wav_pcm16`; the ONE encoding that converter copies through unchanged
/// (`media_convert::is_canonical_pcm16_wav`, sc-22715) is exactly the reader's own acceptance
/// rule, decided inside the converter rather than by a sniff here, so the stored file is the
/// canonical one on every path and a host with no ffmpeg can still import a file that already is.
///
/// A conversion failure is a `BadRequest`: the caller named a file that is not decodable audio (or
/// carries no audio stream), which is a fact about the upload, not about the host. The one host-shaped
/// failure — no reachable ffmpeg — is reported through the same error with ffmpeg's own message, which
/// says so.
fn normalize_audio_upload(
    source_path: &Path,
    work_dir: &Path,
) -> ProjectStoreResult<NormalizedUpload> {
    let temp_wav = work_dir.join(format!("upload-transcode-{}.wav", random_hex(8)?));
    crate::media_convert::transcode_to_wav_pcm16(source_path, &temp_wav).map_err(|error| {
        let _ = fs::remove_file(&temp_wav);
        ProjectStoreError::BadRequest(format!(
            "Could not convert the uploaded audio to a supported format: {error}"
        ))
    })?;
    Ok(NormalizedUpload {
        source_path: temp_wav.clone(),
        content_type: "audio/wav".to_owned(),
        extension: ".wav".to_owned(),
        transcoded_temp: Some(temp_wav),
        // An audio file carries no SceneWorks workflow chunk — that envelope is a PNG ancillary
        // chunk, and nothing writes an audio equivalent.
        workflow: None,
    })
}

/// The clip facts a stored PCM WAV declares in its own header: rate, channel count, and the duration
/// those two plus the `data` chunk length imply.
#[derive(Clone, Copy, Debug, PartialEq)]
struct WavFacts {
    sample_rate: u32,
    channels: u16,
    duration_secs: f64,
}

/// Read [`WavFacts`] out of a RIFF/WAVE header, or `None` if the file is not one this product wrote.
///
/// Header-only: it walks the chunk list for `fmt ` and `data` and reads the `data` chunk's declared
/// LENGTH, never its bytes, so a 200 MB clip costs one small read. It is intentionally NOT a second
/// `read_wav_pcm16` — that one lives in the worker, decodes samples, and answers a different
/// question. This one answers "what does the sidecar record about this file", which the store has to
/// answer without linking a decoder.
fn wav_facts(path: &Path) -> Option<WavFacts> {
    // The fixed prelude plus the largest offset read below (`fmt `'s 16-byte body).
    let mut head = vec![0u8; 4096];
    let read = {
        use std::io::Read;
        let mut file = fs::File::open(path).ok()?;
        let mut filled = 0usize;
        loop {
            match file.read(&mut head[filled..]) {
                Ok(0) => break,
                Ok(count) => filled += count,
                Err(_) => return None,
            }
            if filled == head.len() {
                break;
            }
        }
        filled
    };
    let bytes = &head[..read];
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let le16 = |offset: usize| u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    let le32 = |offset: usize| {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    };
    let mut position = 12usize;
    let mut format: Option<(u16, u32, u16)> = None; // (channels, sample_rate, bits)
    let mut data_len: Option<u32> = None;
    while position + 8 <= bytes.len() {
        let id = &bytes[position..position + 4];
        let size = le32(position + 4);
        let body = position + 8;
        if id == b"fmt " && size >= 16 && body + 16 <= bytes.len() {
            format = Some((le16(body + 2), le32(body + 4), le16(body + 14)));
        } else if id == b"data" {
            data_len = Some(size);
            break;
        }
        // RIFF chunks are word-aligned: an odd body is followed by a pad byte.
        position = body + size as usize + (size as usize & 1);
    }
    let (channels, sample_rate, bits) = format?;
    if channels == 0 || sample_rate == 0 || bits == 0 {
        return None;
    }
    let frame_bytes = u32::from(channels) * u32::from(bits) / 8;
    let duration_secs = match (data_len, frame_bytes) {
        (Some(len), frame) if frame > 0 => f64::from(len / frame) / f64::from(sample_rate),
        _ => 0.0,
    };
    Some(WavFacts {
        sample_rate,
        channels,
        // Three decimals, the precision `build_render_asset` and the audio sidecar already record
        // durations at.
        duration_secs: (duration_secs * 1000.0).round() / 1000.0,
    })
}

/// Look for a SceneWorks workflow envelope in an upload's PNG metadata, degrading EVERY failure to
/// "no workflow found" (sc-15949).
///
/// This file came from a stranger, and it is the epic's main attack surface. The guards are
/// [`crate::workflow_png::read_workflow_chunk_file`]'s and are not re-implemented here: a bounded
/// metadata budget, a 1 MiB cap on the decompressed text so a zip bomb costs a megabyte, a refusal
/// of duplicate chunks and lying chunk lengths, and — the important one — a single parse path
/// through the sc-15946 allow-list, so the envelope is reduced at value granularity on the way IN
/// exactly as it was on the way out. Nothing here deserializes an `ImageJobRequest`, or anything
/// else, from the file.
///
/// Every error becomes `None`, deliberately and with no exception. The user asked to import an
/// image, and they get their image: a hostile or malformed chunk costs them the recipe field and
/// nothing else. `NotPng` is not even unusual — it is what every JPEG and WebP upload returns — so
/// the typed error is logged at debug rather than surfaced.
fn read_upload_workflow(source_path: &Path) -> Option<WorkflowShare> {
    match crate::workflow_png::read_workflow_chunk_file(source_path) {
        Ok(found) => found,
        Err(error) => {
            tracing::debug!(
                path = %source_path.display(),
                %error,
                "upload carries no readable workflow chunk; importing it as a plain asset"
            );
            None
        }
    }
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
    project_id: &str,
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
    maybe_fail_asset_index_db_mutation()?;
    let mut connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    index_asset_on_connection(
        &transaction,
        project_id,
        project_path,
        asset,
        Some(&sidecar_path),
    )?;
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
    maybe_fail_asset_index_db_mutation()?;
    let mut connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute("delete from assets where id = ?1", params![asset_id])?;
    transaction.commit()?;
    Ok(())
}

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
fn confined_deletion_target(
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
fn remove_sibling_poster_no_symlinks(
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
    apply_project_migrations(&connection)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    // (asset id, media path, orphaned sidecar to quarantine) for every row whose media is gone.
    let orphans: Vec<(String, String, Option<PathBuf>)> = {
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
            orphans.push((asset_id, file_rel, sidecar));
        }
        orphans
    };
    for (asset_id, _, _) in &orphans {
        transaction.execute("delete from assets where id = ?1", params![asset_id])?;
    }
    transaction.commit()?;
    // The row(s) are gone from the registry; now move each orphaned sidecar OUT of
    // the scanned asset folders into `.orphaned-sidecars/` (never scanned by
    // `asset_sidecars`). This declutters the project and closes the resurrection
    // edge — a fully-orphaned project would otherwise re-add these rows via the
    // empty-table auto-reindex in `list_assets`. The sidecar is retained (moved, not
    // deleted) so the recipe metadata stays recoverable. Report a move failure after
    // attempting every orphan so dirty repair keeps its marker and retries instead
    // of clearing it while a resurrectable sidecar remains in the scanned tree.
    let mut cleanup_error = None;
    for (asset_id, media_rel, sidecar) in &orphans {
        if let Err(error) = remove_sibling_poster_no_symlinks(project_path, media_rel) {
            cleanup_error.get_or_insert(error);
        }
        let Some(sidecar) = sidecar else {
            continue;
        };
        if let Err(error) = quarantine_orphaned_sidecar(project_path, asset_id, sidecar) {
            tracing::debug!(
                event = "orphaned_sidecar_quarantine_failed",
                asset_id = %asset_id,
                sidecar = %sidecar.display(),
                error = %error,
                "could not move orphaned sidecar out of the asset tree; repair remains pending"
            );
            cleanup_error.get_or_insert(ProjectStoreError::Io(error));
        }
    }
    if let Some(error) = cleanup_error {
        return Err(error);
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

/// Validate the minimum structure the indexer itself requires before a cached
/// envelope can participate in hydration or de-duplication. The SQL primary key
/// is authoritative: a parseable but corrupted `asset_json.id` must never
/// impersonate another row and hide its healthy envelope via `seen_asset_ids`.
fn indexed_asset_envelope_is_valid(
    row_id: &str,
    sidecar_path: Option<&str>,
    asset: &Value,
) -> bool {
    if !is_safe_id(row_id)
        || asset.get("id").and_then(Value::as_str) != Some(row_id)
        || sidecar_path.is_some_and(|path| !is_safe_relative_path(path))
    {
        return false;
    }
    for key in ["type", "displayName", "createdAt"] {
        let Some(value) = asset.get(key).and_then(Value::as_str) else {
            return false;
        };
        if value.is_empty() {
            return false;
        }
    }
    asset
        .pointer("/file/path")
        .and_then(Value::as_str)
        .is_some_and(is_safe_relative_path)
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
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::PathStat);
    if !path.exists() {
        return Ok(Vec::new());
    }
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::DirectoryScan);
    fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()).map_err(Into::into))
        .collect()
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

/// The sidecar `type` an upload's mime maps to. `"audio"` since sc-18650 — the same string
/// `build_generated_asset_sidecar` dispatches on for an Audio Studio render, so an imported clip and
/// a rendered one are one asset kind rather than two.
fn media_type_for_mime(mime_type: &str) -> ProjectStoreResult<&'static str> {
    if mime_type == crate::tensor_artifact::MIME {
        return Ok("tensor");
    }
    if mime_type.starts_with("image/") {
        return Ok("image");
    }
    if mime_type.starts_with("video/") {
        return Ok("video");
    }
    if mime_type.starts_with("audio/") {
        return Ok("audio");
    }
    Err(ProjectStoreError::BadRequest(
        "Only image, video and audio uploads are supported".to_owned(),
    ))
}

fn guess_mime_from_filename(filename: &str) -> Option<String> {
    if Path::new(filename)
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("npy"))
    {
        return Some(crate::tensor_artifact::MIME.to_owned());
    }
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
                // OpenEXR (sc-18790). `image/x-exr` is inert and starts with `image/`, which
                // is what admits it to SAFE_UPLOAD_EXTENSIONS and to the `?thumbnail=` route.
                Some("exr") => Some("image/x-exr".to_owned()),
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
/// `image/`, `video/` or `audio/` and is NOT `image/svg+xml` — the property the tests pin.
///
/// `wav` is the only audio entry and that is not an oversight (sc-18650): `normalize_audio_upload`
/// converts EVERY audio upload to PCM-16 RIFF/WAVE, so `.wav` is the only extension an audio asset
/// is ever stored under. An `.mp3` is admitted at the door and stored `.wav`; the source extension
/// never reaches the filesystem, so it never needs to be on this list.
const SAFE_UPLOAD_EXTENSIONS: &[&str] = &[
    // Raster images (SVG deliberately omitted — it is script-capable).
    "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff", "avif", "heic", "heif",
    // Scene-linear HDR frames (sc-18790) — stored as-is, previewed as a derivative.
    "exr", // Video.
    "mp4", "m4v", "mov", "webm", "mkv", "avi", "ogv", "mpeg", "mpg", "wmv", "flv", "3gp",
    "3g2", // Audio: the single canonical stored form.
    "wav",
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
    if mime_type == crate::tensor_artifact::MIME {
        return ".npy".to_owned();
    }
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
#[path = "project_store_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "project_store_timeline_revision_tests.rs"]
mod timeline_revision_tests;
