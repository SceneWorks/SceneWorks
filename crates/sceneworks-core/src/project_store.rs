//! Public project persistence interface; aggregate implementations are private modules.

mod asset_indexing;
mod asset_mutations;
mod asset_queries;
mod asset_sidecars;
mod audio;
mod characters;
mod errors;
mod film_drafts;
mod film_runs;
mod indexing;
mod keypoints;
mod media;
mod person_tracks;
mod poses;
mod registry;
mod schema;
mod storage;
mod timelines;
mod training;
mod uploads;

#[cfg(test)]
use asset_indexing::apply_upscale_variant_link;
use asset_indexing::asset_index_is_dirty;
use asset_indexing::backfill_upscale_variant_lineage;
use asset_indexing::current_thread_owns_asset_index_mutation;
#[cfg(test)]
use asset_indexing::fail_next_asset_index_db_mutation;
#[cfg(test)]
pub(crate) use asset_indexing::fail_next_asset_poster_db_read;
#[cfg(test)]
use asset_indexing::fail_next_asset_sidecar_remove;
#[cfg(test)]
use asset_indexing::fail_next_upscale_lineage_db_read;
use asset_indexing::find_asset_sidecar_path;
use asset_indexing::index_asset;
use asset_indexing::maybe_fail_asset_index_db_mutation;
pub(crate) use asset_indexing::maybe_fail_asset_poster_db_read;
use asset_indexing::purge_asset_record;
use asset_indexing::remove_asset_sidecar;
use asset_indexing::upscale_lineage_group;
pub(crate) use asset_indexing::AssetIndexMutation;
#[cfg(test)]
use asset_indexing::ASSET_INDEX_DIRTY_MARKER;
use asset_indexing::ASSET_INDEX_VERSION_KEY;
#[cfg(test)]
use asset_indexing::UPSCALE_LINEAGE_QUERY;
pub(crate) use asset_queries::record_asset_list_filesystem_operation;
use asset_sidecars::build_generated_asset_sidecar;
#[cfg(test)]
use asset_sidecars::build_image_sidecar_parts;
#[cfg(test)]
use asset_sidecars::build_video_sidecar_parts;
use film_drafts::copy_film_reference_files;
#[cfg(test)]
use film_drafts::stage_draft_file;
use indexing::project_has_sidecars;
use indexing::reindex_project_path;
#[cfg(test)]
use indexing::ORPHANED_SIDECAR_DIR;
pub use keypoints::GLOBAL_KEYPOINTS_PROJECT_ID;
pub use keypoints::GLOBAL_KEYPOINTS_PROJECT_NAME;
pub use keypoints::KEYPOINT_UPLOADS_CACHE_DIR;
use media::confined_deletion_target;
use media::remove_sibling_poster_no_symlinks;
pub use poses::GLOBAL_POSES_PROJECT_ID;
pub use poses::GLOBAL_POSES_PROJECT_NAME;
pub use poses::POSE_UPLOADS_CACHE_DIR;
use registry::read_project_summary;
#[cfg(test)]
use registry::read_registry_payload;
use registry::RegistryCache;
#[cfg(test)]
use registry::MAX_REGISTRY_BYTES;
pub use registry::PROJECT_FOLDERS;
pub use schema::apply_project_migrations;
use schema::connect_project_db;
pub(crate) use schema::connect_project_db_migrated;
use schema::connect_project_db_ready_locked;
pub use schema::ensure_project_db_ready;
#[cfg(test)]
use schema::install_ensure_ready_before_lock_hook;
#[cfg(test)]
use schema::project_schema_version;
#[cfg(test)]
use schema::run_ensure_ready_before_lock_hook;
use schema::PROJECT_SCHEMA_VERSION;
#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use storage::is_write_denied;
use storage::move_or_copy_file;
pub use storage::probe_storage_writable;
use storage::read_dir_paths;
use storage::storage_not_writable_message;
#[cfg(test)]
use timelines::find_timeline_file;
#[cfg(test)]
use timelines::index_timeline;
pub use timelines::TIMELINE_ASPECT_RATIOS;
pub use timelines::TIMELINE_GENERATED_AUDIO;
pub use timelines::TIMELINE_TRACK_ROLES;
use uploads::guess_mime_from_filename;
#[cfg(test)]
use uploads::is_safe_upload_extension;
use uploads::move_normalized_upload;
use uploads::normalize_asset_tags;
use uploads::normalize_image_upload;
use uploads::safe_filename;
#[cfg(test)]
use uploads::upload_extension;
use uploads::WorkflowScan;
pub use uploads::IMPORTED_WORKFLOW_KEY;
use uploads::SAFE_UPLOAD_EXTENSIONS;

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
    optional_str, optional_u64, random_hex, read_json, relative_string, required_str, sql_int,
    write_json,
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

#[cfg(test)]
#[path = "project_store_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "project_store_timeline_revision_tests.rs"]
mod timeline_revision_tests;
