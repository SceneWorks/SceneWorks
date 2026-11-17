use super::*;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JobsQuery {
    pub(crate) project_id: Option<String>,
    pub(crate) status: Option<String>,
    pub(crate) limit: Option<u32>,
}

/// Register a saved voice (Voice Clone registry, sc-13517): a name + the library audio asset whose
/// voice is cloned. `dedupThreshold` optionally overrides the cosine-similarity cutoff used to warn
/// about near-duplicate voices already saved in the project.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SavedVoiceCreateRequest {
    pub(crate) name: String,
    pub(crate) reference_audio_asset_id: String,
    #[serde(default)]
    pub(crate) dedup_threshold: Option<f32>,
}

/// Filters for the aggregate generation-metrics feed (epic 10402,
/// `GET /api/v1/metrics`). All optional; the charts pass a subset.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MetricsQuery {
    #[serde(rename = "type")]
    pub(crate) job_type: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) quant: Option<String>,
    pub(crate) limit: Option<u32>,
}

/// Paging of a job's per-item results (SW-7082, `GET /api/v1/jobs/:job_id/result-items`).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JobResultItemsQuery {
    pub(crate) offset: Option<usize>,
    pub(crate) limit: Option<usize>,
}

/// One per-item result a worker appends: its order among the job's items, and the item object
/// (which carries its own `itemId`).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JobResultItemBody {
    pub(crate) position: u64,
    pub(crate) item: serde_json::Map<String, Value>,
}

/// Worker → API append of per-item results (SW-7082, `POST /api/v1/jobs/:job_id/result-items`).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JobResultItemsAppend {
    pub(crate) worker_id: Option<String>,
    pub(crate) items: Vec<JobResultItemBody>,
}

pub(crate) use sceneworks_core::generation_contracts::PromptRefineRequest;

/// On-demand "compare image to another" likeness request (epic 4406, sc-4415). Scores a CANDIDATE
/// asset against a SOURCE identity reference asset (both project assets) through the shared
/// SCRFD+ArcFace face-likeness scorer in the worker. The handler enqueues a `face_likeness_compare`
/// job; the client reads the result (`{ score, detected, method, sourceRef, reason? }`) from the
/// completed job.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FaceLikenessCompareRequest {
    /// Project owning both assets (so the worker can resolve each asset's `file.path`).
    pub(crate) project_id: String,
    /// The SOURCE identity reference asset (an approved character Reference Asset). Its face is
    /// embedded once and the candidate is scored against it.
    pub(crate) source_asset_id: String,
    /// The CANDIDATE asset (any image the user picked) whose face is compared to the source.
    pub(crate) candidate_asset_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AssetsQuery {
    pub(crate) include_rejected: Option<bool>,
    pub(crate) include_trashed: Option<bool>,
    /// Scope to one character: assets generated in association with it
    /// (recipe.normalizedSettings.characterId) or referencing it
    /// (metadata.characterReferences[].characterId).
    pub(crate) character_id: Option<String>,
    /// View scope (sc-2024). `library` excludes Character Studio test outputs;
    /// anything else (the default) returns all assets.
    pub(crate) scope: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharactersQuery {
    pub(crate) include_archived: Option<bool>,
}

/// Query parameters for the dataset readiness report (sc-6533). The training UI passes what it has
/// A page of a dataset's batch describe jobs, newest first: skip `offset` jobs, return at most
/// `limit` (default and maximum 200); `unfinished=true` keeps only jobs not yet finished.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BatchDescribeJobsQuery {
    pub(crate) unfinished: Option<bool>,
    pub(crate) offset: Option<u32>,
    pub(crate) limit: Option<u32>,
}

/// selected; all are optional and fall back to per-kind defaults in `readiness_context`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ReadinessQuery {
    /// The chosen training target's resolution; drives the bucket the scalars are measured at.
    pub(crate) target_resolution: Option<u32>,
    /// Comma-separated preset/target `recommendedFor` tags (e.g. `"character,style"`).
    pub(crate) recommended_for: Option<String>,
    /// The dataset's character kind (e.g. `"person"`).
    pub(crate) character_type: Option<String>,
    /// Override for the preset's minimum item count.
    pub(crate) min_items: Option<u32>,
}

/// Body for the per-image quality override (sc-6534): the checks the user dismissed for one image.
/// The store strips the non-acknowledgeable ones (`decode`, `count`); an empty list clears the ack.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct QualityAckBody {
    #[serde(default)]
    pub(crate) checks: Vec<sceneworks_core::dataset_quality::QualityCheck>,
    /// Client-side freshness guard. The ack applies only to the exact image bytes the user reviewed.
    #[serde(default)]
    pub(crate) expected_content_hash: Option<String>,
    /// Freshness guard for caption-dependent checks such as `caption_alignment`.
    #[serde(default)]
    pub(crate) expected_caption_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LorasQuery {
    pub(crate) model_family: Option<String>,
    pub(crate) project_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CatalogDeleteQuery {
    pub(crate) project_id: Option<String>,
    pub(crate) scope: Option<String>,
    /// When true, permanently unlink the artifacts instead of moving them to the OS
    /// trash. The UI only sets this on the second call after a "move to trash" failure
    /// prompts the user to confirm a permanent delete. Defaults to false (trash first).
    pub(crate) permanent: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AssetPurgeQuery {
    /// See `CatalogDeleteQuery::permanent` — false trashes the media/sidecar, true unlinks.
    pub(crate) permanent: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RecipePresetsQuery {
    pub(crate) project_id: Option<String>,
    pub(crate) include_archived: Option<bool>,
    pub(crate) model: Option<String>,
    pub(crate) workflow: Option<String>,
    pub(crate) scope: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PromptBatchesQuery {
    pub(crate) project_id: Option<String>,
    pub(crate) include_archived: Option<bool>,
    pub(crate) scope: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EventsQuery {
    pub(crate) ticket: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CreateEventTicketRequest {
    #[serde(default)]
    pub(crate) active_job_ids: Vec<String>,
    #[serde(default)]
    pub(crate) known_terminal_job_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HealthResponse {
    pub(crate) status: &'static str,
    pub(crate) service: &'static str,
    pub(crate) runtime: String,
    pub(crate) version: String,
    pub(crate) options_validator_schema_version: u32,
    pub(crate) auth_required: bool,
    // Absolute host paths are withheld from the public health endpoint when a token is
    // configured, so a LAN client can't map the host filesystem despite auth being on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) directories: Option<DirectoriesResponse>,
    pub(crate) interrupted_jobs_on_startup: usize,
    pub(crate) readiness: StartupReadinessResponse,
    pub(crate) startup_maintenance: crate::startup::StartupMaintenanceResponse,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StartupReadinessResponse {
    pub(crate) status: &'static str,
    pub(crate) criticality: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DirectoriesResponse {
    pub(crate) data: String,
    pub(crate) config: String,
    pub(crate) projects: String,
    pub(crate) jobs_db: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AccessResponse {
    pub(crate) auth_required: bool,
    pub(crate) token_header: &'static str,
}

/// Host memory for remote-browser model gating (epic 4484 story 9). A remote browser
/// can't call the desktop-only Tauri `get_gpu_info`, so it reads the host's memory
/// from here — derived from the registered GPU worker's reported utilization. Carries
/// only aggregate memory totals + the platform string (no paths/secrets), and is
/// auth-protected (not a public path).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HostCapabilitiesResponse {
    /// API host OS (`macos`/`windows`/`linux`) so the client gates with the right
    /// memory concept (unified vs VRAM).
    pub(crate) platform: &'static str,
    /// Active memory pool and its kind. Kept as a pair so clients cannot silently treat
    /// shared system memory as dedicated VRAM (or the reverse).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) memory_kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) memory_gb: Option<f64>,
    /// Unified system memory (GB) — the macOS MLX worker reports `sysctl hw.memsize`.
    /// `None` when no MLX worker is registered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) unified_memory_gb: Option<f64>,
    /// Largest discrete GPU VRAM (GB) across registered GPU workers (Windows candle /
    /// CUDA). `None` when no such worker is registered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) gpu_memory_gb: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct VerifyResponse {
    pub(crate) ok: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ProjectCreateRequest {
    pub(crate) name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterCreateRequest {
    pub(crate) name: String,
    #[serde(default = "default_character_type", rename = "type")]
    pub(crate) character_type: String,
    #[serde(default)]
    pub(crate) description: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterUpdateRequest {
    pub(crate) name: Option<String>,
    #[serde(default, rename = "type")]
    pub(crate) character_type: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) archived: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterReferenceRequest {
    pub(crate) asset_id: String,
    #[serde(default)]
    pub(crate) approved: bool,
    #[serde(default = "default_reference_role")]
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) notes: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterReferenceUpdateRequest {
    pub(crate) approved: Option<bool>,
    pub(crate) role: Option<String>,
    pub(crate) notes: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterLookRequest {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) description: String,
    #[serde(default)]
    pub(crate) approved_reference_ids: Vec<String>,
    #[serde(default)]
    pub(crate) recipe_settings: JsonObject,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterLookUpdateRequest {
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) approved_reference_ids: Option<Vec<String>>,
    pub(crate) recipe_settings: Option<JsonObject>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterLoraRequest {
    #[serde(default)]
    pub(crate) lora_id: Option<String>,
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) source_path: Option<String>,
    #[serde(default)]
    pub(crate) trigger_words: Vec<String>,
    #[serde(default = "default_character_lora_weight")]
    pub(crate) default_weight: f64,
    #[serde(default)]
    pub(crate) compatibility: JsonObject,
    #[serde(default = "default_project_lora_scope")]
    pub(crate) scope: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterLoraUpdateRequest {
    pub(crate) name: Option<String>,
    pub(crate) trigger_words: Option<Vec<String>>,
    pub(crate) default_weight: Option<f64>,
    pub(crate) compatibility: Option<JsonObject>,
    pub(crate) scope: Option<String>,
}

pub(crate) use sceneworks_core::generation_contracts::CharacterTestRequest;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TimelineCreateRequest {
    #[serde(default = "default_timeline_name")]
    pub(crate) name: String,
    #[serde(default = "default_aspect_ratio")]
    pub(crate) aspect_ratio: String,
    #[serde(default = "default_timeline_fps")]
    pub(crate) fps: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TimelineSaveRequest {
    pub(crate) timeline: Value,
    pub(crate) expected_revision: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TimelineExportRequest {
    #[serde(default = "default_export_resolution")]
    pub(crate) resolution: u32,
    #[serde(default = "default_timeline_fps")]
    pub(crate) fps: u32,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// The export is for a commercial use (sc-22999, epic 22988 E2). An export that places an
    /// asset whose usage policy refuses commercial use (YuE2 audio) is refused with the pointer
    /// to the commercially eligible alternative. Absent ⇒ not declared commercial.
    #[serde(default)]
    pub(crate) commercial_use: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FrameExtractRequest {
    pub(crate) playhead_seconds: f64,
    #[serde(default = "default_frame_intended_use")]
    pub(crate) intended_use: String,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
}

pub(crate) use sceneworks_core::generation_contracts::{
    PersonDetectionJobRequest, PersonTrackJobRequest,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersonTrackCorrectionsRequest {
    /// The UI's full correction set for the track. Each entry targets a frame by
    /// index and adjusts its box and/or rejects the frame; the store validates
    /// ranges and stamps author/createdAt/source. Kept as raw values so the
    /// schema-flexible `corrections` array can evolve without an API change.
    #[serde(default)]
    pub(crate) corrections: Vec<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TrainingCaptionJobRequest {
    #[serde(default = "default_training_captioner")]
    pub(crate) captioner: String,
    #[serde(default = "default_training_caption_model")]
    pub(crate) model_name_or_path: String,
    /// Caption mode (sc-24829): `default` (the full-scene JoyCaption prompt), `subjectOnly` (the
    /// worker's subject-only prompt) or `triggerOnly` (exactly the trigger words, no captioner).
    /// Unknown strings deserialize to `CaptionMode::Unknown` and are refused by validation.
    #[serde(default = "default_training_caption_mode")]
    pub(crate) mode: sceneworks_core::training::CaptionMode,
    /// Trigger words for the job (sc-24829): filled into every captioned item that carries none of
    /// its own, in every mode — so a trigger-only caption has words to write, and default /
    /// subject-only captions get them prepended. Limits mirror the caption dialog
    /// (`captionTriggerWordLimits` in `joyCaptionPrompts.js`).
    #[serde(default)]
    pub(crate) trigger_words: Vec<String>,
    #[serde(default)]
    pub(crate) recaption: bool,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    #[serde(default)]
    pub(crate) options: TrainingCaptionOptions,
    /// Restrict the job to these dataset item ids (sc-2025 per-image Re-Caption).
    /// When present, only those items are captioned and they are always recaptioned;
    /// when absent, the dataset-wide `recaption`/missing-caption rule applies.
    #[serde(default)]
    pub(crate) item_ids: Option<Vec<String>>,
}

/// Request to run the Dataset Doctor CLIP-embedding analysis over a training dataset (sc-6535).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetAnalysisJobRequest {
    /// The image-embedder provider id to run. Only `clip_vit_l14` is supported today (validated
    /// server-side); the field exists so a future EVA-CLIP/SigLIP swap is a payload change.
    #[serde(default = "default_dataset_analysis_embedder")]
    pub(crate) embedder: String,
    /// Optional weights snapshot path/id for the embedder; resolved worker-side when absent.
    #[serde(default)]
    pub(crate) model_name_or_path: Option<String>,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// Restrict the analysis to these dataset item ids; when absent, every item is embedded.
    #[serde(default)]
    pub(crate) item_ids: Option<Vec<String>>,
}

/// Import a bounded set of image/caption rows from local Parquet shards into an empty training
/// dataset. The source may be one `.parquet` file or a directory whose direct children are shards.
/// Column matching is case-insensitive, so LAION's `URL`/`TEXT` defaults also accept `url`/`text`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetParquetImportJobRequest {
    pub(crate) source_path: String,
    #[serde(default = "default_parquet_url_column")]
    pub(crate) url_column: String,
    #[serde(default = "default_parquet_caption_column")]
    pub(crate) caption_column: String,
    #[serde(default = "default_parquet_max_items")]
    pub(crate) max_items: usize,
    #[serde(default = "default_parquet_min_edge")]
    pub(crate) min_edge: u32,
    #[serde(default = "default_parquet_concurrency")]
    pub(crate) concurrency: usize,
    /// Case-insensitive caption substrings. A row is retained when any include
    /// term matches; an empty list disables the include filter.
    #[serde(default)]
    pub(crate) caption_includes: Vec<String>,
    /// Case-insensitive caption substrings. A matching row is always rejected.
    #[serde(default)]
    pub(crate) caption_excludes: Vec<String>,
    /// Extra columns imported as per-item named captions (`captions: {column: text}`), the
    /// HF-style multi-field caption source the Iris trainer's `caption_field(s)` read (sc-25685).
    #[serde(default)]
    pub(crate) caption_field_columns: Vec<String>,
}

fn default_parquet_url_column() -> String {
    "URL".to_owned()
}

fn default_parquet_caption_column() -> String {
    "TEXT".to_owned()
}

fn default_parquet_max_items() -> usize {
    1_000
}

fn default_parquet_min_edge() -> u32 {
    512
}

fn default_parquet_concurrency() -> usize {
    16
}

/// Request to run the Dataset Doctor face-embedding pass over a (Person) training dataset (sc-6538).
/// Simpler than the CLIP analysis request — the face stack is fixed (SCRFD + ArcFace), so there is no
/// embedder choice; only GPU routing + an optional item subset.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetFaceAnalysisJobRequest {
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// Restrict the pass to these dataset item ids; when absent, every item is analyzed.
    #[serde(default)]
    pub(crate) item_ids: Option<Vec<String>>,
}

/// Request to describe a training dataset's images for a prompt batch (SW-7082). Every field is
/// optional: `model` defaults to the Vision Captioner (JoyCaption is the other accepted value,
/// SW-7083), an absent or blank `prompt` uses that model's default describing instruction, and
/// `itemIds` restricts the run to those items (e.g. re-describing one image).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetBatchDescribeJobRequest {
    #[serde(default)]
    pub(crate) model: Option<String>,
    /// The model tier (`q4`, `q8`, `bf16`), named `variant` like every other request that picks a
    /// tier. Only the Qwen3.8 27B describer has tiers; absent is its default (`q4`), and naming
    /// one for any other model is refused.
    #[serde(default)]
    pub(crate) variant: Option<String>,
    /// The user's describing instruction. The JSON output contract is always appended by the
    /// worker and is not part of this text.
    #[serde(default)]
    pub(crate) prompt: Option<String>,
    #[serde(default)]
    pub(crate) item_ids: Option<Vec<String>>,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// "Fit to" (SW-7163): the catalog id of an image model that declares
    /// `image.maxPromptTokens`. Each described image is then condensed into one prompt within that
    /// limit by the Vision Captioner; the limit is read from the server's catalog entry.
    #[serde(default)]
    pub(crate) fit_to: Option<String>,
    /// Condense-only (SW-7163): rows already described, as the dialog holds them. Only the
    /// condensing step runs for them; no image is opened. Needs `fitTo`; excludes `itemIds`.
    #[serde(default)]
    pub(crate) condense_rows: Option<Vec<CondenseRowRequest>>,
    /// Measure-only (SW-7163): re-count each `condenseRows` row's `condensedText` (a prompt the
    /// user edited) against the target. Nothing is generated and no model loads.
    #[serde(default)]
    pub(crate) measure_only: bool,
}

/// One already-described row of a condense-only request: its seven fields (the user may have
/// edited them) and the image's pixel size.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CondenseRowRequest {
    pub(crate) item_id: String,
    #[serde(default)]
    pub(crate) fields: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub(crate) width: Option<u32>,
    #[serde(default)]
    pub(crate) height: Option<u32>,
    /// The row's cut-off notice from its describe run, kept on the condensed item.
    #[serde(default)]
    pub(crate) notice: Option<String>,
    /// Measure-only: the edited condensed prompt to re-count.
    #[serde(default)]
    pub(crate) condensed_text: Option<String>,
}

/// `?model=<repo>` on the describing-prompt routes (SW-7085, SW-7086): which describer's default
/// instruction and limit to serve. Absent means the Vision Captioner.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct DescribePromptQuery {
    #[serde(default)]
    pub(crate) model: Option<String>,
}

/// Request to generate SAM3 subject masks over a training dataset (sc-2126). The segmenter is fixed
/// (SAM3 "person"), so the request only carries GPU routing and an optional item subset.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetSubjectMaskJobRequest {
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// Restrict the pass to these dataset item ids; when absent, every item is masked.
    #[serde(default)]
    pub(crate) item_ids: Option<Vec<String>>,
}

/// The subject-mask worker POSTs its generated masks here (sc-2126). Each mask is a base64 PNG keyed
/// by its image's content hash.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetSubjectMasksBody {
    /// The segmentation space the masks came from (e.g. `sam3-person`). Recorded for diagnostics.
    pub(crate) space: String,
    pub(crate) items: Vec<DatasetSubjectMaskRecordBody>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetSubjectMaskRecordBody {
    pub(crate) content_hash: String,
    /// Standard-alphabet base64 of the mask PNG.
    pub(crate) mask_png: String,
}

/// The analysis worker POSTs its computed CLIP embeddings here to persist the sidecar (sc-6535) —
/// the embedding-side analog of the caption job's `/caption-sidecars` write.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetEmbeddingsBody {
    /// The embedding space (e.g. `clip-vit-l14`).
    pub(crate) space: String,
    pub(crate) items: Vec<DatasetEmbeddingRecord>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetEmbeddingRecord {
    /// The item's content hash (the sidecar key — survives dataset edits).
    pub(crate) content_hash: String,
    /// The raw (un-normalized) embedding vector.
    pub(crate) embedding: Vec<f32>,
    /// SHA-256 of the caption text embedded by the worker. Required when `textEmbedding` is present.
    #[serde(default)]
    pub(crate) caption_hash: Option<String>,
    /// The raw (un-normalized) CLIP text embedding for the caption.
    #[serde(default)]
    pub(crate) text_embedding: Option<Vec<f32>>,
}

/// The worker face pass POSTs its largest-face records here to persist the face sidecar (sc-6538) —
/// the face-stack analog of the analysis job's embedding write.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetFaceRecordsBody {
    /// The face-embedding space (e.g. `arcface-r100`).
    pub(crate) space: String,
    pub(crate) items: Vec<DatasetFaceRecord>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetFaceRecord {
    /// The item's content hash (the sidecar key — survives dataset edits).
    pub(crate) content_hash: String,
    /// The largest face's raw (un-normalized) ArcFace embedding. Empty ⇒ examined, no face found.
    #[serde(default)]
    pub(crate) embedding: Vec<f32>,
    /// The largest face's bbox area as a fraction of the frame, in `[0, 1]`. `0.0` when no face.
    #[serde(default)]
    pub(crate) face_fraction: f64,
}

/// Body for the synchronous one-tap image fixes (sc-6539 smart-crop / EXIF-strip). `itemIds` scopes
/// the fix; when absent it applies to every item in the dataset (EXIF-strip-all).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetImageFixBody {
    #[serde(default)]
    pub(crate) item_ids: Option<Vec<String>>,
}

/// Request to upscale flagged low-resolution items in a training dataset (sc-6539 one-tap fix).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetUpscaleJobRequest {
    /// Integer upscale factor (Real-ESRGAN x2 / x4). Defaults to x2.
    #[serde(default = "default_dataset_upscale_factor")]
    pub(crate) factor: u8,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// The dataset item ids to upscale — the readiness report's low-resolution-flagged items.
    pub(crate) item_ids: Vec<String>,
}

pub(crate) fn default_dataset_upscale_factor() -> u8 {
    2
}

/// The upscale worker POSTs its results here to re-point each item at the upscaled child asset
/// (sc-6539) — the dataset-mutation analog of the caption job's `/caption-sidecars` write.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetRepointBody {
    pub(crate) items: Vec<DatasetRepointRecord>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetRepointRecord {
    pub(crate) item_id: String,
    /// The upscaled child asset (lineage back to the original).
    #[serde(default)]
    pub(crate) asset_id: Option<String>,
    /// Project-relative path of the upscaled bytes the worker wrote.
    pub(crate) source_path: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TrainingCaptionOptions {
    #[serde(default = "default_training_caption_type")]
    pub(crate) caption_type: String,
    #[serde(default = "default_training_caption_length")]
    pub(crate) caption_length: String,
    #[serde(default)]
    pub(crate) extra_options: Vec<String>,
    #[serde(default)]
    pub(crate) name_input: String,
    #[serde(default = "default_training_caption_temperature")]
    pub(crate) temperature: f64,
    #[serde(default = "default_training_caption_top_p")]
    pub(crate) top_p: f64,
    #[serde(default = "default_training_caption_max_new_tokens")]
    pub(crate) max_new_tokens: u32,
    #[serde(default)]
    pub(crate) caption_prompt: String,
    #[serde(default)]
    pub(crate) low_vram: bool,
}

impl Default for TrainingCaptionOptions {
    fn default() -> Self {
        Self {
            caption_type: default_training_caption_type(),
            caption_length: default_training_caption_length(),
            extra_options: Vec::new(),
            name_input: String::new(),
            temperature: default_training_caption_temperature(),
            top_p: default_training_caption_top_p(),
            max_new_tokens: default_training_caption_max_new_tokens(),
            caption_prompt: String::new(),
            low_vram: false,
        }
    }
}

pub(crate) use sceneworks_core::generation_contracts::ImageJobRequest;

pub(crate) use sceneworks_core::generation_contracts::{VectorMode, VectorRequest, VectorSampling};

/// Disclosed prompt-to-vector composition. This never means native text-to-SVG: the prompt is
/// rendered by an ordinary text-to-image model and the resulting retained raster is submitted to
/// the vector model in image_to_svg mode.
pub(crate) use sceneworks_core::generation_contracts::VectorPromptWorkflowRequest;

pub(crate) use sceneworks_core::generation_contracts::VqaJobRequest;

pub(crate) use sceneworks_core::generation_contracts::InterleaveJobRequest;

pub(crate) use sceneworks_core::generation_contracts::VideoJobRequest;

/// A `POST /api/v1/audio/jobs` request (SceneWorks Audio Studio, epic 13400 / sc-13404 + sc-13409).
/// The audio analogue of [`VideoJobRequest`]: `prompt` is the script/sound description, and the typed
/// audio knobs (`voice` / `language` / `targetDurationSecs`) map to the worker-side
/// [`gen_core::AudioParams`] sub-block a `Modality::Audio` generator reads. Diffusion-audio models
/// (MOSS-SoundEffect, the Sound FX mode) additionally read the sampling knobs `guidance` (CFG scale)
/// and `steps` off the *top-level* [`gen_core::GenerationRequest`] — not `AudioParams` — so those two
/// are carried here and mapped onto the request by the worker (sc-13409). Every audio knob is optional
/// — an omitted voice/language falls back to the model's default (Kokoro's `af_heart`, `en`), an
/// omitted duration lets the model synthesize its natural length, and an omitted guidance/steps lets
/// the model use its own sampler default (MOSS: CFG 4.0 / 100 steps). `Serialize` too so the whole
/// request round-trips into the job payload the worker claims (mirroring how the video request is
/// `to_json_object`'d).
pub(crate) use sceneworks_core::generation_contracts::AudioJobRequest;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ConditionalComponentsDownloadRequest {
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// The caller asserts the user accepted the declaring entry's licence (sc-23002); required when
    /// the entry carries `requiresLicenseAcknowledgment`, exactly as on the model install.
    #[serde(default)]
    pub(crate) license_acknowledged: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelDownloadRequest {
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// Quant tier to install for a quant-matrix model (sc-8508): `bf16` / `q8` / `q4`, matching
    /// `downloads[].variant`. Absent installs the default tier (the entry marked `default: true`,
    /// else the first supported entry) — the back-compat single-variant behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) variant: Option<String>,
    /// Repair one missing resident row of a plan-backed imported H3 assembly. The server binds
    /// this identifier to its persisted component closure; it never accepts a repo or file path.
    #[serde(default)]
    pub(crate) component_id: Option<String>,
    /// The caller asserts the user has accepted the license of the weights this request FETCHES
    /// (sc-17227). Read by `create_model_download_job` and `create_lora_download_job`, which share
    /// this body. Consulted when the catalog entry the path id names carries
    /// `requiresLicenseAcknowledgment`, AND when any repo the request would queue resolves against
    /// `license_acknowledgment_repo_index` — the same repo-keyed gate `POST /api/v1/jobs` applies,
    /// so the typed and generic doors answer the same repo the same way. The gated repos are
    /// PUBLIC and therefore have no credential check to fail behind. Defaults to `false`, so a
    /// client that does not know about the field is refused rather than let through: the
    /// acknowledgment must be affirmatively asserted, never assumed.
    #[serde(default)]
    pub(crate) license_acknowledged: bool,
    /// The option to install for each co-requisite choice group the model declares (sc-22998),
    /// e.g. `{"decoder": "legacy"}` for YuE2. A group left out installs its manifest default; an
    /// unknown group or option is a 400, never replaced by another option. Only model downloads
    /// read it: the LoRA route that shares this body refuses a non-empty map.
    #[serde(default)]
    pub(crate) choices: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelConvertRequest {
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// MLX quantization bit width (4 or 8). When set, the convert job quantizes the
    /// MLX weights; absent leaves them at the requested float dtype.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) quantize_bits: Option<u32>,
    /// MLX quantization group size (defaults to 64 in the convert tool when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) quantize_group_size: Option<u32>,
}

/// Which ownership mode an import installs under (epic 20398).
///
/// Only `managed` — SceneWorks copies the bytes into its own storage and owns their lifecycle — is
/// served by this route, and it is the default, so every request that predates the field (and every
/// legacy flat `repo`/`sourceUrl`/`sourcePath` body) normalises to it. Linked ownership references
/// a user's library in place and is compiled by
/// `sceneworks_core::checkpoint_plan_store::compile_linked` against an APPROVED root, which is a
/// different operation with a different trust boundary — it never arrives as an import job, so an
/// unknown or `linked` value is refused here rather than silently treated as managed (E7).
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) enum OwnershipModeV1 {
    #[default]
    Managed,
}

/// The discriminated import source (epic 20398, sc-20636). One variant per managed source, so a
/// request states which source it is instead of the server inferring it from which of three
/// mutually-exclusive flat fields happens to be set.
///
/// The legacy flat fields on [`ModelImportRequest`] remain accepted and normalise onto exactly
/// these variants, so both spellings reach the same ingest.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub(crate) enum ModelImportSourceV1 {
    /// A file uploaded in the same multipart request. The staged path is filled in server-side from
    /// the `file` part and is never accepted from client JSON.
    #[serde(rename_all = "camelCase")]
    Upload {
        #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
        staged_path: Option<String>,
    },
    /// A copy of a file or directory the user already has in an app-managed root. The source is
    /// only ever read.
    #[serde(rename_all = "camelCase")]
    LocalPath { path: String },
    /// A plain download URL. A stored credential matching its host is applied by the worker.
    #[serde(rename_all = "camelCase")]
    Url {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expected_sha256: Option<String>,
    },
    /// A Hugging Face repo snapshot.
    #[serde(rename_all = "camelCase")]
    HuggingFace {
        repo: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revision: Option<String>,
        #[serde(default)]
        files: Vec<String>,
    },
    /// A Civitai download. Distinct from [`Self::Url`] only in what it RECORDS: the model version
    /// and file identities Civitai assigns, so the install's provenance names the exact artifact
    /// rather than just a URL that may be re-pointed. The transfer itself is the same
    /// credential-attaching URL download — Civitai has no separate download stack.
    #[serde(rename_all = "camelCase")]
    Civitai {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model_version_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expected_sha256: Option<String>,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelImportRequest {
    /// Absent means `managed`: a client that predates the field gets the ownership every import
    /// already had.
    #[serde(default)]
    pub(crate) ownership_mode: OwnershipModeV1,
    /// The discriminated source. When absent, the flat `repo`/`sourceUrl`/`sourcePath` fields below
    /// are normalised into one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source: Option<ModelImportSourceV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(default, alias = "type", skip_serializing_if = "Option::is_none")]
    pub(crate) model_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_path: Option<String>,
    /// Linked (in-place) ownership: the approved library root the checkpoint lives under
    /// (epic 20398, sc-20635). Paired with [`Self::linked_relative_path`]; the two together are
    /// the checkpoint's durable identity, and no absolute path is ever accepted or persisted (E6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) linked_root_id: Option<String>,
    /// The checkpoint's path relative to its approved root, `/`-separated. Confined by
    /// `CheckpointPlanStore`: absolute, `..`-bearing and backslash-separated paths are refused,
    /// and a path whose canonical target escapes the root is refused again at compile time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) linked_relative_path: Option<String>,
    #[serde(default)]
    pub(crate) files: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) family: Option<String>,
    /// Explicit MiniMax-H3 compatibility: tensor architecture cannot identify its operation partition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) base_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) minimax_h3_partition: Option<String>,
    /// Optional caller-supplied SHA-256 of the imported file. When present, the worker
    /// verifies the downloaded file against it and fails on a mismatch (sc-6137). HF
    /// repo imports are verified automatically from HF's own per-file digests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) expected_sha256: Option<String>,
    /// The caller asserts the user has accepted the licence of the model this import FETCHES
    /// (sc-17227). Consulted only when `repo`/`sourceUrl` resolves to a catalog entry carrying
    /// `requiresLicenseAcknowledgment` — the same repo-keyed gate `POST /api/v1/jobs` applies —
    /// and defaults to `false` so a client that has never heard of the field is refused rather
    /// than let through. Serialized onto the queued job (and only when true) so a retry of an
    /// authorized import re-validates against the assertion the original request carried.
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub(crate) license_acknowledged: bool,
    #[serde(default, skip_deserializing, skip_serializing_if = "bool_is_false")]
    pub(crate) uploaded_source_path: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LoraImportRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) lora_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_path: Option<String>,
    #[serde(default)]
    pub(crate) files: Vec<String>,
    /// Activation keywords collected at import so the catalog entry carries them
    /// from the start. JSON imports send an array; multipart uploads send a
    /// comma-delimited string that the import handler splits.
    #[serde(default)]
    pub(crate) trigger_words: Vec<String>,
    /// Free-text usage guidance (how to combine keywords, recommended weights,
    /// etc.) persisted onto the manifest entry's `notes` field.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) notes: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) family: Option<String>,
    /// Specific base model the LoRA targets (e.g. `wan_2_2_t2v_14b`). Recorded on
    /// the manifest entry so the loader's base-model gating can match it (sc-1955).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) base_model: Option<String>,
    /// Optional caller-supplied SHA-256 of the imported file. When present, the worker
    /// verifies the downloaded file against it and fails on a mismatch (sc-6137). HF
    /// repo imports are verified automatically from HF's own per-file digests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) expected_sha256: Option<String>,
    /// The caller asserts the user has accepted the licence of the model this import FETCHES
    /// (sc-17227) — the same field, default, and meaning as on [`ModelImportRequest`]. A LoRA
    /// import takes a caller-supplied `repo`/`sourceUrl` and never looks it up in the LoRA catalog,
    /// so it can name a licence-restricted MODEL repo; the gate is keyed on that repo, not on
    /// anything the LoRA catalog declares. Serialized onto the queued job only when true, so a
    /// retry of an authorized import re-validates against the assertion the request carried.
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub(crate) license_acknowledged: bool,
    #[serde(default = "default_lora_scope")]
    pub(crate) scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) project_id: Option<String>,
    #[serde(default, skip_deserializing, skip_serializing_if = "bool_is_false")]
    pub(crate) uploaded_source_path: bool,
    /// Staged path of the low-noise expert half for a Wan A14B MoE upload (sc-1991).
    /// Set server-side from the `secondaryFile` multipart part only; never from
    /// client JSON. Its presence flags the import as a paired MoE write.
    #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub(crate) secondary_source_path: Option<String>,
}

/// PATCH body for editing a catalog LoRA's user-facing metadata after import.
/// Only the fields present are changed (absent field = left untouched), so the UI
/// can send just the keywords, just the notes, or both.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LoraUpdateRequest {
    #[serde(default)]
    pub(crate) trigger_words: Option<Vec<String>>,
    #[serde(default)]
    pub(crate) notes: Option<String>,
    /// User-assigned architecture family (sc-24163). Accepted ONLY when the adapter file's own
    /// header does not resolve a family — a detected family always wins.
    #[serde(default)]
    pub(crate) family: Option<String>,
}

/// Query params identifying a single catalog LoRA across scopes, shared by the
/// metadata PATCH and the embedded-tags lookup. Mirrors `CatalogDeleteQuery`'s
/// scope/project routing without the delete-only `permanent` flag.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LoraCatalogItemQuery {
    pub(crate) project_id: Option<String>,
    pub(crate) scope: Option<String>,
}

#[cfg(test)]
pub(crate) use sceneworks_core::generation_contracts::VectorDetailBudget;
