use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::io::SeekFrom;
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body};
use axum::extract::rejection::JsonRejection;
use axum::extract::{
    DefaultBodyLimit, FromRequest, MatchedPath, Multipart, Path, Query, Request as AxumRequest,
    State,
};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post, put};
use axum::{Json, Router};
use futures_util::{future::join_all, stream};
use parking_lot::Mutex;
use sceneworks_core::contracts::{
    CancelPendingJobsRequest, CancelPendingJobsResponse, ClaimRequest, ClaimResponse,
    ClearJobsRequest, ClearJobsResponse, ContractNumber, DuplicateJobRequest, GenerationMetrics,
    GenerationMetricsRow, ImageUpscaleRequest, JobCreateRequest, JobSnapshot, JobStatus, JobType,
    JsonObject, PrioritizeJobsRequest, PrioritizeJobsResponse, ProgressRequest, QueueSummary,
    RetryJobRequest, WorkerCapability, WorkerHeartbeatRequest, WorkerRegisterRequest,
    WorkerSnapshot, WorkerStatus, WorkerTerminationRequest,
};
use sceneworks_core::hf_home::{huggingface_hub_cache_dir, huggingface_repo_cache_path};
use sceneworks_core::image_request::{
    default_count as image_default_count, default_resolution as image_default_resolution,
};
use sceneworks_core::jobs_store::{
    candle_supported, mac_capabilities, mac_rust_supported, model_candle_support,
    model_mac_support, video_job_type_for_mode, video_request_is_claimable_by_any_lane, CreateJob,
    DuplicateJob, JobsStore, JobsStoreError, MacCapabilities, ProgressUpdate, RegisterWorker,
    RetryJob, RouteDecision, StaleSweep, UnsupportedReason, WorkerHeartbeat, JOB_STATUSES,
};
use sceneworks_core::lora_family::{
    accepted_lora_families, apply_adapter_metadata_to_manifest_entry,
    apply_model_manifest_defaults, canonical_lora_family, detect_lora_family, detect_model_family,
    first_safetensors_path, read_adapter_metadata, read_safetensors_header,
    reconcile_detected_family, AdapterFileMetadata, SafetensorsHeaderError,
};
use sceneworks_core::lora_url::{lora_source_url_file_stem, parse_lora_source_url, LoraUrlError};
use sceneworks_core::project_store::{
    AssetStatusPatch, AssetTagsPatch, CharacterCreateInput, CharacterLookInput,
    CharacterLookUpdateInput, CharacterLoraInput, CharacterLoraUpdateInput,
    CharacterReferenceInput, CharacterReferenceUpdateInput, CharacterUpdateInput, ProjectStore,
    ProjectStoreError, UploadAsset, KEYPOINT_UPLOADS_CACHE_DIR, POSE_UPLOADS_CACHE_DIR,
};
use sceneworks_core::time::{format_unix_seconds, now_unix_seconds};
use sceneworks_core::training::{
    build_training_plan, builtin_training_presets, builtin_training_targets, BuildTrainingPlan,
    LoraTrainingRequest, TrainingDataset, TrainingPresetProvenance, TrainingTarget,
    TrainingTargetRegistry,
};
use sceneworks_core::training_store::{
    DatasetItemRepoint, TrainingCaptionSidecarsResult, TrainingDatasetBatchRenameInput,
    TrainingDatasetCaptionSidecarsInput, TrainingDatasetCreateInput, TrainingDatasetMutationResult,
    TrainingDatasetSummary, TrainingDatasetUpdateInput,
};
use sceneworks_core::video_request::{
    classify_reference_set, default_resolution, duration_limit_error, fps_limit_error,
    reference_limit_error, requested_steps, resolve_duration, resolve_fps, steps_limit_error,
    ReferenceSetVerdict,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio::time::{Instant as TokioInstant, MissedTickBehavior};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tokio_util::io::ReaderStream;
use tower_http::compression::{
    predicate::{Predicate, SizeAbove},
    CompressionLayer, CompressionLevel,
};
use tower_http::cors::{AllowOrigin, CorsLayer};
use uuid::Uuid;

mod auth;
use auth::{access_control, cors_layer, is_authorized, AuthThrottle};
mod catalog_scan_supervisor;
mod startup;
use startup::{StartupCriticality, StartupMaintenance, StartupPhaseTimer};
mod saved_voices;
use saved_voices::{create_saved_voice, delete_saved_voice, list_saved_voices};
mod characters;
use characters::{
    add_character_reference, archive_character, attach_character_lora, create_character,
    create_character_look, create_character_test_job, delete_character_look, detach_character_lora,
    get_character, list_characters, purge_character, remove_character_reference, update_character,
    update_character_look, update_character_lora, update_character_reference,
};
mod timelines;
use timelines::{
    create_timeline, create_timeline_export, extract_timeline_frame, get_timeline, list_timelines,
    update_timeline,
};
mod person;
use person::{
    create_person_detection_job, create_person_track_job, get_person_track, list_person_tracks,
    save_person_track_corrections,
};
mod projects;
use projects::{create_project, get_project, list_projects, reindex_project_endpoint};
mod catalogs;
use catalogs::{
    attach_catalog, catalog_curation_facets, catalog_facets, catalog_record_thumbnail,
    create_catalog, create_catalog_saved_view, curate_catalog, delete_catalog_on_disk,
    delete_catalog_saved_view, detach_catalog, get_catalog, get_catalog_status,
    list_catalog_saved_views, list_catalogs, materialize_catalog_results, pause_catalog,
    query_catalog, resume_catalog, review_catalog_record, run_catalog_analysis,
    update_catalog_analyzer_config, update_catalog_saved_view,
};
mod assets;
use assets::{
    delete_asset, get_asset, get_asset_poster, import_asset, list_assets, move_asset_to_character,
    move_asset_to_library, purge_asset, sweep_stale_asset_uploads, update_asset_status,
    update_asset_tags, write_upload_field_to_dir, write_upload_field_to_temp_file,
};
mod workflows;
use workflows::{get_asset_workflow, inspect_workflow, max_inspect_multipart_body_bytes};
// Test-only crate-root imports: the `tests` module reaches these helpers via
// `super::` (either `use super::{...}` or a fully-qualified `super::fn(...)` call).
// Gating them keeps the non-test build warning-free — they have no non-test
// crate-root consumer.
#[cfg(test)]
use assets::sweep_stale_asset_uploads_before;
mod training;
use training::{
    batch_rename_training_dataset_items, create_training_dataset,
    create_training_dataset_analysis_job, create_training_dataset_caption_job,
    create_training_dataset_face_analysis_job, create_training_dataset_parquet_import_job,
    create_training_dataset_upscale_job, create_training_job, delete_training_dataset,
    finalize_training_dataset_parquet_import, get_training_dataset, get_training_dataset_readiness,
    list_training_datasets, list_training_presets, list_training_targets,
    repoint_training_dataset_items, resolve_control_overlay_output_location,
    resolve_finetune_output_location, resolve_training_output_location,
    set_training_dataset_item_quality_ack, smart_crop_training_dataset_items,
    strip_exif_training_dataset_items, trusted_adapter_files, trusted_base_checkpoint_files,
    update_training_dataset, upload_training_dataset_item, validate_lora_id_component,
    write_training_dataset_analysis_embeddings, write_training_dataset_caption_sidecars,
    write_training_dataset_face_embeddings,
};
mod generation;
use generation::{
    create_audio_job, create_image_job, create_interleave_job, create_video_job, create_vqa_job,
    parse_recipe_preset_resolution, typed_generation_route, JobCatalogSnapshot,
};
#[cfg(test)]
use generation::{validate_interleave_job, validate_vqa_job};
mod ideogram;
mod jobs;
use jobs::{
    cancel_job, cancel_pending_jobs, claim_job, clear_job, clear_jobs, create_job, duplicate_job,
    get_job, get_job_metrics, invalidate_model_catalog_for_terminal_jobs, list_jobs, list_metrics,
    prioritize_jobs, retry_job, update_job_progress, upsert_job_metrics,
};
mod workers;
use workers::{
    heartbeat_worker, host_capabilities, list_workers, mac_capability_support,
    person_capability_readiness, queue_summary, register_worker, request_worker_restart,
    worker_terminated,
};
mod events;
use events::{create_event_ticket, job_events, EventHub, EventMessage};
mod tickets;
use tickets::{create_media_ticket, EventTicketContext, TicketResponse, TicketStore};
mod dto;
use dto::{
    AccessResponse, AssetPurgeQuery, AssetsQuery, AudioJobRequest, CatalogDeleteQuery,
    CharacterCreateRequest, CharacterLookRequest, CharacterLookUpdateRequest, CharacterLoraRequest,
    CharacterLoraUpdateRequest, CharacterReferenceRequest, CharacterReferenceUpdateRequest,
    CharacterTestRequest, CharacterUpdateRequest, CharactersQuery, CreateEventTicketRequest,
    DatasetAnalysisJobRequest, DatasetEmbeddingsBody, DatasetFaceAnalysisJobRequest,
    DatasetFaceRecordsBody, DatasetImageFixBody, DatasetParquetImportJobRequest,
    DatasetRepointBody, DatasetUpscaleJobRequest, DirectoriesResponse, EventsQuery,
    FaceLikenessCompareRequest, FrameExtractRequest, HealthResponse, HostCapabilitiesResponse,
    ImageJobRequest, InterleaveJobRequest, JobsQuery, LoraCatalogItemQuery, LoraImportRequest,
    LoraUpdateRequest, LorasQuery, MetricsQuery, ModelConvertRequest, ModelDownloadRequest,
    ModelImportRequest, ModelImportSourceV1, OwnershipModeV1, PersonDetectionJobRequest,
    PersonTrackCorrectionsRequest, PersonTrackJobRequest, ProjectCreateRequest, PromptBatchesQuery,
    PromptRefineRequest, QualityAckBody, ReadinessQuery, RecipePresetsQuery,
    SavedVoiceCreateRequest, StartupReadinessResponse, TimelineCreateRequest,
    TimelineExportRequest, TimelineSaveRequest, TrainingCaptionJobRequest, VerifyResponse,
    VideoJobRequest, VqaJobRequest,
};
mod manifest;
// The linked-library lifecycle seam (epic 20398, sc-20635): approve, rename, relink, scan, rescan
// and forget an approved checkpoint library. AC1's six verbs, reachable from a client.
mod checkpoint_library;
// The single model-source seam every job-creation path calls (sc-19708): generic carrier
// attachment + typed external-library availability preflight, all data-driven.
mod model_library;
mod model_sources;
// Read + control surface for the app-owned resolved-model hot cache (sc-19711): status for the
// Settings storage card, and the per-model keep/remove operations the Model Manager drives.
mod model_cache;
use manifest::{
    acquire_manifest_file_lock, load_manifest_entries, manifest_write_lock, merge_entries_by_id,
    merge_object, mutate_manifest_entries, remove_catalog_manifest_entry, write_manifest_atomic,
    ManifestCache,
};
#[cfg(test)]
use manifest::{strip_jsonc_comments, API_MANAGED_MANIFEST_HEADER};
use model_cache::{
    get_model_cache, preview_model_cache_removal, remove_model_cache_entry, set_model_cache_pin,
};
mod models;
use models::{
    create_model_convert_job, create_model_download_job, create_model_import_job, delete_model,
    delete_model_variant, list_models, model_catalog, model_is_installed,
    resolve_model_manifest_entry, resolve_selected_image_text_encoder, ModelCatalogCache,
    ModelSizeCache,
};
#[cfg(test)]
use models::{
    download_size_from_siblings, inject_converted_model_path, manifest_download_size_bytes,
    merge_model_manifest_entry, mlx_catalog_status, model_co_requisite_downloads, model_download,
    retain_downloads_for_os,
};
mod control_overlays;
mod external_base_models;
mod external_loras;
use control_overlays::list_control_overlays;
mod loras;
use loras::{
    create_lora_download_job, create_lora_import_job, delete_lora, list_loras, lora_catalog,
    lora_embedded_tags, lora_url_error_message, sweep_stale_lora_uploads, update_lora,
    validate_job_lora_compatibility, validate_job_lora_compatibility_with,
    validate_lora_specs_for_model,
};
#[cfg(test)]
use loras::{lora_artifact_paths, lora_families, sweep_stale_lora_uploads_before};
mod manifest_entity;
mod recipe_presets;
use recipe_presets::{
    create_recipe_preset, delete_recipe_preset, duplicate_recipe_preset, get_recipe_preset,
    list_recipe_presets, preset_lora_id, preset_lora_weight, preset_prompt,
    recipe_preset_catalog_with, recipe_preset_loras, serialize_preset_lora,
    stamp_recipe_preset_used, update_recipe_preset,
};
mod prompt_batches;
use prompt_batches::{
    create_prompt_batch, delete_prompt_batch, duplicate_prompt_batch, get_prompt_batch,
    list_prompt_batches, update_prompt_batch,
};
mod styles;
use styles::list_styles;
mod credentials;
use credentials::{delete_credential, list_credentials, set_credential};
mod preferences;
use preferences::{get_ui_preferences, set_ui_preferences};
mod prompts;
use prompts::create_prompt_refine_job;
// On-demand "compare image to another" likeness tool (epic 4406, sc-4415): enqueues a
// `face_likeness_compare` job scoring a candidate asset against a source identity reference asset.
mod face_likeness;
use face_likeness::create_face_likeness_compare_job;
mod poses;
use poses::{create_pose_sources, create_poses, get_pose_preview, sweep_stale_pose_uploads};
mod keypoints;
use keypoints::{
    create_keypoint, create_keypoint_sources, delete_keypoint_collection,
    list_keypoint_collections, list_keypoint_presets, set_default_keypoint_collection,
    sweep_stale_keypoint_uploads, upsert_keypoint_collection,
};
mod logs;
use logs::list_logs;
// The shared HTTP error type (sc-8890, F-088), re-exported so the `use super::*`
// in every handler module keeps resolving `ApiError` unchanged.
mod error;
pub(crate) use error::ApiError;
// Serde `#[serde(default = "...")]` value providers for the DTOs (sc-8890, F-088),
// re-exported so the `#[serde(default = "default_x")]` string paths and sibling
// call sites keep resolving unchanged.
mod defaults;
pub(crate) use defaults::*;
// The process-lifecycle surface — `Settings`, `AppState`, and the `run`/`run_worker`
// binary entrypoints (sc-9736, the deferred remainder of F-088). Re-exported so
// `main.rs` (`run`/`run_worker`), the handler modules' `use super::*` (`AppState`),
// and `tests.rs` (`Settings`) keep resolving these paths unchanged.
mod server;
pub use server::{run, run_worker, AppState, Settings};

// The theme-preferences route. Its GET (the pre-auth theme read) is public, but its
// PUT writes `ui-preferences.json` to disk, so the exemption is method-aware — the
// PUT is gated when a token is configured (sc-8869, F-067). See `auth::requires_token`.
const UI_PREFERENCES_PATH: &str = "/api/v1/ui-preferences";
const PUBLIC_PATHS: &[&str] = &[
    "/api/v1/health",
    "/api/v1/access",
    "/api/v1/auth/verify",
    "/api/v1/jobs/events",
    // Non-sensitive UI state (theme); the GET is loaded before auth to avoid a
    // flash. The PUT is method-gated in `auth::requires_token`, not here.
    UI_PREFERENCES_PATH,
];
const DEFAULT_CORS_ORIGINS: &str = concat!(
    "http://localhost:5173,http://127.0.0.1:5173,",
    "http://localhost:5174,http://127.0.0.1:5174,",
    "http://localhost:5175,http://127.0.0.1:5175,",
    "http://localhost:5176,http://127.0.0.1:5176"
);
const EVENT_BUFFER_SIZE: usize = 100;
// SSE tickets are single-use and consumed on connect, so a tight window suffices.
const EVENT_TICKET_TTL_SECONDS: u64 = 30;
// Reconnect context is bounded independently from the router-wide 10 MiB JSON
// allowance. 1,024 active UUIDs preserves the tested 600-job reconnect case;
// terminal IDs mirror the web client's 200-row retained-history cap.
const MAX_EVENT_TICKET_ACTIVE_JOB_IDS: usize = 1024;
const MAX_EVENT_TICKET_TERMINAL_JOB_IDS: usize = 200;
const MAX_EVENT_TICKET_JOB_ID_BYTES: usize = 128;
const MAX_EVENT_TICKET_CONTEXT_BYTES: usize = 48 * 1024;
const MAX_EVENT_TICKET_BODY_BYTES: usize = 64 * 1024;
const MAX_OUTSTANDING_EVENT_TICKETS: usize = 128;
// Media tickets ride in <img>/<video>/<a download> URLs (headers impossible), so
// they are multi-use; the web client re-arms the sliding ticket every TTL/3, and a
// leaked URL dies at most one TTL after the last authenticated refresh (sc-8810).
const MEDIA_TICKET_TTL_SECONDS: u64 = 300;
const HEARTBEAT_SSE_DATA: &str = "{}";
#[cfg(test)]
const HEARTBEAT_SSE_WIRE: &str = "event: heartbeat\ndata: {}\n\n";
// sc-4201 (F-API-1): default to loopback so a bare/server run that doesn't set
// SCENEWORKS_API_HOST isn't exposed to the whole LAN with auth off. Docker and the
// desktop wrapper set the host explicitly (0.0.0.0 / 127.0.0.1 respectively), so this
// only changes the out-of-the-box default for a direct binary run.
const DEFAULT_API_HOST: &str = "127.0.0.1";
// sc-8812 (F-010): router-wide default body limit for JSON/non-upload routes. The
// 2 GiB `MAX_UPLOAD_BYTES` cap is sized for streaming multipart asset uploads and is
// far too large to apply globally — every JSON route (`POST /jobs`, `/image/jobs`,
// progress, presets, keypoint collections, ...) buffers the whole body into memory
// before parsing, so a router-wide 2 GiB cap lets any authenticated/loopback caller
// drive multi-GiB memory spikes (a one-request DoS lever). 10 MiB leaves generous
// headroom for the largest legitimate JSON payloads (batch keypoints, timelines,
// person-track corrections) while shrinking the per-request ceiling ~200x. The large
// limit is re-attached per-route ONLY to the multipart/upload endpoints below.
const MAX_JSON_BODY_BYTES: usize = 10 * 1024 * 1024;
// The trusted worker may finalize up to 100k URL/caption records in one atomic
// dataset replacement. Captions are bounded separately, but the JSON envelope can
// still exceed the ordinary API ceiling by hundreds of megabytes.
const MAX_PARQUET_FINALIZE_BODY_BYTES: usize = 512 * 1024 * 1024;
const MAX_UPLOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_MODEL_UPLOAD_BYTES: usize = 256 * 1024 * 1024 * 1024;
const MAX_LORA_MULTIPART_BODY_BYTES: usize = MAX_UPLOAD_BYTES + 16 * 1024 * 1024;
const MAX_MODEL_MULTIPART_BODY_BYTES: usize = MAX_MODEL_UPLOAD_BYTES + 16 * 1024 * 1024;
// sc-15950: `POST /api/v1/workflows/inspect` deliberately does NOT get `MAX_UPLOAD_BYTES`.
//
// The asset upload route pays 2 GiB because the bytes BECOME the asset — the write is the point.
// Inspect streams the body to `cache/uploads` before any PNG check, throws it away, and returns a
// small JSON report, so a 2 GiB body buys 2 GiB of transient disk and one blocking thread for a
// result measured in kilobytes. There is no rate limit, concurrency limit or timeout layer in this
// app (tower-http is compression + cors only) and the desktop sets `SCENEWORKS_TRUST_LOOPBACK`, so
// any local process can drive N of those concurrently without a token.
//
// 512 MiB is sized against what users actually drop, not against a round number. The largest PNG
// SceneWorks itself can write is a 4096² generation (`image_request::MAX_DIMENSION`) through a 4×
// upscale = 16384², 8-bit RGB (`workflow_png::write_workflow_chunk` takes an `RgbImage`) = 805 MB
// raw, which PNG's filtered deflate puts well under 512 MiB on rendered content; an ordinary 4096²
// share is ~50 MB, and a phone photo is single-digit MB. So the cap rejects nothing a user would
// plausibly drop while cutting the transient-disk exposure 4×.
const MAX_WORKFLOW_INSPECT_BYTES: usize = 512 * 1024 * 1024;
// The route limit must sit ABOVE the per-field cap, not equal to it: a body exactly at the cap
// plus multipart framing (boundaries, headers, the trailing CRLF) exceeds an equal router limit,
// so axum's own limit trips first and `field.chunk()` surfaces it as a plain 400 — the typed 413
// would then be unreachable at the only boundary that matters. Same headroom, for the same reason,
// as `MAX_LORA_MULTIPART_BODY_BYTES`. The route reads it through
// `workflows::max_inspect_multipart_body_bytes`, which derives it from the live per-field cap so
// the two cannot drift back into equality and so a test can exercise the real boundary at a
// sendable size.
const MULTIPART_FRAMING_HEADROOM_BYTES: usize = 16 * 1024 * 1024;
const MAX_WORKFLOW_INSPECT_MULTIPART_BODY_BYTES: usize =
    MAX_WORKFLOW_INSPECT_BYTES + MULTIPART_FRAMING_HEADROOM_BYTES;
// sc-8885 (F-083): the shared max age for every `cache/*-uploads` staging area (asset,
// lora, model, pose, keypoint) before the startup sweep reclaims it. Named for uploads
// in general — the old `STALE_LORA_UPLOAD_SECONDS` misleadingly implied LoRA-only.
const STALE_UPLOAD_SECONDS: u64 = 24 * 60 * 60;
// sc-8884 (F-082): the char cap applied to every free-text prompt field (`prompt` and
// `negativePrompt`). Both are persisted into jobs.db and re-broadcast over SSE on every
// `job.updated`, so an uncapped field bloats the row and every subscriber's payload.
const MAX_PROMPT_CHARS: usize = 4000;
// sc-8884 (F-082): serialized-size ceiling for the free-form `advanced` object. It is a
// pass-through bag threaded to the worker, so it has no per-key schema — bound its total
// serialized size instead. 64 KiB is generous for legitimate advanced settings.
const MAX_ADVANCED_JSON_BYTES: usize = 64 * 1024;
// Thread-local (not a process-global atomic) so a test overriding the cap to
// exercise the size limit can't leak that value into other LoRA-upload tests
// running concurrently on sibling threads. `#[tokio::test]` uses a current-thread
// runtime, so the upload handler runs on the same thread that sets the override.
#[cfg(test)]
thread_local! {
    static TEST_MAX_LORA_UPLOAD_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    // sc-15950: same override, same reasoning, for `POST /api/v1/workflows/inspect`. The real cap
    // is `MAX_UPLOAD_BYTES` (2 GiB), which no test can send, so the oversized-body branch is only
    // reachable through a lowered cap.
    static TEST_MAX_WORKFLOW_INSPECT_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
static TEST_MAX_MODEL_UPLOAD_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// sc-8819 (F-017): count how many times the model/LoRA catalogs are assembled (which
// each trigger the whole per-model filesystem install-state probe sweep) so a test can
// assert a preset job-create builds each catalog once, not 2–3×. Thread-local, and the
// counter is bumped on the caller's async task thread (before the catalog's inner
// `spawn_blocking`), so under the `#[tokio::test]` current-thread runtime the count is
// observed on the test thread and is immune to parallel tests on sibling threads.
#[cfg(test)]
thread_local! {
    static TEST_MODEL_CATALOG_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_LORA_CATALOG_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn test_reset_catalog_build_counters() {
    TEST_MODEL_CATALOG_BUILDS.with(|cell| cell.set(0));
    TEST_LORA_CATALOG_BUILDS.with(|cell| cell.set(0));
}

#[cfg(test)]
pub(crate) fn test_model_catalog_builds() -> usize {
    TEST_MODEL_CATALOG_BUILDS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn test_lora_catalog_builds() -> usize {
    TEST_LORA_CATALOG_BUILDS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn test_note_model_catalog_build() {
    TEST_MODEL_CATALOG_BUILDS.with(|cell| cell.set(cell.get() + 1));
}

#[cfg(test)]
pub(crate) fn test_note_lora_catalog_build() {
    TEST_LORA_CATALOG_BUILDS.with(|cell| cell.set(cell.get() + 1));
}

struct ApiJson<T>(T);

#[axum::async_trait]
impl<S, T> FromRequest<S> for ApiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(request: AxumRequest, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(request, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) => Err(json_rejection_response(rejection)),
        }
    }
}

// sc-4201 (F-API-1): true when the API would serve every endpoint without auth to
// the network — no access token AND a non-loopback bind address. Pure so the security
// decision is unit-tested without spinning up a listener.
fn should_warn_open_bind(access_token: &str, ip: std::net::IpAddr) -> bool {
    access_token.trim().is_empty() && !ip.is_loopback()
}

// sc-5720 (API-001): an operator may knowingly opt into an unauthenticated wider
// bind (e.g. a trusted-network deployment that fronts its own auth) by setting
// `SCENEWORKS_ALLOW_OPEN_BIND=1`. Pure + tested alongside `should_warn_open_bind`.
fn open_bind_override_enabled(value: &str) -> bool {
    matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES")
}

/// Choose the builtin-manifest seed mode from the raw `SCENEWORKS_CONFIG_DIR` and
/// `SCENEWORKS_OWN_MANIFESTS` env values (sc-10212, sc-15504).
///
/// An explicit, non-empty `SCENEWORKS_CONFIG_DIR` marks an operator-owned config dir — a repo checkout
/// or a Compose bind mount / RunPod persistent volume — so seed `SyncFromEmbedded`: refresh each builtin
/// manifest when it is missing or has drifted from the binary's embedded copy, but leave a byte-identical
/// file untouched so a matching checkout is never dirtied. The builtin manifests are app-owned (nothing
/// edits them at runtime — customizations live in `user.*.jsonc`), so a persisted copy that no longer
/// matches the running binary is normally stale and must be refreshed. The old always-`IfMissing`
/// behavior — never rewriting an existing file — left an upgraded binary serving a months-old
/// `builtin.models.jsonc` off a persisted volume: it hid the sc-10193 img2img flag once and the Krea
/// Turbo memory-ladder curves again (the ladder was bypassed, so a 24 GB card wrongly rejected a
/// q4/1024² render).
///
/// A deployment that intentionally SHIPS its own `builtin.*.jsonc` and wants it used verbatim (a
/// customized bind mount, or the contract-snapshot test harness) opts out with a truthy
/// `SCENEWORKS_OWN_MANIFESTS` — that forces `IfMissing`: fill only genuinely-missing manifests, never
/// self-heal what the operator provided. The opt-out only applies with an explicit config dir; on the
/// platform-default app-owned dir it is ignored.
///
/// Unset or blank `SCENEWORKS_CONFIG_DIR` means `config_dir` fell back to the platform-default app-owned
/// dir (the same one the desktop seeds `Overwrite`), so `Overwrite` there refreshes the builtin catalog
/// unconditionally on launch. Pure so the choice is unit-tested without touching process env or the
/// filesystem.
///
/// The trim/non-empty rule mirrors [`env_path_or`] exactly, so the seed mode and the resolved
/// `config_dir` always agree on whether the override was actually applied.
fn seed_mode_for_config_dir(
    config_dir_env: Option<&str>,
    own_manifests_env: Option<&str>,
) -> sceneworks_core::builtin_manifests::SeedMode {
    use sceneworks_core::builtin_manifests::SeedMode;
    let explicit_config_dir = config_dir_env
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    if !explicit_config_dir {
        return SeedMode::Overwrite;
    }
    let own_manifests = own_manifests_env
        .map(str::trim)
        .is_some_and(|value| matches!(value, "1" | "true" | "TRUE" | "yes" | "YES"));
    if own_manifests {
        SeedMode::IfMissing
    } else {
        SeedMode::SyncFromEmbedded
    }
}

fn json_rejection_response(rejection: JsonRejection) -> Response {
    // sc-8812 (F-010): a body over the route's `DefaultBodyLimit` surfaces here as a
    // `BytesRejection` whose own status is 413. Preserve the rejection's status code
    // instead of flattening everything to 422, so an oversized body is reported as
    // PAYLOAD_TOO_LARGE (and the DoS-guard is observable), while genuine decode/parse
    // failures keep their existing 422 shape.
    let status = rejection.status();
    if status == StatusCode::PAYLOAD_TOO_LARGE {
        return (status, Json(json!({ "detail": rejection.body_text() }))).into_response();
    }
    let detail = match rejection {
        JsonRejection::JsonDataError(error) => error.body_text(),
        JsonRejection::JsonSyntaxError(error) => error.body_text(),
        other => other.body_text(),
    };
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({
            "detail": [{
                "type": "json_invalid",
                "loc": ["body", 0],
                "msg": "JSON decode error",
                "input": {},
                "ctx": { "error": detail }
            }]
        })),
    )
        .into_response()
}

/// Run this binary as a standalone worker process instead of the HTTP API.
/// One-shot GPU preflight (sc-8411 Metal, sc-16247 CUDA). Dispatched from `main` when
/// `SCENEWORKS_GPU_CHECK=1`: a probe that the desktop spawns at startup. Reuses this
/// binary because it already links the GPU backends (the desktop crate does not), and
/// runs the probe in the SAME process/spawn context the real worker uses, so it
/// faithfully predicts whether the worker can acquire a GPU. `Ok(())` when usable;
/// `Err(message)` is the user-facing reason the desktop relays onto the setup screen.
///
/// Exactly one of the two probes is ever live in a given build: `metal_preflight` is a
/// no-op off-Mac, and `cuda_preflight` is a no-op on macOS and on any off-Mac build
/// without `backend-candle`. Calling both keeps this dispatch platform-agnostic — the
/// desktop decides *whether* to spawn the probe, not which one runs.
pub fn gpu_check() -> Result<(), GpuCheckFailure> {
    if let Err(message) = sceneworks_worker::metal_preflight() {
        // Metal's contract is unchanged from sc-8411: any failure blocks startup.
        return Err(GpuCheckFailure {
            message,
            blocking: true,
        });
    }
    sceneworks_worker::cuda_preflight().map_err(|message| GpuCheckFailure {
        blocking: sceneworks_worker::cuda_failure_is_blocking(&message),
        message,
    })
}

/// A failed [`gpu_check`], plus whether it should stop the app from starting.
///
/// `blocking` is the whole reason this isn't a bare `String`. The desktop can't make the call
/// itself — it doesn't link the worker crate and so has no access to the CUDA error table — and
/// getting it wrong is asymmetric: over-blocking locks the user out of the entire application
/// over what may be a transient GPU state, while under-blocking costs one failed job that now
/// carries an actionable message anyway. See `sceneworks_worker::cuda_failure_is_blocking`.
/// Crosses the process boundary as an exit code (see `main.rs`), the message on stdout.
pub struct GpuCheckFailure {
    pub message: String,
    pub blocking: bool,
}

/// Spawns the in-process CPU utility worker pool ([`sceneworks_worker::run_worker_loop`])
/// as tokio tasks in this process, pointed at the local API over loopback. Each loop
/// observes the same Ctrl+C/SIGTERM as the HTTP server (via the worker's own shutdown
/// handling), so `shutdown()` only bounds the wait by the worker's configured grace
/// period.
///
/// The count comes from [`inprocess_utility_worker_count`] (default 2). A single worker
/// claims one job at a time, so a lone in-process worker serialized *all* CPU utility
/// work — most visibly, model/LoRA downloads queued one-at-a-time on the desktop
/// (sc-10723). Running ≥2 loops lets independent downloads proceed in parallel; the
/// per-file `DownloadLock` (sc-8900) still serializes two jobs resolving the *same*
/// cache target, so concurrency never corrupts a shared file.
async fn spawn_inprocess_utility_worker(
    port: u16,
    data_dir: PathBuf,
) -> Result<InProcessUtilityWorker, sceneworks_worker::WorkerError> {
    let mut worker_settings = sceneworks_worker::Settings::from_env();
    worker_settings.api_url = format!("http://127.0.0.1:{port}");
    worker_settings.data_dir = data_dir;
    worker_settings.gpu_id =
        inprocess_worker_gpu_id(std::env::var("SCENEWORKS_RUST_WORKER_GPU_ID").ok());
    // This API process is the top-level owner of the in-process utility loops, which deliberately
    // call `run_worker_loop` directly. Recover once here before any loop can claim a conversion;
    // never put recovery inside each child loop, where sibling restarts could sweep live backups.
    {
        let _phase = StartupPhaseTimer::start(
            "utility_conversion_recovery",
            StartupCriticality::ReadinessCritical,
        );
        sceneworks_worker::recover_stranded_model_conversions(&worker_settings.data_dir).await?;
    }
    let grace = Duration::from_secs(worker_settings.shutdown_timeout_seconds.max(1));
    let count = inprocess_utility_worker_count();
    let base_worker_id = worker_settings.worker_id.clone();
    let handles = (0..count)
        .map(|index| {
            let mut settings = worker_settings.clone();
            settings.worker_id = inprocess_utility_worker_id(&base_worker_id, index);
            tracing::info!(
                event = "utility_worker_inprocess",
                apiUrl = %settings.api_url,
                workerId = %settings.worker_id,
                index,
                count,
                "SceneWorks utility worker running in-process (loopback)"
            );
            tokio::spawn(async move { sceneworks_worker::run_worker_loop(settings).await })
        })
        .collect();
    Ok(InProcessUtilityWorker { handles, grace })
}

/// Number of in-process CPU utility worker loops to run. Defaults to **2** so desktop
/// model/LoRA downloads (and other CPU utility jobs) run two-at-a-time instead of
/// serializing behind a single worker; `SCENEWORKS_UTILITY_WORKERS` overrides it
/// (clamped to >= 1). This default is intentionally more conservative than the
/// standalone/Docker worker pool's `Settings::utility_workers` default of 4, because the
/// same knob also governs CPU/RAM-heavy conversions/imports that share this pool.
fn inprocess_utility_worker_count() -> usize {
    parse_inprocess_utility_worker_count(std::env::var("SCENEWORKS_UTILITY_WORKERS").ok())
}

/// Pure parser behind [`inprocess_utility_worker_count`] (env split out so it is unit
/// testable): a present, parseable value wins (clamped to >= 1 so `0`/negative-ish input
/// never yields a zero-worker pool); a missing/blank/unparseable value falls back to 2.
fn parse_inprocess_utility_worker_count(raw: Option<String>) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(2)
        .max(1)
}

/// Distinct worker id for the `index`-th in-process utility worker. Index 0 keeps the
/// configured `worker_id` unchanged (so a single-worker setup registers exactly as
/// before); each additional worker is suffixed `-1`, `-2`, ... to avoid a registration
/// collision. Mirrors the standalone pool's `utility_worker_id` scheme.
fn inprocess_utility_worker_id(base_worker_id: &str, index: usize) -> String {
    if index == 0 {
        base_worker_id.to_owned()
    } else {
        format!("{base_worker_id}-{index}")
    }
}

/// GPU id for the in-process utility worker. Defaults to `cpu` so the embedded
/// worker advertises CPU utility capabilities (downloads, imports, ffmpeg,
/// person detect/track) regardless of the ambient `SCENEWORKS_GPU_ID` — which on
/// a GPU host would otherwise make it register as a GPU worker that never claims
/// utility jobs. `SCENEWORKS_RUST_WORKER_GPU_ID` overrides for the rare case of
/// wanting the embedded worker on a specific GPU.
fn inprocess_worker_gpu_id(override_var: Option<String>) -> String {
    override_var
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "cpu".to_owned())
}

struct InProcessUtilityWorker {
    handles: Vec<tokio::task::JoinHandle<sceneworks_worker::WorkerResult<()>>>,
    grace: Duration,
}

impl InProcessUtilityWorker {
    async fn shutdown(self) {
        let InProcessUtilityWorker { handles, grace } = self;
        // The loops observe the shared shutdown signal concurrently, so awaiting them
        // in sequence just collects results — each is already stopping (or stopped) by
        // the time we reach it. The per-handle timeout bounds a stuck loop.
        for handle in handles {
            match tokio::time::timeout(grace, handle).await {
                Ok(Ok(Ok(()))) => {}
                Ok(Ok(Err(error))) => tracing::error!(
                    event = "in_process_worker_exited_error",
                    error = %error,
                    "in-process utility worker exited with error"
                ),
                Ok(Err(join_error)) => tracing::error!(
                    event = "in_process_worker_task_failed",
                    error = %join_error,
                    "in-process utility worker task failed"
                ),
                Err(_) => tracing::warn!(
                    event = "in_process_worker_shutdown_timeout",
                    graceSeconds = grace.as_secs(),
                    "in-process utility worker did not stop within the grace period"
                ),
            }
        }
    }
}

/// Poll cadence for the parent-death watchdog (see [`shutdown_signal`]).
const PARENT_POLL_INTERVAL: Duration = Duration::from_secs(3);

/// The parent PID this process should watch, parsed from `SCENEWORKS_PARENT_PID`.
/// `None` when the var is unset/blank/unparseable or `<= 1`: a value of 0 or 1
/// (init/launchd) means "already reparented or no real parent", so the watchdog
/// must not fire. Server/Docker deployments leave the var unset.
fn parent_pid_to_watch() -> Option<i32> {
    let pid: i64 = std::env::var("SCENEWORKS_PARENT_PID")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    (pid > 1 && pid <= i64::from(i32::MAX)).then_some(pid as i32)
}

/// True while `pid` names a live process. `kill(pid, None)` checks for the
/// process without delivering a signal: `Ok` means it's alive; `EPERM` means it
/// exists but we may not signal it (still alive); `ESRCH` is the only "gone"
/// case and yields false.
#[cfg(unix)]
fn pid_alive(pid: i32) -> bool {
    match nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None) {
        Ok(()) => true,
        Err(errno) => errno == nix::errno::Errno::EPERM,
    }
}

/// True while `pid` names a live process. The workspace forbids `unsafe`, so we
/// can't `OpenProcess`/`WaitForSingleObject` directly; instead we shell out to
/// `tasklist` (the same liveness probe the desktop shell uses to reap sidecars).
/// `tasklist /FO CSV` quotes every field, so a live PID appears as `"<pid>"` in a
/// data row while the no-match case prints only an `INFO:` banner — anchoring on
/// the quoted PID is locale-proof and immune to the digits colliding with another
/// column. A probe we can't even launch is treated as "alive" so a transient
/// failure never makes the worker self-terminate spuriously.
#[cfg(windows)]
fn pid_alive(pid: i32) -> bool {
    let Ok(output) = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
    else {
        return true;
    };
    String::from_utf8_lossy(&output.stdout).contains(&format!("\"{pid}\""))
}

/// Resolves once the watched parent process disappears, polling every
/// [`PARENT_POLL_INTERVAL`]. With no parent to watch (`None`) it stays pending
/// forever, so the `select!` branch in [`shutdown_signal`] never fires.
async fn parent_death(parent_pid: Option<i32>) {
    let Some(parent_pid) = parent_pid else {
        std::future::pending::<()>().await;
        return;
    };
    while pid_alive(parent_pid) {
        tokio::time::sleep(PARENT_POLL_INTERVAL).await;
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    // Parent-death watchdog: when launched as a desktop sidecar the Tauri shell
    // sets SCENEWORKS_PARENT_PID to its own PID. A force-quit/crash skips the
    // shell's graceful teardown (`begin_shutdown`), so without this the API
    // orphans (reparented to PID 1 / the Windows session) — holding its
    // OS-assigned port and a jobs.db handle until the next launch reaps it. Unset
    // (server/Docker) -> the future stays pending and this branch never fires.
    let parent_gone = parent_death(parent_pid_to_watch());

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
        _ = parent_gone => {
            tracing::info!(
                event = "api_parent_exited",
                "SceneWorks API: parent process exited; shutting down"
            );
        }
    }
}

/// Stream a multipart field to `temp_path`, enforcing `max_bytes` (returning
/// `413` with `limit_msg` when exceeded), then flush. sc-8886 (F-084): the single
/// implementation behind every multipart upload writer (asset / lora / model), which
/// were three copy-pasted chunk loops differing only in cap source, destination, and
/// message. On ANY error path (chunk read, write, flush, or size cap) the file handle
/// is dropped and `cleanup` runs before the error is returned, so an aborted or
/// malformed multi-gigabyte upload never leaks a temp file (sc-4204). `cleanup` lets a
/// caller remove more than the file itself (e.g. the per-upload parent directory).
/// The parent directory of `temp_path` must already exist.
pub(crate) async fn stream_multipart_field_to_file<Fut>(
    mut field: axum::extract::multipart::Field<'_>,
    temp_path: &FsPath,
    max_bytes: usize,
    limit_msg: impl FnOnce() -> String,
    cleanup: impl FnOnce() -> Fut,
) -> Result<(), ApiError>
where
    Fut: std::future::Future<Output = ()>,
{
    let mut file = match tokio::fs::File::create(temp_path).await {
        Ok(file) => file,
        Err(error) => {
            cleanup().await;
            return Err(ApiError::internal(error.to_string()));
        }
    };
    let mut uploaded_bytes = 0usize;
    let write_result = async {
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            uploaded_bytes = uploaded_bytes.saturating_add(chunk.len());
            if uploaded_bytes > max_bytes {
                return Err(ApiError::payload_too_large(limit_msg()));
            }
            file.write_all(&chunk)
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?;
        }
        file.flush()
            .await
            .map_err(|error| ApiError::internal(error.to_string()))
    }
    .await;
    if let Err(error) = write_result {
        drop(file);
        cleanup().await;
        return Err(error);
    }
    Ok(())
}

/// Remove stale `upload-*` entries under `<data_dir>/cache/<subdir>` older than
/// `cutoff`. sc-8885 (F-083): the single implementation behind every per-area startup
/// sweep (asset, lora, model, pose, keypoint) — previously four/five copy-pasted loops
/// that had already drifted (some skipped non-directories, some didn't). Handles both
/// files and directories so a staging area holding either is fully reclaimed. A missing
/// root is not an error (nothing was ever staged). Returns the number of entries removed.
///
/// Per-entry reclamation is best-effort: a single unremovable stale entry (locked,
/// permission-denied) is logged and skipped so the rest of the sweep still runs — the
/// original per-area sweepers used `let _ =` and continued the loop. Only the outer
/// `read_dir` failure remains fatal (nothing else could have been reclaimed anyway).
pub(crate) fn sweep_stale_uploads(
    data_dir: &FsPath,
    subdir: &str,
    cutoff: SystemTime,
) -> std::io::Result<usize> {
    sweep_stale_uploads_cancellable(data_dir, subdir, cutoff, || true)
}

pub(crate) fn sweep_stale_uploads_cancellable(
    data_dir: &FsPath,
    subdir: &str,
    cutoff: SystemTime,
    should_continue: impl Fn() -> bool,
) -> std::io::Result<usize> {
    let upload_root = data_dir.join("cache").join(subdir);
    let entries = match std::fs::read_dir(upload_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut removed = 0usize;
    for entry in entries {
        if !should_continue() {
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(
                    event = "stale_upload_entry_read_failed",
                    sweep = subdir,
                    error = %error,
                    "could not read a stale-upload dir entry; skipping it"
                );
                continue;
            }
        };
        if !entry.file_name().to_string_lossy().starts_with("upload-") {
            continue;
        }
        if !should_continue() {
            break;
        }
        let is_dir = match entry.file_type() {
            Ok(file_type) => file_type.is_dir(),
            Err(error) => {
                tracing::warn!(
                    event = "stale_upload_stat_failed",
                    sweep = subdir,
                    error = %error,
                    "could not stat a stale-upload entry; skipping it"
                );
                continue;
            }
        };
        if !should_continue() {
            break;
        }
        let modified = match entry.metadata() {
            Ok(metadata) => metadata.modified().unwrap_or(UNIX_EPOCH),
            Err(error) => {
                tracing::warn!(
                    event = "stale_upload_stat_failed",
                    sweep = subdir,
                    error = %error,
                    "could not read a stale-upload entry's mtime; skipping it"
                );
                continue;
            }
        };
        if modified <= cutoff && should_continue() {
            let path = entry.path();
            let removal = if is_dir {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
            match removal {
                Ok(()) => removed += 1,
                Err(error) => {
                    // Best-effort: one locked/permission-denied temp must not block
                    // reclaiming the rest of the stale entries in this sweep.
                    tracing::warn!(
                        event = "stale_upload_remove_failed",
                        sweep = subdir,
                        error = %error,
                        "could not remove a stale upload entry; leaving it and continuing"
                    );
                }
            }
        }
    }
    Ok(removed)
}

/// Log (but never fail on) a startup directory-creation error. sc-8882 (F-080): the
/// old `let _ =` swallowed permissions/disk problems, so they only ever surfaced as
/// downstream 500s. Startup stays best-effort — a missing dir errors where it is used.
fn warn_on_startup_err(label: &str, path: &FsPath, result: std::io::Result<()>) {
    if let Err(error) = result {
        tracing::warn!(
            event = "startup_create_dir_failed",
            dir = label,
            path = %path.display(),
            error = %error,
            "could not create startup directory"
        );
    }
}

/// issue #1435 / sc-11855: `create_dir_all` on an existing-but-non-writable data
/// dir returns `Ok`, so the startup dir checks above pass even when the workspace
/// folder silently rejects the in-place writes a `project.db` needs — the failure
/// only surfaces later as an opaque `SQLITE_READONLY` on the first project
/// creation. Run the SAME faithful rollback-mode probe project creation uses,
/// against the projects tree, and log the resolved path + result so the condition
/// is diagnosable from `api.log` without reproducing it. Purely diagnostic —
/// never fails startup (the app must still boot so the user can reach Settings and
/// repoint the workspace folder).
fn probe_data_dir_writable(data_dir: &FsPath) {
    let projects = data_dir.join("projects");
    match sceneworks_core::project_store::probe_storage_writable(&projects) {
        Ok(()) => tracing::info!(
            event = "startup_data_dir_writable",
            path = %projects.display(),
            "workspace projects folder is writable"
        ),
        Err(error) => tracing::warn!(
            event = "startup_data_dir_not_writable",
            path = %projects.display(),
            error = %error,
            "workspace projects folder is NOT writable — creating a project will fail; \
             the user must pick a different workspace folder or fix this folder's permissions"
        ),
    }
}

/// Log (but never fail on) a stale-upload sweep error. sc-8882 (F-080): a failed sweep
/// silently leaves leaked multi-GB upload temps unreclaimed; a warning makes that
/// diagnosable without aborting startup.
fn warn_on_sweep_err(kind: &str, result: std::io::Result<usize>) {
    if let Err(error) = result {
        tracing::warn!(
            event = "stale_upload_sweep_failed",
            sweep = kind,
            error = %error,
            "stale upload sweep failed; leaked temp uploads may remain"
        );
    }
}

/// Return only bounded, source-owned route labels. Axum's matched path contains
/// route parameter *names* (`:project_id`), never their values. Unknown API and
/// MCP paths deliberately collapse to one label rather than copying a raw URI
/// that could contain IDs, filesystem-shaped path tails, or secret-bearing query
/// values.
fn normalized_api_route(path: &str, matched_path: Option<&str>) -> Option<String> {
    if path == "/mcp" || path.starts_with("/mcp/") {
        return Some(
            matched_path
                .filter(|route| route.starts_with("/mcp"))
                .unwrap_or("/mcp/<unmatched>")
                .to_owned(),
        );
    }
    if path == "/api" || path.starts_with("/api/") {
        return Some(
            matched_path
                .filter(|route| route.starts_with("/api/"))
                .unwrap_or("/api/<unmatched>")
                .to_owned(),
        );
    }
    None
}

/// Worker claim is a one-second idle poll, so logging every successful sub-millisecond response
/// floods the session ring buffer and pushes actionable events out of the Logs screen. Keep the
/// Server-Timing header on every response, but emit the duration event for this route only when the
/// request fails or is unexpectedly slow. All other API/MCP routes retain per-request timing logs.
const SLOW_CLAIM_REQUEST_MS: f64 = 1_000.0;

fn should_log_api_request_duration(route: &str, status: StatusCode, elapsed_ms: f64) -> bool {
    route != "/api/v1/jobs/claim" || !status.is_success() || elapsed_ms >= SLOW_CLAIM_REQUEST_MS
}

async fn api_request_timing(request: AxumRequest, next: Next) -> Response {
    let method = request.method().clone();
    // `Uri::path()` excludes the query by contract. Retain it only long enough
    // to classify API versus frontend traffic; it is never emitted.
    let path = request.uri().path().to_owned();
    let matched_path = request
        .extensions()
        .get::<MatchedPath>()
        .map(|matched| matched.as_str().to_owned());
    let started = Instant::now();
    let mut response = next.run(request).await;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    if let Some(route) = normalized_api_route(&path, matched_path.as_deref()) {
        let server_timing = format!("app;dur={elapsed_ms:.3}");
        if let Ok(value) = HeaderValue::from_str(&server_timing) {
            response
                .headers_mut()
                .insert(HeaderName::from_static("server-timing"), value);
        }
        if should_log_api_request_duration(&route, response.status(), elapsed_ms) {
            tracing::info!(
                event = "api_request_duration",
                method = %method,
                route,
                status = response.status().as_u16(),
                duration_ms = elapsed_ms,
                "SceneWorks API response completed"
            );
        }
    }

    response
}

/// Response-local marker applied only to `/api` requests. The compression
/// predicate reads this extension after the handler returns, so embedded web
/// assets retain SC-14785's precompressed representation/ETag policy instead of
/// being passed through a second, dynamic compressor.
#[derive(Clone, Copy, Debug)]
struct ApiResponseCompressionCandidate;

async fn mark_api_response_for_compression(request: AxumRequest, next: Next) -> Response {
    let is_api = request.uri().path() == "/api" || request.uri().path().starts_with("/api/");
    let mut response = next.run(request).await;
    if is_api {
        response
            .extensions_mut()
            .insert(ApiResponseCompressionCandidate);
    }
    response
}

const API_COMPRESSION_MIN_BYTES: u16 = 1024;

fn is_json_api_response(
    _status: StatusCode,
    _version: axum::http::Version,
    headers: &HeaderMap,
    extensions: &axum::http::Extensions,
) -> bool {
    extensions
        .get::<ApiResponseCompressionCandidate>()
        .is_some()
        && headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value
                    .split(';')
                    .next()
                    .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
            })
}

pub fn create_app(settings: Settings) -> Result<Router, JobsStoreError> {
    Ok(create_app_with_state(settings)?.0)
}

// Like create_app but also returns a clone of the AppState (the same Arc-shared
// stores + event hub the router uses), so tests can subscribe to the event hub and
// assert on what the handlers publish (sc-4203).
pub(crate) fn create_app_with_state(
    settings: Settings,
) -> Result<(Router, AppState), JobsStoreError> {
    create_app_with_state_mode(settings, false)
}

#[cfg(test)]
pub(crate) fn create_app_with_deferred_startup_maintenance(
    settings: Settings,
    timeout: Duration,
) -> Result<(Router, AppState), JobsStoreError> {
    let (router, state) = create_app_with_pending_startup_maintenance(settings)?;
    state
        .startup_maintenance
        .start(state.settings.data_dir.clone(), timeout);
    Ok((router, state))
}

pub(crate) fn create_app_with_pending_startup_maintenance(
    settings: Settings,
) -> Result<(Router, AppState), JobsStoreError> {
    create_app_with_state_mode(settings, true)
}

fn create_app_with_state_mode(
    settings: Settings,
    defer_upload_sweeps: bool,
) -> Result<(Router, AppState), JobsStoreError> {
    let _filesystem_phase = StartupPhaseTimer::start(
        "filesystem_preflight",
        StartupCriticality::ReadinessCritical,
    );
    // sc-8882 (F-080): a permissions/disk failure here is otherwise invisible until a
    // downstream 500 — surface it as a warning so it is diagnosable. Non-fatal: startup
    // continues (a missing dir surfaces later where it is actually used).
    warn_on_startup_err(
        "data_dir",
        &settings.data_dir,
        std::fs::create_dir_all(&settings.data_dir),
    );
    // create_dir_all above is a no-op (and returns Ok) for an existing but
    // non-writable data dir, so probe the projects tree for real (sc-11855 C).
    probe_data_dir_writable(&settings.data_dir);
    warn_on_startup_err(
        "config_dir",
        &settings.config_dir,
        std::fs::create_dir_all(&settings.config_dir),
    );
    if let Some(jobs_db_parent) = settings.jobs_db_path.parent() {
        warn_on_startup_err(
            "jobs_db_parent",
            jobs_db_parent,
            std::fs::create_dir_all(jobs_db_parent),
        );
    }
    if !defer_upload_sweeps {
        let _phase = StartupPhaseTimer::start("upload_sweeps", StartupCriticality::BackgroundSafe);
        // sc-8882 (F-080): a failed sweep leaves leaked multi-GB upload temps unreclaimed
        // and was previously silent. WARN (never fatal) so the operator can investigate.
        warn_on_sweep_err("lora", sweep_stale_lora_uploads(&settings.data_dir));
        warn_on_sweep_err("pose", sweep_stale_pose_uploads(&settings.data_dir));
        warn_on_sweep_err("keypoint", sweep_stale_keypoint_uploads(&settings.data_dir));
        // sc-4204 (F-API-6): asset-import temp files (cache/uploads) had no startup sweep.
        warn_on_sweep_err("asset", sweep_stale_asset_uploads(&settings.data_dir));
    }
    drop(_filesystem_phase);
    let (jobs_store, interrupted_jobs_on_startup) = {
        let _phase = StartupPhaseTimer::start(
            "jobs_retention_recovery",
            StartupCriticality::ReadinessCritical,
        );
        let jobs_store = Arc::new(JobsStore::new(&settings.jobs_db_path));
        jobs_store.initialize()?;
        let purged_terminal_jobs =
            jobs_store.purge_terminal_jobs_older_than(settings.jobs_retention_days)?;
        tracing::info!(
            event = "terminal_job_retention",
            retention_days = settings.jobs_retention_days,
            purged_terminal_jobs,
            "applied terminal job retention"
        );
        let interrupted_jobs_on_startup = jobs_store.mark_interrupted_on_startup()?.len();
        (jobs_store, interrupted_jobs_on_startup)
    };
    let project_store = Arc::new(ProjectStore::new(
        settings.data_dir.clone(),
        settings.app_version.clone(),
    ));
    {
        let _phase = StartupPhaseTimer::start(
            "reserved_project_initialization",
            StartupCriticality::ReadinessCritical,
        );
        // Reserved global pose library (epic 2282): created up front so its assets
        // endpoint returns [] (not 404) before any pose is saved. Best-effort.
        if let Err(error) = project_store.ensure_global_poses_project() {
            tracing::error!(
                event = "ensure_global_poses_project_failed",
                error = %error,
                "could not ensure global pose library project"
            );
        }
        // Reserved global Key Point Library (epic 4422): created up front so its assets +
        // collections endpoints return seeded data before any preset is saved. Best-effort.
        if let Err(error) = project_store.ensure_global_keypoints_project() {
            tracing::error!(
                event = "ensure_global_keypoints_project_failed",
                error = %error,
                "could not ensure global keypoint library project"
            );
        }
    }
    // Startup data-integrity pass: drop index rows for assets whose media was
    // purged from disk but whose row/sidecar lingered, so the Library never fetches
    // a file that 404s on every open (the source of the app-startup 404 log spam).
    // Runs before the server binds, so the first `list_assets` is already clean.
    // Best-effort and non-fatal — a failure just leaves the stale rows for next
    // startup; the sidecars are untouched, so nothing is lost.
    {
        let _phase = StartupPhaseTimer::start(
            "orphaned_asset_maintenance",
            StartupCriticality::ReadinessCritical,
        );
        match project_store.prune_all_orphaned_assets() {
            Ok(0) => {}
            Ok(pruned) => tracing::info!(
                event = "orphaned_assets_pruned",
                count = pruned,
                "pruned purged-but-referenced assets from project registries at startup"
            ),
            Err(error) => tracing::warn!(
                event = "orphaned_assets_prune_failed",
                error = %error,
                "startup orphaned-asset prune failed; the Library may still request purged media"
            ),
        }
    }
    let state = AppState {
        settings,
        jobs_store,
        project_store,
        events: Arc::new(EventHub::default()),
        queue_snapshot_lock: Arc::new(AsyncMutex::new(())),
        event_tickets: Arc::new(TicketStore::with_max_outstanding(
            EVENT_TICKET_TTL_SECONDS,
            MAX_OUTSTANDING_EVENT_TICKETS,
        )),
        media_tickets: Arc::new(TicketStore::new(MEDIA_TICKET_TTL_SECONDS)),
        thumbnail_generation_slots: Arc::new(tokio::sync::Semaphore::new(2)),
        workflow_strip_slots: Arc::new(tokio::sync::Semaphore::new(WORKFLOW_STRIP_SLOTS)),
        auth_throttle: Arc::new(AuthThrottle::default()),
        resolved_cache_session: Arc::new(AsyncMutex::new(None)),
        manifest_cache: Arc::new(Mutex::new(ManifestCache::default())),
        manifest_write_locks: Arc::new(Mutex::new(HashMap::new())),
        model_catalog_cache: Arc::new(ModelCatalogCache::default()),
        model_size_cache: Arc::new(Mutex::new(ModelSizeCache::default())),
        #[cfg(test)]
        model_size_estimate_test_hook: Arc::new(Mutex::new(None)),
        #[cfg(test)]
        model_size_estimate_disabled_override: Arc::new(Mutex::new(None)),
        #[cfg(test)]
        video_platform_override: Arc::new(Mutex::new(None)),
        external_lora_cache: Arc::new(Mutex::new(external_loras::ExternalLoraCache::default())),
        external_base_model_cache: Arc::new(Mutex::new(
            external_base_models::ExternalBaseModelCache::default(),
        )),
        http_client: reqwest::Client::new(),
        interrupted_jobs_on_startup,
        startup_maintenance: if defer_upload_sweeps {
            StartupMaintenance::pending()
        } else {
            StartupMaintenance::complete()
        },
        progress_side_effects_lock: Arc::new(AsyncMutex::new(())),
        catalog_scan_supervisor: Arc::new(catalog_scan_supervisor::CatalogScanSupervisor::default()),
        catalog_scan_invalid_recovery_reported: Arc::new(AsyncMutex::new(
            std::collections::HashSet::new(),
        )),
        catalog_scan_preflight_slots: Arc::new(tokio::sync::Semaphore::new(2)),
        catalog_scan_work_slots: Arc::new(tokio::sync::Semaphore::new(2)),
        #[cfg(test)]
        catalog_scan_before_driver_start_once: Arc::new(Mutex::new(None)),
        #[cfg(test)]
        catalog_scan_stop_after_pass_once: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        #[cfg(test)]
        catalog_scan_before_terminal_exit_once: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        #[cfg(test)]
        catalog_scan_terminal_exit_reached: Arc::new(tokio::sync::Notify::new()),
        #[cfg(test)]
        catalog_scan_terminal_exit_release: Arc::new(tokio::sync::Notify::new()),
        #[cfg(test)]
        catalog_scan_preflight_delay_ms_once: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        #[cfg(test)]
        catalog_scan_preflight_started: Arc::new(tokio::sync::Notify::new()),
        #[cfg(test)]
        catalog_scan_preflight_admission_timeout_ms: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        #[cfg(test)]
        catalog_scan_preflight_execution_timeout_ms: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        #[cfg(test)]
        catalog_scan_preflight_test_ticks: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        #[cfg(test)]
        catalog_scan_injected_sqlite_busy_failures: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        #[cfg(test)]
        catalog_scan_contention_backoff_started: Arc::new(tokio::sync::Notify::new()),
        #[cfg(test)]
        progress_before_accept_once: Arc::new(Mutex::new(None)),
        #[cfg(test)]
        sse_snapshot_before_subscribe_once: Arc::new(Mutex::new(None)),
        #[cfg(test)]
        progress_side_effects_fail_once: Arc::new(Mutex::new(false)),
        #[cfg(test)]
        progress_side_effects_fail_job_ids: Arc::new(Mutex::new(std::collections::HashSet::new())),
        #[cfg(test)]
        progress_side_effects_attempts: Arc::new(Mutex::new(HashMap::new())),
    };
    let cors = cors_layer(&state.settings);
    let returned_state = state.clone();

    // MCP server (epic 10231, sc-10233): the rmcp streamable-HTTP service is
    // nested at `/mcp` INSIDE this router, so the `access_control` layer below
    // gates it exactly like every `/api/v1` route (`requires_token` includes
    // `/mcp`) — token header, loopback trust, and the brute-force throttle all
    // apply unchanged. Its tools call back into this API over plain HTTP
    // (`settings.mcp_api_url`, i.e. `SCENEWORKS_API_URL` or our own loopback
    // port) carrying the access token, so there is no second engine/DB path.
    // Blocking-job wait policy comes from Settings (sc-10277: SCENEWORKS_MCP_JOB_*
    // env knobs), clamped to the invariants the poll loop needs.
    // F-040 (sc-11236): restore the transport's Host-header (DNS-rebinding)
    // defense. `/mcp` rides `access_control`, but that gate performs NO
    // Host/Origin validation, so in the loopback/loopback-trust/no-token desktop
    // posture a malicious page could DNS-rebind a browser onto `/mcp`. Derive the
    // allowed Host set from the SAME bind config that decides where the API
    // listens: loopback is always allowed; a concrete interface host adds itself;
    // a wildcard LAN bind honors `SCENEWORKS_MCP_ALLOWED_HOSTS` (and otherwise
    // disables the check, relying on the mandatory LAN access token).
    let mcp_allowed_hosts = sceneworks_mcp::mcp_allowed_hosts(
        &state.settings.host,
        state.settings.port,
        &state.settings.mcp_allowed_hosts_extra,
    );
    let mcp_service = sceneworks_mcp::streamable_http_service_with_hosts(
        sceneworks_mcp::ApiClientConfig {
            base_url: state.settings.mcp_api_url.clone(),
            access_token: Some(state.settings.access_token.clone()),
        },
        sceneworks_mcp::JobWaitConfig::clamped(
            state.settings.mcp_job_poll_interval,
            state.settings.mcp_job_timeout,
        ),
        mcp_allowed_hosts,
    );

    let router = Router::new()
        .nest_service("/mcp", mcp_service)
        .route("/api/v1/health", get(health))
        .route("/api/v1/access", get(access))
        .route("/api/v1/auth/verify", post(verify_access))
        .route("/api/v1/training/targets", get(list_training_targets))
        .route("/api/v1/training/presets", get(list_training_presets))
        .route("/api/v1/catalogs", get(list_catalogs).post(create_catalog))
        .route("/api/v1/catalogs/attach", post(attach_catalog))
        .route(
            "/api/v1/catalogs/:catalog_id",
            get(get_catalog).delete(detach_catalog),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/status",
            get(get_catalog_status),
        )
        .route("/api/v1/catalogs/:catalog_id/query", post(query_catalog))
        .route(
            "/api/v1/catalogs/:catalog_id/curation/query",
            post(curate_catalog),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/curation/facets",
            post(catalog_curation_facets),
        )
        .route("/api/v1/catalogs/:catalog_id/facets", post(catalog_facets))
        .route(
            "/api/v1/catalogs/:catalog_id/saved-views",
            get(list_catalog_saved_views).post(create_catalog_saved_view),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/saved-views/:view_id",
            put(update_catalog_saved_view).delete(delete_catalog_saved_view),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/records/:record_id/review",
            put(review_catalog_record),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/records/:record_id/thumbnail",
            get(catalog_record_thumbnail),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/analyzer-config",
            put(update_catalog_analyzer_config),
        )
        .route("/api/v1/catalogs/:catalog_id/pause", post(pause_catalog))
        .route("/api/v1/catalogs/:catalog_id/resume", post(resume_catalog))
        .route(
            "/api/v1/catalogs/:catalog_id/analyze",
            post(run_catalog_analysis),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/on-disk",
            delete(delete_catalog_on_disk),
        )
        .route("/api/v1/projects", get(list_projects).post(create_project))
        .route("/api/v1/projects/:project_id", get(get_project))
        .route(
            "/api/v1/projects/:project_id/reindex",
            post(reindex_project_endpoint),
        )
        .route(
            "/api/v1/projects/:project_id/assets",
            get(list_assets)
                .post(import_asset)
                // sc-8812 (F-010): streaming multipart asset upload needs the large
                // limit; re-attach it per-route since the router default is now the
                // small JSON cap. GET has no body, so this is harmless for listing.
                .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        // sc-15950: read a shared image's embedded workflow WITHOUT creating an asset. Registered
        // beside the asset routes because it shares their multipart shape and staging area, but it
        // is deliberately not project-scoped — it mutates nothing, so there is nothing to scope.
        // Same auth posture as the upload route above; a larger-than-JSON per-route limit is
        // re-attached for the same reason (the router default is the small JSON cap), but it is
        // this route's OWN cap plus framing headroom rather than the 2 GiB asset ceiling — see
        // `MAX_WORKFLOW_INSPECT_BYTES` for why both numbers are what they are.
        .route(
            "/api/v1/workflows/inspect",
            post(inspect_workflow).layer(DefaultBodyLimit::max(max_inspect_multipart_body_bytes())),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id",
            get(get_asset).delete(delete_asset),
        )
        // sc-15952: the resolution report for the envelope an IMPORTED asset is already carrying
        // at `extra.importedWorkflow`. The envelope needs no route — it rides on the asset — but
        // the report is computed against catalogs that live behind this API, and it goes stale the
        // moment a download finishes, so re-resolving after an install is a plain refetch of this.
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/workflow",
            get(get_asset_workflow),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/poster/:poster_sha256",
            get(get_asset_poster),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/purge",
            delete(purge_asset),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/move-to-library",
            post(move_asset_to_library),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/move-to-character",
            post(move_asset_to_character),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/status",
            patch(update_asset_status),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/tags",
            patch(update_asset_tags),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets",
            get(list_training_datasets).post(create_training_dataset),
        )
        .route(
            "/api/v1/catalogs/:catalog_id/materialize",
            post(materialize_catalog_results),
        )
        .route(
            "/api/v1/projects/:project_id/training/uploads",
            // sc-8812 (F-010): streaming multipart training-dataset upload; needs the
            // large per-route limit against the small JSON router default.
            post(upload_training_dataset_item).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id",
            get(get_training_dataset)
                .patch(update_training_dataset)
                .delete(delete_training_dataset),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/readiness",
            get(get_training_dataset_readiness),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/items/:item_id/quality-ack",
            post(set_training_dataset_item_quality_ack),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/batch-rename",
            post(batch_rename_training_dataset_items),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/caption-sidecars",
            post(write_training_dataset_caption_sidecars),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/caption-jobs",
            post(create_training_dataset_caption_job),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/parquet-import-jobs",
            post(create_training_dataset_parquet_import_job),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/parquet-import-finalize",
            post(finalize_training_dataset_parquet_import)
                .layer(DefaultBodyLimit::max(MAX_PARQUET_FINALIZE_BODY_BYTES)),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/analysis-jobs",
            post(create_training_dataset_analysis_job),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/face-analysis-jobs",
            post(create_training_dataset_face_analysis_job),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/analysis-embeddings",
            post(write_training_dataset_analysis_embeddings),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/face-embeddings",
            post(write_training_dataset_face_embeddings),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/upscale-jobs",
            post(create_training_dataset_upscale_job),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/repoint",
            post(repoint_training_dataset_items),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/smart-crop",
            post(smart_crop_training_dataset_items),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/strip-exif",
            post(strip_exif_training_dataset_items),
        )
        .route(
            "/api/v1/projects/:project_id/training/jobs",
            post(create_training_job),
        )
        .route(
            "/api/v1/projects/:project_id/files/*relative_path",
            get(get_project_file),
        )
        .route(
            "/api/v1/projects/:project_id/voices",
            get(list_saved_voices).post(create_saved_voice),
        )
        .route(
            "/api/v1/projects/:project_id/voices/:voice_id",
            delete(delete_saved_voice),
        )
        .route(
            "/api/v1/projects/:project_id/characters",
            get(list_characters).post(create_character),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id",
            get(get_character)
                .patch(update_character)
                .delete(archive_character),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/archive",
            post(archive_character),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/purge",
            delete(purge_character),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/references",
            post(add_character_reference),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/references/:asset_id",
            patch(update_character_reference).delete(remove_character_reference),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/looks",
            post(create_character_look),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/looks/:look_id",
            patch(update_character_look).delete(delete_character_look),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/loras",
            post(attach_character_lora),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/loras/:link_id",
            patch(update_character_lora).delete(detach_character_lora),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/test-jobs",
            post(create_character_test_job),
        )
        .route(
            "/api/v1/projects/:project_id/timelines",
            get(list_timelines).post(create_timeline),
        )
        .route(
            "/api/v1/projects/:project_id/timelines/:timeline_id",
            get(get_timeline).put(update_timeline),
        )
        .route(
            "/api/v1/projects/:project_id/timelines/:timeline_id/exports",
            post(create_timeline_export),
        )
        .route(
            "/api/v1/projects/:project_id/timelines/:timeline_id/items/:item_id/frames",
            post(extract_timeline_frame),
        )
        .route(
            "/api/v1/projects/:project_id/person-tracks",
            get(list_person_tracks),
        )
        .route(
            "/api/v1/projects/:project_id/person-tracks/detections",
            post(create_person_detection_job),
        )
        .route(
            "/api/v1/projects/:project_id/person-tracks/jobs",
            post(create_person_track_job),
        )
        .route(
            "/api/v1/projects/:project_id/person-tracks/:track_id",
            get(get_person_track),
        )
        .route(
            "/api/v1/projects/:project_id/person-tracks/:track_id/corrections",
            post(save_person_track_corrections),
        )
        .route("/api/v1/image/jobs", post(create_image_job))
        .route("/api/v1/image/vqa/jobs", post(create_vqa_job))
        .route("/api/v1/image/interleave/jobs", post(create_interleave_job))
        .route("/api/v1/video/jobs", post(create_video_job))
        .route("/api/v1/audio/jobs", post(create_audio_job))
        .route("/api/v1/prompts/refine", post(create_prompt_refine_job))
        .route(
            "/api/v1/face-likeness/compare",
            post(create_face_likeness_compare_job),
        )
        .route("/api/v1/poses", post(create_poses))
        .route(
            "/api/v1/poses/sources",
            // sc-8812 (F-010): multipart pose-source image upload; large per-route limit.
            post(create_pose_sources).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        .route(
            "/api/v1/poses/preview/:job_id/:file_name",
            get(get_pose_preview),
        )
        .route("/api/v1/keypoints", post(create_keypoint))
        .route(
            "/api/v1/keypoints/sources",
            // sc-8812 (F-010): multipart keypoint-source image upload; large per-route limit.
            post(create_keypoint_sources).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        .route("/api/v1/keypoints/presets", get(list_keypoint_presets))
        .route(
            "/api/v1/keypoints/collections",
            get(list_keypoint_collections).post(upsert_keypoint_collection),
        )
        .route(
            "/api/v1/keypoints/collections/:collection_id",
            delete(delete_keypoint_collection),
        )
        .route(
            "/api/v1/keypoints/collections/:collection_id/default",
            put(set_default_keypoint_collection),
        )
        .route(
            "/api/v1/credentials",
            get(list_credentials).put(set_credential),
        )
        .route("/api/v1/credentials/:host", delete(delete_credential))
        .route(
            UI_PREFERENCES_PATH,
            get(get_ui_preferences).put(set_ui_preferences),
        )
        // The model source library's live status + relocation seam (sc-19709). Its own path
        // prefix, not `/models/...`, so a library operation can never collide with a model id.
        .route(
            "/api/v1/model-library",
            get(model_library::get_model_library),
        )
        .route(
            "/api/v1/model-library/relocate",
            post(model_library::relocate_model_library),
        )
        .route("/api/v1/models", get(list_models))
        // Linked checkpoint libraries (epic 20398, sc-20635). A STATIC sibling of
        // `/api/v1/models/:model_id`, exactly like `/api/v1/models/import`, so a library
        // operation is never mistaken for a model id.
        .route(
            "/api/v1/models/library-roots",
            get(checkpoint_library::list_library_roots)
                .post(checkpoint_library::approve_library_root),
        )
        .route(
            "/api/v1/models/library-roots/:root_id",
            patch(checkpoint_library::update_library_root)
                .delete(checkpoint_library::remove_library_root),
        )
        .route(
            "/api/v1/models/library-roots/:root_id/scan",
            get(checkpoint_library::scan_library_root),
        )
        .route(
            "/api/v1/models/library-roots/:root_id/rescan",
            post(checkpoint_library::rescan_library_checkpoint),
        )
        .route("/api/v1/models/:model_id", delete(delete_model))
        .route(
            "/api/v1/models/:model_id/variants/:variant",
            delete(delete_model_variant),
        )
        .route(
            "/api/v1/models/:model_id/download",
            post(create_model_download_job),
        )
        .route(
            "/api/v1/models/:model_id/convert",
            post(create_model_convert_job),
        )
        .route(
            "/api/v1/models/import",
            post(create_model_import_job)
                .layer(DefaultBodyLimit::max(MAX_MODEL_MULTIPART_BODY_BYTES)),
        )
        // Resolved-model hot cache (sc-19711). The GET is UI-polled, so it is deliberately one
        // cheap write-free listing; the three POSTs are single deliberate user actions.
        .route("/api/v1/model-cache", get(get_model_cache))
        .route(
            "/api/v1/model-cache/removal-preview",
            post(preview_model_cache_removal),
        )
        .route("/api/v1/model-cache/remove", post(remove_model_cache_entry))
        .route("/api/v1/model-cache/pin", post(set_model_cache_pin))
        .route("/api/v1/control-overlays", get(list_control_overlays))
        .route("/api/v1/styles", get(list_styles))
        .route("/api/v1/loras", get(list_loras))
        .route(
            "/api/v1/loras/:lora_id",
            delete(delete_lora).patch(update_lora),
        )
        .route(
            "/api/v1/loras/:lora_id/embedded-tags",
            get(lora_embedded_tags),
        )
        .route(
            "/api/v1/loras/:lora_id/download",
            post(create_lora_download_job),
        )
        .route(
            "/api/v1/loras/import",
            post(create_lora_import_job)
                .layer(DefaultBodyLimit::max(MAX_LORA_MULTIPART_BODY_BYTES)),
        )
        .route(
            "/api/v1/recipe-presets",
            get(list_recipe_presets).post(create_recipe_preset),
        )
        .route(
            "/api/v1/recipe-presets/:preset_id",
            get(get_recipe_preset)
                .patch(update_recipe_preset)
                .delete(delete_recipe_preset),
        )
        .route(
            "/api/v1/recipe-presets/:preset_id/duplicate",
            post(duplicate_recipe_preset),
        )
        .route(
            "/api/v1/prompt-batches",
            get(list_prompt_batches).post(create_prompt_batch),
        )
        .route(
            "/api/v1/prompt-batches/:batch_id",
            get(get_prompt_batch)
                .patch(update_prompt_batch)
                .delete(delete_prompt_batch),
        )
        .route(
            "/api/v1/prompt-batches/:batch_id/duplicate",
            post(duplicate_prompt_batch),
        )
        .route("/api/v1/jobs", get(list_jobs).post(create_job))
        .route("/api/v1/jobs/claim", post(claim_job))
        // Clear completed items from the queue (sc-12231, issue #1556). A static
        // segment like `/claim`, so it takes priority over `/jobs/:job_id`.
        .route("/api/v1/jobs/clear", post(clear_jobs))
        // Cancel every pending (queued / pending_caption) item at once (sc-13448).
        // Static segment like `/clear`, so it takes priority over `/jobs/:job_id`.
        .route("/api/v1/jobs/cancel-pending", post(cancel_pending_jobs))
        // Move selected not-yet-started jobs to the front. This is a static segment and must be
        // registered before the per-job routes below.
        .route("/api/v1/jobs/prioritize", post(prioritize_jobs))
        .route("/api/v1/jobs/events", get(job_events))
        .route("/api/v1/jobs/events/ticket", post(create_event_ticket))
        // Media ticket (sc-8810): auth-protected mint endpoint; the ticket is honored
        // as a query param by the project-files and pose-preview GETs (see auth.rs).
        .route("/api/v1/files/ticket", post(create_media_ticket))
        .route("/api/v1/jobs/:job_id", get(get_job))
        .route("/api/v1/jobs/:job_id/cancel", post(cancel_job))
        // Per-job "clear" (sc-12231, issue #1556) — the per-card × dismiss. Distinct
        // from the bulk `/api/v1/jobs/clear` above (2 segments vs 3, no conflict).
        .route("/api/v1/jobs/:job_id/clear", post(clear_job))
        .route("/api/v1/jobs/:job_id/retry", post(retry_job))
        .route("/api/v1/jobs/:job_id/duplicate", post(duplicate_job))
        .route("/api/v1/jobs/:job_id/progress", post(update_job_progress))
        // Per-run generation metrics (epic 10402): worker POSTs on completion;
        // GET returns a single job's block; the aggregate feed powers the
        // Generation Stats comparison charts.
        .route(
            "/api/v1/jobs/:job_id/metrics",
            get(get_job_metrics).post(upsert_job_metrics),
        )
        .route("/api/v1/metrics", get(list_metrics))
        .route("/api/v1/queue", get(queue_summary))
        .route("/api/v1/logs", get(list_logs))
        .route("/api/v1/workers", get(list_workers))
        .route(
            "/api/v1/capabilities/person",
            get(person_capability_readiness),
        )
        .route("/api/v1/capabilities/mac", get(mac_capability_support))
        // Host memory for remote-browser model gating (epic 4484 story 9).
        .route("/api/v1/host-capabilities", get(host_capabilities))
        // Remote-admin GPU worker restart (epic 4484 story 12).
        .route("/api/v1/worker/restart", post(request_worker_restart))
        .route("/api/v1/workers/register", post(register_worker))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_worker),
        )
        .route(
            "/api/v1/workers/:worker_id/terminated",
            post(worker_terminated),
        )
        .fallback(app_fallback)
        .with_state(state.clone())
        // sc-8812 (F-010): small router-wide default so JSON routes can't buffer
        // multi-GiB bodies. Multipart/upload routes re-attach the large limit per
        // route (asset import, training uploads, pose/keypoint sources, and the
        // model/lora import routes above).
        .layer(DefaultBodyLimit::max(MAX_JSON_BODY_BYTES))
        .layer(middleware::from_fn_with_state(state, access_control))
        .layer(cors)
        // Marking is inside CompressionLayer on the response path. This keeps
        // dynamic compression API-only even when `embed-web` serves a JSON
        // file through the same router.
        .layer(middleware::from_fn(mark_api_response_for_compression))
        .layer(
            CompressionLayer::new()
                // Only the broadly supported encodings required by SC-14798.
                // Dynamic Brotli defaults to level 4 and gzip to its library
                // default; see docs/api-response-compression.md.
                .br(true)
                .gzip(true)
                .no_deflate()
                .no_zstd()
                .quality(CompressionLevel::Default)
                .compress_when(SizeAbove::new(API_COMPRESSION_MIN_BYTES).and(is_json_api_response)),
        )
        // Outermost so auth failures, CORS responses, and handler responses are
        // all timed. Non-API frontend traffic is ignored by the middleware.
        .layer(middleware::from_fn(api_request_timing));
    Ok((router, returned_state))
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let token_configured = !state.settings.access_token.is_empty();
    Json(HealthResponse {
        status: "ok",
        service: "sceneworks-api",
        runtime: "rust".to_owned(),
        version: state.settings.app_version.clone(),
        auth_required: token_configured,
        // When a token is configured the endpoint is public but the deployment expects
        // auth, so don't leak absolute host paths to unauthenticated LAN callers.
        directories: if token_configured {
            None
        } else {
            Some(DirectoriesResponse {
                data: state.settings.data_dir.display().to_string(),
                config: state.settings.config_dir.display().to_string(),
                projects: state.settings.projects_dir().display().to_string(),
                jobs_db: state.settings.jobs_db_path.display().to_string(),
            })
        },
        interrupted_jobs_on_startup: state.interrupted_jobs_on_startup,
        readiness: StartupReadinessResponse {
            status: "ready",
            criticality: "readiness-critical",
        },
        startup_maintenance: state.startup_maintenance.snapshot(),
    })
}

async fn access(State(state): State<AppState>) -> Json<AccessResponse> {
    Json(AccessResponse {
        auth_required: !state.settings.access_token.is_empty(),
        token_header: "X-SceneWorks-Token",
    })
}

async fn verify_access(
    State(state): State<AppState>,
    // `Option<…>` mirrors the auth middleware: unit-test oneshot requests have no
    // connect info, so the peer is absent and the throttle is a no-op for them.
    connect_info: Option<axum::extract::ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Json<VerifyResponse> {
    // sc-8870 (F-068): this endpoint is public and answers `{ok}` for any candidate
    // token, so it is the cheapest brute-force oracle. The access-control middleware
    // already refuses a peer that is over its failure budget (its entry check runs on
    // every request, public ones included), so a throttled caller never reaches here;
    // this handler only has to feed the counter — a wrong token is a failed attempt,
    // a valid one clears the peer's record. Loopback-trusted peers still get counted
    // here on a bad guess, but the desktop UI only ever sends the real token (or none,
    // when auth is off), so in practice only a remote guesser accrues failures.
    let peer_ip = connect_info.map(|axum::extract::ConnectInfo(addr)| addr.ip());
    let ok = is_authorized(&headers, &state.settings);
    // Only meter when a token is actually configured; with auth off every check is
    // trivially `ok` and there is nothing to brute-force.
    if !state.settings.access_token.is_empty() {
        if ok {
            state.auth_throttle.record_success(peer_ip);
        } else {
            let failures = state.auth_throttle.record_failure(peer_ip);
            tracing::warn!(
                event = "auth_verify_failed",
                failures,
                "rejected token via /auth/verify oracle"
            );
        }
    }
    Json(VerifyResponse { ok })
}

const GRID_THUMBNAIL_SIZE: u32 = 384;
const PROJECT_MEDIA_CACHE_CONTROL: &str = "private, max-age=31536000, immutable";

/// How many `?stripWorkflow=true` downloads may be in flight at once (sc-15953).
///
/// Two, mirroring `thumbnail_generation_slots` — the other derived representation of this same
/// route, and the precedent for gating one. A strip is not CPU-bound the way a thumbnail is; what
/// it holds is MEMORY, for as long as the response body is being written to a client that may be
/// on a slow link. Ungated, the ceiling was the product of the file size and the number of
/// simultaneous callers, with nothing in the path to say no.
const WORKFLOW_STRIP_SLOTS: usize = 2;

/// Largest PNG this route will rewrite in memory to remove a workflow chunk. 128 MiB.
///
/// The walk needs the tail of the file, so the whole thing is read; the cost therefore follows the
/// asset. A generated image is nowhere near this — measured through the repo's own writer on an
/// INCOMPRESSIBLE render, the worst case for a PNG, 1024² is 3.0 MiB, 2048² is 12.0 MiB and 4096²
/// is 48.0 MiB, and 4096² is the largest thing SceneWorks renders. An IMPORTED asset is bounded
/// only by `MAX_UPLOAD_BYTES` = 2 GiB, which is what this exists to refuse. 128 MiB is ~2.7× the
/// largest file the writer produces, so nothing SceneWorks made is ever turned away, and with
/// [`WORKFLOW_STRIP_SLOTS`] it bounds the route at 256 MiB rather than at multiple gigabytes.
///
/// Refused rather than silently served whole: "here is your copy without the workflow" answered
/// with the original file is the one outcome this feature must never produce, so an image too
/// large to rewrite gets a 413 that says why.
const MAX_WORKFLOW_STRIP_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProjectFileQuery {
    thumbnail: Option<u32>,
    /// Serve the file with any embedded `sceneworks:workflow` chunk removed (sc-15953).
    ///
    /// A query param on the existing route rather than a new endpoint, because the browser
    /// "Save As without the workflow" is a bare `<a download>` and cannot transform bytes or set
    /// headers. That constraint also decides the shape: in remote/LAN mode the anchor authenticates
    /// with a `?ticket=` media ticket, whose allow-list (`auth::is_ticketed_media_path`) matches on
    /// the PATH and is GET-only — so keeping the strip on this path means the download works
    /// remotely with no change to the ticket rules at all. A separate endpoint would have needed a
    /// new entry in that allow-list, which is the surface hardest to widen safely.
    strip_workflow: Option<bool>,
}

async fn get_project_file(
    State(state): State<AppState>,
    Path((project_id, relative_path)): Path<(String, String)>,
    Query(query): Query<ProjectFileQuery>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let project_file = project_call(state.clone(), {
        let project_id = project_id.clone();
        let relative_path = relative_path.clone();
        move |store| store.project_file(&project_id, &relative_path)
    })
    .await
    .inspect_err(|error| {
        // The generic 4xx logger (error.rs) records only `status` + `detail`, so a
        // bare "File not found" line can't be traced back to a file. Name the missing
        // resource here — mirroring the `auth_rejected`/`auth_throttled` structured
        // logs — so operators can see which asset the web UI requested. The common
        // startup culprits are a video's `<name>.poster.jpg` that was never generated
        // and an asset purged from disk but still referenced by the project; the web
        // UI degrades both to a placeholder (assetMedia.jsx), so the only trace is here.
        if error.status == StatusCode::NOT_FOUND {
            tracing::debug!(
                event = "project_file_missing",
                project_id = %project_id,
                relative_path = %relative_path,
                status = error.status.as_u16(),
                "requested project file not found"
            );
        }
    })?;

    // "Download without the workflow" (sc-15953). Answered here, before the thumbnail and
    // streaming branches, because it is the one variant whose body is not the file on disk.
    if query.strip_workflow.unwrap_or(false) {
        if query.thumbnail.is_some() {
            return Err(ApiError::bad_request(
                "stripWorkflow cannot be combined with thumbnail",
            ));
        }
        return stripped_project_file_response(&state, &project_file, &headers).await;
    }

    // One fixed representation prevents unbounded cache variants. Derivatives
    // are generated on first use, so assets written before this route existed
    // backfill without a migration.
    let (served_path, content_type) = if let Some(size) = query.thumbnail {
        if size != GRID_THUMBNAIL_SIZE {
            return Err(ApiError::bad_request(format!(
                "thumbnail must be {GRID_THUMBNAIL_SIZE}"
            )));
        }
        if !project_file.content_type.starts_with("image/") {
            return Err(ApiError::bad_request(
                "thumbnail is only available for image media",
            ));
        }
        let permit = state
            .thumbnail_generation_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let source_path = project_file.path.clone();
        let cache_root = state
            .settings
            .data_dir
            .join("cache")
            .join("media-thumbnails")
            .join("v1");
        let thumbnail_path = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            ensure_grid_thumbnail(&source_path, &cache_root)
        })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(|error| {
            tracing::debug!(
                event = "project_thumbnail_unavailable",
                project_id = %project_id,
                relative_path = %relative_path,
                error = %error,
                "could not create bounded project thumbnail"
            );
            ApiError::bad_request("Thumbnail unavailable for this media")
        })?;
        (thumbnail_path, "image/png".to_owned())
    } else {
        (project_file.path, project_file.content_type)
    };

    let mut file = tokio::fs::File::open(&served_path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let metadata = file
        .metadata()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let total = metadata.len();
    let etag = project_file_etag(&metadata);
    let last_modified = metadata.modified().ok();
    let base_headers = project_file_response_headers(&content_type, total, &etag, last_modified)?;

    if project_file_is_not_modified(&headers, &etag, last_modified) {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = StatusCode::NOT_MODIFIED;
        *response.headers_mut() = base_headers;
        response.headers_mut().remove(header::CONTENT_LENGTH);
        return Ok(response);
    }

    // WebKit/WKWebView (the macOS desktop webview) requires HTTP byte-range
    // responses to play <video>: it probes with `Range: bytes=0-1` and treats
    // any 200 reply as a non-seekable source it won't play. Honor a single
    // range with 206 Partial Content; advertise Accept-Ranges otherwise.
    if let Some(range_header) = headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
        match parse_single_byte_range(range_header, total) {
            Some((start, end)) => {
                let len = end - start + 1;
                file.seek(SeekFrom::Start(start))
                    .await
                    .map_err(|error| ApiError::internal(error.to_string()))?;
                let stream = ReaderStream::new(file.take(len));
                let mut response = (
                    StatusCode::PARTIAL_CONTENT,
                    [
                        (header::CONTENT_TYPE, content_type.clone()),
                        (header::ACCEPT_RANGES, "bytes".to_string()),
                        (
                            header::CONTENT_RANGE,
                            format!("bytes {start}-{end}/{total}"),
                        ),
                        (header::CONTENT_LENGTH, len.to_string()),
                        // sc-9674 (sc-8872 follow-up): forbid MIME sniffing so a
                        // user-controlled project file can't be reinterpreted by the
                        // browser as a different (e.g. active) content type than the
                        // Content-Type we derived. Kept inline (no attachment
                        // disposition) so <img>/<video> preview and byte-range
                        // playback still work — the assets are served for inline
                        // display, not forced download.
                        (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()),
                    ],
                    Body::from_stream(stream),
                )
                    .into_response();
                response.headers_mut().extend(base_headers);
                response.headers_mut().insert(
                    header::CONTENT_LENGTH,
                    HeaderValue::from_str(&len.to_string())
                        .map_err(|error| ApiError::internal(error.to_string()))?,
                );
                return Ok(response);
            }
            None => {
                let mut response = (
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    [(header::CONTENT_RANGE, format!("bytes */{total}"))],
                )
                    .into_response();
                response.headers_mut().extend(base_headers);
                response.headers_mut().remove(header::CONTENT_LENGTH);
                return Ok(response);
            }
        }
    }

    let stream = ReaderStream::new(file);
    let mut response = (
        [
            (header::CONTENT_TYPE, content_type),
            (header::ACCEPT_RANGES, "bytes".to_string()),
            (header::CONTENT_LENGTH, total.to_string()),
            // sc-9674: forbid MIME sniffing (see the range branch above). Inline
            // disposition is kept intentionally so image/video preview still works.
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()),
        ],
        Body::from_stream(stream),
    )
        .into_response();
    response.headers_mut().extend(base_headers);
    Ok(response)
}

fn ensure_grid_thumbnail(source_path: &FsPath, cache_root: &FsPath) -> Result<PathBuf, String> {
    let metadata = std::fs::metadata(source_path).map_err(|error| error.to_string())?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    let mut hash = Sha256::new();
    hash.update(source_path.to_string_lossy().as_bytes());
    hash.update(metadata.len().to_le_bytes());
    hash.update(modified.as_nanos().to_le_bytes());
    hash.update(GRID_THUMBNAIL_SIZE.to_le_bytes());
    let key = format!("{:x}", hash.finalize());
    let target = cache_root.join(format!("{key}.png"));
    if target.is_file() {
        return Ok(target);
    }

    std::fs::create_dir_all(cache_root).map_err(|error| error.to_string())?;
    let decoded = match image::open(source_path) {
        Ok(decoded) => decoded,
        Err(decode_error) => {
            // Project imports normalize these formats today, but old projects
            // can still contain AVIF/HEIC/HEIF (and other platform-decodable
            // raster files). Reuse the worker's cross-platform compatibility
            // path before declaring the thumbnail unavailable.
            let converted =
                cache_root.join(format!("{key}.{}.converted.png", Uuid::new_v4().simple()));
            let transcode_result =
                sceneworks_core::media_convert::transcode_to_png(source_path, &converted);
            let decoded = transcode_result
                .map_err(|error| format!("{decode_error}; {error}"))
                .and_then(|()| image::open(&converted).map_err(|error| error.to_string()));
            let _ = std::fs::remove_file(&converted);
            decoded?
        }
    };
    let thumbnail = decoded.thumbnail(GRID_THUMBNAIL_SIZE, GRID_THUMBNAIL_SIZE);
    let temporary = cache_root.join(format!("{key}.{}.tmp", Uuid::new_v4().simple()));
    thumbnail
        .save_with_format(&temporary, image::ImageFormat::Png)
        .map_err(|error| error.to_string())?;
    if let Err(error) = std::fs::rename(&temporary, &target) {
        if target.is_file() {
            let _ = std::fs::remove_file(&temporary);
        } else {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.to_string());
        }
    }
    Ok(target)
}

/// Serve a project file with any embedded workflow chunk excised (sc-15953).
///
/// The stripping itself is `sceneworks_core::workflow_png::strip_workflow_chunk` — a byte-level
/// walk of the PNG chunk framing that copies IHDR, every IDAT and IEND through unchanged, so the
/// served body is the file minus one slice rather than a re-encode. Nothing here decodes a pixel.
///
/// Five things this branch has to get right that the streaming path does not:
///
/// * **Only a PNG is read into memory.** The workflow chunk only ever exists in a PNG, so anything
///   else is served by the ordinary path — a video does not get buffered because someone appended
///   the flag to its URL. A non-PNG asked for stripped is not an error either: it genuinely carries
///   no workflow, and saying so by serving it is the honest answer.
/// * **The ETag must differ from the full file's.** `project_file_etag` is derived from the source
///   metadata, and the route sets `immutable` caching — so reusing it would let a cache answer a
///   strip request with the unstripped body it already holds. The variant tag is folded in.
/// * **Revalidation is ETag-ONLY.** The distinct ETag is half a fix on its own, and the missing
///   half was a live hole: both representations are derived from the same file, so they carry the
///   same `Last-Modified`, and a client that had done a plain GET could hand that date back on a
///   strip request and be told `304 Not Modified` — reusing the full body, workflow intact. The
///   date cannot distinguish the two representations, so it does not get a vote here; only the
///   variant tag does. `project_file_is_not_modified` still owns the ordinary path, where one URL
///   means one body and a date is a sound answer.
/// * **No byte ranges.** The body is a rewritten buffer whose offsets do not match the file on
///   disk, so `Accept-Ranges: none` and a `Range` header is ignored rather than answered against
///   the wrong length. This is a download, not a `<video src>`.
/// * **The memory is bounded, twice over.** The whole file is read at once — a walk of the chunk
///   framing has to see the tail — so the cost scales with the asset, and an imported PNG is
///   bounded only by `MAX_UPLOAD_BYTES` = 2 GiB. [`WORKFLOW_STRIP_SLOTS`] caps the concurrency
///   the way `thumbnail_generation_slots` does for the sibling derived representation on this same
///   route, and [`MAX_WORKFLOW_STRIP_BYTES`] caps the per-request size. Between them the route's
///   ceiling is a fixed number rather than "however many clients ask at once".
///
/// The body is assembled without a second copy. `strip_workflow_chunk` would build a new buffer
/// of its own — ~2N peak for a file of N bytes — so this calls the walk that underlies it
/// (`workflow_chunk_spans`) and serves the KEPT spans as `Bytes` slices of the one buffer already
/// read. `Bytes` slices are refcounted views, so the peak is N and the excision costs nothing.
async fn stripped_project_file_response(
    state: &AppState,
    project_file: &sceneworks_core::project_store::ProjectFile,
    request_headers: &HeaderMap,
) -> Result<Response, ApiError> {
    let path = project_file.path.clone();
    let metadata = tokio::fs::metadata(&path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let last_modified = metadata.modified().ok();

    if project_file.content_type != "image/png" {
        // Nothing of ours can be in it. Fall back to the ordinary streaming response so a video or
        // a JPEG is served exactly as it always was, under its own ETag — and under the ordinary
        // revalidation rules too, because this is not a variant: the bytes are the file's.
        let etag = project_file_etag(&metadata);
        let headers = project_file_response_headers(
            &project_file.content_type,
            metadata.len(),
            &etag,
            last_modified,
        )?;
        if project_file_is_not_modified(request_headers, &etag, last_modified) {
            return Ok(not_modified_response(headers));
        }
        let file = tokio::fs::File::open(&path)
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let mut response = Body::from_stream(ReaderStream::new(file)).into_response();
        response.headers_mut().extend(headers);
        return Ok(response);
    }

    let etag = variant_etag(&project_file_etag(&metadata), "nw");
    if if_none_match_matches(request_headers, &etag) {
        let headers = project_file_response_headers(
            &project_file.content_type,
            metadata.len(),
            &etag,
            last_modified,
        )?;
        return Ok(not_modified_response(headers));
    }

    if metadata.len() > MAX_WORKFLOW_STRIP_BYTES {
        return Err(ApiError::payload_too_large(format!(
            "This image is {} MiB, over the {} MiB SceneWorks will rewrite in memory to remove a \
             workflow. Save it normally and remove the block with another tool.",
            metadata.len() / (1024 * 1024),
            MAX_WORKFLOW_STRIP_BYTES / (1024 * 1024)
        )));
    }

    let permit = state
        .workflow_strip_slots
        .clone()
        .acquire_owned()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;

    let (bytes, spans) = tokio::task::spawn_blocking(move || {
        let original = std::fs::read(&path)?;
        // A PNG that cannot be accounted for chunk by chunk is an ERROR, never the original bytes:
        // "here is your copy without the workflow" must not be answered with a file that may still
        // carry one. An empty span list is the honest "nothing of ours is in it".
        let spans = sceneworks_core::workflow_png::workflow_chunk_spans(&original)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok::<_, std::io::Error>((original, spans))
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))?
    .map_err(|error| {
        tracing::debug!(
            event = "project_file_strip_failed",
            error = %error,
            "could not serve a project file without its workflow"
        );
        ApiError::bad_request("This image could not be rewritten without its workflow.")
    })?;

    // One allocation for the whole response: `Bytes::from` takes the read buffer over, and every
    // slice below is a refcounted view into it rather than a copy.
    let bytes = axum::body::Bytes::from(bytes);
    let kept = sceneworks_core::workflow_png::kept_spans(bytes.len(), &spans);
    let served: Vec<axum::body::Bytes> = kept
        .iter()
        .map(|span| bytes.slice(span.start..span.end))
        .collect();
    let served_len: u64 = served.iter().map(|slice| slice.len() as u64).sum();

    let mut headers = project_file_response_headers(
        &project_file.content_type,
        served_len,
        &etag,
        last_modified,
    )?;
    headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("none"));
    // The permit rides with the stream rather than being dropped here: the buffer is alive until
    // the last slice has been written, so releasing the slot at this line would let the next
    // request in while this one's memory is still held.
    let mut response = Body::from_stream(stream::iter(served.into_iter().map(move |slice| {
        let _permit = &permit;
        Ok::<_, std::convert::Infallible>(slice)
    })))
    .into_response();
    response.headers_mut().extend(headers);
    Ok(response)
}

/// Whether `If-None-Match` names `etag` (or `*`).
///
/// The ETag half of [`project_file_is_not_modified`], on its own, for the derived representations
/// whose `Last-Modified` is the source file's and therefore cannot tell one representation from
/// another.
fn if_none_match_matches(request_headers: &HeaderMap, etag: &str) -> bool {
    request_headers
        .get(header::IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(',')
                .map(str::trim)
                .any(|candidate| candidate == "*" || candidate == etag)
        })
}

/// A 304 carrying `headers` minus the content length, which a body-less response must not declare.
fn not_modified_response(headers: HeaderMap) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = StatusCode::NOT_MODIFIED;
    *response.headers_mut() = headers;
    response.headers_mut().remove(header::CONTENT_LENGTH);
    response
}

/// Distinguish a derived representation's ETag from the source file's.
///
/// The route caches `immutable`, so two representations of one URL sharing a tag is a correctness
/// bug and not a missed optimization: a client holding the full file would revalidate a strip
/// request as `304 Not Modified` and reuse the body with the workflow still in it.
fn variant_etag(base: &str, variant: &str) -> String {
    match base.strip_suffix('"') {
        Some(head) => format!("{head}-{variant}\""),
        None => format!("{base}-{variant}"),
    }
}

fn project_file_etag(metadata: &std::fs::Metadata) -> String {
    let modified = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    // Weak avoids reading and hashing a multi-gigabyte original before it can
    // stream; project media is immutable after publication.
    format!(
        "W/\"{:x}-{:x}-{:x}\"",
        metadata.len(),
        modified.as_secs(),
        modified.subsec_nanos()
    )
}

fn project_file_response_headers(
    content_type: &str,
    total: u64,
    etag: &str,
    last_modified: Option<SystemTime>,
) -> Result<HeaderMap, ApiError> {
    let mut headers = HeaderMap::new();
    for (name, value) in [
        (header::CONTENT_TYPE, content_type.to_owned()),
        (header::ACCEPT_RANGES, "bytes".to_owned()),
        (header::CONTENT_LENGTH, total.to_string()),
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_owned()),
        (
            header::CACHE_CONTROL,
            PROJECT_MEDIA_CACHE_CONTROL.to_owned(),
        ),
        (header::ETAG, etag.to_owned()),
    ] {
        headers.insert(
            name,
            HeaderValue::from_str(&value).map_err(|error| ApiError::internal(error.to_string()))?,
        );
    }
    if let Some(modified) = last_modified {
        headers.insert(
            header::LAST_MODIFIED,
            HeaderValue::from_str(&httpdate::fmt_http_date(modified))
                .map_err(|error| ApiError::internal(error.to_string()))?,
        );
    }
    Ok(headers)
}

/// Ordinary revalidation for a representation whose bytes ARE the file's: the ETag when the client
/// sent one, the modification date otherwise.
///
/// Deliberately not used by the strip variant. `If-Modified-Since` is an answer about the FILE, and
/// a derived representation shares its file with the original — see
/// [`stripped_project_file_response`], which revalidates on the tag alone.
fn project_file_is_not_modified(
    request_headers: &HeaderMap,
    etag: &str,
    last_modified: Option<SystemTime>,
) -> bool {
    if request_headers.contains_key(header::IF_NONE_MATCH) {
        return if_none_match_matches(request_headers, etag);
    }
    let Some(modified) = last_modified else {
        return false;
    };
    request_headers
        .get(header::IF_MODIFIED_SINCE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| httpdate::parse_http_date(value).ok())
        .is_some_and(|since| {
            modified
                .duration_since(since)
                .map_or(true, |delta| delta < Duration::from_secs(1))
        })
}

/// Parse a single HTTP byte range (`bytes=start-end`, `bytes=start-`, or
/// `bytes=-suffix`) against a known total size, returning an inclusive
/// `(start, end)` clamped to the file. Returns `None` for unsatisfiable or
/// multi-range requests (callers answer 416).
fn parse_single_byte_range(value: &str, total: u64) -> Option<(u64, u64)> {
    let spec = value.strip_prefix("bytes=")?.trim();
    if spec.is_empty() || spec.contains(',') || total == 0 {
        return None;
    }
    let (start_str, end_str) = spec.split_once('-')?;
    let (start, end) = if start_str.is_empty() {
        // Suffix range: last `suffix` bytes.
        let suffix: u64 = end_str.parse().ok()?;
        if suffix == 0 {
            return None;
        }
        let start = total.saturating_sub(suffix);
        (start, total - 1)
    } else {
        let start: u64 = start_str.parse().ok()?;
        let end = if end_str.is_empty() {
            total - 1
        } else {
            end_str.parse::<u64>().ok()?.min(total - 1)
        };
        (start, end)
    };
    if start > end || start >= total {
        return None;
    }
    Some((start, end))
}

/// Embedded production web bundle (apps/web/dist), compiled in only under the
/// `embed-web` feature so default/server/test builds need no web build.
#[cfg(feature = "embed-web")]
mod web_assets {
    use std::collections::HashSet;
    use std::sync::OnceLock;

    use axum::body::Body;
    use axum::http::{header, HeaderMap, HeaderValue, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};
    use rust_embed::RustEmbed;

    #[derive(RustEmbed)]
    #[folder = "../web/dist"]
    struct WebAssets;

    #[cfg(test)]
    pub(super) fn first_static_asset_path() -> String {
        WebAssets::iter()
            .find(|path| {
                path.starts_with("assets/") && !path.ends_with(".br") && !path.ends_with(".gz")
            })
            .expect("production web bundle contains a static asset")
            .into_owned()
    }

    #[cfg(test)]
    pub(super) fn first_compressible_static_asset_path() -> String {
        WebAssets::iter()
            .find(|path| {
                path.starts_with("assets/")
                    && !path.ends_with(".br")
                    && !path.ends_with(".gz")
                    && WebAssets::get(&format!("{path}.br")).is_some()
                    && WebAssets::get(&format!("{path}.gz")).is_some()
            })
            .expect("production web bundle contains a precompressed static asset")
            .into_owned()
    }

    #[cfg(test)]
    pub(super) fn first_uncompressed_static_asset_path() -> String {
        WebAssets::iter()
            .find(|path| {
                !path.ends_with(".br")
                    && !path.ends_with(".gz")
                    && path.as_ref() != IMMUTABLE_ASSET_MANIFEST
                    && WebAssets::get(&format!("{path}.br")).is_none()
                    && WebAssets::get(&format!("{path}.gz")).is_none()
            })
            .expect("production web bundle contains an uncompressed static asset")
            .into_owned()
    }

    // The desktop shell navigates its privileged webview to this server, so the embedded
    // UI runs from this origin and its CSP must come from here (tauri.conf.json only
    // governs the bundled setup screen). Kept narrow: scripts only from this origin (the
    // theme bootstrap was moved to /theme-init.js so no inline script is needed), fonts
    // self-hosted from this origin (no third-party font host — sc-8956), images/media as
    // self/data/blob, IPC for the Tauri webview. Same-origin API + SSE are covered by
    // connect-src 'self'.
    pub(super) const CONTENT_SECURITY_POLICY: &str = "default-src 'self'; \
script-src 'self'; \
style-src 'self' 'unsafe-inline'; \
font-src 'self'; \
img-src 'self' data: blob:; \
media-src 'self' data: blob:; \
connect-src 'self' ipc: http://ipc.localhost; \
object-src 'none'; \
base-uri 'self'; \
frame-ancestors 'none'; \
form-action 'self'";

    const IMMUTABLE_CACHE: &str = "public, max-age=31536000, immutable";
    const REVALIDATE_CACHE: &str = "no-cache";
    const IMMUTABLE_ASSET_MANIFEST: &str = ".sceneworks-immutable-assets";

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Encoding {
        Brotli,
        Gzip,
        Identity,
    }

    impl Encoding {
        fn suffix(self) -> &'static str {
            match self {
                Self::Brotli => ".br",
                Self::Gzip => ".gz",
                Self::Identity => "",
            }
        }

        fn header_value(self) -> Option<&'static str> {
            match self {
                Self::Brotli => Some("br"),
                Self::Gzip => Some("gzip"),
                Self::Identity => None,
            }
        }
    }

    #[derive(Default)]
    struct EncodingPreferences {
        header_present: bool,
        brotli: Option<f32>,
        gzip: Option<f32>,
        identity: Option<f32>,
        wildcard: Option<f32>,
    }

    impl EncodingPreferences {
        fn parse(headers: &HeaderMap) -> Self {
            let mut preferences = Self::default();
            for value in headers.get_all(header::ACCEPT_ENCODING) {
                preferences.header_present = true;
                let Ok(value) = value.to_str() else {
                    continue;
                };
                for item in value.split(',') {
                    let mut parts = item.trim().split(';');
                    let name = parts.next().unwrap_or_default().trim();
                    if name.is_empty() {
                        continue;
                    }
                    let mut quality = 1.0;
                    for parameter in parts {
                        let Some((key, value)) = parameter.trim().split_once('=') else {
                            continue;
                        };
                        if key.trim().eq_ignore_ascii_case("q") {
                            quality = value
                                .trim()
                                .parse::<f32>()
                                .ok()
                                .filter(|quality| (0.0..=1.0).contains(quality))
                                .unwrap_or(0.0);
                        }
                    }
                    if name.eq_ignore_ascii_case("br") {
                        preferences.brotli = Some(quality);
                    } else if name.eq_ignore_ascii_case("gzip") {
                        preferences.gzip = Some(quality);
                    } else if name.eq_ignore_ascii_case("identity") {
                        preferences.identity = Some(quality);
                    } else if name == "*" {
                        preferences.wildcard = Some(quality);
                    }
                }
            }
            preferences
        }

        fn quality(&self, encoding: Encoding) -> f32 {
            if !self.header_present {
                return if encoding == Encoding::Identity {
                    1.0
                } else {
                    0.0
                };
            }
            match encoding {
                Encoding::Brotli => self.brotli.or(self.wildcard).unwrap_or(0.0),
                Encoding::Gzip => self.gzip.or(self.wildcard).unwrap_or(0.0),
                // RFC 9110: identity is acceptable by default unless it is
                // explicitly excluded, or a wildcard q=0 excludes every
                // unlisted coding.
                Encoding::Identity => match self.identity {
                    Some(quality) => quality,
                    None if self.wildcard == Some(0.0) => 0.0,
                    None => 1.0,
                },
            }
        }
    }

    fn select_encoding(headers: &HeaderMap, requested: &str) -> Option<Encoding> {
        let preferences = EncodingPreferences::parse(headers);
        // Deterministic tie-break: Brotli, then gzip, then identity. A client
        // advertising a coding at q=1 therefore receives compression rather
        // than an equally acceptable identity response.
        let candidates = [
            (
                Encoding::Brotli,
                WebAssets::get(&format!("{requested}.br")).is_some(),
            ),
            (
                Encoding::Gzip,
                WebAssets::get(&format!("{requested}.gz")).is_some(),
            ),
            (Encoding::Identity, WebAssets::get(requested).is_some()),
        ];
        let mut selected = None;
        for (encoding, available) in candidates {
            let quality = preferences.quality(encoding);
            let is_better = match selected {
                Some((_, best_quality)) => quality > best_quality,
                None => true,
            };
            if available && quality > 0.0 && is_better {
                selected = Some((encoding, quality));
            }
        }
        selected.map(|(encoding, _)| encoding)
    }

    fn immutable_assets() -> &'static HashSet<String> {
        static ASSETS: OnceLock<HashSet<String>> = OnceLock::new();
        ASSETS.get_or_init(|| {
            WebAssets::get(IMMUTABLE_ASSET_MANIFEST)
                .map(|manifest| {
                    String::from_utf8_lossy(&manifest.data)
                        .lines()
                        .map(str::trim)
                        .filter(|path| !path.is_empty())
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default()
        })
    }

    fn is_immutable_asset(path: &str) -> bool {
        immutable_assets().contains(path)
    }

    #[cfg(test)]
    pub(super) fn immutable_asset_manifest_path() -> &'static str {
        IMMUTABLE_ASSET_MANIFEST
    }

    fn apply_common_headers(
        response: &mut axum::http::response::Builder,
        mime: &str,
        cache_control: &'static str,
    ) {
        let headers = response
            .headers_mut()
            .expect("embedded response headers are available");
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_str(mime).expect("embedded MIME type is a valid header"),
        );
        headers.insert(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static(CONTENT_SECURITY_POLICY),
        );
        headers.insert(header::VARY, HeaderValue::from_static("Accept-Encoding"));
        headers.insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static(cache_control),
        );
    }

    fn not_acceptable_response(mime: &str, cache_control: &'static str) -> Response {
        let mut response = Response::builder().status(StatusCode::NOT_ACCEPTABLE);
        apply_common_headers(&mut response, mime, cache_control);
        response
            .body(Body::empty())
            .expect("not-acceptable embedded response constructs")
    }

    fn etag(hash: [u8; 32]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut value = String::with_capacity(66);
        value.push('"');
        for byte in hash {
            value.push(HEX[(byte >> 4) as usize] as char);
            value.push(HEX[(byte & 0x0f) as usize] as char);
        }
        value.push('"');
        value
    }

    fn if_none_match(headers: &HeaderMap, current: &str) -> bool {
        headers
            .get_all(header::IF_NONE_MATCH)
            .iter()
            .filter_map(|value| value.to_str().ok())
            .flat_map(|value| value.split(','))
            .map(str::trim)
            .any(|candidate| {
                candidate == "*"
                    || candidate == current
                    || candidate
                        .strip_prefix("W/")
                        .is_some_and(|candidate| candidate == current)
            })
    }

    fn embedded_response(
        requested: &str,
        mime: &str,
        fallback: bool,
        request_headers: &HeaderMap,
    ) -> Option<Response> {
        let cache_control = if !fallback && is_immutable_asset(requested) {
            IMMUTABLE_CACHE
        } else {
            REVALIDATE_CACHE
        };
        let Some(encoding) = select_encoding(request_headers, requested) else {
            return Some(not_acceptable_response(mime, cache_control));
        };
        let representation_path = format!("{requested}{}", encoding.suffix());
        let file = WebAssets::get(&representation_path)?;
        let etag = etag(file.metadata.sha256_hash());
        let status = if if_none_match(request_headers, &etag) {
            StatusCode::NOT_MODIFIED
        } else {
            StatusCode::OK
        };
        let mut response = Response::builder().status(status);
        apply_common_headers(&mut response, mime, cache_control);
        let headers = response
            .headers_mut()
            .expect("embedded response headers are available");
        headers.insert(
            header::ETAG,
            HeaderValue::from_str(&etag).expect("SHA-256 ETag is a valid header"),
        );
        if let Some(content_encoding) = encoding.header_value() {
            headers.insert(
                header::CONTENT_ENCODING,
                HeaderValue::from_static(content_encoding),
            );
        }
        let body = if status == StatusCode::NOT_MODIFIED {
            Body::empty()
        } else {
            Body::from(file.data.into_owned())
        };
        Some(
            response
                .body(body)
                .expect("embedded response body constructs"),
        )
    }

    pub(super) async fn serve(uri: &Uri, request_headers: &HeaderMap) -> Response {
        let requested = uri.path().trim_start_matches('/');
        let requested = if requested.is_empty() {
            "index.html"
        } else {
            requested
        };
        // Encoded siblings and the cache allowlist are internal build
        // metadata, never public paths.
        if !requested.ends_with(".br")
            && !requested.ends_with(".gz")
            && requested != IMMUTABLE_ASSET_MANIFEST
            && WebAssets::get(requested).is_some()
        {
            let mime = mime_guess::from_path(requested).first_or_octet_stream();
            if let Some(response) =
                embedded_response(requested, mime.as_ref(), false, request_headers)
            {
                return response;
            }
        }
        // Single-page app: unknown non-API paths resolve to index.html so
        // client-side deep links (e.g. project routes) load correctly.
        embedded_response(
            "index.html",
            "text/html; charset=utf-8",
            true,
            request_headers,
        )
        .unwrap_or_else(|| StatusCode::NOT_FOUND.into_response())
    }
}

/// Router fallback. With `embed-web`, non-API paths are served from the embedded
/// web bundle (SPA fallback); API paths and all default-feature builds keep the
/// existing JSON not-found behavior.
async fn app_fallback(request: Request<axum::body::Body>) -> Response {
    #[cfg(feature = "embed-web")]
    {
        if !request.uri().path().starts_with("/api/") {
            return web_assets::serve(request.uri(), request.headers()).await;
        }
    }
    route_not_found(request).await
}

async fn route_not_found(request: Request<axum::body::Body>) -> Response {
    let path = request.uri().path();
    let lower_path = path.to_ascii_lowercase();
    if path.contains("/files/")
        && (path.contains("..")
            || lower_path.contains("%2e")
            || lower_path.contains("%2f")
            || lower_path.contains("%5c"))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "detail": "Invalid project file path" })),
        )
            .into_response();
    }
    if path.contains("/person-tracks/")
        && (path.contains("..")
            || lower_path.contains("%2e")
            || lower_path.contains("%2f")
            || lower_path.contains("%5c"))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "detail": "Invalid person track ID" })),
        )
            .into_response();
    }
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "detail": "Not Found" })),
    )
        .into_response()
}

async fn store_call<T, F>(state: AppState, operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(Arc<JobsStore>, u64) -> Result<T, JobsStoreError> + Send + 'static,
{
    let timeout = state.settings.worker_timeout_seconds;
    let store = state.jobs_store.clone();
    tokio::task::spawn_blocking(move || operation(store, timeout))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(Into::into)
}

async fn project_call<T, F>(state: AppState, operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(Arc<ProjectStore>) -> Result<T, ProjectStoreError> + Send + 'static,
{
    let store = state.project_store.clone();
    tokio::task::spawn_blocking(move || operation(store))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(Into::into)
}

/// Project a stored/worker-owned job into the public HTTP/SSE shape.
///
/// A selected image text encoder is resolved to an exact filesystem source only after the API has
/// accepted the client's opaque option id. That resolution must remain available to the worker
/// claim, but it is never browser state: exposing it in a job response would turn the otherwise
/// opaque option back into a host path and would let replay clients persist server-private
/// metadata. Keep the stored row intact and remove only that private resolution from public clones.
fn public_job_snapshot(mut job: JobSnapshot) -> JobSnapshot {
    let private_resolution = job
        .payload
        .get_mut("modelManifestEntry")
        .and_then(Value::as_object_mut)
        .and_then(|manifest_entry| manifest_entry.remove("resolvedTextEncoder"));
    if let Some(selected_path) = private_resolution
        .as_ref()
        .and_then(|resolution| resolution.get("path"))
        .and_then(Value::as_str)
    {
        let source_kind = private_resolution
            .as_ref()
            .and_then(|resolution| resolution.get("sourceKind"))
            .and_then(Value::as_str);
        let selected_paths = private_text_encoder_path_spellings(selected_path, source_kind);
        redact_private_text_encoder_diagnostic(&mut job.message);
        if let Some(error) = job.error.as_mut() {
            redact_private_text_encoder_diagnostic(error);
        }
        if let Some(title) = job.title.as_mut() {
            redact_selected_text_encoder_paths(title, &selected_paths);
        }
        redact_selected_text_encoder_paths_in_map(&mut job.payload, &selected_paths);
        redact_selected_text_encoder_paths_in_map(&mut job.result, &selected_paths);
        redact_selected_text_encoder_paths_in_btree_map(&mut job.extra, &selected_paths);
    }
    job
}

#[derive(Clone, Debug)]
struct PrivatePathSpelling {
    value: String,
    directory: bool,
    case_insensitive: bool,
}

fn private_text_encoder_path_spellings(
    path: &str,
    source_kind: Option<&str>,
) -> Vec<PrivatePathSpelling> {
    // Missing/unknown private metadata is not expected, but public projection must fail closed:
    // directory semantics are the safe superset because they also scrub admitted descendants.
    let directory = source_kind != Some("file");
    let path_bytes = path.as_bytes();
    let case_insensitive = (path_bytes.first().is_some_and(u8::is_ascii_alphabetic)
        && path_bytes.get(1) == Some(&b':')
        && path_bytes
            .get(2)
            .is_some_and(|separator| matches!(separator, b'/' | b'\\')))
        || path.starts_with("\\\\");
    let mut spellings = vec![path.to_owned()];
    if case_insensitive {
        spellings.push(path.replace('\\', "/"));
        spellings.push(path.replace('/', "\\"));
    }
    if directory {
        for spelling in &mut spellings {
            while spelling.len() > 1
                && spelling.ends_with(['/', '\\'])
                && !spelling.trim_matches(['/', '\\']).is_empty()
                && !(spelling.len() == 3 && spelling.as_bytes().get(1) == Some(&b':'))
            {
                spelling.pop();
            }
        }
    }
    spellings.retain(|spelling| !spelling.is_empty());
    spellings.sort_by_key(|spelling| std::cmp::Reverse(spelling.len()));
    spellings.dedup();
    spellings
        .into_iter()
        .map(|value| PrivatePathSpelling {
            value,
            directory,
            case_insensitive,
        })
        .collect()
}

fn redact_private_text_encoder_diagnostic(value: &mut String) {
    redact_absolute_paths(value);
}

/// Redact only the admitted selected source spelling from public contract data.
///
/// Payload/result/extra can legitimately contain unrelated filesystem fields (for example an
/// installed LoRA or generated output). They must remain stable when a text encoder is selected.
/// HTTP(S) and scheme-relative URL spans are skipped so a URL whose path happens to contain the
/// same spelling is not rewritten.
fn redact_selected_text_encoder_paths(value: &mut String, spellings: &[PrivatePathSpelling]) {
    const REDACTED: &str = "[selected text encoder]";
    let original = value.as_bytes();
    let is_url_end = |byte: u8| {
        byte.is_ascii_whitespace()
            || matches!(
                byte,
                b'"' | b'\'' | b'`' | b'<' | b'>' | b'|' | b')' | b']' | b'}' | b',' | b';'
            )
    };
    let starts_with_ignore_ascii_case = |index: usize, needle: &[u8]| {
        original
            .get(index..index.saturating_add(needle.len()))
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(needle))
    };
    let url_boundary = |index: usize| {
        index == 0
            || original[index - 1].is_ascii_whitespace()
            || matches!(
                original[index - 1],
                b'(' | b'[' | b'{' | b'=' | b',' | b';' | b'\'' | b'"' | b'`' | b'<'
            )
    };
    let is_web_url = |index: usize| {
        (url_boundary(index)
            && (starts_with_ignore_ascii_case(index, b"http://")
                || starts_with_ignore_ascii_case(index, b"https://")))
            || ((index == 0 || url_boundary(index))
                && original.get(index..index + 2) == Some(b"//")
                && original
                    .get(index + 2)
                    .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'['))
    };
    let is_file_url = |index: usize| {
        url_boundary(index)
            && starts_with_ignore_ascii_case(index, b"file:")
            && original
                .get(index + 5)
                .is_some_and(|separator| matches!(separator, b'/' | b'\\'))
    };

    let mut output = String::with_capacity(value.len());
    let mut copied_through = 0;
    let mut index = 0;
    while index < original.len() {
        if is_file_url(index) {
            output.push_str(&value[copied_through..index]);
            output.push_str(REDACTED);
            index += 5;
            while index < original.len() && !is_url_end(original[index]) {
                index += 1;
            }
            copied_through = index;
            continue;
        }
        let after_scheme_slashes = || {
            let mut cursor = index;
            while cursor > 0 && matches!(original[cursor - 1], b'/' | b'\\') {
                cursor -= 1;
            }
            cursor > 0 && original[cursor - 1] == b':'
        };
        let Some(spelling) = spellings.iter().find(|spelling| {
            (index == 0
                || after_scheme_slashes()
                || !matches!(
                    original[index - 1],
                    b'.' | b'-' | b'_' | b'/' | b'\\' | b'0'..=b'9' | b'A'..=b'Z'
                        | b'a'..=b'z'
                ))
                && original
                    .get(index..index.saturating_add(spelling.value.len()))
                    .is_some_and(|candidate| {
                        if spelling.case_insensitive {
                            candidate.eq_ignore_ascii_case(spelling.value.as_bytes())
                        } else {
                            candidate == spelling.value.as_bytes()
                        }
                    })
                && original
                    .get(index + spelling.value.len())
                    .map_or(true, |next| {
                        if spelling.directory {
                            matches!(next, b'/' | b'\\')
                                || next.is_ascii_whitespace()
                                || matches!(
                                    next,
                                    b'"' | b'\''
                                        | b'`'
                                        | b'<'
                                        | b'>'
                                        | b'|'
                                        | b')'
                                        | b']'
                                        | b'}'
                                        | b','
                                        | b';'
                                        | b':'
                                        | b'='
                                        | b'?'
                                        | b'#'
                                        | b'&'
                                )
                        } else if *next == b'.' {
                            original
                                .get(index + spelling.value.len() + 1)
                                .map_or(true, |after| {
                                    after.is_ascii_whitespace()
                                        || matches!(
                                            after,
                                            b'"' | b'\''
                                                | b'`'
                                                | b'<'
                                                | b'>'
                                                | b'|'
                                                | b')'
                                                | b']'
                                                | b'}'
                                                | b','
                                                | b';'
                                        )
                                })
                        } else {
                            next.is_ascii_whitespace()
                                || matches!(
                                    next,
                                    b'"' | b'\''
                                        | b'`'
                                        | b'<'
                                        | b'>'
                                        | b'|'
                                        | b')'
                                        | b']'
                                        | b'}'
                                        | b','
                                        | b';'
                                        | b':'
                                        | b'='
                                        | b'?'
                                        | b'#'
                                        | b'&'
                                        | b'!'
                                )
                        }
                    })
        }) else {
            if is_web_url(index) {
                index += 1;
                while index < original.len() && !is_url_end(original[index]) {
                    index += 1;
                }
                continue;
            }
            index += 1;
            continue;
        };
        output.push_str(&value[copied_through..index]);
        output.push_str(REDACTED);
        index += spelling.value.len();
        copied_through = index;
    }
    if copied_through != 0 {
        output.push_str(&value[copied_through..]);
        *value = output;
    }
}

/// Remove any remaining absolute filesystem token from a selected-encoder job's public clone.
///
/// A confinement error can enumerate unrelated allowed roots and a later symlink error can name a
/// target that no longer matches the selected source. This pure lexical pass is deliberately
/// scoped to jobs carrying the server-private encoder resolution; it performs no filesystem I/O
/// while projecting an HTTP/SSE response. Once a filesystem location is found, the remainder is
/// dropped rather than guessing where an unquoted `Path::display()` value containing spaces ends.
/// Web URLs and relative paths remain intact, while `file://` URLs are filesystem locations and
/// are therefore private.
fn redact_absolute_paths(value: &mut String) {
    const REDACTED: &str = "[selected text encoder]";
    let bytes = value.as_bytes();
    let boundary = |index: usize| match index.checked_sub(1).and_then(|i| bytes.get(i)) {
        None => true,
        Some(previous) => {
            !previous.is_ascii_alphanumeric()
                && !matches!(previous, b'_' | b'.' | b'-' | b'/' | b'\\')
        }
    };
    let starts_with_ignore_ascii_case = |index: usize, needle: &[u8]| {
        bytes
            .get(index..index.saturating_add(needle.len()))
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(needle))
    };
    let is_web_url = |index: usize| {
        boundary(index)
            && (starts_with_ignore_ascii_case(index, b"http://")
                || starts_with_ignore_ascii_case(index, b"https://")
                || (bytes.get(index..index + 2) == Some(b"//")
                    && bytes
                        .get(index + 2)
                        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'[')))
    };
    let is_file_url =
        |index: usize| boundary(index) && starts_with_ignore_ascii_case(index, b"file://");
    let is_home_relative = |index: usize| {
        if bytes.get(index.wrapping_sub(1)) == Some(&b'~') {
            return true;
        }
        if bytes.get(index.wrapping_sub(1)) != Some(&b'}') {
            return false;
        }
        bytes[..index]
            .iter()
            .rposition(|byte| *byte == b'{')
            .is_some_and(|open| {
                open > 0
                    && bytes[open - 1] == b'$'
                    && bytes[open + 1..index - 1]
                        .iter()
                        .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            })
    };
    let is_absolute_path = |index: usize, require_boundary: bool| {
        if require_boundary && !boundary(index) {
            return false;
        }
        let starts_component = |byte: &u8| {
            !byte.is_ascii_whitespace()
                && !matches!(
                    byte,
                    b'/' | b'\\'
                        | b'"'
                        | b'\''
                        | b'`'
                        | b'<'
                        | b'>'
                        | b'|'
                        | b')'
                        | b']'
                        | b'}'
                        | b','
                        | b';'
                )
        };
        let unix = !is_home_relative(index)
            && bytes.get(index) == Some(&b'/')
            && bytes.get(index + 1).is_some_and(starts_component);
        let drive = bytes.get(index).is_some_and(u8::is_ascii_alphabetic)
            && bytes.get(index + 1) == Some(&b':')
            && bytes
                .get(index + 2)
                .is_some_and(|separator| matches!(separator, b'/' | b'\\'))
            && bytes.get(index + 3).is_some_and(starts_component);
        let unc = bytes.get(index..index + 2) == Some(b"\\\\")
            && bytes.get(index + 2).is_some_and(starts_component);
        unix || drive || unc
    };
    let is_url_end = |byte: u8| {
        byte.is_ascii_whitespace()
            || matches!(
                byte,
                b'"' | b'\'' | b'`' | b'<' | b'>' | b'|' | b')' | b']' | b'}' | b',' | b';'
            )
    };
    let web_token_end = |start: usize| {
        let mut end = start;
        while end < bytes.len() && !is_url_end(bytes[end]) {
            end += 1;
        }
        end
    };
    let mut index = 0;
    while index < bytes.len() {
        if is_web_url(index) {
            index = web_token_end(index);
            continue;
        }
        let file_url = is_file_url(index);
        if !file_url && !is_absolute_path(index, true) {
            index += 1;
            continue;
        }

        let prefix_end = index
            .checked_sub(1)
            .filter(|previous| matches!(bytes[*previous], b'"' | b'\'' | b'`' | b'<'));
        value.truncate(prefix_end.unwrap_or(index));
        value.push_str(REDACTED);
        return;
    }
}

fn redact_selected_text_encoder_paths_in_map(
    object: &mut serde_json::Map<String, Value>,
    spellings: &[PrivatePathSpelling],
) {
    let original = std::mem::take(object);
    let mut redacted = serde_json::Map::with_capacity(original.len());
    let mut collisions = std::collections::HashSet::new();
    for (mut key, mut value) in original {
        redact_selected_text_encoder_paths(&mut key, spellings);
        redact_selected_text_encoder_paths_in_value(&mut value, spellings);
        if collisions.contains(&key) {
            continue;
        }
        if redacted.contains_key(&key) {
            redacted.insert(
                key.clone(),
                Value::String("[redacted collision]".to_owned()),
            );
            collisions.insert(key);
        } else {
            redacted.insert(key, value);
        }
    }
    *object = redacted;
}

fn redact_selected_text_encoder_paths_in_btree_map(
    object: &mut std::collections::BTreeMap<String, Value>,
    spellings: &[PrivatePathSpelling],
) {
    let original = std::mem::take(object);
    let mut redacted = std::collections::BTreeMap::new();
    let mut collisions = std::collections::HashSet::new();
    for (mut key, mut value) in original {
        redact_selected_text_encoder_paths(&mut key, spellings);
        redact_selected_text_encoder_paths_in_value(&mut value, spellings);
        if collisions.contains(&key) {
            continue;
        }
        if redacted.contains_key(&key) {
            redacted.insert(
                key.clone(),
                Value::String("[redacted collision]".to_owned()),
            );
            collisions.insert(key);
        } else {
            redacted.insert(key, value);
        }
    }
    *object = redacted;
}

fn redact_selected_text_encoder_paths_in_value(
    value: &mut Value,
    spellings: &[PrivatePathSpelling],
) {
    match value {
        Value::String(value) => redact_selected_text_encoder_paths(value, spellings),
        Value::Array(values) => {
            for value in values {
                redact_selected_text_encoder_paths_in_value(value, spellings);
            }
        }
        Value::Object(object) => redact_selected_text_encoder_paths_in_map(object, spellings),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn public_job_snapshots(jobs: Vec<JobSnapshot>) -> Vec<JobSnapshot> {
    jobs.into_iter().map(public_job_snapshot).collect()
}

fn public_queue_summary(mut summary: QueueSummary) -> QueueSummary {
    summary.active_jobs = public_job_snapshots(summary.active_jobs);
    summary
}

/// Defense in depth for live publications. Most producers already pass typed public projections,
/// but `job.updated` also originates at worker/stale-sweep seams that intentionally retain the raw
/// job for lifecycle work. Redact at the serialization boundary so no new producer can accidentally
/// publish the worker-private selection receipt. `jobs.snapshot` is currently built directly in the
/// SSE bootstrap; supporting it here keeps this boundary safe if that snapshot becomes live later.
fn redact_private_job_metadata_from_event(event: &str, value: &mut Value) {
    fn redact_job(value: &mut Value) {
        let Ok(job) = serde_json::from_value::<JobSnapshot>(value.clone()) else {
            // A job event that cannot be projected through the public typed contract is not safe to
            // publish. Clear it so the caller's serialization remains path-free and fail-closed.
            *value = Value::Null;
            return;
        };
        *value = serde_json::to_value(public_job_snapshot(job)).unwrap_or(Value::Null);
    }

    match event {
        "job.updated" => redact_job(value),
        "queue.updated" => {
            if let Some(jobs) = value.get_mut("activeJobs").and_then(Value::as_array_mut) {
                for job in jobs {
                    redact_job(job);
                }
            }
        }
        "jobs.snapshot" => {
            if let Some(jobs) = value.get_mut("jobs").and_then(Value::as_array_mut) {
                for job in jobs {
                    redact_job(job);
                }
            }
        }
        _ => {}
    }
}

async fn queue_summary_snapshot(state: AppState) -> Result<QueueSummary, ApiError> {
    queue_summary_snapshot_inner(state, false).await
}

/// Build the queue summary, optionally SKIPPING the stale-worker sweep.
///
/// The sweep is a second blocking round-trip that mutates jobs to `interrupted`.
/// Callers that already ran `mark_stale_workers_interrupted` in their own
/// transaction this request (currently `claim_job`) pass `skip_sweep = true` so
/// the queue refresh doesn't sweep a SECOND time on the same request (sc-8889 /
/// F-087). Every other caller passes `skip_sweep = false` and gets the sweep, so
/// a plain queue read (GET /queue) or a mutation that didn't sweep still reaps
/// stale workers.
async fn queue_summary_snapshot_inner(
    state: AppState,
    skip_sweep: bool,
) -> Result<QueueSummary, ApiError> {
    let (sweep, summary) = store_call(state.clone(), move |store, timeout| {
        // When the caller already swept this request, don't pay for a second
        // sweep — just read the summary. The empty StaleSweep means the
        // job.updated fan-out below is a no-op (the caller emitted those
        // events off its own sweep result).
        let sweep = if skip_sweep {
            StaleSweep::default()
        } else {
            store.mark_stale_workers_interrupted(timeout)?
        };
        let summary = store.queue_summary();
        Ok((sweep, summary))
    })
    .await?;
    handle_stale_sweep(&state, &sweep);
    let summary = summary?;
    // The stale-sweep mutates jobs to `interrupted` in the DB but — unlike a worker-reported
    // terminal status (`update_job_progress`) or the supervisor crash path (`worker_terminated`) —
    // emits no per-job event. Broadcast `job.updated` for each swept job so a live client's job card
    // flips to "Interrupted" instead of showing its last running state forever: the frontend's job
    // list is driven by `job.updated`, while `queue.updated` only refreshes the summary/workers
    // (sc-8186). The sweep returns each job exactly once (it also flips the owning worker offline, so
    // a later sweep can't re-select it), so this neither spams nor double-fires. When skip_sweep is
    // set the sweep is empty, so nothing is broadcast here.
    Ok(public_queue_summary(summary))
}

/// Apply the API-visible consequences of a stale-worker sweep exactly once at
/// every route that can trigger one. The store returns the terminal job
/// snapshots so callers can invalidate model install state and notify live
/// clients rather than silently discarding those transitions.
fn handle_stale_sweep(state: &AppState, sweep: &StaleSweep) {
    invalidate_model_catalog_for_terminal_jobs(state, &sweep.jobs);
    for job in &sweep.jobs {
        publish(state, "job.updated", job);
    }
}

async fn create_generation_job(
    state: AppState,
    job_type: JobType,
    project_id: Option<String>,
    project_name: Option<String>,
    payload: JsonObject,
    requested_gpu: String,
) -> Result<JobSnapshot, ApiError> {
    create_generation_job_with_status(
        state,
        job_type,
        project_id,
        project_name,
        payload,
        requested_gpu,
        None,
    )
    .await
}

/// Like [`create_generation_job`], but creates the job in an explicit initial status.
/// `None` is the default `queued` (immediately claimable); `Some(JobStatus::PendingCaption)`
/// creates the job NON-claimable so an API-side async pre-step can rewrite its payload and
/// promote it to `queued` before any worker sees it (sc-9120, Ideogram 4 auto-caption). The
/// job.updated/queue.updated events fire either way, so a `pending_caption` job appears in the
/// queue view immediately.
async fn create_generation_job_with_status(
    state: AppState,
    job_type: JobType,
    project_id: Option<String>,
    project_name: Option<String>,
    mut payload: JsonObject,
    requested_gpu: String,
    initial_status: Option<JobStatus>,
) -> Result<JobSnapshot, ApiError> {
    model_sources::ensure_runtime_model_sources(&state, &job_type, &mut payload).await?;
    let job = store_call(state.clone(), move |store, _timeout| {
        store.create_job(CreateJob {
            job_type,
            project_id,
            project_name,
            payload,
            requested_gpu,
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status,
        })
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok(job)
}

async fn publish_queue(state: &AppState) -> Result<(), ApiError> {
    let _snapshot_guard = state.queue_snapshot_lock.lock().await;
    let queue = queue_summary_snapshot(state.clone()).await?;
    publish(state, "queue.updated", &queue);
    Ok(())
}

/// Like [`publish_queue`], but skips the stale-worker sweep because the caller
/// already ran one in its own transaction this request (sc-8889 / F-087). Use
/// only right after a `mark_stale_workers_interrupted` call — otherwise stale
/// workers won't be reaped on this refresh.
async fn publish_queue_skip_sweep(state: &AppState) -> Result<(), ApiError> {
    let _snapshot_guard = state.queue_snapshot_lock.lock().await;
    let queue = queue_summary_snapshot_inner(state.clone(), true).await?;
    publish(state, "queue.updated", &queue);
    Ok(())
}

fn publish<T: Serialize>(state: &AppState, event: &str, data: &T) {
    let data = if matches!(event, "job.updated" | "queue.updated" | "jobs.snapshot") {
        let Ok(mut value) = serde_json::to_value(data) else {
            return;
        };
        redact_private_job_metadata_from_event(event, &mut value);
        serde_json::to_string(&value)
    } else {
        serde_json::to_string(data)
    };
    if let Ok(data) = data {
        // Publishing with no subscribers is expected; slow subscribers are dropped so they reconnect.
        state.events.publish(EventMessage {
            event: event.to_owned(),
            data,
            revision: 0,
        });
    }
}

async fn project_path_for_id(state: AppState, project_id: &str) -> Result<PathBuf, ApiError> {
    let project_id = project_id.to_owned();
    let project = project_call(state, move |store| store.get_project(&project_id)).await?;
    Ok(PathBuf::from(project.path))
}

/// Resolve a project asset record to an on-disk path without allowing absolute
/// paths or traversal outside the owning project.
async fn resolve_project_confined_asset_path(
    state: AppState,
    project_id: &str,
    asset_id: &str,
    project_path: &FsPath,
) -> Result<PathBuf, ApiError> {
    let project_id = project_id.to_owned();
    let asset_id = asset_id.to_owned();
    let asset = project_call(state, move |store| store.get_asset(&project_id, &asset_id)).await?;
    let relative_path = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("Asset has no file path"))?;
    let mut resolved = project_path.to_path_buf();
    for component in FsPath::new(relative_path).components() {
        match component {
            std::path::Component::Normal(value) => resolved.push(value),
            _ => {
                return Err(ApiError::bad_request(
                    "Asset path must stay inside the project directory",
                ))
            }
        }
    }
    if !resolved.exists() {
        return Err(ApiError::bad_request("Asset file not found on disk"));
    }
    Ok(resolved)
}

fn model_lora_families(model: &Value) -> Vec<String> {
    families_from_value_chain(
        model,
        &["families", "compatibleFamilies", "modelFamilies"],
        Some("loraCompatibility"),
    )
}

/// Every LoRA family this model can load: the families it DECLARES plus each one's
/// extra-compatible families from the shared registry (`sceneworks-core`'s
/// `accepted_lora_families`). Normalized and de-duplicated, declared entries first.
///
/// 🔴 The generate-time gate must use this, not [`model_lora_families`]. Keying the gate on the
/// declared set alone made the API stricter than the registry it shares with the worker: Krea
/// Realtime 14B declares only `krea-realtime` and additionally accepts `wan-video`, so a Wan style
/// LoRA the engine installs happily was rejected at submit with "appears to be a wan-video LoRA,
/// which is not compatible" (sc-15017 — caught by running it, not by a test). The same divergence
/// applied to `chroma`←`flux` and `flux2-klein`/`flux2-dev`←`flux2`.
///
/// One-directional, like the registry: nothing here gives a Wan model Krea-Realtime LoRAs.
///
/// ⚠️ Every family that comes back is re-normalized through [`normalize_lora_family`] (the API's
/// CANONICAL spelling — `krea_2`, underscore) because `accepted_lora_families` returns core's
/// hyphenated `normalize_model_family` form (`krea-2`). Both sides of the membership test in
/// `validate_lora_specs_for_model` are in the canonical form, so skipping this re-normalization
/// silently un-does sc-8185 and falsely rejects a Krea 2 LoRA.
fn accepted_model_lora_families(model: &Value) -> Vec<String> {
    let mut accepted: Vec<String> = Vec::new();
    for declared in model_lora_families(model) {
        for family in sceneworks_core::lora_family::accepted_lora_families(&declared) {
            let family = normalize_lora_family(&family);
            if !accepted.contains(&family) {
                accepted.push(family);
            }
        }
    }
    accepted
}

fn families_from_value_chain(
    value: &Value,
    direct_fields: &[&str],
    compatibility_field: Option<&str>,
) -> Vec<String> {
    let compatibility = compatibility_field
        .and_then(|field| value.get(field))
        .unwrap_or(&Value::Null);
    let values = direct_fields
        .iter()
        .find_map(|field| value.get(*field).filter(|value| !value.is_null()))
        .or_else(|| {
            compatibility
                .get("families")
                .filter(|value| !value.is_null())
        })
        .or_else(|| value.get("family").filter(|value| !value.is_null()));
    let mut families = match values {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(normalize_lora_family)
            .collect(),
        Some(Value::String(value)) => vec![normalize_lora_family(value)],
        _ => Vec::new(),
    };
    families.sort();
    families.dedup();
    families
}

fn job_lora_id(lora: &Value) -> Option<&str> {
    lora.as_str()
        .or_else(|| lora.get("id").and_then(Value::as_str))
        .or_else(|| lora.get("loraId").and_then(Value::as_str))
}

async fn catalog_delete_warnings(
    state: &AppState,
    kind: &str,
    id: &str,
    project_id: Option<&str>,
    catalogs: Option<&JobCatalogSnapshot>,
) -> Result<Vec<String>, ApiError> {
    let mut warnings = Vec::new();
    let presets = recipe_preset_catalog_with(state, project_id, catalogs).await?;
    let preset_names = presets
        .iter()
        .filter(|preset| match kind {
            "model" => preset.get("model").and_then(Value::as_str) == Some(id),
            "lora" => recipe_preset_loras(preset)
                .iter()
                .any(|lora| job_lora_id(lora) == Some(id) || preset_lora_id(lora) == Some(id)),
            _ => false,
        })
        .filter_map(|preset| {
            preset
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| preset.get("id").and_then(Value::as_str))
        })
        .take(5)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if !preset_names.is_empty() {
        warnings.push(format!(
            "Recipe presets reference this {kind}: {}",
            preset_names.join(", ")
        ));
    }

    let item_id = id.to_owned();
    let (sweep, jobs) = store_call(state.clone(), move |store, timeout| {
        let sweep = store.mark_stale_workers_interrupted(timeout)?;
        let jobs = store.list_jobs(None, None, 100);
        Ok((sweep, jobs))
    })
    .await?;
    handle_stale_sweep(state, &sweep);
    let jobs = jobs?;
    let job_ids = jobs
        .iter()
        .filter(|job| job_references_catalog_item(job, kind, &item_id))
        .filter_map(|job| {
            if job.id.is_empty() {
                None
            } else {
                Some(job.id.clone())
            }
        })
        .take(5)
        .collect::<Vec<_>>();
    if !job_ids.is_empty() {
        warnings.push(format!(
            "Recent or queued jobs reference this {kind}: {}",
            job_ids.join(", ")
        ));
    }
    Ok(warnings)
}

fn job_references_catalog_item(job: &JobSnapshot, kind: &str, id: &str) -> bool {
    match kind {
        "model" => {
            job.payload.get("model").and_then(Value::as_str) == Some(id)
                || job.payload.get("modelId").and_then(Value::as_str) == Some(id)
        }
        "lora" => {
            job.payload.get("loraId").and_then(Value::as_str) == Some(id)
                || job
                    .payload
                    .get("loras")
                    .and_then(Value::as_array)
                    .is_some_and(|loras| loras.iter().any(|lora| job_lora_id(lora) == Some(id)))
        }
        _ => false,
    }
}

fn serialize_job_lora(lora: &Value, selected_lora: &Value, lora_id: &str) -> Value {
    json!({
        "id": lora_id,
        "name": preferred_lora_str(selected_lora, lora, "name", lora_id),
        "scope": preferred_lora_str(selected_lora, lora, "scope", "global"),
        "weight": preset_lora_weight(lora, selected_lora),
        "family": preferred_lora_value(selected_lora, lora, "family"),
        "families": preferred_lora_value(selected_lora, lora, "families"),
        "compatibleFamilies": preferred_lora_value(selected_lora, lora, "compatibleFamilies"),
        "modelFamilies": preferred_lora_value(selected_lora, lora, "modelFamilies"),
        // The specific base model the LoRA was trained for (e.g. wan_2_2 vs
        // wan_2_2_t2v_14b). The worker gates Wan 5B-vs-14B on this since both share
        // family `wan-video`. Absent for LoRAs that don't record one.
        "baseModel": preferred_lora_value(selected_lora, lora, "baseModel"),
        // Adapter network type (epic 2193). Carried into the generation payload so
        // the worker can route LoKr off the MLX backend without opening the file.
        "networkType": preferred_lora_value(selected_lora, lora, "networkType"),
        "triggerWords": preferred_lora_array(selected_lora, lora, "triggerWords"),
        "compatibility": preferred_lora_object(selected_lora, lora, "compatibility"),
        "icLora": preferred_lora_value(selected_lora, lora, "icLora"),
        "conditioningRole": preferred_lora_value(selected_lora, lora, "conditioningRole"),
        // Sampling-regime role (`accelerator`, sc-13882): carried into the generation payload so the
        // worker can switch a Krea 2 Raw t2i job to the turbo sampling regime (epic 13879 S3, sc-13883)
        // — the sibling of `conditioningRole`, round-tripped identically.
        "role": preferred_lora_value(selected_lora, lora, "role"),
        "installedPath": preferred_lora_value(selected_lora, lora, "installedPath"),
        "sourcePath": preferred_lora_value(selected_lora, lora, "sourcePath"),
        // Declared adapter filename(s): lets the worker load the record's final adapter
        // from its folder instead of an arbitrary sibling — e.g. a trained LoRA's final
        // `<stem>.safetensors` over a `<stem>-stepNNN` checkpoint (sc-10221).
        "files": preferred_lora_value(selected_lora, lora, "files"),
        "source": preferred_lora_value(selected_lora, lora, "source"),
        "presetManaged": selected_lora.get("presetManaged").and_then(Value::as_bool).unwrap_or(false)
    })
}

fn preferred_lora_str<'a>(
    selected_lora: &'a Value,
    catalog_lora: &'a Value,
    field: &str,
    fallback: &'a str,
) -> &'a str {
    selected_lora
        .get(field)
        .and_then(Value::as_str)
        .or_else(|| catalog_lora.get(field).and_then(Value::as_str))
        .unwrap_or(fallback)
}

fn preferred_lora_value(selected_lora: &Value, catalog_lora: &Value, field: &str) -> Value {
    selected_lora
        .get(field)
        .filter(|value| !value.is_null())
        .or_else(|| catalog_lora.get(field))
        .cloned()
        .unwrap_or(Value::Null)
}

fn preferred_lora_array(selected_lora: &Value, catalog_lora: &Value, field: &str) -> Value {
    selected_lora
        .get(field)
        .filter(|value| value.is_array())
        .or_else(|| catalog_lora.get(field).filter(|value| value.is_array()))
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()))
}

fn preferred_lora_object(selected_lora: &Value, catalog_lora: &Value, field: &str) -> Value {
    selected_lora
        .get(field)
        .filter(|value| value.is_object())
        .or_else(|| catalog_lora.get(field).filter(|value| value.is_object()))
        .cloned()
        .unwrap_or_else(|| Value::Object(JsonObject::new()))
}

fn normalize_inline_job_lora(lora: &Value, lora_id: &str) -> Value {
    match lora {
        Value::Object(object) => {
            let mut object = object.clone();
            object.insert("id".to_owned(), Value::String(lora_id.to_owned()));
            Value::Object(object)
        }
        _ => json!({ "id": lora_id }),
    }
}

fn json_size_to_u64(value: &Value) -> Option<u64> {
    if let Some(value) = value.as_u64() {
        return Some(value);
    }
    value.as_str().and_then(|value| value.parse::<u64>().ok())
}

fn allow_pattern_matches(path: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    patterns
        .iter()
        .any(|pattern| pattern_matches(pattern, path))
}

fn pattern_matches(pattern: &str, value: &str) -> bool {
    let (pattern, value) = if cfg!(windows) {
        (pattern.to_ascii_lowercase(), value.to_ascii_lowercase())
    } else {
        (pattern.to_owned(), value.to_owned())
    };
    glob::Pattern::new(&pattern).is_ok_and(|pattern| pattern.matches(&value))
}

fn quote_huggingface_repo(repo: &str) -> String {
    let mut output = String::new();
    for byte in repo.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            output.push(char::from(byte));
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

fn format_bytes(value: u64) -> String {
    let mut size = value as f64;
    for unit in ["B", "KB", "MB", "GB", "TB"] {
        if size < 1024.0 || unit == "TB" {
            if unit == "B" {
                return format!("{} {unit}", size as u64);
            }
            return format!("{size:.1} {unit}");
        }
        size /= 1024.0;
    }
    format!("{size:.1} TB")
}

fn string_array_field(payload: &Value, field: &str) -> Vec<String> {
    payload
        .get(field)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn safe_download_dir(repo: &str) -> String {
    let mut output = String::new();
    let mut in_replacement = false;
    for character in repo.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            output.push(character);
            in_replacement = false;
        } else if !in_replacement {
            output.push_str("__");
            in_replacement = true;
        }
    }
    let output = output.trim_matches('_').to_owned();
    if output.is_empty() {
        "download".to_owned()
    } else {
        output
    }
}

fn sanitized_upload_filename(filename: &str) -> String {
    let filename = filename
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(filename)
        .trim();
    let sanitized = safe_download_dir(filename);
    if sanitized.is_empty() || sanitized == "download" {
        "lora.safetensors".to_owned()
    } else {
        sanitized
    }
}

fn validate_lora_import_source_path(
    source_path: &str,
    allowed_roots: &[PathBuf],
) -> Result<(), ApiError> {
    let source = FsPath::new(source_path);
    if !source.is_absolute() {
        return Err(ApiError::bad_request("LoRA sourcePath must be absolute"));
    }
    let source = std::fs::canonicalize(source)
        .map_err(|_| ApiError::bad_request(format!("LoRA sourcePath not found: {source_path}")))?;
    let metadata = std::fs::metadata(&source)
        .map_err(|error| ApiError::bad_request(format!("Invalid LoRA sourcePath: {error}")))?;
    if !metadata.is_file() && !metadata.is_dir() {
        return Err(ApiError::bad_request(
            "LoRA sourcePath must point to a file or directory",
        ));
    }
    for root in allowed_roots {
        if let Ok(root) = std::fs::canonicalize(root) {
            if source.starts_with(root) {
                return Ok(());
            }
        }
    }
    Err(ApiError::bad_request(
        "LoRA sourcePath must be inside app-managed data/loras, project/loras, or staged upload folders",
    ))
}

fn validate_source_url(source_url: &str) -> Result<(), ApiError> {
    parse_lora_source_url(source_url)
        .map(|_| ())
        .map_err(|error| ApiError::bad_request(lora_url_error_message(error)))
}

fn validate_lora_family(models: &[Value], family: &str) -> Result<String, ApiError> {
    let normalized = normalize_lora_family(family);
    if normalized.is_empty() {
        return Err(ApiError::bad_request(
            "LoRA family is required when provided",
        ));
    }
    let known = known_lora_families(models);
    if !known.is_empty() && !known.iter().any(|known_family| known_family == &normalized) {
        return Err(ApiError::bad_request(format!(
            "Unsupported LoRA family: {family}"
        )));
    }
    Ok(normalized)
}

fn normalize_lora_family(family: &str) -> String {
    // Delegate to the shared canonical resolver so the API agrees with the worker
    // and the catalog on one token per family (Krea 2's `krea2`/`krea-2`/`krea_2`
    // all become `krea_2`). Applied symmetrically to every family string the API
    // compares, so membership tests stay consistent (see `validate_lora_specs_for_model`).
    canonical_lora_family(family)
}

fn known_lora_families(models: &[Value]) -> Vec<String> {
    let mut families = Vec::new();
    for model in models {
        families.extend(model_lora_families(model));
    }
    families.sort();
    families.dedup();
    families
}

/// LoRA families accepted by installed models, read directly from the model
/// manifests. Unlike `known_lora_families(&model_catalog(..))`, this does no
/// Hugging Face size-estimation, so callers on hot/offline paths (the training
/// submit guardrail) stay local.
async fn known_lora_families_from_manifests(state: &AppState) -> Result<Vec<String>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let mut models =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    models.extend(
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?,
    );
    Ok(known_lora_families(&models))
}

fn slugify_lora_id(value: &str) -> String {
    let mut output = String::new();
    let mut previous_separator = false;
    for character in value.trim().chars() {
        if character.is_ascii_alphanumeric() {
            output.push(character.to_ascii_lowercase());
            previous_separator = false;
        } else if !previous_separator && !output.is_empty() {
            output.push('_');
            previous_separator = true;
        }
    }
    while output.ends_with('_') {
        output.pop();
    }
    if output.is_empty() {
        "lora".to_owned()
    } else {
        output
    }
}

fn now_rfc3339() -> String {
    format_unix_seconds(now_unix_seconds())
}

fn huggingface_repo_cache_exists(path: &FsPath) -> bool {
    path.join("snapshots").is_dir() || path.join("blobs").is_dir()
}

fn huggingface_snapshot_dirs(repo_root: &FsPath) -> Vec<PathBuf> {
    let snapshots = repo_root.join("snapshots");
    // Rank by materialization (most files first) so the `.into_iter().next()` / `.find()` callers — the
    // training gate/resolver, `model_catalog`, download-state scans — get the FULLEST snapshot and never
    // an empty/torn one; a partial download or a test that clobbered `refs/main` must not win over a
    // fully-materialized sibling (sc-13915, porting the worker's sc-13834 `resolve_huggingface_snapshot_dir`
    // hardening to the rust-api side). Empty dirs are deprioritized, NOT dropped, so iterate-all callers
    // (dir scans) still see every snapshot; their order is irrelevant to them. Deterministic path tiebreak.
    let mut counted: Vec<(usize, PathBuf)> = std::fs::read_dir(&snapshots)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .map(|path| (snapshot_file_count(&path), path))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    counted.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut snapshot_dirs: Vec<PathBuf> = counted.into_iter().map(|(_, path)| path).collect();
    // `refs/main` fronts the list ONLY when the snapshot it names is materialized (holds files); a
    // polluted/empty `refs/main` must not front an empty dir over a full one.
    if let Some(main_snapshot) = huggingface_main_snapshot_dir(repo_root) {
        if snapshot_file_count(&main_snapshot) > 0 {
            snapshot_dirs.retain(|path| path != &main_snapshot);
            snapshot_dirs.insert(0, main_snapshot);
        }
    }
    snapshot_dirs
}

/// Recursively count regular files under `dir` (symlinks counted — HF snapshots symlink into `blobs/`);
/// `0` when the dir is absent, unreadable, or holds only empty subdirectories. Lets
/// [`huggingface_snapshot_dirs`] rank a fully materialized snapshot above an empty/torn placeholder
/// (sc-13915; mirrors the worker's `snapshot_file_count`, sc-13834).
fn snapshot_file_count(dir: &FsPath) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut count = 0;
    for entry in entries.flatten() {
        match entry.file_type() {
            Ok(file_type) if file_type.is_dir() => count += snapshot_file_count(&entry.path()),
            Ok(_) => count += 1,
            Err(_) => {}
        }
    }
    count
}

fn huggingface_main_snapshot_dir(repo_root: &FsPath) -> Option<PathBuf> {
    let revision = std::fs::read_to_string(repo_root.join("refs").join("main")).ok()?;
    let revision = revision.trim();
    if revision.is_empty() {
        return None;
    }
    let snapshot = repo_root.join("snapshots").join(revision);
    snapshot.is_dir().then_some(snapshot)
}

fn unique_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique = Vec::new();
    for path in paths {
        if !unique.iter().any(|item| item == &path) {
            unique.push(path);
        }
    }
    unique
}

/// Result of attempting to remove a batch of SceneWorks-owned artifact paths.
#[derive(Default)]
struct ArtifactRemoval {
    /// Paths successfully moved to the OS trash (or permanently unlinked).
    removed_paths: Vec<String>,
    /// Paths left in place because they are not inside a SceneWorks-owned root
    /// (e.g. a shared Hugging Face cache blob referenced by another model).
    retained_paths: Vec<String>,
    /// Owned paths that could NOT be moved to the OS trash (recycle bin disabled,
    /// unsupported volume, item too large, …). Nothing was deleted for these, so the
    /// caller can prompt the user before falling back to a permanent delete.
    trash_failed_paths: Vec<String>,
}

#[cfg(test)]
static TEST_TRASH_OUTCOMES: std::sync::OnceLock<Mutex<HashMap<PathBuf, bool>>> =
    std::sync::OnceLock::new();

#[cfg(test)]
pub(crate) struct TestTrashOutcomesGuard {
    paths: Vec<PathBuf>,
}

#[cfg(test)]
impl Drop for TestTrashOutcomesGuard {
    fn drop(&mut self) {
        let mut outcomes = TEST_TRASH_OUTCOMES.get_or_init(Default::default).lock();
        for path in &self.paths {
            outcomes.remove(path);
        }
    }
}

/// Install one-shot deterministic OS-trash outcomes for route tests. A `true`
/// outcome removes the path as if the trash move succeeded; `false` leaves it
/// in place and reports the same recoverable failure as `trash::delete`.
#[cfg(test)]
pub(crate) fn test_trash_outcomes(
    entries: impl IntoIterator<Item = (PathBuf, bool)>,
) -> TestTrashOutcomesGuard {
    let mut outcomes = TEST_TRASH_OUTCOMES.get_or_init(Default::default).lock();
    let mut paths = Vec::new();
    for (path, succeeds) in entries {
        assert!(
            outcomes.insert(path.clone(), succeeds).is_none(),
            "test trash outcome already registered for {}",
            path.display()
        );
        paths.push(path);
    }
    TestTrashOutcomesGuard { paths }
}

/// Move a single path to the operating-system trash (Windows Recycle Bin / macOS
/// Trash / Linux XDG trash). `trash::delete` is blocking, so it runs on the blocking
/// pool to avoid stalling the async runtime.
async fn move_path_to_os_trash(path: PathBuf) -> Result<(), String> {
    #[cfg(test)]
    let test_outcome = {
        TEST_TRASH_OUTCOMES
            .get_or_init(Default::default)
            .lock()
            .remove(&path)
    };
    #[cfg(test)]
    if let Some(succeeds) = test_outcome {
        if !succeeds {
            return Err("injected OS-trash failure".to_owned());
        }
        let metadata = tokio::fs::symlink_metadata(&path)
            .await
            .map_err(|error| format!("injected trash could not inspect path: {error}"))?;
        if metadata.is_dir() {
            tokio::fs::remove_dir_all(&path)
                .await
                .map_err(|error| format!("injected trash could not remove directory: {error}"))?;
        } else {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|error| format!("injected trash could not remove file: {error}"))?;
        }
        return Ok(());
    }

    tokio::task::spawn_blocking(move || trash::delete(&path))
        .await
        .map_err(|error| format!("trash task failed: {error}"))?
        .map_err(|error| error.to_string())
}

/// Remove a batch of artifact paths, moving each SceneWorks-owned path to the OS
/// trash unless `permanent` is set (then unlink it). Paths outside the allowed roots
/// are retained. A trash failure is non-fatal: the path is recorded in
/// `trash_failed_paths` so the caller can offer a permanent-delete confirmation.
async fn remove_owned_artifacts(
    paths: Vec<PathBuf>,
    allowed_roots: &[PathBuf],
    permanent: bool,
) -> Result<ArtifactRemoval, ApiError> {
    let mut removal = ArtifactRemoval::default();
    for path in paths {
        remove_owned_artifact_path(path, allowed_roots, permanent, &mut removal).await?;
    }
    Ok(removal)
}

async fn remove_owned_artifact_path(
    path: PathBuf,
    allowed_roots: &[PathBuf],
    permanent: bool,
    removal: &mut ArtifactRemoval,
) -> Result<(), ApiError> {
    let metadata = match tokio::fs::symlink_metadata(&path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(ApiError::internal(format!(
                "Failed to inspect artifact path {}: {error}",
                path.display()
            )))
        }
    };
    let canonical_path = tokio::fs::canonicalize(&path).await.map_err(|error| {
        ApiError::internal(format!(
            "Failed to resolve artifact path {}: {error}",
            path.display()
        ))
    })?;
    let mut owned = false;
    for root in allowed_roots {
        if let Ok(canonical_root) = tokio::fs::canonicalize(root).await {
            if canonical_path.starts_with(&canonical_root) && canonical_path != canonical_root {
                owned = true;
                break;
            }
        }
    }
    if !owned {
        removal.retained_paths.push(path.display().to_string());
        return Ok(());
    }
    if permanent {
        if metadata.is_dir() {
            tokio::fs::remove_dir_all(&path).await.map_err(|error| {
                ApiError::internal(format!(
                    "Failed to remove artifact directory {}: {error}",
                    path.display()
                ))
            })?;
        } else {
            tokio::fs::remove_file(&path).await.map_err(|error| {
                ApiError::internal(format!(
                    "Failed to remove artifact file {}: {error}",
                    path.display()
                ))
            })?;
        }
        removal.removed_paths.push(path.display().to_string());
        return Ok(());
    }
    match move_path_to_os_trash(path.clone()).await {
        Ok(()) => removal.removed_paths.push(path.display().to_string()),
        Err(error) => {
            tracing::warn!(
                event = "artifact_trash_failed",
                path = %path.display(),
                error = %error,
                "Failed to move artifact to the OS trash; awaiting permanent-delete confirmation"
            );
            removal.trash_failed_paths.push(path.display().to_string());
        }
    }
    Ok(())
}

fn requested_gpu_or_auto(value: String) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        "auto".to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn option_str_is_empty(value: Option<&str>) -> bool {
    value.map(str::trim).unwrap_or_default().is_empty()
}

fn number_to_f64(number: &serde_json::Number, field: &'static str) -> Result<f64, ApiError> {
    number
        .as_f64()
        .ok_or_else(|| ApiError::bad_request(format!("Invalid numeric value for {field}")))
}

fn optional_number_to_f64(
    number: Option<&serde_json::Number>,
    field: &'static str,
) -> Result<Option<f64>, ApiError> {
    number.map(|value| number_to_f64(value, field)).transpose()
}

fn validate_timeline_export(payload: &TimelineExportRequest) -> Result<(), ApiError> {
    if ![640, 720, 1024, 1280].contains(&payload.resolution) {
        return Err(ApiError::bad_request(
            "Resolution must be one of 640, 720, 1024, or 1280.",
        ));
    }
    if !(1..=60).contains(&payload.fps) {
        return Err(ApiError::bad_request("FPS must be between 1 and 60"));
    }
    Ok(())
}

fn validate_frame_extract(payload: &FrameExtractRequest) -> Result<(), ApiError> {
    if !payload.playhead_seconds.is_finite() || payload.playhead_seconds < 0.0 {
        return Err(ApiError::bad_request(
            "playheadSeconds must be greater than or equal to 0",
        ));
    }
    if ![
        "reuse",
        "first_frame",
        "last_frame",
        "video_studio",
        "image_studio",
        "bridge",
        "extension",
    ]
    .contains(&payload.intended_use.as_str())
    {
        return Err(ApiError::bad_request("Unsupported intendedUse"));
    }
    Ok(())
}

fn validate_person_detection_job(payload: &PersonDetectionJobRequest) -> Result<(), ApiError> {
    if payload.source_asset_id.is_empty() {
        return Err(ApiError::bad_request("Source clip is required"));
    }
    if payload
        .source_timestamp
        .is_some_and(|timestamp| !timestamp.is_finite() || timestamp < 0.0)
    {
        return Err(ApiError::bad_request(
            "sourceTimestamp must be greater than or equal to 0",
        ));
    }
    Ok(())
}

fn validate_person_track_job(payload: &PersonTrackJobRequest) -> Result<(), ApiError> {
    if payload.source_asset_id.is_empty() {
        return Err(ApiError::bad_request("Source clip is required"));
    }
    if payload.representative_frame_asset_id.is_empty() {
        return Err(ApiError::bad_request(
            "Representative frame asset is required",
        ));
    }
    if payload.track_name.is_empty() || payload.track_name.chars().count() > 120 {
        return Err(ApiError::bad_request(
            "trackName must be between 1 and 120 characters",
        ));
    }
    if !payload.detection.contains_key("id") {
        return Err(ApiError::bad_request(
            "Selected detection metadata is required",
        ));
    }
    Ok(())
}

/// sc-8884 (F-082): `negativePrompt` and the free-form `advanced` bag previously escaped
/// all length validation (only `prompt` was capped), so an oversized field was persisted
/// to jobs.db and re-serialized to every SSE subscriber on each status change. Cap the
/// negative prompt at the same char limit as `prompt` and bound `advanced`'s serialized
/// size. Shared by `validate_image_job` / `validate_video_job`.
fn validate_prompt_extras(negative_prompt: &str, advanced: &JsonObject) -> Result<(), ApiError> {
    if negative_prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(format!(
            "negativePrompt must be at most {MAX_PROMPT_CHARS} characters"
        )));
    }
    // Serialize once to measure the on-the-wire size of the pass-through bag.
    let advanced_bytes = serde_json::to_vec(advanced)
        .map(|bytes| bytes.len())
        .unwrap_or(0);
    if advanced_bytes > MAX_ADVANCED_JSON_BYTES {
        return Err(ApiError::bad_request(format!(
            "advanced settings must serialize to at most {MAX_ADVANCED_JSON_BYTES} bytes"
        )));
    }
    Ok(())
}

const PROMPT_ENHANCE_MAX_TOKENS: u64 = 2048;
const PROMPT_ENHANCE_MAX_TEMPERATURE: f64 = 2.0;
const PROMPT_ENHANCEMENT_FACT_KEY: &str = "promptEnhancement";

/// Validate the typed, bounded part of the FLUX.2 prompt-enhancement request at every image enqueue
/// boundary. `advanced` is otherwise intentionally extensible, but these fields cross into a native
/// LLM sampler and must never inherit the old truthy/coercing behavior.
fn validate_prompt_enhancement_fields(advanced: &JsonObject) -> Result<bool, ApiError> {
    if advanced.contains_key(PROMPT_ENHANCEMENT_FACT_KEY) {
        return Err(ApiError::bad_request(format!(
            "advanced.{PROMPT_ENHANCEMENT_FACT_KEY} is worker-owned and cannot be supplied by a client"
        )));
    }
    let enabled = match advanced.get("enhancePrompt") {
        None => false,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => {
            return Err(ApiError::bad_request(
                "advanced.enhancePrompt must be a boolean",
            ));
        }
    };
    if let Some(value) = advanced.get("enhanceTemperature") {
        let temperature = value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                ApiError::bad_request("advanced.enhanceTemperature must be a finite number")
            })?;
        if !(0.0..=PROMPT_ENHANCE_MAX_TEMPERATURE).contains(&temperature) {
            return Err(ApiError::bad_request(format!(
                "advanced.enhanceTemperature must be between 0 and {PROMPT_ENHANCE_MAX_TEMPERATURE}"
            )));
        }
        if !enabled {
            return Err(ApiError::bad_request(
                "advanced.enhanceTemperature requires advanced.enhancePrompt=true",
            ));
        }
    }
    if let Some(value) = advanced.get("enhanceMaxTokens") {
        let tokens = value
            .as_u64()
            .ok_or_else(|| ApiError::bad_request("advanced.enhanceMaxTokens must be an integer"))?;
        if !(1..=PROMPT_ENHANCE_MAX_TOKENS).contains(&tokens) {
            return Err(ApiError::bad_request(format!(
                "advanced.enhanceMaxTokens must be between 1 and {PROMPT_ENHANCE_MAX_TOKENS}"
            )));
        }
        if !enabled {
            return Err(ApiError::bad_request(
                "advanced.enhanceMaxTokens requires advanced.enhancePrompt=true",
            ));
        }
    }
    Ok(enabled)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn prompt_enhancement_payload_string(payload: &JsonObject, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn prompt_enhancement_payload_references(payload: &JsonObject) -> bool {
    prompt_enhancement_payload_string(payload, "referenceAssetId")
        || payload
            .get("referenceAssetIds")
            .and_then(Value::as_array)
            .is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_str().is_some_and(|value| !value.trim().is_empty()))
            })
}

fn validate_prompt_enhancement_route(payload: &JsonObject) -> Result<(), ApiError> {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    let mode = payload
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("text_to_image");

    #[cfg(target_os = "macos")]
    if !matches!(
        mode,
        "text_to_image" | "edit_image" | "character_image" | "style_variations"
    ) {
        return Err(ApiError::bad_request(format!(
            "advanced.enhancePrompt on MLX does not support image mode {mode}"
        )));
    }

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if !matches!(mode, "text_to_image" | "edit_image") {
        return Err(ApiError::bad_request(format!(
            "advanced.enhancePrompt on Candle supports only text_to_image and edit_image; mode {mode} is unsupported"
        )));
    }

    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        let _ = payload;
        Err(ApiError::bad_request(
            "advanced.enhancePrompt requires a native MLX or Candle image backend",
        ))
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        let has_references = prompt_enhancement_payload_references(payload);
        let has_edit_input =
            has_references || prompt_enhancement_payload_string(payload, "sourceAssetId");
        if mode == "text_to_image" && has_edit_input {
            return Err(ApiError::bad_request(
                "advanced.enhancePrompt text_to_image cannot include source or reference image assets",
            ));
        }
        if mode == "edit_image" && !has_edit_input {
            return Err(ApiError::bad_request(
                "advanced.enhancePrompt edit_image requires a source or reference image asset",
            ));
        }
        if matches!(mode, "character_image" | "style_variations") && !has_references {
            return Err(ApiError::bad_request(format!(
                "advanced.enhancePrompt {mode} requires a reference image asset"
            )));
        }
        Ok(())
    }
}

/// Validate the canonical, post-preset image payload. Enhancement is deliberately scoped to the
/// actual native backend route: MLX owns base/edit/character plus the defensive legacy style alias,
/// while Candle owns only base and its bespoke `edit_image` lane. It is not inherited by Klein,
/// strict control, backendless builds,
/// or a reference-bearing mode that would fall through to a plain base render. Keeping this check
/// on the final payload also covers presets and retry/duplicate's shallow-merged canonical payload.
fn validate_prompt_enhancement_payload(payload: &JsonObject) -> Result<(), ApiError> {
    let empty = JsonObject::new();
    let advanced = payload
        .get("advanced")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    if !validate_prompt_enhancement_fields(advanced)? {
        return Ok(());
    }
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if model != "flux2_dev" {
        return Err(ApiError::bad_request(
            "advanced.enhancePrompt is supported only by FLUX.2-dev; FLUX.2-Klein and other models reject it",
        ));
    }
    let strict_control = advanced
        .get("poses")
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
        || advanced.contains_key("controlWeights")
        || advanced.contains_key("controlImage")
        || advanced.contains_key("controlMode");
    if strict_control {
        return Err(ApiError::bad_request(
            "advanced.enhancePrompt cannot be combined with FLUX.2-dev strict control",
        ));
    }
    validate_prompt_enhancement_route(payload)
}

/// Reject a `model` id that is not a safe single path component (F-003 / sc-11159).
///
/// The id flows verbatim from the untrusted job payload into the worker's asset
/// filename — `write_image_asset` / `write_upscaled_asset` / `VideoPlan::new` each
/// `format!(".._{model}_..")` a project-relative path, then `create_dir_all` +
/// atomic-rename the file into place. A `model` containing `../`, `..\`, or an
/// absolute path would therefore traverse out of the project dir and hand a remote
/// caller an arbitrary-write primitive. The worker now slugifies the id as a last
/// line of defense, but the API is the first gate and rejects such ids outright,
/// mirroring the worker's `safe_weight_filename` posture for payload-supplied
/// filename components.
///
/// SCOPE (sc-11159): this enforces *path-safety only*, NOT catalog membership. The
/// worker's stub lane deliberately serves uncatalogued model ids for dev/testing,
/// and the API test harness (which does not seed the shipped model manifests)
/// routinely enqueues real-but-uncatalogued ids and expects success, so a catalog
/// rejection here would break that legitimate lane. Path-safety alone fully closes
/// the traversal/arbitrary-write vulnerability: an uncatalogued-but-path-safe id
/// can never escape the project dir.
fn validate_model_id(model: &str) -> Result<(), ApiError> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return Err(ApiError::bad_request("model is required"));
    }
    let mut components = std::path::Path::new(trimmed).components();
    let single_normal = matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none();
    if !single_normal || trimmed.contains('/') || trimmed.contains('\\') {
        return Err(ApiError::bad_request(
            "model must be a plain model id (no path separators or '..')",
        ));
    }
    Ok(())
}

fn validate_image_job(payload: &ImageJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    validate_model_id(&payload.model)?;
    if payload.prompt.is_empty() || payload.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    validate_prompt_extras(&payload.negative_prompt, &payload.advanced)?;
    validate_prompt_enhancement_fields(&payload.advanced)?;
    validate_image_pose_count(&payload.advanced)?;
    if payload.loras.len() > sceneworks_core::lora_family::MAX_JOB_LORAS {
        return Err(ApiError::bad_request(format!(
            "loras must contain at most {} entries",
            sceneworks_core::lora_family::MAX_JOB_LORAS
        )));
    }
    if ![
        "text_to_image",
        "edit_image",
        "character_image",
        "style_variations",
    ]
    .contains(&payload.mode.as_str())
    {
        return Err(ApiError::bad_request("Unsupported image mode"));
    }
    // Only a *named* count is bounded here: an omitted one resolves to the model's declared
    // `defaults.count` in `create_image_job`, like the size below.
    if let Some(count) = payload.count {
        if !(1..=8).contains(&count) {
            return Err(ApiError::bad_request("count must be between 1 and 8"));
        }
    }
    // Only a *named* dimension is bounded here: an omitted side is resolved from the model's
    // declared `defaults.resolution` in `create_image_job` (sc-12400), the same shape as the video
    // route's duration/fps/size.
    if let Some(width) = payload.width {
        validate_dimension(width, "width", MAX_IMAGE_DIMENSION)?;
    }
    if let Some(height) = payload.height {
        validate_dimension(height, "height", MAX_IMAGE_DIMENSION)?;
    }
    if payload.upscale.enabled {
        if ![2, 4].contains(&payload.upscale.factor) {
            return Err(ApiError::bad_request("upscale.factor must be 2 or 4"));
        }
        if payload.upscale.engine.trim().is_empty() {
            return Err(ApiError::bad_request("upscale.engine is required"));
        }
    }
    Ok(())
}

/// Bound strict-pose fan-out at the image-job creation boundary. Each `advanced.poses` entry
/// renders one image, so this is an output-count contract rather than a JSON-size guard.
///
/// Existing stored jobs are immutable and remain readable. Retry/duplicate pass their merged
/// payload through this same helper in `jobs.rs`, so a legacy over-limit job must be reduced before
/// it can create new work instead of silently reproducing the old unbounded behavior.
fn validate_image_pose_count(advanced: &JsonObject) -> Result<(), ApiError> {
    let Some(poses) = advanced.get("poses").and_then(Value::as_array) else {
        return Ok(());
    };
    if poses.len() > sceneworks_core::image_request::MAX_JOB_POSES {
        return Err(ApiError::bad_request(format!(
            "advanced.poses must contain at most {} entries; each pose renders one image",
            sceneworks_core::image_request::MAX_JOB_POSES
        )));
    }
    Ok(())
}

fn validate_character_test_job(payload: &CharacterTestRequest) -> Result<(), ApiError> {
    if payload.prompt.is_empty() || payload.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(format!(
            "prompt must be between 1 and {MAX_PROMPT_CHARS} characters"
        )));
    }
    if !(1..=8).contains(&payload.count) {
        return Err(ApiError::bad_request("count must be between 1 and 8"));
    }
    validate_dimension(payload.width, "width", MAX_IMAGE_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_IMAGE_DIMENSION)?;
    Ok(())
}

/// Every `mode` a `POST /api/v1/video/jobs` request may name — the enqueue allow-list, and a
/// SEPARATE reachability gate from the catalog: a mode absent HERE 400s with "Unsupported video
/// mode" no matter what the model's manifest `capabilities` declare, what `VIDEO_UI_MODES` offers,
/// or what the worker can render.
///
/// It is a named `const` rather than the inline array literal it was so a test can enumerate the
/// REAL list. A guard that retypes the modes asserts against its own copy and stays green while
/// this list drifts — the false-green shape that let GH #2074 ship. `every_declared_video_capability_is_submittable`
/// (tests/jobs.rs) reads this constant and the shipped manifest, so a new family's capability that
/// is not admitted here is RED at the source.
///
/// ⚠️ Adding a family means checking SIX surfaces, not one (sc-17159): manifest `capabilities`,
/// [`sceneworks_core::jobs_store::routing`]'s `VIDEO_UI_MODES`, `video_mode_is_mlx_eligible` +
/// `VIDEO_MODEL_CAPS`, the candle claim gate, THIS list plus the per-mode required-asset `match`
/// below it, and the worker's dispatch arm.
pub(crate) const VIDEO_JOB_MODES: &[&str] = &[
    "image_to_video",
    "text_to_video",
    "first_last_frame",
    "extend_clip",
    "video_bridge",
    "replace_person",
    // Bernini editing / reference-driven video modes (sc-4703).
    "video_to_video",
    "reference_to_video",
    "reference_video_to_video",
    // Bernini multi-source-video modes (sc-5425): mv2v (multiple source clips)
    // and ads2v (source video + reference video + reference images).
    "multi_video_to_video",
    "ads2v",
    // SCAIL-2 standalone character animation (sc-5448 / sc-5449, epic 5439): reference
    // character image + driving video → animated clip. It was wired end-to-end — catalog
    // `capabilities`, `VIDEO_UI_MODES`, `video_mode_is_mlx_eligible`, the candle claim gate,
    // the worker's `generate_scail2` — and offered in the Video Studio, but never added
    // HERE, so every submission 400'd on "Unsupported video mode" and the mode was
    // unreachable from the moment it shipped (GH #2074).
    "animate_character",
];

fn validate_video_job(payload: &VideoJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    validate_model_id(&payload.model)?;
    if payload.prompt.is_empty() || payload.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    validate_prompt_extras(&payload.negative_prompt, &payload.advanced)?;
    if payload.loras.len() > sceneworks_core::lora_family::MAX_JOB_LORAS {
        return Err(ApiError::bad_request(format!(
            "loras must contain at most {} entries",
            sceneworks_core::lora_family::MAX_JOB_LORAS
        )));
    }
    if !VIDEO_JOB_MODES.contains(&payload.mode.as_str()) {
        return Err(ApiError::bad_request("Unsupported video mode"));
    }
    if payload
        .reference_asset_ids
        .iter()
        .any(|id| id.trim().is_empty())
    {
        return Err(ApiError::bad_request(
            "referenceAssetIds must not contain blank ids",
        ));
    }
    if payload.reference_asset_ids.len() > MAX_VIDEO_REFERENCE_ASSET_IDS {
        return Err(ApiError::bad_request(format!(
            "referenceAssetIds must contain at most {MAX_VIDEO_REFERENCE_ASSET_IDS} ids"
        )));
    }
    if payload
        .source_clip_asset_ids
        .iter()
        .any(|id| id.trim().is_empty())
    {
        return Err(ApiError::bad_request(
            "sourceClipAssetIds must not contain blank ids",
        ));
    }
    if payload.source_clip_asset_ids.len() > MAX_VIDEO_SOURCE_CLIP_ASSET_IDS {
        return Err(ApiError::bad_request(format!(
            "sourceClipAssetIds must contain at most {MAX_VIDEO_SOURCE_CLIP_ASSET_IDS} ids"
        )));
    }
    // The audio references (sc-17160), rejected blank-first and bounded exactly like the two
    // id lists above — same order, same wording — because "consistent with the existing lists"
    // is the contract a caller reads off one list and applies to the next.
    if payload
        .reference_audio_asset_ids
        .iter()
        .any(|id| id.trim().is_empty())
    {
        return Err(ApiError::bad_request(
            "referenceAudioAssetIds must not contain blank ids",
        ));
    }
    if payload.reference_audio_asset_ids.len() > MAX_VIDEO_REFERENCE_AUDIO_ASSET_IDS {
        return Err(ApiError::bad_request(format!(
            "referenceAudioAssetIds must contain at most {MAX_VIDEO_REFERENCE_AUDIO_ASSET_IDS} ids"
        )));
    }
    // There is deliberately NO combined blanket here, only the three per-list ones above. 12 is
    // MiniMax-H3's number, not a product-wide truth: today's caps admit 8 images AND 8 clips on
    // one request, so a blanket 12 would refuse a 16-file shape this route accepts right now —
    // narrowing every existing video model, which is precisely what the cap change must not do.
    // The combined budget is a per-model fact and lives with the model, as
    // `limits.maxCombinedReferenceAssets` (see `create_video_job`).
    // Only a *named* duration is bounded here, for the same reason as fps below: an omitted one is
    // resolved from the model's declared `defaults.duration` in `create_video_job`. This blanket
    // stays a payload-sanity check; the model's own `limits.hardMaxDuration` is enforced there
    // (sc-12297 / sc-12400).
    if let Some(duration) = payload.duration.as_ref() {
        let duration = duration
            .as_f64()
            .ok_or_else(|| ApiError::bad_request("duration must be a number between 1 and 30"))?;
        if !duration.is_finite() || !(1.0..=30.0).contains(&duration) {
            return Err(ApiError::bad_request("duration must be between 1 and 30"));
        }
    }
    // Only a *named* fps is bounded here: an omitted one is resolved from the model's declared
    // `defaults.fps` in `create_video_job`, which is the first point that knows the model (this
    // runs before preset expansion, so the model here may be stale — sc-12300). This blanket stays
    // a payload-sanity check; the model's own `limits.fps` menu is enforced there (sc-12347).
    if let Some(fps) = payload.fps {
        if !(1..=60).contains(&fps) {
            return Err(ApiError::bad_request("fps must be between 1 and 60"));
        }
    }
    // Only a *named* dimension is bounded here, like duration and fps above: an omitted side is
    // resolved from the model's declared `defaults.resolution` in `create_video_job` (sc-12400).
    if let Some(width) = payload.width {
        validate_dimension(width, "width", MAX_VIDEO_DIMENSION)?;
    }
    if let Some(height) = payload.height {
        validate_dimension(height, "height", MAX_VIDEO_DIMENSION)?;
    }
    match payload.mode.as_str() {
        "image_to_video" if payload.source_asset_id.is_none() => Err(ApiError::bad_request(
            "Image to Video requires a source image.",
        )),
        "first_last_frame"
            if payload.source_asset_id.is_none() || payload.last_frame_asset_id.is_none() =>
        {
            Err(ApiError::bad_request(
                "First/Last Frame requires first and last image assets.",
            ))
        }
        "extend_clip" if payload.source_clip_asset_id.is_none() => {
            Err(ApiError::bad_request("Extend Clip requires a source clip."))
        }
        "video_bridge"
            if payload.source_clip_asset_id.is_none()
                || payload.bridge_right_clip_asset_id.is_none() =>
        {
            Err(ApiError::bad_request(
                "Bridge generation requires left and right source clips.",
            ))
        }
        "replace_person" if payload.source_clip_asset_id.is_none() => Err(ApiError::bad_request(
            "Replace Person requires a source clip.",
        )),
        "replace_person" if payload.person_track_id.is_none() => Err(ApiError::bad_request(
            "Replace Person requires a selected person track.",
        )),
        "replace_person" if payload.character_id.is_none() => Err(ApiError::bad_request(
            "Replace Person requires a Character.",
        )),
        // Bernini editing / reference-driven video modes (sc-4703): each requires its
        // source media so the worker never falls through to an unconditioned t2v render.
        "video_to_video" if payload.source_clip_asset_id.is_none() => Err(ApiError::bad_request(
            "Video to Video requires a source clip.",
        )),
        // `reference_to_video` requires at least one VISUAL reference — an image or a video clip.
        // Audio references are admitted alongside them and never instead of them.
        //
        // Two corrections in one line. sc-17159 widened this from `reference_asset_ids.is_empty()`
        // because Bernini, the only r2v model at the time, takes images alone, so "at least one
        // reference image" and "at least one reference" were the same sentence — true for the
        // clips MiniMax-H3 Ref2VA added, and WRONG for the audio it added at the same time.
        //
        // sc-19574 settled it against the reference implementation rather than by argument.
        // diffusers `MiniMaxH3` (upstream PR #14355, `0.40.0.dev0 @ 7564fb01`) states the rule on
        // `MiniMaxH3AudioReference` — "never on its own — an audio reference has to be paired with
        // at least one image or video reference. It never reaches the conditioner and is encoded by
        // the audio VAE alone" — and ENFORCES it in `before_encoder.py`:
        //
        //     if set(kinds) == {"audio"}:
        //         raise ValueError("An audio reference has to be paired with at least one image or
        //                           video reference and cannot be used on its own.")
        //
        // So the engine was right and this layer was wrong: an audio-only set leaves the visual
        // conditioner with nothing to read, and the render it would produce is unconditioned. The
        // worker refuses it too (sc-19508, `minimax_h3_validate_partition`); refusing it HERE is
        // what makes the user find out at submission instead of after a queued job fails.
        //
        // The rule itself is `sceneworks_core::video_request::classify_reference_set`, which the
        // MCP tool and the worker call too — one predicate, three layers, so they cannot drift back
        // into disagreement. Only the WORDING is this layer's. It does NOT loosen Bernini: its own
        // conditioning assembly (`resolve_bernini_conditioning`, both lanes) still refuses an r2v
        // with no `referenceAssetIds`, naming bernini — the model-specific half of the requirement
        // belongs with the model, exactly like `limits.maxReferenceAssets`.
        "reference_to_video"
            if classify_reference_set(
                payload.reference_asset_ids.len(),
                payload.source_clip_asset_ids.len(),
                payload.reference_audio_asset_ids.len(),
            ) != ReferenceSetVerdict::Conditionable =>
        {
            Err(ApiError::bad_request(
                "Reference to Video requires at least one reference image or video clip. Audio \
                 references condition the soundtrack and cannot be the only reference.",
            ))
        }
        "reference_video_to_video" if payload.source_clip_asset_id.is_none() => Err(
            ApiError::bad_request("Reference + Video requires a source clip."),
        ),
        "reference_video_to_video" if payload.reference_asset_ids.is_empty() => Err(
            ApiError::bad_request("Reference + Video requires at least one reference image."),
        ),
        // Bernini multi-source-video modes (sc-5425): mv2v blends multiple source clips;
        // ads2v edits a source clip using a reference video + reference images. Each
        // requires its full media set so the worker never falls through to an
        // unconditioned render.
        "multi_video_to_video" if payload.source_clip_asset_ids.len() < 2 => Err(
            ApiError::bad_request("Multi-Clip → Video requires at least two source clips."),
        ),
        "ads2v" if payload.source_clip_asset_id.is_none() => Err(ApiError::bad_request(
            "Source + Reference Video requires a source clip.",
        )),
        "ads2v" if payload.reference_clip_asset_id.is_none() => Err(ApiError::bad_request(
            "Source + Reference Video requires a reference video.",
        )),
        "ads2v" if payload.reference_asset_ids.is_empty() => Err(ApiError::bad_request(
            "Source + Reference Video requires at least one reference image.",
        )),
        // SCAIL-2 standalone character animation (sc-5449): the same required-media contract the
        // worker's `resolve_scail2_conditioning` enforces — the character is `referenceAssetIds[0]`
        // (preferred) or the i2v `sourceAssetId`, and the motion comes from `sourceClipAssetId`.
        // Both are hard engine inputs (`Reference` + `ControlClip`), so a missing one is a rejected
        // enqueue rather than a job that fails minutes later inside the worker.
        "animate_character" if payload.source_clip_asset_id.is_none() => Err(
            ApiError::bad_request("Animate Character requires a driving video."),
        ),
        "animate_character"
            if payload.reference_asset_ids.is_empty() && payload.source_asset_id.is_none() =>
        {
            Err(ApiError::bad_request(
                "Animate Character requires a reference character image.",
            ))
        }
        _ => Ok(()),
    }
}

/// Payload-sanity validation for a `POST /api/v1/audio/jobs` request (sc-13404), the audio analogue
/// of [`validate_video_job`]. Bounds the script prompt + the `advanced` bag exactly as the image/
/// video validators do; the model's own advertised audio surface (voices / languages / max
/// duration) is enforced by the shared gen-core validation floor at generate time, so a *named*
/// duration is only bounded to the sane blanket here (the model's `audio.maxDurationSecs` is the
/// real cap the worker applies). Voice/language are not allow-listed here — an unknown value is
/// rejected by the generator's `validate`, which owns the per-model surface.
fn validate_audio_job(payload: &AudioJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    validate_model_id(&payload.model)?;
    // A multi-speaker request (sc-13676) carries the text in its `script`, so the `prompt` may be
    // empty then; a single-voice request still requires a prompt. The prompt length ceiling always
    // applies. This keeps a single-voice request (script: None) byte-for-byte on the original path.
    let has_script = payload
        .script
        .as_deref()
        .is_some_and(|segments| !segments.is_empty());
    if payload.prompt.chars().count() > MAX_PROMPT_CHARS
        || (payload.prompt.trim().is_empty() && !has_script)
    {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters (or provide a multi-speaker script)",
        ));
    }
    // Multi-speaker dialogue script (sc-13676): when present, every segment must carry non-empty text,
    // and the segment text totals plus the speaker/style labels are bounded so a runaway payload can't
    // slip past the prompt-size floor. The model's advertised `audio.maxSpeakers` is the real
    // speaker-count gate the generator's `validate` applies at the gen-core floor (a script sent to a
    // model that does not advertise `supports_multi_speaker` is a typed Unsupported there); this only
    // rejects a structurally malformed script up front.
    if let Some(segments) = payload.script.as_deref() {
        if segments.is_empty() {
            return Err(ApiError::bad_request(
                "script must have at least one segment",
            ));
        }
        let mut total = 0usize;
        for segment in segments {
            if segment.text.trim().is_empty() {
                return Err(ApiError::bad_request(
                    "each script segment must have non-empty text",
                ));
            }
            total += segment.text.chars().count();
            if segment
                .speaker
                .as_deref()
                .is_some_and(|speaker| speaker.chars().count() > 64)
            {
                return Err(ApiError::bad_request(
                    "script segment speaker label must be at most 64 characters",
                ));
            }
            if segment
                .style
                .as_deref()
                .is_some_and(|style| style.chars().count() > 64)
            {
                return Err(ApiError::bad_request(
                    "script segment style must be at most 64 characters",
                ));
            }
        }
        if total > MAX_PROMPT_CHARS {
            return Err(ApiError::bad_request(format!(
                "script text must total at most {MAX_PROMPT_CHARS} characters"
            )));
        }
    }
    // Reuse the shared negative-prompt + `advanced`-size guard. Music models (ACE-Step) can carry a
    // negative prompt when they advertise support; the size guard bounds it exactly like image/video.
    validate_prompt_extras(
        payload.negative_prompt.as_deref().unwrap_or(""),
        &payload.advanced,
    )?;
    if let Some(target) = payload.target_duration_secs {
        if !target.is_finite() || !(0.0..=300.0).contains(&target) {
            return Err(ApiError::bad_request(
                "targetDurationSecs must be between 0 and 300",
            ));
        }
    }
    // Music describe-the-music sub-block (ACE-Step, sc-13410). BPM must be a real positive tempo; the
    // model's own `validate` re-checks it (finite & > 0), so this is a blanket sanity bound. key/lyrics
    // are free-form — bound their length so a runaway payload can't slip past the prompt-size floor.
    if let Some(bpm) = payload.bpm {
        if !bpm.is_finite() || !(0.0..=1000.0).contains(&bpm) || bpm <= 0.0 {
            return Err(ApiError::bad_request("bpm must be between 0 and 1000"));
        }
    }
    if payload
        .musical_key
        .as_deref()
        .is_some_and(|key| key.chars().count() > 64)
    {
        return Err(ApiError::bad_request(
            "musicalKey must be at most 64 characters",
        ));
    }
    if payload
        .lyrics
        .as_deref()
        .is_some_and(|lyrics| lyrics.chars().count() > MAX_PROMPT_CHARS)
    {
        return Err(ApiError::bad_request(format!(
            "lyrics must be at most {MAX_PROMPT_CHARS} characters"
        )));
    }
    // Extend/edit source band (Conditioning::AudioEdit, sc-13410). Source id + edit mode are a pair: one
    // without the other is a malformed edit request. The mode must be a known token (the model's
    // advertised `audio.editModes` is the real gate the generator's `validate` applies — this only
    // rejects a garbage token up front). Region seconds must be finite and well-ordered; strength is a
    // 0..=1 weight.
    validate_audio_edit_fields(payload)?;
    // Diffusion-audio sampling knobs (Sound FX / MOSS-SoundEffect, sc-13409). Bounded to a blanket
    // sane range here — the same "API blanket vs. model's real cap" split the duration uses: the
    // generator's `validate` owns the per-model guidance range (MOSS: 1.0..=20.0) and step ceiling
    // (MOSS: 1000), so this only rejects nonsense (NaN / non-positive / absurdly large) up front.
    if let Some(guidance) = payload.guidance {
        if !guidance.is_finite() || !(0.0..=100.0).contains(&guidance) {
            return Err(ApiError::bad_request(
                "guidance (CFG scale) must be between 0 and 100",
            ));
        }
    }
    if let Some(steps) = payload.steps {
        if !(1..=10_000).contains(&steps) {
            return Err(ApiError::bad_request("steps must be between 1 and 10000"));
        }
    }
    // Voice Clone (sc-13411 C4). The match strength overrides OpenVoice V2's posterior-sampling
    // temperature τ — bounded to a blanket sane 0..=1 range here (the UI-facing knob range); the
    // converter re-checks it (finite, >= 0). `baseModel`, when supplied, must be a well-formed model id
    // (the route additionally asserts it is a `type: "audio"` model). `referenceAudioAssetId` is a plain
    // library asset id resolved (project-scoped) by the worker; no membership check belongs here.
    if let Some(strength) = payload.match_strength {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(ApiError::bad_request(
                "matchStrength must be between 0 and 1",
            ));
        }
    }
    validate_model_id(&payload.base_model)?;
    Ok(())
}

/// The edit operations the audio route accepts up front (the union across audio models — each model's
/// advertised `audio.editModes` is the real gate the generator's `validate` applies). Matches the
/// [`gen_core::AudioEditMode`] variant set.
const AUDIO_EDIT_MODES: &[&str] = &["inpaint", "repaint", "extend", "cover"];

/// Sanity-bound the extend/edit source-band fields on an [`AudioJobRequest`] (Conditioning::AudioEdit,
/// sc-13410). This is the API blanket floor: the source id + edit mode must be a well-formed pair, the
/// mode a known token, the region seconds finite/ordered, and the strength a 0..=1 weight. The
/// per-model gates (mode ∈ advertised `audio.editModes`, region inside the clip, 48 kHz source) belong
/// to the worker/provider, which knows the model surface and the loaded clip.
fn validate_audio_edit_fields(payload: &AudioJobRequest) -> Result<(), ApiError> {
    let has_source = payload
        .source_audio_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    let has_mode = payload
        .edit_mode
        .as_deref()
        .is_some_and(|mode| !mode.trim().is_empty());
    // A source without a mode (or vice versa) is a malformed edit request — one names WHAT to edit, the
    // other HOW. Reject the half-specified pair rather than silently dropping the edit.
    if has_source != has_mode {
        return Err(ApiError::bad_request(
            "an audio edit needs both sourceAudioAssetId and editMode (or neither)",
        ));
    }
    if let Some(mode) = payload
        .edit_mode
        .as_deref()
        .filter(|m| !m.trim().is_empty())
    {
        // Match the worker's case handling (`edit_mode.map(|m| m.to_lowercase())` at deserialize,
        // then `parse_audio_edit_mode`): the token is case-insensitive, so lowercase before the
        // membership check. Otherwise a mixed-case value like "Extend" is 400'd here even though the
        // worker would accept it — the two validation seams must agree.
        if !AUDIO_EDIT_MODES.contains(&mode.to_lowercase().as_str()) {
            return Err(ApiError::bad_request(format!(
                "editMode must be one of {AUDIO_EDIT_MODES:?}"
            )));
        }
    }
    for (field, value) in [
        ("editRegionStartSecs", payload.edit_region_start_secs),
        ("editRegionEndSecs", payload.edit_region_end_secs),
    ] {
        if let Some(secs) = value {
            if !secs.is_finite() || !(0.0..=3600.0).contains(&secs) {
                return Err(ApiError::bad_request(format!(
                    "{field} must be between 0 and 3600"
                )));
            }
        }
    }
    if let (Some(start), Some(end)) = (payload.edit_region_start_secs, payload.edit_region_end_secs)
    {
        if end <= start {
            return Err(ApiError::bad_request(
                "editRegionEndSecs must be greater than editRegionStartSecs",
            ));
        }
    }
    if let Some(strength) = payload.edit_strength {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(ApiError::bad_request(
                "editStrength must be between 0 and 1",
            ));
        }
    }
    Ok(())
}

/// Upper bound for image width/height. A backstop only — per-model resolution is
/// governed by manifest `limits.resolutions` + the UI. Covers SenseNova-U1's
/// largest trained bucket (3456) with headroom; video uses its own lower cap.
const MAX_IMAGE_DIMENSION: u32 = 4096;

/// Upper bound for video width/height — a lower backstop than images, matching
/// the cap enforced when validating a video job request.
const MAX_VIDEO_DIMENSION: u32 = 1920;
// ---------------------------------------------------------------------------
// Reference-media payload-sanity blankets (sc-17160).
//
// These three are the OUTER bound — "no video model in the product accepts more than
// this" — and play exactly the role the `1..=30` duration and `1..=60` fps blankets do
// a few lines up in `validate_video_job`: they run BEFORE the model is known (a recipe
// preset can still replace it — sc-12300), so they cannot be the per-model answer.
//
// The BINDING per-model cap is `sceneworks_core::video_request::reference_limit_error`,
// enforced in `create_video_job` against the resolved manifest entry, and mirrored in the
// worker's `video_preflight`. Its defaults are 8 images / 8 clips / 0 audio / no combined
// ceiling, so raising the image blanket from 8 to 9 here does NOT hand a 9th reference to
// any already-shipped model — bernini and every other family still refuse it, one layer
// down, on `limits.maxReferenceAssets`.
//
// The COMBINED cap has no blanket at all, only a per-model declaration, for the same reason
// in the opposite direction: see the note where the per-list checks end.
// ---------------------------------------------------------------------------

/// 9 — MiniMax-H3 Ref2VA's image-reference ceiling, the largest of any shipped video model
/// (every other family stops at [`sceneworks_core::video_request::DEFAULT_MAX_REFERENCE_ASSETS`]).
const MAX_VIDEO_REFERENCE_ASSET_IDS: usize = 9;
const MAX_VIDEO_SOURCE_CLIP_ASSET_IDS: usize = 8;
/// 3 — Ref2VA's audio-reference ceiling. No model declares more; models that declare
/// nothing take none at all (`DEFAULT_MAX_REFERENCE_AUDIO_ASSETS` is 0).
const MAX_VIDEO_REFERENCE_AUDIO_ASSET_IDS: usize = 3;

/// Validate the exact model/mode spelling and reference array every video creation boundary will
/// persist. The typed submit route and retry/duplicate all call this after preset/patch merging and
/// current manifest resolution, so replay cannot drift into the worker parser's deliberately
/// tolerant behavior (which trims strings and drops blank/non-string list entries for legacy reads).
///
/// This helper is validation-only: it never trims, filters, sorts, or truncates, preserving every
/// accepted id byte-for-byte and in caller order. SCAIL-2 additionally consumes strict ordered
/// Reference/Mask pairs, has six source positions, and may expose multiple characters only once the
/// current server-owned entry says the paired inference descriptor is installed.
pub(crate) fn validate_video_reference_asset_ids_payload(
    payload: &JsonObject,
    model_manifest_entry: &Value,
) -> Result<(), ApiError> {
    let reference_asset_ids = match payload.get("referenceAssetIds") {
        None => &[][..],
        Some(Value::Array(values)) => values.as_slice(),
        Some(_) => {
            return Err(ApiError::bad_request(
                "referenceAssetIds must be an array of string ids",
            ));
        }
    };
    for value in reference_asset_ids {
        let id = value.as_str().ok_or_else(|| {
            ApiError::bad_request("referenceAssetIds must contain only string ids")
        })?;
        if id.trim().is_empty() {
            return Err(ApiError::bad_request(
                "referenceAssetIds must not contain blank ids",
            ));
        }
        if id != id.trim() {
            return Err(ApiError::bad_request(
                "referenceAssetIds must not contain leading or trailing whitespace",
            ));
        }
    }
    if reference_asset_ids.len() > MAX_VIDEO_REFERENCE_ASSET_IDS {
        return Err(ApiError::bad_request(format!(
            "referenceAssetIds must contain at most {MAX_VIDEO_REFERENCE_ASSET_IDS} ids"
        )));
    }

    let model = match payload.get("model") {
        None => "",
        Some(Value::String(model)) if model == model.trim() => model.as_str(),
        Some(Value::String(_)) => {
            return Err(ApiError::bad_request(
                "model must not contain leading or trailing whitespace",
            ));
        }
        Some(_) => return Err(ApiError::bad_request("model must be a string")),
    };
    let mode = match payload.get("mode") {
        None => "",
        Some(Value::String(mode)) if mode == mode.trim() => mode.as_str(),
        Some(Value::String(_)) => {
            return Err(ApiError::bad_request(
                "mode must not contain leading or trailing whitespace",
            ));
        }
        Some(_) => return Err(ApiError::bad_request("mode must be a string")),
    };
    let reference_count = reference_asset_ids.len();
    if model == "scail2_14b"
        && mode == "animate_character"
        && reference_count > sceneworks_core::video_request::MAX_SCAIL2_REFERENCE_CHARACTERS
    {
        return Err(ApiError::bad_request(format!(
            "SCAIL-2 Animate Character supports at most {} reference characters.",
            sceneworks_core::video_request::MAX_SCAIL2_REFERENCE_CHARACTERS
        )));
    }
    if model == "scail2_14b"
        && mode == "animate_character"
        && reference_count > 1
        && !model_manifest_entry
            .get("ui")
            .and_then(|ui| ui.get("scail2MultiReference"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        return Err(ApiError::bad_request(
            "SCAIL-2 Animate Character multi-reference is unavailable until the paired inference descriptor is installed.",
        ));
    }
    Ok(())
}

fn validate_dimension(value: u32, field: &'static str, max: u32) -> Result<(), ApiError> {
    if !(256..=max).contains(&value) {
        return Err(ApiError::bad_request(format!(
            "{field} must be between 256 and {max}"
        )));
    }
    Ok(())
}

fn to_json_object<T: Serialize>(payload: &T) -> Result<JsonObject, ApiError> {
    serde_json::to_value(payload)
        .map_err(|error| ApiError::internal(error.to_string()))?
        .as_object()
        .cloned()
        .ok_or_else(|| ApiError::internal("Serialized payload was not an object"))
}

fn random_image_seeds(count: u32) -> Value {
    Value::Array(
        (0..count)
            .map(|_| {
                let bytes = *Uuid::new_v4().as_bytes();
                Value::Number(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]).into())
            })
            .collect(),
    )
}

fn find_timeline_item<'a>(timeline: &'a Value, item_id: &str) -> Result<&'a Value, ApiError> {
    timeline
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|track| track.get("items").and_then(Value::as_array))
        .flatten()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(item_id))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Timeline item not found".to_owned(),
            context: None,
            code: None,
        })
}

fn source_timestamp_for_item(item: &Value, playhead_seconds: f64) -> Result<f64, ApiError> {
    let timeline_start = required_finite_f64_field(item, "timelineStart")?;
    let timeline_end = required_finite_f64_field(item, "timelineEnd")?;
    let source_in = required_finite_f64_field(item, "sourceIn")?;
    let speed = required_finite_f64_field(item, "speed")?;
    if timeline_end <= timeline_start {
        return Err(ApiError::bad_request(
            "timelineEnd must be greater than timelineStart.",
        ));
    }
    let clamped = playhead_seconds.clamp(timeline_start, timeline_end);
    Ok(source_in + ((clamped - timeline_start) * speed))
}

fn required_string_field<'a>(payload: &'a Value, field: &str) -> Result<&'a str, ApiError> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request(format!("Missing required field: {field}")))
}

fn optional_f64_field(payload: &Value, field: &str) -> Option<f64> {
    payload.get(field).and_then(Value::as_f64)
}

fn required_finite_f64_field(payload: &Value, field: &str) -> Result<f64, ApiError> {
    let value = optional_f64_field(payload, field)
        .ok_or_else(|| ApiError::bad_request(format!("Missing required field: {field}")))?;
    if !value.is_finite() {
        return Err(ApiError::bad_request(format!(
            "Invalid numeric value for {field}"
        )));
    }
    Ok(value)
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn env_path_or(name: &str, default: &FsPath) -> PathBuf {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| default.to_path_buf())
}

#[cfg(test)]
mod tests;
