//! API assembly and compatibility facade. New helper implementations belong to an area owner;
//! `api_lib_does_not_gain_helper_implementations` enforces the existing root boundary.
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
    GenerationMetricsRow, HiresFixRequest, ImageUpscaleRequest, JobCreateRequest, JobSnapshot,
    JobStatus, JobType, JsonObject, PrioritizeJobsRequest, PrioritizeJobsResponse, ProgressRequest,
    QueueSummary, RetryJobRequest, WorkerCapability, WorkerHeartbeatRequest, WorkerRegisterRequest,
    WorkerSnapshot, WorkerStatus, WorkerTerminationRequest,
};
#[cfg(test)]
use sceneworks_core::hf_home::huggingface_hub_cache_dir;
use sceneworks_core::hf_home::huggingface_repo_cache_path;
use sceneworks_core::image_request::{
    default_count as image_default_count, default_resolution as image_default_resolution,
};
use sceneworks_core::jobs_store::{
    candle_supported, mac_capabilities, mac_rust_supported, model_candle_support,
    model_mac_support, video_job_type_for_mode, video_request_is_claimable_by_any_lane, CreateJob,
    DuplicateJob, JobsStore, JobsStoreError, MacCapabilities, ProgressUpdate, RegisterWorker,
    RetryJob, RouteDecision, StaleSweep, UnsupportedReason, WorkerHeartbeat, JOB_STATUSES,
    MAX_LISTED_JOBS,
};
use sceneworks_core::limits::{requested_steps, steps_limit_error};
use sceneworks_core::lora_family::{
    accepted_lora_families, apply_adapter_metadata_to_manifest_entry,
    apply_model_manifest_defaults, canonical_lora_family, detect_lora_family, detect_model_family,
    first_safetensors_path, read_adapter_metadata, read_safetensors_header,
    reconcile_detected_family, validate_minimax_h3_trainer_header, AdapterFileMetadata,
    SafetensorsHeaderError,
};
use sceneworks_core::lora_url::{lora_source_url_file_stem, parse_lora_source_url, LoraUrlError};
use sceneworks_core::project_store::{
    AssetStatusPatch, AssetTagsPatch, CharacterCreateInput, CharacterLookInput,
    CharacterLookUpdateInput, CharacterLoraInput, CharacterLoraUpdateInput,
    CharacterReferenceInput, CharacterReferenceUpdateInput, CharacterUpdateInput,
    FilmReferenceInput, ProjectStore, ProjectStoreError, UploadAsset, KEYPOINT_UPLOADS_CACHE_DIR,
    POSE_UPLOADS_CACHE_DIR,
};
use sceneworks_core::time::now_unix_seconds;
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
    reference_limit_error, requested_auto_duration, requested_ltx25_vae_decoder,
    requested_temporal_upsample_rounds, resolve_duration, resolve_fps, ReferenceSetVerdict,
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
use tokio_util::sync::CancellationToken;
use tower_http::compression::{
    predicate::{Predicate, SizeAbove},
    CompressionLayer, CompressionLevel,
};
use tower_http::cors::{AllowOrigin, CorsLayer};
use uuid::Uuid;

mod auth;
mod file_serving;
mod job_redaction;
mod project_files;
mod router;
mod uploads;
use file_serving::*;
mod validators;
use auth::{access_control, cors_layer, is_authorized, AuthThrottle};
use job_redaction::*;
use project_files::*;
use validators::*;
mod catalog_scan_supervisor;
mod startup;
use startup::{StartupCriticality, StartupMaintenance, StartupPhaseTimer};
mod saved_voices;
use saved_voices::{create_saved_voice, delete_saved_voice, list_saved_voices};
mod yue2_scores;
use yue2_scores::{
    create_yue2_comparison, create_yue2_score_version, edit_yue2_score_version,
    get_yue2_comparison, get_yue2_score_version, get_yue2_transcription,
    get_yue2_transcription_file, inspect_yue2_score, inspect_yue2_score_version,
    list_yue2_comparisons, list_yue2_score_renders, list_yue2_score_versions,
    list_yue2_transcriptions, record_yue2_score_render,
};
mod yue2_jobs;
use yue2_jobs::{
    create_yue2_jobs, delete_license_acknowledgment, get_job_yue2_eligibility,
    get_license_acknowledgment, put_license_acknowledgment,
};
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
    update_asset_tags, write_upload_field_to_temp_file,
};
mod workflows;
use workflows::{get_asset_workflow, inspect_workflow, max_inspect_multipart_body_bytes};
// Test-only crate-root imports: the `tests` module reaches these helpers via
// `super::` (either `use super::{...}` or a fully-qualified `super::fn(...)` call).
// Gating them keeps the non-test build warning-free — they have no non-test
// crate-root consumer.
#[cfg(test)]
use assets::sweep_stale_asset_uploads_before;
mod iris_curricula;
mod iris_task_training;
mod iris_training;
mod training;
use training::{
    batch_rename_training_dataset_items, create_training_dataset,
    create_training_dataset_analysis_job, create_training_dataset_caption_job,
    create_training_dataset_face_analysis_job, create_training_dataset_parquet_import_job,
    create_training_dataset_subject_mask_job, create_training_dataset_upscale_job,
    create_training_job, delete_training_dataset, finalize_training_dataset_parquet_import,
    get_training_dataset, get_training_dataset_readiness, get_training_dataset_subject_masks,
    list_training_datasets, list_training_presets, list_training_targets,
    repoint_training_dataset_items, resolve_control_overlay_output_location,
    resolve_finetune_output_location, resolve_training_output_location,
    set_training_dataset_item_quality_ack, smart_crop_training_dataset_items,
    strip_exif_training_dataset_items, trusted_adapter_files, trusted_base_checkpoint_files,
    update_training_dataset, upload_ltx_prepared_bundle, upload_training_dataset_item,
    upload_training_dataset_subject_mask, validate_lora_id_component,
    write_training_dataset_analysis_embeddings, write_training_dataset_caption_sidecars,
    write_training_dataset_face_embeddings, write_training_dataset_subject_masks,
};
mod generation;
use generation::{
    create_audio_job, create_generation_job_with_binding, create_image_job, create_interleave_job,
    create_vector_job, create_vector_prompt_workflow, create_video_job, create_vqa_job,
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
mod http_boundary;
mod tickets;
use tickets::{create_media_ticket, EventTicketContext, TicketResponse, TicketStore};
mod dto;
use dto::{
    AccessResponse, AssetPurgeQuery, AssetsQuery, AudioJobRequest, CatalogDeleteQuery,
    CharacterCreateRequest, CharacterLookRequest, CharacterLookUpdateRequest, CharacterLoraRequest,
    CharacterLoraUpdateRequest, CharacterReferenceRequest, CharacterReferenceUpdateRequest,
    CharacterTestRequest, CharacterUpdateRequest, CharactersQuery,
    ConditionalComponentsDownloadRequest, CreateEventTicketRequest, DatasetAnalysisJobRequest,
    DatasetEmbeddingsBody, DatasetFaceAnalysisJobRequest, DatasetFaceRecordsBody,
    DatasetImageFixBody, DatasetParquetImportJobRequest, DatasetRepointBody,
    DatasetSubjectMaskJobRequest, DatasetSubjectMasksBody, DatasetUpscaleJobRequest,
    DirectoriesResponse, EventsQuery, FaceLikenessCompareRequest, FrameExtractRequest,
    HealthResponse, HostCapabilitiesResponse, ImageJobRequest, InterleaveJobRequest, JobsQuery,
    LoraCatalogItemQuery, LoraImportRequest, LoraUpdateRequest, LorasQuery, MetricsQuery,
    ModelConvertRequest, ModelDownloadRequest, ModelImportRequest, ModelImportSourceV1,
    OwnershipModeV1, PersonDetectionJobRequest, PersonTrackCorrectionsRequest,
    PersonTrackJobRequest, ProjectCreateRequest, PromptBatchesQuery, PromptRefineRequest,
    QualityAckBody, ReadinessQuery, RecipePresetsQuery, SavedVoiceCreateRequest,
    StartupReadinessResponse, TimelineCreateRequest, TimelineExportRequest, TimelineSaveRequest,
    TrainingCaptionJobRequest, VectorMode, VectorPromptWorkflowRequest, VectorRequest,
    VerifyResponse, VideoJobRequest, VqaJobRequest,
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
    create_conditional_components_download_jobs, create_model_convert_job,
    create_model_download_job, create_model_import_job, delete_model, delete_model_variant,
    list_models, model_catalog, model_is_installed, repair_model_import_job,
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
use styles::{list_appearance, list_styles};
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
// Iris-3B monocular depth (sc-25682): the typed `image_depth` route behind the Image Editor and
// Library depth actions.
mod depth_jobs;
use depth_jobs::create_image_depth_job;
// Iris-3B restoration pre-submission plan preview (sc-25683): the engine's pure planner, no weights.
mod image_restoration;
use image_restoration::plan_image_restoration;
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
// Local filmmaking harness (epic 22708, sc-22710): plan -> jobs -> assets -> timeline -> export.
pub mod film_harness;
mod film_lifecycle;
pub mod film_planner;
mod film_planner_connections;
mod film_planning;
mod film_review;
mod films;
mod openai_planner;
pub(crate) use error::ApiError;
use film_lifecycle::{cancel_film_run, get_film_run_progress, list_film_runs, resume_film_run};
use film_planner_connections::{
    list_film_planner_connections, save_film_planner_connection, test_film_planner_connection,
};
use film_planning::{
    apply_film_planning_candidate, cancel_film_planning, film_planner_availability,
    get_film_planning_operation, parse_film_draft_script, start_film_planning,
};
use film_review::{
    analyze_film_take, decide_film_take, get_film_review, repair_film_take, replace_film_take,
    swap_film_take,
};
use films::{
    add_film_reference, add_film_sound, create_film_draft, create_film_run, export_film_run,
    get_film_draft, get_film_render_options, get_film_run, get_reference_pack, list_film_drafts,
    preflight_film_draft, preview_film_render_options, start_film_run, update_film_draft,
    update_reference_pack,
};
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
#[cfg(test)]
mod test_hooks;
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
// sc-2126: the subject-mask worker POSTs a dataset's masks as base64 PNG in chunks of at most
// 64 MiB of encoded items (`SUBJECT_MASK_POST_CHUNK_BYTES` in the worker's subject_mask_jobs.rs),
// so any dataset size fits; this per-body ceiling leaves the chunk generous headroom.
const MAX_SUBJECT_MASKS_BODY_BYTES: usize = 256 * 1024 * 1024;
// One replacement-mask upload: the store's encoded-mask cap plus multipart framing headroom. The
// store re-checks the exact limit and answers a field-level error past it.
const MAX_SUBJECT_MASK_MULTIPART_BODY_BYTES: usize =
    sceneworks_core::training_subject_masks::SUBJECT_MASK_MAX_UPLOAD_BYTES as usize + 1024 * 1024;
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
// Declared in sceneworks-core (sc-22710) so the pre-dispatch plan validator in
// `sceneworks_core::film_plan` refuses exactly what this route refuses instead of hand-copying
// the number and drifting.
pub(crate) use sceneworks_core::MAX_PROMPT_CHARS;
// sc-8884 (F-082): serialized-size ceiling for the free-form `advanced` object. It is a
// pass-through bag threaded to the worker, so it has no per-key schema — bound its total
// serialized size instead. 64 KiB is generous for legitimate advanced settings.
const MAX_ADVANCED_JSON_BYTES: usize = 64 * 1024;

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
    sceneworks_core::env_util::flag_value("SCENEWORKS_ALLOW_OPEN_BIND", Some(value), false)
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
    let own_manifests =
        sceneworks_core::env_util::flag_value("SCENEWORKS_OWN_MANIFESTS", own_manifests_env, false);
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
    json_decode_error_response(detail)
}

/// The 422 body a failed JSON decode answers with, as ONE definition (sc-24029).
///
/// A handler that has to look at its body BEFORE the typed decode extracts a `Value` and decodes
/// the typed shape itself, so it also has to report a decode failure itself. Without this it would
/// invent a second shape for the same failure, and the same malformed body would be reported two
/// ways depending on which route received it.
pub(crate) fn json_decode_error_response(detail: String) -> Response {
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

/// The HTTP API's Tokio runtime.
///
/// With `SCENEWORKS_RUN_UTILITY_INPROCESS` (the desktop default) the API's runtime threads poll the
/// in-process utility workers' complete dispatch future, the same future the standalone worker
/// gives an explicitly sized thread ([`sceneworks_worker::WORKER_ENTRY_STACK_BYTES`]). Tokio's
/// default worker stack is 2 MiB, and a claimed MiniMax-H3 model import overflowed it, aborting
/// the whole API. Size every runtime thread the same as the worker entry thread.
pub fn api_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(sceneworks_worker::WORKER_ENTRY_STACK_BYTES)
        .build()
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
    let mut worker_settings = sceneworks_worker::Settings::try_from_env()?;
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
        // the time we reach it. ONE deadline bounds the whole pool (sc-26691): a
        // per-handle timeout let N stuck loops hold shutdown for N × grace.
        let deadline = tokio::time::Instant::now() + grace;
        for handle in handles {
            match tokio::time::timeout_at(deadline, handle).await {
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

use sceneworks_core::process::pid_alive;

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

async fn shutdown_signal(
    film_controller_shutdown: film_harness::FilmControllerShutdown,
    api_shutdown: CancellationToken,
) {
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
    // Set intent before Axum starts draining. Film controllers call this API over loopback and can
    // therefore exit as soon as the listener closes; their lease Drop must already distinguish
    // process shutdown from an ordinary handled controller error.
    film_controller_shutdown.request();
    // sc-26691: the in-process utility loops call `run_worker_loop` directly and only hear OS
    // signals on their own. Whatever stopped THIS process — Ctrl-C, SIGTERM or the parent dying
    // (the campaign harness closing the guardian's stdin) — must stop them too, or
    // `InProcessUtilityWorker::shutdown` waits out its whole grace for loops never told to stop.
    sceneworks_worker::request_shutdown();
    // Axum stops accepting connections when this future returns, but waits for existing
    // connections indefinitely. End app-owned long-lived streams immediately and wake the
    // server's finite drain deadline before returning control to Axum.
    api_shutdown.cancel();
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
    sweep_stale_uploads_retaining(data_dir, subdir, cutoff, should_continue, &[])
}

pub(crate) fn sweep_stale_model_uploads(
    data_dir: &FsPath,
    jobs: &JobsStore,
    cutoff: SystemTime,
    should_continue: impl Fn() -> bool,
) -> std::io::Result<usize> {
    // A database failure must never turn an owned retry input into abandoned cache data.
    let root = data_dir.join("cache/model-uploads");
    let canonical_root = match root.canonicalize() {
        Ok(root) => root,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut retained = Vec::new();
    for source in jobs
        .model_upload_sources_for_recovery()
        .map_err(|error| std::io::Error::other(error.to_string()))?
    {
        match PathBuf::from(source).canonicalize() {
            Ok(source) if source.starts_with(&canonical_root) => retained.push(source),
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    sweep_stale_uploads_retaining(
        data_dir,
        "model-uploads",
        cutoff,
        should_continue,
        &retained,
    )
}

fn sweep_stale_uploads_retaining(
    data_dir: &FsPath,
    subdir: &str,
    cutoff: SystemTime,
    should_continue: impl Fn() -> bool,
    retained: &[PathBuf],
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
        if entry
            .path()
            .canonicalize()
            .is_ok_and(|path| retained.iter().any(|source| source.starts_with(&path)))
        {
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
/// request fails or is unexpectedly slow. Worker heartbeats are the same kind of liveness traffic,
/// and a successful one is never logged at all; only failures surface. All other API/MCP routes
/// retain per-request timing logs.
const SLOW_CLAIM_REQUEST_MS: f64 = 1_000.0;

fn should_log_api_request_duration(route: &str, status: StatusCode, elapsed_ms: f64) -> bool {
    if !status.is_success() {
        return true;
    }
    match route {
        "/api/v1/workers/:worker_id/heartbeat" => false,
        "/api/v1/jobs/claim" => elapsed_ms >= SLOW_CLAIM_REQUEST_MS,
        _ => true,
    }
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
    create_app_with_state_mode(
        settings,
        false,
        film_harness::FilmControllerShutdown::default(),
        CancellationToken::new(),
    )
}

#[cfg(test)]
pub(crate) fn create_app_with_deferred_startup_maintenance(
    settings: Settings,
    timeout: Duration,
) -> Result<(Router, AppState), JobsStoreError> {
    let (router, state) = create_app_with_pending_startup_maintenance(settings)?;
    state.startup_maintenance.start(
        state.settings.data_dir.clone(),
        state.jobs_store.clone(),
        timeout,
    );
    Ok((router, state))
}

#[cfg(test)]
pub(crate) fn create_app_with_pending_startup_maintenance(
    settings: Settings,
) -> Result<(Router, AppState), JobsStoreError> {
    create_app_with_state_mode(
        settings,
        true,
        film_harness::FilmControllerShutdown::default(),
        CancellationToken::new(),
    )
}

pub(crate) fn create_app_with_pending_startup_maintenance_with_shutdowns(
    settings: Settings,
    film_controller_shutdown: film_harness::FilmControllerShutdown,
    api_shutdown: CancellationToken,
) -> Result<(Router, AppState), JobsStoreError> {
    create_app_with_state_mode(settings, true, film_controller_shutdown, api_shutdown)
}

fn create_app_with_state_mode(
    settings: Settings,
    defer_upload_sweeps: bool,
    film_controller_shutdown: film_harness::FilmControllerShutdown,
    api_shutdown: CancellationToken,
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
        let jobs_store = Arc::new(
            JobsStore::new(&settings.jobs_db_path).with_session_started_at(now_unix_seconds()),
        );
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
    if !defer_upload_sweeps {
        warn_on_sweep_err(
            "model",
            sweep_stale_model_uploads(
                &settings.data_dir,
                &jobs_store,
                SystemTime::now() - Duration::from_secs(STALE_UPLOAD_SECONDS),
                || true,
            ),
        );
    }
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
    // The configured self-call URL can name a stable LAN/proxy hostname even
    // when the listener itself binds all interfaces.
    let mut configured_hosts = settings.mcp_allowed_hosts_extra.clone();
    if let Ok(url) = settings.mcp_api_url.parse::<axum::http::Uri>() {
        if let Some(authority) = url.authority() {
            configured_hosts.push(authority.as_str().to_owned());
        }
    }
    let allowed_hosts = Arc::new(sceneworks_mcp::discovered_allowed_hosts(
        &settings.host,
        settings.port,
        &configured_hosts,
    ));
    let state = AppState {
        allowed_hosts,
        settings,
        jobs_store,
        project_store,
        events: Arc::new(EventHub::default()),
        queue_snapshot_lock: Arc::new(AsyncMutex::new(())),
        event_tickets: Arc::new(TicketStore::with_max_outstanding(
            EVENT_TICKET_TTL_SECONDS,
            MAX_OUTSTANDING_EVENT_TICKETS,
        )),
        media_tickets: Arc::new(TicketStore::media(MEDIA_TICKET_TTL_SECONDS, 24 * 60 * 60)),
        thumbnail_generation_slots: Arc::new(tokio::sync::Semaphore::new(2)),
        workflow_strip_slots: Arc::new(tokio::sync::Semaphore::new(WORKFLOW_STRIP_SLOTS)),
        auth_throttle: Arc::new(AuthThrottle::default()),
        resolved_cache_session: Arc::new(AsyncMutex::new(None)),
        manifest_cache: Arc::new(Mutex::new(ManifestCache::default())),
        manifest_write_locks: Arc::new(Mutex::new(HashMap::new())),
        model_catalog_cache: Arc::new(ModelCatalogCache::default()),
        model_size_cache: Arc::new(Mutex::new(ModelSizeCache::default())),
        #[cfg(test)]
        test_hooks: Arc::default(),

        external_lora_cache: Arc::new(Mutex::new(external_loras::ExternalLoraCache::default())),
        external_base_model_cache: Arc::new(Mutex::new(
            external_base_models::ExternalBaseModelCache::default(),
        )),
        http_client: http_boundary::outbound_client()
            .map_err(|error| JobsStoreError::Io(std::io::Error::other(error)))?,
        planner_http_client: http_boundary::planner_client()
            .map_err(|error| JobsStoreError::Io(std::io::Error::other(error)))?,
        interrupted_jobs_on_startup,
        startup_maintenance: if defer_upload_sweeps {
            StartupMaintenance::pending()
        } else {
            StartupMaintenance::complete()
        },
        progress_side_effects_lock: Arc::new(AsyncMutex::new(())),
        catalog_scan_supervisor: Arc::new(catalog_scan_supervisor::CatalogScanSupervisor::default()),
        film_controller_shutdown,
        api_shutdown,
        catalog_scan_invalid_recovery_reported: Arc::new(AsyncMutex::new(
            std::collections::HashSet::new(),
        )),
        catalog_scan_preflight_slots: Arc::new(tokio::sync::Semaphore::new(2)),
        catalog_scan_work_slots: Arc::new(tokio::sync::Semaphore::new(2)),
    };
    let cors = cors_layer(&state.settings);
    let returned_state = state.clone();
    generation::spawn_vector_prompt_workflow_recovery(state.clone());

    // MCP server (epic 10231, sc-10233): the rmcp streamable-HTTP service is
    // nested at `/mcp` INSIDE this router, so the `access_control` layer below
    // gates it exactly like every `/api/v1` route (`requires_token` includes
    // `/mcp`) — token header, loopback trust, and the brute-force throttle all
    // apply unchanged. Its tools call back into this API over plain HTTP
    // (`settings.mcp_api_url`, i.e. `SCENEWORKS_API_URL` or our own loopback
    // port) carrying the access token, so there is no second engine/DB path.
    // Blocking-job wait policy comes from Settings (sc-10277: SCENEWORKS_MCP_JOB_*
    // env knobs), clamped to the invariants the poll loop needs.
    // Both API and MCP use the same concrete Host allow-list, including LAN interfaces.
    let mcp_allowed_hosts = state.allowed_hosts.as_ref().clone();
    let mcp_service = sceneworks_mcp::streamable_http_service_with_origins(
        sceneworks_mcp::ApiClientConfig {
            base_url: state.settings.mcp_api_url.clone(),
            access_token: Some(state.settings.access_token.clone()),
        },
        sceneworks_mcp::JobWaitConfig::clamped(
            state.settings.mcp_job_poll_interval,
            state.settings.mcp_job_timeout,
        ),
        mcp_allowed_hosts,
        state.settings.cors_origins.clone(),
    );

    let router = Router::new()
        .nest_service("/mcp", mcp_service)
        .merge(router::routes())
        .fallback(app_fallback)
        .with_state(state.clone())
        // sc-8812 (F-010): small router-wide default so JSON routes can't buffer
        // multi-GiB bodies. Multipart/upload routes re-attach the large limit per
        // route (asset import, training uploads, pose/keypoint sources, and the
        // model/lora import routes above).
        .layer(DefaultBodyLimit::max(MAX_JSON_BODY_BYTES))
        .layer(middleware::from_fn(
            http_boundary::request_body_idle_timeout,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            access_control,
        ))
        .layer(cors)
        .layer(middleware::from_fn_with_state(state, auth::browser_control))
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
        .layer(middleware::from_fn(api_request_timing))
        .layer(tower::limit::GlobalConcurrencyLimitLayer::new(
            http_boundary::MAX_CONCURRENT_REQUESTS,
        ))
        .layer(middleware::from_fn(http_boundary::no_referrer));
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

/// Embedded production web bundle (apps/web/dist), compiled in only under the
/// `embed-web` feature so default/server/test builds need no web build.
#[cfg(feature = "embed-web")]
mod web_assets;

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

/// Compact replay handle projected ONLY from S11's validated immutable plan provenance. The
/// checkpoint id selects its persisted record; a display/catalog model id alone is not identity.
fn imported_reference_from_provenance(
    entry: &JsonObject,
    provenance: Option<&Value>,
) -> Option<Value> {
    let checkpoint_id = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(entry)?;
    let provenance = provenance?;
    let mut reference = json!({
        "modelId": provenance.get("modelId")?,
        "checkpointId": checkpoint_id,
        "planId": provenance.pointer("/plan/planId")?,
        "semanticDigest": provenance.pointer("/plan/semanticDigest")?,
        "sourceBindingIdentity": provenance.pointer("/plan/sourceBindingIdentity")?,
    });
    for (key, path) in [
        ("family", "/family"),
        ("baseModel", "/baseLineage/baseModel"),
        ("declaredPartition", "/declaredSemantics/minimaxH3Partition"),
    ] {
        if let Some(value) = provenance.pointer(path) {
            reference[key] = value.clone();
        }
    }
    Some(reference)
}

fn validate_expected_imported_model_reference(
    payload: &JsonObject,
    entry: &JsonObject,
    provenance: Option<&Value>,
) -> Result<(), ApiError> {
    let Some(expected) = payload
        .get("expectedImportedModelRef")
        .filter(|value| !value.is_null())
    else {
        return Ok(());
    };
    let actual = imported_reference_from_provenance(entry, provenance);
    sceneworks_core::checkpoint_import::validate_replay_import_reference(expected, actual.as_ref())
        .map_err(ApiError::bad_request)
}

/// Called immediately after canonical model resolution as well as at atomic enqueue. Never
/// resolve a caller's modelManifestEntry or silently replay using today's different checkpoint.
pub(crate) fn ensure_expected_imported_model_reference(
    state: &AppState,
    payload: &JsonObject,
    entry: &Value,
) -> Result<(), ApiError> {
    if payload
        .get("expectedImportedModelRef")
        .is_none_or(Value::is_null)
    {
        return Ok(());
    }
    let empty = JsonObject::new();
    let entry = entry.as_object().unwrap_or(&empty);
    let provenance = sceneworks_core::model_artifacts::provenance::resolved_model_provenance(
        &state.settings.data_dir, entry,
    ).map_err(|error| ApiError::bad_request(format!(
        "Cannot verify imported replay identity: {error}. Reimport or relink the original checkpoint; no replacement weights were queued."
    )))?;
    validate_expected_imported_model_reference(payload, entry, provenance.as_ref())
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
    payload: JsonObject,
    requested_gpu: String,
    initial_status: Option<JobStatus>,
) -> Result<JobSnapshot, ApiError> {
    create_generation_job_with_binding(
        state,
        job_type,
        project_id,
        project_name,
        payload,
        requested_gpu,
        (initial_status, None),
    )
    .await
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

fn model_lora_families(model: &Value) -> Vec<String> {
    model
        .as_object()
        .map(sceneworks_core::lora_family::declared_model_lora_families)
        .unwrap_or_default()
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
    model
        .as_object()
        .map(sceneworks_core::lora_family::accepted_model_lora_families)
        .unwrap_or_default()
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
        // Exact model-id allowlist. Keep both accepted spellings so the API compatibility gate and
        // the worker see the same conjunctive constraint after catalog hydration.
        "modelIds": preferred_lora_value(selected_lora, lora, "modelIds"),
        "model_ids": preferred_lora_value(selected_lora, lora, "model_ids"),
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

use sceneworks_core::payload_util::json_size_to_u64;

use sceneworks_core::model_artifacts::artifact_selection::allow_pattern_matches;

use sceneworks_core::model_artifacts::artifact_selection::pattern_matches;

use sceneworks_core::display_util::quote_path as quote_huggingface_repo;

use sceneworks_core::display_util::format_bytes;

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

use sceneworks_core::model_artifacts::artifact_selection::safe_download_dir;

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

fn validate_import_source_path(
    source_path: &str,
    allowed_roots: &[PathBuf],
) -> Result<(), ApiError> {
    let rejected = || ApiError::bad_request("sourcePath must be inside an approved folder");
    let source = FsPath::new(source_path);
    if !source.is_absolute() {
        return Err(rejected());
    }
    let source = std::fs::canonicalize(source).map_err(|_| rejected())?;
    let metadata = std::fs::metadata(&source).map_err(|_| rejected())?;
    if !metadata.is_file() && !metadata.is_dir() {
        return Err(rejected());
    }
    if allowed_roots
        .iter()
        .filter_map(|root| std::fs::canonicalize(root).ok())
        .any(|root| source.starts_with(root))
    {
        return Ok(());
    }
    Err(rejected())
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
    sceneworks_core::slug::slugify_with(value, "lora", None, '_')
}

use sceneworks_core::time::utc_now as now_rfc3339;

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

use sceneworks_core::env_util::env_string;

use sceneworks_core::env_util::env_path_or;

#[cfg(test)]
mod tests;

#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use test_hooks::{
    test_lora_catalog_builds, test_model_catalog_builds, test_note_lora_catalog_build,
    test_note_model_catalog_build, test_reset_catalog_build_counters, test_trash_outcomes,
    TestTrashOutcomesGuard,
};
#[cfg(test)]
use test_hooks::{
    TEST_MAX_LORA_UPLOAD_BYTES, TEST_MAX_MODEL_UPLOAD_BYTES, TEST_MAX_WORKFLOW_INSPECT_BYTES,
    TEST_TRASH_OUTCOMES,
};
