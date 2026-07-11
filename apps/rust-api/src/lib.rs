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
    DefaultBodyLimit, FromRequest, Multipart, Path, Query, Request as AxumRequest, State,
};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post, put};
use axum::{Json, Router};
use futures_util::future::join_all;
use parking_lot::Mutex;
use sceneworks_core::contracts::{
    ClaimRequest, ClaimResponse, ContractNumber, DuplicateJobRequest, GenerationMetrics,
    GenerationMetricsRow, ImageUpscaleRequest, JobCreateRequest, JobSnapshot, JobStatus, JobType,
    JsonObject, ProgressRequest, QueueSummary, RetryJobRequest, WorkerCapability,
    WorkerHeartbeatRequest, WorkerRegisterRequest, WorkerSnapshot, WorkerStatus,
    WorkerTerminationRequest,
};
use sceneworks_core::hf_home::{huggingface_hub_cache_dir, huggingface_repo_cache_path};
use sceneworks_core::jobs_store::{
    candle_supported, mac_capabilities, mac_rust_supported, model_mac_support, CreateJob,
    DuplicateJob, JobsStore, JobsStoreError, MacCapabilities, ProgressUpdate, RegisterWorker,
    RetryJob, RouteDecision, StaleSweep, UnsupportedReason, WorkerHeartbeat, JOB_STATUSES,
};
use sceneworks_core::lora_family::{
    apply_model_manifest_defaults, canonical_lora_family, detect_lora_family, detect_model_family,
    first_safetensors_path, read_safetensors_header, reconcile_detected_family,
    SafetensorsHeaderError,
};
use sceneworks_core::lora_url::{lora_source_url_file_stem, parse_lora_source_url, LoraUrlError};
use sceneworks_core::project_store::{
    AssetStatusPatch, AssetTagsPatch, CharacterCreateInput, CharacterLookInput,
    CharacterLookUpdateInput, CharacterLoraInput, CharacterLoraUpdateInput,
    CharacterReferenceInput, CharacterReferenceUpdateInput, CharacterUpdateInput, ProjectStore,
    ProjectStoreError, UploadAsset,
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
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio::time::{Instant as TokioInstant, MissedTickBehavior};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tokio_util::io::ReaderStream;
use tower_http::cors::{AllowOrigin, CorsLayer};
use uuid::Uuid;

mod auth;
use auth::{access_control, cors_layer, is_authorized, AuthThrottle};
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
mod assets;
use assets::{
    delete_asset, get_asset, import_asset, list_assets, move_asset_to_character,
    move_asset_to_library, purge_asset, sweep_stale_asset_uploads, update_asset_status,
    update_asset_tags, write_upload_field_to_dir, write_upload_field_to_temp_file,
};
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
    create_training_dataset_face_analysis_job, create_training_dataset_upscale_job,
    create_training_job, delete_training_dataset, get_training_dataset,
    get_training_dataset_readiness, list_training_datasets, list_training_presets,
    list_training_targets, repoint_training_dataset_items, resolve_control_overlay_output_location,
    resolve_training_output_location, set_training_dataset_item_quality_ack,
    smart_crop_training_dataset_items, strip_exif_training_dataset_items, trusted_adapter_files,
    update_training_dataset, upload_training_dataset_item, validate_lora_id_component,
    write_training_dataset_analysis_embeddings, write_training_dataset_caption_sidecars,
    write_training_dataset_face_embeddings,
};
mod generation;
use generation::{
    create_image_job, create_interleave_job, create_video_job, create_vqa_job,
    parse_recipe_preset_resolution, JobCatalogSnapshot,
};
#[cfg(test)]
use generation::{validate_interleave_job, validate_vqa_job};
mod ideogram;
mod jobs;
use jobs::{
    cancel_job, claim_job, create_job, duplicate_job, get_job, get_job_metrics, list_jobs,
    list_metrics, retry_job, update_job_progress, upsert_job_metrics,
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
use tickets::{create_media_ticket, TicketResponse, TicketStore};
mod dto;
use dto::{
    AccessResponse, AssetPurgeQuery, AssetsQuery, CatalogDeleteQuery, CharacterCreateRequest,
    CharacterLookRequest, CharacterLookUpdateRequest, CharacterLoraRequest,
    CharacterLoraUpdateRequest, CharacterReferenceRequest, CharacterReferenceUpdateRequest,
    CharacterTestRequest, CharacterUpdateRequest, CharactersQuery, DatasetAnalysisJobRequest,
    DatasetEmbeddingsBody, DatasetFaceAnalysisJobRequest, DatasetFaceRecordsBody,
    DatasetImageFixBody, DatasetRepointBody, DatasetUpscaleJobRequest, DirectoriesResponse,
    EventsQuery, FaceLikenessCompareRequest, FrameExtractRequest, HealthResponse,
    HostCapabilitiesResponse, ImageJobRequest, InterleaveJobRequest, JobsQuery,
    LoraCatalogItemQuery, LoraImportRequest, LoraUpdateRequest, LorasQuery, MetricsQuery,
    ModelConvertRequest, ModelDownloadRequest, ModelImportRequest, PersonDetectionJobRequest,
    PersonTrackCorrectionsRequest, PersonTrackJobRequest, ProjectCreateRequest, PromptBatchesQuery,
    PromptRefineRequest, QualityAckBody, ReadinessQuery, RecipePresetsQuery, TimelineCreateRequest,
    TimelineExportRequest, TimelineSaveRequest, TrainingCaptionJobRequest, VerifyResponse,
    VideoJobRequest, VqaJobRequest,
};
mod manifest;
use manifest::{
    acquire_manifest_file_lock, load_manifest_entries, manifest_write_lock, merge_entries_by_id,
    merge_object, mutate_manifest_entries, remove_catalog_manifest_entry, write_manifest_atomic,
    ManifestCache,
};
#[cfg(test)]
use manifest::{strip_jsonc_comments, API_MANAGED_MANIFEST_HEADER};
mod models;
use models::{
    create_model_convert_job, create_model_download_job, create_model_import_job, delete_model,
    list_models, model_catalog, model_is_installed, resolve_model_manifest_entry, ModelSizeCache,
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
mod recipe_presets;
use recipe_presets::{
    create_recipe_preset, delete_recipe_preset, duplicate_recipe_preset, get_recipe_preset,
    list_recipe_presets, preset_lora_id, preset_lora_weight, preset_prompt, recipe_preset_catalog,
    recipe_preset_catalog_with, recipe_preset_loras, serialize_preset_lora,
    stamp_recipe_preset_used, update_recipe_preset,
};
mod prompt_batches;
use prompt_batches::{
    create_prompt_batch, delete_prompt_batch, duplicate_prompt_batch, get_prompt_batch,
    list_prompt_batches, update_prompt_batch,
};
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
const MAX_UPLOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_MODEL_UPLOAD_BYTES: usize = 256 * 1024 * 1024 * 1024;
const MAX_LORA_MULTIPART_BODY_BYTES: usize = MAX_UPLOAD_BYTES + 16 * 1024 * 1024;
const MAX_MODEL_MULTIPART_BODY_BYTES: usize = MAX_MODEL_UPLOAD_BYTES + 16 * 1024 * 1024;
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

/// Choose the builtin-manifest seed mode from the raw `SCENEWORKS_CONFIG_DIR` value (sc-10212).
///
/// An explicit, non-empty override marks an operator-owned config dir — a repo checkout or a Compose
/// bind mount — which must stay authoritative, so seed `IfMissing` (fill gaps, never clobber an edited
/// copy or dirty a checked-out `config/`). Unset or blank means `config_dir` fell back to the
/// platform-default app-owned dir (the same one the desktop seeds `Overwrite`), so `Overwrite` there
/// refreshes the builtin catalog on launch instead of serving a stale seed after an upgrade — the
/// sc-10193 img2img flag was invisible on a directly-launched API because the months-old seed was
/// never rewritten. Pure so the choice is unit-tested without touching process env or the filesystem.
///
/// The trim/non-empty rule mirrors [`env_path_or`] exactly, so the seed mode and the resolved
/// `config_dir` always agree on whether the override was actually applied.
fn seed_mode_for_config_dir(
    config_dir_env: Option<&str>,
) -> sceneworks_core::builtin_manifests::SeedMode {
    use sceneworks_core::builtin_manifests::SeedMode;
    match config_dir_env.map(str::trim) {
        Some(value) if !value.is_empty() => SeedMode::IfMissing,
        _ => SeedMode::Overwrite,
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
/// Apple-Silicon Metal preflight (sc-8411). Dispatched from `main` when
/// `SCENEWORKS_GPU_CHECK=1`: a one-shot probe that the desktop spawns at startup —
/// the macOS counterpart of the Windows `nvidia-smi` `cuda_preflight`. Reuses this
/// binary because it already links MLX (the desktop crate does not), and runs the
/// probe in the SAME process/spawn context the real worker uses, so it faithfully
/// predicts whether the worker can acquire a Metal GPU. `Ok(())` when usable;
/// `Err(message)` is the user-facing reason the desktop relays onto the setup screen.
pub fn gpu_check() -> Result<(), String> {
    sceneworks_worker::metal_preflight()
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
fn spawn_inprocess_utility_worker(port: u16) -> InProcessUtilityWorker {
    let mut worker_settings = sceneworks_worker::Settings::from_env();
    worker_settings.api_url = format!("http://127.0.0.1:{port}");
    worker_settings.gpu_id =
        inprocess_worker_gpu_id(std::env::var("SCENEWORKS_RUST_WORKER_GPU_ID").ok());
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
    InProcessUtilityWorker { handles, grace }
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
    let upload_root = data_dir.join("cache").join(subdir);
    let entries = match std::fs::read_dir(upload_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut removed = 0usize;
    for entry in entries {
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
        let is_dir = match entry.file_type() {
            Ok(file_type) => file_type.is_dir(),
            Err(error) => {
                tracing::warn!(
                    event = "stale_upload_stat_failed",
                    sweep = subdir,
                    path = %entry.path().display(),
                    error = %error,
                    "could not stat a stale-upload entry; skipping it"
                );
                continue;
            }
        };
        let modified = match entry.metadata() {
            Ok(metadata) => metadata.modified().unwrap_or(UNIX_EPOCH),
            Err(error) => {
                tracing::warn!(
                    event = "stale_upload_stat_failed",
                    sweep = subdir,
                    path = %entry.path().display(),
                    error = %error,
                    "could not read a stale-upload entry's mtime; skipping it"
                );
                continue;
            }
        };
        if modified <= cutoff {
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
                        path = %path.display(),
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

pub fn create_app(settings: Settings) -> Result<Router, JobsStoreError> {
    Ok(create_app_with_state(settings)?.0)
}

// Like create_app but also returns a clone of the AppState (the same Arc-shared
// stores + event hub the router uses), so tests can subscribe to the event hub and
// assert on what the handlers publish (sc-4203).
pub(crate) fn create_app_with_state(
    settings: Settings,
) -> Result<(Router, AppState), JobsStoreError> {
    // sc-8882 (F-080): a permissions/disk failure here is otherwise invisible until a
    // downstream 500 — surface it as a warning so it is diagnosable. Non-fatal: startup
    // continues (a missing dir surfaces later where it is actually used).
    warn_on_startup_err(
        "data_dir",
        &settings.data_dir,
        std::fs::create_dir_all(&settings.data_dir),
    );
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
    // sc-8882 (F-080): a failed sweep leaves leaked multi-GB upload temps unreclaimed
    // and was previously silent. WARN (never fatal) so the operator can investigate.
    warn_on_sweep_err("lora", sweep_stale_lora_uploads(&settings.data_dir));
    warn_on_sweep_err("pose", sweep_stale_pose_uploads(&settings.data_dir));
    warn_on_sweep_err("keypoint", sweep_stale_keypoint_uploads(&settings.data_dir));
    // sc-4204 (F-API-6): asset-import temp files (cache/uploads) had no startup sweep.
    warn_on_sweep_err("asset", sweep_stale_asset_uploads(&settings.data_dir));
    let jobs_store = Arc::new(JobsStore::new(&settings.jobs_db_path));
    jobs_store.initialize()?;
    let interrupted_jobs_on_startup = jobs_store.mark_interrupted_on_startup()?.len();
    let project_store = Arc::new(ProjectStore::new(
        settings.data_dir.clone(),
        settings.app_version.clone(),
    ));
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
    // Startup data-integrity pass: drop index rows for assets whose media was
    // purged from disk but whose row/sidecar lingered, so the Library never fetches
    // a file that 404s on every open (the source of the app-startup 404 log spam).
    // Runs before the server binds, so the first `list_assets` is already clean.
    // Best-effort and non-fatal — a failure just leaves the stale rows for next
    // startup; the sidecars are untouched, so nothing is lost.
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
    let state = AppState {
        settings,
        jobs_store,
        project_store,
        events: Arc::new(EventHub::default()),
        event_tickets: Arc::new(TicketStore::new(EVENT_TICKET_TTL_SECONDS)),
        media_tickets: Arc::new(TicketStore::new(MEDIA_TICKET_TTL_SECONDS)),
        auth_throttle: Arc::new(AuthThrottle::default()),
        manifest_cache: Arc::new(Mutex::new(ManifestCache::default())),
        manifest_write_locks: Arc::new(Mutex::new(HashMap::new())),
        model_size_cache: Arc::new(Mutex::new(ModelSizeCache::default())),
        external_lora_cache: Arc::new(Mutex::new(external_loras::ExternalLoraCache::default())),
        external_base_model_cache: Arc::new(Mutex::new(
            external_base_models::ExternalBaseModelCache::default(),
        )),
        http_client: reqwest::Client::new(),
        interrupted_jobs_on_startup,
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
    let mcp_service = sceneworks_mcp::streamable_http_service_with(
        sceneworks_mcp::ApiClientConfig {
            base_url: state.settings.mcp_api_url.clone(),
            access_token: Some(state.settings.access_token.clone()),
        },
        sceneworks_mcp::JobWaitConfig::clamped(
            state.settings.mcp_job_poll_interval,
            state.settings.mcp_job_timeout,
        ),
    );

    let router = Router::new()
        .nest_service("/mcp", mcp_service)
        .route("/api/v1/health", get(health))
        .route("/api/v1/access", get(access))
        .route("/api/v1/auth/verify", post(verify_access))
        .route("/api/v1/training/targets", get(list_training_targets))
        .route("/api/v1/training/presets", get(list_training_presets))
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
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id",
            get(get_asset).delete(delete_asset),
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
        .route("/api/v1/models", get(list_models))
        .route("/api/v1/models/:model_id", delete(delete_model))
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
        .route("/api/v1/control-overlays", get(list_control_overlays))
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
        .route("/api/v1/jobs/events", get(job_events))
        .route("/api/v1/jobs/events/ticket", post(create_event_ticket))
        // Media ticket (sc-8810): auth-protected mint endpoint; the ticket is honored
        // as a query param by the project-files and pose-preview GETs (see auth.rs).
        .route("/api/v1/files/ticket", post(create_media_ticket))
        .route("/api/v1/jobs/:job_id", get(get_job))
        .route("/api/v1/jobs/:job_id/cancel", post(cancel_job))
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
        .layer(cors);
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

async fn get_project_file(
    State(state): State<AppState>,
    Path((project_id, relative_path)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let project_file = project_call(state, {
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
    let mut file = tokio::fs::File::open(&project_file.path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let total = file
        .metadata()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .len();
    let content_type = project_file.content_type;

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
                return Ok((
                    StatusCode::PARTIAL_CONTENT,
                    [
                        (header::CONTENT_TYPE, content_type),
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
                    .into_response());
            }
            None => {
                return Ok((
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    [(header::CONTENT_RANGE, format!("bytes */{total}"))],
                )
                    .into_response());
            }
        }
    }

    let stream = ReaderStream::new(file);
    Ok((
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
        .into_response())
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
    use axum::http::{header, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};
    use rust_embed::RustEmbed;

    #[derive(RustEmbed)]
    #[folder = "../web/dist"]
    struct WebAssets;

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

    pub(super) async fn serve(uri: Uri) -> Response {
        let requested = uri.path().trim_start_matches('/');
        let requested = if requested.is_empty() {
            "index.html"
        } else {
            requested
        };
        if let Some(file) = WebAssets::get(requested) {
            let mime = mime_guess::from_path(requested).first_or_octet_stream();
            return (
                [
                    (header::CONTENT_TYPE, mime.as_ref()),
                    (header::CONTENT_SECURITY_POLICY, CONTENT_SECURITY_POLICY),
                ],
                file.data.into_owned(),
            )
                .into_response();
        }
        // Single-page app: unknown non-API paths resolve to index.html so
        // client-side deep links (e.g. project routes) load correctly.
        match WebAssets::get("index.html") {
            Some(index) => (
                [
                    (header::CONTENT_TYPE, "text/html; charset=utf-8"),
                    (header::CONTENT_SECURITY_POLICY, CONTENT_SECURITY_POLICY),
                ],
                index.data.into_owned(),
            )
                .into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        }
    }
}

/// Router fallback. With `embed-web`, non-API paths are served from the embedded
/// web bundle (SPA fallback); API paths and all default-feature builds keep the
/// existing JSON not-found behavior.
async fn app_fallback(request: Request<axum::body::Body>) -> Response {
    #[cfg(feature = "embed-web")]
    {
        if !request.uri().path().starts_with("/api/") {
            return web_assets::serve(request.uri().clone()).await;
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
    let (sweep, summary): (StaleSweep, QueueSummary) =
        store_call(state.clone(), move |store, timeout| {
            // When the caller already swept this request, don't pay for a second
            // sweep — just read the summary. The empty StaleSweep means the
            // job.updated fan-out below is a no-op (the caller emitted those
            // events off its own sweep result).
            let sweep = if skip_sweep {
                StaleSweep::default()
            } else {
                store.mark_stale_workers_interrupted(timeout)?
            };
            let summary = store.queue_summary()?;
            Ok((sweep, summary))
        })
        .await?;
    // The stale-sweep mutates jobs to `interrupted` in the DB but — unlike a worker-reported
    // terminal status (`update_job_progress`) or the supervisor crash path (`worker_terminated`) —
    // emits no per-job event. Broadcast `job.updated` for each swept job so a live client's job card
    // flips to "Interrupted" instead of showing its last running state forever: the frontend's job
    // list is driven by `job.updated`, while `queue.updated` only refreshes the summary/workers
    // (sc-8186). The sweep returns each job exactly once (it also flips the owning worker offline, so
    // a later sweep can't re-select it), so this neither spams nor double-fires. When skip_sweep is
    // set the sweep is empty, so nothing is broadcast here.
    for job in &sweep.jobs {
        publish(&state, "job.updated", job);
    }
    Ok(summary)
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
    payload: JsonObject,
    requested_gpu: String,
    initial_status: Option<JobStatus>,
) -> Result<JobSnapshot, ApiError> {
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
    let queue = queue_summary_snapshot(state.clone()).await?;
    publish(state, "queue.updated", &queue);
    Ok(())
}

/// Like [`publish_queue`], but skips the stale-worker sweep because the caller
/// already ran one in its own transaction this request (sc-8889 / F-087). Use
/// only right after a `mark_stale_workers_interrupted` call — otherwise stale
/// workers won't be reaped on this refresh.
async fn publish_queue_skip_sweep(state: &AppState) -> Result<(), ApiError> {
    let queue = queue_summary_snapshot_inner(state.clone(), true).await?;
    publish(state, "queue.updated", &queue);
    Ok(())
}

fn publish<T: Serialize>(state: &AppState, event: &str, data: &T) {
    if let Ok(data) = serde_json::to_string(data) {
        // Publishing with no subscribers is expected; slow subscribers are dropped so they reconnect.
        state.events.publish(EventMessage {
            event: event.to_owned(),
            data,
        });
    }
}

async fn project_path_for_id(state: AppState, project_id: &str) -> Result<PathBuf, ApiError> {
    let project_id = project_id.to_owned();
    let project = project_call(state, move |store| store.get_project(&project_id)).await?;
    Ok(PathBuf::from(project.path))
}

fn model_lora_families(model: &Value) -> Vec<String> {
    families_from_value_chain(
        model,
        &["families", "compatibleFamilies", "modelFamilies"],
        Some("loraCompatibility"),
    )
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
) -> Result<Vec<String>, ApiError> {
    let mut warnings = Vec::new();
    let presets = recipe_preset_catalog(state, project_id).await?;
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
    let jobs = store_call(state.clone(), move |store, timeout| {
        store.mark_stale_workers_interrupted(timeout)?;
        store.list_jobs(None, None, 100)
    })
    .await?;
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
    let mut snapshot_dirs = std::fs::read_dir(&snapshots)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    snapshot_dirs.sort();
    if let Some(main_snapshot) = huggingface_main_snapshot_dir(repo_root) {
        let mut ordered = vec![main_snapshot.clone()];
        ordered.extend(
            snapshot_dirs
                .into_iter()
                .filter(|path| path != &main_snapshot),
        );
        return ordered;
    }
    snapshot_dirs
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

fn validate_image_job(payload: &ImageJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.prompt.is_empty() || payload.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    validate_prompt_extras(&payload.negative_prompt, &payload.advanced)?;
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
    if !(1..=8).contains(&payload.count) {
        return Err(ApiError::bad_request("count must be between 1 and 8"));
    }
    validate_dimension(payload.width, "width", MAX_IMAGE_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_IMAGE_DIMENSION)?;
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

fn validate_character_test_job(payload: &CharacterTestRequest) -> Result<(), ApiError> {
    if payload.prompt.is_empty() || payload.prompt.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    if !(1..=8).contains(&payload.count) {
        return Err(ApiError::bad_request("count must be between 1 and 8"));
    }
    validate_dimension(payload.width, "width", MAX_IMAGE_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_IMAGE_DIMENSION)?;
    Ok(())
}

fn validate_video_job(payload: &VideoJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.prompt.is_empty() || payload.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    validate_prompt_extras(&payload.negative_prompt, &payload.advanced)?;
    if ![
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
    ]
    .contains(&payload.mode.as_str())
    {
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
    let duration = payload
        .duration
        .as_f64()
        .ok_or_else(|| ApiError::bad_request("duration must be a number between 1 and 30"))?;
    if !duration.is_finite() || !(1.0..=30.0).contains(&duration) {
        return Err(ApiError::bad_request("duration must be between 1 and 30"));
    }
    if !(1..=60).contains(&payload.fps) {
        return Err(ApiError::bad_request("fps must be between 1 and 60"));
    }
    validate_dimension(payload.width, "width", MAX_VIDEO_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_VIDEO_DIMENSION)?;
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
        "reference_to_video" if payload.reference_asset_ids.is_empty() => Err(
            ApiError::bad_request("Reference to Video requires at least one reference image."),
        ),
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
        _ => Ok(()),
    }
}

/// Upper bound for image width/height. A backstop only — per-model resolution is
/// governed by manifest `limits.resolutions` + the UI. Covers SenseNova-U1's
/// largest trained bucket (3456) with headroom; video uses its own lower cap.
const MAX_IMAGE_DIMENSION: u32 = 4096;

/// Upper bound for video width/height — a lower backstop than images, matching
/// the cap enforced when validating a video job request.
const MAX_VIDEO_DIMENSION: u32 = 1920;
const MAX_VIDEO_REFERENCE_ASSET_IDS: usize = 8;
const MAX_VIDEO_SOURCE_CLIP_ASSET_IDS: usize = 8;

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
