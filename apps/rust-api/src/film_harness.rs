//! Local filmmaking harness (epic 22708, sc-22710): render a hand-authored production plan into a
//! SceneWorks sequence through the EXISTING API seams — projects, asset import, `/api/v1/video/jobs`,
//! job polling, timelines and timeline export — and leave a versioned run record behind.
//!
//! The driver is written against [`ApiTransport`], a two-method abstraction over the HTTP API, so
//! the same code runs in-process against `create_app` in tests (with a fake worker claiming the
//! jobs) and over loopback against a live API + GPU worker from the `film-harness` binary. It never
//! renders anything itself and never talks to an engine: every take is produced by whatever worker
//! claims the job, exactly as a Video Studio submission would be.
//!
//! Order of operations, and the guarantee each step gives:
//!
//! 1. read + validate the plan and reference pack (structure, cross-references, files on disk);
//! 2. read the API HOST's platform and memory from `GET /api/v1/host-capabilities` — `--api` may
//!    point at another machine, so the host's platform, not `cfg!(target_os)`, decides the lane
//!    and is what the record claims as the render hardware;
//! 3. resolve the model's catalog entry and validate every shot against its declared modes,
//!    menus, caps and lane memory minimum ([`sceneworks_core::film_plan`]), plus the enqueue
//!    route's own platform-reachability and reference-payload gates, and confirm a registered
//!    worker advertises `video_generate` —
//!    **no job is created while any finding is outstanding** (the run record is still written,
//!    with outcome `rejected`);
//! 4. create/reuse the project and import every reference as a project asset tagged with its role
//!    (an unapproved one is tagged apart and never used as conditioning);
//! 5. dispatch the selected shots one at a time under the plan's wall-clock, attempt and memory
//!    limits — a limit that trips cancels the in-flight job (cooperatively, through the API) and
//!    stops new dispatch; the observed peak comes from the job's metrics block
//!    (`GET /api/v1/jobs/:id/metrics`), which is where a real worker reports it;
//! 6. assemble the rendered takes on a timeline and export it through the `timeline_export` job;
//! 7. write `run.json` (shot -> attempt -> job -> asset, timeline, export, observed
//!    model/backend/hardware) beside copies of the two source documents — on every path past
//!    step 1, refusals and mid-run failures included.
//!
//! **One controller per run directory.** Every mutating library entry point takes the same OS-held
//! filesystem lease, so API and CLI callers cannot drive, resume, review, repair, decide, or edit
//! the same run concurrently, and a crashed process releases ownership without an age guess.

pub mod references;
pub mod review;

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sceneworks_core::file_lock::FileLock;
use sceneworks_core::film_compile::{
    compile_plan, CompileInputs, CompiledPlan, CompiledRequest, DispatchContext,
    PlannerExecutionRecord, ResolvedConditioning,
};
use sceneworks_core::film_plan::{
    self, AttemptRecord, ConditioningAssets, ExportPending, ExportRecord, GeneratedAudio,
    HardwareRecord, IntendedState, ModelEntries, ModelLane, ModelRecord, PlanDiagnostic,
    ProductionDecision, ProductionPlan, ReferenceAssetRecord, ReferencePack, ReviewFlag,
    RunOutcome, RunRecord, RunState, RunStop, ShotOutcome, ShotRunRecord, SoundBed, SoundBus,
    SourceDocument, TakeRecord, TakeRejection, TimelineEditRecord, TimelineItemRecord,
    TimelineRecord, TimelineTrackRecord, RUN_RECORD_SCHEMA_VERSION,
};
use sceneworks_core::film_planner::{capabilities_for, PlannerCapabilities};
use sceneworks_core::jobs_store::TERMINAL_STATUSES;
use sceneworks_core::time::{parse_utc_seconds, utc_now};
use serde_json::{json, Map as JsonObject, Value};
use tokio::time::Instant;

mod transport;

mod plan;

mod run;

mod timeline;

mod provenance;

pub use transport::{
    encode_asset_upload, sanitize_multipart_filename, ApiRequest, ApiResponse, ApiTransport,
    BytesResponse, BytesTransportFuture, HarnessError, HttpTransport, RequestBody, TransportFuture,
};

pub(crate) use transport::{expect_ok_on, host_facts_for, job_peak_memory_bytes};

use transport::{
    api_detail, download_media, Client, JobView, PollBounds, PollStop, TERMINAL_ATTEMPT_STATUSES,
};
#[cfg(test)]
use transport::{merge_job_pages, newest_export_job};

pub use plan::{validate, LIVE_STATUSES};

pub(crate) use plan::{
    catalog_entry_findings, installed_lora_ids, live_worker_advertising, model_tier_installed,
    plan_catalog_for, preflight_documents, stale_workers_detail, FilmDocumentPreflight, HostFacts,
    PlanCatalog, BYTES_PER_GB,
};

use plan::{
    compiled_for_run, discover_host, memory_observation, partition_weights, prepare,
    primary_weights, read_compiled_for, MemoryObservation, Prepared,
};
#[cfg(test)]
use plan::{platform_reachability_finding, reference_payload_findings};

pub use run::{
    clear_cancel_request, replace_take, request_cancel, resume, run, run_with_control,
    run_with_control_and_lease, ControllerLease, ResumeOptions, RunControl, RunOptions,
    CANCEL_SENTINEL_FILE, CONTROLLER_LOCK_FILE,
};

pub(crate) use run::{replace_take_with_lease, resume_with_lease, FilmControllerShutdown};

#[cfg(test)]
use run::attempt_spent_seconds;
use run::{
    flag_dependents, replace_take_operation, Session, ASSET_SETTLE_GRACE, CANCEL_GRACE,
    METRICS_GRACE, METRICS_RETRY_INTERVAL,
};

pub use timeline::{
    edit_timeline, finish_explicit_export, start_explicit_export, EditOptions, ExplicitExportTask,
    TimelineEdit, AMBIENCE_TRACK_ID, DIALOGUE_TRACK_ID, MUSIC_TRACK_ID, PICTURE_TRACK_ID,
    SFX_TRACK_PREFIX,
};

pub(crate) use timeline::reconcile_saved_cut;

#[cfg(test)]
use timeline::{
    aspect_ratio_for, bed_track, export_resolution_for, harness_block, merge_harness_audio_track,
    reduced_aspect_ratio, relayout_timeline, HARNESS_KEY, ROLE_DIALOGUE, ROLE_MUSIC, ROLE_SFX,
};
use timeline::{audio_content_type, resolved_generated_audio, SoundAsset, SOUND_TAG};

pub use provenance::{
    dialogue_idempotency_key, fixture_plate_png, fixture_plate_png_sized, fixture_sound_wav,
    idempotency_key, read_action_operation, read_run_record, synthesized_sound_file,
    write_fixture_images, write_fixture_sound, ActionOperation, FIXTURE_PLATE_SIZE,
    FIXTURE_REFERENCES, FIXTURE_SOUNDS, FIXTURE_SOUND_RATE, RUN_RECORD_FILE,
};

pub(crate) use provenance::{record_action, sha256_hex};

use provenance::{
    audio_asset_from_result, base_record, dropped_audio_layers, harness_asset_tag, persist_record,
    read_source, rejected_record, seconds_since, take_from_result, write_atomically, REFERENCE_TAG,
    UNAPPROVED_REFERENCE_TAG,
};

#[cfg(test)]
use provenance::persisted_picture_items;

#[cfg(test)]
mod unit_tests;

#[cfg(test)]
mod controller_lease_io_tests;

// Keep the existing crate-internal facade stable even where the split now consumes
// these names directly inside their owner module.
#[allow(unused_imports)]
pub(crate) use plan::{model_entry_findings, WorkerAdvert};
#[allow(unused_imports)]
pub(crate) use run::replace_take_operation_with_lease;
