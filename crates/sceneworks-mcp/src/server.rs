//! The SceneWorks MCP tool surface (epic 10231, sc-10233 catalog + sc-10234
//! generate_image + sc-10235 video submit/poll).
//!
//! `SceneWorksMcp` is the rmcp server/service struct: a `#[tool_router]` impl
//! holds one method per MCP tool, and `#[tool_handler]` wires that router into
//! the `ServerHandler` the streamable-HTTP transport serves. Every tool is a
//! thin wrapper over an existing `/api/v1/*` route via [`ApiClient`] — later
//! stories add methods to the `#[tool_router]` block, nothing else.
//!
//! The catalog endpoints return large manifest-derived objects (multi-KB per
//! model: downloads, footprints, platform notes …). Tools re-shape them into
//! compact JSON an LLM can actually use — ids/names plus the values a job
//! request needs — via the pure `compact_*` mappers below (unit-tested).
//!
//! `generate_image` (sc-10234) is the first BLOCKING job tool: it submits a
//! real `POST /api/v1/image/jobs`, polls `GET /api/v1/jobs/:id` to a terminal
//! status (relaying JobSnapshot progress as MCP progress notifications), then
//! fetches the produced media through the project files route and returns it
//! inline as base64 image content.
//!
//! Video generation runs minutes and outlives a single blocking call, so
//! sc-10235 adds a NON-blocking submit/poll trio instead: `submit_video_job`
//! (`POST /api/v1/video/jobs` → job id + initial snapshot), `get_job_status`
//! (a generic `GET /api/v1/jobs/:id` view usable for image jobs too), and
//! `get_job_result` (ticketed download links via `POST /api/v1/files/ticket`
//! — media bytes are never inlined for these).

use std::time::Duration;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, ContentBlock, Extensions, ProgressNotificationParam, ProgressToken,
        Resource, ServerCapabilities, ServerInfo,
    },
    schemars,
    service::{RequestContext, RoleServer},
    tool, tool_handler, tool_router, ErrorData, Peer, ServerHandler,
};
use sceneworks_core::video_request::{classify_reference_set, ReferenceSetVerdict};
use serde_json::{json, Map, Value};

use crate::api_client::{ApiClient, ApiClientError};

/// How long a continuous `GET /api/v1/jobs/:id` outage is tolerated. Counting
/// failures made tolerance accidentally depend on the configured poll cadence:
/// a faster poller abandoned the same server outage much sooner.
const MAX_POLL_OUTAGE: Duration = Duration::from_secs(15);

fn poll_outage_exceeded(outage_started: tokio::time::Instant, now: tokio::time::Instant) -> bool {
    now.duration_since(outage_started) >= MAX_POLL_OUTAGE
}

/// F-041 (sc-11236) — inline-payload caps for `generate_image`. The tool
/// base64-inlines every produced image into the JSON-RPC tool result. base64
/// inflates the bytes by ~33% and the encoded response is held twice in memory,
/// so a large batch (`count` up to 8, caller-chosen dimensions up to 2048²) can
/// balloon to tens of MB — enough to exceed an MCP client's message-size limit or
/// blow the model's context window. When a single image exceeds
/// [`MAX_INLINE_IMAGE_BYTES`] OR the running total exceeds
/// [`MAX_INLINE_TOTAL_BYTES`], `generate_image` returns the exact same
/// ticketed-download-link shape as `get_job_result` (resource links + a JSON
/// summary, no inline bytes) instead of inlining. Thresholds are on the RAW
/// (pre-base64) byte count and picked conservatively: a typical 1–2 image job of a
/// few-MP PNG still inlines; only genuinely heavy batches spill to links.
const MAX_INLINE_IMAGE_BYTES: usize = 4 * 1024 * 1024;
const MAX_INLINE_TOTAL_BYTES: usize = 10 * 1024 * 1024;

/// MCP image egress is privacy-first by default (sc-16202). Unlike a person choosing Save As, an
/// MCP recipient is commonly model-provider infrastructure, so silently forwarding authored
/// prompts, model settings, LoRA repositories, and pose/face coordinates is the riskier default.
/// Agents that deliberately need recipe inspection can opt in per result with `includeWorkflow`.
const DEFAULT_INCLUDE_WORKFLOW: bool = false;

fn workflow_policy(include_workflow: bool) -> &'static str {
    if include_workflow {
        "preserve-if-present"
    } else {
        "strip-requested"
    }
}

/// How the blocking job tools (generate_image) wait for a terminal JobSnapshot:
/// poll `GET /api/v1/jobs/:id` every `poll_interval` until terminal, and give up
/// with a clear tool error after `timeout` so a stuck job can never hang the MCP
/// call forever. Tests shrink both; production uses the defaults.
#[derive(Debug, Clone)]
pub struct JobWaitConfig {
    pub poll_interval: Duration,
    pub timeout: Duration,
}

impl Default for JobWaitConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(1),
            // Generous: a cold first run legitimately spends minutes in
            // `downloading`/`loading_model` before it ever renders.
            timeout: Duration::from_secs(30 * 60),
        }
    }
}

impl JobWaitConfig {
    /// Build a config from deployment-supplied values (sc-10277), enforcing the
    /// invariants the poll loop relies on: a zero/absent poll interval would spin
    /// the CPU (or, as a `sleep(0)`, hammer the API), and a timeout below the
    /// interval would fire before the first poll. A zero interval falls back to
    /// the default cadence; the timeout is raised to at least one interval.
    pub fn clamped(poll_interval: Duration, timeout: Duration) -> Self {
        let poll_interval = if poll_interval.is_zero() {
            Self::default().poll_interval
        } else {
            poll_interval
        };
        Self {
            poll_interval,
            timeout: timeout.max(poll_interval),
        }
    }
}

#[derive(Clone)]
pub struct SceneWorksMcp {
    api: ApiClient,
    job_wait: JobWaitConfig,
    allowed_hosts: Vec<String>,
    tool_router: ToolRouter<Self>,
}

impl SceneWorksMcp {
    pub fn new(api: ApiClient) -> Self {
        Self {
            api,
            job_wait: JobWaitConfig::default(),
            allowed_hosts: crate::loopback_allowed_hosts(),
            tool_router: Self::tool_router(),
        }
    }

    /// Override the blocking-job polling cadence/deadline (tests).
    pub fn with_job_wait(mut self, job_wait: JobWaitConfig) -> Self {
        self.job_wait = job_wait;
        self
    }

    pub fn with_allowed_hosts(mut self, allowed_hosts: Vec<String>) -> Self {
        self.allowed_hosts = allowed_hosts;
        self
    }
}

/// Optional filters for `list_loras`, forwarded verbatim to the
/// `GET /api/v1/loras` query params (`LorasQuery` in the API is camelCase).
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ListLorasArgs {
    #[schemars(
        description = "Only return LoRAs compatible with this model family (e.g. \"sdxl\", \"z-image\", \"flux\")."
    )]
    pub model_family: Option<String>,
    #[schemars(
        description = "Also include LoRAs trained/imported in this project (by project id)."
    )]
    pub project_id: Option<String>,
}

/// Arguments for `generate_image`, mapped 1:1 onto the API's `ImageJobRequest`
/// (`apps/rust-api/src/dto.rs`). Only the provided fields are sent, so the API's
/// serde defaults stay authoritative — except `count`, which defaults to 1 here
/// (the API's default of 4 is a web-UI batch size; 4 inline base64 images is a
/// lot of tokens to return unasked).
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GenerateImageArgs {
    #[schemars(description = "Project to generate into (from list_projects).")]
    pub project_id: String,
    #[schemars(description = "The image prompt (1-4000 characters).")]
    pub prompt: String,
    #[schemars(
        description = "\"generate\" (default, text-to-image) or \"edit_image\" (image-to-image; needs sourceAssetId or referenceAssetIds)."
    )]
    pub mode: Option<String>,
    #[schemars(description = "Things to avoid in the image.")]
    pub negative_prompt: Option<String>,
    #[schemars(description = "Model id from list_models. Omit for the server default.")]
    pub model: Option<String>,
    #[schemars(description = "How many images to generate (1-8, default 1).")]
    pub count: Option<u32>,
    #[schemars(description = "Seed for reproducible output. Omit for random per-image seeds.")]
    pub seed: Option<i64>,
    #[schemars(description = "Output width in pixels (default 1024).")]
    pub width: Option<u32>,
    #[schemars(description = "Output height in pixels (default 1024).")]
    pub height: Option<u32>,
    #[schemars(description = "Style preset name (default \"cinematic\").")]
    pub style_preset: Option<String>,
    #[schemars(
        description = "LoRA adapters to apply: [{\"id\": <from list_loras>, \"weight\": 0.0-2.0}]."
    )]
    pub loras: Option<Vec<Value>>,
    #[schemars(description = "Character to condition on (character id).")]
    pub character_id: Option<String>,
    #[schemars(description = "Edit base image asset id (edit_image mode).")]
    pub source_asset_id: Option<String>,
    #[schemars(
        description = "Reference image asset id for IP-Adapter style/identity conditioning."
    )]
    pub reference_asset_id: Option<String>,
    #[schemars(
        description = "Multi-image reference set for a multi-reference edit (each id jointly conditions the edit)."
    )]
    pub reference_asset_ids: Option<Vec<String>>,
    #[schemars(
        description = "Inpaint mask asset id (white = edit region; inpaint-capable models only)."
    )]
    pub mask_asset_id: Option<String>,
    #[schemars(
        description = "Include SceneWorks workflow metadata in returned images. Default false: MCP outputs often travel to model-provider infrastructure, so workflow metadata is stripped unless explicitly requested."
    )]
    pub include_workflow: Option<bool>,
}

/// Arguments for `submit_video_job`, mapped onto the API's `VideoJobRequest`
/// (`apps/rust-api/src/dto.rs`). The tool exposes task-level modes and maps them to the API's wire
/// modes in [`video_job_body`]; only provided fields are sent so the API's serde defaults
/// (duration 6s, 25fps, 768x512, ltx_2_3 …) stay authoritative.
///
/// sc-19576 completed the mapping: every mode `VIDEO_JOB_MODES` admits is now reachable from here,
/// pinned by [`VIDEO_TOOL_MODES`] and the round-trip tests, so the count cannot drift back.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SubmitVideoJobArgs {
    #[schemars(description = "Project to generate into (from list_projects).")]
    pub project_id: String,
    #[schemars(description = "The video prompt (1-4000 characters).")]
    pub prompt: String,
    #[schemars(
        description = "\"generate\" (default: text-to-video, or image-to-video when sourceAssetId is set, or first/last-frame when lastFrameAssetId is also set), \"reference\" (render from reference media; needs at least one referenceAssetIds or sourceClipAssetIds entry, with optional referenceAudioAssetIds), \"extend\" (continue a clip; needs sourceClipAssetId), \"bridge\" (fill between two clips; needs sourceClipAssetId + bridgeRightClipAssetId), \"person_replace\" (swap a tracked person; needs sourceClipAssetId + personTrackId + characterId), \"video_to_video\" (edit a clip; needs sourceClipAssetId), \"reference_video_to_video\" (edit a clip guided by reference images; needs sourceClipAssetId + referenceAssetIds), \"multi_video_to_video\" (blend several clips; needs 2+ sourceClipAssetIds), \"ads2v\" (edit a clip using a reference video + reference images; needs sourceClipAssetId + referenceClipAssetId + referenceAssetIds), or \"animate_character\" (animate a character with a driving video; needs sourceClipAssetId + referenceAssetIds or sourceAssetId)."
    )]
    pub mode: Option<String>,
    #[schemars(description = "Things to avoid in the video.")]
    pub negative_prompt: Option<String>,
    #[schemars(
        description = "Video model id from list_models (type \"video\"). Omit for the server default."
    )]
    pub model: Option<String>,
    #[schemars(description = "Clip length in seconds (1-30, default 6).")]
    pub duration: Option<f64>,
    #[schemars(description = "Frames per second (1-60, default 25).")]
    pub fps: Option<u32>,
    #[schemars(description = "Output width in pixels (256-1920, default 768).")]
    pub width: Option<u32>,
    #[schemars(description = "Output height in pixels (256-1920, default 512).")]
    pub height: Option<u32>,
    #[schemars(description = "Quality preset (\"draft\", \"balanced\" (default) or \"high\").")]
    pub quality: Option<String>,
    #[schemars(description = "Seed for reproducible output. Omit for a random seed.")]
    pub seed: Option<i64>,
    /// sc-17161. Denoise step count, carried in `advanced.steps` — the ONLY key the API reads it
    /// from (`sceneworks_core::video_request::requested_steps`). Without it this tool could only
    /// ever submit the model's `defaults.steps`, which for MiniMax-H3 is 50 and measured in HOURS
    /// at the default canvas; the cheap draft an agent would reach for was unreachable.
    /// Per-model floors are real (`limits.hardMinSteps`), so `list_models` reports them.
    #[schemars(
        description = "Denoise steps. Omit for the model's own default. Fewer steps render proportionally faster; a model may declare a minimum (minSteps from list_models) below which the request is refused rather than raised. If a step-distill adapter is attached (see loras), omitting this runs that adapter's own trained step count; setting it overrides that count."
    )]
    pub steps: Option<u32>,
    /// sc-18727. The turbo variant selector IS this field — a step-distill adapter is selected the
    /// same way any other LoRA is, which is what keeps the MCP surface, the Video Studio control and
    /// the generic picker on one payload key and one server-side resolver. The description says so
    /// explicitly because the effect is not guessable from an id: attaching
    /// `minimax_h3_turbo_4step_768p` changes the schedule, not just the weights.
    #[schemars(
        description = "LoRA adapters to apply: [{\"id\": <from list_loras>, \"weight\": 0.0-2.0}]. An adapter whose list_loras entry carries role=\"accelerator\" and a sampling block is a step-distill accelerator: attaching it also switches the render to that adapter's trained schedule (its sampling.steps and sampling.schedulerShift), which is how a MiniMax-H3 job renders in 4-8 steps instead of 50. Attach at most one accelerator per job — two asking for different schedules are refused."
    )]
    pub loras: Option<Vec<Value>>,
    #[schemars(
        description = "Character to condition on (character id; required for person_replace)."
    )]
    pub character_id: Option<String>,
    #[schemars(description = "Starting image asset id (generate mode: makes it image-to-video).")]
    pub source_asset_id: Option<String>,
    #[schemars(
        description = "Ending image asset id (generate mode: with sourceAssetId, makes it a first/last-frame generation)."
    )]
    pub last_frame_asset_id: Option<String>,
    #[schemars(
        description = "Source video clip asset id (extend: the clip to continue; bridge: the LEFT clip; person_replace: the clip to edit)."
    )]
    pub source_clip_asset_id: Option<String>,
    #[schemars(description = "RIGHT video clip asset id for bridge mode.")]
    pub bridge_right_clip_asset_id: Option<String>,
    /// sc-19576. The `ads2v` reference-VIDEO slot — a second source video, distinct from
    /// `sourceClipAssetId` (the clip being edited) and from `sourceClipAssetIds` (the reference
    /// clips of `reference` mode). Without this field `ads2v` was unreachable at the FIELD level,
    /// not merely the mode level: `validate_video_job` hard-requires `referenceClipAssetId` for it,
    /// so no combination of the existing arguments could have produced a valid payload.
    #[schemars(
        description = "Reference video clip asset id for \"ads2v\" mode — the video whose look/motion guides the edit, distinct from sourceClipAssetId (the clip being edited)."
    )]
    pub reference_clip_asset_id: Option<String>,
    /// sc-17161. `reference` mode's subject images. The API's `reference_to_video` arm takes them
    /// as a LIST (Bernini encodes each as a subject reference; MiniMax-H3 Ref2VA labels them
    /// `<Picture 1>`, `<Picture 2>` … IN THE ORDER GIVEN, so the order is part of the request).
    #[schemars(
        description = "Reference image asset ids for \"reference\" mode (subjects, characters, wardrobe, locations). Order is meaningful for models that label references in the prompt."
    )]
    pub reference_asset_ids: Option<Vec<String>>,
    /// sc-17161. Reference video clips for `reference` mode. Distinct from `sourceClipAssetId`,
    /// which is the single clip extend/bridge/person_replace edit.
    #[schemars(
        description = "Reference video clip asset ids for \"reference\" mode, on a model whose reference modes take motion clips as well as images. Models that do not take reference clips reject a non-empty list."
    )]
    pub source_clip_asset_ids: Option<Vec<String>>,
    /// sc-17160. Only models that DECLARE `limits.maxReferenceAudioAssets` accept these; every
    /// other video model refuses a non-empty list at enqueue, which is why the description says
    /// so rather than letting the caller discover it from a 400.
    #[schemars(
        description = "Audio clip asset ids to condition the render on (up to 3), for a model whose reference modes take audio as well as images. Every other video model rejects a non-empty list."
    )]
    pub reference_audio_asset_ids: Option<Vec<String>>,
    #[schemars(description = "Person track id to replace (person_replace mode).")]
    pub person_track_id: Option<String>,
    #[schemars(description = "person_replace scope: \"face_only\" (default) or \"full_body\".")]
    pub replacement_mode: Option<String>,
}

/// Arguments for the generic job-polling tools (`get_job_status` /
/// `get_job_result`) — they work for any job type (video AND image).
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct JobIdArgs {
    #[schemars(
        description = "The job id returned by submit_video_job (or any other job-submitting call)."
    )]
    pub job_id: String,
}

/// Arguments for `get_job_result`. Image links follow the same privacy-first workflow metadata
/// policy as inline `generate_image` results; video and other non-PNG bytes are unchanged.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct JobResultArgs {
    #[schemars(description = "The completed job id returned by a job-submitting call.")]
    pub job_id: String,
    #[schemars(
        description = "Include SceneWorks workflow metadata in linked images. Default false: MCP outputs often travel to model-provider infrastructure, so workflow metadata is stripped unless explicitly requested."
    )]
    pub include_workflow: Option<bool>,
}

#[tool_router]
impl SceneWorksMcp {
    #[tool(
        description = "List SceneWorks projects. Returns [{id, name, createdAt}]; use the id as projectId in other calls."
    )]
    async fn list_projects(&self) -> Result<CallToolResult, ErrorData> {
        let projects = self
            .api
            .get_json("/api/v1/projects", &[])
            .await
            .map_err(api_error)?;
        json_result(compact_projects(&projects))
    }

    #[tool(
        description = "List the generation model catalog. Returns compact entries: id (use as the model for a job), name, family, type (image/video), capabilities, installState, defaults (resolution/steps/guidanceScale/count), the supported resolutions/durations/fps menus, minSteps, the reference-media caps (maxReferenceAssets / maxSourceClipAssets / maxReferenceAudioAssets / maxCombinedReferenceAssets), promptGuide, and any licence-required attribution string. An fps menu is EXHAUSTIVE — an off-menu fps is refused, never rounded. A durations menu lists the model's exactly renderable clip lengths: a duration outside the model's hard min/max bounds is refused, but an in-range off-menu value is silently snapped onto the model's frame lattice (MiniMax-H3 rounds UP to the next 17n+5 rung) — send a menu value to get exactly the length you asked for."
    )]
    async fn list_models(&self) -> Result<CallToolResult, ErrorData> {
        let models = self
            .api
            .get_json("/api/v1/models", &[])
            .await
            .map_err(api_error)?;
        json_result(compact_models(&models))
    }

    #[tool(
        description = "List the LoRA adapter catalog (built-in, imported and trained). Returns compact entries: id, name, family, compatibleFamilies, triggerWords, defaultWeight, installState. Optionally filter by model family and/or project."
    )]
    async fn list_loras(
        &self,
        Parameters(args): Parameters<ListLorasArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let loras = self
            .api
            .get_json(
                "/api/v1/loras",
                &[
                    ("modelFamily", args.model_family.as_deref()),
                    ("projectId", args.project_id.as_deref()),
                ],
            )
            .await
            .map_err(api_error)?;
        json_result(compact_loras(&loras))
    }

    #[tool(
        description = "Generate images (or edit an existing image) and return them inline. Submits an image job, waits for it to finish (emitting progress notifications while it runs), and returns each generated image as base64 image content plus a JSON summary with the asset ids. Workflow metadata is stripped by default because MCP outputs often travel to model-provider infrastructure; set includeWorkflow=true only when the recipient needs the embedded recipe. The same choice applies if an oversize result falls back to resource links. Long-running: seconds to minutes depending on the model."
    )]
    async fn generate_image(
        &self,
        Parameters(args): Parameters<GenerateImageArgs>,
        ctx: RequestContext<RoleServer>,
        extensions: Extensions,
    ) -> Result<CallToolResult, ErrorData> {
        let body =
            image_job_body(&args).map_err(|message| ErrorData::invalid_params(message, None))?;
        let include_workflow = args.include_workflow.unwrap_or(DEFAULT_INCLUDE_WORKFLOW);
        let submitted = self
            .api
            .post_json("/api/v1/image/jobs", &body)
            .await
            .map_err(api_error)?;
        let job_id = submitted
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ErrorData::internal_error("image job submission returned no job id", None)
            })?
            .to_owned();

        // Progress notifications ride the client-supplied progressToken; without
        // one we just poll silently (the spec forbids inventing a token).
        let progress_token = ctx.meta.get_progress_token();
        let mut reporter = ProgressReporter::new(ctx.peer.clone(), progress_token);
        reporter.report(&submitted).await;

        let started = tokio::time::Instant::now();
        let mut poll_outage_started: Option<tokio::time::Instant> = None;
        let job = loop {
            // Cooperative cancellation (sc-10276): rmcp cancels `ctx.ct` when the
            // client sends `notifications/cancelled` (it does NOT abort the tool
            // future, so a drop-guard would never fire). Catch it at the top of the
            // wait and ask the job to stop, freeing the worker/GPU instead of
            // letting a canceled render run to completion.
            tokio::select! {
                biased;
                () = ctx.ct.cancelled() => {
                    // Best-effort: the job may already be terminal, in which case the
                    // cancel route is a harmless no-op.
                    let _ = self
                        .api
                        .post_json(&format!("/api/v1/jobs/{job_id}/cancel"), &json!({}))
                        .await;
                    return tool_error(format!(
                        "Image job {job_id} was canceled: the client canceled the \
                         request, so the job was asked to stop."
                    ));
                }
                () = tokio::time::sleep(self.job_wait.poll_interval) => {}
            }
            // Tolerate transient poll failures (sc-10279): the render keeps running
            // server-side, so a single blip must not abort the whole tool call.
            let job = match self
                .api
                .get_json(&format!("/api/v1/jobs/{job_id}"), &[])
                .await
            {
                Ok(job) => {
                    poll_outage_started = None;
                    job
                }
                Err(error) => {
                    let now = tokio::time::Instant::now();
                    let outage_started = *poll_outage_started.get_or_insert(now);
                    if poll_outage_exceeded(outage_started, now) {
                        return Err(api_error(error));
                    }
                    if started.elapsed() >= self.job_wait.timeout {
                        return tool_error(format!(
                            "Image job {job_id} status polling kept failing (last error: \
                             {error}) and the {}s deadline elapsed. The job may still be \
                             running; it was not canceled.",
                            self.job_wait.timeout.as_secs()
                        ));
                    }
                    continue;
                }
            };
            reporter.report(&job).await;
            let status = job.get("status").and_then(Value::as_str).unwrap_or("");
            match status {
                "completed" => break job,
                "failed" => {
                    let detail = job_error_detail(&job);
                    return tool_error(format!("Image job {job_id} failed: {detail}"));
                }
                "canceled" => {
                    return tool_error(format!(
                        "Image job {job_id} was canceled before it finished."
                    ));
                }
                "interrupted" => {
                    return tool_error(format!(
                        "Image job {job_id} was interrupted (worker restarted mid-run); \
                         call generate_image again to retry."
                    ));
                }
                _ => {}
            }
            if started.elapsed() >= self.job_wait.timeout {
                return tool_error(format!(
                    "Image job {job_id} did not reach a terminal state within {}s \
                     (last status: {status}). The job may still be running; it was \
                     not canceled.",
                    self.job_wait.timeout.as_secs()
                ));
            }
        };

        // The job row is authoritative for the project id (mirrors the API).
        let project_id = job
            .get("projectId")
            .and_then(Value::as_str)
            .unwrap_or(&args.project_id)
            .to_owned();
        if !valid_project_id(&project_id) {
            return tool_error(format!(
                "Image job {job_id} returned an invalid project id, so its files cannot be located."
            ));
        }
        let assets: Vec<&Value> = job
            .pointer("/result/assets")
            .and_then(Value::as_array)
            .map(|assets| {
                assets
                    .iter()
                    .filter(|asset| is_image_asset(asset))
                    .collect()
            })
            .unwrap_or_default();
        if assets.is_empty() {
            return tool_error(format!(
                "Image job {job_id} completed but reported no image assets."
            ));
        }

        let mut blocks = Vec::with_capacity(assets.len() + 1);
        let mut summary_assets = Vec::with_capacity(assets.len());
        let mut total_bytes = 0usize;
        for asset in assets {
            let Some(media_path) = asset_media_path(asset) else {
                continue;
            };
            let remaining_total = MAX_INLINE_TOTAL_BYTES.saturating_sub(total_bytes);
            let body_limit = MAX_INLINE_IMAGE_BYTES.min(remaining_total);
            let media_url = format!(
                "/api/v1/projects/{project_id}/files/{}{}",
                encode_media_path(&media_path),
                if include_workflow {
                    ""
                } else {
                    "?stripWorkflow=true"
                }
            );
            let (bytes, header_mime) = self
                .api
                .get_bytes_bounded(&media_url, body_limit)
                .await
                .map_err(api_error)?;
            let Some(bytes) = bytes else {
                let link_base = self.request_link_base(&extensions);
                return self
                    .job_result_links(&job_id, &job, link_base, include_workflow)
                    .await;
            };
            let mime_type = image_mime_type(
                &media_path,
                asset.pointer("/file/mimeType").and_then(Value::as_str),
                header_mime.as_deref(),
            );
            // F-041 (sc-11236): an over-cap result must not be inlined — the
            // base64 JSON-RPC payload (held twice in memory) would exceed MCP
            // client message limits / the model context. Above EITHER the
            // per-image or running-total cap, switch the WHOLE response to the
            // get_job_result ticketed-link shape (bytes downloaded so far are
            // simply dropped). The ticket links reach the same media without
            // inlining, so no result is lost.
            total_bytes += bytes.len();
            summary_assets.push(json!({
                "id": asset.get("id").cloned().unwrap_or(Value::Null),
                "path": &media_path,
                "mimeType": &mime_type,
                "workflowPolicy": workflow_policy(include_workflow),
            }));
            blocks.push(ContentBlock::image(BASE64.encode(&bytes), mime_type));
        }
        if blocks.is_empty() {
            return tool_error(format!(
                "Image job {job_id} completed but its assets carried no media paths."
            ));
        }
        blocks.push(ContentBlock::json(json!({
            "jobId": job_id,
            "projectId": project_id,
            "assets": summary_assets,
        }))?);
        Ok(CallToolResult::success(blocks))
    }

    #[tool(
        description = "Submit a video generation job WITHOUT waiting for it (video renders for minutes). Modes: \"generate\" (text-to-video; add sourceAssetId for image-to-video, plus lastFrameAssetId for first/last-frame), \"reference\" (render from reference media — images and/or video clips, optionally with audio clips), \"extend\" (continue a clip), \"bridge\" (fill between two clips), \"person_replace\" (swap a tracked person for a Character), \"video_to_video\" (edit a clip), \"reference_video_to_video\" (edit a clip guided by reference images), \"multi_video_to_video\" (blend two or more clips), \"ads2v\" (edit a clip using a reference video plus reference images), \"animate_character\" (animate a reference character with a driving video). Check list_models first: each model serves only some of these, its fps menu is exhaustive (an off-menu fps is refused, never rounded), its durations menu lists the exactly renderable clip lengths (out-of-bounds durations are refused; an in-range off-menu value is silently snapped onto the model's frame lattice — MiniMax-H3 rounds UP to the next rung — so send a menu value), and its resolutions menu lists the geometry buckets it renders at. Returns the job id + initial snapshot; poll get_job_status, then fetch links with get_job_result once completed."
    )]
    async fn submit_video_job(
        &self,
        Parameters(args): Parameters<SubmitVideoJobArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let body =
            video_job_body(&args).map_err(|message| ErrorData::invalid_params(message, None))?;
        let job = self
            .api
            .post_json("/api/v1/video/jobs", &body)
            .await
            .map_err(api_error)?;
        if job.get("id").and_then(Value::as_str).is_none() {
            return Err(ErrorData::internal_error(
                "video job submission returned no job id",
                None,
            ));
        }
        let mut snapshot = compact_job_status(&job);
        if let Some(out) = snapshot.as_object_mut() {
            out.insert(
                "next".to_owned(),
                json!(
                    "Video jobs run for minutes. Poll get_job_status with this jobId; \
                     once status is \"completed\", call get_job_result for download links."
                ),
            );
        }
        json_result(snapshot)
    }

    #[tool(
        description = "Get the current status of a submitted job (works for video AND image jobs): status (queued/running/completed/failed/canceled/interrupted), stage, progressPercent, etaSeconds, and the error message when the job failed."
    )]
    async fn get_job_status(
        &self,
        Parameters(args): Parameters<JobIdArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let job_id = valid_job_id(&args.job_id)
            .map_err(|message| ErrorData::invalid_params(message, None))?;
        let job = self
            .api
            .get_json(&format!("/api/v1/jobs/{job_id}"), &[])
            .await
            .map_err(api_error)?;
        json_result(compact_job_status(&job))
    }

    #[tool(
        description = "Fetch the result of a COMPLETED job (video or image) as downloadable links. Mints a short-lived media ticket and returns one resource link per result asset — the URL works from any machine that can reach the SceneWorks API (no auth header needed while the ticket is valid). Workflow metadata is stripped from linked images by default because MCP outputs often travel to model-provider infrastructure; set includeWorkflow=true only when the recipient needs the embedded recipe. Video/image bytes are never inlined by this tool. If the job is still running it reports ready=false; if it failed, the job error."
    )]
    async fn get_job_result(
        &self,
        Parameters(args): Parameters<JobResultArgs>,
        extensions: Extensions,
    ) -> Result<CallToolResult, ErrorData> {
        let job_id = valid_job_id(&args.job_id)
            .map_err(|message| ErrorData::invalid_params(message, None))?;
        let job = self
            .api
            .get_json(&format!("/api/v1/jobs/{job_id}"), &[])
            .await
            .map_err(api_error)?;
        match job.get("status").and_then(Value::as_str).unwrap_or("") {
            "completed" => {}
            "failed" => {
                let detail = job_error_detail(&job);
                return tool_error(format!("Job {job_id} failed: {detail}"));
            }
            "canceled" => {
                return tool_error(format!("Job {job_id} was canceled before it finished."));
            }
            "interrupted" => {
                return tool_error(format!(
                    "Job {job_id} was interrupted (worker restarted mid-run); resubmit it."
                ));
            }
            // Not terminal yet: a clear ready=false answer, NOT an error — the
            // caller simply keeps polling get_job_status.
            _ => {
                let mut snapshot = compact_job_status(&job);
                if let Some(out) = snapshot.as_object_mut() {
                    out.insert("ready".to_owned(), json!(false));
                    out.insert(
                        "note".to_owned(),
                        json!(
                            "The job has not completed yet — keep polling get_job_status \
                             and call get_job_result again once status is \"completed\"."
                        ),
                    );
                }
                return json_result(snapshot);
            }
        }

        let link_base = self.request_link_base(&extensions);
        self.job_result_links(
            job_id,
            &job,
            link_base,
            args.include_workflow.unwrap_or(DEFAULT_INCLUDE_WORKFLOW),
        )
        .await
    }

    /// Absolute URL base for ticketed media links (sc-10290). `/mcp` and
    /// `/api/v1` are the SAME axum app, so the host the client used to reach
    /// `/mcp` is exactly the host that serves the media — derive it from the
    /// incoming request so the URL is reachable by THIS client regardless of how
    /// `SCENEWORKS_API_URL` is configured (e.g. a loopback-default desktop
    /// answering a LAN client). Falls back to the configured API base when the
    /// request parts / Host aren't available. The Host is only reflected back to
    /// the caller that supplied it (never stored / shown to other users), and
    /// `/mcp` is already gated by access_control, so reflecting it is safe.
    fn request_link_base(&self, extensions: &Extensions) -> String {
        extensions
            .get::<http::request::Parts>()
            .and_then(|parts| request_base_url(&parts.headers, &parts.uri, &self.allowed_hosts))
            .unwrap_or_else(|| self.api.base_url().to_owned())
    }

    /// Build the ticketed-download-link result for a COMPLETED `job`: mint one
    /// sliding multi-use media ticket and emit one `resource_link` block per
    /// result asset plus a JSON summary, with `link_base` as the absolute URL
    /// base. This is the response shape `get_job_result` returns, and the
    /// oversize-payload fallback `generate_image` uses (F-041, sc-11236) instead
    /// of inlining tens of MB of base64.
    async fn job_result_links(
        &self,
        job_id: &str,
        job: &Value,
        link_base: String,
        include_workflow: bool,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(project_id) = job
            .get("projectId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        else {
            return tool_error(format!(
                "Job {job_id} completed but carries no project id, so its files \
                 cannot be located."
            ));
        };
        if !valid_project_id(project_id) {
            return tool_error(format!(
                "Job {job_id} returned an invalid project id, so its files cannot be located."
            ));
        }
        let assets: Vec<(&Value, String)> = job
            .pointer("/result/assets")
            .and_then(Value::as_array)
            .map(|assets| {
                assets
                    .iter()
                    .filter_map(|asset| asset_media_path(asset).map(|path| (asset, path)))
                    .collect()
            })
            .unwrap_or_default();
        if assets.is_empty() {
            return tool_error(format!(
                "Job {job_id} completed but reported no downloadable result assets."
            ));
        }

        // One sliding multi-use media ticket covers every link (sc-8810 flavor):
        // it authorizes GET /api/v1/projects/:id/files/* via `?ticket=` with no
        // auth header, so the URL is fetchable from any machine that can reach
        // the API — exactly what a remote MCP client needs.
        let ticket_response = self
            .api
            .post_json("/api/v1/files/ticket", &json!({}))
            .await
            .map_err(api_error)?;
        let Some(ticket) = ticket_response
            .get("ticket")
            .and_then(Value::as_str)
            .filter(|ticket| !ticket.is_empty())
        else {
            return Err(ErrorData::internal_error(
                "the media ticket endpoint returned no ticket",
                None,
            ));
        };
        let expires_in_seconds = ticket_response.get("expiresInSeconds").cloned();

        let mut blocks = Vec::with_capacity(assets.len() + 1);
        let mut summary_assets = Vec::with_capacity(assets.len());
        for (asset, media_path) in assets {
            let image_asset = is_image_asset(asset);
            let asset_workflow_policy = if image_asset {
                workflow_policy(include_workflow)
            } else {
                "not-applicable"
            };
            let workflow_query = if include_workflow || !image_asset {
                ""
            } else {
                "stripWorkflow=true&"
            };
            let relative_url = format!(
                "/api/v1/projects/{project_id}/files/{}?{workflow_query}ticket={ticket}",
                encode_media_path(&media_path)
            );
            let url = format!("{link_base}{relative_url}");
            let mime_type = media_mime_type(
                &media_path,
                asset.pointer("/file/mimeType").and_then(Value::as_str),
            );
            let name = media_path
                .rsplit('/')
                .next()
                .filter(|name| !name.is_empty())
                .unwrap_or(&media_path)
                .to_owned();
            let mut link = Resource::new(&url, name).with_description(format!(
                "SceneWorks {} asset {} from job {job_id}",
                asset.get("type").and_then(Value::as_str).unwrap_or("media"),
                asset.get("id").and_then(Value::as_str).unwrap_or("?"),
            ));
            if let Some(mime) = &mime_type {
                link = link.with_mime_type(mime);
            }
            blocks.push(ContentBlock::resource_link(link));
            summary_assets.push(json!({
                "id": asset.get("id").cloned().unwrap_or(Value::Null),
                "type": asset.get("type").cloned().unwrap_or(Value::Null),
                "mimeType": mime_type,
                "url": url,
                "relativeUrl": relative_url,
                "workflowPolicy": asset_workflow_policy,
            }));
        }
        blocks.push(ContentBlock::json(json!({
            "jobId": job_id,
            "projectId": project_id,
            "status": "completed",
            "assets": summary_assets,
            "ticketExpiresInSeconds": expires_in_seconds,
            "note": format!(
                "Each url embeds a short-lived media ticket — download promptly \
                 (call get_job_result again for fresh links). The urls use the base \
                 \"{link_base}\" (derived from the host you used to reach this MCP \
                 server, so they should be directly fetchable); if that base is not \
                 reachable, apply relativeUrl to the base you use to reach /mcp \
                 (everything before /mcp). Each asset's workflowPolicy reports the \
                 requested handling without claiming metadata was present."
            ),
        }))?);
        Ok(CallToolResult::success(blocks))
    }
}

/// Sends MCP progress notifications for JobSnapshot polls, deduplicated on
/// (percent, stage) so a queued job doesn't spam identical updates. A `None`
/// token (client sent no progressToken) makes every call a no-op. Notification
/// failures are ignored — progress is advisory, never worth failing the job for.
struct ProgressReporter {
    peer: Peer<RoleServer>,
    token: Option<ProgressToken>,
    last: Option<(u32, String)>,
}

impl ProgressReporter {
    fn new(peer: Peer<RoleServer>, token: Option<ProgressToken>) -> Self {
        Self {
            peer,
            token,
            last: None,
        }
    }

    async fn report(&mut self, job: &Value) {
        let Some(token) = &self.token else {
            return;
        };
        let (percent, message) = job_progress(job);
        let key = (percent, message.clone());
        if self.last.as_ref() == Some(&key) {
            return;
        }
        self.last = Some(key);
        let _ = self
            .peer
            .notify_progress(
                ProgressNotificationParam::new(token.clone(), f64::from(percent))
                    .with_total(100.0)
                    .with_message(message),
            )
            .await;
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SceneWorksMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "SceneWorks local generation studio. Use list_projects for project ids, \
             list_models for the generation model catalog (model ids + job defaults), and \
             list_loras for LoRA adapters compatible with a model family. generate_image \
             blocks until the images are ready; video runs minutes, so use \
             submit_video_job, poll get_job_status, then get_job_result for ticketed \
             download links (get_job_status/get_job_result work for image jobs too).",
        )
    }
}

/// A tool result whose single content block is the compact JSON payload. Plain
/// text-JSON (not `structured_content`) for the widest MCP-client compatibility.
fn json_result(value: Value) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::success(vec![ContentBlock::json(&value)?]))
}

/// Surface an API failure as a JSON-RPC internal error; the Display impl already
/// includes the upstream status + detail, and never the token.
fn api_error(error: ApiClientError) -> ErrorData {
    ErrorData::internal_error(error.to_string(), None)
}

/// A domain failure (failed/canceled job, timeout) as a tool-level error result
/// — `isError: true` with a plain-text explanation — so the calling LLM sees a
/// message it can react to, rather than a raw JSON-RPC protocol error.
fn tool_error(message: String) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::error(vec![ContentBlock::text(message)]))
}

/// Map `generate_image` args onto the `ImageJobRequest` wire shape (camelCase).
/// Only provided fields are emitted so the API's serde defaults apply; `count`
/// deliberately defaults to 1 (see [`GenerateImageArgs`]). The tool-facing
/// `"generate"` mode maps to the API's `"text_to_image"`.
pub(crate) fn image_job_body(args: &GenerateImageArgs) -> Result<Value, String> {
    let mode = match args.mode.as_deref().map(str::trim).unwrap_or("generate") {
        "" | "generate" | "text_to_image" => "text_to_image",
        "edit_image" => "edit_image",
        other => {
            return Err(format!(
                "unsupported mode \"{other}\": use \"generate\" or \"edit_image\""
            ))
        }
    };
    if mode == "edit_image"
        && args.source_asset_id.is_none()
        && args
            .reference_asset_ids
            .as_deref()
            .map_or(true, |ids| ids.is_empty())
    {
        return Err(
            "edit_image mode requires a sourceAssetId (or referenceAssetIds for a \
             multi-reference edit)"
                .to_owned(),
        );
    }
    let mut body = Map::new();
    body.insert("projectId".to_owned(), json!(args.project_id));
    body.insert("mode".to_owned(), json!(mode));
    body.insert("prompt".to_owned(), json!(args.prompt));
    body.insert("count".to_owned(), json!(args.count.unwrap_or(1)));
    let optional = [
        (
            "negativePrompt",
            args.negative_prompt.as_ref().map(|v| json!(v)),
        ),
        ("model", args.model.as_ref().map(|v| json!(v))),
        ("seed", args.seed.map(|v| json!(v))),
        ("width", args.width.map(|v| json!(v))),
        ("height", args.height.map(|v| json!(v))),
        ("stylePreset", args.style_preset.as_ref().map(|v| json!(v))),
        ("loras", args.loras.as_ref().map(|v| json!(v))),
        ("characterId", args.character_id.as_ref().map(|v| json!(v))),
        (
            "sourceAssetId",
            args.source_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "referenceAssetId",
            args.reference_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "referenceAssetIds",
            args.reference_asset_ids.as_ref().map(|v| json!(v)),
        ),
        ("maskAssetId", args.mask_asset_id.as_ref().map(|v| json!(v))),
    ];
    for (key, value) in optional {
        if let Some(value) = value {
            body.insert(key.to_owned(), value);
        }
    }
    Ok(Value::Object(body))
}

/// Every `mode` string [`video_job_body`] accepts, in the order the menu presents them (sc-19576).
///
/// It exists so the unsupported-mode error can BUILD its menu instead of restating it: the
/// hand-typed menu was already one mode stale when this const was added, which is the same
/// "a message asserts what the code does not do" defect the five missing modes were.
///
/// It is NOT the source of truth for what is reachable — the `match` below is, and its arms are
/// what `video_job_body_reaches_every_mode_the_api_admits` drives to collect the emitted wire
/// modes. A name here with no arm would be an invisible lie, so that test calls the real mapper
/// for every entry rather than trusting this list.
pub(crate) const VIDEO_TOOL_MODES: &[&str] = &[
    "generate",
    "reference",
    "extend",
    "bridge",
    "person_replace",
    // sc-19576: the five wire modes `VIDEO_JOB_MODES` admitted and this tool could not reach. They
    // take their WIRE names rather than new task-level aliases — `generate` / `reference` /
    // `extend` / `bridge` / `person_replace` earned their aliases by being the shapes an agent asks
    // for in plain language, while these five have no shorter honest name than the mode itself, and
    // inventing one would put a second vocabulary between the agent and the `capabilities` array
    // `list_models` reports.
    "video_to_video",
    "reference_video_to_video",
    "multi_video_to_video",
    "ads2v",
    "animate_character",
];

/// Map `submit_video_job` args onto the `VideoJobRequest` wire shape (camelCase). The tool's
/// task-level modes map to the API's wire modes — `generate` picks `text_to_video` /
/// `image_to_video` / `first_last_frame` from the provided image assets — and the mode-specific
/// required assets are checked here so a bad call fails fast with a precise message instead of a
/// submitted-then-rejected job. Only provided optional fields are emitted so the API's serde
/// defaults stay authoritative.
///
/// Every required-asset check below mirrors the corresponding arm of `validate_video_job`
/// (`apps/rust-api/src/lib.rs`). Mirroring, not replacing: the API arm still runs, and the
/// per-model half (which media a given model actually takes, and how many) stays on the model's
/// declared caps at enqueue.
pub(crate) fn video_job_body(args: &SubmitVideoJobArgs) -> Result<Value, String> {
    let mode = match args.mode.as_deref().map(str::trim).unwrap_or("generate") {
        "" | "generate" => {
            if args.last_frame_asset_id.is_some() {
                if args.source_asset_id.is_none() {
                    return Err(
                        "generate mode with a lastFrameAssetId also needs a sourceAssetId \
                         (the first frame)"
                            .to_owned(),
                    );
                }
                "first_last_frame"
            } else if args.source_asset_id.is_some() {
                "image_to_video"
            } else {
                "text_to_video"
            }
        }
        // sc-17161 made `reference_to_video` reachable — it is Ref2VA's ONLY mode, and the mode
        // the `referenceAudioAssetIds` field sc-17160 added exists to feed, so the field was
        // reachable while the mode that consumes it was not.
        //
        // sc-19574 corrected WHICH sets are legal. sc-17161 mirrored `validate_video_job`'s
        // then-current "at least one reference of ANY kind", audio alone included. The reference
        // implementation says otherwise, in a docstring and in an assertion: diffusers `MiniMaxH3`
        // documents `MiniMaxH3AudioReference` as "never on its own … It never reaches the
        // conditioner", and `before_encoder.py` raises on `set(kinds) == {"audio"}`. An audio-only
        // set leaves the VISUAL stream unconditioned; the worker refuses it (sc-19508) and the API
        // now 400s it, so this refuses it first — which is the whole point of checking here.
        //
        // The rule is `sceneworks_core::video_request::classify_reference_set`, shared with the API
        // and the worker; only the wording is this tool's.
        "reference" => {
            let count = |ids: &Option<Vec<String>>| ids.as_deref().map_or(0, <[String]>::len);
            if classify_reference_set(
                count(&args.reference_asset_ids),
                count(&args.source_clip_asset_ids),
                count(&args.reference_audio_asset_ids),
            ) != ReferenceSetVerdict::Conditionable
            {
                return Err(
                    "reference mode requires at least one referenceAssetIds or sourceClipAssetIds \
                     entry — an audio reference conditions the soundtrack and cannot be the only \
                     reference"
                        .to_owned(),
                );
            }
            "reference_to_video"
        }
        "extend" => {
            if args.source_clip_asset_id.is_none() {
                return Err(
                    "extend mode requires a sourceClipAssetId (the clip to continue)".to_owned(),
                );
            }
            "extend_clip"
        }
        "bridge" => {
            if args.source_clip_asset_id.is_none() || args.bridge_right_clip_asset_id.is_none() {
                return Err(
                    "bridge mode requires a sourceClipAssetId (left clip) and a \
                     bridgeRightClipAssetId (right clip)"
                        .to_owned(),
                );
            }
            "video_bridge"
        }
        "person_replace" => {
            if args.source_clip_asset_id.is_none() {
                return Err(
                    "person_replace mode requires a sourceClipAssetId (the clip to edit)"
                        .to_owned(),
                );
            }
            if args.person_track_id.is_none() {
                return Err(
                    "person_replace mode requires a personTrackId (the person to replace)"
                        .to_owned(),
                );
            }
            if args.character_id.is_none() {
                return Err(
                    "person_replace mode requires a characterId (the replacement Character)"
                        .to_owned(),
                );
            }
            "replace_person"
        }
        // ---- sc-19576: the five modes `VIDEO_JOB_MODES` admitted and this tool could not reach.
        // Each carries the required-asset check from its own `validate_video_job` arm, so the tool
        // keeps its fail-fast contract instead of shipping a name whose every payload 400s — a
        // mode you can NAME but cannot submit a valid payload for is not reachable.
        "video_to_video" => {
            if args.source_clip_asset_id.is_none() {
                return Err(
                    "video_to_video mode requires a sourceClipAssetId (the clip to edit)"
                        .to_owned(),
                );
            }
            "video_to_video"
        }
        "reference_video_to_video" => {
            if args.source_clip_asset_id.is_none() {
                return Err(
                    "reference_video_to_video mode requires a sourceClipAssetId (the clip to edit)"
                        .to_owned(),
                );
            }
            if args
                .reference_asset_ids
                .as_deref()
                .map_or(true, <[String]>::is_empty)
            {
                return Err(
                    "reference_video_to_video mode requires at least one referenceAssetIds entry"
                        .to_owned(),
                );
            }
            "reference_video_to_video"
        }
        // mv2v blends the PLURAL `sourceClipAssetIds`, not the singular slot the edit modes use —
        // two fields with confusingly similar names, and the API counts the plural one.
        "multi_video_to_video" => {
            if args
                .source_clip_asset_ids
                .as_deref()
                .map_or(0, <[String]>::len)
                < 2
            {
                return Err(
                    "multi_video_to_video mode requires at least two sourceClipAssetIds entries \
                     (the clips to blend)"
                        .to_owned(),
                );
            }
            "multi_video_to_video"
        }
        "ads2v" => {
            if args.source_clip_asset_id.is_none() {
                return Err("ads2v mode requires a sourceClipAssetId (the clip to edit)".to_owned());
            }
            if args.reference_clip_asset_id.is_none() {
                return Err(
                    "ads2v mode requires a referenceClipAssetId (the reference video)".to_owned(),
                );
            }
            if args
                .reference_asset_ids
                .as_deref()
                .map_or(true, <[String]>::is_empty)
            {
                return Err("ads2v mode requires at least one referenceAssetIds entry".to_owned());
            }
            "ads2v"
        }
        // The character is `referenceAssetIds[0]` (preferred) or the i2v `sourceAssetId`; the
        // motion is `sourceClipAssetId`. Both halves are hard engine inputs.
        "animate_character" => {
            if args.source_clip_asset_id.is_none() {
                return Err(
                    "animate_character mode requires a sourceClipAssetId (the driving video)"
                        .to_owned(),
                );
            }
            if args
                .reference_asset_ids
                .as_deref()
                .map_or(true, <[String]>::is_empty)
                && args.source_asset_id.is_none()
            {
                return Err(
                    "animate_character mode requires a reference character image \
                     (referenceAssetIds or sourceAssetId)"
                        .to_owned(),
                );
            }
            "animate_character"
        }
        other => {
            // The menu is BUILT from `VIDEO_TOOL_MODES`, never retyped: the hand-written one was
            // already a mode stale, and a caller told to use a mode set that omits the one it needs
            // is worse off than one told nothing.
            let menu = VIDEO_TOOL_MODES
                .iter()
                .map(|mode| format!("\"{mode}\""))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!("unsupported mode \"{other}\": use {menu}"));
        }
    };
    let mut body = Map::new();
    body.insert("projectId".to_owned(), json!(args.project_id));
    body.insert("mode".to_owned(), json!(mode));
    body.insert("prompt".to_owned(), json!(args.prompt));
    let optional = [
        (
            "negativePrompt",
            args.negative_prompt.as_ref().map(|v| json!(v)),
        ),
        ("model", args.model.as_ref().map(|v| json!(v))),
        ("duration", args.duration.map(|v| json!(v))),
        ("fps", args.fps.map(|v| json!(v))),
        ("width", args.width.map(|v| json!(v))),
        ("height", args.height.map(|v| json!(v))),
        ("quality", args.quality.as_ref().map(|v| json!(v))),
        ("seed", args.seed.map(|v| json!(v))),
        ("loras", args.loras.as_ref().map(|v| json!(v))),
        ("characterId", args.character_id.as_ref().map(|v| json!(v))),
        (
            "sourceAssetId",
            args.source_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "lastFrameAssetId",
            args.last_frame_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "sourceClipAssetId",
            args.source_clip_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "bridgeRightClipAssetId",
            args.bridge_right_clip_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "referenceAssetIds",
            args.reference_asset_ids.as_ref().map(|v| json!(v)),
        ),
        (
            "sourceClipAssetIds",
            args.source_clip_asset_ids.as_ref().map(|v| json!(v)),
        ),
        (
            "referenceAudioAssetIds",
            args.reference_audio_asset_ids.as_ref().map(|v| json!(v)),
        ),
        (
            "referenceClipAssetId",
            args.reference_clip_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "personTrackId",
            args.person_track_id.as_ref().map(|v| json!(v)),
        ),
        (
            "replacementMode",
            args.replacement_mode.as_ref().map(|v| json!(v)),
        ),
    ];
    for (key, value) in optional {
        if let Some(value) = value {
            body.insert(key.to_owned(), value);
        }
    }
    // `steps` rides in `advanced`, which is a verbatim passthrough map — `requested_steps` reads
    // `advanced.steps` and nothing else, so a top-level key would be silently dropped. Emitted
    // only when the caller named one, so an omitted step count still means "the model's own
    // default" rather than a blanket this tool invented (the same rule the optional keys follow).
    if let Some(steps) = args.steps {
        body.insert("advanced".to_owned(), json!({ "steps": steps }));
    }
    Ok(Value::Object(body))
}

/// The compact, job-type-agnostic status view of a JobSnapshot the polling
/// tools return: identity + status/stage/progress/eta plus the error when the
/// job failed. Works identically for video and image jobs.
pub(crate) fn compact_job_status(job: &Value) -> Value {
    let mut out = Map::new();
    if let Some(id) = job.get("id").filter(|id| !id.is_null()) {
        out.insert("jobId".to_owned(), id.clone());
    }
    copy_keys(
        job,
        &[
            "type",
            "status",
            "projectId",
            "stage",
            "message",
            "etaSeconds",
            "elapsedSeconds",
            "error",
            "createdAt",
            "completedAt",
        ],
        &mut out,
    );
    // Drop empty message/stage strings — they carry no information for an LLM.
    for key in ["message", "stage"] {
        if out.get(key).and_then(Value::as_str) == Some("") {
            out.remove(key);
        }
    }
    let (percent, _) = job_progress(job);
    out.insert("progressPercent".to_owned(), json!(percent));
    Value::Object(out)
}

/// A job's failure detail for error surfaces (failed status), with a stable
/// fallback when the worker recorded nothing.
pub(crate) fn job_error_detail(job: &Value) -> &str {
    job.get("error")
        .and_then(Value::as_str)
        .filter(|error| !error.is_empty())
        .unwrap_or("the worker reported no error detail")
}

/// Validate a caller-supplied job id before splicing it into a `/api/v1/jobs/…`
/// path: SceneWorks ids are `job_<hex uuid>`-shaped, so anything outside
/// `[A-Za-z0-9_-]` (path separators, query metacharacters …) is rejected.
pub(crate) fn valid_job_id(job_id: &str) -> Result<&str, String> {
    let job_id = job_id.trim();
    if job_id.is_empty()
        || !job_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        return Err(format!(
            "\"{job_id}\" is not a valid job id (expected letters, digits, '-' or '_')"
        ));
    }
    Ok(job_id)
}

/// Best-effort mime type for a result download link: the asset sidecar's
/// recorded `file.mimeType` wins, then the file extension; `None` (omit the
/// field) when neither identifies the media — a link stays useful without one.
pub(crate) fn media_mime_type(path: &str, sidecar_mime: Option<&str>) -> Option<String> {
    if let Some(mime) = sidecar_mime.map(str::trim).filter(|mime| !mime.is_empty()) {
        return Some(mime.to_owned());
    }
    let extension = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    let mime = match extension.as_str() {
        "mp4" | "m4v" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        _ => return None,
    };
    Some(mime.to_owned())
}

/// The absolute base (`scheme://authority`) a returned ticket URL should use so
/// the CALLING client can fetch it (sc-10290): prefer what the client actually
/// used to reach `/mcp` — `X-Forwarded-Host` then `Host` for the authority (with
/// the request-target authority as a last resort), and `X-Forwarded-Proto` then
/// the request scheme (default `http`) for the scheme. Returns `None` when no
/// authority is available, so the caller falls back to the configured API base.
pub(crate) fn request_base_url(
    headers: &http::HeaderMap,
    uri: &http::Uri,
    allowed_hosts: &[String],
) -> Option<String> {
    if let Some(raw_forwarded) = header_str(headers, "x-forwarded-host") {
        let authority = first_csv(raw_forwarded);
        if !crate::forwarded_authority_is_allowed(authority, allowed_hosts) {
            // Deliberately fall back to SCENEWORKS_API_URL rather than silently
            // reflecting the proxy's internal Host when its public authority is
            // not configured.
            return None;
        }
        let scheme = header_str(headers, "x-forwarded-proto")
            .map(first_csv)
            .map(str::to_ascii_lowercase)
            .filter(|scheme| matches!(scheme.as_str(), "http" | "https"))?;
        return Some(format!("{scheme}://{authority}"));
    }

    let raw_authority = header_str(headers, "host")
        .or_else(|| uri.authority().map(http::uri::Authority::as_str))?;
    let authority = http::uri::Authority::try_from(raw_authority.trim()).ok()?;
    let scheme = uri
        .scheme_str()
        .filter(|scheme| matches!(*scheme, "http" | "https"))
        .unwrap_or("http");
    Some(format!("{scheme}://{authority}"))
}

/// A request header as a trimmed, non-empty `&str` (ASCII values only).
fn header_str<'a>(headers: &'a http::HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// The first entry of a possibly comma-separated forwarded header, trimmed — a
/// proxy chain sets e.g. `X-Forwarded-Host: client-host, inner-host`.
fn first_csv(value: &str) -> &str {
    value.split(',').next().unwrap_or(value).trim()
}

/// Keep an asset for the inline result: image-typed (or untyped legacy) records.
fn is_image_asset(asset: &Value) -> bool {
    match asset.get("type").and_then(Value::as_str) {
        Some(media_type) => media_type == "image",
        None => true,
    }
}

/// Percent-encode each segment of a project-relative media path for splicing into
/// a `/files/*relative_path` URL, preserving the `/` separators (sc-10279).
/// Generated media paths are slug-safe today, so this is byte-identical for them;
/// it defends a future path segment containing a space, `%`, `#`, or `?` from
/// silently mis-resolving. The set is the URL path percent-encode set plus `%`
/// itself (the input is a raw filesystem path, not an already-escaped URL).
pub(crate) fn encode_media_path(path: &str) -> String {
    path.split('/')
        .map(encode_path_segment)
        .collect::<Vec<_>>()
        .join("/")
}

/// Project identifiers are filesystem and URL route identifiers, not arbitrary path segments.
/// Match the core store's safe-id contract before interpolating a job-supplied id into either the
/// inline file request or a ticket-bearing result URL. Rejecting `%` also prevents a downstream
/// decoder from turning single- or double-encoded traversal into route separators.
pub(crate) fn valid_project_id(project_id: &str) -> bool {
    !project_id.trim().is_empty()
        && project_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

/// Percent-encode one untrusted media-path segment.
fn encode_path_segment(segment: &str) -> String {
    use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
    const SEGMENT: &AsciiSet = &CONTROLS
        .add(b' ')
        .add(b'"')
        .add(b'#')
        .add(b'%')
        .add(b'/')
        .add(b'<')
        .add(b'>')
        .add(b'?')
        .add(b'`')
        .add(b'{')
        .add(b'}');
    utf8_percent_encode(segment, SEGMENT).to_string()
}

/// The project-relative media path of a result asset, normalized for the
/// `/files/*relative_path` route: prefers the sidecar's `file.path`, falls back
/// to a top-level `path`, converts backslashes and strips any leading slash.
pub(crate) fn asset_media_path(asset: &Value) -> Option<String> {
    let path = asset
        .pointer("/file/path")
        .and_then(Value::as_str)
        .or_else(|| asset.get("path").and_then(Value::as_str))?;
    let normalized = path.replace('\\', "/");
    let trimmed = normalized.trim_start_matches('/');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// The mime type for an inline image block: the asset sidecar's recorded
/// `file.mimeType` wins, then the file extension, then the file route's
/// `Content-Type` header; `image/png` as the final fallback (the worker's own
/// default). Non-image values are skipped — an ImageContent block must be
/// renderable.
pub(crate) fn image_mime_type(
    path: &str,
    sidecar_mime: Option<&str>,
    header_mime: Option<&str>,
) -> String {
    if let Some(mime) = sidecar_mime.filter(|mime| mime.starts_with("image/")) {
        return mime.to_owned();
    }
    let extension = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    let from_extension = match extension.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        "bmp" => Some("image/bmp"),
        _ => None,
    };
    if let Some(mime) = from_extension {
        return mime.to_owned();
    }
    if let Some(mime) = header_mime.filter(|mime| mime.starts_with("image/")) {
        return mime.to_owned();
    }
    "image/png".to_owned()
}

/// (percent 0..=100, human message) for a JobSnapshot poll. `progress` is the
/// contract's 0..1 fraction; the message is "stage" or "stage: message".
pub(crate) fn job_progress(job: &Value) -> (u32, String) {
    let fraction = job
        .get("progress")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let percent = (fraction * 100.0).round() as u32;
    let stage = job
        .get("stage")
        .and_then(Value::as_str)
        .filter(|stage| !stage.is_empty())
        .unwrap_or("queued");
    let message = job
        .get("message")
        .and_then(Value::as_str)
        .filter(|message| !message.is_empty());
    let message = match message {
        Some(detail) => format!("{stage}: {detail}"),
        None => stage.to_owned(),
    };
    (percent, message)
}

/// Map an API array response item-by-item; anything non-array (defensive — the
/// routes today always return arrays) passes through unchanged so a future shape
/// change degrades to "verbose" rather than "wrong".
fn compact_array(value: &Value, compact_one: impl Fn(&Value) -> Value) -> Value {
    match value.as_array() {
        Some(items) => Value::Array(items.iter().map(compact_one).collect()),
        None => value.clone(),
    }
}

/// Copy the given top-level keys, skipping absent/null ones.
fn copy_keys(item: &Value, keys: &[&str], out: &mut Map<String, Value>) {
    for key in keys {
        if let Some(value) = item.get(*key).filter(|value| !value.is_null()) {
            out.insert((*key).to_owned(), value.clone());
        }
    }
}

pub(crate) fn compact_projects(projects: &Value) -> Value {
    compact_array(projects, |project| {
        let mut out = Map::new();
        copy_keys(project, &["id", "name", "createdAt"], &mut out);
        Value::Object(out)
    })
}

pub(crate) fn compact_models(models: &Value) -> Value {
    compact_array(models, |model| {
        let mut out = Map::new();
        copy_keys(
            model,
            &[
                "id",
                "name",
                "family",
                "type",
                "capabilities",
                "installState",
                "gated",
                "defaults",
            ],
            &mut out,
        );
        // The menus and floors a caller needs to build a request the enqueue gate ACCEPTS.
        // Resolutions were the only one carried before sc-17161, and that was survivable while
        // every video model took a continuous duration range. MiniMax-H3 is not that shape: its
        // fourteen `17n + 5` rungs are the only renderable lengths, its fps is a one-entry menu,
        // and its step floor is 2 — so an agent that could not see them would guess "6 seconds,
        // 25 fps" and get a 400 for every call. Sampler/scheduler menus and the download/footprint
        // blocks stay on the full API response; these are the request-shaping keys.
        for (key, pointer) in [
            ("resolutions", "/limits/resolutions"),
            ("durations", "/limits/durations"),
            ("fps", "/limits/fps"),
            ("minSteps", "/limits/hardMinSteps"),
            ("maxReferenceAssets", "/limits/maxReferenceAssets"),
            ("maxSourceClipAssets", "/limits/maxSourceClipAssets"),
            ("maxReferenceAudioAssets", "/limits/maxReferenceAudioAssets"),
            (
                "maxCombinedReferenceAssets",
                "/limits/maxCombinedReferenceAssets",
            ),
        ] {
            if let Some(value) = model.pointer(pointer) {
                out.insert(key.to_owned(), value.clone());
            }
        }
        // Licence-required UI attribution (sc-17227 §IV.2 / sc-17161). An MCP client renders the
        // model it used in ITS OWN interface, so the string has to travel with the catalog entry
        // rather than living only on SceneWorks' own screens.
        if let Some(attribution) = model.pointer("/ui/attribution") {
            out.insert("attribution".to_owned(), attribution.clone());
        }
        // Where the model's prompting guidance lives. Prompt shape is not generic across video
        // families — MiniMax-H3 generates its soundtrack from the same prompt, so an unprompted
        // soundscape is invented rather than absent — and an agent has no other way to find out.
        if let Some(guide) = model.pointer("/ui/promptGuide") {
            out.insert("promptGuide".to_owned(), guide.clone());
        }
        // Which LoRA families this model accepts — pairs with list_loras.
        if let Some(families) = model.pointer("/loraCompatibility/families") {
            out.insert("loraFamilies".to_owned(), families.clone());
        }
        Value::Object(out)
    })
}

pub(crate) fn compact_loras(loras: &Value) -> Value {
    compact_array(loras, |lora| {
        let mut out = Map::new();
        copy_keys(
            lora,
            &[
                "id",
                "name",
                "family",
                "triggerWords",
                "defaultWeight",
                "installState",
                // sc-18727 — the sampling-regime marker and the recipe it carries. Without these an
                // agent listing LoRAs sees "MiniMax-H3 Turbo (4-step)" as a name and nothing else:
                // it cannot tell that selecting it takes the render from the model's `defaults.steps`
                // of 50 to 4 (a measured 2.42 h to 12.6 min), nor which of the three published
                // variants runs which schedule. That is the same class of gap sc-17161 closed for
                // `steps` — a knob reachable through `generate_video`'s `loras` field but invisible
                // in the catalog that is supposed to describe it. `role` alone is not enough,
                // because the numbers differ per adapter and no naming convention carries the shift.
                "role",
                "sampling",
            ],
            &mut out,
        );
        if let Some(families) = lora.pointer("/compatibility/families") {
            out.insert("compatibleFamilies".to_owned(), families.clone());
        }
        Value::Object(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeSet;

    #[test]
    fn compact_projects_keeps_only_id_name_created_at() {
        let full = json!([{
            "id": "p1",
            "name": "My Film",
            "path": "/data/projects/p1",
            "createdAt": "2026-07-07T00:00:00Z"
        }]);
        assert_eq!(
            compact_projects(&full),
            json!([{ "id": "p1", "name": "My Film", "createdAt": "2026-07-07T00:00:00Z" }])
        );
    }

    #[test]
    fn compact_models_keeps_job_request_fields_and_flattens_menus() {
        let full = json!([{
            "id": "z_image_turbo",
            "name": "Z-Image-Turbo",
            "family": "z-image",
            "type": "image",
            "capabilities": ["text_to_image"],
            "installState": "installed",
            "gated": false,
            "defaults": { "resolution": "1024x1024", "steps": 8, "guidanceScale": 0, "count": 4 },
            "limits": {
                "resolutions": ["768x768", "1024x1024"],
                "samplers": ["default", "euler"]
            },
            "loraCompatibility": { "families": ["z-image"], "types": ["style"] },
            // Verbose catalog fields that must be dropped:
            "downloads": [{ "repo": "SceneWorks/z-image-turbo-mlx", "files": ["q4/*"] }],
            "mlx": { "minMemoryGb": 40 },
            "candle": { "minMemoryGb": 40 }
        }]);
        assert_eq!(
            compact_models(&full),
            json!([{
                "id": "z_image_turbo",
                "name": "Z-Image-Turbo",
                "family": "z-image",
                "type": "image",
                "capabilities": ["text_to_image"],
                "installState": "installed",
                "gated": false,
                "defaults": { "resolution": "1024x1024", "steps": 8, "guidanceScale": 0, "count": 4 },
                "resolutions": ["768x768", "1024x1024"],
                "loraFamilies": ["z-image"]
            }])
        );
    }

    #[test]
    fn compact_loras_keeps_trigger_and_compatibility_fields() {
        let full = json!([{
            "id": "ltx_2_3_ic_hdr",
            "name": "LTX-2.3 IC-LoRA HDR",
            "family": "ltx-video",
            "triggerWords": [],
            "compatibility": { "families": ["ltx-video"] },
            "icLora": true,
            "defaultWeight": 0.8,
            "installState": "missing",
            "source": { "provider": "huggingface", "repo": "Lightricks/x", "file": "y.safetensors" }
        }]);
        assert_eq!(
            compact_loras(&full),
            json!([{
                "id": "ltx_2_3_ic_hdr",
                "name": "LTX-2.3 IC-LoRA HDR",
                "family": "ltx-video",
                "triggerWords": [],
                "defaultWeight": 0.8,
                "installState": "missing",
                "compatibleFamilies": ["ltx-video"]
            }])
        );
    }

    /// sc-18727 — a step-distill adapter's REGIME (`role`) and its RECIPE (`sampling`) survive the
    /// compaction, because an agent that cannot see them cannot tell a 4-step render from a 50-step
    /// one before submitting it.
    ///
    /// Asserted with a NON-DEFAULT recipe (4 steps at video shift 6.0, against the model's
    /// `defaults.steps: 50` and the engine's own shift 12.0) so the arm cannot pass against a
    /// mapper that emits placeholder or default values. The negative half is the arm above: a plain
    /// LoRA carries neither key and gains neither.
    #[test]
    fn compact_loras_keeps_the_accelerator_role_and_its_sampling_recipe() {
        let full = json!([{
            "id": "minimax_h3_turbo_4step_768p",
            "name": "MiniMax-H3 Turbo (4-step, 768p)",
            "family": "minimax-h3",
            "role": "accelerator",
            "sampling": { "steps": 4, "schedulerShift": 6.0, "audioSchedulerShift": 3.0 },
            "defaultWeight": 1.0,
            "installState": "installed",
            "compatibility": { "families": ["minimax-h3"] },
            "source": { "provider": "huggingface", "repo": "lightx2v/Minimax-h3-Turbo" }
        }]);
        assert_eq!(
            compact_loras(&full),
            json!([{
                "id": "minimax_h3_turbo_4step_768p",
                "name": "MiniMax-H3 Turbo (4-step, 768p)",
                "family": "minimax-h3",
                "defaultWeight": 1.0,
                "installState": "installed",
                "role": "accelerator",
                "sampling": { "steps": 4, "schedulerShift": 6.0, "audioSchedulerShift": 3.0 },
                "compatibleFamilies": ["minimax-h3"]
            }])
        );
    }

    #[test]
    fn compact_mappers_skip_absent_and_null_fields() {
        let sparse = json!([{ "id": "m1", "name": null }]);
        assert_eq!(compact_models(&sparse), json!([{ "id": "m1" }]));
        assert_eq!(compact_loras(&sparse), json!([{ "id": "m1" }]));
    }

    #[test]
    fn compact_mappers_pass_non_arrays_through() {
        // Defensive: an unexpected shape must degrade to verbose, not panic/lie.
        let detail = json!({ "detail": "unexpected" });
        assert_eq!(compact_projects(&detail), detail);
        assert_eq!(compact_models(&detail), detail);
        assert_eq!(compact_loras(&detail), detail);
    }

    // -----------------------------------------------------------------------
    // generate_image (sc-10234): args → ImageJobRequest mapping.
    // -----------------------------------------------------------------------

    fn args_from(value: Value) -> GenerateImageArgs {
        serde_json::from_value(value).expect("args deserialize")
    }

    #[test]
    fn image_job_body_maps_every_optional_field() {
        let args = args_from(json!({
            "projectId": "p1",
            "prompt": "a city at night",
            "mode": "edit_image",
            "negativePrompt": "blurry",
            "model": "z_image_turbo",
            "count": 3,
            "seed": 42,
            "width": 1280,
            "height": 768,
            "stylePreset": "photoreal",
            "loras": [{ "id": "lora1", "weight": 0.8 }],
            "characterId": "char1",
            "sourceAssetId": "asset_src",
            "referenceAssetId": "asset_ref",
            "referenceAssetIds": ["asset_r1", "asset_r2"],
            "maskAssetId": "asset_mask"
        }));
        assert_eq!(
            image_job_body(&args).expect("body builds"),
            json!({
                "projectId": "p1",
                "mode": "edit_image",
                "prompt": "a city at night",
                "count": 3,
                "negativePrompt": "blurry",
                "model": "z_image_turbo",
                "seed": 42,
                "width": 1280,
                "height": 768,
                "stylePreset": "photoreal",
                "loras": [{ "id": "lora1", "weight": 0.8 }],
                "characterId": "char1",
                "sourceAssetId": "asset_src",
                "referenceAssetId": "asset_ref",
                "referenceAssetIds": ["asset_r1", "asset_r2"],
                "maskAssetId": "asset_mask"
            })
        );
    }

    #[test]
    fn image_job_body_minimal_defaults_to_text_to_image_count_1() {
        // Absent optionals must be OMITTED (the API's serde defaults are
        // authoritative), except count where the MCP default is 1.
        let args = args_from(json!({ "projectId": "p1", "prompt": "hi" }));
        assert_eq!(
            image_job_body(&args).expect("body builds"),
            json!({
                "projectId": "p1",
                "mode": "text_to_image",
                "prompt": "hi",
                "count": 1
            })
        );
    }

    #[test]
    fn image_job_body_maps_generate_mode_to_text_to_image() {
        let args = args_from(json!({ "projectId": "p1", "prompt": "hi", "mode": "generate" }));
        let body = image_job_body(&args).expect("body builds");
        assert_eq!(body["mode"], "text_to_image");
    }

    #[test]
    fn image_job_body_rejects_unknown_mode() {
        let args =
            args_from(json!({ "projectId": "p1", "prompt": "hi", "mode": "style_variations" }));
        let error = image_job_body(&args).expect_err("unknown mode rejected");
        assert!(error.contains("style_variations"), "{error}");
    }

    #[test]
    fn image_job_body_rejects_edit_without_a_source() {
        let args = args_from(json!({ "projectId": "p1", "prompt": "hi", "mode": "edit_image" }));
        let error = image_job_body(&args).expect_err("sourceless edit rejected");
        assert!(error.contains("sourceAssetId"), "{error}");

        // ... but a multi-reference edit (no sourceAssetId) is valid.
        let args = args_from(json!({
            "projectId": "p1",
            "prompt": "hi",
            "mode": "edit_image",
            "referenceAssetIds": ["asset_r1"]
        }));
        assert!(image_job_body(&args).is_ok());
    }

    // -----------------------------------------------------------------------
    // generate_image (sc-10234): result asset → file fetch mapping.
    // -----------------------------------------------------------------------

    #[test]
    fn asset_media_path_prefers_file_path_and_normalizes() {
        // The persisted sidecar shape: the path lives at file.path.
        let sidecar = json!({ "id": "a1", "file": { "path": "assets/images/g1/img_0001.png" } });
        assert_eq!(
            asset_media_path(&sidecar).as_deref(),
            Some("assets/images/g1/img_0001.png")
        );
        // Fallback to a top-level path; backslashes and a leading slash normalize.
        let flat = json!({ "path": "\\assets\\images\\g1\\img_0001.png" });
        assert_eq!(
            asset_media_path(&flat).as_deref(),
            Some("assets/images/g1/img_0001.png")
        );
        assert_eq!(asset_media_path(&json!({ "id": "a1" })), None);
        assert_eq!(asset_media_path(&json!({ "path": "/" })), None);
    }

    #[test]
    fn image_mime_type_prefers_sidecar_then_extension_then_header() {
        // Sidecar mimeType wins.
        assert_eq!(
            image_mime_type("assets/x.png", Some("image/webp"), Some("image/gif")),
            "image/webp"
        );
        // A non-image sidecar value is ignored; extension decides.
        assert_eq!(
            image_mime_type("assets/x.jpg", Some("application/json"), None),
            "image/jpeg"
        );
        assert_eq!(image_mime_type("assets/x.PNG", None, None), "image/png");
        // No sidecar/extension signal → the response header.
        assert_eq!(
            image_mime_type("assets/x.bin", None, Some("image/webp")),
            "image/webp"
        );
        // Nothing usable → the worker's png default.
        assert_eq!(
            image_mime_type("assets/x.bin", None, Some("text/html")),
            "image/png"
        );
    }

    // -----------------------------------------------------------------------
    // submit_video_job (sc-10235): args → VideoJobRequest mapping, all four
    // tool modes + their mode-specific required fields.
    // -----------------------------------------------------------------------

    fn video_args_from(value: Value) -> SubmitVideoJobArgs {
        serde_json::from_value(value).expect("video args deserialize")
    }

    #[test]
    fn video_job_body_minimal_generate_is_text_to_video() {
        // Absent optionals must be OMITTED — the API's serde defaults
        // (duration 6, fps 25, 768x512, ltx_2_3 …) are authoritative.
        let args = video_args_from(json!({ "projectId": "p1", "prompt": "a storm" }));
        assert_eq!(
            video_job_body(&args).expect("body builds"),
            json!({ "projectId": "p1", "mode": "text_to_video", "prompt": "a storm" })
        );
    }

    #[test]
    fn video_job_body_generate_with_source_image_is_image_to_video() {
        let args = video_args_from(json!({
            "projectId": "p1",
            "prompt": "a storm",
            "mode": "generate",
            "sourceAssetId": "img_1"
        }));
        let body = video_job_body(&args).expect("body builds");
        assert_eq!(body["mode"], "image_to_video");
        assert_eq!(body["sourceAssetId"], "img_1");
    }

    #[test]
    fn video_job_body_generate_with_both_frames_is_first_last_frame() {
        let args = video_args_from(json!({
            "projectId": "p1",
            "prompt": "a storm",
            "sourceAssetId": "img_first",
            "lastFrameAssetId": "img_last"
        }));
        let body = video_job_body(&args).expect("body builds");
        assert_eq!(body["mode"], "first_last_frame");
        assert_eq!(body["sourceAssetId"], "img_first");
        assert_eq!(body["lastFrameAssetId"], "img_last");

        // A last frame without a first frame is ambiguous → rejected.
        let args = video_args_from(json!({
            "projectId": "p1",
            "prompt": "a storm",
            "lastFrameAssetId": "img_last"
        }));
        let error = video_job_body(&args).expect_err("last frame alone rejected");
        assert!(error.contains("sourceAssetId"), "{error}");
    }

    #[test]
    fn video_job_body_extend_requires_and_threads_the_source_clip() {
        let args =
            video_args_from(json!({ "projectId": "p1", "prompt": "keep going", "mode": "extend" }));
        let error = video_job_body(&args).expect_err("clipless extend rejected");
        assert!(error.contains("sourceClipAssetId"), "{error}");

        let args = video_args_from(json!({
            "projectId": "p1",
            "prompt": "keep going",
            "mode": "extend",
            "sourceClipAssetId": "clip_1"
        }));
        let body = video_job_body(&args).expect("body builds");
        assert_eq!(body["mode"], "extend_clip");
        assert_eq!(body["sourceClipAssetId"], "clip_1");
    }

    #[test]
    fn video_job_body_bridge_requires_both_clips() {
        let args = video_args_from(json!({
            "projectId": "p1",
            "prompt": "bridge",
            "mode": "bridge",
            "sourceClipAssetId": "clip_left"
        }));
        let error = video_job_body(&args).expect_err("one-sided bridge rejected");
        assert!(error.contains("bridgeRightClipAssetId"), "{error}");

        let args = video_args_from(json!({
            "projectId": "p1",
            "prompt": "bridge",
            "mode": "bridge",
            "sourceClipAssetId": "clip_left",
            "bridgeRightClipAssetId": "clip_right"
        }));
        let body = video_job_body(&args).expect("body builds");
        assert_eq!(body["mode"], "video_bridge");
        assert_eq!(body["sourceClipAssetId"], "clip_left");
        assert_eq!(body["bridgeRightClipAssetId"], "clip_right");
    }

    #[test]
    fn video_job_body_person_replace_requires_clip_track_and_character() {
        let base = json!({ "projectId": "p1", "prompt": "swap", "mode": "person_replace" });

        let error = video_job_body(&video_args_from(base.clone()))
            .expect_err("clipless person_replace rejected");
        assert!(error.contains("sourceClipAssetId"), "{error}");

        let mut with_clip = base.clone();
        with_clip["sourceClipAssetId"] = json!("clip_1");
        let error = video_job_body(&video_args_from(with_clip.clone()))
            .expect_err("trackless person_replace rejected");
        assert!(error.contains("personTrackId"), "{error}");

        with_clip["personTrackId"] = json!("track_1");
        let error = video_job_body(&video_args_from(with_clip.clone()))
            .expect_err("characterless person_replace rejected");
        assert!(error.contains("characterId"), "{error}");

        with_clip["characterId"] = json!("char_1");
        with_clip["replacementMode"] = json!("full_body");
        let body = video_job_body(&video_args_from(with_clip)).expect("body builds");
        assert_eq!(body["mode"], "replace_person");
        assert_eq!(body["sourceClipAssetId"], "clip_1");
        assert_eq!(body["personTrackId"], "track_1");
        assert_eq!(body["characterId"], "char_1");
        assert_eq!(body["replacementMode"], "full_body");
    }

    /// sc-17160: the audio references reach the request body verbatim, and are OMITTED when the
    /// caller names none rather than being sent as an empty array.
    ///
    /// The omission matters: `video_job_body` only inserts keys the caller supplied, and the API's
    /// DTO defaults an absent `referenceAudioAssetIds` to empty. Sending `[]` unconditionally would
    /// be indistinguishable in behaviour but would put a key on every MCP video request that names
    /// no audio, which is not what the other optional fields do.
    #[test]
    fn video_job_body_carries_reference_audio_asset_ids() {
        let args = video_args_from(json!({
            "projectId": "p1",
            "prompt": "the subject speaks",
            "referenceAudioAssetIds": ["asset_voice", "asset_music"]
        }));
        let body = video_job_body(&args).expect("body builds");
        assert_eq!(
            body["referenceAudioAssetIds"],
            json!(["asset_voice", "asset_music"])
        );

        let without = video_args_from(json!({ "projectId": "p1", "prompt": "a storm" }));
        let body = video_job_body(&without).expect("body builds");
        assert!(
            body.get("referenceAudioAssetIds").is_none(),
            "an unnamed optional field stays off the body: {body}"
        );
    }

    /// sc-17161: `reference` is a real mode, not a synonym for `generate`.
    ///
    /// `reference_to_video` is Ref2VA's ONLY mode, and the mode sc-17160's `referenceAudioAssetIds`
    /// field exists to feed — the field was on the tool while the single mode that consumes it was
    /// not, so an agent could set audio references and never render with them.
    ///
    /// All three reference kinds together is the shape the checkpoint serves; the audio kind is
    /// a COMPANION, and an audio-only set is refused (sc-19574, asserted below).
    #[test]
    fn video_job_body_reference_mode_carries_all_three_reference_kinds() {
        let args = video_args_from(json!({
            "projectId": "p1",
            "prompt": "the woman from <Picture 1> speaks with the voice from <Audio 1>",
            "mode": "reference",
            "referenceAssetIds": ["img_1", "img_2"],
            "sourceClipAssetIds": ["clip_1"],
            "referenceAudioAssetIds": ["aud_1"],
            "model": "minimax_h3_ref"
        }));
        let body = video_job_body(&args).expect("body builds");
        assert_eq!(body["mode"], "reference_to_video");
        assert_eq!(body["referenceAssetIds"], json!(["img_1", "img_2"]));
        assert_eq!(body["sourceClipAssetIds"], json!(["clip_1"]));
        assert_eq!(body["referenceAudioAssetIds"], json!(["aud_1"]));
        assert_eq!(body["model"], "minimax_h3_ref");
    }

    /// The gate is "at least one VISUAL reference" — an image or a video clip — with audio as a
    /// companion (sc-19574). It delegates to
    /// `sceneworks_core::video_request::classify_reference_set`, the same predicate
    /// `validate_video_job` and the worker read, so this asserts the tool's END of the shared rule.
    ///
    /// sc-17161 wrote this test the other way round, admitting an audio-only set, because
    /// `validate_video_job` did. The reference implementation settles it: diffusers `MiniMaxH3`
    /// documents `MiniMaxH3AudioReference` as "never on its own … It never reaches the conditioner"
    /// and `before_encoder.py` raises on `set(kinds) == {"audio"}`. An audio-only set leaves the
    /// visual stream unconditioned, so refusing it here is what stops an agent building a request
    /// the tool accepts and the worker then rejects.
    #[test]
    fn video_job_body_reference_mode_needs_a_visual_reference_not_audio_alone() {
        let empty =
            video_args_from(json!({ "projectId": "p1", "prompt": "x", "mode": "reference" }));
        let error = video_job_body(&empty).expect_err("referenceless reference mode rejected");
        assert!(error.contains("referenceAssetIds"), "{error}");
        assert!(error.contains("sourceClipAssetIds"), "{error}");

        // An explicitly EMPTY list is the same as naming none — otherwise `[]` would slip past the
        // gate here and be refused by the API instead, which is the fail-fast this check exists for.
        let blank = video_args_from(json!({
            "projectId": "p1", "prompt": "x", "mode": "reference", "referenceAssetIds": []
        }));
        assert!(
            video_job_body(&blank).is_err(),
            "an empty list is not a reference"
        );

        // THE sc-19574 SHAPE. Audio alone, no image and no clip: the tool must refuse it, and the
        // message must be the audio one — an `is_err()` here would also be satisfied by the
        // no-references-at-all arm above, which is a different (and, for this caller, wrong) reason.
        let audio_only = video_args_from(json!({
            "projectId": "p1",
            "prompt": "the voice from <Audio 1>",
            "mode": "reference",
            "model": "minimax_h3_ref",
            "referenceAudioAssetIds": ["aud_1", "aud_2"]
        }));
        let error =
            video_job_body(&audio_only).expect_err("an audio-only reference set is refused");
        assert!(
            error.contains("cannot be the only reference"),
            "the refusal must name the audio rule, not the empty-set one: {error}"
        );

        // Each VISUAL kind alone is enough, and audio riding along with either is fine.
        for field in ["referenceAssetIds", "sourceClipAssetIds"] {
            let mut payload = json!({ "projectId": "p1", "prompt": "x", "mode": "reference" });
            payload[field] = json!(["only_one"]);
            let body = video_job_body(&video_args_from(payload.clone()))
                .unwrap_or_else(|error| panic!("{field} alone is a valid reference set: {error}"));
            assert_eq!(body["mode"], "reference_to_video");
            assert_eq!(body[field], json!(["only_one"]));

            payload["referenceAudioAssetIds"] = json!(["aud_1"]);
            let body = video_job_body(&video_args_from(payload)).unwrap_or_else(|error| {
                panic!("{field} + audio is a valid reference set: {error}")
            });
            assert_eq!(body["referenceAudioAssetIds"], json!(["aud_1"]));
        }
    }

    /// sc-17161: `steps` rides in `advanced`, the only place `requested_steps` reads it from.
    ///
    /// Without this the tool could submit nothing but the model's `defaults.steps` — 50 for
    /// MiniMax-H3, measured in HOURS at its default canvas — so the cheap draft an agent would
    /// reach for first was unreachable through MCP. A top-level `steps` key would be accepted by
    /// the DTO's `deny_unknown_fields`-free serde and then silently ignored, which is worse than
    /// a rejection.
    #[test]
    fn video_job_body_puts_steps_in_advanced_and_omits_it_otherwise() {
        let args = video_args_from(json!({
            "projectId": "p1", "prompt": "a storm", "steps": 4
        }));
        let body = video_job_body(&args).expect("body builds");
        assert_eq!(body["advanced"], json!({ "steps": 4 }));
        assert!(
            body.get("steps").is_none(),
            "a top-level steps key would be read by nothing: {body}"
        );

        let without = video_args_from(json!({ "projectId": "p1", "prompt": "a storm" }));
        let body = video_job_body(&without).expect("body builds");
        assert!(
            body.get("advanced").is_none(),
            "an unnamed step count still means the model's own default: {body}"
        );
    }

    /// The compacted catalog carries the facts a caller needs to build an ACCEPTED request.
    ///
    /// Resolutions alone were survivable while every video model took a continuous duration range.
    /// MiniMax-H3 is not that shape — fourteen `17n + 5` rungs, a one-entry fps menu and a 2-step
    /// floor — so a caller that cannot see the menus guesses an off-menu value and is refused (an
    /// off-menu fps, an out-of-bounds duration) or silently snapped up the frame lattice (an
    /// in-range off-menu duration) — either way, not the request it made.
    /// The reference caps are here for the same reason: MiniMax-H3 Ref2VA's per-list
    /// caps sum PAST its combined ceiling, so the combined number is not derivable from the others.
    #[test]
    fn compact_models_carries_the_request_shaping_limits_and_attribution() {
        let models = json!([{
            "id": "minimax_h3_ref",
            "name": "MiniMax-H3 References",
            "family": "minimax-h3",
            "type": "video",
            "capabilities": ["reference_to_video"],
            "installState": "installed",
            "defaults": { "duration": 5.1667, "fps": 24, "resolution": "1344x768", "steps": 50 },
            "limits": {
                "durations": [5.1667, 14.375],
                "fps": [24],
                "hardMinSteps": 2,
                "resolutions": ["1344x768"],
                "maxReferenceAssets": 9,
                "maxSourceClipAssets": 3,
                "maxReferenceAudioAssets": 3,
                "maxCombinedReferenceAssets": 12,
                "samplers": ["default"]
            },
            "ui": {
                "attribution": "Powered by MiniMax H3",
                "description": "…",
                "promptGuide": { "title": "MiniMax-H3 Prompt Guide", "path": "/prompt-guides/minimax-h3.md" }
            },
            "downloads": [{ "repo": "SceneWorks/minimax-h3-mlx" }]
        }]);
        let compacted = compact_models(&models);
        let entry = &compacted[0];

        assert_eq!(entry["durations"], json!([5.1667, 14.375]));
        assert_eq!(entry["fps"], json!([24]));
        assert_eq!(entry["minSteps"], 2);
        assert_eq!(entry["maxReferenceAssets"], 9);
        assert_eq!(entry["maxSourceClipAssets"], 3);
        assert_eq!(entry["maxReferenceAudioAssets"], 3);
        assert_eq!(entry["maxCombinedReferenceAssets"], 12);
        assert_eq!(entry["attribution"], "Powered by MiniMax H3");
        assert_eq!(entry["promptGuide"]["path"], "/prompt-guides/minimax-h3.md");
        // Still compact: the verbose blocks stay on the full API response.
        assert!(entry.get("downloads").is_none(), "{entry}");
        assert!(entry.get("limits").is_none(), "{entry}");
        assert!(entry.get("samplers").is_none(), "{entry}");

        // A model that declares none of it gains no keys — this is per-model, not a shape change.
        let bare =
            json!([{ "id": "svd", "type": "video", "limits": { "resolutions": ["1024x576"] } }]);
        let entry = &compact_models(&bare)[0];
        assert_eq!(entry["resolutions"], json!(["1024x576"]));
        for key in ["durations", "fps", "minSteps", "attribution", "promptGuide"] {
            assert!(
                entry.get(key).is_none(),
                "{key} must not be invented: {entry}"
            );
        }
    }

    #[test]
    fn video_job_body_rejects_unknown_mode() {
        let args =
            video_args_from(json!({ "projectId": "p1", "prompt": "x", "mode": "style_remix" }));
        let error = video_job_body(&args).expect_err("unknown mode rejected");
        assert!(error.contains("style_remix"), "{error}");
        // The menu the message offers must list every mode the tool actually serves, or a caller
        // is told to use a mode set that is missing the one it needs (sc-17161). It is BUILT from
        // `VIDEO_TOOL_MODES` now (sc-19576) rather than hand-typed, so this iterates the real list
        // instead of spot-checking three names — the spot check is what let it go stale.
        for mode in VIDEO_TOOL_MODES {
            assert!(
                error.contains(mode),
                "the menu omits `{mode}`, a mode the tool serves: {error}"
            );
        }
    }

    /// The minimal argument set each `VIDEO_TOOL_MODES` entry needs to build a body, so the
    /// reachability guard below drives the REAL mapper rather than asserting over a name list.
    /// `generate` appears three times because it is the one entry that picks its wire mode from the
    /// media it was handed.
    fn video_tool_mode_cases() -> Vec<(&'static str, Value)> {
        vec![
            ("generate", json!({})),
            ("generate", json!({ "sourceAssetId": "img_1" })),
            (
                "generate",
                json!({ "sourceAssetId": "img_1", "lastFrameAssetId": "img_2" }),
            ),
            ("reference", json!({ "referenceAssetIds": ["img_1"] })),
            ("extend", json!({ "sourceClipAssetId": "clip_1" })),
            (
                "bridge",
                json!({ "sourceClipAssetId": "clip_1", "bridgeRightClipAssetId": "clip_2" }),
            ),
            (
                "person_replace",
                json!({ "sourceClipAssetId": "clip_1", "personTrackId": "track_1", "characterId": "char_1" }),
            ),
            ("video_to_video", json!({ "sourceClipAssetId": "clip_1" })),
            (
                "reference_video_to_video",
                json!({ "sourceClipAssetId": "clip_1", "referenceAssetIds": ["img_1"] }),
            ),
            (
                "multi_video_to_video",
                json!({ "sourceClipAssetIds": ["clip_1", "clip_2"] }),
            ),
            (
                "ads2v",
                json!({
                    "sourceClipAssetId": "clip_1",
                    "referenceClipAssetId": "clip_2",
                    "referenceAssetIds": ["img_1"]
                }),
            ),
            (
                "animate_character",
                json!({ "sourceClipAssetId": "clip_1", "referenceAssetIds": ["img_1"] }),
            ),
        ]
    }

    /// **THE REACHABILITY GUARD (sc-19576).** Every mode the API's `VIDEO_JOB_MODES` allow-list
    /// admits is reachable from this tool, and nothing here trusts a comment or a count.
    ///
    /// Both sides are read from the real thing. The REACHABLE set is collected by CALLING
    /// `video_job_body` for every `VIDEO_TOOL_MODES` entry and recording the wire mode it actually
    /// emitted — so a name with no `match` arm, or an arm that emits the wrong string, is red. The
    /// ADMITTED set is parsed out of the shipped `apps/rust-api/src/lib.rs` bytes, because
    /// `VIDEO_JOB_MODES` is `pub(crate)` to that crate and this one cannot link it; retyping the
    /// twelve here would be a guard asserting against its own copy, which is precisely the
    /// false-green shape that let GH #2074 ship and let this tool sit at 7 of 12 unnoticed.
    ///
    /// The shipped comment that made this necessary claimed `reference_to_video` was "the ONLY one
    /// of the twelve this tool could not reach" while five others were also unreachable. A comment
    /// asserting completeness stops the next reader from checking; this does the checking.
    #[test]
    fn video_job_body_reaches_every_mode_the_api_admits() {
        let source = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/rust-api/src/lib.rs"),
        )
        .expect("the API crate's lib.rs is readable from the workspace");
        let start = source
            .find("pub(crate) const VIDEO_JOB_MODES: &[&str] = &[")
            .expect("VIDEO_JOB_MODES is declared in apps/rust-api/src/lib.rs");
        let body = &source[start..];
        let end = body.find("\n];").expect("VIDEO_JOB_MODES terminates");
        // Every quoted literal on the line, not the first. The per-line parser this replaces had
        // exactly one open-failure mode among the realistic edits: TWO entries on one source line
        // (`"a", "b",`). Rename, re-visibility, reformat, added or dropped entry all fail CLOSED
        // (the parse breaks or the set changes and the `assert_eq!` reds), but a second literal on
        // a taken line silently vanished from `admitted` — and if it also had no MCP arm it
        // vanished from `reachable` too, so both sets missed it and the guard stayed green. The
        // vacuity floor cannot catch that either once the list is longer than the floor. rustfmt
        // normally forbids that layout, which is why the risk was low and the fix is cheap.
        let admitted: BTreeSet<String> = body[..end]
            .lines()
            .skip(1)
            .filter(|line| !line.trim_start().starts_with("//"))
            .flat_map(|line| {
                let mut found = Vec::new();
                let mut rest = line;
                while let Some(open) = rest.find('"') {
                    let after = &rest[open + 1..];
                    let Some(close) = after.find('"') else { break };
                    found.push(after[..close].to_owned());
                    rest = &after[close + 1..];
                }
                found
            })
            .collect();
        assert!(
            admitted.len() >= 12,
            "only {} modes were parsed out of VIDEO_JOB_MODES — the parse is wrong and this guard \
             is vacuous: {admitted:?}",
            admitted.len()
        );

        let mut reachable: BTreeSet<String> = BTreeSet::new();
        for (mode, media) in video_tool_mode_cases() {
            let mut payload = json!({ "projectId": "p1", "prompt": "x", "mode": mode });
            payload
                .as_object_mut()
                .expect("payload object")
                .extend(media.as_object().expect("media object").clone());
            let body = video_job_body(&video_args_from(payload))
                .unwrap_or_else(|error| panic!("`{mode}` must build a body: {error}"));
            reachable.insert(
                body["mode"]
                    .as_str()
                    .expect("every body names a wire mode")
                    .to_owned(),
            );
        }

        assert_eq!(
            reachable, admitted,
            "the modes `submit_video_job` can emit no longer match the modes the API admits. A \
             mode in `admitted` and not in `reachable` is one an agent cannot ask for at all; one \
             in `reachable` and not in `admitted` would 400 on every call with \"Unsupported video \
             mode\"."
        );

        // Every entry of the tool's own menu must have produced something — a name in
        // `VIDEO_TOOL_MODES` with no case here would otherwise be silently untested.
        let covered: BTreeSet<&str> = video_tool_mode_cases()
            .into_iter()
            .map(|(mode, _)| mode)
            .collect();
        let declared: BTreeSet<&str> = VIDEO_TOOL_MODES.iter().copied().collect();
        assert_eq!(
            covered, declared,
            "every VIDEO_TOOL_MODES entry needs a case in `video_tool_mode_cases`"
        );
    }

    /// Each of the five modes sc-19576 added carries its own required-asset check, and each names
    /// the field it is missing. Asserted per-refusal rather than with `is_err()`: these calls each
    /// omit ONE field, so a coarse assertion would be satisfied by the wrong arm rejecting for the
    /// wrong reason — the exact false green this epic has already shipped once.
    #[test]
    fn the_five_added_modes_each_refuse_their_own_missing_media() {
        let cases: &[(&str, Value, &str)] = &[
            ("video_to_video", json!({}), "sourceClipAssetId"),
            (
                "reference_video_to_video",
                json!({ "referenceAssetIds": ["img_1"] }),
                "sourceClipAssetId",
            ),
            (
                "reference_video_to_video",
                json!({ "sourceClipAssetId": "clip_1" }),
                "referenceAssetIds",
            ),
            (
                "multi_video_to_video",
                json!({ "sourceClipAssetIds": ["clip_1"] }),
                "sourceClipAssetIds",
            ),
            (
                "ads2v",
                json!({ "referenceClipAssetId": "clip_2", "referenceAssetIds": ["img_1"] }),
                "sourceClipAssetId",
            ),
            (
                "ads2v",
                json!({ "sourceClipAssetId": "clip_1", "referenceAssetIds": ["img_1"] }),
                "referenceClipAssetId",
            ),
            (
                "ads2v",
                json!({ "sourceClipAssetId": "clip_1", "referenceClipAssetId": "clip_2" }),
                "referenceAssetIds",
            ),
            (
                "animate_character",
                json!({ "referenceAssetIds": ["img_1"] }),
                "sourceClipAssetId",
            ),
            (
                "animate_character",
                json!({ "sourceClipAssetId": "clip_1" }),
                "referenceAssetIds",
            ),
        ];
        for (mode, media, missing) in cases {
            let mut payload = json!({ "projectId": "p1", "prompt": "x", "mode": mode });
            payload
                .as_object_mut()
                .expect("payload object")
                .extend(media.as_object().expect("media object").clone());
            let error = match video_job_body(&video_args_from(payload)) {
                Err(error) => error,
                Ok(body) => panic!("`{mode}` without {missing} must be refused, got {body}"),
            };
            assert!(
                error.contains(missing),
                "`{mode}` must name the field it is missing ({missing}): {error}"
            );
        }

        // `animate_character` takes the character from EITHER list, so the `sourceAssetId` spelling
        // must be accepted — a check that demanded `referenceAssetIds` would refuse a shape
        // `validate_video_job` admits.
        let body = video_job_body(&video_args_from(json!({
            "projectId": "p1",
            "prompt": "x",
            "mode": "animate_character",
            "sourceClipAssetId": "clip_1",
            "sourceAssetId": "img_1"
        })))
        .expect("the i2v spelling of the character reference is accepted");
        assert_eq!(body["mode"], "animate_character");
    }

    #[test]
    fn video_job_body_maps_every_optional_field() {
        let args = video_args_from(json!({
            "projectId": "p1",
            "prompt": "a storm",
            "negativePrompt": "static",
            "model": "ltx_2_3",
            "duration": 8.5,
            "fps": 24,
            "width": 1280,
            "height": 720,
            "quality": "high",
            "seed": 42,
            "loras": [{ "id": "lora1", "weight": 0.8 }],
            "characterId": "char_1"
        }));
        assert_eq!(
            video_job_body(&args).expect("body builds"),
            json!({
                "projectId": "p1",
                "mode": "text_to_video",
                "prompt": "a storm",
                "negativePrompt": "static",
                "model": "ltx_2_3",
                "duration": 8.5,
                "fps": 24,
                "width": 1280,
                "height": 720,
                "quality": "high",
                "seed": 42,
                "loras": [{ "id": "lora1", "weight": 0.8 }],
                "characterId": "char_1"
            })
        );
    }

    // -----------------------------------------------------------------------
    // get_job_status / get_job_result (sc-10235): snapshot + result mapping.
    // -----------------------------------------------------------------------

    #[test]
    fn compact_job_status_keeps_the_generic_polling_fields() {
        let job = json!({
            "id": "job_abc",
            "type": "video_generate",
            "status": "running",
            "projectId": "p1",
            "stage": "generating",
            "message": "step 12/40",
            "progress": 0.3,
            "etaSeconds": 95,
            "elapsedSeconds": 41,
            "error": null,
            "createdAt": "2026-07-07T00:00:00Z",
            "completedAt": null,
            // Verbose snapshot fields that must be dropped:
            "payload": { "prompt": "x" },
            "result": {},
            "workerId": "w1",
            "attempts": 1
        });
        assert_eq!(
            compact_job_status(&job),
            json!({
                "jobId": "job_abc",
                "type": "video_generate",
                "status": "running",
                "projectId": "p1",
                "stage": "generating",
                "message": "step 12/40",
                "etaSeconds": 95,
                "elapsedSeconds": 41,
                "createdAt": "2026-07-07T00:00:00Z",
                "progressPercent": 30
            })
        );
    }

    #[test]
    fn compact_job_status_surfaces_the_failure_error_and_drops_empty_strings() {
        let job = json!({
            "id": "job_abc",
            "status": "failed",
            "stage": "",
            "message": "",
            "progress": 0.2,
            "error": "CUDA out of memory on gpu0"
        });
        assert_eq!(
            compact_job_status(&job),
            json!({
                "jobId": "job_abc",
                "status": "failed",
                "error": "CUDA out of memory on gpu0",
                "progressPercent": 20
            })
        );
    }

    #[test]
    fn job_error_detail_falls_back_when_the_worker_recorded_nothing() {
        assert_eq!(job_error_detail(&json!({ "error": "boom" })), "boom");
        assert_eq!(
            job_error_detail(&json!({ "error": "" })),
            "the worker reported no error detail"
        );
        assert_eq!(
            job_error_detail(&json!({})),
            "the worker reported no error detail"
        );
    }

    #[test]
    fn valid_job_id_rejects_path_and_query_metacharacters() {
        assert_eq!(valid_job_id("job_ab12cd34"), Ok("job_ab12cd34"));
        assert_eq!(valid_job_id("  job-1  "), Ok("job-1"));
        assert!(valid_job_id("").is_err());
        assert!(valid_job_id("../secrets").is_err());
        assert!(valid_job_id("job_1?x=1").is_err());
        assert!(valid_job_id("job 1").is_err());
    }

    #[test]
    fn request_base_url_accepts_only_configured_forwarded_authority() {
        use http::{HeaderMap, HeaderValue, Uri};
        let uri: Uri = "/mcp".parse().unwrap();
        let allowed = vec!["studio.example.com".to_owned()];

        // A plain Host header → http scheme by default.
        let mut headers = HeaderMap::new();
        headers.insert("host", HeaderValue::from_static("192.168.4.97:8000"));
        assert_eq!(
            request_base_url(&headers, &uri, &allowed).as_deref(),
            Some("http://192.168.4.97:8000")
        );

        // A configured reverse-proxy authority wins, and only the first CSV entry is used.
        headers.insert(
            "x-forwarded-host",
            HeaderValue::from_static("studio.example.com, inner:8000"),
        );
        headers.insert("x-forwarded-proto", HeaderValue::from_static("https"));
        assert_eq!(
            request_base_url(&headers, &uri, &allowed).as_deref(),
            Some("https://studio.example.com")
        );

        // An arbitrary forwarded authority never receives ticket-bearing URLs:
        // None deliberately selects the configured API base.
        headers.insert(
            "x-forwarded-host",
            HeaderValue::from_static("attacker.example"),
        );
        assert_eq!(request_base_url(&headers, &uri, &allowed), None);
        assert_eq!(request_base_url(&headers, &uri, &[]), None);

        headers.insert(
            "x-forwarded-host",
            HeaderValue::from_static("studio.example.com"),
        );
        headers.insert("x-forwarded-proto", HeaderValue::from_static("ftp"));
        assert_eq!(request_base_url(&headers, &uri, &allowed), None);

        // No authority anywhere → None so the caller keeps the configured base.
        assert_eq!(request_base_url(&HeaderMap::new(), &uri, &allowed), None);
    }

    #[test]
    fn encode_media_path_encodes_unsafe_segments_but_keeps_separators() {
        // The common case: slug-safe generated paths are byte-identical.
        assert_eq!(
            encode_media_path("assets/images/g1/img_0001.png"),
            "assets/images/g1/img_0001.png"
        );
        // Separators survive; space/#/% inside a segment are encoded.
        assert_eq!(
            encode_media_path("assets/my folder/a#b%c.png"),
            "assets/my%20folder/a%23b%25c.png"
        );
        // A '?' in a segment can't be mistaken for the query string.
        assert_eq!(encode_media_path("a/b?c.png"), "a/b%3Fc.png");
    }

    #[test]
    fn project_id_validation_rejects_normalization_and_encoding_escapes() {
        assert!(valid_project_id("project_123-ABC"));
        for invalid in [
            ".",
            "..",
            "..\\other",
            "../other",
            "project/other",
            "%2e%2e",
            "%252e%252e",
        ] {
            assert!(!valid_project_id(invalid), "{invalid:?} must be rejected");
        }
    }

    #[test]
    fn media_mime_type_prefers_sidecar_then_extension() {
        assert_eq!(
            media_mime_type("clips/c.mp4", Some("video/webm")).as_deref(),
            Some("video/webm")
        );
        assert_eq!(
            media_mime_type("clips/c.MP4", None).as_deref(),
            Some("video/mp4")
        );
        assert_eq!(
            media_mime_type("images/i.png", None).as_deref(),
            Some("image/png")
        );
        // Unknown extension + no sidecar → omit rather than guess.
        assert_eq!(media_mime_type("clips/c.bin", Some("")), None);
    }

    #[test]
    fn job_wait_config_clamped_enforces_invariants() {
        // Normal values pass through untouched.
        let c = JobWaitConfig::clamped(Duration::from_secs(2), Duration::from_secs(600));
        assert_eq!(c.poll_interval, Duration::from_secs(2));
        assert_eq!(c.timeout, Duration::from_secs(600));

        // A zero interval falls back to the default cadence (never sleep(0)).
        let c = JobWaitConfig::clamped(Duration::ZERO, Duration::from_secs(600));
        assert_eq!(c.poll_interval, JobWaitConfig::default().poll_interval);
        assert_eq!(c.timeout, Duration::from_secs(600));

        // A timeout below the interval is raised so the loop polls at least once.
        let c = JobWaitConfig::clamped(Duration::from_secs(10), Duration::from_secs(3));
        assert_eq!(c.poll_interval, Duration::from_secs(10));
        assert_eq!(c.timeout, Duration::from_secs(10));
    }

    #[test]
    fn poll_outage_tolerance_is_elapsed_time_based() {
        let started = tokio::time::Instant::now();
        assert!(!poll_outage_exceeded(
            started,
            started + MAX_POLL_OUTAGE - Duration::from_millis(1)
        ));
        assert!(poll_outage_exceeded(started, started + MAX_POLL_OUTAGE));
    }

    #[test]
    fn job_progress_scales_the_contract_fraction_to_percent() {
        let job = json!({ "progress": 0.375, "stage": "generating", "message": "step 3/8" });
        assert_eq!(job_progress(&job), (38, "generating: step 3/8".to_owned()));
        // Missing fields degrade to a queued zero, and out-of-range clamps.
        assert_eq!(job_progress(&json!({})), (0, "queued".to_owned()));
        assert_eq!(
            job_progress(&json!({ "progress": 7.5, "stage": "saving" })),
            (100, "saving".to_owned())
        );
    }
}
