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
        CallToolResult, ContentBlock, Meta, ProgressNotificationParam, ProgressToken, Resource,
        ServerCapabilities, ServerInfo,
    },
    schemars,
    service::RoleServer,
    tool, tool_handler, tool_router, ErrorData, Peer, ServerHandler,
};
use serde_json::{json, Map, Value};

use crate::api_client::{ApiClient, ApiClientError};

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
    tool_router: ToolRouter<Self>,
}

impl SceneWorksMcp {
    pub fn new(api: ApiClient) -> Self {
        Self {
            api,
            job_wait: JobWaitConfig::default(),
            tool_router: Self::tool_router(),
        }
    }

    /// Override the blocking-job polling cadence/deadline (tests).
    pub fn with_job_wait(mut self, job_wait: JobWaitConfig) -> Self {
        self.job_wait = job_wait;
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
}

/// Arguments for `submit_video_job`, mapped onto the API's `VideoJobRequest`
/// (`apps/rust-api/src/dto.rs`). The tool exposes four task-level modes and maps
/// them to the API's wire modes in [`video_job_body`]; only provided fields are
/// sent so the API's serde defaults (duration 6s, 25fps, 768x512, ltx_2_3 …)
/// stay authoritative.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SubmitVideoJobArgs {
    #[schemars(description = "Project to generate into (from list_projects).")]
    pub project_id: String,
    #[schemars(description = "The video prompt (1-4000 characters).")]
    pub prompt: String,
    #[schemars(
        description = "\"generate\" (default: text-to-video, or image-to-video when sourceAssetId is set, or first/last-frame when lastFrameAssetId is also set), \"extend\" (continue a clip; needs sourceClipAssetId), \"bridge\" (fill between two clips; needs sourceClipAssetId + bridgeRightClipAssetId), or \"person_replace\" (swap a tracked person; needs sourceClipAssetId + personTrackId + characterId)."
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
    #[schemars(
        description = "LoRA adapters to apply: [{\"id\": <from list_loras>, \"weight\": 0.0-2.0}]."
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
        description = "List the generation model catalog. Returns compact entries: id (use as the model for a job), name, family, type (image/video), capabilities, installState, defaults (resolution/steps/guidanceScale/count) and supported resolutions."
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
        description = "Generate images (or edit an existing image) and return them inline. Submits an image job, waits for it to finish (emitting progress notifications while it runs), and returns each generated image as base64 image content plus a JSON summary with the asset ids. Long-running: seconds to minutes depending on the model."
    )]
    async fn generate_image(
        &self,
        Parameters(args): Parameters<GenerateImageArgs>,
        meta: Meta,
        peer: Peer<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let body =
            image_job_body(&args).map_err(|message| ErrorData::invalid_params(message, None))?;
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
        let progress_token = meta.get_progress_token();
        let mut reporter = ProgressReporter::new(peer, progress_token);
        reporter.report(&submitted).await;

        let started = tokio::time::Instant::now();
        let job = loop {
            tokio::time::sleep(self.job_wait.poll_interval).await;
            let job = self
                .api
                .get_json(&format!("/api/v1/jobs/{job_id}"), &[])
                .await
                .map_err(api_error)?;
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
        for asset in assets {
            let Some(media_path) = asset_media_path(asset) else {
                continue;
            };
            let (bytes, header_mime) = self
                .api
                .get_bytes(&format!("/api/v1/projects/{project_id}/files/{media_path}"))
                .await
                .map_err(api_error)?;
            let mime_type = image_mime_type(
                &media_path,
                asset.pointer("/file/mimeType").and_then(Value::as_str),
                header_mime.as_deref(),
            );
            summary_assets.push(json!({
                "id": asset.get("id").cloned().unwrap_or(Value::Null),
                "path": &media_path,
                "mimeType": &mime_type,
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
        description = "Submit a video generation job WITHOUT waiting for it (video renders for minutes). Modes: \"generate\" (text-to-video; add sourceAssetId for image-to-video, plus lastFrameAssetId for first/last-frame), \"extend\" (continue a clip), \"bridge\" (fill between two clips), \"person_replace\" (swap a tracked person for a Character). Returns the job id + initial snapshot; poll get_job_status, then fetch links with get_job_result once completed."
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
        description = "Fetch the result of a COMPLETED job (video or image) as downloadable links. Mints a short-lived media ticket and returns one resource link per result asset — the URL works from any machine that can reach the SceneWorks API (no auth header needed while the ticket is valid). Video/image bytes are never inlined by this tool. If the job is still running it reports ready=false; if it failed, the job error."
    )]
    async fn get_job_result(
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
            let relative_url =
                format!("/api/v1/projects/{project_id}/files/{media_path}?ticket={ticket}");
            let url = format!("{}{relative_url}", self.api.base_url());
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
                 (call get_job_result again for fresh links). The urls use the \
                 SceneWorks API base \"{}\"; if that host is not reachable from \
                 your machine, apply relativeUrl to the base you use to reach \
                 this MCP server (everything before /mcp).",
                self.api.base_url()
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

/// Map `submit_video_job` args onto the `VideoJobRequest` wire shape
/// (camelCase). The tool's four task-level modes map to the API's wire modes —
/// `generate` picks `text_to_video` / `image_to_video` / `first_last_frame`
/// from the provided image assets — and the mode-specific required assets are
/// checked here so a bad call fails fast with a precise message instead of a
/// submitted-then-rejected job. Only provided optional fields are emitted so
/// the API's serde defaults stay authoritative.
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
        other => {
            return Err(format!(
                "unsupported mode \"{other}\": use \"generate\", \"extend\", \"bridge\" or \
                 \"person_replace\""
            ))
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

/// Keep an asset for the inline result: image-typed (or untyped legacy) records.
fn is_image_asset(asset: &Value) -> bool {
    match asset.get("type").and_then(Value::as_str) {
        Some(media_type) => media_type == "image",
        None => true,
    }
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
        // The resolution menu is the one `limits` field a job request needs; the
        // rest (sampler/scheduler menus, counts) stays on the full API response.
        if let Some(resolutions) = model.pointer("/limits/resolutions") {
            out.insert("resolutions".to_owned(), resolutions.clone());
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

    #[test]
    fn video_job_body_rejects_unknown_mode() {
        let args =
            video_args_from(json!({ "projectId": "p1", "prompt": "x", "mode": "style_remix" }));
        let error = video_job_body(&args).expect_err("unknown mode rejected");
        assert!(error.contains("style_remix"), "{error}");
        assert!(error.contains("person_replace"), "{error}");
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
