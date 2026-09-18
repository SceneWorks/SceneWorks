//! Vision-assisted take review and the human loop around it (epic 22708, sc-22714).
//!
//! The reviewer reads a rendered take back through the **existing** understanding seams and writes
//! what it saw into a document of its own:
//!
//! 1. `POST /api/v1/projects/:p/timelines/:t/items/:i/frames` — the `frame_extract` CPU/FFmpeg job,
//!    the only video → still seam this API has — samples the take at the review plan's declared
//!    positions and persists each frame as a project asset with a timestamp;
//! 2. `POST /api/v1/image/vqa/jobs` — the `image_vqa` job, SenseNova-U1-8B on the MLX/candle
//!    understanding path — is asked ONE declared question per frame and answers in text.
//!
//! Both are routes the app already serves; this adds no inference dependency, no new job type and
//! no new model. The vision half sits behind [`ReviewVision`] so the whole flow runs against a
//! scripted backend with no weights at all — which is what the deterministic tests and the labeled
//! evaluation's rehearsal use.
//!
//! What comes out is an [`ObservedState`] document per reviewed take, written to
//! `<run_dir>/reviews/`. Three properties of it are the point of this story, and each is enforced
//! here rather than left to convention:
//!
//! * **observed is not intended.** The observed-state document references the intended state by
//!   [`IntendedRef`] — run id, plan id/version/hash and a JSON pointer into the run record — and
//!   never copies it. The run record gains only a [`TakeReviewSummary`] index entry: a path and
//!   some counts, no observed values.
//! * **uncertainty never becomes a fact.** An answer the backend hedged, or could not give, is
//!   [`Verdict::Unobserved`] and carries no value ([`sceneworks_core::film_review`] holds that
//!   invariant). In particular an unobserved parcel handoff is flagged `unobserved`, never recorded
//!   as completed.
//! * **observed state never conditions anything.** Nothing in this module writes
//!   `ShotRunRecord::intended` or `ConditioningAssets`; those are derived from the plan and the
//!   reference pack in [`super::Session::ensure_shot_records`] and nowhere else. The only thing
//!   that moves a selected take is a human decision recorded through [`decide_take`] or
//!   [`super::replace_take`].
//!
//! The human loop is three commands, all of which record a [`ProductionDecision`]:
//!
//! * [`decide_take`] with [`Decision::Accept`] — the person looked and is happy. Clears the shot's
//!   own `needsReview` flags; touches no other shot.
//! * [`decide_take`] with [`Decision::Reject`] — the take is rejected (and kept, with its job and
//!   its asset), the shot's selection is cleared, the shots that DECLARED a dependency on it are
//!   flagged `needsReview` and the export is marked stale. Nothing is re-rendered.
//! * [`request_repair`] — a bounded repair: exactly ONE new attempt through
//!   [`super::replace_take`], with the review's own mismatch flags folded into the reason so the
//!   record says what the repair was for. It does not loop, and a failed repair does not retry.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

use sceneworks_core::film_plan::{
    self, HumanTakeDecision, ProductionPlan, ReferencePack, RunRecord, ShotRunRecord,
    TakeRejection, TakeReviewSummary,
};
use sceneworks_core::film_review::{
    aggregate_cut_observation, aggregate_observation, flag_for, format_eval_report, grade_answer,
    parse_eval_set, score_case, tally, unasked_observation, unsafe_media_path, validate_eval_set,
    validate_review_plan, AdjacentTake, CaseOutcome, EvalCase, EvalResults, EvalSet, FrameAnswer,
    FrameEvidence, IntendedRef, MismatchFlag, Observation, ObservedState, ReviewBackendRecord,
    ReviewLimits, ReviewPlan, ReviewQuestion, ReviewSourceRef, ASSISTIVE_NOTICE,
    OBSERVED_STATE_SCHEMA_VERSION, REVIEW_EVAL_SCHEMA_VERSION,
};
use sceneworks_core::time::utc_now;
use serde_json::{json, Value};
use tokio::time::Instant;

/// Re-exported from the harness root, where every preflight now reads it (sc-22715).
pub use super::LIVE_STATUSES;
use super::{
    api_detail, encode_asset_upload, flag_dependents, live_worker_advertising, persist_record,
    read_run_record, read_source, sha256_hex, stale_workers_detail, ApiRequest, ApiTransport,
    Client, HarnessError, PollBounds, PollStop, RequestBody, RunControl, ASSET_SETTLE_GRACE,
    CANCEL_GRACE,
};

/// Directory, inside the run directory, holding one observed-state document per review.
pub const REVIEWS_DIR: &str = "reviews";

/// Default review document name, looked for beside the plan.
pub const REVIEW_PLAN_FILE: &str = "review.jsonc";

/// The catalog id the `image_vqa` route resolves for the understanding path.
pub const VQA_MODEL_ID: &str = "sensenova_u1_8b";

/// The route one question goes through.
pub const VQA_ROUTE: &str = "POST /api/v1/image/vqa/jobs";

// Tokens an answer is truncated to, and the memory the review declares it needs, both come off the
// review document's own `limits` (`ReviewLimits::max_new_tokens`, `ReviewLimits::max_memory_gb`)
// rather than a constant here: a bound nobody can read off the document is not a declared bound.

// ---------------------------------------------------------------------------------------------
// The vision seam
// ---------------------------------------------------------------------------------------------

/// One frame the reviewer wants an answer about.
#[derive(Debug, Clone)]
pub struct FrameRef {
    /// Stable id within the observed-state document.
    pub id: String,
    /// Project asset id, when the frame is already an asset (the `frame_extract` path).
    pub asset_id: Option<String>,
    /// On-disk path, when the frame is a file the caller supplied (the labeled-set path).
    pub path: PathBuf,
    pub timestamp_seconds: f64,
}

pub type VisionFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, HarnessError>> + Send + 'a>>;

/// One answer, plus whether a model actually produced it.
#[derive(Debug, Clone)]
pub struct VisionAnswer {
    pub answer: String,
    /// What the worker reported. `false` means no weights ran — a scripted rehearsal.
    pub real_model_inference: bool,
    pub elapsed_seconds: f64,
}

/// What one `ask` came back with: an answer, or the fact that none arrived inside the review's
/// `limits.maxAnswerSeconds` (sc-22715). A timeout is NOT an error and NOT a value: the reviewer
/// records that question `unobserved` with the timeout as its note and moves on, because "the
/// model did not answer in time" and "the model saw nothing" must both read as nothing seen,
/// never as agreement — and one slow answer must not throw away a take's other evidence.
#[derive(Debug, Clone)]
pub enum VisionOutcome {
    Answered(VisionAnswer),
    /// The backend was asked and cancelled after `after_seconds` with no answer. `detail` names
    /// the job and its last status, for the note.
    TimedOut {
        after_seconds: f64,
        detail: String,
    },
}

/// How the reviewer reaches a vision model.
///
/// Two implementations ship: [`VqaVision`] over the `image_vqa` route, and [`ScriptedVision`],
/// which answers from a table and runs no model. The trait exists so the state transitions, the
/// unobserved handling and the labeled evaluation's arithmetic are all testable without a GPU.
pub trait ReviewVision: Send + Sync {
    /// How this backend describes itself in the record.
    fn record(&self) -> ReviewBackendRecord;

    /// Make `frame` reachable by the backend, returning the asset id the evidence cites. A frame
    /// that already has an asset id is returned unchanged.
    fn prepare_frame<'a>(
        &'a self,
        project_id: &'a str,
        frame: &'a FrameRef,
    ) -> VisionFuture<'a, String>;

    /// Put one question about one frame, within the review's own declared bounds: an answer that
    /// does not arrive inside `limits.max_answer_seconds` comes back as
    /// [`VisionOutcome::TimedOut`] (the job cancelled), and an answer is truncated to
    /// `limits.max_new_tokens`.
    fn ask<'a>(
        &'a self,
        project_id: &'a str,
        asset_id: &'a str,
        question: &'a str,
        limits: ReviewLimits,
    ) -> VisionFuture<'a, VisionOutcome>;
}

/// The production backend: the existing `image_vqa` job type, SenseNova-U1-8B, through the route
/// the Library's own VQA affordance uses.
pub struct VqaVision<'a> {
    transport: &'a dyn ApiTransport,
    model: String,
    poll_interval: Duration,
    /// Held by value, so the borrowed [`Client`] this backend builds per call cannot outlive it.
    control: RunControl,
}

impl<'a> VqaVision<'a> {
    pub fn new(
        transport: &'a dyn ApiTransport,
        poll_interval: Duration,
        control: RunControl,
    ) -> Self {
        Self {
            transport,
            model: VQA_MODEL_ID.to_owned(),
            poll_interval,
            control,
        }
    }

    fn client(&self) -> Client<'_> {
        Client {
            transport: self.transport,
            control: &self.control,
        }
    }

    /// Refuse before anything is dispatched unless a LIVE registered worker advertises `image_vqa`.
    /// A review that queues questions no worker can claim would sit at the poll deadline and then
    /// record every question as unobserved, which reads as evidence and is not.
    ///
    /// The liveness half is not decoration: the sc-22714 smoke ran against a data dir seeded from
    /// an earlier run, whose `film-harness-smoke-gpu` row still advertised `image_vqa` with
    /// `status: "offline"`. The capability-only check passed it — the exact failure this function
    /// exists to prevent — so a worker only counts while its status is one of [`LIVE_STATUSES`].
    ///
    /// It also checks the review's declared `limits.maxMemoryGb` against what
    /// `GET /api/v1/host-capabilities` reports for the API HOST (`--api` may point at another
    /// machine), the same way a production plan's budget is checked before a render.
    pub async fn preflight(&self, limits: ReviewLimits) -> Result<(), HarnessError> {
        let workers = self
            .client()
            .expect_ok("GET", "/api/v1/workers", None)
            .await?;
        // The ONE liveness rule every harness preflight shares (sc-22715).
        let advert = live_worker_advertising(&workers, "image_vqa");
        if advert.live.is_some() {
            return self.preflight_memory(limits).await;
        }
        let detail = stale_workers_detail(&advert.stale);
        Err(HarnessError::Refused(format!(
            "no live registered worker advertises image_vqa{detail}, so {VQA_MODEL_ID} cannot \
             answer anything; start the GPU worker (SCENEWORKS_WORKER_ONLY=1) and wait for it to \
             register, or clear a stale worker row that is shadowing it"
        )))
    }

    /// Refuse before anything is dispatched when the API host has less memory than the review
    /// declares it needs.
    ///
    /// A review that starts on a host too small for the understanding model does not fail fast: it
    /// creates a project, imports frames and then dies inside the loader, or swaps for minutes and
    /// hits `maxAnswerSeconds` on every question — which records the whole take as unobserved and
    /// READS LIKE EVIDENCE. A host that reports nothing at all is refused too: an unchecked ceiling
    /// is not a checked one.
    async fn preflight_memory(&self, limits: ReviewLimits) -> Result<(), HarnessError> {
        let host = self
            .client()
            .expect_ok("GET", "/api/v1/host-capabilities", None)
            .await?;
        let reported = ["memoryGb", "unifiedMemoryGb", "gpuMemoryGb"]
            .iter()
            .find_map(|key| host.get(*key).and_then(Value::as_f64))
            .filter(|gb| gb.is_finite() && *gb > 0.0);
        match reported {
            Some(available) if limits.max_memory_gb > available => {
                Err(HarnessError::Refused(format!(
                    "the review declares limits.maxMemoryGb {:.1} GB but the API host reports \
                     {available:.1} GB; lower the ceiling only if {VQA_MODEL_ID} really fits, or \
                     point --api at a host that has the memory",
                    limits.max_memory_gb
                )))
            }
            Some(_) => Ok(()),
            None => Err(HarnessError::Refused(format!(
                "no registered worker reports host memory, so the review's declared \
                 limits.maxMemoryGb of {:.1} GB cannot be checked before anything is dispatched; \
                 start the GPU worker and wait for its first heartbeat",
                limits.max_memory_gb
            ))),
        }
    }

    /// Refuse before the first question unless the catalog reports the understanding model
    /// installed on this host.
    ///
    /// Also not decoration: the weights ship as PER-TIER subdirectories with no config at the
    /// snapshot root, and the tier is bound by the DOWNLOAD RECEIPT in the data dir — so a host
    /// with the model in its Hugging Face cache but no receipt fails inside the loader
    /// ("cannot bind numeric tier without .../config.json") on the first question, after the
    /// review has already created a project and imported frames. Catching it here costs one
    /// request and says what to do.
    pub async fn preflight_model(&self) -> Result<(), HarnessError> {
        let catalog = self
            .client()
            .expect_ok("GET", "/api/v1/models", None)
            .await?;
        let Some(entry) = catalog
            .as_array()
            .into_iter()
            .flatten()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(self.model.as_str()))
        else {
            return Err(HarnessError::Refused(format!(
                "{} is not in this API's model catalog, so no question can be answered",
                self.model
            )));
        };
        if entry.get("installState").and_then(Value::as_str) == Some("installed") {
            return Ok(());
        }
        Err(HarnessError::Refused(format!(
            "{} is not installed on this host (catalog installState is {:?}); download it in the \
             Model Manager, or point --api at an API whose data dir holds its download receipt — \
             the weights ship as per-tier subdirectories and the receipt is what binds the tier",
            self.model,
            entry
                .get("installState")
                .and_then(Value::as_str)
                .unwrap_or("absent")
        )))
    }
}

impl ReviewVision for VqaVision<'_> {
    fn record(&self) -> ReviewBackendRecord {
        ReviewBackendRecord {
            kind: "image_vqa".to_owned(),
            model: self.model.clone(),
            route: VQA_ROUTE.to_owned(),
            real_model_inference: true,
        }
    }

    fn prepare_frame<'a>(
        &'a self,
        project_id: &'a str,
        frame: &'a FrameRef,
    ) -> VisionFuture<'a, String> {
        Box::pin(async move {
            if let Some(asset_id) = &frame.asset_id {
                return Ok(asset_id.clone());
            }
            import_frame_asset(self.transport, project_id, frame).await
        })
    }

    fn ask<'a>(
        &'a self,
        project_id: &'a str,
        asset_id: &'a str,
        question: &'a str,
        limits: ReviewLimits,
    ) -> VisionFuture<'a, VisionOutcome> {
        Box::pin(async move {
            let started = Instant::now();
            let max_seconds = limits.max_answer_seconds;
            let client = self.client();
            let created = client
                .expect_ok(
                    "POST",
                    "/api/v1/image/vqa/jobs",
                    Some(json!({
                        "projectId": project_id,
                        "sourceAssetId": asset_id,
                        "question": question,
                        "model": self.model,
                        "maxNewTokens": limits.max_new_tokens,
                        "requestedGpu": "auto",
                    })),
                )
                .await?;
            let job_id = created
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    HarnessError::Transport(format!("vqa job response has no id: {created}"))
                })?
                .to_owned();
            let deadline = started + Duration::from_secs(max_seconds);
            let (view, poll_stop) = client
                .wait_for_job(&job_id, poll_bounds(deadline, self.poll_interval))
                .await?;
            // `limits.maxAnswerSeconds` ran out: the job was cancelled through the API and this
            // question has no answer. Reported as a timeout, never as an error (sc-22715).
            if matches!(poll_stop, PollStop::ShotBudget | PollStop::RunBudget) {
                return Ok(VisionOutcome::TimedOut {
                    after_seconds: started.elapsed().as_secs_f64(),
                    detail: format!(
                        "vqa job {job_id} was still {} and was cancelled",
                        view.status
                    ),
                });
            }
            if view.status != "completed" {
                return Err(HarnessError::Transport(format!(
                    "vqa job {job_id} ended {}: {}",
                    view.status,
                    view.failure_text()
                )));
            }
            Ok(VisionOutcome::Answered(VisionAnswer {
                answer: view
                    .result
                    .get("answer")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                real_model_inference: view
                    .result
                    .get("realModelInference")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                elapsed_seconds: started.elapsed().as_secs_f64(),
            }))
        })
    }
}

/// A backend that answers from a table and runs no model.
///
/// Keys are matched most specific first: `"<question id>@<frame id>"`, then `"<question id>"`, then
/// the fallback. An unmatched question gets the fallback, which defaults to an explicit "I cannot
/// tell" — so a scripted review that forgets a question records it as UNOBSERVED rather than
/// silently as agreement.
#[derive(Debug, Clone)]
pub struct ScriptedVision {
    answers: BTreeMap<String, String>,
    fallback: String,
    label: String,
}

impl Default for ScriptedVision {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptedVision {
    pub fn new() -> Self {
        Self {
            answers: BTreeMap::new(),
            fallback: "I cannot tell from this frame.".to_owned(),
            label: "scripted".to_owned(),
        }
    }

    /// Answer `key` (a question id, or `"<question id>@<frame id>"`) with `answer`.
    pub fn answer(mut self, key: &str, answer: &str) -> Self {
        self.answers.insert(key.to_owned(), answer.to_owned());
        self
    }

    /// The key the reviewer stamps into the question text so this backend can route on it. The
    /// production backend ignores it; it is a plain prefix of the question and the model reads it
    /// as context.
    fn key_of(question: &str) -> (String, String) {
        let Some(rest) = question.strip_prefix('[') else {
            return (String::new(), String::new());
        };
        let Some((head, _)) = rest.split_once(']') else {
            return (String::new(), String::new());
        };
        match head.split_once('@') {
            Some((question_id, frame_id)) => (question_id.to_owned(), frame_id.to_owned()),
            None => (head.to_owned(), String::new()),
        }
    }
}

impl ReviewVision for ScriptedVision {
    fn record(&self) -> ReviewBackendRecord {
        ReviewBackendRecord {
            kind: self.label.clone(),
            model: "none (scripted)".to_owned(),
            route: "none".to_owned(),
            real_model_inference: false,
        }
    }

    fn prepare_frame<'a>(
        &'a self,
        _project_id: &'a str,
        frame: &'a FrameRef,
    ) -> VisionFuture<'a, String> {
        let id = frame
            .asset_id
            .clone()
            .unwrap_or_else(|| format!("scripted_asset_{}", frame.id));
        Box::pin(async move { Ok(id) })
    }

    fn ask<'a>(
        &'a self,
        _project_id: &'a str,
        _asset_id: &'a str,
        question: &'a str,
        _limits: ReviewLimits,
    ) -> VisionFuture<'a, VisionOutcome> {
        let (question_id, frame_id) = Self::key_of(question);
        let answer = self
            .answers
            .get(&format!("{question_id}@{frame_id}"))
            .or_else(|| self.answers.get(&question_id))
            .cloned()
            .unwrap_or_else(|| self.fallback.clone());
        Box::pin(async move {
            Ok(VisionOutcome::Answered(VisionAnswer {
                answer,
                real_model_inference: false,
                elapsed_seconds: 0.0,
            }))
        })
    }
}

/// One deadline bounds a review's job wait in every direction: the shot budget, the run budget and
/// the cancel/settle graces are all the review plan's own `limits`, not a plan's render budget.
fn poll_bounds(deadline: Instant, poll_interval: Duration) -> PollBounds {
    let remaining = deadline.saturating_duration_since(Instant::now());
    PollBounds {
        shot_deadline: deadline,
        run_deadline: Some(deadline),
        poll_interval,
        cancel_grace: CANCEL_GRACE.min(remaining.max(Duration::from_secs(1))),
        settle_grace: ASSET_SETTLE_GRACE.min(remaining.max(Duration::from_secs(1))),
    }
}

/// Prefix the reviewer stamps onto every question so a scripted backend can route on it and a
/// person reading the record can tell which question an answer belongs to.
fn tagged_question(question_id: &str, frame_id: &str, ask: &str) -> String {
    format!("[{question_id}@{frame_id}] {ask}")
}

async fn import_frame_asset(
    transport: &dyn ApiTransport,
    project_id: &str,
    frame: &FrameRef,
) -> Result<String, HarnessError> {
    let bytes = std::fs::read(&frame.path).map_err(|error| {
        HarnessError::Refused(format!(
            "cannot read review frame {}: {error}",
            frame.path.display()
        ))
    })?;
    let filename = frame
        .path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("{}.png", frame.id));
    let content_type = match frame
        .path
        .extension()
        .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    };
    let provenance = json!({
        "filmHarness": {
            "kind": "review_frame",
            "frameId": frame.id,
            "timestampSeconds": frame.timestamp_seconds,
            "sha256": sha256_hex(&bytes),
        }
    });
    let (boundary, body) = encode_asset_upload(&filename, content_type, &bytes, &provenance);
    let route = format!("/api/v1/projects/{project_id}/assets");
    let response = transport
        .call(ApiRequest {
            method: "POST",
            path: route.clone(),
            body: RequestBody::Multipart {
                boundary,
                bytes: body,
            },
        })
        .await?;
    if !(200..300).contains(&response.status) {
        return Err(HarnessError::Api {
            method: "POST",
            path: route,
            status: response.status,
            detail: api_detail(&response.body),
        });
    }
    response
        .body
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            HarnessError::Transport(format!(
                "review frame import response has no id: {}",
                response.body
            ))
        })
}

// ---------------------------------------------------------------------------------------------
// Opening a run for review
// ---------------------------------------------------------------------------------------------

/// Everything a review, an accept/reject or a repair needs off disk, with the source documents
/// re-verified against the hashes the run recorded.
pub struct ReviewContext {
    pub record: RunRecord,
    pub plan: ProductionPlan,
    pub pack: ReferencePack,
    pub plan_path: PathBuf,
    pub pack_path: PathBuf,
    pub out_dir: PathBuf,
    pub review: Option<ReviewPlan>,
    pub review_path: PathBuf,
    pub review_sha256: String,
}

impl ReviewContext {
    /// Read the run record and its sources. `review_plan_path` defaults to `review.jsonc` beside
    /// the plan; `need_review_plan` is false for the pure-decision commands, which need no
    /// questions at all.
    pub fn open(
        out_dir: &Path,
        review_plan_path: Option<&Path>,
        need_review_plan: bool,
    ) -> Result<Self, HarnessError> {
        let record = read_run_record(out_dir)?;
        if record.schema_version != film_plan::RUN_RECORD_SCHEMA_VERSION {
            return Err(HarnessError::Refused(format!(
                "run record schema version {} (this build reads {})",
                record.schema_version,
                film_plan::RUN_RECORD_SCHEMA_VERSION
            )));
        }
        let (plan_path, plan_bytes) = read_source(
            Path::new(&record.plan.path),
            &out_dir.join("plan.json"),
            "plan",
        )?;
        let (pack_path, pack_bytes) = read_source(
            Path::new(&record.reference_pack.path),
            &out_dir.join("references.json"),
            "reference pack",
        )?;
        for (label, bytes, recorded) in [
            ("plan", &plan_bytes, &record.plan.sha256),
            ("reference pack", &pack_bytes, &record.reference_pack.sha256),
        ] {
            let actual = sha256_hex(bytes);
            if &actual != recorded {
                return Err(HarnessError::Refused(format!(
                    "the {label} changed since run {} started (recorded {recorded}, found \
                     {actual}); reviewing a take against a different {label} would compare it to \
                     an intent it was never rendered for",
                    record.run_id
                )));
            }
        }
        let plan = film_plan::parse_plan(std::str::from_utf8(&plan_bytes).unwrap_or_default())
            .map_err(|error| HarnessError::Refused(format!("plan: {error}")))?;
        let pack =
            film_plan::parse_reference_pack(std::str::from_utf8(&pack_bytes).unwrap_or_default())
                .map_err(|error| HarnessError::Refused(format!("reference pack: {error}")))?;

        let review_path = review_plan_path.map(Path::to_path_buf).unwrap_or_else(|| {
            plan_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(REVIEW_PLAN_FILE)
        });
        let (review, review_sha256) = if need_review_plan {
            let bytes = std::fs::read(&review_path).map_err(|error| {
                HarnessError::Refused(format!(
                    "cannot read the review plan at {}: {error}; pass --review-plan FILE",
                    review_path.display()
                ))
            })?;
            let document = sceneworks_core::film_review::parse_review_plan(
                std::str::from_utf8(&bytes).unwrap_or_default(),
            )
            .map_err(|error| HarnessError::Refused(format!("review plan: {error}")))?;
            let findings = validate_review_plan(&document, &plan);
            if !findings.is_empty() {
                return Err(HarnessError::Validation(findings));
            }
            (Some(document), sha256_hex(&bytes))
        } else {
            (None, String::new())
        };

        Ok(Self {
            record,
            plan,
            pack,
            plan_path,
            pack_path,
            out_dir: out_dir.to_path_buf(),
            review,
            review_path,
            review_sha256,
        })
    }

    fn review_plan(&self) -> Result<&ReviewPlan, HarnessError> {
        self.review
            .as_ref()
            .ok_or_else(|| HarnessError::Refused("this command needs a review plan".to_owned()))
    }

    fn source_ref(&self) -> ReviewSourceRef {
        let review = self.review.as_ref();
        ReviewSourceRef {
            id: review.map(|r| r.id.clone()).unwrap_or_default(),
            version: review.map(|r| r.version).unwrap_or_default(),
            path: self.review_path.display().to_string(),
            sha256: self.review_sha256.clone(),
        }
    }

    fn persist(&self) -> Result<(), HarnessError> {
        persist_record(
            &self.record,
            &self.out_dir,
            &self.plan_path,
            &self.pack_path,
        )?;
        Ok(())
    }

    /// The intended state of `shot_id`, referenced rather than copied.
    fn intended_ref(&self, shot_id: &str) -> IntendedRef {
        let index = self
            .record
            .shots
            .iter()
            .position(|shot| shot.shot_id == shot_id)
            .unwrap_or_default();
        IntendedRef {
            run_id: self.record.run_id.clone(),
            shot_id: shot_id.to_owned(),
            plan_id: self.plan.id.clone(),
            plan_version: self.plan.version,
            plan_sha256: self.record.plan.sha256.clone(),
            record_pointer: format!("/shots/{index}/intended"),
            record_path: format!("../{}", super::RUN_RECORD_FILE),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Review
// ---------------------------------------------------------------------------------------------

/// What one `film-harness review` invocation needs beyond the transport and the backend.
#[derive(Debug, Clone)]
pub struct ReviewOptions {
    pub out_dir: PathBuf,
    /// Defaults to `review.jsonc` beside the plan.
    pub review_plan_path: Option<PathBuf>,
    /// Review only these shots. Empty reviews every selected shot that has a take.
    pub shot_ids: Vec<String>,
    pub poll_interval: Duration,
    /// The same cooperative control a run uses: an in-process flag plus the run directory's cancel
    /// sentinel, so `film-harness cancel --out DIR` stops a review from another shell too.
    pub control: RunControl,
}

impl ReviewOptions {
    pub fn new(out_dir: PathBuf) -> Self {
        let control = RunControl::watching(&out_dir);
        Self {
            out_dir,
            review_plan_path: None,
            shot_ids: Vec::new(),
            poll_interval: Duration::from_secs(3),
            control,
        }
    }
}

/// Review the selected take of each requested shot and write one observed-state document each.
///
/// Nothing here dispatches a render, moves a selection or writes an intended state. The bounds in
/// the review plan's `limits` are declared before the first frame is asked for and are what stops
/// a review; a review that runs out records `stop` and keeps the partial evidence, because partial
/// evidence of a fault is still evidence.
pub async fn review(
    transport: &dyn ApiTransport,
    options: &ReviewOptions,
    vision: &dyn ReviewVision,
) -> Result<RunRecord, HarnessError> {
    let lease = super::ControllerLease::acquire_new_action(
        &options.out_dir,
        format!("review_{}", uuid::Uuid::new_v4().simple()),
    )?;
    review_with_lease(transport, options, vision, lease).await
}

pub(crate) async fn review_with_lease(
    transport: &dyn ApiTransport,
    options: &ReviewOptions,
    vision: &dyn ReviewVision,
    _lease: super::ControllerLease,
) -> Result<RunRecord, HarnessError> {
    let prior_reviews: BTreeMap<String, usize> = super::read_run_record(&options.out_dir)?
        .shots
        .iter()
        .map(|shot| (shot.shot_id.clone(), shot.reviews.len()))
        .collect();
    write_review_operation(options, "running", None)?;
    let result = review_inner(transport, options, vision).await;
    let detail = match &result {
        Err(error) => Some(error.to_string()),
        Ok(record) => {
            let stops: Vec<String> = record
                .shots
                .iter()
                .flat_map(|shot| {
                    shot.reviews
                        .iter()
                        .skip(*prior_reviews.get(&shot.shot_id).unwrap_or(&0))
                        .filter_map(|review| {
                            review
                                .stop
                                .as_ref()
                                .map(|stop| format!("{}: {stop}", shot.shot_id))
                        })
                })
                .collect();
            (!stops.is_empty()).then(|| stops.join("; "))
        }
    };
    write_review_operation(
        options,
        if detail.is_none() {
            "completed"
        } else {
            "failed"
        },
        detail.as_deref(),
    )?;
    result
}

pub(crate) const REVIEW_OPERATION_FILE: &str = "review-operation.json";

pub(crate) fn write_review_operation(
    options: &ReviewOptions,
    status: &str,
    detail: Option<&str>,
) -> Result<(), HarnessError> {
    let value = serde_json::json!({"status": status, "shotIds": options.shot_ids, "updatedAt": utc_now(), "detail": detail});
    super::write_atomically(
        &options.out_dir.join(REVIEW_OPERATION_FILE),
        &serde_json::to_vec_pretty(&value).map_err(|error| HarnessError::Io(error.to_string()))?,
    )
}

/// Refuse invalid shot/question contracts before accepting work or consulting a model.
pub(crate) fn validate_review_request(options: &ReviewOptions) -> Result<(), HarnessError> {
    let context = ReviewContext::open(&options.out_dir, options.review_plan_path.as_deref(), true)?;
    review_targets(&context, options)?;
    Ok(())
}

fn review_targets(
    context: &ReviewContext,
    options: &ReviewOptions,
) -> Result<Vec<String>, HarnessError> {
    let review_plan = context.review_plan()?;
    let targets: Vec<String> = if options.shot_ids.is_empty() {
        context
            .record
            .shots
            .iter()
            .filter(|shot| shot.selected_attempt.is_some())
            .map(|shot| shot.shot_id.clone())
            .collect()
    } else {
        options.shot_ids.clone()
    };
    if targets.is_empty() {
        return Err(HarnessError::Refused(format!(
            "run {} has no shot with a selected take to review",
            context.record.run_id
        )));
    }
    for shot_id in &targets {
        if !review_plan.shots.contains_key(shot_id) {
            return Err(HarnessError::Refused(format!(
                "review plan {:?} asks no questions about shot {shot_id}; add them (or review a \
                 different shot) rather than reviewing it against nothing",
                review_plan.id
            )));
        }
        if context
            .record
            .shot(shot_id)
            .and_then(|shot| shot.selected_attempt)
            .is_none()
        {
            return Err(HarnessError::Refused(format!(
                "shot {shot_id} has no selected take in run {}; render one first",
                context.record.run_id
            )));
        }
    }

    Ok(targets)
}

async fn review_inner(
    transport: &dyn ApiTransport,
    options: &ReviewOptions,
    vision: &dyn ReviewVision,
) -> Result<RunRecord, HarnessError> {
    let mut context =
        ReviewContext::open(&options.out_dir, options.review_plan_path.as_deref(), true)?;
    let project_id = context.record.project_id.clone().ok_or_else(|| {
        HarnessError::Refused(format!(
            "run {} never created a project, so it has no take to review",
            context.record.run_id
        ))
    })?;
    let review_plan = context.review_plan()?.clone();

    let targets = review_targets(&context, options)?;
    let client = Client {
        transport,
        control: &options.control,
    };
    let timeline_id = ensure_review_timeline(&client, &project_id, &context.record).await?;

    for shot_id in &targets {
        let observed = review_one(
            transport,
            &client,
            vision,
            &context,
            &review_plan,
            &project_id,
            &timeline_id,
            shot_id,
            options,
        )
        .await?;
        record_review(&mut context, shot_id, &observed)?;
        context.persist()?;
    }
    Ok(context.record)
}

/// The one-item timeline the reviewer extracts frames through.
///
/// Deliberately NOT the run's export timeline: frame extraction is scaffolding, and rewriting the
/// export timeline to point at whichever take is under review would corrupt the thing the run is
/// for. It is found by the name it was created under, so a second review adopts it.
async fn ensure_review_timeline(
    client: &Client<'_>,
    project_id: &str,
    record: &RunRecord,
) -> Result<String, HarnessError> {
    let name = format!("film-harness review ({})", record.run_id);
    let existing = client
        .expect_ok(
            "GET",
            &format!("/api/v1/projects/{project_id}/timelines"),
            None,
        )
        .await?
        .as_array()
        .into_iter()
        .flatten()
        .find(|timeline| timeline.get("name").and_then(Value::as_str) == Some(name.as_str()))
        .and_then(|timeline| timeline.get("id"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    if let Some(id) = existing {
        return Ok(id);
    }
    let aspect_ratio = record
        .timeline
        .as_ref()
        .map(|timeline| timeline.aspect_ratio.clone())
        .unwrap_or_else(|| "16:9".to_owned());
    let fps = record
        .timeline
        .as_ref()
        .map(|timeline| timeline.fps)
        .or_else(|| record.model.as_ref().map(|model| model.fps))
        .unwrap_or(24);
    let created = client
        .expect_ok(
            "POST",
            &format!("/api/v1/projects/{project_id}/timelines"),
            Some(json!({ "name": name, "aspectRatio": aspect_ratio, "fps": fps })),
        )
        .await?;
    created
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| HarnessError::Transport(format!("timeline response has no id: {created}")))
}

/// Point the review timeline at one take, as a single item starting at 0 — which makes the frame
/// route's `playheadSeconds` equal to the timestamp within the take.
async fn point_review_timeline_at(
    client: &Client<'_>,
    project_id: &str,
    timeline_id: &str,
    shot_id: &str,
    asset_id: &str,
    length: f64,
) -> Result<String, HarnessError> {
    let mut timeline = client
        .expect_ok(
            "GET",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
            None,
        )
        .await?;
    let item_id = format!("review_{}", shot_id.to_ascii_lowercase());
    let item = json!({
        "id": item_id,
        "trackId": "track_main",
        "assetId": asset_id,
        "type": "video",
        "displayName": format!("review {shot_id}"),
        "sourceIn": 0.0,
        "sourceOut": length,
        "timelineStart": 0.0,
        "timelineEnd": length,
        "speed": 1.0,
        "fit": "fit",
        "volume": 1.0,
    });
    if let Some(track) = timeline
        .get_mut("tracks")
        .and_then(Value::as_array_mut)
        .and_then(|tracks| {
            tracks
                .iter_mut()
                .find(|track| track.get("id").and_then(Value::as_str) == Some("track_main"))
        })
    {
        track["items"] = Value::Array(vec![item]);
    }
    client
        .expect_ok(
            "PUT",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
            Some(json!({ "timeline": timeline })),
        )
        .await?;
    Ok(item_id)
}

#[allow(clippy::too_many_arguments)]
async fn review_one(
    transport: &dyn ApiTransport,
    client: &Client<'_>,
    vision: &dyn ReviewVision,
    context: &ReviewContext,
    review_plan: &ReviewPlan,
    project_id: &str,
    timeline_id: &str,
    shot_id: &str,
    options: &ReviewOptions,
) -> Result<ObservedState, HarnessError> {
    let started = Instant::now();
    let limits = review_plan.limits;
    let deadline = started + Duration::from_secs(limits.max_seconds);
    let shot = context
        .record
        .shot(shot_id)
        .expect("checked by the caller")
        .clone();
    let attempt = shot.selected_attempt.expect("checked by the caller");
    let take = shot
        .selected()
        .and_then(|record| record.take.clone())
        .ok_or_else(|| {
            HarnessError::Refused(format!("shot {shot_id}'s selected attempt has no take"))
        })?;
    let length = take
        .encoded_duration_seconds
        .filter(|seconds| *seconds > 0.0)
        .unwrap_or(shot.intended.target_duration_seconds);

    let item_id = point_review_timeline_at(
        client,
        project_id,
        timeline_id,
        shot_id,
        &take.asset_id,
        length,
    )
    .await?;

    let mut stop: Option<String> = None;
    let mut frames: Vec<FrameEvidence> = Vec::new();
    let mut frame_refs: Vec<FrameRef> = Vec::new();
    for (index, position) in review_plan.sampling.positions.iter().enumerate() {
        if index as u32 >= limits.max_frames_per_shot {
            stop.get_or_insert(format!(
                "frame_budget: limits.maxFramesPerShot is {}, so sampling stopped after {index} \
                 frames",
                limits.max_frames_per_shot
            ));
            break;
        }
        if Instant::now() >= deadline || options.control.is_canceled() {
            stop.get_or_insert_with(|| review_stop_reason(&options.control, limits));
            break;
        }
        let timestamp = (position * length).clamp(0.0, (length - 0.001).max(0.0));
        let frame_id = format!("{shot_id}-a{attempt}-f{}", index + 1);
        let Some(evidence) = extract_frame(
            client,
            project_id,
            timeline_id,
            &item_id,
            shot_id,
            attempt,
            &frame_id,
            timestamp,
            deadline,
            options,
        )
        .await?
        else {
            // The budget ran out mid-extraction: keep the frames already sampled, say so.
            stop.get_or_insert_with(|| review_stop_reason(&options.control, limits));
            break;
        };
        frame_refs.push(FrameRef {
            id: evidence.id.clone(),
            asset_id: Some(evidence.asset_id.clone()),
            path: PathBuf::from(&evidence.path),
            timestamp_seconds: evidence.timestamp_seconds,
        });
        frames.push(evidence);
    }

    // The adjacent selected take a cut-continuity question compares against: the shot this one
    // DECLARED a dependency on. Read off the plan and the current selection only — never off an
    // earlier review.
    let adjacent = adjacent_take(&context.plan, &context.record, shot_id);
    let mut adjacent_frame: Option<(FrameEvidence, FrameRef)> = None;
    let wants_cut = review_plan.shots[shot_id]
        .questions
        .iter()
        .any(|question| question.across_cut)
        && stop.is_none();
    if let Some(previous) = adjacent.as_ref().filter(|_| wants_cut) {
        let previous_shot = context.record.shot(&previous.shot_id).expect("resolved");
        let previous_take = previous_shot
            .selected()
            .and_then(|record| record.take.clone());
        if let Some(previous_take) = previous_take {
            let previous_length = previous_take
                .encoded_duration_seconds
                .filter(|seconds| *seconds > 0.0)
                .unwrap_or(previous_shot.intended.target_duration_seconds);
            let previous_item = point_review_timeline_at(
                client,
                project_id,
                timeline_id,
                &previous.shot_id,
                &previous_take.asset_id,
                previous_length,
            )
            .await?;
            let frame_id = format!(
                "{}-a{}-cut",
                previous.shot_id,
                previous_shot.selected_attempt.unwrap_or(0)
            );
            match extract_frame(
                client,
                project_id,
                timeline_id,
                &previous_item,
                &previous.shot_id,
                previous_shot.selected_attempt.unwrap_or(0),
                &frame_id,
                (previous_length - 0.05).max(0.0),
                deadline,
                options,
            )
            .await?
            {
                Some(evidence) => {
                    let reference = FrameRef {
                        id: evidence.id.clone(),
                        asset_id: Some(evidence.asset_id.clone()),
                        path: PathBuf::from(&evidence.path),
                        timestamp_seconds: evidence.timestamp_seconds,
                    };
                    adjacent_frame = Some((evidence, reference));
                }
                // Out of budget before the neighbour's frame landed: the cut question is then
                // recorded unobserved by `answer_questions` (no adjacent frame), and the stop
                // says why the review ended.
                None => {
                    stop.get_or_insert_with(|| review_stop_reason(&options.control, limits));
                }
            }
        }
    }
    let adjacent = adjacent.map(|mut adjacent| {
        adjacent.frame_id = adjacent_frame
            .as_ref()
            .map(|(evidence, _)| evidence.id.clone());
        adjacent
    });
    if let Some((evidence, _)) = &adjacent_frame {
        frames.push(evidence.clone());
    }

    let questions = &review_plan.shots[shot_id].questions;
    let answered = answer_questions(
        vision,
        project_id,
        shot_id,
        questions,
        &frame_refs,
        adjacent_frame.as_ref().map(|(_, reference)| reference),
        limits,
        review_plan.uncertain_below,
        deadline,
        &options.control,
    )
    .await?;
    let Answered {
        observations,
        mismatches,
        stop: asked_stop,
        real_model_inference,
    } = answered;
    if stop.is_none() {
        stop = asked_stop;
    }

    let _ = transport;
    let mut backend = vision.record();
    // What the worker actually reported wins over what the backend claims about itself.
    backend.real_model_inference &= real_model_inference;

    Ok(ObservedState {
        schema_version: OBSERVED_STATE_SCHEMA_VERSION,
        review_id: format!(
            "{}:{shot_id}:a{attempt}:r{}",
            context.record.run_id,
            shot.reviews.len() + 1
        ),
        reviewed_at: utc_now(),
        shot_id: shot_id.to_owned(),
        attempt,
        take_asset_id: take.asset_id.clone(),
        intended: context.intended_ref(shot_id),
        review_plan: context.source_ref(),
        backend,
        limits,
        adjacent,
        frames,
        observations,
        mismatches,
        stop,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        notice: ASSISTIVE_NOTICE.to_owned(),
    })
}

fn review_stop_reason(control: &RunControl, limits: ReviewLimits) -> String {
    if control.is_canceled() {
        "canceled: a cancel was requested while the review was running".to_owned()
    } else {
        format!(
            "review_budget: limits.maxSeconds is {}s and the review spent it; the partial evidence \
             below is kept",
            limits.max_seconds
        )
    }
}

/// What one question to one frame produced: a graded answer, or the reason no answer arrived.
enum Asked {
    Answer(FrameAnswer),
    /// The note an `unobserved` observation carries for this question.
    TimedOut(String),
}

/// Put one question to one frame and grade the answer, keeping the token the grader matched and
/// the polarity it read it with so a mis-grade is debuggable from the record alone.
async fn ask_one(
    vision: &dyn ReviewVision,
    project_id: &str,
    question: &ReviewQuestion,
    frame: &FrameRef,
    limits: ReviewLimits,
    real_model_inference: &mut bool,
) -> Result<Asked, HarnessError> {
    let asset_id = vision.prepare_frame(project_id, frame).await?;
    let text = tagged_question(&question.id, &frame.id, &question.ask);
    let answer = match vision.ask(project_id, &asset_id, &text, limits).await? {
        VisionOutcome::Answered(answer) => answer,
        VisionOutcome::TimedOut {
            after_seconds,
            detail,
        } => {
            return Ok(Asked::TimedOut(format!(
                "answer_timeout: limits.maxAnswerSeconds is {}s and frame {} got no answer in \
                 {after_seconds:.1}s ({detail}); nothing was read, so nothing is recorded as seen",
                limits.max_answer_seconds, frame.id
            )));
        }
    };
    *real_model_inference &= answer.real_model_inference;
    let grade = grade_answer(question, &answer.answer);
    Ok(Asked::Answer(FrameAnswer {
        frame_id: frame.id.clone(),
        answer: answer.answer,
        verdict: grade.verdict,
        matched: grade.matched,
        polarity: grade.polarity,
        confidence: grade.confidence,
        hedged: grade.hedged,
        elapsed_seconds: answer.elapsed_seconds,
    }))
}

/// What one pass of questions produced.
struct Answered {
    observations: Vec<Observation>,
    mismatches: Vec<MismatchFlag>,
    stop: Option<String>,
    /// Whether EVERY answer came back from a real model run. `false` the moment one did not — a
    /// record that says weights ran when they did not is the one lie these documents must not
    /// tell, so the worker's own report wins over the backend's self-description.
    real_model_inference: bool,
}

/// Ask every question of every frame its scope names, within the declared bounds.
///
/// `limits.maxQuestionsPerShot` is enforced by [`validate_review_plan`] when the document is read,
/// so a plan that reaches here cannot exceed it; what bounds this loop is the wall-clock deadline
/// and the cancel token.
#[allow(clippy::too_many_arguments)]
async fn answer_questions(
    vision: &dyn ReviewVision,
    project_id: &str,
    shot_id: &str,
    questions: &[ReviewQuestion],
    frames: &[FrameRef],
    adjacent_frame: Option<&FrameRef>,
    limits: ReviewLimits,
    uncertain_below: f64,
    deadline: Instant,
    control: &RunControl,
) -> Result<Answered, HarnessError> {
    let mut observations = Vec::new();
    let mut mismatches = Vec::new();
    let mut stop = None;
    let mut real_model_inference = !questions.is_empty();
    for question in questions.iter() {
        if Instant::now() >= deadline || control.is_canceled() {
            stop.get_or_insert_with(|| review_stop_reason(control, limits));
            break;
        }
        // An across-the-cut question is COMPARED, not aggregated: the same closed question goes to
        // this take's frames and to the adjacent selected take's frame, and the two answers are
        // held against each other. Without an adjacent take there is nothing to compare, so the
        // question is never answered from one side (which would report a clean cut on the strength
        // of never having looked at the other one) — but it IS recorded, as an unobserved
        // observation naming why. Omitting it left the document silent about a declared question,
        // which reads exactly like a question nobody asked for.
        let comparing = question.across_cut.then_some(adjacent_frame).flatten();
        if question.across_cut && comparing.is_none() {
            let observation = unasked_observation(
                question,
                "no adjacent selected take to compare against, so the cut was never looked at",
            );
            if let Some(flag) = flag_for(shot_id, question, &observation, uncertain_below) {
                mismatches.push(flag);
            }
            observations.push(observation);
            continue;
        }
        // A question whose backend answer timed out on ANY of its frames is recorded `unobserved`
        // with the timeout as its note (sc-22715) — never a value, and never an error that would
        // discard the take's other evidence. The remaining questions are still asked, under the
        // review's own deadline.
        let mut answers = Vec::new();
        let mut timed_out: Option<String> = None;
        for frame in question.frames.select(frames) {
            match ask_one(
                vision,
                project_id,
                question,
                frame,
                limits,
                &mut real_model_inference,
            )
            .await?
            {
                Asked::Answer(answer) => answers.push(answer),
                Asked::TimedOut(note) => {
                    timed_out = Some(note);
                    break;
                }
            }
        }
        let theirs = match (timed_out.is_none(), comparing) {
            (true, Some(reference)) => {
                match ask_one(
                    vision,
                    project_id,
                    question,
                    reference,
                    limits,
                    &mut real_model_inference,
                )
                .await?
                {
                    Asked::Answer(answer) => Some(answer),
                    Asked::TimedOut(note) => {
                        timed_out = Some(note);
                        None
                    }
                }
            }
            _ => None,
        };
        let observation = match (timed_out, theirs) {
            (Some(note), _) => {
                // No model produced this question's (absent) answer, so the document must not
                // claim every answer came from a real model run.
                real_model_inference = false;
                unasked_observation(question, &note)
            }
            (None, Some(theirs)) => aggregate_cut_observation(question, answers, theirs),
            (None, None) => aggregate_observation(question, answers),
        };
        // Unconditional, not a `debug_assert!`: the "unobserved carries no value" invariant is the
        // one this whole module exists to hold, and a release build is exactly where an observation
        // claiming an unseen handoff would do harm.
        if let Some(error) = observation.well_formed_error() {
            return Err(HarnessError::Refused(format!(
                "refusing to record a malformed observation for shot {shot_id}: {error}"
            )));
        }
        if let Some(flag) = flag_for(shot_id, question, &observation, uncertain_below) {
            mismatches.push(flag);
        }
        observations.push(observation);
    }
    Ok(Answered {
        observations,
        mismatches,
        stop,
        real_model_inference,
    })
}

#[allow(clippy::too_many_arguments)]
async fn extract_frame(
    client: &Client<'_>,
    project_id: &str,
    timeline_id: &str,
    item_id: &str,
    shot_id: &str,
    attempt: u32,
    frame_id: &str,
    timestamp: f64,
    deadline: Instant,
    options: &ReviewOptions,
) -> Result<Option<FrameEvidence>, HarnessError> {
    let created = client
        .expect_ok(
            "POST",
            &format!(
                "/api/v1/projects/{project_id}/timelines/{timeline_id}/items/{item_id}/frames"
            ),
            Some(json!({ "playheadSeconds": timestamp, "intendedUse": "reuse" })),
        )
        .await?;
    let job_id = created
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| HarnessError::Transport(format!("frame job response has no id: {created}")))?
        .to_owned();
    let (view, poll_stop) = client
        .wait_for_job(&job_id, poll_bounds(deadline, options.poll_interval))
        .await?;
    // The review's `limits.maxSeconds` (or a cancel) ran out while a frame was being extracted:
    // the job was cancelled, and that is a STOP the caller records with the partial evidence
    // (sc-22715) — not a transport error that would throw the evidence away.
    if matches!(
        poll_stop,
        PollStop::ShotBudget | PollStop::RunBudget | PollStop::Operator
    ) {
        return Ok(None);
    }
    if view.status != "completed" {
        return Err(HarnessError::Transport(format!(
            "frame extraction job {job_id} ended {}: {}",
            view.status,
            view.failure_text()
        )));
    }
    let asset = view
        .result
        .get("assets")
        .and_then(Value::as_array)
        .and_then(|assets| assets.first())
        .cloned()
        .ok_or_else(|| {
            HarnessError::Transport(format!(
                "frame extraction job {job_id} completed without an asset: {}",
                view.result
            ))
        })?;
    Ok(Some(FrameEvidence {
        id: frame_id.to_owned(),
        shot_id: shot_id.to_owned(),
        attempt,
        timestamp_seconds: timestamp,
        asset_id: asset
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        path: asset
            .pointer("/file/path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        source: "frame_extract".to_owned(),
        job_id: Some(job_id),
        captured_at: utc_now(),
    }))
}

/// The shot whose selected take this one sits against across a cut, from the plan's declared edges
/// and the CURRENT selection. Never from a review.
fn adjacent_take(plan: &ProductionPlan, record: &RunRecord, shot_id: &str) -> Option<AdjacentTake> {
    let shot = plan.shots.iter().find(|shot| shot.id == shot_id)?;
    for edge in &shot.depends_on {
        let Some(previous) = record.shot(&edge.shot_id) else {
            continue;
        };
        let Some(attempt) = previous.selected_attempt else {
            continue;
        };
        let Some(take) = previous.selected().and_then(|record| record.take.as_ref()) else {
            continue;
        };
        return Some(AdjacentTake {
            shot_id: edge.shot_id.clone(),
            attempt,
            asset_id: take.asset_id.clone(),
            dependency: edge.kind.clone(),
            frame_id: None,
        });
    }
    None
}

/// Write the observed-state document and index it in the run record.
///
/// The index entry is a POINTER and some counts. Nothing observed is written into the record, and
/// in particular nothing is written into `intended` or `conditioningAssets`.
fn record_review(
    context: &mut ReviewContext,
    shot_id: &str,
    observed: &ObservedState,
) -> Result<PathBuf, HarnessError> {
    let dir = context.out_dir.join(REVIEWS_DIR);
    std::fs::create_dir_all(&dir)?;
    let file = format!(
        "{}-a{}-r{}.json",
        shot_id.to_ascii_lowercase(),
        observed.attempt,
        context
            .record
            .shot(shot_id)
            .map(|shot| shot.reviews.len() + 1)
            .unwrap_or(1)
    );
    let path = dir.join(&file);
    let json = serde_json::to_string_pretty(&observed.to_json())
        .map_err(|error| HarnessError::Io(error.to_string()))?;
    super::write_atomically(&path, json.as_bytes())?;

    let mut topics: Vec<String> = observed
        .actionable()
        .iter()
        .map(|flag| flag.topic.clone())
        .collect();
    topics.sort();
    topics.dedup();
    let summary = TakeReviewSummary {
        review_id: observed.review_id.clone(),
        reviewed_at: observed.reviewed_at.clone(),
        attempt: observed.attempt,
        record_path: format!("{REVIEWS_DIR}/{file}"),
        backend: observed.backend.kind.clone(),
        model: observed.backend.model.clone(),
        observations: observed.observations.len() as u32,
        unobserved: observed.unobserved_count() as u32,
        actionable_flags: observed.actionable().len() as u32,
        topics_flagged: topics,
        stop: observed.stop.clone(),
    };
    let detail = format!(
        "reviewed attempt {} ({} question(s), {} unobserved, {} actionable flag(s)) via {} -> {}",
        summary.attempt,
        summary.observations,
        summary.unobserved,
        summary.actionable_flags,
        summary.backend,
        summary.record_path
    );
    if let Some(shot) = context.record.shot_mut(shot_id) {
        shot.reviews.push(summary);
    }
    context
        .record
        .decisions
        .push(sceneworks_core::film_plan::ProductionDecision {
            at: utc_now(),
            action: "review".to_owned(),
            shot_id: Some(shot_id.to_owned()),
            detail,
        });
    Ok(path)
}

/// Read an observed-state document back off disk, refusing one whose observations break the
/// "unobserved carries no value" invariant.
///
/// The pair (`unobserved`, `observed`) is independently settable in the serialized shape, so a
/// hand-edited or foreign document CAN say that an unseen handoff completed. Parsing one and
/// handing it to `request-repair` (which reads its flags) or to a person would launder that into a
/// fact, which is the one thing this module must not do.
pub fn read_observed_state(path: &Path) -> Result<ObservedState, HarnessError> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        HarnessError::Refused(format!("cannot read {}: {error}", path.display()))
    })?;
    let observed: ObservedState = serde_json::from_str(&text).map_err(|error| {
        HarnessError::Refused(format!(
            "{} is not an observed-state document: {error}",
            path.display()
        ))
    })?;
    for observation in &observed.observations {
        if let Some(error) = observation.well_formed_error() {
            return Err(HarnessError::Refused(format!(
                "{} is not a usable observed-state document: {error}",
                path.display()
            )));
        }
    }
    Ok(observed)
}

// ---------------------------------------------------------------------------------------------
// The human loop
// ---------------------------------------------------------------------------------------------

/// The two decisions a person can record without re-rendering anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// The person looked and is happy with the selected take.
    Accept,
    /// The person rejects it. The take, its job and its asset stay in the record.
    Reject,
}

impl Decision {
    fn action(&self) -> &'static str {
        match self {
            Self::Accept => "accept_take",
            Self::Reject => "reject_take",
        }
    }

    fn state(&self) -> &'static str {
        match self {
            Self::Accept => "accepted",
            Self::Reject => "rejected",
        }
    }
}

/// Record a human's accept or reject of a shot's take.
///
/// Touches the API not at all — this is a decision, not a render — and touches no shot but the one
/// named and the shots that DECLARED a dependency on it:
///
/// * **accept** clears the shot's own `needsReview` flags (the person read them) and leaves the
///   selection where it is;
/// * **reject** marks the take `rejection` (it stays, with its job and its asset), clears the
///   selection, flags the declared dependents and marks the export stale. Nothing is re-rendered:
///   `film-harness request-repair` or `resume` is a separate, explicit act — and because nothing
///   is, the decision says in so many words that the shot stays in the cut carrying the take that
///   was just rejected, exactly as the failed-replacement path does (sc-22715).
pub fn decide_take(
    out_dir: &Path,
    shot_id: &str,
    decision: Decision,
    reason: &str,
) -> Result<RunRecord, HarnessError> {
    let _lease = super::ControllerLease::acquire(
        out_dir,
        format!("decision_{}", uuid::Uuid::new_v4().simple()),
    )?;
    decide_take_with_lease(out_dir, shot_id, decision, reason)
}

fn decide_take_with_lease(
    out_dir: &Path,
    shot_id: &str,
    decision: Decision,
    reason: &str,
) -> Result<RunRecord, HarnessError> {
    let mut context = ReviewContext::open(out_dir, None, false)?;
    let Some(index) = context
        .record
        .shots
        .iter()
        .position(|shot| shot.shot_id == shot_id)
    else {
        return Err(HarnessError::Refused(format!(
            "run {} has no record for shot {shot_id}",
            context.record.run_id
        )));
    };
    let shot = &context.record.shots[index];
    let attempt = match decision {
        Decision::Accept => accept_target_attempt(shot),
        Decision::Reject => target_attempt(shot),
    }
    .ok_or_else(|| {
        HarnessError::Refused(format!(
            "shot {shot_id} has no take to accept or reject in run {}",
            context.record.run_id
        ))
    })?;
    // Accepting a REJECTED take would re-select it — rejection record and all — and clear the
    // shot's `needsReview` flags while its dependents keep theirs and the export stays stale. The
    // record would then say the same attempt was both rejected and accepted, and no verb would
    // have moved the selection to anything anybody approved.
    if decision == Decision::Accept {
        if let Some(rejection) = shot
            .attempts
            .iter()
            .find(|candidate| candidate.attempt == attempt)
            .and_then(|candidate| candidate.rejection.as_ref())
        {
            return Err(HarnessError::Refused(format!(
                "attempt {attempt} of shot {shot_id} was rejected at {} ({}), so accepting it \
                 would confirm a take this run already threw away; render another with \
                 `film-harness replace-take` (or `request-repair`), or point the shot at a take \
                 that exists with `film-harness swap-take --shot {shot_id} --asset ASSET_ID`",
                rejection.at, rejection.reason
            )));
        }
    }
    let at = utc_now();
    let mut detail = format!("attempt {attempt}: {reason}");

    match decision {
        Decision::Accept => {
            context.record.shots[index].selected_attempt = Some(attempt);
            let cleared = context.record.shots[index].needs_review.len();
            context.record.shots[index].needs_review.clear();
            if cleared > 0 {
                detail.push_str(&format!(
                    " (cleared {cleared} needsReview flag(s) this shot was carrying)"
                ));
            }
        }
        Decision::Reject => {
            if let Some(record) = context.record.shots[index]
                .attempts
                .iter_mut()
                .find(|candidate| candidate.attempt == attempt)
            {
                record.rejection = Some(TakeRejection {
                    at: at.clone(),
                    reason: reason.to_owned(),
                });
            }
            context.record.shots[index].selected_attempt = None;
            let flagged = flag_dependents(
                &mut context.record,
                &context.plan,
                shot_id,
                &format!("its take was rejected by hand ({reason})"),
            );
            if let Some(export) = context.record.export.as_mut() {
                export.stale = true;
            }
            detail.push_str(&format!(
                " (the take, its job and its asset are kept; {flagged} declared dependent(s) \
                 flagged needsReview; nothing was re-rendered)"
            ));
            // The SAME annotation the failed-replacement path writes (`super::replace_take`), and
            // for the same reason (sc-22715): rejecting a take does not take the shot out of the
            // sequence, so the timeline — and any MP4 exported from it — still carry the take the
            // human just threw away. Without this the record left the shot `rendered` with
            // `selectedAttempt: null` and nothing anywhere said what the cut actually shows.
            detail.push_str(&format!(
                "; shot {shot_id} stays in the sequence, so the timeline and the exported MP4 \
                 still carry the REJECTED take until `film-harness replace-take --shot {shot_id}` \
                 (or `request-repair`) renders another"
            ));
        }
    }

    context.record.shots[index].human_decision = Some(HumanTakeDecision {
        state: decision.state().to_owned(),
        at: at.clone(),
        attempt,
        reason: reason.to_owned(),
    });
    context
        .record
        .decisions
        .push(sceneworks_core::film_plan::ProductionDecision {
            at,
            action: decision.action().to_owned(),
            shot_id: Some(shot_id.to_owned()),
            detail,
        });
    context.persist()?;
    Ok(context.record)
}

/// The attempt a decision is about: the selected one, else the last one that still holds a take.
fn target_attempt(shot: &ShotRunRecord) -> Option<u32> {
    shot.selected_attempt.or_else(|| {
        shot.attempts
            .iter()
            .rev()
            .find(|attempt| attempt.has_live_take())
            .map(|attempt| attempt.attempt)
    })
}

/// The attempt an ACCEPT is about. The same resolution, plus a last fallback to an attempt whose
/// take was rejected — resolved only so [`decide_take`] can refuse it BY NAME. Without the
/// fallback, accepting a shot whose only take was rejected reports "no take to accept", which is
/// both wrong (the take is right there, kept on purpose) and says nothing about what to do next.
fn accept_target_attempt(shot: &ShotRunRecord) -> Option<u32> {
    target_attempt(shot).or_else(|| {
        shot.attempts
            .iter()
            .rev()
            .find(|attempt| attempt.take.is_some())
            .map(|attempt| attempt.attempt)
    })
}

/// One bounded repair: exactly one new attempt for `shot_id`, with the review's own mismatch flags
/// folded into the reason so the record says what the repair was for.
///
/// It is [`super::replace_take`] underneath — the same single, human-authorised, non-looping
/// attempt, the same retention of the rejected take, the same flagging (never regenerating) of
/// declared dependents. This adds the WHY and nothing else; in particular it does not read the
/// observed state into the prompt, because a vision model's reading of a take is not an input to
/// the next one.
pub async fn request_repair(
    transport: &dyn ApiTransport,
    options: &super::ResumeOptions,
    shot_id: &str,
    reason: &str,
) -> Result<RunRecord, HarnessError> {
    let lease = super::ControllerLease::acquire(
        &options.out_dir,
        format!("repair_{}", uuid::Uuid::new_v4().simple()),
    )?;
    request_repair_with_lease(transport, options, shot_id, reason, lease).await
}

pub(crate) async fn request_repair_with_lease(
    transport: &dyn ApiTransport,
    options: &super::ResumeOptions,
    shot_id: &str,
    reason: &str,
    _lease: super::ControllerLease,
) -> Result<RunRecord, HarnessError> {
    super::record_action(
        &options.out_dir,
        "repair",
        Some(shot_id),
        request_repair_inner(transport, options, shot_id, reason),
    )
    .await
}

async fn request_repair_inner(
    transport: &dyn ApiTransport,
    options: &super::ResumeOptions,
    shot_id: &str,
    reason: &str,
) -> Result<RunRecord, HarnessError> {
    let mut context = ReviewContext::open(&options.out_dir, None, false)?;
    let Some(shot) = context.record.shot(shot_id).cloned() else {
        return Err(HarnessError::Refused(format!(
            "run {} has no record for shot {shot_id}",
            context.record.run_id
        )));
    };
    let folded = fold_repair_reason(&context.out_dir, &shot, reason);
    context
        .record
        .decisions
        .push(sceneworks_core::film_plan::ProductionDecision {
            at: utc_now(),
            action: "request_repair".to_owned(),
            shot_id: Some(shot_id.to_owned()),
            detail: format!("one bounded repair attempt authorised: {folded}"),
        });
    // Written before dispatch so the authorisation survives a controller that dies during it, and
    // so `replace_take` — which re-reads the record from disk — starts from it.
    context.persist()?;
    super::replace_take_operation(transport, options, shot_id, &folded, "repair").await
}

/// Build the repair reason: the person's words, plus the actionable flags of the most recent
/// review OF THE SELECTED TAKE. A review of a take that has since been replaced says nothing about
/// the one being repaired, so it is not folded in.
fn fold_repair_reason(out_dir: &Path, shot: &ShotRunRecord, reason: &str) -> String {
    let reason = if reason.trim().is_empty() {
        "repair requested by hand"
    } else {
        reason.trim()
    };
    let Some(summary) = shot.review_of_selected() else {
        return reason.to_owned();
    };
    let Ok(observed) = read_observed_state(&out_dir.join(&summary.record_path)) else {
        return reason.to_owned();
    };
    let flags: Vec<String> = observed
        .actionable()
        .iter()
        .map(|flag| {
            format!(
                "{} [{}] intended {:?}, observed {:?}",
                flag.topic, flag.severity, flag.intended, flag.observed
            )
        })
        .collect();
    if flags.is_empty() {
        return format!(
            "{reason} (review {} raised no actionable flags)",
            summary.review_id
        );
    }
    format!(
        "{reason} — review {} flagged: {}",
        summary.review_id,
        flags.join("; ")
    )
}

// ---------------------------------------------------------------------------------------------
// Labeled evaluation
// ---------------------------------------------------------------------------------------------

/// What one `film-harness review-eval` invocation needs.
#[derive(Debug, Clone)]
pub struct EvalOptions {
    /// The labeled-set document.
    pub set_path: PathBuf,
    /// Where the results, the report and the per-case observed-state documents are written.
    pub out_dir: PathBuf,
    /// Override the set's own `mediaRoot`. What points the shipped labels at real takes whose
    /// media is too large to check in.
    pub media_root: Option<PathBuf>,
    /// Project to import frames into. Created (or adopted) by name when absent.
    pub project_id: Option<String>,
    pub poll_interval: Duration,
    pub control: RunControl,
}

impl EvalOptions {
    pub fn new(set_path: PathBuf, out_dir: PathBuf) -> Self {
        Self {
            set_path,
            out_dir,
            media_root: None,
            project_id: None,
            poll_interval: Duration::from_secs(3),
            control: RunControl::new(),
        }
    }
}

/// A labeled set and the review plan it scores, both read and validated against each other.
struct LabeledSet {
    set: EvalSet,
    set_dir: PathBuf,
    review_path: PathBuf,
    review_bytes: Vec<u8>,
    review_plan: ReviewPlan,
}

/// Read a labeled set and the review plan it names, refusing a set the plan cannot score.
fn read_labeled_set(set_path: &Path) -> Result<LabeledSet, HarnessError> {
    let set_text = std::fs::read_to_string(set_path).map_err(|error| {
        HarnessError::Refused(format!(
            "cannot read the labeled set at {}: {error}",
            set_path.display()
        ))
    })?;
    let set: EvalSet = parse_eval_set(&set_text)
        .map_err(|error| HarnessError::Refused(format!("labeled set: {error}")))?;
    let set_dir = set_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let review_path = resolve_under(&set_dir, &set.review_plan);
    let review_bytes = std::fs::read(&review_path).map_err(|error| {
        HarnessError::Refused(format!(
            "cannot read the review plan at {}: {error}",
            review_path.display()
        ))
    })?;
    let review_plan = sceneworks_core::film_review::parse_review_plan(
        std::str::from_utf8(&review_bytes).unwrap_or_default(),
    )
    .map_err(|error| HarnessError::Refused(format!("review plan: {error}")))?;
    let findings = validate_eval_set(&set, &review_plan);
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    Ok(LabeledSet {
        set,
        set_dir,
        review_path,
        review_bytes,
        review_plan,
    })
}

/// The bounds a review of `out_dir` will run under, read off the review document before anything is
/// dispatched. What [`VqaVision::preflight`] checks the host against.
pub fn review_limits(
    out_dir: &Path,
    review_plan_path: Option<&Path>,
) -> Result<ReviewLimits, HarnessError> {
    let context = ReviewContext::open(out_dir, review_plan_path, true)?;
    Ok(context.review_plan()?.limits)
}

/// The bounds the labeled set at `set_path` is scored under, for the same preflight.
pub fn eval_review_limits(set_path: &Path) -> Result<ReviewLimits, HarnessError> {
    Ok(read_labeled_set(set_path)?.review_plan.limits)
}

/// Run the reviewer over a fixed labeled set and report what it caught, missed and cried wolf over.
///
/// This measures the REVIEWER, not the takes: the frames are already on disk and already labeled.
/// Its report always ends with [`ASSISTIVE_NOTICE`], because a table of detection counts is
/// exactly the artefact someone would otherwise read as a quality bar.
pub async fn review_eval(
    transport: &dyn ApiTransport,
    options: &EvalOptions,
    vision: &dyn ReviewVision,
) -> Result<EvalResults, HarnessError> {
    let started = Instant::now();
    let LabeledSet {
        set,
        set_dir,
        review_path,
        review_bytes,
        review_plan,
    } = read_labeled_set(&options.set_path)?;
    let media_root = options
        .media_root
        .clone()
        .or_else(|| {
            set.media_root
                .as_ref()
                .map(|root| resolve_under(&set_dir, root))
        })
        .unwrap_or_else(|| set_dir.clone());

    let client = Client {
        transport,
        control: &options.control,
    };
    let project_id = ensure_eval_project(&client, options, &set).await?;
    std::fs::create_dir_all(&options.out_dir)?;

    let mut observed_states: Vec<(EvalCase, ObservedState)> = Vec::new();
    for case in &set.cases {
        let spec = review_plan
            .shots
            .get(&case.shot_id)
            .expect("validated against the review plan");
        let mut frames = Vec::new();
        let mut refs = Vec::new();
        for (index, frame) in case.frames.iter().enumerate() {
            let path = resolve_media_path(&media_root, &case.id, &frame.file)?;
            if !path.exists() {
                return Err(HarnessError::Refused(format!(
                    "labeled case {} names a frame that is not on this host: {} — pass \
                     --media-root DIR pointing at the media this set describes",
                    case.id,
                    path.display()
                )));
            }
            let id = format!("{}-f{}", case.id, index + 1);
            frames.push(FrameEvidence {
                id: id.clone(),
                shot_id: case.shot_id.clone(),
                attempt: 0,
                timestamp_seconds: frame.timestamp_seconds,
                asset_id: String::new(),
                path: path.display().to_string(),
                source: "labeled_set".to_owned(),
                job_id: None,
                captured_at: utc_now(),
            });
            refs.push(FrameRef {
                id,
                asset_id: None,
                path,
                timestamp_seconds: frame.timestamp_seconds,
            });
        }
        // The across-cut question compares against the take this one cuts FROM, which the case
        // declares. Comparing a case against its OWN last frame would score a comparison nobody
        // asked about; a case that declares no neighbour simply does not score its cut question.
        let mut adjacent_frame = None;
        if let Some(frame) = case.adjacent_frames.last() {
            let path = resolve_media_path(&media_root, &case.id, &frame.file)?;
            if !path.exists() {
                return Err(HarnessError::Refused(format!(
                    "labeled case {} names an adjacent frame that is not on this host: {} — pass \
                     --media-root DIR pointing at the media this set describes",
                    case.id,
                    path.display()
                )));
            }
            let id = format!("{}-adjacent", case.id);
            frames.push(FrameEvidence {
                id: id.clone(),
                shot_id: case.shot_id.clone(),
                attempt: 0,
                timestamp_seconds: frame.timestamp_seconds,
                asset_id: String::new(),
                path: path.display().to_string(),
                source: "labeled_set_adjacent".to_owned(),
                job_id: None,
                captured_at: utc_now(),
            });
            // Deliberately NOT pushed into `refs`: those are the take's OWN frames, and a
            // `frames: "last"` question must not end up asking about the neighbour.
            adjacent_frame = Some(FrameRef {
                id,
                asset_id: None,
                path,
                timestamp_seconds: frame.timestamp_seconds,
            });
        }
        let case_started = Instant::now();
        let deadline = case_started + Duration::from_secs(review_plan.limits.max_seconds);
        let answered = answer_questions(
            vision,
            &project_id,
            &case.shot_id,
            &spec.questions,
            &refs,
            adjacent_frame.as_ref(),
            review_plan.limits,
            review_plan.uncertain_below,
            deadline,
            &options.control,
        )
        .await?;
        let Answered {
            observations,
            mismatches,
            stop,
            real_model_inference,
        } = answered;
        // Fill in the asset ids the backend actually cited, so the evidence is openable.
        for frame in frames.iter_mut() {
            if frame.asset_id.is_empty() {
                if let Some(reference) = refs
                    .iter()
                    .chain(adjacent_frame.iter())
                    .find(|reference| reference.id == frame.id)
                {
                    frame.asset_id = vision.prepare_frame(&project_id, reference).await?;
                }
            }
        }
        let observed = ObservedState {
            schema_version: OBSERVED_STATE_SCHEMA_VERSION,
            review_id: format!("eval:{}:{}", set.id, case.id),
            reviewed_at: utc_now(),
            shot_id: case.shot_id.clone(),
            attempt: 0,
            take_asset_id: String::new(),
            intended: IntendedRef {
                run_id: format!("eval:{}", set.id),
                shot_id: case.shot_id.clone(),
                plan_id: review_plan.id.clone(),
                plan_version: review_plan.version,
                plan_sha256: sha256_hex(&review_bytes),
                record_pointer: format!("/shots/{}", case.shot_id),
                record_path: review_path.display().to_string(),
            },
            review_plan: ReviewSourceRef {
                id: review_plan.id.clone(),
                version: review_plan.version,
                path: review_path.display().to_string(),
                sha256: sha256_hex(&review_bytes),
            },
            backend: {
                let mut backend = vision.record();
                backend.real_model_inference &= real_model_inference;
                backend
            },
            limits: review_plan.limits,
            adjacent: None,
            frames,
            observations,
            mismatches,
            stop,
            elapsed_seconds: case_started.elapsed().as_secs_f64(),
            notice: ASSISTIVE_NOTICE.to_owned(),
        };
        observed_states.push((case.clone(), observed));
    }

    let mut per_case: Vec<CaseOutcome> = Vec::new();
    for (case, observed) in &observed_states {
        let path = options.out_dir.join(format!("{}.observed.json", case.id));
        let json = serde_json::to_string_pretty(&observed.to_json())
            .map_err(|error| HarnessError::Io(error.to_string()))?;
        super::write_atomically(&path, json.as_bytes())?;
        let mut outcome = score_case(case, observed);
        outcome.observed_state_path = path.display().to_string();
        per_case.push(outcome);
    }
    let scored: Vec<(&EvalCase, &ObservedState, CaseOutcome)> = observed_states
        .iter()
        .zip(per_case.iter())
        .map(|((case, observed), outcome)| (case, observed, outcome.clone()))
        .collect();
    let (totals, per_question, per_topic) = tally(&review_plan, &set, &scored);
    // The report's own backend line is honest about the whole run: one scripted answer anywhere
    // means no model ran for this evaluation.
    let mut backend = vision.record();
    backend.real_model_inference = !observed_states.is_empty()
        && observed_states
            .iter()
            .all(|(_, observed)| observed.backend.real_model_inference);

    let results = EvalResults {
        schema_version: REVIEW_EVAL_SCHEMA_VERSION,
        ran_at: utc_now(),
        set_id: set.id.clone(),
        set_version: set.version,
        review_plan: ReviewSourceRef {
            id: review_plan.id.clone(),
            version: review_plan.version,
            path: review_path.display().to_string(),
            sha256: sha256_hex(&review_bytes),
        },
        backend,
        totals,
        per_question,
        per_topic,
        per_case,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        notice: ASSISTIVE_NOTICE.to_owned(),
    };
    let json = serde_json::to_string_pretty(&results)
        .map_err(|error| HarnessError::Io(error.to_string()))?;
    super::write_atomically(&options.out_dir.join("review-eval.json"), json.as_bytes())?;
    super::write_atomically(
        &options.out_dir.join("review-eval.txt"),
        format_eval_report(&results).as_bytes(),
    )?;
    Ok(results)
}

/// Resolve a document-declared path (the review plan, the media root) against the document's own
/// directory. These MAY be absolute or `~`-relative: the real-takes set points at media outside the
/// repository on purpose.
fn resolve_under(base: &Path, value: &str) -> PathBuf {
    let candidate = PathBuf::from(shellexpand_home(value));
    if candidate.is_absolute() {
        candidate
    } else {
        base.join(candidate)
    }
}

/// Resolve one labeled FRAME under the media root, refusing anything that could leave it.
///
/// A frame path is a name under the root, never a path in its own right: `review-eval` reads these
/// and `review-fixtures` WRITES them, so `..` or an absolute value would make a labels file a read
/// and write primitive for any directory on the host. [`validate_eval_set`] refuses such a document
/// up front; this is the second half of the same rule, held at the moment the path is built, so no
/// caller can reach a file outside the root even if it skipped validation.
fn resolve_media_path(root: &Path, case_id: &str, file: &str) -> Result<PathBuf, HarnessError> {
    if let Some(reason) = unsafe_media_path(file) {
        return Err(HarnessError::Refused(format!(
            "labeled case {case_id} names the frame {file:?}, which {reason}"
        )));
    }
    let path = root.join(file);
    if !path.starts_with(root) {
        return Err(HarnessError::Refused(format!(
            "labeled case {case_id} resolves the frame {file:?} to {}, which is outside the media \
             root {}",
            path.display(),
            root.display()
        )));
    }
    Ok(path)
}

/// `~`-relative paths are how a labels file points at media outside the repository without
/// hard-coding somebody's home directory.
fn shellexpand_home(value: &str) -> String {
    let Some(rest) = value.strip_prefix("~/") else {
        return value.to_owned();
    };
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() => format!("{home}/{rest}"),
        _ => value.to_owned(),
    }
}

async fn ensure_eval_project(
    client: &Client<'_>,
    options: &EvalOptions,
    set: &EvalSet,
) -> Result<String, HarnessError> {
    if let Some(id) = &options.project_id {
        return Ok(id.clone());
    }
    let name = format!("film-harness review-eval ({} v{})", set.id, set.version);
    let existing = client
        .expect_ok("GET", "/api/v1/projects", None)
        .await?
        .as_array()
        .into_iter()
        .flatten()
        .find(|project| project.get("name").and_then(Value::as_str) == Some(name.as_str()))
        .and_then(|project| project.get("id"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    if let Some(id) = existing {
        return Ok(id);
    }
    let created = client
        .expect_ok("POST", "/api/v1/projects", Some(json!({ "name": name })))
        .await?;
    created
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| HarnessError::Transport(format!("project response has no id: {created}")))
}

/// Write a deterministic placeholder frame for every frame the labeled set at `set_path` names,
/// and return the paths.
///
/// These are **placeholders, not footage**: flat plates from the same generator as the reference
/// fixtures, one colour per case. They exist so the shipped labeled set is self-contained and the
/// deterministic tests (which drive [`ScriptedVision`], and never look at a pixel) can run on any
/// machine. Pointing a real vision model at them measures nothing; for that, point `review-eval`
/// at a labels file whose `mediaRoot` holds real takes — `real-takes.jsonc` in the shipped set
/// directory is exactly that.
pub fn write_review_fixture_frames(
    set_path: &Path,
    media_root: Option<&Path>,
) -> Result<Vec<PathBuf>, HarnessError> {
    let text = std::fs::read_to_string(set_path).map_err(|error| {
        HarnessError::Refused(format!(
            "cannot read the labeled set at {}: {error}",
            set_path.display()
        ))
    })?;
    let set: EvalSet = parse_eval_set(&text)
        .map_err(|error| HarnessError::Refused(format!("labeled set: {error}")))?;
    let set_dir = set_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let root = media_root
        .map(Path::to_path_buf)
        .or_else(|| {
            set.media_root
                .as_ref()
                .map(|root| resolve_under(&set_dir, root))
        })
        .unwrap_or_else(|| set_dir.clone());
    let mut written = Vec::new();
    for case in &set.cases {
        for frame in case.frames.iter().chain(case.adjacent_frames.iter()) {
            // This loop WRITES a file per named frame, so the containment rule matters more here
            // than anywhere else in the module.
            let path = resolve_media_path(&root, &case.id, &frame.file)?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            // Seeded on the file name, so the frames of one case differ from each other the way
            // frames of a take do, and every plate is reproducible byte for byte from this file.
            let stem = path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| case.id.clone());
            let bytes = super::fixture_plate_png(&stem, case_plate_rgb(&case.label))?;
            std::fs::write(&path, bytes)?;
            written.push(path);
        }
    }
    Ok(written)
}

/// A per-label colour, so the placeholder plates are at least distinguishable in a file browser.
fn case_plate_rgb(label: &str) -> [u8; 3] {
    let seed = label.bytes().fold(11_u32, |acc, byte| {
        acc.wrapping_mul(37).wrapping_add(u32::from(byte))
    });
    [
        64 + (seed % 96) as u8,
        64 + ((seed / 97) % 96) as u8,
        64 + ((seed / 9409) % 96) as u8,
    ]
}

/// One screen of what a shot's reviews say, for `film-harness status` and `review`.
pub fn format_shot_reviews(shot: &ShotRunRecord) -> String {
    let mut out = String::new();
    for review in &shot.reviews {
        out.push_str(&format!(
            "           review {} attempt {} via {} — {} question(s), {} unobserved, {} \
             actionable flag(s){}\n             {}\n",
            review.review_id,
            review.attempt,
            review.backend,
            review.observations,
            review.unobserved,
            review.actionable_flags,
            review
                .stop
                .as_deref()
                .map(|stop| format!("  STOPPED: {stop}"))
                .unwrap_or_default(),
            review.record_path
        ));
        if !review.topics_flagged.is_empty() {
            out.push_str(&format!(
                "             flagged: {}\n",
                review.topics_flagged.join(", ")
            ));
        }
    }
    if let Some(decision) = &shot.human_decision {
        out.push_str(&format!(
            "           HUMAN {} attempt {} ({}): {}\n",
            decision.state.to_uppercase(),
            decision.attempt,
            decision.at,
            decision.reason
        ));
    }
    out
}
