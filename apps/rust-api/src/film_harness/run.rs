//! Controller leases, cancellation and resumable run orchestration.

use super::*;

/// How long to wait for a canceled job to reach a terminal state before the harness gives up on it.
/// The worker cancels cooperatively between stages, so this bounds the wait rather than the worker.
/// Capped at the plan's own per-shot budget, so a plan that declares a short shot also gets a short
/// grace.
pub(super) const CANCEL_GRACE: Duration = Duration::from_secs(30);

/// How long a TERMINAL job may keep raw `assetWrites` in its result — the window between the
/// worker's terminal status and the API's asset persistence (`persist_reported_assets`) — before
/// the harness stops waiting and records the attempt as terminal-but-unsettled. Also capped at the
/// per-shot budget.
pub(super) const ASSET_SETTLE_GRACE: Duration = Duration::from_secs(30);

/// How long to keep re-reading `GET /api/v1/jobs/:id/metrics` after a terminal attempt. The worker
/// POSTs its metrics block AFTER the terminal progress update, so the first read can legitimately
/// come back `null`.
pub(super) const METRICS_GRACE: Duration = Duration::from_secs(5);

/// Cadence of the metrics re-read inside [`METRICS_GRACE`] (independent of the job poll cadence,
/// which a real run sets to seconds).
pub(super) const METRICS_RETRY_INTERVAL: Duration = Duration::from_millis(250);

/// File `film-harness cancel` drops in the run directory. A controller in another process polls for
/// it, which is what makes cancellation work without a shared handle to the running controller
/// (sc-22711).
pub const CANCEL_SENTINEL_FILE: &str = "cancel.requested";
pub const CONTROLLER_LOCK_FILE: &str = "controller.lock";

/// Per-API-process shutdown intent shared with its Film controller leases.
///
/// SIGTERM is a graceful API shutdown, so Rust destructors still run even though a separately
/// hosted worker may continue the exact Film job. This process-local flag lets an active run keep
/// its durable owner marker through that teardown. Ordinary controller completion, failure, and
/// operator cancellation never set it and continue to clear the marker.
#[derive(Debug, Clone, Default)]
pub(crate) struct FilmControllerShutdown {
    pub(super) requested: Arc<AtomicBool>,
}

impl FilmControllerShutdown {
    pub(crate) fn request(&self) {
        self.requested.store(true, Ordering::Release);
    }

    pub(super) fn requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub struct ControllerLease {
    pub(super) path: PathBuf,
    pub(super) owner: String,
    pub(super) api_shutdown: Option<FilmControllerShutdown>,
    pub(super) _lock: FileLock,
}

impl ControllerLease {
    pub fn acquire(run_dir: &Path, owner: impl Into<String>) -> Result<Self, HarnessError> {
        let (path, lock) = Self::lock(run_dir)?;
        Self::claim(path, owner.into(), lock, None)
    }

    /// Acquire the controller for a separately authorized action and consume the cancel request
    /// addressed to the previous controller.
    ///
    /// Acquisition must happen first: otherwise a competing action could erase a fresh cancel
    /// while the controller it targets still owns the run. A cancel written after this returns is
    /// retained and observed by the new action's watching [`RunControl`]. Automatic startup
    /// adoption uses interrupted acquisition instead: it continues the canceled operation.
    pub fn acquire_new_action(
        run_dir: &Path,
        owner: impl Into<String>,
    ) -> Result<Self, HarnessError> {
        let lease = Self::acquire(run_dir, owner)?;
        clear_cancel_request(run_dir)?;
        Ok(lease)
    }

    pub(crate) fn acquire_for_api(
        run_dir: &Path,
        owner: impl Into<String>,
        api_shutdown: FilmControllerShutdown,
    ) -> Result<Self, HarnessError> {
        let (path, lock) = Self::lock(run_dir)?;
        Self::claim(path, owner.into(), lock, Some(api_shutdown))
    }

    /// Explicit API actions consume only the previous controller's cancellation, before any
    /// asynchronous admission or accepted response, while retaining shutdown recovery metadata.
    pub(crate) fn acquire_new_action_for_api(
        run_dir: &Path,
        owner: impl Into<String>,
        api_shutdown: FilmControllerShutdown,
    ) -> Result<Self, HarnessError> {
        let lease = Self::acquire_for_api(run_dir, owner, api_shutdown)?;
        clear_cancel_request(run_dir)?;
        Ok(lease)
    }

    /// Acquire a run only when its unlocked lease still names the controller that crashed.
    ///
    /// A clean controller release empties the metadata while it still holds the advisory lock;
    /// an abrupt process exit cannot run that cleanup, although the OS releases the lock. Reading
    /// the marker only after obtaining the lock makes this distinction atomic with takeover: a
    /// controller that finishes while startup is scanning cannot be mistaken for an interrupted
    /// one and restarted.
    #[cfg(test)]
    pub(crate) fn acquire_interrupted(
        run_dir: &Path,
        owner: impl Into<String>,
    ) -> Result<Option<Self>, HarnessError> {
        Self::acquire_interrupted_with_shutdown(run_dir, owner, None)
    }

    pub(crate) fn acquire_interrupted_for_api(
        run_dir: &Path,
        owner: impl Into<String>,
        api_shutdown: FilmControllerShutdown,
    ) -> Result<Option<Self>, HarnessError> {
        Self::acquire_interrupted_with_shutdown(run_dir, owner, Some(api_shutdown))
    }

    pub(super) fn acquire_interrupted_with_shutdown(
        run_dir: &Path,
        owner: impl Into<String>,
        api_shutdown: Option<FilmControllerShutdown>,
    ) -> Result<Option<Self>, HarnessError> {
        let (path, lock) = Self::lock(run_dir)?;
        // Windows byte-range locks also reject reads through a second handle in this process.
        // Inspect metadata through the handle that owns the lock, without moving the read before
        // acquisition (which would race a clean release or another controller's takeover).
        use std::io::{Read as _, Seek as _};
        let mut locked_file = lock.file();
        locked_file.rewind()?;
        let mut metadata = String::new();
        locked_file.read_to_string(&mut metadata)?;
        let interrupted = metadata.lines().any(|line| {
            line.strip_prefix("owner=")
                .is_some_and(|owner| !owner.is_empty())
        });
        if !interrupted {
            return Ok(None);
        }
        Self::claim(path, owner.into(), lock, api_shutdown).map(Some)
    }

    pub(super) fn lock(run_dir: &Path) -> Result<(PathBuf, FileLock), HarnessError> {
        std::fs::create_dir_all(run_dir)?;
        let path = run_dir.join(CONTROLLER_LOCK_FILE);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let lock = FileLock::try_exclusive(file).map_err(|error| {
            let contention = fs2::lock_contended_error().raw_os_error();
            if error.kind() == std::io::ErrorKind::WouldBlock
                || contention.is_some() && error.raw_os_error() == contention
            {
                HarnessError::Refused(format!(
                    "run mutation refused: another controller holds {}",
                    path.display()
                ))
            } else {
                HarnessError::Io(error.to_string())
            }
        })?;
        Ok((path, lock))
    }

    pub(super) fn claim(
        path: PathBuf,
        owner: String,
        lock: FileLock,
        api_shutdown: Option<FilmControllerShutdown>,
    ) -> Result<Self, HarnessError> {
        lock.file().set_len(0)?;
        use std::io::{Seek as _, Write as _};
        let mut locked_file = lock.file();
        // Truncation does not reset the cursor advanced by interrupted-owner inspection.
        locked_file.rewind()?;
        locked_file.write_all(
            format!(
                "owner={owner}\npid={}\nacquiredAt={}\n",
                std::process::id(),
                utc_now()
            )
            .as_bytes(),
        )?;
        locked_file.sync_all()?;
        Ok(Self {
            path,
            owner,
            api_shutdown,
            _lock: lock,
        })
    }

    /// Whether another process currently owns the run. The advisory lock, rather than the age or
    /// mere presence of its metadata file, is authoritative: the OS releases it after a crash.
    pub fn is_active(run_dir: &Path) -> Result<bool, HarnessError> {
        std::fs::create_dir_all(run_dir)?;
        let path = run_dir.join(CONTROLLER_LOCK_FILE);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        match FileLock::try_exclusive(file) {
            Ok(probe) => {
                drop(probe);
                Ok(false)
            }
            Err(error) => {
                let contention = fs2::lock_contended_error().raw_os_error();
                if error.kind() == std::io::ErrorKind::WouldBlock
                    || contention.is_some() && error.raw_os_error() == contention
                {
                    Ok(true)
                } else {
                    Err(HarnessError::Io(error.to_string()))
                }
            }
        }
    }
}

impl Drop for ControllerLease {
    fn drop(&mut self) {
        tracing::debug!(path = %self.path.display(), owner = %self.owner, "releasing film controller lease");
        let preserve_for_restart = self
            .api_shutdown
            .as_ref()
            .is_some_and(FilmControllerShutdown::requested)
            && self
                .path
                .parent()
                .and_then(|run_dir| read_run_record(run_dir).ok())
                .is_some_and(|record| record.state == RunState::Running);
        if preserve_for_restart {
            tracing::info!(
                path = %self.path.display(),
                owner = %self.owner,
                "preserving active film controller ownership for API restart"
            );
            return;
        }
        // A clean release erases its diagnostic owner while the advisory lock is still held. If
        // the process crashes this Drop never runs, so the surviving owner identifies which
        // operation startup found interrupted without being mistaken for live ownership.
        let _ = self._lock.file().set_len(0);
        let _ = self._lock.file().sync_all();
    }
}

/// Everything a run needs beyond the transport.
#[derive(Debug, Clone)]
pub struct RunOptions {
    pub plan_path: PathBuf,
    pub reference_pack_path: PathBuf,
    /// Compiled requests to dispatch. `None` looks for `compiled.json` beside the plan and, failing
    /// that, compiles the plan's own prompts in memory — which is exactly what a hand-authored plan
    /// has always done.
    pub compiled_path: Option<PathBuf>,
    /// Reuse this project instead of creating one named after the plan.
    pub project_id: Option<String>,
    /// Render only these shot ids (plan order is kept). `None` renders every shot.
    pub shot_ids: Option<Vec<String>>,
    /// Where `run.json` and the copied source documents are written.
    pub out_dir: PathBuf,
    /// Job polling cadence.
    pub poll_interval: Duration,
    /// Assemble and export the timeline after the shots render.
    pub export: bool,
    /// Refuse a model whose catalog entry is not `installState: "installed"`. Tests drive a fake
    /// worker that owns no weights and turn this off; a real run keeps it on so an uninstalled
    /// tier is refused here rather than by a failed job.
    pub require_installed: bool,
}

/// Cooperative cancellation for a run. The `film-harness` binary flips it from its `ctrl_c`
/// handler; the harness then cancels the in-flight job through the API, stops dispatching, and
/// still writes the run record — rather than leaving a 45-minute render on the GPU with no record
/// of it, which is the opposite of what the plan's limits are for.
/// sc-22711 adds the second source: a controller can also be asked to stop from ANOTHER process, by
/// `film-harness cancel --out DIR` dropping [`CANCEL_SENTINEL_FILE`] in the run directory. A
/// controller checks both before every dispatch and on every poll, so a cancel lands within one
/// poll interval either way.
#[derive(Debug, Clone, Default)]
pub struct RunControl {
    pub(super) canceled: Arc<AtomicBool>,
    pub(super) sentinel: Option<PathBuf>,
}

impl RunControl {
    pub fn new() -> Self {
        Self::default()
    }

    /// A control that is also tripped by the presence of `<run_dir>/cancel.requested`.
    pub fn watching(run_dir: &Path) -> Self {
        Self {
            canceled: Arc::new(AtomicBool::new(false)),
            sentinel: Some(run_dir.join(CANCEL_SENTINEL_FILE)),
        }
    }

    /// Ask the in-flight run to cancel its job and stop dispatching. Idempotent.
    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::SeqCst);
    }

    pub fn is_canceled(&self) -> bool {
        if self.canceled.load(Ordering::SeqCst) {
            return true;
        }
        self.sentinel
            .as_deref()
            .is_some_and(|path| path.try_exists().unwrap_or(false))
    }
}

/// Ask the run in `run_dir` to stop, from outside the process running it. Returns the sentinel it
/// wrote. A run that is not currently held picks this up on its next start, so
/// [`clear_cancel_request`] runs before a resume.
///
/// A directory with no run record in it is REFUSED rather than created: `cancel --out /typo/path`
/// otherwise printed "cancel requested" and exited 0 while the real 45-minute render kept going,
/// which is the one thing a cancel must never do.
pub fn request_cancel(run_dir: &Path) -> Result<PathBuf, HarnessError> {
    let record = run_dir.join(RUN_RECORD_FILE);
    if !record.try_exists().unwrap_or(false) {
        return Err(HarnessError::Refused(format!(
            "no run record in {} — nothing there to cancel (a run writes {RUN_RECORD_FILE} before \
             its first API call; check the --out path)",
            run_dir.display()
        )));
    }
    let path = run_dir.join(CANCEL_SENTINEL_FILE);
    std::fs::write(&path, format!("{}\n", utc_now()))?;
    Ok(path)
}

/// Remove a stale cancel sentinel so a resume is not cancelled by the request that stopped the
/// previous controller.
pub fn clear_cancel_request(run_dir: &Path) -> Result<(), HarnessError> {
    let path = run_dir.join(CANCEL_SENTINEL_FILE);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// What `resume` and `replace-take` need. Everything else — which plan, which pack, which project,
/// which shots, which limits — comes from the run record in `out_dir`, because those are properties
/// of the run being continued, not of the invocation continuing it (sc-22711).
#[derive(Debug, Clone)]
pub struct ResumeOptions {
    /// The run directory holding `run.json`.
    pub out_dir: PathBuf,
    pub poll_interval: Duration,
    /// Assemble and export the timeline. For `replace-take`, whether to re-export after the new
    /// take is selected (without it the existing export is simply marked stale).
    pub export: bool,
    pub require_installed: bool,
    pub control: RunControl,
}

impl ResumeOptions {
    /// Defaults for a run directory: watch that directory's cancel sentinel, poll every 5s, export.
    pub fn new(out_dir: PathBuf) -> Self {
        let control = RunControl::watching(&out_dir);
        Self {
            out_dir,
            poll_interval: Duration::from_secs(5),
            export: true,
            require_installed: true,
            control,
        }
    }
}

/// An asset `run_id` already imported for `role`, matched on the provenance the import stamped.
/// This is what keeps a replay from importing the same reference twice when the controller died
/// between the upload and the record write (sc-22711).
pub(super) fn find_imported_reference(
    assets: &[Value],
    run_id: &str,
    role: &str,
    sha256: &str,
) -> Option<String> {
    assets
        .iter()
        .find(|asset| {
            let field = |name: &str| {
                asset
                    .pointer("/extra/filmHarness")
                    .and_then(|block| block.get(name))
                    .and_then(Value::as_str)
            };
            field("runId") == Some(run_id)
                && field("role") == Some(role)
                && field("sha256") == Some(sha256)
        })
        .and_then(|asset| asset.get("id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// How long an attempt has already been in flight.
///
/// A controller that died mid-poll never wrote `elapsedSeconds`, so the recorded value alone would
/// hand a hung job a fresh per-shot budget on every restart. `startedAt` IS persisted at dispatch,
/// and the job kept running while nothing was watching it, so wall-clock since then is the honest
/// number; the larger of the two wins.
///
/// An attempt with NO job has spent nothing, whatever its `startedAt` says. The attempt record is
/// written BEFORE the job is created (that is what makes the idempotency key work), so a controller
/// that died in that window leaves `status: "dispatching", jobId: null` — nothing was rendered, and
/// charging it the wall clock since then would make a resume the next morning compute
/// `spent > maxShotSeconds`, dispatch a real job and cancel it on the first poll, spending the
/// shot's attempt on zero work.
pub(super) fn attempt_spent_seconds(attempt: &AttemptRecord) -> f64 {
    if attempt.job_id.is_none() {
        return 0.0;
    }
    let recorded = attempt.elapsed_seconds.max(0.0);
    let by_clock = parse_utc_seconds(&attempt.started_at)
        .map(|started| (sceneworks_core::time::now_unix_seconds() - started).max(0) as f64)
        .unwrap_or(0.0);
    recorded.max(by_clock)
}

/// One controller holding one run.
///
/// Everything that reaches the API goes through here, and every state change is followed by a
/// [`Session::persist`], so the record on disk is never more than one transition behind what the
/// API has. That is what makes the three entry points ([`run`], [`resume`], [`replace_take`]) the
/// same code: they differ only in the record they start from and the deadline they run under.
pub(super) struct Session<'a> {
    pub(super) client: Client<'a>,
    pub(super) transport: &'a dyn ApiTransport,
    pub(super) plan_path: PathBuf,
    pub(super) pack_path: PathBuf,
    pub(super) out_dir: PathBuf,
    pub(super) poll_interval: Duration,
    pub(super) export: bool,
    pub(super) plan: ProductionPlan,
    pub(super) pack: ReferencePack,
    /// The requests this session dispatches (sc-22713): the compiled document beside the plan, or
    /// the plan compiled in memory. Every job body comes from here, so what a reviewer reads in
    /// `compiled.json` is what the API receives — on a resume and a replacement exactly as on the
    /// first run.
    pub(super) compiled: CompiledPlan,
    pub(super) facts: HostFacts,
    pub(super) fps: u32,
    pub(super) record: RunRecord,
    pub(super) started: Instant,
    /// Wall-clock earlier controllers already spent on this run. The plan's `maxRunSeconds` bounds
    /// the run, not one attempt at it, so a resume inherits the spend.
    pub(super) prior_elapsed: f64,
    /// Wall-clock earlier HUMAN-REQUESTED work already spent on this run
    /// (`humanRequestedElapsedSeconds`), inherited the same way.
    pub(super) prior_human_elapsed: f64,
    /// Whether this controller's own wall-clock is charged to the run's automatic budget
    /// (`elapsedSeconds`) or booked as human-requested work (`humanRequestedElapsedSeconds`).
    /// `run` and `resume` charge the run; `replace-take` / `request-repair` do not (sc-22715): a
    /// replacement the person asked for must not spend the budget the run's own `resume` needs.
    pub(super) charges_run_budget: bool,
    /// `None` for a session outside the run budget (a replacement), so nothing it polls — the
    /// attempt or the export — can be classified `run_budget`.
    pub(super) run_deadline: Option<Instant>,
    pub(super) role_assets: BTreeMap<String, String>,
    /// The pack's sound clips this run has imported, by role (sc-22712). Rebuilt from the record on
    /// a resume exactly as [`Session::role_assets`] is, so a re-layout after a replacement places
    /// the same clips the first controller did without importing (and transcoding) them twice.
    pub(super) sound_assets: BTreeMap<String, SoundAsset>,
    /// Set the moment dispatch stops, with the outcome that stop implies.
    pub(super) stop: Option<(RunOutcome, RunStop)>,
}

impl Session<'_> {
    /// The model id one shot dispatches as and why, read straight off the compiled request so the
    /// attempt record, the job payload and `compiled.json` carry one string, not three derivations
    /// of it (sc-23402). A shot with no compiled request cannot be dispatched at all, so the
    /// fallback is only ever reached by a caller that is about to refuse.
    pub(super) fn resolved_partition(&self, shot_id: &str) -> (String, String) {
        match self.compiled.request(shot_id) {
            Some(request) => (request.model.clone(), request.partition_reason.clone()),
            None => (self.plan.model.id.clone(), String::new()),
        }
    }

    /// The EFFECTIVE reference-image short edge one shot dispatches at, or `None` for a shot that
    /// encodes no reference (sc-23402). Read off the compiled request for the same reason
    /// [`Session::resolved_partition`] is: the payload the route receives and the number the record
    /// keeps are then one resolution, not two.
    pub(super) fn resolved_reference_short_edge(&self, shot_id: &str) -> Option<u32> {
        self.compiled
            .request(shot_id)
            .and_then(CompiledRequest::effective_reference_image_short_edge)
    }

    /// The EFFECTIVE reference-image resize mode one shot dispatches with, or `None` for a shot
    /// that encodes no reference (sc-26691) — the companion of
    /// [`Session::resolved_reference_short_edge`], read off the same compiled request.
    pub(super) fn resolved_reference_resize_mode(
        &self,
        shot_id: &str,
    ) -> Option<sceneworks_core::video_request::ReferenceImageResizeMode> {
        self.compiled
            .request(shot_id)
            .and_then(CompiledRequest::effective_reference_image_resize_mode)
    }

    /// The LoRAs one shot dispatches with, the step count it renders at, and the turbo video shift
    /// that governs it (sc-23406) — all three read off the compiled request for the same reason
    /// [`Session::resolved_partition`] is, so the payload the route receives and the numbers the
    /// record keeps are one resolution rather than two.
    pub(super) fn resolved_sampling(
        &self,
        shot_id: &str,
    ) -> (Vec<String>, Option<u32>, Option<f64>) {
        match self.compiled.request(shot_id) {
            Some(request) => (
                request.loras.clone(),
                request.effective_steps,
                request.turbo_scheduler_shift,
            ),
            None => (Vec::new(), None, None),
        }
    }

    /// The partition one RECORDED attempt dispatched as, for the take it produced.
    ///
    /// A run record written before sc-23402 carries no `resolvedModelId` at all, and
    /// `#[serde(default)]` reads that as `""` — so a resume that adopted such an attempt used to
    /// record its take with `model: ""`, losing the only statement of which checkpoint made it.
    /// An empty value falls back to this shot's resolved partition, which is exactly what the
    /// first controller would have written, and the attempt row is BACKFILLED so the record
    /// self-heals on the resume that touched it rather than staying blank forever.
    ///
    /// `job` is the attempt's job as the API reports it now; it settles the attempt's resize mode
    /// (sc-26691) from what the job itself records — see
    /// [`AttemptRecord::settle_reference_image_resize_mode`].
    pub(super) fn dispatched_model_for(
        &mut self,
        shot_id: &str,
        shot_index: usize,
        attempt_index: usize,
        job: &JobView,
    ) -> String {
        let job_mode = job.reference_image_resize_mode;
        let job_rendered = job.status == "completed";
        let recorded = self.record.shots[shot_index].attempts[attempt_index]
            .resolved_model_id
            .clone();
        if !recorded.is_empty() {
            self.record.shots[shot_index].attempts[attempt_index]
                .settle_reference_image_resize_mode(job_mode, job_rendered);
            return recorded;
        }
        let (model, reason) = self.resolved_partition(shot_id);
        // Backfilled with the partition, and by the same rule (sc-23402): a pre-story record has no
        // `referenceImageShortEdge` either, and the value this shot resolves to is what the first
        // controller would have written. A base-partition shot resolves to `None`, so the absence
        // stays an absence rather than becoming a number that never applied.
        let short_edge = self.resolved_reference_short_edge(shot_id);
        let sampling = self.resolved_sampling(shot_id);
        let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
        attempt.resolved_model_id = model.clone();
        if attempt.partition_reason.is_empty() {
            attempt.partition_reason = reason;
        }
        if attempt.reference_image_short_edge.is_none() {
            attempt.reference_image_short_edge = short_edge;
        }
        // Same rule for the sampling provenance (sc-23406): a pre-story record carries none, and
        // what this shot resolves to is what the first controller would have written.
        if attempt.effective_steps.is_none() {
            let (loras, steps, shift) = sampling;
            if attempt.loras.is_empty() {
                attempt.loras = loras;
            }
            attempt.effective_steps = steps;
            attempt.turbo_scheduler_shift = shift;
        }
        // NOT the same rule for the resize mode (sc-26691): what this shot resolves to TODAY is not
        // what a pre-story attempt rendered. The job's own record decides — after the short edge
        // above says whether this was a reference attempt at all.
        attempt.settle_reference_image_resize_mode(job_mode, job_rendered);
        model
    }

    /// Total AUTOMATIC wall-clock this run has consumed, across every controller that has held it.
    /// A controller that does not charge the run budget contributes nothing here.
    pub(super) fn elapsed(&self) -> f64 {
        if self.charges_run_budget {
            self.prior_elapsed + self.started.elapsed().as_secs_f64()
        } else {
            self.prior_elapsed
        }
    }

    /// Total human-requested wall-clock, the mirror of [`Session::elapsed`].
    pub(super) fn human_requested_elapsed(&self) -> f64 {
        if self.charges_run_budget {
            self.prior_human_elapsed
        } else {
            self.prior_human_elapsed + self.started.elapsed().as_secs_f64()
        }
    }

    /// Write the record. Called after every state transition — this is the durability contract.
    pub(super) fn persist(&mut self) -> Result<(), HarnessError> {
        self.record.elapsed_seconds = self.elapsed();
        self.record.human_requested_elapsed_seconds = self.human_requested_elapsed();
        persist_record(
            &self.record,
            &self.out_dir,
            &self.plan_path,
            &self.pack_path,
        )?;
        Ok(())
    }

    pub(super) fn note_decision(&mut self, action: &str, shot_id: Option<&str>, detail: String) {
        self.record.decisions.push(ProductionDecision {
            at: utc_now(),
            action: action.to_owned(),
            shot_id: shot_id.map(str::to_owned),
            detail,
        });
    }

    pub(super) fn halt(
        &mut self,
        outcome: RunOutcome,
        reason: &str,
        detail: String,
        resumable: bool,
    ) {
        if self.stop.is_none() {
            self.stop = Some((
                outcome,
                RunStop {
                    reason: reason.to_owned(),
                    detail,
                    resumable,
                },
            ));
        }
    }

    pub(super) fn canceled(&self) -> bool {
        self.client.control.is_canceled()
    }

    /// Whether `memory` is over the plan's budget — and, when it is, halt the run with the memory
    /// stop.
    ///
    /// Both paths that settle an attempt go through here: [`Session::work_attempt`], which watched
    /// the render land, and [`Session::reconcile_shot`], which adopts one a dead controller was
    /// watching. A crash around an over-budget render must stop new dispatch exactly as observing
    /// it does — otherwise a resume adopts the take and keeps dispatching against a budget the
    /// evidence already says was blown.
    pub(super) fn memory_over_budget(&mut self, shot_id: &str, memory: &MemoryObservation) -> bool {
        if !memory
            .gb
            .is_some_and(|observed| observed > self.plan.limits.max_memory_gb)
        {
            return false;
        }
        self.halt(
            RunOutcome::StoppedMemoryLimit,
            "memory_limit",
            format!(
                "shot {shot_id} peaked at {:.1} GB ({}), over the plan's {} GB budget; raise \
                 limits.maxMemoryGb or pick a cheaper tier and start a new run",
                memory.gb.unwrap_or_default(),
                memory.source.as_deref().unwrap_or("no source"),
                self.plan.limits.max_memory_gb
            ),
            false,
        );
        true
    }

    /// The per-shot budget, and the graces derived from it: a cancel is never given more room to
    /// settle than the plan gave the whole attempt.
    pub(super) fn shot_budget(&self) -> Duration {
        Duration::from_secs(self.plan.limits.max_shot_seconds)
    }

    pub(super) fn bounds(&self, shot_deadline: Instant) -> PollBounds {
        let shot_budget = self.shot_budget();
        PollBounds {
            shot_deadline,
            run_deadline: self.run_deadline,
            poll_interval: self.poll_interval,
            cancel_grace: CANCEL_GRACE.min(shot_budget),
            settle_grace: ASSET_SETTLE_GRACE.min(shot_budget),
        }
    }

    pub(super) fn project_id(&self) -> Result<String, HarnessError> {
        self.record
            .project_id
            .clone()
            .ok_or_else(|| HarnessError::Transport("the run has no project yet".to_owned()))
    }

    // -----------------------------------------------------------------------------------------
    // Project and references — idempotent, so a replay adopts instead of re-creating
    // -----------------------------------------------------------------------------------------

    /// The project this run writes into. A run that created its own names it after the plan AND the
    /// run, which is what lets a controller that died right after `POST /projects` find that exact
    /// project again instead of creating a second one (sc-22711).
    pub(super) fn project_name(&self) -> String {
        format!("{} ({})", self.plan.title, self.record.run_id)
    }

    pub(super) async fn ensure_project(&mut self) -> Result<(), HarnessError> {
        if let Some(id) = self.record.project_id.clone() {
            let project = self
                .client
                .expect_ok("GET", &format!("/api/v1/projects/{id}"), None)
                .await?;
            self.record.project_path = project
                .get("path")
                .and_then(Value::as_str)
                .map(str::to_owned);
            self.persist()?;
            return Ok(());
        }
        let name = self.project_name();
        let existing = self
            .client
            .expect_ok("GET", "/api/v1/projects", None)
            .await?
            .as_array()
            .into_iter()
            .flatten()
            .find(|project| project.get("name").and_then(Value::as_str) == Some(name.as_str()))
            .cloned();
        let project = match existing {
            Some(project) => project,
            None => {
                self.client
                    .expect_ok("POST", "/api/v1/projects", Some(json!({ "name": name })))
                    .await?
            }
        };
        self.record.project_id = Some(
            project
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    HarnessError::Transport(format!("project response has no id: {project}"))
                })?
                .to_owned(),
        );
        self.record.project_path = project
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_owned);
        self.persist()?;
        Ok(())
    }

    /// Import every pack reference the record does not already name. An import the record missed
    /// (imported, then the controller died) is found by its `filmHarness` provenance rather than
    /// imported twice, so replay never doubles a project's reference assets.
    ///
    /// ONE ASSET PER DISTINCT FILE (sc-24024). Roles that name the same pack `file` are one image
    /// with several subjects in it, so they are uploaded once and every one of them resolves to
    /// that single asset id — which is what makes the dispatched `referenceAssetIds` carry one
    /// entry for a shared photograph, matching the one `<Picture N>` the compiler wrote for it.
    /// The record still carries one [`ReferenceAssetRecord`] per ROLE, all pointing at that asset:
    /// the record answers "what was this role conditioned on", and dropping the extra rows would
    /// lose the answer for every role but the first.
    ///
    /// "The same file" is the pack's `file` string, compared literally, exactly as
    /// `film_compile::shot_reference_pictures` compares it. Nothing here canonicalizes through the
    /// filesystem: two entries that mean one image must spell its path one way, or the compiler
    /// would number two pictures while the import created one asset.
    pub(super) async fn ensure_references(&mut self) -> Result<(), HarnessError> {
        let project_id = self.project_id()?;
        for existing in &self.record.references {
            // Only APPROVED roles resolve into a shot's conditioning slots, on a resume exactly as
            // on a first run.
            if existing.approved {
                self.role_assets
                    .insert(existing.role.clone(), existing.asset_id.clone());
            }
        }
        let imported: BTreeSet<String> = self
            .record
            .references
            .iter()
            .map(|reference| reference.role.clone())
            .collect();
        let pack_dir = self
            .pack_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        // Only roles with an IMAGE are imported (sc-24025). A described-only role has no bytes to
        // upload, no asset to tag and nothing to put in `role_assets` — it reaches the model as
        // words the compiler writes into the prompt, and the run record's `references` stays what
        // it has always been: the list of images this run imported.
        let references: Vec<film_plan::ReferenceEntry> = self
            .pack
            .references
            .iter()
            .filter(|reference| reference.file().is_some())
            .cloned()
            .collect();
        // One listing for the whole pass, not one per reference: anything imported later in this
        // loop is this controller's own and is already recorded.
        // `includeRejected` / `includeTrashed` default to FALSE on the route, and a human reviewing
        // the project between two controllers can reject or trash an imported plate. Adopting one
        // the listing hid is the point of this lookup, so ask for them.
        let already_imported = if references
            .iter()
            .any(|reference| !imported.contains(&reference.role))
        {
            self.client
                .expect_ok(
                    "GET",
                    &format!(
                        "/api/v1/projects/{project_id}/assets\
                         ?includeRejected=true&includeTrashed=true"
                    ),
                    None,
                )
                .await?
                .as_array()
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        // file -> the asset that file is already imported as, seeded from the record so a resume
        // that got halfway through a shared group reuses the asset its first role landed on
        // instead of uploading the same bytes a second time.
        let mut file_assets: BTreeMap<String, String> = self
            .record
            .references
            .iter()
            .map(|existing| (existing.file.clone(), existing.asset_id.clone()))
            .collect();
        // The roles each file carries, in pack order. Computed up front because the tag PATCH
        // REPLACES an asset's tag set: tagging a shared asset role by role would leave it wearing
        // only the last role's tag, and a query for "the asset the courier was conditioned on"
        // would miss it.
        let mut roles_by_file: BTreeMap<&str, Vec<&film_plan::ReferenceEntry>> = BTreeMap::new();
        for reference in &references {
            let Some(file) = reference.file() else {
                continue;
            };
            roles_by_file.entry(file).or_default().push(reference);
        }
        for reference in &references {
            if imported.contains(&reference.role) {
                continue;
            }
            // Total, though `references` was already filtered to roles that name one: the file is
            // read, hashed, uploaded and recorded below, and every one of those steps wants the
            // path itself rather than an `Option` unwrapped four times (sc-24025).
            let Some(file) = reference.file() else {
                continue;
            };
            let path = pack_dir.join(file);
            let bytes = std::fs::read(&path)?;
            let sha256 = sha256_hex(&bytes);
            let sharing = roles_by_file
                .get(file)
                .map(Vec::as_slice)
                .unwrap_or_default();
            let asset_id = match file_assets.get(file) {
                // Already uploaded in this pass (or by an earlier controller): the shared image is
                // sent once, so this role adopts that asset rather than creating a second copy of
                // the same picture.
                Some(asset_id) => asset_id.clone(),
                None => match find_imported_reference(
                    &already_imported,
                    &self.record.run_id,
                    // The FIRST role on this file is the one the upload stamped its provenance
                    // with, so that is the role a crash-window adoption looks for.
                    sharing
                        .first()
                        .map_or(reference.role.as_str(), |first| first.role.as_str()),
                    &sha256,
                ) {
                    Some(asset_id) => asset_id,
                    None => {
                        self.import_reference(&project_id, reference, sharing, &path, &sha256)
                            .await?
                    }
                },
            };
            file_assets.insert(file.to_owned(), asset_id.clone());
            // On BOTH branches: the upload and the tag PATCH are two writes, so a controller that
            // died between them leaves an asset the adoption finds but nothing has tagged. The
            // PATCH replaces the tag set, so re-applying it to an already-tagged asset is a no-op —
            // and it carries EVERY role sharing the file, for the same reason.
            self.tag_reference(&project_id, &asset_id, reference, sharing)
                .await?;
            if reference.approved {
                self.role_assets
                    .insert(reference.role.clone(), asset_id.clone());
            }
            self.record.references.push(ReferenceAssetRecord {
                role: reference.role.clone(),
                kind: reference.kind.clone(),
                file: file.to_owned(),
                sha256,
                asset_id,
                approved: reference.approved,
            });
            self.persist()?;
        }
        Ok(())
    }

    /// Import every pack sound clip this run will PLACE, and re-adopt the ones an earlier
    /// controller already imported (sc-22712, under sc-22711's resumable session).
    ///
    /// Only what the run PLACES is imported — the two beds plus the dialogue of the SELECTED shots.
    /// Unlike a reference image, importing an audio clip costs an ffmpeg transcode, and a pack
    /// legitimately carries sound for shots a `--shots` run left out. `record.sound` is therefore
    /// the list of clips that were actually available to the mix.
    ///
    /// Resume-safe exactly the way [`Session::ensure_references`] is: a clip the record already
    /// names is adopted rather than re-imported, and one imported by a controller that died before
    /// recording it is found by its `filmHarness` provenance instead of being imported twice. The
    /// durations the beds and lines are laid out from are measured by the import route off the
    /// STORED wav, so they are read back off the project's assets rather than re-derived here —
    /// which is also what lets a resume place the same spans the first controller did.
    pub(super) async fn ensure_sound(&mut self) -> Result<(), HarnessError> {
        let project_id = self.project_id()?;
        let placed: BTreeSet<String> = self
            .plan
            .sound
            .ambience
            .iter()
            .chain(self.plan.sound.music.iter())
            .chain(self.plan.sound.sfx.iter())
            .map(|bed| bed.role.clone())
            .chain(
                self.plan
                    .shots
                    .iter()
                    .filter(|shot| self.record.selected_shot_ids.contains(&shot.id))
                    .filter_map(|shot| shot.dialogue_clip.as_ref().map(|clip| clip.role.clone())),
            )
            .collect();
        let mut entries: Vec<film_plan::SoundEntry> = self
            .pack
            .sound
            .iter()
            .filter(|entry| placed.contains(&entry.role))
            .cloned()
            .collect();
        if entries.is_empty() {
            return Ok(());
        }
        // Speak every placed line that has no clip yet, BEFORE the listing below: synthesis creates
        // assets and writes files, and the import pass that follows must see a pack directory that
        // already holds them (sc-23404). Each entry comes back with the `file` synthesis wrote, so
        // from here down a spoken line and a recorded one are the same thing.
        //
        // The worker preflight comes first and is scoped to the lines still OWED: a run whose clips
        // are all already spoken needs no TTS worker at all (that is the `replace-take` case), and
        // a run that does need one must be told so here rather than enqueue a job nobody claims and
        // spend the whole per-job budget waiting for it. Same rule as the `video_generate` and
        // `image_vqa` preflights — LIVE rows only.
        let owed: Vec<&film_plan::SoundEntry> = entries
            .iter()
            .filter(|entry| entry.is_synthesized() && !self.line_already_spoken(entry))
            .collect();
        if !owed.is_empty() {
            let roles: Vec<&str> = owed.iter().map(|entry| entry.role.as_str()).collect();
            let workers = self
                .client
                .expect_ok("GET", "/api/v1/workers", None)
                .await?;
            let audio = live_worker_advertising(&workers, "audio_generate");
            if audio.live.is_none() {
                self.halt(
                    RunOutcome::Failed,
                    "no_audio_worker",
                    format!(
                        "the pack asks this run to speak {} ({}) but no live registered worker \
                         advertises audio_generate{}; start a worker with the audio lane and \
                         `film-harness resume` speaks them",
                        roles.len(),
                        roles.join(", "),
                        stale_workers_detail(&audio.stale)
                    ),
                    true,
                );
                return Ok(());
            }
        }
        for entry in &mut entries {
            if !entry.is_synthesized() {
                continue;
            }
            let Some(file) = self.synthesize_dialogue(&project_id, entry).await? else {
                // Halted: the stop is recorded, the clips already spoken and imported stay where
                // they are, and `drive_inner` stops before dispatching a render.
                return Ok(());
            };
            entry.file = Some(file);
        }
        // ONE listing for the whole pass. It does double duty: it carries the stored duration of
        // every clip an earlier controller already imported, and the provenance that finds one it
        // imported but died before recording.
        let assets = self
            .client
            .expect_ok(
                "GET",
                &format!(
                    "/api/v1/projects/{project_id}/assets\
                     ?includeRejected=true&includeTrashed=true"
                ),
                None,
            )
            .await?
            .as_array()
            .cloned()
            .unwrap_or_default();
        let stored_duration = |asset_id: &str| -> Option<f64> {
            assets
                .iter()
                .find(|asset| asset.get("id").and_then(Value::as_str) == Some(asset_id))
                .and_then(|asset| asset.pointer("/file/duration"))
                .and_then(Value::as_f64)
                .filter(|seconds| *seconds > 0.0)
        };
        // Keyed on (role, FILE), not on the role alone. `record.sound` is append-only and a role's
        // clip is not immutable — a pack can re-cast a `dialogue` line or swap the recording a role
        // points at — and a map keyed on the role alone would hand the bus back the asset made from
        // the OLD file. The file is what the asset was made from, so it is what the adoption has to
        // agree on.
        //
        // A GUARD rather than a live path: the pack-sha check in `open_session` means a changed
        // pack is a new run, so today nothing can reach `ensure_sound` with a `record.sound` entry
        // the entries disagree with. It costs one tuple and it means the adoption is correct on its
        // own terms instead of correct only because something upstream refuses.
        let recorded: BTreeMap<(String, String), String> = self
            .record
            .sound
            .iter()
            .map(|clip| {
                (
                    (clip.role.clone(), clip.file.clone()),
                    clip.asset_id.clone(),
                )
            })
            .collect();
        let pack_dir = self
            .pack_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        for entry in &entries {
            // Set on every entry by now: a recorded clip declares it, and a synthesized one was
            // given it by `synthesize_dialogue` above. An entry with neither is a validation
            // finding the run never gets past.
            let file = entry.file.clone().ok_or_else(|| {
                HarnessError::Transport(format!(
                    "sound {:?} has no file to import; the pack declares neither `file` nor `text`",
                    entry.role
                ))
            })?;
            if let Some(asset_id) = recorded.get(&(entry.role.clone(), file.clone())) {
                self.sound_assets.insert(
                    entry.role.clone(),
                    SoundAsset {
                        asset_id: asset_id.clone(),
                        duration_seconds: stored_duration(asset_id),
                    },
                );
                continue;
            }
            let path = pack_dir.join(&file);
            let bytes = std::fs::read(&path)?;
            let sha256 = sha256_hex(&bytes);
            // A clip imported by THIS pass is not in the listing above (it was fetched before the
            // upload), so its duration comes back with the import response. Only an ADOPTED clip —
            // one an earlier controller uploaded — is measured off the listing.
            let (asset_id, duration_seconds) =
                match find_imported_reference(&assets, &self.record.run_id, &entry.role, &sha256) {
                    Some(asset_id) => {
                        let duration = stored_duration(&asset_id);
                        (asset_id, duration)
                    }
                    None => {
                        self.import_sound(&project_id, entry, &file, &path, &sha256)
                            .await?
                    }
                };
            // As with a reference: the upload and the tag PATCH are two writes, so re-applying the
            // tags to an adopted asset is the no-op that closes the window between them.
            self.client
                .expect_ok(
                    "PATCH",
                    &format!("/api/v1/projects/{project_id}/assets/{asset_id}/tags"),
                    Some(json!({
                        "tags": [
                            SOUND_TAG,
                            harness_asset_tag("role", &entry.role),
                            harness_asset_tag("pack", &self.pack.id)
                        ]
                    })),
                )
                .await?;
            self.sound_assets.insert(
                entry.role.clone(),
                SoundAsset {
                    asset_id: asset_id.clone(),
                    duration_seconds,
                },
            );
            self.record.sound.push(ReferenceAssetRecord {
                role: entry.role.clone(),
                kind: entry.kind.clone(),
                file: file.clone(),
                sha256,
                asset_id,
                // A sound entry has no approval flag of its own: approval gates CONDITIONING, and
                // sound is never conditioning. Everything in the pack's `sound` array is placeable.
                approved: true,
            });
            self.persist()?;
        }
        Ok(())
    }

    /// Speak one `dialogue` entry's line through `POST /api/v1/audio/jobs` and leave the WAV in the
    /// pack directory, so the import pass that follows treats it exactly as a pre-recorded clip
    /// (sc-23404).
    ///
    /// Returns the pack-relative path the clip is at, or `None` when the run has HALTED — the job
    /// failed, or it ran past a declared limit. A halt leaves everything already spoken and imported
    /// in the record and stops `drive_inner` before the first render, so a resume picks the sequence
    /// up where it is rather than re-speaking what is already there.
    ///
    /// Resume discipline is the renders': the idempotency key is stamped into the dispatched body's
    /// `advanced.filmHarness` block and a controller that died between the POST and the record write
    /// finds its OWN job by that key. What the key covers is model + voice + text + attempt, not
    /// just the role — so a retry after a failure is a new job rather than a re-read of the same
    /// failure.
    ///
    /// RE-CASTING a line is not something this run does: `resume` and `replace-take` refuse a pack
    /// whose bytes no longer hash to what the run started from ("a changed reference pack is a new
    /// run, not a resume"), so an edited line reaches this code only through a fresh `run`. The
    /// content half of the key still earns its place inside ONE run: a line whose text or voice does
    /// not match the record's is a new attempt rather than an adoption, which is what a record
    /// carried over from an aborted earlier shape of the pack needs. When that happens the stale
    /// `record.sound` entry is dropped with it — `record.sound` is append-only, and an entry naming
    /// the old asset would otherwise be re-adopted by `ensure_sound` even where the pack pins
    /// `file` and the clip's name never changed.
    ///
    /// The clip's filename is `<role>.<sha256(text)[..12]>.wav` unless the entry pins one with
    /// `file`, in which case synthesis writes THERE — that is how a pack keeps a stable name for a
    /// line it means to check in.
    pub(super) async fn synthesize_dialogue(
        &mut self,
        project_id: &str,
        entry: &film_plan::SoundEntry,
    ) -> Result<Option<String>, HarnessError> {
        let role = entry.role.clone();
        let text = entry
            .synthesis_text()
            .ok_or_else(|| {
                HarnessError::Transport(format!("sound {role:?} declares an empty `text`"))
            })?
            .to_owned();
        let model = entry.synthesis_model().to_owned();
        let voice = entry.synthesis_voice().map(str::to_owned);
        let text_sha256 = sha256_hex(text.as_bytes());
        let file = entry
            .file
            .clone()
            .unwrap_or_else(|| synthesized_sound_file(&role, &text_sha256));
        let pack_dir = self
            .pack_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let destination = pack_dir.join(&file);

        // 1. Adopt, continue, or start a new attempt. Exactly one record per role — the record says
        //    what the film HAS, and two entries for one role would leave a reader guessing.
        //
        //    - The line this run already spoke, whose WAV is still where it was written: adopt it.
        //      No second job, however many times the run is resumed.
        //    - A record for the same line whose job has NOT ended: continue it under its own key,
        //      which is the "died between the POST and the record write" window.
        //    - A record whose job ENDED without a clip, or one made for a line the pack has since
        //      re-cast: a NEW attempt under a NEW key. Re-polling the finished job under the old
        //      key would make every resume re-read the same failure and never speak the line.
        let existing = self
            .record
            .synthesized_sound
            .iter()
            .position(|line| line.role == role);
        let continues = existing.is_some_and(|index| {
            let line = &self.record.synthesized_sound[index];
            line.matches(&model, voice.as_deref(), &text)
        });
        if continues {
            let index = existing.expect("continues implies a record");
            // Already IMPORTED: the clip is a project asset the dialogue bus is playing, so whether
            // the WAV is still in the pack directory no longer matters — a cleaned pack, or a
            // replacement running months later, must adopt rather than speak the line again.
            let imported = self
                .record
                .sound
                .iter()
                .any(|clip| clip.role == role)
                .then(|| self.record.synthesized_sound[index].file.clone())
                .flatten();
            if self.record.synthesized_sound[index].is_usable() {
                if let Some(file) = imported {
                    return Ok(Some(file));
                }
                if destination.is_file() {
                    return Ok(Some(file));
                }
            }
        }
        let index = match existing {
            Some(index) if continues && !self.record.synthesized_sound[index].is_terminal() => {
                index
            }
            other => {
                if self.canceled() {
                    self.halt(
                        RunOutcome::Canceled,
                        "canceled",
                        format!("canceled before the dialogue line for {role:?} was synthesized"),
                        true,
                    );
                    return Ok(None);
                }
                let attempt = other
                    .map(|index| self.record.synthesized_sound[index].attempt + 1)
                    .unwrap_or(1);
                let fresh = film_plan::SynthesizedSoundRecord {
                    role: role.clone(),
                    text: text.clone(),
                    text_sha256: text_sha256.clone(),
                    model: model.clone(),
                    voice: voice.clone(),
                    attempt,
                    idempotency_key: dialogue_idempotency_key(
                        &self.record.run_id,
                        &role,
                        &model,
                        voice.as_deref(),
                        &text,
                        attempt,
                    ),
                    job_id: None,
                    status: "dispatching".to_owned(),
                    asset_id: None,
                    file: None,
                    error: None,
                    started_at: utc_now(),
                    finished_at: None,
                };
                let index = match other {
                    Some(index) => {
                        self.record.synthesized_sound[index] = fresh;
                        index
                    }
                    None => {
                        self.record.synthesized_sound.push(fresh);
                        self.record.synthesized_sound.len() - 1
                    }
                };
                // A fresh attempt means the clip this film will play is NEW — a line this record
                // does not match, or a retry after one that failed. `record.sound` is append-only,
                // so an entry left from an earlier attempt still names the OLD asset and the import
                // pass in `ensure_sound` would adopt it: the bus would go on saying the old thing
                // in the old voice. Dropping it here is the half the (role, file) key cannot do,
                // because a pack that PINS `file` re-casts a line without the clip's NAME ever
                // changing — what changed is the `text_sha256` this record was just rewritten with.
                //
                // A GUARD, like that key: `open_session` refuses a pack whose bytes moved, so a
                // re-cast is a new run and nothing today reaches this line with a stale clip. It is
                // two statements, and it means the record cannot describe a film that says one
                // thing and plays another.
                self.record.sound.retain(|clip| clip.role != role);
                self.sound_assets.remove(&role);
                // Persisted BEFORE the job exists, exactly as an attempt is: that is what makes the
                // key findable by the controller that comes back.
                self.persist()?;
                index
            }
        };
        let key = self.record.synthesized_sound[index].idempotency_key.clone();
        // 2. The job. Adopt one already created under this key before creating anything.
        let mut job_id = self.record.synthesized_sound[index].job_id.clone();
        if job_id.is_none() {
            job_id = self
                .client
                .find_job_by_idempotency_key(project_id, &key)
                .await?;
        }
        let created_here = job_id.is_none();
        if job_id.is_none() {
            if self.canceled() {
                let detail =
                    format!("canceled before the dialogue line for {role:?} was synthesized");
                self.record.synthesized_sound[index].error = Some(detail.clone());
                self.halt(RunOutcome::Canceled, "canceled", detail, true);
                self.persist()?;
                return Ok(None);
            }
            let mut body = json!({
                "projectId": project_id,
                "prompt": text,
                "model": model,
                "requestedGpu": "auto",
                "advanced": {
                    "filmHarness": {
                        "idempotencyKey": key,
                        "kind": "dialogue",
                        "role": role,
                        "runId": self.record.run_id,
                        "planId": self.plan.id,
                        "planVersion": self.plan.version,
                        "referencePackId": self.pack.id,
                        "referencePackVersion": self.pack.version,
                        "textSha256": text_sha256,
                    }
                }
            });
            if let Some(voice) = &voice {
                body["voice"] = json!(voice);
            }
            let response = self
                .client
                .json("POST", "/api/v1/audio/jobs", Some(body))
                .await?;
            if !(200..300).contains(&response.status) {
                // A refused enqueue is deterministic — an unknown voice, a model with no weights —
                // so retrying it would refuse identically. Say which line, and stop.
                let detail = format!(
                    "POST /api/v1/audio/jobs -> {}: {}",
                    response.status,
                    api_detail(&response.body)
                );
                self.fail_synthesis(index, "rejected", detail.clone());
                self.persist()?;
                self.halt(
                    RunOutcome::Failed,
                    "dialogue_synthesis_refused",
                    format!(
                        "the dialogue line for sound role {role:?} could not be enqueued: \
                         {detail}; fix the pack entry and `film-harness resume` speaks it"
                    ),
                    true,
                );
                return Ok(None);
            }
            job_id = Some(
                response
                    .body
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        HarnessError::Transport(format!(
                            "audio job response has no id: {}",
                            response.body
                        ))
                    })?
                    .to_owned(),
            );
        }
        let job_id = job_id.expect("set on every branch above");
        {
            let line = &mut self.record.synthesized_sound[index];
            if created_here {
                // Nothing has ever run under this key, so the clock starts here — the same reason a
                // render's does (a record written by a controller that died before its POST landed
                // must not be charged every hour since).
                line.started_at = utc_now();
            }
            line.job_id = Some(job_id.clone());
            line.status = "running".to_owned();
        }
        self.persist()?;

        // 3. Wait for it, under the plan's own declared limits: one synthesis is bounded by
        //    `limits.maxShotSeconds` (the same per-job budget the export runs under) and the run by
        //    `limits.maxRunSeconds`. Neither is a new knob — a speech job is a job.
        let started = Instant::now();
        let (view, poll_stop) = self
            .client
            .wait_for_job(&job_id, self.bounds(started + self.shot_budget()))
            .await?;
        let status = match poll_stop {
            PollStop::Terminal | PollStop::AssetsUnsettled => view.status.clone(),
            PollStop::Operator => "canceled_by_operator".to_owned(),
            PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
        };
        let asset = (status == "completed")
            .then(|| audio_asset_from_result(&view.result))
            .flatten();
        let Some((asset_id, media_path)) = asset else {
            let detail = match poll_stop {
                PollStop::Terminal => view.failure_text(),
                PollStop::AssetsUnsettled => format!(
                    "the synthesis job reached {} but its asset never settled",
                    view.status
                ),
                PollStop::Operator => "canceled by operator during synthesis".to_owned(),
                PollStop::ShotBudget => format!(
                    "synthesis exceeded the per-job budget of {}s",
                    self.plan.limits.max_shot_seconds
                ),
                PollStop::RunBudget => format!(
                    "the run's {}s budget ran out during synthesis",
                    self.plan.limits.max_run_seconds
                ),
            };
            self.fail_synthesis(index, &status, detail.clone());
            self.persist()?;
            let (outcome, reason) = match poll_stop {
                PollStop::Operator => (RunOutcome::Canceled, "canceled"),
                PollStop::RunBudget => (RunOutcome::StoppedRunBudget, "run_budget"),
                _ => (RunOutcome::Failed, "dialogue_synthesis_failed"),
            };
            // Resumable on every one of these: the clip is missing, nothing downstream has been
            // written against it, and a resume re-dispatches this one line and continues. The rest
            // of the sequence — the clips already spoken, the imports already made — is untouched.
            self.halt(
                outcome,
                reason,
                format!(
                    "the dialogue line for sound role {role:?} was not synthesized ({detail}); \
                     `film-harness resume` speaks it and continues the sequence"
                ),
                true,
            );
            return Ok(None);
        };

        // 4. The WAV, into the pack directory. The worker writes canonical PCM-16 RIFF/WAVE, which
        //    is the one encoding `media_convert::is_canonical_pcm16_wav` copies through — so the
        //    import below needs no ffmpeg for a clip this run spoke, which is what lets the hosted
        //    macOS lane (no ffmpeg) exercise the whole path.
        //
        //    The bytes come over the TRANSPORT, like every other media hop the harness makes.
        //    `--api` may legitimately name a private-network address, a `.local` name or a bare
        //    hostname (docs/film-harness.md), and the API host "may be a different machine" — on
        //    any such host reaching into its project directory is an io error, not a clip. The
        //    local read survives only as a FAST PATH for the common loopback case, where the file
        //    really is right there and copying it through HTTP buys nothing.
        let source = self
            .record
            .project_path
            .as_deref()
            .map(|path| Path::new(path).join(&media_path))
            .filter(|source| source.is_file());
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = match &source {
            Some(source) => std::fs::read(source).map_err(|error| {
                HarnessError::Io(format!(
                    "the synthesis job for {role:?} reported {} but it could not be read: {error}",
                    source.display()
                ))
            })?,
            None => download_media(self.client.transport, project_id, &media_path).await?,
        };
        std::fs::write(&destination, &bytes)?;
        {
            let line = &mut self.record.synthesized_sound[index];
            line.status = "completed".to_owned();
            line.asset_id = Some(asset_id);
            line.file = Some(file.clone());
            line.error = None;
            line.finished_at = Some(utc_now());
        }
        self.persist()?;
        Ok(Some(file))
    }

    /// Whether this run has already spoken exactly this entry's line and still has the clip — the
    /// same adoption test [`Session::synthesize_dialogue`] applies, read-only, so the TTS worker
    /// preflight can be scoped to the lines that are actually still owed.
    pub(super) fn line_already_spoken(&self, entry: &film_plan::SoundEntry) -> bool {
        let Some(text) = entry.synthesis_text() else {
            return false;
        };
        let Some(line) = self
            .record
            .synthesized_sound
            .iter()
            .find(|line| line.role == entry.role)
        else {
            return false;
        };
        if !line.matches(entry.synthesis_model(), entry.synthesis_voice(), text)
            || !line.is_usable()
        {
            return false;
        }
        if self.record.sound.iter().any(|clip| clip.role == entry.role) {
            return true;
        }
        let pack_dir = self
            .pack_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        line.file
            .as_deref()
            .is_some_and(|file| pack_dir.join(file).is_file())
    }

    /// Settle one synthesis record as unfinished, keeping whatever it already knew.
    pub(super) fn fail_synthesis(&mut self, index: usize, status: &str, detail: String) {
        let line = &mut self.record.synthesized_sound[index];
        line.status = status.to_owned();
        line.error = Some(detail);
        line.finished_at = Some(utc_now());
    }

    /// Upload one pack sound clip, with the same `filmHarness` provenance a reference carries so a
    /// controller that dies before recording it can find it again.
    ///
    /// Returns the asset id and the duration the import route MEASURED off the stored PCM-16 wav —
    /// not anything the plan claimed — which is what the beds and lines are then laid out from.
    pub(super) async fn import_sound(
        &self,
        project_id: &str,
        entry: &film_plan::SoundEntry,
        file: &str,
        path: &Path,
        sha256: &str,
    ) -> Result<(String, Option<f64>), HarnessError> {
        let bytes = std::fs::read(path)?;
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("sound.wav")
            .to_owned();
        let provenance = json!({
            "filmHarness": {
                "kind": "sound",
                "role": entry.role,
                "soundKind": entry.kind,
                "referencePackId": self.pack.id,
                "referencePackVersion": self.pack.version,
                "planId": self.plan.id,
                "planVersion": self.plan.version,
                "runId": self.record.run_id,
                "sourceFile": file,
                // Whether the clip was SPOKEN by this run rather than put on disk by a human
                // (sc-23404), so a reader of the project's assets can tell them apart without the
                // run record in hand.
                "synthesized": entry.is_synthesized(),
                "sha256": sha256,
            }
        });
        let (boundary, body) =
            encode_asset_upload(&filename, audio_content_type(path), &bytes, &provenance);
        let route = format!("/api/v1/projects/{project_id}/assets");
        let response = self
            .transport
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
        let asset_id = response
            .body
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                HarnessError::Transport(format!(
                    "sound import response has no id: {}",
                    response.body
                ))
            })?;
        let duration_seconds = response
            .body
            .pointer("/file/duration")
            .and_then(Value::as_f64)
            .filter(|seconds| *seconds > 0.0);
        Ok((asset_id, duration_seconds))
    }

    /// Upload one reference IMAGE. `sharing` is every pack role that names this file, in pack
    /// order, which for the ordinary case is just `reference` itself (sc-24024).
    ///
    /// The top-level `role` / `referenceKind` / `approved` stay the FIRST sharing role's, because
    /// that triple is what [`find_imported_reference`] matches a crash-window adoption on. The
    /// per-role truth — which subject each role is, and its locator — goes in `roles`, so the asset
    /// itself says the photograph holds two people rather than only naming one of them.
    pub(super) async fn import_reference(
        &self,
        project_id: &str,
        reference: &film_plan::ReferenceEntry,
        sharing: &[&film_plan::ReferenceEntry],
        path: &Path,
        sha256: &str,
    ) -> Result<String, HarnessError> {
        let bytes = std::fs::read(path)?;
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("reference.png")
            .to_owned();
        let content_type = match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref()
        {
            Some("jpg" | "jpeg") => "image/jpeg",
            Some("webp") => "image/webp",
            _ => "image/png",
        };
        let head = sharing.first().copied().unwrap_or(reference);
        let roles: Vec<Value> = sharing
            .iter()
            .map(|entry| {
                json!({
                    "role": entry.role,
                    "referenceKind": entry.kind,
                    "locator": entry.locator(),
                })
            })
            .collect();
        let mut provenance = json!({
            "filmHarness": {
                "kind": "reference",
                "role": head.role,
                "referenceKind": head.kind,
                "approved": head.approved,
                // Every role this ONE image is bound to (sc-24024). A single-role image lists one.
                "roles": roles,
                // sc-23403: whether this plate was GENERATED as a fixture or supplied by a person,
                // carried onto the asset so the answer survives the pack document. It changes
                // nothing else — a generated reference is imported, tagged and conditioned on
                // exactly like any other, and `approved` remains the only gate.
                "generated": reference.generated,
                "referencePackId": self.pack.id,
                "referencePackVersion": self.pack.version,
                "planId": self.plan.id,
                "planVersion": self.plan.version,
                "runId": self.record.run_id,
                "sourceFile": reference.file,
                "sha256": sha256,
            }
        });
        if let Some(generation) = reference.generation.as_ref() {
            // A provenance block that cannot be serialized is a refusal, not a `null`: the whole
            // point of `generated` is that the answer survives the pack document, and an asset
            // stamped `"generation": null` would say the plate came from nowhere.
            let block = serde_json::to_value(generation).map_err(|error| {
                HarnessError::Io(format!(
                    "cannot serialize the generation provenance for reference {:?}: {error}",
                    reference.role
                ))
            })?;
            provenance
                .get_mut("filmHarness")
                .and_then(Value::as_object_mut)
                .expect("the provenance literal has a filmHarness object")
                .insert("generation".to_owned(), block);
        }
        let (boundary, body) = encode_asset_upload(&filename, content_type, &bytes, &provenance);
        let route = format!("/api/v1/projects/{project_id}/assets");
        let response = self
            .transport
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
        let asset_id = response
            .body
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                HarnessError::Transport(format!(
                    "asset import response has no id: {}",
                    response.body
                ))
            })?
            .to_owned();
        Ok(asset_id)
    }

    /// Tag an imported reference with its kind, role and pack.
    ///
    /// An unapproved reference is imported (so the record can point at it and a human can review
    /// it) but tagged distinctly, so a query for the conditioning-eligible references cannot pick
    /// it up. Applied by [`Session::ensure_references`] to a freshly imported asset AND to an
    /// adopted one, because the tags are a second write the crash window can swallow.
    /// `sharing` is every pack role that names this asset's file, and it always CONTAINS
    /// `reference` — `ensure_references` builds it by looking `reference` up in a map keyed on
    /// every reference's own file. A shared image wears a `role:` tag for EACH of them
    /// (sc-24024): the PATCH replaces the whole tag set, so tagging a shared asset one role at a
    /// time would leave it wearing only the last, and a query for the asset the courier was
    /// conditioned on would come back empty.
    pub(super) async fn tag_reference(
        &self,
        project_id: &str,
        asset_id: &str,
        reference: &film_plan::ReferenceEntry,
        sharing: &[&film_plan::ReferenceEntry],
    ) -> Result<(), HarnessError> {
        // Roles sharing a file are validated to AGREE on `approved`, so any of them answers this.
        let kind_tag = if reference.approved {
            REFERENCE_TAG
        } else {
            UNAPPROVED_REFERENCE_TAG
        };
        debug_assert!(
            sharing.iter().any(|entry| entry.role == reference.role),
            "the group a reference is tagged under always contains that reference"
        );
        let mut tags = vec![json!(kind_tag)];
        for entry in sharing {
            tags.push(json!(harness_asset_tag("role", &entry.role)));
        }
        tags.push(json!(harness_asset_tag("pack", &self.pack.id)));
        self.client
            .expect_ok(
                "PATCH",
                &format!("/api/v1/projects/{project_id}/assets/{asset_id}/tags"),
                Some(json!({ "tags": tags })),
            )
            .await?;
        Ok(())
    }

    // -----------------------------------------------------------------------------------------
    // Shots
    // -----------------------------------------------------------------------------------------

    /// Make sure the record holds one entry per plan shot, in plan order, without disturbing the
    /// entries a previous controller wrote.
    ///
    /// Refuses the run when a shot's conditioning role was never imported as an asset (sc-24023):
    /// see [`Session::resolve_new_shot_conditioning`].
    pub(super) fn ensure_shot_records(&mut self) -> Result<(), HarnessError> {
        let mut resolved = self.resolve_new_shot_conditioning()?;
        let mut ordered: Vec<ShotRunRecord> = Vec::with_capacity(self.plan.shots.len());
        for shot in &self.plan.shots {
            if let Some(index) = self
                .record
                .shots
                .iter()
                .position(|existing| existing.shot_id == shot.id)
            {
                ordered.push(self.record.shots.remove(index));
                continue;
            }
            // Geometry, timing and mode come from the COMPILED request, not from the plan read a
            // second time: the compiled document is what the job body is built from (sc-22713), so
            // the record's `intended` has to be the same document or it describes something the
            // route was never asked for.
            let request = self
                .compiled
                .request(&shot.id)
                .expect("every plan shot compiled a request");
            let (width, height) = (request.width, request.height);
            // Resolved by `CompiledRequest::resolve_conditioning` above — the SAME function that
            // walks `shot_reference_pictures`, which the compiler numbered this request's
            // `<Picture N>` with (sc-24023) — so the position an asset takes here is the number the
            // prompt already promised for it. The record is what `work_attempt` dispatches from, so
            // this is the list the engine sees.
            let conditioning = resolved
                .remove(&shot.id)
                .expect("every shot without a record resolved its conditioning above");
            let assets = ConditioningAssets {
                first_frame_asset_id: conditioning.first_frame_asset_id,
                last_frame_asset_id: conditioning.last_frame_asset_id,
                reference_asset_ids: conditioning.reference_asset_ids,
            };
            ordered.push(ShotRunRecord {
                shot_id: shot.id.clone(),
                outcome: ShotOutcome::NotSelected,
                intended: IntendedState {
                    mode: request.mode.clone(),
                    start_state: shot.start_state.clone(),
                    end_state: shot.end_state.clone(),
                    target_duration_seconds: request.duration_seconds,
                    width,
                    height,
                    fps: request.fps,
                    dialogue: shot.dialogue.clone(),
                    // The record keeps its own v2 field name while the plan's moved to `audio`
                    // (sc-24026): renaming it would refuse every in-flight run's resume for a
                    // spelling, and the value it records is the same sentence it always was.
                    sound: Some(shot.audio.clone()),
                    // The policy the export will obey for this shot — its own override, or the
                    // run-level default it inherits (sc-22712). Resolved here, at the same moment
                    // the rest of the intended state is, so the record says what was intended
                    // rather than what a later reader would re-derive from the plan.
                    generated_audio: resolved_generated_audio(&self.plan, shot),
                },
                conditioning_assets: assets,
                attempts: Vec::new(),
                selected_attempt: None,
                needs_review: Vec::new(),
                reviews: Vec::new(),
                human_decision: None,
            });
        }
        self.record.shots = ordered;
        Ok(())
    }

    /// The conditioning assets for every plan shot that does not have a record yet, resolved
    /// through [`CompiledRequest::resolve_conditioning`] — the ONE resolver, shared
    /// with the job-body builder, so the dispatched `referenceAssetIds` and the `<Picture N>` the
    /// compiler wrote into the prompt come out of one walk of `shot_reference_pictures` (sc-24023).
    ///
    /// A role with no imported asset is a finding that refuses the run. Dropping it instead would
    /// shift every LATER asset one position off the picture number the prompt already promised, and
    /// nothing downstream — not the record, not the route, not a reviewer reading `compiled.json` —
    /// can see that the model was bound to the wrong image. `validate_plan_against_pack` makes it
    /// unreachable today; this is the site that enforces it rather than assuming it.
    ///
    /// Runs as a whole pass BEFORE `ensure_shot_records` reorders anything, so a refusal cannot
    /// leave the record half-drained.
    pub(super) fn resolve_new_shot_conditioning(
        &self,
    ) -> Result<BTreeMap<String, ResolvedConditioning>, HarnessError> {
        let mut resolved = BTreeMap::new();
        let mut findings: Vec<PlanDiagnostic> = Vec::new();
        for shot in &self.plan.shots {
            if self
                .record
                .shots
                .iter()
                .any(|existing| existing.shot_id == shot.id)
            {
                continue;
            }
            let request = self
                .compiled
                .request(&shot.id)
                .expect("every plan shot compiled a request");
            match request.resolve_conditioning(&self.pack, &self.role_assets) {
                Ok(assets) => {
                    resolved.insert(shot.id.clone(), assets);
                }
                Err(mut shot_findings) => findings.append(&mut shot_findings),
            }
        }
        if findings.is_empty() {
            Ok(resolved)
        } else {
            Err(HarnessError::Validation(findings))
        }
    }

    pub(super) fn is_selected(&self, shot_id: &str) -> bool {
        self.record.selected_shot_ids.iter().any(|id| id == shot_id)
    }

    /// Reconcile and, where the limits still allow it, dispatch every selected shot.
    pub(super) async fn work_shots(&mut self) -> Result<(), HarnessError> {
        self.ensure_shot_records()?;
        self.persist()?;
        for index in 0..self.plan.shots.len() {
            let shot = self.plan.shots[index].clone();
            if !self.is_selected(&shot.id) {
                if let Some(record) = self.record.shot_mut(&shot.id) {
                    record.outcome = ShotOutcome::NotSelected;
                }
                continue;
            }
            self.work_shot(&shot).await?;
            self.persist()?;
            // Delivery precedes the next dispatch and never starts an export.
            self.assemble_timeline().await?;
        }
        Ok(())
    }

    /// One shot: adopt whatever the API already holds for it, then dispatch what is still owed and
    /// still affordable. Never creates a job for an attempt that already has one.
    pub(super) async fn work_shot(&mut self, shot: &film_plan::Shot) -> Result<(), HarnessError> {
        self.reconcile_shot(shot).await?;
        loop {
            let Some(index) = self.record.shots.iter().position(|s| s.shot_id == shot.id) else {
                return Ok(());
            };
            if self.record.shots[index].selected_attempt.is_some() {
                self.record.shots[index].outcome = ShotOutcome::Rendered;
                return Ok(());
            }
            if self.stop.is_some() {
                break;
            }
            // An attempt a previous controller recorded but never finished is continued, not
            // replaced: it already holds (or is about to hold) a job.
            let pending = self.record.shots[index]
                .attempts
                .iter()
                .position(|attempt| !TERMINAL_ATTEMPT_STATUSES.contains(&attempt.status.as_str()));
            let attempt_index = match pending {
                Some(pending) => pending,
                None => {
                    if self.record.shots[index].automatic_attempts()
                        >= self.plan.limits.max_attempts_per_shot
                    {
                        break;
                    }
                    if self.canceled() {
                        self.halt(
                            RunOutcome::Canceled,
                            "canceled",
                            format!("canceled before shot {} was dispatched", shot.id),
                            true,
                        );
                        break;
                    }
                    if self
                        .run_deadline
                        .is_some_and(|deadline| Instant::now() >= deadline)
                    {
                        self.halt(
                            RunOutcome::StoppedRunBudget,
                            "run_budget",
                            format!(
                                "the run's {}s wall-clock budget was spent before shot {} could be \
                                 dispatched; raise limits.maxRunSeconds and start a new run",
                                self.plan.limits.max_run_seconds, shot.id
                            ),
                            false,
                        );
                        break;
                    }
                    let number = self.record.shots[index].next_attempt_number();
                    let key = idempotency_key(&self.record.run_id, &shot.id, number);
                    let (resolved_model_id, partition_reason) = self.resolved_partition(&shot.id);
                    let reference_image_short_edge = self.resolved_reference_short_edge(&shot.id);
                    let reference_image_resize_mode = self.resolved_reference_resize_mode(&shot.id);
                    let (loras, effective_steps, turbo_scheduler_shift) =
                        self.resolved_sampling(&shot.id);
                    self.record.shots[index].attempts.push(AttemptRecord {
                        attempt: number,
                        idempotency_key: key,
                        resolved_model_id,
                        partition_reason,
                        reference_image_short_edge,
                        reference_image_resize_mode,
                        loras,
                        effective_steps,
                        turbo_scheduler_shift,
                        job_id: None,
                        status: "dispatching".to_owned(),
                        started_at: utc_now(),
                        finished_at: None,
                        elapsed_seconds: 0.0,
                        peak_gpu_memory_pct: None,
                        peak_memory_gb: None,
                        peak_memory_source: None,
                        error: None,
                        take: None,
                        rejection: None,
                        human_requested: false,
                    });
                    // Persisted BEFORE the job exists: this is the record a replay reads to know a
                    // job may already be out there under this key.
                    self.persist()?;
                    self.record.shots[index].attempts.len() - 1
                }
            };
            let progressed = self
                .work_attempt(shot, index, attempt_index, pending.is_none())
                .await?;
            if !progressed {
                break;
            }
        }
        let Some(index) = self.record.shots.iter().position(|s| s.shot_id == shot.id) else {
            return Ok(());
        };
        let record = &mut self.record.shots[index];
        record.outcome = match record
            .attempts
            .last()
            .map(|attempt| attempt.status.as_str())
        {
            _ if record.selected_attempt.is_some() => ShotOutcome::Rendered,
            None => ShotOutcome::NotDispatched,
            Some("timed_out") => ShotOutcome::TimedOut,
            Some("canceled" | "canceled_by_operator") => ShotOutcome::Canceled,
            Some(_) => ShotOutcome::Failed,
        };
        // A shot that spent its attempts does NOT stop the run: the remaining shots are independent
        // work and the human decides what to do about this one. Only a limit, a cancel or a refused
        // enqueue stops dispatch; `finish` turns a shot with no take into the run's outcome.
        Ok(())
    }

    /// Bring one recorded attempt to a terminal state. Returns whether the shot may keep going
    /// (another attempt is worth trying); `false` means a limit, a cancel or a refusal ended it.
    pub(super) async fn work_attempt(
        &mut self,
        shot: &film_plan::Shot,
        shot_index: usize,
        attempt_index: usize,
        fresh: bool,
    ) -> Result<bool, HarnessError> {
        let project_id = self.project_id()?;
        let key = self.record.shots[shot_index].attempts[attempt_index]
            .idempotency_key
            .clone();
        let attempt_number = self.record.shots[shot_index].attempts[attempt_index].attempt;

        // 1. The job. Adopt one already created under this key before creating anything.
        let mut job_id = self.record.shots[shot_index].attempts[attempt_index]
            .job_id
            .clone();
        if job_id.is_none() {
            job_id = self
                .client
                .find_job_by_idempotency_key(&project_id, &key)
                .await?;
        }
        // Neither the record nor the API has a job for this attempt, so this pass creates it: the
        // attempt has been in flight for zero seconds however old its record is.
        let created_here = job_id.is_none();
        if job_id.is_none() {
            if self.canceled() {
                // Keep this undispatched intention pending under its original key. Resume must
                // not spend another attempt just because cancellation won the enqueue race.
                let detail = format!("canceled before shot {} was dispatched", shot.id);
                self.record.shots[shot_index].attempts[attempt_index].error = Some(detail.clone());
                self.record.shots[shot_index].outcome = ShotOutcome::Canceled;
                self.halt(RunOutcome::Canceled, "canceled", detail, true);
                self.persist()?;
                return Ok(false);
            }
            let assets = self.record.shots[shot_index].conditioning_assets.clone();
            // The body is the COMPILED request's, always (sc-22713): the document a reviewer reads
            // in `compiled.json` is what the route receives, on a resume and a replacement exactly
            // as on the first run. The record's own resolved conditioning is passed in rather than
            // re-resolved, so an attempt dispatched after a crash conditions on the same assets the
            // record already names.
            let request = self
                .compiled
                .request(&shot.id)
                .expect("every plan shot compiled a request");
            let body = request.to_job_body_with(
                &DispatchContext {
                    project_id: &project_id,
                    run_id: &self.record.run_id,
                    plan_id: &self.plan.id,
                    plan_version: self.plan.version,
                    attempt: attempt_number,
                    tier: self.plan.model.tier.as_deref(),
                    idempotency_key: Some(&key),
                    pack: &self.pack,
                    role_assets: &self.role_assets,
                },
                &ResolvedConditioning {
                    first_frame_asset_id: assets.first_frame_asset_id.clone(),
                    last_frame_asset_id: assets.last_frame_asset_id.clone(),
                    reference_asset_ids: assets.reference_asset_ids.clone(),
                },
            );
            let response = self
                .client
                .json("POST", "/api/v1/video/jobs", Some(body))
                .await?;
            if !(200..300).contains(&response.status) {
                // A refused enqueue is deterministic: the same body would be refused again, so it
                // consumes the shot rather than every remaining attempt.
                let detail = format!(
                    "POST /api/v1/video/jobs -> {}: {}",
                    response.status,
                    api_detail(&response.body)
                );
                let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
                attempt.status = "rejected".to_owned();
                attempt.error = Some(detail.clone());
                attempt.finished_at = Some(utc_now());
                self.persist()?;
                self.halt(
                    RunOutcome::Failed,
                    "enqueue_refused",
                    format!("shot {} could not be enqueued: {detail}", shot.id),
                    false,
                );
                return Ok(false);
            }
            job_id = Some(
                response
                    .body
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        HarnessError::Transport(format!(
                            "video job response has no id: {}",
                            response.body
                        ))
                    })?
                    .to_owned(),
            );
        }
        let job_id = job_id.expect("set on every branch above");
        {
            let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
            if created_here {
                // Nothing has ever run under this key: neither the record nor the API held a job
                // for it, and the one that exists now was created a moment ago. The clock starts
                // HERE. An attempt recorded by a controller that died before its POST reached the
                // API would otherwise be charged every hour since, and a resume the next morning
                // would dispatch a real render only to cancel it on the first poll.
                attempt.started_at = utc_now();
            }
            attempt.job_id = Some(job_id.clone());
            attempt.status = "running".to_owned();
        }
        self.persist()?;

        // 2. Wait for it, under whatever is left of this attempt's and the run's budgets.
        let attempt_started = Instant::now();
        // An attempt this controller just created has spent nothing; one it adopted from a
        // controller that died has been in flight since its recorded `startedAt`.
        let spent = if fresh {
            0.0
        } else {
            attempt_spent_seconds(&self.record.shots[shot_index].attempts[attempt_index])
        };
        let shot_deadline = attempt_started
            + Duration::from_secs_f64((self.plan.limits.max_shot_seconds as f64 - spent).max(0.0));
        let (view, poll_stop) = self
            .client
            .wait_for_job(&job_id, self.bounds(shot_deadline))
            .await?;

        // 3. Settle it into the record. The observed peak comes off the job's metrics block, which
        // the worker POSTs after its terminal progress — so read it once the attempt is terminal,
        // whatever ended it.
        let metrics = self.client.job_metrics(&job_id, self.poll_interval).await;
        let memory = memory_observation(metrics.as_ref(), &view, self.facts.host_memory_gb);
        // Whether the cancel the harness posted was actually honoured. A job still running after
        // the grace is a render in flight that nothing here can stop.
        let cancel_honoured = view.is_terminal();
        let settle_grace = ASSET_SETTLE_GRACE.min(self.shot_budget());
        let cancel_grace = CANCEL_GRACE.min(self.shot_budget());
        // The take names the partition that actually rendered it, not the plan's declared family
        // model (sc-23402).
        let dispatched_model =
            self.dispatched_model_for(&shot.id, shot_index, attempt_index, &view);
        let take = (poll_stop == PollStop::Terminal && view.status == "completed")
            .then(|| take_from_result(&view.result, &dispatched_model, view.backend.as_deref()))
            .flatten();
        {
            let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
            attempt.finished_at = Some(utc_now());
            attempt.elapsed_seconds = spent + attempt_started.elapsed().as_secs_f64();
            attempt.peak_gpu_memory_pct = memory.pct;
            attempt.peak_memory_gb = memory.gb;
            attempt.peak_memory_source = memory.source.clone();
            attempt.status = match poll_stop {
                // Terminal either way — for `AssetsUnsettled` the JOB finished and it is the
                // API's asset handoff that stalled, so the attempt records the job's own status
                // and says what stalled in `error`. A resume re-reads a completed attempt that
                // carries no take, which gives the handoff another chance to have landed.
                PollStop::Terminal | PollStop::AssetsUnsettled => view.status.clone(),
                PollStop::Operator => "canceled_by_operator".to_owned(),
                PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
            };
            attempt.error = match poll_stop {
                PollStop::Terminal if view.status == "completed" && take.is_none() => {
                    Some(format!(
                        "job {job_id} completed without an asset in its result: {}",
                        view.result
                    ))
                }
                PollStop::Terminal if view.status == "completed" => None,
                PollStop::Terminal => Some(view.failure_text()),
                PollStop::AssetsUnsettled => Some(format!(
                    "job {job_id} reached {} but its assets never settled within {:.0}s (the \
                     result still carries raw assetWrites): {}",
                    view.status,
                    settle_grace.as_secs_f64(),
                    view.result
                )),
                PollStop::Operator => Some(if cancel_honoured {
                    format!("canceled by operator (job {job_id} is {})", view.status)
                } else {
                    format!(
                        "canceled by operator; job {job_id} was still {} {:.0}s after the cancel \
                         was posted",
                        view.status,
                        cancel_grace.as_secs_f64()
                    )
                }),
                PollStop::ShotBudget if !cancel_honoured => Some(format!(
                    "attempt exceeded the per-shot budget of {}s and job {job_id} is still \
                     running {:.0}s after the cancel was posted — stopping dispatch rather than \
                     running a second render beside it",
                    self.plan.limits.max_shot_seconds,
                    cancel_grace.as_secs_f64()
                )),
                PollStop::ShotBudget => Some(format!(
                    "attempt exceeded the per-shot budget of {}s (last status {})",
                    self.plan.limits.max_shot_seconds, view.status
                )),
                PollStop::RunBudget if !cancel_honoured => Some(format!(
                    "run exceeded its budget of {}s and job {job_id} is still running {:.0}s \
                     after the cancel was posted — stopping dispatch rather than running a second \
                     render beside it",
                    self.plan.limits.max_run_seconds,
                    cancel_grace.as_secs_f64()
                )),
                PollStop::RunBudget => Some(format!(
                    "run exceeded its budget of {}s while this attempt was in flight (last status \
                     {})",
                    self.plan.limits.max_run_seconds, view.status
                )),
            };
            attempt.take = take.clone();
        }
        if let Some(take) = &take {
            // The first usable take of a shot becomes its selected take. Replay never reaches this
            // line for a shot that already has one, and nothing but `replace-take` moves it.
            if self.record.shots[shot_index].selected_attempt.is_none() {
                self.record.shots[shot_index].selected_attempt = Some(attempt_number);
            }
            if let Some(model) = self.record.model.as_mut() {
                if model.backend_observed.is_none() {
                    model.backend_observed = take.backend.clone();
                }
            }
        }
        self.persist()?;

        // 4. Decide whether the run may keep dispatching.
        match poll_stop {
            PollStop::AssetsUnsettled => {
                // Retrying would dispatch a fresh render against a server-side condition a retry
                // cannot fix.
                self.halt(
                    RunOutcome::Failed,
                    "assets_unsettled",
                    format!(
                        "shot {}'s job finished but the API never persisted its assets; the take \
                         cannot be adopted and a retry would not fix it",
                        shot.id
                    ),
                    false,
                );
                return Ok(false);
            }
            PollStop::Operator => {
                self.halt(
                    RunOutcome::Canceled,
                    "canceled",
                    if cancel_honoured {
                        format!("canceled while shot {} was in flight", shot.id)
                    } else {
                        format!(
                            "canceled while shot {} was in flight, and the job was still {} when \
                             the grace ran out — check the GPU before resuming",
                            shot.id, view.status
                        )
                    },
                    true,
                );
                return Ok(false);
            }
            PollStop::RunBudget => {
                self.halt(
                    RunOutcome::StoppedRunBudget,
                    "run_budget",
                    format!(
                        "the run's {}s wall-clock budget ran out while shot {} was in flight; \
                         raise limits.maxRunSeconds and start a new run",
                        self.plan.limits.max_run_seconds, shot.id
                    ),
                    false,
                );
                return Ok(false);
            }
            PollStop::ShotBudget if !cancel_honoured => {
                // The render is still on the GPU. Neither the next attempt nor the next shot may go
                // out beside it: the plan declared ONE memory budget, and two renders in flight is
                // exactly what it is there to prevent.
                self.halt(
                    RunOutcome::Failed,
                    "cancel_not_honoured",
                    format!(
                        "shot {}'s job was still {} after the cancel grace, so a render is in \
                         flight that nothing here can stop; no further dispatch",
                        shot.id, view.status
                    ),
                    false,
                );
                return Ok(false);
            }
            _ => {}
        }
        if self.memory_over_budget(&shot.id, &memory) {
            return Ok(false);
        }
        Ok(true)
    }

    /// Adopt whatever the API holds for attempts a previous controller left unfinished.
    ///
    /// Every unfinished attempt is looked up (by its recorded job id, or by its idempotency key
    /// when the controller died before recording one) and read to its current state: a completed
    /// job is imported as a take, a failed one counts as a spent attempt, and a job still running
    /// is left for the dispatch loop to resume polling. Nothing here creates a job.
    pub(super) async fn reconcile_shot(
        &mut self,
        shot: &film_plan::Shot,
    ) -> Result<(), HarnessError> {
        let project_id = self.project_id()?;
        let Some(shot_index) = self.record.shots.iter().position(|s| s.shot_id == shot.id) else {
            return Ok(());
        };
        let unfinished: Vec<usize> = self.record.shots[shot_index]
            .attempts
            .iter()
            .enumerate()
            .filter(|(_, attempt)| {
                !TERMINAL_ATTEMPT_STATUSES.contains(&attempt.status.as_str())
                    || (attempt.status == "completed" && attempt.take.is_none())
            })
            .map(|(index, _)| index)
            .collect();
        for attempt_index in unfinished {
            let attempt = &self.record.shots[shot_index].attempts[attempt_index];
            let key = attempt.idempotency_key.clone();
            let job_id = match attempt.job_id.clone() {
                Some(job_id) => Some(job_id),
                None => {
                    self.client
                        .find_job_by_idempotency_key(&project_id, &key)
                        .await?
                }
            };
            let Some(job_id) = job_id else {
                // The controller died before the job was created. The attempt stays pending and the
                // dispatch loop creates it under the same key.
                continue;
            };
            let view = self.client.get_job(&job_id).await?;
            let terminal = view.is_terminal() && view.is_settled();
            let dispatched_model =
                self.dispatched_model_for(&shot.id, shot_index, attempt_index, &view);
            let take = (terminal && view.status == "completed")
                .then(|| take_from_result(&view.result, &dispatched_model, view.backend.as_deref()))
                .flatten();
            let mut adopted_memory = None;
            if terminal {
                let metrics = self.client.job_metrics(&job_id, self.poll_interval).await;
                let memory = memory_observation(metrics.as_ref(), &view, self.facts.host_memory_gb);
                let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
                attempt.peak_gpu_memory_pct = memory.pct;
                attempt.peak_memory_gb = memory.gb;
                attempt.peak_memory_source = memory.source.clone();
                adopted_memory = Some(memory);
                attempt.job_id = Some(job_id.clone());
                attempt.status = view.status.clone();
                attempt.finished_at.get_or_insert_with(utc_now);
                attempt.error = match (&view.status[..], take.is_some()) {
                    ("completed", true) => None,
                    ("completed", false) => Some(format!(
                        "job {job_id} completed without an asset in its result: {}",
                        view.result
                    )),
                    _ => Some(view.failure_text()),
                };
                attempt.take = take.clone();
            } else {
                let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
                attempt.job_id = Some(job_id.clone());
                attempt.status = "running".to_owned();
            }
            if take.is_some() && self.record.shots[shot_index].selected_attempt.is_none() {
                let number = self.record.shots[shot_index].attempts[attempt_index].attempt;
                self.record.shots[shot_index].selected_attempt = Some(number);
            }
            if let Some(take) = &take {
                if let Some(model) = self.record.model.as_mut() {
                    if model.backend_observed.is_none() {
                        model.backend_observed = take.backend.clone();
                    }
                }
            }
            self.persist()?;
            // The plan's memory budget is judged on the EVIDENCE, not on who was watching when it
            // landed: an adopted terminal attempt whose peak is over budget stops new dispatch here
            // exactly as it would in `work_attempt`. The take itself is kept — it exists — but the
            // run goes no further (E5 / AC3).
            if let Some(memory) = adopted_memory {
                self.memory_over_budget(&shot.id, &memory);
            }
        }
        Ok(())
    }

    /// Close the run: classify the outcome, stamp the record `finished` and write it one last time.
    pub(super) fn finish(&mut self, export_ok: bool) -> Result<(), HarnessError> {
        let all_rendered = self
            .record
            .shots
            .iter()
            .filter(|shot| self.is_selected(&shot.shot_id))
            .all(|shot| shot.selected_attempt.is_some());
        let (outcome, stop) = match self.stop.take() {
            Some((outcome, stop)) => (outcome, Some(stop)),
            None if all_rendered && export_ok => (RunOutcome::Completed, None),
            None if all_rendered => (
                RunOutcome::Failed,
                Some(RunStop {
                    reason: "export_failed".to_owned(),
                    detail: "every selected shot rendered but the timeline export did not \
                             complete; `film-harness resume` retries the export"
                        .to_owned(),
                    resumable: true,
                }),
            ),
            None => (
                RunOutcome::Failed,
                Some(RunStop {
                    reason: "attempts_exhausted".to_owned(),
                    detail: "a selected shot has no usable take and no attempts left; \
                             `film-harness replace-take --shot <id>` authorises one more"
                        .to_owned(),
                    resumable: false,
                }),
            ),
        };
        self.record.outcome = outcome;
        self.record.stop = stop;
        self.record.state = RunState::Finished;
        self.record.finished_at = Some(utc_now());
        self.persist()
    }

    /// Close a `replace-take` invocation, which is scoped to ONE shot.
    ///
    /// [`Session::finish`] classifies the RUN: its `all_rendered` is computed over every selected
    /// shot, so closing a run that still has unrendered shots through it overwrites whatever stop
    /// the run actually had with `attempts_exhausted` / `resumable: false` — permanently blocking
    /// the `resume` that was going to render them. A replacement decides one shot's outcome and
    /// nothing else: when the run carried a resumable stop and other selected shots are still
    /// outstanding, that stop is left exactly where it was.
    ///
    /// The same rule holds for a run that had NO stop, which is to say one that `completed`
    /// (sc-22715). `all_rendered` is false the moment any selected shot has no selection — a
    /// `reject-take` leaves exactly that — so classifying through [`Session::finish`] turned a
    /// completed run into `failed` / `attempts_exhausted` because a DIFFERENT shot was replaced.
    /// That is the `edit_timeline` rule stated for the generation side: a replacement is not a
    /// run, and a verdict the run reached is not re-opened by one. The evaluation film's own
    /// record read `failed` for exactly this reason while its snapshots read `completed`.
    pub(super) fn finish_replacement(
        &mut self,
        export_ok: bool,
        replaced_shot_id: &str,
        prior: (RunOutcome, Option<RunStop>),
    ) -> Result<(), HarnessError> {
        let outstanding: Vec<String> = self
            .record
            .shots
            .iter()
            .filter(|shot| shot.shot_id != replaced_shot_id)
            .filter(|shot| self.is_selected(&shot.shot_id) && shot.selected_attempt.is_none())
            .map(|shot| shot.shot_id.clone())
            .collect();
        // Did THIS replacement land a take on the shot it was asked about? That, and not the state
        // of every other shot, is what this invocation decided.
        let replacement_succeeded = self
            .record
            .shots
            .iter()
            .find(|shot| shot.shot_id == replaced_shot_id)
            .is_some_and(|shot| shot.selected_attempt.is_some());
        // An `export_failed` stop is about an export this replacement did not redo (no
        // `--export`): the run still has no MP4, so the stop — and the `resume` that retries the
        // export — must survive too (sc-22715).
        let export_still_owed = !self.export
            && prior
                .1
                .as_ref()
                .is_some_and(|stop| stop.resumable && stop.reason == "export_failed");
        match prior {
            (outcome, Some(stop))
                if !stop.resumable && replacement_succeeded && self.stop.is_none() =>
            {
                self.record.outcome = outcome;
                self.record.stop = Some(stop);
                self.record.state = RunState::Finished;
                self.record.finished_at = Some(utc_now());
                self.persist()
            }
            (outcome, Some(stop))
                if stop.resumable && (!outstanding.is_empty() || export_still_owed) =>
            {
                self.note_decision(
                    "replace_take",
                    Some(replaced_shot_id),
                    if outstanding.is_empty() {
                        format!(
                            "the run's own stop ({}) is left in place: the export was not redone, \
                             so the run is still resumable",
                            stop.reason
                        )
                    } else {
                        format!(
                            "the run's own stop ({}) is left in place: {} still {} no take, so the \
                             run is still resumable",
                            stop.reason,
                            outstanding.join(", "),
                            if outstanding.len() == 1 { "has" } else { "have" }
                        )
                    },
                );
                self.stop = None;
                self.record.outcome = outcome;
                self.record.stop = Some(stop);
                self.record.state = RunState::Finished;
                self.record.finished_at = Some(utc_now());
                self.persist()
            }
            // The run reached its verdict with no stop of its own — it COMPLETED — and this
            // replacement landed. `finish` would re-derive that verdict from every selected shot,
            // so a shot whose take a human rejected flips the whole run to `attempts_exhausted`
            // because another shot was replaced. Only the export moves: it no longer matches the
            // sequence, which `replace_take` already recorded as `stale`.
            (RunOutcome::Completed, None) if replacement_succeeded && self.stop.is_none() => {
                if export_ok {
                    self.note_decision(
                        "replace_take",
                        Some(replaced_shot_id),
                        format!(
                            "the run's own outcome (completed) is left in place: this replacement \
                             decided shot {replaced_shot_id} and no other shot's state was \
                             re-judged"
                        ),
                    );
                    self.record.outcome = RunOutcome::Completed;
                    self.record.stop = None;
                } else {
                    // The take landed and the re-export did not: that, and only that, is what
                    // this invocation failed at — `attempts_exhausted` would name a different
                    // shot's state as the reason and refuse the `resume` that retries the export.
                    self.record.outcome = RunOutcome::Failed;
                    self.record.stop = Some(RunStop {
                        reason: "export_failed".to_owned(),
                        detail: "the replacement landed but its re-export did not complete; \
                                 `film-harness resume` retries the export"
                            .to_owned(),
                        resumable: true,
                    });
                }
                self.record.state = RunState::Finished;
                self.record.finished_at = Some(utc_now());
                self.persist()
            }
            _ => self.finish(export_ok),
        }
    }

    /// Cancellation does not abandon jobs an interrupted controller may already have created.
    /// Resolve durable intentions (including lost POST responses) and settle existing jobs under
    /// the usual grace. Each worker's dispatch boundary refuses to create a missing job.
    pub(super) async fn reconcile_canceled_work(&mut self) -> Result<(), HarnessError> {
        let Some(project_id) = self.record.project_id.clone() else {
            self.halt(
                RunOutcome::Canceled,
                "canceled",
                "canceled before project creation".to_owned(),
                true,
            );
            return Ok(());
        };
        let entries: Vec<_> = self
            .pack
            .sound
            .iter()
            .filter(|entry| {
                self.record
                    .synthesized_sound
                    .iter()
                    .any(|line| line.role == entry.role && !line.is_terminal())
            })
            .cloned()
            .collect();
        for entry in entries {
            self.synthesize_dialogue(&project_id, &entry).await?;
        }
        for shot in self.plan.shots.clone() {
            let Some(index) = self
                .record
                .shots
                .iter()
                .position(|record| record.shot_id == shot.id)
            else {
                continue;
            };
            for attempt in 0..self.record.shots[index].attempts.len() {
                if !TERMINAL_ATTEMPT_STATUSES
                    .contains(&self.record.shots[index].attempts[attempt].status.as_str())
                {
                    self.work_attempt(&shot, index, attempt, false).await?;
                    if self.record.shots[index].selected_attempt.is_none()
                        && matches!(
                            self.record.shots[index].attempts[attempt].status.as_str(),
                            "canceled" | "canceled_by_operator"
                        )
                    {
                        self.record.shots[index].outcome = ShotOutcome::Canceled;
                    }
                }
            }
        }
        if self.record.export_pending.is_some()
            || self
                .record
                .export
                .as_ref()
                .is_some_and(|export| export.status == "running")
        {
            self.run_export().await?;
        }
        self.halt(
            RunOutcome::Canceled,
            "canceled",
            "operator canceled; existing jobs reconciled without new dispatch".to_owned(),
            true,
        );
        Ok(())
    }

    /// Project -> references -> shots -> timeline -> export -> close.
    pub(super) async fn drive(mut self) -> Result<RunRecord, HarnessError> {
        let outcome = if self.canceled() {
            self.reconcile_canceled_work().await.map(|()| false)
        } else {
            let result = self.drive_inner().await;
            if result.is_ok() && self.canceled() {
                self.reconcile_canceled_work().await.map(|()| false)
            } else {
                result
            }
        };
        match outcome {
            Ok(export_ok) => {
                self.finish(export_ok)?;
                Ok(self.record)
            }
            // A transport/API failure partway through is exactly when the record matters most: by
            // then the run may have created a project, imported assets and dispatched jobs. It
            // stays `running`, because that is what it is — a resume reconciles it.
            Err(error) => {
                self.record.diagnostics.push(PlanDiagnostic::plan(
                    "run",
                    format!("the run stopped on an error: {error}"),
                ));
                // Best effort: an io failure while writing the record must not replace the failure
                // that caused it with a less informative one.
                let _ = self.persist();
                Err(error)
            }
        }
    }

    pub(super) async fn drive_inner(&mut self) -> Result<bool, HarnessError> {
        self.ensure_project().await?;
        self.ensure_references().await?;
        // Before the shots, not after: a resume and a `replace-take` both re-lay the timeline at
        // the end of this same method, and the re-layout needs the clips to place. Importing here
        // means the sound of a run is settled by the time the first take exists, whichever entry
        // point is driving (sc-22712).
        self.ensure_sound().await?;
        // A synthesis that failed, was cancelled or ran past a declared limit has already recorded
        // its stop (sc-23404). Dispatching renders against a sequence that is missing a line would
        // spend GPU hours on a film the operator would have to re-export anyway, so the run stops
        // here with everything it has and `resume` speaks the missing line.
        if self.stop.is_some() {
            return Ok(false);
        }
        self.work_shots().await?;
        self.assemble_and_export().await
    }

    /// Flag every shot that declared a dependency on `shot_id`. See [`flag_dependents`], which the
    /// human reject path in [`review::decide_take`] shares, so a take that stops being the selected
    /// one raises exactly the same signal however it stopped.
    pub(super) fn flag_dependents(&mut self, shot_id: &str, reason: &str) {
        flag_dependents(
            &mut self.record,
            &self.plan,
            shot_id,
            &format!("its selected take was replaced ({reason})"),
        );
    }
}

/// Execute `options` end to end. Returns the run record (also written to `options.out_dir`) on
/// every path that got past validation, including runs that stopped on a limit; a refused plan
/// returns [`HarnessError::Validation`] after writing a `rejected` record.
pub async fn run(
    transport: &dyn ApiTransport,
    options: &RunOptions,
) -> Result<RunRecord, HarnessError> {
    run_with_control(transport, options, &RunControl::default()).await
}

/// [`run`] with an operator cancellation handle. The `film-harness` binary passes one so SIGINT
/// cancels the in-flight job through the API and still leaves a record, instead of orphaning a
/// render on the GPU.
///
/// Every path past document validation writes `run.json` — sc-22711 writes it at every state
/// transition, not only at the end, so a controller killed anywhere leaves a record `resume` can
/// reconcile against the API.
pub async fn run_with_control(
    transport: &dyn ApiTransport,
    options: &RunOptions,
    control: &RunControl,
) -> Result<RunRecord, HarnessError> {
    let lease = ControllerLease::acquire(
        &options.out_dir,
        format!("controller_{}", uuid::Uuid::new_v4().simple()),
    )?;
    run_with_control_and_lease(transport, options, control, lease).await
}

/// Run with a lease acquired by the caller before it acknowledged a mutating request.
pub async fn run_with_control_and_lease(
    transport: &dyn ApiTransport,
    options: &RunOptions,
    control: &RunControl,
    _lease: ControllerLease,
) -> Result<RunRecord, HarnessError> {
    record_action(
        &options.out_dir,
        "start",
        None,
        run_with_control_inner(transport, options, control),
    )
    .await
}

pub(super) async fn run_with_control_inner(
    transport: &dyn ApiTransport,
    options: &RunOptions,
    control: &RunControl,
) -> Result<RunRecord, HarnessError> {
    // A `run` over a directory that already holds a record would mint a NEW run id and persist over
    // the previous run's state, while `persist_record` keeps the plan/pack copies it finds (they
    // are written once and then left alone) — so the surviving documents would belong to the old
    // run and the record's hashes to the new one, and a later resume would refuse or mis-hash.
    // `scripts/film-harness-smoke.sh` pins `--out "$SMOKE_DIR/run"`, so a second invocation is
    // exactly this. The takes, decisions and provenance of a run are not something a typo may
    // overwrite (E2).
    if let Ok(existing) = read_run_record(&options.out_dir) {
        return Err(HarnessError::Refused(format!(
            "{} already holds run {} ({:?}); `film-harness status --out {}` prints it, \
             `film-harness resume --out {}` continues it, and a NEW run needs a different --out",
            options.out_dir.join(RUN_RECORD_FILE).display(),
            existing.run_id,
            existing.outcome,
            options.out_dir.display(),
            options.out_dir.display()
        )));
    }
    let started = Instant::now();
    let run_id = format!("run_{}", uuid::Uuid::new_v4().simple());
    let plan_bytes = std::fs::read(&options.plan_path)?;
    let pack_bytes = std::fs::read(&options.reference_pack_path)?;
    let client = Client { transport, control };

    // Steps 1-3: refuse before the first write.
    let (plan, pack) = match validate(None, options).await {
        Ok(documents) => documents,
        Err(HarnessError::Validation(findings)) => {
            let record = rejected_record(
                &run_id,
                options,
                &plan_bytes,
                &pack_bytes,
                findings.clone(),
                started,
            );
            persist_record(
                &record,
                &options.out_dir,
                &options.plan_path,
                &options.reference_pack_path,
            )?;
            return Err(HarnessError::Validation(findings));
        }
        Err(other) => return Err(other),
    };
    let prepared = match prepare(
        &client,
        &plan,
        &pack,
        options.export,
        options.require_installed,
        options.shot_ids.as_deref(),
    )
    .await?
    {
        Ok(prepared) => prepared,
        Err(findings) => {
            let mut record = base_record(&run_id, &plan, &pack, options, &plan_bytes, &pack_bytes);
            record.state = RunState::Finished;
            record.outcome = RunOutcome::Rejected;
            record.stop = Some(RunStop {
                reason: "rejected".to_owned(),
                detail: "the plan was refused against this API's catalog and host".to_owned(),
                resumable: false,
            });
            record.diagnostics = findings.clone();
            record.finished_at = Some(utc_now());
            record.elapsed_seconds = seconds_since(started);
            persist_record(
                &record,
                &options.out_dir,
                &options.plan_path,
                &options.reference_pack_path,
            )?;
            return Err(HarnessError::Validation(findings));
        }
    };
    let lane = prepared.facts.lane();
    let mut record = base_record(&run_id, &plan, &pack, options, &plan_bytes, &pack_bytes);
    // The requests this run dispatches: the compiled document beside the plan when there is one,
    // else the plan's own prompts compiled in memory (sc-22713). `record.plan.sha256` is the hash
    // of the plan file as read, which is what a compiled document pins. A resume re-reads the same
    // document and re-checks the same hash, so the requests a replay dispatches are the ones the
    // first controller did.
    let supplied_compiled = read_compiled_for(options)?;
    let supplied_compiled_source = match supplied_compiled.as_ref() {
        // The hash is what makes the run reproducible, so an unreadable file is an error rather
        // than an empty string on the record. It was read successfully microseconds ago in
        // `read_compiled_for`, so `?` here can only surface a real io failure.
        Some((compiled, path)) => Some(SourceDocument {
            id: compiled.plan_id.clone(),
            version: compiled.plan_version,
            path: path.display().to_string(),
            sha256: sha256_hex(&std::fs::read(path)?),
        }),
        None => None,
    };
    let compiled = compiled_for_run(
        &plan,
        &pack,
        &prepared.entries(),
        lane,
        &record.plan.sha256,
        supplied_compiled.map(|(compiled, _)| compiled),
    )?;
    record.compiled = match supplied_compiled_source {
        Some(source) => Some(source),
        None => {
            let path = options.out_dir.join("compiled.json");
            let bytes = serde_json::to_vec_pretty(&compiled)
                .map_err(|error| HarnessError::Io(error.to_string()))?;
            write_atomically(&path, &bytes)?;
            Some(SourceDocument {
                id: compiled.plan_id.clone(),
                version: compiled.plan_version,
                path: path.display().to_string(),
                sha256: sha256_hex(&bytes),
            })
        }
    };
    record.model = Some(ModelRecord {
        id: plan.model.id.clone(),
        tier_requested: plan.model.tier.clone(),
        fps: prepared.fps,
        lane: lane.manifest_key().to_owned(),
        backend_observed: None,
        weights: primary_weights(prepared.base_entry(), plan.model.tier.as_deref()),
        partition_weights: partition_weights(
            &plan,
            &prepared.entries(),
            plan.model.tier.as_deref(),
            record.selected_shot_ids.as_slice(),
        ),
        hardware: HardwareRecord {
            platform: prepared.facts.platform_or_local().to_owned(),
            // The host-capabilities route reports no arch, so this process's arch is the truth only
            // when the API is on this platform; otherwise claiming one would be a fabrication.
            arch: if prepared.facts.platform_or_local() == std::env::consts::OS {
                std::env::consts::ARCH.to_owned()
            } else {
                "unknown".to_owned()
            },
            host_memory_gb: prepared.facts.host_memory_gb,
            gpu_name: prepared.facts.video_gpu_name.clone(),
            worker_id: prepared.facts.video_worker_id.clone(),
        },
    });
    let run_deadline = started + Duration::from_secs(plan.limits.max_run_seconds);
    let mut session = Session {
        client,
        transport,
        plan_path: options.plan_path.clone(),
        pack_path: options.reference_pack_path.clone(),
        out_dir: options.out_dir.clone(),
        poll_interval: options.poll_interval,
        export: options.export,
        plan,
        pack,
        compiled,
        facts: prepared.facts,
        fps: prepared.fps,
        record,
        started,
        prior_elapsed: 0.0,
        prior_human_elapsed: 0.0,
        charges_run_budget: true,
        run_deadline: Some(run_deadline),
        role_assets: BTreeMap::new(),
        sound_assets: BTreeMap::new(),
        stop: None,
    };
    // The record exists before the first API write, so even a controller killed during project
    // creation leaves something `resume` can reconcile.
    session.persist()?;
    session.drive().await
}

/// What `resume` and `replace_take` both need: the record, its documents re-read and re-hashed, and
/// the catalog/host judged again (either can have changed while the run was not held).
pub(super) struct Continued {
    pub(super) record: RunRecord,
    pub(super) plan: ProductionPlan,
    pub(super) pack: ReferencePack,
    /// The requests the run was dispatching, re-read from the document the record names (and
    /// re-checked against its recorded hash) or recompiled from the plan when it named none — so a
    /// resume dispatches what the first controller did, not a fresh compilation of an edited file.
    pub(super) compiled: CompiledPlan,
    pub(super) plan_path: PathBuf,
    pub(super) pack_path: PathBuf,
    pub(super) prepared: Prepared,
}

/// Read `<out_dir>/run.json`, re-read the documents it was started from, and refuse unless they
/// still hash to what the run recorded.
///
/// The hash check is the load-bearing part: resuming against an edited plan would silently render
/// shots the record's takes were never meant to sit beside. An edited plan is a new run, and the
/// refusal says so.
pub(super) async fn continue_run(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
) -> Result<Continued, HarnessError> {
    let record = read_run_record(&options.out_dir)?;
    if record.schema_version != RUN_RECORD_SCHEMA_VERSION {
        return Err(HarnessError::Refused(format!(
            "run record schema version {} (this build reads {RUN_RECORD_SCHEMA_VERSION})",
            record.schema_version
        )));
    }
    let (plan_path, plan_bytes) = read_source(
        Path::new(&record.plan.path),
        &options.out_dir.join("plan.json"),
        "plan",
    )?;
    let (pack_path, pack_bytes) = read_source(
        Path::new(&record.reference_pack.path),
        &options.out_dir.join("references.json"),
        "reference pack",
    )?;
    for (label, bytes, recorded) in [
        ("plan", &plan_bytes, &record.plan.sha256),
        ("reference pack", &pack_bytes, &record.reference_pack.sha256),
    ] {
        let actual = sha256_hex(bytes);
        if &actual != recorded {
            return Err(HarnessError::Refused(format!(
                "the {label} changed since run {} started (recorded {recorded}, found {actual}); a \
                 changed {label} is a new run, not a resume",
                record.run_id
            )));
        }
    }
    let plan = film_plan::parse_plan(std::str::from_utf8(&plan_bytes).unwrap_or_default())
        .map_err(|error| HarnessError::Refused(format!("plan: {error}")))?;
    let pack =
        film_plan::parse_reference_pack(std::str::from_utf8(&pack_bytes).unwrap_or_default())
            .map_err(|error| HarnessError::Refused(format!("reference pack: {error}")))?;
    let pack_dir = pack_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let findings = film_plan::validate_all(&plan, &pack, Some(&pack_dir), None);
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let client = Client {
        transport,
        control: &options.control,
    };
    // Admission follows the saved operation's scope. A replacement never requires another
    // pending shot's partition, and its saved export choice survives API startup adoption.
    let selection = record
        .active_take_operation
        .as_ref()
        .map(|operation| std::slice::from_ref(&operation.shot_id))
        .unwrap_or(&record.selected_shot_ids);
    let export = record
        .active_take_operation
        .as_ref()
        .map(|operation| operation.export)
        .unwrap_or(options.export);
    let prepared = prepare(
        &client,
        &plan,
        &pack,
        export,
        options.require_installed,
        Some(selection),
    )
    .await?
    .map_err(HarnessError::Validation)?;
    // The requests the run was dispatching. A record that named a compiled document is held to it,
    // hash and all — re-reading the file the run started from is what keeps a resume from
    // dispatching an edited `compiled.json` under the takes the first controller already made, the
    // same rule the plan and the pack are held to. A record that named none compiles the plan in
    // memory exactly as the first controller did (sc-22713).
    let supplied = match &record.compiled {
        Some(document) => {
            let (_, bytes) = read_source(
                Path::new(&document.path),
                &options.out_dir.join("compiled.json"),
                "compiled requests",
            )?;
            let actual = sha256_hex(&bytes);
            if actual != document.sha256 {
                return Err(HarnessError::Refused(format!(
                    "the compiled requests changed since run {} started (recorded {}, found \
                     {actual}); recompile and start a new run",
                    record.run_id, document.sha256
                )));
            }
            Some(
                serde_json::from_str::<CompiledPlan>(
                    &sceneworks_core::jsonc::strip_jsonc_comments(
                        std::str::from_utf8(&bytes).unwrap_or_default(),
                    ),
                )
                .map_err(|error| HarnessError::Refused(format!("compiled requests: {error}")))?,
            )
        }
        None => None,
    };
    let compiled = compiled_for_run(
        &plan,
        &pack,
        &prepared.entries(),
        prepared.facts.lane(),
        &record.plan.sha256,
        supplied,
    )?;
    Ok(Continued {
        record,
        plan,
        pack,
        compiled,
        plan_path,
        pack_path,
        prepared,
    })
}

/// `run_deadline: None` is a session that runs OUTSIDE the plan's automatic run budget (a
/// replacement): its wall-clock is booked as human-requested and nothing it polls can stop on
/// `run_budget`.
pub(super) fn session_from<'a>(
    transport: &'a dyn ApiTransport,
    options: &'a ResumeOptions,
    continued: Continued,
    started: Instant,
    run_deadline: Option<Instant>,
) -> Session<'a> {
    Session {
        client: Client {
            transport,
            control: &options.control,
        },
        transport,
        plan_path: continued.plan_path,
        pack_path: continued.pack_path,
        out_dir: options.out_dir.clone(),
        poll_interval: options.poll_interval,
        export: options.export,
        plan: continued.plan,
        pack: continued.pack,
        compiled: continued.compiled,
        facts: continued.prepared.facts,
        fps: continued.prepared.fps,
        prior_elapsed: continued.record.elapsed_seconds,
        prior_human_elapsed: continued.record.human_requested_elapsed_seconds,
        charges_run_budget: run_deadline.is_some(),
        record: continued.record,
        started,
        run_deadline,
        role_assets: BTreeMap::new(),
        sound_assets: BTreeMap::new(),
        stop: None,
    }
}

/// Pick a run back up where a crash, a cancel or a failed export left it.
///
/// Every completed take is reused, every job the record names is read back from the API and adopted
/// at whatever state it actually reached, and the run continues under whatever is LEFT of the
/// plan's wall-clock budget and per-shot attempt cap — so a restart loop cannot turn a bounded run
/// into an unbounded one. The selected take of every shot is left exactly as it was.
pub async fn resume(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
) -> Result<RunRecord, HarnessError> {
    let lease = ControllerLease::acquire_new_action(
        &options.out_dir,
        format!("resume_{}", uuid::Uuid::new_v4().simple()),
    )?;
    resume_with_lease(transport, options, lease).await
}

pub(crate) async fn resume_with_lease(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
    _lease: ControllerLease,
) -> Result<RunRecord, HarnessError> {
    record_action(
        &options.out_dir,
        "resume",
        None,
        resume_inner(transport, options),
    )
    .await
}

pub(super) async fn resume_inner(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
) -> Result<RunRecord, HarnessError> {
    let started = Instant::now();
    let continued = continue_run(transport, options).await?;
    if continued.record.active_take_operation.is_some() {
        return recover_take_operation(transport, options, continued, started).await;
    }
    if !continued.record.is_resumable() {
        let detail = continued
            .record
            .stop
            .as_ref()
            .map(|stop| format!("{}: {}", stop.reason, stop.detail))
            .unwrap_or_else(|| format!("outcome {:?}", continued.record.outcome));
        return Err(HarnessError::Refused(format!(
            "run {} is not resumable ({detail})",
            continued.record.run_id
        )));
    }
    let remaining =
        (continued.plan.limits.max_run_seconds as f64 - continued.record.elapsed_seconds).max(0.0);
    let run_deadline = started + Duration::from_secs_f64(remaining);
    let mut session = session_from(transport, options, continued, started, Some(run_deadline));
    session.record.state = RunState::Running;
    session.record.stop = None;
    session.record.finished_at = None;
    session.note_decision(
        "resume",
        None,
        format!(
            "resumed with {remaining:.0}s of the plan's {}s budget left",
            session.plan.limits.max_run_seconds
        ),
    );
    session.persist()?;
    // Even an exhausted run can still own a job. A pending Cancel must settle that job;
    // a later explicit resume still encounters the unchanged exhausted budget.
    if remaining <= 0.0 && !session.canceled() {
        session.halt(
            RunOutcome::StoppedRunBudget,
            "run_budget",
            format!(
                "the run already spent its {}s wall-clock budget; raise limits.maxRunSeconds and \
                 start a new run",
                session.plan.limits.max_run_seconds
            ),
            false,
        );
        session.finish(false)?;
        return Ok(session.record);
    }
    session.drive().await
}

/// Reject the take a shot is currently carrying and render exactly one more, for that shot alone.
///
/// This is a human decision, not a retry: it authorises ONE bounded attempt (it does not loop, and
/// it does not spend or respect the plan's automatic attempt cap), it touches no other shot's
/// takes, jobs or assets, and it leaves the rejected take in the record beside the new one. When
/// the replacement lands, every shot that DECLARED a dependency on this one is flagged
/// `needs_review` and the export is marked stale — flagged, never regenerated.
pub async fn replace_take(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
    shot_id: &str,
    reason: &str,
) -> Result<RunRecord, HarnessError> {
    let lease = ControllerLease::acquire_new_action(
        &options.out_dir,
        format!("replace_{}", uuid::Uuid::new_v4().simple()),
    )?;
    replace_take_with_lease(transport, options, shot_id, reason, lease).await
}

pub(crate) async fn replace_take_with_lease(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
    shot_id: &str,
    reason: &str,
    _lease: ControllerLease,
) -> Result<RunRecord, HarnessError> {
    replace_take_operation_with_lease(transport, options, shot_id, reason, "replacement", _lease)
        .await
}

pub(crate) async fn replace_take_operation_with_lease(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
    shot_id: &str,
    reason: &str,
    kind: &str,
    _lease: ControllerLease,
) -> Result<RunRecord, HarnessError> {
    record_action(
        &options.out_dir,
        kind,
        Some(shot_id),
        replace_take_operation(transport, options, shot_id, reason, kind),
    )
    .await
}

pub(super) async fn replace_take_operation(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
    shot_id: &str,
    reason: &str,
    kind: &str,
) -> Result<RunRecord, HarnessError> {
    let started = Instant::now();
    let continued = continue_run(transport, options).await?;
    if continued.record.active_take_operation.is_some() {
        return Err(HarnessError::Refused(
            "A saved take operation must be resumed before requesting another replacement"
                .to_owned(),
        ));
    }
    if continued.record.project_id.is_none() {
        return Err(HarnessError::Refused(format!(
            "run {} never created a project, so it has no take to replace",
            continued.record.run_id
        )));
    }
    let Some(shot) = continued
        .plan
        .shots
        .iter()
        .find(|shot| shot.id == shot_id)
        .cloned()
    else {
        return Err(HarnessError::Refused(format!(
            "{shot_id:?} is not a shot in plan {:?}",
            continued.plan.id
        )));
    };
    if !continued
        .record
        .selected_shot_ids
        .iter()
        .any(|id| id == shot_id)
    {
        return Err(HarnessError::Refused(format!(
            "shot {shot_id} is not in run {}'s selection",
            continued.record.run_id
        )));
    }
    if options.control.is_canceled() {
        return Err(HarnessError::Canceled(
            "take action canceled during admission; the previous take and attempt budget are unchanged".to_owned(),
        ));
    }
    // One replacement runs OUTSIDE the run budget (sc-22715): the attempt is bounded by the
    // per-shot budget and the re-export by its own, and neither is charged to `elapsedSeconds` —
    // the human just authorised this one attempt, and a replacement that spent the run's remaining
    // wall-clock would make the run's own `resume` refuse with "raise limits.maxRunSeconds". Its
    // wall-clock is booked in `humanRequestedElapsedSeconds` instead, so the total cost is still
    // on the record. With no run deadline, an export that overruns is `export_failed` (resumable),
    // never `run_budget` (terminal).
    let mut session = session_from(transport, options, continued, started, None);
    // What the RUN said before this replacement: a replacement is scoped to one shot and must not
    // re-classify the run, so a resumable stop (a cancel, a crash, a failed export) is restored by
    // `finish_replacement` when other selected shots are still outstanding — and a run that
    // finished with no stop at all (`completed`) keeps that verdict, which is why the OUTCOME is
    // carried here even when there is no stop beside it (sc-22715).
    let prior_verdict = (session.record.outcome, session.record.stop.clone());
    session.record.state = RunState::Running;
    session.record.stop = None;
    session.record.finished_at = None;

    let Some(index) = session
        .record
        .shots
        .iter()
        .position(|record| record.shot_id == shot_id)
    else {
        return Err(HarnessError::Refused(format!(
            "run {} has no record for shot {shot_id}",
            session.record.run_id
        )));
    };
    // An attempt that has not settled is one this shot is still owed — from a controller that died,
    // or from a `run` that is still going. Starting a second one here would leave the first
    // orphaned, so say so and point at the command that settles it.
    if let Some(pending) = session.record.shots[index]
        .attempts
        .iter()
        .find(|attempt| !TERMINAL_ATTEMPT_STATUSES.contains(&attempt.status.as_str()))
    {
        return Err(HarnessError::Refused(format!(
            "shot {shot_id} still has attempt {} in flight (job {}); run `film-harness resume \
             --out {}` to settle it before replacing the take",
            pending.attempt,
            pending.job_id.as_deref().unwrap_or("not yet created"),
            options.out_dir.display()
        )));
    }
    // Reject whatever this shot is carrying: the selected take, or the last take it produced.
    let rejected = session.record.shots[index].selected_attempt.or_else(|| {
        session.record.shots[index]
            .attempts
            .iter()
            .rev()
            .find(|attempt| attempt.has_live_take())
            .map(|attempt| attempt.attempt)
    });
    if let Some(number) = rejected {
        if let Some(attempt) = session.record.shots[index]
            .attempts
            .iter_mut()
            .find(|attempt| attempt.attempt == number)
        {
            attempt.rejection = Some(TakeRejection {
                at: utc_now(),
                reason: reason.to_owned(),
            });
        }
    }
    session.record.shots[index].selected_attempt = None;
    // The standing decision was about the take we just rejected. Keep the append-only decision
    // entry, but do not let an acceptance of attempt N read as acceptance of replacement N+1.
    session.record.shots[index].human_decision = None;
    let previous_asset = rejected
        .and_then(|number| {
            session.record.shots[index]
                .attempts
                .iter()
                .find(|attempt| attempt.attempt == number)
        })
        .and_then(|attempt| attempt.take.as_ref())
        .map(|take| take.asset_id.clone());
    session.note_decision(
        "replace_take",
        Some(shot_id),
        match (&rejected, &previous_asset) {
            (Some(number), Some(asset)) => {
                format!("rejected attempt {number} (asset {asset}): {reason}")
            }
            (Some(number), None) => format!("rejected attempt {number}: {reason}"),
            _ => format!("no take to reject; rendering one: {reason}"),
        },
    );

    // Exactly one attempt, marked as the human decision it is so it never spends the plan's cap.
    let number = session.record.shots[index].next_attempt_number();
    let key = idempotency_key(&session.record.run_id, shot_id, number);
    let (resolved_model_id, partition_reason) = session.resolved_partition(shot_id);
    let reference_image_short_edge = session.resolved_reference_short_edge(shot_id);
    let reference_image_resize_mode = session.resolved_reference_resize_mode(shot_id);
    let (loras, effective_steps, turbo_scheduler_shift) = session.resolved_sampling(shot_id);
    session.record.shots[index].attempts.push(AttemptRecord {
        attempt: number,
        idempotency_key: key,
        resolved_model_id,
        partition_reason,
        reference_image_short_edge,
        reference_image_resize_mode,
        loras,
        effective_steps,
        turbo_scheduler_shift,
        job_id: None,
        status: "dispatching".to_owned(),
        started_at: utc_now(),
        finished_at: None,
        elapsed_seconds: 0.0,
        peak_gpu_memory_pct: None,
        peak_memory_gb: None,
        peak_memory_source: None,
        error: None,
        take: None,
        rejection: None,
        human_requested: true,
    });
    session.record.active_take_operation = Some(film_plan::TakeOperation {
        kind: kind.to_owned(),
        shot_id: shot_id.to_owned(),
        attempt: number,
        idempotency_key: idempotency_key(&session.record.run_id, shot_id, number),
        prior_outcome: prior_verdict.0,
        prior_stop: prior_verdict.1.clone(),
        reason: reason.to_owned(),
        export: options.export,
        max_shot_seconds: session.plan.limits.max_shot_seconds,
    });
    session.persist()?;
    let attempt_index = session.record.shots[index].attempts.len() - 1;
    session
        .work_attempt(&shot, index, attempt_index, true)
        .await?;

    complete_take_operation(session, index, shot_id, reason, prior_verdict).await
}

pub(super) async fn recover_take_operation(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
    continued: Continued,
    started: Instant,
) -> Result<RunRecord, HarnessError> {
    let operation = continued
        .record
        .active_take_operation
        .clone()
        .expect("checked by caller");
    let shot = continued
        .plan
        .shots
        .iter()
        .find(|shot| shot.id == operation.shot_id)
        .cloned()
        .ok_or_else(|| HarnessError::Refused("Saved take operation has no plan shot".to_owned()))?;
    let mut session = session_from(transport, options, continued, started, None);
    session.export = operation.export;
    session.plan.limits.max_shot_seconds = operation.max_shot_seconds;
    let index = session
        .record
        .shots
        .iter()
        .position(|shot| shot.shot_id == operation.shot_id)
        .ok_or_else(|| {
            HarnessError::Refused("Saved take operation has no recorded shot".to_owned())
        })?;
    let attempt_index = session.record.shots[index]
        .attempts
        .iter()
        .position(|attempt| {
            attempt.attempt == operation.attempt
                && attempt.idempotency_key == operation.idempotency_key
                && attempt.human_requested
        })
        .ok_or_else(|| {
            HarnessError::Refused(
                "Saved take operation has no matching authorized attempt".to_owned(),
            )
        })?;
    session.note_decision(
        "resume_take_operation",
        Some(&operation.shot_id),
        format!(
            "recovering {} attempt {} under its original human bounds",
            operation.kind, operation.attempt
        ),
    );
    session.persist()?;
    if !TERMINAL_ATTEMPT_STATUSES.contains(
        &session.record.shots[index].attempts[attempt_index]
            .status
            .as_str(),
    ) {
        session
            .work_attempt(&shot, index, attempt_index, false)
            .await?;
    }
    complete_take_operation(
        session,
        index,
        &operation.shot_id,
        &operation.reason,
        (operation.prior_outcome, operation.prior_stop),
    )
    .await
}

pub(super) async fn complete_take_operation(
    mut session: Session<'_>,
    index: usize,
    shot_id: &str,
    reason: &str,
    prior_verdict: (RunOutcome, Option<RunStop>),
) -> Result<RunRecord, HarnessError> {
    if session.canceled()
        && session
            .record
            .active_take_operation
            .as_ref()
            .is_some_and(|operation| {
                session.record.shots[index]
                    .attempts
                    .iter()
                    .any(|attempt| attempt.attempt == operation.attempt && attempt.job_id.is_none())
            })
    {
        session.finish(false)?;
        return Ok(session.record);
    }
    let replaced = session.record.shots[index].selected_attempt.is_some();
    session.record.shots[index].outcome = if replaced {
        ShotOutcome::Rendered
    } else {
        ShotOutcome::Failed
    };
    if replaced {
        session.flag_dependents(shot_id, reason);
        if let Some(export) = session.record.export.as_mut() {
            export.stale = true;
        }
        session.persist()?;
        // The re-assembly re-derives the harness's own dialogue and bed items from the plan and
        // needs the sound assets to place them (sc-22715 evaluation). Without this the merge
        // re-derived an EMPTY dialogue track over the saved one and every line the run had placed
        // was dropped from the sequence — the beds only survived because a bed track with no
        // asset is skipped and then kept as "not the harness's". `ensure_sound` adopts the clips
        // the record already names — including every line this run SPOKE (sc-23404) — so it
        // uploads nothing and speaks nothing on this path.
        session.ensure_sound().await?;
        // A clip that could not be re-hydrated (its file gone from the pack, a re-synthesis that
        // failed, no live TTS worker) leaves the session's map short, and re-assembling from a
        // short map is exactly the deletion the call above exists to prevent. Leave the saved
        // timeline alone and let the stop say so: the sequence keeps the sound it has, and the
        // replacement's take is still recorded.
        if session.stop.is_none() {
            // Rewriting the timeline is a PUT, not a job: every other shot's item keeps its asset.
            session.assemble_timeline().await?;
        }
    } else {
        // The shot now has NO selected take, and the timeline was deliberately not rewritten (that
        // would drop the shot out of the sequence entirely). So the timeline — and the MP4 rendered
        // from it — still carry the take the human just rejected, and the record has to say so
        // rather than leave `stale: false` claiming the export is current.
        let rejected_note = format!(
            "the replacement attempt for shot {shot_id} produced no take, so the timeline and the \
             exported MP4 still carry the REJECTED take"
        );
        if let Some(export) = session.record.export.as_mut() {
            export.stale = true;
        }
        session.note_decision("replace_take", Some(shot_id), rejected_note.clone());
        session.persist()?;
        if session.stop.is_none() {
            session.halt(
                RunOutcome::Failed,
                "replacement_failed",
                format!(
                    "{rejected_note}; the rejected take is still recorded and `film-harness \
                     replace-take --shot {shot_id}` authorises another"
                ),
                false,
            );
        }
    }
    // Not re-exporting is a deliberate choice, not a failure: the replacement succeeded, the
    // existing MP4 is marked stale, and the run is as finished as this invocation was asked to make
    // it. Only a failed replacement leaves the run failed (with the stop set above).
    //
    // `session.stop.is_none()` gates the export for the same reason it gates the re-assembly above:
    // a clip that could not be re-hydrated left the saved timeline deliberately untouched, so it is
    // the STALE timeline — the one carrying the take the human just replaced — that an export would
    // render from, and `run_export` writes `stale: false` over the `stale: true` set above,
    // recording a fresh MP4 of old material as current. Leave the MP4 that exists, keep the record
    // saying it is stale, and let the stop say why.
    let export_ok = if replaced && session.export && session.stop.is_none() {
        session.run_export().await?
    } else {
        replaced
    };
    session.record.active_take_operation = None;
    session.finish_replacement(export_ok, shot_id, prior_verdict)?;
    Ok(session.record)
}

/// Flag every shot that declared a dependency on `shot_id` AND already has a take of its own.
/// A shot that has not rendered yet needs no flag: it will be rendered against the current
/// state. Direct dependents only — a flagged shot's own take did not change, so the signal does
/// not cascade on its own. Returns how many shots were flagged.
///
/// Shared by [`replace_take`] and by the human reject path in [`review::decide_take`], so a take
/// that stops being the selected one raises exactly the same signal however it stopped.
pub(super) fn flag_dependents(
    record: &mut RunRecord,
    plan: &ProductionPlan,
    shot_id: &str,
    what_happened: &str,
) -> usize {
    let dependents: Vec<(String, String, String)> = film_plan::direct_dependents(plan, shot_id)
        .into_iter()
        .map(|(shot, edge)| (shot.id.clone(), edge.kind.clone(), edge.note.clone()))
        .collect();
    let raised_at = utc_now();
    let mut flagged = 0;
    for (dependent_id, kind, note) in dependents {
        let Some(shot_record) = record.shot_mut(&dependent_id) else {
            continue;
        };
        if shot_record.selected_attempt.is_none() {
            continue;
        }
        // One STANDING flag per (source shot, dependency kind): the same unread signal raised
        // twice is still one thing to look at. It is keyed on what is standing rather than on
        // history, so a flag a person RESOLVED (`film-harness accept-take`, sc-22714) is raised
        // again by the next change upstream — which is the whole point of resolving it.
        if shot_record
            .needs_review
            .iter()
            .any(|flag| flag.source_shot_id == shot_id && flag.dependency == kind)
        {
            continue;
        }
        let detail = if note.trim().is_empty() {
            String::new()
        } else {
            format!(" ({note})")
        };
        shot_record.needs_review.push(ReviewFlag {
            raised_at: raised_at.clone(),
            source_shot_id: shot_id.to_owned(),
            dependency: kind.clone(),
            reason: format!(
                "shot {shot_id}: {what_happened}; this shot's {kind} depends on it{detail} — \
                 review it and replace it too if it no longer matches"
            ),
        });
        flagged += 1;
    }
    flagged
}
