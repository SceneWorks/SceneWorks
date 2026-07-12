//! ControlNet Training Studio orchestration job (Training Studio B1, sc-10162, epic 10159).
//!
//! One job that turns "an existing captioned training dataset + a control type" into a trained
//! control branch. It chains the A1/A2 data-prep (render the per-image control condition) into the
//! SAME native-training executor the LoRA path uses ([`crate::training_jobs::run_training_execution`]):
//!
//! 1. Parse the resolved [`TrainingPlan`] the API stamped into the payload (kernel `krea_control`).
//! 2. Render the control condition for every dataset item via the A1 preprocessor registry / A2
//!    folder-ingest core ([`crate::control_dataset_prep::prepare_control_dataset`]) — pose skeleton
//!    (DWPose) or canny — into an app-managed control dataset (square-canonical `(target, control,
//!    caption)` triples + `manifest.jsonl`).
//! 3. Rewrite the plan's dataset onto that rendered control dataset, threading each item's rendered
//!    condition onto [`TrainingPlanItem::control_image_path`].
//! 4. Hand the rewritten plan to `run_training_execution` → `gen_core::load_trainer("krea_2_control")`
//!    → the candle-gen-krea `ControlTrainer` writes the overlay.
//!
//! Backend gating mirrors [`crate::training_jobs`]: the real orchestration exists only on the
//! neural-inference builds (macOS, or an off-Mac `backend-candle` build). Today only the candle lane
//! has a control trainer (`krea_2_control`), and routing (`training_job_is_candle_eligible`) keeps a
//! `control_training` job on a candle worker; on a build with no native control trainer the stub at
//! the bottom fails loudly. Bring-your-own prepared/COCO datasets (A3,
//! [`crate::control_dataset_byo`]) are a separate source wired by a follow-up (see the module decl in
//! `lib.rs`); B1 ships the render-from-an-existing-dataset path.

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod imp {
    use super::super::*;

    use gen_core::ControlKind;
    use sceneworks_core::training::{TrainingPlan, TrainingPlanItem};

    use crate::control_dataset_prep::{
        prepare_control_dataset, ControlPrepConfig, ControlPrepInput, PersonFilter,
        DEFAULT_MIN_EDGE, PREP_CANCEL_MESSAGE,
    };
    use crate::control_preprocess::{control_kind_label, PreprocessResources};
    use crate::downloads::DownloadContext;
    use crate::paths::resolve_dataset_item_path;
    use crate::training_jobs::{parse_plan, run_training_execution, training_progress};

    /// The only kernel the studio job trains today — the Krea 2 pose-ControlNet branch. The API stamps
    /// it into the plan (`krea_control` target); a plan with any other kernel is rejected here rather
    /// than mis-dispatched (the executor maps `krea_control` → the `krea_2_control` engine trainer).
    const CONTROL_KERNEL: &str = "krea_control";

    /// YOLO person-gate confidence floor when `advanced.personGate` is on — drop a target with no
    /// detected person so a pose branch isn't trained on empty skeletons. Deliberately permissive
    /// (Dataset Doctor runs the richer quality pass).
    const PERSON_GATE_MIN_CONF: f32 = 0.25;

    /// RAII guard for the job-scoped rendered-control-dataset work dir (sc-11186, F-015). The rendered
    /// dataset under `cache/control-datasets/<job.id>` is a full letterboxed copy of the training corpus
    /// — a `.target.png` + `.pose.png` pair per image, easily GBs — and it is consumed only by the
    /// training executor that runs to completion inside [`run_control_training_job`]. Nothing else in
    /// the repo sweeps that tree, so before this guard every `control_training` run leaked the whole
    /// rendered corpus onto disk (success, failure, AND cancel), silently growing the app data dir.
    /// `Drop` removes the tree on EVERY exit path (success, prep/train error, deferred cancel, panic
    /// unwind), so no early return can skip cleanup. Best-effort: a removal failure is logged, never
    /// fatal (a cleanup error must not fail an otherwise-complete job). `Drop` can't be async, so it
    /// uses the sync `std::fs` API. The dir is keyed by `job.id`, so this only ever removes the current
    /// job's own tree — never a concurrent sibling worker's active dataset.
    struct WorkDirGuard {
        path: std::path::PathBuf,
    }

    impl WorkDirGuard {
        fn new(path: std::path::PathBuf) -> Self {
            Self { path }
        }
    }

    impl Drop for WorkDirGuard {
        fn drop(&mut self) {
            match std::fs::remove_dir_all(&self.path) {
                Ok(()) => {}
                // A missing dir is a benign NotFound (nothing was rendered before an early bail) —
                // silently ignore it. Anything else is a real cleanup failure worth surfacing, but the
                // job's outcome is already decided, so we only warn.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    tracing::warn!(
                        event = "control_dataset_cleanup_failed",
                        work_dir = %self.path.display(),
                        %error,
                        "failed to remove the rendered control-dataset work dir; it may leak on disk"
                    );
                }
            }
        }
    }

    /// Run a `control_training` job: render conditions, then train the control branch.
    pub(crate) async fn run_control_training_job(
        api: &ApiClient,
        settings: &Settings,
        http_client: &reqwest::Client,
        job: &JobSnapshot,
    ) -> WorkerResult<()> {
        let backend = backend_label(&settings.gpu_id);
        let mut plan = parse_plan(&job.payload)?;
        if plan.target.kernel != CONTROL_KERNEL {
            return Err(WorkerError::InvalidPayload(format!(
                "control_training expects a '{CONTROL_KERNEL}' plan, got kernel '{}'.",
                plan.target.kernel
            )));
        }
        if plan.dataset.items.is_empty() {
            return Err(WorkerError::InvalidPayload(
                "control_training plan has no dataset items to render conditions from.".to_owned(),
            ));
        }

        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
        update_job(
            api,
            &job.id,
            training_progress(
                JobStatus::Preparing,
                ProgressStage::Preparing,
                0.02,
                "Preparing ControlNet training.",
                None,
                backend,
            ),
        )
        .await?;
        check_cancel(
            api,
            &job.id,
            "ControlNet training canceled before it started.",
        )
        .await?;

        let kind = control_kind_from_plan(&plan);
        let label = control_kind_label(&kind);

        // Resolve the preprocessor resources this control type needs (pose → DWPose weights; canny is
        // weightless). Depth is not yet offered by the studio target (Phase C), so reject it clearly
        // rather than silently produce nothing.
        let mut resources = PreprocessResources::new();
        match kind {
            ControlKind::Pose => {
                resources.dwpose =
                    Some(pose_jobs::ensure_dwpose_weights(api, settings, http_client, job).await?);
            }
            ControlKind::Canny => {}
            other => {
                return Err(WorkerError::InvalidPayload(format!(
                    "control type '{}' is not available in the training studio yet (pose and canny \
                     only); depth/other are a Phase-C follow-up.",
                    control_kind_label(&other)
                )));
            }
        }

        // Optional person-presence gate (pose/people corpora), resolved through the shared YOLO11
        // detector the person-detect job uses.
        let person_filter = if plan_advanced_bool(&plan, "personGate") {
            let context = DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "ControlNet training canceled while fetching the person detector.",
                fresh_download: false,
            };
            let weights_path = person_jobs::ensure_detector_weights(settings, &context).await?;
            Some(PersonFilter {
                weights_path,
                min_conf: PERSON_GATE_MIN_CONF,
            })
        } else {
            None
        };

        // Build the render inputs from the plan's dataset items — the (already trigger-composed)
        // caption rides straight through; the target image is re-confined under the source dataset
        // root before it is read.
        let source_root = plan.dataset.root_path.clone();
        let inputs = plan
            .dataset
            .items
            .iter()
            .map(|item| {
                Ok(ControlPrepInput {
                    target_path: resolve_dataset_item_path(
                        settings,
                        &source_root,
                        &item.image_path,
                        "ControlNet training source imagePath",
                    )?,
                    caption: item.caption.clone(),
                })
            })
            .collect::<WorkerResult<Vec<_>>>()?;
        let input_count = inputs.len();

        // The rendered control dataset lands in an app-managed cache dir keyed by job id, so it is
        // confined (the executor re-resolves item paths under this root).
        let work_dir = settings
            .data_dir
            .join("cache")
            .join("control-datasets")
            .join(&job.id);
        tokio::fs::create_dir_all(&work_dir).await?;
        // Guard the job-scoped tree so it is removed on EVERY exit path below — success, prep/train
        // error, deferred cancel, or panic (sc-11186, F-015). Nothing else sweeps
        // `cache/control-datasets/<job.id>`, so without this the rendered corpus leaked on disk.
        let _work_dir_guard = WorkDirGuard::new(work_dir.clone());

        update_job(
            api,
            &job.id,
            training_progress(
                JobStatus::Preparing,
                ProgressStage::Preparing,
                0.05,
                &format!("Rendering {label} conditions for {input_count} image(s)."),
                None,
                backend,
            ),
        )
        .await?;

        // A2 prep is a multi-minute blocking phase (per-image decode + neural preprocess + PNG encode);
        // run it off the async runtime under the same heartbeat + cancel + per-item progress scaffold
        // the batched analysis jobs use (`analysis_jobs_common`, sc-8836), so the API's ~90s stale-sweep
        // can't false-interrupt a moderately sized dataset mid-render (sc-11155). The keepalive is the
        // `heartbeat` ping on the interval tick below — posting job progress does NOT refresh the
        // worker's `last_seen` (only the heartbeat does), so the loop must tick even between per-item
        // progress posts. The prep loop is worker code, so it takes a REAL `CancelFlag` (sc-9123) that
        // `prepare_control_dataset` polls between items and bails with the typed `Canceled`; any pairs
        // already written sit in the job-scoped cache dir and are inert. `on_progress` is streamed over
        // `tx` (previously discarded) so the studio job shows render progress.
        let config = ControlPrepConfig {
            kind,
            resources,
            min_edge: DEFAULT_MIN_EDGE,
            person_filter,
        };
        let prep_dir = work_dir.clone();
        let cancel = gen_core::CancelFlag::new();
        let task_cancel = cancel.clone();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<(usize, usize)>(64);
        let blocking = tokio::task::spawn_blocking(move || {
            prepare_control_dataset(&inputs, &config, &prep_dir, &task_cancel, |done, total| {
                // A closed channel means the consumer loop returned early (POST failure / 409 stale-
                // sweep reclaim); trip the cancel flag so the prep bails at its next per-item check
                // instead of rendering the whole corpus to its natural end (sc-8804, F-003 — the
                // swallowed-closed-channel leak).
                if tx.blocking_send((done, total)).is_err() {
                    task_cancel.cancel();
                }
            })
        });

        // Bind the blocking prep to its cancel flag (sc-8804, F-003): every `update_job`/`heartbeat`
        // `?` below returns early on a transient POST failure or a 409 reclaim; on that early return the
        // guard trips `cancel` and bounded-joins the prep thread instead of dropping-and-running it.
        let mut guard = CancelJoinGuard::new(cancel.clone(), blocking);
        let mut interval = tokio::time::interval(progress_report_interval(settings));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Defer the terminal `Canceled`: a cancel poll only trips the engine flag (so the prep stops at
        // its next item boundary) and marks `canceled`; the terminal write happens AFTER the task stops,
        // so the worker row isn't freed while the render is still on the GPU (sc-8917, mirrors
        // `consume_training_events` / `run_batched_analysis_job`).
        let mut canceled = false;
        let loop_result: WorkerResult<()> = async {
            loop {
                tokio::select! {
                    event = rx.recv() => {
                        match event {
                            Some((done, total)) => {
                                // Per-item render progress scaled into the prep band 0.05..0.20 (the
                                // plan rewrite + training take the rest). This is a UI signal only; the
                                // stale-sweep keepalive is the heartbeat on the interval tick.
                                let frac = done as f64 / total.max(1) as f64;
                                update_job(
                                    api,
                                    &job.id,
                                    training_progress(
                                        JobStatus::Preparing,
                                        ProgressStage::Preparing,
                                        0.05 + 0.15 * frac,
                                        &format!(
                                            "Rendering {label} conditions ({done}/{total})."
                                        ),
                                        None,
                                        backend,
                                    ),
                                )
                                .await?;
                            }
                            None => break,
                        }
                    }
                    _ = interval.tick() => {
                        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                        // sc-9618: a process shutdown is a cancel checkpoint too — short-circuit the API
                        // poll (a local flag read) so a quit stops the render at its next item check.
                        if !canceled
                            && (shutdown_requested() || cancel_requested_peek(api, &job.id).await)
                        {
                            cancel.cancel();
                            canceled = true;
                        }
                    }
                }
            }
            Ok(())
        }
        .await;
        if let Err(error) = loop_result {
            guard.cancel_and_join().await;
            return Err(error);
        }

        // Loop exited cleanly (channel closed) — reclaim the handle (disarming the drop-guard) and join.
        let join_result = guard
            .into_handle()
            .await
            .map_err(|error| task_join_error("control prep task", error))?;
        if canceled {
            // The prep has actually stopped now, so post the TERMINAL `Canceled` here.
            mark_job_canceled(api, &job.id, PREP_CANCEL_MESSAGE).await?;
            return Err(WorkerError::Canceled(PREP_CANCEL_MESSAGE.to_owned()));
        }
        let report = join_result?;

        // Surface the per-image skip tally rather than silently shrinking the corpus (the report
        // exists for exactly this). The manifest path + the individual reasons go to the log; the
        // studio job's user-facing progress carries the counts below.
        if !report.skipped.is_empty() {
            let detail = report
                .skipped
                .iter()
                .map(|(path, reason)| format!("{}: {}", path.display(), reason.describe()))
                .collect::<Vec<_>>()
                .join("; ");
            tracing::info!(
                event = "control_prep_skipped",
                job_id = %job.id,
                skipped = report.skipped.len(),
                total = input_count,
                %detail,
                "some source images were skipped while rendering control conditions"
            );
        }
        tracing::debug!(
            job_id = %job.id,
            manifest = %report.manifest_path.display(),
            rendered = report.items.len(),
            "wrote control training dataset"
        );

        if report.items.is_empty() {
            return Err(WorkerError::InvalidPayload(format!(
                "No usable training pairs after rendering conditions — all {input_count} source \
                 image(s) were skipped (undecodable, too small, no person, or preprocess failure)."
            )));
        }

        // Rewrite the plan's dataset onto the rendered control dataset: point the root at the work
        // dir and thread each item's rendered condition onto `control_image_path`. The (letterboxed,
        // square) rendered target replaces the source image so target + control share geometry.
        plan.dataset.root_path = work_dir.display().to_string();
        plan.dataset.items = report
            .items
            .iter()
            .map(|item| TrainingPlanItem {
                image_path: item.target_rel.clone(),
                caption: item.caption.clone(),
                control_image_path: Some(item.control_rel.clone()),
                width: None,
                height: None,
                extra: Default::default(),
            })
            .collect();

        let skipped = report.skipped.len();
        update_job(
            api,
            &job.id,
            training_progress(
                JobStatus::Preparing,
                ProgressStage::Preparing,
                0.1,
                &format!(
                    "Rendered {} {label} condition(s){}. Starting control-branch training.",
                    report.items.len(),
                    if skipped > 0 {
                        format!(" ({skipped} source image(s) skipped)")
                    } else {
                        String::new()
                    }
                ),
                None,
                backend,
            ),
        )
        .await?;

        // Hand off to the shared native-training executor (progress streaming, cancellation, overlay
        // write + result post-back). It maps `krea_control` → the `krea_2_control` engine trainer.
        run_training_execution(api, settings, job, &plan).await
    }

    /// The control type to render, from the plan config's `advanced.controlType` (default pose — the
    /// only type the `krea_control` target advertises today).
    fn control_kind_from_plan(plan: &TrainingPlan) -> ControlKind {
        control_kind_from_label(
            plan.config
                .advanced
                .get("controlType")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .unwrap_or("pose"),
        )
    }

    /// Map a `controlType` label onto a [`ControlKind`]. Absent/blank defaults to pose (the studio
    /// target's default); an unknown label flows through as [`ControlKind::Other`] and is rejected by
    /// the caller with a precise "not available yet" message rather than silently mis-rendered.
    fn control_kind_from_label(label: &str) -> ControlKind {
        match label {
            "" | "pose" => ControlKind::Pose,
            "canny" => ControlKind::Canny,
            "depth" => ControlKind::Depth,
            other => ControlKind::Other(other.to_owned()),
        }
    }

    /// Read a boolean flag from the plan config's `advanced` bag (absent/non-bool ⇒ false).
    fn plan_advanced_bool(plan: &TrainingPlan, key: &str) -> bool {
        plan.config
            .advanced
            .get(key)
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
    }

    #[cfg(test)]
    mod tests {
        use super::{control_kind_from_label, ControlKind, WorkDirGuard};

        /// A unique temp path per test (process id + line) so parallel/leftover runs don't collide.
        fn unique_temp(tag: &str, line: u32) -> std::path::PathBuf {
            std::env::temp_dir().join(format!(
                "sw_control_work_{tag}_{}_{line}",
                std::process::id()
            ))
        }

        #[test]
        fn work_dir_guard_removes_populated_tree_on_drop() {
            // A rendered control dataset is a nested tree of PNG pairs + a manifest — the guard must
            // remove the WHOLE subtree, not just the top dir, on drop (sc-11186, F-015).
            let root = unique_temp("drop", line!());
            let nested = root.join("images");
            std::fs::create_dir_all(&nested).expect("create nested work dir");
            std::fs::write(nested.join("0001.target.png"), b"target").expect("write target");
            std::fs::write(nested.join("0001.pose.png"), b"pose").expect("write pose");
            std::fs::write(root.join("manifest.jsonl"), b"{}\n").expect("write manifest");
            assert!(root.exists(), "precondition: work dir exists before drop");

            {
                let _guard = WorkDirGuard::new(root.clone());
                // Still present inside the guarded scope — cleanup happens on drop, not construction.
                assert!(root.exists(), "guard construction must not remove the dir");
            }

            assert!(
                !root.exists(),
                "guard Drop must remove the entire rendered-dataset tree"
            );
        }

        #[test]
        fn work_dir_guard_drop_on_missing_dir_is_a_noop() {
            // An early bail before anything is rendered leaves no dir; Drop must treat the resulting
            // NotFound as benign (best-effort cleanup) and NOT panic.
            let root = unique_temp("missing", line!());
            assert!(!root.exists(), "precondition: dir was never created");
            // Would panic here if Drop unwrapped the io::Error instead of tolerating NotFound.
            drop(WorkDirGuard::new(root.clone()));
            assert!(
                !root.exists(),
                "a missing dir stays absent, no side effects"
            );
        }

        #[test]
        fn control_kind_label_maps_studio_types_and_defaults_to_pose() {
            // Absent/blank ⇒ pose (the krea_control target's default).
            assert_eq!(control_kind_from_label(""), ControlKind::Pose);
            assert_eq!(control_kind_from_label("pose"), ControlKind::Pose);
            assert_eq!(control_kind_from_label("canny"), ControlKind::Canny);
            assert_eq!(control_kind_from_label("depth"), ControlKind::Depth);
            // An unknown label is preserved as Other so the caller can reject it precisely (never
            // silently coerced to pose).
            assert_eq!(
                control_kind_from_label("scribble"),
                ControlKind::Other("scribble".to_owned())
            );
        }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) use imp::run_control_training_job;

/// Stub for a build with no native control trainer (off-Mac without `backend-candle`). The routing
/// gate (`training_job_is_candle_eligible`) keeps a `control_training` job off such a worker, so this
/// is a loud backstop rather than a reachable path.
#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
pub(crate) async fn run_control_training_job(
    _api: &crate::ApiClient,
    _settings: &crate::Settings,
    _http_client: &reqwest::Client,
    job: &sceneworks_core::contracts::JobSnapshot,
) -> crate::WorkerResult<()> {
    Err(crate::WorkerError::InvalidPayload(format!(
        "control_training job {} requires a native control trainer (macOS mlx or off-Mac \
         backend-candle); this worker has neither.",
        job.id
    )))
}
