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
        DEFAULT_MIN_EDGE,
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
        // confined (the executor re-resolves item paths under this root) and cleaned with the cache.
        let work_dir = settings
            .data_dir
            .join("cache")
            .join("control-datasets")
            .join(&job.id);
        tokio::fs::create_dir_all(&work_dir).await?;

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

        // A2 prep is blocking (decode + neural preprocess + PNG encode per image); run it off the
        // async runtime. Ownership moves into the closure; the report + work dir come back out.
        let config = ControlPrepConfig {
            kind,
            resources,
            min_edge: DEFAULT_MIN_EDGE,
            person_filter,
        };
        let prep_dir = work_dir.clone();
        let report = tokio::task::spawn_blocking(move || {
            prepare_control_dataset(&inputs, &config, &prep_dir, |_done, _total| {})
        })
        .await
        .map_err(|error| WorkerError::Engine(format!("control prep task panicked: {error}")))??;

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
        use super::{control_kind_from_label, ControlKind};

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
