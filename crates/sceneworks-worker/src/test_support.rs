//! Opt-in test access to the production training execution seam.
//!
//! This changes only trainer construction. Plan validation, source resolution,
//! dataset preparation, progress, cancellation and API completion use the same
//! execution path as an ordinary worker. It does not start a worker claim loop.

use crate::{api_client::ApiClient, Settings, WorkerResult};
use sceneworks_core::contracts::{JobSnapshot, JsonObject};
use sceneworks_core::training::TrainingPlan;

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub use candle_gen_moss_image::test_support::{generation_request, Fixture};
pub use gen_core;
pub use gen_core_testkit::moss_training_probe::TrainingProbe;
#[cfg(target_os = "macos")]
pub use mlx_gen_moss_image::test_support::{generation_request, Fixture};

/// Exact production adapter selection with its source tokens retained for the
/// full fixture load/generate scope. No parallel path, kind or scale mapper.
pub struct PreparedAdapterSelection {
    specs: Vec<gen_core::AdapterSpec>,
    pins: Vec<gen_core::PinnedWeightsFile>,
}

impl PreparedAdapterSelection {
    pub fn specs(&self) -> &[gen_core::AdapterSpec] {
        &self.specs
    }

    pub fn ensure_unchanged(&self) -> WorkerResult<()> {
        for pin in &self.pins {
            pin.ensure_unchanged().map_err(|error| {
                crate::WorkerError::InvalidPayload(format!(
                    "Prepared adapter source changed: {error}"
                ))
            })?;
        }
        Ok(())
    }
}

pub fn prepare_selected_adapters(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<PreparedAdapterSelection> {
    let request = sceneworks_core::image_request::ImageRequest::from_payload(payload);
    let (specs, pins) = crate::image_jobs::prepared_adapter_selection_for_test(&request, settings)?;
    Ok(PreparedAdapterSelection { specs, pins })
}

pub async fn run_training_execution_using(
    settings: &Settings,
    job: &JobSnapshot,
    plan: &TrainingPlan,
    load_trainer: impl FnOnce(&str, &gen_core::LoadSpec) -> gen_core::Result<Box<dyn gen_core::Trainer>>
        + Send
        + 'static,
) -> WorkerResult<()> {
    let api = ApiClient::new(settings);
    crate::training_jobs::run_training_execution_using(&api, settings, job, plan, load_trainer)
        .await
}

/// Observe the original request token for causal cancellation assertions. The
/// observer must not cancel, clear or replace it; cancellation still comes from
/// the ordinary API poll and the production event consumer.
pub async fn run_training_execution_observing_cancel(
    settings: &Settings,
    job: &JobSnapshot,
    plan: &TrainingPlan,
    load_trainer: impl FnOnce(&str, &gen_core::LoadSpec) -> gen_core::Result<Box<dyn gen_core::Trainer>>
        + Send
        + 'static,
    observe_cancel: impl FnOnce(&gen_core::CancelFlag) + Send + 'static,
) -> WorkerResult<()> {
    run_training_execution_observing_request(settings, job, plan, load_trainer, move |request| {
        observe_cancel(&request.cancel)
    })
    .await
}

/// Inspect the same prepared native request before loading. CPU fixture callers
/// can bind header-derived demand to the real API dataset/config without a second
/// request mapper. Production passes no observer and no request is altered here.
pub async fn run_training_execution_observing_request(
    settings: &Settings,
    job: &JobSnapshot,
    plan: &TrainingPlan,
    load_trainer: impl FnOnce(&str, &gen_core::LoadSpec) -> gen_core::Result<Box<dyn gen_core::Trainer>>
        + Send
        + 'static,
    observe_request: impl FnOnce(&gen_core::TrainingRequest) + Send + 'static,
) -> WorkerResult<()> {
    let api = ApiClient::new(settings);
    crate::training_jobs::run_training_execution_drive_observing_cancel(
        &api,
        settings,
        job,
        plan,
        load_trainer,
        std::sync::Arc::new(crate::iris_dist::CurrentExeLauncher),
        observe_request,
    )
    .await
}
