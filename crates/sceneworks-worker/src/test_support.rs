//! Opt-in test access to the production training execution seam.
//!
//! This changes only trainer construction. Plan validation, source resolution,
//! dataset preparation, progress, cancellation and API completion use the same
//! execution path as an ordinary worker. It does not start a worker claim loop.

use crate::{api_client::ApiClient, Settings, WorkerResult};
use sceneworks_core::contracts::{JobSnapshot, JsonObject};
use sceneworks_core::training::TrainingPlan;

mod acceptance_progress;
pub use acceptance_progress::{
    acceptance_api_cancel_controller, acceptance_progress_snapshot, observe_acceptance_trainer,
    request_acceptance_api_cancel, AcceptanceProgressLedger, AcceptanceProgressObservation,
};

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub use candle_gen_moss_image::test_support::{generation_request, Fixture};
pub use gen_core;
pub use gen_core_testkit::moss_training_probe::{EvaluationManifest, TrainingProbe};
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

/// Register through the real startup-health/discovery/registry chain. No fake
/// capability advertisement and no CPU fallback can count as GPU acceptance.
/// Call only against the caller's isolated loopback API and settings.
pub async fn register_real_acceptance_worker(
    settings: &Settings,
) -> WorkerResult<sceneworks_core::contracts::WorkerSnapshot> {
    require_acceptance_loopback(settings)?;
    let health = crate::cuda_startup_health(settings).await;
    if !health.is_usable() {
        return Err(crate::WorkerError::InvalidPayload(
            "real GPU startup health refused acceptance registration".into(),
        ));
    }
    let gpu = crate::gpu::discover_gpu(settings, &health).await;
    if gpu.id == "cpu"
        || gpu.name == format!("GPU {}", gpu.id)
        || !gpu
            .capabilities
            .contains(&sceneworks_core::contracts::WorkerCapability::Gpu)
        || !crate::worker_capabilities(&gpu)
            .contains(&sceneworks_core::contracts::WorkerCapability::LoraTrainExecute)
    {
        return Err(crate::WorkerError::InvalidPayload(
            "discovered device does not expose real GPU training execution".into(),
        ));
    }
    crate::register_worker(&ApiClient::new(settings), settings, &gpu).await
}

/// Register the separate production CPU utility identity for local adapter imports. Register
/// immediately before each import: a preceding GPU training run may outlast the idle-worker lease.
/// This does not alter the GPU worker's identity or advertise utility capabilities on it.
pub async fn register_acceptance_cpu_utility_worker(
    settings: &Settings,
) -> WorkerResult<(Settings, sceneworks_core::contracts::WorkerSnapshot)> {
    require_acceptance_loopback(settings)?;
    let mut utility = settings.clone();
    utility.worker_id = crate::gpu::cpu_worker_id(&settings.worker_id);
    utility.gpu_id = "cpu".into();
    let gpu = crate::gpu::cpu_gpu();
    let registered = crate::register_worker(&ApiClient::new(&utility), &utility, &gpu).await?;
    Ok((utility, registered))
}

fn require_acceptance_loopback(settings: &Settings) -> WorkerResult<()> {
    let url = reqwest::Url::parse(&settings.api_url)
        .map_err(|error| crate::WorkerError::InvalidPayload(error.to_string()))?;
    if settings.access_token.is_some()
        || !settings.credentials.is_empty()
        || url.scheme() != "http"
        || !matches!(url.host_str(), Some("127.0.0.1" | "::1"))
    {
        return Err(crate::WorkerError::InvalidPayload(
            "acceptance registration requires credential-free owned loopback settings".into(),
        ));
    }
    Ok(())
}

/// Full production utility dispatcher, including source guard, import/image
/// routing, terminal reconciliation and metrics. Unit return is NOT success:
/// callers must read and validate the actual API terminal job snapshot.
pub async fn run_acceptance_utility_job(settings: &Settings, job: JobSnapshot) {
    let api = ApiClient::new(settings);
    crate::run_utility_job(
        &api,
        settings,
        api.http_client(),
        job,
        gen_core::CancelFlag::new(),
    )
    .await;
}

/// Header-only forecast through the same selected spec, component/adapter pins
/// and native structural demand routines used by the production image handler.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub fn moss_generation_preload(
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<serde_json::Value> {
    crate::image_jobs::moss_generation_preload_for_test(settings, job)
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

/// Device/cap observation without trainer construction or a training memory plan.
/// Cross-load consumes ordinary generation only; its demand is checked separately
/// through the production-selected generation headers, including the actual adapter.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub fn moss_generation_acceptance_device(settings: &Settings) -> WorkerResult<serde_json::Value> {
    let (bytes, fraction) = match settings.requested_gpu_memory_limit()? {
        sceneworks_core::app_paths::GpuMemoryLimitRequest::Bytes(bytes) => {
            ((bytes != 0).then_some(bytes), None)
        }
        sceneworks_core::app_paths::GpuMemoryLimitRequest::Fraction(fraction) => {
            (None, Some(fraction))
        }
    };
    moss_acceptance_device(bytes, fraction)
        .map(|(metadata, _device)| metadata)
        .map_err(|error| crate::WorkerError::InvalidPayload(error.to_string()))
}

#[cfg(target_os = "macos")]
fn moss_acceptance_device(
    memory_limit_bytes: Option<u64>,
    memory_limit_fraction: Option<f32>,
) -> gen_core::Result<(serde_json::Value, ())> {
    let unsupported =
        |error: &dyn std::fmt::Display| gen_core::Error::Unsupported(error.to_string());
    if !cfg!(target_arch = "aarch64") {
        return Err(gen_core::Error::Unsupported(
            "MLX acceptance requires Apple Silicon".into(),
        ));
    }
    let device = mlx_rs::Device::try_default().map_err(|error| unsupported(&error))?;
    if !matches!(
        device.get_type().map_err(|error| unsupported(&error))?,
        mlx_rs::DeviceType::Gpu
    ) {
        return Err(gen_core::Error::Unsupported(
            "MLX acceptance refuses a CPU default device".into(),
        ));
    }
    let stream = mlx_rs::Stream::task_local_or_default()
        .get_index()
        .map_err(|error| unsupported(&error))?;
    let gpu_stream = mlx_rs::Stream::gpu()
        .get_index()
        .map_err(|error| unsupported(&error))?;
    if stream != gpu_stream {
        return Err(gen_core::Error::Unsupported(
            "MLX acceptance requires its original default GPU stream".into(),
        ));
    }
    let total = gen_core::moss_image_training::recovery::host::total_bytes()
        .map_err(|error| unsupported(&error))?;
    let ceiling = gen_core::train::effective_memory_budget_with_fraction(
        total,
        Some(total),
        memory_limit_bytes,
        memory_limit_fraction,
    )
    .map_err(|error| unsupported(&error))?;
    let frozen = gen_core::train::effective_memory_budget_with_fraction(
        total,
        Some(total),
        None,
        Some(0.60),
    )
    .map_err(|error| unsupported(&error))?;
    if ceiling > frozen {
        return Err(gen_core::Error::Unsupported(
            "acceptance request exceeds the frozen global MLX fraction".into(),
        ));
    }
    let metadata = serde_json::json!({
        "backend":"mlx", "deviceIdentity":format!("mlx-default-gpu/stream-{stream}"),
        "totalCapacityBytes":total, "resolvedCapBytes":ceiling,
        "resolvedCapProvenance":"product gpuMemoryLimitFraction -> actual unified total via recovery::host::total_bytes -> gen_core::train::effective_memory_budget_with_fraction",
        "weightsLoaded":false
    });
    Ok((metadata, ()))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn moss_acceptance_device(
    memory_limit_bytes: Option<u64>,
    memory_limit_fraction: Option<f32>,
) -> gen_core::Result<(serde_json::Value, candle_gen::candle_core::Device)> {
    let unsupported =
        |error: &dyn std::fmt::Display| gen_core::Error::Unsupported(error.to_string());
    let device = candle_gen::default_device().map_err(|error| unsupported(&error))?;
    if !device.is_cuda() {
        return Err(gen_core::Error::Unsupported(
            "Candle acceptance refuses a CPU/non-CUDA allocator device".into(),
        ));
    }
    let (total, identity) = {
        let cuda = device
            .as_cuda_device()
            .map_err(|error| unsupported(&error))?;
        let stream = cuda.cuda_stream();
        let context = stream.context();
        let (_, total) = context
            .mem_get_info()
            .map_err(|error| unsupported(&error))?;
        let uuid = context.uuid().map_err(|error| unsupported(&error))?;
        let mut identity = String::from("GPU-");
        use std::fmt::Write as _;
        for (index, byte) in uuid.bytes.map(|byte| byte as u8).iter().enumerate() {
            if matches!(index, 4 | 6 | 8 | 10) {
                identity.push('-');
            }
            write!(identity, "{byte:02x}").map_err(|error| unsupported(&error))?;
        }
        (
            u64::try_from(total).map_err(|error| unsupported(&error))?,
            identity,
        )
    };
    let ceiling = gen_core::train::effective_memory_budget_with_fraction(
        total,
        Some(total),
        memory_limit_bytes,
        memory_limit_fraction,
    )
    .map_err(|error| unsupported(&error))?;
    let frozen = gen_core::train::effective_memory_budget_with_fraction(
        total,
        Some(total),
        None,
        Some(0.85),
    )
    .map_err(|error| unsupported(&error))?;
    if ceiling > frozen {
        return Err(gen_core::Error::Unsupported(
            "acceptance request exceeds the frozen global CUDA fraction".into(),
        ));
    }
    let metadata = serde_json::json!({
        "backend":"cuda", "deviceIdentity":identity,
        "totalCapacityBytes":total, "resolvedCapBytes":ceiling,
        "resolvedCapProvenance":"actual allocator-selected CUDA context mem_get_info TOTAL -> product gen_core::train::effective_memory_budget_with_fraction; never host/free/max-across-devices",
        "weightsLoaded":false
    });
    Ok((metadata, device))
}

/// Official-source, header-only acceptance construction. Call this inside the
/// existing blocking loader closure: the returned Trainer is intentionally not Send.
/// This never selects a Fixture, overrides a discovered budget, loads model
/// tensors, or changes a device/stream. The native plan performs normal admission.
pub fn real_moss_trainer_with_probe(
    engine_id: &str,
    spec: &gen_core::LoadSpec,
    request: &gen_core::TrainingRequest,
    probe: TrainingProbe,
) -> gen_core::Result<(Box<dyn gen_core::Trainer>, serde_json::Value)> {
    let variant = match engine_id {
        "moss_image_base" => gen_core::moss_image::MossImageVariant::Base,
        "moss_image_sft" => gen_core::moss_image::MossImageVariant::Sft,
        _ => {
            return Err(gen_core::Error::Unsupported(
                "acceptance requires an exact MOSS Base/SFT provider".into(),
            ))
        }
    };
    let unsupported =
        |error: &dyn std::fmt::Display| gen_core::Error::Unsupported(error.to_string());
    #[cfg(target_os = "macos")]
    {
        let (mut metadata, _device) = moss_acceptance_device(
            request.config.memory_limit_bytes,
            request.config.memory_limit_fraction,
        )?;
        let mut trainer = mlx_gen_moss_image::training::MossImageTrainer::load(variant, spec)
            .map_err(|error| unsupported(&error))?;
        trainer.set_test_probe(Some(probe));
        let plan = trainer
            .test_memory_plan(request)
            .map_err(|error| unsupported(&error))?;
        metadata["nativePlan"] = serde_json::to_value(plan).map_err(|error| unsupported(&error))?;
        metadata["headerDemandProvenance"] = serde_json::json!(
            "mlx MossImageTrainer::test_memory_plan (probe attached before planning)"
        );
        metadata["observationSchema"] = moss_acceptance_observation_schema();
        Ok((Box::new(trainer), metadata))
    }
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        let (mut metadata, device) = moss_acceptance_device(
            request.config.memory_limit_bytes,
            request.config.memory_limit_fraction,
        )?;
        let mut trainer = candle_gen_moss_image::training::MossImageTrainer::load_on_device(
            variant, spec, device,
        )
        .map_err(|error| unsupported(&error))?;
        trainer.set_test_probe(Some(probe));
        let plan = trainer
            .test_memory_plan(request)
            .map_err(|error| unsupported(&error))?;
        metadata["nativePlan"] = serde_json::to_value(plan).map_err(|error| unsupported(&error))?;
        metadata["headerDemandProvenance"] = serde_json::json!(
            "cuda MossImageTrainer::test_memory_plan (probe attached before planning)"
        );
        metadata["observationSchema"] = moss_acceptance_observation_schema();
        Ok((Box::new(trainer), metadata))
    }
    #[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
    {
        let _ = (variant, spec, request, probe, unsupported);
        Err(gen_core::Error::Unsupported(
            "real MOSS acceptance requires the linked MLX or Candle/CUDA backend".into(),
        ))
    }
}

/// Type-schema inspection only, never a fabricated run or a measurement. A
/// consumer must fail closed when required native diagnostic fields are absent.
pub fn moss_acceptance_observation_schema() -> serde_json::Value {
    // TYPE-SCHEMA ONLY: Some(empty pair) survives a round trip only when the
    // linked receipt understands this optional diagnostic. Default serializes
    // None by omission; checking Default alone misidentifies a capable engine.
    // This value is never persisted or used as native execution evidence.
    let mut fields =
        serde_json::to_value(gen_core_testkit::moss_training_probe::RunReceipt::default())
            .expect("native diagnostic receipt schema serializes");
    fields["prepared_evaluation"] = serde_json::json!({"before":null,"after":null});
    let schema: gen_core_testkit::moss_training_probe::RunReceipt =
        serde_json::from_value(fields).expect("native diagnostic receipt type-schema round trip");
    let fields = serde_json::to_value(schema).expect("native diagnostic type-schema serializes");
    serde_json::json!({
        "gradientObservations":fields.get("gradient_tensors").is_some(),
        "fixedEvaluationPair":fields.get("evaluation_before").is_some() && fields.get("evaluation_after").is_some(),
        "numericFactorDelta":fields.get("final_parameter_delta").is_some(),
        "preparedTensorSnapshots":fields.get("prepared_evaluation").is_some_and(|pair| pair.is_object()),
        "schemaOnlyNotRunEvidence":true
    })
}
