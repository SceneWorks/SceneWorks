use super::adapters::resolve_adapters;
use super::artifact_resolution::model_repo;
use super::artifact_resolution::resolve_weights_dir;
use super::artifact_resolution::resolved_mlx_artifact_tier;
use super::conditioning::parse_poses;
use super::conditioning::pose_entries;
use super::conditioning::resolve_control_identity_source;
use super::conditioning::stage_likeness;
use super::events::consume_gen_events;
use super::load_policy::release_declared_sequential_for_request;
#[cfg(test)]
use super::load_spec::load_control_engine;
use super::mlx_model;
use super::quantization::resolve_quant;
use super::resolve_seed;
use super::routing::mlx_flux_strict_control_engine_id;
use super::sampling::resolve_guidance;
use super::sampling::resolve_steps;
use super::stream::drive_gen_items_scored;
use super::stream::start_cached_gen_stream_with_request_state_using_prepared;
use super::strict_control::build_control_conditioning;
use super::strict_control::ensure_depth_estimator_dir;
use super::strict_control::preprocess_control_entry;
use super::strict_control::requested_control_kind;
use super::strict_control::resolve_control_source;
use super::strict_control::resolve_user_control_map;
use super::strict_control::strict_control_default_repo;
use super::strict_control::trusted_control_weight_revision;
use super::strict_control::validate_control_kind;
use super::AdapterSpec;
use super::CancelFlag;
use super::Conditioning;
use super::ControlKind;
use super::GenerationOutput;
use super::GenerationRequest;
use super::Generator;
use super::Image;
use super::ImagePlan;
use super::ImageRequest;
use super::LoadSpec;
use super::Progress;
use super::Quant;
use super::WeightsSource;
use crate::advanced;
use crate::safe_weight_filename;
use crate::ApiClient;
use crate::JobSnapshot;
use crate::JsonObject;
use crate::Path;
use crate::PathBuf;
use crate::Settings;
use crate::Value;
use crate::WorkerError;
use crate::WorkerResult;
use serde_json::json;
// ---------------------------------------------------------------------------
// FLUX.1-dev strict-control Fun-Controlnet-Union (macOS, sc-8244; engine E2 sc-8239):
// the `flux1_dev_control` registry generator — the Shakker
// `FLUX.1-dev-ControlNet-Union-Pro-2.0` branch on the FLUX.1-dev base. One image per
// library pose (pose tier) — or, with `advanced.controlMode = canny|depth` + an input
// image, a canny edge / Depth-Anything-V2 control map auto-derived from that image — fed
// to the union ControlNet (TRUE structural lock, not the best-effort `[skeleton, reference]`
// edit tier). FLUX.1-dev on macOS routes here only for a strict-control job; plain txt2img
// + IP-Adapter keep their existing paths. Mirrors `generate_flux2_dev_control_stream`.
// ---------------------------------------------------------------------------

/// The engine registry id for the FLUX.1-dev Fun-Controlnet-Union variant (E2 sc-8239).
pub(super) const FLUX1_DEV_CONTROL_ENGINE_ID: &str = "flux1_dev_control";
/// The Shakker Union-Pro-2.0 control-weights filename — the single `.safetensors` shipped in the repo
/// (the diffusers checkpoint). The default *repo* is the shared strict-control table (single source of
/// truth — `STRICT_CONTROL_ENGINES`).
pub(super) const FLUX1_CONTROL_FILE: &str = "diffusion_pytorch_model.safetensors";
/// Pinned revision for the default `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` control-weights
/// repo (sc-9879, F-077 follow-up). Fetching the mutable `main` branch means a re-push (or a compromised
/// token) could silently swap the ControlNet checkpoint we load; pin the exact commit for defense-in-depth
/// (mirrors the SeedVR2/Real-ESRGAN pins, sc-8879/sc-9682). Registered overlays carry their own
/// catalog-authorized immutable revision. HF's tree API still reports the file's `lfs.oid`, which
/// `ensure_hf_cached_file` verifies against.
#[cfg(test)]
pub(super) const FLUX1_CONTROL_REVISION: &str = "5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5";
/// The asset `adapter` id recorded on FLUX.1-dev strict-control assets (the dev base MLX label —
/// shared with the plain FLUX.1 path).
const FLUX1_CONTROL_ADAPTER_LABEL: &str = "mlx_flux";

/// True when this is a FLUX.1-dev strict-control job (`flux_dev` + ≥1 pose, not edit mode) whose base
/// weights resolve — routed to the Shakker Fun-Controlnet-Union path rather than plain txt2img or the
/// IP-Adapter reference tier. Gated to `flux_dev` (schnell has no control checkpoint). Control-weights
/// presence is NOT part of the gate: they are fetched on first use in the stream (a missing checkpoint
/// downloads, then errors loudly only on a real failure — never silently drops the poses). The pose
/// entry is the structural hint carrier for every kind (pose renders a skeleton; canny/depth pair the
/// pose with `advanced.controlMode` + an input image — the pose set still drives the per-image loop).
pub(super) fn flux1_dev_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    mlx_flux_strict_control_engine_id(&request.model) == Some(FLUX1_DEV_CONTROL_ENGINE_ID)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// The (repo, filename) of the FLUX.1-dev control weights — `advanced.controlWeights.{repo,filename}`
/// overrides, else the Shakker Union-Pro-2.0 default (parity with the FLUX.2 / Z-Image resolvers).
/// The payload filename must be a plain component (sc-8821 / F-019).
pub(super) fn flux1_control_repo_file(request: &ImageRequest) -> WorkerResult<(String, String)> {
    let cw = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object);
    let pick = |key: &str, default: &str| {
        cw.and_then(|m| m.get(key))
            .and_then(crate::payload::trimmed_value_str)
            .unwrap_or(default)
            .to_owned()
    };
    let repo = pick(
        "repo",
        strict_control_default_repo(FLUX1_DEV_CONTROL_ENGINE_ID),
    );
    let file = safe_weight_filename(
        &pick("filename", FLUX1_CONTROL_FILE),
        "advanced.controlWeights.filename",
    )?;
    trusted_control_weight_revision(request, FLUX1_DEV_CONTROL_ENGINE_ID, &repo, &file)?;
    Ok((
        // Default repo from the shared strict-control table (single source of truth); the file stays
        // engine-specific.
        repo, file,
    ))
}

/// Resolve the Shakker Union-Pro-2.0 checkpoint the engine loads, downloading on first use. Order: an
/// env-pinned file (`SCENEWORKS_CONTROLNET_FLUX1`) → a whole-repo HF cache snapshot → download into the
/// app cache. Mirrors [`ensure_flux2_control_weights`]. The control checkpoint is lazy-fetched only on
/// the first strict-control job (vs bloating the base download).
async fn ensure_flux1_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file) = flux1_control_repo_file(request)?;
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_FLUX1") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
    let revision =
        trusted_control_weight_revision(request, FLUX1_DEV_CONTROL_ENGINE_ID, &repo, &file)?;
    if let Some(snapshot) =
        crate::model_jobs::huggingface_pinned_snapshot_dir(&settings.data_dir, &repo, &revision)
    {
        let f = snapshot.join(&file);
        if f.is_file() {
            return Ok(f);
        }
    }
    let client = crate::downloads::streaming_download_client();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message:
            "FLUX.1-dev strict-control generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-flux1")
        .join(&file);
    // Pin the exact commit for the default table control repo so `main` moving under us can't swap the
    // ControlNet checkpoint (sc-9879). Registered overlays carry their own immutable pin.
    crate::downloads::ensure_hf_cached_file(&context, &repo, &revision, &file, &dst).await
}

/// Control lock strength for FLUX.1-dev: `advanced.controlScale` (default 0.7, clamp [0,2]). The Shakker
/// Union-Pro-2.0 README recommends ~0.7 (the engine default too), so the worker default matches.
fn flux1_control_scale(request: &ImageRequest) -> f32 {
    advanced::f32_clamped(&request.advanced, "controlScale", 0.7, 0.0..=2.0)
}

/// The FLUX.1-dev strict-control request for one pose: the pre-built `conditioning` (the required
/// `Control` / `Depth`, assembled by the shared [`build_control_conditioning`] driver) drives the
/// Shakker Union-Pro-2.0 branch. dev is guidance-distilled (embedded scalar) — `guidance` rides the
/// transformer's guidance embedder (no true-CFG). FLUX.1 control has no img2img-init seam (unlike
/// FLUX.2), so no identity `Reference` is appended. The pre-load planner and the render both build
/// their request here, so a plan is compiled from the request it runs.
#[allow(clippy::too_many_arguments)]
fn flux1_control_request(
    sampling: &GenerationRequest,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    conditioning: Vec<Conditioning>,
    memory: Option<gen_core::GenerationMemory>,
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        conditioning,
        memory,
        preview,
        cancel: cancel.clone(),
        sampler: sampling.sampler.clone(),
        scheduler: sampling.scheduler.clone(),
        scheduler_shift: sampling.scheduler_shift,
        ..Default::default()
    }
}

/// Render one FLUX.1-dev strict-control image from the request [`flux1_control_request`] built and
/// the render plan compiled for it before load (SW-6936), inside the admitted memory scope.
pub(super) fn flux1_control_generate_one(
    generator: &dyn Generator,
    mut request: GenerationRequest,
    plan: &gen_core::RenderPlan,
    memory_evaluation: Option<&crate::mlx_fit_gate::MlxRequestEvaluation>,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let report = crate::memory_strategy::generate_with_scope_report_planned(
        generator,
        &mut request,
        plan,
        memory_evaluation.map(|evaluation| &evaluation.context),
        on_progress,
    )
    .map_err(|error| {
        WorkerError::Engine(format!("FLUX.1-dev control generation failed: {error}"))
    })?;
    match report.output {
        Some(GenerationOutput::Images(mut images)) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("FLUX.1-dev control generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "FLUX.1-dev control generator returned non-image output".to_owned(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn flux1_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(FLUX1_DEV_CONTROL_ENGINE_ID.to_owned()),
    );
    raw
}

/// Build the FLUX.1-dev control LoadSpec: the base dev snapshot + the Shakker Union-Pro-2.0 overlay
/// (+ quant + adapters). The dev base + control overlay load dense bf16 and quantize in place under
/// `with_quant` (the FLUX.1 control loader only wires dense bf16 + spec.quantize).
fn flux1_control_spec(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::File(control_weights));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    spec
}

/// Real-weight test seam: the loaded control generator and the load spec its render plans bind to.
#[cfg(all(target_os = "macos", test))]
pub(super) fn flux1_control_load(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<(Box<dyn Generator>, LoadSpec)> {
    let spec = flux1_control_spec(weights_dir, control_weights, quant, adapters);
    Ok((
        load_control_engine(FLUX1_DEV_CONTROL_ENGINE_ID, &spec)?,
        spec,
    ))
}

/// Real-weight test seam: render one image through the worker's planned control path (compile the
/// plan, then [`flux1_control_generate_one`]). An empty `conditioning` is the control-free baseline
/// the matrix tests measure steering against; no worker route renders it, and the control
/// provider has no plan for it, so it calls the generator directly.
#[cfg(all(target_os = "macos", test))]
#[allow(clippy::too_many_arguments)]
pub(super) fn flux1_control_render_for_test(
    generator: &dyn Generator,
    spec: &LoadSpec,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    conditioning: Vec<Conditioning>,
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let cancel = CancelFlag::new();
    let baseline = conditioning.is_empty();
    let request = flux1_control_request(
        &GenerationRequest::default(),
        prompt,
        width,
        height,
        seed,
        steps,
        guidance,
        conditioning,
        None,
        gen_core::PreviewSink::default(),
        &cancel,
    );
    if baseline {
        let output = generator
            .generate(&request, &mut |_| {})
            .map_err(|error| WorkerError::Engine(format!("baseline render failed: {error}")))?;
        return match output {
            GenerationOutput::Images(mut images) => {
                let image = images
                    .pop()
                    .ok_or_else(|| WorkerError::Engine("baseline produced no image".into()))?;
                Ok((image.width, image.height, image.pixels))
            }
            _ => Err(WorkerError::Engine(
                "baseline returned non-image output".into(),
            )),
        };
    }
    let plan = super::render::compile_required_render_plan(
        crate::inference_runtime::media(),
        FLUX1_DEV_CONTROL_ENGINE_ID,
        spec,
        &request,
    )?;
    flux1_control_generate_one(generator, request, &plan, None, &mut |_| {})
}

/// Real FLUX.1-dev strict-control generation: one image per pose, each conditioned on the requested
/// control map locked by the Shakker Union-Pro-2.0 branch (sc-8244; engine E2 sc-8239). Mirrors
/// [`generate_flux2_dev_control_stream`] — the control checkpoint is fetched on first use, then the dev
/// control engine loads once on the blocking thread and renders one image per pose (shared seed so only
/// the control changes across the set). dev keeps its embedded guidance (no CFG). `advanced.controlMode`
/// (pose | canny | depth) selects the preprocessor; pose is the default (byte-preserved skeleton path).
pub(super) async fn generate_flux1_dev_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let sampling = super::sampling::explicit_sampling_request(request);

    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("FLUX.1-dev weights not found".to_owned()))?;
    let control_weights = ensure_flux1_control_weights(api, settings, job, request).await?;
    let (quant, quant_bits) = resolve_quant(request, Some(&weights_dir));
    let model = mlx_model("flux_dev")
        .ok_or_else(|| WorkerError::InvalidPayload("flux_dev model row missing".to_owned()))?;
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let control_scale = flux1_control_scale(request);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &model);
    // Shared strict-control driver: validate the requested ControlKind against the engine's
    // supported_kinds (flux1_dev_control = {Pose, Canny, Depth}) + resolve an optional user-supplied
    // control-map passthrough. A pose-only job sets no `controlMode`, so `kind == Pose` and the skeleton
    // preprocessor runs.
    let control_kind = requested_control_kind(request)?;
    validate_control_kind(FLUX1_DEV_CONTROL_ENGINE_ID, &control_kind)?;
    let user_control = resolve_user_control_map(request, settings, project_path)?;
    // sc-8244 source threading: for canny/depth WITHOUT a user-supplied control map, the control map is
    // auto-derived from the input image. The pose tier never needs a source (the skeleton is synthetic).
    let control_source = resolve_control_source(request, settings, project_path)?;
    // Auto depth-estimator weights: provisioned only when this is a depth job WITHOUT a user-supplied
    // depth map (the passthrough short-circuits estimation). Shared across the set; fetched once on the
    // first depth job (sc-8242).
    let depth_weights_dir = if control_kind == ControlKind::Depth && user_control.is_none() {
        Some(ensure_depth_estimator_dir(api, settings, job).await?)
    } else {
        None
    };
    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings = flux1_control_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance,
        control_scale,
        count,
    );
    // Strict control shares one seed across the set so noise-derived attributes (hair, wardrobe,
    // lighting) stay constant while only the control changes (FLUX.2 / Z-Image parity).
    let seed = resolve_seed(request, 0);

    // Identity-likeness scoring (epic 4406, sc-4410): a FLUX.1-dev strict-control pose set is a
    // Character-Studio pose-library job; when it carries a character identity `referenceAssetId` (FLUX.1
    // control has no img2img-init seam, but the job may still carry a character source face), score every
    // finished pose against it through the SHARED seam. Source decode + face-stack staging are non-fatal
    // (missing reference / failure → no scorer → scores omitted, set still renders). The `!Send` scorer
    // is built ONCE in the closure (source embedded once, reused across all poses).
    let likeness_source = resolve_control_identity_source(request, settings, project_path);
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        likeness_source.is_some(),
        "pose-set face-stack staging failed; likeness scores omitted",
    )
    .await;

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    let resolved_tier = resolved_mlx_artifact_tier(&weights_dir, quant_bits);
    let route_mode = crate::memory_route_registry::MemoryRouteMode::from_request(&request.mode)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "FLUX.1 control request mode '{}' is not a declared memory route",
                request.mode
            ))
        })?;
    let route_context = crate::memory_route_registry::MemoryRouteRequestContext {
        mode: route_mode,
        reference_count: 1,
        use_pid: false,
        has_phases: false,
    };
    let mut spec = flux1_control_spec(weights_dir, control_weights, quant, adapters)
        .with_resolved_route(request.model.clone());
    let pre_policy_spec = spec.clone();
    spec = crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request(
        FLUX1_DEV_CONTROL_ENGINE_ID,
        resolved_tier,
        Some(route_mode),
        &request.model_manifest_entry,
        spec,
        route_context,
    );
    spec = crate::memory_route_registry::apply_declared_mlx_load_policy_for_request(
        FLUX1_DEV_CONTROL_ENGINE_ID,
        resolved_tier,
        Some(route_mode),
        &request.model_manifest_entry,
        spec,
        route_context,
    );
    let memory_plan = crate::mlx_fit_gate::MlxRequestPlan::for_spec_and_manifest(
        FLUX1_DEV_CONTROL_ENGINE_ID,
        &request.model,
        &spec,
        Some(&request.model_manifest_entry),
        None,
    );
    let memory_plan = match resolved_tier {
        Some(tier) => memory_plan.with_resolved_artifact_tier(Some(tier))?,
        None => memory_plan,
    };
    let memory_inputs = crate::mlx_fit_gate::MlxRequestInputs {
        native_batch: 1,
        width,
        height,
        count: 1,
        mode: request.mode.clone(),
        overlay: crate::mlx_fit_gate::provider_overlay_for_load_spec(
            FLUX1_DEV_CONTROL_ENGINE_ID,
            &spec,
            Some("control".to_owned()),
        ),
        adapter_count,
        has_reference: true,
        reference_count: 1,
        use_pid: false,
        has_phases: false,
        conditioning_windows: None,
        negative_branch: false,
        prefix_kv_cache_eligible: false,
        reference_resources: None,
    };
    let (spec, memory_plan) = release_declared_sequential_for_request(
        FLUX1_DEV_CONTROL_ENGINE_ID,
        &request.model,
        &pre_policy_spec,
        spec,
        memory_plan,
        &memory_inputs,
        |spec| {
            let plan = crate::mlx_fit_gate::MlxRequestPlan::for_spec_and_manifest(
                FLUX1_DEV_CONTROL_ENGINE_ID,
                &request.model,
                spec,
                Some(&request.model_manifest_entry),
                None,
            );
            match resolved_tier {
                Some(tier) => plan.with_resolved_artifact_tier(Some(tier)),
                None => Ok(plan),
            }
        },
    )
    .await?;
    // Every pose's control map is preprocessed and its render plan compiled on the generator's own
    // owner command before the cache loads any weights (SW-6936), so a request the provider refuses
    // fails pre-load, and each pose renders from the plan compiled for its exact map.
    let planning_spec = spec.clone();
    let planning_sampling = sampling.clone();
    let planning_prompt = prompt.clone();
    let prepare_poses = move |planning_cancel: &CancelFlag| {
        poses
            .iter()
            .map(|pose| {
                let control = preprocess_control_entry(
                    &control_kind,
                    user_control.as_ref(),
                    Some(pose),
                    control_source.as_ref(),
                    width,
                    height,
                    stickwidth,
                    depth_weights_dir.as_deref(),
                )?;
                // FLUX.1 control has no img2img-init seam — no identity Reference (None).
                let conditioning =
                    build_control_conditioning(control, control_kind.clone(), control_scale, None);
                // Preview never shapes the trajectory; the live sink is bound at generation time.
                let candidate = flux1_control_request(
                    &planning_sampling,
                    &planning_prompt,
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    conditioning.clone(),
                    None,
                    gen_core::PreviewSink::default(),
                    planning_cancel,
                );
                let plan = super::render::compile_required_render_plan(
                    crate::inference_runtime::media(),
                    FLUX1_DEV_CONTROL_ENGINE_ID,
                    &planning_spec,
                    &candidate,
                )?;
                Ok((conditioning, plan))
            })
            .collect::<WorkerResult<Vec<_>>>()
    };
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state_using_prepared(
        job.id.clone(),
        FLUX1_DEV_CONTROL_ENGINE_ID,
        adapter_count,
        spec,
        "FLUX.1-dev control load failed".to_owned(),
        prepare_poses,
        crate::inference_runtime::load,
        move |generator,
              initial_cache_state,
              loaded_policy,
              warm_policy,
              external_committed_bytes,
              planned_poses,
              tx,
              cancel| {
            // Per-job identity-likeness scorer built ONCE; source embedded once, reused across every
            // pose (sc-4410). `None` ⇒ no identity reference / non-fatal construction failure.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            let mut cache_state = initial_cache_state;
            // sc-18317: ONE warm hit is ONE decision, but this lane evaluates the request once per
            // item. Hand the real proposal to the first evaluation and an inert one to the rest, so a
            // multi-image job settles exactly one decision and emits exactly one event.
            let mut warm_policy = crate::execution_planner::WarmPolicyOnce::new(warm_policy);
            drive_gen_items_scored(
                tx,
                planned_poses,
                move |_index, (conditioning, render_plan), preview, on_progress| {
                    let memory_evaluation = crate::mlx_fit_gate::evaluate_request(
                        generator,
                        &memory_plan,
                        &memory_inputs,
                        cache_state,
                        loaded_policy.offload_policy,
                        warm_policy.take(),
                        external_committed_bytes,
                    )?;
                    cache_state = gen_core::MemoryCacheState::Warm;
                    let _request_memory_limit = memory_evaluation
                        .process_limit_bytes
                        .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
                    let request = flux1_control_request(
                        &sampling,
                        &prompt,
                        width,
                        height,
                        seed,
                        steps,
                        guidance,
                        conditioning,
                        Some(memory_evaluation.memory),
                        preview,
                        &cancel,
                    );
                    let (out_w, out_h, pixels) = flux1_control_generate_one(
                        generator,
                        request,
                        &render_plan,
                        Some(&memory_evaluation),
                        on_progress,
                    )?;
                    // Score this finished pose against the cached source embedding (sc-4410). The strict-
                    // control lane produces the FINAL image directly (no face-restore pass), so this scores
                    // what the user sees. Clone paid ONLY when a scorer exists; a full-body / turned pose
                    // with no reliable frontal face → honest detected:false N/A.
                    let face_likeness = scorer.as_ref().and_then(|scorer| {
                        crate::face_likeness::score_generated_image(
                            Some(scorer),
                            &Image {
                                width: out_w,
                                height: out_h,
                                pixels: pixels.clone(),
                            },
                            likeness_source_ref.as_deref(),
                        )
                    });
                    Ok(Some((seed, out_w, out_h, pixels, face_likeness)))
                },
            )
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        FLUX1_CONTROL_ADAPTER_LABEL,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
