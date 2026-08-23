// Generic SDXL OpenPose ControlNet, shared by MLX and Candle.
//
// The two platform bundles both register the ordinary `sdxl` provider. A `LoadSpec` carrying one
// control checkpoint selects its ControlNet implementation; each generation then carries exactly one
// `Conditioning::Control(Pose)`. The public image DTO remains singular-control: multiple library poses
// are a batch of independent one-control requests, never an `extra_controls` composition.

const SDXL_CONTROL_ENGINE_ID: &str = "sdxl";
const SDXL_CONTROL_REPO: &str = "xinsir/controlnet-openpose-sdxl-1.0";
const SDXL_CONTROL_FILE: &str = "diffusion_pytorch_model.safetensors";
const SDXL_CONTROL_ADAPTER_LABEL: &str = "sdxl_control";
const SDXL_CONTROL_DEFAULT_SCALE: f32 = 1.0;
const SDXL_CONTROL_DEFAULT_STEPS: u32 = 30;
const SDXL_CONTROL_DEFAULT_GUIDANCE: f32 = 7.0;
const SDXL_CONTROL_LIGHTNING_STEPS: u32 = 4;
const SDXL_CONTROL_LIGHTNING_GUIDANCE: f32 = 1.0;

// The currently pinned mlx-gen-sdxl provider rejects ControlNet whenever the `lightning`
// acceleration sampler is selected. Keep the route itself exact and backend-neutral, but fail
// before downloads/load on Mac until the inference follow-up makes this composition truthful.
// Candle's registered SDXL control provider already implements the Lightning composition.
const MLX_LIGHTNING_CONTROLNET_READY: bool = false;

const SDXL_CONTROL_MODELS: &[&str] = &[
    "sdxl",
    "realvisxl",
    "realvisxl_lightning",
];

fn is_sdxl_control_model(model: &str) -> bool {
    SDXL_CONTROL_MODELS.contains(&model)
}

/// A material pose carrier on one of the exact three supported models. This deliberately does not
/// validate or weight-gate: it wins routing first, then [`validate_sdxl_control_request`] reports a
/// typed error for conflicts/malformed/count violations and the stream reports missing weights.
fn sdxl_control_candidate(request: &ImageRequest) -> bool {
    if !is_sdxl_control_model(&request.model) {
        return false;
    }
    let poses_are_material = match request.advanced.get("poses") {
        None | Some(Value::Null) => false,
        Some(Value::Array(poses)) if poses.is_empty() => false,
        Some(_) => true,
    };
    let named_control_is_material = match request.advanced.get("controlMode") {
        None | Some(Value::Null) => false,
        Some(Value::String(mode)) => !mode.trim().is_empty(),
        Some(_) => true,
    };
    poses_are_material
        || named_control_is_material
        || request
            .advanced
            .get("controlImage")
            .is_some_and(|value| !value.is_null())
        || request
            .advanced
            .get("controlWeights")
            .is_some_and(|value| !value.is_null())
}

fn validate_sdxl_control_backend(model: &str, backend: &str) -> WorkerResult<()> {
    if backend == "mlx"
        && model == "realvisxl_lightning"
        && !MLX_LIGHTNING_CONTROLNET_READY
    {
        return Err(WorkerError::InvalidPayload(
            "RealVisXL Lightning pose control is not available on the currently pinned MLX provider; refusing instead of exposing a broken Mac render path"
                .to_owned(),
        ));
    }
    Ok(())
}

/// The terminal CUDA campaign proved only the q4 composition. Missing selectors resolve to each
/// accepted model's default q4 package; explicit selectors must name q4 exactly so q8/bf16 cannot
/// borrow the q4 receipt or fall through to a dense/on-the-fly load.
fn validate_sdxl_control_candle_tier(request: &ImageRequest) -> WorkerResult<()> {
    let quant_tier_is_q4 = match request.advanced.get("quantTier") {
        None | Some(Value::Null) => true,
        Some(Value::String(tier)) => tier.trim().eq_ignore_ascii_case("q4"),
        Some(_) => false,
    };
    let quant_bits_are_q4 = match request.advanced.get("mlxQuantize") {
        None | Some(Value::Null) => true,
        Some(value) => value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse::<i64>().ok())
            == Some(4),
    };
    if quant_tier_is_q4 && quant_bits_are_q4 {
        return Ok(());
    }
    Err(WorkerError::InvalidPayload(
        "SDXL pose control on Candle is proven only for the exact q4 package; advanced.quantTier and advanced.mlxQuantize must resolve to q4"
            .to_owned(),
    ))
}

fn sdxl_control_native_backend() -> &'static str {
    if cfg!(target_os = "macos") {
        "mlx"
    } else {
        "candle"
    }
}

fn strict_optional_number(
    advanced: &JsonObject,
    key: &str,
) -> WorkerResult<Option<f32>> {
    let Some(value) = advanced.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let number = value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("advanced.{key} must be a finite number"))
        })? as f32;
    if !number.is_finite() {
        return Err(WorkerError::InvalidPayload(format!(
            "advanced.{key} must fit in a finite 32-bit number"
        )));
    }
    Ok(Some(number))
}

fn strict_optional_steps(advanced: &JsonObject) -> WorkerResult<Option<u32>> {
    let Some(value) = advanced.get("steps") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let steps = value
        .as_u64()
        .or_else(|| value.as_str()?.trim().parse::<u64>().ok())
        .filter(|steps| (1..=80).contains(steps))
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "advanced.steps must be an integer between 1 and 80".to_owned(),
            )
        })?;
    Ok(Some(steps as u32))
}

fn optional_advanced_name(advanced: &JsonObject, key: &str) -> WorkerResult<Option<String>> {
    let Some(value) = advanced.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let value = value.as_str().ok_or_else(|| {
        WorkerError::InvalidPayload(format!("advanced.{key} must be a string"))
    })?;
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("default") {
        Ok(None)
    } else {
        Ok(Some(value.to_ascii_lowercase()))
    }
}

fn validate_sdxl_control_request(request: &ImageRequest) -> WorkerResult<Vec<PoseInput>> {
    if !is_sdxl_control_model(&request.model) {
        return Err(WorkerError::InvalidPayload(format!(
            "SDXL pose control is not supported for model '{}'",
            request.model
        )));
    }
    if !matches!(request.mode.as_str(), "text_to_image" | "image_generation") {
        return Err(WorkerError::InvalidPayload(
            "SDXL pose control is text-to-image only and cannot be combined with edit, reference, or character modes"
                .to_owned(),
        ));
    }
    if request.source_asset_id.is_some()
        || request.reference_asset_id.is_some()
        || !request.reference_asset_ids.is_empty()
        || request.mask_asset_id.is_some()
        || request.character_id.is_some()
        || request.character_look_id.is_some()
    {
        return Err(WorkerError::InvalidPayload(
            "SDXL pose control cannot be combined with source, reference, mask, or character conditioning"
                .to_owned(),
        ));
    }
    if request.hires_fix.enabled {
        return Err(WorkerError::InvalidPayload(
            "SDXL pose control cannot be combined with Hires.fix".to_owned(),
        ));
    }
    for key in ["controlImage", "phases"] {
        if request
            .advanced
            .get(key)
            .is_some_and(|value| !value.is_null())
        {
            return Err(WorkerError::InvalidPayload(format!(
                "advanced.{key} cannot be combined with SDXL pose control"
            )));
        }
    }
    if request.advanced.get("usePid").is_some_and(|value| {
        value.as_bool() != Some(false) && !value.is_null()
    }) {
        return Err(WorkerError::InvalidPayload(
            "advanced.usePid cannot be combined with SDXL pose control".to_owned(),
        ));
    }
    if request.advanced.get("decoder").is_some_and(|value| {
        !value.is_null()
            && value
                .as_str()
                .map_or(true, |value| !matches!(value.trim(), "" | "native"))
    }) {
        return Err(WorkerError::InvalidPayload(
            "advanced.decoder cannot be combined with SDXL pose control".to_owned(),
        ));
    }

    match request.advanced.get("controlMode") {
        None | Some(Value::Null) => {}
        Some(Value::String(mode)) if mode.trim().eq_ignore_ascii_case("pose") => {}
        Some(Value::String(mode)) => {
            return Err(WorkerError::InvalidPayload(format!(
                "SDXL control supports pose only; advanced.controlMode '{}' is unsupported",
                mode.trim()
            )));
        }
        Some(_) => {
            return Err(WorkerError::InvalidPayload(
                "advanced.controlMode must be the string 'pose' for SDXL control".to_owned(),
            ));
        }
    }

    let poses = request.advanced.get("poses").ok_or_else(|| {
        WorkerError::InvalidPayload("SDXL pose control requires advanced.poses".to_owned())
    })?;
    let poses = poses.as_array().ok_or_else(|| {
        WorkerError::InvalidPayload("advanced.poses must be an array".to_owned())
    })?;
    if poses.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "SDXL pose control requires at least one pose".to_owned(),
        ));
    }
    if poses.len() > sceneworks_core::image_request::MAX_JOB_POSES {
        return Err(WorkerError::InvalidPayload(format!(
            "advanced.poses exceeds the {}-pose job limit",
            sceneworks_core::image_request::MAX_JOB_POSES
        )));
    }
    if !poses.iter().all(Value::is_object) {
        return Err(WorkerError::InvalidPayload(
            "every advanced.poses entry must be an object".to_owned(),
        ));
    }
    Ok(parse_poses(request))
}

fn sdxl_control_scale(request: &ImageRequest) -> WorkerResult<f32> {
    let scale = strict_optional_number(&request.advanced, "controlScale")?
        .unwrap_or(SDXL_CONTROL_DEFAULT_SCALE);
    if !(0.0..=2.0).contains(&scale) {
        return Err(WorkerError::InvalidPayload(
            "advanced.controlScale must be between 0 and 2".to_owned(),
        ));
    }
    Ok(scale)
}

struct SdxlControlSampling {
    steps: u32,
    guidance: Option<f32>,
    sampler: Option<String>,
    scheduler: Option<String>,
}

fn sdxl_control_sampling(request: &ImageRequest) -> WorkerResult<SdxlControlSampling> {
    let requested_steps = strict_optional_steps(&request.advanced)?;
    let requested_guidance = strict_optional_number(&request.advanced, "guidanceScale")?;
    let requested_sampler = optional_advanced_name(&request.advanced, "sampler")?;
    let requested_scheduler = optional_advanced_name(&request.advanced, "scheduler")?;

    if request.model == "realvisxl_lightning" {
        let guidance = requested_guidance.unwrap_or(SDXL_CONTROL_LIGHTNING_GUIDANCE);
        if guidance > 1.0 {
            return Err(WorkerError::InvalidPayload(
                "RealVisXL Lightning pose control requires guidanceScale <= 1.0".to_owned(),
            ));
        }
        if requested_scheduler
            .as_deref()
            .is_some_and(|scheduler| scheduler != "normal")
        {
            return Err(WorkerError::InvalidPayload(
                "RealVisXL Lightning pose control supports only the default/normal scheduler"
                    .to_owned(),
            ));
        }
        return Ok(SdxlControlSampling {
            // The standalone checkpoint's accepted route is the fixed 4-step recipe. Ignore the
            // catalog's ordinary txt2img default (historically 5) and any caller override rather
            // than feeding an unsupported step count into the Lightning policy.
            steps: SDXL_CONTROL_LIGHTNING_STEPS,
            guidance: Some(guidance),
            // The standalone distilled checkpoint always uses its Euler-trailing Lightning recipe.
            sampler: Some("lightning".to_owned()),
            scheduler: requested_scheduler,
        });
    }

    let model = mlx_model(&request.model).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{} model row missing", request.model))
    })?;
    let caps = &model.descriptor.capabilities;
    let sampler = requested_sampler
        .map(|name| {
            if caps.samplers.contains(&name.as_str()) {
                Ok(name)
            } else {
                Err(WorkerError::InvalidPayload(format!(
                    "sampler '{name}' is not supported by the SDXL control provider"
                )))
            }
        })
        .transpose()?;
    let scheduler = requested_scheduler
        .map(|name| {
            if caps.schedulers.contains(&name.as_str()) {
                Ok(name)
            } else {
                Err(WorkerError::InvalidPayload(format!(
                    "scheduler '{name}' is not supported by the SDXL control provider"
                )))
            }
        })
        .transpose()?;
    Ok(SdxlControlSampling {
        steps: requested_steps.unwrap_or(SDXL_CONTROL_DEFAULT_STEPS),
        guidance: Some(requested_guidance.unwrap_or(SDXL_CONTROL_DEFAULT_GUIDANCE)),
        sampler,
        scheduler,
    })
}

fn sdxl_control_repo_file(request: &ImageRequest) -> WorkerResult<(String, String, String)> {
    let control = match request.advanced.get("controlWeights") {
        None | Some(Value::Null) => None,
        Some(Value::Object(control)) => Some(control),
        Some(_) => {
            return Err(WorkerError::InvalidPayload(
                "advanced.controlWeights must be an object".to_owned(),
            ));
        }
    };
    let string_field = |key: &str| -> WorkerResult<Option<String>> {
        let Some(value) = control.and_then(|control| control.get(key)) else {
            return Ok(None);
        };
        let value = value.as_str().ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "advanced.controlWeights.{key} must be a string"
            ))
        })?;
        let value = value.trim();
        if value.is_empty() {
            return Err(WorkerError::InvalidPayload(format!(
                "advanced.controlWeights.{key} cannot be empty"
            )));
        }
        Ok(Some(value.to_owned()))
    };
    let default_repo = strict_control_default_repo(SDXL_CONTROL_ENGINE_ID);
    debug_assert_eq!(default_repo, SDXL_CONTROL_REPO);
    let repo = string_field("repo")?.unwrap_or_else(|| default_repo.to_owned());
    let file = safe_weight_filename(
        &string_field("filename")?.unwrap_or_else(|| SDXL_CONTROL_FILE.to_owned()),
        "advanced.controlWeights.filename",
    )?;
    let revision = trusted_control_weight_revision(
        request,
        SDXL_CONTROL_ENGINE_ID,
        &repo,
        &file,
    )?;
    Ok((repo, file, revision))
}

/// Resolve the immutable ControlNet artifact installed through Model Manager. This helper has only
/// [`Settings`] and is synchronous by design: a render job can inspect the HF cache, but it cannot
/// construct a download client or create an app-private fallback destination.
fn require_sdxl_control_weights(
    settings: &Settings,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file, revision) = sdxl_control_repo_file(request)?;
    crate::downloads::resolve_hf_component_file(settings, &repo, &revision, &file).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "SDXL pose-control weights are not installed. Install \"SDXL OpenPose ControlNet\" in Model Manager when using the shipped default, or install the selected control overlay, then retry (required {repo}@{revision}/{file})"
        ))
    })
}

fn sdxl_control_spec(
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

/// Resolve every weight path carried by the finalized SDXL ControlNet load. The base is returned
/// separately because the Candle conditioning gate reports base/overlay bytes independently; all
/// other typed and named sources are overlays held by the same provider load. The gate's
/// `ConditioningFootprint::from_paths` containment pass then makes a component nested under the base
/// (or a path repeated in two slots) count exactly once.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn sdxl_control_admission_paths(spec: &LoadSpec) -> WorkerResult<(&Path, Vec<&Path>)> {
    let base = match &spec.weights {
        WeightsSource::Dir(path) => path.as_path(),
        WeightsSource::File(_) => {
            return Err(WorkerError::Engine(
                "SDXL pose-control admission requires a directory base".to_owned(),
            ));
        }
    };
    let mut overlays = Vec::new();
    if let Some(control) = &spec.control {
        overlays.push(crate::conditioning_fit::weights_source_path(control));
    }
    overlays.extend(
        spec.extra_controls
            .iter()
            .map(crate::conditioning_fit::weights_source_path),
    );
    if let Some(ip_adapter) = &spec.ip_adapter {
        overlays.push(crate::conditioning_fit::weights_source_path(ip_adapter));
    }
    overlays.extend(spec.adapters.iter().map(|adapter| adapter.path.as_path()));
    if let Some(pid) = &spec.pid {
        overlays.push(crate::conditioning_fit::weights_source_path(&pid.checkpoint));
        overlays.push(crate::conditioning_fit::weights_source_path(&pid.gemma));
    }
    if let Some(identity) = &spec.identity {
        overlays.extend(
            [
                identity.encoder.as_ref(),
                identity.eva.as_ref(),
                identity.face_dir.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(crate::conditioning_fit::weights_source_path),
        );
    }
    if let Some(text_encoder) = &spec.text_encoder {
        overlays.push(crate::conditioning_fit::weights_source_path(text_encoder));
    }
    overlays.extend(
        spec.components
            .values()
            .map(crate::conditioning_fit::weights_source_path),
    );
    Ok((base, overlays))
}

#[cfg(any(target_os = "macos", test))]
fn apply_sdxl_control_mlx_residency(spec: LoadSpec) -> WorkerResult<LoadSpec> {
    crate::mlx_fit_gate::apply_residency_policy(spec, SDXL_CONTROL_ENGINE_ID)
}

#[cfg(any(
    test,
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn sdxl_control_flash_attn(request: &ImageRequest) -> bool {
    request
        .advanced
        .get("flashAttn")
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn apply_sdxl_control_flash_attn(enabled: bool) {
    runtime_cuda::providers::sdxl::set_flash_attn(enabled);
}

#[allow(clippy::too_many_arguments)]
fn sdxl_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    sampling: &SdxlControlSampling,
    conditioning: Vec<Conditioning>,
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(sampling.steps),
        guidance: sampling.guidance,
        sampler: sampling.sampler.clone(),
        scheduler: sampling.scheduler.clone(),
        conditioning,
        preview,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::Engine(format!("SDXL pose-control generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("SDXL pose-control generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "SDXL pose-control generator returned non-image output".to_owned(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
async fn generate_sdxl_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let poses = validate_sdxl_control_request(request)?;
    // `backend` is the telemetry device label (`metal`, `cuda:0`, ...), not the provider family.
    // Select the native provider from the compiled bundle so the MLX Lightning refusal cannot be
    // bypassed by a device-specific label.
    validate_sdxl_control_backend(&request.model, sdxl_control_native_backend())?;
    if sdxl_control_native_backend() == "candle" {
        validate_sdxl_control_candle_tier(request)?;
    }
    let weights_dir = resolve_weights_dir(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} weights not found for SDXL pose control",
            request.model
        ))
    })?;
    let control_weights = require_sdxl_control_weights(settings, request)?;
    let (quant, quant_bits) = resolve_quant(request, Some(&weights_dir));
    let adapters = resolve_adapters(request, settings)?;
    let adapter_count = adapters.len();
    let sampling = sdxl_control_sampling(request)?;
    let control_scale = sdxl_control_scale(request)?;
    let model = mlx_model(&request.model).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{} model row missing", request.model))
    })?;
    let repo = model_repo(request, &model);
    let negative_prompt = if request.model == "realvisxl_lightning" {
        None
    } else {
        resolve_negative_prompt(request, &model)
    };
    let (control_repo, _, _) = sdxl_control_repo_file(request)?;
    let mut raw_settings = mlx_raw_settings(
        request,
        &repo,
        sampling.steps,
        quant_bits,
        sampling.guidance,
    );
    raw_settings.insert("controlEngine".to_owned(), Value::String(SDXL_CONTROL_ENGINE_ID.to_owned()));
    raw_settings.insert("controlRepo".to_owned(), Value::String(control_repo));
    raw_settings.insert("controlMode".to_owned(), Value::String("pose".to_owned()));
    raw_settings.insert("controlScale".to_owned(), json!(control_scale));
    raw_settings.insert("poseCount".to_owned(), json!(poses.len()));
    raw_settings.insert(
        "sampler".to_owned(),
        sampling.sampler.clone().map(Value::String).unwrap_or(Value::Null),
    );
    raw_settings.insert(
        "scheduler".to_owned(),
        sampling.scheduler.clone().map(Value::String).unwrap_or(Value::Null),
    );

    let mut spec = sdxl_control_spec(
        weights_dir.clone(),
        control_weights.clone(),
        quant,
        adapters,
    );
    spec = attach_required_components(
        spec,
        SDXL_CONTROL_ENGINE_ID,
        &request.model_manifest_entry,
        settings,
    )?;
    spec = attach_manifest_text_encoder(spec, SDXL_CONTROL_ENGINE_ID, request, settings)?;
    spec = spec.with_resolved_route(request.model.clone());

    // This bespoke route bypasses the generator cache, so apply the same provider-derived MLX
    // residency/fit contract here on the final spec before its uncached load.
    #[cfg(target_os = "macos")]
    let spec = apply_sdxl_control_mlx_residency(spec)?;

    // Every Candle conditioning route is admitted before its uncached provider load. Derive the
    // footprint from the FINAL spec, after request adapters, required named components, and any
    // selected text encoder have been attached. `from_paths` deduplicates nested/repeated paths.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        let (base, overlays) = sdxl_control_admission_paths(&spec)?;
        admit_conditioning_paths(
            settings,
            &request.model,
            "SDXL OpenPose ControlNet",
            base,
            &overlays,
        )
        .await?;
    }

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let seed = resolve_seed(request, 0);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let total = poses.len();
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let flash_attn = sdxl_control_flash_attn(request);
    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        SDXL_CONTROL_ENGINE_ID,
        adapter_count,
        move || {
            // The registered txt2img route applies this process-global setting before load, but this
            // bespoke ControlNet route bypasses that seam. Set it for every job (including `false`)
            // immediately before load so a previous job's value can never leak into this pipeline.
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            apply_sdxl_control_flash_attn(flash_attn);
            crate::inference_runtime::load(SDXL_CONTROL_ENGINE_ID, &spec).map_err(|error| {
                WorkerError::Engine(format!("SDXL pose-control load failed: {error}"))
            })
        },
        move |generator, tx, cancel| {
            drive_gen_items(tx, poses, move |_index, pose, preview, on_progress| {
                let control = preprocess_control_entry(
                    &ControlKind::Pose,
                    None,
                    Some(&pose),
                    None,
                    width,
                    height,
                    stickwidth,
                    None,
                )?;
                let conditioning = build_control_conditioning(
                    control,
                    ControlKind::Pose,
                    control_scale,
                    None,
                );
                let (out_w, out_h, pixels) = sdxl_control_generate_one(
                    generator.as_ref(),
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    &sampling,
                    conditioning,
                    preview,
                    &cancel,
                    on_progress,
                )?;
                Ok(Some((seed, out_w, out_h, pixels)))
            })
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        SDXL_CONTROL_ADAPTER_LABEL,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

#[cfg(test)]
mod sdxl_control_tests {
    use super::*;

    fn request(value: Value) -> ImageRequest {
        ImageRequest::from_payload(value.as_object().expect("payload"))
    }

    fn isolate_hf_cache() -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ])
    }

    fn offline_settings_at(data_dir: &Path) -> Settings {
        Settings {
            data_dir: data_dir.to_path_buf(),
            ..crate::test_env::offline_settings()
        }
    }

    fn stage_control_snapshot(data_dir: &Path, revision: &str) -> PathBuf {
        let snapshot = crate::huggingface_repo_cache_path(data_dir, SDXL_CONTROL_REPO)
            .expect("repo cache path")
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).expect("create snapshot");
        let weights = snapshot.join(SDXL_CONTROL_FILE);
        std::fs::write(&weights, b"installed control weights").expect("stage weights");
        weights
    }

    #[test]
    fn exact_three_model_family_and_material_candidate_are_pinned() {
        assert_eq!(
            SDXL_CONTROL_MODELS,
            ["sdxl", "realvisxl", "realvisxl_lightning"]
        );
        for model in SDXL_CONTROL_MODELS {
            assert!(sdxl_control_candidate(&request(json!({
                "model": model,
                "advanced": { "poses": [{ "keypoints": [] }] }
            }))));
        }
        assert!(!sdxl_control_candidate(&request(json!({
            "model": "instantid_realvisxl",
            "advanced": { "poses": [{ "keypoints": [] }] }
        }))));
        for model in ["illustrious_xl_v1", "illustrious_xl_v2"] {
            assert!(!sdxl_control_candidate(&request(json!({
                "model": model,
                "advanced": { "poses": [{ "keypoints": [] }] }
            }))));
        }
        assert!(!sdxl_control_candidate(&request(json!({
            "model": "sdxl",
            "advanced": { "poses": [] }
        }))));
        for advanced in [
            json!({ "controlMode": "pose" }),
            json!({ "controlMode": "canny" }),
            json!({ "controlMode": false }),
            json!({ "controlImage": "asset" }),
            json!({ "controlWeights": {} }),
        ] {
            assert!(sdxl_control_candidate(&request(json!({
                "model": "sdxl",
                "advanced": advanced,
            }))));
        }
        assert!(!sdxl_control_candidate(&request(json!({
            "model": "sdxl",
            "advanced": { "controlMode": "  " }
        }))));
    }

    #[test]
    fn load_spec_is_one_dense_control_over_a_directory_base() {
        let root = tempfile::tempdir().expect("tempdir");
        let spec = sdxl_control_spec(
            root.path().join("q8"),
            root.path().join(SDXL_CONTROL_FILE),
            Some(Quant::Q8),
            Vec::new(),
        );
        assert!(matches!(spec.weights, WeightsSource::Dir(_)));
        assert!(matches!(spec.control, Some(WeightsSource::File(_))));
        assert!(spec.extra_controls.is_empty(), "public route is singular-control only");
        assert_eq!(spec.quantize, Some(Quant::Q8));
    }

    #[test]
    fn installed_control_component_resolves_from_the_exact_pinned_snapshot() {
        let _env = isolate_hf_cache();
        let root = tempfile::tempdir().expect("data dir");
        let artifact = sceneworks_core::control_weights::shipped_control_weight(
            SDXL_CONTROL_ENGINE_ID,
            SDXL_CONTROL_REPO,
            SDXL_CONTROL_FILE,
        )
        .expect("shipped SDXL control authority");
        let installed = stage_control_snapshot(root.path(), artifact.revision);
        let resolved = require_sdxl_control_weights(
            &offline_settings_at(root.path()),
            &request(json!({ "model": "sdxl", "advanced": { "poses": [{}] } })),
        )
        .expect("installed component resolves without network");
        assert_eq!(resolved, installed);
    }

    #[test]
    fn missing_control_component_is_an_actionable_model_manager_refusal() {
        let _env = isolate_hf_cache();
        let root = tempfile::tempdir().expect("data dir");
        let error = require_sdxl_control_weights(
            &offline_settings_at(root.path()),
            &request(json!({ "model": "sdxl", "advanced": { "poses": [{}] } })),
        )
        .expect_err("missing component must refuse before provider load");
        let message = error.to_string();
        assert!(message.contains("Model Manager"), "{message}");
        assert!(message.contains(SDXL_CONTROL_REPO), "{message}");
        assert!(message.contains(SDXL_CONTROL_FILE), "{message}");
        assert!(message.contains(
            sceneworks_core::control_weights::default_control_revision(SDXL_CONTROL_ENGINE_ID)
                .expect("shipped revision")
        ));
    }

    #[test]
    fn control_component_is_cache_only_with_no_legacy_or_mutable_fallback() {
        let _env = isolate_hf_cache();
        let root = tempfile::tempdir().expect("data dir");
        let repo_cache = crate::huggingface_repo_cache_path(root.path(), SDXL_CONTROL_REPO)
            .expect("repo cache path");

        // Neither a mutable `refs/main` snapshot nor the unpublished route's former private-cache
        // destination is an installed immutable component. Both must remain invisible.
        let stale = stage_control_snapshot(root.path(), "mutable-main");
        std::fs::create_dir_all(repo_cache.join("refs")).expect("refs dir");
        std::fs::write(repo_cache.join("refs/main"), "mutable-main").expect("refs main");
        let legacy = root
            .path()
            .join("cache")
            .join("controlnet-sdxl")
            .join(SDXL_CONTROL_FILE);
        std::fs::create_dir_all(legacy.parent().expect("legacy parent")).expect("legacy dir");
        std::fs::write(&legacy, b"legacy job cache").expect("legacy weights");
        assert!(stale.is_file() && legacy.is_file());

        let error = require_sdxl_control_weights(
            &offline_settings_at(root.path()),
            &request(json!({ "model": "sdxl", "advanced": { "poses": [{}] } })),
        )
        .expect_err("only the exact Model Manager snapshot may satisfy the render");
        assert!(error.to_string().contains("Model Manager"), "{error}");
    }

    #[test]
    fn render_weight_resolver_has_no_network_or_job_context_capability() {
        // This type assertion locks the cache-only seam: reintroducing an async resolver or giving
        // it API/job/download context fails to compile. The repository-wide job-time download guard
        // separately scans the reachable body and rejects any network-capable call site.
        let resolver: fn(&Settings, &ImageRequest) -> WorkerResult<PathBuf> =
            require_sdxl_control_weights;
        let _ = resolver;
    }

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn finalized_admission_counts_adapter_component_and_selected_te_once() {
        let root = tempfile::tempdir().expect("tempdir");
        let base = root.path().join("base");
        let nested = base.join("vae");
        std::fs::create_dir_all(&nested).expect("base dirs");
        std::fs::write(base.join("model.safetensors"), vec![0_u8; 11]).expect("base");
        std::fs::write(nested.join("model.safetensors"), vec![0_u8; 13]).expect("nested");
        let write = |name: &str, bytes: usize| {
            let path = root.path().join(name);
            std::fs::write(&path, vec![0_u8; bytes]).expect("overlay");
            path
        };
        let control = write("control.safetensors", 17);
        let adapter = write("adapter.safetensors", 19);
        let component = write("component.safetensors", 23);
        let text_encoder = write("text-encoder.safetensors", 29);

        let spec = sdxl_control_spec(
            base,
            control.clone(),
            Some(Quant::Q8),
            vec![AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)],
        )
        // The nested component is already covered by the recursively scanned base.
        .with_component("nested_vae", WeightsSource::Dir(nested))
        .with_component("external_component", WeightsSource::File(component))
        // A repeated source in another named slot must not be priced twice.
        .with_component("repeated_control", WeightsSource::File(control))
        .with_text_encoder(WeightsSource::File(text_encoder));
        let (base, overlays) = sdxl_control_admission_paths(&spec).expect("admission paths");
        let footprint = crate::conditioning_fit::ConditioningFootprint::from_paths(
            "sdxl",
            "SDXL OpenPose ControlNet",
            base,
            &overlays,
        );
        assert_eq!(footprint.base_bytes, 11 + 13);
        assert_eq!(
            footprint.overlay_bytes,
            17 + 19 + 23 + 29,
            "control + adapter + external component + selected TE, each exactly once"
        );
    }

    #[test]
    fn mlx_fit_gate_refuses_the_complete_control_composition_under_a_small_cap() {
        use std::fs::OpenOptions;

        const GIB: u64 = 1024 * 1024 * 1024;
        let root = tempfile::tempdir().expect("tempdir");
        let sparse = |path: PathBuf, bytes: u64| {
            let file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .expect("sparse source");
            file.set_len(bytes).expect("set sparse length");
            path
        };
        let base = root.path().join("base");
        std::fs::create_dir_all(&base).expect("base dir");
        sparse(base.join("model.safetensors"), GIB);
        let control = sparse(root.path().join("control.safetensors"), GIB / 8);
        let adapter = sparse(root.path().join("adapter.safetensors"), GIB / 8);
        let component = sparse(root.path().join("component.safetensors"), GIB / 4);
        let selected_te = sparse(root.path().join("selected-te.safetensors"), 5 * GIB / 2);

        let base_composition = sdxl_control_spec(
            base,
            control,
            Some(Quant::Q8),
            vec![AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)],
        )
        .with_component("vae_fp16_fix", WeightsSource::File(component));
        let complete = base_composition
            .clone()
            .with_text_encoder(WeightsSource::File(selected_te));

        crate::test_env::temp_env_var(
            crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV,
            "4",
            || {
                assert!(
                    apply_sdxl_control_mlx_residency(base_composition).is_ok(),
                    "the otherwise identical composition fits without the selected encoder"
                );
                let error = apply_sdxl_control_mlx_residency(complete)
                    .expect_err("attaching the selected encoder alone must cross the fit threshold");
                assert!(error.to_string().contains("unified memory"), "{error}");
            },
        );
    }

    #[test]
    fn flash_attention_defaults_on_and_preserves_explicit_false() {
        let default = request(json!({ "model": "sdxl", "advanced": { "poses": [{}] } }));
        let enabled = request(json!({
            "model": "sdxl", "advanced": { "poses": [{}], "flashAttn": true }
        }));
        let disabled = request(json!({
            "model": "sdxl", "advanced": { "poses": [{}], "flashAttn": false }
        }));
        assert!(sdxl_control_flash_attn(&default));
        assert!(sdxl_control_flash_attn(&enabled));
        assert!(!sdxl_control_flash_attn(&disabled));
    }

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn candle_flash_attention_is_reset_for_true_and_false_jobs() {
        apply_sdxl_control_flash_attn(true);
        assert!(runtime_cuda::providers::sdxl::flash_attn_enabled());
        apply_sdxl_control_flash_attn(false);
        assert!(!runtime_cuda::providers::sdxl::flash_attn_enabled());
        // Restore the process default for any later SDXL test.
        apply_sdxl_control_flash_attn(true);
    }

    #[test]
    fn validation_rejects_conflicts_malformed_counts_and_non_pose_modes() {
        let valid = request(json!({
            "model": "sdxl", "mode": "text_to_image",
            "advanced": { "controlMode": "pose", "poses": [{ "keypoints": [] }] }
        }));
        assert_eq!(validate_sdxl_control_request(&valid).unwrap().len(), 1);

        for bad in [
            json!({ "model": "sdxl", "mode": "edit_image", "sourceAssetId": "source", "advanced": { "poses": [{}] } }),
            json!({ "model": "sdxl", "referenceAssetId": "ref", "advanced": { "poses": [{}] } }),
            json!({ "model": "sdxl", "advanced": { "controlMode": "canny", "poses": [{}] } }),
            json!({ "model": "sdxl", "advanced": { "controlMode": "canny" } }),
            json!({ "model": "sdxl", "advanced": { "controlMode": "pose" } }),
            json!({ "model": "sdxl", "advanced": { "poses": [] } }),
            json!({ "model": "sdxl", "advanced": { "poses": "bad" } }),
            json!({ "model": "sdxl", "advanced": { "poses": [null] } }),
            json!({ "model": "sdxl", "advanced": { "poses": [{}], "usePid": true } }),
            json!({ "model": "sdxl", "advanced": { "poses": [{}], "phases": [] } }),
        ] {
            assert!(validate_sdxl_control_request(&request(bad)).is_err());
        }

        let overflowing_scale = request(json!({
            "model": "sdxl",
            "advanced": { "poses": [{}], "controlScale": "1e100" }
        }));
        assert!(sdxl_control_scale(&overflowing_scale).is_err());

        let too_many = vec![json!({}); sceneworks_core::image_request::MAX_JOB_POSES + 1];
        let over_limit = request(json!({
            "model": "sdxl",
            "advanced": { "poses": too_many }
        }));
        assert!(validate_sdxl_control_request(&over_limit).is_err());
    }

    #[test]
    fn candle_control_accepts_only_the_receipt_backed_q4_tier() {
        for advanced in [
            json!({ "poses": [{}] }),
            json!({ "poses": [{}], "mlxQuantize": 4 }),
            json!({ "poses": [{}], "mlxQuantize": "4", "quantTier": "q4" }),
        ] {
            let request = request(json!({ "model": "sdxl", "advanced": advanced }));
            assert!(validate_sdxl_control_candle_tier(&request).is_ok());
        }
        for advanced in [
            json!({ "poses": [{}], "mlxQuantize": 8 }),
            json!({ "poses": [{}], "mlxQuantize": 0 }),
            json!({ "poses": [{}], "quantTier": "q8" }),
            json!({ "poses": [{}], "quantTier": "bf16" }),
            json!({ "poses": [{}], "mlxQuantize": 4, "quantTier": "q8" }),
        ] {
            let request = request(json!({ "model": "sdxl", "advanced": advanced }));
            assert!(validate_sdxl_control_candle_tier(&request).is_err());
        }
    }

    #[test]
    fn lightning_recipe_defaults_and_rejections_are_exact() {
        let default = request(json!({
            "model": "realvisxl_lightning",
            "advanced": { "poses": [{}], "steps": 8 }
        }));
        let sampling = sdxl_control_sampling(&default).expect("default lightning recipe");
        assert_eq!(sampling.steps, 4);
        assert_eq!(sampling.guidance, Some(1.0));
        assert_eq!(sampling.sampler.as_deref(), Some("lightning"));
        assert_eq!(sampling.scheduler, None);

        let explicit = request(json!({
            "model": "realvisxl_lightning",
            "advanced": {
                "poses": [{}], "guidanceScale": 1.0,
                "sampler": "ddim", "scheduler": "normal"
            }
        }));
        let sampling = sdxl_control_sampling(&explicit).expect("explicit accepted ceiling");
        assert_eq!(sampling.steps, 4);
        assert_eq!(sampling.guidance, Some(1.0));
        assert_eq!(sampling.sampler.as_deref(), Some("lightning"));
        assert_eq!(sampling.scheduler.as_deref(), Some("normal"));

        for advanced in [
            json!({ "poses": [{}], "guidanceScale": 1.01 }),
            json!({ "poses": [{}], "scheduler": "karras" }),
        ] {
            let request = request(json!({
                "model": "realvisxl_lightning",
                "advanced": advanced,
            }));
            assert!(sdxl_control_sampling(&request).is_err());
        }
    }

    #[test]
    fn current_mlx_lightning_gap_fails_closed_without_disabling_candle_or_other_models() {
        assert!(validate_sdxl_control_backend("realvisxl_lightning", "mlx").is_err());
        assert!(validate_sdxl_control_backend("realvisxl_lightning", "candle").is_ok());
        if cfg!(target_os = "macos") {
            assert_eq!(sdxl_control_native_backend(), "mlx");
        } else {
            assert_eq!(sdxl_control_native_backend(), "candle");
        }
        for model in ["sdxl", "realvisxl"] {
            assert!(validate_sdxl_control_backend(model, "mlx").is_ok());
        }
    }
}
