// Shared MLX/Candle in-place loader for a fused SDXL LDM/A1111 checkpoint (sc-14024).
// The checkpoint supplies UNet + CLIP-L + OpenCLIP-bigG + VAE. MLX borrows only tokenizer assets
// from the installed SDXL turnkey; candle stages its existing tokenizer + fp16-fix VAE components.

#[cfg(target_os = "macos")]
const SDXL_IMPORTED_ENGINE: &str = "mlx_sdxl_imported";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SDXL_IMPORTED_ENGINE: &str = "candle_sdxl_imported";

const SDXL_IMPORTED_DEFAULT_STEPS: u32 = 30;
const SDXL_IMPORTED_DEFAULT_GUIDANCE: f32 = 7.0;
const SDXL_CLIP_L_REPO: &str = "openai/clip-vit-large-patch14";
const SDXL_CLIP_L_REVISION: &str = "32bd64288804d66eefd0ccbe215aa642df71cc41";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SDXL_CLIP_BIGG_REPO: &str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SDXL_CLIP_BIGG_REVISION: &str = "743c27bd53dfe508a0ade0f50698f99b39d03bec";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SDXL_VAE_REPO: &str = "madebyollin/sdxl-vae-fp16-fix";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SDXL_VAE_REVISION: &str = "207b116dae70ace3637169f1ddd2434b91b3a8cd";

fn resolve_imported_sdxl_file(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        != Some("sdxl")
        || mlx_model(&request.model).is_some()
    {
        return Ok(None);
    }
    let Some(raw_path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .or_else(|| {
            request
                .model_manifest_entry
                .get("paths")
                .and_then(|paths| paths.get("model"))
        })
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
    else {
        return Ok(None);
    };
    let path = crate::paths::normalize_app_managed_model_path(
        settings,
        raw_path,
        "Imported SDXL checkpoint",
    )?;
    Ok(imported_dit_file(&path))
}

fn sdxl_imported_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.mode != "edit_image"
        && request.reference_asset_id.is_none()
        && request.reference_asset_ids.is_empty()
        && request.source_asset_id.is_none()
        && request.mask_asset_id.is_none()
        && request.character_id.is_none()
        && request.character_look_id.is_none()
        && pose_entries(request).is_empty()
        && matches!(
            resolve_imported_sdxl_file(request, settings),
            Ok(Some(_))
        )
}

async fn stage_sdxl_component_file(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    repo: &str,
    revision: &str,
    file: &str,
    destination: &Path,
) -> WorkerResult<()> {
    if destination.is_file() {
        return Ok(());
    }
    let client = crate::downloads::streaming_download_client();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "SDXL generation canceled while staging shared components.",
        fresh_download: false,
    };
    crate::downloads::ensure_hf_cached_file(
        &context,
        repo,
        revision,
        file,
        destination,
    )
    .await?;
    Ok(())
}

#[cfg(target_os = "macos")]
async fn stage_sdxl_tokenizer(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let root = settings
        .data_dir
        .join("cache")
        .join("sdxl-imported-components");
    for file in ["vocab.json", "merges.txt"] {
        stage_sdxl_component_file(
            api,
            settings,
            job,
            SDXL_CLIP_L_REPO,
            SDXL_CLIP_L_REVISION,
            file,
            &root.join("tokenizer").join(file),
        )
        .await?;
    }
    Ok(root)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn attach_imported_sdxl_components(
    spec: LoadSpec,
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<LoadSpec> {
    let root = settings
        .data_dir
        .join("cache")
        .join("sdxl-imported-components");
    let clip_l = root.join("clip-l");
    let clip_bigg = root.join("clip-bigg");
    let vae = root.join("vae-fp16-fix");
    for (repo, revision, file, destination) in [
        (
            SDXL_CLIP_L_REPO,
            SDXL_CLIP_L_REVISION,
            "tokenizer.json",
            clip_l.join("tokenizer.json"),
        ),
        (
            SDXL_CLIP_BIGG_REPO,
            SDXL_CLIP_BIGG_REVISION,
            "tokenizer.json",
            clip_bigg.join("tokenizer.json"),
        ),
        (
            SDXL_VAE_REPO,
            SDXL_VAE_REVISION,
            "diffusion_pytorch_model.safetensors",
            vae.join("diffusion_pytorch_model.safetensors"),
        ),
    ] {
        stage_sdxl_component_file(
            api,
            settings,
            job,
            repo,
            revision,
            file,
            &destination,
        )
        .await?;
    }
    Ok(spec
        .with_component(
            "tokenizer_clip_l",
            WeightsSource::Dir(clip_l),
        )
        .with_component(
            "tokenizer_clip_bigg",
            WeightsSource::Dir(clip_bigg),
        )
        .with_component(
            "vae_fp16_fix",
            WeightsSource::Dir(vae),
        ))
}

async fn generate_sdxl_imported_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let file = resolve_imported_sdxl_file(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Imported SDXL checkpoint could not be resolved as one confined .safetensors file"
                .to_owned(),
        )
    })?;
    let adapters = resolve_adapters(request, settings)?;
    let adapter_count = adapters.len();
    let steps =
        resolve_advanced_or_manifest_u32(request, "steps", SDXL_IMPORTED_DEFAULT_STEPS, 1..=100);
    let guidance = resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        SDXL_IMPORTED_DEFAULT_GUIDANCE,
        0.0..=30.0,
    );
    let (sampler, scheduler, scheduler_shift) = read_advanced_sampling_knobs(&request.advanced);
    let descriptor = crate::inference_runtime::media_descriptor("sdxl").ok_or_else(|| {
        WorkerError::Engine("native SDXL generator is not registered".to_owned())
    })?;
    let caps = &descriptor.capabilities;
    let sampler = normalize_sampling_knob(
        sampler,
        &caps.samplers,
        "sampler",
        &request.model,
        &job.id,
        backend,
    );
    let scheduler = normalize_sampling_knob(
        scheduler,
        &caps.schedulers,
        "scheduler",
        &request.model,
        &job.id,
        backend,
    );
    let guidance_method = normalize_sampling_knob(
        read_advanced_guidance_method(&request.advanced),
        &caps.supported_guidance_methods,
        "guidanceMethod",
        &request.model,
        &job.id,
        backend,
    );
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, "sdxl")?;
    let use_pid = pid_weights.is_some();
    let negative_prompt = (!request.negative_prompt.trim().is_empty())
        .then(|| request.negative_prompt.clone());
    let (width, height) = (request.width, request.height);
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let mut raw_settings = request.advanced.clone();
    raw_settings.insert("realModelInference".to_owned(), Value::Bool(true));
    raw_settings.insert("numInferenceSteps".to_owned(), json!(steps));
    raw_settings.insert("guidanceScale".to_owned(), json!(guidance));
    raw_settings.insert(
        "engine".to_owned(),
        Value::String(SDXL_IMPORTED_ENGINE.to_owned()),
    );
    raw_settings.insert(
        "importedCheckpoint".to_owned(),
        Value::String(request.model.clone()),
    );

    let mut spec = LoadSpec::new(WeightsSource::File(file.clone())).with_adapters(adapters);
    if let Some(pid) = pid_weights {
        spec = spec.with_pid(pid.checkpoint, pid.gemma);
    }
    #[cfg(target_os = "macos")]
    let spec = crate::mlx_fit_gate::apply_residency_policy(spec, "sdxl")?;
    #[cfg(target_os = "macos")]
    let tokenizer_root = stage_sdxl_tokenizer(api, settings, job).await?;
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let spec = attach_imported_sdxl_components(spec, api, settings, job).await?;

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        SDXL_IMPORTED_ENGINE,
        adapter_count,
        move || {
            #[cfg(target_os = "macos")]
            let loaded = runtime_macos::providers::sdxl::load_from_ldm_file(
                &spec,
                &tokenizer_root,
            );
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            let loaded = runtime_cuda::providers::sdxl::load(&spec);
            loaded.map_err(|error| {
                WorkerError::Engine(format!("SDXL imported checkpoint load failed: {error}"))
            })
        },
        move |model, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let request = GenerationRequest {
                    prompt,
                    negative_prompt: negative_prompt.clone(),
                    width,
                    height,
                    count: 1,
                    seed: Some(seed as u64),
                    steps: Some(steps),
                    guidance: Some(guidance),
                    sampler: sampler.clone(),
                    scheduler: scheduler.clone(),
                    scheduler_shift,
                    guidance_method: guidance_method.clone(),
                    use_pid,
                    cancel: cancel.clone(),
                    ..Default::default()
                };
                let output = model.generate(&request, &mut *on_progress).map_err(|error| {
                    WorkerError::Engine(format!(
                        "SDXL imported checkpoint generation failed: {error}"
                    ))
                })?;
                match output {
                    GenerationOutput::Images(mut images) => {
                        let image = images.pop().ok_or_else(|| {
                            WorkerError::Engine(
                                "SDXL imported checkpoint produced no image".to_owned(),
                            )
                        })?;
                        Ok(Some((seed, image.width, image.height, image.pixels)))
                    }
                    _ => Err(WorkerError::Engine(
                        "SDXL imported checkpoint returned non-image output".to_owned(),
                    )),
                }
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
        SDXL_IMPORTED_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
