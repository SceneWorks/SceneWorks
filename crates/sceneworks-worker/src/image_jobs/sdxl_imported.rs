// Shared MLX/Candle in-place loader for a fused SDXL LDM/A1111 checkpoint (sc-14024).
// The checkpoint supplies UNet + CLIP-L + OpenCLIP-bigG + VAE. MLX borrows only tokenizer assets
// from the installed SDXL turnkey; candle stages its existing tokenizer + fp16-fix VAE components.

#[cfg(target_os = "macos")]
const SDXL_IMPORTED_ENGINE: &str = "mlx_sdxl_imported";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SDXL_IMPORTED_ENGINE: &str = "candle_sdxl_imported";

const SDXL_IMPORTED_DEFAULT_STEPS: u32 = 30;
const SDXL_IMPORTED_DEFAULT_GUIDANCE: f32 = 7.0;
// On macOS these shared defaults are supplied by the advanced SDXL route included in the same
// module. The Windows Candle build does not include that MLX-only file, so keep its imported SDXL
// conditioning defaults explicit and byte-identical here.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SDXL_EDIT_STRENGTH: f32 = 0.6;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SDXL_INPAINT_STRENGTH: f32 = 0.85;
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

struct PreparedSdxlImportedSources {
    checkpoint: gen_core::PinnedWeightsFile,
    adapters: PreparedAdapters,
}

fn resolve_imported_sdxl_pin(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<gen_core::PinnedWeightsFile>> {
    // A plan-backed entry (`importPlan.checkpointId`, epic 20398) belongs to the plan-driven
    // route; this bespoke lane never also claims it, so one entry has exactly one owner. Explicit
    // rather than left to arm ordering in the resolver (sc-20634 review): ordering is not a claim.
    if request_is_checkpoint_plan_backed(request) {
        return Ok(None);
    }
    if request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        != Some("sdxl")
    {
        return Ok(None);
    }
    // A plan-backed entry whose request the plan route does NOT claim (edit, LoRA, Hires.fix,
    // reference-guided init) still runs here, on the plan's verified FUSED layer rather than a
    // directory scan — the same inputs the plan route would hand the same provider, so a fixed seed
    // renders equal on either route, and a LINKED fused checkpoint reaches those shapes at all
    // (sc-20644 SDXL row).
    if let Some(pin) = checkpoint_plan_bespoke_primary_pin(request, settings, "sdxl")? {
        return Ok(Some(pin));
    }
    // A builtin SDXL engine id (in `MODEL_TABLE`) loads from its snapshot turnkey via the normal MLX
    // lane — never through the single-file entrypoint. Checked AFTER the plan pin, mirroring
    // `resolve_imported_krea_dit_pin`: a plan-backed entry must be offered its verified layer before
    // a builtin-id test can send it down the turnkey path (feature-end round 1, ordering minor).
    if mlx_model(&request.model).is_some() {
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
    let confined = crate::paths::normalize_app_managed_model_path(
        settings,
        raw_path,
        "Imported SDXL checkpoint",
    )?;
    let candidate = if confined.is_dir() {
        // Select inside the already-confined target, then reconstruct the caller-visible child and
        // re-confine/pin that exact lexical entry. A child symlink must never inherit trust from its
        // parent directory.
        imported_dit_file(&confined).and_then(|file| {
            file.file_name()
                .map(|name| Path::new(raw_path).join(name))
        })
    } else if imported_dit_file(Path::new(raw_path)).is_some() {
        Some(PathBuf::from(raw_path))
    } else {
        None
    };
    candidate
        .map(|file| {
            crate::paths::pin_app_managed_model_file(
                settings,
                &file,
                "Imported SDXL checkpoint",
            )
        })
        .transpose()
}

fn sdxl_imported_request_shape_available(request: &ImageRequest) -> bool {
    if sceneworks_core::jobs_store::imported_control_intent_is_material(&request.advanced) {
        return false;
    }
    let operation = if request.mode == "edit_image" {
        gen_core::ImportedModelOperation::Edit
    } else {
        gen_core::ImportedModelOperation::Generate
    };
    let Some(descriptor) = crate::inference_runtime::imported_model_descriptor(
        "sdxl",
        gen_core::ImportedModelSource::FusedCheckpoint,
        operation,
    ) else {
        return false;
    };
    let caps = &descriptor.capabilities;
    if imported_model_quant(request, &descriptor, "Imported SDXL").is_err() {
        return false;
    }
    let has_phases = request
        .advanced
        .get("phases")
        .and_then(Value::as_array)
        .is_some_and(|phases| !phases.is_empty());
    if !pose_entries(request).is_empty()
        || !request.reference_asset_ids.is_empty()
        || request.character_id.is_some()
        || request.character_look_id.is_some()
        || has_phases
        || (!request.loras.is_empty() && !(caps.supports_lora || caps.supports_lokr))
    {
        return false;
    }
    let has_reference = non_empty(&request.reference_asset_id);
    let has_source = non_empty(&request.source_asset_id);
    if request.mode == "edit_image" {
        if !(has_source || has_reference)
            || !caps
                .conditioning
                .contains(&gen_core::ConditioningKind::Reference)
        {
            return false;
        }
        if non_empty(&request.mask_asset_id)
            && !caps
                .conditioning
                .contains(&gen_core::ConditioningKind::Mask)
        {
            return false;
        }
    } else if has_source
        || non_empty(&request.mask_asset_id)
        || (has_reference
            && !caps
                .conditioning
                .contains(&gen_core::ConditioningKind::Reference))
    {
        return false;
    }
    true
}

#[cfg(test)]
fn resolve_imported_sdxl_file(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    Ok(resolve_imported_sdxl_pin(request, settings)?
        .map(|pin| pin.loader_path().to_path_buf()))
}

#[cfg(test)]
fn sdxl_imported_available(request: &ImageRequest, settings: &Settings) -> bool {
    sdxl_imported_request_shape_available(request)
        && matches!(
            resolve_imported_sdxl_file(request, settings),
            Ok(Some(_))
        )
}

/// Resolve imported SDXL img2img/edit/inpaint/outpaint conditioning. The exact registered provider
/// owns the capability claim; this helper only turns SceneWorks asset ids into that contract.
fn resolve_imported_sdxl_conditioning(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<(Vec<Conditioning>, &'static str, Option<f32>)> {
    let (width, height) = (request.width, request.height);
    if request.mode != "edit_image" {
        let Some(asset_id) = request
            .reference_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
        else {
            return Ok((Vec::new(), "text_to_image", None));
        };
        let image = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            asset_id,
            project_path,
        )?;
        let image = fit_engine_image(image, width, height, &request.fit_mode)?;
        let strength = advanced::f32_clamped(
            &request.advanced,
            "strength",
            SDXL_EDIT_STRENGTH,
            0.0..=1.0,
        );
        return Ok((
            vec![Conditioning::Reference {
                image,
                strength: Some(strength),
            }],
            "img2img",
            Some(strength),
        ));
    }

    let source_id = request
        .source_asset_id
        .as_deref()
        .or(request.reference_asset_id.as_deref())
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Imported SDXL edit requires a source image".to_owned())
        })?;
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        source_id,
        project_path,
    )?;
    let outpaint = request.fit_mode == "outpaint";
    let masked = non_empty(&request.mask_asset_id);
    let (src_w, src_h) = (source.width, source.height);
    let source = fit_engine_image(source, width, height, &request.fit_mode)?;
    let strength = advanced::f32_clamped(
        &request.advanced,
        "strength",
        if masked || outpaint {
            SDXL_INPAINT_STRENGTH
        } else {
            SDXL_EDIT_STRENGTH
        },
        0.0..=1.0,
    );
    let mut conditioning = vec![Conditioning::Reference {
        image: source,
        strength: Some(strength),
    }];
    if masked || outpaint {
        let mut mask = if outpaint {
            gen_core::imageops::outpaint_border_mask(src_w, src_h, width, height)
        } else {
            let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
            let mask = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                mask_id,
                project_path,
            )?;
            fit_engine_image(mask, width, height, &request.fit_mode)?
        };
        if outpaint && masked {
            let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
            let user_mask = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                mask_id,
                project_path,
            )?;
            let user_mask = fit_engine_image(user_mask, width, height, "pad")?;
            mask = gen_core::imageops::union_masks(&mask, &user_mask).map_err(|error| {
                WorkerError::Engine(format!("Imported SDXL outpaint mask union failed: {error}"))
            })?;
        }
        conditioning.push(Conditioning::Mask { image: mask });
    }
    Ok((
        conditioning,
        if outpaint {
            "outpaint"
        } else if masked {
            "inpaint"
        } else {
            "edit"
        },
        Some(strength),
    ))
}

fn prepare_sdxl_imported_sources(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedSdxlImportedSources>> {
    if !sdxl_imported_request_shape_available(request) {
        return Ok(None);
    }
    let Some(checkpoint) = resolve_imported_sdxl_pin(request, settings)? else {
        return Ok(None);
    };
    Ok(Some(PreparedSdxlImportedSources {
        checkpoint,
        adapters: resolve_prepared_adapters(request, settings)?,
    }))
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

/// The `ldm_tokenizer` directory a fused SDXL checkpoint pairs with, resolved **cache-only** for the
/// plan-driven route: a root whose `tokenizer/` subdir holds CLIP-L's `vocab.json` + `merges.txt`
/// (`mlx-gen-sdxl`'s `loader::load_tokenizer` joins `tokenizer` onto whatever directory it is
/// handed).
///
/// Lives in THIS file, not in the plan route, for two reasons: the staging root and the pinned
/// CLIP-L revision are this lane's, so both routes hand the loader one directory rather than two
/// copies; and `<data>/cache/sdxl-imported-components` is a destination epic 17625's guard already
/// attributes to `image_jobs/sdxl_imported.rs` — a second file naming the same literal would count
/// as a second weights destination against a ratchet that only goes down, for no new destination.
///
/// This never DOWNLOADS. It mirrors two ~1 MB model-agnostic assets that are already in the
/// app-managed Hugging Face cache, and refuses with an actionable install-it-from-the-Model-Manager
/// diagnostic when they are not — during PLANNING, not inside a loader (epic 17625 AC9 / E7).
///
/// Without it the plan route CLAIMED a plan-backed fused SDXL text-to-image request — the SDXL
/// adapter binds `FusedCheckpoint`/`Generate` on this backend — and then refused it with
/// `[checkpoint-plan:missing-component] … 'ldm_tokenizer'`, so the family had no servable shape at
/// all on the universal route (sc-20644 SDXL row).
fn resolve_sdxl_ldm_tokenizer_root_cache_only(
    settings: &Settings,
    checkpoint_id: &str,
) -> WorkerResult<PathBuf> {
    let root = settings
        .data_dir
        .join("cache")
        .join("sdxl-imported-components");
    let tokenizer_dir = root.join("tokenizer");
    let staging_failed = |path: &Path, error: std::io::Error| {
        WorkerError::Engine(format!(
            "Checkpoint plan source preparation failed: {} ({error})",
            path.display()
        ))
    };
    for file in ["vocab.json", "merges.txt"] {
        let destination = tokenizer_dir.join(file);
        if destination.is_file() {
            continue;
        }
        let cached = crate::downloads::resolve_hf_component_file(
            settings,
            SDXL_CLIP_L_REPO,
            SDXL_CLIP_L_REVISION,
            file,
        )
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "[checkpoint-plan:missing-component] checkpoint {checkpoint_id:?} (sdxl family) \
                 requires the fused-checkpoint tokenizer assets, and {SDXL_CLIP_L_REPO}/{file} is \
                 not installed — install it from the Model Manager, then run this checkpoint \
                 again. A fused SDXL checkpoint carries its own UNet, encoders and VAE; only the \
                 model-agnostic CLIP tokenizer vocabulary comes from outside it."
            ))
        })?;
        std::fs::create_dir_all(&tokenizer_dir)
            .map_err(|error| staging_failed(&tokenizer_dir, error))?;
        // Copy to a pid-keyed sibling and rename, so two concurrent jobs can never observe a
        // half-written vocabulary at the destination path.
        let staging = tokenizer_dir.join(format!("{file}.{}.partial", std::process::id()));
        std::fs::copy(&cached, &staging)
            .and_then(|_| std::fs::rename(&staging, &destination))
            .map_err(|error| {
                let _ = std::fs::remove_file(&staging);
                staging_failed(&destination, error)
            })?;
    }
    Ok(root)
}

/// Where each SDXL component id's bytes come from: `(component id, repo, revision, file, subdir)`.
///
/// Catalog data only. WHICH of these the imported fused route actually needs is the registered
/// provider's `required_components`, never this table: a fused LDM checkpoint carries its own VAE,
/// so the candle binding declares `LDM_REQUIRED_COMPONENTS` — the two model-agnostic tokenizers —
/// while the ordinary snapshot route additionally requires the fp16-fix VAE. Staging all three
/// regardless downloaded ~335 MB from `madebyollin/sdxl-vae-fp16-fix` on every imported render and
/// handed the loader a component it drops: `SdxlComponents::from_spec` makes `vae_fp16_fix` optional
/// for a fused source and `load_components` takes the VAE from `ldm.vae`. A pure dead download,
/// removed by asking the descriptor (sc-20644 SDXL row).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SDXL_IMPORTED_COMPONENT_SOURCES: &[(&str, &str, &str, &str, &str)] = &[
    (
        "tokenizer_clip_l",
        SDXL_CLIP_L_REPO,
        SDXL_CLIP_L_REVISION,
        "tokenizer.json",
        "clip-l",
    ),
    (
        "tokenizer_clip_bigg",
        SDXL_CLIP_BIGG_REPO,
        SDXL_CLIP_BIGG_REVISION,
        "tokenizer.json",
        "clip-bigg",
    ),
    (
        "vae_fp16_fix",
        SDXL_VAE_REPO,
        SDXL_VAE_REVISION,
        "diffusion_pytorch_model.safetensors",
        "vae-fp16-fix",
    ),
];

/// Stage exactly the components the RESOLVED provider declares, in its declared order.
///
/// A declared id this build cannot source refuses by name rather than being skipped: skipping would
/// hand the loader an incomplete spec and turn a staging gap into a mid-load failure (E7).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn attach_imported_sdxl_components(
    spec: LoadSpec,
    descriptor: &gen_core::ModelDescriptor,
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<LoadSpec> {
    let root = settings
        .data_dir
        .join("cache")
        .join("sdxl-imported-components");
    let mut spec = spec;
    for component in descriptor.required_components {
        let Some((_, repo, revision, file, subdir)) = SDXL_IMPORTED_COMPONENT_SOURCES
            .iter()
            .find(|(id, ..)| id == component)
        else {
            return Err(WorkerError::InvalidPayload(format!(
                "The registered imported SDXL provider {:?} declares required component \
                 {component:?}, which this build cannot stage",
                descriptor.id
            )));
        };
        let destination_dir = root.join(subdir);
        stage_sdxl_component_file(
            api,
            settings,
            job,
            repo,
            revision,
            file,
            &destination_dir.join(file),
        )
        .await?;
        spec = spec.with_component(*component, WeightsSource::Dir(destination_dir));
    }
    Ok(spec)
}

async fn generate_sdxl_imported_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    dispatch: PreparedFileDispatch<'_, PreparedSdxlImportedSources>,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let PreparedFileDispatch { plan, sources } = dispatch;
    let request = &plan.request;
    let PreparedSdxlImportedSources {
        checkpoint: file_pin,
        adapters: prepared_adapters,
    } = sources;
    let file = file_pin.loader_path().to_path_buf();
    let adapter_count = prepared_adapters.specs.len();
    let steps =
        resolve_advanced_or_manifest_u32(request, "steps", SDXL_IMPORTED_DEFAULT_STEPS, 1..=100);
    let guidance = resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        SDXL_IMPORTED_DEFAULT_GUIDANCE,
        0.0..=30.0,
    );
    let (sampler, scheduler, scheduler_shift) = read_advanced_sampling_knobs(&request.advanced);
    let operation = if request.mode == "edit_image" {
        gen_core::ImportedModelOperation::Edit
    } else {
        gen_core::ImportedModelOperation::Generate
    };
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        "sdxl",
        gen_core::ImportedModelSource::FusedCheckpoint,
        operation,
    )
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "The active backend has no registered imported SDXL {operation:?} provider"
        ))
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
    let (conditioning, mode_tag, strength) =
        resolve_imported_sdxl_conditioning(request, settings, project_path)?;
    let hires_fix = resolve_hires_fix_plan(request, steps, Some(guidance), None);
    if !conditioning.is_empty() && hires_fix.is_some() {
        return Err(WorkerError::InvalidPayload(
            "Imported SDXL img2img/edit conditioning cannot be combined with Hires.fix".to_owned(),
        ));
    }
    let (quant, quant_bits) = imported_model_quant(request, &descriptor, "Imported SDXL")?;
    let enhance = PromptEnhance::default();
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
        Value::String(descriptor.id.to_owned()),
    );
    raw_settings.insert(
        "importedCheckpoint".to_owned(),
        Value::String(request.model.clone()),
    );
    raw_settings.insert("mode".to_owned(), Value::String(mode_tag.to_owned()));
    raw_settings.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map_or(Value::Null, Value::from),
    );
    if let Some(strength) = strength {
        raw_settings.insert("strength".to_owned(), json!(strength));
    }
    if request.hires_fix.enabled {
        raw_settings.insert(
            "hiresFix".to_owned(),
            serde_json::to_value(&request.hires_fix).expect("HiresFixRequest is serializable"),
        );
    }

    let mut spec = LoadSpec::new(WeightsSource::File(file.clone()));
    if !prepared_adapters.specs.is_empty() {
        spec = spec.with_adapters(prepared_adapters.specs);
    }
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    crate::paths::prepare_load_spec_with_file_pins(
        &mut spec,
        std::iter::once(file_pin).chain(prepared_adapters.pins),
        "SDXL imported source preparation failed",
    )?;
    if let Some(pid) = pid_weights {
        spec = spec.with_pid(pid.checkpoint, pid.gemma);
    }
    #[cfg(target_os = "macos")]
    let spec = crate::mlx_fit_gate::apply_residency_policy(spec, descriptor.id)?;
    #[cfg(target_os = "macos")]
    let spec = spec.with_component(
        "ldm_tokenizer",
        WeightsSource::Dir(stage_sdxl_tokenizer(api, settings, job).await?),
    );
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let spec = attach_imported_sdxl_components(spec, &descriptor, api, settings, job).await?;
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    admit_candle_load_spec_floor(&request.model, "SDXL imported", settings, &spec).await?;

    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        descriptor.id,
        adapter_count,
        spec,
        "SDXL imported checkpoint load failed".to_owned(),
        move |model, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                if let Some(hires_fix) = hires_fix {
                    let (out_width, out_height, pixels) = generate_one_with_hires(
                        model,
                        &prompt,
                        width,
                        height,
                        seed,
                        steps,
                        Some(guidance),
                        negative_prompt.clone(),
                        None,
                        &[],
                        None,
                        None,
                        sampler.as_deref(),
                        scheduler.as_deref(),
                        scheduler_shift,
                        guidance_method.as_deref(),
                        use_pid,
                        None,
                        None,
                        None,
                        &enhance,
                        Some(hires_fix),
                        preview,
                        gen_core::PromptEnhancementSink::default(),
                        &cancel,
                        on_progress,
                    )?;
                    return Ok(Some((seed, out_width, out_height, pixels)));
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
                    conditioning: conditioning.clone(),
                    preview,
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
