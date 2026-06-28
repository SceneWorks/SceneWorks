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
const FLUX1_DEV_CONTROL_ENGINE_ID: &str = "flux1_dev_control";
/// The Shakker Union-Pro-2.0 control-weights filename — the single `.safetensors` shipped in the repo
/// (the diffusers checkpoint). The default *repo* is the shared strict-control table (single source of
/// truth — `STRICT_CONTROL_ENGINES`).
const FLUX1_CONTROL_FILE: &str = "diffusion_pytorch_model.safetensors";
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
fn flux1_dev_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "flux_dev"
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// The (repo, filename) of the FLUX.1-dev control weights — `advanced.controlWeights.{repo,filename}`
/// overrides, else the Shakker Union-Pro-2.0 default (parity with the FLUX.2 / Z-Image resolvers).
fn flux1_control_repo_file(request: &ImageRequest) -> (String, String) {
    let cw = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object);
    let pick = |key: &str, default: &str| {
        cw.and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    (
        // Default repo from the shared strict-control table (single source of truth); the file stays
        // engine-specific.
        pick(
            "repo",
            strict_control_default_repo(FLUX1_DEV_CONTROL_ENGINE_ID),
        ),
        pick("filename", FLUX1_CONTROL_FILE),
    )
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
    let (repo, file) = flux1_control_repo_file(request);
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_FLUX1") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, &repo) {
        let f = snapshot.join(&file);
        if f.is_file() {
            return Ok(f);
        }
    }
    let client = reqwest::Client::new();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "FLUX.1-dev strict-control generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-flux1")
        .join(&file);
    crate::downloads::ensure_hf_cached_file(&context, &repo, "main", &file, &dst).await
}

/// Control lock strength for FLUX.1-dev: `advanced.controlScale` (default 0.7, clamp [0,2]). The Shakker
/// Union-Pro-2.0 README recommends ~0.7 (the engine default too), so the worker default matches.
fn flux1_control_scale(request: &ImageRequest) -> f32 {
    advanced::f32_clamped(&request.advanced, "controlScale", 0.7, 0.0..=2.0)
}

/// Generate one FLUX.1-dev strict-control image: the pre-built `conditioning` (the required `Control` /
/// `Depth`, assembled by the shared [`build_control_conditioning`] driver) drives the Shakker
/// Union-Pro-2.0 branch. dev is guidance-distilled (embedded scalar) — `guidance` rides the
/// transformer's guidance embedder (no true-CFG). FLUX.1 control has no img2img-init seam (unlike
/// FLUX.2), so no identity `Reference` is appended.
#[allow(clippy::too_many_arguments)]
fn flux1_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    conditioning: Vec<Conditioning>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::Engine(format!("FLUX.1-dev control generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
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

#[cfg(all(target_os = "macos", test))]
fn flux1_control_load(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<Box<dyn Generator>> {
    let spec = flux1_control_spec(weights_dir, control_weights, quant, adapters);
    gen_core::load(FLUX1_DEV_CONTROL_ENGINE_ID, &spec)
        .map_err(|error| WorkerError::Engine(format!("FLUX.1-dev control load failed: {error}")))
}

/// Real FLUX.1-dev strict-control generation: one image per pose, each conditioned on the requested
/// control map locked by the Shakker Union-Pro-2.0 branch (sc-8244; engine E2 sc-8239). Mirrors
/// [`generate_flux2_dev_control_stream`] — the control checkpoint is fetched on first use, then the dev
/// control engine loads once on the blocking thread and renders one image per pose (shared seed so only
/// the control changes across the set). dev keeps its embedded guidance (no CFG). `advanced.controlMode`
/// (pose | canny | depth) selects the preprocessor; pose is the default (byte-preserved skeleton path).
async fn generate_flux1_dev_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;

    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("FLUX.1-dev weights not found".to_owned()))?;
    let control_weights = ensure_flux1_control_weights(api, settings, job, request).await?;
    let (quant, quant_bits) = resolve_quant(request);
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
    let face_stack_dir = if likeness_source.is_some() {
        match ensure_face_stack_dir(api, settings, job).await {
            Ok(dir) => Some(dir),
            Err(error) => {
                tracing::warn!(error = %error, "pose-set face-stack staging failed; likeness scores omitted");
                None
            }
        }
    } else {
        None
    };

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    let spec = flux1_control_spec(weights_dir, control_weights, quant, adapters);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        FLUX1_DEV_CONTROL_ENGINE_ID,
        adapter_count,
        spec,
        "FLUX.1-dev control load failed".to_owned(),
        move |generator, tx, cancel| {
            let user_control = user_control.as_ref();
            let control_source = control_source.as_ref();
            let depth_weights_dir = depth_weights_dir.as_deref();
            // Per-job identity-likeness scorer built ONCE; source embedded once, reused across every
            // pose (sc-4410). `None` ⇒ no identity reference / non-fatal construction failure.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            drive_gen_items_scored(tx, poses, move |_index, pose, on_progress| {
                let control = preprocess_control_entry(
                    &control_kind,
                    user_control,
                    Some(&pose),
                    control_source,
                    width,
                    height,
                    stickwidth,
                    depth_weights_dir,
                )?;
                // FLUX.1 control has no img2img-init seam — no identity Reference (None).
                let conditioning =
                    build_control_conditioning(control, control_kind.clone(), control_scale, None);
                let (out_w, out_h, pixels) = flux1_control_generate_one(
                    generator,
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    conditioning,
                    &cancel,
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
