// ---------------------------------------------------------------------------
// Krea 2 single-phase "turbo-on-Raw" (macOS, epic 13879 S3, sc-13883): the Raw base
// DiT run through the DISTILLED TURBO SAMPLING REGIME when the accelerator-role turbo
// LoRA (sc-13882 `krea2_turbo_accel`) is selected on a `krea_2_raw` t2i job. Routed here
// from `ImageRoute::KreaTurboOnRaw`.
//
// The Krea engine keys its sampling regime on the DESCRIPTOR id, not per-generation params:
// the Raw generator (`krea_2_raw`) always runs the resolution-DYNAMIC-mu, 52-step, true-CFG
// `base_schedule`, while the Turbo generator (`krea_2_turbo`) always runs the FIXED mu 1.15,
// 8-step, CFG-free `turbo_schedule` (single conditional forward — no negative branch). Both
// share one loader/pipeline (`load_variant`) and load their weights straight off the
// `LoadSpec`, so the descriptor id and the weights are independent. This lane exploits that:
// it resolves the RAW weights + tier (the job's model IS `krea_2_raw`), applies the accelerator
// LoRA (plus any user Raw-trained LoRA) ADDITIVELY, but hands the whole spec to the `krea_2_turbo`
// ENGINE so the sampler runs the turbo regime. Net effect = "Raw base + LoRA additive, sampled
// as Turbo" — exactly the community `raw+lora` recipe (fixed mu 1.15 / ~8 steps / CFG off).
//
// This is the t2i sibling of `krea_edit.rs::krea_edit_engine_id`, which likewise picks the
// engine id (`krea_2_turbo_edit` vs `krea_2_edit`) by job shape. No inference-monorepo change
// is required — the fixed-mu / CFG-off behavior is entirely the existing `krea_2_turbo`
// generator's, reached by routing rather than by a new per-generation flag. (The whole file is
// `include!`d only on macOS, so — like `krea_edit.rs` — nothing here needs a per-item cfg.)
// ---------------------------------------------------------------------------

/// The SceneWorks model id whose LoRA-training/Raw base this accelerator lane runs on. The
/// accelerator LoRA is family `krea_2`, surfaced under Raw (`krea_2_raw.loraCompatibility.types`
/// gained `"acceleration"`, sc-13882).
const KREA_RAW_MODEL_ID: &str = "krea_2_raw";

/// The registered engine id supplying the TURBO sampling regime (fixed mu 1.15 / 8 steps /
/// CFG-free). Routing a Raw job to this engine is what selects `render_turbo` over `render_base`
/// in the pinned Krea engine — the whole mechanism this lane relies on.
const KREA_TURBO_ENGINE_ID: &str = "krea_2_turbo";

/// True when a selected LoRA declares the ACCELERATOR sampling-regime role (`role: accelerator`,
/// e.g. the builtin `krea2_turbo_accel`, sc-13882). The sampling-regime sibling of
/// [`lora_declares_image_edit_role`] (`conditioningRole: image_edit`): where that marks a LoRA
/// that changes how a job is CONDITIONED, this marks one that changes how the sampler RUNS (fewer
/// steps, CFG off). Selecting it auto-switches a Raw job to the turbo regime (this lane). Reads the
/// marker straight off the payload `loras` entry — it must round-trip via `serializeLora`
/// (web) / `serialize_job_lora` (rust-api), which forward `role` exactly like `conditioningRole`.
fn lora_declares_accelerator_role(lora: &Value) -> bool {
    lora.as_object()
        .and_then(|obj| obj.get("role"))
        .and_then(Value::as_str)
        .map(|role| role.trim().to_lowercase().replace('-', "_") == "accelerator")
        .unwrap_or(false)
}

/// Whether the job carries an accelerator LoRA (any selected LoRA with `role: accelerator`).
fn request_has_accelerator_lora(request: &ImageRequest) -> bool {
    request.loras.iter().any(lora_declares_accelerator_role)
}

/// True when this is a plain Krea 2 **Raw** t2i job carrying the accelerator (turbo) LoRA — the
/// single-phase turbo-on-Raw lane (S3). Keyed on model `krea_2_raw`, a plain t2i shape (NOT
/// `edit_image`, and no strict poses — those divert to the Krea edit / reject lanes ABOVE this arm),
/// a selected accelerator-role LoRA, and resolvable Raw weights. Mirrors [`krea_edit_available`]'s
/// job-shape gate; placed AFTER the edit/control lanes and BEFORE the generic `mlx_available` arm so
/// it wins over plain Raw t2i (which would run the 52-step true-CFG regime and never accelerate).
fn krea_turbo_on_raw_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == KREA_RAW_MODEL_ID
        && request.mode != "edit_image"
        && pose_entries(request).is_empty()
        && request_has_accelerator_lora(request)
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Flat telemetry for a turbo-on-Raw generation (parity with [`mlx_raw_settings`] /
/// [`krea_edit_raw_settings`]). Records the Raw repo/tier that actually loaded, the turbo step count,
/// a null `guidanceScale` (CFG off), and a `samplingRegime` marker naming the engine the sampler ran.
fn krea_turbo_on_raw_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // CFG off: the turbo regime is a single conditional forward, so no guidance scale is forwarded.
    raw.insert("guidanceScale".to_owned(), Value::Null);
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    // Name the sampler regime that ran (the Raw DiT sampled as Turbo) so the recipe/telemetry is
    // unambiguous about the accelerated path — distinct from a plain `krea_2_raw` render.
    raw.insert(
        "samplingRegime".to_owned(),
        Value::String(KREA_TURBO_ENGINE_ID.to_owned()),
    );
    raw
}

/// Generate one turbo-on-Raw image (plain t2i, no reference/edit conditioning). CFG-free: no negative
/// prompt and no guidance are forwarded — the `krea_2_turbo` generator runs a single conditional
/// forward per step on the fixed-mu `turbo_schedule`. Mirrors [`krea_edit_generate_one`] minus the edit
/// conditioning.
#[allow(clippy::too_many_arguments)]
fn krea_turbo_on_raw_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    text_style_gain: Option<f32>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        // CFG off — no negative branch (the distilled regime baked guidance into the weights).
        negative_prompt: None,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        // CFG off — the turbo generator is single-forward; a guidance scale would be rejected.
        guidance: None,
        text_style_gain,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::Engine(format!("Krea turbo-on-Raw generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("Krea turbo-on-Raw generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "Krea turbo-on-Raw generator returned non-image output".to_owned(),
        )),
    }
}

/// Real single-phase turbo-on-Raw generation (epic 13879 S3, sc-13883): load the RAW weights + the
/// accelerator LoRA (additive) into the `krea_2_turbo` ENGINE so the sampler runs the distilled regime
/// (fixed mu 1.15 / ~8 steps / CFG-free), then one output per requested count. Mirrors
/// [`generate_krea_edit_stream`]'s blocking-thread + streamed-events shape and reuses
/// [`consume_gen_events`]; differs in the plain t2i shape (no source references) and the engine-id
/// swap that selects the turbo sampling regime while keeping the Raw base weights.
async fn generate_krea_turbo_on_raw_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    // The job's model is `krea_2_raw`: resolve the RAW weights + tier + repo. The accelerator LoRA
    // stacks ADDITIVELY on this frozen Raw DiT (the core reads rank off the tensor shapes at load,
    // sc-13882), so the fidelity base is unchanged — only the sampler regime differs.
    let raw_model = mlx_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an MLX-backed model".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Krea 2 Raw weights not found".to_owned()))?;

    // The `krea_2_turbo` engine supplies the sampling REGIME. Same architecture / snapshot layout as
    // Raw, so it loads the Raw weights + adapters unchanged; its stored descriptor id is what makes
    // the pinned engine run `render_turbo` (fixed mu 1.15, 8-step, single conditional forward) instead
    // of `render_base` (dynamic-mu, 52-step, true CFG). Resolving the regime knobs against this Turbo
    // descriptor yields the whole regime for free: `resolve_steps` defaults to the distilled 8 (an
    // explicit `advanced.steps` is still honored). Guidance / negative are left off unconditionally —
    // the Turbo generator is single-forward and rejects both.
    let turbo = mlx_model(KREA_TURBO_ENGINE_ID).ok_or_else(|| {
        WorkerError::InvalidPayload("Krea 2 Turbo engine is not registered".to_owned())
    })?;
    let engine_id = turbo.engine_id();

    let (quant, quant_bits) = resolve_quant(request, Some(&weights_dir));
    let steps = resolve_steps(request, &turbo);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &raw_model);
    // Raw and Turbo share the `mlx_krea` adapter label, so telemetry names the family either way.
    let adapter_label = raw_model.adapter_label();
    // Krea "text style" tap-reweight gain (sc-11878) — self-gates on `ui.textStyleGain`; a no-op at the
    // default. Forwarded for parity with the plain Krea t2i path (the Turbo generator honors it).
    let text_style_gain = resolve_text_style_gain(request);

    // Plain per-image work: `request.count` renders, each its own seed + the base prompt.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let raw_settings = krea_turbo_on_raw_raw_settings(request, &repo, steps, quant_bits);

    let (width, height) = (request.width, request.height);
    let adapter_count = adapters.len();
    let spec = load_spec(weights_dir, quant, adapters, None);

    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let (out_w, out_h, pixels) = krea_turbo_on_raw_generate_one(
                    generator,
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    text_style_gain,
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
        adapter_label,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
