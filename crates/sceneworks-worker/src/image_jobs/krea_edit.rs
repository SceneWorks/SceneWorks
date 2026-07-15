// ---------------------------------------------------------------------------
// Krea 2 image-edit (macOS, epic 10871): the Kontext-style dual-conditioned edit
// surface. Routed here from `ImageRoute::KreaEdit` — an `edit_image` job on
// `krea_2_raw` carrying a source image. Loads the `krea_2_edit` registry generator
// (the Raw pipeline routed to `generate_edit_with_progress`: each source rides as
// in-context VAE tokens at a distinct RoPE frame AND grounds the Qwen3-VL vision
// tower). Conditions on one source (`Conditioning::Reference`) or two in FIXED order
// — image 1 (required) + image 2 (optional) (`Conditioning::MultiReference`, epic 10871
// P1.3); either image can be a person. One output per requested count, each an edit of
// the same source(s) under the instruction prompt. Krea is MLX-only, so this is the only
// Krea edit path (the candle mirror is a separate slice, P1.2/P2.2 + the sc-11085 seam
// registration). Mirrors
// `generate_flux2_edit_stream`'s blocking-thread + streamed-events shape and reuses
// `consume_gen_events`.
// ---------------------------------------------------------------------------

/// The engine registry id the edit lane loads, chosen by model: the undistilled Raw pipeline routed to
/// the Kontext edit entrypoint (`krea_2_edit`, full-CFG ~52 steps, epic 10871) for `krea_2_raw`, or the
/// distilled CFG-free few-step edit (`krea_2_turbo_edit`, guidance=0 ~8 steps, sc-11640) for
/// `krea_2_turbo`. Distinct from the plain `krea_2_*` t2i/img2img generators — the distinct id is what
/// tells the engine to treat the source `Reference` as an edit, not an img2img init.
fn krea_edit_engine_id(model: &str) -> &'static str {
    if model == "krea_2_turbo" {
        "krea_2_turbo_edit"
    } else {
        "krea_2_edit"
    }
}

/// The most source images a Krea edit conditions on (epic 10871 P1.3): image 1 (required) + image 2
/// (optional), a FIXED order (swapping degrades identity per the LoRA authors); either can be a
/// person. Mirrors the engine cap
/// (`mlx-gen-krea` / `candle-gen-krea` `MAX_EDIT_REFERENCES`); the worker rejects a longer list up
/// front so the error is clear rather than surfacing from the engine seam.
const KREA_MAX_EDIT_REFERENCES: usize = 2;

/// True when a selected LoRA declares the image-edit conditioning role (`conditioningRole: image_edit`,
/// e.g. the builtin `krea2_identity_edit`). The image-edit sibling of the LTX IC-LoRA detector
/// (`lora_looks_like_ic_lora`): this LoRA is what makes the in-context source actually steer the edit —
/// the base weights leave it inert (R5).
fn lora_declares_image_edit_role(lora: &Value) -> bool {
    lora.as_object()
        .and_then(|obj| obj.get("conditioningRole"))
        .and_then(Value::as_str)
        .map(|role| role.trim().to_lowercase().replace('-', "_") == "image_edit")
        .unwrap_or(false)
}

/// Whether the job carries an image-edit LoRA (any selected LoRA with `conditioningRole: image_edit`).
fn request_has_image_edit_lora(request: &ImageRequest) -> bool {
    request.loras.iter().any(lora_declares_image_edit_role)
}

/// True when this is a Krea edit job whose weights resolve — routed to the `krea_2_edit` /
/// `krea_2_turbo_edit` engine rather than the plain `krea_2_*` t2i/img2img path. Both Krea image
/// variants edit: **Raw** on the full-CFG `krea_2_edit` (epic 10871) and **Turbo** on the CFG-free
/// distilled `krea_2_turbo_edit` (sc-11640); both drive the same `krea2_identity_edit` LoRA. Keyed on
/// `edit_image` mode + a source asset. Mirrors the core router's `krea_mlx_eligible` edit branch.
fn krea_edit_available(request: &ImageRequest, settings: &Settings) -> bool {
    matches!(request.model.as_str(), "krea_2_raw" | "krea_2_turbo")
        && request.mode == "edit_image"
        && !edit_reference_ids(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Flat telemetry for a Krea edit generation (parity with `flux2_edit_raw_settings`).
fn krea_edit_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    reference_count: usize,
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
    raw.insert(
        "editEngine".to_owned(),
        Value::String(krea_edit_engine_id(&request.model).to_owned()),
    );
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw
}

/// Generate one Krea edit image conditioned on `conditioning` (the source `Reference`). Full-CFG: passes
/// the negative prompt + `guidance`. The `krea_2_edit` generator routes the `Reference` to the Kontext
/// edit entrypoint. Mirrors [`flux2_edit_generate_one`].
#[allow(clippy::too_many_arguments)]
fn krea_edit_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    conditioning: Vec<Conditioning>,
    text_style_gain: Option<f32>,
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
        steps: Some(steps),
        guidance,
        conditioning,
        text_style_gain,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::Engine(format!("Krea edit generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("Krea edit generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "Krea edit generator returned non-image output".to_owned(),
        )),
    }
}

/// Real Krea 2 edit generation: load the `krea_2_edit` generator once, then one output per requested
/// count, each an edit of the shared source(s) under the instruction prompt. Mirrors
/// [`generate_flux2_edit_stream`]'s blocking-thread + streamed-events shape and reuses
/// [`consume_gen_events`]; differs in the required edit LoRA (R5) and the two-reference edit
/// conditioning (one `Reference` or a two-source `MultiReference`, epic 10871 P1.3).
async fn generate_krea_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let model = mlx_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an MLX-backed model".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Krea 2 weights not found".to_owned()))?;

    // R5 (epic 10871): the base cannot edit without the edit LoRA — its in-context/grounded source
    // conditioning is inert without the trained weights (a VAE-only render is off-distribution, not
    // shippable). Require an `image_edit`-role LoRA (the builtin `krea2_identity_edit`), mirroring the
    // LTX IC-LoRA hard requirement. Checked before loading weights so the error is fast + clear.
    if !request_has_image_edit_lora(request) {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires the Krea 2 Identity Edit LoRA (or another image-edit LoRA): without \
             it the source-image conditioning is inert. Select it in the LoRA picker."
                .to_owned(),
        ));
    }

    let (quant, quant_bits) = resolve_quant(request, Some(&weights_dir));
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &model);
    let adapter_label = model.adapter_label();

    // Resolve the source image(s) on the async side (decode → Send Image moved into the worker thread).
    // Krea edit conditions on image 1 (required) + image 2 (optional) (fixed order, epic 10871 P1.3): the
    // multi-image picker sends the plural `referenceAssetIds`, and a single Image-Edit `sourceAssetId`
    // is the one-source case (both via `edit_reference_ids`). Capped at [`KREA_MAX_EDIT_REFERENCES`] —
    // `build_edit_conditioning` then emits a single `Reference` (1) or a `MultiReference` (2), which the
    // `krea_2_edit` generator VAE-encodes at successive RoPE frames.
    let reference_ids = edit_reference_ids(request);
    if reference_ids.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires a source image (sourceAssetId).".to_owned(),
        ));
    }
    if reference_ids.len() > KREA_MAX_EDIT_REFERENCES {
        return Err(WorkerError::InvalidPayload(format!(
            "Krea 2 edit takes at most {KREA_MAX_EDIT_REFERENCES} images (image 1, then image 2)."
        )));
    }
    let mut sources = Vec::with_capacity(reference_ids.len());
    for id in &reference_ids {
        sources.push(load_reference_image(
            &settings.data_dir,
            &request.project_id,
            id,
            project_path,
        )?);
    }
    // Pre-fit each source to the target W×H (crop / pad / outpaint→pad) so an off-aspect source isn't
    // squished into the latent grid; `stretch` keeps the legacy resize. Fixed order preserved. Shared
    // with the other edit lanes.
    let sources = fit_edit_references(sources, request, request.width, request.height)?;
    let reference_count = sources.len();

    // Plain per-image work: `request.count` edits of the same source(s), each its own seed + the base
    // instruction prompt (Krea edit has no angle/pose grouping — that is the character_image path).
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let raw_settings =
        krea_edit_raw_settings(request, &repo, steps, quant_bits, guidance, reference_count);

    let (width, height) = (request.width, request.height);
    let adapter_count = adapters.len();
    let spec = load_spec(weights_dir, quant, adapters, None);
    // Raw → `krea_2_edit` (full-CFG); Turbo → `krea_2_turbo_edit` (CFG-free distilled, sc-11640).
    let engine_id = krea_edit_engine_id(&request.model);
    // Krea "text style" tap-reweight gain (sc-12009) — self-gates on `ui.textStyleGain` (Krea only),
    // applied to the edit lane's POSITIVE grounded context by the engine (inference sc-12009).
    let text_style_gain = resolve_text_style_gain(request);
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
                let conditioning = build_edit_conditioning(&sources);
                let (out_w, out_h, pixels) = krea_edit_generate_one(
                    generator,
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    conditioning,
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
