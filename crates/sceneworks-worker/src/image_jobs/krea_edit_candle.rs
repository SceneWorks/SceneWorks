// Candle (Windows/CUDA) Krea 2 image-edit route (epic 10871) — the Kontext-style dual-conditioned edit
// surface off-Mac via `candle_gen_krea::pipeline::{load_components, load_edit_components, render_edit}`.
// The candle sibling of the macOS `krea_edit.rs` (`generate_krea_edit_stream`): an `edit_image` job on
// `krea_2_raw` carrying a source image (+ optional 2nd) is routed here rather than the plain Raw t2i path.
// The source(s) ride as in-context VAE tokens at distinct RoPE frames AND ground the Qwen3-VL vision
// tower (R2 dual conditioning); the trained `krea2_identity_edit` LoRA is what makes that conditioning
// steer the edit (R5 — the base leaves it inert).
//
// **Candle-only.** macOS keeps the MLX Krea edit path (`krea_edit.rs`, `krea_2_edit` registry generator);
// the candle Krea edit is a bespoke pipeline (no registered `krea_2_edit` candle generator — the candle
// edit providers are all driven directly by the worker), so this whole file is gated to the Windows/CUDA
// candle build (the `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs`
// module, so it shares that module's imports (ImageRequest/Settings/WorkerResult/WorkerError/
// `load_reference_image`/`fit_engine_image`/`resolve_weights_dir`/`resolve_adapters`/`resolve_seed`/
// `resolve_advanced_or_manifest_u32`/`start_gen_stream`/`drive_gen_items`/`consume_gen_events`/`non_empty`/
// `gen_core`/`AdapterSpec`/… all in scope).
//
// Both Krea image variants edit: **Raw** on the full-CFG loop (epic 10871) and **Turbo** on the CFG-free
// distilled few-step loop (`render_edit(distilled = true)`, sc-11640 — the fast-path); the same
// `krea2_identity_edit` LoRA drives both (family match, no base gating). One or two references in fixed
// order — scene = image 1, person = image 2 ([`KREA_EDIT_CANDLE_MAX_REFERENCES`]).

/// Krea edit denoise steps default — Raw undistilled, full-CFG (mirrors `candle_gen_krea` `RAW_STEPS`).
const KREA_EDIT_CANDLE_DEFAULT_STEPS: u32 = 52;
/// Krea Turbo edit denoise steps default — the distilled CFG-free few-step student (sc-11640; mirrors
/// `candle_gen_krea` `TURBO_STEPS`).
const KREA_EDIT_CANDLE_TURBO_STEPS: u32 = 8;
/// Raw full-CFG guidance default (mirrors `candle_gen_krea` `RAW_GUIDANCE`).
const KREA_EDIT_CANDLE_DEFAULT_GUIDANCE: f32 = 3.5;
/// The adapter/engine id recorded on candle Krea edit assets + telemetry (distinct from the `candle_krea`
/// txt2img lane — the edit is a separate surface with the required edit LoRA).
const KREA_EDIT_CANDLE_ENGINE: &str = "candle_krea_edit";
/// Reference cap — the edit LoRA's fixed-order contract: scene = image 1, person = image 2. Swapping the
/// order degrades results (the LoRA authors' note); more than two is off-contract.
const KREA_EDIT_CANDLE_MAX_REFERENCES: usize = 2;

/// True when a selected LoRA declares the image-edit conditioning role (`conditioningRole: image_edit`,
/// e.g. the builtin `krea2_identity_edit`). The candle twin of the macOS
/// `krea_edit.rs::lora_declares_image_edit_role` — that file is `#[cfg(target_os = "macos")]`, so the
/// tiny check is duplicated here for the candle build. This LoRA is what makes the in-context source
/// actually steer the edit; the base weights leave it inert (R5).
fn krea_edit_candle_lora_role(lora: &Value) -> bool {
    lora.as_object()
        .and_then(|obj| obj.get("conditioningRole"))
        .and_then(Value::as_str)
        .map(|role| role.trim().to_lowercase().replace('-', "_") == "image_edit")
        .unwrap_or(false)
}

/// Whether the job carries an image-edit LoRA (any selected LoRA with `conditioningRole: image_edit`).
fn krea_edit_candle_has_lora(request: &ImageRequest) -> bool {
    request.loras.iter().any(krea_edit_candle_lora_role)
}

/// Reference asset ids for a Krea edit, in fixed order (scene = image 1, person = image 2), capped at
/// [`KREA_EDIT_CANDLE_MAX_REFERENCES`]. The multi-image picker sends the plural `referenceAssetIds`; with
/// no plural list it falls back to the single Image-Edit `sourceAssetId`. Mirrors
/// `flux2_edit_candle_reference_ids` (capped to the Krea 1..=2 contract).
fn krea_edit_candle_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        return request
            .reference_asset_ids
            .iter()
            .take(KREA_EDIT_CANDLE_MAX_REFERENCES)
            .cloned()
            .collect();
    }
    if let Some(id) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
    }
    Vec::new()
}

/// True when this is a Krea edit job: `krea_2_raw` / `krea_2_turbo` + `edit_image` mode + at least one
/// source reference. Both Krea image variants edit — Raw on the full-CFG loop, Turbo on the CFG-free
/// distilled few-step loop (sc-11640). Mirrors the core router's `krea_edit_candle_eligible` (gated to the
/// two models by its caller) + the macOS `krea_edit_available` (minus the weight-resolve check).
fn krea_edit_candle_mode(request: &ImageRequest) -> bool {
    matches!(request.model.as_str(), "krea_2_raw" | "krea_2_turbo")
        && request.mode == "edit_image"
        && !krea_edit_candle_reference_ids(request).is_empty()
}

/// Whether this Krea edit runs the distilled CFG-free Turbo recipe (`krea_2_turbo` — few-step
/// `turbo_schedule`, guidance forced to 0) rather than the undistilled full-CFG Raw loop (sc-11640).
fn krea_edit_candle_distilled(request: &ImageRequest) -> bool {
    request.model == "krea_2_turbo"
}

/// True when this is a candle-eligible Krea edit job whose Raw weights resolve locally. Mirrors
/// `flux2_edit_candle_available` (Krea resolves through the shared `resolve_weights_dir` → the
/// `krea_2_raw` tier subdir).
fn krea_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    krea_edit_candle_mode(request)
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=100) → manifest `steps` → the per-variant default
/// (Raw 52 / Turbo 8, sc-11640).
fn krea_edit_candle_steps(request: &ImageRequest) -> u32 {
    let default = if krea_edit_candle_distilled(request) {
        KREA_EDIT_CANDLE_TURBO_STEPS
    } else {
        KREA_EDIT_CANDLE_DEFAULT_STEPS
    };
    resolve_advanced_or_manifest_u32(request, "steps", default, 1..=100)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → the Raw default (3.5).
fn krea_edit_candle_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        KREA_EDIT_CANDLE_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// Load the Krea edit reference set: the 1..=2 references (plural `referenceAssetIds`, else the single
/// `sourceAssetId`), each pre-fit to the render W×H (crop / pad / outpaint→pad; `stretch` keeps the legacy
/// resize). `render_edit` VAE-encodes each at the target resolution, so pre-fitting keeps an off-aspect
/// source from stretching. Errors if no source. Shares the geometry with the other edit lanes
/// ([`fit_engine_image`]).
fn load_krea_edit_candle_references(
    request: &ImageRequest,
    project_path: &Path,
    settings: &Settings,
    width: u32,
    height: u32,
) -> WorkerResult<Vec<Image>> {
    let ids = krea_edit_candle_reference_ids(request);
    if ids.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires a source image (sourceAssetId).".to_owned(),
        ));
    }
    let mut references = Vec::with_capacity(ids.len());
    for id in &ids {
        let source =
            load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
        let fitted = if request.fit_mode == "stretch" {
            source
        } else {
            fit_engine_image(source, width, height, &request.fit_mode)?
        };
        references.push(fitted);
    }
    Ok(references)
}

/// Flat telemetry recorded on candle Krea edit assets (parity with the macOS `krea_edit_raw_settings`).
fn krea_edit_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw.insert(
        "editEngine".to_owned(),
        Value::String(KREA_EDIT_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// Real candle Krea 2 edit generation: resolve the Raw weights + source references on the async side,
/// pre-fit each to the render geometry, then load the Krea components (the edit LoRA folded into the DiT)
/// plus the edit components (Qwen3-VL vision tower and VAE encoder) once on the blocking thread and render
/// each image — `request.count` edits of the same reference set, each its own seed. Mirrors
/// [`generate_candle_flux2_edit_stream`]'s blocking-thread + streamed-events shape and reuses
/// [`consume_gen_events`]; differs in the required edit LoRA (R5) and the direct pipeline-function API
/// (`render_edit` with `count = 1` per streamed item).
async fn generate_candle_krea_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    if !krea_edit_candle_mode(request) {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires edit_image mode + a source image".to_owned(),
        ));
    }
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Krea 2 Raw weights not found".to_owned()))?;

    // R5 (epic 10871): the base cannot edit without the edit LoRA — its in-context/grounded source
    // conditioning is inert without the trained weights (a VAE-only render is off-distribution, not
    // shippable). Require an `image_edit`-role LoRA (the builtin `krea2_identity_edit`). Checked before
    // loading weights so the error is fast + clear (parity with the macOS lane).
    if !krea_edit_candle_has_lora(request) {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires the Krea 2 Identity Edit LoRA (or another image-edit LoRA): without \
             it the source-image conditioning is inert. Select it in the LoRA picker."
                .to_owned(),
        ));
    }

    let steps = krea_edit_candle_steps(request);
    // Turbo edit is CFG-free — `render_edit(distilled = true)` forces guidance 0 and runs the few-step
    // `turbo_schedule`; Raw honors the resolved guidance on the full-CFG loop (sc-11640).
    let distilled = krea_edit_candle_distilled(request);
    let guidance = if distilled {
        0.0
    } else {
        krea_edit_candle_guidance(request)
    };
    let negative = request.negative_prompt.clone();
    // The selected LoRAs → adapter specs (the edit LoRA + any user LoRAs), folded into the DiT at load.
    let adapters = resolve_adapters(request, settings)?;
    let adapter_count = adapters.len();
    // Telemetry `repo` — the manifest `repo` else the Krea 2 Raw default.
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("SceneWorks/krea-2-raw-mlx")
        .to_owned();

    let (width, height) = (request.width, request.height);
    let references = load_krea_edit_candle_references(request, project_path, settings, width, height)?;
    let raw_settings =
        krea_edit_candle_raw_settings(request, &repo, steps, guidance, references.len());

    // Per-image work items: (seed, prompt) — `request.count` edits of the same reference set.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "krea_edit",
        adapter_count,
        move || {
            let device = candle_gen::default_device()
                .map_err(|error| WorkerError::Engine(format!("Krea edit device init: {error}")))?;
            // Load the Krea Raw components with the edit LoRA merged into the DiT (R5), plus the edit-only
            // components (Qwen3-VL vision tower + VAE encoder) — both cached for the whole batch.
            let comps = candle_gen_krea::pipeline::load_components(
                &weights_dir,
                &device,
                &adapters,
                None,
            )
            .map_err(|error| WorkerError::Engine(format!("Krea edit load failed: {error}")))?;
            let edit = candle_gen_krea::pipeline::load_edit_components(&weights_dir, &device)
                .map_err(|error| {
                    WorkerError::Engine(format!("Krea edit components load failed: {error}"))
                })?;
            Ok(((comps, edit, device), references))
        },
        move |(components, references), tx, cancel| {
            let (comps, edit, device) = components;
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                // One image per streamed item: `render_edit` batches on `count`, so pass `count = 1` and
                // the item's seed. The scene/person references are passed directly (not via
                // `conditioning`) in fixed order.
                let req = gen_core::GenerationRequest {
                    prompt,
                    // The worker's `negative_prompt` is a plain String; the engine wants `Option` (empty
                    // ⇒ no user negative, the Raw CFG uncond branch falls back to "").
                    negative_prompt: if negative.trim().is_empty() {
                        None
                    } else {
                        Some(negative.clone())
                    },
                    width,
                    height,
                    count: 1,
                    seed: Some(seed as u64),
                    steps: Some(steps),
                    guidance: Some(guidance),
                    cancel: cancel.clone(),
                    ..Default::default()
                };
                let result = candle_gen_krea::pipeline::render_edit(
                    &comps,
                    &edit,
                    &req,
                    &references,
                    distilled,
                    &device,
                    &mut *on_progress,
                );
                let mut images = match result {
                    Ok(images) => images,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "Krea edit generation failed: {error}"
                        )));
                    }
                };
                let image = images.pop().ok_or_else(|| {
                    WorkerError::Engine("Krea edit produced no image".to_owned())
                })?;
                Ok(Some((seed, image.width, image.height, image.pixels)))
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
        KREA_EDIT_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
