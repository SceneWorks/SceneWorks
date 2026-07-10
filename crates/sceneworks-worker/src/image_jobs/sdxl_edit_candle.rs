// Candle (Windows/CUDA) SDXL img2img / inpaint / outpaint edit route (sc-5487, epic 5480) — pixel-
// conditioned editing on SDXL/RealVisXL off-Mac via `candle_gen_sdxl::SdxlEdit`. The edit sibling of the
// candle SDXL IP-Adapter lane (sdxl_ipadapter.rs): the same SDXL base resolution + stream plumbing, but
// a SOURCE (and optional mask) instead of a reference, across the three edit sub-modes.
//
// **Candle-only.** macOS keeps the MLX SDXL advanced path (sdxl.rs `SdxlSubMode::{Edit,Inpaint,Outpaint}`);
// the candle `SdxlEdit` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle
// build (the `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs`
// module, so it shares that module's imports (ImageRequest/Settings/WorkerResult/`advanced`/
// `load_reference_image`/`huggingface_snapshot_dir`/`resolve_app_managed_model_dir`/`resolve_seed`/
// `start_gen_stream`/`drive_gen_items`/`consume_gen_events`/`non_empty`/`gen_core`/… all in scope).

/// img2img strength for a plain SDXL edit (torch `SdxlDiffusersAdapter` default 0.6).
const SDXL_EDIT_CANDLE_EDIT_STRENGTH: f32 = 0.6;
/// Strength for masked inpaint / outpaint (torch default 0.85 — the repaint region is mostly regenerated).
const SDXL_EDIT_CANDLE_INPAINT_STRENGTH: f32 = 0.85;
/// Denoise steps default (SDXL production).
const SDXL_EDIT_CANDLE_DEFAULT_STEPS: u32 = 30;
/// CFG default — base SDXL uses ~7; 5.0 matches the candle SDXL reference-conditioned envelope.
const SDXL_EDIT_CANDLE_DEFAULT_GUIDANCE: f32 = 5.0;
/// The adapter/engine id recorded on candle SDXL edit assets + telemetry (distinct from the txt2img
/// `candle_sdxl` and the `candle_sdxl_ipadapter` lanes).
const SDXL_EDIT_CANDLE_ENGINE: &str = "candle_sdxl_edit";

/// SDXL model ids the candle edit route accepts (the txt2img-eligible SDXL family). Must stay in
/// lockstep with `jobs_store::routing::candle::image_request_candle_lane`'s `sdxl_edit_candle_eligible`
/// guard — a model the router sends here but this rejects lands in a lane that then refuses it.
fn is_sdxl_edit_candle_model(model: &str) -> bool {
    matches!(
        model,
        "sdxl" | "realvisxl" | "illustrious_xl_v1" | "illustrious_xl_v2"
    )
}

/// Default SDXL base repo for a model id when the manifest omits `repo`.
///
/// `sdxl` and `realvisxl` name FLAT upstream diffusers snapshots. Illustrious has no such upstream —
/// OnomaAI ship a single-file LDM checkpoint — so it names its tiered turnkey, and
/// `dense_tier_subdir` descends into the dense `bf16/` tier (sc-10614). This lane is dense-only:
/// `IMAGE_MODEL_CAPS` marks the whole SDXL family `candle_quant: false`.
fn sdxl_edit_candle_default_repo(model: &str) -> &'static str {
    match model {
        "realvisxl" => "SG161222/RealVisXL_V5.0",
        "illustrious_xl_v1" => "SceneWorks/illustrious-xl-v1-mlx",
        "illustrious_xl_v2" => "SceneWorks/illustrious-xl-v2-mlx",
        _ => "stabilityai/stable-diffusion-xl-base-1.0",
    }
}

/// Which SDXL edit sub-mode a request maps onto — the candle subset of the macOS `sdxl_sub_mode` (the
/// IP-Adapter case is the separate sdxl_ipadapter lane). `None` = not a candle SDXL edit job.
#[derive(Clone, Copy)]
enum SdxlEditCandleMode {
    /// Plain img2img (source init only).
    Edit,
    /// Masked inpaint (source init + mask).
    Inpaint,
    /// Outpaint = inpaint with a worker-built border mask (+ optional user-mask union).
    Outpaint,
}

/// Classify a candle SDXL edit job: an `edit_image` job with a source. Outpaint wins over a plain mask
/// when `fit_mode == "outpaint"` (matching the torch path); a mask without outpaint is inpaint; neither
/// is a plain img2img edit. Mirrors the macOS `sdxl_sub_mode` (minus the `Ip` case).
fn sdxl_edit_candle_mode(request: &ImageRequest) -> Option<SdxlEditCandleMode> {
    if request.mode != "edit_image" || !non_empty(&request.source_asset_id) {
        return None;
    }
    if request.fit_mode == "outpaint" {
        return Some(SdxlEditCandleMode::Outpaint);
    }
    if non_empty(&request.mask_asset_id) {
        return Some(SdxlEditCandleMode::Inpaint);
    }
    Some(SdxlEditCandleMode::Edit)
}

/// Resolve the SDXL base snapshot for the edit route: an explicit `modelPath` dir (advanced or manifest)
/// wins, else the HF cache snapshot for the manifest `repo` (default by model id). `None` means the base
/// is not present locally, so the job is not candle-runnable (falls through to torch). Mirrors
/// `resolve_sdxl_ipadapter_base`.
fn resolve_sdxl_edit_candle_base(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if let Some(path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
    {
        return resolve_app_managed_model_dir(settings, &path, "SDXL edit modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| sdxl_edit_candle_default_repo(&request.model));
    // A tiered turnkey re-host has no component tree at its root — descend to the dense `bf16/`
    // tier. Flat upstream diffusers snapshots pass through untouched (sc-10614).
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo).map(dense_tier_subdir))
}

/// True when this is a candle-eligible SDXL edit job: an sdxl-family `edit_image` job with a source (an
/// img2img / inpaint / outpaint shape, NOT a pure reference — that is the sdxl_ipadapter lane) whose base
/// resolves locally. Mirrors `jobs_store::sdxl_edit_candle_eligible` so the worker and router agree.
fn sdxl_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_sdxl_edit_candle_model(&request.model)
        && sdxl_edit_candle_mode(request).is_some()
        && matches!(
            resolve_sdxl_edit_candle_base(request, settings),
            Ok(Some(_))
        )
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=80) → manifest `steps` → default (30).
fn sdxl_edit_candle_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", SDXL_EDIT_CANDLE_DEFAULT_STEPS, 1..=80)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (5.0), clamped.
fn sdxl_edit_candle_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        SDXL_EDIT_CANDLE_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

// Fit/letterbox geometry is the SHARED `fit_engine_image` (base.rs) — the same helper the macOS
// SDXL edit lane and the candle Z-Image edit lane use — rather than a per-lane twin (sc-8824). Its
// pad/outpaint arm and `gen_core::imageops::outpaint_border_mask` both call
// `gen_core::imageops::contain_box`, so the letterboxed source and the outpaint mask stay pixel-
// aligned with no per-lane rounding drift.

/// Flat telemetry recorded on candle SDXL edit assets (the sub-mode + strength that drove it; parity with
/// the macOS SDXL advanced recipe keys).
fn sdxl_edit_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    strength: f32,
    mode_tag: &str,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("strength".to_owned(), json!(strength));
    raw.insert("editMode".to_owned(), Value::String(mode_tag.to_owned()));
    raw.insert(
        "editEngine".to_owned(),
        Value::String(SDXL_EDIT_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// Load the SDXL edit source asset (the `sourceAssetId` is required for every edit sub-mode) as an engine
/// [`Image`].
fn load_sdxl_edit_source(
    request: &ImageRequest,
    project_path: &Path,
    settings: &Settings,
) -> WorkerResult<Image> {
    let source_id = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("SDXL edit requires a source image".to_owned())
        })?;
    load_reference_image(
        &settings.data_dir,
        &request.project_id,
        source_id,
        project_path,
    )
}

/// Real candle SDXL edit generation: resolve the source (+ mask) + base on the async side, build the
/// sub-mode's `(source, mask)` (img2img: a fitted source; inpaint: fitted source + mask; outpaint: the
/// padded canvas + the border mask, unioned with any user mask), then load `SdxlEdit` once + generate
/// each image on the blocking thread. `request.count` images, each its own seed. `generate` takes `&self`
/// (no per-call UNet mutation), so the per-item closure needs no `mut`. Reuses [`consume_gen_events`].
async fn generate_candle_sdxl_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let sdxl_base = resolve_sdxl_edit_candle_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("SDXL edit base (SDXL/RealVisXL) not found".to_owned())
    })?;
    let mode = sdxl_edit_candle_mode(request).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "SDXL edit requires edit_image mode + a source image".to_owned(),
        )
    })?;
    // Per-generation PiD decode (epic 7840, sc-8044) + output tier (sc-10054), resolved BEFORE the
    // source/mask fit so a 2K tier sizes the effective base and the edit source + mask are fit to THAT
    // base (source, mask, and latent stay aligned). `use_pid`/`with_pid` stay paired at the load below.
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();
    let (width, height) =
        pid_effective_dims(request.width, request.height, use_pid, pid_output_tier(request));
    let source = load_sdxl_edit_source(request, project_path, settings)?;

    let is_inpaint = matches!(
        mode,
        SdxlEditCandleMode::Inpaint | SdxlEditCandleMode::Outpaint
    );
    let strength = advanced::f32_clamped(
        &request.advanced,
        "strength",
        if is_inpaint {
            SDXL_EDIT_CANDLE_INPAINT_STRENGTH
        } else {
            SDXL_EDIT_CANDLE_EDIT_STRENGTH
        },
        0.0..=1.0,
    );

    // Build the sub-mode's (source, optional mask) at the render geometry. The provider re-resizes
    // internally, so the source/mask only need to be at `width`×`height` with aligned geometry.
    let (gen_source, gen_mask, mode_tag): (Image, Option<Image>, &str) = match mode {
        SdxlEditCandleMode::Edit => (
            fit_engine_image(source, width, height, &request.fit_mode)?,
            None,
            "edit",
        ),
        SdxlEditCandleMode::Inpaint => {
            let src = fit_engine_image(source, width, height, &request.fit_mode)?;
            let mask_id = request
                .mask_asset_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("SDXL inpaint requires a mask image".to_owned())
                })?;
            let mask = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                mask_id,
                project_path,
            )?;
            let mask = fit_engine_image(mask, width, height, &request.fit_mode)?;
            (src, Some(mask), "inpaint")
        }
        SdxlEditCandleMode::Outpaint => {
            // Pad the source onto the render canvas; the white border is the region to regenerate.
            // Fit + mask MUST share `gen_core::imageops::contain_box` (sc-8824): `fit_engine_image`'s
            // "outpaint"/pad arm and `outpaint_border_mask` both call it, so the kept rect and the
            // mask's keep rect are pixel-identical.
            let (src_w, src_h) = (source.width, source.height);
            let canvas = fit_engine_image(source, width, height, "outpaint")?;
            // White = regenerate (the padded border), black = keep (the centered source).
            let mut mask = gen_core::imageops::outpaint_border_mask(src_w, src_h, width, height);
            if non_empty(&request.mask_asset_id) {
                // Union any user edit region into the border (white wins) — pad-fit the user mask onto
                // the same contained geometry first.
                let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
                let user_mask = load_reference_image(
                    &settings.data_dir,
                    &request.project_id,
                    mask_id,
                    project_path,
                )?;
                let user_mask = fit_engine_image(user_mask, width, height, "pad")?;
                mask = gen_core::imageops::union_masks(&mask, &user_mask).map_err(|error| {
                    WorkerError::Engine(format!("SDXL outpaint mask union failed: {error}"))
                })?;
            }
            (canvas, Some(mask), "outpaint")
        }
    };

    let steps = sdxl_edit_candle_steps(request);
    let guidance = sdxl_edit_candle_guidance(request);
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| sdxl_edit_candle_default_repo(&request.model))
        .to_owned();
    // `pid_weights`/`use_pid`/`width`/`height` were resolved above (ahead of the source+mask fit) so the
    // PiD output tier (sc-10054) could size the effective base. SDXL edit composes the SDXL VAE, so it
    // shares the one `sdxl` student; the inpaint/outpaint mask blend ends in a single decode, so PiD sees
    // the same final latent as the VAE path — the output is just the tier's 2K/4K.
    let mut raw_settings =
        sdxl_edit_candle_raw_settings(request, &repo, steps, guidance, strength, mode_tag);
    // Mark PiD output on the sidecar (epic 7840): the NSCLv1 non-commercial restriction flows to PiD-
    // decoded output. Record whether PiD ACTUALLY ran (not merely whether it was requested) — a request
    // that opted in but has no cached snapshots decodes on the native VAE.
    raw_settings.insert("usePid".to_owned(), Value::Bool(use_pid));

    // Per-image work items: (seed, prompt) — `request.count` edits of the same source.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let negative = request.negative_prompt.clone();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "sdxl_edit",
        0,
        move || {
            let model = SdxlEdit::load(&SdxlEditPaths { sdxl_base })
                .map_err(|error| WorkerError::Engine(format!("SDXL edit load failed: {error}")))?;
            // Attach the optional PiD decoder (sc-8044): `Some` only when this generation opted in AND the
            // snapshots are cached, so a native-VAE edit is a no-op here.
            let model = match &pid_weights {
                Some(pid) => model.with_pid(pid).map_err(|error| {
                    WorkerError::Engine(format!("SDXL edit PiD decoder load failed: {error}"))
                })?,
                None => model,
            };
            Ok((model, gen_source, gen_mask))
        },
        move |(model, source, mask), tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let req = SdxlEditRequest {
                    prompt,
                    negative: negative.clone(),
                    width,
                    height,
                    steps: steps as usize,
                    guidance,
                    strength,
                    seed: seed as u64,
                    // PiD opt-in (sc-8044): in lockstep with the `with_pid` load above — the engine errors
                    // if set without a loaded student, so `use_pid` is `pid_weights.is_some()`.
                    use_pid,
                    cancel: cancel.clone(),
                };
                let result = match &mask {
                    Some(mask) => model.generate_masked(&req, &source, mask, &mut *on_progress),
                    None => model.generate(&req, &source, &mut *on_progress),
                };
                let out = match result {
                    Ok(out) => out,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "SDXL edit generation failed: {error}"
                        )));
                    }
                };
                Ok(Some((seed, out.width, out.height, out.pixels)))
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
        SDXL_EDIT_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
