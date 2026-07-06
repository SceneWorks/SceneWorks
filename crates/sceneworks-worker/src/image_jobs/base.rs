// Image fit/crop/pad geometry shared by the MLX edit handlers (flux2/qwen/sdxl/kolors/sensenova/
// zimage), the candle edit handlers (*_edit_candle.rs), and the video I2V resolve paths
// (video_jobs.rs). Kept in base.rs — included on macOS AND the `backend-candle` lane (and nowhere
// else) — so `crate::image_jobs::fit_engine_image` resolves on exactly the lanes that call it. Moved
// here from the macOS-only flux2.rs (sc-6231; the sc-6139 fit-mode refactor left it macOS-gated, which
// broke the candle build because video_jobs.rs / the candle edit handlers call it). No `#[cfg]` here:
// availability follows base.rs's own include cfg, which matches the callers'.

/// Resize an RGB image to exactly `width`×`height` honoring `mode` without distorting it
/// (parity with Python `fit_image`, RGB path only — no inpaint mask exists on the MLX
/// FLUX.2 edit path, so `outpaint` degrades to `pad` geometry):
///   - `crop`:    scale to COVER (short edge fits), center-crop the overflow.
///   - `pad`/`outpaint`: scale to CONTAIN (long edge fits), center on a black canvas.
///   - `stretch`: legacy non-aspect-preserving resize.
///
/// The pad/outpaint arm's contain geometry is the engine's [`gen_core::imageops::contain_box`]
/// (sc-8824) — the SINGLE source of truth shared with the outpaint mask
/// ([`gen_core::imageops::outpaint_border_mask`] calls the same `contain_box`), so the letterboxed
/// kept-rect and the mask's keep-rect are pixel-identical. It rounds half-to-even in f64 (matching
/// Python `round`) and returns i32 offsets; the old local copy rounded half-away-from-zero in f32,
/// which could disagree by a pixel at an exact `.5` and desync fit vs. mask on outpaint edges.
fn fit_rgb(source: &image::RgbImage, width: u32, height: u32, mode: &str) -> image::RgbImage {
    use image::imageops::FilterType::Lanczos3;
    let width = width.max(1);
    let height = height.max(1);
    let (src_w, src_h) = (source.width(), source.height());
    match mode {
        "stretch" => image::imageops::resize(source, width, height, Lanczos3),
        "crop" => {
            let ratio = (width as f32 / src_w as f32).max(height as f32 / src_h as f32);
            // Ceil so the scaled image always fully covers the target before cropping.
            let new_w = width.max((src_w as f32 * ratio).ceil() as u32);
            let new_h = height.max((src_h as f32 * ratio).ceil() as u32);
            let resized = image::imageops::resize(source, new_w, new_h, Lanczos3);
            let left = (new_w - width) / 2;
            let top = (new_h - height) / 2;
            image::imageops::crop_imm(&resized, left, top, width, height).to_image()
        }
        // "pad" / "outpaint": contain + center on a black canvas (letterbox). The engine's
        // `contain_box` is the shared geometry the outpaint mask also uses, so fit + mask agree.
        _ => {
            let (new_w, new_h, left, top) =
                gen_core::imageops::contain_box(src_w, src_h, width, height);
            let resized = image::imageops::resize(source, new_w.max(1), new_h.max(1), Lanczos3);
            let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
            image::imageops::overlay(&mut canvas, &resized, left as i64, top as i64);
            canvas
        }
    }
}

/// Fit an engine [`Image`] (RGB8) to `width`×`height` by `mode` via [`fit_rgb`].
/// `pub(crate)` so the video I2V resolve paths (`video_jobs.rs`, sc-6139) can pre-fit a
/// starting image to the output dims with the same crop/pad geometry as the image-edit lane.
pub(crate) fn fit_engine_image(
    source: Image,
    width: u32,
    height: u32,
    mode: &str,
) -> WorkerResult<Image> {
    let rgb =
        image::RgbImage::from_raw(source.width, source.height, source.pixels).ok_or_else(|| {
            WorkerError::InvalidPayload("edit source buffer size mismatch".to_owned())
        })?;
    let fitted = fit_rgb(&rgb, width, height, mode);
    Ok(Image {
        width: fitted.width(),
        height: fitted.height(),
        pixels: fitted.into_raw(),
    })
}

#[cfg(target_os = "macos")]
fn mlx_available(request: &ImageRequest, settings: &Settings) -> bool {
    mlx_model(&request.model).is_some()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageRoute {
    ZImageControl,
    ZImageBaseControl,
    QwenControl,
    KolorsControl,
    Flux1DevControl,
    Flux2DevControl,
    Flux2Edit,
    QwenEdit,
    InstantId,
    PulidFlux,
    SdxlAdvanced,
    SensenovaEdit,
    Bernini,
    Mlx,
}

#[cfg(target_os = "macos")]
fn resolve_image_route(request: &ImageRequest, settings: &Settings) -> Option<ImageRoute> {
    if zimage_control_available(request, settings) {
        Some(ImageRoute::ZImageControl)
    } else if zimage_base_control_available(request, settings) {
        // Base (non-distilled, full-CFG) Z-Image strict control (advanced.poses on `z_image`) →
        // base Fun-Controlnet-Union (`z_image_control`). The base mirror of the Turbo control arm
        // above; keyed on the base model id so the Turbo path is untouched (sc-8251).
        Some(ImageRoute::ZImageBaseControl)
    } else if qwen_control_available(request, settings) {
        Some(ImageRoute::QwenControl)
    } else if kolors_control_available(request, settings) {
        Some(ImageRoute::KolorsControl)
    } else if flux1_dev_control_available(request, settings) {
        // FLUX.1-dev strict control (advanced.poses on flux_dev) → Shakker Union-Pro-2.0. Wins over the
        // PuLID-FLUX / generic MLX arms below: a flux_dev pose job is the real ControlNet path (sc-8244).
        Some(ImageRoute::Flux1DevControl)
    } else if flux2_dev_control_available(request, settings) {
        // FLUX.2-dev strict pose (advanced.poses) → Fun-Controlnet-Union. Wins over the edit/
        // best-effort pose tier below (`flux2_edit_available` needs a reference; a flux2_dev pose
        // job is the real ControlNet path, with the reference an opt-in img2img-init).
        Some(ImageRoute::Flux2DevControl)
    } else if flux2_edit_available(request, settings) {
        Some(ImageRoute::Flux2Edit)
    } else if qwen_edit_available(request, settings) {
        Some(ImageRoute::QwenEdit)
    } else if instantid_available(request, settings) {
        Some(ImageRoute::InstantId)
    } else if pulid_flux_available(request, settings) {
        Some(ImageRoute::PulidFlux)
    } else if sdxl_advanced_available(request, settings) {
        Some(ImageRoute::SdxlAdvanced)
    } else if sensenova_edit_available(request, settings) {
        Some(ImageRoute::SensenovaEdit)
    } else if bernini_image_available(request, settings) {
        // Bernini still-image companion (sc-5424): t2i / i2i on the `bernini_image` id. Must win
        // over the generic `mlx_available` arm below — `bernini_image` is in MODEL_TABLE (so
        // `mlx_available` would match it), but the generic `generate_stream` leaves `frames`/
        // `video_mode` unset, which the engine treats as a multi-frame video request.
        Some(ImageRoute::Bernini)
    } else if mlx_available(request, settings) {
        Some(ImageRoute::Mlx)
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
impl ImageRoute {
    fn image_count(self, request: &ImageRequest, settings: &Settings) -> u32 {
        match self {
            ImageRoute::ZImageControl
            | ImageRoute::ZImageBaseControl
            | ImageRoute::QwenControl
            | ImageRoute::KolorsControl
            | ImageRoute::Flux1DevControl
            | ImageRoute::Flux2DevControl => pose_entries(request).len() as u32,
            ImageRoute::Flux2Edit | ImageRoute::QwenEdit => grouped_edit_image_count(request),
            ImageRoute::InstantId => instantid_image_count(request, settings),
            ImageRoute::SensenovaEdit => match edit_grouping(request) {
                EditGrouping::Angles => CHARACTER_ANGLE_SET_ORDER.len() as u32,
                // SenseNova has no strict-pose (ControlNet) path; pose jobs are excluded
                // upstream, so any residual grouping preserves the requested image count.
                EditGrouping::Poses(_) | EditGrouping::Plain => request.count,
            },
            // PuLID-FLUX is one identity image per seed (no angle/pose grouping) — like the base
            // MLX + SDXL-advanced + Bernini paths, the effective count is the requested count.
            ImageRoute::PulidFlux
            | ImageRoute::SdxlAdvanced
            | ImageRoute::Bernini
            | ImageRoute::Mlx => request.count,
        }
    }
}

/// The candle (Windows/CUDA/Linux) image engine a `run_image_generate_job` request routes to — the
/// candle-lane sibling of [`ImageRoute`] (sc-8828, F-026). Each variant maps 1:1 to a bespoke candle
/// stream handler with the uniform `(api, settings, job, &plan, &project_path, backend, &mut asset_writes)`
/// signature; [`CandleImageRoute::PoseReject`] is the no-silent-T2I reject arm (sc-5968). Every arm is
/// gated on `settings.backend_candle_enabled`; when off (default) the resolver returns `None` so the job
/// falls through to the stub, exactly as before.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandleImageRoute {
    /// InstantID identity (sc-5491) — the off-Mac sibling of `ImageRoute::InstantId`. Checked first
    /// because `instantid_realvisxl` is not an `is_candle_engine` txt2img id.
    InstantId,
    /// SDXL img2img / inpaint / outpaint edit (sc-5487).
    SdxlEdit,
    /// FLUX.2-klein reference / img2img edit (sc-5487).
    Flux2Edit,
    /// Qwen-Image-Edit reference / dual-latent edit (sc-5487).
    QwenEdit,
    /// Z-Image img2img / edit (sc-6595).
    ZimageEdit,
    /// Z-Image identity-init for Image Studio "With Character" (sc-8409).
    ZimageIdentity,
    /// SDXL IP-Adapter-Plus reference conditioning (sc-5488).
    SdxlIpAdapter,
    /// Kolors IP-Adapter-Plus reference conditioning (sc-5488).
    KolorsIpAdapter,
    /// FLUX XLabs IP-Adapter reference conditioning (sc-5872).
    FluxIpAdapter,
    /// PuLID-FLUX face identity (sc-5492).
    Pulid,
    /// Qwen-Image strict-pose ControlNet (sc-5489).
    QwenControl,
    /// Kolors strict-pose ControlNet (sc-5489).
    KolorsControl,
    /// Z-Image strict-pose Fun-ControlNet (sc-5489).
    ZimageControl,
    /// FLUX.2-dev strict-pose Fun-Controlnet-Union (sc-7736).
    Flux2Control,
    /// FLUX.1-dev strict-control Shakker Union-Pro-2.0 (sc-8412).
    Flux1Control,
    /// A strict-pose job on a candle model with NO pose lane → reject loudly, never silent T2I (sc-5968).
    PoseReject,
    /// A plain candle txt2img engine id → `generate_candle_stream`.
    CandleTxt2Img,
}

/// Run the candle image dispatch predicate ladder ONCE and return the [`CandleImageRoute`] (or `None`
/// when candle is disabled / no candle engine matches → the job stubs). Mirrors the historical inline
/// `else if settings.backend_candle_enabled && <predicate>` ladder EXACTLY — same predicate order,
/// same `backend_candle_enabled` gating, same handler per family — so routing is byte-identical
/// (sc-8828). Pure decision: no I/O, no generation.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_image_route(
    request: &ImageRequest,
    settings: &Settings,
) -> Option<CandleImageRoute> {
    if !settings.backend_candle_enabled {
        return None;
    }
    // Order matches the historical ladder: the edit / reference / identity / control lanes are all
    // checked BEFORE the generic `is_candle_engine` txt2img arm (they share candle txt2img model ids, so
    // without diverting first they'd be silently rendered as plain txt2img, dropping the source / poses).
    if instantid_available(request, settings) {
        Some(CandleImageRoute::InstantId)
    } else if sdxl_edit_candle_available(request, settings) {
        Some(CandleImageRoute::SdxlEdit)
    } else if flux2_edit_candle_available(request, settings) {
        Some(CandleImageRoute::Flux2Edit)
    } else if qwen_edit_candle_available(request, settings) {
        Some(CandleImageRoute::QwenEdit)
    } else if zimage_edit_candle_available(request, settings) {
        Some(CandleImageRoute::ZimageEdit)
    } else if zimage_identity_candle_available(request, settings) {
        Some(CandleImageRoute::ZimageIdentity)
    } else if sdxl_ipadapter_available(request, settings) {
        Some(CandleImageRoute::SdxlIpAdapter)
    } else if kolors_ipadapter_available(request, settings) {
        Some(CandleImageRoute::KolorsIpAdapter)
    } else if flux_ipadapter_available(request, settings) {
        Some(CandleImageRoute::FluxIpAdapter)
    } else if pulid_candle_available(request, settings) {
        Some(CandleImageRoute::Pulid)
    } else if qwen_control_available(request, settings) {
        Some(CandleImageRoute::QwenControl)
    } else if kolors_control_available(request, settings) {
        Some(CandleImageRoute::KolorsControl)
    } else if zimage_control_available(request, settings) {
        Some(CandleImageRoute::ZimageControl)
    } else if flux2_control_candle_available(request, settings) {
        Some(CandleImageRoute::Flux2Control)
    } else if flux1_control_candle_available(request, settings) {
        Some(CandleImageRoute::Flux1Control)
    } else if is_candle_engine(&request.model)
        && !matches!(
            request.model.as_str(),
            "qwen_image" | "kolors" | "z_image_turbo" | "z_image" | "flux2_dev" | "flux_dev"
        )
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
    {
        // No-silent-T2I (sc-5968): a strict-pose job on a candle model with NO pose lane (e.g. sdxl) must
        // be REJECTED, not silently rendered as plain txt2img. Checked BEFORE the txt2img arm below.
        Some(CandleImageRoute::PoseReject)
    } else if is_candle_engine(&request.model) {
        Some(CandleImageRoute::CandleTxt2Img)
    } else {
        None
    }
}

/// How a native edit job batches its iterations (sc-8946 (F-144): renamed from `Flux2Grouping` and
/// moved here from flux2.rs — it is the SHARED grouping for the FLUX.2 / Qwen-Edit / SenseNova-U1 edit
/// lanes, not FLUX.2-specific, so a reader auditing Qwen/SenseNova grouping finds it in base.rs).
#[cfg(target_os = "macos")]
enum EditGrouping {
    /// `count` independent images (per-image seeds), the plain reference/edit path.
    Plain,
    /// The 11-angle Character-Studio set: shared seed, per-angle prompt augment.
    Angles,
    /// The best-effort pose tier: `n` poses, shared seed, `[skeleton, reference]` sets.
    Poses(usize),
}

/// Decide the grouping for a native edit job (parity with the `Mlx*Adapter` decision: pose set >
/// angle set > plain, all gated to `character_image` mode — an `edit_image` job is never grouped).
/// The caller only reaches this with a reference present, so `is_character_image` reduces to the mode
/// check. Shared by the FLUX.2 / Qwen-Edit / SenseNova-U1 edit lanes (sc-8946 moved it from flux2.rs).
#[cfg(target_os = "macos")]
fn edit_grouping(request: &ImageRequest) -> EditGrouping {
    if request.mode != "character_image" {
        return EditGrouping::Plain;
    }
    let poses = pose_entries(request).len();
    if poses > 0 {
        return EditGrouping::Poses(poses);
    }
    if advanced::flag(&request.advanced, "angleSet") {
        return EditGrouping::Angles;
    }
    EditGrouping::Plain
}

/// Upper bound on reference images for a multi-reference edit (sc-6211). Even with the engine's
/// sequence-gated activation chunking (sc-6266), the FLUX.2-dev edit stays activation-bound: 4
/// references at 1024² peak ~93 GB and 5 would exceed the 96 GB floor (measured). The per-machine
/// `flux2_dev_edit_memory_guard` rejects over-budget combinations with an actionable message; this
/// caps absurd inputs (and bounds the DiT sequence) before that.
#[cfg(target_os = "macos")]
const MAX_EDIT_REFERENCES: usize = 4;

/// Reference asset ids for a native edit (sc-8946 moved it from flux2.rs — shared by the FLUX.2 /
/// SenseNova-via-grouping lanes, not FLUX.2-specific). The FLUX.2-dev multi-image picker (sc-6211)
/// sends the plural `referenceAssetIds` — take all of them in order, capped at [`MAX_EDIT_REFERENCES`].
/// With no plural list it falls back to the single-reference flows: the character `referenceAssetId`,
/// else the Image-Edit `sourceAssetId` (edit_image mode). Mirrors the Python
/// `ref_id = referenceAssetId or (sourceAssetId if edit_image)`, plus the new multi-reference set.
#[cfg(target_os = "macos")]
fn edit_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        // Parsed list is already trimmed + non-empty (sceneworks-core `string_list`).
        return request
            .reference_asset_ids
            .iter()
            .take(MAX_EDIT_REFERENCES)
            .cloned()
            .collect();
    }
    if let Some(id) = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
    }
    if request.mode == "edit_image" {
        if let Some(id) = request
            .source_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
        {
            return vec![id.to_owned()];
        }
    }
    Vec::new()
}

#[cfg(target_os = "macos")]
fn grouped_edit_image_count(request: &ImageRequest) -> u32 {
    match edit_grouping(request) {
        EditGrouping::Angles => CHARACTER_ANGLE_SET_ORDER.len() as u32,
        EditGrouping::Poses(count) => count as u32,
        EditGrouping::Plain => request.count,
    }
}

/// The HuggingFace repo for the model: the manifest entry's `repo` wins, else the
/// family default. Shared by the MLX path and the candle lane (sc-5096).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn model_repo(request: &ImageRequest, model: &ResolvedModel) -> String {
    request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(model.default_repo())
        .to_owned()
}

/// The separate `SceneWorks/ideogram-4` repo that hosts Ideogram 4's bf16 tree under `bf16/`, the
/// selectable full-precision tier (sc-8513). The `q4/`/`q8/` packed turnkey lives in
/// `SceneWorks/ideogram-4-mlx`; bf16 is resolved from THIS repo rather than duplicated. sc-9650 wires it
/// on the candle lane too: the `bf16/` subdir is in the SAME single-file `transformer/model.safetensors`
/// layout the candle loader reads, and `linear_detect` takes its dense arm (no `.scales` sibling), so
/// candle dense-loads bf16 exactly like macOS/MLX — while the packed q4/q8 tiers still come from the
/// `-mlx` turnkey via `ideogram_model_subdir`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const IDEOGRAM_BF16_REPO: &str = "SceneWorks/ideogram-4";

/// Resolve the weights snapshot directory: an explicit `modelPath` dir wins, else the
/// HuggingFace cache snapshot for the model repo. `None` when the model is not a known
/// engine family or its snapshot is absent. Available on the candle lane too (sc-5501): the
/// off-Mac SenseNova-U1 VQA / interleave handlers resolve their snapshot through it.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) fn resolve_weights_dir(
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
        return resolve_app_managed_model_dir(settings, &path, "Image modelPath").map(Some);
    }
    let Some(model) = mlx_model(&request.model) else {
        return Ok(None);
    };
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, &model_repo(request, &model));
    // Ideogram 4 ships a turnkey with packed `q4/` (default) + `q8/` self-contained subdirs; point
    // the engine at the chosen quant's subdir rather than the repo root (epic 4725 / sc-5992),
    // mirroring the LTX bundle pattern. The packed weights auto-detect their quant on load. The
    // turbo variant (mlx-gen #488) shares the same turnkey — each subdir also carries the bundled
    // `turbo_lora.safetensors` the `ideogram_4_turbo` engine installs at load.
    if request.model == "ideogram_4" || request.model == "ideogram_4_turbo" {
        // bf16 (sc-8513, epic 8506) is the SHARED `SceneWorks/ideogram-4` repo's `bf16/` subdir — NOT
        // duplicated into the MLX turnkey. When a request opts into bf16 (advanced mlxQuantize<=0) AND it
        // is downloaded, resolve there (the dense weights load with no quantize); else the q4 (default)/q8
        // turnkey subdir. A partial bf16 download falls back rather than half-loading. sc-9650: wired on
        // the candle lane too — the candle Ideogram loader reads the same `transformer/model.safetensors`
        // layout and dense-loads bf16 via `linear_detect` (`resolve_quant` returns None for mlxQuantize<=0,
        // so no on-the-fly quant runs). The packed q4/q8 tiers still resolve from the `-mlx` turnkey below.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        {
            let wants_bf16 = request
                .advanced
                .get("mlxQuantize")
                .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
                .is_some_and(|bits| bits <= 0);
            if wants_bf16 {
                if let Some(bf16) = huggingface_snapshot_dir(&settings.data_dir, IDEOGRAM_BF16_REPO)
                    .map(|root| root.join("bf16"))
                    .filter(|dir| dir.join("transformer/model.safetensors").is_file())
                {
                    return Ok(Some(bf16));
                }
            }
        }
        return Ok(snapshot.map(|root| ideogram_model_subdir(&root, request)));
    }
    // Boogu (epic 6387) ships a turnkey with pre-packed Q8 `base/ turbo/ edit/` subfolders (default) +
    // full-precision `*-bf16/`; point the engine at the variant's subfolder rather than the repo root
    // (the packed weights auto-detect their quant on load).
    if matches!(
        request.model.as_str(),
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit"
    ) {
        return Ok(snapshot.map(|root| boogu_model_subdir(&root, request)));
    }
    // Krea 2 Turbo (epic 7565) ships a turnkey with packed `q8/` (default) + `q4/` self-contained subdirs;
    // point the engine at the chosen quant's subdir rather than the repo root. The packed weights
    // auto-detect their quant on load, so the resolved `spec.quantize` is a no-op on them.
    if request.model == "krea_2_turbo" {
        return Ok(snapshot.map(|root| krea_model_subdir(&root, request)));
    }
    // Catalog-wide quant-matrix models (sc-8513, epic 8506) ship as SceneWorks pre-quantized
    // turnkeys with self-contained `q4/` (default) + `q8/` + `bf16/` subdirs (replacing any
    // install-time convert); point the engine at the chosen tier's subdir rather than the repo root.
    // FLUX.2-dev was the pilot; the rollout registers each model in [`STANDARD_TIER_MODELS`] OR (the
    // sc-8508 manifest-driven form) flags `mlx.standardTierLayout: true` in its catalog entry.
    if uses_standard_tier_layout(request) {
        return Ok(snapshot.map(|root| standard_tier_subdir(&root, request)));
    }
    Ok(snapshot)
}

/// Models that ship the standard SceneWorks quant-matrix turnkey layout: self-contained `q4/`
/// (manifest default) + `q8/` + `bf16/` subdirs, each a complete `from_snapshot`-loadable tree
/// (packed or dense `transformer/` + the dense text encoder(s)/VAE/tokenizer). Registering a model
/// here routes it through [`standard_tier_subdir`] (sc-8513, epic 8506) — the generalization of the
/// FLUX.2-dev pilot's bespoke resolver. Legacy turnkeys with non-standard defaults/variants
/// (Ideogram q4-only, Krea q8-default, Boogu per-variant + on-demand bf16) keep their own resolvers
/// above.
///
/// sc-8508 makes this catalog-driven: [`uses_standard_tier_layout`] also honors a manifest
/// `mlx.standardTierLayout: true` flag, so a NEW quant-matrix model can opt in from the manifest
/// alone. This registry remains the zero-manifest-change path for every already-wired model.
const STANDARD_TIER_MODELS: &[&str] = &[
    "flux2_dev",
    "sd3_5_large",
    "sd3_5_large_turbo",
    "sd3_5_medium",
    // Z-Image (sc-8670, Group-B pilot): turbo + base ship the standard q4/q8/bf16 turnkey; the
    // edit id reuses the turbo turnkey (engine_id z_image_turbo, same repo).
    "z_image_turbo",
    "z_image",
    "z_image_edit",
    // FLUX.2-klein (sc-8711): the two distilled weight variants ship the standard q4/q8/bf16
    // turnkey, but with a DENSE bf16 Qwen3 text encoder in every tier (only the transformer is
    // packed) — so they additionally appear in [`DENSE_TE_TIER_MODELS`], which forces the load
    // Quant to None so the dense TE is never re-quantized. `_true_v2` stays on its install-time
    // single-file→diffusers convert (candle-only) and is not a turnkey yet.
    "flux2_klein_9b",
    "flux2_klein_9b_kv",
    // Qwen-Image (sc-8669, Group-B): base T2I + the two Edit-2511 ids ship the standard
    // q4/q8/bf16 turnkey. Like FLUX.2-klein only the transformer is packed (the Qwen2.5-VL text
    // encoder is skip_quantization, the VAE is all-conv), so the TE/VAE stay dense bf16 in every
    // tier — but, UNLIKE klein, the qwen loader never quantizes the TE regardless of the load
    // Quant, so these do NOT need a DENSE_TE_TIER_MODELS guard: the q4/q8 load-quant is a harmless
    // no-op on the already-packed transformer, and the bf16 tier resolves to Quant::None anyway.
    // `qwen_image_edit_2511` + `_2511_lightning` share one repo (same Edit-2511 checkpoint).
    "qwen_image",
    "qwen_image_edit_2511",
    "qwen_image_edit_2511_lightning",
    // FLUX.1 (sc-8669, Group-B): schnell + dev ship the standard q4/q8/bf16 turnkey. FLUX quantizes
    // all four components (DiT transformer + CLIP + T5 + VAE attention), so the TE is packed too —
    // hence NOT in DENSE_TE_TIER_MODELS (the q4/q8 load-quant is a harmless no-op on already-packed
    // weights, bf16 resolves to Quant::None). Replaces the gated BFL download + install-time quantize.
    "flux_schnell",
    "flux_dev",
    // PuLID-FLUX (sc-9947, epic 8506): the MLX lane's FLUX.1-dev backbone now loads from the SAME
    // `SceneWorks/flux1-dev-mlx` q4/q8/bf16 turnkey as base `flux_dev` (its bespoke `pulid.rs` resolver
    // calls `standard_tier_subdir` directly; `mlx-gen-pulid` delegates the backbone to `load_flux1`, which
    // packed-detects the tier). Registering it here makes `uses_standard_tier_layout` true for that
    // resolver. The candle (Windows/Linux) PuLID lane keeps the upstream dense BFL backbone and never
    // reaches the base tier path, so this is inert there (epic-9083 covers the candle packed lane).
    "pulid_flux_dev",
    // Lens / Lens-Turbo (sc-9092, epic 9083 gap #3): the SceneWorks re-hosted `SceneWorks/lens-mlx` /
    // `SceneWorks/lens-turbo-mlx` turnkeys are standard q4/q8/bf16 tiers (their manifests already flag
    // `mlx.standardTierLayout: true`, so `uses_standard_tier_layout` was already true via the manifest —
    // registering them here is the zero-manifest-change form + documents the candle-lane opt-in). As of
    // the candle-gen packed-load rollout (sc-8799) the candle Lens loader packed-detects the SAME
    // turnkey subdir the macOS path loads, so the ad-hoc `candle_lens_repo` (a separate bf16 diffusers
    // rehost) is retired and both lanes resolve Lens through `standard_tier_subdir`. Lens is the lone
    // candle family that ALSO advertises `supported_quants` (Q4/Q8) today, so `resolve_quant` engages on
    // its candle lane; ideogram/boogu/krea keep their legacy per-family subdir resolvers (non-standard
    // q4-default / per-variant / q8-default layouts) and are NOT registered here.
    "lens",
    "lens_turbo",
    // SANA + SANA-Sprint (sc-8489/sc-8513, epic 8506): the `SceneWorks/Sana_1600M_1024px_mlx` /
    // `Sana_Sprint_1.6B_1024px_mlx` turnkeys ship standard q4/q8/bf16 tiers. mlx-gen #653 packs the
    // Linear-DiT transformer + the Gemma-2 CHI TE and packed-detects on load; the DC-AE VAE stays
    // dense in every tier. Like flux1/qwen (and UNLIKE the dense-TE klein class) the q4/q8 load-quant
    // is a harmless no-op on the already-packed weights and bf16 resolves to Quant::None — so these do
    // NOT need a DENSE_TE_TIER_MODELS guard. The SANA descriptor now advertises supported_quants
    // Q4/Q8 (mlx-gen #654), so `supports_quant()` is true and they flow through the same
    // resolve_quant + reconcile path as every other matrix model (no more no-quant special case).
    "sana_1600m",
    "sana_sprint_1600m",
    // Kolors (sc-9946, epic 8506): the `SceneWorks/kolors-mlx` turnkey ships standard q4/q8/bf16
    // tiers. mlx-gen #659 packs the SDXL-style UNet + the ChatGLM3-6B `ChatGlmLinear` projections
    // and packed-detects on load; the SDXL VAE stays dense in every tier. Like flux1/sana (and
    // UNLIKE the dense-TE klein class) the ChatGLM3 TE is packed, so the q4/q8 load-quant is a
    // harmless no-op on the already-packed weights and bf16 resolves to Quant::None — no
    // DENSE_TE_TIER_MODELS guard. The kolors descriptor already advertises supported_quants Q4/Q8,
    // so it flows through the same resolve_quant + reconcile path as every other matrix model.
    "kolors",
];

/// Standard-tier models whose text encoder ships DENSE bf16 in EVERY tier (epic 8506, sc-8711:
/// quantize the transformer, keep the TE bf16). Their pre-packed transformer self-describes its
/// quant on load, so [`resolve_quant`] must return `None` for them — otherwise the load-time
/// `.quantize()` would re-quantize the dense bf16 TE down to Q4/Q8. Contrast flux2_dev / sd3.5 /
/// z-image, whose text encoders are packed too, so their Q4/Q8 load-quant is a harmless no-op on
/// already-packed weights.
const DENSE_TE_TIER_MODELS: &[&str] = &["flux2_klein_9b", "flux2_klein_9b_kv"];

/// Whether a request's model ships the standard SceneWorks quant-matrix turnkey layout (sc-8508):
/// true when it is registered in [`STANDARD_TIER_MODELS`] OR its manifest entry declares
/// `mlx.standardTierLayout: true`. The manifest flag is the first-class, catalog-driven form of the
/// hardcoded registry (epic 8506) — a new quant-matrix model can opt in from the manifest alone,
/// while the registry keeps every already-wired model working with zero manifest change.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn uses_standard_tier_layout(request: &ImageRequest) -> bool {
    STANDARD_TIER_MODELS.contains(&request.model.as_str())
        || request
            .model_manifest_entry
            .get("mlx")
            .and_then(|mlx| mlx.get("standardTierLayout"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
}

/// Whether a standard-tier request keeps a DENSE bf16 text encoder in every tier (sc-8508): true
/// when it is in [`DENSE_TE_TIER_MODELS`] OR its manifest declares `mlx.denseTextEncoderTier: true`.
/// The manifest flag mirrors the hardcoded set so a new dense-TE turnkey opts in from the catalog.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn is_dense_te_tier(request: &ImageRequest) -> bool {
    DENSE_TE_TIER_MODELS.contains(&request.model.as_str())
        || request
            .model_manifest_entry
            .get("mlx")
            .and_then(|mlx| mlx.get("denseTextEncoderTier"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
}

/// Pick the engine-complete tier subdir of a standard SceneWorks quant-matrix turnkey `root`:
/// `bf16/` when the request opts out of quantization (`advanced.mlxQuantize <= 0` / "none"), `q8/`
/// when it opts into Q8 (`> 4`), else the default `q4/`. Falls back through q4 → q8 → bf16 → `root`
/// so a partially-downloaded turnkey surfaces as a load error rather than a silent half-load.
///
/// Tier presence is filename-agnostic: a tier is "present" when its backbone component holds any
/// `*.safetensors` (packed single-file OR a `*-00001-of-*.safetensors` shard) or a `*.index.json`
/// (dense sharded). The backbone component is `transformer/` for the DiT turnkeys
/// (flux/qwen/z-image/sd3.5) or `unet/` for the SDXL-family turnkeys (sc-8746) — SDXL packs its UNet
/// under `unet/`, never `transformer/`. This covers every backbone regardless of its packed filename
/// (`diffusion_pytorch_model.safetensors`, `model.safetensors`, …), so a new model needs only a
/// [`STANDARD_TIER_MODELS`] entry (or `mlx.standardTierLayout`), no bespoke resolver.
///
/// Unified-model turnkeys (SenseNova-U1 MoT, sc-8771) have NO component subdir: the whole backbone
/// is a flat `model.safetensors` (or sharded `*.index.json`) directly in the tier dir. The presence
/// check also accepts weights at the tier root so a flat unified tier resolves like a component one.
fn standard_tier_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    // A component dir "has weights" when it holds a packed/dense safetensors or a shard index.
    let component_has_weights = |dir: &Path| -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
        entries.flatten().any(|entry| {
            let file = entry.file_name();
            let name = file.to_string_lossy();
            name.ends_with(".safetensors") || name.ends_with(".index.json")
        })
    };
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        // DiT turnkeys pack the backbone under `transformer/`; SDXL-family turnkeys under `unet/`.
        // Unified-model turnkeys (SenseNova-U1 MoT, sc-8771) have NO component subdirs — the whole
        // backbone is a flat `model.safetensors` (or sharded `*.index.json`) directly in the tier
        // dir. Accept any of the three so a flat unified tier resolves like a component one.
        let has_backbone = component_has_weights(&dir.join("transformer"))
            || component_has_weights(&dir.join("unet"))
            || component_has_weights(&dir);
        has_backbone.then_some(dir)
    };
    // bits<=0 (advanced.mlxQuantize: 0 / "none") → bf16; bits>4 → q8; else the q4 default.
    let preferred = match bits {
        Some(b) if b <= 0 => "bf16",
        Some(b) if b > 4 => "q8",
        _ => "q4",
    };
    present(preferred)
        .or_else(|| present("q4"))
        .or_else(|| present("q8"))
        .or_else(|| present("bf16"))
        .unwrap_or_else(|| root.to_path_buf())
}

/// The Ideogram 4 tier subdir a `mlxQuantize` request needs fetched ON DEMAND — `Some("q8")` when the
/// request opts into Q8 (`> 4`), else `None` (the shipped default `q4/`, which the catalog download
/// pulls; bf16 is a SEPARATE catalog repo the user opts into on the Models page, never an on-demand
/// fetch). Shared by [`ideogram_model_subdir`] (which subdir to load) and [`ensure_ideogram_tier_present`]
/// (which to fetch) so the two stay in lockstep, mirroring [`boogu_tier_subdir`].
fn ideogram_tier_subdir(bits: Option<i64>) -> Option<&'static str> {
    match bits {
        Some(b) if b > 4 => Some("q8"),
        _ => None,
    }
}

/// Pick the engine-complete packed subdir of an Ideogram 4 turnkey `root`: `q8/` when the request
/// opts into Q8 (`advanced.mlxQuantize: 8`) AND it is downloaded, else the default `q4/`. Falls back
/// to `root` if neither subdir is present (a partially-downloaded bundle surfaces as a load error
/// rather than a silent half-load). The non-default `q8/` tier is an on-demand download fetched by
/// [`ensure_ideogram_tier_present`] before this resolves (sc-9607); `q4/` is the manifest default.
fn ideogram_model_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        dir.join("transformer/model.safetensors")
            .is_file()
            .then_some(dir)
    };
    if let Some(tier) = ideogram_tier_subdir(bits) {
        if let Some(dir) = present(tier) {
            return dir;
        }
    }
    present("q4")
        .or_else(|| present("q8"))
        .unwrap_or_else(|| root.to_path_buf())
}

/// The Boogu subfolder for a `mlxQuantize` request — `None` keeps the Q8 default. Shared by
/// [`boogu_model_subdir`] (which subfolder to load) and [`ensure_boogu_tier_present`] (which to
/// fetch on demand): `<=0` → `<variant>-bf16/` (dense full precision), `1..=4` → `<variant>-q4/`
/// (packed Q4, sc-8513), anything else → the default `<variant>/` (packed Q8). Returns the subfolder
/// name relative to the turnkey root.
fn boogu_tier_subdir(variant: &str, bits: Option<i64>) -> Option<String> {
    match bits {
        Some(b) if b <= 0 => Some(format!("{variant}-bf16")),
        Some(b) if b <= 4 => Some(format!("{variant}-q4")),
        _ => None,
    }
}

/// Pick the engine-complete subfolder of a Boogu turnkey `root` for the requested variant. Each
/// catalog id maps to a variant folder: `boogu_image`→`base`, `boogu_image_turbo`→`turbo`,
/// `boogu_image_edit`→`edit`. **Q8 is the shipped default** (the pre-packed `<variant>/` folder); an
/// explicit advanced `mlxQuantize` selects another tier (sc-8513, epic 8506): `<=0` → the dense
/// `<variant>-bf16/`, `1..=4` → the packed `<variant>-q4/`. Falls back through Q8 → q4 → bf16 → `root`
/// when the requested tier isn't downloaded, so a partial bundle surfaces as a load error rather than
/// a silent half-load. (The non-default tiers are on-demand downloads fetched by
/// [`ensure_boogu_tier_present`] before this resolves, sc-6568/sc-8513.)
fn boogu_model_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let variant = match request.model.as_str() {
        "boogu_image_turbo" => "turbo",
        "boogu_image_edit" => "edit",
        _ => "base",
    };
    let bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    // q4/q8 ship a single packed transformer file; bf16 is the dense diffusers tree (SHARDED → only
    // the `.index.json`). Accept either shape.
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        let packed = dir
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file();
        let sharded = dir
            .join("transformer/diffusion_pytorch_model.safetensors.index.json")
            .is_file();
        (packed || sharded).then_some(dir)
    };
    let q4 = format!("{variant}-q4");
    let bf16 = format!("{variant}-bf16");
    if let Some(tier) = boogu_tier_subdir(variant, bits) {
        if let Some(dir) = present(&tier) {
            return dir;
        }
    }
    present(variant)
        .or_else(|| present(&q4))
        .or_else(|| present(&bf16))
        .unwrap_or_else(|| root.to_path_buf())
}

/// Pick the engine-complete packed subdir of a Krea 2 Turbo turnkey `root`: `q4/` when the request opts
/// into Q4 (`advanced.mlxQuantize <= 4`) AND it is downloaded, else the default `q8/` (the shipped
/// default — the P1-validated near-lossless quant). Falls back to whichever subdir is present, then
/// `root`, so a partially-downloaded bundle surfaces as a load error rather than a silent half-load. The
/// turnkey (`SceneWorks/krea-2-turbo-mlx`, sc-7573) carries one `from_snapshot`-loadable subdir per quant
/// (each with a packed `transformer/diffusion_pytorch_model.safetensors`); the loader auto-detects the
/// packed quant, so the resolved `spec.quantize` is a no-op on it. Mirrors [`ideogram_model_subdir`]
/// (q4/q8 subdirs) with Boogu's packed-transformer filename and a Q8-default selection.
fn krea_model_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    // q4/q8 ship a single packed transformer file; the bf16 tier (sc-8513) is the dense diffusers
    // tree (SHARDED → only the `.index.json`). Accept either shape.
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        let packed = dir
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file();
        let sharded = dir
            .join("transformer/diffusion_pytorch_model.safetensors.index.json")
            .is_file();
        (packed || sharded).then_some(dir)
    };
    // bits<=0 → bf16 (the dense base; krea's loader takes it with no quantize); bits<=4 → q4; else the
    // default q8 (the P1-validated near-lossless quant).
    let preferred = match bits {
        Some(b) if b <= 0 => "bf16",
        Some(b) if b <= 4 => "q4",
        _ => "q8",
    };
    present(preferred)
        .or_else(|| present("q8"))
        .or_else(|| present("q4"))
        .or_else(|| present("bf16"))
        .unwrap_or_else(|| root.to_path_buf())
}

/// The private HF repo hosting the Krea 2 INT8-ConvRot DiT single-file checkpoint (sc-9300, epic
/// 9083). Authed download with the SceneWorks HF token, like every private SceneWorks tier.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_CONVROT_REPO: &str = "SceneWorks/krea-2-turbo-int8-convrot";

/// The ConvRot DiT filename inside [`KREA_CONVROT_REPO`] (mirrors the manifest download `files`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_CONVROT_DIT_FILE: &str = "krea2_turbo_int8_convrot.safetensors";

/// Whether this Krea 2 request selected the candle-only INT8-ConvRot tier (sc-9300). The studio's tier
/// picker sends `advanced.convRot: true` for the `int8-convrot` variant (it has no `mlxQuantize` — the
/// online-rotation int8 DiT isn't a bits-based quant). Candle-lane only: the tier is `platforms`-scoped
/// off macOS and the worker only advertises the `int8_convrot` capability on the candle lane, so this
/// never fires on the MLX path even if a stray flag arrives. Confined to `krea_2_turbo`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn wants_krea_convrot(request: &ImageRequest) -> bool {
    request.model == "krea_2_turbo"
        && request
            .advanced
            .get("convRot")
            .and_then(Value::as_bool)
            .unwrap_or(false)
}

/// Resolve the INT8-ConvRot LoadSpec inputs for a Krea 2 request (sc-9300): the canonical bf16 Krea 2
/// snapshot DIR (the LoadSpec `weights` root — tokenizer / Qwen3-VL TE / Qwen-Image VAE / config + the
/// non-quantized surface) and the downloaded ConvRot DiT single-file (the LoadSpec `text_encoder`
/// `File`, which the candle-gen krea engine's `convrot_selector` routes to `load_components_convrot`).
///
/// `None` when the request didn't select ConvRot, OR either artifact isn't present yet (the bf16
/// `bf16/` subdir of the `krea-2-turbo-mlx` turnkey, or the ConvRot DiT `.safetensors`) — the caller
/// then falls back to the normal dense/packed path rather than half-loading. The bf16 base is fetched
/// on demand by [`ensure_krea_convrot_base_present`] before this resolves. The sm_89 compute-cap floor
/// is enforced ENGINE-side (`ensure_int8_floor` inside `load_components_convrot`) AND surfaced as the
/// worker's `int8_convrot` capability, so an ineligible card never reaches here (the picker hides it).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_krea_convrot(
    request: &ImageRequest,
    settings: &Settings,
) -> Option<(PathBuf, PathBuf)> {
    if !wants_krea_convrot(request) {
        return None;
    }
    // The bf16 base surface: the `bf16/` subdir of the shared `krea-2-turbo-mlx` turnkey (the SAME dir
    // the bf16 tier ships), a candle-readable `transformer/ text_encoder/ vae/ tokenizer/` root. The
    // ConvRot DiT replaces only the transformer at load, but the pipeline still reads the dense TE/VAE/
    // tokenizer/config from here.
    let base_dir = huggingface_snapshot_dir(&settings.data_dir, KREA_MLX_TURNKEY_REPO)
        .map(|root| root.join("bf16"))
        .filter(|dir| {
            dir.join("model_index.json").is_file()
                && dir.join("text_encoder").is_dir()
                && dir.join("vae").is_dir()
        })?;
    // The ConvRot DiT single-file inside the private repo's snapshot.
    let convrot_dit = huggingface_snapshot_dir(&settings.data_dir, KREA_CONVROT_REPO)
        .map(|root| root.join(KREA_CONVROT_DIT_FILE))
        .filter(|file| file.is_file())?;
    Some((base_dir, convrot_dit))
}

/// The shared `krea-2-turbo-mlx` turnkey repo (its `bf16/` subdir supplies the ConvRot base surface).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_MLX_TURNKEY_REPO: &str = "SceneWorks/krea-2-turbo-mlx";
/// Pinned revision for the fixed [`KREA_MLX_TURNKEY_REPO`] (sc-9879, F-077 follow-up). The repo is a
/// hard-coded const (no manifest/payload override reaches this on-demand ConvRot-base fetch), so pulling
/// the mutable `main` branch would let an upstream re-push silently swap the bf16 DiT / Qwen3-VL TE /
/// Qwen-Image VAE we load. Pin the exact commit for defense-in-depth (mirrors the SeedVR2/Real-ESRGAN
/// pins, sc-8879/sc-9682). The `hf` CLI still verifies each file's own hash on download.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_MLX_TURNKEY_REVISION: &str = "d009674080cc1bccf2b629d834c34bf5eccdb723";

/// On-demand fetch of the canonical bf16 Krea 2 base surface for the INT8-ConvRot tier (sc-9300),
/// the sibling of [`ensure_boogu_tier_present`]. The ConvRot catalog download pulls only the DiT
/// single-file; the bf16 `bf16/` subdir of the `krea-2-turbo-mlx` turnkey (tokenizer / Qwen3-VL TE /
/// Qwen-Image VAE / config) is fetched here when the ConvRot tier is selected and it isn't present —
/// so q4/q8 users are never forced to download the 35 GB bf16 base (it isn't a global co-requisite).
/// No-op when the request isn't a ConvRot job, the bf16 base is already complete, or the `hf` CLI is
/// absent (then `resolve_krea_convrot` returns `None` and the job falls back / surfaces a load error).
/// Fails loud on a real download error — fast, before any compute.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn ensure_krea_convrot_base_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<()> {
    if !wants_krea_convrot(request) {
        return Ok(());
    }
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, KREA_MLX_TURNKEY_REPO) else {
        // Turnkey never fetched → `hf` may still pull it below; probe the eventual bf16 subdir path.
        // (If the repo is entirely absent, the fetch below installs the requested bf16 leaf dirs.)
        return fetch_krea_convrot_base(api, settings, job).await;
    };
    let bf16 = root.join("bf16");
    // Present already (dense sharded transformer index + the dense TE/VAE) → no fetch.
    if bf16.join("model_index.json").is_file()
        && bf16.join("text_encoder").is_dir()
        && bf16.join("vae").is_dir()
    {
        return Ok(());
    }
    fetch_krea_convrot_base(api, settings, job).await
}

/// Pull the bf16 base leaf dirs of the `krea-2-turbo-mlx` turnkey into the HF cache (sc-9300). Same
/// leaf-glob shape as the manifest bf16 tier download; scratched marker dir keyed by job id.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn fetch_krea_convrot_base(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let scratch = settings
        .data_dir
        .join("cache")
        .join(format!(".krea-convrot-base-fetch-{}", job.id));
    tokio::fs::create_dir_all(&scratch).await?;
    let files = vec![
        "bf16/transformer/*".to_owned(),
        "bf16/text_encoder/*".to_owned(),
        "bf16/vae/*".to_owned(),
        "bf16/tokenizer/*".to_owned(),
        "bf16/scheduler/*".to_owned(),
        "bf16/model_index.json".to_owned(),
    ];
    let result = crate::model_jobs::download_model_with_hf_cli(
        api,
        settings,
        job,
        KREA_MLX_TURNKEY_REPO,
        KREA_MLX_TURNKEY_REVISION,
        &files,
        &scratch,
    )
    .await;
    let _ = tokio::fs::remove_dir_all(&scratch).await;
    result.map(|_| ())
}

/// On-demand fetch of a non-default Boogu tier subfolder (sc-6568 / sc-8513). The catalog download
/// pulls only the packed Q8 `<variant>/` subfolder, so when a job opts into another tier
/// ([`boogu_tier_subdir`]: `<=0` → `<variant>-bf16/` dense, `1..=4` → `<variant>-q4/` packed) and that
/// subfolder isn't present yet, pull just its files into the HF cache so [`boogu_model_subdir`]
/// resolves it. No-op when the Q8 default is requested, the model isn't Boogu, the turnkey snapshot
/// isn't downloaded yet (`boogu_model_subdir` then falls back to Q8 / surfaces the load error), or the
/// tier subfolder is already complete. Fails loud on a real download error — fast, before any compute;
/// a missing `hf` CLI leaves the subfolder absent so the request gracefully falls back to Q8. Mirrors
/// [`crate::video_jobs::ensure_ltx_q8_present`].
///
/// sc-9607 (epic 9083): also runs on the candle lane (off-Mac) — `generate_candle_stream` calls it
/// before snapshot resolution, so Windows/Linux users get the SAME on-demand `-q4/-bf16` fetch as
/// macOS. Previously `#[cfg(target_os = "macos")]`, so off-Mac only the shipped Q8 `base/` default was
/// installable and a non-default tier silently fell back to Q8.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn ensure_boogu_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<()> {
    let variant = match request.model.as_str() {
        "boogu_image" => "base",
        "boogu_image_turbo" => "turbo",
        "boogu_image_edit" => "edit",
        _ => return Ok(()),
    };
    let bits = request.advanced.get("mlxQuantize").and_then(quant_int);
    let Some(tier) = boogu_tier_subdir(variant, bits) else {
        // Q8 default ships in the catalog download — nothing to fetch.
        return Ok(());
    };
    let Some(model) = mlx_model(&request.model) else {
        return Ok(());
    };
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, &model_repo(request, &model))
    else {
        // Turnkey not downloaded at all → leave it to the load path's "weights not found" error.
        return Ok(());
    };
    let tier_dir = root.join(&tier);
    // Present already (packed single-file q4 OR sharded-dense bf16 `.index.json`) → no fetch.
    if tier_dir
        .join("transformer/diffusion_pytorch_model.safetensors")
        .is_file()
        || tier_dir
            .join("transformer/diffusion_pytorch_model.safetensors.index.json")
            .is_file()
    {
        return Ok(());
    }
    let scratch = settings
        .data_dir
        .join("cache")
        .join(format!(".boogu-tier-fetch-{}", job.id));
    tokio::fs::create_dir_all(&scratch).await?;
    // The tier subfolder nests transformer/mllm/vae (leaf-dir globs, like the catalog Q8 entry).
    let files = vec![
        format!("{tier}/transformer/*"),
        format!("{tier}/mllm/*"),
        format!("{tier}/vae/*"),
    ];
    let result = crate::model_jobs::download_model_with_hf_cli(
        api,
        settings,
        job,
        &model_repo(request, &model),
        "main",
        &files,
        &scratch,
    )
    .await;
    let _ = tokio::fs::remove_dir_all(&scratch).await;
    result.map(|_| ())
}

/// On-demand fetch of Ideogram 4's non-default `q8/` tier (sc-9607, epic 9083). The catalog download
/// pulls only the default `q4/` subdir (`files: ["q4/*"]`), so a job that opts into Q8
/// ([`ideogram_tier_subdir`]: `> 4` → `q8/`) needs the `q8/` subdir pulled into the HF cache before
/// [`ideogram_model_subdir`] can resolve it. No-op when the default q4 is requested, the model isn't
/// Ideogram, the turnkey snapshot isn't downloaded yet (`ideogram_model_subdir` then falls back to q4 /
/// surfaces the load error), or `q8/` is already complete. The `q8/*` glob is recursive (matches
/// `q8/transformer/…`, and for `ideogram_4_turbo` the bundled `q8/turbo_lora.safetensors`), mirroring
/// the catalog q4 entry and [`crate::video_jobs::ensure_ltx_q8_present`]. bf16 is NOT fetched here — it
/// lives in the separate `SceneWorks/ideogram-4` catalog repo the user opts into on the Models page
/// (and is macOS-only). This is the on-demand `q8/` download the [`ideogram_model_subdir`] docstring
/// flagged as a follow-up; it runs on BOTH the MLX (`generate_stream`) and candle
/// (`generate_candle_stream`) lanes, so off-Mac gets the same q4/q8 picker as macOS. Fails loud on a
/// real download error; a missing `hf` CLI leaves `q8/` absent so the request gracefully falls back to q4.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn ensure_ideogram_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<()> {
    if request.model != "ideogram_4" && request.model != "ideogram_4_turbo" {
        return Ok(());
    }
    let bits = request.advanced.get("mlxQuantize").and_then(quant_int);
    let Some(tier) = ideogram_tier_subdir(bits) else {
        // Default q4 ships in the catalog download — nothing to fetch.
        return Ok(());
    };
    let Some(model) = mlx_model(&request.model) else {
        return Ok(());
    };
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, &model_repo(request, &model))
    else {
        // Turnkey not downloaded at all → leave it to the load path's "weights not found" error.
        return Ok(());
    };
    // Present already (the packed single-file transformer) → no fetch.
    if root
        .join(tier)
        .join("transformer/model.safetensors")
        .is_file()
    {
        return Ok(());
    }
    let scratch = settings
        .data_dir
        .join("cache")
        .join(format!(".ideogram-tier-fetch-{}", job.id));
    tokio::fs::create_dir_all(&scratch).await?;
    let files = vec![format!("{tier}/*")];
    let result = crate::model_jobs::download_model_with_hf_cli(
        api,
        settings,
        job,
        &model_repo(request, &model),
        "main",
        &files,
        &scratch,
    )
    .await;
    let _ = tokio::fs::remove_dir_all(&scratch).await;
    result.map(|_| ())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn quant_int(value: &Value) -> Option<i64> {
    if value.is_boolean() {
        return None;
    }
    value
        .as_i64()
        .or_else(|| value.as_str()?.trim().parse().ok())
}

/// Resolve quantization: `advanced.mlxQuantize` → `manifest.mlx.quantize` → Q8
/// default. The engine supports Q4/Q8; map (<=0 → dense, <=4 → Q4, else Q8). Returns the
/// engine quant + the effective bit count for the recipe (None = dense bf16).
///
/// Shared by the MLX path and the candle lane (sc-5126). On the candle lane it is called ONLY for a
/// family whose descriptor advertises `supported_quants` (i.e. Lens — see `generate_candle_stream`'s
/// `model.supports_quant()` gate), so the Q8 default applies to Lens exactly like the MLX families;
/// the sc-3675/sc-5096 candle families advertise no quant and never reach this resolver (stay dense).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_quant(request: &ImageRequest) -> (Option<Quant>, Option<i64>) {
    // Dense-TE turnkeys (FLUX.2-klein, sc-8711): the tier subdir already holds a packed transformer
    // + a DENSE bf16 text encoder, so the load Quant must be None — quantizing here would crush the
    // dense bf16 TE we deliberately kept full-precision. The packed transformer self-describes its
    // quant regardless. Tier selection (q4/q8/bf16) is driven by the resolved subdir, not this.
    if is_dense_te_tier(request) {
        return (None, None);
    }
    let raw = request
        .advanced
        .get("mlxQuantize")
        .and_then(quant_int)
        .or_else(|| {
            request
                .model_manifest_entry
                .get("mlx")
                .and_then(|mlx| mlx.get("quantize"))
                .and_then(quant_int)
        });
    match raw {
        None => (Some(Quant::Q8), Some(8)),
        Some(bits) if bits <= 0 => (None, None),
        Some(bits) if bits <= 4 => (Some(Quant::Q4), Some(4)),
        Some(_) => (Some(Quant::Q8), Some(8)),
    }
}

/// The transformer-tier bit count a dense-TE turnkey (FLUX.2-klein, the [`DENSE_TE_TIER_MODELS`]
/// class) actually asked for, derived from `advanced.mlxQuantize` the SAME way [`standard_tier_subdir`]
/// picks its `bf16`/`q8`/`q4` tier (`<=0 → bf16`, `>4 → q8`, else the q4 default). Returns the recipe
/// bit count of the REQUESTED tier: `None` (bf16) / `Some(8)` / `Some(4)`.
///
/// sc-9362 (F-018 follow-up): [`resolve_quant`] returns `(None, None)` for every dense-TE job (the
/// load quant must stay `None` so the deliberately-dense bf16 text encoder is never re-quantized), so
/// the request-derived recipe bits are ALWAYS bf16 even though the transformer is packed at q4/q8. If
/// [`reconcile_resolved_tier_quant`] compared the resolved transformer tier against that always-`None`
/// value, every straight (non-fallback) dense-TE job would look like a bf16→qN "downgrade" — firing a
/// spurious `quant_tier_downgraded` event while the asset telemetry still hid the true transformer
/// precision. Comparing the resolved tier against THIS requested-tier value instead lets the reconcile
/// record the actual transformer precision on every job and warn/emit ONLY on a genuine fallback
/// (requested tier absent → resolver fell through to an adjacent tier).
///
/// macOS-only: consumed on the MLX `generate_stream` reconcile path (the candle lane has no tier
/// layout), alongside [`tier_quant_from_resolved_dir`].
#[cfg(target_os = "macos")]
fn dense_te_requested_tier_bits(request: &ImageRequest) -> Option<i64> {
    let bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    match bits {
        Some(b) if b <= 0 => None,
        Some(b) if b > 4 => Some(8),
        _ => Some(4),
    }
}

/// The `(engine Quant, recipe bit count)` a resolved turnkey tier subdir ACTUALLY loads at, parsed
/// from the tier dir's basename (sc-8820). The tier resolvers ([`standard_tier_subdir`],
/// [`ideogram_model_subdir`], [`boogu_model_subdir`], [`krea_model_subdir`]) fall through q4→q8→bf16
/// (or the Boogu `<variant>`-shaped names) when the requested tier isn't downloaded, so the resolved
/// path can be a DIFFERENT precision than the request asked for. This maps the resolved basename back
/// to its precision so the caller can record the tier that ran, not the one requested:
///   - `bf16` / `<variant>-bf16` → dense (`None`, `None`)
///   - `q4`   / `<variant>-q4`   → Q4 (`4`)
///   - `q8`   / `<variant>-q8`   → Q8 (`8`)
///   - a bare Boogu `<variant>/` (no `-q4`/`-q8`/`-bf16` suffix) → the shipped **Q8** default
///
/// `None` when the basename is not a recognizable tier name — e.g. the resolver fell all the way back
/// to the repo `root` (a partial/absent turnkey the engine will error on), or the dir is a `modelPath`
/// override. In that case the caller keeps the request-derived quant rather than inventing one.
///
/// macOS-only: its sole caller is [`reconcile_resolved_tier_quant`] on the MLX `generate_stream` path.
/// The candle lane has no quant-tier layout to reconcile, so this would be dead code there.
#[cfg(target_os = "macos")]
fn tier_quant_from_resolved_dir(dir: &Path) -> Option<(Option<Quant>, Option<i64>)> {
    let name = dir.file_name()?.to_str()?;
    // Match the trailing tier token: `q4` / `q8` / `bf16`, whether the whole basename (standard/
    // ideogram/krea) or the suffix of a Boogu `<variant>-<tier>` folder.
    let tier = name.rsplit('-').next().unwrap_or(name);
    match tier {
        "bf16" => Some((None, None)),
        "q4" => Some((Some(Quant::Q4), Some(4))),
        "q8" => Some((Some(Quant::Q8), Some(8))),
        // A bare Boogu `base`/`turbo`/`edit` folder (no tier suffix) IS the packed Q8 default.
        "base" | "turbo" | "edit" => Some((Some(Quant::Q8), Some(8))),
        _ => None,
    }
}

/// Reconcile the request-derived `(quant, quant_bits)` against the tier subdir the resolver ACTUALLY
/// landed on (sc-8820). The tier resolvers fall through q4→q8→bf16 when the preferred tier isn't
/// downloaded, but the recipe quant is derived from the REQUEST — so a user who selected bf16 with
/// only `q4/` present would silently render Q4 while the sidecar records dense. That makes the epic
/// 8506 quant A/B workflow lie about precision (and can compare a tier against itself). This corrects
/// the recorded quant to the resolved tier, and — when a downgrade actually happened — `warn!`s and
/// emits a `quant_tier_downgraded` event so the UI/telemetry surfaces the fallback instead of hiding
/// it. We do NOT hard-error: a working render at an adjacent tier beats failing because the preferred
/// tier is missing (the finding prefers warn+fallback+correct-recording). Returns the
/// `(quant, quant_bits)` to record + load with.
///
/// `allow_quant_change` gates whether the LOAD quant may be rewritten to match the resolved tier.
/// It's `true` for the ordinary packed-turnkey families (the load-quant is a no-op on already-packed
/// weights, so correcting it to the resolved tier is safe/right). It's `false` for the DENSE-TE
/// turnkeys (FLUX.2-klein, sc-8711): their load quant MUST stay `None` so the deliberately-dense bf16
/// text encoder is never re-quantized — but the *recorded* bit count still gets corrected to the
/// packed transformer's resolved tier so the sidecar tells the truth. The event/`warn!` still fire so
/// the fallback is surfaced either way.
#[cfg(target_os = "macos")]
fn reconcile_resolved_tier_quant(
    requested: (Option<Quant>, Option<i64>),
    weights_dir: &Path,
    allow_quant_change: bool,
    model_id: &str,
    job_id: &str,
    engine: &str,
) -> (Option<Quant>, Option<i64>) {
    let Some((actual_quant, actual_bits)) = tier_quant_from_resolved_dir(weights_dir) else {
        // Not a recognizable tier dir (fell back to the repo root, or a modelPath override) — keep
        // the request-derived quant; the engine will surface any missing-weights error itself.
        return requested;
    };
    if actual_bits == requested.1 {
        return requested;
    }
    // The resolved tier differs from what the request asked for → a silent fallback. Surface it and
    // record the precision that actually ran.
    let requested_label = requested.1.map_or("bf16".to_owned(), |b| format!("q{b}"));
    let actual_label = actual_bits.map_or("bf16".to_owned(), |b| format!("q{b}"));
    tracing::warn!(
        "{engine}: requested quant tier {requested_label} for {model_id} is not downloaded; \
         fell back to {actual_label} — recording the tier that actually ran"
    );
    emit_event(
        "quant_tier_downgraded",
        json!({
            "jobId": job_id,
            "engine": engine,
            "model": model_id,
            "requested": requested_label,
            "resolved": actual_label,
            "requestedBits": requested.1,
            "resolvedBits": actual_bits,
        }),
    );
    // Always correct the recorded bits; only rewrite the load quant when it's safe to (packed
    // turnkeys), so a dense-TE turnkey keeps its `None` load quant while still recording the truth.
    let load_quant = if allow_quant_change {
        actual_quant
    } else {
        requested.0
    };
    (load_quant, actual_bits)
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=80) else the family default.
/// Shared by the MLX path and the candle lane (sc-5096).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_steps(request: &ImageRequest, model: &ResolvedModel) -> u32 {
    request
        .advanced
        .get("steps")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|steps| (steps as u32).clamp(1, 80))
        .unwrap_or(model.default_steps())
}

/// Resolve the guidance scale. Distilled variants (z-image-turbo, flux schnell) take
/// no guidance — the engine rejects `Some(_)` on them — so this returns `None`. For a
/// guided variant (flux dev) it is `advanced.guidanceScale` else the family default.
/// Shared by the MLX path and the candle lane (sc-5096); the descriptor's `supports_guidance` is the
/// candle descriptor on the Windows lane, so a distilled candle family (z-image, flux schnell) still
/// gets `None` and a guided one (flux dev, flux2, qwen, sdxl) gets the scale.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_guidance(request: &ImageRequest, model: &ResolvedModel) -> Option<f32> {
    if !model.supports_guidance() {
        return None;
    }
    let scale = request
        .advanced
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(model.default_guidance());
    Some(scale)
}

/// Resolve an unsigned advanced knob with a manifest fallback (sc-8825). The single mechanism the
/// bespoke edit/adapter/control lanes (`*_edit_candle.rs`, `sdxl_ipadapter.rs`, `kolors_*.rs`,
/// `qwen_control.rs`, `instantid.rs`, `pulid.rs`) had each re-implemented as an inline parse closure:
/// `advanced[key]` (JSON uint OR numeric string) → the manifest `[key]` (same parse) → `default`.
/// The parsed **advanced-or-manifest** value is clamped to `range`; the `default` is returned
/// **unclamped** (it is a trusted per-lane constant, and clamping it would silently change a lane
/// whose default sits outside its own historical range). Each caller passes its OWN `range`/`default`,
/// so the drifting steps bounds (1..=80 / 1..=50 / 1..=100) are preserved byte-for-byte — this is a
/// dedup-of-mechanism refactor, not a policy change.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_advanced_or_manifest_u32(
    request: &ImageRequest,
    key: &str,
    default: u32,
    range: std::ops::RangeInclusive<u32>,
) -> u32 {
    let parse = |value: &Value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    };
    request
        .advanced
        .get(key)
        .and_then(parse)
        .or_else(|| request.model_manifest_entry.get(key).and_then(parse))
        .map(|value| value.clamp(*range.start() as u64, *range.end() as u64) as u32)
        .unwrap_or(default)
}

/// Resolve a float advanced knob with a manifest fallback (sc-8825). The guidance twin of
/// [`resolve_advanced_or_manifest_u32`]: the manifest `[key]` (JSON float OR numeric string) supplies
/// the effective default (else the per-lane `default`), then [`advanced::f32_clamped`] reads
/// `advanced[key]` — falling back to that manifest default — and clamps the result to `range`. Unlike
/// the u32 twin, the resolved value here is always clamped (matching the historical `f32_clamped`
/// call, which clamps the manifest/default fallback too). Each caller passes its OWN `range`/`default`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_advanced_or_manifest_f32(
    request: &ImageRequest,
    key: &str,
    default: f32,
    range: std::ops::RangeInclusive<f32>,
) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default);
    advanced::f32_clamped(&request.advanced, key, manifest_default, range)
}

/// Closure-default twin of [`resolve_advanced_or_manifest_u32`] (sc-8825). Identical mechanism —
/// `advanced[key]` → manifest `[key]` (parsed, clamped to `range`) — except the fallback is a
/// per-lane `default_fn` closure evaluated **only** when both advanced and manifest are absent (and,
/// like the const twin, returned **unclamped**). This covers the lanes whose default is model-dependent
/// (`flux_ipadapter` variant steps, `qwen_edit_candle`/`zimage_control` per-variant steps) rather than a
/// bare constant. Every caller passes its OWN `range`/`default_fn`, so per-lane bounds stay byte-for-byte.
///
/// Unlike the const twins (which have macOS-live callers in `pulid.rs`/`instantid.rs`), these variants
/// are only *called* by the candle-exclusive lanes, so the macOS non-test lib build has no caller —
/// hence the gate is `candle-lane OR test` (the sc-8825 unit tests exercise it on both build lanes).
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
fn resolve_advanced_or_manifest_u32_with(
    request: &ImageRequest,
    key: &str,
    default_fn: impl FnOnce() -> u32,
    range: std::ops::RangeInclusive<u32>,
) -> u32 {
    let parse = |value: &Value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    };
    request
        .advanced
        .get(key)
        .and_then(parse)
        .or_else(|| request.model_manifest_entry.get(key).and_then(parse))
        .map(|value| value.clamp(*range.start() as u64, *range.end() as u64) as u32)
        .unwrap_or_else(default_fn)
}

/// Closure-default twin of [`resolve_advanced_or_manifest_f32`] (sc-8825). Identical mechanism —
/// manifest `[key]` supplies the effective default (else the per-lane `default_fn` closure, evaluated
/// only when the manifest key is absent), then [`advanced::f32_clamped`] reads `advanced[key]` (falling
/// back to that default) and clamps to `range`. Covers `flux_ipadapter`, whose guidance fallback is a
/// per-variant fn. Each caller passes its OWN `range`/`default_fn`. Gated `candle-lane OR test` for the
/// same reason as the u32 twin: no macOS non-test caller, but the sc-8825 tests exercise it on both lanes.
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
fn resolve_advanced_or_manifest_f32_with(
    request: &ImageRequest,
    key: &str,
    default_fn: impl FnOnce() -> f32,
    range: std::ops::RangeInclusive<f32>,
) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or_else(default_fn);
    advanced::f32_clamped(&request.advanced, key, manifest_default, range)
}

/// True for a TRUE-CFG family whose engine reads the CFG scale from `true_cfg` (with a real
/// negative prompt) and **rejects** the distilled `guidance` scalar — i.e. Chroma (epic 3531),
/// uniquely identified by `supports_guidance=false` + `supports_negative_prompt=true`. The
/// guidance-distilled families (`z_image_turbo`, `flux_schnell`) are `false`/`false` (no CFG at
/// all), and the `guidance`-scalar families (qwen / sdxl / flux2 …) are `true`/*. For a true-CFG
/// family the worker forwards `advanced.guidanceScale` as `true_cfg`, not `guidance`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn uses_true_cfg(model: &ResolvedModel) -> bool {
    !model.supports_guidance() && model.supports_negative_prompt()
}

/// Resolve the true-CFG scale for a true-CFG family (Chroma). `None` for every other family
/// (their CFG, if any, flows through [`resolve_guidance`]). The scale is `advanced.guidanceScale`
/// (the same user knob) else the family default — forwarded to the engine as `GenerationRequest.true_cfg`.
/// Shared by the MLX path and the candle lane (sc-5096).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_true_cfg(request: &ImageRequest, model: &ResolvedModel) -> Option<f32> {
    if !uses_true_cfg(model) {
        return None;
    }
    let scale = request
        .advanced
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(model.default_guidance());
    Some(scale)
}

/// The negative prompt to pass to the engine. `None` for variants without true CFG
/// (the engine rejects `negative_prompt` on the distilled families) and for an empty
/// prompt (the true-CFG engines fall back to their own neutral negative).
/// Shared by the MLX path and the candle lane (sc-5096); on the Windows lane `supports_negative_prompt`
/// is the candle descriptor, so distilled candle families (z-image, flux schnell) get `None`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_negative_prompt(request: &ImageRequest, model: &ResolvedModel) -> Option<String> {
    if !model.supports_negative_prompt() {
        return None;
    }
    let trimmed = request.negative_prompt.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// First non-empty of installedPath/sourcePath/path/source.path on a LoRA spec.
/// Shared by the MLX path and the candle Lens lane (sc-5126).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn lora_path(lora: &Value) -> Option<PathBuf> {
    for key in ["installedPath", "sourcePath", "path"] {
        if let Some(value) = lora
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Some(PathBuf::from(value));
        }
    }
    lora.get("source")
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Classify a LoRA file into the mlx-gen adapter `kind`. SceneWorks peft-LoKr (stamped
/// `networkType: lokr`) → `Lokr` (the engine's metadata-gated `apply_lokr` peft path). Everything
/// else → `Lora`, INCLUDING third-party LyCORIS (LoHa / kohya non-peft LoKr): since epic 3641
/// (sc-3642/3643/3671) the engine's `apply_adapter_specs_autoprefix` detects `lokr_*` / `hada_*`
/// keys by sniff and routes them to its third-party reconstruction regardless of the declared kind,
/// so `Lora` is the correct hint and the worker no longer rejects them. (A LyCORIS algo the engine
/// doesn't implement — e.g. (IA)³/OFT — has no `lokr_*`/`hada_*` keys, so the engine's LoRA loader
/// finds nothing and surfaces a loud "matched nothing" error rather than mis-applying.)
///
/// Shared by the MLX path and the candle Lens lane (sc-5126): candle-gen-lens's `merge_adapters`
/// dispatches on this `kind` (a `lokr`-metadata file declared `Lora` would find no lora_A/B keys and
/// it surfaces the mismatch loudly), so the same `networkType: lokr` classification feeds both lanes.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn classify_adapter(file: &Path) -> WorkerResult<AdapterKind> {
    let header = read_safetensors_header(file)
        .map_err(|error| WorkerError::InvalidPayload(format!("LoRA header: {error}")))?;
    let network_type = header
        .get("__metadata__")
        .and_then(|meta| meta.get("networkType"))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase());
    if network_type.as_deref() == Some("lokr") {
        return Ok(AdapterKind::Lokr);
    }
    Ok(AdapterKind::Lora)
}

/// Resolve up to 3 request LoRAs into engine adapter specs (path + scale + kind).
/// Shared by the MLX path and the candle Lens lane (sc-5126).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_adapters(request: &ImageRequest, settings: &Settings) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs = Vec::with_capacity(request.loras.len());
    for lora in &request.loras {
        let raw = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        // The path is attacker-controllable payload; confine it to an app-managed
        // root before any on-disk use (sc-5723 / WKA-002).
        let path = crate::normalize_app_managed_lora_path(settings, &raw)?;
        let file = if path.is_dir() {
            first_safetensors_path(&path).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "LoRA has no .safetensors under {}",
                    path.display()
                ))
            })?
        } else {
            path
        };
        if !file.exists() {
            return Err(WorkerError::InvalidPayload(format!(
                "LoRA file is missing: {}",
                file.display()
            )));
        }
        let kind = classify_adapter(&file)?;
        let scale = lora
            .get("weight")
            .and_then(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_str()?.trim().parse().ok())
            })
            .unwrap_or(0.8) as f32;
        specs.push(AdapterSpec::new(file, scale, kind));
    }
    Ok(specs)
}

fn mlx_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // Distilled variants run without CFG (guidance == None → null in the recipe).
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw
}

fn load_spec(
    weights_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    ip_adapter_dir: Option<PathBuf>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    if let Some(dir) = ip_adapter_dir {
        spec = spec.with_ip_adapter(WeightsSource::Dir(dir));
    }
    spec
}

/// Registry-only generator load (epic 3720, sc-3724): resolve `engine_id` through the
/// backend-neutral `gen_core::load` seam and return a `Box<dyn gen_core::Generator>`. Optionally
/// installs an IP-Adapter from `ip_adapter_dir` (`LoadSpec::with_ip_adapter`) — the FLUX.1 XLabs
/// IP-Adapter reference path (epic 3621), after which the engine treats a `Conditioning::Reference`
/// as the image prompt. `cfg(target_os)` decides which provider crate registered the engine, not
/// this call.
#[cfg(all(target_os = "macos", test))]
fn load_engine(
    engine_id: &str,
    weights_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    ip_adapter_dir: Option<PathBuf>,
) -> WorkerResult<Box<dyn Generator>> {
    let spec = load_spec(weights_dir, quant, adapters, ip_adapter_dir);
    load_control_engine(engine_id, &spec)
}

/// Shared real-weight smoke loader: resolve `engine_id` through the backend-neutral
/// `gen_core::load` seam and wrap a failure as `WorkerError::Engine`. Every image
/// control/base lane's `#[cfg(test)]` load wrapper funnels through here so the
/// `gen_core::load` + `map_err` tail lives in one place (sc-8954). `cfg(target_os)`
/// still decides which provider crate registered the engine, not this call.
#[cfg(all(target_os = "macos", test))]
fn load_control_engine(engine_id: &str, spec: &LoadSpec) -> WorkerResult<Box<dyn Generator>> {
    gen_core::load(engine_id, spec)
        .map_err(|error| WorkerError::Engine(format!("{engine_id} load failed: {error}")))
}

/// XLabs FLUX IP-Adapter repos (epic 3621). The torch `flux_dev` path already declares +
/// downloads these (the `ipAdapter` block in `image_adapters`); the MLX path reuses the same
/// HF-cache snapshots — there is no new weight to ship.
#[cfg(target_os = "macos")]
const FLUX_IP_ADAPTER_REPO: &str = "XLabs-AI/flux-ip-adapter";
#[cfg(target_os = "macos")]
const FLUX_IP_IMAGE_ENCODER_REPO: &str = "openai/clip-vit-large-patch14";
/// IP-Adapter scale when the request omits `ipAdapterScale` (XLabs resemblance tier 0.7, matching
/// the torch `FluxDiffusersAdapter`).
#[cfg(target_os = "macos")]
const FLUX_IP_SCALE: f32 = 0.7;
/// `trueCfgScale` default for the FLUX.1-dev IP-Adapter path (real CFG; torch default ~4.0).
#[cfg(target_os = "macos")]
const FLUX_IP_TRUE_CFG: f32 = 4.0;

/// The FLUX.1 engine families that carry the XLabs IP-Adapter (both variants — the Rust engine has
/// no diffusers `load_ip_adapter` schnell limitation).
#[cfg(target_os = "macos")]
fn is_flux_model(model: &str) -> bool {
    matches!(model, "flux_schnell" | "flux_dev")
}

/// The SenseNova-U1 SceneWorks ids (base + 8-step distill), both served by the unified
/// `mlx-gen-sensenova` engine (sc-3900).
#[cfg(target_os = "macos")]
fn is_sensenova_model(model: &str) -> bool {
    matches!(
        model,
        "sensenova_u1_8b"
            | "sensenova_u1_8b_infographic_v2"
            | "sensenova_u1_8b_fast"
            | "sensenova_u1_8b_infographic_v2_fast"
    )
}

/// Stage the engine's IP-Adapter dir contract from the two cached HF snapshots:
/// `<staged>/ip_adapter.safetensors` (XLabs) + `<staged>/image_encoder/model.safetensors`
/// (openai CLIP-ViT-L). Errors loudly if either snapshot is missing — mirrors the SDXL IP path
/// (`resolve_ip_adapter_dir`); the repos reach the cache via the model-download flow / the torch
/// `flux_dev` path, not a new provisioning step.
#[cfg(target_os = "macos")]
fn resolve_flux_ip_adapter_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    let missing = || {
        WorkerError::InvalidPayload(format!(
            "FLUX IP-Adapter weights not found (download {FLUX_IP_ADAPTER_REPO} + {FLUX_IP_IMAGE_ENCODER_REPO})."
        ))
    };
    let adapter_snap =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, FLUX_IP_ADAPTER_REPO)
            .ok_or_else(missing)?;
    let clip_snap =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, FLUX_IP_IMAGE_ENCODER_REPO)
            .ok_or_else(missing)?;
    let ip_file = adapter_snap.join("ip_adapter.safetensors");
    let clip_file = clip_snap.join("model.safetensors");
    if !ip_file.exists() || !clip_file.exists() {
        return Err(missing());
    }
    let staged = settings.data_dir.join("staged").join("flux-ip-adapter");
    let encoder_dir = staged.join("image_encoder");
    std::fs::create_dir_all(&encoder_dir)
        .map_err(|e| WorkerError::InvalidPayload(format!("stage flux ip-adapter dir: {e}")))?;
    // Re-link each call: the HF-cache targets are immutable, so a stable staged dir is reusable.
    let link = |src: &Path, dst: PathBuf| -> WorkerResult<()> {
        let _ = std::fs::remove_file(&dst);
        std::os::unix::fs::symlink(src, &dst)
            .map_err(|e| WorkerError::InvalidPayload(format!("stage flux ip-adapter link: {e}")))
    };
    link(&ip_file, staged.join("ip_adapter.safetensors"))?;
    link(&clip_file, encoder_dir.join("model.safetensors"))?;
    Ok(staged)
}

/// Emit an `image_pipeline_load_{start,complete}` event from inside a blocking
/// generation closure (sc-3450), parity with the Python worker's pipeline-load
/// events. On the backend path `gen_core::load` is a single atomic call that also fuses
/// any distill LoRA and applies user LoRAs (`spec.with_adapters`), so there is no
/// separable fuse/apply step to bracket: the adapter total (`adapter_count` =
/// distill + user) is reported here instead of via the torch worker's separate
/// `image_distill_lora_fuse_*` / `image_lora_apply_*` sub-phase events. A `start`
/// with no matching `complete` means the load failed (the error propagates via `?`).
pub(crate) fn emit_load_event(event: &str, job_id: &str, engine: &str, adapter_count: usize) {
    emit_event(
        event,
        json!({
            "jobId": job_id,
            "engine": engine,
            "adapterCount": adapter_count,
        }),
    );
}

/// N3 (epic 7114): a per-generation `sampler` / `scheduler` knob that names something the engine does
/// NOT advertise must never hard-fail the generation. `gen_core::Capabilities::validate_request` (and
/// each engine's own `validate`) rejects an unadvertised name with an `Err`, so the worker pre-filters
/// the knob here against the linked descriptor's advertised surface (`Capabilities.samplers` /
/// `.schedulers`): an advertised name passes through untouched; an unknown one — a stale recipe, a
/// per-backend capability gap (candle advertises a narrower set than mlx until P4), or manifest drift —
/// is dropped back to the engine default (`None`) and a `sampling_knob_unsupported` worker event is
/// emitted for observability. `None` and the `"default"` sentinel are already stripped at the read site,
/// so this only fires on a real, unsupported name. Shared by the MLX (`generate_stream`) + candle
/// (`generate_candle_stream`) image lanes and the video lane (`run_loaded_video_generation`).
pub(crate) fn normalize_sampling_knob(
    requested: Option<String>,
    advertised: &[&str],
    knob: &str,
    model_id: &str,
    job_id: &str,
    engine: &str,
) -> Option<String> {
    let name = requested?;
    if advertised.contains(&name.as_str()) {
        return Some(name);
    }
    tracing::warn!(
        "{engine}: requested {knob} {name:?} is not advertised (supported: {advertised:?}); \
         falling back to the engine default"
    );
    emit_event(
        "sampling_knob_unsupported",
        json!({
            "jobId": job_id,
            "engine": engine,
            "model": model_id,
            "knob": knob,
            "requested": name,
            "supported": advertised,
        }),
    );
    None
}

/// Read the raw per-generation sampler / scheduler / schedule-shift knobs from a job's `advanced`
/// block (the 1753 front-half carrier). `sampler` / `scheduler` strip the `"default"` sentinel + blanks
/// to `None`, so the engine default — N1's guaranteed no-op — is the ABSENCE of a name, not a magic
/// string; `scheduler_shift` accepts the `schedulerShift` (or legacy `timestepShift`) key as a number or
/// numeric string. Shared by the MLX (`generate_stream`) + candle (`generate_candle_stream`) image lanes
/// — the result is then realvisxl-forced (the lightning checkpoint) and N3-guarded via
/// [`normalize_sampling_knob`]. Returns `(sampler, scheduler, scheduler_shift)`.
pub(crate) fn read_advanced_sampling_knobs(
    advanced: &JsonObject,
) -> (Option<String>, Option<String>, Option<f32>) {
    let name = |key: &str| {
        advanced
            .get(key)
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty() && *value != "default")
            .map(str::to_owned)
    };
    let scheduler_shift = advanced
        .get("schedulerShift")
        .or_else(|| advanced.get("timestepShift"))
        .and_then(|value| value.as_f64().or_else(|| value.as_str()?.trim().parse().ok()))
        .map(|value| value as f32);
    (name("sampler"), name("scheduler"), scheduler_shift)
}

/// Read the per-generation guidance method (epic 7434 P5, sc-7448) from a job's `advanced` block —
/// the 4th sampling axis (`cfg` / `cfg_rescale` / `apg` / `cfg_pp`), alongside the sampler/scheduler
/// knobs. Strips the `"default"` sentinel + blanks to `None` so the engine default (the N1 no-op) is
/// the ABSENCE of a method. The result is then N3-guarded via [`normalize_sampling_knob`] against the
/// model descriptor's `supported_guidance_methods` and threaded onto `GenerationRequest.guidance_method`.
pub(crate) fn read_advanced_guidance_method(advanced: &JsonObject) -> Option<String> {
    advanced
        .get("guidanceMethod")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "default")
        .map(str::to_owned)
}

/// The curated sampler/scheduler menu (epic 7114 decision 2) the **bespoke** conditioned image paths
/// honor — the shared `gen_core` solver/scheduler vocabulary the unified-sampler engines gate on
/// (`Solver::from_name` / the additive `denoise_curated` path; mlx #537/#538/#539, candle #130). The
/// bespoke per-family paths (InstantID, Kolors-conditioned, PuLID — sc-7432) build CUSTOM request
/// structs OUTSIDE `generate_stream`'s generic plumbing, so they N3-normalize the per-request knob
/// against THIS menu instead of a `Capabilities` list: every engine's advertised set is a superset of
/// it (their native default is the only extra, and `"default"`/`None` already strip to the engine
/// default), so a name that survives [`normalize_sampling_knob`] here also passes the engine's own
/// `validate_request`. This is also the single source of truth the manifest⊆engine drift guard
/// (`engines.rs`) checks these out-of-`MODEL_TABLE` models against, so the runtime and the guard never
/// disagree. Derived from `gen_core` (the engines' own vocab), so it tracks the framework on BOTH
/// backends rather than hard-coding names. Returns `(samplers, schedulers)`.
pub(crate) fn curated_image_menu() -> (Vec<&'static str>, Vec<&'static str>) {
    (
        gen_core::sampling::Solver::ALL
            .iter()
            .map(|solver| solver.name())
            .collect(),
        gen_core::sampling::Scheduler::ALL
            .iter()
            .map(|scheduler| scheduler.name())
            .collect(),
    )
}

#[cfg(test)]
mod sampling_knob_tests {
    use super::*;

    #[test]
    fn advertised_name_passes_through() {
        let advertised = ["euler", "dpmpp_2m", "uni_pc"];
        assert_eq!(
            normalize_sampling_knob(
                Some("dpmpp_2m".to_owned()),
                &advertised,
                "sampler",
                "qwen_image",
                "job-1",
                "mlx",
            ),
            Some("dpmpp_2m".to_owned())
        );
    }

    #[test]
    fn unadvertised_name_falls_back_to_default() {
        // N3: a name the engine doesn't advertise (a legacy `dpmpp`/`unipc` recipe, or a candle
        // per-backend gap) is dropped to the engine default (`None`) instead of hard-failing the
        // generation in `validate_request`.
        let advertised = ["lightning"];
        assert_eq!(
            normalize_sampling_knob(
                Some("dpmpp".to_owned()),
                &advertised,
                "sampler",
                "qwen_image",
                "job-1",
                "mlx",
            ),
            None
        );
    }

    #[test]
    fn unset_knob_stays_unset() {
        let advertised = ["euler"];
        assert_eq!(
            normalize_sampling_knob(None, &advertised, "scheduler", "m", "j", "mlx"),
            None
        );
    }

    // sc-7448 — the guidance method rides the same `normalize_sampling_knob` N3 guard as sampler/scheduler.
    #[test]
    fn guidance_method_advertised_passes_through() {
        let advertised = ["cfg", "cfg_pp"];
        assert_eq!(
            normalize_sampling_knob(
                Some("cfg_pp".to_owned()),
                &advertised,
                "guidanceMethod",
                "sdxl",
                "job-1",
                "mlx",
            ),
            Some("cfg_pp".to_owned())
        );
    }

    #[test]
    fn guidance_method_unadvertised_falls_back_to_default() {
        // N3: cfg_pp requested on a model that doesn't advertise it (an engine that only does plain `cfg`,
        // or a stale recipe) drops to the engine default — never a `validate_request` hard-fail.
        let advertised = ["cfg"];
        assert_eq!(
            normalize_sampling_knob(
                Some("cfg_pp".to_owned()),
                &advertised,
                "guidanceMethod",
                "chroma",
                "job-1",
                "mlx",
            ),
            None
        );
    }

    // N1: the read strips the `"default"` sentinel + blanks to `None` (the absence of a method = the
    // engine default = the guaranteed no-op), exactly like the sampler/scheduler read.
    #[test]
    fn read_guidance_method_strips_default_and_blank() {
        assert_eq!(read_advanced_guidance_method(&advanced(serde_json::json!({}))), None);
        assert_eq!(
            read_advanced_guidance_method(&advanced(serde_json::json!({"guidanceMethod": "default"}))),
            None
        );
        assert_eq!(
            read_advanced_guidance_method(&advanced(serde_json::json!({"guidanceMethod": "  "}))),
            None
        );
        assert_eq!(
            read_advanced_guidance_method(&advanced(serde_json::json!({"guidanceMethod": "cfg_pp"}))),
            Some("cfg_pp".to_owned())
        );
    }

    fn advanced(value: serde_json::Value) -> JsonObject {
        value.as_object().expect("object").clone()
    }

    // N1 (epic 7114): the guaranteed no-op default. A job with no sampling knobs — or the explicit
    // `"default"` sentinel the UI sends for "Model default" — must resolve to ALL `None`, i.e. the engine
    // runs its existing native path byte-for-byte. This guards the worker read against a future change
    // that silently injects a non-default sampler onto the default path.
    #[test]
    fn n1_default_advanced_is_a_no_op() {
        assert_eq!(
            read_advanced_sampling_knobs(&advanced(serde_json::json!({}))),
            (None, None, None)
        );
        assert_eq!(
            read_advanced_sampling_knobs(&advanced(serde_json::json!({
                "sampler": "default",
                "scheduler": "default",
                "steps": 30
            }))),
            (None, None, None)
        );
        // Blank / whitespace-only names are also treated as the default (no name).
        assert_eq!(
            read_advanced_sampling_knobs(&advanced(serde_json::json!({"sampler": "  ", "scheduler": ""}))),
            (None, None, None)
        );
    }

    #[test]
    fn read_passes_real_names_and_shift_through() {
        assert_eq!(
            read_advanced_sampling_knobs(&advanced(serde_json::json!({
                "sampler": "dpmpp_2m",
                "scheduler": "sgm_uniform",
                "schedulerShift": 2.5
            }))),
            (
                Some("dpmpp_2m".to_owned()),
                Some("sgm_uniform".to_owned()),
                Some(2.5)
            )
        );
        // schedulerShift accepts a numeric string and the legacy `timestepShift` key.
        let (_, _, shift) = read_advanced_sampling_knobs(&advanced(serde_json::json!({
            "timestepShift": "1.5"
        })));
        assert_eq!(shift, Some(1.5));
    }
}

/// Optional prompt-enhancement settings resolved from a job request's `advanced` block and threaded
/// into a [`GenerationRequest`] (sc-6135). Mirrors the LTX-2.3 video path (`advanced.enhancePrompt` /
/// `enhanceTemperature` / `enhanceMaxTokens`). Only FLUX.2-dev / FLUX.2-dev-edit act on it — the
/// Mistral3 caption upsampler (sc-6030), text-only for txt2img and image-conditioned on the
/// reference image(s) for edit; every other engine ignores the fields, and the dev Image-Studio
/// toggle (manifest `ui.promptEnhance`) is the only surface that sets `enhancePrompt`, so this is a
/// no-op for all other models.
#[derive(Clone, Default)]
pub(crate) struct PromptEnhance {
    enabled: bool,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl PromptEnhance {
    /// Resolve from a job request's `advanced` settings (same keys as the LTX-2.3 video path).
    pub(crate) fn from_advanced(advanced: &JsonObject) -> Self {
        PromptEnhance {
            enabled: advanced::bool(advanced, "enhancePrompt"),
            temperature: advanced
                .get("enhanceTemperature")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            max_tokens: advanced
                .get("enhanceMaxTokens")
                .and_then(Value::as_u64)
                .map(|value| value as u32),
        }
    }

    /// Write the resolved enhancement settings onto a `GenerationRequest`.
    fn apply(&self, request: &mut GenerationRequest) {
        request.enhance_prompt = self.enabled;
        request.enhance_temperature = self.temperature;
        request.enhance_max_tokens = self.max_tokens;
    }
}

/// Generate one image (RGB8) at the given seed; `on_progress` streams denoise steps.
/// `guidance` is `None` for distilled variants (the engine rejects it on them).
///
/// `reference` is the optional identity img2img-init (sc-3619): `(image, strength)` adds a
/// `Reference` conditioning that seeds the denoise from the reference latents — the plain
/// (no-ControlNet) Z-Image reference-without-pose path, reusing the same engine img2img the
/// strict-pose tier already drives. `None` → plain txt2img. `enhance` carries the optional
/// caption-upsampling settings (sc-6135; only FLUX.2-dev acts on them).
#[allow(clippy::too_many_arguments)]
fn generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    edit_mask: Option<&Image>,
    true_cfg: Option<f32>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    // The guidance method (epic 7434 P5, sc-7448): `None` is the engine default (N1 no-op); a value
    // is already N3-normalized against the descriptor's `supported_guidance_methods` by the caller.
    guidance_method: Option<&str>,
    // Per-generation PiD super-resolving decode (epic 7840, sc-7849). Must be `true` only when the
    // generator was loaded with `LoadSpec::with_pid` (the engine rejects a mismatch); the caller keeps
    // the two in lockstep. The candle path passes `false` (candle PiD is Phase 4, sc-7853).
    use_pid: bool,
    enhance: &PromptEnhance,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    // `multi_references` (Boogu instruction edit, sc-7645) takes precedence when present: one image →
    // `Reference` (byte-identical to the single-reference path); 2–5 → `MultiReference`. Every other
    // family passes `&[]` and keeps the single `reference` (img2img init / IP-Adapter) path unchanged.
    let mut conditioning = if !multi_references.is_empty() {
        build_reference_conditioning(multi_references)
    } else {
        match reference {
            Some((image, strength)) => vec![Conditioning::Reference {
                image: image.clone(),
                strength: Some(*strength),
            }],
            None => Vec::new(),
        }
    };
    // Inpaint / outpaint mask (Ideogram 4 edit, sc-6303): a `Conditioning::Mask` (white = repaint)
    // alongside the source `Reference`. Only the Ideogram edit path supplies one today; every other
    // base-path family passes `None`.
    if let Some(mask) = edit_mask {
        conditioning.push(Conditioning::Mask {
            image: mask.clone(),
        });
    }
    let mut request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        true_cfg,
        sampler: sampler.map(str::to_owned),
        scheduler: scheduler.map(str::to_owned),
        scheduler_shift,
        guidance_method: guidance_method.map(str::to_owned),
        use_pid,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    enhance.apply(&mut request);
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::Engine(format!("generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images
                .pop()
                .ok_or_else(|| WorkerError::Engine("generator produced no image".to_owned()))?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "generator returned non-image output".to_owned(),
        )),
    }
}

/// Within-image step fraction mapped into the 0.10..0.95 generation band.
fn step_fraction(index: usize, current: u32, total: u32, count: u32) -> f64 {
    let per = 0.85 / count.max(1) as f64;
    let within = if total > 0 {
        (current as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0.1 + per * (index as f64 + within)).min(0.95)
}

/// Resolve a reference/source asset id to an in-memory RGB8 image (the engine VAE-encodes + resizes
/// it). Uses the indexed `ProjectStore::get_asset` → `file.path`. Shared by the MLX image/video
/// conditioning paths and the candle video i2v conditioning (sc-5175), so it lives here (both lanes)
/// rather than in a macOS-only include.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn load_reference_image(
    data_dir: &Path,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<Image> {
    let asset = ProjectStore::new(data_dir.to_path_buf(), "worker")
        .get_asset(project_id, asset_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("reference asset {asset_id}: {error}"))
        })?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("reference asset {asset_id} has no media path"))
        })?;
    // The asset's file.path comes from an on-disk sidecar the user can edit, so
    // route it through safe_project_path (rejects `..`/absolute components) rather
    // than a bare join — matching the media-jobs reads and keeping a poisoned
    // sidecar from reading an arbitrary file as the reference (sc-4278 / F-MLXW-14).
    let path = crate::safe_project_path(project_path, rel)?;
    let decoded = crate::image_decode::decode_image_any(&path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("reference image {}: {error}", path.display()))
        })?
        .to_rgb8();
    Ok(Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}

/// The clamped identity img2img-init strength for a strict-pose set, or `None` for the pose-only tier.
/// `Some(strength)` iff `advanced.referenceStrength > 0` AND a non-empty `referenceAssetId` is present;
/// `strength` is the user value clamped to `[0.05, 1.0]`, carrying the mflux `image_strength`
/// convention **verbatim** (higher strength → later denoise start → output stays closer to the init).
///
/// sc-8946 (F-144): this was duplicated line-for-line as `zimage_identity_strength` (Z-Image, sc-3146)
/// and `flux2_identity_strength` (FLUX.2-dev control) — an identity-gate change had to be made twice.
/// The gate + clamp are IDENTICAL across the two lanes (both mirror `MlxZImageAdapter`), so it lives
/// here once. Pure (request only), so the parity-sensitive gate + clamp stay unit-testable without I/O.
#[cfg(target_os = "macos")]
fn identity_strength(request: &ImageRequest) -> Option<f32> {
    let strength = request
        .advanced
        .get("referenceStrength")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .filter(|strength| *strength > 0.0)?;
    let has_asset = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .is_some_and(|id| !id.is_empty());
    has_asset.then(|| (strength as f32).clamp(0.05, 1.0))
}

/// Resolve the optional identity img2img-init for a strict-pose set: `Some((image, strength))` when
/// [`identity_strength`] engages (decoding `referenceAssetId` via [`load_reference_image`]), else
/// `None` (the default pose-only tier). The reference is shared across the whole pose set — identity is
/// constant; only the per-pose skeleton changes.
///
/// sc-8946 (F-144): the shared body of the former `resolve_identity_init` /
/// `resolve_flux2_identity_init` (both line-for-line copies). The Z-Image strict-pose stream and the
/// FLUX.2-dev control stream both call this.
#[cfg(target_os = "macos")]
fn resolve_identity_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    let Some(strength) = identity_strength(request) else {
        return Ok(None);
    };
    let asset_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .expect("identity_strength guarantees a non-empty referenceAssetId");
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    Ok(Some((image, strength)))
}

/// The source identity reference (decoded image + its asset id) a strict-control pose set scores its
/// finished poses against (epic 4406, sc-4410), or `None` when the job carries no identity reference.
///
/// A strict-control pose-library job locks the body pose via ControlNet but optionally carries a
/// character identity `referenceAssetId` (the same one the opt-in img2img init uses). When present, the
/// pose set is part of Character Studio and each finished pose is scored against that source identity
/// face through the shared [`crate::face_likeness`] seam — independent of whether the img2img init is
/// engaged (scoring observes the FINAL pose; the init only seeds it). `None` (a bare pose set with no
/// identity reference) ⇒ no scorer ⇒ the `faceLikeness` field is omitted from each sidecar — there is no
/// identity to compare against, which is honest, not an error.
///
/// Decoding is non-fatal: a reference that fails to load logs and yields `None` (scores omitted, the
/// set still renders) — scoring NEVER aborts a generation (the sc-4407 non-fatal AC). The source image
/// is decoded here ONCE and handed to the per-job scorer, which embeds it ONCE (the caching AC).
///
/// Lives in `base.rs` (compiled under BOTH the macOS pose lanes and the off-Mac candle-control lanes)
/// rather than the macOS-only `strict_control.rs` include, so the not-macOS candle strict-pose siblings
/// (`zimage_control` / `qwen_control` / `kolors_control` / `flux2_control_candle`) can resolve it.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn resolve_control_identity_source(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> Option<(Image, String)> {
    let asset_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())?
        .to_owned();
    match load_reference_image(&settings.data_dir, &request.project_id, &asset_id, project_path) {
        Ok(image) => Some((image, asset_id)),
        Err(error) => {
            tracing::warn!(
                error = %error,
                "strict-control identity reference decode failed; likeness scores omitted \
                 (generation continues)"
            );
            None
        }
    }
}

/// The source identity reference (decoded image + its asset id) an Image Studio "With Character"
/// (`character_image`) generation scores each finished image against (epic 4406, sc-4411), or `None`
/// when the job is not a plain character image with a reference face.
///
/// This is the GENERAL With-Character case — a regular `character_image` generation against a
/// `referenceAssetId` (the character reference shown in the Image Studio reference thumbnail), NOT an
/// angle set (`advanced.angleSet`) and NOT a pose-library set (`advanced.poses`). Those two are ALSO
/// `character_image` jobs but are already scored by sc-4409 (angles) / sc-4410 (poses) through the same
/// shared seam; this resolver deliberately returns `None` for them so the plain case never
/// double-attaches or conflicts on an angle/pose job. The gate is therefore:
/// `mode == "character_image"` AND a non-empty `referenceAssetId` AND no angle/pose grouping.
///
/// The asset id returned is the CURRENT job's `referenceAssetId`, so changing the reference asset
/// changes the source the score is computed against (an explicit sc-4411 acceptance criterion) — the
/// source is never cached across jobs or hardcoded.
///
/// Decoding is non-fatal (the sc-4407 contract): a reference that fails to load logs and yields `None`
/// (scores omitted, the generation still runs) — scoring NEVER aborts a generation. The source image is
/// decoded here ONCE and handed to the per-job scorer, which embeds it ONCE (the caching AC).
///
/// Lives in `base.rs` (compiled under BOTH the macOS routes and the off-Mac candle-control lanes) so the
/// candle siblings can resolve it identically.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn resolve_character_image_likeness_source(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> Option<(Image, String)> {
    if request.mode != "character_image" {
        return None;
    }
    // Angle / pose sets are already scored by sc-4409 / sc-4410 through the same shared seam; this is
    // the PLAIN With-Character path only, so exclude both groupings to avoid double-attaching.
    if !pose_entries(request).is_empty() || advanced::flag(&request.advanced, "angleSet") {
        return None;
    }
    resolve_control_identity_source(request, settings, project_path)
}

/// img2img (Remix) strength for a plain Ideogram 4 edit with no mask — mirrors the sdxl/z-image 0.6
/// edit default and the engine's `DEFAULT_IMG2IMG_STRENGTH`. Shared by the macOS MLX edit path and the
/// candle in-lane edit (sc-6598), so it compiles off-Mac under `backend-candle` too.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const IDEOGRAM_EDIT_STRENGTH: f32 = 0.6;
/// Heavier img2img strength for masked inpaint / outpaint (regenerate the painted region) — mirrors
/// the sdxl 0.85 inpaint default and the engine's `DEFAULT_INPAINT_STRENGTH`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const IDEOGRAM_INPAINT_STRENGTH: f32 = 0.85;

/// Upper bound on reference images for a Boogu instruction edit (sc-7645). The DiT's
/// `image_index_embedding` carries 5 per-image index slots (OmniGen2 lineage), so `N ∈ [1, 5]`
/// references can be packed into one edit (e.g. subject-from-A composed into scene-from-B); a plural
/// picker beyond that is capped here. Matches the `mlx-gen-boogu` / `candle-gen-boogu` `MAX_EDIT_REFS`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const BOOGU_MAX_EDIT_REFERENCES: usize = 5;

/// One `Conditioning::Reference` (a single reference) or one `Conditioning::MultiReference` (2–5) from
/// the resolved Boogu edit references (cloned per output). Empty references → empty (T2I fallback).
/// The single case stays a `Reference` so it is byte-identical to the pre-sc-7645 single-reference
/// path. Mirrors `flux2.rs::build_edit_conditioning`. The per-reference strength is inert for Boogu
/// (the edit is structural), so `None` is used. Not cfg-gated — called from the un-gated [`generate_one`].
fn build_reference_conditioning(references: &[Image]) -> Vec<Conditioning> {
    match references {
        [] => Vec::new(),
        [single] => vec![Conditioning::Reference {
            image: single.clone(),
            strength: None,
        }],
        many => vec![Conditioning::MultiReference {
            images: many.to_vec(),
        }],
    }
}

/// Reference asset ids for a Boogu instruction edit, in order. The multi-image picker sends the plural
/// `referenceAssetIds` — take all of them, capped at [`BOOGU_MAX_EDIT_REFERENCES`]; with no plural list
/// it falls back to the single Image-Edit `sourceAssetId` (`edit_image` mode). Mirrors
/// `edit_reference_ids`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn boogu_edit_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        // The parsed list is already trimmed + non-empty (sceneworks-core `string_list`).
        return request
            .reference_asset_ids
            .iter()
            .take(BOOGU_MAX_EDIT_REFERENCES)
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

/// Resolve the Boogu instruction-edit sources: the `N ∈ [1, 5]` reference images (plural
/// `referenceAssetIds`, else the single `sourceAssetId` — [`boogu_edit_reference_ids`]), each fit to the
/// output W×H (so it satisfies the engine's multiple-of-16 guard and aligns to the target aspect).
/// Returns the references in order; **empty** when not an edit / no source. The engine treats one
/// reference as `Conditioning::Reference` and 2–5 as `Conditioning::MultiReference` (each read by the
/// Qwen3-VL vision tower + VAE-encoded into the DiT spatial sequence). The per-reference strength is
/// inert for Boogu (the edit is structural — the engine ignores `Conditioning::Reference.strength`). No
/// mask / outpaint path (the descriptor accepts only `Reference` / `MultiReference`).
///
/// Shared by the macOS MLX `generate_stream` and the off-Mac candle `generate_candle_stream` (sc-7524):
/// Boogu is the same engine family for T2I and edit on both backends (the registered `boogu_image_edit`
/// resolves the source `Reference`(s) in-lane, like Ideogram), so both lanes resolve the edit sources the
/// same way. Its deps (`load_reference_image`, `fit_engine_image`) already compile off-Mac under
/// `backend-candle` (the Ideogram edit path uses them too).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_boogu_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Vec<Image>> {
    if request.mode != "edit_image" {
        return Ok(Vec::new());
    }
    let ids = boogu_edit_reference_ids(request);
    let mut references = Vec::with_capacity(ids.len());
    for id in &ids {
        let source = load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
        let source = fit_engine_image(source, request.width, request.height, &request.fit_mode)?;
        references.push(source);
    }
    Ok(references)
}

/// Resolve the Ideogram 4 `edit_image` conditioning (sc-6303) into the base MLX path's
/// `(source, strength, optional-mask)` shape (→ the engine's `Conditioning::Reference` +
/// `Conditioning::Mask`). Three sub-shapes, mirroring the sdxl edit classification:
///   * **img2img / Remix** — `sourceAssetId`, no mask: pre-fit the source to the output W×H
///     (crop/pad, never stretch) → `(source, 0.6, None)`.
///   * **masked inpaint** — `+ maskAssetId`: the mask fit with the same geometry → `(source, 0.85,
///     Some(mask))` (white = repaint).
///   * **outpaint** — `fit_mode == "outpaint"`: contain-pad the source onto the canvas and generate
///     the border via [`gen_core::imageops::outpaint_border_mask`] (using the ORIGINAL source dims so
///     it lines up), unioning any user mask (white wins).
///
/// `None` when not an edit job or no source asset (the caller falls back to plain txt2img).
///
/// Shared by the macOS MLX `generate_stream` and the off-Mac candle `generate_candle_stream`
/// (sc-6598): Ideogram is the same engine for T2I and edit on both backends, so both lanes resolve the
/// edit conditioning the same way. Its deps (`load_reference_image`, `fit_engine_image`, `non_empty`,
/// the `gen_core::imageops` mask helpers) are all already compiled off-Mac under `backend-candle`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_ideogram_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32, Option<Image>)>> {
    if request.mode != "edit_image" {
        return Ok(None);
    }
    let Some(asset_id) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    let is_outpaint = request.fit_mode == "outpaint";
    let has_user_mask = non_empty(&request.mask_asset_id);
    let strength = advanced::f32_clamped(
        &request.advanced,
        "strength",
        if is_outpaint || has_user_mask {
            IDEOGRAM_INPAINT_STRENGTH
        } else {
            IDEOGRAM_EDIT_STRENGTH
        },
        0.05..=1.0,
    );

    if is_outpaint {
        // Pad the source onto the target canvas (contain) and regenerate the border. The border mask
        // uses the ORIGINAL source dims so it lines up with the padded canvas (same contain geometry
        // as `fit_engine_image`'s "outpaint"/pad). Any user mask unions into the border (white wins).
        let (src_w, src_h) = (source.width, source.height);
        let canvas = fit_engine_image(source, request.width, request.height, "outpaint")?;
        let mut mask =
            gen_core::imageops::outpaint_border_mask(src_w, src_h, request.width, request.height);
        if has_user_mask {
            let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
            let user_mask = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                mask_id,
                project_path,
            )?;
            let user_mask = fit_engine_image(user_mask, request.width, request.height, "pad")?;
            mask = gen_core::imageops::union_masks(&mask, &user_mask).map_err(|error| {
                WorkerError::Engine(format!("ideogram outpaint mask union failed: {error}"))
            })?;
        }
        return Ok(Some((canvas, strength, Some(mask))));
    }

    // img2img / inpaint: pre-fit the source to the output W×H so an off-aspect edit doesn't stretch.
    let source = fit_engine_image(source, request.width, request.height, &request.fit_mode)?;
    let mask = if has_user_mask {
        let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
        let user_mask = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            mask_id,
            project_path,
        )?;
        // Align the mask to the source with the SAME fit geometry.
        Some(fit_engine_image(
            user_mask,
            request.width,
            request.height,
            &request.fit_mode,
        )?)
    } else {
        None
    };
    Ok(Some((source, strength, mask)))
}

// ---------------------------------------------------------------------------
// Shared pose + angle-prompt helpers. Used by the macOS Z-Image strict-pose control path
// (`zimage.rs`) AND the InstantID lane (`instantid.rs`) on BOTH backends — the candle InstantID
// provider (sc-5491) needs them off-Mac, so they live here in the shared include rather than in the
// macOS-only `zimage.rs` (same reason `load_reference_image` does). All `include!`d image-job files
// share one module, so moving these here keeps them visible to `zimage.rs` on macOS unchanged.
// ---------------------------------------------------------------------------

/// True for a present, non-blank optional asset id (the conditioning-asset presence test shared by
/// the SDXL advanced sub-mode, PuLID, and InstantID gates). Moved here from the macOS-only `sdxl.rs`
/// so the candle InstantID lane (sc-5491) can use it off-Mac.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn non_empty(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|id| !id.trim().is_empty())
}

/// The object-shaped `advanced.poses` entries (the strict-pose tier; empty otherwise).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn pose_entries(request: &ImageRequest) -> Vec<&Value> {
    request
        .advanced
        .get("poses")
        .and_then(Value::as_array)
        .map(|poses| poses.iter().filter(|pose| pose.is_object()).collect())
        .unwrap_or_default()
}

/// A pose's parsed keypoints, ready for [`crate::openpose_skeleton::draw_wholebody`].
// The candle InstantID pose lane reads only `keypoints` (→ OpenPose body skeleton); `hands`/`face` are
// the Z-Image whole-body strict-pose path's (macOS), so allow them dead off-Mac.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct PoseInput {
    keypoints: Vec<crate::openpose_skeleton::Keypoint>,
    hands: Option<Vec<crate::openpose_skeleton::Hand>>,
    face: Option<Vec<crate::openpose_skeleton::Keypoint>>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn parse_poses(request: &ImageRequest) -> Vec<PoseInput> {
    use crate::openpose_skeleton::{normalize_face, normalize_hands, normalize_keypoints};
    pose_entries(request)
        .into_iter()
        .map(|entry| PoseInput {
            keypoints: entry
                .get("keypoints")
                .map(normalize_keypoints)
                .unwrap_or_else(|| vec![None; 18]),
            hands: entry.get("hands").and_then(normalize_hands),
            face: entry.get("face").and_then(normalize_face),
        })
        .collect()
}

/// Stage the antelopev2 face stack for identity-likeness scoring (epic 4406), collapsing the
/// warn-and-`None` staging block duplicated across every scored image lane (F-024, sc-8826). When
/// `should_stage` is false the stack is never fetched and this is `None`; when true it downloads the
/// shared InstantID bundle (a no-op if already cached) and returns its dir. Staging is **non-fatal**:
/// a download failure logs `warn_message` and yields `None`, so the scorer is simply skipped and the
/// generation still renders (no scores). `warn_message` is the per-lane phrasing (e.g. the
/// `character_image` edit streams vs. the `pose-set` control lanes) so the log line is unchanged from
/// the hand-written blocks this replaces.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn stage_likeness(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    should_stage: bool,
    warn_message: &'static str,
) -> Option<PathBuf> {
    if !should_stage {
        return None;
    }
    match ensure_face_stack_dir(api, settings, job).await {
        Ok(dir) => Some(dir),
        Err(error) => {
            tracing::warn!(error = %error, "{warn_message}");
            None
        }
    }
}

/// The per-angle continuation clause appended to the user's prompt (parity with
/// `character_studio_angles.ANGLE_PROMPT_AUGMENTS`). Unknown angle → empty.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn angle_prompt_augment(angle: &str) -> &'static str {
    match angle {
        "front" => {
            "frontal portrait, looking directly at the camera, head and shoulders, neutral expression"
        }
        "three_quarter_left" => {
            "three-quarter left profile, head turned slightly to the left, three-quarter view"
        }
        "three_quarter_right" => {
            "three-quarter right profile, head turned slightly to the right, three-quarter view"
        }
        "left_profile" => {
            "full left profile, head turned 90 degrees to the left, side view of the head"
        }
        "right_profile" => {
            "full right profile, head turned 90 degrees to the right, side view of the head"
        }
        "up" => "looking up, head tilted slightly upward toward the sky",
        "down" => "looking down, head tilted slightly downward toward the floor",
        "up_left" => {
            "looking up and to the left, head tilted slightly upward and turned slightly to the left"
        }
        "up_right" => {
            "looking up and to the right, head tilted slightly upward and turned slightly to the right"
        }
        "down_left" => {
            "looking down and to the left, head tilted slightly downward and turned slightly to the left"
        }
        "down_right" => {
            "looking down and to the right, head tilted slightly downward and turned slightly to the right"
        }
        _ => "",
    }
}

/// Strip the user's base prompt for augmentation: trim whitespace, then trailing
/// `,`/`.`/`;` — exactly Python's `(base or "").strip().rstrip(",.;")` (which can
/// leave a trailing space, e.g. `"a . "` → `"a "`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn strip_base_prompt(base: &str) -> &str {
    base.trim().trim_end_matches([',', '.', ';'])
}

/// Append the per-angle clause to the user's base prompt (parity with
/// `augment_prompt_for_angle`). Empty base + unknown angle → empty string.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn augment_prompt_for_angle(base: &str, angle: &str) -> String {
    let augment = angle_prompt_augment(angle);
    let base = strip_base_prompt(base);
    if !base.is_empty() && !augment.is_empty() {
        format!("{base}, {augment}")
    } else if !augment.is_empty() {
        augment.to_owned()
    } else {
        base.to_owned()
    }
}

/// Per-family reference conditioning for the generic MLX lane (`generate_stream`), resolved once
/// (constant across the generation set). Bundles the four values the family dispatch produces so the
/// caller does one `resolve_generic_lane_conditioning(..)` call instead of the inline 5-way match —
/// the historical place per-family drift bugs land (sc-8828, F-026). The families served:
///  • Z-Image reference-identity img2img-init / edit-init (sc-3619 / epic 3529),
///  • FLUX.1 XLabs IP-Adapter (epic 3621 — schnell + dev; `strength = ipAdapterScale`, real CFG via
///    `trueCfgScale` on dev),
///  • Kolors img2img (sc-4765) + IP-Adapter-Plus reference (sc-4767), and
///  • Ideogram 4 img2img (Remix) + mask inpaint/outpaint (sc-6303).
/// Every other family (plain t2i, Boogu multi-reference) resolves to the all-`None`/`Vec::new` default.
#[cfg(target_os = "macos")]
#[derive(Default)]
struct LaneConditioning {
    /// Single img2img-init / IP-Adapter reference image + strength (Z-Image, FLUX.1 IP, Kolors,
    /// Ideogram) fed to the engine as `Conditioning::Reference`. `None` for plain t2i.
    identity_init: Option<(Image, f32)>,
    /// FLUX.1 / Kolors IP-Adapter weights directory threaded into the [`load_spec`]. `None` unless the
    /// IP-Adapter reference path is active.
    flux_ip_dir: Option<PathBuf>,
    /// FLUX.1-dev reference path's real-CFG scale (`trueCfgScale`). `None` for distilled/guidance-scalar
    /// families; the caller folds it into the effective `true_cfg` alongside the true-CFG-family scale.
    flux_true_cfg: Option<f32>,
    /// Ideogram 4 inpaint/outpaint mask (white = repaint) threaded to `generate_one` as
    /// `Conditioning::Mask`. `None` for every other family / plain img2img.
    ideogram_edit_mask: Option<Image>,
}

/// Resolve the [`LaneConditioning`] for `request` on the generic MLX lane. Mirrors the historical
/// inline family dispatch EXACTLY — same predicate order, same per-family values — so routing is
/// byte-identical (sc-8828). The strict-pose ControlNet / edit tiers divert earlier (in
/// `resolve_image_route`), so only the reference/identity/img2img families reach here.
#[cfg(target_os = "macos")]
fn resolve_generic_lane_conditioning(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
    has_reference: bool,
) -> WorkerResult<LaneConditioning> {
    if matches!(request.model.as_str(), "z_image_turbo" | "z_image_edit") {
        // Z-Image base path: `edit_image` → img2img-edit (sourceAssetId + strength, epic 3529);
        // otherwise the identity-init reference (referenceAssetId + referenceStrength, sc-3619).
        // Both feed the engine's single `Reference` conditioning; only the source + strength
        // keying differs. The strict-pose ControlNet tier diverts earlier (zimage_control_available).
        let init = if request.mode == "edit_image" {
            resolve_zimage_edit_init(request, settings, project_path)?
        } else {
            resolve_identity_init(request, settings, project_path)?
        };
        Ok(LaneConditioning {
            identity_init: init,
            ..Default::default()
        })
    } else if is_flux_model(&request.model) && has_reference && request.mode != "edit_image" {
        let reference_id = request
            .reference_asset_id
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .to_owned();
        let image = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            &reference_id,
            project_path,
        )?;
        let scale = advanced::f32_clamped(
            &request.advanced,
            "ipAdapterScale",
            FLUX_IP_SCALE,
            0.0..=1.0,
        );
        let ip_dir = resolve_flux_ip_adapter_dir(settings)?;
        // Real CFG only on dev (schnell is distilled — no CFG).
        let true_cfg = (request.model == "flux_dev").then(|| {
            advanced::f32_clamped(
                &request.advanced,
                "trueCfgScale",
                FLUX_IP_TRUE_CFG,
                1.0..=10.0,
            )
        });
        Ok(LaneConditioning {
            identity_init: Some((image, scale)),
            flux_ip_dir: Some(ip_dir),
            flux_true_cfg: true_cfg,
            ideogram_edit_mask: None,
        })
    } else if request.model == "kolors" && request.mode == "edit_image" {
        // Kolors img2img (sc-4765): `sourceAssetId` + `strength` → the engine's `Reference`
        // (img2img init, no IP-Adapter loaded). Kolors carries CFG through `guidance` + negative
        // prompt (resolved above), not `true_cfg`.
        let init = resolve_kolors_edit_init(request, settings, project_path)?;
        Ok(LaneConditioning {
            identity_init: init,
            ..Default::default()
        })
    } else if request.model == "kolors" && has_reference {
        // Kolors IP-Adapter-Plus reference (sc-4767): `referenceAssetId` → the IP image prompt at
        // `ipAdapterScale`. `with_ip_adapter` makes the engine treat the `Reference` as the image
        // prompt (decoupled cross-attn) rather than an img2img init.
        let (image, scale) = resolve_kolors_ip_reference(request, settings, project_path)?;
        let ip_dir = resolve_kolors_ip_adapter_dir(settings)?;
        Ok(LaneConditioning {
            identity_init: Some((image, scale)),
            flux_ip_dir: Some(ip_dir),
            flux_true_cfg: None,
            ideogram_edit_mask: None,
        })
    } else if matches!(request.model.as_str(), "ideogram_4" | "ideogram_4_turbo")
        && request.mode == "edit_image"
    {
        // Ideogram 4 img2img (Remix) + mask inpaint / outpaint (Edit), sc-6303: `sourceAssetId` →
        // the engine's `Reference` (img2img init); a `maskAssetId` (inpaint) or `fit_mode ==
        // "outpaint"` adds a `Conditioning::Mask` (white = repaint), threaded via `ideogram_edit_mask`.
        // Works in both quality (`ideogram_4`) and turbo (same base + TurboTime LoRA). No IP-Adapter.
        match resolve_ideogram_edit(request, settings, project_path)? {
            Some((source, strength, mask)) => Ok(LaneConditioning {
                identity_init: Some((source, strength)),
                flux_ip_dir: None,
                flux_true_cfg: None,
                ideogram_edit_mask: mask,
            }),
            None => Ok(LaneConditioning::default()),
        }
    } else {
        // Boogu instruction edit resolves its (1..5) references separately into `boogu_refs` below —
        // it uses the `MultiReference`-capable path, not the single `identity_init` reference.
        Ok(LaneConditioning::default())
    }
}

/// Real MLX generation: load once on a blocking thread, generate each image, and
/// stream step/decode/image events back to the async worker (which saves PNGs, emits
/// `assetWrites`, and polls cancel). MLX runs entirely on the blocking thread (the
/// `Box<dyn Generator>` is `!Send` and the MLX device is single-thread).
#[allow(clippy::too_many_arguments)]
#[cfg(target_os = "macos")]
async fn generate_stream(
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
    // sc-6568: a bf16 opt-in for Boogu fetches the full-precision `<variant>-bf16/` subfolder on
    // demand (the catalog ships only the Q8 default) before snapshot resolution. No-op for every
    // other model / the default Q8 path. sc-9607: the same on-demand pattern for Ideogram's `q8/`
    // tier (the catalog ships only q4) — was a documented follow-up, now wired on both lanes.
    ensure_boogu_tier_present(api, settings, job, request).await?;
    ensure_ideogram_tier_present(api, settings, job, request).await?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("model weights not found".to_owned()))?;
    // sc-3723: surface the descriptor-derived backend ("mlx" for every linked family today; a
    // future candle row would self-describe) over the gpu-id-derived label. Falls back to the
    // passed-in label only if a descriptor ever advertised an empty backend (never today).
    let backend = if model.backend().is_empty() {
        backend
    } else {
        model.backend()
    };
    // Descriptor-gated quant (mirrors the candle lane below): the MLX families advertise Q4/Q8
    // (`supported_quants`) and tolerate the Q8 default (a real quant on a dense convert, a no-op on an
    // already-packed turnkey). SANA joined this set in mlx-gen #654 (sc-8489): its descriptor now
    // advertises Q4/Q8 and its `load` ACCEPTS an advisory `spec.quantize` (the pre-quantized tier is
    // packed-detected from disk, #653), so it flows through the normal resolve_quant path like every
    // other matrix model. The `else` arm stays for any future engine that genuinely advertises no
    // quant — such a model loads dense.
    let (quant, quant_bits) = if model.supports_quant() {
        resolve_quant(request)
    } else {
        (None, None)
    };
    // sc-8820: the tier resolvers ([`standard_tier_subdir`] & friends) silently fall through
    // q4→q8→bf16 when the preferred tier isn't downloaded, but the quant above is derived from the
    // REQUEST — so a bf16 pick with only `q4/` present would render Q4 while the recipe records dense,
    // lying to the epic 8506 quant A/B workflow. Reconcile against the tier subdir actually resolved:
    // record the precision that ran + `warn!`/emit `quant_tier_downgraded` on a real fallback. SANA
    // (sc-8489) now ships standard q4/q8/bf16 turnkey tiers and advertises Q4/Q8, so it reconciles here
    // exactly like the other matrix models.
    //
    // sc-9362 (F-018 follow-up): dense-TE turnkeys (FLUX.2-klein) always derive `(None, None)` from
    // `resolve_quant` (the load quant must stay `None` so the dense bf16 TE is never re-quantized),
    // but their transformer is packed at q4/q8. Reconciling against that always-bf16 value made every
    // straight dense-TE job read as a bf16→qN "downgrade" — a spurious event, and pre-8820 the recipe
    // recorded bf16 for a q4/q8 transformer. Feed reconcile the transformer tier the request ACTUALLY
    // asked for ([`dense_te_requested_tier_bits`], mirroring the `standard_tier_subdir` mapping) so it
    // records the resolved transformer precision on EVERY job and only warns/emits on a genuine
    // fallback. `allow_quant_change=false` keeps the load quant `None` (TE stays dense bf16).
    let (quant, quant_bits) = if model.supports_quant() {
        let requested_for_reconcile = if is_dense_te_tier(request) {
            (None, dense_te_requested_tier_bits(request))
        } else {
            (quant, quant_bits)
        };
        reconcile_resolved_tier_quant(
            requested_for_reconcile,
            &weights_dir,
            !is_dense_te_tier(request),
            &request.model,
            &job.id,
            backend,
        )
    } else {
        (quant, quant_bits)
    };
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let (sampler, scheduler, scheduler_shift) = read_advanced_sampling_knobs(&request.advanced);
    // RealVisXL Lightning (sc-6075): a standalone few-step *distilled checkpoint* (the
    // SDXL-Lightning distillation is baked into the weights — no acceleration LoRA). It must run
    // on the engine's `lightning` (Euler-trailing) few-step schedule, not the 30-step
    // `euler_ancestral` default, so the schedule matches the checkpoint regardless of the UI
    // payload — mirrors the qwen `*_lightning` sampler forcing. The engine then applies the
    // CFG-off, few-step recipe (steps/guidance come from the manifest defaults via the model row).
    let sampler = if request.model == "realvisxl_lightning" {
        Some("lightning".to_owned())
    } else {
        sampler
    };
    // N3 (epic 7114): drop a sampler/scheduler the linked engine descriptor doesn't advertise back to
    // the engine default + emit an event, instead of letting `validate_request` hard-fail the whole
    // generation over a sampling knob (a stale recipe, manifest drift, or a per-backend gap). The forced
    // `realvisxl_lightning` sampler above is always in that family's advertised set, so it passes through.
    let caps = &model.descriptor.capabilities;
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
    // Guidance method (epic 7434 P5, sc-7448): the 4th sampling axis. N3-guarded against the engine's
    // `supported_guidance_methods` exactly like sampler/scheduler — an unadvertised method (a stale
    // recipe, a per-backend gap, or a method gated to an incompatible sampler) drops to the engine
    // default + a `sampling_knob_unsupported` event, never a `validate_request` hard-fail.
    let guidance_method = normalize_sampling_knob(
        read_advanced_guidance_method(&request.advanced),
        &caps.supported_guidance_methods,
        "guidanceMethod",
        &request.model,
        &job.id,
        backend,
    );
    // True-CFG families (Chroma) carry the CFG scale in `true_cfg`, not `guidance` (which their
    // engine rejects); `None` for every other family. The recipe records the effective CFG knob.
    let model_true_cfg = resolve_true_cfg(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &model);
    let raw_settings = mlx_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance.or(model_true_cfg),
    );
    let engine_id = model.engine_id();
    let adapter_label = model.adapter_label();
    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count)
        .map(|index| resolve_seed(request, index))
        .collect();
    // Reference conditioning for the base MLX path, resolved once (constant across the set):
    //  • Z-Image reference-identity img2img-init (sc-3619),
    //  • FLUX.1 XLabs IP-Adapter (epic 3621 — both schnell + dev; `strength = ipAdapterScale`, plus
    //    real CFG via `trueCfgScale` on dev), and
    //  • Kolors img2img (sc-4765, `edit_image` + `sourceAssetId`) + the IP-Adapter-Plus reference
    //    (sc-4767, `referenceAssetId` → image prompt at `ipAdapterScale`). Qwen/SDXL reference
    //    divert to their own advanced branches before reaching here.
    let has_reference = request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    // Per-family reference conditioning (Z-Image identity/edit-init, FLUX.1/Kolors IP-Adapter, Kolors
    // img2img, Ideogram edit + mask), resolved once — same predicate order + per-family values as the
    // historical inline 5-way match, now table-ized into one resolver (sc-8828, F-026). The strict-pose
    // ControlNet / edit tiers divert earlier in `resolve_image_route`.
    let LaneConditioning {
        identity_init,
        flux_ip_dir,
        flux_true_cfg,
        ideogram_edit_mask,
    } = resolve_generic_lane_conditioning(request, settings, project_path, has_reference)?;
    // Boogu instruction edit (epic 6387, multi-reference sc-7645): resolve the 1..5 source references
    // (the Qwen3-VL vision tower reads each + they VAE-encode into the DiT spatial sequence); the prompt
    // is the edit instruction. Threaded to `generate_one` as `Conditioning::Reference` (one reference) /
    // `MultiReference` (2–5). No mask / IP-Adapter (the descriptor accepts only Reference/MultiReference).
    let boogu_refs: Vec<Image> = if request.model == "boogu_image_edit" {
        resolve_boogu_edit(request, settings, project_path)?
    } else {
        Vec::new()
    };
    // The CFG scale passed to the engine as `true_cfg`: the FLUX.1-dev reference path's scale if
    // present, otherwise the true-CFG family scale (Chroma). `None` for the guidance-scalar and
    // distilled families, which carry CFG (if any) through `guidance` instead.
    let true_cfg = flux_true_cfg.or(model_true_cfg);

    // Ideogram 4 (epic 4725, sc-6501) is JSON-caption-only: a raw plain-text prompt is
    // out-of-distribution and stochastically renders the "Image blocked by safety filter"
    // placeholder (sc-6307, reference-confirmed faithful). The web Image Studio auto-expands plain
    // prompts into rich captions; this is the worker-side HARD GUARANTEE that raw plain text never
    // tokenizes — it wraps a non-caption prompt into a minimal valid caption (covers the API path
    // and any UI bypass). A prompt that is already a caption passes through unchanged. No-op for
    // every other family.
    let is_ideogram = crate::ideogram_caption::is_ideogram_model(&request.model);
    let prompt = if is_ideogram {
        crate::ideogram_caption::ensure_caption_prompt(&request.prompt)
    } else {
        request.prompt.clone()
    };
    let (width, height) = (request.width, request.height);
    let adapter_count = adapters.len();
    // sc-6135: caption upsampling (FLUX.2-dev only; every other engine ignores it). Resolved from
    // the request's advanced `enhancePrompt` toggle, gated to dev by the manifest `ui.promptEnhance`.
    let enhance = PromptEnhance::from_advanced(&request.advanced);
    // Per-generation PiD decode (epic 7840, sc-7849): resolve the PiD checkpoint + Gemma for this
    // model's latent space when `advanced.usePid` is set and the snapshots are cached; otherwise keep
    // the native VAE. `use_pid` and `spec.pid` stay in lockstep (the engine rejects a mismatch).
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();
    // PiD output tier (sc-10054): PiD super-resolves the base latent by a fixed 4×, so the effective
    // base picks whether the output lands on ~2K or ~4K. `4k`/native leave the requested dims untouched;
    // `2k` caps the base (also lowering the F-013 decode peak). Rebind before `generate_one`.
    let (width, height) = pid_effective_dims(width, height, use_pid, pid_output_tier(request));
    let mut spec = load_spec(weights_dir, quant, adapters, flux_ip_dir);
    if let Some(pid) = pid_weights {
        spec = spec.with_pid(pid.checkpoint, pid.gemma);
    }

    // Identity-likeness scoring (epic 4406, sc-4411 plain With-Character): the generic MLX lane serves
    // the remaining With-Character identity generators — Z-Image identity-init (`referenceAssetId` ⇒
    // img2img init), the FLUX.1 XLabs IP-Adapter, and the Kolors IP-Adapter-Plus reference — all of
    // which carry a character `referenceAssetId`. Score every output against that reference face through
    // the SHARED generator-agnostic seam, but ONLY for an Image Studio "With Character"
    // (`character_image`) generation; a z-image / kolors `edit_image` job (its source is `sourceAssetId`,
    // not an identity reference) is excluded by `resolve_character_image_likeness_source` (mode gate),
    // which also resolves the CURRENT job's reference (so changing it changes the scored source) and is
    // non-fatal. The `!Send` scorer is built ONCE inside the closure and reused across the N outputs.
    let likeness_source = resolve_character_image_likeness_source(request, settings, project_path);
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        likeness_source.is_some(),
        "character_image face-stack staging failed; likeness scores omitted",
    )
    .await;
    // Keep the source only if the face stack staged (otherwise no scorer can be built).
    let likeness_source = face_stack_dir.as_ref().and(likeness_source);

    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            // Per-job identity-likeness scorer built ONCE on the generator-worker thread (the `!Send`
            // face stack lives here); source embedded once, reused across every output (sc-4411). `None`
            // ⇒ not a With-Character generation, or non-fatal staging/construction failure ⇒ omitted.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            drive_gen_items_scored(tx, seeds, move |_index, seed, on_progress| {
                let render = |seed: i64, on_progress: &mut dyn FnMut(Progress)| {
                    generate_one(
                        generator,
                        &prompt,
                        width,
                        height,
                        seed,
                        steps,
                        guidance,
                        negative_prompt.clone(),
                        identity_init.as_ref(),
                        &boogu_refs,
                        ideogram_edit_mask.as_ref(),
                        true_cfg,
                        sampler.as_deref(),
                        scheduler.as_deref(),
                        scheduler_shift,
                        guidance_method.as_deref(),
                        use_pid,
                        &enhance,
                        &cancel,
                        on_progress,
                    )
                };
                let (mut out_w, mut out_h, mut pixels) = render(seed, on_progress)?;
                let mut final_seed = seed;
                // Detect-and-recover safety net (sc-6501): the caption guard makes the placeholder
                // rare, but a residual one can still occur even with a caption. Detect it via the
                // baked-text heuristic (NOT a std/flatness check — the text lifts std to ~10) and
                // reseed transparently, keeping the first clean render. Gated to Ideogram 4; a no-op
                // elsewhere (and on turbo, which is CFG-free and cannot produce the placeholder).
                if is_ideogram
                    && crate::ideogram_caption::looks_like_placeholder(&pixels, out_w, out_h)
                {
                    let retries = crate::ideogram_caption::placeholder_recovery_retries();
                    for attempt in 0..retries {
                        if cancel.is_cancelled() {
                            break;
                        }
                        let retry_seed = crate::ideogram_caption::recovery_seed(seed, attempt);
                        tracing::warn!(
                            "ideogram 4 placeholder detected (seed {seed}); reseeding {retry_seed} \
                             (attempt {}/{retries})",
                            attempt + 1,
                        );
                        let (rw, rh, rpixels) = render(retry_seed, on_progress)?;
                        let recovered =
                            !crate::ideogram_caption::looks_like_placeholder(&rpixels, rw, rh);
                        out_w = rw;
                        out_h = rh;
                        pixels = rpixels;
                        final_seed = retry_seed;
                        if recovered {
                            break;
                        }
                    }
                }
                // Score this finished image against the cached source embedding (sc-4411). Image build +
                // pixel clone is paid ONLY when a scorer exists (a With-Character generation) — a plain
                // t2i / edit job has no scorer, so this is a no-op with no clone. Non-frontal → honest
                // detected:false N/A; `None` scorer ⇒ field omitted.
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
                Ok(Some((final_seed, out_w, out_h, pixels, face_likeness)))
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
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// Whether `model` is served by the candle (Windows/CUDA) backend's generic lane (txt2img, plus the
/// Ideogram 4 in-lane edit, below). SDXL/RealVisXL (sc-3675) plus the four image families wired in
/// sc-5096 — z-image, flux schnell/dev, flux2-klein, qwen-image — plus Lens / Lens-Turbo (sc-5126, the
/// first candle family with quant + LoRA/LoKr) plus Ideogram 4 + Turbo (sc-6597/sc-6598, epic 6561) plus
/// the two FLUX.2-klein weight variants (sc-7459: `_kv` / `_true_v2`, sharing the `flux2_klein_9b` loader).
/// `realvisxl` (and the klein `_kv` / `_true_v2`) share an existing candle engine via a weights swap;
/// every other id maps 1:1 to its
/// `MODEL_TABLE` engine id. For the OTHER families, edit/control/reference shapes route to their bespoke
/// candle lanes (checked before this gate in the dispatch) or to the Python torch worker; Ideogram is
/// the exception — its img2img/mask edit is the SAME engine as its T2I, so `generate_candle_stream`
/// resolves the edit conditioning in-lane (mirroring the MLX `generate_stream`), no separate stream.
/// Lens is pure T2I; only quant + adapters, which `generate_candle_stream` resolves from the descriptor.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn is_candle_engine(model: &str) -> bool {
    matches!(
        model,
        "sdxl"
            | "realvisxl"
            // RealVisXL Lightning (sc-7176): shares the candle `sdxl` engine via a weights swap; the
            // few-step `lightning` sampler is forced in `generate_candle_stream`. txt2img-only (the
            // router defers its conditioning shapes to torch), so it rides the base candle txt2img lane.
            | "realvisxl_lightning"
            | "z_image_turbo"
            // Base (non-distilled, full-CFG) Z-Image (sc-8679, epic 8236): the registered candle
            // `z_image` base generator (shift-6.0 / ~50-step / real CFG) rides the generic candle txt2img
            // lane, the base sibling of `z_image_turbo`. Shares the `standard_tier_subdir` turnkey layout;
            // `generate_candle_stream` resolves its own repo/steps/guidance/negative from the descriptor.
            // txt2img only — edit / identity / strict-pose shapes have their own bespoke lanes (the
            // `z_image` control lane is `advanced.poses`, branched out before this gate).
            | "z_image"
            | "flux_schnell"
            | "flux_dev"
            | "flux2_klein_9b"
            // FLUX.2-klein weight variants (sc-7459): same candle `flux2_klein_9b` loader/arch, a
            // weights swap. `_kv` loads its own full diffusers tree; `_true_v2` loads the locally
            // converted diffusers dir (`modelPath` seam). txt2img only — `_kv`'s reference-edit /
            // KV-cache accel and every edit/reference shape defer to torch (`image_request_candle_eligible`).
            | "flux2_klein_9b_kv"
            | "flux2_klein_9b_true_v2"
            // FLUX.2-dev (sc-7458): the 32B flagship rides the generic candle txt2img lane like klein.
            // `generate_candle_stream` resolves Q4 (manifest `mlx.quantize: 4` + the dev descriptor's
            // `supported_quants`) so the dense snapshot is staged in CPU RAM and quantized onto the GPU
            // at load. Edit/control/reference shapes route to their bespoke lanes or torch (story 4).
            | "flux2_dev"
            | "qwen_image"
            | "chroma1_hd"
            | "chroma1_base"
            | "chroma1_flash"
            | "lens"
            | "lens_turbo"
            | "kolors"
            | "sensenova_u1_8b"
            | "sensenova_u1_8b_infographic_v2"
            | "sensenova_u1_8b_fast"
            | "sensenova_u1_8b_infographic_v2_fast"
            | "ideogram_4"
            | "ideogram_4_turbo"
            // Boogu-Image-0.1 (sc-7524, epic 6831): Base + Turbo (txt2img) and the Edit checkpoint, all
            // on the generic candle lane. Like Ideogram, `boogu_image_edit`'s instruction edit is in-lane
            // (the engine resolves the source `Reference`), not a separate bespoke stream.
            | "boogu_image"
            | "boogu_image_turbo"
            | "boogu_image_edit"
            // Krea 2 Turbo (sc-7581, epic 7565 P4): txt2img + inference LoRA/LoKr on the generic candle
            // lane (CFG-free 8-step). sc-9092: the candle lane now packed-loads the SAME
            // `SceneWorks/krea-2-turbo-mlx` q8/q4 turnkey subdir the macOS path loads (candle-gen sc-9411
            // packed-detect, via the shared `resolve_weights_dir`/`krea_model_subdir`) — the ad-hoc
            // `candle_krea_repo` bf16 diffusers rehost is retired. The `candle-gen-krea` descriptor
            // advertises supports_lora/supports_lokr (sc-7836), so `generate_candle_stream` resolves a
            // `krea_2_raw`-trained adapter via `model.supports_adapters()`. No edit/reference/control
            // shapes; on-the-fly quant stays descriptor-gated (`supported_quants` still `&[]` — the
            // packed turnkey self-describes its tier at load).
            | "krea_2_turbo"
            // Stable Diffusion 3.5 (sc-7880, epic 7982): Large / Large Turbo / Medium all ride the generic
            // candle txt2img lane (the `candle-gen-sd3` provider). `generate_candle_stream` resolves Q4/Q8
            // from the descriptor + manifest (`mlx.quantize` 8) — the dense MMDiT folds to Q4_0/Q8_0 at
            // load (sc-7879). Pure txt2img; no edit/reference/control/LoRA candle path (those defer to
            // torch via `image_request_candle_eligible`).
            | "sd3_5_large"
            | "sd3_5_large_turbo"
            | "sd3_5_medium"
    )
}

/// The per-asset `adapter` id recorded for a candle image engine (`candle_<family>`), the candle
/// sibling of the `MODEL_TABLE` `mlx_<family>` labels. Used both per-asset (`generate_candle_stream`)
/// and at the generation-set level (`adapter_id`) so the sidecar + result agree on the backend.
/// (sc-5099 extends this same labeling to the video + caption engines.)
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_adapter_label(model: &str) -> &'static str {
    match model {
        // Base z_image (sc-8679) shares the candle z-image family label with Turbo.
        "z_image_turbo" | "z_image" => "candle_z_image",
        "flux_schnell" | "flux_dev" => "candle_flux",
        // The base klein + its `_kv` / `_true_v2` weight variants (sc-7459) + dev all run candle FLUX.2.
        "flux2_klein_9b" | "flux2_klein_9b_kv" | "flux2_klein_9b_true_v2" | "flux2_dev" => {
            "candle_flux2"
        }
        "qwen_image" => "candle_qwen",
        "chroma1_hd" | "chroma1_base" | "chroma1_flash" => "candle_chroma",
        "lens" | "lens_turbo" => "candle_lens",
        "kolors" => "candle_kolors",
        "sensenova_u1_8b"
        | "sensenova_u1_8b_infographic_v2"
        | "sensenova_u1_8b_fast"
        | "sensenova_u1_8b_infographic_v2_fast" => "candle_sensenova",
        "ideogram_4" | "ideogram_4_turbo" => "candle_ideogram",
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit" => "candle_boogu",
        "krea_2_turbo" => "candle_krea",
        // Stable Diffusion 3.5 (sc-7880): Large / Large Turbo / Medium share the candle SD3.5 engine.
        "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium" => "candle_sd3",
        // sdxl / realvisxl share the candle "sdxl" engine.
        _ => CANDLE_ADAPTER,
    }
}

/// Windows/CUDA candle execution path (sc-3675 SDXL, generalized in sc-5096). The macOS dispatch is
/// MLX-bound; candle is a narrow **txt2img-only** lane, so this is a trimmed sibling of
/// [`generate_stream`] that drives the SAME neutral streaming harness (`start_cached_gen_stream` →
/// `generate_one` → `consume_gen_events`) against the registry-resolved candle generator.
///
/// Backend-neutral resolution (sc-5096): the per-engine repo / steps / guidance / negative prompt all
/// come from the shared [`mlx_model`] join (`MODEL_TABLE` row + the linked candle descriptor), exactly
/// like the MLX path — so adding a family needs no new dispatch logic, just its provider crate linked.
/// Quant + LoRA/LoKr are **descriptor-gated** (sc-5126): resolved (via the same `resolve_quant` /
/// `resolve_adapters` the MLX path uses) only when the linked candle descriptor advertises them — i.e.
/// for Lens (Q4/Q8 + LoRA/LoKr); the sc-3675/sc-5096 families advertise neither, so they stay dense +
/// adapter-free exactly as before. No reference/img2img/control — those shapes fall back to the Python
/// worker upstream (`image_request_candle_eligible`). Reached only when `backend_candle_enabled`
/// (default off → production routing unchanged until parity).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    _device_backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let adapter_label = candle_adapter_label(&request.model);
    // Join the MODEL_TABLE row with the linked candle descriptor (same resolver the MLX path uses).
    // `None` means the candle provider crate for this id wasn't linked/registered — fail loud rather
    // than silently stubbing.
    let model = mlx_model(&request.model).ok_or_else(|| {
        WorkerError::Engine(format!(
            "candle backend not linked for model {} (no registered generator)",
            request.model
        ))
    })?;
    let engine_id = model.engine_id();
    // Report the descriptor's tensor backend ("candle"), not the gpu-id device label
    // (`_device_backend`), on the streamed progress + inference events (sc-3678) — parity with the
    // macOS path's `model.backend()` override, so the worker log + the UI architecture pill clearly
    // attribute the run to Candle.
    let backend = if model.backend().is_empty() {
        "candle"
    } else {
        model.backend()
    };
    let is_ideogram = crate::ideogram_caption::is_ideogram_model(&request.model);
    // Standard-tier weight resolution, SHARED with the MLX lane (sc-9092, epic 9083 gap #3). Every
    // candle image family — Ideogram / Boogu / Krea / Lens included — now packed-loads the SAME
    // SceneWorks MLX-packed per-tier turnkey the macOS path uses: as of the candle-gen rollout all 11
    // packed-load-capable crates read a packed `q4/q8/bf16` (or the Ideogram/Boogu/Krea legacy-layout)
    // turnkey subdir directly, so the four ad-hoc `candle_{ideogram,boogu,krea,lens}_repo` resolvers
    // (which pointed candle at a SEPARATE bf16 diffusers rehost because it couldn't read the packed
    // turnkeys) are retired. `resolve_weights_dir` applies the identical dispatch the MLX path uses —
    // an explicit `modelPath` override (FLUX.2-klein `_true_v2` convert-at-install), then the Ideogram
    // (`ideogram_model_subdir`) / Boogu (`boogu_model_subdir`) / Krea (`krea_model_subdir`) per-family
    // subdir, then `standard_tier_subdir` (Lens + the STANDARD_TIER_MODELS registry) resolving the
    // requested `advanced.mlxQuantize` → q4/q8/bf16 tier — so the candle lane needs no bespoke repo
    // logic. `model` is already resolved via `mlx_model` above, so a `None` here means only the
    // snapshot is absent (unfetched turnkey), which stays a loud load error.
    //
    // sc-9607 (epic 9083): off-Mac on-demand fetch of the non-default Ideogram/Boogu tiers before
    // resolution — the catalog pulls only the shipped default (ideogram q4, boogu Q8), so a candle
    // job that opts into another tier (`advanced.mlxQuantize`) needs its subdir pulled first. These
    // were `#[cfg(target_os = "macos")]` (boogu) / absent (ideogram), so off-Mac previously fell back
    // to the default tier; now Windows/Linux gets the same q4/q8/bf16 picker as macOS. No-op for the
    // default tier / every other family / an unfetched turnkey (falls through to the load error below).
    ensure_boogu_tier_present(api, settings, job, request).await?;
    ensure_ideogram_tier_present(api, settings, job, request).await?;
    // Krea 2 INT8-ConvRot tier (sc-9300, epic 9083): fetch the canonical bf16 base surface on demand
    // (the ConvRot catalog download pulls only the DiT single-file), then resolve the two LoadSpec
    // inputs. `None` = not a ConvRot job (or an artifact still absent) → the normal dense/packed path
    // below. When it resolves, the LoadSpec `weights` becomes the bf16 base DIR and `text_encoder` the
    // ConvRot DiT `File` — the exact shape the candle-gen krea engine's `convrot_selector` expects.
    ensure_krea_convrot_base_present(api, settings, job, request).await?;
    let convrot = resolve_krea_convrot(request, settings);
    let weights_dir = if let Some((base_dir, _)) = convrot.as_ref() {
        base_dir.clone()
    } else {
        resolve_weights_dir(request, settings)?.ok_or_else(|| {
            let repo = model_repo(request, &model);
            WorkerError::InvalidPayload(format!("candle weights snapshot not found for {repo}"))
        })?
    };

    // Descriptor-derived denoise/guidance surface (distilled families → no guidance/negative; guided
    // families → the scale + negative prompt). Identical to the MLX path; quant + LoRA are omitted.
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let true_cfg = resolve_true_cfg(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);

    // Per-payload flash/accel-attention (sc-3674): the UI Advanced toggle sends `advanced.flashAttn`
    // (default on). Process-global toggle, set before the generator loads (the candle pipeline reads
    // it at load) — race-free because the worker runs image jobs sequentially. The providers expose
    // the runtime knob under different names (SDXL `set_flash_attn`, Z-Image `set_accel_attn`); the
    // diffusion-transformer families (flux/flux2/qwen) bake it via the build feature with no runtime
    // toggle. No effect unless the crate was built with its flash/accel feature.
    let flash_attn = request
        .advanced
        .get("flashAttn")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    match request.model.as_str() {
        // realvisxl_lightning shares the candle `sdxl` engine (sc-7176), so the SDXL flash toggle applies.
        "sdxl" | "realvisxl" | "realvisxl_lightning" => candle_gen_sdxl::set_flash_attn(flash_attn),
        // Base z_image (sc-8679) shares the candle z-image accel-attention toggle with Turbo.
        "z_image_turbo" | "z_image" => candle_gen_z_image::set_accel_attn(flash_attn),
        _ => {}
    }

    // Descriptor-gated quant + adapters (sc-5126). Lens advertises Q4/Q8 (Q8 default) + LoRA/LoKr, so
    // it resolves them like the MLX path; the sc-3675/sc-5096 families advertise neither and skip both
    // (dense bf16/fp16, no adapters) — preserving their shipped behavior. The router only lets a quant
    // request / LoRA reach this worker for a family that supports it (`image_request_candle_eligible`).
    let (quant, quant_bits) = if convrot.is_some() {
        // INT8-ConvRot (sc-9300): the int8 DiT replaces the dense transformer wholesale — a bits-based
        // load-time `Quant` is meaningless (and the candle-gen krea engine rejects a quant overlay on
        // the ConvRot path). Force dense-None; the recipe records no `mlxQuantize` bits for this tier.
        (None, None)
    } else if model.supports_quant() {
        resolve_quant(request)
    } else {
        (None, None)
    };
    let adapters = if convrot.is_some() {
        // ConvRot does not combine with LoRA/LoKr (the int8 DiT is not adapter-wired); skip adapters.
        Vec::new()
    } else if model.supports_adapters() {
        resolve_adapters(request, settings)?
    } else {
        Vec::new()
    };
    let adapter_count = adapters.len();

    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count).map(|index| resolve_seed(request, index)).collect();
    // Ideogram 4 (epic 4725, sc-6501) is JSON-caption-only: a raw plain-text prompt is out-of-
    // distribution and stochastically renders the "Image blocked by safety filter" placeholder. Wrap a
    // non-caption prompt into a minimal valid caption — the same worker-side guarantee the macOS path
    // applies via `ideogram_caption::ensure_caption_prompt`. No-op (a clone) for every other family.
    // (`is_ideogram` was resolved above with the weights repo.)
    let prompt = if is_ideogram {
        crate::ideogram_caption::ensure_caption_prompt(&request.prompt)
    } else {
        request.prompt.clone()
    };
    // In-lane edit conditioning (sc-6598 Ideogram / sc-7524 Boogu): resolve the source `Reference`
    // (+ optional `Mask` for Ideogram) + strength once, seed-independent — the candle sibling of the MLX
    // `generate_stream` edit path. Both families edit on the SAME engine as their T2I (no separate bespoke
    // stream), so the generic lane resolves the source here. `resolve_ideogram_edit` / `resolve_boogu_edit`
    // return `None` for a non-edit (T2I) job, and each is gated to its family so a stray job reaching this
    // generic lane is untouched. Boogu has no mask (the `boogu_image_edit` descriptor accepts only
    // `Reference` — the Qwen3-VL vision tower reads it + it VAE-encodes into the DiT reference latent).
    // Other candle edit families (sdxl/flux2/qwen/z-image) have their own bespoke streams (checked before
    // this dispatch).
    let (edit_reference, edit_mask) = if is_ideogram {
        match resolve_ideogram_edit(request, settings, project_path)? {
            Some((source, strength, mask)) => (Some((source, strength)), mask),
            None => (None, None),
        }
    } else {
        (None, None)
    };
    // Boogu instruction edit (sc-7524, multi-reference sc-7645): resolve the 1..5 references here (each
    // read by the Qwen3-VL vision tower + VAE-encoded into the DiT spatial sequence) — the
    // `MultiReference`-capable path, not the single `edit_reference`. Threaded to `generate_one` as
    // `Conditioning::Reference` (one) / `MultiReference` (2–5). Empty for non-Boogu / non-edit jobs.
    let boogu_refs: Vec<Image> = if request.model == "boogu_image_edit" {
        resolve_boogu_edit(request, settings, project_path)?
    } else {
        Vec::new()
    };
    let (width, height) = (request.width, request.height);
    // Per-payload sampler / scheduler / schedule-shift, mirroring the MLX `generate_stream` lane (the
    // 1753 front-half advanced carrier — epic 7114 P5, sc-7127). RealVisXL Lightning (sc-7176) forces the
    // few-step `lightning` id regardless of the payload: candle-gen-sdxl advertises `["ddim", "lightning"]`,
    // so it survives the N3 guard below. Every value is then run through `normalize_sampling_knob` against
    // this family's advertised surface — a name candle doesn't honor (candle adopts the unified framework in
    // P4, so most families advertise only their family default today) is dropped back to the engine default
    // + a `sampling_knob_unsupported` event, never a hard-fail. The curated knobs light up per-family with
    // zero worker change as the candle engines are adopted.
    let (sampler, scheduler, scheduler_shift) = read_advanced_sampling_knobs(&request.advanced);
    let sampler = if request.model == "realvisxl_lightning" {
        Some("lightning".to_owned())
    } else {
        sampler
    };
    let caps = &model.descriptor.capabilities;
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
    // Guidance method (epic 7434 P5, sc-7448), N3-guarded against the candle descriptor's advertised set.
    let guidance_method = normalize_sampling_knob(
        read_advanced_guidance_method(&request.advanced),
        &caps.supported_guidance_methods,
        "guidanceMethod",
        &request.model,
        &job.id,
        backend,
    );
    // sc-6135 / sc-7458: caption upsampling is FLUX.2-dev-only. On candle (off-Mac) dev now runs here,
    // but the Mistral3/Pixtral caption-upsampler vision tower is NOT ported (deferred to epic 6564
    // story 4), so `enhance` degrades to **passthrough**: it is carried onto the `GenerationRequest`
    // for uniformity, but the candle `Flux2Generator` ignores `enhance_prompt`, so the raw prompt is
    // used verbatim. Critically this is a no-op, NOT a fall-back to the Python torch worker — the dev
    // T2I job stays on candle (a future candle enhancer lights up here with no router change). Every
    // other candle family ignores the fields too.
    let enhance = PromptEnhance::from_advanced(&request.advanced);
    // Record the effective CFG knob (guidance for guided families, else true_cfg) + quant bits in the
    // recipe, so a Lens asset's sidecar reflects the Q4/Q8 it ran at (parity with the MLX path). The
    // recorded repo is the resolved model repo (the MLX turnkey the candle lane now packed-loads from,
    // sc-9092) — the same `model_repo` the MLX path records.
    let repo = model_repo(request, &model);
    let raw_settings = mlx_raw_settings(request, &repo, steps, quant_bits, guidance.or(true_cfg));
    // Per-generation PiD decode (epic 7840): resolve the PiD checkpoint + Gemma for this model's latent
    // space when `advanced.usePid` is set and the snapshots are cached; otherwise keep the native VAE.
    // `use_pid` and `spec.pid` stay in lockstep (the engine rejects a mismatch). Every candle image
    // provider reads `spec.pid` (candle-gen sc-7853: sdxl/flux/flux2/qwen-image/z-image/chroma/boogu/
    // ideogram/kolors/krea/lens), so this un-gates the toggle across the whole off-Mac catalog — using the
    // SAME `resolve_pid_weights` model→backbone gate as the macOS lane (a non-eligible model → `None` →
    // native VAE, so this is a no-op for anything without a PiD backbone).
    // ConvRot (sc-9300) does not combine with a PiD decode overlay — the int8 DiT consume path replaces
    // the transformer wholesale and is not PiD-wired (the candle-gen krea engine rejects the combo). So
    // suppress PiD when ConvRot is selected; every non-ConvRot job resolves PiD exactly as before.
    let pid_weights = if convrot.is_some() {
        None
    } else {
        resolve_pid_weights(request, &settings.data_dir, &request.model)?
    };
    let use_pid = pid_weights.is_some();
    // PiD output tier (sc-10054): 2K caps the effective base so PiD's fixed 4× lands on ~2048 (default
    // 4K/native leaves the requested dims untouched). Rebind before `generate_one`.
    let (width, height) = pid_effective_dims(width, height, use_pid, pid_output_tier(request));
    let mut spec = load_spec(weights_dir, quant, adapters, None);
    if let Some(pid) = pid_weights {
        spec = spec.with_pid(pid.checkpoint, pid.gemma);
    }
    // INT8-ConvRot LoadSpec seam (sc-9300, epic 9083): ride the ConvRot DiT single-file on the shared,
    // already-optional `LoadSpec::text_encoder` as a `WeightsSource::File` while `spec.weights` stays the
    // canonical Krea 2 bf16 snapshot `Dir` (set as `weights_dir` above). The candle-gen krea engine's
    // `convrot_selector` decodes a `File` here → `load_components_convrot` (which enforces the sm_89
    // compute-cap floor); a `Dir`/`None` there is the normal dense/packed path. Other engines ignore it.
    if let Some((_, convrot_dit)) = convrot {
        spec.text_encoder = Some(WeightsSource::File(convrot_dit));
    }

    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("candle {engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(tx, seeds, move |_index, seed, on_progress| {
                let render = |seed: i64, on_progress: &mut dyn FnMut(Progress)| {
                    generate_one(
                        generator,
                        &prompt,
                        width,
                        height,
                        seed,
                        steps,
                        guidance,
                        negative_prompt.clone(),
                        edit_reference.as_ref(),
                        &boogu_refs,
                        edit_mask.as_ref(),
                        true_cfg,
                        // Per-payload sampler / scheduler / schedule-shift (sc-7127), already N3-guarded
                        // against this family's advertised surface above. RealVisXL Lightning forces
                        // `lightning`; most candle families advertise only their default until P4, so an
                        // unsupported request was dropped to `None` (the engine default) before reaching here.
                        sampler.as_deref(),
                        scheduler.as_deref(),
                        scheduler_shift,
                        // Guidance method, N3-guarded against this family's advertised surface above
                        // (sc-7448). candle adopts cfg_pp/cfg_rescale/apg in P4; until then an unsupported
                        // method was already dropped to `None` (the engine default) before reaching here.
                        guidance_method.as_deref(),
                        // Per-generation PiD decode (epic 7840): route the final latent through the
                        // `spec.pid` super-resolving student when resolved (opt-in + snapshots cached),
                        // else the native VAE. Every candle image provider reads `spec.pid` (sc-7853), so
                        // the whole off-Mac catalog honors the toggle in lockstep with `spec.pid` above.
                        use_pid,
                        &enhance,
                        &cancel,
                        on_progress,
                    )
                };
                let (mut out_w, mut out_h, mut pixels) = render(seed, on_progress)?;
                let mut final_seed = seed;
                // Ideogram 4 placeholder detect-and-reseed (sc-6858, parity with the macOS
                // `generate_stream` net, sc-6501): the caption guard above makes it rare, but a residual
                // "Image blocked by safety filter" placeholder can still occur even with a caption.
                // Detect via the baked-text heuristic and reseed transparently, keeping the first clean
                // render. Gated to Ideogram 4; a no-op for every other candle family, for turbo (CFG-free,
                // cannot produce it), and for an edit (the output is anchored to a real source latent, so
                // `looks_like_placeholder` returns false).
                if is_ideogram
                    && crate::ideogram_caption::looks_like_placeholder(&pixels, out_w, out_h)
                {
                    let retries = crate::ideogram_caption::placeholder_recovery_retries();
                    for attempt in 0..retries {
                        if cancel.is_cancelled() {
                            break;
                        }
                        let retry_seed = crate::ideogram_caption::recovery_seed(seed, attempt);
                        tracing::warn!(
                            "ideogram 4 placeholder detected (seed {seed}); reseeding {retry_seed} \
                             (attempt {}/{retries})",
                            attempt + 1,
                        );
                        let (rw, rh, rpixels) = render(retry_seed, on_progress)?;
                        let recovered =
                            !crate::ideogram_caption::looks_like_placeholder(&rpixels, rw, rh);
                        out_w = rw;
                        out_h = rh;
                        pixels = rpixels;
                        final_seed = retry_seed;
                        if recovered {
                            break;
                        }
                    }
                }
                Ok(Some((final_seed, out_w, out_h, pixels)))
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
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// Consume the streamed generation events (step / decoding / image) from the blocking
/// thread: write each finished image as an asset fact, stream progress, and poll cancel
/// ~every 2s (draining the channel after a cancel so the blocking sender never blocks).
/// Shared by the base txt2img path ([`generate_stream`]) and the Z-Image strict-pose
/// control path ([`generate_zimage_control_stream`]). `total` is the number of images
/// the job produces (the request count, or the pose count).
#[allow(clippy::too_many_arguments)]
async fn consume_gen_events(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    adapter_label: &str,
    raw_settings: &JsonObject,
    total: usize,
    mut rx: tokio::sync::mpsc::Receiver<GenEvent>,
    cancel: CancelFlag,
    blocking: tokio::task::JoinHandle<WorkerResult<()>>,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let total_u32 = total as u32;
    let mut canceled = false;
    let mut last_cancel_check = Instant::now();
    // Per-image inference lifecycle events (sc-3450), parity with the Python worker's
    // `image_inference_start`/`image_inference_complete`. The first event for an index
    // marks its start; `GenEvent::Image` marks completion. This is the single shared
    // streaming seam, so every MLX image family reports the same phases on mlx-worker.log
    // + the in-app Logs screen.
    let mut started: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut mark_started = |index: usize| {
        if started.insert(index) {
            emit_event(
                "image_inference_start",
                json!({
                    "jobId": job.id,
                    "imageIndex": index,
                    "imageCount": total,
                    "backend": backend,
                }),
            );
        }
    };
    // Bind the blocking generation task to its cancel flag (sc-8804, F-003): every `update_job`/
    // `heartbeat` `?` in the loop below returns early on a transient POST failure or a 409
    // (stale-sweep reclaim); on that early return this guard trips the engine `CancelFlag` and
    // aborts the still-running denoise instead of leaving it burning GPU memory alongside the next
    // claimed job. `cancel` is kept alongside (it's `Clone`) for the in-loop `begin_image_cancel`
    // poller; the guard drives only the drop-time teardown.
    let mut guard = CancelJoinGuard::new(cancel.clone(), blocking);
    // Heartbeat + cancel-poll on a fixed interval, not only when the blocking
    // thread emits an event. The cold model-load phase (multi-GB load + quantize)
    // emits nothing, so without an interval arm the worker reports no Busy
    // heartbeat and honors no cancel until the first denoise step — long enough
    // for the API's staleness check to think it died (sc-4276 / F-MLXW-12;
    // mirrors the caption-job select!-with-interval).
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Run the event loop capturing its Result so any `?`-error path performs the explicit awaited
    // bounded-join teardown BEFORE returning, instead of drop-and-run (sc-8804, F-003).
    let loop_result: WorkerResult<()> = async {
        loop {
        tokio::select! {
            maybe_event = rx.recv() => {
                let Some(event) = maybe_event else {
                    break;
                };
                if canceled {
                    continue; // drain remaining events so the blocking sender never blocks.
                }
                match event {
            GenEvent::Step {
                index,
                current,
                total: step_total,
            } => {
                mark_started(index);
                if last_cancel_check.elapsed() >= Duration::from_secs(2) {
                    last_cancel_check = Instant::now();
                    // sc-9618: a process shutdown is a cancel checkpoint too — short-circuit the API
                    // poll so a quit stops the gen at this step, matching a user cancel.
                    if shutdown_requested() || cancel_requested_peek(api, &job.id).await {
                        // Trip the flag + show "Cancelling…", but stay non-terminal until the
                        // in-flight image actually stops (terminal Canceled posted after the
                        // blocking run returns) — sc-5515.
                        begin_image_cancel(api, &job.id, &cancel, plan, asset_writes, backend).await;
                        canceled = true;
                        continue;
                    }
                }
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        step_fraction(index, current, step_total, total_u32),
                        &format!("Image {}/{total} — step {current}/{step_total}.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Decoding { index } => {
                mark_started(index);
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        step_fraction(index, 1, 1, total_u32),
                        &format!("Image {}/{total} — decoding.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Image {
                index,
                seed,
                width,
                height,
                pixels,
                face_likeness,
            } => {
                // The identity-likeness post-pass (sc-4409) scores each image on the blocking thread
                // and hands the pre-built `faceLikeness` block back through the event. Attach it to a
                // PER-IMAGE clone of the shared raw settings under the sidecar key so each angle's
                // asset carries its own honest score (an N/A `detected:false` block for profile/up/
                // down views), while every non-scoring path leaves `face_likeness` `None` ⇒ the field
                // is omitted entirely (the sc-4408 omit-when-absent contract).
                let mut image_raw_settings = raw_settings.clone();
                if let Some(block) = face_likeness {
                    image_raw_settings.insert(
                        crate::face_likeness::FACE_LIKENESS_FACT_KEY.to_owned(),
                        Value::Object(block),
                    );
                }
                // Encode + write the asset PNG off the async runtime thread (sc-8909 / F-107).
                let plan_for_task = plan.clone();
                let adapter_for_task = adapter_label.to_owned();
                let project_path_for_task = project_path.to_owned();
                let fact = tokio::task::spawn_blocking(move || {
                    write_image_asset(
                        &plan_for_task,
                        index,
                        seed,
                        width,
                        height,
                        pixels,
                        &adapter_for_task,
                        image_raw_settings,
                        &project_path_for_task,
                    )
                })
                .await
                .map_err(|error| crate::task_join_error("image asset write task", error))??;
                asset_writes.push(Value::Object(fact));
                emit_event(
                    "image_inference_complete",
                    json!({
                        "jobId": job.id,
                        "imageIndex": index,
                        "backend": backend,
                    }),
                );
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        0.1 + 0.85 * ((index + 1) as f64 / total as f64),
                        &format!("Generated image {}/{total}.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
            }
                }
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                // sc-9618: honor a process shutdown on every tick (a local flag read, no API cost, so
                // not throttled by the 2s user-cancel poll) so a quit trips the engine cancel promptly.
                if !canceled && (shutdown_requested()
                    || (last_cancel_check.elapsed() >= Duration::from_secs(2) && {
                        last_cancel_check = Instant::now();
                        cancel_requested_peek(api, &job.id).await
                    }))
                {
                    begin_image_cancel(api, &job.id, &cancel, plan, asset_writes, backend).await;
                    canceled = true;
                }
            }
        }
        }
        Ok(())
    }
    .await;
    if let Err(error) = loop_result {
        guard.cancel_and_join().await;
        return Err(error);
    }

    // Loop exited cleanly — reclaim the handle (disarming the drop-guard) and join the finished task.
    let task_result = guard
        .into_handle()
        .await
        .map_err(|error| task_join_error("generation task join", error))?;
    if canceled {
        // The generation has now actually stopped, so post the TERMINAL Canceled here
        // (not at the earlier cancel poll, which only tripped the flag + showed
        // "Cancelling…"). This terminal write is what frees the worker row
        // (`jobs_store::update_job_progress`), so it lands exactly as the worker process
        // returns to its claim loop — the next queued job waits only until the GPU is
        // genuinely free, and the UI shows "Cancelling…" until completion (sc-5515).
        // result=None lets `coalesce` keep any partial images already streamed.
        let message = "Image generation canceled by user.";
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                message,
                None,
                backend,
            ),
        )
        .await?;
        return Err(WorkerError::Canceled(message.to_owned()));
    }
    task_result
}

// ---------------------------------------------------------------------------
// Z-Image strict-pose ControlNet (macOS, sc-3028): the Fun-Controlnet-Union
// `z_image_turbo_control` variant. One image per pose, each driven by a DWPose
// skeleton rendered from the pose's keypoints (see `openpose_skeleton`).
// ---------------------------------------------------------------------------

// Candle image lane labeling + engine-gate unit tests (sc-5099). Windows/candle-gated (the functions
// only exist on that build); pure string maps, no GPU.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod candle_label_tests {
    use super::*;

    #[test]
    fn candle_image_adapter_labels_are_per_family() {
        assert_eq!(candle_adapter_label("z_image_turbo"), "candle_z_image");
        assert_eq!(candle_adapter_label("flux_schnell"), "candle_flux");
        assert_eq!(candle_adapter_label("flux_dev"), "candle_flux");
        assert_eq!(candle_adapter_label("flux2_klein_9b"), "candle_flux2");
        assert_eq!(candle_adapter_label("flux2_dev"), "candle_flux2");
        // sc-7459: the klein weight variants share the FLUX.2 family label.
        assert_eq!(candle_adapter_label("flux2_klein_9b_kv"), "candle_flux2");
        assert_eq!(
            candle_adapter_label("flux2_klein_9b_true_v2"),
            "candle_flux2"
        );
        assert_eq!(candle_adapter_label("qwen_image"), "candle_qwen");
        assert_eq!(candle_adapter_label("chroma1_hd"), "candle_chroma");
        assert_eq!(candle_adapter_label("chroma1_base"), "candle_chroma");
        assert_eq!(candle_adapter_label("chroma1_flash"), "candle_chroma");
        assert_eq!(candle_adapter_label("lens"), "candle_lens");
        assert_eq!(candle_adapter_label("lens_turbo"), "candle_lens");
        assert_eq!(candle_adapter_label("kolors"), "candle_kolors");
        assert_eq!(candle_adapter_label("sensenova_u1_8b"), "candle_sensenova");
        assert_eq!(
            candle_adapter_label("sensenova_u1_8b_fast"),
            "candle_sensenova"
        );
        assert_eq!(candle_adapter_label("ideogram_4"), "candle_ideogram");
        assert_eq!(candle_adapter_label("ideogram_4_turbo"), "candle_ideogram");
        // Boogu (sc-7524): all three variants share the `candle_boogu` asset stamp.
        assert_eq!(candle_adapter_label("boogu_image"), "candle_boogu");
        assert_eq!(candle_adapter_label("boogu_image_turbo"), "candle_boogu");
        assert_eq!(candle_adapter_label("boogu_image_edit"), "candle_boogu");
        // Krea 2 Turbo (sc-7581): the candle asset stamp.
        assert_eq!(candle_adapter_label("krea_2_turbo"), "candle_krea");
        assert_eq!(candle_adapter_label("sdxl"), "candle_sdxl");
        assert_eq!(candle_adapter_label("realvisxl"), "candle_sdxl");
        // Every wired engine carries a `candle_`-prefixed label, distinct from the `mlx_` labels.
        for model in [
            "z_image_turbo",
            "flux_schnell",
            "flux_dev",
            "flux2_klein_9b",
            "flux2_dev",
            "qwen_image",
            "chroma1_hd",
            "chroma1_base",
            "chroma1_flash",
            "lens",
            "lens_turbo",
            "kolors",
            "sensenova_u1_8b",
            "sensenova_u1_8b_fast",
            "ideogram_4",
            "ideogram_4_turbo",
            "boogu_image",
            "boogu_image_turbo",
            "boogu_image_edit",
            // SD3.5 (sc-7880): Large / Large Turbo / Medium carry the `candle_sd3` stamp.
            "sd3_5_large",
            "sd3_5_large_turbo",
            "sd3_5_medium",
            "sdxl",
            "realvisxl",
        ] {
            assert!(candle_adapter_label(model).starts_with("candle_"));
        }
        // SD3.5 (sc-7880): the candle asset stamp.
        assert_eq!(candle_adapter_label("sd3_5_large"), "candle_sd3");
        assert_eq!(candle_adapter_label("sd3_5_large_turbo"), "candle_sd3");
        assert_eq!(candle_adapter_label("sd3_5_medium"), "candle_sd3");
    }

    #[test]
    fn is_candle_engine_covers_only_the_wired_txt2img_families() {
        for model in [
            "sdxl",
            "realvisxl",
            "realvisxl_lightning",
            "z_image_turbo",
            // Base (non-distilled, full-CFG) Z-Image txt2img (sc-8679): the base sibling of z_image_turbo.
            "z_image",
            "flux_schnell",
            "flux_dev",
            "flux2_klein_9b",
            "flux2_dev",
            // sc-7459: the two klein weight variants share the candle FLUX.2 engine.
            "flux2_klein_9b_kv",
            "flux2_klein_9b_true_v2",
            "qwen_image",
            "chroma1_hd",
            "chroma1_base",
            "chroma1_flash",
            "lens",
            "lens_turbo",
            "kolors",
            "sensenova_u1_8b",
            "sensenova_u1_8b_fast",
            "ideogram_4",
            "ideogram_4_turbo",
            // Boogu (sc-7524): Base + Turbo (txt2img) AND `boogu_image_edit` — unlike z_image_edit /
            // qwen_image_edit (bespoke streams), Boogu's instruction edit is in-lane on the generic
            // candle stream (like Ideogram), so `boogu_image_edit` IS a candle engine.
            "boogu_image",
            "boogu_image_turbo",
            "boogu_image_edit",
            // Krea 2 Turbo (sc-7581): pure txt2img on the generic candle lane.
            "krea_2_turbo",
            // SD3.5 (sc-7880): Large / Large Turbo / Medium ride the generic candle txt2img lane.
            "sd3_5_large",
            "sd3_5_large_turbo",
            "sd3_5_medium",
        ] {
            assert!(is_candle_engine(model), "{model} should be a candle engine");
        }
        // Non-candle families + still-unwired variants (bespoke-stream edit ids) are not in the generic
        // lane. (kolors / sensenova ARE candle engines now — sc-5576 — for their base txt2img shape;
        // `boogu_image_edit` IS — sc-7524 — because its edit is in-lane, not bespoke; the klein
        // `_kv` / `_true_v2` weight variants are candle engines too — sc-7459 — for txt2img.)
        for model in [
            "bernini_image",
            "z_image_edit",
            "qwen_image_edit",
            "wan_2_2",
        ] {
            assert!(!is_candle_engine(model), "{model} must not be a candle engine");
        }
    }
}

#[cfg(test)]
mod boogu_tier_tests {
    use super::*;

    #[test]
    fn tier_subdir_selects_by_quant_bits() {
        // Q8 default (no opt-in / a >4 request) → None (the `<variant>/` folder ships in the catalog
        // download). 1..=4 → packed q4; <=0 → dense bf16. Consistent with krea/ideogram (sc-8513).
        assert_eq!(boogu_tier_subdir("base", None), None);
        assert_eq!(boogu_tier_subdir("base", Some(8)), None);
        assert_eq!(boogu_tier_subdir("base", Some(4)), Some("base-q4".to_owned()));
        assert_eq!(boogu_tier_subdir("turbo", Some(2)), Some("turbo-q4".to_owned()));
        assert_eq!(boogu_tier_subdir("edit", Some(0)), Some("edit-bf16".to_owned()));
        assert_eq!(
            boogu_tier_subdir("base", Some(-1)),
            Some("base-bf16".to_owned())
        );
    }
}

#[cfg(test)]
mod standard_tier_tests {
    use super::*;
    use serde_json::json;

    fn request(advanced: serde_json::Value) -> ImageRequest {
        ImageRequest::from_payload(
            json!({ "model": "sd3_5_large", "advanced": advanced })
                .as_object()
                .unwrap(),
        )
    }

    /// Write a minimal present `<tier>/transformer/<file>` so [`standard_tier_subdir`]'s
    /// filename-agnostic probe sees the tier as downloaded.
    fn seed_tier(root: &Path, tier: &str, file: &str) {
        let dir = root.join(tier).join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), b"x").unwrap();
    }

    #[test]
    fn defaults_to_q4_and_honors_quantize_selection() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Packed q4/q8 single-file + dense sharded bf16 (only the index.json shape).
        seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "bf16", "diffusion_pytorch_model.safetensors.index.json");

        // No selection → q4 default.
        assert_eq!(
            standard_tier_subdir(root, &request(json!({}))),
            root.join("q4")
        );
        // mlxQuantize 8 → q8; 0/"none" → bf16; numeric-string accepted.
        assert_eq!(
            standard_tier_subdir(root, &request(json!({ "mlxQuantize": 8 }))),
            root.join("q8")
        );
        assert_eq!(
            standard_tier_subdir(root, &request(json!({ "mlxQuantize": 0 }))),
            root.join("bf16")
        );
        assert_eq!(
            standard_tier_subdir(root, &request(json!({ "mlxQuantize": "8" }))),
            root.join("q8")
        );
    }

    #[test]
    fn falls_back_when_preferred_tier_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Only q4 downloaded: a q8/bf16 request still resolves to the present q4 rather than a
        // half-empty subdir, so a partial turnkey surfaces as a load error, not a silent half-load.
        seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
        assert_eq!(
            standard_tier_subdir(root, &request(json!({ "mlxQuantize": 8 }))),
            root.join("q4")
        );
        // Nothing present → the repo root (engine surfaces the missing-weights error).
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(
            standard_tier_subdir(empty.path(), &request(json!({}))),
            empty.path().to_path_buf()
        );
    }

    /// sc-8746: the SDXL-family turnkeys pack their backbone under `unet/`, not `transformer/`, so
    /// [`standard_tier_subdir`]'s probe must recognize a `unet/` component as a present tier.
    #[test]
    fn resolves_sdxl_unet_backbone_tiers() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let seed_unet = |tier: &str| {
            let dir = root.join(tier).join("unet");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), b"x").unwrap();
        };
        seed_unet("q4");
        seed_unet("q8");
        seed_unet("bf16");
        // Default q4, q8 selection, bf16 opt-out all resolve to the unet-backed tier subdir.
        assert_eq!(
            standard_tier_subdir(root, &request(json!({}))),
            root.join("q4")
        );
        assert_eq!(
            standard_tier_subdir(root, &request(json!({ "mlxQuantize": 8 }))),
            root.join("q8")
        );
        assert_eq!(
            standard_tier_subdir(root, &request(json!({ "mlxQuantize": 0 }))),
            root.join("bf16")
        );
    }

    /// sc-8771: SenseNova-U1 is a unified MoT turnkey — no `transformer/`/`unet/` component, the whole
    /// backbone is a flat `model.safetensors` (q4/q8) or sharded `*.index.json` (bf16) directly in the
    /// tier dir. [`standard_tier_subdir`]'s probe must recognize weights at the tier root itself.
    #[test]
    fn resolves_flat_unified_backbone_tiers() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let seed_flat = |tier: &str, file: &str| {
            let dir = root.join(tier);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(file), b"x").unwrap();
        };
        // Packed q4/q8 single-file, dense sharded bf16 (index.json shape) — the SenseNova layout.
        seed_flat("q4", "model.safetensors");
        seed_flat("q8", "model.safetensors");
        seed_flat("bf16", "model.safetensors.index.json");
        assert_eq!(
            standard_tier_subdir(root, &request(json!({}))),
            root.join("q4")
        );
        assert_eq!(
            standard_tier_subdir(root, &request(json!({ "mlxQuantize": 8 }))),
            root.join("q8")
        );
        assert_eq!(
            standard_tier_subdir(root, &request(json!({ "mlxQuantize": 0 }))),
            root.join("bf16")
        );
    }

    /// sc-8746 on-device verify (MLX): drive the ACTUAL worker seam against a downloaded SceneWorks
    /// realvisxl-mlx turnkey — `standard_tier_subdir` resolves the `q4/` subdir from the tier root,
    /// then `gen_core::load("sdxl", …)` with `Quant::Q4` loads the packed tier and renders. Asserts a
    /// non-degenerate image (per-pixel std above the all-black/NaN floor). `#[ignore]`d — run by hand
    /// on a Mac with the tier downloaded:
    /// ```text
    /// hf download SceneWorks/realvisxl-mlx --include "q4/*" --local-dir /tmp/realvisxl-q4
    /// SDXL_TIER_ROOT=/tmp/realvisxl-q4 cargo test -p sceneworks-worker --lib \
    ///   sdxl_realvisxl_q4_tier_mlx_smoke -- --ignored --nocapture
    /// ```
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight MLX smoke; needs a downloaded SceneWorks/realvisxl-mlx q4 tier (SDXL_TIER_ROOT)"]
    fn sdxl_realvisxl_q4_tier_mlx_smoke() {
        use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

        let root = PathBuf::from(
            std::env::var("SDXL_TIER_ROOT")
                .expect("set SDXL_TIER_ROOT to the downloaded realvisxl-mlx tier root")
                .trim(),
        );
        // The worker resolution: a `realvisxl` request (q4 default, no mlxQuantize) must land on q4/.
        let req = ImageRequest::from_payload(
            json!({ "model": "realvisxl", "advanced": {} })
                .as_object()
                .unwrap(),
        );
        let tier = standard_tier_subdir(&root, &req);
        assert_eq!(tier, root.join("q4"), "worker must resolve the q4 tier subdir");
        assert!(
            tier.join("model_index.json").is_file() && tier.join("unet").is_dir(),
            "q4 tier subdir missing turnkey layout (model_index.json + unet/): {}",
            tier.display()
        );

        // Load the packed q4 tier through the MLX `sdxl` engine (Quant::Q4 = harmless no-op on the
        // already-packed weights) and render a 768x768 image.
        let spec = LoadSpec::new(WeightsSource::Dir(tier.clone())).with_quant(Quant::Q4);
        let generator = gen_core::load("sdxl", &spec).expect("load MLX sdxl provider on q4 tier");
        let gen_req = GenerationRequest {
            prompt: "a photorealistic portrait of a red fox in a snowy forest, golden hour"
                .to_owned(),
            width: 768,
            height: 768,
            count: 1,
            seed: Some(42),
            steps: Some(20),
            guidance: Some(7.0),
            ..Default::default()
        };
        let output = generator
            .generate(&gen_req, &mut |_p| {})
            .expect("sdxl q4 tier generate");
        let image = match output {
            GenerationOutput::Images(mut images) => images.pop().expect("no image returned"),
            other => panic!("expected Images output, got {other:?}"),
        };
        // Cheap degenerate-floor check: an all-black / NaN-clamped decode collapses toward std 0.
        let n = image.pixels.len() as f64;
        assert!(n > 0.0, "empty image buffer");
        let mean = image.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
        let std = (image.pixels.iter().map(|&p| (p as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
        println!("[sc-8746 smoke] realvisxl q4 tier render {}x{} std {std:.2}", image.width, image.height);
        assert!(std > 5.0, "render looks degenerate (std {std:.2}) — possible NaN / all-black decode");
    }

    /// A model built with a manifest entry so `uses_standard_tier_layout` / `is_dense_te_tier`
    /// can be exercised without touching the hardcoded registries.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    fn manifest_request(model: &str, mlx: serde_json::Value) -> ImageRequest {
        ImageRequest::from_payload(
            json!({ "model": model, "modelManifestEntry": { "mlx": mlx } })
                .as_object()
                .unwrap(),
        )
    }

    /// sc-8508: the standard-tier routing is manifest-driven — a model NOT in
    /// [`STANDARD_TIER_MODELS`] opts in via `mlx.standardTierLayout: true`, while a registry model
    /// stays true without the flag and an ordinary model stays false. Back-compat guard.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn standard_tier_layout_is_manifest_driven_with_registry_backcompat() {
        // Registry member, no manifest flag → still true.
        assert!(uses_standard_tier_layout(&manifest_request(
            "flux2_dev",
            json!({})
        )));
        // Novel id (not in the registry) opts in from the manifest alone.
        assert!(uses_standard_tier_layout(&manifest_request(
            "some_new_matrix_model",
            json!({ "standardTierLayout": true })
        )));
        // Novel id without the flag → not a standard-tier model.
        assert!(!uses_standard_tier_layout(&manifest_request(
            "some_dense_model",
            json!({})
        )));
    }

    /// sc-8508: the dense-TE guard is manifest-driven too — registry members
    /// (`DENSE_TE_TIER_MODELS`) stay true and a novel id opts in via `mlx.denseTextEncoderTier`.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn dense_te_tier_is_manifest_driven_with_registry_backcompat() {
        assert!(is_dense_te_tier(&manifest_request("flux2_klein_9b", json!({}))));
        assert!(is_dense_te_tier(&manifest_request(
            "some_dense_te_model",
            json!({ "denseTextEncoderTier": true })
        )));
        assert!(!is_dense_te_tier(&manifest_request("flux2_dev", json!({}))));
    }

    /// sc-9092 (epic 9083 gap #3): the candle Lens lane no longer resolves a SEPARATE bf16 diffusers
    /// rehost (`SceneWorks/Lens{,-Turbo}`, the retired `candle_lens_repo`) — it packed-loads the SAME
    /// `SceneWorks/lens{,-turbo}-mlx` MLX turnkey the macOS path uses, routed through the shared
    /// `standard_tier_subdir` (`lens`/`lens_turbo` opt in via `mlx.standardTierLayout`, exactly like the
    /// MLX lane). This proves the shared tier resolver picks the requested q4/q8/bf16 subdir of a Lens
    /// turnkey snapshot off-Mac — the candle-lane sibling of the SD3.5 `standard_tier_subdir` tests
    /// above — so the retired resolver is fully replaced by the standard machinery.
    #[test]
    fn candle_lens_resolves_packed_turnkey_tier_subdir() {
        let lens_request = |bits: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": "lens", "advanced": { "mlxQuantize": bits } })
                    .as_object()
                    .unwrap(),
            )
        };
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // A Lens turnkey ships packed per-tier subdirs (transformer + gpt-oss-20b MoE TE + FLUX.2 VAE).
        seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "bf16", "diffusion_pytorch_model.safetensors.index.json");

        // A/B tier toggle: default (q4) / mlxQuantize:8 (q8) / mlxQuantize:0 (bf16) each resolve to
        // their tier subdir — the same q4-default recipe the `lens` manifest declares (`mlx.quantize:4`).
        assert_eq!(
            standard_tier_subdir(root, &lens_request(json!(null))),
            root.join("q4")
        );
        assert_eq!(
            standard_tier_subdir(root, &lens_request(json!(8))),
            root.join("q8")
        );
        assert_eq!(
            standard_tier_subdir(root, &lens_request(json!(0))),
            root.join("bf16")
        );
    }

    /// sc-9092 (epic 9083 gap #3, review fix): Ideogram + Boogu were left `macOnly:true` with only
    /// off-Mac diffusers download entries, which the PR deleted — so off-Mac they resolved to the MLX
    /// turnkey (`SceneWorks/{ideogram-4,boogu-image}-mlx`) whose download entries were macOS-only and
    /// thus never fetched (a "candle weights snapshot not found" load error). The fix flips them
    /// `macOnly:false` and extends the turnkey download `platforms` to windows/linux, so both lanes
    /// packed-load the SAME turnkey the macOS path uses (candle-gen sc-9412 / sc-9410). This asserts the
    /// per-family subdir resolvers pick the shipped tier of a turnkey snapshot regardless of platform —
    /// the ideogram/boogu sibling of `candle_lens_resolves_packed_turnkey_tier_subdir` above.
    #[test]
    fn candle_ideogram_boogu_resolve_packed_turnkey_tier_subdir() {
        let model_request = |model: &str, bits: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": model, "advanced": { "mlxQuantize": bits } })
                    .as_object()
                    .unwrap(),
            )
        };

        // Ideogram turnkey (`SceneWorks/ideogram-4-mlx`): q4 default (candle off-Mac tier) + on-demand
        // q8. `ideogram_model_subdir` probes `<tier>/transformer/model.safetensors`.
        {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path();
            seed_tier(root, "q4", "model.safetensors");
            seed_tier(root, "q8", "model.safetensors");
            for model in ["ideogram_4", "ideogram_4_turbo"] {
                // Default → q4 (the shipped off-Mac tier).
                assert_eq!(
                    ideogram_model_subdir(root, &model_request(model, json!(null))),
                    root.join("q4")
                );
                // mlxQuantize:8 → q8 when present.
                assert_eq!(
                    ideogram_model_subdir(root, &model_request(model, json!(8))),
                    root.join("q8")
                );
            }
        }

        // Boogu turnkey (`SceneWorks/boogu-image-mlx`): Q8 `base/`/`turbo/`/`edit/` is the shipped
        // (off-Mac) default. `boogu_model_subdir` probes `<variant>/transformer/diffusion_pytorch_model.safetensors`.
        {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path();
            for variant in ["base", "turbo", "edit"] {
                seed_tier(root, variant, "diffusion_pytorch_model.safetensors");
            }
            for (model, variant) in [
                ("boogu_image", "base"),
                ("boogu_image_turbo", "turbo"),
                ("boogu_image_edit", "edit"),
            ] {
                // Default (no mlxQuantize) → the shipped Q8 `<variant>/` subfolder.
                assert_eq!(
                    boogu_model_subdir(root, &model_request(model, json!(null))),
                    root.join(variant)
                );
            }
        }
    }
}

/// sc-8820: the recorded quant must reflect the tier subdir ACTUALLY resolved, not the one requested,
/// and a fallback must be surfaced (warn! + `quant_tier_downgraded` event) rather than silently
/// downgrading with lying telemetry.
///
/// macOS-only: exercises [`tier_quant_from_resolved_dir`] / [`reconcile_resolved_tier_quant`], which
/// only compile on the MLX `generate_stream` path. The candle lane has no quant-tier layout.
#[cfg(test)]
#[cfg(target_os = "macos")]
mod quant_tier_reconcile_tests {
    use super::*;
    use serde_json::json;

    /// The resolved basename → precision map used to record the tier that ran (not the one requested).
    #[test]
    fn tier_quant_from_resolved_dir_maps_basename_to_precision() {
        let root = std::path::Path::new("/models/sd3_5_large-mlx");
        // Standard `q4`/`q8`/`bf16` tier dirs → their precision.
        assert_eq!(
            tier_quant_from_resolved_dir(&root.join("q4")),
            Some((Some(Quant::Q4), Some(4)))
        );
        assert_eq!(
            tier_quant_from_resolved_dir(&root.join("q8")),
            Some((Some(Quant::Q8), Some(8)))
        );
        assert_eq!(
            tier_quant_from_resolved_dir(&root.join("bf16")),
            Some((None, None))
        );
        // Boogu `<variant>-<tier>` and bare `<variant>` (= the packed Q8 default).
        assert_eq!(
            tier_quant_from_resolved_dir(&root.join("base-q4")),
            Some((Some(Quant::Q4), Some(4)))
        );
        assert_eq!(
            tier_quant_from_resolved_dir(&root.join("turbo-bf16")),
            Some((None, None))
        );
        assert_eq!(
            tier_quant_from_resolved_dir(&root.join("edit")),
            Some((Some(Quant::Q8), Some(8)))
        );
        // A fell-all-the-way-back-to-root (or modelPath) dir is not a recognizable tier → None, so the
        // caller keeps the request-derived quant.
        assert_eq!(tier_quant_from_resolved_dir(root), None);
    }

    /// The end-to-end reconcile is macOS-only (the MLX generate path). When the resolved tier matches
    /// the request it's a pass-through; when it differs it records the tier that ran, and the
    /// dense-TE guard keeps the load quant `None` while still correcting the recorded bits.
    #[test]
    fn reconcile_records_the_resolved_tier_on_fallback() {
        // Requested q8 present as q8 → pass through, records q8.
        assert_eq!(
            reconcile_resolved_tier_quant(
                (Some(Quant::Q8), Some(8)),
                std::path::Path::new("/m/q8"),
                true,
                "sd3_5_large",
                "job1",
                "mlx",
            ),
            (Some(Quant::Q8), Some(8)),
        );
        // Requested bf16 but only q4 downloaded → resolved dir is `q4`; record Q4 (not dense) and
        // rewrite the load quant to Q4 (safe no-op on already-packed weights).
        assert_eq!(
            reconcile_resolved_tier_quant(
                (None, None),
                std::path::Path::new("/m/q4"),
                true,
                "sd3_5_large",
                "job1",
                "mlx",
            ),
            (Some(Quant::Q4), Some(4)),
        );
        // Dense-TE turnkey: same q4 fallback, but the load quant STAYS `None` (never re-quantize the
        // dense bf16 TE) while the recorded bits are still corrected to Q4.
        assert_eq!(
            reconcile_resolved_tier_quant(
                (None, None),
                std::path::Path::new("/m/q4"),
                false,
                "flux2_klein_9b",
                "job1",
                "mlx",
            ),
            (None, Some(4)),
        );
        // Unrecognized resolved dir (fell back to repo root / modelPath) → keep the request quant.
        assert_eq!(
            reconcile_resolved_tier_quant(
                (Some(Quant::Q8), Some(8)),
                std::path::Path::new("/m/root"),
                true,
                "sd3_5_large",
                "job1",
                "mlx",
            ),
            (Some(Quant::Q8), Some(8)),
        );
    }

    /// End-to-end tier resolution + recording: a bf16 request against a turnkey where ONLY `q4/` is
    /// downloaded resolves to `q4/`, and the reconciled recipe records Q4 — the precision that ran —
    /// not the requested dense bf16. Guards the epic 8506 A/B workflow against telemetry that lies.
    #[test]
    fn bf16_request_with_only_q4_present_records_q4() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let dir = root.join("q4").join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), b"x").unwrap();

        let req = ImageRequest::from_payload(
            json!({ "model": "sd3_5_large", "advanced": { "mlxQuantize": 0 } })
                .as_object()
                .unwrap(),
        );
        // The tier resolver falls through to q4.
        let resolved = standard_tier_subdir(root, &req);
        assert_eq!(resolved, root.join("q4"));
        // The request would have recorded dense bf16 — reconcile corrects it to the resolved Q4 tier.
        let requested = resolve_quant(&req);
        assert_eq!(requested, (None, None), "bf16 request derives dense");
        let (quant, bits) =
            reconcile_resolved_tier_quant(requested, &resolved, true, "sd3_5_large", "job1", "mlx");
        assert_eq!((quant, bits), (Some(Quant::Q4), Some(4)));
    }

    /// sc-9362 (F-018 follow-up): the dense-TE transformer tier the request asks for is derived from
    /// `advanced.mlxQuantize` exactly like [`standard_tier_subdir`] — `<=0 → bf16 (None)`, `>4 → q8`,
    /// else the q4 default — regardless of the always-`None` load quant `resolve_quant` returns for
    /// dense-TE. This is what reconcile compares the resolved tier against so a straight job isn't a
    /// spurious downgrade.
    #[test]
    fn dense_te_requested_tier_bits_mirrors_standard_tier_mapping() {
        let req = |mlx_quantize: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": "flux2_klein_9b", "advanced": { "mlxQuantize": mlx_quantize } })
                    .as_object()
                    .unwrap(),
            )
        };
        let default = ImageRequest::from_payload(
            json!({ "model": "flux2_klein_9b" }).as_object().unwrap(),
        );
        // No selection → the q4 default (matches standard_tier_subdir's preferred).
        assert_eq!(dense_te_requested_tier_bits(&default), Some(4));
        assert_eq!(dense_te_requested_tier_bits(&req(json!(0))), None); // bf16 opt-out
        assert_eq!(dense_te_requested_tier_bits(&req(json!(4))), Some(4));
        assert_eq!(dense_te_requested_tier_bits(&req(json!(8))), Some(8));
        assert_eq!(dense_te_requested_tier_bits(&req(json!("8"))), Some(8)); // numeric-string
        assert_eq!(dense_te_requested_tier_bits(&req(json!(-1))), None);
    }

    /// sc-9362 (F-018 follow-up): a straight (no-fallback) dense-TE job — its q4 transformer tier is
    /// downloaded and resolves as requested — records the ACTUAL transformer tier (Q4) while keeping
    /// the load quant `None` (the dense bf16 TE is never re-quantized). Before the fix the request
    /// derived `(None, None)` and — since dense-TE always requests bf16 in `resolve_quant` — the
    /// resolved q4 tier read as a bf16→q4 downgrade; now the requested tier is the q4 the request
    /// actually asked for, so the resolved tier MATCHES and reconcile pass-throughs it with no event.
    #[test]
    fn dense_te_no_fallback_records_transformer_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // The klein q4 transformer tier is present (dense bf16 TE lives alongside in the same tier).
        let dir = root.join("q4").join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), b"x").unwrap();

        let req = ImageRequest::from_payload(
            json!({ "model": "flux2_klein_9b", "advanced": {} })
                .as_object()
                .unwrap(),
        );
        // The tier resolver lands on the requested q4 tier (no fallback).
        let resolved = standard_tier_subdir(root, &req);
        assert_eq!(resolved, root.join("q4"));
        // resolve_quant keeps dense-TE at `(None, None)` (never re-quantize the dense bf16 TE)…
        assert_eq!(resolve_quant(&req), (None, None));
        // …but reconcile against the REQUESTED transformer tier (q4) records the real transformer
        // precision (Q4) with the load quant still `None`, and — since resolved == requested — with no
        // downgrade (the requested/resolved tiers match, so it's a clean pass-through).
        let requested_for_reconcile = (None, dense_te_requested_tier_bits(&req));
        assert_eq!(requested_for_reconcile, (None, Some(4)));
        let (quant, bits) = reconcile_resolved_tier_quant(
            requested_for_reconcile,
            &resolved,
            false, // dense-TE: keep the load quant None
            "flux2_klein_9b",
            "job1",
            "mlx",
        );
        assert_eq!(
            (quant, bits),
            (None, Some(4)),
            "records the actual q4 transformer tier, load quant stays None"
        );
    }

    /// sc-9362: a GENUINE dense-TE fallback still surfaces — the request asks for q8 but only q4 is
    /// downloaded, so the resolver falls through to q4; reconcile records q4 (the tier that ran) while
    /// the load quant stays `None`. This is the case that legitimately warns/emits.
    #[test]
    fn dense_te_genuine_fallback_records_resolved_tier() {
        let req = ImageRequest::from_payload(
            json!({ "model": "flux2_klein_9b", "advanced": { "mlxQuantize": 8 } })
                .as_object()
                .unwrap(),
        );
        // Requested q8 tier bits, but only the q4 tier exists on disk (resolver fell through).
        assert_eq!(dense_te_requested_tier_bits(&req), Some(8));
        let (quant, bits) = reconcile_resolved_tier_quant(
            (None, dense_te_requested_tier_bits(&req)),
            std::path::Path::new("/m/q4"),
            false,
            "flux2_klein_9b",
            "job1",
            "mlx",
        );
        assert_eq!(
            (quant, bits),
            (None, Some(4)),
            "genuine q8→q4 fallback records the resolved q4 tier, load quant stays None"
        );
    }
}
