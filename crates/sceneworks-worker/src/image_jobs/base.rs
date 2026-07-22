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
    KreaControl,
    Flux1DevControl,
    Flux2DevControl,
    Flux2Edit,
    QwenEdit,
    KreaEdit,
    InstantId,
    PulidFlux,
    SdxlAdvanced,
    SensenovaEdit,
    Bernini,
    /// A strict-pose job on a WIRED MLX pose family (one with a `…_control_available` lane, i.e. a
    /// [`WIRED_MLX_POSE_FAMILIES`] id) whose control base/overlay is NOT installed — its
    /// `…_control_available` gate failed, so it reached the fall-through. Reject loudly instead of
    /// falling through to `Mlx` (plain txt2img) and silently dropping the poses (sc-11796 generalized to
    /// every wired family, sc-11814) — the MLX twin of the candle `CandleImageRoute::PoseControlBaseMissing`.
    PoseControlBaseMissing,
    /// A strict-pose job on an MLX model with NO pose-control lane (e.g. a plain `sdxl` pose job with no
    /// reference — SDXL identity-pose ships via InstantID / IP-Adapter) that `mlx_available` would
    /// otherwise render as plain txt2img, dropping the poses. Reject loudly (sc-5968) — the MLX twin of
    /// the candle `CandleImageRoute::PoseReject`.
    PoseReject,
    Mlx,
}

/// Image model ids the MLX router HAS a bespoke strict-pose control lane for — each is claimed by an
/// `… _control_available` arm in [`resolve_image_route`] BEFORE the generic `mlx_available` txt2img arm,
/// but only when its control base/overlay resolves locally. This is the SINGLE source for the
/// fall-through reject: a wired family that reached the fall-through means its control base is absent
/// (its lane's local weight-gate failed) → [`ImageRoute::PoseControlBaseMissing`], never silent txt2img.
/// The MLX twin of [`WIRED_CANDLE_POSE_FAMILIES`] (sc-11171/F-008), and the SAME id set — every candle
/// wired family has a matching MLX control lane:
///   - `z_image_turbo` → `zimage_control_available` (Turbo Fun-Controlnet-Union)
///   - `z_image`       → `zimage_base_control_available` (base full-CFG Fun-Controlnet-Union, sc-8251)
///   - `qwen_image`    → `qwen_control_available` (2512 Fun-Controlnet-Union)
///   - `kolors`        → `kolors_control_available` (Kolors ControlNet; also needs a reference)
///   - `krea_2_turbo`  → `krea_control_available` (trained control-branch overlay, sc-8465)
///   - `flux_dev`      → `flux1_dev_control_available` (Shakker Union-Pro-2.0)
///   - `flux2_dev`     → `flux2_dev_control_available` (Fun-Controlnet-Union)
///
/// Distinct from a non-wired MLX pose family (e.g. `sdxl`), which reaches the sc-5968
/// [`ImageRoute::PoseReject`] instead. (sc-11814.)
#[cfg(target_os = "macos")]
const WIRED_MLX_POSE_FAMILIES: &[&str] = &[
    "z_image_turbo",
    "z_image",
    "qwen_image",
    "kolors",
    "krea_2_turbo",
    "flux_dev",
    "flux2_dev",
];

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
    } else if krea_control_available(request, settings) {
        // Krea 2 Turbo strict pose (advanced.poses on `krea_2_turbo`) → the trained control-branch
        // overlay (sc-8465, epic 8459 S5). Wins over the generic `mlx_available` arm below — `krea_2_turbo`
        // is in MODEL_TABLE, so `mlx_available` would otherwise render it as plain t2i and silently drop
        // the poses. The MLX twin of the candle `CandleImageRoute::KreaControl` resolver arm.
        Some(ImageRoute::KreaControl)
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
    } else if krea_edit_available(request, settings) {
        // Krea 2 Raw Kontext-style edit (mode edit_image + a source) → the `krea_2_edit` engine
        // (epic 10871). Wins over the generic MLX arm below — `krea_2_raw` is in MODEL_TABLE, so
        // `mlx_available` would otherwise render it as plain t2i and silently drop the source.
        Some(ImageRoute::KreaEdit)
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
    } else if request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && (WIRED_MLX_POSE_FAMILIES.contains(&request.model.as_str())
            || mlx_available(request, settings))
    {
        // A strict-pose job that fell past every `…_control_available` lane above (and the edit / identity /
        // bernini lanes) must be REJECTED, not silently rendered as plain txt2img with the poses dropped —
        // the MLX twin of the candle fall-through reject (base.rs, sc-11171/F-008 + sc-5968). Two sub-cases,
        // distinguished by whether the family has a wired MLX pose lane at all:
        //  - WIRED MLX pose family (`WIRED_MLX_POSE_FAMILIES`): its control lane exists but the base/overlay
        //    snapshot is absent (the lane's `…_control_available` weight-gate failed) → `PoseControlBaseMissing`.
        //    Fires regardless of whether the plain base weights resolve, because the control gate can fail while
        //    `mlx_available` succeeds — for `krea_2_turbo` the control base (`resolve_krea_control_base`) diverges
        //    from the txt2img base (the reported sc-11796 silent-drop), and for `kolors` the lane additionally
        //    needs a `referenceAssetId`. Generalizes the sc-11796 krea-only reject to every wired family (sc-11814).
        //  - A non-wired MLX pose family that `mlx_available` would render as plain txt2img (e.g. a plain `sdxl`
        //    pose job with no reference — SDXL identity-pose ships via InstantID / IP-Adapter, claimed above) →
        //    the sc-5968 no-silent-T2I `PoseReject`.
        // Checked BEFORE the generic `mlx_available` arm.
        if WIRED_MLX_POSE_FAMILIES.contains(&request.model.as_str()) {
            Some(ImageRoute::PoseControlBaseMissing)
        } else {
            Some(ImageRoute::PoseReject)
        }
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
            | ImageRoute::KreaControl
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
            // MLX + SDXL-advanced + Bernini paths, the effective count is the requested count. Krea
            // edit (epic 10871) is likewise plain per-image: `count` edits of the one source. The
            // pose reject arms (`PoseControlBaseMissing` / `PoseReject`) error before generation, so
            // their count is inert.
            ImageRoute::PulidFlux
            | ImageRoute::SdxlAdvanced
            | ImageRoute::Bernini
            | ImageRoute::KreaEdit
            | ImageRoute::PoseControlBaseMissing
            | ImageRoute::PoseReject
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
    /// Krea 2 Kontext-style dual-conditioned image-edit — `krea_2_raw` + `edit_image` + a source, with
    /// the required `krea2_identity_edit` LoRA (epic 10871).
    KreaEdit,
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
    /// Krea 2 pose-ControlNet — a trained control-branch overlay on the frozen Turbo base (sc-8464).
    KreaControl,
    /// A strict-pose job on a candle model with NO pose lane → reject loudly, never silent T2I (sc-5968).
    PoseReject,
    /// A strict-pose job on a WIRED candle pose family (one with a `…_control_available` lane) whose
    /// control base snapshot is NOT installed — the lane's local weight-gate failed, so the job reached
    /// the fall-through arm. Reject loudly ("control base snapshot not installed") rather than silently
    /// rendering plain txt2img and dropping the poses (sc-11171, F-008). Distinct from `PoseReject`,
    /// which is a family that has no candle pose lane at all.
    PoseControlBaseMissing,
    /// An in-place ComfyUI Z-Image base model (`external_base_*`) → `generate_candle_zimage_comfyui_stream`
    /// (epic 10451 Phase 2, sc-10668). Not an `is_candle_engine` id — routed off the forwarded row.
    ZimageComfyui,
    /// An in-place ComfyUI Qwen-Image base model (`external_base_*`) → `generate_candle_qwen_comfyui_stream`
    /// (epic 10451 Phase 2b, sc-10670). Not an `is_candle_engine` id — routed off the forwarded row.
    QwenImageComfyui,
    /// An in-place ComfyUI FLUX.2-dev fp8-mixed base model (`external_base_*`) →
    /// `generate_candle_flux2_comfyui_stream` (epic 10451 Phase 2e, sc-10680). Not an `is_candle_engine`
    /// id — routed off the forwarded row.
    Flux2Comfyui,
    /// Bernini still-image companion (`bernini_image`, engine id `bernini`, `frames:1`) → the dedicated
    /// `generate_candle_bernini_image_stream` (sc-10996, epic 6562). NOT an `is_candle_engine` txt2img id
    /// (the engine is `Modality::Video`, reached with `frames:1`), so — like the MLX `ImageRoute::Bernini`
    /// arm — it is routed on the model id BEFORE the generic txt2img arm; both t2i and i2i (`edit_image`)
    /// route here.
    Bernini,
    /// A plain candle txt2img engine id → `generate_candle_stream`.
    CandleTxt2Img,
}

/// Candle-routed image model ids that HAVE a bespoke worker strict-pose control lane — each is claimed
/// by an `else if …_control_available(…)` arm in [`resolve_candle_image_route`] BEFORE the generic
/// txt2img arm, but only when its control base snapshot resolves locally. This is the SINGLE source for
/// (a) the fall-through reject branch below (a wired family reaching the fall-through means its control
/// base is absent → [`CandleImageRoute::PoseControlBaseMissing`], never silent txt2img) and (b) the
/// reject error message that enumerates the wired families — previously hand-duplicated across the
/// `resolve_candle_image_route` `matches!` guard, the handler comment, and the reject error string, which
/// had already drifted (the handler comment omitted `krea_2_turbo`). (sc-11171, F-008.)
///
/// NOTE: deliberately DISTINCT from the router's `model_has_candle_pose_lane` (sceneworks-core), which
/// omits `krea_2_turbo` so the co-resident torch worker declines krea pose jobs and candle reliably wins
/// them — do not conflate the two lists.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const WIRED_CANDLE_POSE_FAMILIES: &[&str] = &[
    "qwen_image",
    "kolors",
    "z_image_turbo",
    "z_image",
    "flux2_dev",
    "flux_dev",
    "krea_2_turbo",
];

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl CandleImageRoute {
    /// The real image total this candle route produces, baked into the plan's `expectedCount` so the
    /// streamed gallery total matches what actually lands (sc-11171, F-009 — the candle sibling of the
    /// macOS `ImageRoute::image_count`). The strict-pose control lanes each render one image per pose
    /// (`pose_entries().len()`), InstantID renders its active angle/pose collection, every other lane
    /// renders the requested `count`.
    fn image_count(self, request: &ImageRequest, settings: &Settings) -> u32 {
        match self {
            CandleImageRoute::QwenControl
            | CandleImageRoute::KolorsControl
            | CandleImageRoute::ZimageControl
            | CandleImageRoute::Flux2Control
            | CandleImageRoute::Flux1Control
            | CandleImageRoute::KreaControl => pose_entries(request).len() as u32,
            CandleImageRoute::InstantId => instantid_image_count(request, settings),
            // Every other lane (plain txt2img, the edit/reference/identity/comfyui/bernini lanes, and the
            // pose-reject arms — which error before generation) produces the requested count.
            _ => request.count,
        }
    }
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
    } else if krea_control_candle_available(request, settings) {
        // Krea 2 pose-ControlNet (sc-8464): `krea_2_turbo` + `advanced.poses` is the bespoke candle
        // `Krea2Control` lane, diverted before the registry txt2img arm (which would render it as plain
        // txt2img and drop the poses). Mirrors `jobs_store::krea_control_candle_eligible`.
        Some(CandleImageRoute::KreaControl)
    } else if krea_edit_candle_available(request, settings) {
        // Krea 2 Kontext-style edit (epic 10871): `krea_2_raw` + `edit_image` + a source is the bespoke
        // candle `KreaEdit` lane (`generate_candle_krea_edit_stream`), NOT the generic t2i stream. Diverted
        // BEFORE the generic `is_candle_engine` t2i arm below (which `krea_2_raw` now matches, sc-9994/epic
        // 9992) so an edit job runs the dual-conditioning Kontext render instead of being flattened to plain
        // txt2img. Mirrors `jobs_store::krea_edit_candle_eligible`.
        Some(CandleImageRoute::KreaEdit)
    } else if zimage_comfyui_available(request, settings) {
        // In-place ComfyUI Z-Image base (sc-10668): an `external_base_*` id, so it matches no
        // `is_candle_engine` arm below — route it here off the forwarded `modelManifestEntry`.
        Some(CandleImageRoute::ZimageComfyui)
    } else if qwen_comfyui_available(request, settings) {
        // In-place ComfyUI Qwen-Image base (sc-10670): sibling of the Z-Image comfyui lane — an
        // `external_base_*` id routed off the forwarded row (family=="qwen-image", usable).
        Some(CandleImageRoute::QwenImageComfyui)
    } else if flux2_comfyui_available(request, settings) {
        // In-place ComfyUI FLUX.2-dev base (sc-10680): sibling of the Qwen-Image comfyui lane — an
        // `external_base_*` id routed off the forwarded row (family=="flux2", usable).
        Some(CandleImageRoute::Flux2Comfyui)
    } else if bernini_image_candle_available(request) {
        // Bernini still-image companion (sc-10996, epic 6562): t2i / i2i on the `bernini_image` id,
        // routed to the same `engine_id:"bernini"` planner+renderer with `frames:1`. Must win over the
        // generic `is_candle_engine` txt2img arm below — `bernini_image` is NOT an `is_candle_engine` id
        // (its engine is `Modality::Video`, reached with `frames:1`), so it would otherwise fall through
        // to `None` and stub. Routed on the model id alone (like the sdxl txt2img arm) — a missing
        // `SceneWorks/bernini` snapshot fails loud at load, never silently stubs (the MLX
        // `ImageRoute::Bernini` weight-gates instead only because it must fall through to `mlx_available`).
        Some(CandleImageRoute::Bernini)
    } else if is_candle_engine(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
    {
        // A strict-pose candle job that reached here was NOT claimed by any `…_control_available` lane
        // above, so it must be REJECTED, not silently rendered as plain txt2img (poses dropped). Two
        // sub-cases, distinguished by whether the family has a wired candle pose lane at all:
        //  - WIRED pose family (`WIRED_CANDLE_POSE_FAMILIES`): the lane exists but its control base
        //    snapshot is absent (the lane's local weight-gate failed) → `PoseControlBaseMissing`
        //    ("control base snapshot not installed"). Previously this family was excluded from the reject
        //    entirely and fell through to `CandleTxt2Img`, silently dropping the poses (sc-11171, F-008).
        //  - No candle pose lane (e.g. sdxl) → the sc-5968 no-silent-T2I `PoseReject`.
        // Checked BEFORE the txt2img arm below.
        if WIRED_CANDLE_POSE_FAMILIES.contains(&request.model.as_str()) {
            Some(CandleImageRoute::PoseControlBaseMissing)
        } else {
            Some(CandleImageRoute::PoseReject)
        }
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

/// Receipt variant selected by this request.  Prefer an explicit quant request; otherwise use the
/// manifest's default selectable download (or its only selectable download).  Co-requisites are
/// intentionally excluded because each repo resolves its own receipt independently.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn requested_receipt_variant(request: &ImageRequest) -> Option<String> {
    if let Some(bits) = request
        .advanced
        .get("mlxQuantize")
        .and_then(|value| value.as_i64().or_else(|| value.as_str()?.trim().parse().ok()))
    {
        return Some(if bits <= 0 {
            "bf16"
        } else if bits > 4 {
            "q8"
        } else {
            "q4"
        }
        .to_owned());
    }
    let selectable = request
        .model_manifest_entry
        .get("downloads")
        .and_then(Value::as_array)?
        .iter()
        .filter(|entry| entry.get("coRequisite").and_then(Value::as_bool) != Some(true))
        .collect::<Vec<_>>();
    selectable
        .iter()
        .copied()
        .find(|entry| entry.get("default").and_then(Value::as_bool) == Some(true))
        .or_else(|| (selectable.len() == 1).then_some(selectable[0]))
        .and_then(|entry| entry.get("variant").and_then(Value::as_str))
        .map(str::to_owned)
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

/// The whole-repo `Efficient-Large-Model/Sana_1600M_1024px_diffusers` HF snapshot the candle SANA
/// lane loads (sc-11780, epic 8485). The `candle-gen-sana` pipeline reads the diffusers-layout tree
/// (`transformer/` + `vae/` + `text_encoder/`) directly, so the off-Mac lane resolves this repo's
/// snapshot root — NOT the MLX-packed `SceneWorks/Sana_1600M_1024px_mlx` turnkey (the `MODEL_TABLE`
/// `default_repo`, which the macOS/MLX path loads) and NOT a `q4/q8/bf16` tier subdir. Matches the
/// manifest's windows/linux whole-repo download entry.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SANA_CANDLE_DIFFUSERS_REPO: &str = "Efficient-Large-Model/Sana_1600M_1024px_diffusers";

/// The whole-repo `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` HF snapshot the candle
/// SANA-Sprint lane loads (sc-11781, epic 8485). The `candle-gen-sana` Sprint pipeline reads the same
/// diffusers-layout tree (`transformer/` Sprint Linear-DiT + guidance embedder + `vae/` + `text_encoder/`)
/// as base SANA, so the off-Mac lane resolves this repo's snapshot root — NOT the MLX-packed
/// `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` turnkey (the `MODEL_TABLE` `default_repo`, which the macOS/MLX
/// path loads) and NOT a `q4/q8/bf16` tier subdir. Matches the manifest's windows/linux whole-repo
/// download entry.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SANA_SPRINT_CANDLE_DIFFUSERS_REPO: &str =
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers";

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
        let dir = resolve_app_managed_model_dir(settings, &path, "Image modelPath")?;
        // Anima (epic 10512) is convert-at-install with a q4/q8/bf16 MATRIX: unlike other convert
        // models (a single flat dir), its injected `modelPath` is the converted ROOT holding `bf16/`,
        // `q8/`, `q4/` tier subdirs (written by `convert_anima_prequant`). Descend into the requested
        // tier — bespoke, like Boogu/Ideogram — so the packed DiT loads at the chosen precision. Every
        // other `modelPath` model resolves to the flat dir unchanged.
        if is_anima_model(&request.model) {
            return Ok(Some(anima_tier_subdir(&dir, request)));
        }
        return Ok(Some(dir));
    }
    let Some(model) = mlx_model(&request.model) else {
        return Ok(None);
    };
    let repo = model_repo(request, &model);
    let receipt_variant = requested_receipt_variant(request);
    let receipt_snapshot = crate::model_jobs::huggingface_receipt_weights_dir(
        &settings.data_dir,
        &repo,
        Some(&request.model),
        receipt_variant.as_deref(),
    );
    let snapshot = receipt_snapshot
        .clone()
        .or_else(|| huggingface_snapshot_dir(&settings.data_dir, &repo));
    // A tier receipt already resolves to the exact self-contained tier directory.  Returning it
    // before the family pickers prevents a second `q4/q8/bf16` descent and, more importantly, keeps
    // all load inputs on the receipt side of the all-receipt-or-all-current boundary.
    if receipt_snapshot
        .as_deref()
        .and_then(tier_key_from_resolved_dir)
        .is_some()
    {
        return Ok(receipt_snapshot);
    }
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
    // Krea 2 Turbo (epic 7565) + Krea 2 Raw (epic 9992) ship a turnkey with self-contained quant subdirs
    // (Turbo: packed `q8/` default + `q4/`; Raw: packed `q8/` default + `q4/` + dense `bf16/`); point the
    // engine at the chosen quant's subdir rather than the repo root. The packed weights auto-detect their
    // quant on load, so the resolved `spec.quantize` is a no-op on them. `krea_model_subdir` also falls
    // back to any downloaded tier when the preferred one is absent — so Raw generates off the `bf16/`
    // training-base tier when only that is present, instead of failing at the repo root (no `tokenizer/`
    // there). Without this branch Raw fell through to `Ok(snapshot)` (the repo root) and load errored
    // with "tokenizer: No such file or directory" (epic 9992 P5/P6 wiring gap — the `krea_2_raw` engine
    // row already documents this resolver, but the branch was never added).
    if request.model == "krea_2_turbo" || request.model == "krea_2_raw" {
        return Ok(snapshot.map(|root| krea_model_subdir(&root, request)));
    }
    // Anima off-Mac (candle, sc-10676): dense bf16, NOT convert-at-install. There is no converted tier
    // artifact off-Mac (the `anima_quant` converter is macOS-only), so point at the raw
    // `circlestone-labs/Anima` `split_files/` root the candle loader reads directly (its DiT +
    // `text_encoders/qwen_3_06b_base` + `vae/qwen_image_vae`, the exact dir the GPU-validated anima smoke
    // used) and SKIP `anima_tier_subdir` — there are no bf16/q8/q4 tier subdirs off-Mac. The candle
    // loader's `resolve_split_files` also accepts the snapshot parent, so fall back to the snapshot root
    // if `split_files/` is somehow absent (a partial download then surfaces a loud load error, not a
    // silently-wrong dir). macOS never reaches here: a converted Anima install injects `modelPath` and
    // returns early via the `is_anima_model` tier-descent branch at the top of this fn.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if is_anima_model(&request.model) {
        return Ok(snapshot.map(anima_dense_split_files_dir));
    }
    // SANA 1600M off-Mac (candle, sc-11780, epic 8485): the `candle-gen-sana` pipeline
    // (`from_diffusers_snapshot`) loads the WHOLE `Efficient-Large-Model/Sana_1600M_1024px_diffusers`
    // HF snapshot (diffusers layout: `transformer/` + `vae/` + `text_encoder/`) — NOT the MLX-packed
    // `SceneWorks/Sana_1600M_1024px_mlx` turnkey the macOS/MLX path loads (which has no diffusers tree
    // the candle pipeline can read) and NOT a `q4/q8/bf16` tier subdir. So resolve the diffusers repo's
    // snapshot ROOT directly, BYPASSING the `STANDARD_TIER_MODELS` `standard_tier_subdir` descent below
    // (`sana_1600m` is registered there for the MLX turnkey, which would otherwise append a nonexistent
    // `q4/` to the diffusers root). The whole-repo download (manifest windows/linux entry, empty files
    // list) provisions this snapshot; an unfetched repo surfaces as a loud "snapshot not found" load
    // error above. macOS never compiles this branch (it keeps the MLX turnkey path).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if request.model == "sana_1600m" {
        return Ok(huggingface_snapshot_dir(
            &settings.data_dir,
            SANA_CANDLE_DIFFUSERS_REPO,
        ));
    }
    // SANA-Sprint 1.6B off-Mac (candle, sc-11781, epic 8485): identical treatment to base SANA above —
    // the `candle-gen-sana` Sprint pipeline loads the WHOLE `Efficient-Large-Model/
    // Sana_Sprint_1.6B_1024px_diffusers` HF snapshot (same diffusers layout: `transformer/` + `vae/` +
    // `text_encoder/`), NOT the MLX-packed turnkey and NOT a `q4/q8/bf16` tier subdir. Resolve the
    // diffusers repo's snapshot ROOT directly, BYPASSING the `STANDARD_TIER_MODELS` descent below
    // (`sana_sprint_1600m` is registered there for the MLX turnkey, which would otherwise append a
    // nonexistent `q4/`). macOS never compiles this branch (it keeps the MLX turnkey path).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if request.model == "sana_sprint_1600m" {
        return Ok(huggingface_snapshot_dir(
            &settings.data_dir,
            SANA_SPRINT_CANDLE_DIFFUSERS_REPO,
        ));
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

/// Quality rank of a generation quant tier: higher = more faithful (`bf16` = 3, `q8` = 2, `q4` = 1,
/// anything else = 0). Used to CLAMP a resolver's DEFAULT tier UP to a model's per-model quality floor
/// (`mlx.minQualityTier`, sc-10731) — the clamp only ever RAISES, never lowers.
fn tier_quality_rank(tier: &str) -> u8 {
    match tier {
        "bf16" => 3,
        "q8" => 2,
        "q4" => 1,
        _ => 0,
    }
}

/// Normalize a floor tier string to its canonical `&'static str` name (`bf16`/`q8`/`q4`), so a floor
/// borrowed from the request manifest can be returned as a static tier subdir name. An unknown value
/// falls to `q4` (harmless — it never outranks the `q8` default, so it is never actually selected).
fn tier_static_name(tier: &str) -> &'static str {
    match tier {
        "bf16" => "bf16",
        "q8" => "q8",
        _ => "q4",
    }
}

/// The tier subdir name of the NVFP4 tier (sc-11042, epic 11037), and the value of the
/// `advanced.quantTier` label that selects it.
///
/// A DISTINCT, user-selectable tier — **not** an int4-affine equivalent and never an auto-swap of `q4`
/// (epic 11037 SC#5 / the sc-11042 Option A decision). NVFP4 is E2M1 4-bit elements over 16-element
/// blocks with FP8-E4M3 micro-scales + an FP32 per-tensor scale (~4.5 effective bits/weight), a
/// different numeric regime from `q4`; auto-selecting it for a `q4` pick on Blackwell would silently
/// change that tier's output.
pub(crate) const NVFP4_TIER: &str = "nvfp4";

/// The `candle.vramGbByTier` key of the INT8-ConvRot tier (sc-9300), and the tier IDENTITY the VRAM
/// gate sizes a ConvRot render against (sc-12425).
///
/// Like [`NVFP4_TIER`], this is a tier identity with **no honest `mlxQuantize` integer** — the
/// online-rotation int8 DiT is not a point on the bits ladder, so the picker sends
/// `advanced.convRot: true` instead ([`wants_krea_convrot`]). NVFP4's doc calls that "exactly the
/// sc-9300 `convRot` precedent"; sc-12425 is that precedent finally reaching the gate.
///
/// **Why this const has to exist (sc-12425).** `vram_gate::requested_tier_key` is bits-derived and
/// returns only `nvfp4`/`bf16`/`q8`/`q4`, so a ConvRot request — carrying no `mlxQuantize` — fell to
/// its `None => "q8"` arm and was sized against `vramGbByTier["q8"]`. That is the identical aliasing
/// sc-11042 fixed for NVFP4, **but with the sign flipped**: q8 OVER-predicts NVFP4 (a spurious
/// `TooBig`/`Offload`, never an OOM), and UNDER-predicts INT8-ConvRot. Measured on a real trunk
/// (sc-12381, sm_120, 1024²/8-step): the tier peaks at **42.9 GB** while the q8 row predicts
/// 35.9 + 2.0 headroom = 37.9 GB — permissive by 5.0 GB, i.e. it admits a load that OOMs.
/// `vram_gate::predicted_peak_gb`'s own doc names the hazard: "an under-prediction admits a load that
/// can OOM".
///
/// The `vramGbByTier["int8-convrot"]` row existed since sc-9300 but **nothing ever read it** — which is
/// why its unmeasured 31.0 estimate survived without a symptom: a dead row cannot be wrong out loud.
///
/// Candle-lane only — UNLIKE [`NVFP4_TIER`], which is un-gated because macOS-compiled fns
/// (`nvfp4_selected`, `preferred_tier`, …) use it. This const's ONLY users are candle-only
/// (`gate_tier_key`, `vram_gate`), so on the macOS/MLX build it would be dead code (clippy `-D warnings`
/// → error). ConvRot is a candle-only tier (sm_89, sc-9300), so nothing on the MLX path references it.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const INT8_CONVROT_TIER: &str = "int8-convrot";

/// Whether the request EXPLICITLY asked for the NVFP4 tier, via the `advanced.quantTier: "nvfp4"`
/// label (sc-12006 established the label as asset telemetry; sc-11042 makes it a selection input).
///
/// NVFP4 rides `quantTier` and NOT `advanced.mlxQuantize` because `mlxQuantize` is a BITS-VALUED knob:
/// every consumer parses it as an integer that NAMES A TIER (`quant_int` → `<= 0` ⇒ bf16, `<= 4` ⇒ q4,
/// else q8), so **no integer is honest for NVFP4** — `4` would select the int4-affine `q4` tier, and
/// every other value names bf16/q8. So `mlxQuantize` stays `null` for NVFP4 and the tier identity rides
/// this distinct label, exactly the sc-9300 `convRot` precedent.
///
/// PURE + UNGATED (like [`preferred_tier`], its caller): reads only the request, so the "did the user
/// ask for NVFP4" question is testable on every platform without a GPU. The sm_120 host gate is the
/// separate [`nvfp4_host_eligible`], and the on-disk gate is [`tier_dir_is_nvfp4`]; the tier is
/// SELECTED only when all three hold — see [`nvfp4_selected`].
fn nvfp4_requested(request: &ImageRequest) -> bool {
    request
        .advanced
        .get("quantTier")
        .and_then(Value::as_str)
        .is_some_and(|tier| tier.trim().eq_ignore_ascii_case(NVFP4_TIER))
}

/// Whether the RESOLVED tier dir is the distinct `nvfp4/` tier — i.e. whether the NVFP4 tier is
/// actually what [`standard_tier_subdir`] landed on (sc-11042).
///
/// The DISK half of the NVFP4 gate. `standard_tier_subdir` only returns the `nvfp4/` dir when that dir
/// exists with weights in it; a request for a tier that isn't converted yet (sc-11043 owns the
/// convert-at-install loop — **no shipping model packs an `nvfp4/` dir today**) rejoins the clean
/// q8 → bf16 → q4 fallback chain. So "the resolved basename is `nvfp4`" is exactly "the NVFP4 tier is
/// installed AND was chosen", read off the same value the loader will read.
///
/// `None` (tier dir unknown — a lane that resolves no standard-tier subdir: a flat diffusers snapshot,
/// a `modelPath` override, or a caller with no dir in scope) ⇒ **false**. A tier we cannot verify is a
/// tier we must not claim: the conservative direction is to fall through to the request-derived
/// q4/q8/bf16 (which is what such a lane actually loads), never to stamp NVFP4 on it.
///
/// Gated to the lanes that HAVE a quant resolver ([`resolve_quant`]'s cfg): the neither-MLX-nor-candle
/// build resolves no load quant at all, so this would be dead code there (`-D warnings`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn tier_dir_is_nvfp4(tier_dir: Option<&Path>) -> bool {
    tier_dir.and_then(Path::file_name).and_then(|name| name.to_str()) == Some(NVFP4_TIER)
}

/// Whether this job actually SELECTS the distinct NVFP4 tier (sc-11042, epic 11037 SC#5) — the single
/// predicate behind the tier's load quant ([`resolve_quant`]), its recorded label
/// ([`effective_quant_label`]), and its VRAM sizing (`vram_gate::requested_tier_key`), so those three
/// can never disagree about what ran.
///
/// **All THREE halves are required:**
/// 1. [`nvfp4_requested`] — the user EXPLICITLY named the tier. The SC#5 opt-in: never inferred from
///    `bits`, a manifest default, or hardware detection. Being on Blackwell alone selects nothing.
/// 2. [`nvfp4_host_eligible`] — this host can serve FP4 (sm_120 on the candle lane).
/// 3. [`tier_dir_is_nvfp4`] — the `nvfp4/` tier is INSTALLED and is the dir that resolved.
///
/// Half 3 is why this takes the RESOLVED dir rather than trusting the request: halves 1+2 alone say
/// only what the user WANTED on hardware that COULD serve it, and `standard_tier_subdir` independently
/// falls back to q8 when the `nvfp4/` dir is absent — the shipping case on every model today. Deriving
/// the label from 1+2 therefore stamped `"nvfp4"` on a render that actually ran the **q8** weights: a
/// creative choice falsified in the asset record, precisely the SC#5 aliasing this tier exists to
/// avoid. The label must describe what RAN, so it is read off the same dir the loader loads.
///
/// Gated to [`resolve_quant`]'s cfg, like [`tier_dir_is_nvfp4`]: every caller (the quant resolver, the
/// label, and the candle fit gate) lives on the MLX or candle lane, so compiling it on the
/// neither-backend build would be dead code under `-D warnings`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn nvfp4_selected(request: &ImageRequest, nvfp4_host: bool, tier_dir: Option<&Path>) -> bool {
    nvfp4_requested(request) && nvfp4_host && tier_dir_is_nvfp4(tier_dir)
}

/// Whether THIS HOST can serve the NVFP4 tier: the candle lane on a GPU clearing the sm_120
/// consumer-Blackwell compute-cap floor (sc-11042).
///
/// The RUNTIME half of the Blackwell gate, and deliberately defence-in-depth. The web picker already
/// hides the tier unless a live worker advertises the `nvfp4` capability (`gpu.rs`), but `advanced` is
/// free-form pass-through (`rawAdapterSettings` has no strict deserializer), so a hand-crafted API call
/// can put `quantTier: "nvfp4"` on a request to ANY worker. Checking here means such a request falls
/// back cleanly to an installed tier instead of routing an FP4 load at hardware with no FP4 tensor
/// cores. Mirrors ConvRot's belt-and-braces (`int8_convrot` capability + engine-side `ensure_int8_floor`).
///
/// Always `false` off the candle lane: macOS/MLX (Metal has no FP4 hardware — explicitly out of scope
/// for epic 11037, and the runtime's MLX side `reject_quant`s NVFP4) and the non-candle build (no FP4
/// compute) can never serve it, so the tier is never selected there.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn nvfp4_host_eligible() -> bool {
    crate::gpu::compute_cap_meets_nvfp4(crate::gpu::cached_compute_cap())
}

#[cfg(not(all(not(target_os = "macos"), feature = "backend-candle")))]
fn nvfp4_host_eligible() -> bool {
    false
}

/// The tier subdir name a request prefers, given its explicit `advanced.mlxQuantize` `bits`, the
/// model's per-model quality `floor` (`mlx.minQualityTier`, sc-10731 — `None` = no floor), and whether
/// the request is a CANDIDATE for the distinct NVFP4 tier (`nvfp4`, sc-11042 — an explicit
/// [`nvfp4_requested`] label AND an [`nvfp4_host_eligible`] host).
///
/// `nvfp4` here is deliberately the TWO-part gate, NOT the fully-resolved [`nvfp4_selected`]: this
/// function is what CHOOSES the tier dir, so it necessarily runs before any dir exists to probe (asking
/// for `nvfp4_selected` would be circular). The third half of the gate — the `nvfp4/` tier dir actually
/// being installed — is the caller's own `present()` fallback chain, which is exactly what
/// [`nvfp4_selected`] reads back afterwards to confirm what this resolver landed on. So a host-eligible
/// NVFP4 pick with no converted tier on disk falls through this function's chain to an installed tier,
/// and `nvfp4_selected` then reports `false` for it — the two agree by construction.
///
/// An explicit, host-eligible **NVFP4** pick (`nvfp4 == true`) resolves the distinct [`NVFP4_TIER`] and
/// short-circuits the bits map below — NVFP4 has no honest `mlxQuantize` integer, so it cannot be
/// expressed there (epic 11037 SC#5 / sc-11042 Option A). It is NOT floor-clamped: like every explicit
/// pick it is a deliberate creative choice, honored as asked. `nvfp4 == false` — every request that did
/// not explicitly ask for NVFP4, which is all of them today — leaves this function's behavior **exactly**
/// as it was: the mapping below is untouched, so no existing tier's resolution changes (guarded by
/// `preferred_tier_bits_map_is_unchanged_by_the_nvfp4_tier`).
///
/// An EXPLICIT bits pick maps directly (`<= 0` → bf16, `> 4` → q8, `1..=4` → q4) and is HONORED even below
/// the floor — a quant tier is a deliberate quality/creative choice, so the worker never silently
/// overrides it (the web surfaces a non-blocking advisory on a below-floor pick instead). With NO
/// explicit pick, the app-wide default (Q8, epic 10721 / sc-10726) is CLAMPED UP to the floor: a floored
/// model (Anima base/aesthetic = q8) never lets the plain default land below the floor. The floor only
/// ever RAISES the default — a floor at or below Q8 leaves the Q8 default unchanged — and the caller's
/// clean-tier fallback chain still caps the result at what's installed (a floor tier not on disk falls
/// to the best installed tier). This is the SHARED default-tier logic behind both [`standard_tier_subdir`]
/// and [`anima_tier_subdir`]; it REPLACES the sc-10714 anima-specific `None => "q8"` hardcode so Anima's
/// q8 default is now floor-driven from the manifest, not a resolver special-case.
fn preferred_tier(bits: Option<i64>, floor: Option<&str>, nvfp4: bool) -> &'static str {
    // The distinct NVFP4 tier (sc-11042). Checked FIRST and returned as-is: it is not a point on the
    // bf16/q8/q4 fidelity ladder, so it takes no part in the floor clamp or the bits map below.
    if nvfp4 {
        return NVFP4_TIER;
    }
    match bits {
        Some(b) if b <= 0 => "bf16",
        Some(b) if b > 4 => "q8",
        Some(_) => "q4",
        None => match floor {
            Some(f) if tier_quality_rank(f) > tier_quality_rank("q8") => tier_static_name(f),
            _ => "q8",
        },
    }
}

/// The model's per-model quality FLOOR (`mlx.minQualityTier`, sc-10731): the MINIMUM-fidelity tier a
/// DEFAULT resolution may land on, read from the request's forwarded manifest entry. `None` (field
/// absent) means no floor — the app-wide default stands (e.g. Anima turbo is q4-tolerant and declares
/// none). Only `bf16`/`q8`/`q4` are honored; any other value is ignored. Ungated (like
/// [`standard_tier_subdir`], its ungated caller) so every build config that compiles the resolver also
/// compiles this.
fn min_quality_floor(request: &ImageRequest) -> Option<&str> {
    request
        .model_manifest_entry
        .get("mlx")
        .and_then(|mlx| mlx.get("minQualityTier"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|tier| tier_quality_rank(tier) > 0)
}

/// The component subdirs a diffusers-style tier snapshot declares in its own `model_index.json`.
///
/// `None` when the tier ships no readable `model_index.json` — a flat unified turnkey (SenseNova-U1
/// MoT, sc-8771) roots its weights + `tokenizer.json` directly in the tier dir and has no component
/// tree to verify, so callers treat `None` as "nothing to check" and keep the backbone-only probe.
///
/// KNOWN LIMIT (sc-12279): that also means a tier torn badly enough to lack `model_index.json`
/// ITSELF is unverifiable and still resolves on its backbone alone. Deliberate — absence of the index
/// is not evidence the tier is torn, and inferring "component-shaped but no index ⇒ torn" would let a
/// perfectly good tier lose the chain to a sibling that merely ships an index, which is a worse bug
/// than the one this fixes. We only ever demote a tier on POSITIVE evidence: its own index naming a
/// component that is not on disk.
fn tier_declared_components(dir: &Path) -> Option<Vec<String>> {
    let raw = std::fs::read_to_string(dir.join("model_index.json")).ok()?;
    let index: Value = serde_json::from_str(&raw).ok()?;
    Some(
        index
            .as_object()?
            .iter()
            .filter(|(key, value)| !key.starts_with('_') && is_component_entry(value))
            .map(|(key, _)| key.clone())
            .collect(),
    )
}

/// Whether a `model_index.json` value names a COMPONENT — the diffusers `[library, class]` pair.
///
/// Three things in a real index are NOT components, and all three must be rejected here (each is
/// live in a shipping turnkey, so a laxer test fails a good tier):
/// - `[null, null]` marks an ABSENT optional component — `SceneWorks/realvisxl-mlx` declares
///   `feature_extractor` and `image_encoder` this way and ships neither dir.
/// - config arrays — `SceneWorks/krea-2-raw-mlx` declares `text_encoder_select_layers: [2, 5, 8, …]`.
/// - config scalars — krea's `patch_size: 2`, realvisxl's `force_zeros_for_empty_prompt: true`.
fn is_component_entry(value: &Value) -> bool {
    value
        .as_array()
        .is_some_and(|pair| pair.len() == 2 && pair.iter().all(Value::is_string))
}

/// Whether every component `dir`'s own `model_index.json` declares is actually on disk (sc-12279).
///
/// Presence is "the subdir holds at least one non-hidden entry", NOT "holds weights": `tokenizer/`
/// and `scheduler/` are config-only. Hidden entries don't count for the same reason the backbone
/// probes skip them — a dir holding only an AppleDouble `._tokenizer.json` sidecar has no tokenizer
/// (SceneWorks#1333).
fn tier_components_present(dir: &Path) -> bool {
    let Some(components) = tier_declared_components(dir) else {
        return true;
    };
    components
        .iter()
        .all(|component| dir_has_visible_entry(&dir.join(component)))
}

/// Whether `dir` is a directory holding at least one non-hidden entry.
fn dir_has_visible_entry(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| !sceneworks_core::lora_family::is_hidden_file(&entry.path()))
    })
}

// The per-family tier-completeness predicates for the no-`model_index` MLX turnkeys
// (`anima_tier_complete` / `boogu_tier_complete` / `sana_tier_complete`, plus their
// `dir_has_visible_file_ending` helper) live in `sceneworks_core::mlx_tier_completeness` and are
// called fully-qualified below. They are shared with rust-api's catalog completeness so the worker's
// tier resolvers and the /models `installed`-vs-`incomplete` report cannot drift apart (sc-13513).

/// Whether the tier dir `resolve_weights_dir` resolved for `request` is COMPLETE — every component its
/// loader reads is on disk. Used ONLY by [`mlx_weights_gap`] to turn a torn-tier load (which otherwise
/// dies mid-generation with a raw "No such file or directory") into an actionable PRE-FLIGHT message.
/// After the sc-12279-generalized resolvers, `resolve_weights_dir` already falls back to a complete
/// sibling tier whenever one exists, so this only ever returns `false` when NO complete tier is
/// installed for the model.
///
/// Dispatches to the SAME family completeness predicate the resolvers use. An unrecognized family falls
/// back to [`tier_components_present`] (the `model_index.json` guard), which is `true` for a layout
/// without one — so a dispatch miss degrades to "no extra message" (today's raw error at load), NEVER a
/// false "incomplete" that would block a loadable tier.
#[cfg(target_os = "macos")]
fn resolved_tier_is_complete(request: &ImageRequest, dir: &Path) -> bool {
    if is_anima_model(&request.model) {
        return sceneworks_core::mlx_tier_completeness::anima_tier_complete(dir);
    }
    if matches!(
        request.model.as_str(),
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit"
    ) {
        return sceneworks_core::mlx_tier_completeness::boogu_tier_complete(dir);
    }
    if matches!(request.model.as_str(), "sana_1600m" | "sana_sprint_1600m") {
        return tier_components_present(dir)
            && sceneworks_core::mlx_tier_completeness::sana_tier_complete(dir);
    }
    tier_components_present(dir)
}

/// Walk `chain` (a tier-name preference order) and return the first tier that is safe to load: one
/// that is COMPLETE (`complete` returns true) if any qualifies, else the first that merely clears
/// `present`'s backbone probe. `None` when no tier is installed at all.
///
/// The shared tail of every tier resolver (sc-12279). `present` alone accepts a tier on its backbone,
/// so a torn tier — the transformer landed, `tokenizer/` did not — short-circuited the chain and the
/// loader died on `tokenizer: No such file or directory` even when a complete sibling tier was
/// installed (issue #850's symptom). Running the chain twice, rather than folding completeness into
/// `present`, is deliberate: if `complete` ever misjudges a tier shape we haven't seen, pass 2 lands
/// exactly where the pre-sc-12279 code did, so this can never strand a model that loads today.
/// Duplicates in `chain` (the preferred tier is usually also a fallback) are harmless.
///
/// `complete` is family-specific: the diffusers-turnkey families pass [`tier_components_present`]
/// (reads the tier's own `model_index.json`); families with a bespoke on-disk layout and no
/// `model_index.json` (Anima's `diffusion_models/ + text_encoders/ + vae/`, Boogu's packed subfolders,
/// InstantID's `unet/`) pass their own predicate so the completeness half is not a silent no-op for
/// them (the pre-generalization gap that let those families still short-circuit onto a torn tier).
fn pick_loadable_tier(
    chain: &[&str],
    present: &dyn Fn(&str) -> Option<PathBuf>,
    complete: &dyn Fn(&Path) -> bool,
) -> Option<PathBuf> {
    let first = |probe: &dyn Fn(&str) -> Option<PathBuf>| chain.iter().find_map(|name| probe(name));
    if let Some(dir) = first(&|name: &str| present(name).filter(|dir| complete(dir))) {
        return Some(dir);
    }
    // No complete tier: load the torn one anyway (pre-sc-12279 behavior) but say so — the loader
    // error it usually raises names a missing file, never the tier that lacks it.
    let torn = first(present)?;
    tracing::warn!(
        tier = %torn.display(),
        "Tier is missing components it should ship, and no complete tier is installed. Loading it \
         anyway; re-download the tier if load fails."
    );
    Some(torn)
}

/// Pick the engine-complete tier subdir of a standard SceneWorks quant-matrix turnkey `root`:
/// `bf16/` when the request opts out of quantization (`advanced.mlxQuantize <= 0` / "none"), `q8/`
/// when it opts into Q8 (`> 4`), `q4/` for an explicit Q4 pick (`1..=4`), else — with NO explicit
/// `mlxQuantize` — the **`q8/`** default (epic 10721 / sc-10726: the app-wide gen-time default tier,
/// matching [`resolve_quant`]'s Q8 default and [`anima_tier_subdir`], replacing the old blind q4).
/// Falls back through the clean tiers first (q8 → bf16 → q4 → `root`) so a partial install never
/// silently lands on the low-fidelity q4 (and a fully-absent turnkey surfaces as a load error). The
/// Q8 default is CLAMPED to what's installed: with only `q4/` on disk it resolves q4, so a heavy model
/// never OOMs on a tier the user didn't download.
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
    standard_tier_subdir_gated(root, request, nvfp4_host_eligible())
}

/// [`standard_tier_subdir`] with the NVFP4 **host** gate passed in rather than probed (sc-11042).
///
/// Split out ONLY for testability, and the injected value is deliberately the HARDWARE fact
/// (`nvfp4_host` = "this host clears the sm_120 floor"), not the finished decision: the live
/// compute-cap probe is the one thing a test can't control, while reading the request's `quantTier`
/// label stays real. So a test can drive both host classes and still exercise the actual
/// request-parsing + SC#5 opt-in. Production has exactly one caller ([`standard_tier_subdir`]), which
/// passes the real probe.
fn standard_tier_subdir_gated(root: &Path, request: &ImageRequest, nvfp4_host: bool) -> PathBuf {
    let bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    // A component dir "has weights" when it holds a packed/dense safetensors or a shard index.
    // Hidden entries don't count: a dir holding only a `._model.safetensors` AppleDouble sidecar has
    // no weights, and reporting otherwise routes the loader at a tier it cannot load
    // (SceneWorks#1333).
    let component_has_weights = |dir: &Path| -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
        entries.flatten().any(|entry| {
            if sceneworks_core::lora_family::is_hidden_file(&entry.path()) {
                return false;
            }
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
    // No explicit selection → the app-wide q8 default (epic 10721 / sc-10726), CLAMPED UP to the model's
    // per-model quality floor (`mlx.minQualityTier`, sc-10731 — raises only, never lowers); bits<=0
    // (advanced.mlxQuantize: 0 / "none") → bf16; bits>4 → q8; else (an explicit 1..=4) the q4 the user
    // asked for (an explicit below-floor pick is honored — the web flags it, the worker never overrides).
    // Fallback prefers the clean tiers (q8 → bf16 → q4) so a partial install never silently lands on the
    // washed q4, and the (possibly floored) default is clamped to what's on disk.
    //
    // An explicit, host-eligible NVFP4 pick (sc-11042) prefers the distinct `nvfp4/` tier dir and then
    // rejoins the SAME clean-tier fallback chain, so a request for a tier that isn't converted yet
    // (sc-11043 owns the convert-at-install loop; no shipping model packs an `nvfp4/` dir today) lands
    // on an installed tier exactly like an uninstalled q4/bf16 pick does — never a load error, and never
    // an FP4 load on hardware that can't serve it (`nvfp4` is already false off Blackwell).
    //
    // That NVFP4 fallback is currently NOT event-surfaced, and deliberately so — say what is true here:
    // `reconcile_resolved_tier_quant` (which `warn!`s + fires `quant_tier_downgraded` on a tier
    // downgrade) is `#[cfg(target_os = "macos")]`, while `nvfp4_host_eligible()` is hard-`false` on
    // macOS — so the reconcile path and an NVFP4 pick are MUTUALLY EXCLUSIVE by construction and nothing
    // reconciles NVFP4 on any lane. What keeps the fallback HONEST instead is [`nvfp4_selected`]: the
    // recorded label + the load quant are gated on this resolver's OWN output (the resolved dir), so a
    // pick that lands on q8 is recorded `"q8"` — the asset record tells the truth even with no event.
    // Wiring a candle-lane reconcile so the downgrade is also OBSERVABLE (an event, not just an honest
    // record) is worth doing when a tier is actually converted; sc-11043 owns that tier.
    //
    // BOTH halves are required HERE (the SC#5 opt-in): the user explicitly named the tier AND the host
    // can serve it. Neither alone selects NVFP4. The third half — the tier is actually installed — is
    // this function's own `present()` chain below, which is what [`nvfp4_selected`] reads back.
    let preferred = preferred_tier(
        bits,
        min_quality_floor(request),
        nvfp4_requested(request) && nvfp4_host,
    );
    // sc-12279: prefer a tier whose component tree is fully on disk, so a torn tier falls through to a
    // complete sibling instead of reaching the loader. Diffusers-shaped turnkeys (flux/qwen/sd3.5/…)
    // ship a per-tier `model_index.json` that `tier_components_present` reads. SANA is the exception:
    // its `SceneWorks/Sana_*_mlx` turnkeys ship NO `model_index.json`, so that guard is a no-op for it —
    // fold in the concrete `sana_tier_complete` check (transformer + VAE + Gemma TE + tokenizer) so a
    // torn SANA tier is demoted too. Flat unified tiers (SenseNova-U1 MoT) ship no `model_index.json`
    // either but are not SANA, so they resolve on the backbone probe exactly as before.
    let is_sana = matches!(request.model.as_str(), "sana_1600m" | "sana_sprint_1600m");
    let complete = |dir: &Path| {
        tier_components_present(dir)
            && (!is_sana || sceneworks_core::mlx_tier_completeness::sana_tier_complete(dir))
    };
    pick_loadable_tier(&[preferred, "q8", "bf16", "q4"], &present, &complete)
        .unwrap_or_else(|| root.to_path_buf())
}

/// The DENSE (`bf16/`) tier of a SceneWorks quant-matrix turnkey, or `root` unchanged for a flat
/// diffusers snapshot (sc-10614).
///
/// The FALLBACK tier resolver for the candle SDXL edit / IP-Adapter lanes (`sdxl_edit_candle.rs`,
/// `sdxl_ipadapter.rs`) on a NON-standard-layout repo. As of sc-10813 those lanes packed-detect and,
/// for a standard-tier turnkey (`mlx.standardTierLayout`), descend through [`standard_tier_subdir`]
/// (honouring `advanced.mlxQuantize`, exactly like the txt2img lane, sc-10767) — so the packed q4/q8
/// tiers now serve edit / inpaint / IP-Adapter, not just txt2img. This helper stays the else-branch:
/// a flat upstream diffusers repo (`stabilityai/stable-diffusion-xl-base-1.0`, `SG161222/RealVisXL_V5.0`)
/// roots its `unet/` and is returned untouched, and a non-standard tiered turnkey resolves its dense
/// `bf16/` (an SDXL turnkey has no component tree at its root, so the loader would find no `unet/`).
///
/// A flat snapshot roots its backbone dir — `unet/` for the SDXL family, `transformer/` for the DiTs
/// — and is returned untouched, so the existing two models keep resolving exactly as before. Same
/// backbone split the `apps/rust-api` training-readiness gate keys on (sc-10613).
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn dense_tier_subdir(root: PathBuf) -> PathBuf {
    let roots_a_component_tree = root.join("unet").is_dir() || root.join("transformer").is_dir();
    if !roots_a_component_tree && root.join("bf16").is_dir() {
        return root.join("bf16");
    }
    root
}

/// Whether `model` is one of the three Anima catalog ids (epic 10512). Anima is convert-at-install with
/// a bespoke tier resolver, so — like Ideogram/Boogu/Krea — it is NOT in [`STANDARD_TIER_MODELS`].
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn is_anima_model(model: &str) -> bool {
    matches!(model, "anima_base" | "anima_aesthetic" | "anima_turbo")
}

/// Pick the tier subdir of a converted Anima `root` (the injected `modelPath`, holding `bf16/ q8/ q4/`,
/// each a `diffusion_models/<variant>.safetensors` + dense `text_encoders/` + `vae/` tree the Anima
/// loader reads): `bf16/` when the request opts out of quantization (`advanced.mlxQuantize <= 0`), `q4/`
/// when it opts into Q4 (`1..=4`), else the default **`q8/`** (sc-10714). Falls back through q8 → bf16 →
/// q4 → `root` so a partially-written artifact surfaces as a load error rather than a silent half-load
/// onto the low-fidelity q4. A tier is "present" when its `diffusion_models/` holds a `.safetensors` DiT
/// (packed OR dense bf16). Mirrors [`standard_tier_subdir`], but keyed on Anima's `split_files` layout
/// rather than `transformer/`, and — unlike the standard q4-first convention — defaults to q8 (below).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn anima_tier_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        // A hidden `._*.safetensors` AppleDouble sidecar is not a DiT (SceneWorks#1333).
        let has_dit = std::fs::read_dir(dir.join("diffusion_models"))
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|entry| !sceneworks_core::lora_family::is_hidden_file(&entry.path()))
                    .any(|entry| entry.file_name().to_string_lossy().ends_with(".safetensors"))
            })
            .unwrap_or(false);
        has_dit.then_some(dir)
    };
    // Default (no explicit `mlxQuantize`) → the app-wide Q8 default (epic 10721 / sc-10726), CLAMPED UP
    // to the model's per-model quality floor (`mlx.minQualityTier`, sc-10731). This REPLACES the sc-10714
    // anima-specific `None => "q8"` hardcode with the SHARED, floor-driven [`preferred_tier`]: Anima
    // base/aesthetic declare `minQualityTier: q8` in the manifest, so their default is now floor-derived
    // — the fix for base/aesthetic rendering WASHED at q4 (q4 weight-quant error amplified by CFG 4.5 over
    // 30 steps) is a general mechanism, not a resolver special-case. Turbo declares NO floor (it is
    // CFG-free, so q4 is acceptable there) and rides the plain q8 default. bf16 (explicit `<= 0`) is there
    // for max fidelity/speed on this small 2B DiT, and an explicit q4 pick is still honored (the web
    // flags a below-floor pick with an advisory). Fallback prefers the clean tiers (q8 → bf16 → q4) so a
    // partial install never silently lands on the washed q4.
    //
    // `nvfp4: false` (sc-11042): Anima is an MLX convert-at-install family and hosts no NVFP4 tier — the
    // lane is candle/Blackwell-only. Passing `false` keeps this resolver byte-identical to its pre-sc-11042
    // behavior; wire it here if an Anima NVFP4 tier is ever converted.
    let preferred = preferred_tier(bits, min_quality_floor(request), false);
    // sc-12279 generalized: Anima ships no `model_index.json`, so route the chain through the concrete
    // `anima_tier_complete` predicate — a torn tier (DiT present, text-encoder/VAE absent) now falls
    // through to a complete sibling instead of reaching the loader and dying on the missing file.
    pick_loadable_tier(
        &[preferred, "q8", "bf16", "q4"],
        &present,
        &sceneworks_core::mlx_tier_completeness::anima_tier_complete,
    )
        .unwrap_or_else(|| root.to_path_buf())
}

/// The candle off-Mac Anima weights dir (sc-10676): the `split_files/` subdir of the HF snapshot `root`
/// when it holds `diffusion_models/` (the dense DiT tree `runtime_cuda::providers::anima::loader::resolve_split_files`
/// reads — the exact dir the GPU-validated anima smoke used), else `root` itself. The candle loader also
/// accepts the snapshot parent, so falling back to `root` keeps a partial download a loud load error, not
/// a silently-wrong dir. Anima has NO convert-at-install tier off-Mac (the `anima_quant` converter is
/// macOS-only), so this deliberately SKIPS [`anima_tier_subdir`]'s bf16/q8/q4 tier descent — off-Mac is
/// always dense bf16.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn anima_dense_split_files_dir(root: PathBuf) -> PathBuf {
    let split = root.join("split_files");
    if split.join("diffusion_models").is_dir() {
        split
    } else {
        root
    }
}

/// The Ideogram 4 tier subdir a `mlxQuantize` request needs fetched ON DEMAND — `Some("q8")` when the
/// request opts into Q8 (`> 4`), else `None` (the shipped default `q4/`, which the catalog download
/// pulls; bf16 is a SEPARATE catalog repo the user opts into on the Models page, never an on-demand
/// fetch). The FETCH-side helper for [`ensure_ideogram_tier_present`] (which tier to pull). The LOAD-side
/// resolver [`ideogram_model_subdir`] no longer shares this: as of sc-10777 it routes its default through
/// the floor-aware [`preferred_tier`] (so a floored default clamps up to `mlx.minQualityTier`, capped by
/// installed), while this fetch helper stays keyed on the explicit pick only — no shipping Ideogram model
/// declares a floor, so the two still agree for every current model. Mirrors [`boogu_tier_subdir`].
fn ideogram_tier_subdir(bits: Option<i64>) -> Option<&'static str> {
    match bits {
        Some(b) if b > 4 => Some("q8"),
        _ => None,
    }
}

/// Pick the engine-complete packed subdir of an Ideogram 4 turnkey `root`: `q8/` when the request
/// opts into Q8 (`advanced.mlxQuantize: 8`) AND it is downloaded, `q4/` for an explicit Q4 pick
/// (`1..=4`), else — with NO explicit `mlxQuantize` — the **`q8/`** default (epic 10721 / sc-10726),
/// CLAMPED UP to the model's per-model quality floor (`mlx.minQualityTier`, sc-10731 via the shared
/// [`preferred_tier`], sc-10777) and DOWN to what's installed (only `q4/` on disk ⇒ q4). Falls back to
/// `root` if neither subdir is present (a partially-downloaded bundle surfaces as a load error rather
/// than a silent half-load). The `q8/` tier is an on-demand download fetched by
/// [`ensure_ideogram_tier_present`] on an explicit opt-in (sc-9607); this resolver never triggers that
/// fetch for the plain default — it simply prefers q8 when it happens to be on disk. Ideogram's turnkey
/// carries only `q4/`/`q8/` (bf16 is a separate catalog repo), so the clean-tiers fallback here is
/// q8 ⇆ q4 and a bf16 floor has no in-turnkey tier to land on (it falls through to the best installed).
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
    // The tier the request prefers: an explicit `mlxQuantize` pick maps directly (`<=0` → bf16, `>4` →
    // q8, `1..=4` → q4), else the app-wide q8 default (epic 10721 / sc-10726) CLAMPED UP to the model's
    // per-model quality floor (`mlx.minQualityTier`, sc-10731) — the SAME shared, floor-aware logic as
    // [`standard_tier_subdir`] / [`anima_tier_subdir`] (sc-10777, routing this resolver through
    // [`preferred_tier`] instead of its own non-floor-aware q8 default). No shipping Ideogram model
    // declares a floor, so this is byte-identical to the prior q8-first default for every current model —
    // it just keeps the worker default path from silently landing below a floor should one ever be
    // declared. Ideogram's turnkey carries only `q4/`/`q8/` (bf16 is the separate `SceneWorks/ideogram-4`
    // catalog repo, never an in-turnkey subdir), so a preferred `bf16` — an explicit `<=0` pick, or a
    // hypothetical bf16 floor — simply isn't present and falls through the clean-tiers chain (q8 → q4) to
    // the best installed tier, exactly as the prior resolver did.
    //
    // `nvfp4: false` (sc-11042): the Ideogram turnkey hosts no NVFP4 tier. Byte-identical to its
    // pre-sc-11042 behavior; wire it here if one is ever converted.
    let preferred = preferred_tier(bits, min_quality_floor(request), false);
    // sc-12279: the Ideogram turnkey ships a per-tier `model_index.json`, so the shared chain prefers
    // a tier whose declared component tree is fully on disk over a torn one.
    pick_loadable_tier(&[preferred, "q8", "q4"], &present, &tier_components_present)
        .unwrap_or_else(|| root.to_path_buf())
}

/// The Boogu subfolder for a `mlxQuantize` request — `None` keeps the Q8 default. The FETCH-side helper
/// for [`ensure_boogu_tier_present`] (which non-default tier to pull on demand): `<=0` → `<variant>-bf16/`
/// (dense full precision), `1..=4` → `<variant>-q4/` (packed Q4, sc-8513), anything else → `None` (the
/// default `<variant>/` packed Q8 ships in the catalog download). Returns the subfolder name relative to
/// the turnkey root. The LOAD-side resolver [`boogu_model_subdir`] no longer shares this: as of sc-10777 it
/// routes its default through the floor-aware [`preferred_tier`] (so a floored default clamps up to
/// `mlx.minQualityTier`, capped by installed), while this fetch helper stays keyed on the explicit pick
/// only — no shipping Boogu model declares a floor, so the two still agree for every current model.
fn boogu_tier_subdir(variant: &str, bits: Option<i64>) -> Option<String> {
    match bits {
        Some(b) if b <= 0 => Some(format!("{variant}-bf16")),
        Some(b) if b <= 4 => Some(format!("{variant}-q4")),
        _ => None,
    }
}

/// Pick the engine-complete subfolder of a Boogu turnkey `root` for the requested variant. Each
/// catalog id maps to a variant folder: `boogu_image`→`base`, `boogu_image_turbo`→`turbo`,
/// `boogu_image_edit`→`edit`. **Q8 is the shipped default** (the pre-packed `<variant>/` folder),
/// CLAMPED UP to the model's per-model quality floor (`mlx.minQualityTier`, sc-10731 via the shared
/// [`preferred_tier`], sc-10777 — a bf16 floor raises the picker-less default to `<variant>-bf16/`); an
/// explicit advanced `mlxQuantize` selects another tier (sc-8513, epic 8506): `<=0` → the dense
/// `<variant>-bf16/`, `1..=4` → the packed `<variant>-q4/`. Falls back through Q8 → q4 → bf16 → `root`
/// when the requested tier isn't downloaded, so a partial bundle surfaces as a load error rather than
/// a silent half-load (a floor tier not on disk falls to the best installed). (The non-default tiers are
/// on-demand downloads fetched by [`ensure_boogu_tier_present`] before this resolves, sc-6568/sc-8513.)
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
    // Boogu names its tiers `<variant>` (the packed Q8 shipped default), `<variant>-q4` and
    // `<variant>-bf16` — map a generic [`preferred_tier`] tier name onto that layout.
    let variant_folder = |tier: &str| -> String {
        match tier {
            "q8" => variant.to_owned(),
            other => format!("{variant}-{other}"),
        }
    };
    // The tier the request prefers: an explicit `mlxQuantize` pick maps directly (`<=0` → bf16, `1..=4` →
    // q4, else Q8 — the same mapping the old `boogu_tier_subdir` load-path branch used), else the app-wide
    // Q8 default (epic 10721 / sc-10726) CLAMPED UP to the model's per-model quality floor
    // (`mlx.minQualityTier`, sc-10731) — the SAME shared, floor-aware logic as [`standard_tier_subdir`] /
    // [`anima_tier_subdir`] (sc-10777, routing this resolver through [`preferred_tier`] instead of its own
    // non-floor-aware Q8 default). No shipping Boogu model declares a floor, so this is byte-identical to
    // the prior Q8-first default for every current variant; the routing keeps the resolver from silently
    // landing below a floor should one ever be declared — a bf16 floor would raise the picker-less default
    // from the packed Q8 up to the dense `<variant>-bf16`, capped by what's installed. The clean-tiers
    // fallback (Q8 → q4 → bf16) is unchanged, so a partial bundle still surfaces as a load error.
    //
    // `nvfp4: false` (sc-11042): Boogu's `<variant>-bf16` MLX layout hosts no NVFP4 tier. Byte-identical
    // to its pre-sc-11042 behavior; wire it here if one is ever converted.
    let preferred = preferred_tier(bits, min_quality_floor(request), false);
    // sc-12279 generalized: Boogu ships no `model_index.json`, so route the folder chain through the
    // concrete `boogu_tier_complete` predicate — a torn tier (transformer present, `mllm/tokenizer.json`
    // or VAE absent) now falls through to a complete sibling instead of crashing the loader on the
    // missing tokenizer. Chain is folder names (`<variant>`, `<variant>-q4`, `<variant>-bf16`).
    let chain = [variant_folder(preferred), variant.to_owned(), q4, bf16];
    let chain_refs: Vec<&str> = chain.iter().map(String::as_str).collect();
    pick_loadable_tier(
        &chain_refs,
        &present,
        &sceneworks_core::mlx_tier_completeness::boogu_tier_complete,
    )
        .unwrap_or_else(|| root.to_path_buf())
}

/// Pick the engine-complete packed subdir of a Krea 2 Turbo turnkey `root`: `q4/` when the request opts
/// into Q4 (`advanced.mlxQuantize <= 4`) AND it is downloaded, else the default `q8/` (the shipped
/// default — the P1-validated near-lossless quant), CLAMPED UP to the model's per-model quality floor
/// (`mlx.minQualityTier`, sc-10731 via the shared [`preferred_tier`], sc-10845 — a bf16 floor raises the
/// picker-less default to the dense `bf16/`). Falls back to whichever subdir is present, then `root`, so
/// a partially-downloaded bundle surfaces as a load error rather than a silent half-load (a floor tier not
/// on disk falls to the best installed). The turnkey (`SceneWorks/krea-2-turbo-mlx`, sc-7573) carries one
/// `from_snapshot`-loadable subdir per quant (each with a packed
/// `transformer/diffusion_pytorch_model.safetensors`); the loader auto-detects the packed quant, so the
/// resolved `spec.quantize` is a no-op on it. Mirrors [`ideogram_model_subdir`] (q4/q8 subdirs) with
/// Boogu's packed-transformer filename and a Q8-default selection.
fn krea_model_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    krea_model_subdir_gated(root, request, nvfp4_host_eligible())
}

/// [`krea_model_subdir`] with the NVFP4 **host** gate passed in rather than probed (sc-11042). Split
/// out for testability, exactly like [`standard_tier_subdir_gated`]; production has one caller.
fn krea_model_subdir_gated(root: &Path, request: &ImageRequest, nvfp4_host: bool) -> PathBuf {
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
    // The tier the request prefers: an explicit `mlxQuantize` pick maps directly (`<=0` → bf16 the dense
    // base which krea's loader takes with no quantize, `>4` → q8, `1..=4` → q4 — the SAME mapping krea's
    // own match used), else the app-wide q8 default (epic 10721 / sc-10726, the P1-validated near-lossless
    // quant) CLAMPED UP to the model's per-model quality floor (`mlx.minQualityTier`, sc-10731) — the SAME
    // shared, floor-aware logic as `standard_tier_subdir` / `anima_tier_subdir` / `ideogram_model_subdir` /
    // `boogu_model_subdir` (sc-10845, routing this last bespoke resolver through [`preferred_tier`] instead
    // of its own non-floor-aware q8 default). No shipping Krea model declares a floor, so this is
    // byte-identical to the prior q8 default; the routing keeps the resolver from silently landing below a
    // floor should one ever be declared (a bf16 floor would raise the picker-less default from q8 to the
    // dense bf16, capped by what's installed). Fallback stays krea's q8 → q4 → bf16 (bf16 last — it is the
    // heaviest dense tree), so a partial bundle still surfaces as a load error.
    //
    // NVFP4 (sc-11042): wired here as well as in [`standard_tier_subdir`] because Krea 2 Turbo is epic
    // 11037's named SC#1/SC#2 validation vehicle (the 2026-07-15 Sana → Krea redirect, sc-12110) and its
    // per-tier subdir shape takes an `nvfp4/` dir with no new logic — the same packed
    // `transformer/diffusion_pytorch_model.safetensors` the q4/q8 tiers carry. No `nvfp4/` dir exists on
    // disk yet (sc-11043 owns the convert-at-install loop), so today this always falls through the
    // unchanged q8 → q4 → bf16 chain; wiring it now means the tier resolves the moment the converter
    // lands, rather than needing a second edit here.
    let preferred = preferred_tier(
        bits,
        min_quality_floor(request),
        nvfp4_requested(request) && nvfp4_host,
    );
    // sc-12279: a torn tier no longer short-circuits this chain — `pick_loadable_tier` prefers a tier
    // whose `model_index.json` component tree is fully on disk, so Raw with a torn `q8/` and a
    // complete `bf16/` training-base tier now generates off bf16 instead of dying on the absent
    // `q8/tokenizer/` (issue #850's symptom).
    pick_loadable_tier(&[preferred, "q8", "q4", "bf16"], &present, &tier_components_present)
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
/// pins, sc-8879/sc-9682). The native downloader still verifies each file's own hash on download.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_MLX_TURNKEY_REVISION: &str = "d009674080cc1bccf2b629d834c34bf5eccdb723";

/// On-demand fetch of the canonical bf16 Krea 2 base surface for the INT8-ConvRot tier (sc-9300),
/// the sibling of [`ensure_boogu_tier_present`]. The ConvRot catalog download pulls only the DiT
/// single-file; the bf16 `bf16/` subdir of the `krea-2-turbo-mlx` turnkey (tokenizer / Qwen3-VL TE /
/// Qwen-Image VAE / config) is fetched here when the ConvRot tier is selected and it isn't present —
/// so q4/q8 users are never forced to download the 35 GB bf16 base (it isn't a global co-requisite).
/// No-op when the request isn't a ConvRot job or the bf16 base is already complete; a real download
/// error fails loud (otherwise `resolve_krea_convrot` loads the freshly fetched base).
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
    let files = vec![
        "bf16/transformer/*".to_owned(),
        "bf16/text_encoder/*".to_owned(),
        "bf16/vae/*".to_owned(),
        "bf16/tokenizer/*".to_owned(),
        "bf16/scheduler/*".to_owned(),
        "bf16/model_index.json".to_owned(),
    ];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        KREA_MLX_TURNKEY_REPO,
        KREA_MLX_TURNKEY_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// On-demand fetch of a non-default Boogu tier subfolder (sc-6568 / sc-8513). The catalog download
/// pulls only the packed Q8 `<variant>/` subfolder, so when a job opts into another tier
/// ([`boogu_tier_subdir`]: `<=0` → `<variant>-bf16/` dense, `1..=4` → `<variant>-q4/` packed) and that
/// subfolder isn't present yet, pull just its files into the HF cache so [`boogu_model_subdir`]
/// resolves it. No-op when the Q8 default is requested, the model isn't Boogu, the turnkey snapshot
/// isn't downloaded yet (`boogu_model_subdir` then falls back to Q8 / surfaces the load error), or the
/// tier subfolder is already complete. Fails loud on a real download error — fast, before any compute;
/// a tier that isn't published yet stays absent so the request falls back to Q8. Mirrors
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
    // The tier subfolder nests transformer/mllm/vae (leaf-dir globs, like the catalog Q8 entry).
    let files = vec![
        format!("{tier}/transformer/*"),
        format!("{tier}/mllm/*"),
        format!("{tier}/vae/*"),
    ];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        &model_repo(request, &model),
        "main",
        &files,
    )
    .await
    .map(|_| ())
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
/// real download error; a `q8/` tier that isn't published yet stays absent so the request falls back to q4.
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
    let files = vec![format!("{tier}/*")];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        &model_repo(request, &model),
        "main",
        &files,
    )
    .await
    .map(|_| ())
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

/// Resolve quantization: the explicit `advanced.quantTier: "nvfp4"` label → `advanced.mlxQuantize` →
/// `manifest.mlx.quantize` → Q8 default. The engine supports Q4/Q8/NVFP4; map (<=0 → dense, <=4 → Q4,
/// else Q8). Returns the engine quant + the effective bit count for the recipe (None = dense bf16).
///
/// Shared by the MLX path and the candle lane (sc-5126). On the candle lane it is called ONLY for a
/// family whose descriptor advertises `supported_quants` (i.e. Lens — see `generate_candle_stream`'s
/// `model.supports_quant()` gate), so the Q8 default applies to Lens exactly like the MLX families;
/// the sc-3675/sc-5096 candle families advertise no quant and never reach this resolver (stay dense).
///
/// **NVFP4 (sc-11042, epic 11037 SC#5)** is selected ONLY on a full [`nvfp4_selected`] — an explicit
/// pick, a Blackwell-eligible host, AND the `nvfp4/` tier actually resolved on disk — never from
/// `bits`, a manifest default, or hardware detection alone, and never on a `tier_dir` that resolved to
/// some other tier (that would load FP4 against q8 weights and record the wrong tier). Its recipe
/// bit count is `None`, not `Some(4)`: `Quant::Nvfp4::bits()` returns 4 (its E2M1 elements are 4-bit),
/// but NVFP4 is ~4.5 EFFECTIVE bits/weight and is a different tier from int4-affine `q4`, so reporting
/// `4` here would stamp an NVFP4 render with `q4`'s bit count in the recipe — the aliasing SC#5 forbids
/// (the same footgun `video_quant_label` carries; see its note). Every non-NVFP4 request takes the
/// unchanged path below.
///
/// `tier_dir` is the RESOLVED tier subdir this job will load (`standard_tier_subdir`'s output), or
/// `None` on a lane with no standard-tier layout in scope — see [`tier_dir_is_nvfp4`] for why `None`
/// conservatively means "not NVFP4".
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_quant(request: &ImageRequest, tier_dir: Option<&Path>) -> (Option<Quant>, Option<i64>) {
    resolve_quant_gated(request, nvfp4_host_eligible(), tier_dir)
}

/// [`resolve_quant`] with the NVFP4 **host** gate passed in rather than probed (sc-11042). Split out
/// for testability, exactly like [`standard_tier_subdir_gated`] — and necessarily so: the candle CI
/// lane runs ON the sm_120 rig, so a test that let this probe the live cap would assert different
/// things on the rig and on a developer box. Production has one caller.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_quant_gated(
    request: &ImageRequest,
    nvfp4_host: bool,
    tier_dir: Option<&Path>,
) -> (Option<Quant>, Option<i64>) {
    // Dense-TE turnkeys (FLUX.2-klein, sc-8711): the tier subdir already holds a packed transformer
    // + a DENSE bf16 text encoder, so the load Quant must be None — quantizing here would crush the
    // dense bf16 TE we deliberately kept full-precision. The packed transformer self-describes its
    // quant regardless. Tier selection (q4/q8/bf16) is driven by the resolved subdir, not this.
    //
    // Ordered FIRST, ahead of the NVFP4 arm (sc-11042): `flux2_klein_9b`/`_kv` are BOTH in
    // `DENSE_TE_TIER_MODELS` and on the candle txt2img lane (whose `resolve_quant` call is gated only
    // by `model.supports_quant()`), so an NVFP4 arm placed above this could return `Some(Nvfp4)` for a
    // crafted `quantTier: "nvfp4"` on a dense-TE turnkey and skip the carve-out — re-quantizing the
    // bf16 text encoder sc-8711/sc-9362 deliberately kept dense. The carve-out is the wider invariant
    // (the TE must never be quantized by ANY tier), so it wins outright rather than being an exception
    // the NVFP4 arm has to remember.
    if is_dense_te_tier(request) {
        return (None, None);
    }
    // The distinct NVFP4 tier (sc-11042): an EXPLICIT `advanced.quantTier: "nvfp4"` pick AND a
    // Blackwell-eligible host AND the `nvfp4/` tier resolved on disk — all three halves, the SC#5
    // opt-in (see [`nvfp4_selected`]). Checked before the bits map: it is a tier identity, not a point
    // on the bits ladder. Off Blackwell / off the candle lane `nvfp4_host` is false, and no shipping
    // model packs an `nvfp4/` dir today, so this arm is unreachable there and every request resolves
    // exactly as it always has.
    if nvfp4_selected(request, nvfp4_host, tier_dir) {
        return (Some(Quant::Nvfp4), None);
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
/// picks its `bf16`/`q8`/`q4` tier (no explicit pick → `q8`; `<=0 → bf16`; `>4 → q8`; else `q4`).
/// Returns the recipe bit count of the REQUESTED tier: `None` (bf16) / `Some(8)` / `Some(4)`. Kept in
/// lockstep with the q8 default (sc-10726) so a straight default dense-TE job that resolves the q8 tier
/// isn't mis-reported as a bf16/q4→q8 tier change by [`reconcile_resolved_tier_quant`].
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
        None => Some(8),
        Some(b) if b <= 0 => None,
        Some(b) if b > 4 => Some(8),
        _ => Some(4),
    }
}

/// The generation tier (`bf16`/`q8`/`q4`) a resolved turnkey tier subdir ACTUALLY loads at, parsed
/// from its basename (sc-8820 / sc-12090). The tier resolvers ([`standard_tier_subdir`],
/// [`ideogram_model_subdir`], [`boogu_model_subdir`], [`krea_model_subdir`]) fall through q4→q8→bf16
/// (or the Boogu `<variant>`-shaped names) when the requested tier isn't downloaded, so the resolved
/// path can be a DIFFERENT precision than the request asked for. This maps the resolved basename back
/// to its tier:
///   - `bf16` / `<variant>-bf16` → `bf16`
///   - `q4`   / `<variant>-q4`   → `q4`
///   - `q8`   / `<variant>-q8`   → `q8`
///   - a bare Boogu `<variant>/` (no `-q4`/`-q8`/`-bf16` suffix) → the shipped **Q8** default
///
/// `None` when the basename is not a recognizable tier name — e.g. the resolver fell all the way back
/// to the repo `root` (a partial/absent turnkey the engine will error on), or the dir is a `modelPath`
/// override. In that case the caller keeps its request-derived tier rather than inventing one.
///
/// Available on the candle lane too (sc-12090): the candle VRAM fit-gate reads the tier the
/// disk-probing resolver landed on here — instead of re-deriving from the manifest — so it names and
/// budgets against the tier that would actually load, never an uninstalled one.
/// The tier key the candle VRAM fit-gate sizes a render against — the ONE place that decision is made
/// (sc-12425). Pure: extracted out of `generate_candle_stream` so the ConvRot identity below is
/// unit-testable; that function takes an api/settings/job and cannot be exercised from a unit test, so
/// the mapping had no gate of its own.
///
/// Resolution order:
///
/// 1. **A resolved ConvRot load ⇒ its tier IDENTITY** (`convrot_resolved` = the base dir AND the int8
///    DiT are both on disk). This is the sc-12425 fix. It used to hand ConvRot to
///    [`vram_gate::requested_tier_key`](crate::vram_gate::requested_tier_key), which is BITS-derived —
///    and a ConvRot request carries no `mlxQuantize` ([`wants_krea_convrot`]), so it fell to that
///    function's `None => "q8"` arm. That sized a **measured 42.9 GB** render (sc-12381) against q8's
///    35.9 + 2.0 = 37.9 GB and admitted loads that OOM. Identical aliasing to the one sc-11042 fixed for
///    NVFP4, except q8 OVER-predicts NVFP4 ("never an OOM") and UNDER-predicts ConvRot.
/// 2. Else the **on-disk tier** the resolver landed on (sc-12090) — budget the tier that will load, not
///    one the user never installed. A `modelPath`/flat root has no recognizable basename ⇒ `None`.
/// 3. Else the manifest/request key (`requested_tier_key`), whose own `nvfp4` arm is the sibling of (1).
// Candle-lane only — NOT `any(macos, candle)` like `tier_key_from_resolved_dir`, because this fn calls
// `crate::vram_gate`, which is itself `#[cfg(all(not(macos), backend-candle))]`. On the macOS/MLX build
// `vram_gate` doesn't exist, so an `any(macos, ...)` gate here fails to compile (E0433). Its only caller,
// `generate_candle_stream`, is candle-only too, so this loses nothing.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn gate_tier_key(
    convrot_resolved: bool,
    weights_dir: &Path,
    advanced: &JsonObject,
    manifest_entry: &JsonObject,
    nvfp4: bool,
) -> &'static str {
    if convrot_resolved {
        return INT8_CONVROT_TIER;
    }
    tier_key_from_resolved_dir(weights_dir)
        .unwrap_or_else(|| crate::vram_gate::requested_tier_key(advanced, manifest_entry, nvfp4))
}

#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn tier_key_from_resolved_dir(dir: &Path) -> Option<&'static str> {
    let name = dir.file_name()?.to_str()?;
    // Match the trailing tier token: the whole basename (standard/ideogram/krea) or the suffix of a
    // Boogu `<variant>-<tier>` folder; a bare Boogu `<variant>` IS the packed Q8 default.
    match name.rsplit('-').next().unwrap_or(name) {
        "bf16" => Some("bf16"),
        "q4" => Some("q4"),
        "q8" | "base" | "turbo" | "edit" => Some("q8"),
        _ => None,
    }
}

/// The `(engine Quant, recipe bit count)` a resolved turnkey tier subdir ACTUALLY loads at (sc-8820) —
/// the `Quant`-typed view of [`tier_key_from_resolved_dir`], consumed by
/// [`reconcile_resolved_tier_quant`] on the MLX `generate_stream` path. `None` for an unrecognizable
/// basename (kept identical to the pre-sc-12090 behavior: the caller keeps the request-derived quant).
///
/// macOS-only: the candle lane has no quant-tier layout to reconcile, so this would be dead code there.
#[cfg(target_os = "macos")]
fn tier_quant_from_resolved_dir(dir: &Path) -> Option<(Option<Quant>, Option<i64>)> {
    match tier_key_from_resolved_dir(dir)? {
        "bf16" => Some((None, None)),
        "q4" => Some((Some(Quant::Q4), Some(4))),
        "q8" => Some((Some(Quant::Q8), Some(8))),
        _ => None,
    }
}

/// Resolve the on-disk weights dir for `request` FORCED to a specific generation `tier`
/// (`bf16`/`q8`/`q4`), reusing the SAME family disk-probing resolver the default path uses
/// ([`resolve_weights_dir`]). `Some(dir)` only when that tier is actually INSTALLED — the resolver
/// lands on that exact tier; `None` when it isn't (the resolver falls through to a different/absent
/// tier) or the model has no quant-tier layout. This lets the capability downtier (sc-10733) and the
/// reject-message tier suggestions (sc-12090) enumerate installed tiers WITHOUT duplicating each
/// family's bespoke `present()` logic — a forced-bits probe is byte-for-byte what the loader would do.
///
/// The forced `mlxQuantize` is an explicit tier probe, so it bypasses the per-model quality floor
/// (explicit picks are always honored) — the caller filters candidates to `>= floor` before probing.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn resolve_tier_dir(request: &ImageRequest, settings: &Settings, tier: &str) -> Option<PathBuf> {
    let bits: i64 = match tier {
        "bf16" => 0,
        "q4" => 4,
        _ => 8, // q8
    };
    let mut probe = request.clone();
    probe
        .advanced
        .insert("mlxQuantize".to_owned(), Value::from(bits));
    let dir = resolve_weights_dir(&probe, settings).ok().flatten()?;
    (tier_key_from_resolved_dir(&dir) == Some(tier_static_name(tier))).then_some(dir)
}

/// The generation tiers (`bf16`/`q8`/`q4`) currently INSTALLED for `request`'s model, in DESCENDING
/// fidelity (bf16 → q8 → q4). Each is confirmed by [`resolve_tier_dir`] (a forced-bits probe of the
/// real family resolver), so the list is exactly what the loader could load. Empty for a model with no
/// quant-tier layout (a flat / `modelPath` snapshot).
///
/// Candle-only: the sc-12090 reject-message enumeration. The MLX downtier reject names the single
/// smallest evaluated tier from `choose_downtier`, so it needs no separate installed-list scan.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn installed_tier_keys(request: &ImageRequest, settings: &Settings) -> Vec<&'static str> {
    ["bf16", "q8", "q4"]
        .into_iter()
        .filter(|tier| resolve_tier_dir(request, settings, tier).is_some())
        .collect()
}

/// A per-tier capability-fit result for the downtier chooser (sc-10733) — the lane-agnostic reduction
/// of each lane's richer fit decision (candle's resident/offload/reject, MLX's resident/sequential/
/// reject) to "does this tier run at all on this machine."
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum TierFit {
    /// Runs — resident or (where the provider stages components) sequentially.
    Fits,
    /// Won't run even sequentially. `needed_gb`/`available_gb` are for the reject message.
    TooBig { needed_gb: f64, available_gb: f64 },
}

/// The capability-downtier decision (sc-10733).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum DowntierPick {
    /// The resolved default tier fits — load it unchanged.
    Keep,
    /// A LOWER installed tier is the highest-fidelity one that fits — load it instead.
    Downtier(&'static str),
    /// Nothing in `[floor, default]` fits — reject, naming the SMALLEST (least-demanding) tier
    /// evaluated and what it needed.
    Reject {
        tier: &'static str,
        needed_gb: f64,
        available_gb: f64,
    },
}

/// The pure capability-downtier chooser (sc-10733), shared by the candle and MLX gates. `candidates`
/// are the INSTALLED tiers in `[floor, default]` with their per-lane [`TierFit`], in DESCENDING fidelity
/// (so the default tier is first). Returns the highest-fidelity tier that fits: [`DowntierPick::Keep`]
/// when that is the default itself, [`DowntierPick::Downtier`] when a lower tier is the best that fits,
/// or [`DowntierPick::Reject`] (naming the smallest — least-demanding — tier evaluated) when nothing in
/// range fits.
///
/// The floor + installed clamping live in the CANDIDATE list (the caller filters to `rank >= floor` and
/// `installed`), so the quality floor always wins over the downtier — a floor-q8 model's candidates
/// never include q4, so it rejects rather than silently rendering q4 (acceptance #5). An explicit user
/// pick never reaches here (the caller skips the downtier for it, honoring the pick — acceptance #7).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn choose_downtier(default_tier: &str, candidates: &[(&'static str, TierFit)]) -> DowntierPick {
    let mut smallest_reject: Option<(&'static str, f64, f64)> = None;
    for &(tier, fit) in candidates {
        match fit {
            // DESCENDING fidelity ⇒ the first that fits is the highest-fidelity fitting tier.
            TierFit::Fits => {
                return if tier == default_tier {
                    DowntierPick::Keep
                } else {
                    DowntierPick::Downtier(tier)
                };
            }
            TierFit::TooBig {
                needed_gb,
                available_gb,
            } => smallest_reject = Some((tier, needed_gb, available_gb)),
        }
    }
    // Nothing fit — reject, naming the LAST (smallest / least-demanding) tier we tried. `None` only when
    // the candidate list was empty (no installed tier in range — defensive; the default itself is always
    // installed & in range), in which case Keep lets the plain gate handle it.
    match smallest_reject {
        Some((tier, needed_gb, available_gb)) => DowntierPick::Reject {
            tier,
            needed_gb,
            available_gb,
        },
        None => DowntierPick::Keep,
    }
}

/// The installed tiers in `[floor, default]` for `request`'s model, in DESCENDING fidelity, ready for
/// [`choose_downtier`] once each is paired with its per-lane [`TierFit`] (sc-10733). `default_tier` is
/// the disk-clamped resolved tier (the downtier ceiling — the clamp only ever lowers fidelity); `floor`
/// is the per-model quality floor ([`min_quality_floor`], defaulting to `q4`). Excludes tiers not on
/// disk so the downtier never lands on an uninstalled tier.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn downtier_candidate_tiers(
    request: &ImageRequest,
    settings: &Settings,
    default_tier: &str,
    floor: Option<&str>,
) -> Vec<&'static str> {
    let default_rank = tier_quality_rank(default_tier);
    // Floor at least q4 (rank 1) — a downtier must never fall below the lowest real tier.
    let floor_rank = floor.map_or(1, tier_quality_rank).max(1);
    ["bf16", "q8", "q4"]
        .into_iter()
        .filter(|tier| {
            let rank = tier_quality_rank(tier);
            rank <= default_rank && rank >= floor_rank
        })
        .filter(|tier| resolve_tier_dir(request, settings, tier).is_some())
        .collect()
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

/// The adapter filename a LoRA record's manifest `files` list declares (its first
/// entry), if any (sc-10221). When `lora_path` resolves to a record *directory*, this
/// is the specific adapter to load — e.g. a trained LoRA's final `<stem>.safetensors`
/// rather than a `<stem>-stepNNN` checkpoint sharing the folder. Untrusted (rides the
/// job payload); `resolve_adapter_in_dir` re-validates it as a plain in-dir filename.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn declared_adapter_file(lora: &Value) -> Option<&str> {
    lora.get("files")
        .and_then(Value::as_array)
        .and_then(|files| files.first())
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
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
            // Prefer the manifest-declared adapter over an arbitrary directory scan so a
            // trained LoRA loads its final adapter, not a step checkpoint (sc-10221).
            crate::resolve_adapter_in_dir(&path, declared_adapter_file(lora)).ok_or_else(|| {
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

/// Stage a media model's caller-provisioned components (epic 13657, sc-13679) onto its `LoadSpec` —
/// the image/video twin of the audio seam (`audio_jobs.rs run_audio_synthesis_using`). It reads the
/// model's descriptor `required_components` from the media registry and resolves each declared id to
/// its cached `coRequisite` snapshot via [`crate::model_jobs::resolve_co_requisites`] (all-or-nothing:
/// a missing co-requisite fails the job with an actionable error BEFORE the engine load), then stages
/// each in `LoadSpec::components` for the engine's own load-time `require_component` gate.
///
/// DORMANT at this pin: every current image/video descriptor advertises `required_components: &[]`, so
/// the early return keeps this a no-op (no manifest clone, no cache probe). It is a REAL generic path,
/// not a stub — the first image provider to advertise components (SDXL, sc-13682) is provisioned
/// through it with no new plumbing.
fn attach_required_components(
    spec: LoadSpec,
    model_id: &str,
    manifest_entry: &JsonObject,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    let Some(descriptor) = crate::inference_runtime::media_descriptor(model_id) else {
        return Ok(spec);
    };
    if descriptor.required_components.is_empty() {
        return Ok(spec);
    }
    let manifest_value = Value::Object(manifest_entry.clone());
    let components = resolve_co_requisites(&descriptor, &manifest_value, settings)?;
    Ok(components
        .into_iter()
        .fold(spec, |spec, (id, source)| spec.with_component(id, source)))
}

/// Resolve SDXL's three caller-staged components (`tokenizer_clip_l` / `tokenizer_clip_bigg` /
/// `vae_fp16_fix`, epic 13657 / sc-13682) for a BESPOKE candle lane whose provider takes explicit
/// component paths (`SdxlEditPaths` / `IpAdapterSdxlPaths` / `InstantIdPaths` — InstantID reuses the
/// candle SDXL conditioner + VAE) rather than a [`LoadSpec`]. Rides the SAME
/// generic [`crate::model_jobs::resolve_co_requisites`] seam the txt2img [`attach_required_components`]
/// path uses: it reads the registered candle `sdxl` descriptor's `required_components` (the exact three
/// ids the edit / IP-Adapter / trainer providers also consume) and maps each to this model's pinned
/// `coRequisite` download by `componentId`. All-or-nothing — a missing component fails the job BEFORE the
/// engine load with the seam's actionable error naming the component id + repo. Candle-only, because the
/// bespoke `SdxlEdit` / `IpAdapterSdxl` providers are (macOS keeps the self-contained MLX SDXL lane).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_sdxl_components(
    manifest_entry: &JsonObject,
    settings: &Settings,
) -> WorkerResult<(WeightsSource, WeightsSource, WeightsSource)> {
    // The bespoke edit / IP-Adapter providers ARE the candle `sdxl` engine, so resolve that descriptor
    // (id == "sdxl") and let its `required_components` advertisement drive the ids — never hardcoded here.
    let descriptor = crate::inference_runtime::media_descriptor("sdxl").ok_or_else(|| {
        WorkerError::Engine(
            "candle SDXL generator is not registered — cannot resolve its required components"
                .to_owned(),
        )
    })?;
    let manifest_value = Value::Object(manifest_entry.clone());
    let mut components =
        crate::model_jobs::resolve_co_requisites(&descriptor, &manifest_value, settings)?;
    let mut take = |id: &str| -> WorkerResult<WeightsSource> {
        components.remove(id).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "SDXL requires the '{id}' component, but its catalog entry declares no matching \
                 `coRequisite` download — the model manifest entry is misconfigured"
            ))
        })
    };
    Ok((
        take("tokenizer_clip_l")?,
        take("tokenizer_clip_bigg")?,
        take("vae_fp16_fix")?,
    ))
}

/// Registry-only generator load (epic 3720, sc-3724): resolve `engine_id` through the
/// backend-neutral `crate::inference_runtime::load` seam and return a `Box<dyn gen_core::Generator>`. Optionally
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
/// `crate::inference_runtime::load` seam and wrap a failure as `WorkerError::Engine`. Every image
/// control/base lane's `#[cfg(test)]` load wrapper funnels through here so the
/// `crate::inference_runtime::load` + `map_err` tail lives in one place (sc-8954). `cfg(target_os)`
/// still decides which provider crate registered the engine, not this call.
#[cfg(all(target_os = "macos", test))]
fn load_control_engine(engine_id: &str, spec: &LoadSpec) -> WorkerResult<Box<dyn Generator>> {
    crate::inference_runtime::load(engine_id, spec)
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
            | "sensenova_u1_8b_infographic_v3"
            | "sensenova_u1_8b_fast"
            | "sensenova_u1_8b_infographic_v2_fast"
            | "sensenova_u1_8b_infographic_v3_fast"
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
/// events. On the backend path `crate::inference_runtime::load` is a single atomic call that also fuses
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
mod metrics_settings_tests {
    use super::*;
    use serde_json::json;

    fn request(value: serde_json::Value) -> ImageRequest {
        ImageRequest::from_payload(value.as_object().unwrap())
    }

    #[test]
    fn default_run_reports_effective_settings_not_blank() {
        let req = request(json!({
            "projectId": "p", "model": "qwen_image", "prompt": "mist",
            "width": 1024, "height": 1024, "seed": 42
        }));
        let metrics =
            image_settings_metrics(&req, Some(20), Some(4.0), Some("q8".to_owned()), Some(8), 4);
        assert_eq!(metrics.model.as_deref(), Some("qwen_image"));
        assert_eq!(metrics.image_count, Some(4));
        assert_eq!(metrics.quant_label.as_deref(), Some("q8"));
        assert_eq!(metrics.quant_bits, Some(8));
        // A default run is not blank — sampler/scheduler/method carry the effective default.
        assert_eq!(metrics.sampler.as_deref(), Some("default"));
        assert_eq!(metrics.scheduler.as_deref(), Some("default"));
        assert_eq!(metrics.guidance_method.as_deref(), Some("cfg"));
        assert_eq!(metrics.use_pid, Some(false));
        assert_eq!(metrics.steps, Some(20));
        assert_eq!(metrics.seed, Some(42));
        assert_eq!(metrics.width, Some(1024));
        assert_eq!(
            metrics
                .guidance_scale
                .as_ref()
                .and_then(serde_json::Number::as_f64),
            Some(4.0)
        );
    }

    #[test]
    fn advanced_overrides_are_reported() {
        let req = request(json!({
            "projectId": "p", "model": "sdxl", "prompt": "mist", "width": 832, "height": 1216,
            "advanced": {
                "sampler": "dpmpp_2m", "scheduler": "karras", "schedulerShift": 3.0,
                "usePid": true, "pidTarget": "2k", "guidanceMethod": "cfgpp"
            }
        }));
        let metrics = image_settings_metrics(&req, Some(30), None, Some("bf16".to_owned()), None, 1);
        assert_eq!(metrics.sampler.as_deref(), Some("dpmpp_2m"));
        assert_eq!(metrics.scheduler.as_deref(), Some("karras"));
        assert_eq!(
            metrics
                .scheduler_shift
                .as_ref()
                .and_then(serde_json::Number::as_f64),
            Some(3.0)
        );
        assert_eq!(metrics.use_pid, Some(true));
        assert_eq!(metrics.pid_target.as_deref(), Some("2k"));
        assert_eq!(metrics.guidance_method.as_deref(), Some("cfgpp"));
        assert_eq!(metrics.quant_label.as_deref(), Some("bf16"));
        assert_eq!(metrics.quant_bits, None);
    }
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
    // Krea "text style" tap-reweight gain (sc-11878) — `None` for every non-Krea family (the engine
    // ignores the field regardless). The caller resolves it from the manifest `textStyleGain` slider +
    // `advanced` only when the model declares the control, so a non-Krea render passes `None`.
    text_style_gain: Option<f32>,
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
        text_style_gain,
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

/// Resolve the **Krea 2 Turbo img2img** init (epic 8588 slice A, sc-8591): `Some((reference, strength))`
/// from a `referenceAssetId` + `advanced.strength`, or `None` when no reference asset is supplied (the
/// lane then falls back to plain txt2img). `strength` is the full-range 0.0–1.0 reference-fidelity
/// slider (default 0.5); the worker does NOT clamp beyond `[0, 1]` — the usable band is model-specific
/// (A0/sc-8589 mapped Krea Turbo's sweet spot at ~0.35–0.65, but that is guidance, not a hard clamp).
/// The single `Conditioning::Reference` this produces is routed by the engine to
/// `generate_turbo_img2img` (sc-10135), whose `preprocess_init_image` LANCZOS-resizes the reference to
/// the output W×H — so, like Z-Image's [`resolve_identity_init`], the reference is fed raw (the
/// `edit_image`-only [`should_fit_edit_source`] crop/pad-fit never applies to Krea's t2i-only surface).
///
/// Available to the candle lane too (sc-10134): the candle `generate_candle_stream` calls this to resolve
/// the Krea 2 Turbo img2img init off-Mac, feeding the same `(image, strength)` into `generate_one`'s
/// `reference` → `Conditioning::Reference` → the engine's `render_img2img`. (The broader `ui.img2img`
/// candle roll-out for SD3.5 / Z-Image / Boogu / Ideogram is sc-10265.)
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_img2img_init_generic(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    let Some(asset_id) = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    let strength = advanced::f32_clamped(&request.advanced, "strength", 0.5, 0.0..=1.0);
    Ok(Some((image, strength)))
}

/// Whether the model opts into plain-t2i img2img (reference-guided latent-init) via the catalog —
/// the SAME `ui.img2img` manifest flag the web reads to show the "Image reference" tile (epic 8588
/// A4, sc-10195/sc-10189). Manifest-flag-driven rather than an ever-growing model-string match, so a
/// new text-only model gains img2img by flipping its manifest flag + landing its mlx-gen entrypoint —
/// no worker change. Mirrors the existing `uses_standard_tier_layout`/`is_dense_te_tier` pattern.
///
/// This gates the GENERIC img2img arm in [`resolve_generic_lane_conditioning`], which sits AFTER the
/// model-specific reference arms (z-image identity-init, FLUX IP-Adapter, Kolors, Ideogram edit) so
/// those bespoke surfaces keep precedence; the generic arm then catches Krea + SD3.5 + any future
/// `ui.img2img` model uniformly.
///
/// Available to the candle lane too (sc-10134): `generate_candle_stream` gates its Krea 2 Turbo img2img
/// resolve on this same manifest flag off-Mac. (Today the candle router only lets `krea_2_turbo` reach the
/// candle lane with a reference; the other `ui.img2img` families follow in sc-10265.)
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn model_supports_img2img(request: &ImageRequest) -> bool {
    request
        .model_manifest_entry
        .get("ui")
        .and_then(|ui| ui.get("img2img"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Resolve the Krea "text style" tap-reweight gain (sc-11878; gate fixed in sc-12008). Set ONLY when
/// the model's manifest declares the `ui.textStyleGain` slider — Krea/Qwen-Image-family only, so every
/// other family self-gates to `None`. The user value comes from `advanced.textStyleGain`, clamped to
/// the GPU-validated `[0.25, 1.75]`; the manifest slider object is not a scalar, so the `1.0` default
/// (a byte-exact engine no-op) applies when the user leaves it at default.
///
/// NOTE (sc-12008): `model_manifest_entry` is the FULL model entry (`resolve_model_manifest_entry`),
/// so the slider lives at `.ui.textStyleGain`, NOT the top level. Reading `.get("textStyleGain")`
/// directly is always `None` and silently disables the feature — the original end-to-end bug. This is
/// the single seam for both the MLX (`generate_stream`) and candle (`generate_candle_stream`) lanes so
/// the two can't drift apart again.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_text_style_gain(request: &ImageRequest) -> Option<f32> {
    request
        .model_manifest_entry
        .get("ui")
        .and_then(|ui| ui.get("textStyleGain"))
        .is_some()
        .then(|| advanced::f32_clamped(&request.advanced, "textStyleGain", 1.0, 0.25..=1.75))
}

/// Whether a Z-Image t2i request should take the generic `ui.img2img` reference-guided latent-init
/// rather than the Character Studio identity-init (sc-3619). Z-Image already owns a bespoke reference
/// arm in [`resolve_generic_lane_conditioning`] (keyed on `referenceStrength`), so it never reaches the
/// generic img2img arm below — this predicate re-introduces the generic path INSIDE that arm.
///
/// Identity-init keeps precedence (it also drives face-likeness scoring), so the generic img2img fires
/// ONLY when identity-init doesn't engage (no `referenceStrength`) yet the model opts into `ui.img2img`
/// and a reference is present — i.e. the Image Studio "Image reference" tile (`referenceAssetId` +
/// `advanced.strength`). The two surfaces are mutually exclusive by mode (character_image vs
/// text_to_image); encoding the precedence purely keeps it unit-testable without image I/O (epic 8588
/// A4.5, sc-10193). Base `z_image` is NOT in the z-image arm, so it reaches the generic arm directly and
/// never consults this predicate.
#[cfg(target_os = "macos")]
fn zimage_uses_generic_img2img(request: &ImageRequest, has_reference: bool) -> bool {
    identity_strength(request).is_none() && has_reference && model_supports_img2img(request)
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
        // Character Studio identity-init (referenceStrength, sc-3619) is the primary reference surface
        // and keeps precedence — it also drives face-likeness scoring. When it doesn't engage (the Image
        // Studio "Image reference" tile sends referenceAssetId + advanced.strength with NO
        // referenceStrength), the generic `ui.img2img` reference-guided latent-init (epic 8588 A4.5,
        // sc-10193) takes over so the slider actually reaches the engine. The two surfaces are mutually
        // exclusive by mode (character_image vs text_to_image), so this never double-drives; both produce
        // the same single `Conditioning::Reference`.
        let init = if request.mode == "edit_image" {
            resolve_zimage_edit_init(request, settings, project_path)?
        } else if zimage_uses_generic_img2img(request, has_reference) {
            resolve_img2img_init_generic(request, settings, project_path)?
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
    } else if model_supports_img2img(request) && has_reference && request.mode != "edit_image" {
        // Generic plain-t2i img2img latent-init for any `ui.img2img` model (epic 8588 A4, sc-10189):
        // a `referenceAssetId` + `advanced.strength` seeds the denoise from the VAE-encoded reference,
        // which the engine routes to that model's img2img entrypoint via the single
        // `Conditioning::Reference`. Krea 2 Turbo (sc-8591 #666) + SD3.5 large/turbo/medium (sc-10189
        // #667) opt in today; a new text-only model joins by flipping `ui.img2img` + landing its
        // mlx-gen entrypoint. Sits after the model-specific reference arms (z-image/flux/kolors/ideogram)
        // so their bespoke surfaces keep precedence. Candle parity per model is a deferred follow-up.
        Ok(LaneConditioning {
            identity_init: resolve_img2img_init_generic(request, settings, project_path)?,
            ..Default::default()
        })
    } else {
        // Boogu instruction edit resolves its (1..5) references separately into `boogu_refs` below —
        // it uses the `MultiReference`-capable path, not the single `identity_init` reference.
        Ok(LaneConditioning::default())
    }
}

/// MLX per-tier capability fit for the downtier chooser (sc-10733): fold the MLX residency decision
/// (resident-fits / staged-fits / won't-fit-even-staged) for a candidate tier's weights dir down to
/// [`TierFit`]. `Resident`/`Sequential` ⇒ `Fits`; `Reject` ⇒ `TooBig` (carrying the resident need + the
/// machine budget for the message). Uses the SAME `mlx_fit_gate` budget + footprint math the cold-load
/// `apply_residency_policy` runs, so the seam's downtier and the cache's admission never disagree.
#[cfg(target_os = "macos")]
fn mlx_tier_fit(engine_id: &str, weights_dir: &Path) -> TierFit {
    match crate::mlx_fit_gate::residency_for_dir(engine_id, weights_dir) {
        crate::mlx_fit_gate::ResidencyOutcome::Resident
        | crate::mlx_fit_gate::ResidencyOutcome::Sequential => TierFit::Fits,
        crate::mlx_fit_gate::ResidencyOutcome::Reject {
            needed_gb,
            available_gb,
            ..
        } => TierFit::TooBig {
            needed_gb,
            available_gb,
        },
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
    // Hoisted (also used at the load-spec seam below): the registered engine id, the sc-10733
    // capability-downtier fit key + `apply_residency_policy` sequential-capability key.
    let engine_id = model.engine_id();
    // sc-6568: a bf16 opt-in for Boogu fetches the full-precision `<variant>-bf16/` subfolder on
    // demand (the catalog ships only the Q8 default) before snapshot resolution. No-op for every
    // other model / the default Q8 path. sc-9607: the same on-demand pattern for Ideogram's `q8/`
    // tier (the catalog ships only q4) — was a documented follow-up, now wired on both lanes.
    ensure_boogu_tier_present(api, settings, job, request).await?;
    ensure_ideogram_tier_present(api, settings, job, request).await?;
    // `mut` for the sc-10733 capability downtier below: a DEFAULT job whose resolved tier won't fit this
    // machine's unified memory is re-pointed at the highest installed tier that does, BEFORE the quant
    // reconcile + spec build (so both the recorded precision and the load follow the downtiered tier).
    let mut weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("model weights not found".to_owned()))?;
    // sc-10733 capability downtier (MLX): for a DEFAULT job (no explicit per-(screen,model) pick), if the
    // resolved tier won't fit this machine's unified memory even under sequential residency, step DOWN to
    // the highest installed tier that does — floored at the per-model quality floor — rejecting only when
    // nothing >= floor fits. An explicit pick (`mlxQuantizeExplicit`) is HONORED: it skips the downtier
    // (the cold-load `apply_residency_policy` still reject-before-OOMs an unfittable explicit pick). The
    // `reconcile_resolved_tier_quant` below then corrects the recorded quant to the (possibly downtiered)
    // `weights_dir`, so telemetry never lies about the tier that actually ran.
    let explicit_pick = request
        .advanced
        .get("mlxQuantizeExplicit")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !explicit_pick {
        if let Some(default_tier) = tier_key_from_resolved_dir(&weights_dir) {
            let floor = min_quality_floor(request);
            let candidates: Vec<(&'static str, TierFit)> =
                downtier_candidate_tiers(request, settings, default_tier, floor)
                    .into_iter()
                    .filter_map(|cand| {
                        resolve_tier_dir(request, settings, cand)
                            .map(|dir| (cand, mlx_tier_fit(engine_id, &dir)))
                    })
                    .collect();
            match choose_downtier(default_tier, &candidates) {
                DowntierPick::Keep => {}
                DowntierPick::Downtier(chosen) => {
                    if let Some(dir) = resolve_tier_dir(request, settings, chosen) {
                        tracing::warn!(
                            model = %request.model,
                            from = %default_tier,
                            to = %chosen,
                            "MLX fit-gate: default tier won't fit unified memory — downtiering to the \
                             highest installed tier that does (capability clamp, sc-10733)"
                        );
                        weights_dir = dir;
                    }
                }
                DowntierPick::Reject {
                    tier,
                    needed_gb,
                    available_gb,
                } => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "{model} needs ~{needed} GB of unified memory even at the smallest installed \
                         tier it can run ({tier}) but this machine has ~{available} GB. Lower the output \
                         resolution or run on a Mac with more memory.",
                        model = request.model,
                        needed = needed_gb.round() as i64,
                        available = available_gb.round() as i64,
                    )));
                }
            }
        }
    }
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
        // `weights_dir` is the resolved tier subdir (sc-11042). NVFP4 is unreachable on this lane
        // regardless (`nvfp4_host_eligible()` is hard-`false` on macOS — Metal has no FP4 hardware), so
        // this is the same `(quant, bits)` it has always produced; passing the dir keeps the resolver's
        // one contract — the tier is read off what resolved — uniform across both lanes.
        resolve_quant(request, Some(&weights_dir))
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
    // Krea "text style" tap-reweight gain — see `resolve_text_style_gain` (sc-11878, gate fixed sc-12008).
    let text_style_gain = resolve_text_style_gain(request);
    let mut spec = load_spec(weights_dir, quant, adapters, flux_ip_dir);
    if let Some(pid) = pid_weights {
        spec = spec.with_pid(pid.checkpoint, pid.gemma);
    }
    // Named model components (epic 13657, sc-13682): stage a provider's caller-staged components (SDXL's
    // `tokenizer_clip_l` / `tokenizer_clip_bigg` / `vae_fp16_fix`) via the generic seam. Keyed on the
    // resolved `engine_id` (the DESCRIPTOR id) rather than `request.model`, so a finetune sibling that
    // shares one engine under a distinct catalog id resolves the same descriptor (media_descriptor matches
    // on descriptor.id). Inert on macOS: the MLX SDXL turnkey is self-contained (no `required_components`).
    spec = attach_required_components(spec, engine_id, &request.model_manifest_entry, settings)?;

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
                        text_style_gain,
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
            // Illustrious-XL (epic 10609): vanilla-SDXL anime finetunes on the shared candle `sdxl`
            // engine. Their turnkeys are tiered, so the dense lanes resolve `bf16/` (dense_tier_subdir).
            | "illustrious_xl_v1"
            | "illustrious_xl_v2"
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
            | "sensenova_u1_8b_infographic_v3"
            | "sensenova_u1_8b_fast"
            | "sensenova_u1_8b_infographic_v2_fast"
            | "sensenova_u1_8b_infographic_v3_fast"
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
            // Krea 2 Raw (sc-9994, epic 9992): the UNDISTILLED full-CFG base sibling of Turbo, now a
            // registered candle txt2img generator (`candle-gen-krea` `render_base` — two DiT forwards/step,
            // 52 steps, resolution-dynamic mu). Arch-identical to Turbo (only the DiT weights differ), so it
            // rides the SAME generic candle lane: `generate_candle_stream` resolves its repo/tier/steps/
            // guidance/negative from the descriptor + the shared `krea_model_subdir` turnkey resolver, and
            // the img2img `Reference` init (sc-10226 `render_base_img2img`). It is ALSO the LoRA-training
            // base (Path 1). This entry was MISSING when epic 9992 landed the generator — the scheduler
            // routes `krea_2_raw` to candle (`IMAGE_MODEL_CAPS` candle_routed=true, mirroring Turbo), but
            // `resolve_candle_image_route`'s generic arm gates on THIS list, so plain Raw t2i fell through to
            // `None` and the worker emitted a `procedural_preview` STUB gradient (`realModelInference:false`)
            // instead of running the real model — the classic router/worker list skew. Turbo was unaffected
            // (it was in the list); the Raw edit lane (`krea_edit_candle_available`, checked first) hid the
            // gap for edit jobs, and there was no `krea_2_raw` t2i routing test to catch it.
            | "krea_2_raw"
            // Stable Diffusion 3.5 (sc-7880, epic 7982): Large / Large Turbo / Medium all ride the generic
            // candle txt2img lane (the `candle-gen-sd3` provider). `generate_candle_stream` resolves Q4/Q8
            // from the descriptor + manifest (`mlx.quantize` 8) — the dense MMDiT folds to Q4_0/Q8_0 at
            // load (sc-7879). Pure txt2img; no edit/reference/control/LoRA candle path (those defer to
            // torch via `image_request_candle_eligible`).
            | "sd3_5_large"
            | "sd3_5_large_turbo"
            | "sd3_5_medium"
            // SANA 1600M (sc-11780, epic 8485): NVIDIA's 1.6B Linear-DiT true-CFG txt2img rides the
            // generic candle lane (the `candle-gen-sana` provider, candle-gen #495 — the off-Mac sibling
            // of `mlx-gen-sana`). `generate_candle_stream` resolves the whole `Efficient-Large-Model/
            // Sana_1600M_1024px_diffusers` diffusers snapshot (transformer/ + vae/ + text_encoder/) via the
            // candle-sana branch in `resolve_weights_dir` — NOT the MLX-packed turnkey, NOT a tier subdir.
            // Pure txt2img (20 steps / guidance 4.5 + negative prompt); no edit/reference/control/LoRA/quant
            // candle path (those defer to torch via `image_request_candle_eligible`).
            | "sana_1600m"
            // SANA-Sprint 1.6B (sc-11781, epic 8485): NVIDIA's CFG-free few-step distill of SANA rides the
            // generic candle lane too (the `candle-gen-sana` Sprint pipeline, candle-gen #498 — the off-Mac
            // sibling of `mlx-gen-sana`'s Sprint id). `generate_candle_stream` resolves the whole
            // `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` diffusers snapshot (transformer/ +
            // vae/ + text_encoder/) via the candle-sana branch in `resolve_weights_dir` — NOT the MLX-packed
            // turnkey, NOT a tier subdir. Pure txt2img, CFG-free 1–4 step SCM/TrigFlow (guidance embedded,
            // no negative-prompt second pass); no edit/reference/control/LoRA/quant candle path (the Sprint
            // adapter rejects those — they defer to torch via `image_request_candle_eligible`).
            | "sana_sprint_1600m"
            // Anima 2B base / aesthetic / turbo (sc-10676, epic 10512): the candle port (sc-10525,
            // GPU-validated sc-10625) rides the generic candle txt2img lane off-Mac. `generate_candle_\
            // stream` dense-loads bf16 from the raw `circlestone-labs/Anima` split_files/ tree (no
            // convert-at-install off-Mac) and FORCES the load Quant to None — the descriptor advertises
            // Q4/Q8 but there is no packed tier off-Mac, and the loader rejects a quant against the dense
            // DiT. Inference LoRA/LoKr folds onto the dense DiT (`supports_lora`/`supports_lokr`); quant +
            // LoRA together stays rejected (sc-10578). Pure txt2img (no edit/reference/control shapes).
            | "anima_base"
            | "anima_aesthetic"
            | "anima_turbo"
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
        | "sensenova_u1_8b_infographic_v3"
        | "sensenova_u1_8b_fast"
        | "sensenova_u1_8b_infographic_v2_fast"
        | "sensenova_u1_8b_infographic_v3_fast" => "candle_sensenova",
        "ideogram_4" | "ideogram_4_turbo" => "candle_ideogram",
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit" => "candle_boogu",
        "krea_2_turbo" | "krea_2_raw" => "candle_krea",
        // Stable Diffusion 3.5 (sc-7880): Large / Large Turbo / Medium share the candle SD3.5 engine.
        "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium" => "candle_sd3",
        // SANA 1600M + SANA-Sprint (sc-11780 / sc-11781): both share the candle SANA engine (the off-Mac
        // sibling of the `mlx_sana` label).
        "sana_1600m" | "sana_sprint_1600m" => "candle_sana",
        // Anima 2B (sc-10676): base / aesthetic / turbo share the candle Anima engine (the off-Mac
        // sibling of the `mlx_anima` label).
        "anima_base" | "anima_aesthetic" | "anima_turbo" => "candle_anima",
        // sdxl / realvisxl share the candle "sdxl" engine.
        _ => CANDLE_ADAPTER,
    }
}

/// The actionable tail for a candle VRAM-fit reject (sc-12090). Suggests only the tiers that are
/// actually INSTALLED and smaller than the rejected one (`installed_smaller`, descending fidelity),
/// else states that none is installed. Never names the rejected tier itself, and never points at the
/// quant picker — which is hidden by design when ≤1 tier is installed, exactly the case that produced
/// the misleading "Pick a lower tier (Q4/Q8)" text on a single-tier install.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn vram_reject_tail(installed_smaller: &[&str]) -> String {
    if installed_smaller.is_empty() {
        return "No smaller tier is installed — lower the output resolution or run on a GPU with more \
                VRAM."
            .to_owned();
    }
    format!(
        "Select a smaller installed tier ({}), lower the output resolution, or run on a GPU with more \
         VRAM.",
        installed_smaller
            .iter()
            .map(|tier| tier.to_uppercase())
            .collect::<Vec<_>>()
            .join(" / "),
    )
}

/// Candle per-tier capability fit for the downtier chooser (sc-10733): fold the full candle fit decision
/// (predicted resident peak vs the live budget, plus the sequential-residency second stage where the
/// provider stages components) down to [`TierFit`]. `Fits` = runs resident OR sequentially; `TooBig` =
/// won't run even one-component-at-a-time. `Unknown` (no budget / unmeasured tier) counts as `Fits` — the
/// gate never blocks without a signal.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_tier_fit(
    manifest_entry: &JsonObject,
    tier: &'static str,
    budget: Option<crate::vram_gate::VramBudget>,
    sequential_capable: bool,
) -> TierFit {
    let needed = crate::vram_gate::predicted_peak_gb(manifest_entry, tier);
    match crate::vram_gate::resolve_offload(
        crate::vram_gate::fit_decision(needed, budget),
        sequential_capable,
    ) {
        crate::vram_gate::FitDecision::Fits | crate::vram_gate::FitDecision::Unknown => TierFit::Fits,
        crate::vram_gate::FitDecision::Offload {
            available_gb, ..
        } => {
            // Resident won't fit but the provider stages — fits only if the MEASURED sequential peak
            // fits (unmeasured ⇒ best-effort run, so `sequential_overflow_gb` yields None ⇒ Fits).
            let seq_needed =
                crate::vram_gate::predicted_sequential_peak_gb(manifest_entry, tier);
            match crate::vram_gate::sequential_overflow_gb(seq_needed, budget) {
                Some(seq_gb) => TierFit::TooBig {
                    needed_gb: seq_gb,
                    available_gb,
                },
                None => TierFit::Fits,
            }
        }
        crate::vram_gate::FitDecision::TooBig {
            needed_gb,
            available_gb,
        } => TierFit::TooBig {
            needed_gb,
            available_gb,
        },
    }
}

/// The `(load Quant, recipe bit count)` a resolved generation `tier` loads at (sc-10733) — used to
/// correct the recorded quant + telemetry after a capability downtier rewrites the tier, so a
/// downtiered job records the precision it ACTUALLY ran (parity with the MLX
/// [`reconcile_resolved_tier_quant`]), not the requested one. On candle the load quant is advisory (the
/// packed tier is auto-detected on disk), so this is safe to set to the downtiered tier.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn tier_to_quant(tier: &str) -> (Option<Quant>, Option<i64>) {
    match tier {
        "bf16" => (None, None),
        "q4" => (Some(Quant::Q4), Some(4)),
        _ => (Some(Quant::Q8), Some(8)),
    }
}

/// sc-13960 — the evict-then-reclaim gate driver the bespoke (non-cache-evicting) candle image lanes
/// (`qwen_edit_candle` / `krea_control_candle`) share. Runs the lane's PURE `gate(budget)` against a
/// **two-pass** budget and returns the plan to act on plus the budget it was resolved against:
///
///  1. **Raw pass.** Gate against raw live free VRAM (no reclaim). Correct as-is: a resident txt2img
///     generator stays live and co-resident with the bespoke load, so its cudarc pool pages are NOT
///     free — crediting them would over-admit an OOM (sc-13588's documented reason these lanes budget
///     raw). If nothing is reclaimable (cold pool) — or the plan the raw budget yields already stands
///     after folding the pool — return it here, and **the warm txt2img generator cache is preserved**
///     (the deliberate tradeoff: we do NOT evict on every render, only when it changes the outcome).
///  2. **Reclaim pass.** Otherwise gate again against `free + reclaimable_pool` and, when that yields a
///     BETTER plan (`reclaim_improves` — the budget only ever grows, so a changed plan is always a
///     higher residency / an admit-instead-of-reject), EVICT the single-slot generator cache so the
///     resident generator's pages become genuinely free, making the reclaim credit honest, then act on
///     the reclaimed plan. This is the missing half of sc-11023 for the bespoke lanes: a second
///     edit/control render in the same worker no longer sees the first render's dropped-but-pooled
///     pages as unavailable.
///
/// `reclaim_improves(&raw, &reclaimed)` returns whether the reclaimed plan is worth an evict — `false`
/// when it is the same action as raw (including two rejects that differ only in their reported
/// free-VRAM number, which must NOT trigger a pointless evict). Mirrors the candle video comfyui lane's
/// `generator_cache::with_uncached_generator` precedent (evict, then reclaim) — see that function and
/// `video_jobs::candle_video_vram_budget`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn gate_with_evict_reclaim<D>(
    gpu_id: &str,
    raw_budget: Option<crate::vram_gate::VramBudget>,
    gate: impl Fn(Option<crate::vram_gate::VramBudget>) -> D,
    reclaim_improves: impl Fn(&D, &D) -> bool,
) -> WorkerResult<(D, Option<crate::vram_gate::VramBudget>)> {
    let raw = gate(raw_budget);
    let reclaimable = crate::vram_gate::reclaimable_pool_gb(gpu_id);
    // Cold pool: nothing this process pooled to reclaim, hence nothing resident to evict — gate exactly
    // as the pre-sc-13960 lanes did.
    if reclaimable <= 0.0 {
        return Ok((raw, raw_budget));
    }
    let reclaimed_budget =
        raw_budget.map(|budget| crate::vram_gate::with_reclaimable(budget, reclaimable));
    let reclaimed = gate(reclaimed_budget);
    if !reclaim_improves(&raw, &reclaimed) {
        // Reclaim would not change the plan — keep the warm txt2img cache rather than evict for nothing.
        return Ok((raw, raw_budget));
    }
    let evicted = crate::generator_cache::evict_cached_generator().await?;
    tracing::info!(
        gpu_id,
        evicted,
        reclaimable_gb = reclaimable,
        "candle bespoke-lane VRAM gate: evicted the resident generator to reclaim its cudarc pool so \
         this render admits at a higher residency (sc-13960)"
    );
    Ok((reclaimed, reclaimed_budget))
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
    // `mut` for the sc-10733 capability downtier below: a DEFAULT job whose resolved tier won't fit the
    // live VRAM budget is re-pointed at the highest installed tier that does, before the spec is built.
    let mut weights_dir = if let Some((base_dir, _)) = convrot.as_ref() {
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
        // So do the Illustrious-XL finetunes (epic 10609).
        "sdxl"
        | "realvisxl"
        | "realvisxl_lightning"
        | "illustrious_xl_v1"
        | "illustrious_xl_v2" => runtime_cuda::providers::sdxl::set_flash_attn(flash_attn),
        // Base z_image (sc-8679) shares the candle z-image accel-attention toggle with Turbo.
        "z_image_turbo" | "z_image" => runtime_cuda::providers::z_image::set_accel_attn(flash_attn),
        _ => {}
    }

    // Descriptor-gated quant + adapters (sc-5126). Lens advertises Q4/Q8 (Q8 default) + LoRA/LoKr, so
    // it resolves them like the MLX path; the sc-3675/sc-5096 families advertise neither and skip both
    // (dense bf16/fp16, no adapters) — preserving their shipped behavior. The router only lets a quant
    // request / LoRA reach this worker for a family that supports it (`image_request_candle_eligible`).
    // `mut` so the sc-10733 downtier can correct the recorded precision to the tier it lands on.
    let (mut quant, mut quant_bits) = if convrot.is_some() {
        // INT8-ConvRot (sc-9300): the int8 DiT replaces the dense transformer wholesale — a bits-based
        // load-time `Quant` is meaningless (and the candle-gen krea engine rejects a quant overlay on
        // the ConvRot path). Force dense-None; the recipe records no `mlxQuantize` bits for this tier.
        (None, None)
    } else if is_anima_model(&request.model) {
        // Anima off-Mac (sc-10676): the descriptor advertises Q4/Q8, but there is NO packed tier off-Mac
        // — the `anima_quant` converter is macOS-only and the NC license bars publishing one, and the
        // candle loader only CONSUMES an MLX-packed tier (it hard-rejects a quant request against the
        // dense split_files/ DiT: "the DiT checkpoint is DENSE … load the dense tier"). So force dense
        // bf16 here, IGNORING the manifest `mlx.quantize: 4` default that `resolve_quant` would otherwise
        // apply — else every plain candle Anima job would fail the loader's packed-detect. The router
        // keeps `candle_quant = false`, so a deliberate `advanced.mlxQuantize > 0` never reaches this lane
        // (it defers rather than silently running dense); this arm handles the default-quant case the
        // router doesn't strip. A dense DiT + LoRA/LoKr still folds (Quant None ⇒ no sc-10578 reject).
        (None, None)
    } else if model.supports_quant() {
        // `weights_dir` is the tier subdir this lane is about to load (resolved above), so the NVFP4
        // tier is picked only when the `nvfp4/` dir is what actually resolved (sc-11042) — never FP4
        // against a q8 fallback.
        resolve_quant(request, Some(&weights_dir))
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
    // Generic img2img (reference-guided latent-init, sc-10134, epic 8588): a `ui.img2img` model in a
    // NON-edit mode carrying a `referenceAssetId` resolves to the img2img init `(image, advanced.strength)`,
    // threaded to `generate_one` as the single `Conditioning::Reference` the candle engine routes to its
    // img2img entrypoint (VAE-encode the reference → blend at `sigmas[init_time_step]` → denoise; CFG-free
    // for distilled families, two-forward CFG for the base ones like Krea Raw `render_base_img2img`,
    // sc-10226). Model-agnostic here — the candle router gates which ids reach this lane with a reference
    // (`krea_2_turbo`/`krea_2_raw`, SD3.5, Z-Image, Boogu, Ideogram all wired). Disjoint from the Ideogram
    // `edit_reference` (edit_image vs text_to_image) and Boogu's
    // `multi_references` (guarded here so a future overlap never double-drives the single `reference` slot).
    let img2img_reference = if edit_reference.is_none()
        && boogu_refs.is_empty()
        && request.mode != "edit_image"
        && model_supports_img2img(request)
    {
        resolve_img2img_init_generic(request, settings, project_path)?
    } else {
        None
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
    // `mut`: rebuilt with the corrected `quant_bits` if the sc-10733 downtier lands on a lower tier.
    let mut raw_settings = mlx_raw_settings(request, &repo, steps, quant_bits, guidance.or(true_cfg));
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
    // Krea "text style" tap-reweight gain — see `resolve_text_style_gain` (sc-11878, gate fixed sc-12008).
    let text_style_gain = resolve_text_style_gain(request);
    // VRAM fit-gate (epic 10765, sc-10766 Phase 0 + sc-10821 Phase 1b + sc-10856): when the selected
    // tier's predicted resident peak won't fit the card, either RUN SEQUENTIALLY (a provider that
    // supports sequential component residency — the candle FLUX lane, sc-10769 — drops the text encoders
    // before the DiT so peak = DiT+VAE, not TE+DiT+VAE) or, for a family that has not wired it, reject-
    // before-OOM with an actionable message. sc-10856 adds a second stage on the sequential path: when
    // the tier's MEASURED sequential peak (`candle.sequentialPeakGb`) is known and STILL won't fit, reject
    // instead of running into a reactive OOM. Honors `SCENEWORKS_CUDA_VRAM_CAP_GB` to emulate a small
    // card. Unmeasured models (no `candle` block) and non-NVIDIA hosts yield `Unknown` → never block.
    let budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    // sc-11023: the single-slot generator cache evicts its current occupant BEFORE the incoming load,
    // and cudarc's caching allocator reuses those freed pages in-process — nvidia-smi `free` never rises
    // after an in-process evict (the epic 10765 caching-allocator note). So the VRAM this process already
    // holds is RECLAIMABLE by the incoming load; budget against `free + reclaimable` (capped to total),
    // else a warm/swap re-gate falsely rejects a load that will actually fit (the "even with sequential
    // residency" 2nd-run reject a resident bf16 tier hits on the next generation).
    let budget = budget.map(|budget| {
        crate::vram_gate::with_reclaimable(
            budget,
            crate::vram_gate::reclaimable_pool_gb(&settings.gpu_id),
        )
    });
    // sc-12090: budget + name the tier the disk-probing resolver ACTUALLY landed on (`weights_dir`),
    // not a manifest re-derivation that ignores what's installed. `requested_tier_key` re-derived from
    // `mlx.quantize` with no disk check, so a q4-only install was budgeted (and rejected) against a q8
    // the user never downloaded. Read the resolved on-disk tier instead — one value, both named and
    // budgeted. ConvRot loads an int8 DiT over the bf16 base surface (its footprint is neither the bf16
    // nor the q8 tier), and a `modelPath`/flat root has no recognizable tier basename — fall back to the
    // manifest key there, preserving today's behavior.
    //
    // sc-11042: the NVFP4 tier composes with the sc-12090 disk probe rather than bypassing it.
    // `tier_key_from_resolved_dir` only recognizes the bits-based basenames (bf16/q8/q4), so a resolved
    // `nvfp4/` dir yields `None` and falls through to `requested_tier_key`, whose `nvfp4` short-circuit
    // names the tier by IDENTITY (never by bits — `Quant::Nvfp4.bits()` is 4, which would alias q4).
    // `nvfp4_selected` reads that same resolved `weights_dir`, so a `quantTier: "nvfp4"` label that fell
    // back to another tier's dir is sized/named as the tier that will actually load, not as nvfp4.
    let nvfp4_sel = nvfp4_selected(request, nvfp4_host_eligible(), Some(&weights_dir));
    // sc-12425: a resolved ConvRot load is named by its tier IDENTITY (see [`gate_tier_key`]) — it used
    // to be handed to the bits-derived `requested_tier_key`, which aliased it to q8 and under-gated it.
    // The comment above already knew "its footprint is neither the bf16 nor the q8 tier"; now the gate
    // acts on it. Extracted so that mapping has a unit test; this fn cannot be exercised from one.
    let mut tier = gate_tier_key(
        convrot.is_some(),
        &weights_dir,
        &request.advanced,
        &request.model_manifest_entry,
        nvfp4_sel,
    );
    // sc-12130: derive Candle residency support from the provider's weights-free descriptor instead of
    // maintaining a second engine-id allowlist in the worker. The capability bit is the provider's
    // contract that every request shape accepted by this id honors Sequential. Bespoke edit/control,
    // ComfyUI, and strict-control routes are diverted by `resolve_candle_image_route` before this gate;
    // generic txt2img/img2img reaches this point and uses the same registry-derived signal for both the
    // capability downtier and the resident/sequential decision.
    let sequential_capable = crate::mlx_fit_gate::engine_supports_sequential(engine_id);
    // sc-10733 capability downtier: for a DEFAULT job (no explicit per-(screen,model) pick, and not the
    // bespoke ConvRot tier), if the resolved tier won't fit the live budget, step DOWN to the highest
    // installed tier that does — floored at the per-model quality floor — rejecting only when nothing
    // >= floor fits. An explicit pick (`mlxQuantizeExplicit`) is HONORED: it skips the downtier and runs
    // the plain gate below (fits → run; too big → reject-before-OOM), never silently downtiered (#7).
    //
    // sc-11042 (epic 11037 SC#5): a selected NVFP4 tier is never downtiered either, and does not rely on
    // `mlxQuantizeExplicit` to escape — the web omits that flag for nvfp4 (it rides inside the
    // `tierQuantize(quantTier) !== null` bits branch, and nvfp4 has no honest `mlxQuantize` integer).
    // Instead NVFP4 is unrankable ON PURPOSE: `tier_quality_rank("nvfp4")` is 0 because nvfp4 is a
    // distinct numeric regime, not a rung on the bf16/q8/q4 fidelity ladder, so
    // `downtier_candidate_tiers` yields NO candidates in `[floor, nvfp4]` and `choose_downtier` returns
    // `Keep`. Downtiering NVFP4 to q4/q8 would silently swap the numerics of an explicitly-picked tier —
    // exactly the creative-choice violation SC#5 forbids. Pinned by
    // `nvfp4_tier_is_never_downtiered_by_the_capability_clamp`.
    let explicit_pick = request
        .advanced
        .get("mlxQuantizeExplicit")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if convrot.is_none() && !explicit_pick {
        let floor = min_quality_floor(request);
        let candidates: Vec<(&'static str, TierFit)> =
            downtier_candidate_tiers(request, settings, tier, floor)
                .into_iter()
                .map(|candidate| {
                    (
                        candidate,
                        candle_tier_fit(
                            &request.model_manifest_entry,
                            candidate,
                            budget,
                            sequential_capable,
                        ),
                    )
                })
                .collect();
        match choose_downtier(tier, &candidates) {
            DowntierPick::Keep => {}
            DowntierPick::Downtier(chosen) => {
                // The chosen tier came from `downtier_candidate_tiers`, so re-resolving its dir yields
                // Some; fall through defensively rather than unwrap-panic if it somehow doesn't.
                if let Some(dir) = resolve_tier_dir(request, settings, chosen) {
                    tracing::warn!(
                        model = %request.model,
                        from = %tier,
                        to = %chosen,
                        "candle VRAM fit-gate: default tier won't fit — downtiering to the highest \
                         installed tier that does (capability clamp, sc-10733)"
                    );
                    weights_dir = dir;
                    tier = chosen;
                    // Record the precision that ACTUALLY runs (parity with the MLX reconcile) so a
                    // downtiered job's sidecar/telemetry never lies. Candle load quant is advisory (the
                    // packed tier is auto-detected on disk), so this rewrite is safe.
                    let (downtiered_quant, downtiered_bits) = tier_to_quant(chosen);
                    quant = downtiered_quant;
                    quant_bits = downtiered_bits;
                    raw_settings =
                        mlx_raw_settings(request, &repo, steps, quant_bits, guidance.or(true_cfg));
                }
            }
            DowntierPick::Reject {
                tier: smallest,
                needed_gb,
                available_gb,
            } => {
                return Err(WorkerError::InvalidPayload(format!(
                    "{model} needs ~{needed} GB of VRAM even at the smallest installed tier it can run \
                     ({smallest}) but GPU {gpu} has ~{available} GB available. Lower the output \
                     resolution or run on a card with more VRAM.",
                    model = request.model,
                    needed = needed_gb.round() as i64,
                    available = available_gb.round() as i64,
                    gpu = settings.gpu_id,
                )));
            }
        }
    }
    let needed = crate::vram_gate::predicted_peak_gb(&request.model_manifest_entry, tier);
    // sc-12090 AC#4/#5: the reject suggestions name only tiers actually INSTALLED and lower-fidelity than
    // the one being rejected — never the rejected tier, never the picker (hidden when ≤1 tier installed).
    // Reached only on the explicit-pick / ConvRot reject below (the downtier path already rejected above
    // when nothing smaller fits), where suggesting a smaller installed tier the user could pick is apt.
    let installed_smaller: Vec<&'static str> = installed_tier_keys(request, settings)
        .into_iter()
        .filter(|candidate| tier_quality_rank(candidate) < tier_quality_rank(tier))
        .collect();
    let use_sequential = {
        match crate::vram_gate::resolve_offload(
            crate::vram_gate::fit_decision(needed, budget),
            sequential_capable,
        ) {
            crate::vram_gate::FitDecision::Offload {
                needed_gb,
                available_gb,
            } => {
                // Second-stage gate (sc-10856): sequential residency was selected because the resident
                // peak won't fit. If this tier's MEASURED sequential peak is known and STILL exceeds the
                // budget, reject before load instead of running into a reactive OOM. Absent the number
                // (unmeasured tier) keep the best-effort run — the reactive OOM containment backstops it.
                let sequential_needed = crate::vram_gate::predicted_sequential_peak_gb(
                    &request.model_manifest_entry,
                    tier,
                );
                if let Some(seq_gb) =
                    crate::vram_gate::sequential_overflow_gb(sequential_needed, budget)
                {
                    return Err(WorkerError::InvalidPayload(format!(
                        "{model} at the {tier} tier needs ~{seq} GB of VRAM even with sequential \
                         component residency (loading one component at a time), but GPU {gpu} has \
                         ~{available} GB available. {tail}",
                        model = request.model,
                        seq = seq_gb.round() as i64,
                        available = available_gb.round() as i64,
                        gpu = settings.gpu_id,
                        tail = vram_reject_tail(&installed_smaller),
                    )));
                }
                tracing::info!(
                    model = %request.model,
                    needed_gb = needed_gb.round() as i64,
                    available_gb = available_gb.round() as i64,
                    "candle VRAM fit-gate: resident peak exceeds free VRAM — loading with sequential \
                     component residency (text encoders dropped before the DiT)"
                );
                true
            }
            crate::vram_gate::FitDecision::TooBig {
                needed_gb,
                available_gb,
            } => {
                return Err(WorkerError::InvalidPayload(format!(
                    "{model} at the {tier} tier needs ~{needed} GB of VRAM (with headroom) but GPU \
                     {gpu} has ~{available} GB available. {tail}",
                    model = request.model,
                    needed = needed_gb.round() as i64,
                    available = available_gb.round() as i64,
                    gpu = settings.gpu_id,
                    tail = vram_reject_tail(&installed_smaller),
                )));
            }
            _ => false,
        }
    };
    // sc-11023: record this admitted load's incurred peak as the reclaimable high-water for the NEXT
    // gate. Sequential residency peaks at the largest single component; a resident load at the whole-
    // model peak. cudarc's pool never returns pages to the driver, so the max we have ever loaded is
    // exactly what a later swap-in reclaims after the single-slot cache evicts this model. Reached only
    // when the gate ADMITTED the load (the reject arms `return` above), so we never record a peak we
    // didn't actually attempt to allocate.
    let incurred_peak = if use_sequential {
        crate::vram_gate::predicted_sequential_peak_gb(&request.model_manifest_entry, tier)
    } else {
        needed
    };
    if let Some(peak_gb) = incurred_peak {
        crate::vram_gate::note_loaded_peak(&settings.gpu_id, peak_gb);
    }
    let mut spec = load_spec(weights_dir, quant, adapters, None);
    if use_sequential {
        // Ask the provider (candle FLUX) to load→use→drop each component in phase order (sc-10821).
        spec = spec.with_offload_policy(gen_core::OffloadPolicy::Sequential);
    }
    if let Some(pid) = pid_weights {
        spec = spec.with_pid(pid.checkpoint, pid.gemma);
    }
    // Named model components (epic 13657, sc-13682): the candle twin of the mlx attach above — stages
    // SDXL's three caller-provided components on the Windows/CUDA candle `sdxl` engine. Keyed on the
    // resolved `engine_id` (the DESCRIPTOR id), NOT `request.model`, so the finetune siblings that share
    // the candle `sdxl` engine under distinct catalog ids — realvisxl / illustrious_xl_v1|v2 /
    // realvisxl_lightning — resolve the same descriptor and get the components staged. A no-op for every
    // engine that advertises no `required_components`.
    spec = attach_required_components(spec, engine_id, &request.model_manifest_entry, settings)?;
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
                        // Ideogram edit source (edit_image) OR the Krea 2 Turbo img2img init (sc-10134) —
                        // mutually exclusive by mode/family; whichever resolved seeds the single
                        // `Conditioning::Reference` slot.
                        edit_reference.as_ref().or(img2img_reference.as_ref()),
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
                        text_style_gain,
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
/// Best-effort human label for a LoRA entry in a request (epic 10402). Accepts a
/// bare string or an object with an id/name field.
fn lora_label(value: &Value) -> Option<String> {
    match value {
        Value::String(name) => Some(name.clone()),
        Value::Object(map) => map
            .get("id")
            .or_else(|| map.get("name"))
            .or_else(|| map.get("loraId"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

/// Assemble the EFFECTIVE-settings metrics block for an image job (epic 10402,
/// sc-10406). Reports the value the run actually used — not the sparse `advanced`
/// payload where defaults are omitted: sampler/scheduler default to "default"
/// (model-native), guidanceMethod to "cfg", `use_pid` to false, steps come from
/// the observed denoise total, guidance from the resolver, quant from
/// [`effective_quant_label`]. A default-settings run is fully populated, never
/// blank — which is what makes the comparison charts meaningful (sc-10409).
fn image_settings_metrics(
    request: &ImageRequest,
    effective_steps: Option<u32>,
    effective_guidance: Option<f32>,
    quant_label: Option<String>,
    quant_bits: Option<i64>,
    image_count: u32,
) -> GenerationMetrics {
    let adv = &request.advanced;
    let string_or = |key: &str, default: &str| -> Option<String> {
        Some(
            adv.get(key)
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_owned(),
        )
    };
    let number_field = |key: &str| -> Option<serde_json::Number> {
        adv.get(key)
            .and_then(|value| value.as_f64().or_else(|| value.as_str()?.trim().parse().ok()))
            .and_then(serde_json::Number::from_f64)
    };
    let loras: Vec<String> = request.loras.iter().filter_map(lora_label).collect();
    GenerationMetrics {
        model: (!request.model.is_empty()).then(|| request.model.clone()),
        quant_label,
        quant_bits: quant_bits.map(|bits| bits as u32),
        sampler: string_or("sampler", "default"),
        scheduler: string_or("scheduler", "default"),
        scheduler_shift: number_field("schedulerShift"),
        steps: effective_steps,
        image_count: Some(image_count),
        guidance_scale: effective_guidance
            .map(|scale| scale as f64)
            .and_then(serde_json::Number::from_f64),
        true_cfg_scale: number_field("trueCfgScale"),
        guidance_method: string_or("guidanceMethod", "cfg"),
        use_pid: Some(adv.get("usePid").and_then(Value::as_bool).unwrap_or(false)),
        pid_target: adv.get("pidTarget").and_then(Value::as_str).map(str::to_owned),
        width: Some(request.width),
        height: Some(request.height),
        seed: request.seed.or_else(|| request.seeds.first().copied()),
        loras: (!loras.is_empty()).then_some(loras),
        ..Default::default()
    }
}

/// The effective quant label + bit count for a request (epic 10402): the Krea
/// INT8-ConvRot tier, then dense-TE turnkey tiers (which `resolve_quant` reports
/// as bf16 to keep the dense TE full-precision), else `resolve_quant`.
///
/// **The NVFP4 arm matches the VARIANT, not the bit count (sc-11042, epic 11037 SC#5)** — the image
/// lane's instance of the same footgun `video_quant_label` carries, and it fails in the opposite
/// direction. This maps [`resolve_quant`]'s *bits* onto a label, and NVFP4's bits are deliberately
/// `None` (~4.5 EFFECTIVE bits/weight — `Some(4)` would alias it onto q4), so a bits-only match would
/// drop NVFP4 into the `_ => bf16` arm and stamp an NVFP4 render as **`"bf16"`**: a full-precision
/// label on a 4-bit render, the inverse mislabel but the same SC#5 violation. Matching the variant is
/// what makes both directions impossible. The tier's own arm is placed alongside int8-convrot's — both
/// are tiers with no honest integer width, and both report `None` bits for that reason.
///
/// **The label describes what RAN — so it is disk-aware, not just host-aware (sc-11042).** `tier_dir`
/// is the RESOLVED tier subdir, and NVFP4 is labelled only when that dir IS the `nvfp4/` one (the
/// [`nvfp4_selected`] third half). Host-awareness alone was NOT enough: `standard_tier_subdir`
/// independently falls back to q8 when the `nvfp4/` dir is absent — the shipping case on every model
/// today, since sc-11043 has not converted a tier yet — so an explicit pick on a Blackwell host
/// recorded `"nvfp4"` on a render whose weights were **q8**. Same SC#5 creative-choice aliasing this
/// tier exists to eliminate, merely displaced out of selection and into telemetry. Reading the label
/// off the resolver's own output is what makes the two agree by construction.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn effective_quant_label(
    request: &ImageRequest,
    tier_dir: Option<&Path>,
) -> (Option<String>, Option<i64>) {
    effective_quant_label_gated(request, nvfp4_host_eligible(), tier_dir)
}

/// [`effective_quant_label`] with the NVFP4 **host** gate passed in rather than probed (sc-11042).
/// Split out for testability, exactly like [`resolve_quant_gated`], which it delegates to. Production
/// has one caller.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn effective_quant_label_gated(
    request: &ImageRequest,
    nvfp4_host: bool,
    tier_dir: Option<&Path>,
) -> (Option<String>, Option<i64>) {
    if request
        .advanced
        .get("convRot")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return (Some("int8-convrot".to_owned()), None);
    }
    #[cfg(target_os = "macos")]
    if is_dense_te_tier(request) {
        return match dense_te_requested_tier_bits(request) {
            Some(8) => (Some("q8".to_owned()), Some(8)),
            Some(4) => (Some("q4".to_owned()), Some(4)),
            _ => (Some("bf16".to_owned()), None),
        };
    }
    match resolve_quant_gated(request, nvfp4_host, tier_dir) {
        // The distinct NVFP4 tier, matched on the VARIANT (see the note above): its bit count is
        // `None`, so a bits-only match would silently label it "bf16".
        (Some(Quant::Nvfp4), _) => (Some(NVFP4_TIER.to_owned()), None),
        (_, Some(8)) => (Some("q8".to_owned()), Some(8)),
        (_, Some(4)) => (Some("q4".to_owned()), Some(4)),
        _ => (Some("bf16".to_owned()), None),
    }
}

/// Resolve quant + guidance with the generation's own rules and assemble the
/// effective-settings metrics for an image job (epic 10402, sc-10406). A build
/// with neither the MLX nor candle backend reports quant/guidance as none.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn build_image_metrics(
    request: &ImageRequest,
    effective_steps: Option<u32>,
    image_count: u32,
    tier_dir: Option<&Path>,
) -> GenerationMetrics {
    let (quant_label, quant_bits) = effective_quant_label(request, tier_dir);
    let guidance = mlx_model(&request.model).and_then(|model| resolve_guidance(request, &model));
    image_settings_metrics(
        request,
        effective_steps,
        guidance,
        quant_label,
        quant_bits,
        image_count,
    )
}
#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
fn build_image_metrics(
    request: &ImageRequest,
    effective_steps: Option<u32>,
    image_count: u32,
    _tier_dir: Option<&Path>,
) -> GenerationMetrics {
    image_settings_metrics(request, effective_steps, None, None, None, image_count)
}

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
    // Per-phase wall-clock (epic 10402, sc-10405): derive load/sample/decode spans
    // from this shared event stream — load = start→first Step, sample = Step→Decoding,
    // decode = Decoding→Image — summed across the batch's images. Both the MLX and
    // candle image lanes funnel through here, so both get the split. Posted best-effort
    // at clean completion; coalesce-merges with the S2 hardware block server-side.
    let mut phase_timer = crate::job_metrics::PhaseTimer::new(Instant::now());
    // Effective denoise step count (sc-10406): the Step event's `total` is the
    // resolved step count, so a default run reports real steps, not the sparse payload.
    let mut effective_steps: Option<u32> = None;
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
                phase_timer.mark_sample_step(Instant::now());
                if effective_steps.is_none() {
                    effective_steps = Some(step_total);
                }
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
                phase_timer.mark_decoding(Instant::now());
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
            GenEvent::Loading { index, phase } => {
                mark_started(index);
                let component = match phase {
                    LoadPhase::TextEncoder => "text encoder",
                    LoadPhase::Renderer => "render components",
                };
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::LoadingModel,
                        ProgressStage::LoadingModel,
                        step_fraction(index, 0, 1, total_u32),
                        &format!("Image {}/{total} — loading {component}.", index + 1),
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
                phase_timer.mark_item_done(Instant::now());
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
    // Post the effective-settings + per-phase timing block (epic 10402,
    // sc-10405/sc-10406). Best-effort; coalesce-merges with the S2 hardware block
    // (which owns totalMs/backend/peaks) server-side.
    // The RESOLVED tier dir, so the recorded quant label describes the tier that actually RAN rather
    // than the one requested (sc-11042). Re-resolving here is the same pure path-join + `is_dir` probe
    // the lane already did before loading (no fetch, no I/O beyond a stat), and it runs once per job.
    // `.ok().flatten()` because a metrics block must never fail a completed generation: an unresolvable
    // dir yields `None`, which conservatively reports the request-derived q4/q8/bf16 label exactly as
    // it did before this parameter existed.
    let tier_dir = resolve_weights_dir(&plan.request, settings).ok().flatten();
    let mut metrics = build_image_metrics(
        &plan.request,
        effective_steps,
        total as u32,
        tier_dir.as_deref(),
    );
    if let Some(phase) = phase_timer.into_metrics(Instant::now()) {
        metrics.load_ms = phase.load_ms;
        metrics.sample_ms = phase.sample_ms;
        metrics.decode_ms = phase.decode_ms;
    }
    crate::job_metrics::post_generation_metrics(api, &job.id, &metrics).await;
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
        // Anima 2B (sc-10676): the candle asset stamp, the off-Mac sibling of `mlx_anima`.
        assert_eq!(candle_adapter_label("anima_base"), "candle_anima");
        assert_eq!(candle_adapter_label("anima_aesthetic"), "candle_anima");
        assert_eq!(candle_adapter_label("anima_turbo"), "candle_anima");
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
            // Krea 2 Raw (sc-9994, epic 9992): the undistilled full-CFG base rides the SAME generic candle
            // lane as Turbo (`render_base`). Was missing here, so plain Raw t2i stubbed a procedural gradient.
            "krea_2_raw",
            // SD3.5 (sc-7880): Large / Large Turbo / Medium ride the generic candle txt2img lane.
            "sd3_5_large",
            "sd3_5_large_turbo",
            "sd3_5_medium",
            // Anima 2B (sc-10676): base / aesthetic / turbo dense-load bf16 on the generic candle lane.
            "anima_base",
            "anima_aesthetic",
            "anima_turbo",
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

    /// sc-12090: the reject tail suggests only INSTALLED, smaller tiers — never the rejected tier, and
    /// never the picker (hidden when ≤1 tier is installed, the case that produced the misleading
    /// "Pick a lower tier (Q4/Q8)" on the #1516 q4-only install).
    #[test]
    fn vram_reject_tail_names_only_installed_smaller_tiers() {
        // Two smaller tiers installed → both offered, uppercased, highest-fidelity first.
        let tail = vram_reject_tail(&["q8", "q4"]);
        assert!(tail.contains("Q8 / Q4"), "lists installed smaller tiers: {tail}");
        assert!(!tail.contains("picker"), "never points at the picker: {tail}");
        // One smaller tier installed.
        assert!(vram_reject_tail(&["q4"]).contains("(Q4)"));
        // None smaller installed (the single-tier / q4-only case) → says so, no tier list, no picker.
        let none = vram_reject_tail(&[]);
        assert!(none.contains("No smaller tier is installed"), "{none}");
        assert!(!none.contains("Select a smaller"), "{none}");
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

    /// A4 (sc-10189): the generic img2img arm keys off the `ui.img2img` manifest flag the catalog
    /// forwards as `modelManifestEntry`, NOT a hardcoded model string — so Krea + SD3.5 + any future
    /// `ui.img2img` model route uniformly, and a model without the flag stays plain txt2img.
    #[cfg(target_os = "macos")]
    #[test]
    fn model_supports_img2img_reads_the_ui_manifest_flag() {
        let entry = |manifest: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": "m", "modelManifestEntry": manifest })
                    .as_object()
                    .unwrap(),
            )
        };
        // ui.img2img: true → opted in (SD3.5 + Krea shape).
        assert!(model_supports_img2img(&entry(
            json!({ "ui": { "img2img": true } })
        )));
        // Flag explicitly false, or no `ui`, or no flag → plain txt2img.
        assert!(!model_supports_img2img(&entry(
            json!({ "ui": { "img2img": false } })
        )));
        assert!(!model_supports_img2img(&entry(json!({ "family": "sd3" }))));
        assert!(!model_supports_img2img(&entry(json!({ "ui": {} }))));
    }

    /// A4.5 (sc-10193): on Z-Image t2i the Character Studio identity-init (`referenceStrength`, sc-3619)
    /// keeps precedence; the generic `ui.img2img` reference-guided init (Image Studio "Image reference"
    /// tile: `advanced.strength`, no `referenceStrength`) fires only when identity-init doesn't engage,
    /// the model opts into `ui.img2img`, AND a reference is present. Otherwise plain txt2img.
    #[cfg(target_os = "macos")]
    #[test]
    fn zimage_generic_img2img_yields_to_identity_reference() {
        let req = |advanced: serde_json::Value| {
            ImageRequest::from_payload(
                json!({
                    "model": "z_image_turbo",
                    "modelManifestEntry": { "ui": { "img2img": true } },
                    // Both surfaces carry a reference asset; `identity_strength` reads it too.
                    "referenceAssetId": "asset-1",
                    "advanced": advanced,
                })
                .as_object()
                .unwrap(),
            )
        };
        // Image Studio "Image reference": strength only, no referenceStrength, a reference present →
        // generic img2img takes over.
        assert!(zimage_uses_generic_img2img(
            &req(json!({ "strength": 0.6 })),
            true
        ));
        // Character Studio identity: referenceStrength set → identity-init keeps precedence, generic
        // img2img yields (even though ui.img2img is on and a reference is present).
        assert!(!zimage_uses_generic_img2img(
            &req(json!({ "referenceStrength": 0.7 })),
            true
        ));
        // No reference asset present → neither surface engages (plain txt2img).
        assert!(!zimage_uses_generic_img2img(
            &req(json!({ "strength": 0.6 })),
            false
        ));
        // Model without the ui.img2img flag → never the generic path.
        let no_flag = ImageRequest::from_payload(
            json!({ "model": "z_image_turbo", "modelManifestEntry": { "ui": {} }, "advanced": {} })
                .as_object()
                .unwrap(),
        );
        assert!(!zimage_uses_generic_img2img(&no_flag, true));
    }

    /// A4.4 (sc-10192): Ideogram opts into the generic `ui.img2img` surface (the Image Studio "Image
    /// reference" tile, `text_to_image` mode) while ALSO owning the bespoke Remix/inpaint edit arm
    /// (`edit_image` mode, [`resolve_ideogram_edit`], sc-6303). The two arms in
    /// [`resolve_generic_lane_conditioning`] are mutually exclusive by mode — the edit arm is checked
    /// first and gates on `mode == "edit_image"`, the generic img2img arm on `mode != "edit_image"` — so a
    /// plain-t2i reference routes to the generic init (a single `Conditioning::Reference`, no mask, which
    /// the native engine's edit path denoises as plain img2img) while an Edit-tab job keeps the
    /// mask-capable path. No engine change was needed: mlx-gen `resolve_edit` already treats a Reference
    /// with no Mask as img2img. This tripwire locks the flag + mode-split; the disk-backed resolve
    /// (asset decode) is validated on-device.
    #[cfg(target_os = "macos")]
    #[test]
    fn ideogram_img2img_routes_by_mode() {
        let req = |model: &str, mode: &str| {
            ImageRequest::from_payload(
                json!({
                    "model": model,
                    "mode": mode,
                    "modelManifestEntry": { "ui": { "img2img": true } },
                    "referenceAssetId": "asset-1",
                })
                .as_object()
                .unwrap(),
            )
        };
        for model in ["ideogram_4", "ideogram_4_turbo"] {
            let is_ideogram_edit = |r: &ImageRequest| {
                matches!(r.model.as_str(), "ideogram_4" | "ideogram_4_turbo")
                    && r.mode == "edit_image"
            };
            // Plain t2i + reference: the generic img2img arm's gate holds (flag on, non-edit mode) and the
            // earlier Ideogram edit arm does not — so the reference takes the generic img2img init.
            let t2i = req(model, "text_to_image");
            assert!(model_supports_img2img(&t2i) && t2i.mode != "edit_image");
            assert!(!is_ideogram_edit(&t2i));
            // Edit tab: the Ideogram edit arm claims it first and the generic arm yields (edit mode).
            let edit = req(model, "edit_image");
            assert!(is_ideogram_edit(&edit));
            assert!(!(model_supports_img2img(&edit) && edit.mode != "edit_image"));
        }
    }

    /// Write a minimal present `<tier>/transformer/<file>` so [`standard_tier_subdir`]'s
    /// filename-agnostic probe sees the tier as downloaded.
    fn seed_tier(root: &Path, tier: &str, file: &str) {
        let dir = root.join(tier).join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), b"x").unwrap();
    }

    #[test]
    fn defaults_to_q8_and_honors_quantize_selection() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Packed q4/q8 single-file + dense sharded bf16 (only the index.json shape).
        seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "bf16", "diffusion_pytorch_model.safetensors.index.json");

        // No selection → q8 default (epic 10721 / sc-10726), clamped to installed.
        assert_eq!(
            standard_tier_subdir(root, &request(json!({}))),
            root.join("q8")
        );
        // An explicit Q4 pick is still honored (never overridden by the q8 default).
        assert_eq!(
            standard_tier_subdir(root, &request(json!({ "mlxQuantize": 4 }))),
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

    /// **sc-10732 — acceptance #1: the app-wide default-tier revert guard.**
    ///
    /// Epic 10721 moved the gen-time default tier off the old blind `q4` to `q8` (sc-10726): the shared
    /// [`preferred_tier`] returns `"q8"` with no explicit `mlxQuantize` pick and no per-model floor, and
    /// BOTH tier resolvers ([`standard_tier_subdir`] and [`anima_tier_subdir`]) inherit it. If a future
    /// change reverts that default back to `q4` — in `preferred_tier`'s `None => …` arm, or a resolver
    /// special-case — this test FAILS LOUDLY. That is the whole point of the sc-10732 lock: the finer
    /// resolver/floor tests each imply it, but this one names it at the revert site so the intent is
    /// unmissable. Deliberately redundant.
    #[test]
    fn default_tier_is_q8_not_q4_regression() {
        // The shared default-tier primitive: no explicit `mlxQuantize`, no per-model floor → q8, never q4.
        assert_eq!(
            preferred_tier(None, None, false),
            "q8",
            "app-wide gen default MUST be q8 (epic 10721 / sc-10726) — a revert to q4 is the regression \
             this guards"
        );
        assert_ne!(
            preferred_tier(None, None, false),
            "q4",
            "the pre-epic-10721 blind-q4 default has been reverted — do NOT reinstate it (sc-10726/sc-10732)"
        );

        // Disk-backed through the standard resolver: a default job (no mlxQuantize) with ALL tiers
        // installed resolves the q8 subdir, not the washed q4 — the revert is caught end-to-end too.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for tier in ["bf16", "q8", "q4"] {
            seed_tier(root, tier, "diffusion_pytorch_model.safetensors");
        }
        let default_job = request(json!({}));
        assert_eq!(
            standard_tier_subdir(root, &default_job),
            root.join("q8"),
            "standard_tier_subdir default MUST land on q8 (not q4) when all tiers are installed"
        );
        assert_ne!(standard_tier_subdir(root, &default_job), root.join("q4"));

        // The Anima resolver shares the same default (sc-10714 → sc-10731): its no-pick default is q8 too,
        // so a revert to q4 also washes Anima — the exact quality bug epic 10721 fixed.
        #[cfg(any(target_os = "macos", feature = "backend-candle"))]
        {
            let anima_root = tempfile::tempdir().unwrap();
            for tier in ["bf16", "q8", "q4"] {
                let dir = anima_root.path().join(tier).join("diffusion_models");
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(dir.join("anima-base-v1.0.safetensors"), b"x").unwrap();
            }
            let anima_default =
                ImageRequest::from_payload(json!({ "model": "anima_base" }).as_object().unwrap());
            assert_eq!(
                anima_tier_subdir(anima_root.path(), &anima_default),
                anima_root.path().join("q8"),
                "anima_tier_subdir default MUST be q8 (sc-10714), never the washed q4"
            );
            assert_ne!(
                anima_tier_subdir(anima_root.path(), &anima_default),
                anima_root.path().join("q4")
            );
        }
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
        // sc-10726: the q8 default is CLAMPED to installed — with only q4 on disk a default job
        // (no mlxQuantize) resolves q4, never a tier the user didn't download (no OOM risk).
        assert_eq!(
            standard_tier_subdir(root, &request(json!({}))),
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
        // Default q8, q8 selection, bf16 opt-out all resolve to the unet-backed tier subdir.
        assert_eq!(
            standard_tier_subdir(root, &request(json!({}))),
            root.join("q8")
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
            root.join("q8")
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

    /// sc-10517 / sc-10714 / sc-10731: Anima is convert-at-install with a q4/q8/bf16 MATRIX under the
    /// injected `modelPath` root (`bf16/ q8/ q4/`, each a `diffusion_models/<variant>.safetensors` tree —
    /// NOT a `transformer/` component). [`anima_tier_subdir`] picks the tier by `mlxQuantize`
    /// (**default Q8**; `<= 0` → bf16; `1..=4` → q4; `> 4` → q8) and falls back clean-tiers-first through
    /// q8 → bf16 → q4 → root, so a partial install surfaces as a load error, never a silent half-load onto
    /// the washed q4. The Q8 default (sc-10714) is the fix for base/aesthetic rendering smudgy at q4 —
    /// q4 × CFG amplifies quant error; Q8 is near-lossless. [`is_anima_model`] gates only the three ids.
    ///
    /// sc-10731 reconciled the old anima-specific `None => "q8"` hardcode into the shared, floor-driven
    /// [`preferred_tier`]: with no manifest floor the app-wide q8 default still stands (the assertions
    /// above), and the added assertions below prove the floor now DRIVES the default — a manifest
    /// `mlx.minQualityTier` clamps the default UP (capped by installed), while an explicit pick is honored.
    #[test]
    fn anima_tier_subdir_selects_and_falls_back() {
        let anima_request = |bits: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": "anima_base", "advanced": { "mlxQuantize": bits } })
                    .as_object()
                    .unwrap(),
            )
        };
        assert!(is_anima_model("anima_base"));
        assert!(is_anima_model("anima_aesthetic") && is_anima_model("anima_turbo"));
        assert!(!is_anima_model("sd3_5_large"));

        let seed_tier = |root: &std::path::Path, tier: &str| {
            let dir = root.join(tier).join("diffusion_models");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("anima-base-v1.0.safetensors"), b"x").unwrap();
        };
        // All three tiers present → exact selection.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for tier in ["bf16", "q8", "q4"] {
            seed_tier(root, tier);
        }
        assert_eq!(
            anima_tier_subdir(root, &anima_request(json!(null))),
            root.join("q8"),
            "no opt-in → the Q8 default (sc-10714), not the washed q4"
        );
        assert_eq!(
            anima_tier_subdir(root, &anima_request(json!(4))),
            root.join("q4"),
            "an explicit q4 pick is still honored"
        );
        assert_eq!(
            anima_tier_subdir(root, &anima_request(json!(8))),
            root.join("q8")
        );
        assert_eq!(
            anima_tier_subdir(root, &anima_request(json!(0))),
            root.join("bf16"),
            "explicit bf16 (mlxQuantize <= 0) is honored"
        );
        // Only q4 downloaded, but the default (q8) is requested → falls through clean-tiers-first to the
        // present q4, never a bare root. (An all-tiers install lands on q8 above; this is the partial case.)
        let tmp2 = tempfile::tempdir().unwrap();
        seed_tier(tmp2.path(), "q4");
        assert_eq!(
            anima_tier_subdir(tmp2.path(), &anima_request(json!(null))),
            tmp2.path().join("q4")
        );
        // q8 default absent (only bf16 + q4 present) → fallback prefers the clean bf16 over the washed q4.
        let tmp2b = tempfile::tempdir().unwrap();
        seed_tier(tmp2b.path(), "bf16");
        seed_tier(tmp2b.path(), "q4");
        assert_eq!(
            anima_tier_subdir(tmp2b.path(), &anima_request(json!(null))),
            tmp2b.path().join("bf16"),
            "q8 default absent → fallback prefers clean bf16 over washed q4"
        );
        // Nothing present → the root itself (the loader then surfaces a clear error).
        let tmp3 = tempfile::tempdir().unwrap();
        assert_eq!(
            anima_tier_subdir(tmp3.path(), &anima_request(json!(null))),
            tmp3.path().to_path_buf()
        );

        // sc-10731 — the per-model quality FLOOR now DRIVES the anima default (was a hardcode).
        // A floored request carries `advanced.mlxQuantize` AND the forwarded manifest floor.
        let floored_request = |bits: serde_json::Value, floor: &str| {
            ImageRequest::from_payload(
                json!({
                    "model": "anima_base",
                    "advanced": { "mlxQuantize": bits },
                    "modelManifestEntry": { "mlx": { "minQualityTier": floor } }
                })
                .as_object()
                .unwrap(),
            )
        };
        // Anima base's PRODUCTION shape: floor q8, all tiers present, no explicit pick → the default is
        // q8 — now floor-DERIVED (the hardcode is gone), not a resolver special-case.
        assert_eq!(
            anima_tier_subdir(root, &floored_request(json!(null), "q8")),
            root.join("q8"),
            "floor q8 drives the default to q8 (reconciled from the sc-10714 hardcode)"
        );
        // Floor CAPPED by installed: floor q8 but only q4 on disk → q4 (the floor never selects an
        // uninstalled tier — a heavy model never resolves a tier the user didn't download).
        assert_eq!(
            anima_tier_subdir(tmp2.path(), &floored_request(json!(null), "q8")),
            tmp2.path().join("q4"),
            "floor q8 but only q4 installed → q4 (floor capped by installed)"
        );
        // Floor RAISES above the q8 default: a synthetic bf16 floor + no explicit pick → bf16 (proves the
        // clamp genuinely lifts the default, not just coincides with the app-wide q8).
        assert_eq!(
            anima_tier_subdir(root, &floored_request(json!(null), "bf16")),
            root.join("bf16"),
            "floored default → the floor tier (bf16 floor raises above the q8 default)"
        );
        // An EXPLICIT below-floor pick is HONORED even against a q8 floor — the worker never overrides a
        // deliberate quant choice (the web surfaces the advisory instead).
        assert_eq!(
            anima_tier_subdir(root, &floored_request(json!(4), "q8")),
            root.join("q4"),
            "explicit q4 honored despite the q8 floor (below-floor pick is the user's choice)"
        );
    }

    /// sc-10731: the shared, floor-aware default-tier logic behind both [`standard_tier_subdir`] and
    /// [`anima_tier_subdir`]. An explicit `mlxQuantize` pick maps directly and is honored regardless of
    /// the floor; with NO pick, the app-wide q8 default is clamped UP to the floor (never DOWN).
    #[test]
    fn preferred_tier_clamps_default_up_to_the_floor_only() {
        // No explicit pick, no floor → the app-wide q8 default (unchanged, non-floored models).
        assert_eq!(preferred_tier(None, None, false), "q8");
        // Floor at/below q8 leaves the q8 default untouched (the floor only ever RAISES).
        assert_eq!(preferred_tier(None, Some("q8"), false), "q8");
        assert_eq!(preferred_tier(None, Some("q4"), false), "q8");
        // A floor ABOVE q8 raises the default to the floor tier.
        assert_eq!(preferred_tier(None, Some("bf16"), false), "bf16");
        // Explicit picks map directly and are HONORED even below the floor (no clamp on an explicit pick).
        assert_eq!(preferred_tier(Some(4), Some("q8"), false), "q4");
        assert_eq!(preferred_tier(Some(0), Some("q8"), false), "bf16");
        assert_eq!(preferred_tier(Some(8), None, false), "q8");
        assert_eq!(preferred_tier(Some(2), None, false), "q4");
        // Rank order + normalization helpers.
        assert!(tier_quality_rank("bf16") > tier_quality_rank("q8"));
        assert!(tier_quality_rank("q8") > tier_quality_rank("q4"));
        assert_eq!(tier_quality_rank("mystery"), 0);
    }

    /// sc-11042 / epic 11037 SC#5 — **the regression guard for "no existing tier changes"**.
    ///
    /// The whole `mlxQuantize` bits map must resolve EXACTLY as it did before NVFP4 existed, for every
    /// (bits, floor) pair, on every host. `nvfp4: false` is what every non-NVFP4 request passes — i.e.
    /// every request in existence today — so this pins the claim that adding the tier is purely
    /// additive. If someone ever "helpfully" routes q4 → NVFP4 on Blackwell (the Option B this story
    /// rejected), these assertions fail.
    #[test]
    fn preferred_tier_bits_map_is_unchanged_by_the_nvfp4_tier() {
        // The exact pre-sc-11042 mapping, re-asserted with the new parameter defaulted off.
        for floor in [None, Some("bf16"), Some("q8"), Some("q4"), Some("mystery")] {
            // Explicit bits picks: `<= 0` → bf16, `> 4` → q8, `1..=4` → q4 — floor-independent.
            assert_eq!(preferred_tier(Some(0), floor, false), "bf16");
            assert_eq!(preferred_tier(Some(-1), floor, false), "bf16");
            assert_eq!(preferred_tier(Some(4), floor, false), "q4");
            assert_eq!(preferred_tier(Some(1), floor, false), "q4");
            assert_eq!(preferred_tier(Some(8), floor, false), "q8");
            assert_eq!(preferred_tier(Some(16), floor, false), "q8");
        }
        // No pick → the q8 default, clamped UP to a higher floor only.
        assert_eq!(preferred_tier(None, None, false), "q8");
        assert_eq!(preferred_tier(None, Some("q4"), false), "q8");
        assert_eq!(preferred_tier(None, Some("q8"), false), "q8");
        assert_eq!(preferred_tier(None, Some("bf16"), false), "bf16");
        // And NOTHING in the bits map can ever produce the NVFP4 tier: only the explicit flag does.
        for bits in [None, Some(-1), Some(0), Some(1), Some(4), Some(8), Some(16)] {
            for floor in [None, Some("bf16"), Some("q8"), Some("q4")] {
                assert_ne!(
                    preferred_tier(bits, floor, false),
                    NVFP4_TIER,
                    "NVFP4 must never be reachable from the bits map (bits={bits:?}, floor={floor:?})"
                );
            }
        }
    }

    /// sc-11042: an explicit, host-eligible NVFP4 pick resolves the distinct `nvfp4` tier and takes no
    /// part in the bits map or the floor clamp — it is a tier identity, not a rung on the fidelity ladder.
    #[test]
    fn preferred_tier_resolves_the_distinct_nvfp4_tier_on_an_explicit_pick() {
        // Wins regardless of what bits/floor say — NVFP4 has no honest `mlxQuantize` integer, so a
        // stray/stale bits value must not steer a request that explicitly named the NVFP4 tier.
        for bits in [None, Some(0), Some(4), Some(8)] {
            for floor in [None, Some("bf16"), Some("q8")] {
                assert_eq!(preferred_tier(bits, floor, true), NVFP4_TIER);
            }
        }
        // It is NOT floor-clamped: a bf16 floor does not raise/replace an explicit NVFP4 pick.
        assert_eq!(preferred_tier(None, Some("bf16"), true), NVFP4_TIER);
        // The label is distinct from q4 — the aliasing SC#5 forbids (see `video_quant_label`).
        assert_ne!(NVFP4_TIER, "q4");
    }

    /// sc-11042: [`nvfp4_requested`] reads ONLY the explicit `advanced.quantTier: "nvfp4"` label, and no
    /// `mlxQuantize` value — not even `4` — can stand in for it. This is the request-side half of SC#5.
    #[test]
    fn nvfp4_requested_reads_only_the_explicit_quant_tier_label() {
        // The explicit label, tolerant of surrounding whitespace / casing like the sibling parsers.
        assert!(nvfp4_requested(&request(json!({ "quantTier": "nvfp4" }))));
        assert!(nvfp4_requested(&request(json!({ "quantTier": "  nvfp4 " }))));
        assert!(nvfp4_requested(&request(json!({ "quantTier": "NVFP4" }))));
        // Nothing else asks for NVFP4: no label, another tier's label, a non-string, or ANY bits value.
        assert!(!nvfp4_requested(&request(json!({}))));
        assert!(!nvfp4_requested(&request(json!({ "quantTier": "q4" }))));
        assert!(!nvfp4_requested(&request(json!({ "quantTier": "int8-convrot" }))));
        assert!(!nvfp4_requested(&request(json!({ "quantTier": 4 }))));
        assert!(!nvfp4_requested(&request(json!({ "quantTier": true }))));
        for bits in [0, 4, 8] {
            assert!(
                !nvfp4_requested(&request(json!({ "mlxQuantize": bits }))),
                "mlxQuantize {bits} must never request NVFP4 — it is a bits-valued knob naming q4/q8/bf16"
            );
        }
    }

    /// sc-11042 — **the SC#5 opt-in guard at the resolver**: with NO explicit `quantTier` label, a
    /// Blackwell-eligible host resolves EXACTLY the tier it resolved before this story. Being on sm_120
    /// selects nothing by itself; NVFP4 is only ever reached by a deliberate user pick.
    #[test]
    fn nvfp4_never_selected_without_an_explicit_pick() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "bf16", "diffusion_pytorch_model.safetensors.index.json");
        seed_tier(root, NVFP4_TIER, "diffusion_pytorch_model.safetensors");

        // The injected `true` says "the HOST is Blackwell-eligible" — the hardware gate is ON, but no
        // request below carries the `quantTier` label. Every answer is the pre-sc-11042 one, even though
        // an `nvfp4/` tier is sitting right there on disk. That is SC#5: hardware selects nothing.
        for (advanced, expected) in [
            (json!({}), "q8"),                   // default
            (json!({ "mlxQuantize": 4 }), "q4"), // the tier NVFP4 must never silently replace
            (json!({ "mlxQuantize": 8 }), "q8"),
            (json!({ "mlxQuantize": 0 }), "bf16"),
        ] {
            assert_eq!(
                standard_tier_subdir_gated(root, &request(advanced.clone()), true),
                root.join(expected),
                "an unlabeled request must resolve {expected} on a Blackwell host (advanced={advanced})"
            );
        }
    }

    /// sc-11042: the NVFP4 tier resolves on an sm_120 host and falls back CLEANLY off Blackwell.
    ///
    /// The two host classes are exercised through the injected gate rather than a live compute-cap
    /// probe, so the rig isn't needed; [`nvfp4_host_eligible`] is what maps hardware → this bool, and
    /// its floor is pinned separately by `gpu::tests`.
    #[test]
    fn nvfp4_tier_resolves_on_blackwell_and_falls_back_off_it() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
        seed_tier(root, NVFP4_TIER, "diffusion_pytorch_model.safetensors");
        let picked = request(json!({ "quantTier": "nvfp4" }));

        // sm_120 + the tier installed → the distinct `nvfp4/` dir.
        assert_eq!(
            standard_tier_subdir_gated(root, &picked, true),
            root.join(NVFP4_TIER)
        );
        // NOT Blackwell (pre-sm_120 NVIDIA, macOS/MLX, or the neither build) → the label is ignored and
        // the request lands on an installed tier via the normal chain. A clean fallback, not an error.
        assert_eq!(standard_tier_subdir_gated(root, &picked, false), root.join("q8"));

        // sm_120 but the tier ISN'T converted yet (sc-11043 owns the converter; the shipping case today)
        // → rejoins the same clean chain rather than failing the load.
        let bare = tempfile::tempdir().unwrap();
        seed_tier(bare.path(), "q4", "diffusion_pytorch_model.safetensors");
        seed_tier(bare.path(), "q8", "diffusion_pytorch_model.safetensors");
        assert_eq!(
            standard_tier_subdir_gated(bare.path(), &picked, true),
            bare.path().join("q8")
        );
        // …and with only q4 on disk it clamps to q4 — never a half-load, never an FP4 load with no
        // FP4 weights.
        let only_q4 = tempfile::tempdir().unwrap();
        seed_tier(only_q4.path(), "q4", "diffusion_pytorch_model.safetensors");
        assert_eq!(
            standard_tier_subdir_gated(only_q4.path(), &picked, true),
            only_q4.path().join("q4")
        );
    }

    /// sc-11042: [`resolve_quant`] returns the distinct `Quant::Nvfp4` for an explicit, host-eligible
    /// pick — with **no** bit count (NVFP4 is ~4.5 EFFECTIVE bits/weight; `Some(4)` would alias the
    /// recipe onto q4) — and is otherwise completely unchanged.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn resolve_quant_returns_the_distinct_nvfp4_tier_only_on_an_explicit_blackwell_pick() {
        let picked = request(json!({ "quantTier": "nvfp4" }));
        let nvfp4_dir = PathBuf::from("/models/klein").join(NVFP4_TIER);
        let q8_dir = PathBuf::from("/models/klein").join("q8");
        // Explicit pick + Blackwell + the nvfp4 tier RESOLVED → the distinct tier, and NOT `Some(4)` bits.
        assert_eq!(
            resolve_quant_gated(&picked, true, Some(&nvfp4_dir)),
            (Some(Quant::Nvfp4), None)
        );
        // Off Blackwell → the label is ignored; the unchanged q8 default. A clean fallback.
        assert_eq!(
            resolve_quant_gated(&picked, false, Some(&nvfp4_dir)),
            (Some(Quant::Q8), Some(8))
        );
        // On Blackwell but the resolver landed on q8 (the `nvfp4/` dir isn't installed — the shipping
        // case today): the LOAD quant must be q8 too. Loading FP4 against q8 weights is not a mislabel,
        // it's a wrong load.
        assert_eq!(
            resolve_quant_gated(&picked, true, Some(&q8_dir)),
            (Some(Quant::Q8), Some(8))
        );
        // Tier dir unknown ⇒ never claim NVFP4 (see `tier_dir_is_nvfp4`).
        assert_eq!(
            resolve_quant_gated(&picked, true, None),
            (Some(Quant::Q8), Some(8))
        );

        // SC#5: every existing mapping is untouched on BOTH host classes and on EVERY tier dir — being
        // on sm_120, even with the `nvfp4/` tier resolved, never converts an unlabeled q4/q8/bf16
        // request into an NVFP4 one.
        for nvfp4_host in [false, true] {
            for dir in [None, Some(&nvfp4_dir), Some(&q8_dir)] {
                assert_eq!(
                    resolve_quant_gated(&request(json!({})), nvfp4_host, dir.map(PathBuf::as_path)),
                    (Some(Quant::Q8), Some(8))
                );
                assert_eq!(
                    resolve_quant_gated(
                        &request(json!({ "mlxQuantize": 4 })),
                        nvfp4_host,
                        dir.map(PathBuf::as_path)
                    ),
                    (Some(Quant::Q4), Some(4)),
                    "a q4 pick must stay int4-affine q4 on every host (epic 11037 SC#5)"
                );
                assert_eq!(
                    resolve_quant_gated(
                        &request(json!({ "mlxQuantize": 8 })),
                        nvfp4_host,
                        dir.map(PathBuf::as_path)
                    ),
                    (Some(Quant::Q8), Some(8))
                );
                assert_eq!(
                    resolve_quant_gated(
                        &request(json!({ "mlxQuantize": 0 })),
                        nvfp4_host,
                        dir.map(PathBuf::as_path)
                    ),
                    (None, None)
                );
            }
        }
    }

    /// sc-11042 — **the dense-TE carve-out outranks the NVFP4 tier** (sc-8711 / sc-9362).
    ///
    /// `flux2_klein_9b`/`_kv` are in [`DENSE_TE_TIER_MODELS`] AND ride the candle txt2img lane, whose
    /// `resolve_quant` call is gated only by `model.supports_quant()`. With the NVFP4 arm ordered ahead
    /// of the carve-out, a crafted `quantTier: "nvfp4"` returned `Some(Nvfp4)` and skipped it — which
    /// would re-quantize the bf16 text encoder those stories deliberately kept dense. The carve-out is
    /// the wider invariant (the TE must never be quantized by ANY tier), so it wins outright.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn the_nvfp4_arm_never_short_circuits_the_dense_te_carve_out() {
        let nvfp4_dir = PathBuf::from("/models/klein").join(NVFP4_TIER);
        // Both the registry form and the manifest-flag form of a dense-TE turnkey, with EVERY gate for
        // the NVFP4 arm satisfied: explicit pick, Blackwell host, and the `nvfp4/` tier resolved.
        let mut by_id = request(json!({ "quantTier": "nvfp4" }));
        by_id.model = "flux2_klein_9b".to_owned();
        let by_manifest = ImageRequest::from_payload(
            json!({
                "model": "some_matrix_model",
                "advanced": { "quantTier": "nvfp4" },
                "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } }
            })
            .as_object()
            .unwrap(),
        );

        for dense_te in [&by_id, &by_manifest] {
            assert!(is_dense_te_tier(dense_te), "test fixture must be dense-TE");
            assert_eq!(
                resolve_quant_gated(dense_te, true, Some(&nvfp4_dir)),
                (None, None),
                "a dense-TE turnkey must load with quant None — the NVFP4 arm must not preempt the \
                 sc-8711 carve-out that keeps its bf16 text encoder dense"
            );
        }

        // The carve-out is not a general NVFP4 kill-switch: a NON-dense-TE model on the same terms
        // still selects the tier, so the fix can't be masking the arm entirely.
        assert_eq!(
            resolve_quant_gated(&request(json!({ "quantTier": "nvfp4" })), true, Some(&nvfp4_dir)),
            (Some(Quant::Nvfp4), None)
        );
    }

    /// sc-11042 / epic 11037 SC#5 — **the image lane's aliasing guard**, the sibling of
    /// `video_quant_label_never_aliases_nvfp4_to_q4`.
    ///
    /// [`effective_quant_label`] maps [`resolve_quant`]'s BIT COUNT onto a label, and NVFP4's bits are
    /// deliberately `None`, so a bits-only match drops it into the `_ => bf16` arm — stamping a 4-bit
    /// NVFP4 render as full-precision `"bf16"`. The inverse of the video lane's `q4` mislabel, the same
    /// violation. Matching the variant is what pins it.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn effective_quant_label_never_aliases_nvfp4_to_bf16_or_q4() {
        let picked = request(json!({ "quantTier": "nvfp4" }));
        let nvfp4_dir = PathBuf::from("/models/klein").join(NVFP4_TIER);
        let (label, bits) = effective_quant_label_gated(&picked, true, Some(&nvfp4_dir));
        assert_eq!(label, Some("nvfp4".to_owned()));
        // Neither mislabel is possible: not the "bf16" a bits-only match produced…
        assert_ne!(label, Some("bf16".to_owned()));
        // …nor the "q4" the video lane's bits-derived form produced.
        assert_ne!(label, Some("q4".to_owned()));
        // No honest integer width — same reason the tier reports None everywhere else.
        assert_eq!(bits, None);

        // Off Blackwell the label reports what actually ran (the q8 fallback), not the tier asked for.
        assert_eq!(
            effective_quant_label_gated(&picked, false, Some(&nvfp4_dir)),
            (Some("q8".to_owned()), Some(8))
        );
        // SC#5: the existing labels are unchanged on both host classes.
        for nvfp4_host in [false, true] {
            assert_eq!(
                effective_quant_label_gated(
                    &request(json!({ "mlxQuantize": 4 })),
                    nvfp4_host,
                    Some(&nvfp4_dir)
                ),
                (Some("q4".to_owned()), Some(4))
            );
            assert_eq!(
                effective_quant_label_gated(
                    &request(json!({ "mlxQuantize": 8 })),
                    nvfp4_host,
                    Some(&nvfp4_dir)
                ),
                (Some("q8".to_owned()), Some(8))
            );
            assert_eq!(
                effective_quant_label_gated(
                    &request(json!({ "mlxQuantize": 0 })),
                    nvfp4_host,
                    Some(&nvfp4_dir)
                ),
                (Some("bf16".to_owned()), None)
            );
            // The int8-convrot tier still wins its early return.
            assert_eq!(
                effective_quant_label_gated(
                    &request(json!({ "convRot": true })),
                    nvfp4_host,
                    Some(&nvfp4_dir)
                ),
                (Some("int8-convrot".to_owned()), None)
            );
        }
    }

    /// sc-11042 / epic 11037 SC#5 — **the recorded label must describe the tier that RAN.**
    ///
    /// The regression this pins: `effective_quant_label` was host-aware but DISK-BLIND. It returned
    /// `Nvfp4` from `nvfp4_requested && nvfp4_host` alone, while [`standard_tier_subdir`] independently
    /// (and correctly) fell back to `q8/` when the `nvfp4/` dir was absent — **the shipping case on
    /// every model today**, since sc-11043 has not converted a tier yet. So on a Blackwell candle host
    /// the resolver loaded the **q8** weights and the asset record stamped them **`"nvfp4"`**: a q8
    /// render sold as NVFP4. Exactly the SC#5 creative-choice aliasing this tier exists to eliminate,
    /// displaced out of selection and into telemetry.
    ///
    /// The suite already contained both halves of the contradiction — `standard_tier_subdir_gated(bare,
    /// picked, true) == q8` and `effective_quant_label_gated(picked, true) == "nvfp4"` — and never
    /// connected them. This test connects them: it drives the SAME request through the resolver and the
    /// label and asserts they agree, so the two can never drift apart again.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn effective_quant_label_reports_the_resolved_tier_not_the_requested_one() {
        let picked = request(json!({ "quantTier": "nvfp4" }));

        // A Blackwell host whose turnkey has NO `nvfp4/` dir — the shipping case today.
        let bare = tempfile::tempdir().unwrap();
        seed_tier(bare.path(), "q4", "diffusion_pytorch_model.safetensors");
        seed_tier(bare.path(), "q8", "diffusion_pytorch_model.safetensors");

        // The resolver falls back to q8 (this half already passed before the fix)…
        let resolved = standard_tier_subdir_gated(bare.path(), &picked, true);
        assert_eq!(resolved, bare.path().join("q8"));
        // …so the label MUST say q8. Before the fix this returned `Some("nvfp4")` with bits None —
        // a q8 render recorded as an NVFP4 one.
        let (label, bits) = effective_quant_label_gated(&picked, true, Some(&resolved));
        assert_ne!(
            label,
            Some(NVFP4_TIER.to_owned()),
            "an NVFP4 pick that RESOLVED q8 must never be recorded as nvfp4 (epic 11037 SC#5)"
        );
        assert_eq!((label, bits), (Some("q8".to_owned()), Some(8)));

        // Same host, same request, but the tier IS installed → the label is nvfp4. This is what proves
        // the fix pins the label to the DISK and isn't just disabling the tier.
        let converted = tempfile::tempdir().unwrap();
        seed_tier(converted.path(), "q8", "diffusion_pytorch_model.safetensors");
        seed_tier(
            converted.path(),
            NVFP4_TIER,
            "diffusion_pytorch_model.safetensors",
        );
        let resolved = standard_tier_subdir_gated(converted.path(), &picked, true);
        assert_eq!(resolved, converted.path().join(NVFP4_TIER));
        assert_eq!(
            effective_quant_label_gated(&picked, true, Some(&resolved)),
            (Some(NVFP4_TIER.to_owned()), None)
        );

        // The resolver and the label agree on EVERY tier/host/install combination — the invariant, not
        // just the two cases above. Each `expected` is read off the resolver's own output.
        for install in [vec!["q4", "q8"], vec!["q4", "q8", NVFP4_TIER]] {
            let root = tempfile::tempdir().unwrap();
            for tier in &install {
                seed_tier(root.path(), tier, "diffusion_pytorch_model.safetensors");
            }
            for nvfp4_host in [false, true] {
                for advanced in [
                    json!({ "quantTier": "nvfp4" }),
                    json!({}),
                    json!({ "mlxQuantize": 4 }),
                ] {
                    let req = request(advanced.clone());
                    let resolved = standard_tier_subdir_gated(root.path(), &req, nvfp4_host);
                    let (label, _) = effective_quant_label_gated(&req, nvfp4_host, Some(&resolved));
                    let resolved_tier = resolved.file_name().unwrap().to_str().unwrap();
                    assert_eq!(
                        label.as_deref() == Some(NVFP4_TIER),
                        resolved_tier == NVFP4_TIER,
                        "label {label:?} disagrees with the resolved tier {resolved_tier} \
                         (host={nvfp4_host}, advanced={advanced}, installed={install:?})"
                    );
                }
            }
        }
    }

    /// sc-11042: the Krea 2 Turbo resolver (epic 11037's named SC#1/SC#2 validation vehicle, sc-12110)
    /// wires the NVFP4 tier on the same terms — explicit pick + Blackwell, clean fallback otherwise —
    /// and its q4/q8/bf16 selection is unchanged.
    #[test]
    fn krea_model_subdir_wires_nvfp4_without_changing_its_existing_tiers() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for tier in ["q4", "q8", NVFP4_TIER] {
            seed_tier(root, tier, "diffusion_pytorch_model.safetensors");
        }
        let picked = request(json!({ "quantTier": "nvfp4" }));
        // Explicit pick on Blackwell → the distinct tier.
        assert_eq!(
            krea_model_subdir_gated(root, &picked, true),
            root.join(NVFP4_TIER)
        );
        // Off Blackwell → clean fallback to the shipped q8 default.
        assert_eq!(krea_model_subdir_gated(root, &picked, false), root.join("q8"));
        // SC#5: krea's existing tiers resolve exactly as before on a Blackwell host with an `nvfp4/`
        // dir present.
        for (advanced, expected) in [
            (json!({}), "q8"),
            (json!({ "mlxQuantize": 4 }), "q4"),
            (json!({ "mlxQuantize": 8 }), "q8"),
        ] {
            assert_eq!(
                krea_model_subdir_gated(root, &request(advanced), true),
                root.join(expected)
            );
        }
    }

    /// sc-10731: [`min_quality_floor`] reads the forwarded manifest `mlx.minQualityTier`, honoring only a
    /// valid bf16/q8/q4 value and treating an absent/bogus one as no floor.
    #[test]
    fn min_quality_floor_reads_valid_manifest_values_only() {
        let with_floor = |mlx: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": "anima_base", "modelManifestEntry": { "mlx": mlx } })
                    .as_object()
                    .unwrap(),
            )
        };
        assert_eq!(
            min_quality_floor(&with_floor(json!({ "minQualityTier": "q8" }))),
            Some("q8")
        );
        assert_eq!(min_quality_floor(&with_floor(json!({ "quantize": 4 }))), None);
        assert_eq!(
            min_quality_floor(&with_floor(json!({ "minQualityTier": "q2" }))),
            None
        );
        // No manifest entry at all → no floor.
        assert_eq!(min_quality_floor(&request(json!({}))), None);
    }

    /// sc-10731: [`standard_tier_subdir`] applies the same floor clamp — a floored standard-tier model's
    /// DEFAULT lands at the floor (capped by installed), while a non-floored one is unchanged and an
    /// explicit below-floor pick is honored.
    #[test]
    fn standard_tier_subdir_clamps_default_to_the_floor() {
        let floored = |bits: Option<i64>, floor: &str| {
            let advanced = match bits {
                Some(b) => json!({ "mlxQuantize": b }),
                None => json!({}),
            };
            ImageRequest::from_payload(
                json!({
                    "model": "some_matrix_model",
                    "advanced": advanced,
                    "modelManifestEntry": { "mlx": { "minQualityTier": floor } }
                })
                .as_object()
                .unwrap(),
            )
        };
        // All three tiers present.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for tier in ["bf16", "q8", "q4"] {
            seed_tier(root, tier, "diffusion_pytorch_model.safetensors");
        }
        // Floor bf16, no explicit pick → bf16 (floored default → floor tier).
        assert_eq!(
            standard_tier_subdir(root, &floored(None, "bf16")),
            root.join("bf16")
        );
        // Floor q8, no explicit pick → q8; a non-floored default is q8 too, but here it is floor-driven.
        assert_eq!(
            standard_tier_subdir(root, &floored(None, "q8")),
            root.join("q8")
        );
        // Explicit q4 honored despite a bf16 floor (below-floor pick is the user's choice).
        assert_eq!(
            standard_tier_subdir(root, &floored(Some(4), "bf16")),
            root.join("q4")
        );
        // Floor capped by installed: floor bf16 but only q4 on disk → q4.
        let only_q4 = tempfile::tempdir().unwrap();
        seed_tier(only_q4.path(), "q4", "diffusion_pytorch_model.safetensors");
        assert_eq!(
            standard_tier_subdir(only_q4.path(), &floored(None, "bf16")),
            only_q4.path().join("q4")
        );
        // Non-floored model unaffected — the plain q8 default still stands (acceptance #3).
        assert_eq!(
            standard_tier_subdir(root, &request(json!({}))),
            root.join("q8")
        );
    }

    /// Regression guard for **sc-10578** (epic 10512), pinning the worker HALF of the fix that mlx-gen
    /// #681 (this pin bump → `a5c1fcd`) delivered.
    ///
    /// The bug: on mlx-gen ≤ `6a10ae1`, `mlx-gen-anima`'s `load` rejected
    /// `spec.quantize.is_some() && !spec.adapters.is_empty()`. The worker defaults EVERY MLX model's
    /// tier to Q8 ([`resolve_quant`]'s `None` arm), so `spec.quantize` is `Some(..)` on the default
    /// path — which meant **every** Anima LoRA/LoKr generation failed at model load, at the DEFAULT
    /// tier, with no tier selection by the user. The only escape was an explicit bf16 pick
    /// (`mlxQuantize <= 0`). That combination — default tier + adapter — is exactly what no prior test
    /// exercised, which is how the model shipped with `supports_lora`, an official style LoRA, and a
    /// trainer whose output could not be loaded.
    ///
    /// This asserts the two worker-owned facts that made the bug fire, so a future change that
    /// reintroduces either is caught here. The end-to-end proof that the engine now ACCEPTS this spec
    /// lives in mlx-gen's real-weights `tests/packed_adapters.rs` (a Mac + weights are needed to run it,
    /// so it cannot live in this crate).
    #[test]
    fn anima_default_tier_with_adapter_builds_loadable_spec() {
        use gen_core::{AdapterKind, AdapterSpec, Quant};

        // 1. A default Anima request IS quantized — the premise that made the guard fire everywhere.
        let base_default =
            ImageRequest::from_payload(json!({ "model": "anima_base" }).as_object().unwrap());
        assert_eq!(
            resolve_quant(&base_default, None),
            (Some(Quant::Q8), Some(8)),
            "anima_base with no mlxQuantize must default to Q8 — the reason adding an adapter used to \
             fail at load on the DEFAULT tier"
        );
        // aesthetic/turbo ship manifest `mlx.quantize: 4`; still quantized, so still hit the guard.
        let aesthetic = ImageRequest::from_payload(
            json!({ "model": "anima_aesthetic", "modelManifestEntry": { "mlx": { "quantize": 4 } } })
                .as_object()
                .unwrap(),
        );
        assert_eq!(resolve_quant(&aesthetic, None), (Some(Quant::Q4), Some(4)));
        // Only an explicit bf16 opt-out escaped the bug.
        let bf16 = ImageRequest::from_payload(
            json!({ "model": "anima_base", "advanced": { "mlxQuantize": 0 } })
                .as_object()
                .unwrap(),
        );
        assert_eq!(resolve_quant(&bf16, None), (None, None));

        // 2. The LoadSpec the worker hands the engine carries a quant AND the adapter together — the
        //    exact `quantize.is_some() && !adapters.is_empty()` shape mlx-gen-anima rejected on
        //    `6a10ae1` and accepts on `a5c1fcd`.
        let (quant, _) = resolve_quant(&base_default, None);
        let adapters = vec![AdapterSpec::new(
            PathBuf::from("/tmp/anima-style-lora.safetensors"),
            1.0,
            AdapterKind::Lora,
        )];
        let spec = load_spec(PathBuf::from("/tmp/anima-q8-tier"), quant, adapters, None);
        assert!(
            spec.quantize.is_some(),
            "the default Anima tier is quantized, so the spec carries a quant"
        );
        assert!(
            !spec.adapters.is_empty(),
            "the adapter is present alongside the quant — the combination that used to fail"
        );
        // A dense-tier spec (the bf16 escape hatch) never carried a quant — documents the one path
        // that worked before the fix.
        let dense = load_spec(
            PathBuf::from("/tmp/anima-bf16-tier"),
            None,
            vec![AdapterSpec::new(
                PathBuf::from("/tmp/anima-style-lora.safetensors"),
                1.0,
                AdapterKind::Lora,
            )],
            None,
        );
        assert!(dense.quantize.is_none());
    }

    /// sc-10676: off-Mac candle dense-load resolution. [`anima_dense_split_files_dir`] descends into the
    /// `split_files/` subdir of the HF snapshot (the raw dense DiT tree, NOT a converted q4/q8/bf16 tier),
    /// falling back to the snapshot root when `split_files/` is absent (the candle loader accepts the
    /// parent; a partial download stays a loud load error). This is the off-Mac counterpart to the mac
    /// convert-at-install [`anima_tier_subdir`] — there are no tier subdirs off-Mac.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn anima_dense_split_files_dir_descends_into_split_files_else_root() {
        // Snapshot holds `split_files/diffusion_models/...` → resolve there (the loader reads it directly).
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("split_files").join("diffusion_models")).unwrap();
        assert_eq!(
            anima_dense_split_files_dir(root.to_path_buf()),
            root.join("split_files"),
            "descends into split_files/ when present"
        );
        // No `split_files/` yet (partial/absent download) → the snapshot root; the loader surfaces a
        // clear "not an Anima split_files dir" error rather than this silently pointing at a wrong dir.
        let tmp2 = tempfile::tempdir().unwrap();
        assert_eq!(
            anima_dense_split_files_dir(tmp2.path().to_path_buf()),
            tmp2.path().to_path_buf(),
            "falls back to the snapshot root when split_files/ is absent"
        );
    }

    /// sc-8746 on-device verify (MLX): drive the ACTUAL worker seam against a downloaded SceneWorks
    /// realvisxl-mlx turnkey — `standard_tier_subdir` resolves the `q4/` subdir from the tier root,
    /// then `crate::inference_runtime::load("sdxl", …)` with `Quant::Q4` loads the packed tier and renders. Asserts a
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
        // The worker resolution: a default `realvisxl` request (no mlxQuantize) prefers the q8 default
        // (sc-10726) but CLAMPS to installed — only the q4 tier was downloaded here (`--include "q4/*"`),
        // so it lands on q4/.
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
        let generator = crate::inference_runtime::load("sdxl", &spec).expect("load MLX sdxl provider on q4 tier");
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

    /// sc-10614: the candle SDXL lanes (edit / IP-Adapter) read DENSE weights, so a tiered-turnkey
    /// snapshot must resolve to its `bf16/` tier — its root holds no component tree, and the loader
    /// would find no `unet/`. Flat upstream diffusers snapshots (what `sdxl` and `realvisxl` fall
    /// back to today) must pass through untouched.
    #[test]
    fn dense_tier_subdir_descends_into_turnkeys_and_passes_flat_snapshots_through() {
        // Tiered turnkey: tier subdirs, no backbone at the root.
        let turnkey = tempfile::tempdir().unwrap();
        for tier in ["q4", "q8", "bf16"] {
            std::fs::create_dir_all(turnkey.path().join(tier).join("unet")).unwrap();
        }
        assert_eq!(
            dense_tier_subdir(turnkey.path().to_path_buf()),
            turnkey.path().join("bf16"),
            "an SDXL turnkey resolves to its dense bf16 tier, never a quantized one"
        );

        // Flat SDXL diffusers snapshot: `unet/` at the root.
        let flat_unet = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(flat_unet.path().join("unet")).unwrap();
        assert_eq!(
            dense_tier_subdir(flat_unet.path().to_path_buf()),
            flat_unet.path().to_path_buf(),
            "stabilityai/stable-diffusion-xl-base-1.0 and SG161222/RealVisXL_V5.0 pass through"
        );

        // Flat DiT snapshot: `transformer/` at the root — even alongside a stray `bf16/` dir.
        let flat_dit = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(flat_dit.path().join("transformer")).unwrap();
        std::fs::create_dir_all(flat_dit.path().join("bf16")).unwrap();
        assert_eq!(
            dense_tier_subdir(flat_dit.path().to_path_buf()),
            flat_dit.path().to_path_buf(),
            "a rooted component tree wins over a tier dir sitting beside it"
        );

        // Neither shape: return the root and let the loader raise a real load error.
        let bare = tempfile::tempdir().unwrap();
        assert_eq!(
            dense_tier_subdir(bare.path().to_path_buf()),
            bare.path().to_path_buf()
        );
    }

    /// sc-9092 (epic 9083 gap #3): the candle Lens lane no longer resolves a SEPARATE bf16 diffusers
    /// rehost (`SceneWorks/Lens{,-Turbo}`, the retired `candle_lens_repo`) — it packed-loads the SAME
    /// `SceneWorks/lens{,-turbo}-mlx` MLX turnkey the macOS path uses, routed through the shared
    /// `standard_tier_subdir` (`lens`/`lens_turbo` opt in via `mlx.standardTierLayout`, exactly like the
    /// MLX lane). This proves the shared tier resolver picks the requested q4/q8/bf16 subdir of a Lens
    /// turnkey snapshot off-Mac — the candle-lane sibling of the SD3.5 `standard_tier_subdir` tests
    /// above — so the retired resolver is fully replaced by the standard machinery.
    #[test]
    fn krea_model_subdir_selects_tier_and_falls_back_to_downloaded() {
        let krea_request = |bits: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": "krea_2_raw", "advanced": { "mlxQuantize": bits } })
                    .as_object()
                    .unwrap(),
            )
        };
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Krea turnkey: packed q4/q8 single-file transformer + dense sharded bf16 (index.json only).
        seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "bf16", "diffusion_pytorch_model.safetensors.index.json");

        // q8-default (no selection) / q4 (bits<=4) / bf16 (bits<=0) each resolve to their tier subdir.
        assert_eq!(
            krea_model_subdir(root, &krea_request(json!(null))),
            root.join("q8")
        );
        assert_eq!(
            krea_model_subdir(root, &krea_request(json!(4))),
            root.join("q4")
        );
        assert_eq!(
            krea_model_subdir(root, &krea_request(json!(0))),
            root.join("bf16")
        );

        // Regression guard (epic 9992): with ONLY the `bf16/` training-base tier downloaded (the Path-1
        // unify scenario), a q8-default generation must fall back to the present bf16 tier — NOT the repo
        // root (which has no `tokenizer/`, the reported "tokenizer: No such file or directory" load error).
        let bf16_only = tempfile::tempdir().unwrap();
        seed_tier(
            bf16_only.path(),
            "bf16",
            "diffusion_pytorch_model.safetensors.index.json",
        );
        assert_eq!(
            krea_model_subdir(bf16_only.path(), &krea_request(json!(null))),
            bf16_only.path().join("bf16"),
            "q8-default must fall back to the only downloaded tier (bf16), not the repo root"
        );
    }

    /// Seed a diffusers-shaped tier: a backbone, a `model_index.json` declaring `declared` as
    /// `[library, class]` components, and an on-disk dir for each of `on_disk`. The index also carries
    /// the three NON-component value shapes a real one does (see [`is_component_entry`]) so every test
    /// through this helper proves they aren't mistaken for required dirs.
    fn seed_diffusers_tier(root: &Path, tier: &str, declared: &[&str], on_disk: &[&str]) {
        seed_tier(root, tier, "diffusion_pytorch_model.safetensors");
        let dir = root.join(tier);
        let mut index = serde_json::Map::new();
        index.insert("_class_name".to_owned(), json!("Krea2Pipeline"));
        // Config array + scalar (krea's real `text_encoder_select_layers` / `patch_size`), and an
        // ABSENT optional component (realvisxl's real `feature_extractor`). None is a required dir.
        index.insert("text_encoder_select_layers".to_owned(), json!([2, 5, 8]));
        index.insert("patch_size".to_owned(), json!(2));
        index.insert("feature_extractor".to_owned(), json!([null, null]));
        for component in declared {
            index.insert((*component).to_owned(), json!(["transformers", "SomeClass"]));
        }
        std::fs::write(
            dir.join("model_index.json"),
            serde_json::to_vec(&Value::Object(index)).unwrap(),
        )
        .unwrap();
        for component in on_disk {
            let component_dir = dir.join(component);
            std::fs::create_dir_all(&component_dir).unwrap();
            std::fs::write(component_dir.join("config.json"), b"{}").unwrap();
        }
    }

    /// The Krea tier component set, as the real `q8/model_index.json` declares it.
    const KREA_COMPONENTS: &[&str] = &["transformer", "tokenizer", "text_encoder", "vae", "scheduler"];

    /// sc-12279 (issue #850): a TORN tier — backbone landed, `tokenizer/` did not — must not
    /// short-circuit the fallback chain. Before this, `present()` accepted a tier on its transformer
    /// alone, so the q8 default resolved to the torn `q8/` and the loader died on
    /// `tokenizer: No such file or directory (os error 2)` even though a complete `bf16/` was installed.
    #[test]
    fn krea_tier_probe_skips_a_torn_tier_for_a_complete_sibling() {
        let krea_request = |bits: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": "krea_2_raw", "advanced": { "mlxQuantize": bits } })
                    .as_object()
                    .unwrap(),
            )
        };
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // q8 (the default tier) is torn: everything but `tokenizer/`. bf16 is complete.
        seed_diffusers_tier(
            root,
            "q8",
            KREA_COMPONENTS,
            &["text_encoder", "vae", "scheduler"],
        );
        seed_diffusers_tier(
            root,
            "bf16",
            KREA_COMPONENTS,
            &["tokenizer", "text_encoder", "vae", "scheduler"],
        );
        assert_eq!(
            krea_model_subdir(root, &krea_request(json!(null))),
            root.join("bf16"),
            "the q8 default is torn (no tokenizer/), so the chain must land on the complete bf16 tier"
        );
        // An EXPLICIT pick of the torn tier is also redirected — the user asked for a tier that cannot
        // load, and a working render beats an os-error-2 on a request we can serve.
        assert_eq!(
            krea_model_subdir(root, &krea_request(json!(8))),
            root.join("bf16"),
            "an explicit q8 pick still skips the torn tier for the complete sibling"
        );
    }

    /// sc-12279: with NO complete tier, the torn one is still returned — pass 2 of the chain preserves
    /// the pre-sc-12279 result exactly. The user gets the same loader error as before (they have no
    /// loadable tier), never a silent no-op, and the resolver logs which tier is short.
    #[test]
    fn krea_tier_probe_still_returns_a_torn_tier_when_it_is_all_there_is() {
        let request = ImageRequest::from_payload(
            json!({ "model": "krea_2_raw", "advanced": {} }).as_object().unwrap(),
        );
        let tmp = tempfile::tempdir().unwrap();
        seed_diffusers_tier(
            tmp.path(),
            "q8",
            KREA_COMPONENTS,
            &["text_encoder", "vae", "scheduler"],
        );
        assert_eq!(
            krea_model_subdir(tmp.path(), &request),
            tmp.path().join("q8"),
            "no complete tier: the torn tier still resolves (unchanged behavior), not the repo root"
        );
    }

    /// sc-12279: [`tier_components_present`] reads the tier's OWN `model_index.json`, and must not
    /// mistake diffusers' non-component value shapes for required dirs. Each of these is live in a
    /// shipping turnkey, so getting any wrong would fail a perfectly good tier.
    #[test]
    fn tier_components_present_reads_model_index_and_ignores_non_components() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Complete: every declared component on disk. The `[null, null]` optional, the config array and
        // the config scalar that `seed_diffusers_tier` always writes must NOT be required as dirs —
        // this asserting true IS the proof (no `feature_extractor/` dir exists).
        seed_diffusers_tier(
            root,
            "q8",
            KREA_COMPONENTS,
            &["tokenizer", "text_encoder", "vae", "scheduler"],
        );
        assert!(tier_components_present(&root.join("q8")));

        // Torn: `tokenizer/` declared but absent.
        seed_diffusers_tier(root, "q4", KREA_COMPONENTS, &["text_encoder", "vae", "scheduler"]);
        assert!(!tier_components_present(&root.join("q4")));

        // A component dir holding ONLY an AppleDouble sidecar has no tokenizer (SceneWorks#1333).
        let sidecar = root.join("q4").join("tokenizer");
        std::fs::create_dir_all(&sidecar).unwrap();
        std::fs::write(sidecar.join("._tokenizer.json"), b"x").unwrap();
        assert!(
            !tier_components_present(&root.join("q4")),
            "a hidden sidecar must not satisfy a declared component"
        );
        std::fs::write(sidecar.join("tokenizer.json"), b"{}").unwrap();
        assert!(tier_components_present(&root.join("q4")));

        // No `model_index.json` at all (flat unified turnkeys — SenseNova-U1 MoT roots its weights +
        // `tokenizer.json` directly in the tier dir): nothing to verify, so the backbone probe rules.
        seed_tier(root, "bf16", "model.safetensors");
        assert!(
            tier_components_present(&root.join("bf16")),
            "a tier with no model_index.json has no component tree to check"
        );
    }

    // ---- sc-12279 generalized to the no-`model_index.json` families (SANA / Boogu / Anima) ----
    // These turnkeys ship no `model_index.json`, so the shared `tier_components_present` guard is a
    // no-op for them and a torn tier used to short-circuit the chain and crash the loader on the first
    // missing component. Each resolver now routes through a concrete per-family completeness predicate.

    /// Seed a SANA MLX tier (`SceneWorks/Sana_*_mlx` layout): transformer + (when `complete`) the DC-AE
    /// VAE and the Gemma-2 text encoder with its bundled tokenizer. No `model_index.json` (as shipped).
    fn seed_sana_tier(root: &Path, tier: &str, complete: bool) {
        let dir = root.join(tier);
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        std::fs::write(dir.join("transformer/diffusion_pytorch_model.safetensors"), b"x").unwrap();
        if complete {
            std::fs::create_dir_all(dir.join("vae")).unwrap();
            std::fs::write(dir.join("vae/diffusion_pytorch_model.safetensors"), b"x").unwrap();
            std::fs::create_dir_all(dir.join("text_encoder")).unwrap();
            std::fs::write(dir.join("text_encoder/gemma-2-2b-it.safetensors"), b"x").unwrap();
            std::fs::write(dir.join("text_encoder/tokenizer.json"), b"{}").unwrap();
        }
    }

    /// A torn SANA tier (transformer only, TE/VAE absent) must fall through to a complete sibling rather
    /// than reaching the loader, which would die on the missing Gemma text encoder. SANA ships no
    /// `model_index.json`, so this is caught by the concrete `sana_tier_complete` check, not the no-op
    /// `tier_components_present` guard.
    #[test]
    fn sana_torn_tier_falls_through_to_a_complete_sibling() {
        let request = |bits: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": "sana_1600m", "advanced": { "mlxQuantize": bits } })
                    .as_object()
                    .unwrap(),
            )
        };
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_sana_tier(root, "q8", false); // torn: transformer only
        seed_sana_tier(root, "bf16", true); // complete
        assert_eq!(
            standard_tier_subdir(root, &request(json!(null))),
            root.join("bf16"),
            "the torn q8 default must skip to the complete bf16 tier, not crash on the missing Gemma TE"
        );
        assert_eq!(
            standard_tier_subdir(root, &request(json!(8))),
            root.join("bf16"),
            "an explicit q8 pick is redirected too — a working render beats an os-error-2 we can avoid"
        );
    }

    /// Regression contract: with NO complete SANA tier, the torn one still resolves (pass 2), unchanged
    /// from before — the user gets the same loader error they would have, never a silent no-op.
    #[test]
    fn sana_torn_tier_is_returned_when_it_is_all_there_is() {
        let request = ImageRequest::from_payload(
            json!({ "model": "sana_1600m", "advanced": {} }).as_object().unwrap(),
        );
        let tmp = tempfile::tempdir().unwrap();
        seed_sana_tier(tmp.path(), "q8", false);
        assert_eq!(
            standard_tier_subdir(tmp.path(), &request),
            tmp.path().join("q8"),
            "no complete tier: the torn tier still resolves, not the repo root"
        );
    }

    /// Non-SANA standard turnkeys (flux/qwen/…) are unaffected: they DO ship `model_index.json`, so
    /// their completeness stays `tier_components_present` — the added SANA branch never touches them.
    #[test]
    fn non_sana_standard_turnkey_ignores_the_sana_completeness_branch() {
        let request = ImageRequest::from_payload(
            json!({ "model": "flux_dev", "advanced": {} }).as_object().unwrap(),
        );
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // A flux tier with only a transformer and no text_encoder dir would FAIL sana_tier_complete, but
        // flux is not SANA — it resolves on its backbone/model_index exactly as before.
        seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
        assert_eq!(standard_tier_subdir(root, &request), root.join("q8"));
    }

    /// Seed a Boogu tier folder (`<variant>` / `<variant>-q4` / `<variant>-bf16`): packed transformer +
    /// its config, and (when `complete`) the Qwen3-VL `mllm/` with tokenizer and the VAE. No index.
    fn seed_boogu_tier(root: &Path, folder: &str, complete: bool) {
        let dir = root.join(folder);
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        std::fs::write(dir.join("transformer/diffusion_pytorch_model.safetensors"), b"x").unwrap();
        std::fs::write(dir.join("transformer/config.json"), b"{}").unwrap();
        if complete {
            std::fs::create_dir_all(dir.join("mllm")).unwrap();
            std::fs::write(dir.join("mllm/model.safetensors"), b"x").unwrap();
            std::fs::write(dir.join("mllm/tokenizer.json"), b"{}").unwrap();
            std::fs::create_dir_all(dir.join("vae")).unwrap();
            std::fs::write(dir.join("vae/diffusion_pytorch_model.safetensors"), b"x").unwrap();
        }
    }

    /// A torn Boogu tier (transformer present, `mllm/tokenizer.json` + VAE absent) must fall through to a
    /// complete sibling rather than crash the loader on the first-read `mllm/tokenizer.json`.
    #[test]
    fn boogu_torn_tier_falls_through_to_a_complete_sibling() {
        let request = ImageRequest::from_payload(
            json!({ "model": "boogu_image", "advanced": {} }).as_object().unwrap(),
        );
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // `base` is the Q8 default folder (torn); `base-bf16` is complete.
        seed_boogu_tier(root, "base", false);
        seed_boogu_tier(root, "base-bf16", true);
        assert_eq!(
            boogu_model_subdir(root, &request),
            root.join("base-bf16"),
            "the torn Q8 `base/` must skip to the complete `base-bf16`, not crash on `mllm/tokenizer.json`"
        );
    }

    /// `resolved_tier_is_complete` (the pre-flight completeness dispatcher behind the friendly
    /// `mlx_weights_gap` message) reports a torn SANA tier as incomplete and a whole one as complete.
    #[cfg(target_os = "macos")]
    #[test]
    fn resolved_tier_is_complete_flags_a_torn_sana_tier() {
        let request = ImageRequest::from_payload(
            json!({ "model": "sana_1600m", "advanced": {} }).as_object().unwrap(),
        );
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_sana_tier(root, "q8", false);
        seed_sana_tier(root, "bf16", true);
        assert!(!resolved_tier_is_complete(&request, &root.join("q8")));
        assert!(resolved_tier_is_complete(&request, &root.join("bf16")));
    }

    /// Seed an Anima tier (`bf16/ q8/ q4/`, `split_files` shape): the DiT + (when `complete`) the dense
    /// text encoder and the VAE. Tokenizers are vendored into the binary, so none is on disk. No index.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    fn seed_anima_tier(root: &Path, tier: &str, complete: bool) {
        let dir = root.join(tier);
        std::fs::create_dir_all(dir.join("diffusion_models")).unwrap();
        std::fs::write(dir.join("diffusion_models/anima-base-v1.0.safetensors"), b"x").unwrap();
        if complete {
            std::fs::create_dir_all(dir.join("text_encoders")).unwrap();
            std::fs::write(dir.join("text_encoders/qwen_3_06b_base.safetensors"), b"x").unwrap();
            std::fs::create_dir_all(dir.join("vae")).unwrap();
            std::fs::write(dir.join("vae/qwen_image_vae.safetensors"), b"x").unwrap();
        }
    }

    /// A torn Anima tier (DiT present, text-encoder/VAE absent) must fall through to a complete sibling
    /// rather than reaching the loader, which would die on the missing `text_encoders/…` (mlx-gen-anima).
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn anima_torn_tier_falls_through_to_a_complete_sibling() {
        let request = ImageRequest::from_payload(
            json!({ "model": "anima_base", "advanced": {} }).as_object().unwrap(),
        );
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_anima_tier(root, "q8", false); // torn: DiT only
        seed_anima_tier(root, "bf16", true); // complete
        assert_eq!(
            anima_tier_subdir(root, &request),
            root.join("bf16"),
            "the torn q8 default must skip to the complete bf16 tier, not crash on the missing text encoder"
        );
    }

    /// sc-10845 (epic 10721): [`krea_model_subdir`] — the last bespoke tier resolver — routes its DEFAULT
    /// through the shared, floor-aware [`preferred_tier`]`(bits, `[`min_quality_floor`]`(request))`, exactly
    /// like [`standard_tier_subdir`] / [`anima_tier_subdir`] / [`ideogram_model_subdir`] /
    /// [`boogu_model_subdir`], so a floored model's picker-less default clamps UP to `mlx.minQualityTier`
    /// (capped by installed) instead of blindly landing on the q8 default. No shipping Krea model declares
    /// a floor today, so every current model is byte-identical to the prior q8 default; this pins the
    /// generality. Mirrors [`standard_tier_subdir_clamps_default_to_the_floor`].
    #[test]
    fn krea_default_tier_clamps_to_the_floor() {
        // A krea request with the given `bits` (null = no explicit pick) and an optional per-model quality
        // floor forwarded via `modelManifestEntry.mlx.minQualityTier`.
        let floored = |bits: serde_json::Value, floor: Option<&str>| {
            let payload = match floor {
                Some(f) => json!({
                    "model": "krea_2_raw",
                    "advanced": { "mlxQuantize": bits },
                    "modelManifestEntry": { "mlx": { "minQualityTier": f } },
                }),
                None => json!({
                    "model": "krea_2_raw",
                    "advanced": { "mlxQuantize": bits },
                }),
            };
            ImageRequest::from_payload(payload.as_object().unwrap())
        };
        // Krea turnkey carries packed q4/q8 + dense sharded bf16.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "bf16", "diffusion_pytorch_model.safetensors.index.json");

        // Floor bf16, no explicit pick → the dense bf16 (raised above the q8 default).
        assert_eq!(
            krea_model_subdir(root, &floored(json!(null), Some("bf16"))),
            root.join("bf16"),
            "floored Krea default → the bf16 floor tier (raised above the q8 default)"
        );
        // Floor q8 (at the default) → q8, unchanged (the floor only ever RAISES).
        assert_eq!(
            krea_model_subdir(root, &floored(json!(null), Some("q8"))),
            root.join("q8")
        );
        // An explicit below-floor q4 pick is HONORED even against a bf16 floor (the worker never overrides
        // a deliberate quant choice — the web surfaces the advisory instead).
        assert_eq!(
            krea_model_subdir(root, &floored(json!(4), Some("bf16"))),
            root.join("q4")
        );
        // Non-floored default is unchanged — the q8 default still stands.
        assert_eq!(
            krea_model_subdir(root, &floored(json!(null), None)),
            root.join("q8")
        );

        // Floor capped by installed: bf16 floor but only q8 on disk → q8.
        let only_q8 = tempfile::tempdir().unwrap();
        seed_tier(only_q8.path(), "q8", "diffusion_pytorch_model.safetensors");
        assert_eq!(
            krea_model_subdir(only_q8.path(), &floored(json!(null), Some("bf16"))),
            only_q8.path().join("q8"),
            "bf16 floor with only q8 installed → q8 (a floor tier not on disk falls to the best installed)"
        );
    }

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

        // A/B tier toggle: default (q8, sc-10726) / mlxQuantize:8 (q8) / mlxQuantize:0 (bf16) each
        // resolve to their tier subdir — the default now prefers the clean q8 tier (clamped to
        // installed), matching `resolve_quant`'s Q8 default.
        assert_eq!(
            standard_tier_subdir(root, &lens_request(json!(null))),
            root.join("q8")
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

        // Ideogram turnkey (`SceneWorks/ideogram-4-mlx`): q4 + q8 tiers. `ideogram_model_subdir` probes
        // `<tier>/transformer/model.safetensors`.
        {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path();
            seed_tier(root, "q4", "model.safetensors");
            seed_tier(root, "q8", "model.safetensors");
            for model in ["ideogram_4", "ideogram_4_turbo"] {
                // Default (no mlxQuantize) → q8 when installed (epic 10721 / sc-10726).
                assert_eq!(
                    ideogram_model_subdir(root, &model_request(model, json!(null))),
                    root.join("q8")
                );
                // mlxQuantize:8 → q8 when present.
                assert_eq!(
                    ideogram_model_subdir(root, &model_request(model, json!(8))),
                    root.join("q8")
                );
                // An explicit Q4 pick is still honored.
                assert_eq!(
                    ideogram_model_subdir(root, &model_request(model, json!(4))),
                    root.join("q4")
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

    /// sc-10777 (epic 10721): [`ideogram_model_subdir`] / [`boogu_model_subdir`] route their DEFAULT tier
    /// through the shared, floor-aware [`preferred_tier`]`(bits, `[`min_quality_floor`]`(request))` — the
    /// same clamp [`standard_tier_subdir`] / [`anima_tier_subdir`] use — so a floored model's picker-less
    /// default clamps UP to `mlx.minQualityTier` (capped by installed) instead of blindly landing on the
    /// per-family q8/Q8 default. No shipping Ideogram/Boogu model declares a floor today, so every current
    /// model is byte-identical to the prior default; this pins the generality so a future floored model
    /// can't silently regress below its floor on the worker default path. Mirrors
    /// [`standard_tier_subdir_clamps_default_to_the_floor`].
    #[test]
    fn ideogram_boogu_default_tier_clamps_to_the_floor() {
        // A request for `model` with the given `bits` (null = no explicit pick) and an optional per-model
        // quality floor forwarded via `modelManifestEntry.mlx.minQualityTier`.
        let floored = |model: &str, bits: serde_json::Value, floor: Option<&str>| {
            let payload = match floor {
                Some(f) => json!({
                    "model": model,
                    "advanced": { "mlxQuantize": bits },
                    "modelManifestEntry": { "mlx": { "minQualityTier": f } },
                }),
                None => json!({
                    "model": model,
                    "advanced": { "mlxQuantize": bits },
                }),
            };
            ImageRequest::from_payload(payload.as_object().unwrap())
        };

        // --- Boogu: a bf16 floor RAISES the picker-less default above the packed Q8 (the meaningful case,
        // since Boogu's turnkey carries a bf16 tier above its Q8 default). ---
        {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path();
            seed_tier(root, "base", "diffusion_pytorch_model.safetensors");
            seed_tier(root, "base-q4", "diffusion_pytorch_model.safetensors");
            // bf16 ships as the dense diffusers tree (SHARDED → only the `.index.json`), like the loader.
            seed_tier(
                root,
                "base-bf16",
                "diffusion_pytorch_model.safetensors.index.json",
            );
            // Floor bf16, no explicit pick → the dense `base-bf16` (raised above the Q8 default).
            assert_eq!(
                boogu_model_subdir(root, &floored("boogu_image", json!(null), Some("bf16"))),
                root.join("base-bf16"),
                "floored Boogu default → the bf16 floor tier (raised above the packed Q8 default)"
            );
            // Floor q8 (at the default) → the packed Q8 `base/`, unchanged (the floor only ever RAISES).
            assert_eq!(
                boogu_model_subdir(root, &floored("boogu_image", json!(null), Some("q8"))),
                root.join("base")
            );
            // An explicit below-floor q4 pick is HONORED even against a bf16 floor (the worker never
            // overrides a deliberate quant choice — the web surfaces the advisory instead).
            assert_eq!(
                boogu_model_subdir(root, &floored("boogu_image", json!(4), Some("bf16"))),
                root.join("base-q4")
            );
            // Non-floored default is unchanged — the packed Q8 default still stands (acceptance #3).
            assert_eq!(
                boogu_model_subdir(root, &floored("boogu_image", json!(null), None)),
                root.join("base")
            );
        }
        // Boogu floor capped by installed: bf16 floor but only the packed Q8 `base/` on disk → Q8.
        {
            let only_q8 = tempfile::tempdir().unwrap();
            seed_tier(only_q8.path(), "base", "diffusion_pytorch_model.safetensors");
            assert_eq!(
                boogu_model_subdir(
                    only_q8.path(),
                    &floored("boogu_image", json!(null), Some("bf16"))
                ),
                only_q8.path().join("base"),
                "bf16 floor with only Q8 installed → Q8 (a floor tier not on disk falls to the best installed)"
            );
        }

        // --- Ideogram: the turnkey carries only `q4/`/`q8/` (bf16 is a separate repo) and the default
        // already prefers q8, so the floor routing is inert for every in-turnkey floor — but it must still
        // resolve correctly and stay capped by installed. ---
        {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path();
            seed_tier(root, "q4", "model.safetensors");
            seed_tier(root, "q8", "model.safetensors");
            // Floor q8, no explicit pick → q8 (the default already prefers q8; confirms the floor routing
            // resolves through the shared helper).
            assert_eq!(
                ideogram_model_subdir(root, &floored("ideogram_4", json!(null), Some("q8"))),
                root.join("q8")
            );
            // A bf16 floor has NO in-turnkey tier → falls through the clean q8 → q4 chain to the best
            // installed (q8), never erroring on the absent bf16.
            assert_eq!(
                ideogram_model_subdir(root, &floored("ideogram_4", json!(null), Some("bf16"))),
                root.join("q8"),
                "bf16 floor with no in-turnkey bf16 tier → best installed (q8)"
            );
            // An explicit below-floor q4 pick is HONORED even against a q8 floor.
            assert_eq!(
                ideogram_model_subdir(root, &floored("ideogram_4", json!(4), Some("q8"))),
                root.join("q4")
            );
            // Non-floored default is unchanged — q8 when installed (acceptance #3).
            assert_eq!(
                ideogram_model_subdir(root, &floored("ideogram_4", json!(null), None)),
                root.join("q8")
            );
        }
        // Ideogram floor capped by installed: q8 floor but only the shipped q4 on disk → q4.
        {
            let only_q4 = tempfile::tempdir().unwrap();
            seed_tier(only_q4.path(), "q4", "model.safetensors");
            assert_eq!(
                ideogram_model_subdir(only_q4.path(), &floored("ideogram_4", json!(null), Some("q8"))),
                only_q4.path().join("q4"),
                "q8 floor with only q4 installed → q4 (a floor tier not on disk falls to the best installed)"
            );
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
        let requested = resolve_quant(&req, Some(&resolved));
        assert_eq!(requested, (None, None), "bf16 request derives dense");
        let (quant, bits) =
            reconcile_resolved_tier_quant(requested, &resolved, true, "sd3_5_large", "job1", "mlx");
        assert_eq!((quant, bits), (Some(Quant::Q4), Some(4)));
    }

    /// sc-9362 (F-018 follow-up): the dense-TE transformer tier the request asks for is derived from
    /// `advanced.mlxQuantize` exactly like [`standard_tier_subdir`] — no explicit pick → `q8` (sc-10726),
    /// `<=0 → bf16 (None)`, `>4 → q8`, else `q4` — regardless of the always-`None` load quant
    /// `resolve_quant` returns for dense-TE. This is what reconcile compares the resolved tier against so
    /// a straight default job (resolving the q8 tier) isn't flagged as a spurious downgrade.
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
        // No selection → the q8 default (matches standard_tier_subdir's preferred, sc-10726).
        assert_eq!(dense_te_requested_tier_bits(&default), Some(8));
        assert_eq!(dense_te_requested_tier_bits(&req(json!(0))), None); // bf16 opt-out
        assert_eq!(dense_te_requested_tier_bits(&req(json!(4))), Some(4)); // explicit q4 honored
        assert_eq!(dense_te_requested_tier_bits(&req(json!(8))), Some(8));
        assert_eq!(dense_te_requested_tier_bits(&req(json!("8"))), Some(8)); // numeric-string
        assert_eq!(dense_te_requested_tier_bits(&req(json!(-1))), None);
    }

    /// sc-9362 (F-018 follow-up) + sc-10726: a straight (no-fallback) dense-TE default job — its q8
    /// transformer tier is downloaded and resolves as the new q8 default — records the ACTUAL
    /// transformer tier (Q8) while keeping the load quant `None` (the dense bf16 TE is never
    /// re-quantized). The requested tier ([`dense_te_requested_tier_bits`], now q8 by default) MATCHES
    /// the resolved q8 tier, so reconcile pass-throughs it with no spurious `quant_tier_downgraded`.
    #[test]
    fn dense_te_no_fallback_records_transformer_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // The klein q4 + q8 transformer tiers are present (dense bf16 TE lives alongside in each tier);
        // a default job takes the q8 default.
        for tier in ["q4", "q8"] {
            let dir = root.join(tier).join("transformer");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), b"x").unwrap();
        }

        let req = ImageRequest::from_payload(
            json!({ "model": "flux2_klein_9b", "advanced": {} })
                .as_object()
                .unwrap(),
        );
        // The tier resolver lands on the q8 default tier (no fallback).
        let resolved = standard_tier_subdir(root, &req);
        assert_eq!(resolved, root.join("q8"));
        // resolve_quant keeps dense-TE at `(None, None)` (never re-quantize the dense bf16 TE)…
        assert_eq!(resolve_quant(&req, Some(&resolved)), (None, None));
        // …but reconcile against the REQUESTED transformer tier (q8) records the real transformer
        // precision (Q8) with the load quant still `None`, and — since resolved == requested — with no
        // downgrade (the requested/resolved tiers match, so it's a clean pass-through).
        let requested_for_reconcile = (None, dense_te_requested_tier_bits(&req));
        assert_eq!(requested_for_reconcile, (None, Some(8)));
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
            (None, Some(8)),
            "records the actual q8 transformer tier, load quant stays None"
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

/// sc-10733: the shared capability-downtier chooser — walk installed tiers from the resolved default
/// down to the quality floor, pick the highest that fits, reject only when nothing fits. Pure decision,
/// compiled on both lanes (the candle vram gate + the MLX fit gate both feed it their own [`TierFit`]).
#[cfg(all(test, any(target_os = "macos", feature = "backend-candle")))]
mod capability_downtier_tests {
    use super::*;

    /// sc-12425 — **a resolved ConvRot load must be gated by its OWN tier, never aliased to q8.**
    ///
    /// The defect this kills: `generate_candle_stream` handed ConvRot to the BITS-derived
    /// `vram_gate::requested_tier_key`. A ConvRot request carries no `mlxQuantize` (the picker sends
    /// `advanced.convRot: true` — see [`wants_krea_convrot`]), so it hit that function's `None => "q8"`
    /// arm, and a **measured 42.9 GB** render (sc-12381) was sized against q8's 35.9 + 2.0 = 37.9 GB row
    /// — admitting loads that OOM.
    ///
    /// The second assertion pins the ALIASING itself, so this test says why the first one matters
    /// instead of just asserting a constant: q8's row is what was actually being read, which is why
    /// "just correct the manifest row" would have fixed nothing.
    ///
    /// Lives in `capability_downtier_tests` (not the `#[cfg(target_os = "macos")]`
    /// `quant_tier_reconcile_tests`, where it would compile out on the candle lane), but carries its OWN
    /// candle-only gate: it calls [`gate_tier_key`] + `crate::vram_gate`, both `not(macos)` — while this
    /// module is `any(macos, candle)`, so without the attribute below it fails to compile on the MLX
    /// build (E0433, no `vram_gate`).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn gate_tier_key_names_convrot_by_identity_never_q8() {
        // The real shape: the ConvRot base surface IS the bf16 dir, and the request carries no bits.
        let convrot_base = std::path::Path::new("/models/krea-2-turbo-mlx/bf16");
        let advanced = serde_json::json!({ "convRot": true })
            .as_object()
            .expect("object")
            .clone();
        let entry = serde_json::Map::new();

        assert_eq!(
            gate_tier_key(true, convrot_base, &advanced, &entry, false),
            INT8_CONVROT_TIER,
            "a resolved ConvRot load must be named by tier identity, not sized against another tier"
        );

        // THE ALIASING, pinned. If this stops being q8, the aliasing changed shape and sc-12425 needs
        // re-reading — do not just bump it to whatever it now returns.
        assert_eq!(
            crate::vram_gate::requested_tier_key(&advanced, &entry, false),
            "q8",
            "a ConvRot request carries no mlxQuantize, so the bits-derived key aliases it to q8 — the \
             under-prediction sc-12425 fixes"
        );

        // The non-ConvRot path is untouched: the on-disk tier still wins (sc-12090).
        assert_eq!(
            gate_tier_key(false, convrot_base, &advanced, &entry, false),
            "bf16"
        );
    }

    /// sc-12090 / sc-12829: the basename → tier-NAME reader the candle VRAM gate keys off —
    /// `generate_candle_stream` → `gate_tier_key` fallback and the sc-12090 disk-probe path
    /// ([`resolve_tier_dir`] / `installed_tier_keys`) both resolve the on-disk tier through
    /// [`tier_key_from_resolved_dir`]. That fn is `any(macos, candle)`, so its coverage lives HERE (this
    /// cross-lane module), NOT in the `#[cfg(target_os = "macos")]` `quant_tier_reconcile_tests` where it
    /// would compile out on the candle lane — leaving the exact basename→tier mapping the gate depends on
    /// unexercised on `candle-worker`. A regression (e.g. a new Boogu `<variant>-<tier>` shape) now goes
    /// RED on both lanes, not just the Mac runner. (`gate_tier_key`/`installed_tier_keys` are candle-only,
    /// so they're named as plain code — only the cross-lane fns exist on both of this module's lanes.)
    #[test]
    fn tier_key_from_resolved_dir_reads_the_on_disk_tier() {
        let root = std::path::Path::new("/models/krea-2-turbo-mlx");
        assert_eq!(tier_key_from_resolved_dir(&root.join("q4")), Some("q4"));
        assert_eq!(tier_key_from_resolved_dir(&root.join("q8")), Some("q8"));
        assert_eq!(tier_key_from_resolved_dir(&root.join("bf16")), Some("bf16"));
        // Boogu `<variant>-<tier>` suffix + the bare `<variant>` packed Q8 default.
        assert_eq!(tier_key_from_resolved_dir(&root.join("base-q4")), Some("q4"));
        assert_eq!(
            tier_key_from_resolved_dir(&root.join("turbo-bf16")),
            Some("bf16")
        );
        assert_eq!(tier_key_from_resolved_dir(&root.join("edit")), Some("q8"));
        // Unrecognized basename (repo root / modelPath) → None; the gate keeps its manifest key.
        assert_eq!(tier_key_from_resolved_dir(root), None);
    }

    fn too_big(needed: f64, avail: f64) -> TierFit {
        TierFit::TooBig {
            needed_gb: needed,
            available_gb: avail,
        }
    }

    #[test]
    fn keeps_the_default_when_it_fits() {
        // Q8 default fits → Keep, even though a smaller q4 is also installed and would fit.
        let candidates = [("q8", TierFit::Fits), ("q4", TierFit::Fits)];
        assert_eq!(choose_downtier("q8", &candidates), DowntierPick::Keep);
    }

    #[test]
    fn downtiers_to_the_highest_installed_tier_that_fits() {
        // Q8 default won't fit, q4 does → downtier to q4 (acceptance #2).
        let candidates = [("q8", too_big(33.0, 30.0)), ("q4", TierFit::Fits)];
        assert_eq!(
            choose_downtier("q8", &candidates),
            DowntierPick::Downtier("q4")
        );
        // bf16 default won't fit, q8 is the highest that does (q4 also fits but is lower fidelity).
        let three = [
            ("bf16", too_big(72.0, 40.0)),
            ("q8", TierFit::Fits),
            ("q4", TierFit::Fits),
        ];
        assert_eq!(
            choose_downtier("bf16", &three),
            DowntierPick::Downtier("q8")
        );
    }

    #[test]
    fn rejects_naming_the_smallest_evaluated_tier_when_nothing_fits() {
        // Neither q8 nor q4 fits → reject, naming q4 (the smallest / least-demanding tried) + its need.
        let candidates = [("q8", too_big(33.0, 10.0)), ("q4", too_big(28.0, 10.0))];
        assert_eq!(
            choose_downtier("q8", &candidates),
            DowntierPick::Reject {
                tier: "q4",
                needed_gb: 28.0,
                available_gb: 10.0,
            }
        );
    }

    #[test]
    fn floor_wins_over_downtier_via_the_candidate_list() {
        // A floor-q8 model: the caller filters q4 OUT of the candidate list (rank < floor), so even when
        // only q8 is offered and it won't fit, we REJECT rather than silently rendering q4 (acceptance #5).
        let floored = [("q8", too_big(33.0, 20.0))];
        assert_eq!(
            choose_downtier("q8", &floored),
            DowntierPick::Reject {
                tier: "q8",
                needed_gb: 33.0,
                available_gb: 20.0,
            }
        );
    }

    #[test]
    fn empty_candidates_keep_the_default() {
        // No installed candidate in range (defensive — the default itself is normally present) → Keep,
        // deferring to the plain gate.
        assert_eq!(choose_downtier("q8", &[]), DowntierPick::Keep);
    }

    /// sc-11042 (epic 11037 SC#5) × sc-10733: the capability clamp NEVER downtiers a selected NVFP4
    /// tier. Downtiering it to q4/q8 would silently swap the numerics of an explicitly-picked tier —
    /// the exact creative-choice violation SC#5 forbids.
    ///
    /// This is load-bearing BECAUSE nvfp4 does not escape the clamp via `mlxQuantizeExplicit`: the web
    /// emits that flag only inside the `tierQuantize(quantTier) !== null` bits branch, and nvfp4 has no
    /// honest `mlxQuantize` integer, so an nvfp4 job reaches the clamp with `explicit_pick == false`.
    /// What saves it is that nvfp4 is UNRANKABLE on purpose (`tier_quality_rank` ⇒ 0): it is a distinct
    /// numeric regime, not a rung on the bf16/q8/q4 ladder, so no tier is ever in `[floor, nvfp4]` and
    /// the chooser keeps it. These two facts are a silent pair — pin them together, since making nvfp4
    /// rankable would quietly arm the downtier.
    #[test]
    fn nvfp4_tier_is_never_downtiered_by_the_capability_clamp() {
        // Unrankable by construction — the fact the whole guard rests on.
        assert_eq!(tier_quality_rank(NVFP4_TIER), 0);
        assert!(tier_quality_rank(NVFP4_TIER) < tier_quality_rank("q4"));

        // The `downtier_candidate_tiers` range math (installed-filtering aside) admits NOTHING when the
        // default is nvfp4: `rank <= 0 && rank >= floor_rank(>= 1)` is unsatisfiable for every tier.
        let in_range = |tier: &str, default: &str, floor: Option<&str>| {
            let default_rank = tier_quality_rank(default);
            let floor_rank = floor.map_or(1, tier_quality_rank).max(1);
            let rank = tier_quality_rank(tier);
            rank <= default_rank && rank >= floor_rank
        };
        for floor in [None, Some("q4"), Some("q8"), Some("bf16")] {
            for tier in ["bf16", "q8", "q4", NVFP4_TIER] {
                assert!(
                    !in_range(tier, NVFP4_TIER, floor),
                    "nvfp4 must admit NO downtier candidate (tier={tier} floor={floor:?})"
                );
            }
        }

        // …so the chooser is handed an empty candidate list and KEEPS nvfp4, deferring to the plain gate.
        assert_eq!(choose_downtier(NVFP4_TIER, &[]), DowntierPick::Keep);
    }

    #[test]
    fn downtier_candidate_range_is_floor_to_default_descending() {
        // The pure range math behind `downtier_candidate_tiers` (installed-filtering aside): candidates
        // run from the default DOWN to the floor, highest-fidelity first, never above the default.
        let in_range = |tier: &str, default: &str, floor: Option<&str>| {
            let default_rank = tier_quality_rank(default);
            let floor_rank = floor.map_or(1, tier_quality_rank).max(1);
            let rank = tier_quality_rank(tier);
            rank <= default_rank && rank >= floor_rank
        };
        // Default q8, no floor (→ q4): q8 and q4 in range; bf16 (above default) excluded.
        assert!(in_range("q8", "q8", None));
        assert!(in_range("q4", "q8", None));
        assert!(!in_range("bf16", "q8", None));
        // Floor q8: q4 excluded (below floor), q8 in range.
        assert!(!in_range("q4", "q8", Some("q8")));
        assert!(in_range("q8", "q8", Some("q8")));
    }
}

/// sc-10733 acceptance #6 (MLX lane): drive the capability downtier END-TO-END through the real
/// `SCENEWORKS_MLX_MEMORY_CAP_GB` emulation knob — env → `mlx_memory_cap_gb` → `resolve_budget` → real
/// `sum_safetensors_bytes` → `decide_residency` → [`mlx_tier_fit`] → [`choose_downtier`] — not just the
/// isolated pure chooser. Ignored: it reads the ambient knob, so run it with the cap set between the q4
/// and q8 predicted peaks (weights + the 18 GiB headroom):
///
/// ```text
/// SCENEWORKS_MLX_MEMORY_CAP_GB=21 cargo test -p sceneworks-worker --lib -- --ignored --nocapture \
///   mlx_downtier_via_emulation_knob
/// ```
#[cfg(all(test, target_os = "macos"))]
mod mlx_downtier_emulation_tests {
    use super::*;

    #[test]
    #[ignore = "sc-10733 AC#6 e2e; run with SCENEWORKS_MLX_MEMORY_CAP_GB=21"]
    fn mlx_downtier_via_emulation_knob() {
        let cap = crate::mlx_fit_gate::mlx_memory_cap_gb().expect(
            "set SCENEWORKS_MLX_MEMORY_CAP_GB (e.g. 21) — between the q4 (~19) and q8 (~23) peaks",
        );
        // Sparse tier dirs: q8 ~5 GiB, q4 ~1 GiB LOGICAL (set_len ⇒ no real disk on APFS). The gate sums
        // `metadata.len()`, so these read as 5/1 GiB → predicted peaks 23/19 GiB (+18 headroom).
        let root = std::env::temp_dir().join(format!(
            "mlx_downtier_emu_{}_{}",
            std::process::id(),
            line!()
        ));
        let make_tier = |tier: &str, gib: u64| -> PathBuf {
            let dir = root.join(tier).join("transformer");
            std::fs::create_dir_all(&dir).expect("mk tier dir");
            let file = std::fs::File::create(dir.join("model.safetensors")).expect("mk weights");
            file.set_len(gib * 1024 * 1024 * 1024).expect("sparse weights");
            root.join(tier)
        };
        let q8_dir = make_tier("q8", 5);
        let q4_dir = make_tier("q4", 1);
        // Unregistered engine ⇒ no footprint / not sequential-capable ⇒ resident-or-reject (te=0).
        let engine = "unregistered_downtier_probe";
        let q8_fit = mlx_tier_fit(engine, &q8_dir);
        let q4_fit = mlx_tier_fit(engine, &q4_dir);
        assert!(
            matches!(q8_fit, TierFit::TooBig { .. }),
            "q8 (~23 GiB) must exceed the {cap} GB emulated budget: {q8_fit:?}"
        );
        assert_eq!(q4_fit, TierFit::Fits, "q4 (~19 GiB) must fit {cap} GB");
        // The full decision: a q8 DEFAULT downtiers to the installed q4 that fits (acceptance #2).
        let candidates = [("q8", q8_fit), ("q4", q4_fit)];
        assert_eq!(
            choose_downtier("q8", &candidates),
            DowntierPick::Downtier("q4"),
            "q8 default must capability-downtier to q4 under the {cap} GB emulated cap"
        );
        eprintln!("emulation knob cap={cap} GB → q8 default DOWNTIERED to q4 (sc-10733 ✓)");
        std::fs::remove_dir_all(&root).ok();
    }
}

/// sc-10733 acceptance #6 (candle lane): drive the capability downtier END-TO-END through the real
/// `SCENEWORKS_CUDA_VRAM_CAP_GB` emulation knob — env → `cuda_vram_cap_gb` → `apply_vram_cap` →
/// `predicted_peak_gb`/`fit_decision` → [`candle_tier_fit`] → [`choose_downtier`]. The synthetic-cap
/// budget (`apply_vram_cap(None, Some(cap))`) needs no real GPU, so this runs in the candle build under
/// the knob. Ignored (reads the ambient knob); run with the cap between the q4 and q8 peaks:
///
/// ```text
/// SCENEWORKS_CUDA_VRAM_CAP_GB=30 cargo test -p sceneworks-worker --lib --features backend-candle -- \
///   --ignored --nocapture candle_downtier_via_emulation_knob
/// ```
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod candle_downtier_emulation_tests {
    use super::*;
    use serde_json::json;

    #[test]
    #[ignore = "sc-10733 AC#6 e2e; run with SCENEWORKS_CUDA_VRAM_CAP_GB=30"]
    fn candle_downtier_via_emulation_knob() {
        let cap = crate::vram_gate::cuda_vram_cap_gb()
            .expect("set SCENEWORKS_CUDA_VRAM_CAP_GB (e.g. 30) — between the q4 (~28) and q8 (~38) peaks");
        // The knob-emulated budget: no real reading + the cap ⇒ a synthetic `free = total = cap` budget,
        // so this exercises the whole chain without a CUDA card.
        let budget = crate::vram_gate::apply_vram_cap(None, crate::vram_gate::cuda_vram_cap_gb());
        // Krea 2 Turbo candle tiers (builtin.models.jsonc, measured — sc-12126): q4 26.4, q8 35.9 (+2
        // headroom ⇒ peaks 28.4 / 37.9).
        let manifest = json!({ "candle": { "vramGbByTier": { "q4": 26.4, "q8": 35.9 } } })
            .as_object()
            .cloned()
            .unwrap();
        // Not sequential-capable (krea keeps the resident path) ⇒ resident-or-reject.
        let q8_fit = candle_tier_fit(&manifest, "q8", budget, false);
        let q4_fit = candle_tier_fit(&manifest, "q4", budget, false);
        assert!(
            matches!(q8_fit, TierFit::TooBig { .. }),
            "q8 (~38 GB) must exceed the {cap} GB emulated card: {q8_fit:?}"
        );
        assert_eq!(q4_fit, TierFit::Fits, "q4 (~28 GB) must fit {cap} GB");
        assert_eq!(
            choose_downtier("q8", &[("q8", q8_fit), ("q4", q4_fit)]),
            DowntierPick::Downtier("q4"),
            "q8 default must capability-downtier to q4 under the {cap} GB emulated cap"
        );
        eprintln!("emulation knob cap={cap} GB → q8 default DOWNTIERED to q4 (sc-10733 ✓)");
    }
}

// Krea "text style" gain gate (sc-12008): the slider lives at `.ui.textStyleGain` in the FULL model
// entry the worker receives, so the gate MUST read through `ui`. Reading the top level is a silent
// no-op end-to-end (the original bug). This drives a real resolved manifest entry through the shared
// `resolve_text_style_gain` seam — the coverage the engine-only A/B (sc-11878/sc-11884) never had.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
mod text_style_gain_gate_tests {
    use super::*;
    use serde_json::json;

    fn request_with(manifest_entry: Value, advanced: Value) -> ImageRequest {
        let payload = json!({
            "projectId": "p",
            "model": "krea_2_turbo",
            "modelManifestEntry": manifest_entry,
            "advanced": advanced,
        });
        ImageRequest::from_payload(payload.as_object().expect("payload is an object"))
    }

    #[test]
    fn gain_resolves_through_ui_nesting_not_top_level() {
        // Correct nesting → the user value flows through, clamped to the GPU-validated band.
        let declared = request_with(
            json!({ "id": "krea_2_turbo", "ui": { "textStyleGain": { "default": 1.0, "min": 0.25, "max": 1.75 } } }),
            json!({ "textStyleGain": 1.75 }),
        );
        assert_eq!(resolve_text_style_gain(&declared), Some(1.75));

        // Slider declared but user left it at default (web omits the key) → Some(1.0), a byte-exact
        // engine no-op — NOT None, so the field is still wired through.
        let defaulted = request_with(
            json!({ "id": "krea_2_turbo", "ui": { "textStyleGain": { "default": 1.0 } } }),
            json!({}),
        );
        assert_eq!(resolve_text_style_gain(&defaulted), Some(1.0));

        // Out-of-range user value is clamped to [0.25, 1.75].
        let hot = request_with(
            json!({ "id": "krea_2_turbo", "ui": { "textStyleGain": { "default": 1.0 } } }),
            json!({ "textStyleGain": 9.0 }),
        );
        assert_eq!(resolve_text_style_gain(&hot), Some(1.75));

        // MUTATION CHECK (the sc-12008 bug): the slider object at the manifest TOP LEVEL with no `ui`
        // block must NOT resolve — guards against regressing to `.get("textStyleGain")`.
        let top_level_only = request_with(
            json!({ "id": "krea_2_turbo", "textStyleGain": { "default": 1.0 } }),
            json!({ "textStyleGain": 1.75 }),
        );
        assert_eq!(resolve_text_style_gain(&top_level_only), None);

        // A model that doesn't declare the slider (no `ui.textStyleGain`) self-gates to None even when
        // the client sends an `advanced.textStyleGain` — the manifest is the gate.
        let undeclared = request_with(
            json!({ "id": "sana_1600m", "ui": { "img2img": true } }),
            json!({ "textStyleGain": 1.75 }),
        );
        assert_eq!(resolve_text_style_gain(&undeclared), None);
    }
}
