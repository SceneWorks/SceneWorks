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
    /// A Krea 2 **Raw** t2i job carrying the accelerator (turbo) LoRA (`role: accelerator`, sc-13882)
    /// → the single-phase turbo-on-Raw lane (epic 13879 S3, sc-13883): the Raw base + LoRA additive,
    /// sampled through the distilled Turbo regime (fixed mu 1.15 / ~8 steps / CFG-off) by routing to
    /// the `krea_2_turbo` engine. Wins over the generic `Mlx` arm below — `krea_2_raw` is in
    /// MODEL_TABLE, so `mlx_available` would otherwise render it as plain Raw t2i (52-step true-CFG)
    /// and never accelerate. The t2i sibling of [`ImageRoute::KreaEdit`].
    KreaTurboOnRaw,
    /// A Krea 2 **Raw** t2i job carrying an explicit `advanced.phases` list (epic 13879 S4, sc-13884)
    /// → the multi-phase denoise driver: ONE Raw trajectory over ONE global sigma schedule, each phase
    /// a contiguous step slice with its own guidance (per-phase CFG on/off) and its own active subset of
    /// the job's load-time LoRA stack (per-phase toggling). Wins over [`ImageRoute::KreaTurboOnRaw`] and
    /// the generic `Mlx` arm — an explicit phase list is the FINER-GRAINED control and takes precedence
    /// over S3's whole-job turbo-on-Raw regime. Loads the `krea_2_raw` engine (the multi-phase driver
    /// keys on that descriptor id), unlike S3 which swaps to `krea_2_turbo`. Reference/edit/pose/PiD
    /// shapes are rejected loudly by the lane (multi-phase renders from pure noise).
    KreaMultiPhase,
    /// An imported/user single-file Krea 2 checkpoint (epic 14015 S0c, sc-14018): a non-builtin
    /// `krea_2`-family model whose `modelPath` is a single `.safetensors` DiT → the bespoke in-place
    /// assembly lane, which pairs the imported transformer with a resident `krea_2` base tier (shared
    /// Qwen3-VL TE / Qwen VAE / tokenizer) and loads via the S0b MLX native single-file entrypoint. A
    /// builtin Krea id (`krea_2_turbo` / `krea_2_raw`, in `MODEL_TABLE`) never reaches here —
    /// `resolve_imported_krea_dit` returns `None` for it, so the snapshot-dir Krea path is untouched.
    /// txt2img only (a bare imported DiT carries no conditioning components).
    KreaImported,
    /// An imported/user single-file Krea 2 checkpoint carrying a **strict-pose set** (a non-empty
    /// `advanced.poses` outside edit mode): the trained pose control-branch overlay rides the
    /// FILE-LOADED imported DiT via the MLX native control entrypoint
    /// (`load_control_from_native_dit_file`), one image per pose — the imported twin of
    /// [`ImageRoute::KreaControl`]. Claimed BEFORE the plain [`ImageRoute::KreaImported`] arm so a
    /// pose set gets the per-pose count + control render instead of per-image t2i. MLX-only
    /// (`KREA_IMPORTED_SUPPORTS_POSE_CONTROL`); the candle imported lane has no control path.
    KreaImportedControl,
    /// A fused SDXL LDM/A1111 single-file checkpoint. The file carries the UNet, both text encoders,
    /// and VAE; tokenizer assets are borrowed from the installed SDXL base turnkey.
    SdxlImported,
    /// This app's OWN full base fine-tune of Mage-Flow Base (sc-15036, epic 14034 F6): a non-builtin
    /// `mage-flow`-family catalog entry whose `paths.model` is a complete diffusers transformer
    /// component dir → the bespoke fine-tuned lane, which stages the installed base's shared text
    /// encoder + VAE and loads through `mage::load_finetuned`. A builtin Mage id never reaches here
    /// (`resolve_mage_finetuned_transformer` returns `None` for a `MODEL_TABLE` id), so the tiered
    /// snapshot path is untouched. txt2img only — the non-edit Mage variants advertise no
    /// conditioning and the fine-tuned entrypoint refuses adapters.
    MageFinetuned,
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

/// The production FLUX strict-control router, expressed as the exact base-model → dedicated
/// provider mapping used by the two `..._control_available` arms below. Keeping this pure seam next
/// to [`resolve_image_route`] lets source-bound audits ask the router whether a model really has a
/// strict-control lane instead of attaching arbitrary control bytes to a base provider.
///
/// Chroma, FLUX.1 Schnell, and FLUX.2 Klein deliberately resolve to `None`: none ships a control
/// checkpoint. Only the two Dev bases route to their dedicated registry providers.
#[cfg(target_os = "macos")]
pub(crate) fn mlx_flux_strict_control_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "flux_dev" => Some(FLUX1_DEV_CONTROL_ENGINE_ID),
        "flux2_dev" => Some(FLUX2_DEV_CONTROL_ENGINE_ID),
        _ => None,
    }
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
    } else if krea_multiphase_available(request, settings) {
        // Krea 2 Raw t2i + an explicit `advanced.phases` list (epic 13879 S4, sc-13884) → the
        // multi-phase denoise lane. Placed FIRST among the `krea_2_raw` lanes so an explicit phase list
        // takes precedence over the S3 whole-job turbo-on-Raw regime (`krea_turbo_on_raw_available`
        // below) AND the generic Raw t2i (`mlx_available`): multi-phase is the finer-grained control.
        // Only claims `krea_2_raw` + a present `advanced.phases`, so every non-phases job is unaffected;
        // the lane itself rejects edit/pose/reference/PiD shapes loudly (multi-phase renders from noise),
        // so claiming a conflicting shape here surfaces a clear error rather than diverting it silently.
        Some(ImageRoute::KreaMultiPhase)
    } else if krea_edit_available(request, settings) {
        // Krea 2 Raw Kontext-style edit (mode edit_image + a source) → the `krea_2_edit` engine
        // (epic 10871). Wins over the generic MLX arm below — `krea_2_raw` is in MODEL_TABLE, so
        // `mlx_available` would otherwise render it as plain t2i and silently drop the source.
        Some(ImageRoute::KreaEdit)
    } else if krea_turbo_on_raw_available(request, settings) {
        // Krea 2 Raw t2i + the accelerator (turbo) LoRA (sc-13882) → the single-phase turbo-on-Raw
        // lane (epic 13879 S3, sc-13883): the Raw base + LoRA additive, sampled as Turbo (fixed mu
        // 1.15 / ~8 steps / CFG-off) by routing to the `krea_2_turbo` engine. Wins over the generic
        // `mlx_available` arm below for PLAIN t2i — a plain Raw t2i would otherwise run the 52-step
        // true-CFG regime and ignore the accelerator's intent. A `krea_2_raw` job that ALSO carries an
        // img2img reference (`ui.img2img` + `referenceAssetId`) is EXCLUDED by the gate and falls
        // through to the generic `mlx_available` img2img arm (reference honored, accelerator LoRA still
        // additive) — turbo-on-Raw img2img is out of scope for this t2i story (sc-13883). The t2i
        // sibling of the `krea_edit_available` arm above.
        Some(ImageRoute::KreaTurboOnRaw)
    } else if krea_imported_control_available(request, settings) {
        // An imported single-file Krea 2 checkpoint + a strict-pose set: the pose control branch
        // rides the file-loaded imported DiT (the imported twin of the `KreaControl` arm above).
        // Checked BEFORE the plain imported arm so a pose set renders one pose-locked image per
        // pose instead of falling into per-image t2i (which would silently drop the poses).
        Some(ImageRoute::KreaImportedControl)
    } else if krea_imported_available(request, settings) {
        // An imported/user single-file Krea 2 checkpoint (epic 14015 S0c, sc-14018): a non-builtin
        // `krea_2`-family model whose `modelPath` is a single `.safetensors` DiT → the bespoke in-place
        // assembly lane. A builtin Krea id never claims this (`resolve_imported_krea_dit` returns `None`
        // for a `MODEL_TABLE` id), so the generic `mlx_available` snapshot-dir arm below is unchanged for
        // builtin Krea. The imported id is in no `MODEL_TABLE`, so `mlx_available` is `false` for it — this
        // arm is what routes it to real MLX generation at all (S0d marked it Mac-routable; this loads it).
        Some(ImageRoute::KreaImported)
    } else if sdxl_imported_available(request, settings) {
        Some(ImageRoute::SdxlImported)
    } else if mage_finetuned_available(request, settings) {
        // A fine-tuned Mage-Flow base (sc-15036). The fine-tune's id is in no `MODEL_TABLE`, so
        // `mlx_available` is `false` for it — this arm is what routes it to real MLX generation at
        // all. A builtin Mage id is claimed by the generic `mlx_available` arm below, unchanged.
        Some(ImageRoute::MageFinetuned)
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
    /// True only for routes whose actual load path applies the request's LoRA/LoKr stack.
    fn applies_request_loras(self) -> bool {
        matches!(
            self,
            ImageRoute::ZImageControl
                | ImageRoute::ZImageBaseControl
                | ImageRoute::QwenControl
                | ImageRoute::KolorsControl
                | ImageRoute::KreaControl
                | ImageRoute::Flux1DevControl
                | ImageRoute::Flux2DevControl
                | ImageRoute::Flux2Edit
                | ImageRoute::QwenEdit
                | ImageRoute::KreaEdit
                | ImageRoute::KreaTurboOnRaw
                | ImageRoute::KreaMultiPhase
                | ImageRoute::KreaImported
                | ImageRoute::KreaImportedControl
                | ImageRoute::SdxlImported
                | ImageRoute::InstantId
                | ImageRoute::SdxlAdvanced
                | ImageRoute::Mlx
        )
    }

    fn image_count(self, request: &ImageRequest, settings: &Settings) -> u32 {
        match self {
            ImageRoute::ZImageControl
            | ImageRoute::ZImageBaseControl
            | ImageRoute::QwenControl
            | ImageRoute::KolorsControl
            | ImageRoute::KreaControl
            | ImageRoute::Flux1DevControl
            | ImageRoute::Flux2DevControl
            // The imported strict-pose lane renders one image per pose, like every control lane.
            | ImageRoute::KreaImportedControl => pose_entries(request).len() as u32,
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
            // Turbo-on-Raw is plain per-image (like Krea edit + the base MLX path): `count` renders of
            // the base prompt, each its own seed. No angle/pose grouping.
            | ImageRoute::KreaTurboOnRaw
            // Multi-phase (S4) is likewise plain per-image: `count` renders, each its own seed, driven
            // through the phase plan. No angle/pose grouping.
            | ImageRoute::KreaMultiPhase
            // Imported single-file Krea 2 (S0c) is plain per-image txt2img: `count` renders, each its own
            // seed. No angle/pose grouping (a bare imported DiT carries no conditioning).
            | ImageRoute::KreaImported
            | ImageRoute::SdxlImported
            // A fine-tuned Mage-Flow base (sc-15036) is plain per-image txt2img too: `count`
            // renders, each its own seed. No angle/pose grouping (the lane claims no conditioning).
            | ImageRoute::MageFinetuned
            | ImageRoute::PoseControlBaseMissing
            | ImageRoute::PoseReject
            | ImageRoute::Mlx => request.count,
        }
    }

    /// The label written by this route's per-asset stream. Bespoke routes that do not map directly
    /// to the request's registry row are named explicitly; the remaining MLX routes use the same
    /// descriptor label their stream resolves.
    fn adapter_label(self, request: &ImageRequest) -> &'static str {
        match self {
            ImageRoute::KreaControl => KREA_CONTROL_ENGINE_ID,
            ImageRoute::KreaImported | ImageRoute::KreaImportedControl => KREA_IMPORTED_ENGINE,
            ImageRoute::SdxlImported => SDXL_IMPORTED_ENGINE,
            ImageRoute::MageFinetuned => MAGE_FINETUNED_ENGINE,
            ImageRoute::InstantId => INSTANTID_ENGINE,
            ImageRoute::PulidFlux => PULID_ADAPTER_LABEL,
            ImageRoute::PoseControlBaseMissing | ImageRoute::PoseReject => STUB_ADAPTER,
            _ => mlx_model(&request.model)
                .map(|model| model.adapter_label())
                .unwrap_or(STUB_ADAPTER),
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
    /// Mage-Flow Base/RL/Turbo instruction edit. Uses the generic registry stream, but requires
    /// source-first ordered multi-reference conditioning rather than the plain T2I request shape.
    MageEdit,
    /// Krea 2 Kontext-style dual-conditioned image-edit — `krea_2_raw` + `edit_image` + a source, with
    /// the required `krea2_identity_edit` LoRA (epic 10871).
    KreaEdit,
    /// A Krea 2 **Raw** t2i job carrying the accelerator (turbo) LoRA (`role: accelerator`, sc-13882)
    /// → the single-phase turbo-on-Raw lane (epic 13879 S3, sc-13883, brought to candle by sc-13887):
    /// the Raw base + LoRA additive, sampled through the distilled Turbo regime (fixed mu 1.15 / ~8 steps
    /// / CFG-off) by routing to the `krea_2_turbo` candle engine id (which keys that regime, inference PR
    /// #204). The candle twin of the MLX `ImageRoute::KreaTurboOnRaw`; both dispatch to the SHARED,
    /// backend-neutral `generate_krea_turbo_on_raw_stream` (`krea_turbo_raw.rs`). Wins over the generic
    /// `CandleTxt2Img` arm for PLAIN t2i — `krea_2_raw` is a candle txt2img id, so it would otherwise run
    /// the 52-step true-CFG Raw regime and never accelerate.
    KreaTurboOnRaw,
    /// A Krea 2 **Raw** t2i job carrying an explicit `advanced.phases` list (epic 13879 S4, sc-13884,
    /// brought to candle by sc-13887) → the multi-phase denoise driver: ONE Raw trajectory over ONE
    /// global sigma schedule, each phase a contiguous step slice with its own guidance (per-phase CFG
    /// on/off) and its own active subset of the job's load-time LoRA stack (per-phase toggling). The
    /// candle twin of the MLX `ImageRoute::KreaMultiPhase`; both dispatch to the SHARED, backend-neutral
    /// `generate_krea_multiphase_stream` (`krea_multiphase.rs`), which passes the backend-agnostic
    /// `GenerationRequest::phases` the candle Krea engine honors (inference PR #204). Wins over
    /// `KreaTurboOnRaw` and the generic `CandleTxt2Img` arm — an explicit phase list is the finer-grained
    /// control. Loads the `krea_2_raw` engine (the driver keys on that descriptor id). Reference / edit /
    /// pose / PiD shapes are rejected loudly by the lane (multi-phase renders from pure noise).
    KreaMultiPhase,
    /// An imported/user single-file Krea 2 transformer: pair the in-place DiT with the resident
    /// shared-component base tier and load it through Candle's native single-file entrypoint. This is
    /// the off-Mac twin of [`ImageRoute::KreaImported`], required because imported IDs are not registry
    /// engine IDs and would otherwise fall through to the procedural stub.
    KreaImported,
    /// Off-Mac twin of [`ImageRoute::SdxlImported`], loaded by candle from the fused checkpoint plus
    /// the three caller-staged SDXL components.
    SdxlImported,
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
/// NOTE: deliberately DISTINCT from the router's `model_has_candle_pose_lane` (sceneworks-core),
/// which omits `krea_2_turbo` so other GPU descriptors decline Krea pose jobs and candle reliably
/// owns the request — do not conflate the two lists.
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
    /// True only for routes whose actual load path applies the request's LoRA/LoKr stack. Z-Image
    /// edit is the registered Turbo alias and therefore participates; the remaining bespoke edit,
    /// IP-adapter, control, ComfyUI, PuLID and Bernini lanes intentionally do not.
    fn applies_request_loras(self, request: &ImageRequest) -> bool {
        match self {
            CandleImageRoute::InstantId
            | CandleImageRoute::QwenEdit
            | CandleImageRoute::ZimageEdit
            | CandleImageRoute::MageEdit
            | CandleImageRoute::KreaEdit
            | CandleImageRoute::KreaTurboOnRaw
            | CandleImageRoute::KreaMultiPhase
            | CandleImageRoute::KreaImported
            | CandleImageRoute::SdxlImported
            | CandleImageRoute::KreaControl => true,
            CandleImageRoute::CandleTxt2Img => {
                !wants_krea_convrot(request)
                    && mlx_model(&request.model).is_some_and(|model| model.supports_adapters())
            }
            _ => false,
        }
    }

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

    /// The exact label written by this route's per-asset stream. This is intentionally route-based:
    /// imported/external/bespoke IDs are not present in the generic candle catalog.
    fn adapter_label(self, request: &ImageRequest) -> &'static str {
        match self {
            CandleImageRoute::InstantId => INSTANTID_ENGINE,
            CandleImageRoute::SdxlEdit => sdxl_edit_candle::SDXL_EDIT_CANDLE_ENGINE,
            CandleImageRoute::Flux2Edit => flux2_edit_candle::FLUX2_EDIT_CANDLE_ENGINE,
            CandleImageRoute::QwenEdit => qwen_edit_candle::QWEN_EDIT_CANDLE_ENGINE,
            CandleImageRoute::ZimageEdit => candle_adapter_label(&request.model),
            CandleImageRoute::KreaEdit => krea_edit_candle::KREA_EDIT_CANDLE_ENGINE,
            CandleImageRoute::KreaTurboOnRaw | CandleImageRoute::KreaMultiPhase => {
                mlx_model(&request.model)
                    .map(|model| model.adapter_label())
                    .unwrap_or(STUB_ADAPTER)
            }
            CandleImageRoute::KreaImported => KREA_IMPORTED_ENGINE,
            CandleImageRoute::SdxlImported => SDXL_IMPORTED_ENGINE,
            CandleImageRoute::ZimageIdentity => {
                zimage_identity_candle::ZIMAGE_IDENTITY_CANDLE_ENGINE
            }
            CandleImageRoute::SdxlIpAdapter => sdxl_ipadapter::SDXL_IPADAPTER_ENGINE,
            CandleImageRoute::KolorsIpAdapter => kolors_ipadapter::KOLORS_IPADAPTER_ENGINE,
            CandleImageRoute::FluxIpAdapter => flux_ipadapter::FLUX_IPADAPTER_ENGINE,
            CandleImageRoute::Pulid => pulid_candle::PULID_CANDLE_ENGINE,
            CandleImageRoute::QwenControl => qwen_control::QWEN_CONTROL_ENGINE,
            CandleImageRoute::KolorsControl => kolors_control::KOLORS_CONTROL_ENGINE,
            CandleImageRoute::ZimageControl => zimage_control::ZIMAGE_CTRL_ENGINE,
            CandleImageRoute::Flux2Control => {
                flux2_control_candle::FLUX2_CONTROL_CANDLE_ENGINE
            }
            CandleImageRoute::Flux1Control => {
                flux1_control_candle::FLUX1_CONTROL_CANDLE_ENGINE
            }
            CandleImageRoute::KreaControl => krea_control_candle::KREA_CONTROL_ENGINE,
            CandleImageRoute::PoseReject | CandleImageRoute::PoseControlBaseMissing => STUB_ADAPTER,
            CandleImageRoute::ZimageComfyui => {
                zimage_comfyui_candle::ZIMAGE_COMFYUI_CANDLE_ENGINE
            }
            CandleImageRoute::QwenImageComfyui => {
                qwen_comfyui_candle::QWEN_COMFYUI_CANDLE_ENGINE
            }
            CandleImageRoute::Flux2Comfyui => {
                flux2_comfyui_candle::FLUX2_COMFYUI_CANDLE_ENGINE
            }
            CandleImageRoute::Bernini => CANDLE_BERNINI_IMAGE_ADAPTER,
            CandleImageRoute::MageEdit | CandleImageRoute::CandleTxt2Img => {
                candle_adapter_label(&request.model)
            }
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
    } else if is_mage_edit_model(&request.model)
        && request.mode == "edit_image"
        && request
            .source_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
    {
        Some(CandleImageRoute::MageEdit)
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
    } else if krea_multiphase_available(request, settings) {
        // Krea 2 Raw t2i + an explicit `advanced.phases` list (epic 13879 S4, sc-13884; candle sc-13887)
        // → the multi-phase denoise lane (`generate_krea_multiphase_stream`, SHARED with MLX). Placed
        // FIRST among the `krea_2_raw` t2i lanes so an explicit phase list takes precedence over the S3
        // whole-job turbo-on-Raw regime (`krea_turbo_on_raw_available` below) AND the generic Raw t2i
        // (`CandleTxt2Img`): multi-phase is the finer-grained control. Only claims `krea_2_raw` + a present
        // `advanced.phases`, so every non-phases job is unaffected; the lane itself rejects
        // edit/pose/reference/PiD shapes loudly (multi-phase renders from pure noise), so claiming a
        // conflicting shape here surfaces a clear error rather than diverting it silently. The candle twin
        // of the MLX `resolve_image_route` `KreaMultiPhase` arm.
        Some(CandleImageRoute::KreaMultiPhase)
    } else if krea_edit_candle_available(request, settings) {
        // Krea 2 Kontext-style edit (epic 10871): `krea_2_raw` + `edit_image` + a source is the bespoke
        // candle `KreaEdit` lane (`generate_candle_krea_edit_stream`), NOT the generic t2i stream. Diverted
        // BEFORE the generic `is_candle_engine` t2i arm below (which `krea_2_raw` now matches, sc-9994/epic
        // 9992) so an edit job runs the dual-conditioning Kontext render instead of being flattened to plain
        // txt2img. Mirrors `jobs_store::krea_edit_candle_eligible`.
        Some(CandleImageRoute::KreaEdit)
    } else if krea_turbo_on_raw_available(request, settings) {
        // Krea 2 Raw t2i + the accelerator (turbo) LoRA (sc-13882) → the single-phase turbo-on-Raw lane
        // (epic 13879 S3, sc-13883; candle sc-13887): the Raw base + LoRA additive, sampled as Turbo
        // (fixed mu 1.15 / ~8 steps / CFG-off) by routing to the `krea_2_turbo` candle engine
        // (`generate_krea_turbo_on_raw_stream`, SHARED with MLX). Wins over the generic `CandleTxt2Img`
        // arm for PLAIN t2i — a plain Raw t2i would otherwise run the 52-step true-CFG regime and ignore
        // the accelerator's intent. A `krea_2_raw` job that ALSO carries an img2img reference is EXCLUDED
        // by the gate and falls through to the generic `CandleTxt2Img` img2img path (reference honored via
        // `render_base_img2img`, accelerator LoRA still additive) — turbo-on-Raw img2img is out of scope
        // for this t2i story (sc-13883). The candle twin of the MLX `resolve_image_route` `KreaTurboOnRaw`
        // arm; placed AFTER the edit lane, BEFORE the generic txt2img arm.
        Some(CandleImageRoute::KreaTurboOnRaw)
    } else if krea_imported_available(request, settings) {
        // Imported/user Krea 2 single-file t2i: external IDs are absent from `is_candle_engine`, so
        // this bespoke route must claim them before the generic/external fall-through.
        Some(CandleImageRoute::KreaImported)
    } else if sdxl_imported_available(request, settings) {
        Some(CandleImageRoute::SdxlImported)
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
    } else if is_candle_engine(&request.model) && request.mode != "edit_image" {
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
/// FLUX.2's provider safety contract rejects over-budget combinations with an actionable message;
/// this caps absurd inputs (and bounds the DiT sequence) before that.
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

/// Exact snapshots used by the on-demand non-default tier fetches.  The API replaces a submitted
/// `modelManifestEntry` with its resolved builtin/user manifest entry before enqueueing, so a repo
/// different from the engine default is an intentional configured override.  First-party turnkeys
/// never follow a mutable ref; configured override repos retain `main` because SceneWorks cannot know
/// their release commit in advance.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const BOOGU_MLX_TURNKEY_REVISION: &str = "a459e614d408bfdf57089c32cc3da706f5a017de";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const IDEOGRAM_MLX_TURNKEY_REVISION: &str = "a3095855b8819dc0d6b067cb1354aaa7da189ff8";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const ZIMAGE_MLX_TURNKEY_REPO: &str = "SceneWorks/z-image-mlx";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const ZIMAGE_MLX_TURNKEY_REVISION: &str =
    "c74f74c2ad193294fc9ff3f8a5be71daa00d22ab";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const FLUX1_SCHNELL_MLX_TURNKEY_REVISION: &str =
    "bba3ae01dfd94089f173c05edd4e1a4c551f2599";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const FLUX1_DEV_MLX_TURNKEY_REVISION: &str =
    "323fd12d79f78ad444e882e8d8e871914584f2b9";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const FLUX2_DEV_MLX_TURNKEY_REVISION: &str =
    "2868b1461b2b6e6e05d84e52534df3632b4c7d5d";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const FLUX2_KLEIN_9B_MLX_TURNKEY_REVISION: &str =
    "acf05e8d5103838baba6a5e32dc91d6997a56023";

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn turnkey_tier_revision<'a>(
    repo: &str,
    default_repo: &str,
    default_revision: &'a str,
) -> &'a str {
    if repo == default_repo {
        default_revision
    } else {
        "main"
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn pinned_turnkey_snapshot_for_request(
    data_dir: &Path,
    request: &ImageRequest,
    repo: &str,
    default_repo: &str,
    revision: &str,
) -> Option<PathBuf> {
    if repo != default_repo {
        return None;
    }
    let root =
        crate::model_jobs::huggingface_pinned_snapshot_dir(data_dir, repo, revision)?;
    let selected = match request.model.as_str() {
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit" => {
            boogu_model_subdir(&root, request)
        }
        "ideogram_4" | "ideogram_4_turbo" => ideogram_model_subdir(&root, request),
        "z_image" => standard_tier_subdir(&root, request),
        _ => return None,
    };
    let complete = if request.model == "z_image" {
        tier_key_from_resolved_dir(&selected).is_some()
            && (selected
                .join("transformer/diffusion_pytorch_model.safetensors")
                .is_file()
                || selected
                    .join("transformer/diffusion_pytorch_model.safetensors.index.json")
                    .is_file())
    } else if request.model.starts_with("boogu_") {
        selected
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file()
            || selected
                .join("transformer/diffusion_pytorch_model.safetensors.index.json")
                .is_file()
    } else {
        selected.join("transformer/model.safetensors").is_file()
    };
    complete.then_some(root)
}

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

#[cfg(any(target_os = "macos", feature = "backend-candle"))]
// Keep the explicit optional branch: the manifest audit recognizes this shape and proves every
// production modelPath reader preserves the normal repo-resolution fallback.
#[allow(clippy::question_mark)]
fn model_path_override(request: &ImageRequest) -> Option<String> {
    let Some(raw_path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return None;
    };
    Some(raw_path.to_owned())
}

/// Resolve the weights snapshot directory: an explicit `modelPath` dir wins, else the
/// HuggingFace cache snapshot for the model repo. `None` when the model is not a known
/// engine family or its snapshot is absent. Available on the candle lane too (sc-5501): the
/// off-Mac SenseNova-U1 VQA / interleave handlers resolve their snapshot through it.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) fn resolve_weights_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if let Some(path) = model_path_override(request) {
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
    let pinned_turnkey_snapshot = match request.model.as_str() {
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit" => {
            pinned_turnkey_snapshot_for_request(
                &settings.data_dir,
                request,
                &repo,
                model.default_repo(),
                BOOGU_MLX_TURNKEY_REVISION,
            )
        }
        "ideogram_4" | "ideogram_4_turbo" => pinned_turnkey_snapshot_for_request(
            &settings.data_dir,
            request,
            &repo,
            model.default_repo(),
            IDEOGRAM_MLX_TURNKEY_REVISION,
        ),
        "z_image" => pinned_turnkey_snapshot_for_request(
            &settings.data_dir,
            request,
            &repo,
            ZIMAGE_MLX_TURNKEY_REPO,
            ZIMAGE_MLX_TURNKEY_REVISION,
        ),
        _ => None,
    };
    let has_pinned_turnkey_snapshot = pinned_turnkey_snapshot.is_some();
    let snapshot = pinned_turnkey_snapshot
        .or_else(|| receipt_snapshot.clone())
        .or_else(|| huggingface_snapshot_dir(&settings.data_dir, &repo));
    // A tier receipt already resolves to the exact self-contained tier directory.  Returning it
    // before the family pickers prevents a second `q4/q8/bf16` descent and, more importantly, keeps
    // all load inputs on the receipt side of the all-receipt-or-all-current boundary.
    if !has_pinned_turnkey_snapshot
        && receipt_snapshot
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
    // SenseNova-U1 needs NO bespoke branch (sc-14249). It had one from sc-13817 to sc-14249:
    // `candle-gen-sensenova` mmapped its backbone at a hardcoded F32 and hard-rejected
    // `spec.quantize`, so it could read ONLY the dense `bf16/` tier — and `standard_tier_subdir`'s
    // app-wide q8 default handed it packed weights it could not load. That is fixed IN THE ENGINE:
    // every backbone projection now packed-detects its MLX triple, so all three tiers load and the
    // family goes through the ordinary `standard_tier_subdir` descent below like every other
    // quant-matrix model. The old `sensenova_candle_dense_tier` force is deliberately gone rather
    // than left as a harmless-looking guard: leaving it would pin every candle sensenova job to the
    // heaviest tier forever, which is the whole thing sc-14249 exists to undo.
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

#[cfg(target_os = "macos")]
fn resolved_mlx_artifact_tier(
    weights_dir: &Path,
    quant_bits: Option<i64>,
) -> Option<&'static str> {
    tier_key_from_resolved_dir(weights_dir).or_else(|| {
        quant_bits.map(|bits| if bits <= 4 { "q4" } else { "q8" })
    })
}

#[cfg(target_os = "macos")]
fn fixed_artifact_tier_matches(
    fixed_artifact_tier: Option<&str>,
    effective_tier: Option<&str>,
) -> bool {
    match (fixed_artifact_tier, effective_tier) {
        (Some(fixed), Some(actual)) => fixed == actual,
        _ => true,
    }
}

/// Resolve immutable provenance beside the exact artifact the MLX loader will open. HF identity is
/// supplied by a completed download receipt; local/converted identity is supplied only by the
/// worker-owned install receipt. Request and manifest provenance fields are intentionally ignored.
#[cfg(target_os = "macos")]
fn resolved_mlx_artifact_provenance(
    request: &ImageRequest,
    settings: &Settings,
    repo: &str,
    weights_dir: &Path,
    effective_tier: Option<&'static str>,
) -> WorkerResult<Option<crate::model_jobs::ResolvedArtifactProvenance>> {
    if let Some(path) = model_path_override(request) {
        let managed_root = resolve_app_managed_model_dir(settings, &path, "Image modelPath")?;
        let resolved_weights_dir = std::fs::canonicalize(weights_dir)?;
        if !resolved_weights_dir.starts_with(&managed_root) {
            return Ok(None);
        }
        let provenance =
            crate::model_jobs::app_managed_artifact_provenance(&resolved_weights_dir)?;
        return Ok(provenance.and_then(|mut provenance| {
            if provenance.fixed_artifact_tier.is_none() {
                provenance.fixed_artifact_tier =
                    tier_key_from_resolved_dir(&resolved_weights_dir).map(str::to_owned);
            }
            let matches = fixed_artifact_tier_matches(
                provenance.fixed_artifact_tier.as_deref(),
                effective_tier,
            );
            matches.then_some(provenance)
        }));
    }

    let requested_variant = requested_receipt_variant(request);
    let mut variants = vec![requested_variant, effective_tier.map(str::to_owned), None];
    variants.dedup();
    for variant in variants {
        // Probe WITHOUT repair first: this loop walks several candidate variants, and only the one
        // whose path actually holds the weights being loaded has earned a stamp write.
        let Some(mut resolved) = crate::model_jobs::huggingface_receipt_weights(
            &settings.data_dir,
            repo,
            Some(&request.model),
            variant.as_deref(),
            crate::model_jobs::ProvenanceRepair::Skip,
        ) else {
            continue;
        };
        if !weights_dir.starts_with(&resolved.path) {
            continue;
        }
        // This receipt names the artifact being loaded. If it never carried a tree-stamp baseline
        // (a backfilled or pre-stamping install), establish one now rather than leave the model
        // pinned to the legacy selector for good — see `ProvenanceRepair` (sc-16482). A receipt
        // whose stamp is present but MISMATCHED is drift and is left alone to fail closed.
        if resolved.provenance.is_none() {
            if let Some(repaired) = crate::model_jobs::huggingface_receipt_weights(
                &settings.data_dir,
                repo,
                Some(&request.model),
                variant.as_deref(),
                crate::model_jobs::ProvenanceRepair::Allow,
            ) {
                resolved = repaired;
            }
        }
        if let Some(mut provenance) = resolved.provenance {
            if provenance.fixed_artifact_tier.is_none() {
                provenance.fixed_artifact_tier =
                    tier_key_from_resolved_dir(weights_dir).map(str::to_owned);
            }
            if fixed_artifact_tier_matches(
                provenance.fixed_artifact_tier.as_deref(),
                effective_tier,
            ) {
                return Ok(Some(provenance));
            }
        }
    }
    Ok(None)
}

#[cfg(all(test, target_os = "macos"))]
mod resolved_artifact_provenance_tests {
    use super::*;
    use serde_json::json;

    fn settings(data_dir: &Path) -> Settings {
        Settings {
            api_url: "http://127.0.0.1".to_owned(),
            access_token: None,
            data_dir: data_dir.to_path_buf(),
            config_dir: data_dir.join("config"),
            worker_id: "test-worker".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 1,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
            max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: true,
            backend_candle_enabled: false,
            gpu_memory_limit_bytes: 0,
            external_model_roots: Vec::new(),
        }
    }

    fn request(model_path: &Path, spoofed_tier: &str) -> ImageRequest {
        ImageRequest::from_payload(
            json!({
                "model": "fixture_model",
                "advanced": {
                    "modelPath": model_path,
                    "modelPathProvenance": {
                        "repository": "attacker/spoof",
                        "revision": "attacker",
                        "variant": "attacker",
                        "tier": spoofed_tier,
                        "resolvedPathFingerprint": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
                    }
                }
            })
            .as_object()
            .expect("request object"),
        )
    }

    #[test]
    fn local_resolver_uses_only_worker_receipt_and_requires_its_exact_tier() {
        let data = tempfile::tempdir().expect("data dir");
        let installed = data.path().join("models/mlx/fixture/q4");
        std::fs::create_dir_all(&installed).expect("artifact dir");
        std::fs::write(installed.join("weights.safetensors"), b"trusted").expect("weights");
        let written = crate::model_jobs::write_app_managed_artifact_receipt(
            &installed,
            "SceneWorks/fixture",
            "c0ffee",
            "converted-fixture-q4",
            "q4",
        )
        .expect("worker receipt");
        let settings = settings(data.path());
        let spoofed = request(&installed, "q8");

        assert_eq!(
            resolved_mlx_artifact_provenance(
                &spoofed,
                &settings,
                "ignored/repo",
                &installed,
                Some("q4")
            )
            .expect("resolver"),
            Some(written),
            "request-supplied provenance must not replace the worker receipt"
        );
        assert_eq!(
            resolved_mlx_artifact_provenance(
                &spoofed,
                &settings,
                "ignored/repo",
                &installed,
                Some("q8")
            )
            .expect("resolver"),
            None,
            "the trusted q4 receipt must never be rewritten into a q8 identity"
        );

        let unstamped = data.path().join("models/mlx/unstamped/q8");
        std::fs::create_dir_all(&unstamped).expect("unstamped artifact");
        std::fs::write(unstamped.join("weights.safetensors"), b"untrusted").expect("weights");
        assert_eq!(
            resolved_mlx_artifact_provenance(
                &request(&unstamped, "q8"),
                &settings,
                "ignored/repo",
                &unstamped,
                Some("q8")
            )
            .expect("resolver"),
            None,
            "a spoofed payload cannot manufacture missing worker-owned provenance"
        );
    }

    /// A model installed before download-time tree stamps existed carries a BACKFILLED receipt
    /// (`apps/rust-api/src/models.rs::backfill_current_receipt`), which records `resolvedFiles` read
    /// off disk but never an `artifactTreeStamp` — only a download whose job payload carried
    /// `memoryCalibrationProvenanceRequired: true` computes one. Such a receipt resolves a fully
    /// loadable snapshot and still yields NO provenance, and nothing recomputes the stamp later.
    ///
    /// That is the exact state that made every calibration-opted-in model unusable. Two things had
    /// to change, and this pins both: the raw read still reports `None` (nothing is invented), while
    /// the MLX provenance resolver ESTABLISHES the missing baseline so the install can reach the
    /// evidence path instead of being pinned to the legacy selector for good (sc-16482).
    #[test]
    fn a_backfilled_receipt_gains_its_missing_tree_stamp_baseline_on_the_mlx_path() {
        let data = tempfile::tempdir().expect("data dir");
        let hub = data.path().join("hub");
        let _env =
            crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().expect("hub path"))]);
        let repo = "SceneWorks/fixture-turnkey";
        let files = ["q8/model_index.json", "q8/transformer/config.json"];
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(data.path(), repo)
            .expect("cache")
            .join("snapshots/rev-backfilled");
        std::fs::create_dir_all(snapshot.join("q8/transformer")).expect("snapshot tree");
        for file in files {
            std::fs::write(snapshot.join(file), b"{}").expect("snapshot file");
        }
        let marker_dir = data
            .path()
            .join("models")
            .join(crate::paths::safe_download_dir(repo));
        std::fs::create_dir_all(&marker_dir).expect("marker dir");
        // Byte-for-byte the backfill writer's shape: no `snapshotRevision`, no `artifactTreeStamp`.
        std::fs::write(
            marker_dir.join(crate::INSTALL_MARKER),
            serde_json::to_vec(&json!({
                "schemaVersion": 2,
                "repo": repo,
                "modelId": "fixture_model",
                "variant": "q8",
                "manifestFiles": ["q8/*"],
                "resolvedFiles": files,
                "backfilled": true,
            }))
            .expect("receipt json"),
        )
        .expect("marker");

        let weights_dir = snapshot.join("q8");
        let marker = marker_dir.join(crate::INSTALL_MARKER);
        let read_only = crate::model_jobs::huggingface_receipt_weights(
            data.path(),
            repo,
            Some("fixture_model"),
            Some("q8"),
            crate::model_jobs::ProvenanceRepair::Skip,
        )
        .expect("a backfilled receipt still resolves loadable weights");
        assert_eq!(
            read_only.path, weights_dir,
            "the artifact itself is found and loadable"
        );
        assert_eq!(
            read_only.provenance, None,
            "a read that may not repair must never invent an identity"
        );

        let request = ImageRequest::from_payload(
            json!({ "model": "fixture_model" })
                .as_object()
                .expect("request object"),
        );
        let repaired = resolved_mlx_artifact_provenance(
            &request,
            &settings(data.path()),
            repo,
            &weights_dir,
            Some("q8"),
        )
        .expect("resolver")
        .expect("the MLX path must establish the missing baseline, not give up on the install");
        assert_eq!(repaired.identity.repository, repo);
        assert_eq!(repaired.identity.revision, "rev-backfilled");
        assert_eq!(repaired.identity.variant, "q8");
        assert_eq!(repaired.fixed_artifact_tier.as_deref(), Some("q8"));

        // The baseline is PERSISTED, so a later mutation is detectable rather than re-blessed, and
        // the resolved revision is recorded so future resolution no longer leans on file-set
        // uniqueness. The repair anchor is recorded and never mistaken for a download-time stamp.
        let stored: Value =
            serde_json::from_slice(&std::fs::read(&marker).expect("marker")).expect("receipt json");
        assert!(
            crate::model_jobs::is_sha256_fingerprint(
                stored["artifactTreeStamp"].as_str().expect("stamp")
            ),
            "repair must write a well-formed stamp"
        );
        assert_eq!(stored["artifactTreeStampSource"], json!("repair"));
        assert_eq!(stored["snapshotRevision"], json!("rev-backfilled"));

        // Now that a baseline exists, a plain read resolves provenance with no further writing...
        assert!(crate::model_jobs::huggingface_receipt_weights(
            data.path(),
            repo,
            Some("fixture_model"),
            Some("q8"),
            crate::model_jobs::ProvenanceRepair::Skip,
        )
        .expect("resolves")
        .provenance
        .is_some());

        // ...and drift against that baseline fails closed instead of being repaired away.
        std::fs::write(snapshot.join(files[0]), b"{\"mutated\":true}").expect("mutate");
        assert_eq!(
            resolved_mlx_artifact_provenance(
                &request,
                &settings(data.path()),
                repo,
                &weights_dir,
                Some("q8"),
            )
            .expect("resolver"),
            None,
            "a receipt whose stamp MISMATCHES is drift; repair must never overwrite it"
        );
    }
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
    // packed) — so they additionally carry its `mlx.denseTextEncoderTier` declaration, which forces the load
    // Quant to None so the dense TE is never re-quantized. `_true_v2` stays on its install-time
    // single-file→diffusers convert (candle-only) and is not a turnkey yet.
    "flux2_klein_9b",
    "flux2_klein_9b_kv",
    // Qwen-Image (sc-8669, Group-B): base T2I + the two Edit-2511 ids ship the standard
    // q4/q8/bf16 turnkey. Like FLUX.2-klein only the transformer is packed (the Qwen2.5-VL text
    // encoder is skip_quantization, the VAE is all-conv), so the TE/VAE stay dense bf16 in every
    // tier — but, UNLIKE klein, the qwen loader never quantizes the TE regardless of the load
    // Quant, so these do NOT declare `mlx.denseTextEncoderTier`: the q4/q8 load-quant is a harmless
    // no-op on the already-packed transformer, and the bf16 tier resolves to Quant::None anyway.
    // `qwen_image_edit_2511` + `_2511_lightning` share one repo (same Edit-2511 checkpoint).
    "qwen_image",
    "qwen_image_edit_2511",
    "qwen_image_edit_2511_lightning",
    // FLUX.1 (sc-8669, Group-B): schnell + dev ship the standard q4/q8/bf16 turnkey. FLUX quantizes
    // all four components (DiT transformer + CLIP + T5 + VAE attention), so the TE is packed too —
    // hence no `mlx.denseTextEncoderTier` (the q4/q8 load-quant is a harmless no-op on already-packed
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
    // NOT need a `mlx.denseTextEncoderTier` declaration (only `flux2_klein_9b` / `_kv` carry that flag;
    // the "NOT" was dropped by a rename, which left this block contradicting its own UNLIKE clause two
    // lines above). The SANA descriptor now advertises supported_quants
    // Q4/Q8 (mlx-gen #654), so `supports_quant()` is true and they flow through the same
    // resolve_quant + reconcile path as every other matrix model (no more no-quant special case).
    "sana_1600m",
    "sana_sprint_1600m",
    // Kolors (sc-9946, epic 8506): the `SceneWorks/kolors-mlx` turnkey ships standard q4/q8/bf16
    // tiers. mlx-gen #659 packs the SDXL-style UNet + the ChatGLM3-6B `ChatGlmLinear` projections
    // and packed-detects on load; the SDXL VAE stays dense in every tier. Like flux1/sana (and
    // UNLIKE the dense-TE klein class) the ChatGLM3 TE is packed, so the q4/q8 load-quant is a
    // harmless no-op on the already-packed weights and bf16 resolves to Quant::None — no
    // `mlx.denseTextEncoderTier` declaration. The kolors descriptor already advertises supported_quants Q4/Q8,
    // so it flows through the same resolve_quant + reconcile path as every other matrix model.
    "kolors",
];

// The paragraph that used to sit here documented `DENSE_TE_TIER_MODELS` — a const that had already
// moved to `tier_resolver.rs` and is DELETED entirely by sc-15799, so its doc block was orphaned onto
// `dense_tier_subdir` below and read as that function's documentation. The rationale it carried
// (quantize the transformer, keep the text encoder bf16; `resolve_quant` must therefore return `None`)
// now lives with `tier_resolver::is_dense_te_tier`, beside the manifest flag that is the only way to
// request it.

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
    let convrot_dit = resolve_krea_convrot_dit(settings)?;
    Some((base_dir, convrot_dit))
}

/// Resolve only the immutable ConvRot DiT single-file. Strict-pose routing uses this narrower probe so
/// it can claim the Krea control lane before the shared bf16 tokenizer/TE/VAE surface is fetched on
/// demand; generation then calls [`ensure_krea_convrot_base_present`] before resolving the full pair.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_krea_convrot_dit(settings: &Settings) -> Option<PathBuf> {
    huggingface_snapshot_dir(&settings.data_dir, KREA_CONVROT_REPO)
        .map(|root| root.join(KREA_CONVROT_DIT_FILE))
        .filter(|file| file.is_file())
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
    let repo = model_repo(request, &model);
    let root = if repo == model.default_repo() {
        crate::model_jobs::huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            &repo,
            BOOGU_MLX_TURNKEY_REVISION,
        )
    } else {
        huggingface_snapshot_dir(&settings.data_dir, &repo)
    };
    let repo_is_installed = root.is_some()
        || (repo == model.default_repo()
            && huggingface_snapshot_dir(&settings.data_dir, &repo).is_some());
    if !repo_is_installed {
        // Turnkey not downloaded at all → leave it to the load path's "weights not found" error.
        return Ok(());
    }
    let tier_dir = root.as_ref().map(|root| root.join(&tier));
    // Present already (packed single-file q4 OR sharded-dense bf16 `.index.json`) → no fetch.
    if tier_dir.as_ref().is_some_and(|tier_dir| {
        tier_dir
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file()
            || tier_dir
                .join("transformer/diffusion_pytorch_model.safetensors.index.json")
                .is_file()
    })
    {
        return Ok(());
    }
    // The tier subfolder nests transformer/mllm/vae (leaf-dir globs, like the catalog Q8 entry).
    let files = vec![
        format!("{tier}/transformer/*"),
        format!("{tier}/mllm/*"),
        format!("{tier}/vae/*"),
    ];
    let revision =
        turnkey_tier_revision(&repo, model.default_repo(), BOOGU_MLX_TURNKEY_REVISION);
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, &repo, revision, &files)
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
    let repo = model_repo(request, &model);
    let root = if repo == model.default_repo() {
        crate::model_jobs::huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            &repo,
            IDEOGRAM_MLX_TURNKEY_REVISION,
        )
    } else {
        huggingface_snapshot_dir(&settings.data_dir, &repo)
    };
    let repo_is_installed = root.is_some()
        || (repo == model.default_repo()
            && huggingface_snapshot_dir(&settings.data_dir, &repo).is_some());
    if !repo_is_installed {
        // Turnkey not downloaded at all → leave it to the load path's "weights not found" error.
        return Ok(());
    }
    // Present already (the packed single-file transformer) → no fetch.
    if root.as_ref().is_some_and(|root| {
        root.join(tier)
            .join("transformer/model.safetensors")
            .is_file()
    })
    {
        return Ok(());
    }
    let files = vec![format!("{tier}/*")];
    let revision =
        turnkey_tier_revision(&repo, model.default_repo(), IDEOGRAM_MLX_TURNKEY_REVISION);
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, &repo, revision, &files)
    .await
    .map(|_| ())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
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
    // declared dense-TE (`mlx.denseTextEncoderTier`) and on the candle txt2img lane (whose `resolve_quant` call is gated only
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

/// The transformer-tier bit count a dense-TE turnkey (FLUX.2-klein — a declared
/// `mlx.denseTextEncoderTier` entry) actually asked for, derived from `advanced.mlxQuantize` the SAME
/// way [`standard_tier_subdir`]
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

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_certified_artifact_path(
    engine_id: &str,
    settings: &Settings,
    weights_dir: &Path,
    tier: &str,
) -> bool {
    let (repo, revision) = match engine_id {
        "z_image" | "z_image_turbo" =>
            (ZIMAGE_MLX_TURNKEY_REPO, ZIMAGE_MLX_TURNKEY_REVISION),
        "flux1_schnell" =>
            ("SceneWorks/flux1-schnell-mlx", FLUX1_SCHNELL_MLX_TURNKEY_REVISION),
        "flux1_dev" => ("SceneWorks/flux1-dev-mlx", FLUX1_DEV_MLX_TURNKEY_REVISION),
        "flux2_dev" => ("SceneWorks/flux2-dev-mlx", FLUX2_DEV_MLX_TURNKEY_REVISION),
        "flux2_klein_9b" => (
            "SceneWorks/flux2-klein-9b-mlx",
            FLUX2_KLEIN_9B_MLX_TURNKEY_REVISION,
        ),
        _ => return false,
    };
    candle_certified_hf_artifact_path(settings, repo, revision, Path::new(tier), weights_dir)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn is_mage_engine(engine_id: &str) -> bool {
    matches!(
        engine_id,
        "mage_flow_base"
            | "mage_flow"
            | "mage_flow_turbo"
            | "mage_flow_edit_base"
            | "mage_flow_edit"
            | "mage_flow_edit_turbo"
    )
}

/// Certify the complete artifact identity consumed by one registered image load. Most providers
/// are self-contained, so their identity is the tier directory alone. Mage is split: its DiT lives
/// in the model-specific snapshot while the text encoder and VAE live in a shared pinned snapshot.
/// Optimized evidence is admissible only when every descriptor-required component matches the
/// exact manifest repo, revision, tier, and subdirectory that the eventual provider load receives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_certified_load_spec(
    engine_id: &str,
    settings: &Settings,
    spec: &LoadSpec,
    manifest_entry: &JsonObject,
    tier: &str,
) -> bool {
    let WeightsSource::Dir(weights_dir) = &spec.weights else {
        return false;
    };
    if !is_mage_engine(engine_id) {
        return candle_certified_artifact_path(engine_id, settings, weights_dir, tier);
    }

    let Some(downloads) = manifest_entry.get("downloads").and_then(Value::as_array) else {
        return false;
    };
    let Some(descriptor) = crate::inference_runtime::media_descriptor(engine_id) else {
        return false;
    };
    if spec.components.len() != descriptor.required_components.len() {
        return false;
    }

    let certify = |component_id: Option<&str>, actual: &Path| {
        let mut matches = downloads.iter().filter(|download| {
            download.get("provider").and_then(Value::as_str) == Some("huggingface")
                && download.get("variant").and_then(Value::as_str) == Some(tier)
                && download.get("componentId").and_then(Value::as_str) == component_id
                && download.get("coRequisite").and_then(Value::as_bool).unwrap_or(false)
                    == component_id.is_some()
        });
        let Some(download) = matches.next() else {
            return false;
        };
        if matches.next().is_some() {
            return false;
        }
        let (Some(repo), Some(revision)) = (
            download.get("repo").and_then(Value::as_str),
            download.get("revision").and_then(Value::as_str),
        ) else {
            return false;
        };
        let relative = download
            .get("subdir")
            .and_then(Value::as_str)
            .unwrap_or(tier);
        candle_certified_hf_artifact_path(
            settings,
            repo,
            revision,
            Path::new(relative),
            actual,
        )
    };

    certify(None, weights_dir)
        && descriptor.required_components.iter().all(|component_id| {
            matches!(
                spec.components.get(*component_id),
                Some(WeightsSource::Dir(path))
                    if certify(Some(component_id), path)
            )
        })
}

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod mage_artifact_certification_tests {
    use super::*;
    use std::fs;

    fn settings(data_dir: &Path) -> Settings {
        Settings {
            api_url: "http://127.0.0.1".to_owned(),
            access_token: None,
            data_dir: data_dir.to_path_buf(),
            config_dir: data_dir.join("config"),
            worker_id: "test-worker".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 1,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
            max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: false,
            backend_candle_enabled: true,
            gpu_memory_limit_bytes: 0,
            external_model_roots: Vec::new(),
        }
    }

    fn mage_manifest(model_id: &str) -> JsonObject {
        let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, source)| *source)
            .expect("embedded model manifest");
        let root: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(source))
            .expect("model manifest parses");
        root["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|model| model["id"] == model_id)
            .and_then(Value::as_object)
            .cloned()
            .expect("Mage manifest entry")
    }

    fn cached_snapshot(data_dir: &Path, repo: &str, revision: &str) -> PathBuf {
        let root = sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, repo)
            .expect("safe fixture repo")
            .join("snapshots")
            .join(revision);
        fs::create_dir_all(&root).expect("create fixture snapshot");
        root
    }

    #[test]
    fn mage_certification_binds_backbone_and_every_required_component() {
        let data = tempfile::tempdir().expect("temp data dir");
        let settings = settings(data.path());
        let manifest = mage_manifest("mage_flow_base");
        let downloads = manifest["downloads"].as_array().expect("downloads");
        let revision_for = |component_id: Option<&str>| {
            let download = downloads
                .iter()
                .find(|download| {
                    download["variant"] == "q4"
                        && download.get("componentId").and_then(Value::as_str) == component_id
                })
                .expect("tier artifact");
            (
                download["repo"].as_str().expect("repo"),
                download["revision"].as_str().expect("revision"),
            )
        };
        let (backbone_repo, backbone_revision) = revision_for(None);
        let (components_repo, components_revision) = revision_for(Some("text_encoder"));
        let backbone = cached_snapshot(data.path(), backbone_repo, backbone_revision).join("q4");
        let components = cached_snapshot(data.path(), components_repo, components_revision).join("q4");
        let text_encoder = components.join("text_encoder");
        let vae = components.join("vae");
        for path in [&backbone, &text_encoder, &vae] {
            fs::create_dir_all(path).expect("create tier artifact");
        }

        let complete = LoadSpec::new(WeightsSource::Dir(backbone.clone()))
            .with_component("text_encoder", WeightsSource::Dir(text_encoder.clone()))
            .with_component("vae", WeightsSource::Dir(vae.clone()));
        assert!(candle_certified_load_spec(
            "mage_flow_base",
            &settings,
            &complete,
            &manifest,
            "q4",
        ));

        let substituted_vae = LoadSpec::new(WeightsSource::Dir(backbone.clone()))
            .with_component("text_encoder", WeightsSource::Dir(text_encoder))
            .with_component("vae", WeightsSource::Dir(data.path().join("other-vae")));
        assert!(!candle_certified_load_spec(
            "mage_flow_base",
            &settings,
            &substituted_vae,
            &manifest,
            "q4",
        ));

        let incomplete = LoadSpec::new(WeightsSource::Dir(backbone))
            .with_component("vae", WeightsSource::Dir(vae));
        assert!(!candle_certified_load_spec(
            "mage_flow_base",
            &settings,
            &incomplete,
            &manifest,
            "q4",
        ));
    }
}

/// Exact path identity for an artifact inside one immutable Hugging Face snapshot. Optimized memory
/// evidence is artifact-specific: resolving the same filename from `refs/main`, an environment
/// override, or another registered overlay must not inherit the canonical fixture's measurements.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_certified_hf_artifact_path(
    settings: &Settings,
    repo: &str,
    revision: &str,
    relative: &Path,
    actual: &Path,
) -> bool {
    candle_pinned_hf_artifact_path(settings, repo, revision, relative)
        .is_some_and(|expected| candle_artifact_path_matches(actual, &expected))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_pinned_hf_artifact_path(
    settings: &Settings,
    repo: &str,
    revision: &str,
    relative: &Path,
) -> Option<PathBuf> {
    let root = crate::model_jobs::huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        repo,
        revision,
    )?;
    Some(root.join(relative))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_artifact_path_matches(actual: &Path, expected: &Path) -> bool {
    match (
        std::fs::canonicalize(actual),
        std::fs::canonicalize(expected),
    ) {
        (Ok(actual), Ok(expected)) => actual == expected,
        _ => actual == expected,
    }
}

/// Identity of the Krea Turbo calibration this request was admitted under, built from the two
/// constants the selector itself compares against (sc-17097).
///
/// It used to be a literal in two places, so a re-measurement had to remember to edit both; the
/// tracing line and the run context would otherwise disagree about which capture was in force.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn krea_evidence_revision() -> String {
    format!(
        "{}@{}",
        crate::vram_gate::KREA_TURBO_SCENEWORKS_REVISION,
        // sc-17774: the lane's compile-closure digest replaced the frozen inference SHA here too, so
        // the traced receipt names the same thing the selector actually compared.
        sceneworks_core::memory_calibration::packaged_closure_digest("candle", "krea_2_turbo")
            .unwrap_or_default()
    )
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn optimized_shared_memory_context(
    context: gen_core::MemoryRunContext,
) -> Option<gen_core::MemoryRunContext> {
    context.selection.strategy.is_optimized().then_some(context)
}

/// The registered FLUX.2-dev generator is a text-to-image provider even when the catalog surface
/// originated from a style-variation prompt. Keep that provider-specific canonicalization here so
/// Z-Image, Qwen, and FLUX.1 retain their established mode/evidence semantics.
#[cfg(any(
    test,
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn candle_base_memory_request_mode<'a>(engine_id: &str, request_mode: &'a str) -> &'a str {
    if engine_id == "flux2_dev"
        || (engine_id == "flux2_klein_9b"
            && matches!(request_mode, "image_generation" | "text_to_image"))
    {
        "text_to_image"
    } else {
        request_mode
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn shared_image_reference_count(
    edit_reference_count: usize,
    has_single_reference: bool,
    has_edit_mask: bool,
    has_hires_fix: bool,
) -> u32 {
    if has_hires_fix {
        hires_fix_reference_count()
    } else {
        lane_reference_count(
            has_single_reference,
            edit_reference_count,
            has_edit_mask,
        )
    }
}

#[cfg(any(
    test,
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn flux2_klein_reference_bearing_mode(mode: &str) -> bool {
    matches!(
        mode,
        "edit_image"
            | "reference"
            | "image_to_image"
            | "character_image"
            | "style_variations"
    )
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
    resolve_advanced_or_manifest_u32_with(request, key, || default, range)
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
    resolve_advanced_or_manifest_f32_with(request, key, || default, range)
}

/// Closure-default twin of [`resolve_advanced_or_manifest_u32`] (sc-8825). Identical mechanism —
/// `advanced[key]` → manifest `[key]` (parsed, clamped to `range`) — except the fallback is a
/// per-lane `default_fn` closure evaluated **only** when both advanced and manifest are absent (and,
/// like the const twin, returned **unclamped**). This covers the lanes whose default is model-dependent
/// (`flux_ipadapter` variant steps, `qwen_edit_candle`/`zimage_control` per-variant steps) rather than a
/// bare constant. Every caller passes its OWN `range`/`default_fn`, so per-lane bounds stay byte-for-byte.
///
#[cfg(any(
    target_os = "macos",
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
/// per-variant fn. Each caller passes its OWN `range`/`default_fn`.
#[cfg(any(
    target_os = "macos",
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

/// Apply the Ideogram placeholder recovery policy to one completed render. The renderer is injected
/// so the MLX and candle lanes share cancellation, reseeding, retry exhaustion, and error behavior.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn recover_ideogram_placeholder<R>(
    enabled: bool,
    seed: i64,
    cancel: &CancelFlag,
    initial: (u32, u32, Vec<u8>),
    mut render: R,
) -> WorkerResult<(i64, u32, u32, Vec<u8>)>
where
    R: FnMut(i64) -> WorkerResult<(u32, u32, Vec<u8>)>,
{
    recover_ideogram_placeholder_with(
        enabled,
        crate::ideogram_caption::placeholder_recovery_retries(),
        seed,
        cancel,
        initial,
        |pixels, width, height| {
            crate::ideogram_caption::looks_like_placeholder(pixels, width, height)
        },
        &mut render,
    )
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn recover_ideogram_placeholder_with<R, P>(
    enabled: bool,
    retries: u32,
    seed: i64,
    cancel: &CancelFlag,
    initial: (u32, u32, Vec<u8>),
    mut is_placeholder: P,
    mut render: R,
) -> WorkerResult<(i64, u32, u32, Vec<u8>)>
where
    R: FnMut(i64) -> WorkerResult<(u32, u32, Vec<u8>)>,
    P: FnMut(&[u8], u32, u32) -> bool,
{
    let (mut width, mut height, mut pixels) = initial;
    if !enabled || !is_placeholder(&pixels, width, height) {
        return Ok((seed, width, height, pixels));
    }

    let mut final_seed = seed;
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
        (width, height, pixels) = render(retry_seed)?;
        final_seed = retry_seed;
        if !is_placeholder(&pixels, width, height) {
            break;
        }
    }
    Ok((final_seed, width, height, pixels))
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
        let file = resolve_adapter_file(lora, settings)?;
        let kind = classify_adapter(&file)?;
        let scale = lora_weight(lora) as f32;
        specs.push(AdapterSpec::new(file, scale, kind));
    }
    Ok(specs)
}

/// Whether Krea may use calibrated streamed blocks, plus the load-exact adapter bytes supporting
/// that decision. A non-empty stack with unreadable metadata returns `None`; the caller refuses it
/// rather than inheriting evidence from the adapter-free route.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn krea_streamed_blocks_adapter_evidence(adapters: &[AdapterSpec]) -> (bool, Option<u64>) {
    if adapters.is_empty() {
        return (true, Some(0));
    }
    let bytes = adapters.iter().try_fold(0_u64, |total, adapter| {
        let bytes = gen_core::safetensors_path_bytes(&adapter.path);
        (bytes > 0).then(|| total.saturating_add(bytes))
    });
    (false, bytes)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_adapter_resident_bytes(
    engine_id: &str,
    tier: &str,
    measured_source_bytes: u64,
) -> u64 {
    // Krea keeps lazily merged adapter overlays on its resident host. Z-Image likewise installs
    // forward-time residuals for dense bf16 as well as packed tiers, so both families must reserve
    // the exact adapter source bytes at every precision. Other generic Candle providers fold dense
    // factors and retain residuals only on packed tiers.
    if engine_id.starts_with("krea_2_") || engine_id.starts_with("z_image") || tier != "bf16" {
        measured_source_bytes
    } else {
        0
    }
}

fn mlx_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    // This key is worker-owned. A replayed recipe or hostile advanced payload must never be able to
    // forge Auto's rung-4 decision; the trusted Candle path may add it back after final admission.
    scrub_untrusted_memory_strategy_disclosure(&mut raw);
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
    if request.hires_fix.enabled {
        raw.insert(
            "hiresFix".to_owned(),
            serde_json::to_value(&request.hires_fix).expect("HiresFixRequest is serializable"),
        );
    }
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

/// Select deferred materialization for the native Candle/CUDA Qwen routes. Only the uniform
/// base/edit transformer trunk can be reopened one window at a time: control/IP/PiD and adapter
/// overlays keep the established eager shape and therefore make rung 4 explicitly unavailable.
///
/// Gated to exactly the predicate its call sites live under — `generate_candle_stream`, the
/// `qwen_edit_candle` module, and `candle_qwen_load_shape_tests` all carry
/// `all(not(target_os = "macos"), feature = "backend-candle")`. Note this deliberately does NOT
/// use the `any(test, ...)` spelling of the sibling `apply_request_scoped_candle_residency`: this
/// function's test module is itself macOS-excluded, so admitting bare `test` would leave it unused
/// in a macOS test build and dead-code-error again under `-D warnings`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn apply_candle_image_load_shape(engine_id: &str, spec: LoadSpec) -> LoadSpec {
    let directory = matches!(&spec.weights, WeightsSource::Dir(_));
    let qwen_native = matches!(engine_id, "qwen_image" | "qwen_image_edit")
        && directory
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.adapters.is_empty()
        && spec.pid.is_none();
    let flux_supported = matches!(engine_id, "flux1_schnell" | "flux1_dev")
        && directory
        && spec.adapters.is_empty()
        && spec.extra_controls.is_empty()
        && spec.pid.is_none()
        && spec.identity.is_none()
        && !(spec.control.is_some() && spec.ip_adapter.is_some());
    let flux2_supported = matches!(engine_id, "flux2_dev" | "flux2_klein_9b")
        && directory
        && spec.adapters.is_empty()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.pid.is_none()
        && spec.identity.is_none();
    let mage_supported = matches!(
        engine_id,
        "mage_flow_base"
            | "mage_flow"
            | "mage_flow_turbo"
            | "mage_flow_edit_base"
            | "mage_flow_edit"
            | "mage_flow_edit_turbo"
    ) && directory
        && spec.adapters.is_empty()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.pid.is_none()
        && spec.identity.is_none();
    if qwen_native || flux_supported || flux2_supported || mage_supported {
        spec.with_load_shape(gen_core::LoadShape::DeferredMaterialization)
    } else {
        spec
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn apply_candle_qwen_load_shape(engine_id: &str, spec: LoadSpec) -> LoadSpec {
    apply_candle_image_load_shape(engine_id, spec)
}

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod candle_image_load_shape_tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn fixture_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
            "qwen-image-fixture",
        )))
    }

    #[test]
    fn adopting_native_image_routes_use_deferred_materialization() {
        for engine_id in [
            "qwen_image",
            "qwen_image_edit",
            "flux1_schnell",
            "flux1_dev",
            "flux2_dev",
            "flux2_klein_9b",
            "mage_flow_base",
            "mage_flow",
            "mage_flow_turbo",
            "mage_flow_edit_base",
            "mage_flow_edit",
            "mage_flow_edit_turbo",
        ] {
            assert_eq!(
                apply_candle_image_load_shape(engine_id, fixture_spec()).load_shape,
                gen_core::LoadShape::DeferredMaterialization
            );
        }
    }

    #[test]
    fn unsupported_overlays_and_non_streamable_routes_remain_eager() {
        let adapter = gen_core::AdapterSpec::new(
            std::path::PathBuf::from("adapter.safetensors"),
            1.0,
            gen_core::AdapterKind::Lora,
        );
        let cases = [
            ("qwen_image", fixture_spec().with_adapters(vec![adapter])),
            (
                "qwen_image_edit",
                fixture_spec().with_pid(
                    WeightsSource::File(std::path::PathBuf::from("pid.safetensors")),
                    WeightsSource::Dir(std::path::PathBuf::from("gemma")),
                ),
            ),
            (
                "mage_flow_edit",
                fixture_spec().with_pid(
                    WeightsSource::File(std::path::PathBuf::from("pid.safetensors")),
                    WeightsSource::Dir(std::path::PathBuf::from("gemma")),
                ),
            ),
            ("z_image", fixture_spec()),
        ];
        for (engine_id, spec) in cases {
            assert_eq!(
                apply_candle_image_load_shape(engine_id, spec).load_shape,
                gen_core::LoadShape::EagerMaterialization
            );
        }
    }

    #[test]
    fn shared_reference_count_preserves_mage_multi_reference_geometry() {
        assert_eq!(shared_image_reference_count(0, false, false, false), 0);
        assert_eq!(shared_image_reference_count(0, true, false, false), 1);
        assert_eq!(shared_image_reference_count(1, true, false, false), 1);
        assert_eq!(shared_image_reference_count(8, true, false, false), 8);
        assert_eq!(shared_image_reference_count(0, true, true, false), 2);
        assert_eq!(shared_image_reference_count(8, false, true, false), 9);
        assert_eq!(shared_image_reference_count(8, true, true, true), 1);
    }

    #[test]
    fn flux_single_control_or_ip_overlay_keeps_deferred_materialization() {
        for spec in [
            fixture_spec().with_control(WeightsSource::File(std::path::PathBuf::from(
                "control.safetensors",
            ))),
            fixture_spec().with_ip_adapter(WeightsSource::Dir(std::path::PathBuf::from("ip"))),
        ] {
            assert_eq!(
                apply_candle_image_load_shape("flux1_dev", spec).load_shape,
                gen_core::LoadShape::DeferredMaterialization
            );
        }
        let combined = fixture_spec()
            .with_control(WeightsSource::File(std::path::PathBuf::from(
                "control.safetensors",
            )))
            .with_ip_adapter(WeightsSource::Dir(std::path::PathBuf::from("ip")));
        assert_eq!(
            apply_candle_image_load_shape("flux1_dev", combined).load_shape,
            gen_core::LoadShape::EagerMaterialization
        );
        assert_eq!(
            apply_candle_image_load_shape(
                "flux2_dev",
                fixture_spec().with_control(WeightsSource::File(std::path::PathBuf::from(
                    "control.safetensors",
                )))
            )
            .load_shape,
            gen_core::LoadShape::DeferredMaterialization
        );
    }

    struct ResidentOverwriteScope;

    impl gen_core::MemoryRequestScope for ResidentOverwriteScope {
        fn configure_request(
            &mut self,
            request: &mut gen_core::GenerationRequest,
        ) -> gen_core::Result<()> {
            request.memory = Some(gen_core::GenerationMemory::default());
            Ok(())
        }

        fn enter_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }
        fn leave_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }
        fn configure_decode(
            &mut self,
            _tile_edge: u32,
            _overlap: u32,
            _geometry: gen_core::MemoryGeometry,
        ) -> gen_core::Result<()> {
            Ok(())
        }
        fn configure_attention(&mut self, _chunk_size: u32) -> gen_core::Result<()> {
            Ok(())
        }
        fn materialize_transformer_window(
            &mut self,
            _first_block: u32,
            _block_count: u32,
        ) -> gen_core::Result<()> {
            Ok(())
        }
        fn finish(&mut self, _outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
            Ok(())
        }
    }

    struct LegacyFallbackGenerator {
        observed_stage_residency: Arc<Mutex<Option<bool>>>,
        descriptor: gen_core::ModelDescriptor,
    }

    impl LegacyFallbackGenerator {
        fn new(observed_stage_residency: Arc<Mutex<Option<bool>>>) -> Self {
            Self {
                observed_stage_residency,
                descriptor: gen_core::ModelDescriptor {
                    id: "legacy_fallback_fixture",
                    family: "test",
                    backend: "candle",
                    modality: gen_core::Modality::Image,
                    capabilities: Default::default(),
                    required_components: &[],
                    control_kinds: None,
                },
            }
        }
    }

    impl gen_core::Generator for LegacyFallbackGenerator {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }

        fn begin_memory_strategy_request(
            &self,
            _context: &gen_core::MemoryRunContext,
        ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
            Ok(Some(Box::new(ResidentOverwriteScope)))
        }

        fn validate(&self, _request: &gen_core::GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            request: &gen_core::GenerationRequest,
            _on_progress: &mut dyn FnMut(gen_core::Progress),
        ) -> gen_core::Result<gen_core::GenerationOutput> {
            *self.observed_stage_residency.lock().unwrap() =
                Some(request.memory.unwrap_or_default().stage_residency);
            Ok(gen_core::GenerationOutput::Images(Vec::new()))
        }
    }

    fn resident_context() -> gen_core::MemoryRunContext {
        gen_core::MemoryRunContext {
            selection: gen_core::MemorySelection {
                strategy: gen_core::MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: gen_core::MemoryNumericTier {
                    precision: gen_core::Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
            },
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: String::new(),
            load_shape: gen_core::LoadShape::DeferredMaterialization,
            mode: gen_core::MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: gen_core::MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: gen_core::MemoryBudget {
                total_bytes: 24,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 20,
            cache_state: gen_core::MemoryCacheState::Cold,
            evidence_revision: "resident-sentinel".to_owned(),
        }
    }

    #[test]
    fn resident_shared_context_cannot_overwrite_legacy_sequential_request() {
        let observed = Arc::new(Mutex::new(None));
        let generator = LegacyFallbackGenerator::new(Arc::clone(&observed));
        let mut request = gen_core::GenerationRequest {
            memory: Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let context = optimized_shared_memory_context(resident_context());
        assert!(context.is_none(), "Resident is a fallback sentinel, not an execution scope");

        crate::memory_strategy::generate_with_scope(
            &generator,
            &mut request,
            context.as_ref(),
            &mut |_| {},
        )
        .expect("legacy fallback generation");

        assert_eq!(*observed.lock().unwrap(), Some(true));
        assert!(request.memory.unwrap().stage_residency);
    }
}

/// Select deferred materialization for MLX routes with a measured load-exact contract. The fit gate
/// still owns the independent Resident/Sequential decision; a constrained request adds Sequential
/// and can then select the provider's bounded transformer rung. Lens has two deliberately separate
/// measured identities at the current provider pin: base Lens Q4 exposes the full ladder, while
/// Lens-Turbo BF16 retains its legacy text-encoder-only rung. Qwen base/edit also cover Q4/Q8 and
/// forward-time adapters because the block stream replays packed quantization and captured residuals.
/// Request-aware callers use the private helper below so Krea and SDXL base calibration is never
/// applied to reference, edit, or Hires.fix surfaces that share the same engine and weight shape.
#[cfg(target_os = "macos")]
pub(crate) fn apply_measured_mlx_load_shape(engine_id: &str, spec: LoadSpec) -> LoadSpec {
    apply_measured_mlx_load_shape_for_request(engine_id, spec, true)
}

#[cfg(target_os = "macos")]
fn apply_measured_mlx_load_shape_for_request(
    engine_id: &str,
    spec: LoadSpec,
    plain_text_to_image: bool,
) -> LoadSpec {
    let directory_native = matches!(&spec.weights, WeightsSource::Dir(_))
        && spec.precision == gen_core::Precision::Bf16;
    let lens_native = directory_native
        && ((engine_id == "lens" && spec.quantize == Some(gen_core::Quant::Q4))
            || (engine_id == "lens_turbo" && spec.quantize.is_none()))
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.adapters.is_empty()
        && spec.pid.is_none();
    let qwen_native = directory_native
        && matches!(engine_id, "qwen_image" | "qwen_image_edit")
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.pid.is_none();
    let krea_native = directory_native
        && engine_id == "krea_2_turbo"
        && plain_text_to_image
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.adapters.is_empty()
        && spec.pid.is_none();
    let sdxl_native = directory_native
        && engine_id == "sdxl"
        && plain_text_to_image
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.adapters.is_empty()
        && spec.pid.is_none();
    if lens_native || qwen_native || krea_native || sdxl_native {
        spec.with_load_shape(gen_core::LoadShape::DeferredMaterialization)
    } else {
        spec
    }
}

#[cfg(all(test, target_os = "macos"))]
mod measured_mlx_load_shape_tests {
    use super::*;
    use gen_core::{
        AdapterKind, AdapterSpec, MemoryStrategy, MemoryStrategySupport, OffloadPolicy, Quant,
        TransformerComponent,
    };

    fn fixture_spec(root: &std::path::Path, quant_bits: Option<u8>) -> LoadSpec {
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            // A minimal VALID safetensors file carrying one f32 scalar: the asset-facts
            // unification (inference PR #395) parses headers — and requires at least one tensor —
            // when the provider contract is built, so a zero-filled stub now fails the lookup.
            let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(&0f32.to_le_bytes());
            std::fs::write(dir.join("model.safetensors"), &bytes).unwrap();
        }
        std::fs::write(
            root.join("text_encoder/config.json"),
            quant_bits.map_or_else(
                || r#"{"dtype":"bfloat16"}"#.to_owned(),
                |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("transformer/config.json"),
            quant_bits.map_or_else(
                || r#"{"dtype":"bfloat16"}"#.to_owned(),
                |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            ),
        )
        .unwrap();
        LoadSpec::new(WeightsSource::Dir(root.to_owned()))
    }

    fn rung_four_support(engine_id: &str, spec: &LoadSpec) -> MemoryStrategySupport {
        crate::inference_runtime::media()
            .memory_strategy_contract(engine_id, spec)
            .unwrap()
            .expect("measured provider registers a memory-strategy contract")
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .expect("compatibility contract contains rung 4")
            .support
            .clone()
    }

    #[test]
    fn worker_lens_specs_reach_only_their_exact_measured_contracts() {
        let bf16_dir = tempfile::tempdir().unwrap();
        let bf16 = fixture_spec(bf16_dir.path(), None);
        let q4_dir = tempfile::tempdir().unwrap();
        let q4 = fixture_spec(q4_dir.path(), Some(4)).with_quant(Quant::Q4);

        let turbo = apply_measured_mlx_load_shape("lens_turbo", bf16.clone());
        assert_eq!(turbo.load_shape, gen_core::LoadShape::DeferredMaterialization);
        assert_eq!(
            rung_four_support("lens_turbo", &turbo),
            MemoryStrategySupport::Missing,
            "the legacy dense Lens-Turbo rung still requires Sequential"
        );
        let turbo = turbo.with_offload_policy(OffloadPolicy::Sequential);
        let turbo_contract = crate::inference_runtime::media()
            .memory_strategy_contract("lens_turbo", &turbo)
            .unwrap()
            .expect("Lens-Turbo contract");
        let turbo_rung = turbo_contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .expect("Lens-Turbo rung 4");
        assert_eq!(turbo_rung.support, MemoryStrategySupport::Implemented);
        assert_eq!(turbo_rung.parameters.transformer_window_sizes, vec![1]);
        assert_eq!(
            turbo_rung.parameters.transformer_window_components,
            vec![TransformerComponent::TextEncoder]
        );

        let lens = apply_measured_mlx_load_shape("lens", q4.clone());
        assert_eq!(lens.load_shape, gen_core::LoadShape::DeferredMaterialization);
        let lens_contract = crate::inference_runtime::media()
            .memory_strategy_contract("lens", &lens)
            .unwrap()
            .expect("Lens contract");
        let lens_rung = lens_contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .expect("Lens rung 4");
        assert_eq!(lens_rung.support, MemoryStrategySupport::Implemented);
        assert_eq!(lens_rung.parameters.transformer_window_sizes, vec![1]);
        assert_eq!(
            lens_rung.parameters.transformer_window_components,
            vec![TransformerComponent::Both]
        );

        let adapter = AdapterSpec::new(
            q4_dir.path().join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        );
        let unsupported = [
            ("lens", bf16.clone()),
            ("lens_turbo", q4),
            ("lens", bf16.clone().with_quant(Quant::Q8)),
            ("lens", bf16.clone().with_adapters(vec![adapter])),
            (
                "lens",
                bf16.with_pid(
                    WeightsSource::File(q4_dir.path().join("pid.safetensors")),
                    WeightsSource::Dir(q4_dir.path().join("gemma")),
                ),
            ),
        ];
        for (engine_id, spec) in unsupported {
            let shaped = apply_measured_mlx_load_shape(engine_id, spec)
                .with_offload_policy(OffloadPolicy::Sequential);
            assert_eq!(
                shaped.load_shape,
                gen_core::LoadShape::EagerMaterialization,
                "unmeasured Lens entry/tier/overlay combinations must remain eager"
            );
            assert_eq!(
                rung_four_support(engine_id, &shaped),
                MemoryStrategySupport::Missing
            );
        }
    }

    #[test]
    fn worker_qwen_specs_reach_the_shared_base_edit_and_lightning_contract() {
        let bf16_dir = tempfile::tempdir().unwrap();
        let bf16 = fixture_spec(bf16_dir.path(), None);
        let q4_dir = tempfile::tempdir().unwrap();
        let q4 = fixture_spec(q4_dir.path(), None);
        std::fs::write(
            q4_dir.path().join("transformer/config.json"),
            r#"{"quantization":{"bits":4}}"#,
        )
        .unwrap();
        let lightning_adapter = AdapterSpec::new(
            q4_dir.path().join("lightning.safetensors"),
            1.0,
            AdapterKind::Lora,
        );

        for (label, engine_id, spec) in [
            ("qwen_image", "qwen_image", bf16.clone()),
            (
                "qwen_image_edit_2511",
                "qwen_image_edit",
                q4.clone().with_quant(Quant::Q4),
            ),
            (
                "qwen_image_edit_2511_lightning",
                "qwen_image_edit",
                q4.with_quant(Quant::Q4)
                    .with_adapters(vec![lightning_adapter]),
            ),
        ] {
            let shaped = apply_measured_mlx_load_shape(engine_id, spec);
            assert_eq!(
                shaped.load_shape,
                gen_core::LoadShape::DeferredMaterialization,
                "{label} must resolve to the shared Qwen provider load shape"
            );
            let contract = crate::inference_runtime::media()
                .memory_strategy_contract(
                    engine_id,
                    &shaped.with_offload_policy(OffloadPolicy::Sequential),
                )
                .unwrap()
                .expect("Qwen contract");
            let rung = contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .expect("rung 4");
            assert_eq!(rung.support, MemoryStrategySupport::Implemented);
            assert_eq!(rung.parameters.transformer_window_sizes, vec![1]);
            assert_eq!(
                rung.parameters.transformer_window_components,
                vec![TransformerComponent::Dit]
            );
        }

        let pid = bf16.clone().with_pid(
            WeightsSource::File(bf16_dir.path().join("pid.safetensors")),
            WeightsSource::Dir(bf16_dir.path().join("gemma")),
        );
        assert_eq!(
            apply_measured_mlx_load_shape("qwen_image", pid).load_shape,
            gen_core::LoadShape::EagerMaterialization,
            "the native-Qwen-VAE ladder must not opt PiD into its load shape"
        );
        assert_eq!(
            apply_measured_mlx_load_shape("qwen_image_control", bf16).load_shape,
            gen_core::LoadShape::EagerMaterialization,
            "the unbounded five-block control side branch is not advertised as rung 4"
        );
    }

    #[test]
    fn worker_plain_krea_specs_reach_the_full_ladder_without_admitting_other_surfaces() {
        for (tier, quant_bits, quant) in [
            ("bf16", None, None),
            ("q4", Some(4), Some(Quant::Q4)),
            ("q8", Some(8), Some(Quant::Q8)),
        ] {
            let root = tempfile::tempdir().unwrap();
            let mut spec = fixture_spec(root.path(), quant_bits);
            if let Some(quant) = quant {
                spec = spec.with_quant(quant);
            }
            let shaped =
                apply_measured_mlx_load_shape_for_request("krea_2_turbo", spec, true);
            assert_eq!(
                shaped.load_shape,
                gen_core::LoadShape::DeferredMaterialization,
                "plain Krea {tier} must use the production deferred load shape"
            );
            let resident_contract = crate::inference_runtime::media()
                .memory_strategy_contract(
                    "krea_2_turbo",
                    &shaped.clone().with_offload_policy(OffloadPolicy::Resident),
                )
                .unwrap()
                .expect("plain Krea registers a resident/deferred memory-strategy contract");
            assert_eq!(
                resident_contract
                    .capability(MemoryStrategy::Resident)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented,
                "plain Krea {tier} must remain reachable on a roomy resident host"
            );

            let sequential_contract = crate::inference_runtime::media()
                .memory_strategy_contract(
                    "krea_2_turbo",
                    &shaped.with_offload_policy(OffloadPolicy::Sequential),
                )
                .unwrap()
                .expect("plain Krea registers a sequential/deferred memory-strategy contract");
            let rung = sequential_contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .expect("Krea compatibility contract contains rung 4");
            assert_eq!(rung.support, MemoryStrategySupport::Implemented, "{tier}");
            assert_eq!(rung.parameters.transformer_window_sizes, vec![1]);
            assert_eq!(
                rung.parameters.transformer_window_components,
                vec![TransformerComponent::Dit]
            );
        }

        let root = tempfile::tempdir().unwrap();
        let base = fixture_spec(root.path(), Some(4)).with_quant(Quant::Q4);
        assert_eq!(
            apply_measured_mlx_load_shape_for_request("krea_2_turbo", base.clone(), false)
                .load_shape,
            gen_core::LoadShape::EagerMaterialization,
            "a reference/edit/hires request is outside the plain Krea T2I apparatus even when its \
             weight spec is otherwise clean"
        );
        let adapter = AdapterSpec::new(
            root.path().join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        );
        for (engine, spec) in [
            ("krea_2_turbo_edit", base.clone()),
            ("krea_2_turbo_control", base.clone()),
            ("krea_2_turbo", base.clone().with_adapters(vec![adapter])),
            (
                "krea_2_turbo",
                base.with_control(WeightsSource::File(root.path().join("control.safetensors"))),
            ),
        ] {
            assert_eq!(
                apply_measured_mlx_load_shape_for_request(engine, spec, true).load_shape,
                gen_core::LoadShape::EagerMaterialization,
                "{engine} overlay/edit/control surface is outside the base calibration identity"
            );
        }
    }

    #[test]
    fn worker_plain_sdxl_specs_reach_only_the_pinned_three_rung_contract() {
        fn sdxl_spec(root: &std::path::Path, quant_bits: Option<u8>, quant: Option<Quant>) -> LoadSpec {
            for component in ["text_encoder", "text_encoder_2", "unet", "vae"] {
                let directory = root.join(component);
                std::fs::create_dir_all(&directory).unwrap();
                let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
                let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
                bytes.extend_from_slice(header);
                bytes.extend_from_slice(&0_f32.to_le_bytes());
                std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
            }
            if let Some(bits) = quant_bits {
                std::fs::write(
                    root.join("unet").join("config.json"),
                    format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
                )
                .unwrap();
            }
            let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_owned()));
            if let Some(quant) = quant {
                spec = spec.with_quant(quant);
            }
            spec
        }

        for (tier, quant_bits, quant) in [
            ("bf16", None, None),
            ("q4", Some(4), Some(Quant::Q4)),
            ("q8", Some(8), Some(Quant::Q8)),
        ] {
            let root = tempfile::tempdir().unwrap();
            let shaped = apply_measured_mlx_load_shape_for_request(
                "sdxl",
                sdxl_spec(root.path(), quant_bits, quant),
                true,
            );
            assert_eq!(
                shaped.load_shape,
                gen_core::LoadShape::DeferredMaterialization,
                "plain SDXL {tier} must use the production deferred load shape"
            );
            let contract = crate::inference_runtime::media()
                .memory_strategy_contract(
                    "sdxl",
                    &shaped.with_offload_policy(OffloadPolicy::Sequential),
                )
                .unwrap()
                .expect("plain SDXL must resolve its pinned provider contract before GPU work");
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented,
                "{tier}"
            );
            for missing in [MemoryStrategy::BoundedDecode, MemoryStrategy::BoundedAttention] {
                assert_eq!(
                    contract.capability(missing).unwrap().support,
                    MemoryStrategySupport::Missing,
                    "SDXL {tier} must not invent measured-Missing rungs"
                );
            }
        }

        let root = tempfile::tempdir().unwrap();
        let base = sdxl_spec(root.path(), Some(4), Some(Quant::Q4));
        assert_eq!(
            apply_measured_mlx_load_shape_for_request("sdxl", base.clone(), false).load_shape,
            gen_core::LoadShape::EagerMaterialization,
            "SDXL edit/reference/Hires.fix requests are outside the base T2I apparatus"
        );
        let adapter = AdapterSpec::new(
            root.path().join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        );
        for spec in [
            base.clone().with_adapters(vec![adapter]),
            base.with_control(WeightsSource::File(root.path().join("control.safetensors"))),
        ] {
            assert_eq!(
                apply_measured_mlx_load_shape_for_request("sdxl", spec, true).load_shape,
                gen_core::LoadShape::EagerMaterialization,
                "SDXL adapter/control surfaces must not borrow the base calibration identity"
            );
        }
    }
}

/// Stage a media model's caller-provisioned components (epic 13657, sc-13679) onto its `LoadSpec` —
/// the image/video twin of the audio seam (`audio_jobs.rs run_audio_synthesis_using`). It reads the
/// model's descriptor `required_components` from the media registry and resolves each declared id to
/// its cached `coRequisite` snapshot via [`crate::model_jobs::resolve_co_requisites`] (all-or-nothing:
/// a missing co-requisite fails the job with an actionable error BEFORE the engine load), then stages
/// each in `LoadSpec::components` for the engine's own load-time `require_component` gate.
///
/// Providers with a split artifact layout (Mage-Flow and Candle SDXL at this pin) advertise their
/// component ids here; self-contained providers take the early no-op path without a manifest clone
/// or cache probe.
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
    // Mage-Flow's shared text encoder / VAE are themselves per-tier (sc-14980), so the co-requisite
    // must match the tier actually resolved for the BACKBONE — not the tier the request asked for,
    // which `standard_tier_subdir`'s completeness fallback may have stepped away from. Reading it
    // back off the resolved weights dir keeps the two in lockstep by construction: a q4 request that
    // fell back to q8 gets the q8 text encoder, never a mixed pair.
    let tier = resolved_tier_name(&spec);
    let components =
        crate::model_jobs::resolve_co_requisites_for_tier(&descriptor, &manifest_value, settings, tier)?;
    Ok(components
        .into_iter()
        .fold(spec, |spec, (id, source)| spec.with_component(id, source)))
}

/// Resolve a decoder id through the linked provider descriptor. This is the worker's backend and
/// latent-space gate: the inference registry owns provider eligibility, and its typed compatibility
/// check fails closed for z48 or unknown/learned normalization.
fn selected_decoder_option(
    engine_id: &str,
    decoder_id: &str,
) -> WorkerResult<gen_core::DecoderOption> {
    let descriptor = crate::inference_runtime::media_descriptor(engine_id).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "decoder '{decoder_id}' cannot be selected because image engine '{engine_id}' is not registered"
        ))
    })?;
    descriptor
        .compatible_decoder_options()
        .into_iter()
        .find(|option| option.id == decoder_id)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "decoder '{decoder_id}' is not compatible with image engine '{engine_id}' on backend '{}'",
                descriptor.backend
            ))
        })
}

/// Validate selection before routing or touching model weights. PiD and the alternate decoder both
/// own terminal decode, so accepting both would make the recorded recipe disagree with execution.
fn validate_selected_decoder_request(
    engine_id: &str,
    decoder_id: &str,
    advanced: &JsonObject,
) -> WorkerResult<gen_core::DecoderOption> {
    if advanced::flag(advanced, "usePid") {
        return Err(WorkerError::InvalidPayload(
            "advanced.decoder cannot be combined with advanced.usePid; select exactly one decoder"
                .to_owned(),
        ));
    }
    selected_decoder_option(engine_id, decoder_id)
}

/// Stage the selected standalone decoder file in `LoadSpec.components`. The manifest row is a soft
/// co-requisite, so native generation remains installable and runnable without it; selection itself
/// is strict and refuses a missing/stale donor before the fit gate or provider load.
fn attach_selected_decoder(
    spec: LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    let Some(decoder_id) = requested_decoder_id(&request.advanced)? else {
        return Ok(spec);
    };
    let option = validate_selected_decoder_request(engine_id, decoder_id, &request.advanced)?;
    let manifest = Value::Object(request.model_manifest_entry.clone());
    let source = crate::model_jobs::resolve_optional_component(
        &manifest,
        option.component_id,
        settings,
    )
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "decoder '{}' needs its standalone pinned component '{}' to be installed; install or repair this model before generating",
            option.label, option.component_id
        ))
    })?;
    Ok(spec.with_component(option.component_id, source))
}

/// The tier a [`LoadSpec`]'s weights dir resolved to, for matching a per-tier `coRequisite`'s
/// `variant` (sc-14980).
///
/// Delegates to [`tier_key_from_resolved_dir`] — the established "what tier is this dir" answer, so
/// the co-requisite lookup cannot drift from the tier the backbone actually loads at. `None` for a
/// flat snapshot (whose final segment is a commit sha), which keeps single-row co-requisites — every
/// audio model, and Mage on a legacy flat install — on the tier-agnostic path. A `None` here against
/// a model that DOES declare per-tier rows is refused loudly by `resolve_co_requisites_for_tier`
/// rather than guessed, so an unrecognized layout can never silently pair mismatched weights.
fn resolved_tier_name(spec: &LoadSpec) -> Option<&'static str> {
    let WeightsSource::Dir(dir) = &spec.weights else {
        return None;
    };
    tier_key_from_resolved_dir(dir)
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
            WorkerError::Engine(format!(
                "the registered candle SDXL descriptor does not advertise its required '{id}' \
                 component — the SceneWorks inference runtime pin is incompatible with this worker"
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
///
/// sc-13817 widened the gate from macOS-only so the off-Mac candle lane could force a dense tier.
/// sc-14249 retired that force (`candle-gen-sensenova` packed-detects all three tiers now), which
/// removed the only candle-side caller — so the gate is back to **macOS-only**. Its lone remaining
/// user is `image_jobs/sensenova.rs` (the MLX it2i / VQA / interleave routing), whose `include!` is
/// itself `cfg(target_os = "macos")`; leaving the wider cfg makes this dead code on the candle build
/// (`-D warnings` → `function is never used`).
///
/// The id list itself lives in [`sceneworks_core::mlx_tier_completeness::SENSENOVA_MODELS`], beside the
/// per-tier completeness predicates the tier resolver and rust-api's catalog both gate on — two
/// independently-maintained copies of a model-id list is exactly the drift that leaves one gate silently
/// un-wired (sc-14432).
#[cfg(target_os = "macos")]
fn is_sensenova_model(model: &str) -> bool {
    sceneworks_core::mlx_tier_completeness::SENSENOVA_MODELS.contains(&model)
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
    // Quality-preserving, request-scoped memory adaptations selected by the candle Krea Turbo fit
    // ladder. `None` is the historical path for every other provider and unconstrained Krea jobs.
    memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    enhance: &PromptEnhance,
    // Live denoise preview (epic 16624, sc-16904): forwarded frames reach the job's progress
    // stream. Inert for engines that don't emit; the default sink costs one branch per step.
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let conditioning = build_lane_conditioning(reference, multi_references, edit_mask);
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
        memory,
        conditioning,
        preview,
        cancel: cancel.clone(),
        ..Default::default()
    };
    enhance.apply(&mut request);
    let output = crate::memory_strategy::generate_with_scope(
        generator,
        &mut request,
        memory_strategy_context,
        on_progress,
    )
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct HiresFixPlan {
    width: u32,
    height: u32,
    steps: u32,
    guidance: Option<f32>,
    true_cfg: Option<f32>,
    provider_reference_strength: f32,
}

fn resolve_hires_fix_plan(
    request: &ImageRequest,
    first_pass_steps: u32,
    first_pass_guidance: Option<f32>,
    first_pass_true_cfg: Option<f32>,
) -> Option<HiresFixPlan> {
    if request.hires_fix.is_disabled() {
        return None;
    }
    let upscale_by = request.hires_fix.effective_upscale_by();
    let denoising_strength = request.hires_fix.effective_denoising_strength();
    let family = resolve_family(request);
    // SDXL and Ideogram expose conventional denoising strength (higher means more regeneration).
    // The other current img2img providers expose reference fidelity (higher means preserve more),
    // so translate the user-facing A1111 denoising value at this boundary.
    let provider_reference_strength = if matches!(family.as_str(), "sdxl" | "ideogram") {
        denoising_strength
    } else {
        1.0 - denoising_strength
    };
    let cfg = request
        .hires_fix
        .effective_cfg_scale(first_pass_guidance.or(first_pass_true_cfg));
    Some(HiresFixPlan {
        width: hires_fix_target_dimension(request.width, upscale_by),
        height: hires_fix_target_dimension(request.height, upscale_by),
        steps: request.hires_fix.effective_steps(first_pass_steps),
        guidance: first_pass_guidance.map(|_| cfg.unwrap_or_default()),
        true_cfg: first_pass_true_cfg.map(|_| cfg.unwrap_or_default()),
        provider_reference_strength,
    })
}

/// The conditioning one generic-lane render carries. Split out of [`generate_one`] so
/// [`lane_reference_count`] — the count the backend request scope grades the request against — can be
/// tested against the conditioning this lane REALLY sends rather than against a restatement of it.
fn build_lane_conditioning(
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    edit_mask: Option<&Image>,
) -> Vec<Conditioning> {
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
    conditioning
}

/// The image-conditioning count gen-core derives from the request [`generate_one`] builds for these
/// lane inputs — the SAME rule as `GenerationRequest::image_reference_count`: multi-reference
/// carriers contribute their image count, a single reference contributes one, and an edit mask
/// contributes one MORE (it is a `Conditioning::Mask` alongside the source `Reference`, not a
/// replacement for it).
///
/// This is not bookkeeping. The admitted geometry is re-derived from the live request by the
/// backend request scopes (`mlx-gen`'s `MlxRequestScopeCore::configure_request` and its candle twin),
/// which refuse any request whose geometry differs from the one admitted, and gen-core's shared
/// safety check rejects a `has_reference` that disagrees with `reference_count > 0`. Declaring a
/// count the request does not carry fails the render outright. The old formula
/// (`edit_refs.len().max(reference || mask)`) undercounted every reference+mask edit by one.
fn lane_reference_count(
    has_identity_init: bool,
    multi_reference_count: usize,
    has_edit_mask: bool,
) -> u32 {
    let references = if multi_reference_count > 0 {
        multi_reference_count
    } else {
        usize::from(has_identity_init)
    };
    u32::try_from(references.saturating_add(usize::from(has_edit_mask))).unwrap_or(u32::MAX)
}

/// The hires-fix refinement pass conditions on exactly one image: the upscaled first-pass render,
/// passed as the single `Conditioning::Reference` (no multi-reference, no mask). Derived through
/// [`lane_reference_count`] rather than written as a literal so it cannot drift from the rule the
/// request scope grades against.
///
/// Shared by the MLX and Candle admission declarations so both backends describe the same final-pass
/// request identity.
fn hires_fix_reference_count() -> u32 {
    lane_reference_count(true, 0, false)
}

/// The request-scope identity of the hires FIRST pass, derived from the admitted (final-pass)
/// context.
///
/// Admission describes the heaviest pass — the upscaled refinement — because that is what sets the
/// memory ceiling. The first pass renders at the BASE size with the caller's own conditioning, so
/// running it under the admitted context hands the backend request scope a geometry the request does
/// not match, and the scope refuses it (`request geometry … does not fit admitted …`) — which failed
/// the first pass of every hires render on a scope-adopting provider. Only the geometry identity
/// moves: the memory SELECTION is reused verbatim, since a strategy chosen for the larger pass is
/// the conservative choice for the smaller one.
fn hires_first_pass_context(
    admitted: &gen_core::MemoryRunContext,
    width: u32,
    height: u32,
    reference_count: u32,
) -> gen_core::MemoryRunContext {
    let mut context = admitted.clone();
    context.geometry.width = width;
    context.geometry.height = height;
    context.geometry.reference_count = reference_count;
    context.has_reference = reference_count > 0;
    context
}

/// Run the normal first pass followed by an optional high-resolution img2img refinement while
/// keeping progress monotonic across both denoise schedules.
#[allow(clippy::too_many_arguments)]
fn generate_one_with_hires(
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
    guidance_method: Option<&str>,
    use_pid: bool,
    text_style_gain: Option<f32>,
    memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    enhance: &PromptEnhance,
    hires_fix: Option<HiresFixPlan>,
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let Some(hires) = hires_fix else {
        return generate_one(
            generator,
            prompt,
            width,
            height,
            seed,
            steps,
            guidance,
            negative_prompt,
            reference,
            multi_references,
            edit_mask,
            true_cfg,
            sampler,
            scheduler,
            scheduler_shift,
            guidance_method,
            use_pid,
            text_style_gain,
            memory,
            memory_strategy_context,
            enhance,
            preview,
            cancel,
            on_progress,
        );
    };

    let first_pass_context = memory_strategy_context.map(|context| {
        hires_first_pass_context(
            context,
            width,
            height,
            lane_reference_count(reference.is_some(), multi_references.len(), edit_mask.is_some()),
        )
    });
    let combined_steps = steps.saturating_add(hires.steps);
    let mut first_progress = |progress| match progress {
        Progress::Step { current, .. } => on_progress(Progress::Step {
            current,
            total: combined_steps,
        }),
        // The first decode is an internal hand-off. The user-visible decode is the final one.
        Progress::Decoding => {}
        Progress::Loading(phase) => on_progress(Progress::Loading(phase)),
    };
    let (base_width, base_height, base_pixels) = generate_one(
        generator,
        prompt,
        width,
        height,
        seed,
        steps,
        guidance,
        negative_prompt.clone(),
        reference,
        multi_references,
        edit_mask,
        true_cfg,
        sampler,
        scheduler,
        scheduler_shift,
        guidance_method,
        use_pid,
        text_style_gain,
        memory,
        first_pass_context.as_ref(),
        enhance,
        preview.clone(),
        cancel,
        &mut first_progress,
    )?;
    if cancel.is_cancelled() {
        return Err(WorkerError::Engine("generation cancelled".to_owned()));
    }
    let high_res_reference = fit_engine_image(
        Image {
            width: base_width,
            height: base_height,
            pixels: base_pixels,
        },
        hires.width,
        hires.height,
        "stretch",
    )?;
    let reference = (high_res_reference, hires.provider_reference_strength);
    let mut second_progress = |progress| match progress {
        Progress::Step { current, .. } => on_progress(Progress::Step {
            current: steps.saturating_add(current),
            total: combined_steps,
        }),
        Progress::Decoding => on_progress(Progress::Decoding),
        Progress::Loading(phase) => on_progress(Progress::Loading(phase)),
    };
    generate_one(
        generator,
        prompt,
        hires.width,
        hires.height,
        seed,
        hires.steps,
        hires.guidance,
        negative_prompt,
        Some(&reference),
        &[],
        None,
        hires.true_cfg,
        sampler,
        scheduler,
        scheduler_shift,
        guidance_method,
        false,
        text_style_gain,
        memory,
        memory_strategy_context,
        enhance,
        preview,
        cancel,
        &mut second_progress,
    )
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

/// Mage Edit always conditions on the required primary source first, followed by the optional
/// `referenceAssetIds` exactly in client order.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mage_edit_reference_ids(request: &ImageRequest) -> Vec<String> {
    if request.mode != "edit_image" {
        return Vec::new();
    }
    let Some(source) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Vec::new();
    };
    let mut ids = Vec::with_capacity(1 + request.reference_asset_ids.len());
    ids.push(source.to_owned());
    ids.extend(request.reference_asset_ids.iter().cloned());
    ids
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn is_mage_edit_model(model: &str) -> bool {
    matches!(
        model,
        "mage_flow_edit_base" | "mage_flow_edit" | "mage_flow_edit_turbo"
    )
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

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_mage_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Vec<Image>> {
    let ids = mage_edit_reference_ids(request);
    let mut references = Vec::with_capacity(ids.len());
    for id in &ids {
        let source = load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
        references.push(fit_engine_image(
            source,
            request.width,
            request.height,
            &request.fit_mode,
        )?);
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

/// True when an Image-Edit source should be fitted to the requested output geometry before it is
/// handed to an img2img provider. This lives in the cross-backend base lane because both the MLX
/// edit routes and the Candle Z-Image alias consume it.
fn should_fit_edit_source(request: &ImageRequest) -> bool {
    let has_source = request
        .source_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    let no_reference = !request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    request.mode == "edit_image" && has_source && no_reference && request.fit_mode != "stretch"
}

/// Resolve the Z-Image Turbo provider's shared img2img init. Both backends map `z_image_edit` to
/// the registered `z_image_turbo` generator, so this resolver must live in the cross-backend lane.
fn resolve_zimage_edit_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
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
    let image = if should_fit_edit_source(request) {
        fit_engine_image(source, request.width, request.height, &request.fit_mode)?
    } else {
        source
    };
    let strength = advanced::f32_clamped(&request.advanced, "strength", 0.6, 0.05..=1.0);
    Ok(Some((image, strength)))
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
        // Multi-reference instruction editors resolve their ordered references into `edit_refs` below;
        // they use the `MultiReference`-capable path, not the single `identity_init` reference.
        Ok(LaneConditioning::default())
    }
}

/// MLX per-tier capability fit for the downtier chooser (sc-10733): fold the MLX residency decision
/// (resident-fits / staged-fits / won't-fit-even-staged) for a candidate tier's probe [`LoadSpec`]
/// down to [`TierFit`]. `Resident`/`Sequential` ⇒ `Fits`; `Reject` ⇒ `TooBig` (carrying the resident
/// need + the machine budget for the message). Uses the SAME `mlx_fit_gate` budget + footprint math
/// the cold-load `apply_residency_policy` runs, so the seam's downtier and the cache's admission
/// never disagree — which is why it takes the spec [`tier_probe_spec`] builds rather than a bare dir.
#[cfg(target_os = "macos")]
fn mlx_tier_fit(engine_id: &str, spec: &LoadSpec) -> TierFit {
    match crate::mlx_fit_gate::decide_residency_for_spec(engine_id, spec) {
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

/// The [`LoadSpec`] a candidate tier would be loaded with, for fit probing only — the tier's weights
/// dir **plus whatever caller-provisioned components that tier stages** (sc-15154). Never loaded.
///
/// A bare `Dir` spec is right for every model whose components sit under its weights dir, but
/// Mage-Flow's per-tier dir holds the DiT alone: its text encoder and VAE are bit-identical across
/// the six variants and staged from a shared mirror. Probing the bare dir therefore scored a q4 edit
/// install at 2.33 GB instead of 7.00 GB, which both under-quoted the over-budget message and let the
/// permissive weights-fit floor admit budgets the tier does not fit.
///
/// Required-component staging remains best-effort: the real load reports its actionable error. An
/// explicitly selected decoder is different: it must never disappear from the probe, because doing
/// so would under-price the request and silently evaluate the native-decoder composition instead.
#[cfg(target_os = "macos")]
fn tier_probe_spec(
    engine_id: &str,
    weights_dir: &Path,
    request: &ImageRequest,
    settings: &Settings,
    adapters: &[AdapterSpec],
) -> WorkerResult<LoadSpec> {
    let spec = LoadSpec::new(WeightsSource::Dir(weights_dir.to_path_buf()));
    let spec = attach_required_components(
        spec.clone(),
        engine_id,
        &request.model_manifest_entry,
        settings,
    )
    .unwrap_or(spec);
    Ok(attach_selected_decoder(
        spec, engine_id, request, settings,
    )?
    .with_adapters(adapters.to_vec()))
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
    // Capability downtier probes must carry the exact adapter stack the eventual load sees. Resolving
    // here lets a lower adapted tier win instead of keeping a base-only fit that the final gate rejects.
    let adapters = resolve_adapters(request, settings)?;
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
                        resolve_tier_dir(request, settings, cand).map(|dir| (cand, dir))
                    })
                    .map(|(cand, dir)| {
                        let probe =
                            tier_probe_spec(engine_id, &dir, request, settings, &adapters)?;
                        Ok((cand, mlx_tier_fit(engine_id, &probe)))
                    })
                    .collect::<WorkerResult<Vec<_>>>()?;
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
                    // Name the REJECTED TIER's own weight bytes next to the peak (sc-15154). The
                    // peak is `Σweights + HEADROOM_GB`, and on a small budget the flat headroom
                    // dominates it — a q4 install of 7 GB refused with a bare "~25 GB" reads like
                    // the figure belongs to some other tier. Recomputed from the same probe spec
                    // `mlx_tier_fit` scored, so the two numbers cannot drift apart.
                    let weights_note = if let Some(dir) = resolve_tier_dir(request, settings, tier) {
                        let probe =
                            tier_probe_spec(engine_id, &dir, request, settings, &adapters)?;
                        let gb = crate::mlx_fit_gate::spec_weights_gb(engine_id, &probe);
                        (gb > 0.0)
                            .then(|| {
                                format!(
                                    " — ~{} GB of weights plus headroom for activations and the OS",
                                    gb.round() as i64
                                )
                            })
                            .unwrap_or_default()
                    } else {
                        String::new()
                    };
                    return Err(WorkerError::InvalidPayload(format!(
                        "{model} needs ~{needed} GB of unified memory even at the smallest installed \
                         tier it can run ({tier}{weights_note}) but this machine has ~{available} GB. \
                         Lower the output resolution or run on a Mac with more memory.",
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
    // Registry instruction edits: Boogu resolves 1..5 sources; Mage resolves its required primary
    // source followed by every optional reference in client order. Both thread through `generate_one`
    // as `Reference` (one) / `MultiReference` (many), never the single img2img-init slot.
    let edit_refs: Vec<Image> = if request.model == "boogu_image_edit" {
        resolve_boogu_edit(request, settings, project_path)?
    } else if is_mage_edit_model(&request.model) {
        resolve_mage_edit(request, settings, project_path)?
    } else {
        Vec::new()
    };
    // The CFG scale passed to the engine as `true_cfg`: the FLUX.1-dev reference path's scale if
    // present, otherwise the true-CFG family scale (Chroma). `None` for the guidance-scalar and
    // distilled families, which carry CFG (if any) through `guidance` instead.
    let true_cfg = flux_true_cfg.or(model_true_cfg);
    let hires_fix = resolve_hires_fix_plan(request, steps, guidance, true_cfg);

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
    // Admission describes the heaviest pass. Hires fix renders the first pass at `width`/`height`
    // and the refinement at the plan dimensions, so every memory fit and request scope must use the
    // latter. `generate_one_with_hires` derives a base-pass scope from this final-pass identity.
    let (memory_width, memory_height) =
        hires_fix.map_or((width, height), |plan| (plan.width, plan.height));
    // Krea "text style" tap-reweight gain — see `resolve_text_style_gain` (sc-11878, gate fixed sc-12008).
    let text_style_gain = resolve_text_style_gain(request);
    let calibration_opt_in = request
        .model_manifest_entry
        .get("mlx")
        .and_then(|mlx| mlx.get("calibrations"))
        .and_then(Value::as_array)
        .is_some_and(|calibrations| !calibrations.is_empty());
    let resolved_artifact = if calibration_opt_in {
        let effective_tier = resolved_mlx_artifact_tier(&weights_dir, quant_bits);
        resolved_mlx_artifact_provenance(
            request,
            settings,
            &repo,
            &weights_dir,
            effective_tier,
        )?
    } else {
        None
    };
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
    // F3 alternate decoder: attach before both the provider-specific memory contract and the generic
    // MLX fit gate, so donor bytes + normal activation/OS margin are admitted as one composition.
    spec = attach_selected_decoder(spec, engine_id, request, settings)?;
    let plain_text_to_image = matches!(request.mode.as_str(), "image_generation" | "text_to_image")
        && identity_init.is_none()
        && edit_refs.is_empty()
        && ideogram_edit_mask.is_none()
        && hires_fix.is_none();
    spec = apply_measured_mlx_load_shape_for_request(engine_id, spec, plain_text_to_image);
    let mlx_request_plan = crate::mlx_fit_gate::MlxRequestPlan::for_spec_and_manifest(
        engine_id,
        &request.model,
        &spec,
        Some(&request.model_manifest_entry),
        resolved_artifact,
    );
    let has_request_reference =
        identity_init.is_some() || !edit_refs.is_empty() || ideogram_edit_mask.is_some();
    // The admitted geometry describes the HEAVIEST pass: with hires fix that is the final
    // upscaled img2img refinement (one `Reference`, no mask), otherwise the single base pass. The
    // first hires pass renders at the base size with the caller's own conditioning and gets its own
    // request-scope identity inside `generate_one_with_hires`.
    let reference_count = if hires_fix.is_some() {
        hires_fix_reference_count()
    } else {
        lane_reference_count(
            identity_init.is_some(),
            edit_refs.len(),
            ideogram_edit_mask.is_some(),
        )
    };
    let mut memory_overlays = Vec::new();
    if has_request_reference {
        memory_overlays.push(format!("references:{}", edit_refs.len().max(1)));
    }
    if ideogram_edit_mask.is_some() {
        memory_overlays.push("mask".to_owned());
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() {
        memory_overlays.push(format!(
            "control:{}",
            usize::from(spec.control.is_some()) + spec.extra_controls.len()
        ));
    }
    if spec.ip_adapter.is_some() {
        memory_overlays.push("ip_adapter".to_owned());
    }
    if adapter_count > 0 {
        memory_overlays.push(format!("adapters:{adapter_count}"));
    }
    if use_pid {
        memory_overlays.push("pid".to_owned());
    }
    let mlx_request_inputs = crate::mlx_fit_gate::MlxRequestInputs {
        width: memory_width,
        height: memory_height,
        count: request.count,
        mode: request.mode.clone(),
        overlay: (!memory_overlays.is_empty()).then(|| memory_overlays.join("+")),
        adapter_count,
        has_reference: reference_count > 0,
        reference_count,
        use_pid,
        has_phases: false,
    };

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

    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        move |generator,
              cache_state,
              loaded_policy,
              _requested_policy,
              external_committed_bytes,
              tx,
              cancel| {
            // Per-job identity-likeness scorer built ONCE on the generator-worker thread (the `!Send`
            // face stack lives here); source embedded once, reused across every output (sc-4411). `None`
            // ⇒ not a With-Character generation, or non-fatal staging/construction failure ⇒ omitted.
            let scorer_requested = face_stack_dir.is_some() && likeness_source.is_some();
            let scorer_active_before = if scorer_requested {
                mlx_rs::memory::clear_cache();
                mlx_rs::memory::get_active_memory() as u64
            } else {
                0
            };
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let request_external_committed_bytes = if scorer_requested {
                // The MLX face scorer is constructed after the generator cache recorded its
                // pre-load baseline. Add only its active-memory delta so scorer weights cannot be
                // credited as generation weights without double-charging the cached generator.
                // Measure attempted construction too: a failed scorer may still have allocated.
                mlx_rs::memory::clear_cache();
                let scorer_active_after = mlx_rs::memory::get_active_memory() as u64;
                crate::mlx_fit_gate::add_post_load_external_delta(
                    external_committed_bytes,
                    scorer_active_before,
                    scorer_active_after,
                )
            } else {
                external_committed_bytes
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            drive_gen_items_scored(tx, seeds, move |_index, seed, preview, on_progress| {
                let memory_evaluation = crate::mlx_fit_gate::evaluate_request(
                    generator,
                    &mlx_request_plan,
                    &mlx_request_inputs,
                    cache_state,
                    loaded_policy.offload_policy,
                    request_external_committed_bytes,
                )?;
                // Exact promoted MLX evidence may tighten the soft process limit for this request.
                // The RAII guard restores the process-global/user limit after all retries and never
                // touches the wired limit (#1947).
                let _request_memory_limit = memory_evaluation
                    .process_limit_bytes
                    .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
                let render = |seed: i64, on_progress: &mut dyn FnMut(Progress)| {
                    generate_one_with_hires(
                        generator,
                        &prompt,
                        width,
                        height,
                        seed,
                        steps,
                        guidance,
                        negative_prompt.clone(),
                        identity_init.as_ref(),
                        &edit_refs,
                        ideogram_edit_mask.as_ref(),
                        true_cfg,
                        sampler.as_deref(),
                        scheduler.as_deref(),
                        scheduler_shift,
                        guidance_method.as_deref(),
                        use_pid,
                        text_style_gain,
                        Some(memory_evaluation.memory),
                        Some(&memory_evaluation.context),
                        &enhance,
                        hires_fix,
                        preview.clone(),
                        &cancel,
                        on_progress,
                    )
                };
                // Detect-and-recover safety net (sc-6501): the caption guard makes the placeholder
                // rare, but a residual one can still occur even with a caption. Detect it via the
                // baked-text heuristic (NOT a std/flatness check — the text lifts std to ~10) and
                // reseed transparently, keeping the first clean render. Gated to Ideogram 4; a no-op
                // elsewhere (and on turbo, which is CFG-free and cannot produce the placeholder).
                let initial = render(seed, on_progress)?;
                let (final_seed, out_w, out_h, pixels) = recover_ideogram_placeholder(
                    is_ideogram,
                    seed,
                    &cancel,
                    initial,
                    |retry_seed| render(retry_seed, on_progress),
                )?;
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

/// Whether `model` is served by the Candle backend's generic built-in image lane.
///
/// The scheduler's generated catalog is the source of truth. `bernini_image` is the sole built-in
/// exception: the scheduler routes it to Candle, but [`resolve_candle_image_route`] sends it through
/// the dedicated still-image Bernini lane before reaching this gate. Dynamic `external_base_*` ids
/// are likewise claimed by manifest-driven bespoke routes and never appear in the built-in catalog.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn is_candle_engine(model: &str) -> bool {
    model != "bernini_image"
        && sceneworks_core::jobs_store::candle_routed_image_models().contains(&model)
}

/// The per-asset `adapter` id recorded for a candle image engine (`candle_<family>`), the candle
/// sibling of the `MODEL_TABLE` `mlx_<family>` labels. Used by both the generic per-asset stream and
/// [`CandleImageRoute::adapter_label`] so the sidecar and generation-set result agree.
/// (sc-5099 extends this same labeling to the video + caption engines.)
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_adapter_label(model: &str) -> &'static str {
    match model {
        // Base z_image (sc-8679) shares the candle z-image family label with Turbo.
        "z_image_turbo" | "z_image" | "z_image_edit" => "candle_z_image",
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
        "mage_flow_base"
        | "mage_flow"
        | "mage_flow_turbo"
        | "mage_flow_edit_base"
        | "mage_flow_edit"
        | "mage_flow_edit_turbo" => "candle_mage",
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

/// Build the candle reject advice for `rejected_tier` from the tiers the loader can actually resolve.
/// Kept pure so bespoke candle lanes can share the same truthfulness contract as the main image lane.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn vram_reject_tail_for_tier(installed: Vec<&'static str>, rejected_tier: &str) -> String {
    let smaller: Vec<&'static str> = installed
        .into_iter()
        .filter(|candidate| {
            tier_quality_rank(candidate) < tier_quality_rank(rejected_tier)
        })
        .collect();
    vram_reject_tail(&smaller)
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
    adapter_bytes: u64,
) -> TierFit {
    let needed = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        manifest_entry,
        tier,
        adapter_bytes,
    );
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
            let seq_needed = crate::vram_gate::predicted_sequential_peak_gb_with_adapter_bytes(
                manifest_entry,
                tier,
                adapter_bytes,
            );
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

/// Carry the legacy Candle fit gate's resident/staged choice into the request-scoped runtime
/// contract. The load-time policy remains populated for compatibility with older and non-image
/// providers, but current image providers use this bit as the sole lifecycle authority.
#[cfg(any(
    test,
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn apply_request_scoped_candle_residency(
    use_sequential: bool,
    memory: &mut Option<gen_core::GenerationMemory>,
) {
    if use_sequential {
        memory.get_or_insert_with(Default::default).stage_residency = true;
    }
}

#[cfg(any(
    test,
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn rejects_unverified_shared_memory_fallback(
    engine_id: &str,
    optimized_strategy_selected: bool,
) -> bool {
    matches!(
        engine_id,
        "z_image" | "z_image_turbo" | "flux2_dev" | "flux2_klein_9b"
    )
        && !optimized_strategy_selected
}

#[cfg(any(
    test,
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn verified_only_memory_family_label(engine_id: &str) -> Option<&'static str> {
    match engine_id {
        "z_image" | "z_image_turbo" => Some("Z-Image"),
        "flux2_dev" => Some("FLUX.2-dev"),
        "flux2_klein_9b" => Some("FLUX.2 Klein"),
        _ => None,
    }
}

/// Caller-visible explanation for the one Krea memory decision whose latency would otherwise look
/// like a hung render (sc-16104). This is deliberately presentation-small: the worker folds the note
/// into its ordinary progress message rather than creating a dialog, toast, or separate UI surface.
#[derive(Clone, Debug, PartialEq)]
struct AutoTierStreamingDisclosure {
    tier: String,
    measured_materialization_ms_per_block: Option<f64>,
    measured_materialization_ms_per_step: Option<u64>,
}

const AUTO_TIER_STREAMING_DISCLOSURE_KEY: &str = "memoryStrategyDisclosure";

fn scrub_untrusted_memory_strategy_disclosure(raw_settings: &mut JsonObject) {
    raw_settings.remove(AUTO_TIER_STREAMING_DISCLOSURE_KEY);
}

impl AutoTierStreamingDisclosure {
    fn progress_message(&self, progress: &str) -> String {
        format!(
            "{progress} Streaming transformer blocks to hold the auto-selected {} tier.",
            self.tier
        )
    }

    #[cfg(any(
        test,
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn record(&self) -> Value {
        json!({
            "strategy": "bounded_transformer_residency",
            "cause": "streaming_transformer_blocks_to_hold_auto_selected_tier",
            "tier": self.tier,
            "measuredMaterializationMsPerBlock": self.measured_materialization_ms_per_block,
            "measuredMaterializationMsPerStep": self.measured_materialization_ms_per_step,
            "measurementSource": "sc-16096-rtx-pro-6000-blackwell",
        })
    }

    #[cfg(any(
        test,
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn telemetry(&self, job_id: &str, model: &str) -> Value {
        let mut telemetry = self.record();
        let fields = telemetry
            .as_object_mut()
            .expect("memory-strategy disclosure record is an object");
        fields.insert("jobId".to_owned(), Value::String(job_id.to_owned()));
        fields.insert("model".to_owned(), Value::String(model.to_owned()));
        telemetry
    }
}

fn progress_with_auto_tier_streaming(
    progress: &str,
    disclosure: Option<&AutoTierStreamingDisclosure>,
) -> String {
    disclosure.map_or_else(
        || progress.to_owned(),
        |disclosure| disclosure.progress_message(progress),
    )
}

#[cfg(any(
    test,
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoTierFitDecision {
    StreamedBlocks,
    Other,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn auto_tier_fit_decision(
    final_fit: Option<&crate::vram_gate::KreaTurboFit>,
) -> AutoTierFitDecision {
    if matches!(
        final_fit,
        Some(crate::vram_gate::KreaTurboFit::Fits {
            selection: gen_core::MemorySelection {
                strategy: gen_core::MemoryStrategy::BoundedTransformerResidency,
                ..
            },
            ..
        })
    ) {
        AutoTierFitDecision::StreamedBlocks
    } else {
        AutoTierFitDecision::Other
    }
}

/// SC-16096 measured Krea's real packed q4/q8 window materialization after the device-format sidecar
/// fix. Krea Turbo is a fixed 30-block, CFG-free single-forward denoiser, so block median x 30 is the
/// measured materialization component paid by each rung-4 denoise step. Dense bf16 was not measured;
/// disclose the strategy there without inventing a number.
#[cfg(any(
    test,
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn krea_streaming_measurement(tier: &str) -> (Option<f64>, Option<u64>) {
    let block_ms: Option<f64> = match tier {
        "q4" => Some(56.2),
        "q8" => Some(101.7),
        _ => None,
    };
    let step_ms = block_ms.map(|milliseconds| (milliseconds * 30.0).round() as u64);
    (block_ms, step_ms)
}

/// A disclosure exists only for Auto plus the final rung-4 selection. Explicit picks keep bypassing
/// the capability downtier and do not claim that Auto retained their tier; cheaper rungs do not claim
/// transformer streaming. These two guards are mutation-pinned below.
#[cfg(any(
    test,
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn auto_tier_streaming_disclosure(
    explicit_pick: bool,
    tier: &str,
    final_fit: AutoTierFitDecision,
) -> Option<AutoTierStreamingDisclosure> {
    if explicit_pick || final_fit != AutoTierFitDecision::StreamedBlocks {
        return None;
    }
    let (measured_materialization_ms_per_block, measured_materialization_ms_per_step) =
        krea_streaming_measurement(tier);
    Some(AutoTierStreamingDisclosure {
        tier: tier.to_owned(),
        measured_materialization_ms_per_block,
        measured_materialization_ms_per_step,
    })
}

#[cfg(test)]
mod candle_request_residency_tests {
    use super::*;

    #[test]
    fn sequential_admission_sets_the_request_bit_without_erasing_other_rungs() {
        let mut resident = None;
        apply_request_scoped_candle_residency(false, &mut resident);
        assert!(resident.is_none(), "resident admission leaves request memory absent");

        let mut sequential = None;
        apply_request_scoped_candle_residency(true, &mut sequential);
        assert!(
            sequential.is_some_and(|memory| memory.stage_residency),
            "mutation guard: sequential admission must reach the request lifecycle bit"
        );

        let mut composed = Some(gen_core::GenerationMemory {
            tile_vae_decode: true,
            ..Default::default()
        });
        apply_request_scoped_candle_residency(true, &mut composed);
        let composed = composed.expect("existing rung memory is preserved");
        assert!(composed.stage_residency);
        assert!(composed.tile_vae_decode);
    }

    #[test]
    fn verified_only_families_never_fall_back_to_an_unverified_optimized_strategy() {
        assert!(rejects_unverified_shared_memory_fallback(
            "z_image", false
        ));
        assert!(rejects_unverified_shared_memory_fallback(
            "flux2_dev",
            false
        ));
        assert!(rejects_unverified_shared_memory_fallback(
            "flux2_klein_9b",
            false
        ));
        assert!(!rejects_unverified_shared_memory_fallback(
            "z_image", true
        ));
        assert!(!rejects_unverified_shared_memory_fallback(
            "flux2_dev",
            true
        ));
        assert!(!rejects_unverified_shared_memory_fallback(
            "flux2_klein_9b",
            true
        ));
        assert!(!rejects_unverified_shared_memory_fallback(
            "qwen_image",
            false
        ));
        assert_eq!(
            verified_only_memory_family_label("flux2_dev"),
            Some("FLUX.2-dev")
        );
        assert_eq!(
            verified_only_memory_family_label("flux2_klein_9b"),
            Some("FLUX.2 Klein")
        );
    }

    #[test]
    fn flux2_base_mode_is_canonicalized_without_changing_other_families() {
        assert_eq!(
            candle_base_memory_request_mode("flux2_dev", "style_variations"),
            "text_to_image"
        );
        assert_eq!(
            candle_base_memory_request_mode("flux2_klein_9b", "image_generation"),
            "text_to_image"
        );
        for mode in [
            "edit_image",
            "reference",
            "image_to_image",
            "character_image",
            "style_variations",
        ] {
            assert_eq!(candle_base_memory_request_mode("flux2_klein_9b", mode), mode);
            assert!(flux2_klein_reference_bearing_mode(mode));
        }
        for engine in ["z_image", "qwen_image", "flux1_dev"] {
            assert_eq!(
                candle_base_memory_request_mode(engine, "style_variations"),
                "style_variations"
            );
        }
    }

    #[test]
    fn flux_without_exact_evidence_preserves_sequential_fallback_and_never_names_zimage() {
        for engine in ["flux1_schnell", "flux1_dev"] {
            assert!(!rejects_unverified_shared_memory_fallback(engine, false));
            assert_eq!(verified_only_memory_family_label(engine), None);
        }
        assert_eq!(verified_only_memory_family_label("z_image"), Some("Z-Image"));
    }

    #[test]
    fn auto_rung_four_disclosure_names_cause_and_real_krea_cost() {
        let disclosure =
            auto_tier_streaming_disclosure(false, "q8", AutoTierFitDecision::StreamedBlocks)
            .expect("Auto plus rung 4 must disclose streaming");
        assert_eq!(disclosure.measured_materialization_ms_per_block, Some(101.7));
        assert_eq!(disclosure.measured_materialization_ms_per_step, Some(3_051));
        let progress = disclosure.progress_message("Image 1/1 — step 1/8.");
        assert!(progress.contains("Streaming transformer blocks"));
        assert!(progress.contains("hold the auto-selected q8 tier"));
        let trace = disclosure.telemetry("job-1", "krea_2_turbo");
        assert_eq!(trace["strategy"], "bounded_transformer_residency");
        assert_eq!(
            trace["cause"],
            "streaming_transformer_blocks_to_hold_auto_selected_tier"
        );
        assert_eq!(trace["measuredMaterializationMsPerStep"], 3_051);
        let mut raw_settings = JsonObject::new();
        raw_settings.insert(
            AUTO_TIER_STREAMING_DISCLOSURE_KEY.to_owned(),
            disclosure.record(),
        );
        assert_eq!(raw_settings["memoryStrategyDisclosure"], disclosure.record());

        // Mutation guards: removing either predicate would mislabel an explicit choice or a cheaper
        // rung as Auto's block-streaming decision.
        assert!(auto_tier_streaming_disclosure(
            true,
            "q8",
            AutoTierFitDecision::StreamedBlocks
        )
        .is_none());
        assert!(auto_tier_streaming_disclosure(false, "q8", AutoTierFitDecision::Other).is_none());
    }

    #[test]
    fn client_raw_settings_cannot_forge_auto_rung_four_disclosure() {
        let mut advanced = JsonObject::new();
        advanced.insert(
            AUTO_TIER_STREAMING_DISCLOSURE_KEY.to_owned(),
            json!({
                "tier": "q8",
                "cause": "streaming_transformer_blocks_to_hold_auto_selected_tier",
            }),
        );

        scrub_untrusted_memory_strategy_disclosure(&mut advanced);
        assert!(!advanced.contains_key(AUTO_TIER_STREAMING_DISCLOSURE_KEY));
        assert_eq!(
            progress_with_auto_tier_streaming("Image 1/1 — step 1/8.", None),
            "Image 1/1 — step 1/8."
        );
    }

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn final_krea_fit_enum_is_the_trusted_disclosure_source() {
        let fit = |strategy| crate::vram_gate::KreaTurboFit::Fits {
            phases: crate::vram_gate::KreaTurboPhasePeaks {
                text_gb: 1.0,
                denoise_gb: 1.0,
                decode_gb: 1.0,
            },
            needed_gb: 1.0,
            selection: gen_core::MemorySelection {
                strategy,
                parameters: Default::default(),
                tier: gen_core::MemoryNumericTier {
                    precision: gen_core::Precision::Bf16,
                    quant: Some(gen_core::Quant::Q8),
                    component_precision_floors: &[],
                },
            },
            memory: gen_core::GenerationMemory::default(),
            estimate_scoped: false,
        };
        let streamed = fit(gen_core::MemoryStrategy::BoundedTransformerResidency);
        let cheaper = fit(gen_core::MemoryStrategy::BoundedAttention);

        assert_eq!(
            auto_tier_fit_decision(Some(&streamed)),
            AutoTierFitDecision::StreamedBlocks
        );
        assert_eq!(
            auto_tier_fit_decision(Some(&cheaper)),
            AutoTierFitDecision::Other
        );
        assert_eq!(auto_tier_fit_decision(None), AutoTierFitDecision::Other);
    }

    #[test]
    fn streaming_measurements_are_krea_specific_and_do_not_invent_bf16_cost() {
        assert_eq!(krea_streaming_measurement("q4"), (Some(56.2), Some(1_686)));
        assert_eq!(krea_streaming_measurement("q8"), (Some(101.7), Some(3_051)));
        assert_eq!(krea_streaming_measurement("bf16"), (None, None));
    }
}

/// Whether a candle job may use Krea Turbo's request-scoped, quality-preserving memory ladder.
/// Keep every exclusion explicit: these surfaces have distinct component graphs or denoise contracts.
#[cfg(any(
    test,
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
fn krea_turbo_memory_route(
    engine_id: &str,
    has_convrot: bool,
    has_edit_reference: bool,
    has_img2img_reference: bool,
    has_edit_references: bool,
    has_edit_mask: bool,
    has_hires_fix: bool,
    use_pid: bool,
) -> bool {
    engine_id == "krea_2_turbo"
        && !has_convrot
        && !has_edit_reference
        && !has_img2img_reference
        && !has_edit_references
        && !has_edit_mask
        && !has_hires_fix
        && !use_pid
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn krea_runtime_evidence_context(
    request: &ImageRequest,
    settings: &Settings,
    tier: &str,
    resolved_artifact_root: &Path,
) -> Option<crate::vram_gate::KreaRuntimeEvidenceContext> {
    let download = request
        .model_manifest_entry
        .get("downloads")?
        .as_array()?
        .iter()
        .find(|download| download.get("variant").and_then(Value::as_str) == Some(tier))?;
    let provider = download.get("provider")?.as_str()?;
    let repository = download.get("repo")?.as_str()?;
    let revision = download.get("revision")?.as_str()?;
    let pinned_snapshot_root = crate::model_jobs::huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        repository,
        revision,
    )?;
    crate::vram_gate::KreaRuntimeEvidenceContext::inspect(
        "krea_2_turbo",
        "candle",
        &settings.gpu_id,
        crate::gpu::cached_compute_cap(),
        provider,
        repository,
        revision,
        tier,
        resolved_artifact_root,
        &pinned_snapshot_root,
    )
}

/// The constrained Krea ladder preserves the precision selected for the request. Its memory rungs
/// change residency only; they must never make the generic capability clamp silently cross from Q8
/// to Q4 before the ladder can return its measured reject/lower-resolution result.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn capability_downtier_floor<'a>(
    krea_turbo_ladder: bool,
    resolved_tier: &'a str,
    manifest_floor: Option<&'a str>,
) -> Option<&'a str> {
    if krea_turbo_ladder {
        Some(resolved_tier)
    } else {
        manifest_floor
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn log_krea_turbo_evidence_exclusion(
    model: &str,
    tier: &str,
    width: u32,
    height: u32,
    consumer: &str,
    reason: Option<gen_core::MemoryEvidenceVerdict>,
) {
    tracing::warn!(
        model,
        tier,
        width,
        height,
        consumer,
        reason = ?reason.unwrap_or(gen_core::MemoryEvidenceVerdict::Missing),
        "shared Krea memory-strategy evidence excluded; optimized execution is disabled"
    );
}

/// Unverified Krea evidence may retain only the baseline resident behavior. In particular, a legacy
/// sequential estimate fitting does not authorize an optimized load after the shared contract failed.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn krea_unverified_resident_decision(
    needed_gb: Option<f64>,
    budget: Option<crate::vram_gate::VramBudget>,
) -> crate::vram_gate::FitDecision {
    crate::vram_gate::fit_decision(needed_gb, budget)
}

/// Truthful Krea Turbo capability-clamp rejection. The ladder's same-tier floor prevents a silent
/// quality downgrade, but lower installed tiers remain a valid MANUAL escape and must be named when
/// lowering resolution cannot cross the measured fixed-weight floor.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn krea_turbo_capability_reject_message(
    model: &str,
    tier: &str,
    needed_gb: f64,
    available_gb: f64,
    gpu_id: &str,
    lower_resolution: Option<(u32, u32)>,
    installed_smaller: &[&str],
) -> String {
    let mut options = Vec::new();
    if let Some((smaller_w, smaller_h)) = lower_resolution {
        options.push(format!(
            "lower the output resolution to {smaller_w}x{smaller_h} or below"
        ));
    }
    if !installed_smaller.is_empty() {
        options.push(format!(
            "select a smaller installed tier ({})",
            installed_smaller
                .iter()
                .map(|candidate| candidate.to_uppercase())
                .collect::<Vec<_>>()
                .join(" / ")
        ));
    }
    options.push("run on a GPU with more VRAM".to_owned());
    format!(
        "{model} at its {tier} precision floor needs {needed_gb:.2} GiB of VRAM even after every \
         measured constrained-memory rung, but GPU {gpu_id} has {available_gb:.2} GiB available. \
         {options}.",
        options = options.join(", or "),
    )
}

#[cfg(all(
    test,
    not(target_os = "macos"),
    feature = "backend-candle"
))]
mod krea_turbo_memory_route_tests {
    use super::{
        candle_adapter_resident_bytes, capability_downtier_floor,
        krea_streamed_blocks_adapter_evidence, krea_turbo_capability_reject_message,
        krea_turbo_memory_route, krea_unverified_resident_decision,
    };
    use serde_json::{Map, Value};

    fn historical_builtin_krea_turbo_manifest() -> Map<String, Value> {
        let jsonc = include_str!("../../../../config/manifests/builtin.models.jsonc");
        let mut parsed: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(jsonc))
                .expect("builtin model manifest parses");
        let model = parsed["models"]
            .as_array_mut()
            .expect("models array")
            .iter_mut()
            .find(|model| model["id"].as_str() == Some("krea_2_turbo"))
            .expect("Krea 2 Turbo manifest entry");
        model["candle"]["turboFit"]["calibrationFingerprint"] =
            Value::String("krea-turbo-cuda-phase-curves-v1".into());
        // sc-17774: these tests make the ladder historical through the FINGERPRINT, which is their
        // subject. Stamp the live closure so currency is not a second, unintended historical axis —
        // the shipped digest is behind the pin (`gen-core` moved), and letting that decide too would
        // stop these tests exercising the fingerprint path they are named for.
        model["candle"]["turboFit"]["inferenceClosureDigest"] = Value::String(
            sceneworks_core::memory_calibration::packaged_closure_digest("candle", "krea_2_turbo")
                .unwrap_or_default(),
        );
        // sc-17097 removed the ABI re-stamp that used to sit here. Overwriting the shipped
        // `calibrationAbi` with the current constant is what kept every selector test green while
        // production rejected the same manifest: no test in the tree ever read the shipped value.
        model
            .as_object()
            .expect("Krea 2 Turbo manifest object")
            .clone()
    }

    #[test]
    fn only_plain_turbo_text_to_image_uses_the_memory_ladder() {
        assert!(krea_turbo_memory_route(
            "krea_2_turbo",
            false,
            false,
            false,
            false,
            false,
            false,
            false
        ));
        for excluded_surface in 0..7 {
            let mut flags = [false; 7];
            flags[excluded_surface] = true;
            assert!(
                !krea_turbo_memory_route(
                    "krea_2_turbo",
                    flags[0],
                    flags[1],
                    flags[2],
                    flags[3],
                    flags[4],
                    flags[5],
                    flags[6],
                ),
                "surface flag {excluded_surface} must retain its established route"
            );
        }
        for engine in ["krea_2_raw", "krea_2_turbo_edit", "krea_2_turbo_control"] {
            assert!(!krea_turbo_memory_route(
                engine, false, false, false, false, false, false, false
            ));
        }
    }

    #[test]
    fn adapter_bytes_disable_krea_streaming_and_missing_bytes_fail_closed() {
        let root_guard = tempfile::Builder::new()
            .prefix("krea-adapter-evidence-")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        let adapter = root.join("adapter.safetensors");
        std::fs::write(&adapter, vec![0_u8; 321]).unwrap();
        let measured = vec![gen_core::AdapterSpec::new(
            adapter,
            1.0,
            gen_core::AdapterKind::Lora,
        )];
        assert_eq!(krea_streamed_blocks_adapter_evidence(&[]), (true, Some(0)));
        assert_eq!(
            krea_streamed_blocks_adapter_evidence(&measured),
            (false, Some(321))
        );
        assert_eq!(candle_adapter_resident_bytes("krea_2_turbo", "bf16", 321), 321);
        assert_eq!(candle_adapter_resident_bytes("z_image", "bf16", 321), 321);
        assert_eq!(
            candle_adapter_resident_bytes("z_image_control", "bf16", 321),
            321
        );
        assert_eq!(candle_adapter_resident_bytes("sdxl", "bf16", 321), 0);
        assert_eq!(candle_adapter_resident_bytes("sdxl", "q4", 321), 321);

        let missing = vec![gen_core::AdapterSpec::new(
            root.join("missing.safetensors"),
            1.0,
            gen_core::AdapterKind::Lora,
        )];
        assert_eq!(
            krea_streamed_blocks_adapter_evidence(&missing),
            (false, None),
            "missing byte evidence must retain the count-based fail-closed fallback"
        );
    }

    #[test]
    fn adapter_bytes_flip_the_candle_resident_fit_boundary() {
        use crate::vram_gate::{VramBudget, HEADROOM_GB};

        let manifest = serde_json::json!({
            "candle": { "vramGbByTier": { "q4": 7.0 } }
        })
        .as_object()
        .cloned()
        .unwrap();
        let budget = Some(VramBudget {
            free_gb: 7.5 + HEADROOM_GB,
            total_gb: 16.0,
        });
        assert_eq!(
            super::candle_tier_fit(&manifest, "q4", budget, false, 0),
            super::TierFit::Fits
        );
        assert!(matches!(
            super::candle_tier_fit(&manifest, "q4", budget, false, 1024 * 1024 * 1024),
            super::TierFit::TooBig { .. }
        ));
    }

    #[test]
    fn unverified_krea_never_uses_legacy_sequential_even_when_it_would_fit() {
        use crate::vram_gate::{FitDecision, VramBudget};

        let budget = Some(VramBudget {
            free_gb: 24.0,
            total_gb: 24.0,
        });
        assert!(matches!(
            krea_unverified_resident_decision(Some(27.7), budget),
            FitDecision::TooBig {
                needed_gb: 27.7,
                available_gb: 24.0,
            }
        ));
        assert!(
            crate::vram_gate::sequential_overflow_gb(Some(23.9), budget).is_none(),
            "the regression boundary intentionally leaves the legacy sequential estimate fitting"
        );
    }

    #[test]
    fn historical_constrained_q8_reject_never_capability_downtiers_to_q4() {
        use super::{choose_downtier, tier_quality_rank, DowntierPick, TierFit};
        use crate::vram_gate::{krea_turbo_fit_with_runtime as krea_turbo_fit, KreaTurboFit, VramBudget};

        let manifest = historical_builtin_krea_turbo_manifest();
        // sc-17097 re-measurement moved both streamed floors (q4 6.32 GiB, q8 7.01 GiB including the
        // 2 GiB reserve). This budget still sits in the only window that makes the test meaningful:
        // Q8 cannot fit at any rung while Q4 can, so a downtier is available and must NOT be taken.
        let available_gb = 7.00;
        let budget = Some(VramBudget {
            free_gb: available_gb,
            total_gb: available_gb,
        });
        let fit = |tier| {
            let runtime = crate::vram_gate::KreaRuntimeEvidenceContext::verified_for_test(tier);
            match krea_turbo_fit(&manifest, tier, 1024, 1024, budget, true, Some(&runtime))
                .expect("Q8 and Q4 have measured ladder curves")
            {
            KreaTurboFit::Resident { .. } | KreaTurboFit::Fits { .. } => TierFit::Fits,
            KreaTurboFit::Reject { needed_gb, .. } => TierFit::TooBig {
                needed_gb,
                available_gb,
            },
            KreaTurboFit::Unverified { .. } => TierFit::TooBig {
                needed_gb: f64::INFINITY,
                available_gb,
            },
            }
        };
        assert!(matches!(fit("q8"), TierFit::TooBig { .. }));
        assert_eq!(fit("q4"), TierFit::Fits);

        let floor = capability_downtier_floor(true, "q8", None).expect("Q8 precision floor");
        let candidates: Vec<_> = ["q8", "q4"]
            .into_iter()
            .filter(|tier| tier_quality_rank(tier) >= tier_quality_rank(floor))
            .map(|tier| (tier, fit(tier)))
            .collect();
        assert!(matches!(
            choose_downtier("q8", &candidates),
            DowntierPick::Reject { tier: "q8", .. }
        ));
    }

    #[test]
    fn historical_constrained_bf16_reject_never_capability_downtiers_to_q8_or_q4() {
        use super::{choose_downtier, tier_quality_rank, DowntierPick, TierFit};
        use crate::vram_gate::{krea_turbo_fit_with_runtime as krea_turbo_fit, KreaTurboFit, VramBudget};

        let manifest = historical_builtin_krea_turbo_manifest();
        // sc-17097: BF16's streamed floor is 10.42 GiB after the re-measurement, so this budget keeps
        // BF16 unfittable while BOTH lower tiers fit - the case where downtiering is most tempting.
        let available_gb = 10.41;
        let budget = Some(VramBudget {
            free_gb: available_gb,
            total_gb: available_gb,
        });
        let fit = |tier| {
            let runtime = crate::vram_gate::KreaRuntimeEvidenceContext::verified_for_test(tier);
            match krea_turbo_fit(&manifest, tier, 1024, 1024, budget, true, Some(&runtime))
                .expect("BF16, Q8, and Q4 have measured ladder curves")
            {
            KreaTurboFit::Resident { .. } | KreaTurboFit::Fits { .. } => TierFit::Fits,
            KreaTurboFit::Reject { needed_gb, .. } => TierFit::TooBig {
                needed_gb,
                available_gb,
            },
            KreaTurboFit::Unverified { .. } => TierFit::TooBig {
                needed_gb: f64::INFINITY,
                available_gb,
            },
            }
        };
        assert!(matches!(fit("bf16"), TierFit::TooBig { .. }));
        assert_eq!(fit("q8"), TierFit::Fits);
        assert_eq!(fit("q4"), TierFit::Fits);

        let floor = capability_downtier_floor(true, "bf16", None).expect("BF16 precision floor");
        let candidates: Vec<_> = ["bf16", "q8", "q4"]
            .into_iter()
            .filter(|tier| tier_quality_rank(tier) >= tier_quality_rank(floor))
            .map(|tier| (tier, fit(tier)))
            .collect();
        assert!(matches!(
            choose_downtier("bf16", &candidates),
            DowntierPick::Reject { tier: "bf16", .. }
        ));
    }

    #[test]
    fn constrained_bf16_capability_reject_offers_only_truthful_manual_escapes() {
        let immediate_below = krea_turbo_capability_reject_message(
            "Krea 2 Turbo",
            "bf16",
            10.64,
            10.63,
            "0",
            None,
            &["q8", "q4"],
        );
        assert!(
            immediate_below.contains("bf16 precision floor"),
            "{immediate_below}"
        );
        assert!(
            immediate_below.contains("smaller installed tier (Q8 / Q4)"),
            "{immediate_below}"
        );
        assert!(
            immediate_below.contains("needs 10.64 GiB")
                && immediate_below.contains("has 10.63 GiB available"),
            "immediate-below copy must preserve the measured distinction: {immediate_below}"
        );
        assert!(
            !immediate_below.contains("lower the output resolution"),
            "the fixed BF16 text floor makes resolution advice false: {immediate_below}"
        );
        assert!(
            !immediate_below.contains("smallest installed tier"),
            "{immediate_below}"
        );

        let geometry_limited = krea_turbo_capability_reject_message(
            "Krea 2 Turbo",
            "bf16",
            12.0,
            11.0,
            "0",
            Some((768, 768)),
            &[],
        );
        assert!(
            geometry_limited.contains("lower the output resolution to 768x768 or below"),
            "{geometry_limited}"
        );
        assert!(
            !geometry_limited.contains("select a smaller installed tier"),
            "{geometry_limited}"
        );
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
/// adapter-free exactly as before. No reference/img2img/control — unsupported shapes are refused
/// upstream and remain queued (`image_request_candle_eligible`). Reached only when `backend_candle_enabled`
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
    // Every Klein reference-bearing request belongs to the bespoke Flux2Edit route, which resolves
    // and counts the actual reference set. Reaching the generic registered generator means routing
    // could not prove a usable reference/base; fail closed instead of dropping the reference and
    // silently authorizing the request as text-to-image.
    if engine_id == "flux2_klein_9b" && flux2_klein_reference_bearing_mode(&request.mode) {
        return Err(WorkerError::InvalidPayload(format!(
            "{} {} requires the FLUX.2 reference route and at least one resolvable reference image",
            request.model, request.mode
        )));
    }
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
    let hires_fix = resolve_hires_fix_plan(request, steps, guidance, true_cfg);
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
    } else if matches!(request.model.as_str(), "z_image_turbo" | "z_image_edit") {
        // `z_image_edit` is a catalog alias for the registered Turbo provider. Resolve its source
        // into the generic request so memory admission, lifecycle cleanup, and telemetry stay shared.
        (resolve_zimage_edit_init(request, settings, project_path)?, None)
    } else {
        (None, None)
    };
    // Registry instruction edits: resolve Boogu's 1..5 sources or Mage's source-first ordered list.
    // Each uses the `MultiReference`-capable path, not the single `edit_reference` img2img slot.
    let edit_refs: Vec<Image> = if request.model == "boogu_image_edit" {
        resolve_boogu_edit(request, settings, project_path)?
    } else if is_mage_edit_model(&request.model) {
        resolve_mage_edit(request, settings, project_path)?
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
    // `edit_reference` (edit_image vs text_to_image) and the registry editors' `edit_refs` (guarded here
    // so a future overlap never double-drives the single `reference` slot).
    let img2img_reference = if edit_reference.is_none()
        && edit_refs.is_empty()
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
    // Admission describes the heaviest pass. Hires fix renders the first pass at `width`/`height`
    // and the refinement at the plan dimensions, so every memory fit and request scope must use the
    // latter. `generate_one_with_hires` derives a base-pass scope from this final-pass identity.
    let (memory_width, memory_height) =
        hires_fix.map_or((width, height), |plan| (plan.width, plan.height));
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
    let raw_budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    // sc-11023: the single-slot generator cache evicts its current occupant BEFORE the incoming load.
    // This gate reads `free` while that occupant is still resident (so raw `free` still counts the model
    // it is about to replace); crediting the reclaimable pool predicts the `free` the imminent evict will
    // PRODUCE. That prediction is right whether evicting returns the VRAM to the driver (GPU-measured
    // sc-13960: a full generator drop frees most of it back, `nvidia-smi` free RISES) or a within-device
    // component free keeps it pooled in-process. So budget against `free + reclaimable` (capped to total),
    // else a warm/swap re-gate falsely rejects a load that will actually fit (the "even with sequential
    // residency" 2nd-run reject a resident bf16 tier hits on the next generation).
    let reclaimable_gb = crate::vram_gate::reclaimable_pool_gb(&settings.gpu_id);
    let budget =
        raw_budget.map(|budget| crate::vram_gate::with_reclaimable(budget, reclaimable_gb));
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
    let requested_tier = tier;
    // sc-12130: derive Candle residency support from the provider's weights-free descriptor instead of
    // maintaining a second engine-id allowlist in the worker. The capability bit is the provider's
    // contract that every request shape accepted by this id honors Sequential. Bespoke edit/control,
    // ComfyUI, and strict-control routes are diverted by `resolve_candle_image_route` before this gate;
    // generic txt2img/img2img reaches this point and uses the same registry-derived signal for both the
    // capability downtier and the resident/sequential decision.
    let sequential_capable = crate::mlx_fit_gate::engine_supports_sequential(engine_id);
    // SC-15117: the deeper, request-scoped Krea Turbo ladder is intentionally limited to the stock
    // ordinary single-pass txt2img route implemented by candle-gen-krea. Reference/edit/hires/PiD/
    // ConvRot surfaces keep their established paths. Hires is deliberately excluded: its refinement
    // is img2img, which the Krea request scope does not implement. Adapter jobs have no calibrated
    // evidence cells and therefore fail closed to resident-or-reject; evidence from ordinary
    // text-to-image never transfers to them.
    let krea_turbo_ladder = krea_turbo_memory_route(
        engine_id,
        convrot.is_some(),
        edit_reference.is_some(),
        img2img_reference.is_some(),
        !edit_refs.is_empty(),
        edit_mask.is_some(),
        hires_fix.is_some(),
        use_pid,
    );
    let (krea_allow_streamed_blocks, krea_adapter_bytes) =
        krea_streamed_blocks_adapter_evidence(&adapters);
    if adapter_count > 0 && krea_adapter_bytes.is_none() {
        return Err(WorkerError::InvalidPayload(format!(
            "{} includes {adapter_count} adapter(s), but their resident bytes are unavailable; \
             refusing an unbounded Candle request",
            request.model
        )));
    }
    let adapter_source_bytes = krea_adapter_bytes.unwrap_or(0);
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
        let floor = capability_downtier_floor(
            krea_turbo_ladder,
            tier,
            min_quality_floor(request),
        );
        let candidates: Vec<(&'static str, TierFit)> =
            downtier_candidate_tiers(request, settings, tier, floor)
                .into_iter()
                .map(|candidate| {
                    let candidate_adapter_bytes = candle_adapter_resident_bytes(
                        engine_id,
                        candidate,
                        adapter_source_bytes,
                    );
                    let fit = if krea_turbo_ladder && adapter_count == 0 {
                        let candidate_runtime = resolve_tier_dir(request, settings, candidate)
                            .and_then(|dir| {
                                krea_runtime_evidence_context(
                                    request, settings, candidate, &dir,
                                )
                            });
                        match crate::vram_gate::krea_turbo_fit_with_runtime(
                            &request.model_manifest_entry,
                            candidate,
                            memory_width,
                            memory_height,
                            budget,
                            krea_allow_streamed_blocks,
                            candidate_runtime.as_ref(),
                        ) {
                            Some(
                                crate::vram_gate::KreaTurboFit::Resident { .. }
                                | crate::vram_gate::KreaTurboFit::Fits { .. },
                            ) => TierFit::Fits,
                            Some(crate::vram_gate::KreaTurboFit::Reject { needed_gb, .. }) => {
                                TierFit::TooBig {
                                    needed_gb,
                                    available_gb: budget.map_or(0.0, |budget| budget.free_gb),
                                }
                            }
                            Some(crate::vram_gate::KreaTurboFit::Unverified { reason }) => {
                                log_krea_turbo_evidence_exclusion(
                                    &request.model,
                                    candidate,
                                    width,
                                    height,
                                    "capability_downtier",
                                    Some(reason),
                                );
                                candle_tier_fit(
                                    &request.model_manifest_entry,
                                    candidate,
                                    budget,
                                    false,
                                    candidate_adapter_bytes,
                                )
                            }
                            None => {
                                log_krea_turbo_evidence_exclusion(
                                    &request.model,
                                    candidate,
                                    width,
                                    height,
                                    "capability_downtier",
                                    None,
                                );
                                candle_tier_fit(
                                    &request.model_manifest_entry,
                                    candidate,
                                    budget,
                                    false,
                                    candidate_adapter_bytes,
                                )
                            }
                        }
                    } else {
                        candle_tier_fit(
                            &request.model_manifest_entry,
                            candidate,
                            budget,
                            sequential_capable,
                            candidate_adapter_bytes,
                        )
                    };
                    (candidate, fit)
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
                if krea_turbo_ladder && adapter_count == 0 {
                    let smallest_runtime = resolve_tier_dir(request, settings, smallest).and_then(
                        |dir| krea_runtime_evidence_context(request, settings, smallest, &dir),
                    );
                    let lower_resolution = crate::vram_gate::krea_turbo_smaller_fit_with_runtime(
                        &request.model_manifest_entry,
                        smallest,
                        memory_width,
                        memory_height,
                        budget,
                        krea_allow_streamed_blocks,
                        smallest_runtime.as_ref(),
                    );
                    let installed_smaller: Vec<&'static str> =
                        installed_tier_keys(request, settings)
                            .into_iter()
                            .filter(|candidate| {
                                let candidate_runtime =
                                    resolve_tier_dir(request, settings, candidate).and_then(|dir| {
                                        krea_runtime_evidence_context(
                                            request, settings, candidate, &dir,
                                        )
                                    });
                                tier_quality_rank(candidate) < tier_quality_rank(smallest)
                                    && matches!(
                                        crate::vram_gate::krea_turbo_fit_with_runtime(
                                            &request.model_manifest_entry,
                                            candidate,
                                            memory_width,
                                            memory_height,
                                            budget,
                                            krea_allow_streamed_blocks,
                                            candidate_runtime.as_ref(),
                                        ),
                                        Some(
                                            crate::vram_gate::KreaTurboFit::Resident { .. }
                                                | crate::vram_gate::KreaTurboFit::Fits { .. }
                                        )
                                    )
                            })
                            .collect();
                    return Err(WorkerError::InvalidPayload(
                        krea_turbo_capability_reject_message(
                            &request.model,
                            smallest,
                            needed_gb,
                            available_gb,
                            &settings.gpu_id,
                            lower_resolution,
                            &installed_smaller,
                        ),
                    ));
                }
                let resolution_option = "Lower the output resolution, or ";
                return Err(WorkerError::InvalidPayload(format!(
                    "{model} needs ~{needed} GB of VRAM even at the smallest installed tier it can run \
                     ({smallest}) but GPU {gpu} has ~{available} GB available. {resolution_option}Run \
                     on a card with more VRAM.",
                    model = request.model,
                    needed = needed_gb.round() as i64,
                    available = available_gb.round() as i64,
                    gpu = settings.gpu_id,
                )));
            }
        }
    }
    let adapter_resident_bytes =
        candle_adapter_resident_bytes(engine_id, tier, adapter_source_bytes);
    let needed = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        &request.model_manifest_entry,
        tier,
        adapter_resident_bytes,
    );
    let krea_runtime_context = krea_turbo_ladder
        .then(|| krea_runtime_evidence_context(request, settings, tier, &weights_dir))
        .flatten();
    // sc-12090 AC#4/#5: the reject suggestions name only tiers actually INSTALLED and lower-fidelity than
    // the one being rejected — never the rejected tier, never the picker (hidden when ≤1 tier installed).
    // Reached only on the explicit-pick / ConvRot reject below (the downtier path already rejected above
    // when nothing smaller fits), where suggesting a smaller installed tier the user could pick is apt.
    let mut generation_memory: Option<gen_core::GenerationMemory> = None;
    let mut memory_strategy_selection: Option<gen_core::MemorySelection> = None;
    let mut selected_memory_strategy_context: Option<gen_core::MemoryRunContext> = None;
    let mut adapted_peak_gb: Option<f64> = None;
    // Every adopting provider uses the same worker-owned selector. Static
    // Implemented/unverified declarations do not authorize optimized execution: this bridge always
    // submits the conservative resident estimate and adds deeper candidates only when an exact
    // authoritative record exists in the packaged evidence bundle.
    let mut shared_contract_spec = load_spec(weights_dir.clone(), quant, adapters.clone(), None);
    if let Some(pid) = pid_weights.as_ref() {
        shared_contract_spec =
            shared_contract_spec.with_pid(pid.checkpoint.clone(), pid.gemma.clone());
    }
    shared_contract_spec = apply_candle_image_load_shape(engine_id, shared_contract_spec);
    // The selector and the eventual provider load must inspect the SAME artifact set. Mage's tier
    // directory contains only the DiT; its shared text encoder and VAE are caller-staged named
    // components. Evaluating the bare tier dir undercounts the provider contract and can admit a
    // strategy against assets different from those the generator loads.
    shared_contract_spec = attach_required_components(
        shared_contract_spec,
        engine_id,
        &request.model_manifest_entry,
        settings,
    )?;
    let shared_request_mode = candle_base_memory_request_mode(engine_id, &request.mode);
    let reference_count = shared_image_reference_count(
        edit_refs.len(),
        edit_reference.is_some() || img2img_reference.is_some(),
        edit_mask.is_some(),
        hires_fix.is_some(),
    );
    let shared_memory = crate::candle_memory_strategy::evaluate_shared_image(
        engine_id,
        &request.model,
        &shared_contract_spec,
        candle_certified_load_spec(
            engine_id,
            settings,
            &shared_contract_spec,
            &request.model_manifest_entry,
            tier,
        ),
        &request.model_manifest_entry,
        tier,
        shared_request_mode,
        (adapter_count > 0).then_some("lora"),
        gen_core::MemoryGeometry {
            width: memory_width,
            height: memory_height,
            batch: 1,
            frames: 1,
            reference_count,
        },
        reference_count > 0,
        use_pid,
        hires_fix.is_some(),
        budget,
        needed,
        adapter_resident_bytes,
        if reclaimable_gb > 0.0 {
            gen_core::MemoryCacheState::Warm
        } else {
            gen_core::MemoryCacheState::Cold
        },
    )?;
    if let Some(evaluation) = shared_memory {
        memory_strategy_selection = Some(evaluation.context.selection);
        generation_memory = evaluation.memory;
        adapted_peak_gb = Some(evaluation.predicted_peak_gb);
        // Resident is the selector's conservative sentinel, not authority to reconfigure a request.
        // In particular, a later legacy low-VRAM decision may choose sequential residency; carrying a
        // Resident scope would then overwrite that request memory back to resident in configure_request.
        selected_memory_strategy_context = optimized_shared_memory_context(evaluation.context);
    }
    // Krea's shared selector runs before any legacy resident/staged gate and owns the final fit
    // decision whenever its revision-bound evidence is available. A `None` result is the explicit
    // unverified path; only then may the established gate remain as provider-safe fallback.
    let shared_krea_fit = (krea_turbo_ladder && adapter_count == 0)
        .then(|| {
            crate::vram_gate::krea_turbo_fit_with_runtime(
                &request.model_manifest_entry,
                tier,
                memory_width,
                memory_height,
                budget,
                krea_allow_streamed_blocks,
                krea_runtime_context.as_ref(),
            )
        })
        .flatten();
    let auto_tier_streaming = auto_tier_streaming_disclosure(
        explicit_pick,
        tier,
        auto_tier_fit_decision(shared_krea_fit.as_ref()),
    );
    if krea_turbo_ladder {
        match shared_krea_fit {
            Some(crate::vram_gate::KreaTurboFit::Unverified { reason }) => {
                log_krea_turbo_evidence_exclusion(
                    &request.model,
                    tier,
                    width,
                    height,
                    "final_admission",
                    Some(reason),
                );
            }
            None => log_krea_turbo_evidence_exclusion(
                &request.model,
                tier,
                width,
                height,
                "final_admission",
                None,
            ),
            _ => {}
        }
    }
    let use_sequential =
        if let Some(crate::vram_gate::KreaTurboFit::Resident {
            peak_gb,
            needed_gb,
            selection,
        }) =
            shared_krea_fit
        {
            memory_strategy_selection = Some(selection);
            adapted_peak_gb = Some(peak_gb);
            tracing::info!(
                model = %request.model,
                tier,
                width,
                height,
                needed_gb,
                available_gb = budget.map_or(0.0, |budget| budget.free_gb),
                "shared memory-strategy selector retained Krea Turbo resident execution"
            );
            false
        } else {
            // A verified non-resident Krea result bypasses the legacy chooser. Missing or
            // unverified Krea evidence fails closed to resident-or-reject: it must never select a
            // legacy optimized rung.
            let gate_decision = if matches!(
                shared_krea_fit,
                Some(crate::vram_gate::KreaTurboFit::Fits { .. })
                    | Some(crate::vram_gate::KreaTurboFit::Reject { .. })
            ) {
                match (needed, budget) {
                    (Some(needed_gb), Some(budget)) => {
                        crate::vram_gate::FitDecision::Offload {
                            needed_gb,
                            available_gb: budget.free_gb,
                        }
                    }
                    _ => crate::vram_gate::FitDecision::Unknown,
                }
            } else if krea_turbo_ladder {
                krea_unverified_resident_decision(needed, budget)
            } else {
                crate::vram_gate::resolve_offload(
                    crate::vram_gate::fit_decision(needed, budget),
                    sequential_capable,
                )
            };
            match gate_decision {
            crate::vram_gate::FitDecision::Offload {
                needed_gb,
                available_gb,
            } => {
                // The resident peak does not fit. Krea Turbo first evaluates its measured three-stage
                // ladder, even when the older two-stage sequential estimate would fit, so a 24 GB card
                // physically drops the DiT before VAE decode. Other models retain the established
                // sequential-overflow gate below; unverified Krea evidence reaches the explicit
                // resident-or-reject path instead of this arm.
                let sequential_needed = crate::vram_gate::predicted_sequential_peak_gb_with_adapter_bytes(
                    &request.model_manifest_entry,
                    tier,
                    adapter_resident_bytes,
                );
                let krea_selected = if krea_turbo_ladder {
                    match shared_krea_fit {
                        Some(crate::vram_gate::KreaTurboFit::Resident { .. }) => false,
                        Some(crate::vram_gate::KreaTurboFit::Fits {
                            phases,
                            needed_gb,
                            selection,
                            memory,
                            // Measured or estimate-scoped (sc-18097): the selected rung and its
                            // knobs are what this lane acts on, and both are already graded. The
                            // flag exists for refusal ADVICE (`krea_turbo_smaller_fit_*`), which
                            // must not name an estimate-backed geometry.
                            estimate_scoped: _,
                        }) => {
                            memory_strategy_selection = Some(selection);
                            generation_memory = Some(memory);
                            // Reclaim accounting records allocations, not the admission threshold.
                            // `needed_gb` includes the 2 GB safety reserve, which is deliberately never
                            // allocated and therefore cannot be credited back during a model swap.
                            adapted_peak_gb = Some(phases.peak_gb());
                            tracing::info!(
                                model = %request.model,
                                tier,
                                width,
                                height,
                                strategy = ?selection.strategy,
                                text_peak_gb = phases.text_gb,
                                denoise_peak_gb = phases.denoise_gb,
                                decode_peak_gb = phases.decode_gb,
                                needed_gb,
                                available_gb,
                                "Krea Turbo VRAM fit ladder selected the least-cost sufficient rung"
                            );
                            true
                        }
                        Some(crate::vram_gate::KreaTurboFit::Reject { phases, needed_gb }) => {
                            let installed_smaller: Vec<&'static str> =
                                installed_tier_keys(request, settings)
                                    .into_iter()
                                    .filter(|candidate| {
                                        let candidate_runtime =
                                            resolve_tier_dir(request, settings, candidate).and_then(
                                                |dir| {
                                                    krea_runtime_evidence_context(
                                                        request, settings, candidate, &dir,
                                                    )
                                                },
                                            );
                                        tier_quality_rank(candidate) < tier_quality_rank(tier)
                                            && matches!(
                                                crate::vram_gate::krea_turbo_fit_with_runtime(
                                                    &request.model_manifest_entry,
                                                    candidate,
                                                    memory_width,
                                                    memory_height,
                                                    budget,
                                                    krea_allow_streamed_blocks,
                                                    candidate_runtime.as_ref(),
                                                ),
                                                Some(
                                                    crate::vram_gate::KreaTurboFit::Resident { .. }
                                                        | crate::vram_gate::KreaTurboFit::Fits { .. }
                                                )
                                            )
                                    })
                                    .collect();
                            let lower_resolution =
                                crate::vram_gate::krea_turbo_smaller_fit_with_runtime(
                                &request.model_manifest_entry,
                                tier,
                                memory_width,
                                memory_height,
                                budget,
                                krea_allow_streamed_blocks,
                                krea_runtime_context.as_ref(),
                            );
                            let mut options = Vec::new();
                            if let Some((smaller_w, smaller_h)) = lower_resolution {
                                options.push(format!(
                                        "lower the output resolution to {smaller_w}x{smaller_h} or below"
                                    ));
                            }
                            if !installed_smaller.is_empty() {
                                options.push(format!(
                                    "select a smaller installed tier ({})",
                                    installed_smaller
                                        .iter()
                                        .map(|candidate| candidate.to_uppercase())
                                        .collect::<Vec<_>>()
                                        .join(" / ")
                                ));
                            }
                            options.push("run on a GPU with more VRAM".to_owned());
                            return Err(WorkerError::InvalidPayload(format!(
                                    "{model} at {width}x{height} on the {tier} tier needs ~{needed} GB \
                                     of VRAM even with three-stage loading, tiled VAE decode, attention \
                                     chunking, and {streaming}, but GPU {gpu} has ~{available} GB \
                                     available. Measured phase predictions: text {text:.1} GB, denoise \
                                     {denoise:.1} GB, decode {decode:.1} GB. {options}.",
                                    model = request.model,
                                    needed = needed_gb.round() as i64,
                                    available = available_gb.round() as i64,
                                    gpu = settings.gpu_id,
                                    streaming = if krea_allow_streamed_blocks {
                                        "transformer block streaming"
                                    } else {
                                        "the deepest adapter-compatible rung"
                                    },
                                    text = phases.text_gb,
                                    denoise = phases.denoise_gb,
                                    decode = phases.decode_gb,
                                    options = options.join(", or "),
                                )));
                        }
                        Some(crate::vram_gate::KreaTurboFit::Unverified { reason }) => {
                            log_krea_turbo_evidence_exclusion(
                                &request.model,
                                tier,
                                width,
                                height,
                                "optimized_rung",
                                Some(reason),
                            );
                            false
                        }
                        None => {
                            log_krea_turbo_evidence_exclusion(
                                &request.model,
                                tier,
                                width,
                                height,
                                "optimized_rung",
                                None,
                            );
                            false
                        }
                    }
                } else {
                    false
                };
                let shared_memory_selected = memory_strategy_selection
                    .is_some_and(|selection| selection.strategy.is_optimized());
                if rejects_unverified_shared_memory_fallback(
                    engine_id,
                    shared_memory_selected,
                ) {
                    let family = verified_only_memory_family_label(engine_id)
                        .expect("verified-only rejection is restricted to a named family");
                    return Err(WorkerError::InvalidPayload(format!(
                        "{model} at the {tier} tier needs ~{needed} GB of resident VRAM, but GPU \
                         {gpu} has ~{available} GB available and no exact verified {family} memory \
                         strategy fits this request. Install or verify matching calibration evidence, \
                         select a smaller tier or resolution, or use a GPU with more VRAM.",
                        model = request.model,
                        needed = needed_gb.round() as i64,
                        available = available_gb.round() as i64,
                        gpu = settings.gpu_id,
                        family = family,
                    )));
                }
                if !krea_selected && !shared_memory_selected {
                    if let Some(seq_gb) =
                        crate::vram_gate::sequential_overflow_gb(sequential_needed, budget)
                    {
                        let installed_smaller: Vec<&'static str> =
                            installed_tier_keys(request, settings)
                                .into_iter()
                                .filter(|candidate| {
                                    tier_quality_rank(candidate) < tier_quality_rank(tier)
                                })
                                .collect();
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
                let installed_smaller: Vec<&'static str> =
                    installed_tier_keys(request, settings)
                        .into_iter()
                        .filter(|candidate| {
                            let candidate_runtime =
                                resolve_tier_dir(request, settings, candidate).and_then(|dir| {
                                    krea_runtime_evidence_context(
                                        request, settings, candidate, &dir,
                                    )
                                });
                            tier_quality_rank(candidate) < tier_quality_rank(tier)
                                && (!krea_turbo_ladder
                                    || matches!(
                                        crate::vram_gate::krea_turbo_fit_with_runtime(
                                            &request.model_manifest_entry,
                                            candidate,
                                            memory_width,
                                            memory_height,
                                            budget,
                                            krea_allow_streamed_blocks,
                                            candidate_runtime.as_ref(),
                                        ),
                                        Some(
                                            crate::vram_gate::KreaTurboFit::Resident { .. }
                                                | crate::vram_gate::KreaTurboFit::Fits { .. }
                                        )
                                    ))
                        })
                        .collect();
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
        adapted_peak_gb.or_else(|| {
            crate::vram_gate::predicted_sequential_peak_gb_with_adapter_bytes(
                &request.model_manifest_entry,
                tier,
                adapter_resident_bytes,
            )
        })
    } else {
        needed
    };
    if let Some(peak_gb) = incurred_peak {
        crate::vram_gate::note_loaded_peak(&settings.gpu_id, peak_gb);
    }
    let memory_strategy_context = selected_memory_strategy_context.or_else(|| memory_strategy_selection.and_then(|selection| {
        let budget = budget?;
        let predicted_peak_gb = adapted_peak_gb?;
        let turbo_fit = request.model_manifest_entry.get("candle")?.get("turboFit")?;
        let calibration_abi =
            u32::try_from(turbo_fit.get("calibrationAbi")?.as_u64()?).ok()?;
        let calibration_fingerprint = turbo_fit
            .get("calibrationFingerprint")?
            .as_str()?
            .to_owned();
        let gb_to_bytes = |gb: f64| {
            (gb * 1024.0 * 1024.0 * 1024.0)
                .round()
                .clamp(0.0, u64::MAX as f64) as u64
        };
        tracing::info!(
            backend = "candle",
            evidence_revision = krea_evidence_revision(),
            reclaimable_gb,
            raw_available_gb = raw_budget.map_or(0.0, |raw| raw.free_gb),
            effective_available_gb = budget.free_gb,
            requested_tier,
            effective_tier = tier,
            strategy = ?selection.strategy,
            "shared memory-strategy selection admitted"
        );
        // sc-17097: the shape now comes from the manifest, which states what the curves were MEASURED
        // under. Reading it back off the provider - as this did - made the safety check compare the
        // provider against itself, so calibration ABI 2's load-shape axis could never fail here.
        let load_shape = crate::vram_gate::krea_turbo_load_shape(turbo_fit)?;
        Some(gen_core::MemoryRunContext {
            selection,
            calibration_abi,
            calibration_fingerprint,
            load_shape,
            mode: gen_core::MemoryMode::TextToImage,
            has_reference: reference_count > 0,
            use_pid: false,
            has_phases: false,
            geometry: gen_core::MemoryGeometry {
                width: memory_width,
                height: memory_height,
                batch: 1,
                frames: 1,
                reference_count,
            },
            overlay: None,
            budget: gen_core::MemoryBudget {
                total_bytes: gb_to_bytes(budget.total_gb),
                committed_bytes: gb_to_bytes((budget.total_gb - budget.free_gb).max(0.0)),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: gb_to_bytes(2.0),
            },
            predicted_peak_bytes: gb_to_bytes(predicted_peak_gb),
            cache_state: if reclaimable_gb > 0.0 {
                gen_core::MemoryCacheState::Warm
            } else {
                gen_core::MemoryCacheState::Cold
            },
            evidence_revision: krea_evidence_revision(),
        })
    }));
    apply_request_scoped_candle_residency(use_sequential, &mut generation_memory);
    // Reuse the exact selector spec, including Mage's split text-encoder/VAE component paths. Only
    // the post-selection residency policy and optional ConvRot substitution may differ below.
    let mut spec = shared_contract_spec;
    if use_sequential {
        // Ask the provider (candle FLUX) to load→use→drop each component in phase order (sc-10821).
        spec = spec.with_offload_policy(gen_core::OffloadPolicy::Sequential);
    }
    // INT8-ConvRot LoadSpec seam (sc-9300, epic 9083): ride the ConvRot DiT single-file on the shared,
    // already-optional `LoadSpec::text_encoder` as a `WeightsSource::File` while `spec.weights` stays the
    // canonical Krea 2 bf16 snapshot `Dir` (set as `weights_dir` above). The candle-gen krea engine's
    // `convrot_selector` decodes a `File` here → `load_components_convrot` (which enforces the sm_89
    // compute-cap floor); a `Dir`/`None` there is the normal dense/packed path. Other engines ignore it.
    if let Some((_, convrot_dit)) = convrot {
        spec.text_encoder = Some(WeightsSource::File(convrot_dit));
    }

    // Surface the decision before model execution, while the reason for a slow render is still clear,
    // then keep the same compact note on subsequent progress updates. The structured event is the
    // statistics trace; the tracing event remains useful in the worker log. Neither changes admission
    // or tier ordering, and neither creates a dialog/toast UI (Michael's sc-16104 constraint).
    if let Some(disclosure) = auto_tier_streaming.as_ref() {
        raw_settings.insert(
            AUTO_TIER_STREAMING_DISCLOSURE_KEY.to_owned(),
            disclosure.record(),
        );
        tracing::info!(
            job_id = %job.id,
            model = %request.model,
            tier = %disclosure.tier,
            strategy = "bounded_transformer_residency",
            cause = "streaming_transformer_blocks_to_hold_auto_selected_tier",
            measured_materialization_ms_per_block = ?disclosure.measured_materialization_ms_per_block,
            measured_materialization_ms_per_step = ?disclosure.measured_materialization_ms_per_step,
            measurement_source = "sc-16096-rtx-pro-6000-blackwell",
            "streaming transformer blocks to hold the auto-selected tier"
        );
        emit_event(
            "image_memory_strategy_selected",
            disclosure.telemetry(&job.id, &request.model),
        );
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::LoadingModel,
                ProgressStage::LoadingModel,
                0.0,
                &disclosure.progress_message("Preparing render."),
                Some(streaming_result(plan, asset_writes)),
                backend,
            ),
        )
        .await?;
    }

    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("candle {engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(tx, seeds, move |_index, seed, preview, on_progress| {
                let render = |seed: i64, on_progress: &mut dyn FnMut(Progress)| {
                    generate_one_with_hires(
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
                        &edit_refs,
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
                        generation_memory,
                        memory_strategy_context.as_ref(),
                        &enhance,
                        hires_fix,
                        preview.clone(),
                        &cancel,
                        on_progress,
                    )
                };
                // Ideogram 4 placeholder detect-and-reseed (sc-6858, parity with the macOS
                // `generate_stream` net, sc-6501): the caption guard above makes it rare, but a residual
                // "Image blocked by safety filter" placeholder can still occur even with a caption.
                // Detect via the baked-text heuristic and reseed transparently, keeping the first clean
                // render. Gated to Ideogram 4; a no-op for every other candle family, for turbo (CFG-free,
                // cannot produce it), and for an edit (the output is anchored to a real source latent, so
                // `looks_like_placeholder` returns false).
                let initial = render(seed, on_progress)?;
                let (final_seed, out_w, out_h, pixels) = recover_ideogram_placeholder(
                    is_ideogram,
                    seed,
                    &cancel,
                    initial,
                    |retry_seed| render(retry_seed, on_progress),
                )?;
                Ok(Some((final_seed, out_w, out_h, pixels)))
            })
        },
    );

    consume_gen_events_with_disclosure(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        &raw_settings,
        auto_tier_streaming.as_ref(),
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
    let (selected_quant, selected_bits) = resolve_quant_gated(request, nvfp4_host, tier_dir);
    let resolved = match (selected_quant, selected_bits) {
        // The distinct NVFP4 tier, matched on the VARIANT (see the note above): its bit count is
        // `None`, so a bits-only match would silently label it "bf16".
        (Some(Quant::Nvfp4), _) => (Some(NVFP4_TIER.to_owned()), None),
        (_, Some(8)) => (Some("q8".to_owned()), Some(8)),
        (_, Some(4)) => (Some("q4".to_owned()), Some(4)),
        _ => (Some("bf16".to_owned()), None),
    };
    let Some(selected) = selected_quant else {
        return resolved;
    };
    let Some(model) = mlx_model(&request.model) else {
        return resolved;
    };
    let active = model
        .descriptor
        .capabilities
        .component_precision_floors
        .iter()
        .copied()
        .filter(|floor| floor.applies_to(selected))
        .collect::<Vec<_>>();
    if active.is_empty() {
        return resolved;
    }
    let selected_label = match selected {
        Quant::Q4 => "q4",
        Quant::Q8 => "q8",
        Quant::Nvfp4 => NVFP4_TIER,
    };
    let floors = active
        .iter()
        .map(|floor| {
            let resident = match floor.resident_tier {
                Quant::Q4 => "q4",
                Quant::Q8 => "q8",
                Quant::Nvfp4 => NVFP4_TIER,
            };
            format!("{}:{resident}", floor.component.as_str())
        })
        .collect::<Vec<_>>()
        .join(",");
    (
        Some(format!("{selected_label}+[{floors}]")),
        None, // A mixed component profile has no honest single bit width.
    )
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

/// Latest per-step preview frame for the in-flight image (epic 16624, sc-16904). One slot,
/// latest-wins: `ProgressRequest.result` is replaced wholesale on every accepted POST, so the
/// frame rides the existing per-step update and is never accumulated. Cleared when its image's
/// final asset lands (`GenEvent::Image`), and absent from the terminal result by construction.
struct PreviewSlot {
    index: usize,
    current: u32,
    total: u32,
    data_url: String,
}

/// Encode one latent-resolution preview frame (~128×128 RGB8) as a `data:image/jpeg` URL,
/// ~10 KB at quality 70. Decorative by contract: any failure drops the frame, never the job.
fn encode_preview_data_url(frame: &gen_core::PreviewFrame) -> Option<String> {
    use base64::Engine as _;
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut std::io::Cursor::new(&mut jpeg), 70)
        .encode(
            &frame.image.pixels,
            frame.image.width,
            frame.image.height,
            image::ExtendedColorType::Rgb8,
        )
        .inspect_err(|error| {
            tracing::debug!(%error, "preview frame JPEG encode failed; dropping frame");
        })
        .ok()?;
    Some(format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&jpeg)
    ))
}

/// [`streaming_result`] plus the current preview frame, when one is live.
fn streaming_result_with_preview(
    plan: &ImagePlan,
    asset_writes: &[Value],
    preview: Option<&PreviewSlot>,
) -> JsonObject {
    let mut result = streaming_result(plan, asset_writes);
    if let Some(slot) = preview {
        result.insert(
            "previewFrame".to_owned(),
            json!({
                "imageIndex": slot.index,
                "current": slot.current,
                "total": slot.total,
                "dataUrl": slot.data_url,
            }),
        );
    }
    result
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
    rx: tokio::sync::mpsc::Receiver<GenEvent>,
    cancel: CancelFlag,
    blocking: tokio::task::JoinHandle<WorkerResult<()>>,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    consume_gen_events_with_disclosure(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        raw_settings,
        None,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn consume_gen_events_with_disclosure(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    adapter_label: &str,
    raw_settings: &JsonObject,
    auto_tier_streaming: Option<&AutoTierStreamingDisclosure>,
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
    // Live denoise preview (sc-16904): the latest frame rides the next progress POST.
    let mut latest_preview: Option<PreviewSlot> = None;
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
                        &progress_with_auto_tier_streaming(
                            &format!("Image {}/{total} — step {current}/{step_total}.", index + 1),
                            auto_tier_streaming,
                        ),
                        Some(streaming_result_with_preview(
                            plan,
                            asset_writes,
                            latest_preview.as_ref(),
                        )),
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
                        Some(streaming_result_with_preview(
                            plan,
                            asset_writes,
                            latest_preview.as_ref(),
                        )),
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
                        &progress_with_auto_tier_streaming(
                            &format!("Image {}/{total} — loading {component}.", index + 1),
                            auto_tier_streaming,
                        ),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Preview { index, frame } => {
                mark_started(index);
                // Encode off the async runtime thread (parity with the asset write, sc-8909).
                // No POST here: the frame rides the NEXT Step/Decoding update, so preview traffic
                // adds zero requests to the existing per-step cadence.
                let encoded = tokio::task::spawn_blocking(move || {
                    let data_url = encode_preview_data_url(&frame)?;
                    Some(PreviewSlot {
                        index,
                        current: frame.current,
                        total: frame.total,
                        data_url,
                    })
                })
                .await
                .map_err(|error| crate::task_join_error("preview frame encode task", error))?;
                if let Some(slot) = encoded {
                    latest_preview = Some(slot);
                }
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
                // The finished asset supersedes its interim preview: drop the slot before this
                // arm's result POST so the replaced `result` no longer carries the frame (a later
                // image in the batch starts its own stream).
                if latest_preview
                    .as_ref()
                    .is_some_and(|slot| slot.index == index)
                {
                    latest_preview = None;
                }
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
        // The explicit streaming result keeps the partial images already streamed (it IS
        // that fact set; `persist_reported_assets` re-injects the derived asset keys) while
        // dropping any in-flight `previewFrame` from the stored result — a canceled job must
        // not carry a stale interim frame (sc-16904). Before previews this posted result=None
        // for the same partial-image outcome via `coalesce`.
        let message = "Image generation canceled by user.";
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                message,
                Some(streaming_result(plan, asset_writes)),
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
    fn mage_edit_preserves_primary_then_ordered_extra_references() {
        let request = ImageRequest::from_payload(
            json!({
                "mode": "edit_image",
                "model": "mage_flow_edit_turbo",
                "sourceAssetId": "primary",
                "referenceAssetIds": ["second", "third"]
            })
            .as_object()
            .unwrap(),
        );
        assert_eq!(
            mage_edit_reference_ids(&request),
            ["primary", "second", "third"]
        );
        assert_eq!(
            CandleImageRoute::MageEdit.adapter_label(&request),
            "candle_mage",
            "Mage edit assets and generation sets must retain the Candle family label"
        );

        let missing_primary = ImageRequest::from_payload(
            json!({
                "mode": "edit_image",
                "model": "mage_flow_edit",
                "referenceAssetIds": ["orphan"]
            })
            .as_object()
            .unwrap(),
        );
        assert!(
            mage_edit_reference_ids(&missing_primary).is_empty(),
            "Mage edit must not reinterpret an optional reference as its required primary source"
        );
    }

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
        for model in [
            "mage_flow_base",
            "mage_flow",
            "mage_flow_turbo",
            "mage_flow_edit_base",
            "mage_flow_edit",
            "mage_flow_edit_turbo",
        ] {
            assert_eq!(candle_adapter_label(model), "candle_mage");
        }
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

    /// sc-14432: a SenseNova tier's backbone probe passes on the flat `model.safetensors` alone, so a
    /// tier missing the rest of what its loader reads short-circuited the chain and then died at load
    /// ("complete but unloadable"). Under the real `sensenova_u1_8b` id the resolver now folds in
    /// `sensenova_tier_complete` and falls through to a complete sibling instead. Note the tokenizer is
    /// model-wide: a tier that ships none still loads by borrowing a sibling tier's copy, so it is
    /// `config.json` + the backbone that make a tier tier-specifically complete.
    #[test]
    fn sensenova_tier_selection_skips_a_torn_tier_for_a_complete_sibling() {
        let sensenova_request = |bits: serde_json::Value| {
            ImageRequest::from_payload(
                json!({ "model": "sensenova_u1_8b", "advanced": { "mlxQuantize": bits } })
                    .as_object()
                    .unwrap(),
            )
        };
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let seed_tier = |tier: &str, complete: bool| {
            let dir = root.join(tier);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
            if complete {
                std::fs::write(dir.join("config.json"), b"x").unwrap();
                std::fs::write(dir.join("tokenizer.json"), b"x").unwrap();
            }
        };

        // Only a torn q4 is installed: nothing complete anywhere, so the resolver still lands on it
        // (pre-sc-12279 behavior — never strand a model that might load) and warns.
        seed_tier("q4", false);
        assert_eq!(
            standard_tier_subdir(root, &sensenova_request(json!(4))),
            root.join("q4")
        );

        // Add a complete bf16: the explicit q4 pick now falls through to it rather than loading the torn
        // tier. This is the assertion that goes RED without the completeness fold.
        seed_tier("bf16", true);
        assert_eq!(
            standard_tier_subdir(root, &sensenova_request(json!(4))),
            root.join("bf16")
        );

        // Complete q4 with its own `config.json` — it borrows bf16's tokenizer (the engine's sibling
        // resolution), so it is loadable and the explicit pick is honored again.
        std::fs::write(root.join("q4").join("config.json"), b"x").unwrap();
        assert_eq!(
            standard_tier_subdir(root, &sensenova_request(json!(4))),
            root.join("q4")
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
    /// `flux2_klein_9b`/`_kv` declare `mlx.denseTextEncoderTier` AND ride the candle txt2img lane, whose
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
        // The declared dense-TE turnkey (sc-15799: the manifest flag is the only form) and a novel
        // manifest-flag model, with EVERY gate for the NVFP4 arm satisfied: explicit pick, Blackwell
        // host, and the `nvfp4/` tier resolved.
        let by_id = ImageRequest::from_payload(
            json!({
                "model": "flux2_klein_9b",
                "advanced": { "quantTier": "nvfp4" },
                "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } }
            })
            .as_object()
            .unwrap(),
        );
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

    /// sc-16025: provider-local floors are part of the effective numeric identity. The label stored
    /// on the asset/effective-settings record must not collapse Mage's mixed q4/q8 load to plain q4.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn mage_q4_label_includes_every_descriptor_declared_component_floor() {
        let request = ImageRequest::from_payload(
            json!({ "model": "mage_flow", "advanced": { "mlxQuantize": 4 } })
                .as_object()
                .unwrap(),
        );
        let q4_dir = PathBuf::from("/models/mage_flow/q4");
        let (label, bits) = effective_quant_label_gated(&request, false, Some(&q4_dir));
        assert_eq!(
            label.as_deref(),
            Some("q4+[textEncoder:q8,transformerHead:q8]")
        );
        assert_ne!(label.as_deref(), Some("q4"));
        assert_eq!(bits, None, "a mixed-width profile has no single bit count");
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

    /// sc-14249 — the candle SenseNova lane resolves the tier the REQUEST asks for.
    ///
    /// This replaces the two sc-13817 tests that pinned it to `bf16/`. That force existed only
    /// because `candle-gen-sensenova` mmapped at a hardcoded F32 and rejected `spec.quantize`, so
    /// packed weights were unreadable; the engine now packed-detects every backbone projection, so
    /// the family takes the ordinary `standard_tier_subdir` descent and the cheap tiers are reachable
    /// at last. Asserting the descent HERE is what would catch a re-introduced dense force — the
    /// symptom of which is silent: every job still renders, just on the 70.5 GB tier.
    ///
    /// All three tiers are seeded COMPLETE so a result cannot be an accident of a tier being absent
    /// (the completeness fallback would reach a sibling on its own — the shape that masked sc-13817
    /// on a half-provisioned host).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn sensenova_candle_tier_follows_the_request_not_a_forced_dense_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for tier in ["q4", "q8", "bf16"] {
            std::fs::create_dir_all(root.join(tier)).unwrap();
            std::fs::write(root.join(tier).join("model.safetensors"), "weights").unwrap();
        }
        let req = |model: &str, advanced: serde_json::Value| {
            ImageRequest::from_payload(
                json!({
                    "model": model,
                    "advanced": advanced,
                    "modelManifestEntry": { "mlx": { "standardTierLayout": true } }
                })
                .as_object()
                .unwrap(),
            )
        };
        for model in [
            "sensenova_u1_8b",
            "sensenova_u1_8b_fast",
            "sensenova_u1_8b_infographic_v2",
            "sensenova_u1_8b_infographic_v3",
            "sensenova_u1_8b_infographic_v2_fast",
            "sensenova_u1_8b_infographic_v3_fast",
        ] {
            // An explicit pick is honored — the packed tiers are the point of the story.
            for (bits, tier) in [(4, "q4"), (8, "q8"), (0, "bf16")] {
                assert_eq!(
                    standard_tier_subdir(root, &req(model, json!({ "mlxQuantize": bits }))),
                    root.join(tier),
                    "{model} with mlxQuantize {bits} must resolve {tier}"
                );
            }
            // ...and with no pick, the app-wide q8 default — NOT a forced bf16.
            assert_eq!(
                standard_tier_subdir(root, &req(model, json!({}))),
                root.join("q8"),
                "{model} default must be the shared q8, not a dense force"
            );
        }
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

    /// sc-15799: the dense-TE guard is MANIFEST-ONLY. An above-tier residency the shared decision cannot
    /// see is the defect that story removes, so the id alone must no longer grant the carve-out —
    /// `flux2_klein_9b` gets it from the catalog (pinned against the shipped manifest by
    /// `tests/gpu_and_manifest.rs`) or not at all.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn dense_te_tier_is_declared_only_in_the_manifest() {
        assert!(
            !is_dense_te_tier(&manifest_request("flux2_klein_9b", json!({}))),
            "a bare id must NOT grant the carve-out any more — the deleted hardcoded registry is \
             exactly the invisible exception sc-15799 removes"
        );
        assert!(is_dense_te_tier(&manifest_request(
            "flux2_klein_9b",
            json!({ "denseTextEncoderTier": true })
        )));
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

    /// The same pre-flight dispatcher for SenseNova-U1 (sc-14432): a torn tier reads incomplete, and the
    /// distilled `_fast` id additionally requires the pre-merged `distill_merged.json` marker — the base
    /// id must NOT, or every base install would report a spurious gap.
    #[cfg(target_os = "macos")]
    #[test]
    fn resolved_tier_is_complete_flags_a_torn_sensenova_tier_and_a_fast_tier_missing_its_marker() {
        let sensenova = |model: &str| {
            ImageRequest::from_payload(
                json!({ "model": model, "advanced": {} }).as_object().unwrap(),
            )
        };
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let seed = |tier: &str, complete: bool| {
            let dir = root.join(tier);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
            if complete {
                std::fs::write(dir.join("config.json"), b"x").unwrap();
                std::fs::write(dir.join("tokenizer.json"), b"x").unwrap();
            }
        };
        seed("q8", false);
        seed("bf16", true);

        let base = sensenova("sensenova_u1_8b");
        assert!(!resolved_tier_is_complete(&base, &root.join("q8")));
        assert!(resolved_tier_is_complete(&base, &root.join("bf16")));

        // The bf16 tier satisfies the BASE contract but has no distill marker, so it is complete for the
        // base id and incomplete for the fast one — the discriminating pair for the fast arm.
        let fast = sensenova("sensenova_u1_8b_fast");
        assert!(!resolved_tier_is_complete(&fast, &root.join("bf16")));
        std::fs::write(root.join("bf16").join("distill_merged.json"), b"{}").unwrap();
        assert!(resolved_tier_is_complete(&fast, &root.join("bf16")));
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
                json!({ "model": "flux2_klein_9b", "advanced": { "mlxQuantize": mlx_quantize }, "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } } })
                    .as_object()
                    .unwrap(),
            )
        };
        let default = ImageRequest::from_payload(
            json!({ "model": "flux2_klein_9b", "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } } }).as_object().unwrap(),
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
            json!({ "model": "flux2_klein_9b", "advanced": {}, "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } } })
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
            json!({ "model": "flux2_klein_9b", "advanced": { "mlxQuantize": 8 }, "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } } })
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
        let root_guard = tempfile::Builder::new()
            .prefix("mlx_downtier_emu_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
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
        let probe = |dir: &PathBuf| LoadSpec::new(WeightsSource::Dir(dir.clone()));
        let q8_fit = mlx_tier_fit(engine, &probe(&q8_dir));
        let q4_fit = mlx_tier_fit(engine, &probe(&q4_dir));
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
        let q8_fit = candle_tier_fit(&manifest, "q8", budget, false, 0);
        let q4_fit = candle_tier_fit(&manifest, "q4", budget, false, 0);
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
