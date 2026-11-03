// Image fit/crop/pad geometry shared by the MLX edit handlers (flux2/qwen/sdxl/kolors/sensenova/
// zimage), the candle edit handlers (*_edit_candle.rs), and the video I2V resolve paths
// (video_jobs.rs). Kept in base.rs — included on macOS AND the `backend-candle` lane (and nowhere
// else) — so `crate::image_jobs::fit_engine_image` resolves on exactly the lanes that call it. Moved
// here from the macOS MLX flux2.rs (sc-6231; the sc-6139 fit-mode refactor left the shared helper macOS-gated, which
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

/// Resolve one exact imported generate surface and reject every request axis the selected provider
/// cannot execute. Source shape is part of the registry key: a sibling route for the same family is
/// never unioned in. The ComfyUI lanes are ordinary Generate routes; only a singular Reference is
/// admitted when that exact descriptor advertises it.
fn imported_generate_request_has_unsupported_shape(request: &ImageRequest) -> bool {
    request.mode == "edit_image"
        || !pose_entries(request).is_empty()
        || request.source_asset_id.is_some()
        || request.mask_asset_id.is_some()
        || request.character_id.is_some()
        || request.character_look_id.is_some()
        || !request.reference_asset_ids.is_empty()
        || request_has_multiphase(request)
        || sceneworks_core::jobs_store::imported_control_intent_is_material(&request.advanced)
}

fn imported_generate_request_supported(
    request: &ImageRequest,
    family: &str,
    source: gen_core::ImportedModelSource,
) -> Option<gen_core::ModelDescriptor> {
    // Reject shape before registry/path resolution so every shared Generate consumer (Mage and the
    // three ComfyUI Candle lanes) refuses material control intent at the same admission seam.
    if imported_generate_request_has_unsupported_shape(request) {
        return None;
    }
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        family,
        source,
        gen_core::ImportedModelOperation::Generate,
    )?;
    if request.reference_asset_id.is_some()
        && !descriptor
            .capabilities
            .conditioning
            .contains(&gen_core::ConditioningKind::Reference)
    {
        return None;
    }
    if (!request.loras.is_empty())
        && !(descriptor.capabilities.supports_lora || descriptor.capabilities.supports_lokr)
    {
        return None;
    }
    Some(descriptor)
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
    /// FLUX.1's IP-Adapter and strict-control providers are separate graphs.  A reference plus
    /// pose set must not be claimed by the control route, which would otherwise consume the pose
    /// while treating the reference only as post-generation scoring input.
    FluxIpAdapterPoseReject,
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
    /// A user model bound to a persisted checkpoint import plan (epic 20398, sc-20634): its manifest
    /// entry carries `importPlan.checkpointId`, the plan store resolves and re-verifies the plan, and
    /// the provider is selected by family + source shape + operation through the registry. Claimed
    /// BEFORE every bespoke imported lane, but PER REQUEST, not per entry (sc-20636): this skeleton
    /// serves plain text-to-image only, so a managed entry's family lane keeps the edit / pose /
    /// reference / LoRA / Hires.fix shapes it had before the import. Exactly one lane owns each
    /// REQUEST; the entry itself may be served by two. A request no lane claims at all is a typed
    /// refusal, never the stub.
    CheckpointPlan,
    /// An imported/user single-file Krea 2 checkpoint (epic 14015 S0c, sc-14018): a non-builtin
    /// `krea_2`-family model whose `modelPath` is a single `.safetensors` DiT → the bespoke in-place
    /// assembly lane, which pairs the imported transformer with a resident `krea_2` base tier (shared
    /// Qwen3-VL TE / Qwen VAE / tokenizer) and loads through the native MLX or Candle single-file
    /// entrypoint. A
    /// builtin Krea id (`krea_2_turbo` / `krea_2_raw`, in `MODEL_TABLE`) never reaches here —
    /// `resolve_imported_krea_dit` returns `None` for it, so the snapshot-dir Krea path is untouched.
    /// The resident base tier supplies the img2img/edit conditioning and LoRA/LoKr components.
    KreaImported,
    /// An imported/user single-file Krea 2 checkpoint carrying a **strict-pose set** (a non-empty
    /// `advanced.poses` outside edit mode): the trained pose control-branch overlay rides the
    /// FILE-LOADED imported DiT via the native MLX or Candle control entrypoint, one image per pose
    /// — the imported twin of
    /// [`ImageRoute::KreaControl`]. Claimed BEFORE the plain [`ImageRoute::KreaImported`] arm so a
    /// pose set gets the per-pose count + control render instead of per-image generation.
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
    /// Generic SDXL OpenPose ControlNet, shared by the five independently accepted model ids.
    SdxlControl,
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
    /// A strict-pose job on an MLX model with NO pose-control lane that `mlx_available` would
    /// otherwise render as plain txt2img, dropping the poses. Reject loudly (sc-5968) — the MLX twin of
    /// the candle `CandleImageRoute::PoseReject`.
    PoseReject,
    Mlx,
}

/// Image model ids the MLX router HAS a bespoke strict-pose control lane for — each is claimed by an
/// `… _control_available` arm in [`prepare_image_route`] BEFORE the generic `mlx_available` txt2img arm,
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
/// Distinct from a non-wired MLX pose family, which reaches the sc-5968
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

/// FLUX.1-dev has independent IP-Adapter and strict-control providers, neither of which composes
/// the other's conditioning. Keep this predicate at route selection so a combined request is
/// refused before either provider can silently discard the other input.
#[cfg(native_lane)]
fn flux1_ipadapter_pose_combination(request: &ImageRequest) -> bool {
    request.model == "flux_dev"
        && request.mode != "edit_image"
        && non_empty(&request.reference_asset_id)
        && !pose_entries(request).is_empty()
}

/// The production FLUX strict-control router, expressed as the exact base-model → dedicated
/// provider mapping used by the two `..._control_available` arms below. Keeping this pure seam next
/// to [`prepare_image_route`] lets source-bound audits ask the router whether a model really has a
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
#[allow(clippy::too_many_arguments)]
fn resolve_image_route_with_imported_availability(
    request: &ImageRequest,
    settings: &Settings,
    checkpoint_plan_available: bool,
    imported_control_available: bool,
    imported_available: bool,
    sdxl_imported_available: bool,
    mage_finetuned_available: bool,
) -> Option<ImageRoute> {
    if flux1_ipadapter_pose_combination(request) {
        Some(ImageRoute::FluxIpAdapterPoseReject)
    } else if zimage_control_available(request, settings) {
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
    } else if checkpoint_plan_available {
        // A plan-backed user checkpoint (epic 20398, sc-20634): the persisted ImportPlanV1 is the
        // route's sole source of truth. Claimed before every bespoke imported lane.
        Some(ImageRoute::CheckpointPlan)
    } else if imported_control_available {
        // An imported single-file Krea 2 checkpoint + a strict-pose set: the pose control branch
        // rides the file-loaded imported DiT (the imported twin of the `KreaControl` arm above).
        // Checked BEFORE the plain imported arm so a pose set renders one pose-locked image per
        // pose instead of falling into per-image t2i (which would silently drop the poses).
        Some(ImageRoute::KreaImportedControl)
    } else if imported_available {
        // An imported/user single-file Krea 2 checkpoint (epic 14015 S0c, sc-14018): a non-builtin
        // `krea_2`-family model whose `modelPath` is a single `.safetensors` DiT → the bespoke in-place
        // assembly lane. A builtin Krea id never claims this (`resolve_imported_krea_dit` returns `None`
        // for a `MODEL_TABLE` id), so the generic registry-backed snapshot-dir arms are unchanged for
        // builtin Krea. Imported ids are not in `MODEL_TABLE`; the bespoke MLX/Candle gates claim them
        // here for text-to-image, img2img/edit, and adapter-bearing generation.
        Some(ImageRoute::KreaImported)
    } else if sdxl_imported_available {
        Some(ImageRoute::SdxlImported)
    } else if mage_finetuned_available {
        // A fine-tuned Mage-Flow base (sc-15036). The fine-tune's id is in no `MODEL_TABLE`, so
        // `mlx_available` is `false` for it — this arm is what routes it to real MLX generation at
        // all. A builtin Mage id is claimed by the generic `mlx_available` arm below, unchanged.
        Some(ImageRoute::MageFinetuned)
    } else if instantid_available(request, settings) {
        Some(ImageRoute::InstantId)
    } else if sdxl_control_candidate(request) {
        // Keep the generic SDXL ControlNet route after InstantID and before every SDXL edit,
        // IP-Adapter, and generic route. The handler validates conflicts and malformed carriers,
        // so a pose can never be silently discarded by a sibling conditioned path.
        Some(ImageRoute::SdxlControl)
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
        //  - A non-wired MLX pose family that `mlx_available` would render as plain txt2img →
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

/// Plan-route availability for the test-only route probes, with refusals made loud.
///
/// `prepare_checkpoint_plan_sources` has four outcomes: not plan-backed (an empty selection),
/// servable (`is_available()`), a typed refusal (`Err`), and a DECLINE that retains the refusal for
/// the router's zero-lane fall-through. Only the first is "this route is unavailable"; the other
/// three mean the route OWNS the entry. Folding the refusal into `false` made every plan refusal
/// indistinguishable from a missing route in the probes.
///
/// Gated per lane rather than a bare `#[cfg(test)]`: this helper only exists where a route probe
/// does, and `tests/test_builtin_manifest_audit.py` treats the first line-initial `#[cfg(test)]`
/// as the start of this file's test module — a bare one here would truncate the audited region to
/// the first 400 lines and silently drop base.rs's `repo` / `modelPath` lanes from the inventory.
#[cfg(all(test, native_lane))]
fn checkpoint_plan_available_or_panic(request: &ImageRequest, settings: &Settings) -> bool {
    match prepare_checkpoint_plan_sources(request, settings) {
        Ok(selection) => selection.is_available(),
        Err(error) => panic!(
            "the plan-driven route refused {:?} while probing which route claims it; probe with \
             `prepare_checkpoint_plan_sources` directly to assert on a refusal: {error}",
            request.model
        ),
    }
}

/// The companion directories a candle floor prices for an imported single-file load that pairs a
/// user checkpoint with a resident family base snapshot.
///
/// The base snapshot is a COMPANION, never the weights: the imported file REPLACES the snapshot's
/// `transformer/`, so pricing the snapshot directory recursively charges the DiT twice — the exact
/// double-claim `base_admission::imported_file_floor_excludes_the_replaced_snapshot_transformer`
/// exists to forbid. A selected encoder is already represented by the spec's prepared contract
/// receipt, so the bundled `text_encoder/` is priced only when the spec has none of its own.
///
/// Shared rather than duplicated: the legacy Krea imported lane and the plan-driven checkpoint
/// route must be admitted on an identical floor for the same file + base, and equality by
/// construction is the only version of that which cannot drift.
#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
pub(super) fn imported_base_snapshot_companions(
    base_dir: &Path,
    spec_has_selected_text_encoder: bool,
) -> Vec<PathBuf> {
    let mut companions = vec![base_dir.join("vae")];
    if !spec_has_selected_text_encoder {
        companions.push(base_dir.join("text_encoder"));
    }
    companions
}

/// Test-facing pure route probe. Production uses [`prepare_image_route`] so payload-selected File
/// sources stay pinned from selection through dispatch instead of being resolved a second time after
/// the async preamble.
#[cfg(all(target_os = "macos", test))]
fn resolve_image_route(request: &ImageRequest, settings: &Settings) -> Option<ImageRoute> {
    resolve_image_route_with_imported_availability(
        request,
        settings,
        // Loud, not `matches!`: a plan-backed entry that REFUSES is a typed diagnostic, and
        // collapsing it into "route unavailable" would make every refusal look like a missing
        // route in the probes (sc-20634 review). Callers that want to observe a refusal call
        // `prepare_checkpoint_plan_sources` directly.
        checkpoint_plan_available_or_panic(request, settings),
        krea_imported_control_available(request, settings),
        krea_imported_available(request, settings),
        sdxl_imported_available(request, settings),
        mage_finetuned_available(request, settings),
    )
}

#[cfg(target_os = "macos")]
enum PreparedImageRoute {
    Plain(ImageRoute),
    CheckpointPlan(Box<PreparedCheckpointPlanSources>),
    KreaImported(Box<PreparedKreaImportedSources>),
    KreaImportedControl(Box<PreparedKreaImportedControlSources>),
    SdxlImported(Box<PreparedSdxlImportedSources>),
    MageFinetuned(Box<PreparedMageFinetunedTransformer>),
}

#[cfg(target_os = "macos")]
impl PreparedImageRoute {
    fn kind(&self) -> ImageRoute {
        match self {
            Self::Plain(route) => *route,
            Self::CheckpointPlan(_) => ImageRoute::CheckpointPlan,
            Self::KreaImported(_) => ImageRoute::KreaImported,
            Self::KreaImportedControl(_) => ImageRoute::KreaImportedControl,
            Self::SdxlImported(_) => ImageRoute::SdxlImported,
            Self::MageFinetuned(_) => ImageRoute::MageFinetuned,
        }
    }
}

/// Resolve the production route and retain every payload-selected File token it owns. Selection is
/// synchronous and fallible; the resulting route value survives all later awaits and is consumed by
/// the matching handler.
#[cfg(target_os = "macos")]
fn prepare_image_route(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedImageRoute>> {
    // The plan-driven route is offered the request first. It claims the shapes it serves; a shape
    // it does not serve declines to the entry's own bespoke family lane when the entry has loadable
    // bytes, and refuses outright when it does not. The decline RETAINS its refusal, re-raised
    // below if no lane claims the request at all (sc-20636).
    let checkpoint_plan = prepare_checkpoint_plan_sources(request, settings)?;
    let imported_control = prepare_krea_imported_control_sources(request, settings)?;
    let imported = if imported_control.is_none() {
        prepare_krea_imported_sources(request, settings)?
    } else {
        None
    };
    let sdxl = prepare_sdxl_imported_sources(request, settings)?;
    let mage_finetuned = prepare_mage_finetuned_transformer(request, settings)?;
    let Some(kind) = resolve_image_route_with_imported_availability(
        request,
        settings,
        checkpoint_plan.is_available(),
        imported_control.is_some(),
        imported.is_some(),
        sdxl.is_some(),
        mage_finetuned.is_some(),
    ) else {
        // Zero lanes claimed the request. For a plan-backed entry that is never "no route": the
        // plan route declined only because the entry has SceneWorks-owned bytes some bespoke lane
        // could load, and here that lane did not materialize. Falling out with `Ok(None)` lands the
        // job in `generate_stub_stream`, which COMPLETES it with procedural stub output — so
        // re-raise the retained typed refusal instead (sc-20636 review).
        checkpoint_plan.into_unclaimed_refusal(request)?;
        return Ok(None);
    };
    Ok(Some(match kind {
        ImageRoute::CheckpointPlan => PreparedImageRoute::CheckpointPlan(Box::new(
            checkpoint_plan
                .into_sources()
                .expect("prepared checkpoint plan route lost its sources"),
        )),
        ImageRoute::KreaImportedControl => PreparedImageRoute::KreaImportedControl(Box::new(
            imported_control.expect("prepared imported-control route lost its sources"),
        )),
        ImageRoute::KreaImported => PreparedImageRoute::KreaImported(Box::new(
            imported.expect("prepared imported route lost its sources"),
        )),
        ImageRoute::SdxlImported => PreparedImageRoute::SdxlImported(Box::new(
            sdxl.expect("prepared SDXL imported route lost its sources"),
        )),
        ImageRoute::MageFinetuned => PreparedImageRoute::MageFinetuned(Box::new(
            mage_finetuned.expect("prepared Mage fine-tuned route lost its transformer"),
        )),
        route => PreparedImageRoute::Plain(route),
    }))
}

#[cfg(target_os = "macos")]
impl ImageRoute {
    /// True only for routes whose actual load path applies the request's LoRA/LoKr stack.
    fn applies_request_loras(self) -> bool {
        matches!(
            self,
            ImageRoute::ZImageControl
                | ImageRoute::ZImageBaseControl
                | ImageRoute::SdxlControl
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
            | ImageRoute::SdxlControl
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
            // Imported single-file Krea 2 renders `count` images here for text-to-image or reference/edit
            // conditioning; strict-pose requests use the separate control route above.
            | ImageRoute::KreaImported
            | ImageRoute::SdxlImported
            // Plan-driven checkpoints (sc-20634) are plain per-image txt2img: `count` renders.
            | ImageRoute::CheckpointPlan
            // A fine-tuned Mage-Flow base (sc-15036) is plain per-image txt2img too: `count`
            // renders, each its own seed. No angle/pose grouping (the lane claims no conditioning).
            | ImageRoute::MageFinetuned
            | ImageRoute::FluxIpAdapterPoseReject
            | ImageRoute::PoseControlBaseMissing
            | ImageRoute::PoseReject => request.count,
            // The generic lane also serves a native prompt batch (sc-25681): `prompts × count`.
            ImageRoute::Mlx => request.image_count(),
        }
    }

    /// The label written by this route's per-asset stream. Bespoke routes that do not map directly
    /// to the request's registry row are named explicitly; the remaining MLX routes use the same
    /// descriptor label their stream resolves.
    fn adapter_label(self, request: &ImageRequest) -> &'static str {
        match self {
            ImageRoute::SdxlControl => SDXL_CONTROL_ADAPTER_LABEL,
            ImageRoute::KreaControl => KREA_CONTROL_ENGINE_ID,
            ImageRoute::CheckpointPlan => CHECKPOINT_PLAN_ENGINE,
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
    /// Typed native MOSS editing, using the shared MOSS stream rather than generic conditioning.
    MossImageEdit,
    /// InstantID identity (sc-5491) — the off-Mac sibling of `ImageRoute::InstantId`. Checked first
    /// because `instantid_realvisxl` is not an `is_candle_engine` txt2img id.
    InstantId,
    /// Generic SDXL OpenPose ControlNet, shared with the macOS route through registry id `sdxl`.
    SdxlControl,
    /// SDXL img2img / inpaint / outpaint edit (sc-5487).
    SdxlEdit,
    /// FLUX.2-klein reference / img2img edit (sc-5487).
    Flux2Edit,
    /// Qwen-Image-Edit reference / dual-latent edit (sc-5487).
    QwenEdit,
    /// SenseNova-U1 instruction/character edit through the registered Reference/MultiReference path.
    SenseNovaEdit,
    /// Z-Image img2img / edit (sc-6595).
    ZimageEdit,
    /// Mage-Flow Base/RL/Turbo instruction edit. Uses the generic registry stream, but requires
    /// source-first ordered multi-reference conditioning rather than the plain T2I request shape.
    MageEdit,
    /// Qwen-Image 2.1 reference / local editing (sc-24110). `qwen_image_2_1` is ALSO a candle
    /// txt2img id, so a conditioned request must divert here first or the generic arm would render
    /// it as plain text-to-image and silently drop every reference. Uses the same generic registry
    /// stream once the ordered conditioning list is resolved.
    QwenImage21Edit,
    /// Ordered FLUX.3 reference conditioning through the ordinary registered generator.
    Flux3Reference,
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
    /// Strict-pose control over an imported Krea DiT, one image per pose.
    KreaImportedControl,
    /// Off-Mac twin of [`ImageRoute::CheckpointPlan`] (sc-20634): a plan-backed user checkpoint
    /// resolved through the plan store and rendered by the registry-bound candle provider.
    CheckpointPlan,
    /// A generated full-fine-tune Mage transformer paired with the installed Base TE/VAE.
    MageFinetuned,
    /// Off-Mac twin of [`ImageRoute::SdxlImported`], loaded by candle from the fused checkpoint plus
    /// the three caller-staged SDXL components.
    SdxlImported,
    /// SDXL IP-Adapter-Plus reference conditioning (sc-5488).
    SdxlIpAdapter,
    /// Kolors IP-Adapter-Plus reference conditioning (sc-5488).
    KolorsIpAdapter,
    /// FLUX XLabs IP-Adapter reference conditioning (sc-5872).
    FluxIpAdapter,
    /// FLUX.1's IP-Adapter and strict-control providers cannot compose one another's input.
    FluxIpAdapterPoseReject,
    /// PuLID-FLUX face identity (sc-5492).
    Pulid,
    /// Qwen-Image strict-pose ControlNet (sc-5489).
    QwenControl,
    /// Kolors strict-pose ControlNet (sc-5489).
    KolorsControl,
    /// Kolors' bespoke IP and ControlNet providers have no shared composition.  This explicit
    /// reject prevents routing precedence from silently dropping either identity reference, pose,
    /// or an unsupported IP+PiD request.
    KolorsCompositeReject,
    /// Kolors source-image img2img/edit through the registered generator.
    KolorsEdit,
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
/// by an `else if …_control_available(…)` arm in [`prepare_candle_image_route`] BEFORE the generic
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
fn candle_conditioned_angle_set(request: &ImageRequest) -> bool {
    request.mode == "character_image"
        && advanced::flag(&request.advanced, "angleSet")
        && pose_entries(request).is_empty()
}

/// Seeds/prompts for a Candle reference-edit lane. Angle sets use the canonical 11 prompts with one
/// shared seed; every other request preserves the ordinary count/per-image seed contract.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_conditioned_edit_work(request: &ImageRequest) -> Vec<(i64, String)> {
    if candle_conditioned_angle_set(request) {
        let seed = resolve_seed(request, 0);
        return sceneworks_core::angle_kps::BUILTIN_ANGLE_SET_ORDER
            .iter()
            .map(|angle| (seed, augment_prompt_for_angle(&request.prompt, angle)))
            .collect();
    }
    (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect()
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn sensenova_edit_candle_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        return request
            .reference_asset_ids
            .iter()
            .take(5)
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

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_sensenova_candle_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Vec<Image>> {
    let ids = sensenova_edit_candle_reference_ids(request);
    if ids.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "SenseNova edit requires at least one reference image".to_owned(),
        ));
    }
    ids.iter()
        .map(|id| {
            let image =
                load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
            if request.fit_mode == "stretch" {
                Ok(image)
            } else {
                fit_engine_image(image, request.width, request.height, &request.fit_mode)
            }
        })
        .collect()
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_sensenova_candle_true_cfg(request: &ImageRequest) -> f32 {
    let default = if request.mode == "character_image" {
        1.5
    } else {
        1.0
    };
    request
        .advanced
        .get("trueCfgScale")
        .or_else(|| request.advanced.get("imageGuidanceScale"))
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default)
        .clamp(1.0, 10.0)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_kolors_edit_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    let Some(source_id) = request
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
        source_id,
        project_path,
    )?;
    let image = if request.fit_mode == "stretch" {
        source
    } else {
        fit_engine_image(source, request.width, request.height, &request.fit_mode)?
    };
    Ok(Some((
        image,
        advanced::f32_clamped(&request.advanced, "strength", 0.6, 0.05..=1.0),
    )))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn is_sensenova_candle_model(model: &str) -> bool {
    sceneworks_core::mlx_tier_completeness::SENSENOVA_MODELS.contains(&model)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn sensenova_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_sensenova_candle_model(&request.model)
        && matches!(request.mode.as_str(), "edit_image" | "character_image")
        && !sensenova_edit_candle_reference_ids(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn kolors_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "kolors"
        && request.mode == "edit_image"
        && request
            .source_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
        && request.reference_asset_id.is_none()
        && request.reference_asset_ids.is_empty()
        && request.mask_asset_id.is_none()
        && pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl CandleImageRoute {
    /// True only for routes whose actual load path applies the request's LoRA/LoKr stack. This is
    /// also the provenance contract: adding a route here requires the same change to thread the
    /// resolved stack into its inference constructor.
    fn applies_request_loras(self, request: &ImageRequest) -> bool {
        match self {
            CandleImageRoute::InstantId
            | CandleImageRoute::SdxlControl
            | CandleImageRoute::SdxlEdit
            | CandleImageRoute::SdxlIpAdapter
            | CandleImageRoute::KolorsIpAdapter
            | CandleImageRoute::Pulid
            | CandleImageRoute::QwenControl
            | CandleImageRoute::KolorsControl
            | CandleImageRoute::ZimageControl
            | CandleImageRoute::Flux2Control
            | CandleImageRoute::Flux1Control
            | CandleImageRoute::Flux2Edit
            | CandleImageRoute::ZimageComfyui
            | CandleImageRoute::Flux2Comfyui
            | CandleImageRoute::Bernini
            | CandleImageRoute::QwenEdit
            | CandleImageRoute::QwenImageComfyui
            | CandleImageRoute::ZimageEdit
            | CandleImageRoute::KolorsEdit
            | CandleImageRoute::MageEdit
            | CandleImageRoute::KreaEdit
            | CandleImageRoute::KreaTurboOnRaw
            | CandleImageRoute::KreaMultiPhase
            | CandleImageRoute::KreaImported
            | CandleImageRoute::KreaImportedControl
            | CandleImageRoute::SdxlImported
            | CandleImageRoute::KreaControl => true,
            // sc-24163: the Qwen-Image 2.1 reference-edit route is the SAME generic registry stream
            // as `CandleTxt2Img` (`generate_candle_stream`, which threads `resolve_adapters` into the
            // LoadSpec whenever the model supports adapters), so it consumes the stack exactly when
            // the plain arm does. Omitting it refused every LoRA/LoKr on a 2.1 edit (the CUDA
            // real-weight evidence run caught it).
            CandleImageRoute::CandleTxt2Img
            | CandleImageRoute::QwenImage21Edit
            | CandleImageRoute::Flux3Reference => {
                mlx_model(&request.model).is_some_and(|model| model.supports_adapters())
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
            | CandleImageRoute::SdxlControl
            | CandleImageRoute::KolorsControl
            | CandleImageRoute::ZimageControl
            | CandleImageRoute::Flux2Control
            | CandleImageRoute::Flux1Control
            | CandleImageRoute::KreaControl => pose_entries(request).len() as u32,
            CandleImageRoute::KreaImportedControl => pose_entries(request).len() as u32,
            CandleImageRoute::InstantId => instantid_image_count(request, settings),
            CandleImageRoute::QwenEdit
                if qwen_edit_candle::qwen_edit_candle_pose_count(request)
                    .is_some_and(|count| count > 0) =>
            {
                qwen_edit_candle::qwen_edit_candle_pose_count(request).unwrap_or_default() as u32
            }
            CandleImageRoute::Flux2Edit
            | CandleImageRoute::QwenEdit
            | CandleImageRoute::SenseNovaEdit
                if candle_conditioned_angle_set(request) =>
            {
                sceneworks_core::angle_kps::BUILTIN_ANGLE_SET_ORDER.len() as u32
            }
            // The generic lane also serves a native prompt batch (sc-25681): `prompts × count`.
            CandleImageRoute::CandleTxt2Img | CandleImageRoute::MossImageEdit => {
                request.image_count()
            }
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
            CandleImageRoute::SdxlControl => SDXL_CONTROL_ADAPTER_LABEL,
            CandleImageRoute::SdxlEdit => sdxl_edit_candle::SDXL_EDIT_CANDLE_ENGINE,
            CandleImageRoute::Flux2Edit => flux2_edit_candle::FLUX2_EDIT_CANDLE_ENGINE,
            CandleImageRoute::QwenEdit => qwen_edit_candle::QWEN_EDIT_CANDLE_ENGINE,
            CandleImageRoute::SenseNovaEdit | CandleImageRoute::KolorsEdit => {
                candle_adapter_label(&request.model)
            }
            CandleImageRoute::ZimageEdit => candle_adapter_label(&request.model),
            CandleImageRoute::KreaEdit => krea_edit_candle::KREA_EDIT_CANDLE_ENGINE,
            CandleImageRoute::KreaTurboOnRaw | CandleImageRoute::KreaMultiPhase => {
                mlx_model(&request.model)
                    .map(|model| model.adapter_label())
                    .unwrap_or(STUB_ADAPTER)
            }
            CandleImageRoute::KreaImported => KREA_IMPORTED_ENGINE,
            CandleImageRoute::KreaImportedControl => KREA_IMPORTED_ENGINE,
            CandleImageRoute::CheckpointPlan => CHECKPOINT_PLAN_ENGINE,
            CandleImageRoute::MageFinetuned => MAGE_FINETUNED_ENGINE,
            CandleImageRoute::SdxlImported => SDXL_IMPORTED_ENGINE,
            CandleImageRoute::SdxlIpAdapter => sdxl_ipadapter::SDXL_IPADAPTER_ENGINE,
            CandleImageRoute::KolorsIpAdapter => kolors_ipadapter::KOLORS_IPADAPTER_ENGINE,
            CandleImageRoute::FluxIpAdapter => flux_ipadapter::FLUX_IPADAPTER_ENGINE,
            CandleImageRoute::Pulid => pulid_candle::PULID_CANDLE_ENGINE,
            CandleImageRoute::QwenControl => qwen_control::QWEN_CONTROL_ENGINE,
            CandleImageRoute::KolorsControl => kolors_control::KOLORS_CONTROL_ENGINE,
            CandleImageRoute::ZimageControl => zimage_control::ZIMAGE_CTRL_ENGINE,
            CandleImageRoute::Flux2Control => flux2_control_candle::FLUX2_CONTROL_CANDLE_ENGINE,
            CandleImageRoute::Flux1Control => flux1_control_candle::FLUX1_CONTROL_CANDLE_ENGINE,
            CandleImageRoute::KreaControl => krea_control_candle::KREA_CONTROL_ENGINE,
            CandleImageRoute::PoseReject
            | CandleImageRoute::PoseControlBaseMissing
            | CandleImageRoute::FluxIpAdapterPoseReject
            | CandleImageRoute::KolorsCompositeReject => STUB_ADAPTER,
            CandleImageRoute::ZimageComfyui => zimage_comfyui_candle::ZIMAGE_COMFYUI_CANDLE_ENGINE,
            CandleImageRoute::QwenImageComfyui => qwen_comfyui_candle::QWEN_COMFYUI_CANDLE_ENGINE,
            CandleImageRoute::Flux2Comfyui => flux2_comfyui_candle::FLUX2_COMFYUI_CANDLE_ENGINE,
            CandleImageRoute::Bernini => CANDLE_BERNINI_IMAGE_ADAPTER,
            CandleImageRoute::MossImageEdit
            | CandleImageRoute::MageEdit
            | CandleImageRoute::QwenImage21Edit
            | CandleImageRoute::Flux3Reference
            | CandleImageRoute::CandleTxt2Img => candle_adapter_label(&request.model),
        }
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
enum PreparedCandleImageRoute {
    Plain(CandleImageRoute),
    CheckpointPlan(Box<PreparedCheckpointPlanSources>),
    KreaImported(Box<PreparedKreaImportedSources>),
    SdxlImported(Box<PreparedSdxlImportedSources>),
    ZimageComfyui(Box<zimage_comfyui_candle::ComfyuiZImagePaths>),
    QwenImageComfyui(Box<qwen_comfyui_candle::ComfyuiQwenPaths>),
    Flux2Comfyui(Box<flux2_comfyui_candle::ComfyuiFlux2Paths>),
    MageFinetuned(Box<PreparedMageFinetunedTransformer>),
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl PreparedCandleImageRoute {
    fn kind(&self) -> CandleImageRoute {
        match self {
            Self::Plain(route) => *route,
            Self::CheckpointPlan(_) => CandleImageRoute::CheckpointPlan,
            Self::KreaImported(_) => CandleImageRoute::KreaImported,
            Self::SdxlImported(_) => CandleImageRoute::SdxlImported,
            Self::ZimageComfyui(_) => CandleImageRoute::ZimageComfyui,
            Self::QwenImageComfyui(_) => CandleImageRoute::QwenImageComfyui,
            Self::Flux2Comfyui(_) => CandleImageRoute::Flux2Comfyui,
            Self::MageFinetuned(_) => CandleImageRoute::MageFinetuned,
        }
    }
}

/// Run the candle image dispatch predicate ladder ONCE and return the [`CandleImageRoute`] (or `None`
/// when candle is disabled / no candle engine matches → the job stubs). Mirrors the historical inline
/// `else if settings.backend_candle_enabled && <predicate>` ladder EXACTLY — same predicate order,
/// same `backend_candle_enabled` gating, same handler per family — so routing is byte-identical
/// (sc-8828). Pure decision: no I/O, no generation.
// One `…_available` flag per prepared bundle, so the count tracks the number of prepared candle
// lanes rather than any avoidable grouping.
#[allow(clippy::too_many_arguments)]
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_image_route_with_prepared_availability(
    request: &ImageRequest,
    settings: &Settings,
    checkpoint_plan_available: bool,
    imported_available: bool,
    sdxl_imported_available: bool,
    zimage_comfyui_available: bool,
    qwen_comfyui_available: bool,
    flux2_comfyui_available: bool,
    mage_finetuned_available: bool,
) -> Option<CandleImageRoute> {
    if !settings.backend_candle_enabled {
        return None;
    }
    // Order matches the historical ladder: the edit / reference / identity / control lanes are all
    // checked BEFORE the generic `is_candle_engine` txt2img arm (they share candle txt2img model ids, so
    // without diverting first they'd be silently rendered as plain txt2img, dropping the source / poses).
    if sceneworks_core::moss_image::is_moss(&request.model) && request.mode == "edit_image" {
        Some(CandleImageRoute::MossImageEdit)
    } else if flux1_ipadapter_pose_combination(request) {
        Some(CandleImageRoute::FluxIpAdapterPoseReject)
    } else if instantid_available(request, settings) {
        Some(CandleImageRoute::InstantId)
    } else if sdxl_control_candidate(request) {
        // Same precedence as the core router: InstantID stays isolated, then generic SDXL control
        // claims every material pose carrier before edit/IP/generic routes can discard it.
        Some(CandleImageRoute::SdxlControl)
    } else if sdxl_edit_candle_available(request, settings) {
        Some(CandleImageRoute::SdxlEdit)
    } else if matches!(
        request.model.as_str(),
        "flux2_dev"
            | "qwen_image_edit"
            | "qwen_image_edit_2509"
            | "qwen_image_edit_2511"
            | "qwen_image_edit_2511_lightning"
    ) && candle_conditioned_pose_requires_reject(request)
    {
        // Defense in depth for direct worker calls: the scheduler rejects malformed pose carriers
        // and valid pose sets paired with unsupported conditioned modes. The worker must reject the
        // same shapes rather than letting typed parsing erase them or falling through to the stub.
        Some(CandleImageRoute::PoseReject)
    } else if flux2_control_candle_available(request, settings) {
        // Character Studio FLUX.2 pose payloads also carry a reference image. The pose/control lane
        // must win before reference edit so the skeleton is never silently dropped.
        Some(CandleImageRoute::Flux2Control)
    } else if flux2_edit_candle_available(request, settings) {
        Some(CandleImageRoute::Flux2Edit)
    } else if qwen_edit_candle_available(request, settings) {
        Some(CandleImageRoute::QwenEdit)
    } else if sensenova_edit_candle_available(request, settings) {
        Some(CandleImageRoute::SenseNovaEdit)
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
    } else if request
        .flux3_reference_request
        .as_ref()
        .is_some_and(|authored| {
            !authored
                .as_ref()
                .is_ok_and(|references| references.asset_ids.is_empty())
        })
    {
        // Also claim malformed stored reference jobs so the single trusted resolver refuses them
        // before load, rather than a missing/invalid slot falling through to a T2I or stub path.
        Some(CandleImageRoute::Flux3Reference)
    } else if is_qwen_image_2_1_edit(request) {
        // Qwen-Image 2.1 reference / local editing (sc-24110), named by the resolver for the same
        // reason Mage Edit is: the id is a candle txt2img id, so a conditioned request that fell
        // through would be rendered as plain T2I with every reference silently dropped. The core
        // router's `CandleImageLane::QwenImage21Edit` claims exactly these shapes.
        Some(CandleImageRoute::QwenImage21Edit)
    } else if request.model == "kolors"
        && ((non_empty(&request.reference_asset_id) && !pose_entries(request).is_empty())
            || (non_empty(&request.reference_asset_id)
                && advanced::flag(&request.advanced, "usePid")))
    {
        // Kolors IP-Adapter is its own provider (not a base overlay), and ControlNet owns a
        // separate pose provider. Neither implementation composes IP+Control or IP+PiD, so this
        // must fail before either precedence arm can discard an input.
        Some(CandleImageRoute::KolorsCompositeReject)
    } else if zimage_identity_candle_available(request, settings) {
        // The registered Z-Image Turbo generator owns both Reference conditioning and adapters.
        Some(CandleImageRoute::CandleTxt2Img)
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
    } else if kolors_edit_candle_available(request, settings) {
        Some(CandleImageRoute::KolorsEdit)
    } else if zimage_control_available(request, settings) {
        Some(CandleImageRoute::ZimageControl)
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
    } else if checkpoint_plan_available {
        // A plan-backed user checkpoint (epic 20398, sc-20634): claimed before every bespoke imported
        // lane; the plan store is the route's sole source of truth.
        Some(CandleImageRoute::CheckpointPlan)
    } else if krea_imported_control_available(request, settings) {
        // An imported single-file Krea 2 checkpoint + a strict-pose set: the pose control branch rides
        // the file-loaded imported DiT. Checked BEFORE the plain imported arm so a pose set renders one
        // pose-locked image per pose instead of falling into per-image t2i (dropping the poses).
        Some(CandleImageRoute::KreaImportedControl)
    } else if imported_available {
        // Imported/user Krea 2 single-file generation: external IDs are absent from
        // `is_candle_engine`, so this bespoke text-to-image/img2img/edit route must claim them before
        // the generic/external fall-through.
        Some(CandleImageRoute::KreaImported)
    } else if mage_finetuned_available {
        // A fine-tuned Mage-Flow base (sc-15036): its id is in no `MODEL_TABLE` / `is_candle_engine`
        // arm, so this route is what dispatches it to real candle generation at all.
        Some(CandleImageRoute::MageFinetuned)
    } else if sdxl_imported_available {
        Some(CandleImageRoute::SdxlImported)
    } else if zimage_comfyui_available {
        // In-place ComfyUI Z-Image base (sc-10668): an `external_base_*` id, so it matches no
        // `is_candle_engine` arm below — route it here off the forwarded `modelManifestEntry`.
        Some(CandleImageRoute::ZimageComfyui)
    } else if qwen_comfyui_available {
        // In-place ComfyUI Qwen-Image base (sc-10670): sibling of the Z-Image comfyui lane — an
        // `external_base_*` id routed off the forwarded row (family=="qwen-image", usable).
        Some(CandleImageRoute::QwenImageComfyui)
    } else if flux2_comfyui_available {
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
        //  - No candle pose lane → the sc-5968 no-silent-T2I `PoseReject`.
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

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_conditioned_pose_requires_reject(request: &ImageRequest) -> bool {
    let pose_count = match request.advanced.get("poses") {
        None | Some(Value::Null) => return false,
        Some(Value::Array(poses))
            if poses.len() <= sceneworks_core::image_request::MAX_JOB_POSES
                && poses.iter().all(Value::is_object) =>
        {
            poses.len()
        }
        Some(_) => return true,
    };
    if pose_count == 0 {
        return false;
    }
    match request.model.as_str() {
        // FLUX.2 pose ControlNet serves every non-edit mode; instruction edit plus a pose has no
        // composition contract and must reject rather than fall through after both specialists say no.
        "flux2_dev" => request.mode == "edit_image",
        // Qwen Edit's exact pose recipe is Character Studio with one identity reference; every other
        // valid pose-bearing mode/reference shape is unsupported and must likewise reject explicitly.
        "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning" => !qwen_edit_candle::qwen_edit_candle_mode(request),
        _ => false,
    }
}

/// Test-facing pure route probe. Production uses [`prepare_candle_image_route`] so imported File
/// tokens remain owned across admission and dispatch.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle", test))]
fn resolve_candle_image_route(
    request: &ImageRequest,
    settings: &Settings,
) -> Option<CandleImageRoute> {
    resolve_candle_image_route_with_prepared_availability(
        request,
        settings,
        // See the macOS twin: a typed refusal must surface, not read as "route unavailable".
        checkpoint_plan_available_or_panic(request, settings),
        krea_imported_available(request, settings),
        sdxl_imported_available(request, settings),
        zimage_comfyui_candle::zimage_comfyui_available(request, settings),
        qwen_comfyui_candle::qwen_comfyui_available(request, settings),
        flux2_comfyui_candle::flux2_comfyui_available(request, settings),
        mage_finetuned_available(request, settings),
    )
}

/// Resolve a production candle route while retaining each payload-selected File token through every
/// later await. Only the route that wins the existing predicate order consumes its prepared bundle.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn prepare_candle_image_route(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedCandleImageRoute>> {
    if !settings.backend_candle_enabled {
        // This early-out returns BEFORE `prepare_checkpoint_plan_sources` runs, so the
        // `into_unclaimed_refusal` re-raise below never fires and a plan-backed entry falls all the
        // way to `generate_stub_stream` — the job COMPLETES with procedural output under the user's
        // imported checkpoint id (sc-20651). Refuse in the same typed shape instead. The macOS twin
        // `prepare_image_route` has no such early-out, so it needs no counterpart.
        if let Some(checkpoint_id) = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(
            &request.model_manifest_entry,
        ) {
            return Err(WorkerError::InvalidPayload(format!(
                "[checkpoint-plan:no-adapter-binding] checkpoint {checkpoint_id:?}: the candle \
                 backend is disabled on this worker, and a plan-backed entry never renders \
                 procedural stub output"
            )));
        }
        return Ok(None);
    }
    // The plan-driven route is offered the request first (sc-20634). Per-request claim, exactly as
    // in the macOS twin: it takes the shapes it serves, the family's bespoke lane keeps the rest,
    // and the decline's retained refusal is re-raised below when no lane claims it (sc-20636).
    let checkpoint_plan = prepare_checkpoint_plan_sources(request, settings)?;
    // A pose-bearing imported checkpoint is claimed by the `KreaImportedControl` arm, which resolves
    // its own sources; skip pinning a second File token for the plain imported bundle in that case
    // (mirrors the macOS `prepare_image_route` guard).
    let imported = if krea_imported_control_available(request, settings) {
        None
    } else {
        prepare_krea_imported_sources(request, settings)?
    };
    let sdxl = prepare_sdxl_imported_sources(request, settings)?;
    let zimage = zimage_comfyui_candle::prepare_zimage_comfyui_sources(request, settings)?;
    let qwen = qwen_comfyui_candle::prepare_qwen_comfyui_sources(request, settings)?;
    let flux2 = flux2_comfyui_candle::prepare_flux2_comfyui_sources(request, settings)?;
    let mage_finetuned = prepare_mage_finetuned_transformer(request, settings)?;
    let Some(kind) = resolve_candle_image_route_with_prepared_availability(
        request,
        settings,
        checkpoint_plan.is_available(),
        imported.is_some(),
        sdxl.is_some(),
        zimage.is_some(),
        qwen.is_some(),
        flux2.is_some(),
        mage_finetuned.is_some(),
    ) else {
        // See the macOS twin: zero lanes claimed a plan-backed entry means the bespoke lane the
        // plan route deferred to does not exist for this shape. Raise the retained typed refusal
        // rather than dropping through to `generate_stub_stream`'s procedural output (sc-20636).
        checkpoint_plan.into_unclaimed_refusal(request)?;
        return Ok(None);
    };
    Ok(Some(match kind {
        CandleImageRoute::CheckpointPlan => PreparedCandleImageRoute::CheckpointPlan(Box::new(
            checkpoint_plan
                .into_sources()
                .expect("prepared checkpoint plan route lost its sources"),
        )),
        CandleImageRoute::KreaImported => PreparedCandleImageRoute::KreaImported(Box::new(
            imported.expect("prepared imported route lost its sources"),
        )),
        CandleImageRoute::SdxlImported => PreparedCandleImageRoute::SdxlImported(Box::new(
            sdxl.expect("prepared SDXL imported route lost its sources"),
        )),
        CandleImageRoute::ZimageComfyui => PreparedCandleImageRoute::ZimageComfyui(Box::new(
            zimage.expect("prepared Z-Image route lost its sources"),
        )),
        CandleImageRoute::QwenImageComfyui => PreparedCandleImageRoute::QwenImageComfyui(Box::new(
            qwen.expect("prepared Qwen route lost its sources"),
        )),
        CandleImageRoute::Flux2Comfyui => PreparedCandleImageRoute::Flux2Comfyui(Box::new(
            flux2.expect("prepared FLUX.2 route lost its sources"),
        )),
        CandleImageRoute::MageFinetuned => PreparedCandleImageRoute::MageFinetuned(Box::new(
            mage_finetuned.expect("prepared Mage fine-tuned route lost its transformer"),
        )),
        route => PreparedCandleImageRoute::Plain(route),
    }))
}

/// The `!backend_candle_enabled` early-out of [`prepare_candle_image_route`] returns before
/// `prepare_checkpoint_plan_sources` runs, so `into_unclaimed_refusal` never fires and a plan-backed
/// entry falls through to `generate_stub_stream` — the job COMPLETES with procedural output under
/// the user's imported checkpoint id (sc-20651). The refusal is asserted at that seam, and the
/// control proves it is keyed on the PLAN and not on the disabled backend.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/candle_disabled_plan_backed_refusal_tests.rs"]
mod candle_disabled_plan_backed_refusal_tests;

/// Couples a route-owned File source bundle to the immutable generation plan without widening the
/// uniform stream-handler signatures. The bundle is moved into the handler exactly once.
#[cfg(native_lane)]
struct PreparedFileDispatch<'a, Sources> {
    plan: &'a ImagePlan,
    sources: Sources,
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
#[cfg(native_lane)]
fn model_repo(request: &ImageRequest, model: &ResolvedModel) -> String {
    model_repo_with_default(request, model.default_repo())
}

#[cfg(native_lane)]
fn model_repo_with_default(request: &ImageRequest, default_repo: &str) -> String {
    crate::payload::trimmed_string(request.model_manifest_entry.get("repo"))
        .unwrap_or(default_repo)
        .to_owned()
}

/// Receipt variant selected by this request.  Prefer an explicit quant request; otherwise use the
/// manifest's default selectable download (or its only selectable download).  Co-requisites are
/// intentionally excluded because each repo resolves its own receipt independently.
#[cfg(native_lane)]
fn requested_receipt_variant(request: &ImageRequest) -> Option<String> {
    if let Some(bits) = request.quant_bits() {
        return Some(crate::payload::tier_for_bits(Some(bits), "q8").to_owned());
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
#[cfg(native_lane)]
const IDEOGRAM_BF16_REPO: &str = "SceneWorks/ideogram-4";

/// Exact snapshots used by the on-demand non-default tier fetches.  The API replaces a submitted
/// `modelManifestEntry` with its resolved builtin/user manifest entry before enqueueing, so a repo
/// different from the engine default is an intentional configured override.  First-party turnkeys
/// never follow a mutable ref; configured override repos retain `main` because SceneWorks cannot know
/// their release commit in advance.
#[cfg(native_lane)]
const BOOGU_MLX_TURNKEY_REVISION: &str = "a459e614d408bfdf57089c32cc3da706f5a017de";
#[cfg(native_lane)]
const IDEOGRAM_MLX_TURNKEY_REVISION: &str = "a3095855b8819dc0d6b067cb1354aaa7da189ff8";
#[cfg(native_lane)]
pub(super) const ZIMAGE_MLX_TURNKEY_REPO: &str = "SceneWorks/z-image-mlx";
#[cfg(native_lane)]
pub(super) const ZIMAGE_MLX_TURNKEY_REVISION: &str = "c74f74c2ad193294fc9ff3f8a5be71daa00d22ab";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const FLUX1_SCHNELL_MLX_TURNKEY_REVISION: &str = "bba3ae01dfd94089f173c05edd4e1a4c551f2599";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const FLUX1_DEV_MLX_TURNKEY_REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const FLUX2_DEV_MLX_TURNKEY_REVISION: &str = "2868b1461b2b6e6e05d84e52534df3632b4c7d5d";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const FLUX2_KLEIN_9B_MLX_TURNKEY_REVISION: &str = "1902693279fcfb828919370dfac2b8922d99499a";
#[cfg(native_lane)]
fn turnkey_tier_revision<'a>(repo: &str, default_repo: &str, default_revision: &'a str) -> &'a str {
    if repo == default_repo {
        default_revision
    } else {
        "main"
    }
}

#[cfg(native_lane)]
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
    let root = crate::model_jobs::huggingface_pinned_snapshot_dir(data_dir, repo, revision)?;
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
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SANA_CANDLE_DIFFUSERS_REVISION: &str = "ac0da2ff55fbe434795be0dce883042e4d49e2fc";

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
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SANA_SPRINT_CANDLE_DIFFUSERS_REVISION: &str = "19683c58b7ea290e55cedd8950ae1d86ada7ef96";

#[cfg(native_lane)]
// Keep the explicit optional branch: the manifest audit recognizes this shape and proves every
// production modelPath reader preserves the normal repo-resolution fallback.
#[allow(clippy::question_mark)]
fn model_path_override(request: &ImageRequest) -> Option<String> {
    let Some(raw_path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(crate::payload::trimmed_value_str)
    else {
        return None;
    };
    Some(raw_path.to_owned())
}

/// The confined directory a job's `modelPath` (the request's `advanced.modelPath`, else the entry's
/// injected one — [`model_path_override`]) names, through the app-managed resolver that refuses a
/// path outside the data dir. The ONE resolution the load ([`resolve_weights_dir`]) and the
/// failed-load diagnosis (`converted_install::explain_failed_load`) share, so a diagnosis judges —
/// and marks — exactly the bundle the load read. `Ok(None)` when the job names no `modelPath`.
#[cfg(native_lane)]
pub(crate) fn resolve_model_path_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    model_path_override(request)
        .map(|path| resolve_app_managed_model_dir(settings, &path, "Image modelPath"))
        .transpose()
}

/// Typed, actionable preflight for a **convert-at-install** model whose conversion has not produced
/// a local artifact yet (sc-20529).
///
/// Such a model is declared with `mlx.requiresConversion` + `mlx.converter`, and the API injects the
/// converted dir as `modelPath` (`inject_converted_model_path`) once, and only once, the conversion
/// has actually run. So a request with NO `modelPath` means there is no converted artifact — and the
/// source repo it would otherwise fall back to is a bare single-file transformer
/// with no `text_encoder/` / `vae/` / `tokenizer/`. Loading it produced a raw engine-level
/// `"<label>: snapshot missing the text_encoder/ component directory (at …)"` from candle-gen's
/// shared `loader.rs`, which names an internal path and tells the user nothing about what to do.
///
/// This turns that class into one typed [`WorkerError::InvalidPayload`] naming BOTH remedies: run
/// the conversion, and install the base model whose text encoder / VAE / tokenizer the conversion
/// borrows (`mlx.convertBaseRepo` + `mlx.convertBaseSubdir` — for `flux2_klein_9b_true_v2`,
/// `SceneWorks/flux2-klein-9b-mlx` `bf16`). Keyed on the MANIFEST CONTRACT, never on a model id, so
/// every present and future convert-at-install model inherits it.
///
/// **Platform-scoped on purpose.** The gate is [`convert_artifact_required_here`], not a bare
/// `requiresConversion` read: Anima (`anima_base` / `anima_aesthetic` / `anima_turbo`) declares
/// `requiresConversion` and ships windows/linux downloads, but its `anima_quant` converter is
/// macOS-only — off-Mac those models legitimately resolve the raw `circlestone-labs/Anima`
/// `split_files/` tree below with no `modelPath` at all. Refusing them here would break three
/// models that render fine today.
///
/// **Self-contained on purpose.** The `modelPath` test lives INSIDE this function rather than in the
/// caller, so the same verdict can be reached from two seams that see the request at different
/// points: [`resolve_weights_dir`] (which reaches the loader) and [`candle_weights_gap`] (the
/// stub fall-through, which never resolves weights at all — see its docs for why that second seam
/// is required).
#[cfg(native_lane)]
fn unconverted_model_preflight(request: &ImageRequest) -> WorkerResult<()> {
    // An injected `modelPath` IS the converted artifact — nothing outstanding.
    if model_path_override(request).is_some() {
        return Ok(());
    }
    let Some(mlx) = request.model_manifest_entry.get("mlx") else {
        return Ok(());
    };
    if mlx.get("requiresConversion").and_then(Value::as_bool) != Some(true) {
        return Ok(());
    }
    let converter = mlx
        .get("converter")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if !sceneworks_core::jobs_store::convert_artifact_required_here(converter) {
        return Ok(());
    }
    let name = crate::payload::trimmed_string(request.model_manifest_entry.get("name"))
        .unwrap_or(request.model.as_str());
    // The base whose text encoder / VAE / tokenizer the conversion borrows. Named explicitly when
    // the manifest declares it so the user knows the exact second install to make; the sentence is
    // dropped rather than faked when it does not (a dir-sourced converter borrows nothing).
    let base = crate::payload::trimmed_string(mlx.get("convertBaseRepo"))
        .map(|repo| {
            let tier = crate::payload::trimmed_string(mlx.get("convertBaseSubdir"));
            match tier {
                Some(tier) => format!(
                    " It also requires the base model {repo} ({tier}) to be installed — the \
                     conversion borrows that base's text encoder, VAE, and tokenizer, which the \
                     single-file checkpoint does not contain."
                ),
                None => format!(
                    " It also requires the base model {repo} to be installed — the conversion \
                     borrows that base's text encoder, VAE, and tokenizer, which the single-file \
                     checkpoint does not contain."
                ),
            }
        })
        .unwrap_or_default();
    Err(WorkerError::InvalidPayload(format!(
        "{name} has not been converted yet — convert it from the Model Manager before \
         generating. It downloads as a transformer-only checkpoint, so there is nothing to load \
         until the conversion assembles a complete local model.{base}"
    )))
}

/// Typed, actionable preflight for an injected converted artifact that would not load (sc-24216).
///
/// The twin of [`unconverted_model_preflight`] for the case where the conversion DID run but its
/// output is no longer what the engine accepts — truncated, edited, or written by an older build.
/// For a converter with an engine-owned pin contract
/// ([`sceneworks_core::converted_artifacts::CONTRACT_CONVERTERS`], e.g. `supra2_img`) the engine's
/// own CHEAP check runs here (manifest, pin contract, file sizes, and a verification failure a
/// previous failed load recorded), so a damaged bundle is refused before any load with a message
/// naming the remedy instead of the engine's raw refusal. Same-size corruption is caught by the
/// engine's own load-time digests and explained by [`crate::converted_install::explain_failed_load`]
/// — never by hashing here, so a healthy install is hashed once per load. Every other converter
/// passes untouched.
#[cfg(native_lane)]
fn converted_artifact_preflight(request: &ImageRequest, dir: &std::path::Path) -> WorkerResult<()> {
    let Some(mlx) = request.model_manifest_entry.get("mlx") else {
        return Ok(());
    };
    if mlx.get("requiresConversion").and_then(Value::as_bool) != Some(true) {
        return Ok(());
    }
    let converter = mlx
        .get("converter")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    let Some(problem) = sceneworks_core::converted_artifacts::converted_artifact_problem(
        converter,
        dir,
        sceneworks_core::converted_artifacts::ArtifactCheck::Sizes,
    ) else {
        return Ok(());
    };
    Err(WorkerError::InvalidPayload(
        crate::converted_install::converted_files_message(
            crate::converted_install::model_display_name(
                &request.model_manifest_entry,
                &request.model,
            ),
            &problem,
        ),
    ))
}

/// Fail-loud gate for the CANDLE stub fallback (sc-20529) — the off-Mac twin of [`mlx_weights_gap`],
/// consulted by `run_image_generate_job` immediately before it would call `generate_stub_stream`.
///
/// [`unconverted_model_preflight`] fires from [`resolve_weights_dir`], which only runs once a route
/// has CLAIMED the job. That is enough for `text_to_image` / `image_to_image` / `character_image` on
/// an unconverted convert-at-install model: those modes reach the ladder's terminal
/// `is_candle_engine(&request.model) && request.mode != "edit_image"` arm → `CandleTxt2Img` →
/// `generate_candle_stream`, which resolves weights and propagates the typed error.
///
/// `edit_image` does NOT. `flux2_edit_candle_available` (and every sibling `…_available` weight
/// gate) collapses a resolver `Err` to `false`, and every terminal arm of the ladder excludes
/// `mode == "edit_image"` — so an unconverted `edit_image` job is claimed by NO route,
/// `prepare_candle_image_route` returns `None`, and the job fell through to `generate_stub_stream`
/// and COMPLETED with a procedural gradient. A silent stub in place of a typed refusal is exactly
/// the no-silent-fallback posture sc-5099 forbids, and it is worse than the raw loader error it
/// replaced: the user gets a plausible-looking asset instead of an error.
///
/// Returning the preflight's own message keeps ONE wording for the class — the dispatch seam and
/// the loader seam cannot drift into two different explanations of the same missing conversion.
/// `None` for every other job, so a model that legitimately stubs (test ids, unported families)
/// stubs exactly as before.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_weights_gap(request: &ImageRequest) -> Option<String> {
    match unconverted_model_preflight(request) {
        Ok(()) => None,
        Err(WorkerError::InvalidPayload(message)) => Some(message),
        // The preflight raises exactly one class; anything else would be a new variant whose
        // wording has not been reviewed for this seam, so fall through to the existing behaviour
        // rather than surfacing it as a stub-gap message.
        Err(_) => None,
    }
}

/// Resolve the weights snapshot directory: an explicit `modelPath` dir wins, else the
/// HuggingFace cache snapshot for the model repo. `None` when the model is not a known
/// engine family or its snapshot is absent. Available on the candle lane too (sc-5501): the
/// off-Mac SenseNova-U1 VQA / interleave handlers resolve their snapshot through it.
#[cfg(native_lane)]
pub(crate) fn resolve_weights_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    // Canonical FLUX3 identities must be proved by the worker-owned exact-revision receipt.
    // Never let an arbitrary cache HEAD, a distilled snapshot, or a modelPath override impersonate
    // the training base an adapter declares. Other families retain their historical resolution.
    let flux3_identity = match request.model.as_str() {
        "flux3_klein" => Some((
            "black-forest-labs/flux-3-image-klein",
            "8454017f9ebaada30c998fccd9f67a80c7580f68",
        )),
        "flux3_klein_base" => Some((
            "black-forest-labs/flux-3-image-klein-base",
            "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4",
        )),
        _ => None,
    };
    if let Some((repo, revision)) = flux3_identity {
        if model_repo_with_default(request, repo) != repo {
            return Err(WorkerError::InvalidPayload(format!(
                "{} requires pinned {repo}@{revision}; a different repo is not this checkpoint",
                request.model
            )));
        }
        let explicit = model_path_override(request)
            .map(|path| resolve_app_managed_model_dir(settings, &path, "Image modelPath"))
            .transpose()?;
        return Ok(Some(resolve_pinned_native_package(
            &settings.data_dir,
            repo,
            revision,
            &request.model,
            explicit.as_deref(),
        )?));
    }
    if let Some(dir) = resolve_model_path_dir(request, settings)? {
        converted_artifact_preflight(request, &dir)?;
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
    // No `modelPath` on a convert-at-install model means the conversion has not produced its local
    // artifact. Refuse HERE with an actionable error rather than falling through to the raw
    // single-file source snapshot, which loads no further than candle-gen's "snapshot missing the
    // text_encoder/ component directory" (sc-20529).
    unconverted_model_preflight(request)?;
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
        #[cfg(native_lane)]
        {
            let wants_bf16 = request.quant_bits()
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
        return Ok(huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            SANA_CANDLE_DIFFUSERS_REPO,
            SANA_CANDLE_DIFFUSERS_REVISION,
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
        return Ok(huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            SANA_SPRINT_CANDLE_DIFFUSERS_REPO,
            SANA_SPRINT_CANDLE_DIFFUSERS_REVISION,
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
    // Qwen-Image 2.1 (sc-24112) — a SPLIT-REPO tier layout, and the reason it cannot use
    // `standard_tier_subdir`: that resolver descends into `<root>/<tier>/` of ONE repo, and 2.1's
    // three tiers do not live in one repo. bf16 IS the released upstream snapshot at its own root
    // (the converter refuses to emit a `bf16/` copy, and re-hosting unmodified weights would be a
    // §3 redistribution SceneWorks does not need to make), while q8 and q4 are `q8/`/`q4/` subdirs
    // of the SceneWorks re-host. So the tier picks the REPO first, then the subdir.
    //
    // Exactly the shape the Ideogram-4 branch above already has, with the halves swapped: there
    // the packed tiers are the turnkey and bf16 lives in a separate shared repo; here bf16 is the
    // upstream tree and the packed pair is the re-host. Both fall back to the resolved default
    // rather than half-loading when the requested tier is not on disk.
    //
    // Repo and revision come from the MANIFEST's own download rows, not from consts here: the
    // catalog is the pin authority (F-029), and reading it means the terminal story's revision pin
    // reaches the loader by editing one place. A row still carrying the null-SHA placeholder
    // resolves to nothing, so a pending tier falls back to the installed default instead of
    // pointing the loader at a snapshot directory that cannot exist.
    #[cfg(native_lane)]
    // Both 2.1 ids (base and its distilled Turbo sibling, sc-25907) are split-repo tier families:
    // the upstream snapshot is bf16 and the SceneWorks re-host carries q8/q4.
    if sceneworks_core::jobs_store::is_qwen_image_2_1_model(&request.model) {
        if let Some((tier_dir, _tier)) = qwen_image_2_1_tier_dir(settings, request)? {
            return Ok(Some(tier_dir));
        }
        return Ok(snapshot);
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

/// The installed tier directory for a `qwen_image_2_1` request AND the tier it is, or `None` to
/// fall back to the model's default snapshot (sc-24112).
///
/// The identity rides alongside the path because the path alone cannot carry it: bf16 is the
/// upstream snapshot ROOT, whose basename is a commit SHA rather than a tier token, so
/// [`tier_key_from_resolved_dir`] answers `None` for it. Every consumer that prices or loads the
/// resolved directory (the Candle VRAM gate via [`gate_tier_key`], the Candle load quant, the MLX
/// candidate quant and reconcile) reads it back through [`tier_key_for_resolved_dir`], which maps
/// that root to `bf16` from the same catalog rows this resolver descends. The two agree by
/// construction and `resolved_tier_identity_round_trips_through_the_directory` pins it.
///
/// Resolution:
///
/// * an EXPLICIT `advanced.mlxQuantize` (`<= 0` or `>= 9` ⇒ bf16, `1..=4` ⇒ q4, `5..=8` ⇒ q8) is
///   honoured exactly
///   or refused: a pick whose tier is not installed is a typed error naming the tier, never a
///   silent substitution of another one (the FLUX.1 Candle precedent,
///   `candle_flux1_packed_requested_tier`). The Studio only sends a tier it resolved from the
///   installed set, so this fires for a stale replay or a direct API call, not for the picker;
/// * with NO selection, the catalog default (`mlx.quantize`, q8) first, then the remaining
///   installed tiers DENSEST FIRST (bf16 → q8 → q4), so a partial install never lands on the washed
///   q4 while a denser tier is on disk.
///
/// A tier is "installed" when its resolved directory holds a loadable `transformer/`. With nothing
/// installed at all an UNSELECTED request answers `None`, so the caller's ordinary "install the
/// model" path owns that error; an EXPLICIT pick with nothing installed is the install error itself
/// (sc-24114) — answering `None` there let the caller fall back to the default snapshot directory,
/// which loads a bf16 root that may not hold a `transformer/` at all instead of saying "install".
#[cfg(native_lane)]
fn qwen_image_2_1_tier_dir(
    settings: &Settings,
    request: &ImageRequest,
) -> WorkerResult<Option<(PathBuf, &'static str)>> {
    const TIERS: [&str; 3] = ["bf16", "q8", "q4"];
    let qwen_checkpoint_tier = |bits: i64| -> &'static str {
        match bits {
            // A dense bit width (16 = bf16, 32 = f32) is the unquantized tier, never q8 (sc-24114).
            bits if bits <= 0 || bits >= 9 => "bf16",
            bits if bits <= 4 => "q4",
            _ => "q8",
        }
    };
    let explicit = request
        .advanced
        .get("mlxQuantize")
        .filter(|value| !value.is_null())
        .map(|value| {
            crate::payload::quant_int(value)
                .map(qwen_checkpoint_tier)
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "{} advanced.mlxQuantize must be an integer bit count, got {value}",
                        request.model
                    ))
                })
        })
        .transpose()?;
    let installed = |tier: &'static str| -> Option<(PathBuf, &'static str)> {
        let dir = qwen_image_2_1_declared_tier_dir(settings, request, tier)?;
        qwen_image_2_1_has_loadable_transformer(&dir).then_some((dir, tier))
    };
    if let Some(tier) = explicit {
        if let Some(resolved) = installed(tier) {
            return Ok(Some(resolved));
        }
        if TIERS.into_iter().all(|other| installed(other).is_none()) {
            return Err(WorkerError::InvalidPayload(format!(
                "{} is not installed on this machine (no tier is on disk, so tier {tier} cannot \
                 load). Install it in Model Manager, then retry.",
                request.model
            )));
        }
        return Err(WorkerError::InvalidPayload(format!(
            "{} tier {tier} is not installed; install it or pick an installed tier \
             (an explicit tier selection is never substituted with a different tier)",
            request.model
        )));
    }
    let default = request
        .model_manifest_entry
        .get("mlx")
        .and_then(|mlx| mlx.get("quantize"))
        .and_then(Value::as_i64)
        .map_or("q8", qwen_checkpoint_tier);
    Ok(installed(default).or_else(|| {
        TIERS
            .into_iter()
            .filter(|tier| *tier != default)
            .find_map(installed)
    }))
}

/// The "has a loadable backbone" probe the shared resolver applies, spelled for BOTH layouts this
/// family ships (sc-24114):
///
/// * the upstream diffusers bf16 snapshot — a sharded `transformer/` with its index, or one
///   `diffusion_pytorch_model.safetensors`;
/// * a packed q8/q4 tier as the engine's `convert::prequantize_turnkey` writes it — ONE
///   `transformer/model.safetensors` (the sharded index is deliberately absent) beside a
///   `config.json` carrying the `quantization` marker the engine's packed-detect reads.
#[cfg(native_lane)]
fn qwen_image_2_1_has_loadable_transformer(dir: &Path) -> bool {
    let transformer = dir.join("transformer");
    let dense = transformer
        .join("diffusion_pytorch_model.safetensors.index.json")
        .is_file()
        || transformer
            .join("diffusion_pytorch_model.safetensors")
            .is_file();
    let packed = transformer.join("model.safetensors").is_file()
        && std::fs::read_to_string(transformer.join("config.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .is_some_and(|config| config.get("quantization").is_some_and(Value::is_object));
    dense || packed
}

/// The tier a resolved directory IS, reading the request's catalog entry when the basename is not
/// a tier token (sc-24112).
///
/// Only a split-repo family reaches the second arm: `qwen_image_2_1`'s bf16 tier is a whole upstream
/// snapshot ROOT (`files` empty), resolved to `<library>/models--<org>--<name>/snapshots/<rev>`.
/// That directory is matched back to its catalog row by repo AND revision — the same pin
/// [`qwen_image_2_1_declared_tier_dir`] descends — so the gate, the Candle load quant and the MLX
/// reconcile see `bf16` instead of falling back to the request/manifest default (`q8`) and asking a
/// dense root for a packed load.
///
/// Scoped to [`SPLIT_REPO_TIER_MODELS`] on purpose: other catalog rows are also tier-tagged
/// whole-repo roots (the Candle SANA and Wan diffusers snapshots), and relabelling THEIR resolved
/// directories would move their gate pricing and load quant, which nothing here has validated.
#[cfg(native_lane)]
pub(crate) fn tier_key_for_resolved_dir(
    manifest_entry: &JsonObject,
    dir: &Path,
) -> Option<&'static str> {
    tier_key_from_resolved_dir(dir).or_else(|| split_repo_root_tier_key(manifest_entry, dir))
}

/// Catalog ids whose tiers span repositories with a tier that is a whole snapshot root.
#[cfg(native_lane)]
const SPLIT_REPO_TIER_MODELS: &[&str] = sceneworks_core::jobs_store::QWEN_IMAGE_2_1_MODEL_IDS;

#[cfg(native_lane)]
fn split_repo_root_tier_key(manifest_entry: &JsonObject, dir: &Path) -> Option<&'static str> {
    use sceneworks_core::model_artifacts::artifact_selection::{
        is_pending_artifact_download, model_download_for_variant,
    };
    let model_id = manifest_entry.get("id").and_then(Value::as_str)?;
    if !SPLIT_REPO_TIER_MODELS.contains(&model_id) {
        return None;
    }
    let revision = dir.file_name()?.to_str()?;
    let snapshots = dir.parent()?;
    if snapshots.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let repo_dir = snapshots.parent()?.file_name()?.to_str()?;
    let entry = Value::Object(manifest_entry.clone());
    ["bf16", "q8", "q4"].into_iter().find(|tier| {
        let Some(download) = model_download_for_variant(&entry, tier) else {
            return false;
        };
        let root_row = download
            .get("files")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty);
        root_row
            && !is_pending_artifact_download(&download)
            && download.get("revision").and_then(Value::as_str) == Some(revision)
            && download
                .get("repo")
                .and_then(Value::as_str)
                .is_some_and(|repo| repo_dir == format!("models--{}", repo.replace('/', "--")))
    })
}

/// Where `tier`'s snapshot directory would be, read off the request's OWN manifest entry.
///
/// `None` when the catalog declares no such tier, when its revision is still the sc-24112
/// null-SHA placeholder (the artifact is not published, so there is nothing to point at), or when
/// that pinned snapshot is not in the cache. The `files` glob's leading path component is what
/// names the subdir — `["q8/*"]` ⇒ `q8/` — and an empty `files` (the whole-repo upstream bf16 row)
/// means the snapshot ROOT, which is exactly the distinction that makes this family split-repo.
#[cfg(native_lane)]
fn qwen_image_2_1_declared_tier_dir(
    settings: &Settings,
    request: &ImageRequest,
    tier: &str,
) -> Option<PathBuf> {
    let entry = Value::Object(request.model_manifest_entry.clone());
    let download =
        sceneworks_core::model_artifacts::artifact_selection::model_download_for_variant(
            &entry, tier,
        )?;
    if sceneworks_core::model_artifacts::artifact_selection::is_pending_artifact_download(&download)
    {
        return None;
    }
    let repo = download.get("repo").and_then(Value::as_str)?;
    let revision = download.get("revision").and_then(Value::as_str)?;
    let files: Vec<String> = download
        .get("files")
        .and_then(Value::as_array)
        .map(|files| {
            files
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    // An install made at a revision the pin has since SUPERSEDED (sc-25909: the q8/q4 re-host
    // republished to add each tier's NOTICE) is carried forward cache-only before the exact lookup.
    let root = crate::model_jobs::huggingface_pinned_tier_snapshot_dir(
        &settings.data_dir,
        repo,
        revision,
        &files,
    )?;
    let subdir = files
        .first()
        .map(String::as_str)
        .and_then(|pattern| pattern.split('/').next())
        // Confined to a single plain directory name. The value is a checked-in catalog string
        // rather than user input, but a glob is still a string a future edit could widen, and a
        // traversal component here would point the LOADER outside the cache — cheap to refuse.
        .filter(|component| {
            !component.is_empty()
                && *component != "*"
                && !component.starts_with('.')
                && sceneworks_core::ids::is_safe_id(component)
        });
    match subdir {
        Some(component) => Some(root.join(component)),
        None => Some(root),
    }
}

/// FLUX.1's generic Candle route reads only the hosted packed q4/q8 turnkeys. The shared resolver
/// deliberately has a best-available fallback chain for ordinary matrix models; using it unchecked
/// here could turn an explicit FLUX q4/q8 selection into a different tier (or a dense root). Probe the
/// shared resolver with the exact requested tier, then accept its answer only when the resolved
/// directory names that same tier.
///
/// This is intentionally limited to generic Candle text-to-image. The existing FLUX IP-Adapter and
/// strict-control providers retain their own established weight-resolution and load contracts.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_flux1_packed_requested_tier(request: &ImageRequest) -> WorkerResult<&'static str> {
    match request.advanced.get("mlxQuantize") {
        None | Some(Value::Null) => Ok("q4"),
        Some(value) => match crate::payload::quant_int(value)
        {
            Some(4) => Ok("q4"),
            Some(8) => Ok("q8"),
            _ => Err(WorkerError::InvalidPayload(format!(
                "{} on Candle supports only its hosted packed q4 or q8 tiers",
                request.model
            ))),
        },
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_flux1_packed_resolution_is_exact(requested_tier: &str, resolved: &Path) -> bool {
    resolved.file_name().and_then(|name| name.to_str()) == Some(requested_tier)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_flux1_packed_weights_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    let requested_tier = candle_flux1_packed_requested_tier(request)?;
    let mut probe = request.clone();
    probe.advanced.insert(
        "mlxQuantize".to_owned(),
        Value::from(if requested_tier == "q4" { 4 } else { 8 }),
    );
    let resolved = resolve_weights_dir(&probe, settings)?;
    match resolved {
        Some(dir) if candle_flux1_packed_resolution_is_exact(requested_tier, &dir) => Ok(Some(dir)),
        Some(_) | None => Err(WorkerError::InvalidPayload(format!(
            "{} requested Candle packed {requested_tier}, but that exact artifact is not installed; \
             refusing to fall back to a different tier",
            request.model
        ))),
    }
}

/// Whether completed-image telemetry must replay the Candle-only FLUX.1 exact-tier resolver.
///
/// The metrics consumer is shared by MLX and Candle builds, but the direct packed FLUX.1 route is
/// not: macOS must keep using its ordinary MLX resolver and default-tier semantics. Keeping that
/// distinction in a pure cfg-sensitive predicate makes the build boundary testable without making
/// [`resolve_candle_flux1_packed_weights_dir`] (or any Candle production route) available on macOS.
#[cfg(native_lane)]
fn completed_image_metrics_use_candle_flux1_exact_resolution(model: &str) -> bool {
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let exact = matches!(model, "flux_schnell" | "flux_dev");
    #[cfg(target_os = "macos")]
    let exact = {
        let _ = model;
        false
    };
    exact
}

/// Re-resolve the tier used by a completed image for telemetry only. The Candle-only FLUX.1 lane
/// replays its exact q4/q8 resolver; macOS and all other families replay the ordinary resolver that
/// selected their production weights. The cfg around the Candle call is intentional: a shared
/// metrics helper must never make the off-Mac Candle route part of a macOS build.
#[cfg(native_lane)]
fn resolve_completed_image_metrics_tier_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> Option<PathBuf> {
    let _use_candle_exact =
        completed_image_metrics_use_candle_flux1_exact_resolution(&request.model);
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if _use_candle_exact {
        return resolve_candle_flux1_packed_weights_dir(request, settings)
            .ok()
            .flatten();
    }
    #[cfg(target_os = "macos")]
    debug_assert!(
        !_use_candle_exact,
        "macOS metrics must not select the Candle-only FLUX.1 resolver"
    );
    resolve_weights_dir(request, settings).ok().flatten()
}

#[cfg(target_os = "macos")]
pub(super) fn resolved_mlx_artifact_tier(
    weights_dir: &Path,
    quant_bits: Option<i64>,
) -> Option<&'static str> {
    tier_key_from_resolved_dir(weights_dir)
        .or_else(|| quant_bits.map(|bits| if bits <= 4 { "q4" } else { "q8" }))
}

#[cfg(target_os = "macos")]
pub(super) fn resolved_mlx_artifact_tier_for_model(
    model_id: &str,
    weights_dir: &Path,
    quant_bits: Option<i64>,
) -> Option<&'static str> {
    fixed_mlx_artifact_tier(model_id)
        .or_else(|| resolved_mlx_artifact_tier(weights_dir, quant_bits))
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
pub(super) fn resolved_mlx_artifact_provenance(
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
        let provenance = crate::model_jobs::app_managed_artifact_provenance(&resolved_weights_dir)?;
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
#[path = "tests/resolved_artifact_provenance_tests.rs"]
pub(crate) mod resolved_artifact_provenance_tests;

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
    // hence no `mlx.denseTextEncoderTier`. The pinned Candle descriptor advertises no load-time
    // quants and rejects a nonempty `LoadSpec.quantize`; its loader packed-detects q4/q8 directly
    // from the selected directory while the worker records that resolved tier separately. Replaces
    // the gated BFL download + install-time quantize.
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
    // dense in every tier. Unlike the dense-TE klein class, the q4/q8 load-quant is an accepted no-op
    // on the already-packed weights and bf16 resolves to Quant::None — so these do
    // NOT need a `mlx.denseTextEncoderTier` declaration (only `flux2_klein_9b` / `_kv` carry that flag;
    // the "NOT" was dropped by a rename, which left this block contradicting its own UNLIKE clause two
    // lines above). The SANA descriptor now advertises supported_quants
    // Q4/Q8 (mlx-gen #654), so `supports_quant()` is true and they flow through the same
    // resolve_quant + reconcile path as every other matrix model (no more no-quant special case).
    "sana_1600m",
    "sana_sprint_1600m",
    // Kolors (sc-9946, epic 8506): the `SceneWorks/kolors-mlx` turnkey ships standard q4/q8/bf16
    // tiers. mlx-gen #659 packs the SDXL-style UNet + the ChatGLM3-6B `ChatGlmLinear` projections
    // and packed-detects on load; the SDXL VAE stays dense in every tier. Unlike the dense-TE klein
    // class, the ChatGLM3 TE is packed, so the q4/q8 load-quant is an accepted no-op on the
    // already-packed weights and bf16 resolves to Quant::None — no
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
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn dense_tier_subdir(root: PathBuf) -> PathBuf {
    let roots_a_component_tree = root.join("unet").is_dir() || root.join("transformer").is_dir();
    if !roots_a_component_tree && root.join("bf16").is_dir() {
        return root.join("bf16");
    }
    root
}

/// Whether `model` is one of the three Anima catalog ids (epic 10512). Anima is convert-at-install with
/// a bespoke tier resolver, so — like Ideogram/Boogu/Krea — it is NOT in [`STANDARD_TIER_MODELS`].
#[cfg(native_lane)]
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
#[cfg(native_lane)]
fn anima_tier_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let bits = request.quant_bits();
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        // A hidden `._*.safetensors` AppleDouble sidecar is not a DiT (SceneWorks#1333).
        let has_dit = std::fs::read_dir(dir.join("diffusion_models"))
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|entry| !sceneworks_core::lora_family::is_hidden_file(&entry.path()))
                    .any(|entry| {
                        entry
                            .file_name()
                            .to_string_lossy()
                            .ends_with(".safetensors")
                    })
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
    let bits = request.quant_bits();
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

/// Requested Boogu download subfolder. Even the default Q8 variant can be absent when only
/// a sibling variant is installed. The load-side resolver separately chooses among complete tiers.
fn boogu_tier_subdir(variant: &str, bits: Option<i64>) -> String {
    match crate::payload::tier_for_bits(bits, "q8") {
        "bf16" => format!("{variant}-bf16"),
        "q4" => format!("{variant}-q4"),
        _ => variant.to_owned(),
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
    let bits = request.quant_bits();
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
    let bits = request.quant_bits();
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
    pick_loadable_tier(
        &[preferred, "q8", "q4", "bf16"],
        &present,
        &tier_components_present,
    )
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
/// non-quantized surface) and the downloaded ConvRot DiT single-file (the named
/// `krea_convrot_dit` component, leaving `LoadSpec::text_encoder` exclusively for
/// real encoder substitution).
///
/// `None` when the request didn't select ConvRot, OR either artifact isn't present yet (the bf16
/// `bf16/` subdir of the `krea-2-turbo-mlx` turnkey, or the ConvRot DiT `.safetensors`) — the caller
/// then falls back to the normal dense/packed path rather than half-loading. The bf16 base is fetched
/// on demand by [`ensure_krea_convrot_base_present`] before this resolves. The sm_89 compute-cap floor
/// is enforced ENGINE-side (`ensure_int8_floor` inside `load_components_convrot`) AND surfaced as the
/// worker's `int8_convrot` capability, so an ineligible card never reaches here (the picker hides it).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_krea_convrot(request: &ImageRequest, settings: &Settings) -> Option<(PathBuf, PathBuf)> {
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

/// Ensure the requested Boogu variant/tier is complete in the shared snapshot before loading.
/// Base, Turbo and Edit share a repository but have separate component directories. Even Q8
/// may need fetching when only a sibling variant has been installed. Reuse the native cache
/// downloader for missing or partial tiers, then check tokenizer and component-weight presence.
#[cfg(native_lane)]
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
    let bits = request.quant_bits();
    let tier = boogu_tier_subdir(variant, bits);
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
    // Base, Turbo and Edit share a repository, but installation of a sibling is not proof
    // that this variant exists. Repair missing default tiers and incomplete non-default tiers.
    if tier_dir
        .as_ref()
        .is_some_and(|dir| sceneworks_core::mlx_tier_completeness::boogu_tier_complete(dir))
    {
        return Ok(());
    }
    // The tier subfolder nests transformer/mllm/vae (leaf-dir globs, like the catalog Q8 entry).
    let files = vec![
        format!("{tier}/transformer/*"),
        format!("{tier}/mllm/*"),
        format!("{tier}/vae/*"),
    ];
    let revision = turnkey_tier_revision(&repo, model.default_repo(), BOOGU_MLX_TURNKEY_REVISION);
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, &repo, revision, &files).await?;
    let installed = if repo == model.default_repo() {
        crate::model_jobs::huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            &repo,
            BOOGU_MLX_TURNKEY_REVISION,
        )
    } else {
        huggingface_snapshot_dir(&settings.data_dir, &repo)
    };
    if !installed.as_ref().is_some_and(|root| {
        sceneworks_core::mlx_tier_completeness::boogu_tier_complete(&root.join(&tier))
    }) {
        return Err(WorkerError::InvalidPayload(format!(
            "{}: downloaded Boogu tier {tier} is incomplete; expected transformer, mllm tokenizer/weights, and VAE",
            request.model,
        )));
    }
    Ok(())
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
#[cfg(native_lane)]
async fn ensure_ideogram_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<()> {
    if request.model != "ideogram_4" && request.model != "ideogram_4_turbo" {
        return Ok(());
    }
    let bits = request.quant_bits();
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
    }) {
        return Ok(());
    }
    let files = vec![format!("{tier}/*")];
    let revision =
        turnkey_tier_revision(&repo, model.default_repo(), IDEOGRAM_MLX_TURNKEY_REVISION);
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, &repo, revision, &files)
        .await
        .map(|_| ())
}

#[cfg(native_lane)]
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
#[cfg(native_lane)]
pub(crate) fn resolve_quant(
    request: &ImageRequest,
    tier_dir: Option<&Path>,
) -> (Option<Quant>, Option<i64>) {
    resolve_quant_gated(request, nvfp4_host_eligible(), tier_dir)
}

/// [`resolve_quant`] with the NVFP4 **host** gate passed in rather than probed (sc-11042). Split out
/// for testability, exactly like [`standard_tier_subdir_gated`] — and necessarily so: the candle CI
/// lane runs ON the sm_120 rig, so a test that let this probe the live cap would assert different
/// things on the rig and on a developer box. Production has one caller.
#[cfg(native_lane)]
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
    let raw = request.quant_bits()
        .or_else(|| {
            request
                .model_manifest_entry
                .get("mlx")
                .and_then(|mlx| mlx.get("quantize"))
                .and_then(quant_int)
        });
    match crate::payload::tier_for_bits(raw, "q8") {
        "bf16" => (None, None),
        "q4" => (Some(Quant::Q4), Some(4)),
        _ => (Some(Quant::Q8), Some(8)),
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
    match crate::payload::tier_for_bits(request.quant_bits(), "q8") {
        "bf16" => None,
        "q4" => Some(4),
        _ => Some(8),
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
    tier_key_for_resolved_dir(manifest_entry, weights_dir)
        .unwrap_or_else(|| crate::vram_gate::requested_tier_key(advanced, manifest_entry, nvfp4))
}

#[cfg(native_lane)]
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
fn tier_quant_from_resolved_dir(
    manifest_entry: &JsonObject,
    dir: &Path,
) -> Option<(Option<Quant>, Option<i64>)> {
    match tier_key_for_resolved_dir(manifest_entry, dir)? {
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
#[cfg(native_lane)]
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
    (tier_key_for_resolved_dir(&request.model_manifest_entry, &dir) == Some(tier_static_name(tier)))
        .then_some(dir)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_certified_artifact_path(
    engine_id: &str,
    settings: &Settings,
    weights_dir: &Path,
    tier: &str,
) -> bool {
    let (repo, revision) = match engine_id {
        "z_image" | "z_image_turbo" => (ZIMAGE_MLX_TURNKEY_REPO, ZIMAGE_MLX_TURNKEY_REVISION),
        "flux1_schnell" => (
            "SceneWorks/flux1-schnell-mlx",
            FLUX1_SCHNELL_MLX_TURNKEY_REVISION,
        ),
        "flux1_dev" => ("SceneWorks/flux1-dev-mlx", FLUX1_DEV_MLX_TURNKEY_REVISION),
        "flux2_dev" => ("SceneWorks/flux2-dev-mlx", FLUX2_DEV_MLX_TURNKEY_REVISION),
        "flux2_klein_9b" => (
            "SceneWorks/flux2-klein-9b-mlx",
            FLUX2_KLEIN_9B_MLX_TURNKEY_REVISION,
        ),
        // Iris-3B generation (sc-25690). The one dense artifact is the pinned upstream snapshot at
        // its ROOT — there is no tier sub-directory to join — and its text encoder is a manifest
        // co-requisite the worker stages cache-only at its own pinned revision
        // (`attach_required_components`), so the backbone root is the whole caller-chosen identity.
        // Without this arm the route was permanently uncertified: the shared ladder then submits
        // the resident estimate alone, and the staged and bounded-attention rungs
        // `candle-gen-iris` implements could never be selected for a request.
        "iris_3b" => {
            return candle_certified_hf_artifact_path(
                settings,
                gen_core_iris::UPSTREAM_WEIGHTS_REPO,
                gen_core_iris::UPSTREAM_WEIGHTS_REVISION,
                Path::new(""),
                weights_dir,
            );
        }
        _ => return false,
    };
    candle_certified_hf_artifact_path(settings, repo, revision, Path::new(tier), weights_dir)
}

/// The numeric tier key the SHARED memory selector binds a request to.
///
/// The VRAM gate names a tier from the resolved directory's basename, or — for a flat root with no
/// tier basename — from `advanced.mlxQuantize` (q8 by default; `requested_tier_key`). That key only
/// has to index the manifest's per-tier floor rows there. The shared selector is stricter: it turns
/// the key into the `MemoryNumericTier` of the selection, and the provider's safety check refuses a
/// selection whose tier is not the one the generator LOADED. Iris-3B refuses quantization and has
/// exactly one dense artifact, so its loaded tier is bf16 whatever the request's advisory quantize
/// knob says; handing the selector the gate's `q8` would mint a `Q8` selection the loaded bf16
/// generator rejects ("selected tier does not match loaded tier") the moment an optimized rung is
/// chosen.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_shared_memory_tier<'a>(engine_id: &str, gate_tier: &'a str) -> &'a str {
    match engine_id {
        "iris_3b" => "bf16",
        // FLUX 3's gate tier IS its loaded package (sc-26241: `Flux3LoadPlan` names it by
        // identity), so the selection binds the package that loads — FP8R as FP8R, never bf16.
        _ => gate_tier,
    }
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
pub(super) fn candle_certified_load_spec(
    engine_id: &str,
    settings: &Settings,
    spec: &LoadSpec,
    manifest_entry: &JsonObject,
    tier: &str,
) -> bool {
    let WeightsSource::Dir(weights_dir) = &spec.weights else {
        return false;
    };
    let declaration_owned =
        crate::memory_route_registry::candle_declaration_owns_load_shape(engine_id)
            || crate::memory_route_registry::candle_manifest_declares_request_strategy_provider(
                manifest_entry,
                engine_id,
            );
    if !is_mage_engine(engine_id) && !declaration_owned {
        return candle_certified_artifact_path(engine_id, settings, weights_dir, tier);
    }
    let Some(downloads) = manifest_entry.get("downloads").and_then(Value::as_array) else {
        return false;
    };
    let certify = |component_id: Option<&str>, actual: &Path| {
        let mut matches = downloads.iter().filter(|download| {
            download.get("provider").and_then(Value::as_str) == Some("huggingface")
                && (download.get("variant").and_then(Value::as_str) == Some(tier)
                    || (component_id.is_some() && download.get("variant").is_none()))
                && download.get("componentId").and_then(Value::as_str) == component_id
                && download
                    .get("coRequisite")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
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
            .or_else(|| {
                component_id
                    .and_then(|_| download.get("files"))
                    .and_then(Value::as_array)
                    .and_then(|files| match files.as_slice() {
                        [Value::String(file)] if !file.contains('*') => Some(file.as_str()),
                        _ => None,
                    })
            })
            .unwrap_or(tier);
        candle_certified_hf_artifact_path(settings, repo, revision, Path::new(relative), actual)
    };

    // Component certification is descriptor-driven, and `media_descriptor` only ever resolves a
    // registered GENERATOR. Several declaration-owned runtime providers are registered as encoder
    // contract ROUTES over a sibling generator instead — `z_image_control`,
    // `z_image_turbo_control`, `qwen_image_edit`, `qwen_image_control` — so no descriptor carries
    // their id and no component contract exists to certify against. Those routes bind their
    // co-requisites outside `spec.components`; the exact certification for them is the one this
    // function has always applied: the primary matches its pinned download AND nothing else is
    // bound. Reading a missing descriptor as `false` (as this did between the SDXL-derivative
    // change and sc-20799) reports every such route uncertified, which silently strips its
    // receipt-priced admission down to resident-only.
    let descriptor = crate::inference_runtime::media_descriptor(engine_id);
    if !is_mage_engine(engine_id) && descriptor.is_none() {
        return spec.components.is_empty()
            && (spec.text_encoder.is_none() || spec.validate_prepared_file_pins().is_ok())
            && certify(None, weights_dir);
    }
    let Some(descriptor) = descriptor else {
        return false;
    };
    let expected_component_count = descriptor.required_components.len()
        + usize::from(
            engine_id == "sdxl"
                && spec.ip_adapter.is_some()
                && spec.components.contains_key("sdxl_ip_image_encoder"),
        );
    if spec.components.len() != expected_component_count {
        return false;
    }

    certify(None, weights_dir)
        && descriptor.required_components.iter().all(|component_id| {
            spec.components.get(*component_id).is_some_and(|source| {
                certify(
                    Some(component_id),
                    crate::weights_source::WeightsSourceExt::path(source),
                )
            })
        })
}

/// Bind the endpoint price to the complete ordinary SDXL assembly, including the shared VAE and
/// tokenizers actually opened by Candle. ArtifactSeal's existing stat-identity memo avoids repeated
/// multi-GB hashing; these exact tokens then participate in cache identity and guard registry load.
/// Different base-tier members or consumed component bytes lose endpoint authority. Unused files in
/// a shared tokenizer/VAE snapshot do not affect the price of the same loaded assembly.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn seal_sdxl_endpoint_load_spec(
    spec: &mut LoadSpec,
    expected: &[sceneworks_core::memory_anchor::sdxl_cuda_memory_endpoints::EndpointArtifactFile],
) -> WorkerResult<bool> {
    use std::collections::BTreeMap;

    const COMPONENTS: [(&str, &str); 3] = [
        ("tokenizer_clip_l", "tokenizer.json"),
        ("tokenizer_clip_bigg", "tokenizer.json"),
        ("vae_fp16_fix", "diffusion_pytorch_model.safetensors"),
    ];
    // The measured assembly has one base directory and these three component sources. The stock
    // co-requisite resolver returns File for a single literal manifest member; providers also accept
    // its snapshot Dir. Extra encoders, controls, adapters, or imports cannot inherit this endpoint.
    if spec.file_source_paths().len() + spec.directory_source_paths().len() != 4
        || spec.components.len() != COMPONENTS.len()
        || COMPONENTS
            .iter()
            .any(|(id, _)| !spec.components.contains_key(*id))
    {
        return Ok(false);
    }
    let WeightsSource::Dir(base) = &spec.weights else {
        return Ok(false);
    };
    let mut components = Vec::new();
    for (id, file) in COMPONENTS {
        let path = match spec.components.get(id) {
            Some(WeightsSource::Dir(root)) => root.join(file),
            Some(WeightsSource::File(path))
                if path.file_name() == Some(std::ffi::OsStr::new(file)) =>
            {
                path.clone()
            }
            _ => return Ok(false),
        };
        components.push((format!("components/{id}/{file}"), path));
    }

    fn visit(
        root: &Path,
        directory: &Path,
        prefix: &str,
        files: &mut BTreeMap<String, PathBuf>,
    ) -> WorkerResult<()> {
        for entry in std::fs::read_dir(directory).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "SDXL endpoint inventory cannot read {}: {error}",
                directory.display()
            ))
        })? {
            let path = entry
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!("SDXL endpoint inventory failed: {error}"))
                })?
                .path();
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "SDXL endpoint member cannot be inspected: {error}"
                ))
            })?;
            if metadata.is_dir() {
                visit(root, &path, prefix, files)?;
                continue;
            }
            // HF snapshot links to blob files are normal; directory links are not inventory roots.
            if !std::fs::metadata(&path)
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "SDXL endpoint member target cannot be inspected: {error}"
                    ))
                })?
                .is_file()
            {
                return Err(WorkerError::InvalidPayload(format!(
                    "SDXL endpoint member must be a file: {}",
                    path.display()
                )));
            }
            let relative = path
                .strip_prefix(root)
                .expect("inventory child stays below root");
            let relative = relative
                .iter()
                .map(|part| part.to_str())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(
                        "SDXL endpoint member has a non-UTF-8 path".to_owned(),
                    )
                })?
                .join("/");
            files.insert(format!("{prefix}/{relative}"), path);
        }
        Ok(())
    }

    let mut actual = BTreeMap::new();
    let base = std::path::absolute(base).map_err(|error| {
        WorkerError::InvalidPayload(format!("SDXL endpoint root cannot be absolutized: {error}"))
    })?;
    visit(&base, &base, "base", &mut actual)?;
    // These are the exact joins used by SDXL's resolve_tokenizer_file/resolve_vae_file. The same HF
    // shared snapshot may contain additional CLIP weights installed for an unrelated model.
    for (name, path) in components {
        if !path.is_file() {
            return Ok(false);
        }
        actual.insert(name, path);
    }
    if expected.is_empty()
        || expected.len() != actual.len()
        || expected.iter().any(|file| !actual.contains_key(&file.path))
    {
        return Ok(false);
    }
    let mut pins = Vec::with_capacity(expected.len());
    for file in expected {
        let pin = gen_core::PinnedWeightsFile::pin(&actual[&file.path])
            .map_err(|error| crate::classify_engine_error("SDXL endpoint load pin", error))?;
        let digest = sceneworks_core::hashing::hex(pin.content_sha256());
        if pin.target_fingerprint().size != file.bytes || digest != file.sha256 {
            return Ok(false);
        }
        pins.push(pin);
    }
    crate::paths::prepare_load_spec_with_file_pins(spec, pins, "SDXL endpoint load preparation")?;
    Ok(true)
}

#[cfg(test)]
#[path = "tests/sdxl_endpoint_inventory_tests.rs"]
mod sdxl_endpoint_inventory_tests;

/// Seal every file recursively-inventoried receipt provider can read before selector evaluation.
/// The same finalized `LoadSpec` reaches the generator cache, so cold/warm execution cannot
/// silently re-resolve a pathname after admission.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn seal_recursive_receipt_load_spec(
    engine_id: &str,
    artifact_is_certified: bool,
    spec: &mut LoadSpec,
) -> WorkerResult<()> {
    if !matches!(
        engine_id,
        "chroma1_hd"
            | "chroma1_base"
            | "chroma1_flash"
            | "sd3_5_large"
            | "sd3_5_large_turbo"
            | "sd3_5_medium"
    ) || !artifact_is_certified
    {
        return Ok(());
    }

    fn visit(root: &Path, files: &mut Vec<PathBuf>) -> WorkerResult<()> {
        for entry in std::fs::read_dir(root).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Receipt artifact inventory cannot read {}: {error}",
                root.display()
            ))
        })? {
            let path = entry
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "Receipt artifact inventory failed: {error}"
                    ))
                })?
                .path();
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "Receipt artifact member cannot be inspected: {error}"
                ))
            })?;
            if metadata.file_type().is_symlink() {
                let target = std::fs::metadata(&path).map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "Receipt artifact symlink target cannot be inspected: {error}"
                    ))
                })?;
                if target.is_file() {
                    files.push(std::path::absolute(&path).map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "Receipt artifact member cannot be absolutized: {error}"
                        ))
                    })?);
                    continue;
                }
                return Err(WorkerError::InvalidPayload(format!(
                    "Receipt artifact symlink must resolve directly to a file: {}",
                    path.display()
                )));
            }
            if metadata.is_dir() {
                visit(&path, files)?;
            } else if metadata.is_file() {
                files.push(std::path::absolute(&path).map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "Receipt artifact member cannot be absolutized: {error}"
                    ))
                })?);
            }
        }
        Ok(())
    }

    let mut files = spec
        .file_source_paths()
        .into_iter()
        .map(std::path::absolute)
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Receipt file source cannot be absolutized: {error}"
            ))
        })?;
    for root in spec.directory_source_paths() {
        visit(root, &mut files)?;
    }
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Receipt artifact inventory is empty".to_owned(),
        ));
    }
    let pins = files
        .iter()
        .map(|path| {
            gen_core::PinnedWeightsFile::pin(path)
                .map_err(|error| crate::classify_engine_error("Receipt load pin", error))
        })
        .collect::<WorkerResult<Vec<_>>>()?;
    crate::paths::prepare_load_spec_with_file_pins(spec, pins, "Receipt load preparation")?;
    spec.validate_prepared_file_pins()
        .map_err(|error| crate::classify_engine_error("Receipt load preparation", error))
}

/// Seal exactly the files the Ideogram Candle loader reads. Direct component safetensors,
/// transformer configs, the tokenizer, TurboTime residual, ordered user adapters, and PiD assets
/// are load identity; nested files below a component are deliberately ignored because the provider
/// does not enumerate them.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn seal_ideogram_load_spec(
    engine_id: &str,
    artifact_is_certified: bool,
    spec: &mut LoadSpec,
) -> WorkerResult<()> {
    if !matches!(engine_id, "ideogram_4" | "ideogram_4_turbo") || !artifact_is_certified {
        return Ok(());
    }
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(WorkerError::InvalidPayload(
            "Ideogram certified load must resolve to one tier directory".to_owned(),
        ));
    };
    let root = root.clone();
    let mut files = Vec::new();
    for component in [
        "transformer",
        "unconditional_transformer",
        "text_encoder",
        "vae",
    ] {
        let dir = root.join(component);
        if !dir.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&dir).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Ideogram artifact inventory cannot read {}: {error}",
                dir.display()
            ))
        })? {
            let path = entry
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "Ideogram artifact inventory failed: {error}"
                    ))
                })?
                .path();
            if path.is_file()
                && path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
            {
                files.push(path);
            }
        }
    }
    files.push(root.join("transformer/config.json"));
    if engine_id == "ideogram_4" {
        files.push(root.join("unconditional_transformer/config.json"));
    } else {
        files.push(root.join("turbo_lora.safetensors"));
    }
    files.push(root.join("tokenizer/tokenizer.json"));
    files.extend(spec.file_source_paths().into_iter().map(Path::to_path_buf));
    if let Some(pid) = spec.pid.as_ref() {
        let WeightsSource::Dir(gemma) = &pid.gemma else {
            return Err(WorkerError::InvalidPayload(
                "Ideogram PiD Gemma must resolve to one directory".to_owned(),
            ));
        };
        let merged = gemma.join("gemma-2-2b-it.safetensors");
        if merged.is_file() {
            files.push(merged);
        } else {
            for entry in std::fs::read_dir(gemma).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "Ideogram PiD inventory cannot read {}: {error}",
                    gemma.display()
                ))
            })? {
                let path = entry
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "Ideogram PiD inventory failed: {error}"
                        ))
                    })?
                    .path();
                if path.is_file()
                    && path.extension().and_then(|extension| extension.to_str())
                        == Some("safetensors")
                {
                    files.push(path);
                }
            }
        }
        files.push(gemma.join("tokenizer.json"));
    }
    files = files
        .into_iter()
        .map(std::path::absolute)
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Ideogram load member cannot be absolutized: {error}"
            ))
        })?;
    files.sort();
    files.dedup();
    if files.is_empty() || files.iter().any(|path| !path.is_file()) {
        return Err(WorkerError::InvalidPayload(
            "Ideogram loader-visible artifact inventory is incomplete".to_owned(),
        ));
    }
    let pins = files
        .iter()
        .map(|path| {
            gen_core::PinnedWeightsFile::pin(path)
                .map_err(|error| crate::classify_engine_error("Ideogram load pin", error))
        })
        .collect::<WorkerResult<Vec<_>>>()?;
    crate::paths::prepare_load_spec_with_file_pins(spec, pins, "Ideogram load preparation")?;
    spec.validate_prepared_file_pins()
        .map_err(|error| crate::classify_engine_error("Ideogram load preparation", error))
}

#[cfg(test)]
#[path = "tests/chroma_load_seal_tests.rs"]
mod chroma_load_seal_tests;

#[cfg(test)]
#[path = "tests/ideogram_load_seal_tests.rs"]
mod ideogram_load_seal_tests;

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/mage_artifact_certification_tests.rs"]
mod mage_artifact_certification_tests;

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
    let root =
        crate::model_jobs::huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision)?;
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
    engine_id: &str,
    context: gen_core::MemoryRunContext,
) -> Option<gen_core::MemoryRunContext> {
    crate::memory_strategy::executable_context(engine_id, context)
}

/// The registered FLUX.2-dev generator is a text-to-image provider even when the catalog surface
/// originated from a style-variation prompt. Keep that provider-specific canonicalization here so
/// Z-Image, Qwen, and FLUX.1 retain their established mode/evidence semantics.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn candle_base_memory_request_mode<'a>(engine_id: &str, request_mode: &'a str) -> &'a str {
    if engine_id == "flux2_dev"
        || (engine_id == "flux2_klein_9b"
            && matches!(request_mode, "image_generation" | "text_to_image"))
        || (matches!(engine_id, "chroma1_hd" | "chroma1_base" | "chroma1_flash")
            && matches!(request_mode, "image_generation" | "text_to_image"))
        || (matches!(engine_id, "ideogram_4" | "ideogram_4_turbo")
            && matches!(request_mode, "image_generation" | "text_to_image"))
        || (matches!(engine_id, "sana_1600m" | "sana_sprint_1600m")
            && matches!(request_mode, "image_generation" | "text_to_image"))
        || (matches!(
            engine_id,
            "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium"
        ) && matches!(request_mode, "image_generation" | "text_to_image"))
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
        lane_reference_count(has_single_reference, edit_reference_count, has_edit_mask)
    }
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn flux2_klein_reference_bearing_mode(mode: &str) -> bool {
    matches!(
        mode,
        "edit_image" | "reference" | "image_to_image" | "character_image" | "style_variations"
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
#[cfg(any(all(target_os = "macos", test), feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum TierFit {
    /// Runs — resident or (where the provider stages components) sequentially.
    Fits,
    /// Won't run even sequentially. `needed_gb`/`available_gb` are for the reject message.
    TooBig { needed_gb: f64, available_gb: f64 },
}

/// The capability-downtier decision (sc-10733).
#[cfg(any(all(target_os = "macos", test), feature = "backend-candle"))]
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
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
// Scoring a candle tier is a manifest arithmetic lookup, so that lane keeps the eager form and its
// straight-line read. On macOS the only callers left are the ordering tests — the MLX gate scores
// lazily now, because there scoring a tier hashes an encoder.
fn choose_downtier(default_tier: &str, candidates: &[(&'static str, TierFit)]) -> DowntierPick {
    let tiers: Vec<&'static str> = candidates.iter().map(|(tier, _)| *tier).collect();
    let mut fits = candidates.iter().map(|(_, fit)| *fit);
    match choose_downtier_lazy::<_, std::convert::Infallible>(default_tier, &tiers, |_| {
        Ok(fits.next())
    }) {
        Ok(pick) => pick,
        Err(never) => match never {},
    }
}

/// [`choose_downtier`] with the fit scored ON DEMAND, stopping at the first tier that fits.
///
/// Same ordering rule, same result — this is the single implementation and the eager form above
/// delegates to it, so the two cannot drift. What changes is how many tiers get SCORED.
///
/// That matters because on the MLX lane scoring a tier is not cheap. `mlx_tier_fit` reaches
/// `ProviderRegistry::footprint`, which resolves the encoder through `EncoderContract::source_for_load`
/// and seals it with `ArtifactSeal::read_unchanged` — a full SHA-256 of that tier's text encoder. A
/// stack sample of a krea_2_turbo job caught 62% of a 30-second window inside that subtree, 96% of it
/// in the SHA-256 compressor itself and only 3% in `read`: it is CPU, not I/O.
///
/// The caller then collected EVERY candidate before choosing, so a machine with room to spare paid for
/// bf16, q8 and q4 in turn — three encoders hashed to answer a question the first one had already
/// settled. Measured on one job: 15s + 10s + 6s, against a budget with 61 GB unused.
///
/// `Ok(None)` from `fit` SKIPS a tier rather than scoring it, preserving the caller's pre-existing
/// filter for a candidate whose directory does not resolve: such a tier was absent from the eager
/// slice entirely, so it must not become the named rejection either.
#[cfg(any(all(target_os = "macos", test), feature = "backend-candle"))]
fn choose_downtier_lazy<F, E>(
    default_tier: &str,
    tiers: &[&'static str],
    mut fit: F,
) -> Result<DowntierPick, E>
where
    F: FnMut(&'static str) -> Result<Option<TierFit>, E>,
{
    let mut smallest_reject: Option<(&'static str, f64, f64)> = None;
    for &tier in tiers {
        let Some(fit) = fit(tier)? else {
            continue;
        };
        match fit {
            // DESCENDING fidelity ⇒ the first that fits is the highest-fidelity fitting tier, and
            // nothing below it can win — so nothing below it is scored.
            TierFit::Fits => {
                return Ok(if tier == default_tier {
                    DowntierPick::Keep
                } else {
                    DowntierPick::Downtier(tier)
                });
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
    Ok(match smallest_reject {
        Some((tier, needed_gb, available_gb)) => DowntierPick::Reject {
            tier,
            needed_gb,
            available_gb,
        },
        None => DowntierPick::Keep,
    })
}

/// The installed tiers in `[floor, default]` for `request`'s model, in DESCENDING fidelity, ready for
/// [`choose_downtier`] once each is paired with its per-lane [`TierFit`] (sc-10733). `default_tier` is
/// the disk-clamped resolved tier (the downtier ceiling — the clamp only ever lowers fidelity); `floor`
/// is the per-model quality floor ([`min_quality_floor`], defaulting to `q4`). Excludes tiers not on
/// disk so the downtier never lands on an uninstalled tier.
#[cfg(native_lane)]
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
    manifest_entry: &JsonObject,
    allow_quant_change: bool,
    model_id: &str,
    job_id: &str,
    engine: &str,
) -> (Option<Quant>, Option<i64>) {
    let Some((actual_quant, actual_bits)) =
        tier_quant_from_resolved_dir(manifest_entry, weights_dir)
    else {
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

/// Resolve denoise steps: `advanced.steps` (clamped 1..=max(80, family default)) else the family
/// default.
/// Shared by the MLX path and the candle lane (sc-5096).
#[cfg(native_lane)]
fn resolve_steps(request: &ImageRequest, model: &ResolvedModel) -> u32 {
    request
        .advanced
        .get("steps")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        // The historical 80-step ceiling, raised to the model's own default where that is higher
        // (Iris-3B's release default is 100, sc-25679) so the default is always reachable rather
        // than silently clamped below what the model ships with.
        // A declared `limits.hardMaxSteps` (sc-24215) replaces the historical ceiling; the
        // model's own boundary refuses a request above it before this point.
        .map(|steps| {
            let ceiling =
                sceneworks_core::video_request::hard_max_steps(&request.model_manifest_entry)
                    .unwrap_or_else(|| max_requested_steps(model.default_steps()));
            (steps as u32).clamp(1, ceiling)
        })
        .unwrap_or(model.default_steps())
}

/// The largest `advanced.steps` the generic image lane forwards: the historical 80, or the model's
/// own default when that is higher. Mirrored by Image Studio's Steps input (`maxStepsForModel`).
#[cfg(native_lane)]
fn max_requested_steps(default_steps: u32) -> u32 {
    default_steps.max(sceneworks_core::image_request::BASE_MAX_IMAGE_STEPS)
}

/// Resolve the guidance scale. Distilled variants (z-image-turbo, flux schnell) take
/// no guidance — the engine rejects `Some(_)` on them — so this returns `None`. For a
/// guided variant (flux dev) it is `advanced.guidanceScale` else the family default.
/// Shared by the MLX path and the candle lane (sc-5096); the descriptor's `supports_guidance` is the
/// candle descriptor on the Windows lane, so a distilled candle family (z-image, flux schnell) still
/// gets `None` and a guided one (flux dev, flux2, qwen, sdxl) gets the scale.
#[cfg(native_lane)]
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
#[cfg(native_lane)]
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
#[cfg(native_lane)]
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
#[cfg(any(native_lane, test))]
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
#[cfg(any(native_lane, test))]
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
#[cfg(native_lane)]
fn uses_true_cfg(model: &ResolvedModel) -> bool {
    !model.supports_guidance() && model.supports_negative_prompt()
}

/// Apply the Ideogram placeholder recovery policy to one completed render. The renderer is injected
/// so the MLX and candle lanes share cancellation, reseeding, retry exhaustion, and error behavior.
#[cfg(native_lane)]
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

#[cfg(native_lane)]
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
#[cfg(native_lane)]
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
#[cfg(native_lane)]
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
/// Shared by MLX and Candle: every Candle family validates this declared kind against the selected
/// file before application, so the same `networkType: lokr` classification feeds both backends.
#[cfg(native_lane)]
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

#[cfg(any(
    native_lane,
    test
))]
pub(crate) fn classify_prepared_adapter(pin: &gen_core::PinnedWeightsFile) -> WorkerResult<AdapterKind> {
    let header = pin
        .read_unchanged(|file| {
            read_safetensors_header(file)
                .map_err(|error| gen_core::Error::Msg(format!("LoRA header: {error}")))
        })
        .map_err(|error| crate::classify_engine_error("LoRA source validation failed", error))?;
    let network_type = header
        .get("__metadata__")
        .and_then(|meta| meta.get("networkType"))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase());
    Ok(if network_type.as_deref() == Some("lokr") {
        AdapterKind::Lokr
    } else {
        AdapterKind::Lora
    })
}

#[derive(Debug)]
#[cfg(any(native_lane, test))]
struct PreparedAdapters {
    specs: Vec<AdapterSpec>,
    pins: Vec<gen_core::PinnedWeightsFile>,
}

#[cfg(any(native_lane, test))]
impl PreparedAdapters {
    // Gated to exactly where it is used. `imported_edit_requires_the_identity_edit_lora` became
    // cross-platform when the Kontext edit surface started being served on both native backends, so
    // a macOS-only helper left the candle test build without it.
    #[cfg(all(native_lane, test))]
    fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

/// Prepared counterpart of [`resolve_adapters`]. Classification reads through the exact token that
/// is later installed on the `LoadSpec`; repeated references to the same lexical entry share one
/// token, while preserving every requested adapter spec and its scale/order.
#[cfg(any(native_lane, test))]
fn resolve_prepared_adapters(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<PreparedAdapters> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs = Vec::with_capacity(request.loras.len());
    let mut pins = BTreeMap::<PathBuf, gen_core::PinnedWeightsFile>::new();
    for lora in &request.loras {
        let pin = resolve_prepared_adapter_file(lora, settings, &request.model)?;
        let pin = crate::adapter_resolution::retain_adapter_pin(&mut pins, pin)?;
        let kind = classify_prepared_adapter(pin)?;
        let scale = lora_weight(lora) as f32;
        specs.push(AdapterSpec::new(
            pin.loader_path().to_path_buf(),
            scale,
            kind,
        ));
    }
    Ok(PreparedAdapters {
        specs,
        pins: pins.into_values().collect(),
    })
}

/// Resolve up to [`MAX_JOB_LORAS`] request adapters into engine specs (path + scale + kind).
/// Shared by the MLX and Candle paths so both backends enforce the same stack ceiling.
#[cfg(native_lane)]
fn resolve_adapters(request: &ImageRequest, settings: &Settings) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs = Vec::with_capacity(request.loras.len());
    for lora in &request.loras {
        let file = resolve_adapter_file(lora, settings, &request.model)?;
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
    if engine_id.starts_with("krea_2_")
        || engine_id.starts_with("z_image")
        || crate::candle_memory_strategy::is_sd35(engine_id)
        || tier != "bf16"
    {
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
    mlx_raw_settings_for_bundle(request, None, repo, steps, quant_bits, guidance)
}

/// [`mlx_raw_settings`] for a render whose weights LOADED from `bundle` (the generic lane's resolved
/// `weights_dir`): Supra2's unconditional-context record is read off that bundle's manifest, never
/// re-resolved from the request (sc-24217 pre-merge review).
fn mlx_raw_settings_for_bundle(
    request: &ImageRequest,
    bundle: Option<&Path>,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
) -> JsonObject {
    let mut raw = if sceneworks_core::image_request::flux3::is_model(&request.model) {
        flux3_references::caller_raw_settings(&request.advanced)
    } else {
        request.advanced.clone()
    };
    // This key is worker-owned. A replayed recipe or hostile advanced payload must never be able to
    // forge Auto's rung-4 decision; the trusted Candle path may add it back after final admission.
    scrub_untrusted_memory_strategy_disclosure(&mut raw);
    // Worker-owned too (epic 25899): only the image's own render report adds the record back.
    raw.remove(lane_controls::PREFIX_KV_CACHE_KEY);
    raw.remove(sceneworks_core::moss_image::EDIT_REPORT_KEY);
    raw.remove("moss_image_edit_execution_report");
    raw.remove(sceneworks_core::moss_image::REPORT_KEY);
    raw.remove("moss_image_execution_report");
    // Worker-owned (sc-24215): the pinned weights Supra2 rendered from, never a replayed copy.
    // So are the unconditional context CFG steered against and the rendering backend (epic-review
    // follow-up): the reproducibility record, never a replayed copy.
    raw.remove(sceneworks_core::supra2::MODEL_REVISIONS_KEY);
    if sceneworks_core::supra2::is_supra2(&request.model) {
        raw.insert(
            sceneworks_core::supra2::MODEL_REVISIONS_KEY.to_owned(),
            sceneworks_core::supra2::pinned_revisions(&request.model_manifest_entry),
        );
        raw.insert(
            sceneworks_core::supra2::UNCONDITIONAL_KEY.to_owned(),
            sceneworks_core::supra2::unconditional_source(bundle, guidance),
        );
        raw.insert(
            sceneworks_core::supra2::BACKEND_KEY.to_owned(),
            Value::from(if cfg!(target_os = "macos") {
                "mlx"
            } else {
                "candle"
            }),
        );
    }
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

/// Krea's MLX base-DiT turnkeys are already packed in the resolved `bf16/q4/q8`
/// artifact directory. `LoadSpec::quantize` is a load-time transform request, not
/// that artifact-tier identity, so forwarding the UI tier as a quant transform
/// would make the provider predicate reject the otherwise exact production spec.
/// Keep the resolved tier on the request plan/declaration axes instead.
#[cfg(target_os = "macos")]
pub(super) fn mlx_load_quant_for_resolved_artifact(
    engine_id: &str,
    quant: Option<Quant>,
) -> Option<Quant> {
    // FLUX.2 Klein's packed-tier rule is the MLX provider crate's product decision (`product_load`).
    if matches!(
        engine_id,
        "krea_2_raw" | "krea_2_turbo" | "krea_2_edit" | "krea_2_turbo_edit"
    ) || runtime_macos::providers::flux2::product_load::loads_packed_tier_unquantized(engine_id)
    {
        None
    } else {
        quant
    }
}

/// Fixed converted artifacts whose resolved precision is independent of a request preference.
/// Returning `Some` lets callers skip tier reconciliation entirely, avoiding both a false recipe
/// identity and a false downgrade event before the provider sees the load.
#[cfg(target_os = "macos")]
pub(super) fn fixed_mlx_artifact_quant(model_id: &str) -> Option<(Option<Quant>, Option<i64>)> {
    (model_id == "flux2_klein_9b_true_v2").then_some((None, None))
}

#[cfg(target_os = "macos")]
fn fixed_mlx_artifact_tier(model_id: &str) -> Option<&'static str> {
    (model_id == "flux2_klein_9b_true_v2").then_some("bf16")
}

/// Validate the API's fresh opaque resolution against this route's inference descriptor and attach
/// the exact prepared source receipt before any planner, fit gate, or provider loader sees the spec.
/// Default/absent selection is an exact no-op.
///
/// The wrapper is intentional typestate: production routes must explicitly consume the attached
/// spec before handing it to a planner or provider. This keeps route wiring compile-enforced while
/// behavior tests exercise the real resolution seam, without source-text assertions.
#[must_use = "a manifest text-encoder attachment must be consumed by the production route"]
#[derive(Debug)]
pub(super) struct ManifestTextEncoderAttachedSpec(LoadSpec);

impl ManifestTextEncoderAttachedSpec {
    pub(super) fn into_load_spec(self) -> LoadSpec {
        self.0
    }
}

impl std::ops::Deref for ManifestTextEncoderAttachedSpec {
    type Target = LoadSpec;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for ManifestTextEncoderAttachedSpec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub(super) fn attach_manifest_text_encoder(
    spec: LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<ManifestTextEncoderAttachedSpec> {
    crate::text_encoder_selection::prepare_selected_text_encoder(
        spec,
        engine_id,
        &request.advanced,
        &request.model_manifest_entry,
        settings,
    )
    .map(ManifestTextEncoderAttachedSpec)
}

/// Whether the request carries a non-default authored encoder id. Kept beside the attachment seam
/// so bespoke File routes make the same default/invalid-shape decision as registry routes.
pub(super) fn has_authored_text_encoder(request: &ImageRequest) -> WorkerResult<bool> {
    match request.advanced.get("textEncoderModel") {
        None => Ok(false),
        Some(Value::String(id)) if id == "default" => Ok(false),
        Some(Value::String(id)) if !id.trim().is_empty() => Ok(true),
        Some(_) => Err(WorkerError::InvalidPayload(
            "advanced.textEncoderModel must be a non-empty string option id returned by GET /api/v1/models"
                .to_owned(),
        )),
    }
}

/// Finalize a complete bespoke File-route spec. A selected encoder's contract export prepares every
/// configured File slot plus its shard/config/tokenizer receipt atomically; the default path keeps
/// the route's existing explicit tokens. Pre-resolved route tokens are compared against the
/// contract-prepared set so a mutation between dispatch resolution and attachment fails closed.
pub(super) fn prepare_manifest_text_encoder_with_file_pins(
    spec: LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
    pins: impl IntoIterator<Item = gen_core::PinnedWeightsFile>,
    label: &str,
) -> WorkerResult<LoadSpec> {
    let selected = has_authored_text_encoder(request)?;
    let pins = pins.into_iter().collect::<Vec<_>>();
    let mut attached_spec = attach_manifest_text_encoder(spec, engine_id, request, settings)?;
    if !selected {
        crate::paths::prepare_load_spec_with_file_pins(&mut attached_spec, pins, label)?;
        return Ok(attached_spec.into_load_spec());
    }

    for expected in pins {
        expected
            .ensure_unchanged()
            .map_err(|error| crate::classify_engine_error(label, error))?;
        let received = attached_spec
            .prepared_file_pin_for(expected.loader_path())
            .map_err(|error| crate::classify_engine_error(label, error))?;
        if received != Some(&expected) {
            return Err(WorkerError::InvalidPayload(format!(
                "{label}: source changed while the selected text encoder receipt was prepared: {}",
                expected.loader_path().display()
            )));
        }
    }
    Ok(attached_spec.into_load_spec())
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
pub(super) fn apply_declared_candle_image_load_shape(
    engine_id: &str,
    resolved_tier: Option<&str>,
    request_mode: &str,
    manifest: &JsonObject,
    spec: LoadSpec,
    sequential_selected: bool,
) -> LoadSpec {
    let mode = crate::memory_route_registry::MemoryRouteMode::from_request(request_mode);
    let spec = crate::memory_route_registry::evaluate_declared_candle_load_shape(
        engine_id,
        resolved_tier,
        mode,
        manifest,
        spec,
        sequential_selected,
    );
    crate::memory_route_registry::apply_registered_load_shape(
        crate::memory_route_registry::MemoryRouteBackend::Candle,
        engine_id,
        mode.unwrap_or(crate::memory_route_registry::MemoryRouteMode::TextToImage),
        spec,
        sequential_selected,
    )
}

/// Unchanged legacy shaper for bespoke Candle call sites that do not carry a complete manifest
/// declaration coordinate. Declaration-owned routes use [`apply_declared_candle_image_load_shape`]
/// only after assembling their exact route, tier, mode, source, components, and selected encoder.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn apply_candle_image_load_shape(engine_id: &str, spec: LoadSpec) -> LoadSpec {
    crate::memory_route_registry::apply_registered_load_shape(
        crate::memory_route_registry::MemoryRouteBackend::Candle,
        engine_id,
        crate::memory_route_registry::MemoryRouteMode::TextToImage,
        spec,
        false,
    )
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn apply_candle_qwen_load_shape(
    engine_id: &str,
    resolved_tier: &str,
    request_mode: &str,
    manifest: &JsonObject,
    spec: LoadSpec,
) -> LoadSpec {
    apply_declared_candle_image_load_shape(
        engine_id,
        Some(resolved_tier),
        request_mode,
        manifest,
        spec,
        false,
    )
}

/// The Candle shared lane's prefix-KV-cache surcharge for one request (epic 25899): priced from the
/// LINKED provider's descriptor and pricing function, or `None` for every provider without a cache
/// and for a request whose admitted block is not handed to ONE render at exactly this geometry and
/// reference set (`single_pass == false`: hires fix, native prompt batch).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_prefix_kv_cache_surcharge(
    engine_id: &str,
    width: u32,
    height: u32,
    reference_count: u32,
    negative_branch: bool,
    single_pass: bool,
) -> Option<u64> {
    if !single_pass {
        return None;
    }
    crate::inference_runtime::media_descriptor(engine_id).and_then(|descriptor| {
        crate::execution_planner::prefix_kv_cache_surcharge_bytes(
            engine_id,
            &descriptor.capabilities.execution,
            None,
            width,
            height,
            reference_count,
            negative_branch,
        )
    })
}

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/candle_prefix_kv_cache_pricing_tests.rs"]
mod candle_prefix_kv_cache_pricing_tests;

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/candle_image_load_shape_tests.rs"]
mod candle_image_load_shape_tests;

/// Finalize the manifest/provider load-shape intersection before pricing or loading a tier.
/// Shared by production and real-artifact admission tests so neither can force an unshipped strategy.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_mlx_load_policy(
    engine_id: &str,
    effective_tier: Option<&str>,
    declaration_mode: Option<crate::memory_route_registry::MemoryRouteMode>,
    manifest: &serde_json::Map<String, serde_json::Value>,
    mut spec: LoadSpec,
    declaration_context: crate::memory_route_registry::MemoryRouteRequestContext,
    plain_text_to_image: bool,
    supports_sequential_offload: bool,
) -> WorkerResult<LoadSpec> {
    spec = crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request(
        engine_id,
        effective_tier,
        declaration_mode,
        manifest,
        spec,
        declaration_context,
    );
    spec = crate::memory_route_registry::apply_declared_mlx_load_policy_for_request(
        engine_id,
        effective_tier,
        declaration_mode,
        manifest,
        spec,
        declaration_context,
    );
    if let Some(warning) = crate::memory_route_registry::mlx_load_shape_declaration_warning(&spec) {
        tracing::warn!(
            event = "mlx_load_shape_declaration_warning",
            provider = engine_id,
            ?warning,
            "provider refused deferred materialization; retaining the safe eager load path"
        );
    }
    if spec.load_shape_declaration_result == gen_core::LoadShapeDeclarationResult::NotEvaluated {
        spec = apply_measured_mlx_load_shape_for_request(engine_id, spec, plain_text_to_image);
    }
    if spec.offload_policy != gen_core::OffloadPolicy::Sequential {
        let outcome = crate::mlx_fit_gate::decide_residency_for_spec(engine_id, &spec);
        let deferred_can_stage = spec.load_shape_declaration_result
            == gen_core::LoadShapeDeclarationResult::Applied
            && spec.load_shape == gen_core::LoadShape::DeferredMaterialization
            && supports_sequential_offload;
        if deferred_can_stage && !matches!(outcome, crate::mlx_fit_gate::ResidencyOutcome::Resident)
        {
            spec = spec.with_offload_policy(gen_core::OffloadPolicy::Sequential);
            spec = crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request(
                engine_id,
                effective_tier,
                declaration_mode,
                manifest,
                spec,
                declaration_context,
            );
            spec = crate::memory_route_registry::apply_declared_mlx_load_policy_for_request(
                engine_id,
                effective_tier,
                declaration_mode,
                manifest,
                spec,
                declaration_context,
            );
        } else if matches!(outcome, crate::mlx_fit_gate::ResidencyOutcome::Sequential) {
            spec = crate::mlx_fit_gate::apply_residency_policy(spec, engine_id)?;
        }
    }
    Ok(spec)
}

/// The eager Resident counterpart of a spec that [`prepare_mlx_load_policy`] made Sequential only
/// because the manifest's declaration row requires it (sc-24806). `None` unless the policy step
/// introduced Sequential and the fit gate admits the eager spec Resident. The declaration stays
/// `Eligible`, so generic shapers keep ignoring it and only a fresh exact evaluation can apply it.
#[cfg(target_os = "macos")]
pub(crate) fn declared_sequential_resident_alternative(
    engine_id: &str,
    before: &LoadSpec,
    after: &LoadSpec,
) -> Option<LoadSpec> {
    if after.offload_policy != gen_core::OffloadPolicy::Sequential
        || before.offload_policy == gen_core::OffloadPolicy::Sequential
        || !matches!(
            after.load_shape_declaration_result,
            gen_core::LoadShapeDeclarationResult::Applied
                | gen_core::LoadShapeDeclarationResult::Eligible
        )
    {
        return None;
    }
    matches!(
        crate::mlx_fit_gate::decide_residency_for_spec(engine_id, before),
        crate::mlx_fit_gate::ResidencyOutcome::Resident
    )
    .then(|| before.clone().with_eligible_load_shape_declaration())
}

/// Whether an admitted request runs without whole-component staging or transformer block streaming.
#[cfg(target_os = "macos")]
fn admission_needs_no_staging(admission: &crate::mlx_fit_gate::MlxRequestAdmission) -> bool {
    matches!(
        admission,
        crate::mlx_fit_gate::MlxRequestAdmission::Admitted(evaluation)
            if !evaluation.memory.stage_residency && !evaluation.memory.stream_transformer_blocks
    )
}

/// sc-24806 for the MLX lanes that bind the declared load policy themselves (the base lane does
/// the same inside its tier chooser): keep `declared` unless both it and its eager Resident
/// counterpart are admitted for this request without staging, and the Resident load passes the
/// load gate. A failed pre-load probe keeps `declared`, whose own admission still runs at load.
#[cfg(target_os = "macos")]
pub(crate) async fn release_declared_sequential_for_request(
    engine_id: &'static str,
    model_id: &str,
    before: &LoadSpec,
    declared: LoadSpec,
    plan: crate::mlx_fit_gate::MlxRequestPlan,
    inputs: &crate::mlx_fit_gate::MlxRequestInputs,
    rebuild_plan: impl FnOnce(&LoadSpec) -> WorkerResult<crate::mlx_fit_gate::MlxRequestPlan>,
) -> WorkerResult<(LoadSpec, crate::mlx_fit_gate::MlxRequestPlan)> {
    let Some(resident) = declared_sequential_resident_alternative(engine_id, before, &declared)
    else {
        return Ok((declared, plan));
    };
    let budget = crate::generator_cache::mlx_tier_budget(engine_id).await?;
    let admits = |spec: &LoadSpec, plan: &crate::mlx_fit_gate::MlxRequestPlan| {
        crate::mlx_fit_gate::preflight_request(spec, plan, inputs, budget)
            .is_ok_and(|admission| admission_needs_no_staging(&admission))
    };
    if !admits(&declared, &plan) {
        return Ok((declared, plan));
    }
    let resident_plan = rebuild_plan(&resident)?;
    if admits(&resident, &resident_plan)
        && crate::mlx_fit_gate::preflight_load_rejection(engine_id, &resident).is_none()
    {
        tracing::info!(
            event = "mlx_declared_sequential_released_to_resident",
            engine = %engine_id,
            model = %model_id,
            "request needs no staging; loading Resident instead of the declared Sequential policy"
        );
        return Ok((resident, resident_plan));
    }
    Ok((declared, plan))
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
    let mode = if plain_text_to_image {
        crate::memory_route_registry::MemoryRouteMode::TextToImage
    } else {
        crate::memory_route_registry::MemoryRouteMode::EditImage
    };
    crate::memory_route_registry::apply_registered_load_shape(
        crate::memory_route_registry::MemoryRouteBackend::Mlx,
        engine_id,
        mode,
        spec,
        false,
    )
}

#[cfg(all(test, target_os = "macos"))]
#[path = "tests/measured_mlx_load_shape_tests.rs"]
mod measured_mlx_load_shape_tests;

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
    let components = crate::model_jobs::resolve_co_requisites_for_tier(
        &descriptor,
        &manifest_value,
        settings,
        tier,
    )?;
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

/// Stage the selected standalone decoder file without finalizing File identity. Imported routes use
/// this form while assembling their already-confined primary/control/adapter token set atomically.
fn attach_selected_decoder_unprepared(
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

/// Pin the selected decoder through the app-managed model-file trust boundary. The component id is
/// derived from the registered option rather than duplicated in SceneWorks.
fn selected_decoder_component_pin(
    spec: &LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<gen_core::PinnedWeightsFile>> {
    let Some(decoder_id) = requested_decoder_id(&request.advanced)? else {
        return Ok(None);
    };
    let option = validate_selected_decoder_request(engine_id, decoder_id, &request.advanced)?;
    let source = spec.components.get(option.component_id).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "decoder '{}' component '{}' was not staged",
            option.label, option.component_id
        ))
    })?;
    let WeightsSource::File(path) = source else {
        return Err(WorkerError::InvalidPayload(format!(
            "decoder '{}' component '{}' must resolve to one standalone file",
            option.label, option.component_id
        )));
    };
    crate::paths::pin_app_managed_model_file(
        settings,
        path,
        &format!("{} decoder weights", option.label),
    )
    .map(Some)
}

/// Stage and prepare the selected standalone decoder. Native generation remains runnable without
/// the soft co-requisite; a selected job retains the exact confined donor token and finalizes every
/// configured File identity before cache-key construction.
fn attach_selected_decoder(
    spec: LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    let mut spec = attach_selected_decoder_unprepared(spec, engine_id, request, settings)?;
    let Some(decoder_pin) = selected_decoder_component_pin(&spec, engine_id, request, settings)?
    else {
        return Ok(spec);
    };
    let decoder_path = decoder_pin.loader_path().to_path_buf();
    spec.set_prepared_file_pin(decoder_path, decoder_pin)
        .map_err(|error| {
            crate::classify_engine_error("Alternate decoder source preparation failed", error)
        })?;
    spec.prepare_file_sources().map_err(|error| {
        crate::classify_engine_error("Alternate decoder source preparation failed", error)
    })?;
    Ok(spec)
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
    emit_sampling_knob_unsupported(knob, json!(name), advertised, model_id, job_id, engine);
    None
}

/// The `sampling_knob_unsupported` worker event [`normalize_sampling_knob`] emits when it drops a
/// requested knob back to the engine default. Shared with the other per-request drops that remove a
/// value the engine would refuse (Iris-3B's CFG-bound negative prompt and schedule shift, sc-25679),
/// so every silent drop is observable through the one event.
pub(crate) fn emit_sampling_knob_unsupported(
    knob: &str,
    requested: Value,
    advertised: &[&str],
    model_id: &str,
    job_id: &str,
    engine: &str,
) {
    emit_event(
        "sampling_knob_unsupported",
        json!({
            "jobId": job_id,
            "engine": engine,
            "model": model_id,
            "knob": knob,
            "requested": requested,
            "supported": advertised,
        }),
    );
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
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
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
#[path = "tests/metrics_settings_tests.rs"]
mod metrics_settings_tests;

#[cfg(test)]
#[path = "tests/sampling_knob_tests.rs"]
mod sampling_knob_tests;

type PromptEnhancementReports =
    std::collections::HashMap<usize, (String, gen_core::PromptEnhancementReport)>;

fn record_prompt_enhancement_report(
    reports: &mut PromptEnhancementReports,
    enhancement_expected: bool,
    image_count: usize,
    index: usize,
    expected_prompt: String,
    report: gen_core::PromptEnhancementReport,
) -> WorkerResult<()> {
    if !enhancement_expected {
        return Err(WorkerError::Engine(
            "provider emitted a prompt-enhancement report for a disabled request".to_owned(),
        ));
    }
    if index >= image_count {
        return Err(WorkerError::Engine(
            "provider emitted a prompt-enhancement report for an unknown image".to_owned(),
        ));
    }
    match reports.entry(index) {
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert((expected_prompt, report));
        }
        std::collections::hash_map::Entry::Occupied(_) => {
            return Err(WorkerError::Engine(format!(
                "provider emitted duplicate prompt-enhancement reports for image {index}"
            )));
        }
    }
    Ok(())
}

fn take_prompt_enhancement_fact(
    reports: &mut PromptEnhancementReports,
    index: usize,
    model: &str,
) -> WorkerResult<Value> {
    let (expected_prompt, report) = reports.remove(&index).ok_or_else(|| {
        WorkerError::Engine(format!(
            "provider emitted no prompt-enhancement report for image {index}"
        ))
    })?;
    prompt_upsampling::prompt_enhancement_fact_for_model(report, &expected_prompt, model)
}

/// Generate one image (RGB8) at the given seed; `on_progress` streams denoise steps.
/// `guidance` is `None` for distilled variants (the engine rejects it on them).
///
/// `reference` is the optional identity img2img-init (sc-3619): `(image, strength)` adds a
/// `Reference` conditioning that seeds the denoise from the reference latents — the plain
/// (no-ControlNet) Z-Image reference-without-pose path, reusing the same engine img2img the
/// strict-pose tier already drives. `None` → plain txt2img. `enhance` carries the optional
/// caption-upsampling settings (sc-6135; only FLUX.2-dev acts on them).
///
/// Production renders go through [`generate_one_on_surface`]; this RGB-surface form is the one the
/// engine smokes drive.
#[cfg(all(test, target_os = "macos"))]
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
    prompt_enhancement: gen_core::PromptEnhancementSink,
    // Live denoise preview (epic 16624, sc-16904): forwarded frames reach the job's progress
    // stream. Inert for engines that don't emit; the default sink costs one branch per step.
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    generate_one_on_surface(
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
        prompt_enhancement,
        preview,
        &LaneRgbaSurface::default(),
        cancel,
        on_progress,
    )
}

/// The four-channel surface of one generic-lane render (sc-24111 / sc-24113 S4 contract).
///
/// `Default` is the historical three-channel lane, byte-identical: no reference carries alpha and
/// the request asks for `OutputChannels::Rgb`. Only a model whose descriptor advertises
/// `supports_alpha_output` / `ReferenceRgba` (Qwen-Image 2.1) ever gets a non-default value.
#[derive(Default)]
pub(crate) struct LaneRgbaSurface<'a> {
    /// The alpha plane of each `multi_references` entry, positionally; `None` (or a missing entry)
    /// is an ordinary RGB reference. An entry that carries one is sent as
    /// `Conditioning::ReferenceRgba`, UN-flattened, because the VAE encodes all four channels.
    pub(crate) multi_reference_alpha: &'a [Option<image::GrayImage>],
    /// The one trusted FLUX.3 owned-asset resolution used for admission and rendering. Its
    /// original RGB/RGBA carriers and optional size intent must not pass through img2img slots.
    pub(crate) flux3_references: Option<&'a flux3_references::ResolvedReferences>,
    /// Caption-only box controls, validated against those same owned reference slots before load.
    pub(crate) flux3_layout: Option<&'a flux3_references::ResolvedLayout>,
    /// `GenerationRequest::output_channels`. `Rgba` makes the provider answer with
    /// `GenerationOutput::ImagesRgba`, whose four channels the egress types as an RGBA PNG.
    pub(crate) output_channels: gen_core::OutputChannels,
    /// The request controls beyond the positional knobs (sc-25681): CFG interval and
    /// caption-overflow policy, resolved and refused by [`lane_controls`]. `Default` = the
    /// engine's own for every model that declares none.
    pub(crate) controls: lane_controls::LaneRequestControls,
    /// Where the render's non-fatal engine warnings (`GenerationReport::warnings`, e.g. Iris's
    /// `caption_truncated`) are collected for the image's provenance. `None` discards them, as
    /// the lane always has for providers that report none.
    pub(crate) warnings: Option<&'a std::cell::RefCell<Vec<gen_core::GenerationWarning>>>,
    /// Where the render's `GenerationReport::prefix_kv_cache` — the prefix-KV-cache path that
    /// actually RAN, its cache bytes and any shed-over-priced-bound overflow (epic 25899) — is
    /// collected for the image's provenance and the job record. `None` discards it; a provider
    /// with no cache reports none.
    pub(crate) prefix_kv_cache:
        Option<&'a std::cell::RefCell<Option<gen_core::PrefixKvCacheReport>>>,
    /// A native prompt batch (sc-25681): the request carries `prompt_batch` + `count` instead of
    /// one prompt, and the engine returns every image of the batch. `None` everywhere else.
    pub(crate) prompt_batch: Option<&'a lane_controls::NativePromptBatch>,
}

/// [`generate_one`] with an explicit [`LaneRgbaSurface`].
#[allow(clippy::too_many_arguments)]
fn generate_one_on_surface(
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
    prompt_enhancement: gen_core::PromptEnhancementSink,
    preview: gen_core::PreviewSink,
    surface: &LaneRgbaSurface<'_>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    render_on_surface(
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
        prompt_enhancement,
        preview,
        surface,
        cancel,
        on_progress,
    )?
    .pop()
    .ok_or_else(|| WorkerError::Engine("generator produced no image".to_owned()))
}

/// The one generic-lane engine call: builds the request (one `prompt`, or the surface's native
/// `prompt_batch` + `count`) and returns EVERY image the engine produced, in engine order.
#[allow(clippy::too_many_arguments)]
fn render_on_surface(
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
    prompt_enhancement: gen_core::PromptEnhancementSink,
    preview: gen_core::PreviewSink,
    surface: &LaneRgbaSurface<'_>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<Vec<(u32, u32, Vec<u8>)>> {
    let conditioning = if surface.flux3_references.is_some() {
        if reference.is_some() || !multi_references.is_empty() || edit_mask.is_some() {
            return Err(WorkerError::InvalidPayload(
                "FLUX.3 references must use the owned ordered conditioning list, not img2img or mask slots".to_owned(),
            ));
        }
        // The owned render seam below installs these carriers only after checking the admitted facts.
        Vec::new()
    } else {
        build_lane_conditioning_with_alpha(
            reference,
            multi_references,
            surface.multi_reference_alpha,
            edit_mask,
        )
    };
    // A native prompt batch is the prompt list ALONE (the engine refuses `prompt` beside it).
    let (prompt, prompt_batch, count) = match surface.prompt_batch {
        Some(batch) if batch.prompts.len() == 1 => {
            (batch.prompts[0].clone(), Vec::new(), batch.count)
        }
        Some(batch) => (String::new(), batch.prompts.clone(), batch.count),
        None => (prompt.to_owned(), Vec::new(), 1),
    };
    let mut request = GenerationRequest {
        prompt,
        prompt_batch,
        negative_prompt,
        width,
        height,
        count,
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
        flux3_reference_output: None,
        output_channels: surface.output_channels,
        cfg_interval: surface.controls.cfg_interval,
        caption_overflow: surface.controls.caption_overflow,
        preview,
        cancel: cancel.clone(),
        ..Default::default()
    };
    enhance.apply(&mut request, prompt_enhancement);
    if let Some(layout) = surface.flux3_layout {
        layout.apply(&mut request);
    }
    // `generate_with_report` (sc-25681): the default implementation is exactly `generate`, so every
    // provider that reports nothing is unaffected; one that reports (Iris's `caption_truncated`)
    // has its warnings collected for the image's provenance.
    let report = if let Some(references) = surface.flux3_references {
        flux3_references::render_reference_request(
            generator,
            &mut request,
            references,
            memory_strategy_context,
            on_progress,
        )
    } else {
        crate::memory_strategy::generate_with_scope_report(
            generator,
            &mut request,
            memory_strategy_context,
            on_progress,
        )
    }
    .map_err(|error| WorkerError::Engine(format!("generation failed: {error}")))?;
    if let Some(sink) = surface.warnings {
        sink.borrow_mut().extend(report.warnings);
    }
    if let Some(sink) = surface.prefix_kv_cache {
        *sink.borrow_mut() = report.prefix_kv_cache;
    }
    let output = report
        .output
        .ok_or_else(|| WorkerError::Engine("generator produced no output".to_owned()))?;
    match output {
        GenerationOutput::Images(images) => Ok(images
            .into_iter()
            .map(|image| (image.width, image.height, image.pixels))
            .collect()),
        // Only ever produced for an `OutputChannels::Rgba` request (the shared request floor refuses
        // one against a provider without `supports_alpha_output`). The flat interleaved RGBA buffer
        // flows on unchanged: `GeneratedPixels::from_engine_buffer` types it by its channel count.
        GenerationOutput::ImagesRgba(images) => Ok(images
            .into_iter()
            .map(|image| (image.width, image.height, image.pixels))
            .collect()),
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
        width: sceneworks_core::image_request::hires_fix_target_dimension(
            request.width,
            upscale_by,
        ),
        height: sceneworks_core::image_request::hires_fix_target_dimension(
            request.height,
            upscale_by,
        ),
        steps: request.hires_fix.effective_steps(first_pass_steps),
        guidance: first_pass_guidance.map(|_| cfg.unwrap_or_default()),
        true_cfg: first_pass_true_cfg.map(|_| cfg.unwrap_or_default()),
        provider_reference_strength,
    })
}

/// The conditioning one generic-lane render carries. Split out of [`generate_one`] so
/// [`lane_reference_count`] — the count the backend request scope grades the request against — can be
/// tested against the conditioning this lane REALLY sends rather than against a restatement of it.
#[cfg(test)]
fn build_lane_conditioning(
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    edit_mask: Option<&Image>,
) -> Vec<Conditioning> {
    build_lane_conditioning_with_alpha(reference, multi_references, &[], edit_mask)
}

/// [`build_lane_conditioning`] where the ordered `multi_references` may carry alpha planes
/// (Qwen-Image 2.1, sc-24110/sc-24113). With no alpha anywhere it is exactly
/// [`build_lane_conditioning`]; otherwise the ordered list is built per entry by
/// [`build_ordered_reference_conditioning`].
fn build_lane_conditioning_with_alpha(
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    multi_reference_alpha: &[Option<image::GrayImage>],
    edit_mask: Option<&Image>,
) -> Vec<Conditioning> {
    // `multi_references` (Boogu instruction edit, sc-7645) takes precedence when present: one image →
    // `Reference` (byte-identical to the single-reference path); 2–5 → `MultiReference`. Every other
    // family passes `&[]` and keeps the single `reference` (img2img init / IP-Adapter) path unchanged.
    let mut conditioning = if !multi_references.is_empty() {
        build_ordered_reference_conditioning(multi_references, multi_reference_alpha)
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
    context.mode = if reference_count == 0 {
        gen_core::MemoryMode::TextToImage
    } else {
        gen_core::MemoryMode::ImageToImage
    };
    context.has_phases = false;
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
    hires_first_pass_memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    hires_first_pass_memory_context: Option<&gen_core::MemoryRunContext>,
    enhance: &PromptEnhance,
    hires_fix: Option<HiresFixPlan>,
    preview: gen_core::PreviewSink,
    prompt_enhancement: gen_core::PromptEnhancementSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    generate_one_with_hires_on_surface(
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
        hires_first_pass_memory,
        memory_strategy_context,
        hires_first_pass_memory_context,
        enhance,
        hires_fix,
        preview,
        prompt_enhancement,
        &LaneRgbaSurface::default(),
        cancel,
        on_progress,
    )
}

/// [`generate_one_with_hires`] with an explicit [`LaneRgbaSurface`].
///
/// The disposable Hires.fix BASE pass always decodes RGB: it only exists to become the
/// three-channel img2img reference of the refinement pass. The surface's `output_channels` applies
/// to the pass that is persisted.
#[allow(clippy::too_many_arguments)]
fn generate_one_with_hires_on_surface(
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
    hires_first_pass_memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    hires_first_pass_memory_context: Option<&gen_core::MemoryRunContext>,
    enhance: &PromptEnhance,
    hires_fix: Option<HiresFixPlan>,
    preview: gen_core::PreviewSink,
    prompt_enhancement: gen_core::PromptEnhancementSink,
    surface: &LaneRgbaSurface<'_>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let Some(hires) = hires_fix else {
        return generate_one_on_surface(
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
            prompt_enhancement,
            preview,
            surface,
            cancel,
            on_progress,
        );
    };

    let first_pass_context = hires_first_pass_memory_context.cloned().or_else(|| {
        memory_strategy_context.map(|context| {
            hires_first_pass_context(
                context,
                width,
                height,
                lane_reference_count(
                    reference.is_some(),
                    multi_references.len(),
                    edit_mask.is_some(),
                ),
            )
        })
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
        Progress::PostProcessing => on_progress(Progress::PostProcessing),
        Progress::PromptEncodingComplete => on_progress(Progress::PromptEncodingComplete),
        Progress::PromptUpsamplingComplete => on_progress(Progress::PromptUpsamplingComplete),
        Progress::ConditioningComplete => on_progress(Progress::ConditioningComplete),
        Progress::Upsampling { tokens, max_tokens } => {
            on_progress(Progress::Upsampling { tokens, max_tokens })
        }
    };
    // Enhancement belongs to the final persisted pass. Running it on the disposable base pass
    // would produce two reports for one image and could feed two different prompts into one recipe.
    let base_surface = LaneRgbaSurface {
        multi_reference_alpha: surface.multi_reference_alpha,
        flux3_references: surface.flux3_references,
        flux3_layout: surface.flux3_layout,
        output_channels: gen_core::OutputChannels::Rgb,
        controls: surface.controls,
        warnings: surface.warnings,
        prefix_kv_cache: surface.prefix_kv_cache,
        // A hires-fix render is per image; the arms refuse a prompt batch with it.
        prompt_batch: None,
    };
    let (base_width, base_height, base_pixels) = generate_one_on_surface(
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
        hires_first_pass_memory.or(memory),
        first_pass_context.as_ref(),
        &PromptEnhance::default(),
        gen_core::PromptEnhancementSink::default(),
        preview.clone(),
        &base_surface,
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
        Progress::PostProcessing => on_progress(Progress::PostProcessing),
        Progress::PromptEncodingComplete => on_progress(Progress::PromptEncodingComplete),
        Progress::PromptUpsamplingComplete => on_progress(Progress::PromptUpsamplingComplete),
        Progress::ConditioningComplete => on_progress(Progress::ConditioningComplete),
        Progress::Upsampling { tokens, max_tokens } => {
            on_progress(Progress::Upsampling { tokens, max_tokens })
        }
    };
    let refine_surface = LaneRgbaSurface {
        multi_reference_alpha: &[],
        flux3_references: None,
        flux3_layout: None,
        output_channels: surface.output_channels,
        controls: surface.controls,
        warnings: surface.warnings,
        prefix_kv_cache: surface.prefix_kv_cache,
        // A hires-fix render is per image; the arms refuse a prompt batch with it.
        prompt_batch: None,
    };
    generate_one_on_surface(
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
        prompt_enhancement,
        preview,
        &refine_surface,
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
#[cfg(native_lane)]
pub(crate) fn load_reference_image(
    data_dir: &Path,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<Image> {
    // Every caller that existed before sc-24113 keeps upstream-parity truncation, byte-for-byte.
    load_reference_image_with(
        data_dir,
        project_id,
        asset_id,
        project_path,
        FlattenPolicy::Truncate,
    )
}

/// How a reference that CARRIES an alpha channel is reduced to the 3-channel `gen_core::Image`
/// every RGB-only engine takes (sc-24113).
///
/// A PARITY knob, not a quality one, and both answers are correct — for different upstream
/// pipelines:
///
/// * [`Truncate`](FlattenPolicy::Truncate) drops the fourth byte, which is what
///   `DynamicImage::to_rgb8()` does and what upstream `PIL.Image.convert("RGB")` does. Every edit
///   model SceneWorks shipped before 2.1 — SDXL inpaint, FLUX.2 edit, Kolors IP-adapter,
///   `qwen_image_edit_2511` — is compared against a reference implementation that truncates, so
///   this is the only answer that keeps them at parity, and it is the default for that reason.
/// * [`OverWhite`](FlattenPolicy::OverWhite) composites straight (un-premultiplied) alpha over an
///   opaque white backdrop, matching the S4 contract's `RgbaImage::to_rgb_over_white()`.
///   Qwen-Image 2.1 feeds its VISION tower the composited copy, so its reference path wants this.
///
/// ⚠️ The two are IDENTICAL on an opaque image (`A=255`) and differ MAXIMALLY on a transparent one,
/// because 2.1's alpha is straight: `A=0` does NOT zero RGB, so a transparent pixel keeps whatever
/// colour it was authored with. A blanket change of this default would therefore move every
/// existing model silently rather than loudly — which is exactly what the first cut of sc-24113 did.
///
/// Qwen-Image 2.1 does not flatten here at all: an alpha-carrying 2.1 reference travels UN-FLATTENED
/// as `Conditioning::ReferenceRgba` (its RGB via `Truncate`, which is the straight colour, plus its
/// own alpha plane), and the engine composites the vision tower's copy itself.
#[cfg(native_lane)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum FlattenPolicy {
    /// Drop the alpha byte — upstream `convert("RGB")` semantics. Every pre-2.1 lane.
    #[default]
    Truncate,
    /// Composite straight alpha over opaque white — the S4 `to_rgb_over_white()` semantics.
    ///
    /// Constructed only by tests: the one lane whose engine consumes a white composite
    /// (`qwen_image_2_1`'s vision tower) builds that copy engine-side from the un-flattened
    /// `Conditioning::ReferenceRgba`, so no production caller flattens over white. Kept so the
    /// parity decision stays stated and tested beside the default it contrasts with.
    #[allow(dead_code)]
    OverWhite,
}

/// [`load_reference_image`] with an explicit [`FlattenPolicy`].
#[cfg(native_lane)]
pub(crate) fn load_reference_image_with(
    data_dir: &Path,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
    flatten: FlattenPolicy,
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
    let decoded = crate::image_decode::decode_image_any(&path).map_err(|error| {
        WorkerError::InvalidPayload(format!("reference image {}: {error}", path.display()))
    })?;
    // sc-24113: how an alpha-carrying reference is flattened is the CALLER's choice, and the
    // default is upstream-parity truncation. See [`FlattenPolicy`] for why both answers are
    // correct and why the default must not move: `to_rgb8()` drops the fourth byte exactly as
    // `PIL.Image.convert("RGB")` does, which is what every pre-2.1 edit model is compared against.
    //
    // `OverWhite` composites straight alpha over an opaque white backdrop, matching the S4
    // contract's `RgbaImage::to_rgb_over_white()`. It matters only for an image that actually
    // carries alpha, and only where the transparency is real: `A=255` makes the two identical, and
    // `A=0` makes them maximally different because straight alpha leaves the hidden RGB intact.
    //
    // A non-alpha source takes `to_rgb8()` under either policy — the same call, the same bytes —
    // so nothing here can perturb an ordinary opaque reference.
    let rgb = match (flatten, decoded.color().has_alpha()) {
        (FlattenPolicy::OverWhite, true) => {
            let rgba = decoded.to_rgba8();
            let (width, height) = (rgba.width(), rgba.height());
            image::RgbImage::from_fn(width, height, |x, y| {
                let [r, g, b, a] = rgba.get_pixel(x, y).0;
                let over_white = |channel: u8| {
                    // Straight alpha over an opaque white backdrop, rounded half-up:
                    //   out = channel * a/255 + 255 * (1 - a/255)
                    let blended = channel as u32 * a as u32 + 255 * (255 - a as u32);
                    ((blended + 127) / 255) as u8
                };
                image::Rgb([over_white(r), over_white(g), over_white(b)])
            })
        }
        _ => decoded.to_rgb8(),
    };
    Ok(Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    })
}

/// The alpha plane of the same asset [`load_reference_image`] loads, or `None` when it has none
/// (sc-24111).
///
/// A deliberate second read rather than a widening of `load_reference_image`. That function's
/// return type is `gen_core::Image`, whose `pixels` is a flat 3-channel buffer — the engine-side
/// contract, which the inference half of this story owns and which this PR does not touch. The
/// product-side lanes that refine or rescale an image and then write it back as an asset still
/// have to preserve the channel, so the plane travels beside the engine image instead of inside
/// it, and is re-attached after the pass by `image_jobs::reattach_alpha`.
///
/// Reads the same path through the same `safe_project_path` confinement, so a poisoned sidecar
/// cannot reach a different file here than it does there.
#[cfg(native_lane)]
pub(crate) fn load_reference_alpha(
    data_dir: &Path,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<Option<image::GrayImage>> {
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
    let path = crate::safe_project_path(project_path, rel)?;
    let decoded = crate::image_decode::decode_image_any(&path).map_err(|error| {
        WorkerError::InvalidPayload(format!("reference image {}: {error}", path.display()))
    })?;
    Ok(split_alpha(&decoded).1)
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
/// `edit_image`-only [`should_fit_edit_source`] crop/pad-fit never applies to Krea Turbo's
/// reference-guided img2img surface).
///
/// Available to the candle lane too (sc-10134): the candle `generate_candle_stream` calls this to
/// resolve the img2img init off-Mac for every model the candle router admits on a generic img2img
/// lane, feeding the same `(image, strength)` into `generate_one`'s `reference` →
/// `Conditioning::Reference` → the engine's img2img entrypoint. That arm is model-agnostic; the
/// authoritative list of the ids the router actually wires lives beside the `generate_candle_stream`
/// call site, and this resolver serves all of them uniformly through [`model_supports_img2img`].
///
/// NOT every `ui.img2img` model, despite the flag being the gate here: FLUX.2 Klein
/// (`flux2_klein_9b` / `_kv` / `_true_v2`) also declares `ui.img2img`, but
/// `candle_reserves_reference_only_modes` hands its whole reference surface to the bespoke
/// `Flux2Edit` route, and `generate_candle_stream` fails closed on a Klein reference-bearing mode
/// before reaching this resolver — so Klein never takes the generic init.
#[cfg(native_lane)]
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
/// Available to the candle lane too (sc-10134): `generate_candle_stream` gates its GENERIC img2img
/// resolve on this same manifest flag off-Mac, each engine owning its img2img entrypoint. The flag is
/// necessary but not sufficient there — a model reaches that arm only if the candle router also wires
/// it an img2img lane (the wired id list is stated at the `generate_candle_stream` call site). FLUX.2
/// Klein declares `ui.img2img` yet has no such lane: its reference surface is reserved for the bespoke
/// `Flux2Edit` route, so it never reaches [`resolve_img2img_init_generic`] off-Mac.
#[cfg(native_lane)]
fn model_supports_img2img(request: &ImageRequest) -> bool {
    request
        .model_manifest_entry
        .get("ui")
        .and_then(|ui| ui.get("img2img"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Pure worker-side gate shared by the MLX and Candle generic conditioning resolvers. Scheduler
/// routing rejects unsupported plural/edit/control shapes before dispatch; this keeps the worker
/// defensive about the fields it consumes so neither backend silently treats an edit or a missing
/// reference as img2img.
#[cfg(native_lane)]
fn uses_generic_img2img(request: &ImageRequest, has_reference: bool) -> bool {
    model_supports_img2img(request) && has_reference && request.mode != "edit_image"
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
#[cfg(native_lane)]
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
#[cfg(native_lane)]
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
    match load_reference_image(
        &settings.data_dir,
        &request.project_id,
        &asset_id,
        project_path,
    ) {
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
#[cfg(native_lane)]
pub(crate) fn resolve_character_image_likeness_source(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> Option<(Image, String)> {
    if !character_image_likeness_requested(request) {
        return None;
    }
    resolve_control_identity_source(request, settings, project_path)
}

#[cfg(native_lane)]
fn character_image_likeness_requested(request: &ImageRequest) -> bool {
    request.mode == "character_image"
        && request
            .reference_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
        // Angle / pose sets are already scored by sc-4409 / sc-4410 through the shared seam.
        && pose_entries(request).is_empty()
        && !advanced::flag(&request.advanced, "angleSet")
}

/// img2img (Remix) strength for a plain Ideogram 4 edit with no mask — mirrors the sdxl/z-image 0.6
/// edit default and the engine's `DEFAULT_IMG2IMG_STRENGTH`. Shared by the macOS MLX edit path and the
/// candle in-lane edit (sc-6598), so it compiles off-Mac under `backend-candle` too.
#[cfg(native_lane)]
const IDEOGRAM_EDIT_STRENGTH: f32 = 0.6;
/// Heavier img2img strength for masked inpaint / outpaint (regenerate the painted region) — mirrors
/// the sdxl 0.85 inpaint default and the engine's `DEFAULT_INPAINT_STRENGTH`.
#[cfg(native_lane)]
const IDEOGRAM_INPAINT_STRENGTH: f32 = 0.85;

/// Upper bound on reference images for a Boogu instruction edit (sc-7645). The DiT's
/// `image_index_embedding` carries 5 per-image index slots (OmniGen2 lineage), so `N ∈ [1, 5]`
/// references can be packed into one edit (e.g. subject-from-A composed into scene-from-B); a plural
/// picker beyond that is capped here. Matches the `mlx-gen-boogu` / `candle-gen-boogu` `MAX_EDIT_REFS`.
#[cfg(native_lane)]
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

/// The ordered reference list where entries may carry their own alpha (S4 contract, sc-24111).
///
/// With no alpha anywhere this IS [`build_reference_conditioning`], byte for byte. Otherwise every
/// entry is emitted individually and in order — `Conditioning::Reference` for an RGB entry,
/// `Conditioning::ReferenceRgba` for one that carries alpha — because a `MultiReference` holds only
/// RGB images and flattening an alpha-carrying reference would send a DIFFERENT request (the VAE
/// encodes all four channels; the vision tower's white composite is engine-internal). The engine
/// numbers `Reference` / `ReferenceRgba` / `MultiReference` entries in one ordered sequence, so the
/// per-entry form keeps every reference's slot. Strength is `None`: upstream's condition images
/// have none, and a provider without img2img strength requires it unset.
fn build_ordered_reference_conditioning(
    references: &[Image],
    alpha: &[Option<image::GrayImage>],
) -> Vec<Conditioning> {
    if !alpha.iter().any(Option::is_some) {
        return build_reference_conditioning(references);
    }
    references
        .iter()
        .enumerate()
        .map(
            |(slot, image)| match alpha.get(slot).and_then(Option::as_ref) {
                None => Conditioning::Reference {
                    image: image.clone(),
                    strength: None,
                },
                Some(plane) => Conditioning::ReferenceRgba {
                    image: rgba_reference(image, plane),
                    strength: None,
                },
            },
        )
        .collect()
}

/// Interleave an RGB engine image with its (same-geometry) alpha plane into a straight-alpha
/// `RgbaImage`. The plane is decoded from the same asset as the image, so a geometry mismatch here
/// is a programming error; it is resampled defensively rather than misaligned.
fn rgba_reference(image: &Image, plane: &image::GrayImage) -> gen_core::RgbaImage {
    let resized;
    let plane = if plane.dimensions() == (image.width, image.height) {
        plane
    } else {
        resized = image::imageops::resize(
            plane,
            image.width,
            image.height,
            image::imageops::FilterType::Triangle,
        );
        &resized
    };
    let mut pixels = Vec::with_capacity(image.pixels.len() / 3 * 4);
    for (rgb, a) in image.pixels.as_chunks::<3>().0.iter().zip(plane.as_raw().iter()) {
        pixels.extend_from_slice(rgb);
        pixels.push(*a);
    }
    gen_core::RgbaImage {
        width: image.width,
        height: image.height,
        pixels,
    }
}

/// The inverse of [`rgba_reference`]: split a straight-alpha `RgbaImage` into the lane's RGB engine
/// image and its alpha plane, byte for byte.
fn split_rgba_reference(image: &gen_core::RgbaImage) -> WorkerResult<(Image, image::GrayImage)> {
    let mut rgb = Vec::with_capacity(image.pixels.len() / 4 * 3);
    let mut plane = Vec::with_capacity(image.pixels.len() / 4);
    for pixel in image.pixels.as_chunks::<4>().0 {
        rgb.extend_from_slice(&pixel[..3]);
        plane.push(pixel[3]);
    }
    let plane = image::GrayImage::from_raw(image.width, image.height, plane).ok_or_else(|| {
        WorkerError::InvalidPayload("RGBA reference buffer size mismatch".to_owned())
    })?;
    Ok((
        Image {
            width: image.width,
            height: image.height,
            pixels: rgb,
        },
        plane,
    ))
}

/// Reference asset ids for a Boogu instruction edit, in order. The multi-image picker sends the plural
/// `referenceAssetIds` — take all of them, capped at [`BOOGU_MAX_EDIT_REFERENCES`]; with no plural list
/// it falls back to the single Image-Edit `sourceAssetId` (`edit_image` mode). Mirrors
/// `edit_reference_ids`.
#[cfg(native_lane)]
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
#[cfg(native_lane)]
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

#[cfg(native_lane)]
fn is_mage_edit_model(model: &str) -> bool {
    matches!(
        model,
        "mage_flow_edit_base" | "mage_flow_edit" | "mage_flow_edit_turbo"
    )
}

/// One resolved Qwen-Image 2.1 condition image, with the alpha plane it arrived with (sc-24110).
///
/// The plane travels BESIDE the engine image rather than inside it because `gen_core::Image` is a
/// flat 3-channel buffer — the same reason sc-24111 carries it beside the render on the way out.
/// Which of the two conditioning kinds an entry becomes is decided from `alpha` alone, so the
/// classification is a property of the ASSET, never of its ordinal position.
#[cfg(native_lane)]
pub(crate) struct QwenImage21Reference {
    pub(crate) image: Image,
    /// `Some` iff the source asset carried an alpha channel. An RGBA reference is a DIFFERENT
    /// request from the same picture flattened over white (S4 contract), so this is never dropped
    /// silently — see [`build_qwen_image_2_1_conditioning`].
    pub(crate) alpha: Option<image::GrayImage>,
}

/// Is this the model whose edit route is ONE ordered conditioning list (sc-24110)?
///
/// Deliberately the two 2.1 engine ids (base and Turbo, sc-25907 — one request contract) rather
/// than a family predicate: the 2512-weights `qwen_image_edit*` ids are a different engine, a
/// different latent space and a different edit contract, and they keep the `ImageRoute::QwenEdit`
/// / `CandleImageRoute::QwenEdit` lanes they have always had.
#[cfg(native_lane)]
fn is_qwen_image_2_1_edit(request: &ImageRequest) -> bool {
    sceneworks_core::jobs_store::is_qwen_image_2_1_model(&request.model)
        && !qwen_image_2_1_reference_ids(request).is_empty()
}

/// The ORDERED condition-image asset ids of a Qwen-Image 2.1 edit, in the order the engine numbers
/// them. The worker-side twin of `sceneworks_core::jobs_store::qwen_image_2_1_reference_ids`, over
/// the parsed [`ImageRequest`] instead of the raw payload — and pinned against it by
/// `qwen_image_2_1_worker_and_router_agree_on_the_reference_order`, because a disagreement between
/// the two would mean the API validated one list and the worker rendered another.
///
/// Order: `sourceAssetId`, then `maskAssetId` (an ORDINARY reference — 2.1 has no mask tensor),
/// then `referenceAssetIds` in submitted order, then the singular `referenceAssetId`.
///
/// **MODE-INDEPENDENT**, matching the router exactly. A payload with no `mode` and a
/// `sourceAssetId`, or `mode: "image_generation"` with a `referenceAssetIds` list, is claimed by
/// the router as a conditioned request — so if this function consulted the mode it would return
/// empty for a job the router already admitted, and the references would be silently dropped into
/// a plain text-to-image render. There is no mode axis in the upstream contract at all: an empty
/// ordered list IS text-to-image and a non-empty one IS the edit call.
///
/// **DEDUPED by asset id, keeping the first occurrence.** The web's `editReferenceIds` leads
/// `referenceAssetIds` with the working image while `buildEditJobBody` also sets `sourceAssetId`,
/// so the ordinary Image-Editor payload names the same asset twice. Sending it twice is not a
/// harmless duplicate here: every entry occupies one of the ten slots and gets its own number in
/// the prompt template, so a duplicate silently costs a slot AND renumbers every reference after
/// it. First-occurrence wins because the earlier slot is the one the prompt refers to.
///
/// **No `.take(N)`.** Every other edit lane silently truncates an over-long set; here the API
/// refuses it with a 400 that names the cap, so a list that reaches the worker is already inside
/// the cap and truncating would only hide a routing defect.
#[cfg(native_lane)]
fn qwen_image_2_1_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !sceneworks_core::jobs_store::is_qwen_image_2_1_model(&request.model) {
        return Vec::new();
    }
    let scalar = |value: &Option<String>| -> Option<String> {
        value
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
    };
    let mut ids = Vec::with_capacity(2 + request.reference_asset_ids.len());
    ids.extend(scalar(&request.source_asset_id));
    ids.extend(scalar(&request.mask_asset_id));
    ids.extend(request.reference_asset_ids.iter().cloned());
    ids.extend(scalar(&request.reference_asset_id));
    let mut seen = std::collections::HashSet::with_capacity(ids.len());
    ids.retain(|id| seen.insert(id.clone()));
    ids
}

/// Resolve those ids into engine images at their NATIVE geometry, each with the alpha plane its
/// asset carried (same geometry as the RGB it belongs to).
///
/// Unlike the other registry editors, NOTHING here fits a reference to the output W×H (sc-24114).
/// Upstream never crops or letterboxes a condition image: the engine fits each reference itself,
/// aspect-preserved, onto its ~1024²-area 32-px grid (S3 contract), and that one resize feeds both
/// the vision tower and the VAE. A worker-side `fit_engine_image` to the OUTPUT aspect ran first
/// and so centre-cropped (default `fitMode: "crop"`) or black-letterboxed every portrait reference
/// in a landscape request before the engine ever saw it — a different request from upstream's.
#[cfg(native_lane)]
fn resolve_qwen_image_2_1_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Vec<QwenImage21Reference>> {
    let ids = qwen_image_2_1_reference_ids(request);
    // The ceiling comes from the job's own resolved manifest entry — the SAME `limits` key the
    // enqueue gate read — so the worker cannot hold a second opinion about what "too many" means,
    // and a manifest edit moves both. Absent means no cap, exactly as it does at enqueue.
    if let Some(cap) =
        sceneworks_core::video_request::image_max_reference_assets(&request.model_manifest_entry)
    {
        if ids.len() > cap {
            // Unreachable through the API, which refuses this at enqueue. Stated anyway because the
            // worker is also driven directly by tests and by jobs stored before the gate existed,
            // and because the alternative every sibling lane chose — a silent `.take(N)` — renders
            // a DIFFERENT request than the one asked for and reports success.
            return Err(WorkerError::InvalidPayload(format!(
                "qwen_image_2_1: {} reference images were supplied; upstream composes at most {cap}",
                ids.len()
            )));
        }
    }
    let mut references = Vec::with_capacity(ids.len());
    for id in &ids {
        // `FlattenPolicy::Truncate`, named EXPLICITLY rather than taken from the default
        // (sc-24110 answering the note on [`FlattenPolicy`]).
        //
        // Truncation drops the fourth byte and keeps the STRAIGHT RGB, which is exactly the colour
        // half of `Conditioning::ReferenceRgba`: an alpha-carrying reference travels UN-FLATTENED,
        // RGB + its own alpha plane, and the engine composites the vision-tower copy over white
        // itself (S4 contract). `OverWhite` would bake that composite into the colour the VAE
        // encodes — a different request. On an opaque reference the two policies are identical.
        let source = load_reference_image_with(
            &settings.data_dir,
            &request.project_id,
            id,
            project_path,
            FlattenPolicy::Truncate,
        )?;
        // sc-24111's lane, reused verbatim: the same asset, read through the same
        // `safe_project_path` confinement, for its alpha plane alone. Neither half is resampled,
        // so the plane is aligned with the RGB by construction (both are the decoded asset).
        let alpha =
            load_reference_alpha(&settings.data_dir, &request.project_id, id, project_path)?;
        references.push(QwenImage21Reference {
            image: source,
            alpha,
        });
    }
    Ok(references)
}

/// The ONE ordered conditioning list a Qwen-Image 2.1 edit sends, built from the resolved
/// references (sc-24110).
///
/// * No references → empty (the text-to-image call).
/// * All-RGB → exactly what every other registry editor sends: one `Conditioning::Reference` for a
///   single image, one `Conditioning::MultiReference` for many. Both kinds flatten into the same
///   ordered list engine-side, so the single case stays byte-identical to the one-reference path.
/// * **Never `Conditioning::Mask`.** The engine does not declare that kind and refuses it by name;
///   a mask asset is an ordinary ordered reference (see [`qwen_image_2_1_reference_ids`]). Such a
///   carrier reaches this route only from a DIRECT API CALLER OR WORKFLOW REPLAY — the Image
///   Editor gates its mask tool on `image_inpaint`, which this model does not declare.
/// * Strength is always `None` — upstream's condition images have no strength, and the engine
///   refuses anything but an unset-or-1.0 value.
///
/// * An alpha-carrying reference → `Conditioning::ReferenceRgba`, UN-flattened (S4 contract): the
///   VAE encodes all four channels, so flattening it would send a DIFFERENT request. A list with
///   any such entry is emitted per entry, in order ([`build_ordered_reference_conditioning`]).
#[cfg(native_lane)]
fn build_qwen_image_2_1_conditioning(
    references: &[QwenImage21Reference],
) -> WorkerResult<Vec<Conditioning>> {
    let images: Vec<Image> = references.iter().map(|entry| entry.image.clone()).collect();
    let alpha: Vec<Option<image::GrayImage>> =
        references.iter().map(|entry| entry.alpha.clone()).collect();
    let conditioning = build_ordered_reference_conditioning(&images, &alpha);
    // The per-entry classification is `qwen_alpha::reference_conditioning_kind` (sc-24113), so
    // both halves of the epic name the carrier the same way. Asserted rather than assumed: the
    // builder above and the classifier must never disagree about what an entry became.
    if conditioning.len() == references.len() {
        for (entry, built) in references.iter().zip(&conditioning) {
            let built_kind = match built {
                Conditioning::ReferenceRgba { .. } => {
                    crate::qwen_alpha::CONDITIONING_REFERENCE_RGBA
                }
                _ => "Reference",
            };
            if built_kind != crate::qwen_alpha::reference_conditioning_kind(entry.alpha.is_some()) {
                return Err(WorkerError::Engine(format!(
                    "qwen_image_2_1: reference conditioning disagrees with its classification \
                     ({built_kind})"
                )));
            }
        }
    }
    Ok(conditioning)
}

/// The ordered condition images a Qwen-Image 2.1 edit contributes to the generic lane's
/// `edit_refs` slot, with each one's alpha plane (`None` for an opaque asset) — resolved,
/// validated, and in request order.
///
/// [`build_qwen_image_2_1_conditioning`] is the ONE source of truth for what this route sends: its
/// answer is unpacked here into the lane's `edit_refs` slot (plus each entry's alpha plane), and
/// `build_lane_conditioning_with_alpha` re-wraps those into exactly that answer (`Reference` for
/// one opaque image, `MultiReference` for many, per-entry `Reference` / `ReferenceRgba` once any
/// entry carries alpha — pinned by `qwen_image_2_1_lane_conditioning_matches_the_dedicated_builder`
/// and the live-path tests). So the builder's shape is what reaches the engine, and a builder that
/// ever emitted anything else — a `Mask` above all — is refused here rather than silently dropped.
/// Running it here also means a refusal happens before any weights are loaded, next to the asset
/// reads that produced it, rather than as a render failure.
#[cfg(native_lane)]
fn resolve_qwen_image_2_1_edit_images(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<(Vec<Image>, Vec<Option<image::GrayImage>>)> {
    let references = resolve_qwen_image_2_1_edit(request, settings, project_path)?;
    let mut images = Vec::with_capacity(references.len());
    let mut alpha = Vec::with_capacity(references.len());
    for conditioning in build_qwen_image_2_1_conditioning(&references)? {
        match conditioning {
            Conditioning::Reference {
                image,
                strength: None,
            } => {
                images.push(image);
                alpha.push(None);
            }
            Conditioning::MultiReference { images: ordered } => {
                alpha.extend(ordered.iter().map(|_| None));
                images.extend(ordered);
            }
            // S4: an alpha-carrying reference, split back into the lane's RGB slot + its plane.
            Conditioning::ReferenceRgba {
                image,
                strength: None,
            } => {
                let (rgb, plane) = split_rgba_reference(&image)?;
                images.push(rgb);
                alpha.push(Some(plane));
            }
            other => {
                return Err(WorkerError::InvalidPayload(format!(
                    "qwen_image_2_1: the ordered conditioning list may carry only strength-less \
                     reference images, but the builder produced {other:?}"
                )))
            }
        }
    }
    Ok((images, alpha))
}

/// Everything the generic MLX lane conditions one render on, resolved once per job: the
/// per-family slots from [`resolve_generic_lane_conditioning`] plus the registry editors' ordered
/// `edit_refs` (Boogu, Mage, Qwen-Image 2.1).
///
/// Split out of [`generate_stream`] (sc-24110) so the conditioning a job REALLY sends is testable
/// without weights: [`generate_one`] assembles `build_lane_conditioning(identity_init, &edit_refs,
/// mask)` from exactly this tuple, so a test that resolves it for a real on-disk payload is looking
/// at the live path, not at a restatement of it.
#[cfg(target_os = "macos")]
#[allow(clippy::type_complexity)]
fn resolve_generic_lane_inputs(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
    has_reference: bool,
) -> WorkerResult<(LaneConditioning, Vec<Image>, Vec<Option<image::GrayImage>>)> {
    // Per-family reference conditioning (Z-Image identity/edit-init, FLUX.1/Kolors IP-Adapter,
    // Kolors img2img, Ideogram edit + mask), resolved once — same predicate order + per-family
    // values as the historical inline 5-way match, table-ized into one resolver (sc-8828, F-026).
    // The strict-pose ControlNet / edit tiers divert earlier in `resolve_image_route`.
    let lane = resolve_generic_lane_conditioning(request, settings, project_path, has_reference)?;
    // Registry instruction edits: Boogu resolves 1..5 sources; Mage resolves its required primary
    // source followed by every optional reference in client order. Both thread through
    // `generate_one` as `Reference` (one) / `MultiReference` (many), never the img2img-init slot.
    // The alpha plane of each `edit_refs` entry — only Qwen-Image 2.1 ever fills it (S4 contract).
    let mut edit_ref_alpha: Vec<Option<image::GrayImage>> = Vec::new();
    let edit_refs: Vec<Image> = if request.model == "boogu_image_edit" {
        resolve_boogu_edit(request, settings, project_path)?
    } else if is_mage_edit_model(&request.model) {
        resolve_mage_edit(request, settings, project_path)?
    } else if is_qwen_image_2_1_edit(request) {
        // Qwen-Image 2.1 reference / local editing (sc-24110): ONE ordered list of 1..=10 condition
        // images, source → mask → submitted references. No `ImageRoute` variant of its own — the id
        // is in MODEL_TABLE, so an edit lands on the generic `Mlx` arm exactly as Mage Edit does,
        // and the ordering + the never-a-Mask guarantee live in the resolver.
        let (images, alpha) = resolve_qwen_image_2_1_edit_images(request, settings, project_path)?;
        edit_ref_alpha = alpha;
        images
    } else {
        Vec::new()
    };
    // The never-a-Mask / never-a-strength guarantee, on the path that actually runs.
    guard_qwen_image_2_1_lane_slots(
        request,
        lane.identity_init.as_ref(),
        lane.ideogram_edit_mask.as_ref(),
    )?;
    Ok((lane, edit_refs, edit_ref_alpha))
}

/// Refuse a Qwen-Image 2.1 render whose generic-lane slots carry anything but the ordered
/// reference list (sc-24110).
///
/// The never-a-Mask guarantee has to hold on the path that actually RUNS. What `generate_one`
/// sends is [`build_lane_conditioning`]`(identity_init, &edit_refs, edit_mask)` — so
/// [`build_qwen_image_2_1_conditioning`] can state the contract perfectly and still be bypassed if
/// either of the other two slots is ever populated for this model. Today neither is
/// (`resolve_generic_lane_conditioning` is a per-family table and 2.1 is in none of its arms, and
/// the candle lane's `edit_reference` is likewise family-keyed), which is exactly the problem: it
/// holds by accident of a table this model is absent from, and adding an arm for it would silently
/// start sending the engine a `Conditioning::Mask` it refuses by name, or an img2img-init
/// `Reference` carrying a strength it also refuses.
///
/// So the invariant is ASSERTED at the seam rather than inferred. Erroring is the right answer
/// over quietly clearing the slots: a populated slot means some caller believes this model has an
/// img2img or inpaint surface, and it does not — upstream's pipeline takes only an ordered list of
/// condition images (S3 contract).
#[cfg(native_lane)]
fn guard_qwen_image_2_1_lane_slots(
    request: &ImageRequest,
    identity_init: Option<&(Image, f32)>,
    edit_mask: Option<&Image>,
) -> WorkerResult<()> {
    if !sceneworks_core::jobs_store::is_qwen_image_2_1_model(&request.model) {
        return Ok(());
    }
    if edit_mask.is_some() {
        return Err(WorkerError::InvalidPayload(
            "qwen_image_2_1: an inpaint mask reached the generic lane's mask slot, which would be \
             sent as Conditioning::Mask — a carrier this engine does not declare and refuses by \
             name. 2.1 has no mask tensor and performs no inpainting: draw the annotation into the \
             reference, or pass the mask as an ordinary ordered reference the prompt names."
                .to_owned(),
        ));
    }
    if identity_init.is_some() {
        return Err(WorkerError::InvalidPayload(
            "qwen_image_2_1: an img2img-init reference reached the generic lane's single-reference \
             slot, which would be sent with a strength. Upstream's condition images have no \
             strength and this engine refuses one; every 2.1 reference travels in the ordered \
             conditioning list instead."
                .to_owned(),
        ));
    }
    Ok(())
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
#[cfg(native_lane)]
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
        let source =
            load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
        let source = fit_engine_image(source, request.width, request.height, &request.fit_mode)?;
        references.push(source);
    }
    Ok(references)
}

#[cfg(native_lane)]
fn resolve_mage_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Vec<Image>> {
    let ids = mage_edit_reference_ids(request);
    let mut references = Vec::with_capacity(ids.len());
    for id in &ids {
        let source =
            load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
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
#[cfg(native_lane)]
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
#[cfg(native_lane)]
fn non_empty(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|id| !id.trim().is_empty())
}

/// The object-shaped `advanced.poses` entries (the strict-pose tier; empty otherwise).
#[cfg(native_lane)]
fn pose_entries(request: &ImageRequest) -> Vec<&Value> {
    request
        .advanced
        .get("poses")
        .and_then(Value::as_array)
        .map(|poses| poses.iter().filter(|pose| pose.is_object()).collect())
        .unwrap_or_default()
}

/// A pose's parsed keypoints, ready for [`crate::openpose_skeleton::draw_wholebody`].
// Shared strict-control preprocessing reads the whole body on both native backends.
#[cfg(native_lane)]
struct PoseInput {
    keypoints: Vec<crate::openpose_skeleton::Keypoint>,
    hands: Option<Vec<crate::openpose_skeleton::Hand>>,
    face: Option<Vec<crate::openpose_skeleton::Keypoint>>,
}

#[cfg(native_lane)]
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
#[cfg(native_lane)]
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
#[cfg(native_lane)]
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
#[cfg(native_lane)]
fn strip_base_prompt(base: &str) -> &str {
    base.trim().trim_end_matches([',', '.', ';'])
}

/// Append the per-angle clause to the user's base prompt (parity with
/// `augment_prompt_for_angle`). Empty base + unknown angle → empty string.
#[cfg(native_lane)]
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
    } else if uses_generic_img2img(request, has_reference) {
        // Generic plain-t2i img2img latent-init for any `ui.img2img` model (epic 8588 A4, sc-10189):
        // a `referenceAssetId` + `advanced.strength` seeds the denoise from the VAE-encoded reference,
        // which the engine routes to that model's img2img entrypoint via the single
        // `Conditioning::Reference`. Reached today by Krea 2 Turbo (sc-8591 #666), SD3.5
        // large/turbo/medium (sc-10189 #667), SANA base/Sprint (sc-18475), base `z_image` (sc-10265),
        // and Boogu / Ideogram 4 in a plain-t2i shape (Ideogram's bespoke arm above keys on
        // `edit_image`; Boogu's instruction editor is the separate `boogu_image_edit` id resolved
        // through `edit_refs`, so neither intercepts a t2i reference). A new text-only
        // model joins by flipping `ui.img2img` and landing its engine entrypoint. Sits after the
        // model-specific reference arms (z-image/flux/kolors/ideogram) so their bespoke surfaces keep
        // precedence; `z_image_turbo` reaches this same resolver from inside the z-image arm via
        // [`zimage_uses_generic_img2img`].
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

/// Historical whole-component fit, retained as a regression witness. Production tier selection
/// now evaluates the complete request ladder through `choose_mlx_request_tier`.
#[cfg(all(target_os = "macos", test))]
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

/// Complete immutable input to both pre-load tier selection and the eventual cached generation.
#[cfg(target_os = "macos")]
struct PreparedMlxImageTier {
    weights_dir: PathBuf,
    spec: LoadSpec,
    plan: crate::mlx_fit_gate::MlxRequestPlan,
    inputs: crate::mlx_fit_gate::MlxRequestInputs,
    /// The manifest declaration alone made this spec Sequential while the fit gate admits the
    /// eager Resident load (sc-24806); the tier chooser may re-prepare it Resident.
    has_resident_alternative: bool,
}

#[cfg(any(target_os = "macos", test))]
pub(crate) enum MlxTierFit<T> {
    Fits(T),
    TooBig(WorkerError),
}

/// Stop at the highest-fidelity fitting tier. Contract/source failures propagate; only an actual
/// memory refusal can try the next installed tier. Preserve the last evaluated diagnostic.
#[cfg(any(target_os = "macos", test))]
pub(crate) async fn choose_mlx_request_tier<
    T,
    F: std::future::Future<Output = WorkerResult<MlxTierFit<T>>>,
>(
    directories: Vec<PathBuf>,
    mut probe: impl FnMut(PathBuf) -> F,
) -> WorkerResult<T> {
    let mut last_rejection = None;
    for dir in directories {
        let tier = dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        match probe(dir).await? {
            MlxTierFit::Fits(prepared) => return Ok(prepared),
            MlxTierFit::TooBig(error) => {
                last_rejection = Some(WorkerError::InvalidPayload(format!(
                    "{} tier: {error}",
                    tier.as_deref().unwrap_or("resolved"),
                )));
            }
        }
    }
    Err(last_rejection.unwrap_or_else(|| {
        WorkerError::InvalidPayload("no installed MLX tier could be resolved".to_owned())
    }))
}

#[cfg(test)]
#[path = "tests/mlx_request_tier_tests.rs"]
mod mlx_request_tier_tests;

/// Reconcile a probe silently; only the winning tier emits the generation recipe/tier event.
#[cfg(target_os = "macos")]
fn mlx_candidate_quant(
    request: &ImageRequest,
    model: &ResolvedModel,
    dir: &Path,
) -> (Option<Quant>, Option<i64>) {
    if let Some(fixed) = fixed_mlx_artifact_quant(&request.model) {
        return fixed;
    }
    if !model.supports_quant() {
        return (None, None);
    }
    let requested = resolve_quant(request, Some(dir));
    match tier_quant_from_resolved_dir(&request.model_manifest_entry, dir) {
        Some((actual, bits)) => (
            if is_dense_te_tier(request) {
                requested.0
            } else {
                actual
            },
            bits,
        ),
        None => requested,
    }
}

/// Resolve the receipt of the candidate the ladder is about to score. Lower tiers remain untouched
/// if an earlier candidate fits, including when their receipts cannot be verified offline.
#[cfg(target_os = "macos")]
pub(crate) async fn ensure_mlx_candidate_provenance(
    request: &ImageRequest,
    settings: &Settings,
    repo: &str,
    directory: &Path,
) -> WorkerResult<()> {
    if model_path_override(request).is_some() {
        return Ok(());
    }
    let variant = tier_key_from_resolved_dir(directory)
        .map(str::to_owned)
        .or_else(|| requested_receipt_variant(request));
    crate::model_jobs::receipt_verification::ensure_huggingface_receipt_provenance(
        settings,
        repo,
        &request.model,
        variant.as_deref(),
        directory,
    )
    .await
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
    generate_stream_using(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        asset_writes,
        crate::inference_runtime::load,
    )
    .await
}

/// [`generate_stream`] with the generator loader supplied by the caller (the `_using` seam, see
/// [`start_cached_gen_stream_with_request_state_using`]): every pre-load decision, the cached
/// load, the per-item drive, and `consume_gen_events` are the production code; only the engine
/// is injectable, so a test can pin progress / cancel / failure on a real job arm without weights.
#[allow(clippy::too_many_arguments)]
#[cfg(target_os = "macos")]
async fn generate_stream_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<()> {
    if sceneworks_core::moss_image::is_moss(&plan.request.model) {
        return generate_moss_stream_using(
            api,
            settings,
            job,
            plan,
            project_path,
            asset_writes,
            load_generator,
        )
        .await;
    }
    let flux3_references = flux3_references::resolve_owned_references(
        &plan.request,
        &settings.data_dir,
        project_path,
    )?;
    let flux3_reference_count = flux3_references
        .as_ref()
        .map_or(0, |references| references.conditioning.len());
    let mut resolved_request = plan.request.clone();
    if let Some(references) = &flux3_references {
        (resolved_request.width, resolved_request.height) = references.plan.output.canvas;
        // Only numeric admission geometry is normalized. Preserve the authored UI mode and the
        // original optional-size intent; the memory route alone is derived from the references.
    }
    let flux3_layout =
        flux3_references::resolve_owned_layout(&resolved_request, flux3_references.as_ref())?;
    let request = &resolved_request;
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
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("model weights not found".to_owned()))?;
    // sc-26241: distilled FLUX 3 loads the package the request names — the dense bf16 snapshot
    // root or a locally converted MLX affine q4/q8 — never the generic resolver's q8 default and
    // never a request-time quantization of the dense package. `None` for every other model.
    let flux3_plan = flux3_precision::mlx_flux3_plan(
        request,
        settings,
        &weights_dir,
        model.descriptor.capabilities.supported_quants,
    )?;
    let weights_dir = flux3_plan
        .as_ref()
        .map_or(weights_dir, |plan| plan.weights_dir.clone());
    let adapters = resolve_adapters(request, settings)?;
    // Each adapter's identity (request id, path, sha256, scale, kind) for the image provenance,
    // joined after the render with the engine's own apply report (sc-25681).
    let adapter_identities =
        lane_controls::adapter_identities(api, settings, job, request, &adapters).await;
    // sc-3723: surface the descriptor-derived backend ("mlx" for every linked family today; a
    // future candle row would self-describe) over the gpu-id-derived label. Falls back to the
    // passed-in label only if a descriptor ever advertised an empty backend (never today).
    let backend = if model.backend().is_empty() {
        backend
    } else {
        model.backend()
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
    // The request controls beyond the positional knobs (sc-25681): CFG interval and caption-overflow
    // policy against the linked descriptor, plus the CFG-bound negative prompt
    // (`image.negativePromptRequiresGuidance`, Iris-3B) — each refused before load when the engine
    // does not honour it or CFG-off would make it inert, never dropped.
    let lane_controls = lane_controls::resolve_lane_request_controls(
        request,
        &model.descriptor.capabilities,
        guidance,
        negative_prompt.as_deref(),
    )?;
    let repo = model_repo(request, &model);
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
    // The per-family slots + the registry editors' ordered `edit_refs`, resolved once — see
    // `resolve_generic_lane_inputs`, which is also what the sc-24110 live-path tests drive.
    let (
        LaneConditioning {
            identity_init,
            flux_ip_dir,
            flux_true_cfg,
            ideogram_edit_mask,
        },
        edit_refs,
        edit_ref_alpha,
    ) = resolve_generic_lane_inputs(request, settings, project_path, has_reference)?;
    // The S4 four-channel surface (sc-24113): the transparency toggle resolved against the engine's
    // own `supports_alpha_output` (refused by name when it cannot serve it) and assigned onto the
    // request, plus each ordered reference's alpha plane (`edit_ref_alpha`). Default — RGB, no
    // alpha — for every model but Qwen-Image 2.1.
    let rgba_output_channels = crate::qwen_alpha::request_output_channels(
        crate::qwen_alpha::resolve_output_channels(&request.model, &request.advanced)?,
    );
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
    let enhance = PromptEnhance::from_advanced(&request.model, &request.advanced)?;
    // Per-generation PiD decode (epic 7840, sc-7849): resolve the PiD checkpoint + Gemma for this
    // model's latent space when `advanced.usePid` is set and the snapshots are cached; otherwise keep
    // the native VAE. `use_pid` and `spec.pid` stay in lockstep (the engine rejects a mismatch).
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();
    // A native prompt batch (sc-25681): ONE engine call over `prompts × count` images, refused
    // before load when the linked engine cannot render it or a per-image lane feature is in use.
    let native_batch = lane_controls::resolve_native_prompt_batch(
        request,
        &model.descriptor.capabilities,
        &[
            ("hires fix", hires_fix.is_some()),
            (
                "a reference image",
                flux3_reference_count > 0
                    || identity_init.is_some()
                    || !edit_refs.is_empty()
                    || ideogram_edit_mask.is_some(),
            ),
            ("prompt enhancement", enhance.enabled),
        ],
    )?;
    let native_batch_images = native_batch.as_ref().map_or(1, |batch| batch.images());
    // The images this job streams: `count` per-seed renders, or every image of the one batch call.
    let stream_total = if native_batch.is_some() {
        native_batch_images as usize
    } else {
        count
    };
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
    let quality_opt_in = crate::mlx_fit_gate::manifest_declares_decode_quality_policies(
        &request.model_manifest_entry,
    );
    let memory_budget = crate::generator_cache::mlx_tier_budget(engine_id).await?;
    // Resolve every tier through the exact production composition BEFORE scoring it. No candidate
    // loads tensors, and the winning spec/plan are retained rather than rebuilt after selection.
    let prepare = |weights_dir: PathBuf,
                   prefer_resident: bool|
     -> WorkerResult<PreparedMlxImageTier> {
        let (quant, quant_bits) = match flux3_plan.as_ref() {
            Some(plan) => plan.quant_and_bits(),
            None => mlx_candidate_quant(request, &model, &weights_dir),
        };
        // A split-repo snapshot root (sc-24112's `qwen_image_2_1` bf16) names its tier only
        // through the catalog, so ask for it first; `None` for every other family. A FLUX 3
        // package names its tier by identity (sc-26241).
        let effective_tier = flux3_plan
            .as_ref()
            .map(|plan| plan.tier.key())
            .or_else(|| split_repo_root_tier_key(&request.model_manifest_entry, &weights_dir))
            .or_else(|| {
                resolved_mlx_artifact_tier_for_model(&request.model, &weights_dir, quant_bits)
            });
        let resolved_artifact = if calibration_opt_in || quality_opt_in {
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
        #[cfg(target_os = "macos")]
        let load_quant = mlx_load_quant_for_resolved_artifact(engine_id, quant);
        #[cfg(not(target_os = "macos"))]
        let load_quant = quant;
        let mut spec = load_spec(
            weights_dir.clone(),
            load_quant,
            adapters.clone(),
            flux_ip_dir.clone(),
        );
        if let Some(pid) = &pid_weights {
            spec = spec.with_pid(pid.checkpoint.clone(), pid.gemma.clone());
        }
        // Named model components (epic 13657, sc-13682): stage a provider's caller-staged components (SDXL's
        // `tokenizer_clip_l` / `tokenizer_clip_bigg` / `vae_fp16_fix`) via the generic seam. Keyed on the
        // resolved `engine_id` (the DESCRIPTOR id) rather than `request.model`, so a finetune sibling that
        // shares one engine under a distinct catalog id resolves the same descriptor (media_descriptor matches
        // on descriptor.id). The MLX SDXL turnkey is self-contained (no `required_components`); Iris-3B
        // stages its Qwen3-VL `text_encoder` here (sc-25679).
        spec =
            attach_required_components(spec, engine_id, &request.model_manifest_entry, settings)?;
        spec = attach_prompt_upsampler(spec, engine_id, request, settings)?;
        if let Some(plan) = flux3_plan.as_ref() {
            spec = plan.attach_components(spec);
        }
        // F3 alternate decoder: attach before both the provider-specific memory contract and the generic
        // MLX fit gate, so donor bytes + normal activation/OS margin are admitted as one composition.
        spec = attach_selected_decoder(spec, engine_id, request, settings)?;
        // P9: a shared engine such as `sdxl` serves several independently pinned catalog routes. Bind
        // the exact resolved model id, independently resolved artifact tree, and running inference
        // implementation before any semantic quality row reaches the provider contract.
        spec = spec.with_resolved_route(request.model.clone());
        let plain_text_to_image =
            matches!(request.mode.as_str(), "image_generation" | "text_to_image")
                && flux3_references.is_none()
                && identity_init.is_none()
                && edit_refs.is_empty()
                && ideogram_edit_mask.is_none()
                && hires_fix.is_none();
        // Finalize caller-selected text-encoder state before asking the provider about the real
        // candidate. Chroma must see and reject an external encoder rather than being admitted against
        // an incomplete LoadSpec which is mutated afterward.
        let unattached_spec = spec;
        let attached_spec =
            attach_manifest_text_encoder(unattached_spec, engine_id, request, settings)?;
        let mut spec = attached_spec.into_load_spec();
        // SC-18457: provider adoption is an exact three-way intersection. A route-local BTR entry owns
        // the decision: the typed registry enforces mode/overlay/source semantics and the linked
        // provider must return BTR Implemented for this real deferred candidate. A refusal never falls
        // through to legacy shaping; only a manifest with no relevant BTR entry uses that unchanged
        // path. The tier is the resolved artifact tier, not the load-time quant field on the spec.
        let declaration_reference_count = if hires_fix.is_some() {
            hires_fix_reference_count()
        } else {
            lane_reference_count(
                identity_init.is_some(),
                edit_refs.len().max(flux3_reference_count),
                ideogram_edit_mask.is_some(),
            )
        };
        let reference_memory_mode = if flux3_references.is_some() {
            "edit_image"
        } else {
            &request.mode
        };
        let declaration_mode = crate::memory_route_registry::MemoryRouteMode::from_mlx_request(
            engine_id,
            reference_memory_mode,
        );
        let declaration_context = crate::memory_route_registry::MemoryRouteRequestContext {
            mode: declaration_mode
                .unwrap_or(crate::memory_route_registry::MemoryRouteMode::TextToImage),
            reference_count: declaration_reference_count,
            use_pid,
            has_phases: false,
        };
        let pre_policy_spec = spec.clone();
        spec = prepare_mlx_load_policy(
            engine_id,
            effective_tier,
            declaration_mode,
            &request.model_manifest_entry,
            spec,
            declaration_context,
            plain_text_to_image,
            model.descriptor.capabilities.supports_sequential_offload,
        )?;
        let resident_alternative =
            declared_sequential_resident_alternative(engine_id, &pre_policy_spec, &spec);
        let has_resident_alternative = resident_alternative.is_some();
        if let Some(resident) = resident_alternative.filter(|_| prefer_resident) {
            spec = resident;
        }
        let decode_quality_binding =
            crate::mlx_fit_gate::bind_decode_quality_policies_from_manifest(
                &request.model_manifest_entry,
                &request.model,
                resolved_artifact.as_ref(),
            )?;
        spec = crate::mlx_fit_gate::attach_decode_quality_binding(
            spec,
            decode_quality_binding,
            &request.model,
        );
        let mlx_request_plan = crate::mlx_fit_gate::MlxRequestPlan::for_spec_and_manifest(
            engine_id,
            &request.model,
            &spec,
            Some(&request.model_manifest_entry),
            resolved_artifact,
        );
        let mlx_request_plan = if matches!(
            engine_id,
            "krea_2_raw"
                | "krea_2_turbo"
                | "krea_2_edit"
                | "krea_2_turbo_edit"
                | "flux1_schnell"
                | "flux1_dev"
                | "flux2_klein_9b"
        ) {
            mlx_request_plan.with_resolved_artifact_tier(effective_tier)?
        } else {
            mlx_request_plan
        };
        let has_request_reference = flux3_reference_count > 0
            || identity_init.is_some()
            || !edit_refs.is_empty()
            || ideogram_edit_mask.is_some();
        // The admitted geometry describes the HEAVIEST pass: with hires fix that is the final
        // upscaled img2img refinement (one `Reference`, no mask), otherwise the single base pass. The
        // first hires pass renders at the base size with the caller's own conditioning and gets its own
        // request-scope identity inside `generate_one_with_hires`.
        let reference_count = declaration_reference_count;
        let mut memory_overlays = Vec::new();
        if has_request_reference {
            memory_overlays.push(format!(
                "references:{}",
                edit_refs.len().max(flux3_reference_count).max(1)
            ));
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
        let provider_overlay = crate::mlx_fit_gate::provider_overlay_for_load_spec(
            engine_id,
            &spec,
            (!memory_overlays.is_empty()).then(|| memory_overlays.join("+")),
        );
        let mlx_request_inputs = crate::mlx_fit_gate::MlxRequestInputs {
            native_batch: native_batch_images,
            width: memory_width,
            height: memory_height,
            count: request.count,
            mode: reference_memory_mode.to_owned(),
            overlay: provider_overlay,
            adapter_count,
            has_reference: reference_count > 0,
            reference_count,
            use_pid,
            has_phases: false,
            conditioning_windows: Some(crate::mlx_fit_gate::clip_windows_for_spec(
                engine_id,
                &spec,
                &request.prompt,
                &request.negative_prompt,
            )),
            // The prompt branches the engine will evaluate (epic 25899): a prefix-KV-cache provider
            // holds one cache per branch, so its surcharge doubles under true CFG.
            negative_branch: crate::execution_planner::evaluates_negative_branch(
                negative_prompt.as_deref().unwrap_or_default(),
                true_cfg,
                guidance,
            ),
            // A hires refinement hands this block to a first pass at the base size with the caller's
            // own references and re-resolved CFG, and a native prompt batch renders one prefix per
            // prompt in one pass: the cache is priced for neither render.
            prefix_kv_cache_eligible: hires_fix.is_none() && native_batch.is_none(),
            reference_resources: flux3_references
                .as_ref()
                .map(|references| references.resources.clone()),
        };
        Ok(PreparedMlxImageTier {
            weights_dir,
            spec,
            plan: mlx_request_plan,
            inputs: mlx_request_inputs,
            has_resident_alternative,
        })
    };
    let default_tier = tier_key_for_resolved_dir(&request.model_manifest_entry, &weights_dir);
    let explicit_pick = flux3_plan.is_some()
        || request
            .advanced
            .get("mlxQuantizeExplicit")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let tiers = default_tier
        .filter(|_| !explicit_pick)
        .map(|tier| downtier_candidate_tiers(request, settings, tier, min_quality_floor(request)))
        .unwrap_or_default();
    let directories = if tiers.is_empty() {
        vec![weights_dir.clone()]
    } else {
        tiers
            .iter()
            .filter_map(|tier| resolve_tier_dir(request, settings, tier))
            .collect()
    };
    let selected = choose_mlx_request_tier(directories, |dir| {
        let prepare = &prepare;
        let repo = &repo;
        async move {
        if calibration_opt_in || quality_opt_in {
            ensure_mlx_candidate_provenance(request, settings, repo, &dir).await?;
        }
        let mut prepared = prepare(dir.clone(), false)?;
        // Query/validate the provider before considering a legacy load refusal, so malformed
        // artifacts are never mistaken for a reason to silently select another tier.
        let mut admission = crate::mlx_fit_gate::preflight_request(
            &prepared.spec,
            &prepared.plan,
            &prepared.inputs,
            memory_budget,
        )?;
        // sc-24806: a manifest BTR/staging row binds Sequential even when this request needs no
        // staging or block streaming. Re-prepare the same tier eager + Resident and keep it only if
        // it is admitted without staging and passes the load gate; otherwise keep the declared spec.
        if prepared.has_resident_alternative && admission_needs_no_staging(&admission) {
            // A failed Resident probe keeps the admitted declared spec, as the shared helper does.
            let released = prepare(dir, true).ok().and_then(|resident| {
                let resident_admission = crate::mlx_fit_gate::preflight_request(
                    &resident.spec,
                    &resident.plan,
                    &resident.inputs,
                    memory_budget,
                )
                .ok()?;
                (admission_needs_no_staging(&resident_admission)
                    && crate::mlx_fit_gate::preflight_load_rejection(engine_id, &resident.spec)
                        .is_none())
                .then_some((resident, resident_admission))
            });
            if let Some((resident, resident_admission)) = released {
                tracing::info!(
                    event = "mlx_declared_sequential_released_to_resident",
                    engine = %engine_id,
                    model = %request.model,
                    "request needs no staging; loading Resident instead of the declared Sequential policy"
                );
                prepared = resident;
                admission = resident_admission;
            }
        }
        let admission = match admission {
            crate::mlx_fit_gate::MlxRequestAdmission::Admitted(evaluation) => {
                match crate::mlx_fit_gate::preflight_load_rejection(engine_id, &prepared.spec) {
                    Some(error) => crate::mlx_fit_gate::MlxRequestAdmission::Rejected(error),
                    None => crate::mlx_fit_gate::MlxRequestAdmission::Admitted(evaluation),
                }
            }
            rejected => rejected,
        };
        Ok(match admission {
            crate::mlx_fit_gate::MlxRequestAdmission::Admitted(_) => MlxTierFit::Fits(prepared),
            crate::mlx_fit_gate::MlxRequestAdmission::Rejected(error) => MlxTierFit::TooBig(error),
        })
        }
    }).await?;
    if selected.weights_dir != weights_dir {
        tracing::warn!(model = %request.model, from = ?default_tier,
            to = ?tier_key_for_resolved_dir(&request.model_manifest_entry, &selected.weights_dir),
            "MLX request ladder selected a lower installed tier");
    }
    let PreparedMlxImageTier {
        weights_dir,
        spec,
        plan: mlx_request_plan,
        inputs: mlx_request_inputs,
        has_resident_alternative: _,
    } = selected;
    // Descriptor-gated quant (mirrors the candle lane below): the MLX families advertise Q4/Q8
    // (`supported_quants`) and tolerate the Q8 default (a real quant on a dense convert, a no-op on an
    // already-packed turnkey). SANA joined this set in mlx-gen #654 (sc-8489): its descriptor now
    // advertises Q4/Q8 and its `load` ACCEPTS an advisory `spec.quantize` (the pre-quantized tier is
    // packed-detected from disk, #653), so it flows through the normal resolve_quant path like every
    // other matrix model. The `else` arm stays for any future engine that genuinely advertises no
    // quant — such a model loads dense.
    // True-V2's converter consumes the sole BF16 source and writes a dense BF16 transformer. It
    // has no packed tier matrix: fixing the pair before reconciliation means neither the historical
    // catalog default nor a crafted advanced preference can emit a false tier-change event or
    // relabel the declaration, fit, recipe, and provider identities.
    let fixed_artifact_quant = flux3_plan
        .as_ref()
        .map(|plan| plan.quant_and_bits())
        .or_else(|| fixed_mlx_artifact_quant(&request.model));
    let (quant, quant_bits) = if let Some(fixed) = fixed_artifact_quant {
        fixed
    } else if model.supports_quant() {
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
    let (_, quant_bits) = if fixed_artifact_quant.is_some() {
        (quant, quant_bits)
    } else if model.supports_quant() {
        let requested_for_reconcile = if is_dense_te_tier(request) {
            (None, dense_te_requested_tier_bits(request))
        } else {
            (quant, quant_bits)
        };
        reconcile_resolved_tier_quant(
            requested_for_reconcile,
            &weights_dir,
            &request.model_manifest_entry,
            !is_dense_te_tier(request),
            &request.model,
            &job.id,
            backend,
        )
    } else {
        (quant, quant_bits)
    };
    let raw_settings = mlx_raw_settings_for_bundle(
        request,
        Some(&weights_dir),
        &repo,
        steps,
        quant_bits,
        guidance.or(model_true_cfg),
    );

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

    // Keep the exact selected spec across the cache callback. Cache warm policy may change
    // residency, not the artifact/precision identity; reference demand is repriced on every fit.
    let reference_pricing_spec = flux3_references.as_ref().map(|_| spec.clone());
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state_using(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        load_generator,
        move |generator,
              cache_state,
              loaded_policy,
              warm_policy,
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
            // sc-18317: ONE warm hit is ONE decision, but this lane evaluates the request once per
            // item. Hand the real proposal to the first evaluation and an inert one to the rest, so a
            // multi-image job settles exactly one decision and emits exactly one event.
            let mut warm_policy = crate::execution_planner::WarmPolicyOnce::new(warm_policy);
            // A native prompt batch (sc-25681) is ONE engine call over every image of the job,
            // admitted against its `prompts × count` batch (`native_batch` above).
            if let Some(batch) = native_batch.as_ref() {
                return drive_native_prompt_batch(
                    tx,
                    seeds[0],
                    batch.images() as usize,
                    |preview, on_progress| {
                        let memory_evaluation =
                            if let Some(reference_spec) = reference_pricing_spec.as_ref() {
                                crate::mlx_fit_gate::evaluate_request_with_reference_spec(
                                    generator,
                                    &mlx_request_plan,
                                    &mlx_request_inputs,
                                    cache_state,
                                    loaded_policy.offload_policy,
                                    warm_policy.take(),
                                    request_external_committed_bytes,
                                    reference_spec,
                                )
                            } else {
                                crate::mlx_fit_gate::evaluate_request(
                                    generator,
                                    &mlx_request_plan,
                                    &mlx_request_inputs,
                                    cache_state,
                                    loaded_policy.offload_policy,
                                    warm_policy.take(),
                                    request_external_committed_bytes,
                                )
                            }?;
                        let _request_memory_limit = memory_evaluation
                            .process_limit_bytes
                            .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
                        let warnings = std::cell::RefCell::new(Vec::new());
                        let prefix_kv_cache_report = std::cell::RefCell::new(None);
                        let images = render_on_surface(
                            generator,
                            "",
                            width,
                            height,
                            seeds[0],
                            steps,
                            guidance,
                            negative_prompt.clone(),
                            None,
                            &[],
                            None,
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
                            gen_core::PromptEnhancementSink::default(),
                            preview,
                            &LaneRgbaSurface {
                                multi_reference_alpha: &[],
                                flux3_references: None,
                                flux3_layout: None,
                                output_channels: rgba_output_channels,
                                controls: lane_controls,
                                warnings: Some(&warnings),
                                prefix_kv_cache: Some(&prefix_kv_cache_report),
                                prompt_batch: Some(batch),
                            },
                            &cancel,
                            on_progress,
                        )?;
                        let facts = lane_controls::image_facts(
                            &warnings.take(),
                            &adapter_identities,
                            &generator.adapter_apply_reports(),
                            crate::execution_planner::prefix_kv_cache_record(
                                memory_evaluation.prefix_kv_cache,
                                prefix_kv_cache_report.take().as_ref(),
                            ),
                        );
                        Ok((images, facts))
                    },
                );
            }
            drive_gen_items_scored_reported(
                tx,
                seeds,
                move |_index, seed, preview, prompt_enhancement, on_progress| {
                    let memory_evaluation =
                        if let Some(reference_spec) = reference_pricing_spec.as_ref() {
                            crate::mlx_fit_gate::evaluate_request_with_reference_spec(
                                generator,
                                &mlx_request_plan,
                                &mlx_request_inputs,
                                cache_state,
                                loaded_policy.offload_policy,
                                warm_policy.take(),
                                request_external_committed_bytes,
                                reference_spec,
                            )
                        } else {
                            crate::mlx_fit_gate::evaluate_request(
                                generator,
                                &mlx_request_plan,
                                &mlx_request_inputs,
                                cache_state,
                                loaded_policy.offload_policy,
                                warm_policy.take(),
                                request_external_committed_bytes,
                            )
                        }?;
                    // Exact promoted MLX evidence may tighten the soft process limit for this request.
                    // The RAII guard restores the process-global/user limit after all retries and never
                    // touches the wired limit (#1947).
                    let _request_memory_limit = memory_evaluation
                        .process_limit_bytes
                        .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
                    // This image's non-fatal engine warnings (sc-25681), reported with its asset.
                    let warnings = std::cell::RefCell::new(Vec::new());
                    // ...and the prefix-KV-cache path its render reports (epic 25899).
                    let prefix_kv_cache_report = std::cell::RefCell::new(None);
                    let render = |seed: i64, on_progress: &mut dyn FnMut(Progress)| {
                        generate_one_with_hires_on_surface(
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
                            Some(memory_evaluation.memory),
                            Some(&memory_evaluation.context),
                            None,
                            &enhance,
                            hires_fix,
                            preview.clone(),
                            prompt_enhancement.for_prompt(&prompt),
                            &LaneRgbaSurface {
                                multi_reference_alpha: &edit_ref_alpha,
                                flux3_references: flux3_references.as_ref(),
                                flux3_layout: flux3_layout.as_ref(),
                                output_channels: rgba_output_channels,
                                controls: lane_controls,
                                warnings: Some(&warnings),
                                prefix_kv_cache: Some(&prefix_kv_cache_report),
                                prompt_batch: None,
                            },
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
                    // The image's provenance facts (sc-25681): engine warnings, and each adapter's
                    // identity joined with the engine's own applied/skipped report.
                    let mut facts = lane_controls::image_facts(
                        &warnings.take(),
                        &adapter_identities,
                        &generator.adapter_apply_reports(),
                        crate::execution_planner::prefix_kv_cache_record(
                            memory_evaluation.prefix_kv_cache,
                            prefix_kv_cache_report.take().as_ref(),
                        ),
                    )
                    .unwrap_or_default();
                    if let Some(references) = flux3_references.as_ref() {
                        facts.insert(
                            flux3_references::FACT_KEY.to_owned(),
                            references.image_fact((out_w, out_h)),
                        );
                    }
                    if let Some(layout) = flux3_layout.as_ref() {
                        facts.insert("flux3Layout".to_owned(), layout.image_fact());
                    }
                    if !facts.is_empty() {
                        prompt_enhancement.send_image_facts(facts);
                    }
                    Ok(Some((final_seed, out_w, out_h, pixels, face_likeness)))
                },
            )
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
        stream_total,
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
/// exception: the scheduler routes it to Candle, but [`prepare_candle_image_route`] sends it through
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
        // Qwen-Image 2.1 (sc-24109) is its OWN engine, not a newer build of the row above: a
        // different snapshot on a different latent space with a different text tower. It must
        // stamp its own adapter or a replayed asset cannot say which weights produced it.
        //
        // These labels are a SEPARATE namespace from the recipe's adapter ids: this one is the
        // per-asset `adapter` the candle stream writes (`candle_<family>`, the off-Mac sibling of
        // the MODEL_TABLE `mlx_<family>` labels), while the recipe stamp is
        // `sceneworks_core::contracts::RecipeAdapter::QwenImage21` ("qwen_image_2_1"). Both exist
        // for the same reason, and neither is derived from the other.
        "qwen_image_2_1" => "candle_qwen_2_1",
        // Turbo (sc-25907): the distilled weights stamp their own label for the same reason.
        "qwen_image_2_1_turbo" => "candle_qwen_2_1_turbo",
        "flux3_klein" | "flux3_klein_base" => "candle_flux3",
        "moss_image_base" | "moss_image_sft" => moss_adapter_label(model, "candle"),
        // Supra2-IMG (sc-24214): the off-Mac sibling of the `mlx_supra2` MODEL_TABLE label.
        "supra2_img" => "candle_supra2",
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
        // Iris-3B (sc-25680): the off-Mac sibling of the `mlx_iris` label.
        "iris_3b" => "candle_iris",
        // Bernini's production stream stamps this via its own route; the request-only fallback
        // (`adapter_id`) reaches it through here.
        "bernini_image" => CANDLE_BERNINI_IMAGE_ADAPTER,
        // sdxl / realvisxl share the candle "sdxl" engine.
        _ => CANDLE_ADAPTER,
    }
}

/// Whether the GB figure a candle reject just quoted can move with the output geometry — the
/// difference between naming a knob and sending the operator round a loop that cannot pass
/// (issue #3197).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolutionLever {
    /// The number is a per-tier manifest ROW read VERBATIM — `candle.vramGbByTier[tier]` or
    /// `candle.sequentialPeakGb[tier]`, plus [`crate::vram_gate::HEADROOM_GB`], stated once at
    /// `candle.vramMeasuredPixels`. [`crate::vram_gate::predicted_peak_gb`] and
    /// [`crate::vram_gate::predicted_sequential_peak_gb`] take no width and no height, so the
    /// verdict is identical at every output size: a retry at a smaller one re-derives the same GB
    /// and rejects again. This is the resolution-blind contract `vram_gate`'s own
    /// `video_peak_too_big_error` already documents (and already declines to print the advice
    /// against); the image lane kept offering the knob anyway, which is how #3197's reporter spent
    /// a 1024x1024 and a 768x768 attempt on byte-identical errors.
    Blind,
    /// The number came from a geometry-aware selector, so a smaller output can genuinely change the
    /// verdict — including by bringing an out-of-envelope request back inside the measured hull the
    /// shared memory-strategy ladder can price at all. Keep offering it.
    Priced,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl ResolutionLever {
    /// The resolution clause for a reject tail: the knob when it is real, otherwise the sentence
    /// that stops the retry. Saying it outright beats dropping the clause silently — the estimate
    /// reads as activation-shaped to anyone looking at it, so silence invites the very retry this
    /// explains away. The rule is "name a knob only when the gate priced it", never "don't mention
    /// resolution": the lane that CAN price a smaller geometry (Krea Turbo's calibrated ladder)
    /// resolves a concrete `krea_turbo_smaller_fit_with_runtime` target and offers that size.
    fn for_selector(selector_priced: bool) -> Self {
        if selector_priced {
            Self::Priced
        } else {
            Self::Blind
        }
    }

    fn clause(self) -> &'static str {
        match self {
            Self::Blind => {
                " Lowering the output resolution will not help: this gate prices each tier from a \
                 fixed per-tier peak that does not scale with the output size."
            }
            Self::Priced => "",
        }
    }

    /// The lever list that closes a reject tail, in the tail's two historical shapes: `joined`
    /// follows a comma (a smaller tier was just offered), `leading` opens the clause on its own.
    fn levers(self) -> (&'static str, &'static str) {
        match self {
            Self::Blind => (
                "or run on a GPU with more VRAM.",
                "run on a GPU with more VRAM.",
            ),
            Self::Priced => (
                "lower the output resolution, or run on a GPU with more VRAM.",
                "lower the output resolution or run on a GPU with more VRAM.",
            ),
        }
    }
}

/// The actionable tail for a candle VRAM-fit reject (sc-12090). Suggests only the tiers that are
/// actually INSTALLED and smaller than the rejected one (`installed_smaller`, descending fidelity),
/// else states that none is installed. Never names the rejected tier itself, and never points at the
/// quant picker — which is hidden by design when ≤1 tier is installed, exactly the case that produced
/// the misleading "Pick a lower tier (Q4/Q8)" text on a single-tier install. `lever` decides whether
/// the resolution knob is offered or explained away ([`ResolutionLever`], issue #3197).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn vram_reject_tail(installed_smaller: &[&str], lever: ResolutionLever) -> String {
    let (joined, leading) = lever.levers();
    if installed_smaller.is_empty() {
        return format!(
            "No smaller tier is installed — {leading}{note}",
            note = lever.clause(),
        );
    }
    format!(
        "Select a smaller installed tier ({tiers}), {joined}{note}",
        tiers = installed_smaller
            .iter()
            .map(|tier| tier.to_uppercase())
            .collect::<Vec<_>>()
            .join(" / "),
        note = lever.clause(),
    )
}

/// Build the candle reject advice for `rejected_tier` from the tiers the loader can actually resolve.
/// Kept pure so bespoke candle lanes can share the same truthfulness contract as the main image lane.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn vram_reject_tail_for_tier(
    installed: Vec<&'static str>,
    rejected_tier: &str,
    lever: ResolutionLever,
) -> String {
    let smaller: Vec<&'static str> = installed
        .into_iter()
        .filter(|candidate| tier_quality_rank(candidate) < tier_quality_rank(rejected_tier))
        .collect();
    vram_reject_tail(&smaller, lever)
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
    let needed =
        crate::vram_gate::predicted_peak_gb_with_adapter_bytes(manifest_entry, tier, adapter_bytes);
    match crate::vram_gate::resolve_offload(
        crate::vram_gate::fit_decision(needed, budget),
        sequential_capable,
    ) {
        crate::vram_gate::FitDecision::Fits | crate::vram_gate::FitDecision::Unknown => {
            TierFit::Fits
        }
        crate::vram_gate::FitDecision::Offload { available_gb, .. } => {
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

/// A generic SDXL request has a narrower, artifact-bound staging contract than its descriptor.
/// Bespoke edit/IP/detail lanes never reach this path. Keep the legacy tier verdict unless the
/// exact request selector has admitted the candidate; a declaration alone grants no saving.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_tier_fit_after_shared_selection(
    legacy: TierFit,
    selection: Option<&crate::candle_memory_strategy::CandleMemoryEvaluation>,
) -> TierFit {
    if selection.is_some() {
        TierFit::Fits
    } else {
        legacy
    }
}

/// FLUX.2-dev's caption-upsample lifecycle has exact bounded high-waters only for the Q4 and Q8
/// turnkey tiers. `predicted_peak_gb` intentionally falls back to `minMemoryGb` for legacy sparse
/// manifests, but doing that here would admit the unmeasured 113 GB BF16 tier at the 48 GB Q4 floor.
/// Keep this route-specific until another provider rotates its calibration identity and needs the
/// same evidence rule.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn validate_candle_tier_memory_evidence(
    model: &str,
    manifest_entry: &JsonObject,
    tier: &str,
) -> WorkerResult<()> {
    if model == "flux2_dev" && !has_candle_tier_peak_row(manifest_entry, tier) {
        return Err(WorkerError::InvalidPayload(format!(
            "{model} cannot run the resolved {tier} Candle tier because no current caption-aware \
             candle.vramGbByTier evidence exists for it; select Q4/Q8 or update SceneWorks"
        )));
    }
    Ok(())
}

/// Carry the legacy Candle fit gate's resident/staged choice into the request-scoped runtime
/// contract. The load-time policy remains populated for compatibility with older and non-image
/// providers, but current image providers use this bit as the sole lifecycle authority.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn apply_request_scoped_candle_residency(
    use_sequential: bool,
    memory: &mut Option<gen_core::GenerationMemory>,
) {
    if use_sequential {
        memory.get_or_insert_with(Default::default).stage_residency = true;
    }
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn rejects_unverified_shared_memory_fallback(
    engine_id: &str,
    optimized_strategy_selected: bool,
) -> bool {
    matches!(
        engine_id,
        "z_image"
            | "z_image_turbo"
            | "flux2_dev"
            | "flux2_klein_9b"
            | "chroma1_hd"
            | "chroma1_base"
            | "chroma1_flash"
    ) && !optimized_strategy_selected
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn verified_only_memory_family_label(engine_id: &str) -> Option<&'static str> {
    match engine_id {
        "z_image" | "z_image_turbo" => Some("Z-Image"),
        "flux2_dev" => Some("FLUX.2-dev"),
        "flux2_klein_9b" => Some("FLUX.2 Klein"),
        "chroma1_hd" | "chroma1_base" | "chroma1_flash" => Some("Chroma1"),
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

    #[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
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

    #[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
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

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
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
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
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
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
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
#[path = "tests/candle_request_residency_tests.rs"]
mod candle_request_residency_tests;

/// Whether a candle job may use Krea Turbo's request-scoped, quality-preserving memory ladder.
/// Keep every exclusion explicit: these surfaces have distinct component graphs or denoise contracts.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
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
    crate::vram_gate::KreaRuntimeEvidenceContext::inspect_turbo_fit(
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

/// Truthful capability-clamp rejection for the generic candle lane (sc-10733), reached once every
/// installed tier in `[floor, default]` has been scored by [`candle_tier_fit`] and none fits.
///
/// `smallest_tier` is the least-demanding tier that was evaluated, so a smaller tier is not an
/// escape. Whether a smaller output is one depends on `lever`: `candle_tier_fit` prices a manifest
/// row and never reads the request geometry ([`ResolutionLever::Blind`]), but the SDXL admission
/// path overrides that verdict with the shared selector, which prices the geometry
/// ([`ResolutionLever::Priced`]). Kept pure so the contract is testable without a GPU, exactly
/// like [`krea_turbo_capability_reject_message`] beside it.
///
/// It also says WHAT the number covers, in `video_peak_too_big_error`'s established spelling. #3197's
/// reporter read a ~14 GB demand against a tier Model Manager lists at 2.5 GiB and concluded the gate
/// was broken; those are a whole-render peak and an on-disk size, and the SDXL candle lane's own
/// `config/memory-anchors.json` rows (q4 at 10.87 GiB for 768x768, q8 at 13.78 GiB for 1024x1024,
/// decode-dominated) say the big one is real. A reject that quotes a number nobody can reconcile with
/// the model card produces the bug report instead of the answer.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_capability_reject_message(
    model: &str,
    smallest_tier: &str,
    needed_gb: f64,
    available_gb: f64,
    gpu_id: &str,
    lever: ResolutionLever,
) -> String {
    let (_, leading) = lever.levers();
    let mut action = leading.to_owned();
    action[..1].make_ascii_uppercase();
    format!(
        "{model} needs ~{needed} GB of VRAM even at the smallest installed tier it can run \
         ({smallest_tier}) — the peak of the whole render, the resident weights plus the denoise and \
         VAE-decode transient, not the tier's on-disk size — but GPU {gpu_id} has ~{available} GB \
         available. {action}{note}",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
        note = lever.clause(),
    )
}

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/krea_turbo_memory_route_tests.rs"]
mod krea_turbo_memory_route_tests;

/// Resolve the Candle load quant + recipe bit count from the FINAL tier the disk resolver (or the
/// capability clamp) selected. That tier is also handed to the shared memory evaluator, so the
/// provider's `LoadSpec` and request scope cannot describe different numeric artifacts.
///
/// Dense-TE turnkeys keep their full-precision text encoder while still recording the packed DiT's
/// resolved bits. Opaque/flat paths retain the request-derived behavior because there is no tier
/// basename to make a stronger artifact claim.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_resolved_tier_key(
    request: &ImageRequest,
    weights_dir: &Path,
    convrot_resolved: bool,
) -> &'static str {
    gate_tier_key(
        convrot_resolved,
        weights_dir,
        &request.advanced,
        &request.model_manifest_entry,
        nvfp4_selected(request, nvfp4_host_eligible(), Some(weights_dir)),
    )
}

/// Run the shared selector for an uncached, bespoke SDXL provider after every physical component
/// has been assembled. This is the only admission that can authorize the subsequent loader; the
/// older scalar gate remains useful telemetry but cannot mint an optimized execution plan.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[allow(clippy::too_many_arguments)]
pub(super) async fn admit_sdxl_bespoke_memory(
    request: &ImageRequest,
    settings: &Settings,
    mut spec: LoadSpec,
    request_mode: &str,
    width: u32,
    height: u32,
    reference_count: u32,
    use_pid: bool,
    additional_artifacts_certified: bool,
) -> WorkerResult<(
    crate::candle_memory_strategy::CandleMemoryEvaluation,
    LoadSpec,
)> {
    let WeightsSource::Dir(weights_dir) = &spec.weights else {
        return Err(WorkerError::InvalidPayload(
            "SDXL optimized admission requires a catalog snapshot directory".to_owned(),
        ));
    };
    let tier = candle_resolved_tier_key(request, weights_dir, false);
    spec = match tier {
        "q4" => spec.with_quant(Quant::Q4),
        "q8" => spec.with_quant(Quant::Q8),
        "bf16" => spec,
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "SDXL Candle route resolved unsupported physical tier {other:?}"
            )))
        }
    };
    spec = apply_declared_candle_image_load_shape(
        "sdxl",
        Some(tier),
        request_mode,
        &request.model_manifest_entry,
        spec,
        false,
    );
    spec.prepare_file_sources().map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "SDXL could not retain exact component/adapter receipts: {error}"
        ))
    })?;
    let contract = runtime_cuda::providers::sdxl::provider_contract_for_spec(&spec)
        .map_err(|error| WorkerError::InvalidPayload(format!("SDXL receipt refused: {error}")))?;
    let runtime_overlay_bytes = contract.asset_facts.overlay_bytes;
    let artifact_is_certified = additional_artifacts_certified
        && candle_certified_load_spec("sdxl", settings, &spec, &request.model_manifest_entry, tier);
    let budget = crate::vram_gate::live_capped_budget(&settings.gpu_id).await;
    let predicted_peak = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        &request.model_manifest_entry,
        tier,
        runtime_overlay_bytes,
    );
    let overlay = if spec.ip_adapter.is_some() {
        Some("identity")
    } else if spec.control.is_some() {
        Some("control")
    } else if !spec.adapters.is_empty() {
        Some("lora")
    } else {
        None
    };
    let evaluation = crate::candle_memory_strategy::evaluate_shared_bespoke_image(
        "sdxl",
        &request.model,
        &spec,
        artifact_is_certified,
        &request.model_manifest_entry,
        tier,
        request_mode,
        overlay,
        gen_core::MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count,
        },
        reference_count > 0,
        use_pid,
        false,
        budget,
        // This lane gates on the raw probe (no reclaimable credit), so the reserve derives from it.
        budget.map_or(0.0, crate::vram_gate::ladder_reserve_gb),
        predicted_peak,
        runtime_overlay_bytes,
        gen_core::MemoryCacheState::Cold,
        contract,
        crate::candle_memory_strategy::SDXL_REQUEST_EVIDENCE_REVISION,
    )?
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "SDXL {} {} at {tier} has no exact shared memory strategy before load",
            request.model, request_mode
        ))
    })?;
    Ok((evaluation, spec))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_quant_for_resolved_tier(
    request: &ImageRequest,
    tier: &str,
    weights_dir: &Path,
    supports_quant: bool,
    force_dense: bool,
) -> (Option<Quant>, Option<i64>) {
    if matches!(request.model.as_str(), "sana_1600m" | "sana_sprint_1600m") {
        return (None, None);
    }
    let resolved_bits = match tier {
        "bf16" => None,
        "q4" => Some(4),
        "q8" => Some(8),
        // Distinct identities such as NVFP4 and opaque modelPath roots keep the existing resolver,
        // but against the FINAL directory so NVFP4 cannot survive a fallback to another artifact.
        _ if !force_dense && supports_quant => return resolve_quant(request, Some(weights_dir)),
        _ => return (None, None),
    };
    if is_dense_te_tier(request) {
        return (None, resolved_bits);
    }
    // Two families reach here with physically packed q4/q8 transformer tensors that the provider
    // discovers from the selected directory rather than from `LoadSpec.quantize`:
    //   * the Chroma/SD3.5 turnkeys, detected from their exact header/config receipt — setting
    //     `quantize` would request a second on-the-fly quantization and the provider refuses it;
    //   * the pinned candle-gen-flux descriptor, which advertises `supported_quants: []` and
    //     rejects a nonempty `LoadSpec.quantize` outright.
    //   * sc-22732: the two Ideogram 4 routes. `candle-gen-ideogram`'s descriptor advertises
    //     `supported_quants: [Q4, Q8]` so this A/B tier toggle engages, but its exact directory
    //     route — `IdeogramLoadReceipt::capture` -> `validate_load_shape`, on the PRODUCTION
    //     `load`/`load_turbo` path, not a test-only one — refuses a spec whose `quantize` is
    //     `Some(_)` by name ("exact Ideogram directory route requires precision=Bf16 and
    //     quantize=None") and proves the tier off the packed safetensors headers instead. Sending
    //     the resolved quant therefore failed every candle Ideogram q4/q8 load at the loader; the
    //     artifact is the authority on the tier, so the load instruction stays empty and the
    //     resolved bits still reach the recipe.
    // Keep the load instruction empty while retaining the resolved artifact bits for the recipe
    // and later telemetry. This is deliberately narrower than `!supports_quant`: other descriptors
    // with an empty list are dense-only and must not acquire a packed-tier receipt.
    if matches!(
        request.model.as_str(),
        "chroma1_hd"
            | "chroma1_base"
            | "chroma1_flash"
            | "sd3_5_large"
            | "sd3_5_large_turbo"
            | "sd3_5_medium"
            | "flux_schnell"
            | "flux_dev"
            | "ideogram_4"
            | "ideogram_4_turbo"
    ) {
        return (None, resolved_bits);
    }
    if force_dense || !supports_quant {
        return (None, None);
    }
    match tier {
        "bf16" => (None, None),
        "q4" => (Some(Quant::Q4), Some(4)),
        "q8" => (Some(Quant::Q8), Some(8)),
        _ => unreachable!("resolved bits above accepts only bf16/q4/q8"),
    }
}

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/candle_resolved_tier_contract_tests.rs"]
mod candle_resolved_tier_contract_tests;

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

/// Everything the generic Candle lane conditions one render on, resolved once per job: the
/// single-reference (img2img-init) slot + its strength, the inpaint-mask slot, and the registry
/// editors' ordered `edit_refs` — the candle twin of `resolve_generic_lane_inputs`.
///
/// In-lane edit conditioning (sc-6598 Ideogram / sc-7524 Boogu): resolve the source `Reference`
/// (+ optional `Mask` for Ideogram) + strength once, seed-independent. Both families edit on the
/// SAME engine as their T2I (no separate bespoke stream), so the generic lane resolves the source
/// here. `resolve_ideogram_edit` / `resolve_boogu_edit` return `None` for a non-edit (T2I) job, and
/// each is gated to its family so a stray job reaching this generic lane is untouched. Boogu has no
/// mask (the `boogu_image_edit` descriptor accepts only `Reference`). Other candle edit families
/// (sdxl/flux2/qwen/z-image) have their own bespoke streams (checked before this dispatch).
///
/// Split out of [`generate_candle_stream`] (sc-24110) so the conditioning a job REALLY sends is
/// testable without weights — `generate_one` assembles `build_lane_conditioning` from exactly this
/// tuple.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[allow(clippy::type_complexity)]
fn resolve_candle_lane_inputs(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
    is_ideogram: bool,
) -> WorkerResult<(
    Option<(Image, f32)>,
    Option<Image>,
    Vec<Image>,
    Vec<Option<image::GrayImage>>,
)> {
    let (edit_reference, edit_mask) = if zimage_identity_candle_strength(request).is_some() {
        let reference_id = request
            .reference_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "Z-Image identity requires a referenceAssetId".to_owned(),
                )
            })?;
        let reference = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            reference_id,
            project_path,
        )?;
        let reference =
            fit_engine_image(reference, request.width, request.height, &request.fit_mode)?;
        (
            Some((
                reference,
                zimage_identity_candle_strength(request).expect("checked above"),
            )),
            None,
        )
    } else if is_ideogram {
        match resolve_ideogram_edit(request, settings, project_path)? {
            Some((source, strength, mask)) => (Some((source, strength)), mask),
            None => (None, None),
        }
    } else if matches!(request.model.as_str(), "z_image_turbo" | "z_image_edit") {
        // `z_image_edit` is a catalog alias for the registered Turbo provider. Resolve its source
        // into the generic request so memory admission, lifecycle cleanup, and telemetry stay shared.
        (
            resolve_zimage_edit_init(request, settings, project_path)?,
            None,
        )
    } else if request.model == "kolors" && request.mode == "edit_image" {
        (
            resolve_candle_kolors_edit_init(request, settings, project_path)?,
            None,
        )
    } else {
        (None, None)
    };
    // Registry instruction edits: resolve Boogu's 1..5 sources or Mage's source-first ordered list.
    // Each uses the `MultiReference`-capable path, not the single `edit_reference` img2img slot.
    // The alpha plane of each `edit_refs` entry — only Qwen-Image 2.1 ever fills it (S4 contract).
    let mut edit_ref_alpha: Vec<Option<image::GrayImage>> = Vec::new();
    let edit_refs: Vec<Image> = if request.model == "boogu_image_edit" {
        resolve_boogu_edit(request, settings, project_path)?
    } else if is_mage_edit_model(&request.model) {
        resolve_mage_edit(request, settings, project_path)?
    } else if is_qwen_image_2_1_edit(request) {
        // Qwen-Image 2.1 reference / local editing (sc-24110) — the SAME resolver as the MLX lane,
        // because the Candle port registers the same engine id and declares the same
        // `Reference` + `ReferenceRgba` + `MultiReference` conditioning. One request contract, two
        // backends.
        let (images, alpha) = resolve_qwen_image_2_1_edit_images(request, settings, project_path)?;
        edit_ref_alpha = alpha;
        images
    } else if is_sensenova_candle_model(&request.model)
        && matches!(request.mode.as_str(), "edit_image" | "character_image")
    {
        resolve_sensenova_candle_edit(request, settings, project_path)?
    } else {
        Vec::new()
    };
    // The never-a-Mask / never-a-strength guarantee, on the path that actually runs — the candle
    // twin of the MLX call. `edit_reference` is this lane's single-reference (img2img-init) slot
    // and `edit_mask` its mask slot; both must stay empty for 2.1.
    guard_qwen_image_2_1_lane_slots(request, edit_reference.as_ref(), edit_mask.as_ref())?;
    Ok((edit_reference, edit_mask, edit_refs, edit_ref_alpha))
}

/// Windows/CUDA registry-generator path (sc-3675 SDXL, generalized in sc-5096). This is the Candle
/// sibling of [`generate_stream`], driving the same neutral streaming harness
/// (`start_cached_gen_stream` → `generate_one` → `consume_gen_events`) for base generation and the
/// descriptor-backed reference/edit/identity shapes selected by [`resolve_candle_image_route`].
///
/// Backend-neutral resolution (sc-5096): the per-engine repo / steps / guidance / negative prompt all
/// come from the shared [`mlx_model`] join (`MODEL_TABLE` row + the linked candle descriptor), exactly
/// like the MLX path — so adding a family needs no new dispatch logic, just its provider crate linked.
/// Quant + LoRA/LoKr are **descriptor-gated** (sc-5126): resolved through the same `resolve_quant` /
/// `resolve_adapters` seams as MLX only when the linked Candle descriptor advertises them. Bespoke
/// control/IP/ComfyUI providers divert before this function; unsupported request shapes are refused
/// upstream and remain queued (`image_request_candle_eligible`). Reached only when
/// `backend_candle_enabled` (default off → production routing unchanged until parity).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    device_backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    generate_candle_stream_using(
        api,
        settings,
        job,
        plan,
        project_path,
        device_backend,
        asset_writes,
        crate::inference_runtime::load,
        Arc::new(NativeMossStreamRuntime),
    )
    .await
}

/// [`generate_candle_stream`] with the generator loader supplied by the caller — the Candle twin of
/// [`generate_stream_using`] (sc-25680): every pre-load decision, the cached load, the per-item drive
/// and `consume_gen_events` are the production code. Private MOSS source, admission and cache
/// dependencies permit CPU lifecycle tests through this dispatcher without a CUDA device.
#[allow(clippy::too_many_arguments)]
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    all(test, target_os = "macos")
))]
async fn generate_candle_stream_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    _device_backend: &str,
    asset_writes: &mut Vec<Value>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
    runtime: Arc<dyn MossStreamRuntime>,
) -> WorkerResult<()> {
    if sceneworks_core::moss_image::is_moss(&plan.request.model) {
        return generate_moss_stream_with_runtime(
            api,
            settings,
            job,
            plan,
            project_path,
            asset_writes,
            load_generator,
            runtime,
            "candle",
        )
        .await;
    }
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        generate_candle_registry_stream_using(
            api,
            settings,
            job,
            plan,
            project_path,
            _device_backend,
            asset_writes,
            load_generator,
        )
        .await
    }
    #[cfg(target_os = "macos")]
    {
        Err(WorkerError::Engine(
            "generic Candle registry stream is unavailable on macOS".into(),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_registry_stream_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    _device_backend: &str,
    asset_writes: &mut Vec<Value>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<()> {
    let flux3_references = flux3_references::resolve_owned_references(
        &plan.request,
        &settings.data_dir,
        project_path,
    )?;
    let flux3_reference_count = flux3_references
        .as_ref()
        .map_or(0, |references| references.conditioning.len());
    let mut resolved_request = plan.request.clone();
    if let Some(references) = &flux3_references {
        (resolved_request.width, resolved_request.height) = references.plan.output.canvas;
        // Only numeric admission geometry is normalized. Preserve the authored UI mode and the
        // original optional-size intent; the memory route alone is derived from the references.
    }
    let flux3_layout =
        flux3_references::resolve_owned_layout(&resolved_request, flux3_references.as_ref())?;
    let request = &resolved_request;

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
    let is_sana = matches!(engine_id, "sana_1600m" | "sana_sprint_1600m");
    let is_sd35 = crate::candle_memory_strategy::is_sd35(engine_id);
    // Chroma, Ideogram and Kolors are priced from the same exact provider receipts SANA and SD3.5
    // are, so their executed load policy is bound by the same authority (sc-20799). Before this
    // they fell through to `resolve_offload(fit_decision(..))`, a manifest estimate, and an
    // admitted STAGED peak could be executed as a RESIDENT load that the receipt never bounded.
    let receipt_priced_route = crate::candle_memory_strategy::is_receipt_priced(engine_id);
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
    } else if matches!(request.model.as_str(), "flux_schnell" | "flux_dev") {
        resolve_candle_flux1_packed_weights_dir(request, settings)?.ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Candle packed FLUX.1 weights snapshot not found for {}",
                request.model
            ))
        })?
    } else {
        resolve_weights_dir(request, settings)?.ok_or_else(|| {
            let repo = model_repo(request, &model);
            WorkerError::InvalidPayload(format!("candle weights snapshot not found for {repo}"))
        })?
    };
    // sc-26241: distilled FLUX 3 selects its package by the request alone — dense bf16 at the
    // snapshot root, a locally converted GGML q4/q8, or the publisher FP8R/NVFP4 at
    // `variants/<tier>` — with the root's bundled encoder/autoencoder as explicit components for
    // every non-dense package. `None` for every other model.
    let flux3_plan = flux3_precision::candle_flux3_plan(
        request,
        settings,
        &weights_dir,
        model.descriptor.capabilities.supported_quants,
    )?;
    if let Some(plan) = flux3_plan.as_ref() {
        weights_dir = plan.weights_dir.clone();
    }

    // Descriptor-derived denoise/guidance surface (distilled families → no guidance/negative; guided
    // families → the scale + negative prompt). Identical to the MLX path; quant + LoRA are omitted.
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let mut true_cfg = resolve_true_cfg(request, &model);
    let hires_fix = resolve_hires_fix_plan(request, steps, guidance, true_cfg);
    // The same request controls the MLX arm resolves (sc-25681): CFG interval, caption-overflow
    // policy and the CFG-bound negative prompt, each refused before load when the linked engine
    // does not honour it or CFG-off would make it inert. Inert for every model that sends none.
    let negative_prompt = resolve_negative_prompt(request, &model);
    let lane_controls = lane_controls::resolve_lane_request_controls(
        request,
        &model.descriptor.capabilities,
        guidance,
        negative_prompt.as_deref(),
    )?;

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

    // Adapters do not participate in tier identity. Quant is resolved only after every possible
    // weights-directory rewrite below, so the final LoadSpec and memory receipt share one tier.
    let adapters = if model.supports_adapters() {
        // Pinned inference applies Krea adapters as additive residuals over ConvRot projections too.
        resolve_adapters(request, settings)?
    } else {
        Vec::new()
    };
    let adapter_count = adapters.len();
    // Each adapter's identity for the image provenance, joined after the render with the engine's
    // own apply report (sc-25681) — the MLX arm's twin.
    let adapter_identities =
        lane_controls::adapter_identities(api, settings, job, request, &adapters).await;

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
    let work: Vec<(i64, String)> = if is_sensenova_candle_model(&request.model)
        && matches!(request.mode.as_str(), "edit_image" | "character_image")
    {
        candle_conditioned_edit_work(request)
    } else {
        (0..request.count as usize)
            .map(|index| (resolve_seed(request, index), prompt.clone()))
            .collect()
    };
    let total = work.len();
    // The single-reference + mask slots and the registry editors' ordered `edit_refs`, resolved once
    // — see `resolve_candle_lane_inputs`, which is also what the sc-24110 live-path test drives.
    let (edit_reference, edit_mask, edit_refs, edit_ref_alpha) =
        resolve_candle_lane_inputs(request, settings, project_path, is_ideogram)?;
    // The S4 four-channel surface (sc-24113): the transparency toggle resolved against the engine's
    // own `supports_alpha_output` (refused by name when it cannot serve it) and assigned onto the
    // request, plus each ordered reference's alpha plane (`edit_ref_alpha`). Default — RGB, no
    // alpha — for every model but Qwen-Image 2.1.
    let rgba_output_channels = crate::qwen_alpha::request_output_channels(
        crate::qwen_alpha::resolve_output_channels(&request.model, &request.advanced)?,
    );
    if is_sensenova_candle_model(&request.model) && !edit_refs.is_empty() {
        true_cfg = Some(resolve_sensenova_candle_true_cfg(request));
    }
    let has_reference = request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    // Generic img2img (reference-guided latent-init, sc-10134, epic 8588): a `ui.img2img` model in a
    // NON-edit mode carrying a `referenceAssetId` resolves to the img2img init `(image, advanced.strength)`,
    // threaded to `generate_one` as the single `Conditioning::Reference` the candle engine routes to its
    // img2img entrypoint (VAE-encode the reference → blend at `sigmas[init_time_step]` → denoise; CFG-free
    // for distilled families, two-forward CFG for the base ones like Krea Raw `render_base_img2img`,
    // sc-10226). Model-agnostic here — the candle router gates which ids reach this lane with a reference
    // (`krea_2_turbo`/`krea_2_raw`, SD3.5, SANA base/Sprint, Z-Image, Boogu, Ideogram all wired).
    // Disjoint from the Ideogram `edit_reference` (edit_image vs text_to_image) and the registry
    // editors' `edit_refs` (guarded here so a future overlap never double-drives the single slot).
    let img2img_reference = if edit_reference.is_none()
        && edit_refs.is_empty()
        && uses_generic_img2img(request, has_reference)
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
    // sc-18474: FLUX.2-dev prompt enhancement is native on Candle too. The request-local report
    // sink below is the proof seam: an enabled request cannot silently run the raw prompt without a
    // typed, honestly persisted fallback. The stale passthrough-only behavior is no longer allowed.
    let enhance = PromptEnhance::from_advanced(&request.model, &request.advanced)?;
    // Record the effective CFG knob (guidance for guided families, else true_cfg) + quant bits in the
    // recipe, so a Lens asset's sidecar reflects the Q4/Q8 it ran at (parity with the MLX path). The
    // recorded repo is the resolved model repo (the MLX turnkey the candle lane now packed-loads from,
    // sc-9092) — the same `model_repo` the MLX path records.
    let repo = model_repo(request, &model);
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
    // A native prompt batch (sc-25681): ONE engine call over `prompts × count` images, refused
    // before load when the linked engine cannot render it or a per-image lane feature is in use.
    let native_batch = lane_controls::resolve_native_prompt_batch(
        request,
        &model.descriptor.capabilities,
        &[
            ("hires fix", hires_fix.is_some()),
            (
                "a reference image",
                edit_reference.is_some()
                    || img2img_reference.is_some()
                    || !edit_refs.is_empty()
                    || edit_mask.is_some(),
            ),
            ("prompt enhancement", enhance.enabled),
        ],
    )?;
    // The batch renders every image in one call: the stream reports all of them.
    let total = native_batch
        .as_ref()
        .map_or(total, |batch| batch.images() as usize);
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
    //
    // sc-25684: a resident restoration transform (the Iris `upscaler/` task, sc-25683) is evicted by
    // the generator cache's load anyway — evict it BEFORE probing, so its VRAM is not counted as
    // occupied and a generation that fits once it is gone is not refused or demoted to staging
    // (the reclaimable credit below covers only a resident GENERATOR). A no-op when none is loaded.
    if crate::transform_cache::evict_resident_transform().await? {
        tracing::info!(
            event = "candle_generation_evicted_resident_transform",
            job_id = %job.id,
            model = %request.model,
        );
    }
    let raw_budget = crate::vram_gate::live_capped_budget(&settings.gpu_id).await;
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
    // sc-22664: the shared ladder's operational reserve is derived from the RAW probe, never from
    // the reclaimable-credited `budget` above — the credit predicts the free the imminent evict
    // will produce, so on a warm run `total − free` of the credited budget reads the card as
    // nearly idle and the reserve would collapse to the bare margin (`vram_gate::ladder_reserve_gb`).
    let shared_reserve_gb = raw_budget.map_or(0.0, crate::vram_gate::ladder_reserve_gb);
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
    // sc-12425: a resolved ConvRot load is named by its tier IDENTITY (see [`gate_tier_key`]) — it used
    // to be handed to the bits-derived `requested_tier_key`, which aliased it to q8 and under-gated it.
    // The comment above already knew "its footprint is neither the bf16 nor the q8 tier"; now the gate
    // acts on it. Extracted so that mapping has a unit test; this fn cannot be exercised from one.
    let mut tier = candle_resolved_tier_key(request, &weights_dir, convrot.is_some());
    if matches!(engine_id, "sana_1600m" | "sana_sprint_1600m") {
        tier = "bf16";
    }
    // FLUX 3's tier is the request-named package identity: neither the snapshot basename nor the
    // bits-ladder `q8` default the generic key derives for an unrecognized directory.
    if let Some(plan) = flux3_plan.as_ref() {
        tier = plan.tier.key();
    }
    validate_candle_tier_memory_evidence(&request.model, &request.model_manifest_entry, tier)?;
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
    // Ordinary SDXL reaches a request-scoped provider contract whose staging support is narrower
    // than the descriptor shared with its bespoke surfaces. Price each installed candidate with
    // that contract before the resident-only capability clamp can reject it. Other SDXL surfaces
    // retain their existing gates; no descriptor capability is broadened here.
    let sdxl_request_tier_admission = engine_id == "sdxl"
        && request.model == "sdxl"
        && matches!(request.mode.as_str(), "image_generation" | "text_to_image")
        && edit_reference.is_none()
        && img2img_reference.is_none()
        && edit_refs.is_empty()
        && edit_mask.is_none()
        && hires_fix.is_none()
        && !use_pid
        && native_batch.is_none();
    let sdxl_endpoints = if sdxl_request_tier_admission
        && request.count == 1
        && adapter_count == 0
        && flash_attn
        && lane_controls.cfg_interval.is_none()
        && guidance_method.is_none()
        && scheduler.is_none()
        && scheduler_shift.is_none()
    {
        sceneworks_core::memory_anchor::sdxl_cuda_memory_endpoints::packaged_sdxl_cuda_memory_endpoints()
            .filter(|endpoints| {
                endpoints.settings.get("steps").and_then(Value::as_u64) == Some(u64::from(steps))
                    && endpoints.settings.get("guidance").and_then(Value::as_f64)
                        == guidance.map(f64::from)
                    && endpoints.settings.get("flashAttn").and_then(Value::as_bool) == Some(true)
                    && sampler.as_deref().is_none_or(|sampler| {
                        endpoints.resolved_defaults.get("sampler").and_then(Value::as_str)
                            == Some(sampler)
                    })
            })
    } else {
        None
    };
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
    let explicit_pick = (matches!(request.model.as_str(), "flux_schnell" | "flux_dev")
        && request.advanced.contains_key("mlxQuantize"))
        || request
            .advanced
            .get("mlxQuantizeExplicit")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    // FLUX 3's package is request-named and is gated as named: rejected before load if it does
    // not fit, never downtiered to a package the request did not choose.
    if convrot.is_none() && !explicit_pick && flux3_plan.is_none() {
        let floor = capability_downtier_floor(krea_turbo_ladder, tier, min_quality_floor(request));
        let candidates: Vec<(&'static str, TierFit)> =
            downtier_candidate_tiers(request, settings, tier, floor)
                .into_iter()
                .map(|candidate| -> WorkerResult<_> {
                    let candidate_adapter_bytes =
                        candle_adapter_resident_bytes(engine_id, candidate, adapter_source_bytes);
                    let fit = if sdxl_request_tier_admission {
                        let legacy = candle_tier_fit(
                            &request.model_manifest_entry,
                            candidate,
                            budget,
                            sequential_capable,
                            candidate_adapter_bytes,
                        );
                        let selected = if let Some(dir) =
                            resolve_tier_dir(request, settings, candidate)
                        {
                            let (candidate_quant, _) = candle_quant_for_resolved_tier(
                                request,
                                candidate,
                                &dir,
                                model.supports_quant(),
                                false,
                            );
                            let spec = load_spec(dir, candidate_quant, adapters.clone(), None)
                                .with_resolved_route(request.model.clone());
                            let spec = attach_required_components(
                                spec,
                                engine_id,
                                &request.model_manifest_entry,
                                settings,
                            )?;
                            let spec =
                                attach_manifest_text_encoder(spec, engine_id, request, settings)?
                                    .into_load_spec();
                            let mut spec = apply_declared_candle_image_load_shape(
                                engine_id,
                                Some(candidate),
                                &request.mode,
                                &request.model_manifest_entry,
                                spec,
                                false,
                            );
                            let certified = candle_certified_load_spec(
                                engine_id,
                                settings,
                                &spec,
                                &request.model_manifest_entry,
                                candidate,
                            );
                            let candidate_endpoints = match sdxl_endpoints {
                                Some(endpoints) if certified && candidate == "q4" => {
                                    seal_sdxl_endpoint_load_spec(
                                        &mut spec,
                                        &endpoints.artifact.files,
                                    )?
                                    .then_some(endpoints)
                                }
                                _ => None,
                            };
                            let overlay_bytes = candidate_adapter_bytes.saturating_add(
                                lane_controls::declared_activation_overlay_bytes(
                                    request,
                                    None,
                                    memory_width,
                                    memory_height,
                                )?,
                            );
                            crate::candle_memory_strategy::evaluate_shared_image_priced(
                                engine_id,
                                &request.model,
                                &spec,
                                certified,
                                &request.model_manifest_entry,
                                candidate,
                                &request.mode,
                                (adapter_count > 0).then_some("lora"),
                                gen_core::MemoryGeometry {
                                    width: memory_width,
                                    height: memory_height,
                                    batch: 1,
                                    frames: 1,
                                    reference_count: 0,
                                },
                                false,
                                false,
                                false,
                                false,
                                budget,
                                shared_reserve_gb,
                                crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
                                    &request.model_manifest_entry,
                                    candidate,
                                    overlay_bytes,
                                ),
                                overlay_bytes,
                                if reclaimable_gb > 0.0 {
                                    gen_core::MemoryCacheState::Warm
                                } else {
                                    gen_core::MemoryCacheState::Cold
                                },
                                None,
                                candidate_endpoints,
                            )?
                        } else {
                            None
                        };
                        candle_tier_fit_after_shared_selection(legacy, selected.as_ref())
                    } else if krea_turbo_ladder && adapter_count == 0 {
                        let candidate_runtime = resolve_tier_dir(request, settings, candidate)
                            .and_then(|dir| {
                                krea_runtime_evidence_context(request, settings, candidate, &dir)
                            });
                        match crate::vram_gate::krea_turbo_fit_with_runtime(
                            &request.model_manifest_entry,
                            candidate,
                            memory_width,
                            memory_height,
                            budget,
                            shared_reserve_gb,
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
                    Ok((candidate, fit))
                })
                .collect::<WorkerResult<_>>()?;
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
                }
            }
            DowntierPick::Reject {
                tier: smallest,
                needed_gb,
                available_gb,
            } => {
                if krea_turbo_ladder && adapter_count == 0 {
                    let smallest_runtime =
                        resolve_tier_dir(request, settings, smallest).and_then(|dir| {
                            krea_runtime_evidence_context(request, settings, smallest, &dir)
                        });
                    let lower_resolution = crate::vram_gate::krea_turbo_smaller_fit_with_runtime(
                        &request.model_manifest_entry,
                        smallest,
                        memory_width,
                        memory_height,
                        budget,
                        shared_reserve_gb,
                        krea_allow_streamed_blocks,
                        smallest_runtime.as_ref(),
                    );
                    let installed_smaller: Vec<&'static str> =
                        installed_tier_keys(request, settings)
                            .into_iter()
                            .filter(|candidate| {
                                let candidate_runtime = resolve_tier_dir(
                                    request, settings, candidate,
                                )
                                .and_then(|dir| {
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
                                            shared_reserve_gb,
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
                return Err(WorkerError::InvalidPayload(
                    candle_capability_reject_message(
                        &request.model,
                        smallest,
                        needed_gb,
                        available_gb,
                        &settings.gpu_id,
                        // The SDXL admission path scores each candidate through the shared
                        // selector, which prices the request geometry: a smaller output can
                        // bring it inside what the measured endpoints cover.
                        ResolutionLever::for_selector(sdxl_request_tier_admission),
                    ),
                ));
            }
        }
    }
    // Reconcile only after the capability clamp has made its final directory/tier decision. The
    // resolved artifact drives recipe bits, MemoryNumericTier, active component floors, and the
    // provider begin-request check. `LoadSpec.quantize` remains descriptor-governed: in particular,
    // FLUX.1 packed-detects q4/q8 and therefore receives `None` while its receipt keeps the tier bits.
    let (quant, quant_bits) = match flux3_plan.as_ref() {
        // FP8R/NVFP4 are tier identities with no honest bit count on the q4/q8 ladder, so the
        // recipe's `mlxQuantize` bits stay empty and the load quant alone carries the format.
        Some(plan) => plan.quant_and_bits(),
        None => candle_quant_for_resolved_tier(
            request,
            tier,
            &weights_dir,
            model.supports_quant(),
            convrot.is_some() || is_anima_model(&request.model),
        ),
    };
    let mut raw_settings = mlx_raw_settings_for_bundle(
        request,
        Some(&weights_dir),
        &repo,
        steps,
        quant_bits,
        guidance.or(true_cfg),
    );
    // The declared pixel-space activations beyond the one 1024² image the floors carry — a native
    // prompt batch's extra images (sc-25681) and an output above 1024² (sc-25684) — ride every
    // resident/sequential prediction below exactly like resident adapter bytes do:
    // `candle.imageActivationGbPerMegapixel × (images × max(1, MP) − 1)`. Zero otherwise.
    let batch_overlay_bytes = lane_controls::declared_activation_overlay_bytes(
        request,
        native_batch.as_ref(),
        memory_width,
        memory_height,
    )?;
    let source_activation_bytes = crate::vram_gate::flux3_source_activation_allowance_bytes(
        engine_id,
        memory_width,
        memory_height,
    );
    let adapter_resident_bytes =
        candle_adapter_resident_bytes(engine_id, tier, adapter_source_bytes)
            .saturating_add(batch_overlay_bytes)
            .saturating_add(source_activation_bytes);
    let needed = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        &request.model_manifest_entry,
        tier,
        adapter_resident_bytes,
    );
    // Chroma's selector prices the exact provider receipt after the load spec is sealed. Passing
    // generic source-file adapter bytes here would double-charge adapters, omit PiD's materialized
    // student/Gemma footprint, and let the public `lora` cell stand in for load identity. Other
    // providers retain their established source-byte accounting.
    let provider_receipt_priced = matches!(
        engine_id,
        "flux3_klein"
            | "flux3_klein_base"
            | "chroma1_hd"
            | "chroma1_base"
            | "chroma1_flash"
            | "ideogram_4"
            | "ideogram_4_turbo"
            | "sana_1600m"
            | "sana_sprint_1600m"
            | "sd3_5_large"
            | "sd3_5_large_turbo"
            | "sd3_5_medium"
    );
    let shared_predicted_peak_gb = if provider_receipt_priced {
        let base = crate::vram_gate::predicted_peak_gb(&request.model_manifest_entry, tier);
        if matches!(engine_id, "flux3_klein" | "flux3_klein_base") {
            // Provider receipts replace only adapter/PUP weights. Keep source-derived activation
            // work exactly once in the base peak; never re-label it as adapter residency.
            base.map(|peak| {
                peak + batch_overlay_bytes.saturating_add(source_activation_bytes) as f64
                    / 1_073_741_824.0
            })
        } else {
            base
        }
    } else {
        needed
    };
    let shared_runtime_overlay_bytes = if provider_receipt_priced {
        0
    } else {
        adapter_resident_bytes
    };
    let krea_runtime_context = krea_turbo_ladder
        .then(|| krea_runtime_evidence_context(request, settings, tier, &weights_dir))
        .flatten();
    // sc-12090 AC#4/#5: the reject suggestions name only tiers actually INSTALLED and lower-fidelity than
    // the one being rejected — never the rejected tier, never the picker (hidden when ≤1 tier installed).
    // Reached only on the explicit-pick / ConvRot reject below (the downtier path already rejected above
    // when nothing smaller fits), where suggesting a smaller installed tier the user could pick is apt.
    let mut generation_memory: Option<gen_core::GenerationMemory> = None;
    let mut hires_first_pass_generation_memory: Option<gen_core::GenerationMemory> = None;
    let mut memory_strategy_selection: Option<gen_core::MemorySelection> = None;
    let mut selected_memory_strategy_context: Option<gen_core::MemoryRunContext> = None;
    let mut hires_first_pass_memory_context: Option<gen_core::MemoryRunContext> = None;
    let mut ideogram_warm_staged: Option<
        crate::candle_memory_strategy::CandleWarmStagedEvaluation,
    > = None;
    let mut ideogram_hires_first_warm_staged: Option<
        crate::candle_memory_strategy::CandleWarmStagedEvaluation,
    > = None;
    let mut adapted_peak_gb: Option<f64> = None;
    // Every adopting provider uses the same worker-owned selector. Static
    // Implemented/unverified declarations do not authorize optimized execution: this bridge always
    // submits the conservative resident estimate and adds deeper candidates only when an exact
    // authoritative record exists in the packaged evidence bundle.
    let mut shared_contract_spec = load_spec(weights_dir.clone(), quant, adapters.clone(), None)
        .with_resolved_route(request.model.clone());
    if let Some(pid) = pid_weights.as_ref() {
        shared_contract_spec =
            shared_contract_spec.with_pid(pid.checkpoint.clone(), pid.gemma.clone());
    }
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
    shared_contract_spec =
        attach_prompt_upsampler(shared_contract_spec, engine_id, request, settings)?;
    if let Some(plan) = flux3_plan.as_ref() {
        shared_contract_spec = plan.attach_components(shared_contract_spec);
    }
    if let Some((_, convrot_dit)) = convrot.as_ref() {
        shared_contract_spec = shared_contract_spec.with_component(
            gen_core::KREA_CONVROT_DIT_COMPONENT,
            WeightsSource::File(convrot_dit.clone()),
        );
    }
    // Export the selected encoder receipt only after the entire load shape is complete; this exact
    // prepared spec is reused by the selector and eventual provider/cache load.
    let unattached_shared_contract_spec = shared_contract_spec;
    let attached_shared_contract_spec = attach_manifest_text_encoder(
        unattached_shared_contract_spec,
        engine_id,
        request,
        settings,
    )?;
    let mut shared_contract_spec = attached_shared_contract_spec.into_load_spec();
    let shared_request_mode = candle_base_memory_request_mode(
        engine_id,
        if flux3_references.is_some() {
            "edit_image"
        } else {
            &request.mode
        },
    );
    // The numeric tier the shared selector binds this request to — the tier the provider LOADS,
    // which for a single-artifact dense route is not the gate's bits-derived key (see
    // `candle_shared_memory_tier`).
    let shared_tier = candle_shared_memory_tier(engine_id, tier);
    // A supported Hires refinement is a fresh one-reference image-to-image request even when the
    // caller's first pass was Edit or Inpaint. Admit that final-pass identity independently; the
    // original mode/reference/geometry is admitted below for the first pass.
    let shared_admission_mode = if hires_fix.is_some()
        && matches!(
            engine_id,
            "ideogram_4" | "ideogram_4_turbo" | "sana_1600m" | "sana_sprint_1600m" | "sdxl"
        ) {
        "image_to_image"
    } else {
        shared_request_mode
    };
    shared_contract_spec = apply_declared_candle_image_load_shape(
        engine_id,
        Some(shared_tier),
        shared_admission_mode,
        &request.model_manifest_entry,
        shared_contract_spec,
        false,
    );
    let first_pass_reference_count = lane_reference_count(
        edit_reference.is_some() || img2img_reference.is_some(),
        edit_refs.len().max(flux3_reference_count),
        edit_mask.is_some(),
    );
    let reference_count = shared_image_reference_count(
        edit_refs.len().max(flux3_reference_count),
        edit_reference.is_some() || img2img_reference.is_some(),
        edit_mask.is_some(),
        hires_fix.is_some(),
    )
    .saturating_add(
        u32::try_from(
            usize::from(shared_contract_spec.control.is_some())
                + shared_contract_spec.extra_controls.len(),
        )
        .unwrap_or(u32::MAX),
    );
    let artifact_is_certified = candle_certified_load_spec(
        engine_id,
        settings,
        &shared_contract_spec,
        &request.model_manifest_entry,
        shared_tier,
    );
    seal_recursive_receipt_load_spec(engine_id, artifact_is_certified, &mut shared_contract_spec)?;
    seal_ideogram_load_spec(engine_id, artifact_is_certified, &mut shared_contract_spec)?;
    let sdxl_endpoints = match sdxl_endpoints {
        Some(endpoints) if artifact_is_certified && shared_tier == "q4" => {
            seal_sdxl_endpoint_load_spec(&mut shared_contract_spec, &endpoints.artifact.files)?
                .then_some(endpoints)
        }
        _ => None,
    };
    // Epic 25899: the provider's prefix KV cache, priced from the linked provider for THIS request
    // (`None` for every provider without one — its admission is unchanged). The Candle surcharge
    // follows the build's production device width, not the tier.
    let prefix_kv_cache_surcharge = candle_prefix_kv_cache_surcharge(
        engine_id,
        memory_width,
        memory_height,
        reference_count,
        crate::execution_planner::evaluates_negative_branch(
            negative_prompt.as_deref().unwrap_or_default(),
            true_cfg,
            guidance,
        ),
        // A hires refinement is a second, differently shaped pass, and a native prompt batch holds
        // one prefix per prompt: the cache is priced for neither (the MLX lane's same rule,
        // `MlxRequestInputs::prefix_kv_cache_eligible`).
        hires_fix.is_none() && native_batch.is_none(),
    );
    let shared_memory = if let Some(references) = flux3_references.as_ref() {
        crate::candle_memory_strategy::evaluate_shared_image_priced_with_resources(
            engine_id,
            &request.model,
            &shared_contract_spec,
            artifact_is_certified,
            &request.model_manifest_entry,
            shared_tier,
            shared_admission_mode,
            (adapter_count > 0).then_some("lora"),
            gen_core::MemoryGeometry {
                width: memory_width,
                height: memory_height,
                // A native prompt batch (sc-25681) is ONE provider call over `prompts × count`
                // images, so that is the batch the request is admitted at — the provider's request
                // scope bounds the images it will render by exactly this figure (`candle-gen-iris`
                // refuses `prompts × count > MemoryGeometry::batch`). The batch's memory is charged
                // once, by the declared activation overlay above; the selector does not scale the
                // same activations by this batch again (`candle_memory_strategy::law_geometry`).
                batch: native_batch
                    .as_ref()
                    .map_or(1, lane_controls::NativePromptBatch::images),
                frames: 1,
                reference_count,
            },
            reference_count > 0,
            use_pid && hires_fix.is_none(),
            hires_fix.is_some(),
            hires_fix.is_some(),
            budget,
            shared_reserve_gb,
            shared_predicted_peak_gb,
            shared_runtime_overlay_bytes,
            if reclaimable_gb > 0.0 {
                gen_core::MemoryCacheState::Warm
            } else {
                gen_core::MemoryCacheState::Cold
            },
            prefix_kv_cache_surcharge,
            Some(&references.resources),
            sdxl_endpoints,
        )
    } else {
        crate::candle_memory_strategy::evaluate_shared_image_priced(
            engine_id,
            &request.model,
            &shared_contract_spec,
            artifact_is_certified,
            &request.model_manifest_entry,
            shared_tier,
            shared_admission_mode,
            (adapter_count > 0).then_some("lora"),
            gen_core::MemoryGeometry {
                width: memory_width,
                height: memory_height,
                // A native prompt batch (sc-25681) is ONE provider call over `prompts × count`
                // images, so that is the batch the request is admitted at — the provider's request
                // scope bounds the images it will render by exactly this figure (`candle-gen-iris`
                // refuses `prompts × count > MemoryGeometry::batch`). The batch's memory is charged
                // once, by the declared activation overlay above; the selector does not scale the
                // same activations by this batch again (`candle_memory_strategy::law_geometry`).
                batch: native_batch
                    .as_ref()
                    .map_or(1, lane_controls::NativePromptBatch::images),
                frames: 1,
                reference_count,
            },
            reference_count > 0,
            use_pid && hires_fix.is_none(),
            hires_fix.is_some(),
            hires_fix.is_some(),
            budget,
            shared_reserve_gb,
            shared_predicted_peak_gb,
            shared_runtime_overlay_bytes,
            if reclaimable_gb > 0.0 {
                gen_core::MemoryCacheState::Warm
            } else {
                gen_core::MemoryCacheState::Cold
            },
            prefix_kv_cache_surcharge,
            sdxl_endpoints,
        )
    }?;
    if flux3_references.is_some() && shared_memory.is_none() {
        return Err(WorkerError::InvalidPayload(
            "FLUX.3 references require a matching provider memory contract and an available device budget; this request cannot use legacy unpriced admission".to_owned(),
        ));
    }
    let first_pass_shared_memory = if hires_fix.is_some()
        && matches!(
            engine_id,
            "ideogram_4" | "ideogram_4_turbo" | "sana_1600m" | "sana_sprint_1600m" | "sdxl"
        ) {
        crate::candle_memory_strategy::evaluate_shared_image(
            engine_id,
            &request.model,
            &shared_contract_spec,
            artifact_is_certified,
            &request.model_manifest_entry,
            shared_tier,
            shared_request_mode,
            (adapter_count > 0).then_some("lora"),
            gen_core::MemoryGeometry {
                width,
                height,
                batch: 1,
                frames: 1,
                reference_count: first_pass_reference_count,
            },
            first_pass_reference_count > 0,
            use_pid,
            false,
            false,
            budget,
            shared_reserve_gb,
            shared_predicted_peak_gb,
            shared_runtime_overlay_bytes,
            if reclaimable_gb > 0.0 {
                gen_core::MemoryCacheState::Warm
            } else {
                gen_core::MemoryCacheState::Cold
            },
        )?
    } else {
        None
    };
    // The receipt-priced selection, kept even when it is Resident. `optimized_shared_memory_context`
    // below deliberately drops a Resident context for families outside its allowlist, so it cannot
    // answer "did the receipt price this request?" — and that is exactly the question the executed
    // load policy has to agree with.
    let mut receipt_priced_selection: Option<gen_core::MemoryStrategy> = None;
    // What the admission decided about the prefix KV cache, recorded with each image (epic 25899).
    let mut prefix_kv_cache_plan: Option<crate::execution_planner::PrefixKvCachePlan> = None;
    if let Some(evaluation) = shared_memory {
        prefix_kv_cache_plan = evaluation.prefix_kv_cache;
        if receipt_priced_route {
            receipt_priced_selection = Some(evaluation.context.selection.strategy);
        } else {
            // sc-22664 (epic 22657 E7): the shared ladder's selection, with the rung the selector
            // chose and the three phase peaks the law derived for it, so an OOM under this
            // selection is attributable to the exact estimate that admitted it. The receipt-priced
            // families emit their own exact-receipt event after the load (below) and are not
            // duplicated here.
            let telemetry = evaluation.selection_telemetry(engine_id, shared_tier);
            tracing::info!(
                event = "image_memory_strategy_selected",
                job_id = %job.id,
                model = %request.model,
                route = engine_id,
                actual_tier = shared_tier,
                strategy = crate::candle_memory_strategy::strategy_label(
                    evaluation.context.selection.strategy
                ),
                basis = evaluation.basis.as_key(),
                predicted_peak_bytes = evaluation.context.predicted_peak_bytes,
                conditioning_peak_bytes = evaluation.phase_peaks.map(|phases| phases.conditioning),
                denoise_peak_bytes = evaluation.phase_peaks.map(|phases| phases.denoise),
                decode_peak_bytes = evaluation.phase_peaks.map(|phases| phases.decode),
                admitted_peak_gb = evaluation.admitted.needed_gb,
                available_gb = evaluation.admitted.available_gb,
                reserve_gb = evaluation.admitted.reserve_gb,
                "shared memory-strategy ladder selected a candle strategy"
            );
            emit_event("image_memory_strategy_selected", telemetry);
        }
        memory_strategy_selection = Some(evaluation.context.selection);
        generation_memory = evaluation.memory;
        if sdxl_request_tier_admission
            && evaluation.context.selection.strategy == gen_core::MemoryStrategy::Resident
        {
            // The measured resident endpoint is literal: disable the provider's legacy automatic
            // decode fallback by carrying an explicit all-false request memory block.
            generation_memory = Some(gen_core::GenerationMemory::default());
        }
        adapted_peak_gb = Some(evaluation.predicted_peak_gb);
        ideogram_warm_staged = evaluation.warm_staged;
        // Resident is the selector's conservative sentinel, not authority to reconfigure a request.
        // In particular, a later legacy low-VRAM decision may choose sequential residency; carrying a
        // Resident scope would then overwrite that request memory back to resident in configure_request.
        selected_memory_strategy_context = if sdxl_request_tier_admission {
            Some(evaluation.context)
        } else {
            optimized_shared_memory_context(engine_id, evaluation.context)
        };
    }
    if let Some(evaluation) = first_pass_shared_memory {
        hires_first_pass_generation_memory = evaluation.memory;
        ideogram_hires_first_warm_staged = evaluation.warm_staged;
        hires_first_pass_memory_context =
            optimized_shared_memory_context(engine_id, evaluation.context);
    }
    if (is_sana || is_sd35)
        && (selected_memory_strategy_context.is_none()
            || (hires_fix.is_some() && hires_first_pass_memory_context.is_none()))
    {
        return Err(WorkerError::InvalidPayload(format!(
            "{} requires an exact receipt-priced pre-load memory selection for every render pass",
            request.model
        )));
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
                shared_reserve_gb,
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
    // Every receipt-priced family, not just SANA/SD3.5 (sc-20799). `receipt_priced_selection` is
    // Some exactly when the receipt produced a graded selection for this request, including a
    // Resident one — which is a real decision the load must honor, not an absence of one.
    let receipt_selector_authoritative = receipt_priced_selection.is_some();
    // The generic SDXL selector validates the exact artifact, request and composition, including
    // resident scratch-bounding rungs. The descriptor's deliberately broad staging flag and the
    // old geometry-free resident row must not veto that same decision after it passed the tier
    // chooser. Missing or refused selection still reaches the unchanged legacy resident gate.
    let request_selector_authoritative = receipt_selector_authoritative
        || (sdxl_request_tier_admission && memory_strategy_selection.is_some());
    // The plain gate's legacy rows are geometry-blind, but a request the shared selector priced
    // (or that SDXL admission scored against the measured endpoints) can pass at a smaller output;
    // only a request that never reached the selector is truly resolution-blind (issue #3197).
    let plain_gate_lever = ResolutionLever::for_selector(
        sdxl_request_tier_admission || memory_strategy_selection.is_some(),
    );
    let use_sequential = if request_selector_authoritative {
        generation_memory.is_some_and(|memory| memory.stage_residency)
    } else if let Some(
        fit @ crate::vram_gate::KreaTurboFit::Resident {
            peak_gb,
            needed_gb,
            selection,
            ..
        },
    ) = shared_krea_fit
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
        // sc-22667 (E7): the Krea lane's selection event, in the shared ladder's spelling.
        if let Some(telemetry) = fit.selection_telemetry(tier, memory_width, memory_height) {
            emit_event("image_memory_strategy_selected", telemetry);
        }
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
                (Some(needed_gb), Some(budget)) => crate::vram_gate::FitDecision::Offload {
                    needed_gb,
                    available_gb: budget.free_gb,
                },
                _ => crate::vram_gate::FitDecision::Unknown,
            }
        } else if krea_turbo_ladder {
            krea_unverified_resident_decision(needed, budget)
        } else if sequential_capable
            && memory_strategy_selection.is_some_and(|selection| selection.strategy.is_optimized())
            && generation_memory.is_some_and(|memory| memory.stage_residency)
        {
            // sc-22664: the shared ladder SELECTED an optimized rung that ENGAGES staged
            // residency for this request, priced per rung from the law against the
            // reserve-charged budget, on a provider that supports sequential offload. That
            // decision stands: the legacy resident-vs-free comparison (the padded
            // `vramGbByTier` row) must not re-refuse it, and the Offload arm below already
            // defers its sequential-overflow gate to a shared selection. The figures name the
            // resident peak the ladder moved off, as the Krea Fits arm above does.
            //
            // A staging-FREE optimized selection (a bounded rung whose composition excludes
            // `StagedResidency`) runs whole-model resident, so the resident-vs-free gate
            // below still applies to it; and a provider without `supportsSequentialOffload`
            // cannot run the staged shape the ladder chose, so it keeps its `TooBig` refusal
            // rather than being sent down an Offload path it does not implement.
            match (needed, budget) {
                (Some(needed_gb), Some(budget)) => crate::vram_gate::FitDecision::Offload {
                    needed_gb,
                    available_gb: budget.free_gb,
                },
                _ => crate::vram_gate::FitDecision::Unknown,
            }
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
                let sequential_needed =
                    crate::vram_gate::predicted_sequential_peak_gb_with_adapter_bytes(
                        &request.model_manifest_entry,
                        tier,
                        adapter_resident_bytes,
                    );
                let krea_selected = if krea_turbo_ladder {
                    match shared_krea_fit {
                        Some(crate::vram_gate::KreaTurboFit::Resident { .. }) => false,
                        Some(
                            fit @ crate::vram_gate::KreaTurboFit::Fits {
                                phases,
                                needed_gb,
                                selection,
                                memory,
                                // `estimate_scoped` is deliberately not read here (sc-18097):
                                // the selected rung and its knobs are what this lane acts on,
                                // and both are already graded. The flag exists for refusal
                                // ADVICE (`krea_turbo_smaller_fit_*`), which must not name an
                                // estimate-backed geometry.
                                ..
                            },
                        ) => {
                            memory_strategy_selection = Some(selection);
                            generation_memory = Some(memory);
                            // Reclaim accounting records allocations, not the admission threshold.
                            // `needed_gb` includes the ladder's operational reserve
                            // (`vram_gate::ladder_reserve_gb`, sc-22667), which is deliberately
                            // never allocated and therefore cannot be credited back during a
                            // model swap.
                            adapted_peak_gb = Some(phases.peak_gb());
                            tracing::info!(
                                event = "image_memory_strategy_selected",
                                job_id = %job.id,
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
                            // sc-22667 (epic 22657 E7): the Krea lane's selection event — the
                            // rung, its basis and the three phase peaks the selector graded, in
                            // the shared ladder's spelling — so an OOM under this selection is
                            // attributable to the exact estimate that admitted it.
                            if let Some(telemetry) =
                                fit.selection_telemetry(tier, memory_width, memory_height)
                            {
                                emit_event("image_memory_strategy_selected", telemetry);
                            }
                            true
                        }
                        Some(crate::vram_gate::KreaTurboFit::Reject { phases, needed_gb }) => {
                            let installed_smaller: Vec<&'static str> = installed_tier_keys(
                                request, settings,
                            )
                            .into_iter()
                            .filter(|candidate| {
                                let candidate_runtime = resolve_tier_dir(
                                    request, settings, candidate,
                                )
                                .and_then(|dir| {
                                    krea_runtime_evidence_context(
                                        request, settings, candidate, &dir,
                                    )
                                });
                                tier_quality_rank(candidate) < tier_quality_rank(tier)
                                    && matches!(
                                        crate::vram_gate::krea_turbo_fit_with_runtime(
                                            &request.model_manifest_entry,
                                            candidate,
                                            memory_width,
                                            memory_height,
                                            budget,
                                            shared_reserve_gb,
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
                                    shared_reserve_gb,
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
                if rejects_unverified_shared_memory_fallback(engine_id, shared_memory_selected) {
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
                            tail = vram_reject_tail(&installed_smaller, plain_gate_lever),
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
                let installed_smaller: Vec<&'static str> = installed_tier_keys(request, settings)
                    .into_iter()
                    .filter(|candidate| {
                        let candidate_runtime = resolve_tier_dir(request, settings, candidate)
                            .and_then(|dir| {
                                krea_runtime_evidence_context(request, settings, candidate, &dir)
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
                                        shared_reserve_gb,
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
                    tail = vram_reject_tail(&installed_smaller, plain_gate_lever),
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
    let incurred_peak = if sdxl_request_tier_admission && memory_strategy_selection.is_some() {
        // An endpoint-priced resident/scratch-bound run must not credit the old padded resident
        // manifest row during the next eviction. Only the admitted raw working set is reclaimable.
        adapted_peak_gb
    } else if use_sequential {
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
    let mut memory_strategy_context = selected_memory_strategy_context.or_else(|| {
        memory_strategy_selection.and_then(|selection| {
            let budget = budget?;
            let predicted_peak_gb = adapted_peak_gb?;
            let turbo_fit = request
                .model_manifest_entry
                .get("candle")?
                .get("turboFit")?;
            let calibration_abi = u32::try_from(turbo_fit.get("calibrationAbi")?.as_u64()?).ok()?;
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
                optimization_authority: gen_core::MemoryOptimizationAuthority::Calibrated,
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
                reference_resources: None,
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
        })
    });
    if !receipt_selector_authoritative {
        apply_request_scoped_candle_residency(use_sequential, &mut generation_memory);
    }
    // Reuse the exact selector spec, including Mage's split text-encoder/VAE component paths. Only
    // the post-selection residency policy and optional ConvRot substitution may differ below.
    let mut spec = shared_contract_spec;
    spec = apply_declared_candle_image_load_shape(
        engine_id,
        Some(tier),
        shared_admission_mode,
        &request.model_manifest_entry,
        spec,
        use_sequential,
    );
    if use_sequential {
        // Ask the provider (candle FLUX) to load→use→drop each component in phase order (sc-10821).
        spec = spec.with_offload_policy(gen_core::OffloadPolicy::Sequential);
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

    // The registered Z-Image identity route retains the shared, non-fatal likeness post-pass from the
    // retired bespoke stream. This stages only the existing face stack, not an obsolete Z-Image path.
    let likeness_source = resolve_character_image_likeness_source(request, settings, project_path);
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        likeness_source.is_some(),
        "registered Candle character_image face-stack staging failed; likeness scores omitted",
    )
    .await;
    let likeness_source = face_stack_dir.as_ref().and(likeness_source);

    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state_using(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("candle {engine_id} load failed"),
        load_generator,
        move |generator,
              cache_state,
              loaded_policy,
              warm_policy,
              external_committed_bytes,
              tx,
              cancel| {
            if is_ideogram || is_sana || is_sd35 || receipt_selector_authoritative {
                // Receipt-priced providers require the exact pre-load selector result to reach the
                // cached generator. Ideogram may preserve a tighter pre-admitted staged sibling;
                // SANA's cache key must exactly retain the selector's own resident/staged policy.
                //
                // Ideogram/SANA/SD3.5 additionally REQUIRE a pre-load context: for them an absent
                // one is a bug, not a fallback. Chroma and Kolors reach here only when the receipt
                // actually priced the request (`receipt_selector_authoritative`); an uncertified
                // import legitimately produced no receipt and keeps the historical resident path
                // through the `else` arm below.
                if is_ideogram {
                    let context = memory_strategy_context.as_mut().ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "{engine_id} cache access is missing its exact pre-load memory context"
                        ))
                    })?;
                    crate::candle_memory_strategy::bind_ideogram_cache_execution(
                        context,
                        &mut generation_memory,
                        &mut ideogram_warm_staged,
                        cache_state,
                        loaded_policy.offload_policy,
                    )?;
                    if let Some(first_pass) = hires_first_pass_memory_context.as_mut() {
                        crate::candle_memory_strategy::bind_ideogram_cache_execution(
                            first_pass,
                            &mut hires_first_pass_generation_memory,
                            &mut ideogram_hires_first_warm_staged,
                            cache_state,
                            loaded_policy.offload_policy,
                        )?;
                    }
                } else if is_sana || is_sd35 {
                    let context = memory_strategy_context.as_mut().ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "{engine_id} cache access is missing its exact pre-load memory context"
                        ))
                    })?;
                    context.cache_state = cache_state;
                    if let Some(first_pass) = hires_first_pass_memory_context.as_mut() {
                        first_pass.cache_state = cache_state;
                    }
                } else {
                    // Chroma/Kolors: `optimized_shared_memory_context` drops a Resident context,
                    // so `None` here means the receipt selected Resident, not that it was silent.
                    if let Some(context) = memory_strategy_context.as_mut() {
                        context.cache_state = cache_state;
                    }
                    if let Some(first_pass) = hires_first_pass_memory_context.as_mut() {
                        first_pass.cache_state = cache_state;
                    }
                }
                // sc-20799: the policy the receipt admitted must EQUAL the policy that loaded, for
                // every receipt-priced family — not only SANA and SD3.5. An admitted staged peak
                // does not bound a resident load, and a warm cache entry can present either shape.
                // Evaluated AFTER the Ideogram warm-staged rebinding above, so the comparison is
                // against the strategy this request will actually execute.
                if let Some(selected_strategy) = receipt_priced_selection {
                    let effective_strategy = memory_strategy_context
                        .as_ref()
                        .map(|context| context.selection.strategy)
                        .unwrap_or(selected_strategy);
                    let selected_staged = effective_strategy
                        == gen_core::MemoryStrategy::StagedResidency
                        || generator
                            .memory_strategy_contract()
                            .is_some_and(|contract| {
                                contract.engages(
                                    effective_strategy,
                                    gen_core::MemoryStrategy::StagedResidency,
                                )
                            });
                    let loaded_staged =
                        loaded_policy.offload_policy == gen_core::OffloadPolicy::Sequential;
                    if selected_staged != loaded_staged {
                        return Err(WorkerError::InvalidPayload(format!(
                            "{engine_id} warm cache policy crossed its exact pre-load selection"
                        )));
                    }
                }
                let stages = generation_memory.is_some_and(|memory| memory.stage_residency);
                warm_policy.settle_with_selection(if stages {
                    crate::execution_planner::GrantOutcome::AlreadyStaged
                } else {
                    crate::execution_planner::GrantOutcome::NoStagedCandidate
                });
                let cache_state_label = match cache_state {
                    gen_core::MemoryCacheState::Cold => "cold",
                    gen_core::MemoryCacheState::Warm => "warm",
                };
                let contract = generator.memory_strategy_contract().ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "{engine_id} loaded provider omitted its sealed memory contract"
                    ))
                })?;
                let facts = contract.asset_facts;
                // Only these three families expose a singular canonical physical receipt to name in
                // the disclosure. Chroma and Kolors seal their identity as an ordered overlay
                // receipt instead, already bound into the selection above; they take the policy
                // guard without minting a second, weaker receipt here.
                if is_ideogram || is_sana || is_sd35 {
                    let physical_receipt = if is_sana {
                        crate::candle_memory_strategy::sana_physical_receipt_identity(
                            engine_id, contract,
                        )?
                    } else if is_sd35 {
                        crate::candle_memory_strategy::sd35_physical_receipt_identity(
                            engine_id, contract,
                        )?
                    } else {
                        crate::candle_memory_strategy::ideogram_physical_receipt_identity(contract)?
                    };
                    let context = memory_strategy_context.as_ref().ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "{engine_id} cache access is missing its exact pre-load memory context"
                        ))
                    })?;
                    let passes = match hires_first_pass_memory_context.as_ref() {
                        Some(first) => vec![("hires_first", first), ("hires_final", context)],
                        None => vec![("single", context)],
                    };
                    for (pass_identity, pass_context) in passes {
                        let strategy_label = match pass_context.selection.strategy {
                            gen_core::MemoryStrategy::Resident => "resident",
                            gen_core::MemoryStrategy::StagedResidency => "staged_residency",
                            gen_core::MemoryStrategy::BoundedDecode => "bounded_decode",
                            gen_core::MemoryStrategy::BoundedAttention => "bounded_attention",
                            gen_core::MemoryStrategy::BoundedTransformerResidency => {
                                "bounded_transformer_residency"
                            }
                        };
                        let authority_label = match pass_context.optimization_authority {
                            gen_core::MemoryOptimizationAuthority::Resident => "resident",
                            gen_core::MemoryOptimizationAuthority::Estimated => "estimated",
                            gen_core::MemoryOptimizationAuthority::Calibrated => "calibrated",
                        };
                        let carrier =
                            match (&pass_context.mode, pass_context.geometry.reference_count) {
                                (_, 0) => "prompt_only",
                                (gen_core::MemoryMode::Edit, 2) => "reference_mask",
                                _ => "reference",
                            };
                        tracing::info!(
                            event = "image_memory_strategy_selected",
                            route = engine_id,
                            actual_tier = tier,
                            pass = pass_identity,
                            mode = pass_context.mode.as_key(),
                            carrier,
                            reference_count = pass_context.geometry.reference_count,
                            use_pid = pass_context.use_pid,
                            overlay = pass_context.overlay.as_deref().unwrap_or("none"),
                            physical_receipt = physical_receipt.as_str(),
                            strategy = strategy_label,
                            authority = authority_label,
                            cache_state = cache_state_label,
                            predicted_peak_bytes = pass_context.predicted_peak_bytes,
                            base_bytes = facts.base_bytes,
                            conditioning_bytes = facts.conditioning_bytes,
                            transformer_bytes = facts.transformer_bytes,
                            decoder_bytes = facts.decoder_bytes,
                            overlay_bytes = facts.overlay_bytes,
                            external_committed_bytes,
                            evidence_revision = %pass_context.evidence_revision,
                            "exact receipt-backed image memory strategy selected"
                        );
                        emit_event(
                            "image_memory_strategy_selected",
                            serde_json::json!({
                                "backend": "candle",
                                "route": engine_id,
                                "actualTier": tier,
                                "passIdentity": pass_identity,
                                "mode": pass_context.mode.as_key(),
                                "carrier": carrier,
                                "referenceCount": pass_context.geometry.reference_count,
                                "geometry": {
                                    "width": pass_context.geometry.width,
                                    "height": pass_context.geometry.height,
                                    "batch": pass_context.geometry.batch,
                                    "frames": pass_context.geometry.frames,
                                },
                                "usePid": pass_context.use_pid,
                                "overlay": pass_context.overlay.clone(),
                                "physicalReceipt": physical_receipt.clone(),
                                "strategy": strategy_label,
                                "authority": authority_label,
                                "cacheState": cache_state_label,
                                "predictedPeakBytes": pass_context.predicted_peak_bytes,
                                "physicalFloor": {
                                    "baseBytes": facts.base_bytes,
                                    "conditioningBytes": facts.conditioning_bytes,
                                    "transformerBytes": facts.transformer_bytes,
                                    "decoderBytes": facts.decoder_bytes,
                                    "overlayBytes": facts.overlay_bytes,
                                },
                                "evidenceRevision": pass_context.evidence_revision.clone(),
                            }),
                        );
                    }
                }
            } else {
                warm_policy.decline(
                    crate::execution_planner::ServedAsIsReason::RouteHasNoRequestScopedMemory,
                );
            }
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            // A native prompt batch (sc-25681) is ONE engine call over every image of the job,
            // admitted against its `prompts × count` batch (`batch_overlay_bytes` above).
            if let Some(batch) = native_batch.as_ref() {
                let seed = work[0].0;
                return drive_native_prompt_batch(
                    tx,
                    seed,
                    batch.images() as usize,
                    |preview, on_progress| {
                        let warnings = std::cell::RefCell::new(Vec::new());
                        let prefix_kv_cache_report = std::cell::RefCell::new(None);
                        let images = render_on_surface(
                            generator,
                            "",
                            width,
                            height,
                            seed,
                            steps,
                            guidance,
                            negative_prompt.clone(),
                            None,
                            &[],
                            None,
                            true_cfg,
                            sampler.as_deref(),
                            scheduler.as_deref(),
                            scheduler_shift,
                            guidance_method.as_deref(),
                            use_pid,
                            text_style_gain,
                            generation_memory,
                            memory_strategy_context.as_ref(),
                            &enhance,
                            gen_core::PromptEnhancementSink::default(),
                            preview,
                            &LaneRgbaSurface {
                                multi_reference_alpha: &[],
                                flux3_references: None,
                                flux3_layout: None,
                                output_channels: rgba_output_channels,
                                controls: lane_controls,
                                warnings: Some(&warnings),
                                prefix_kv_cache: Some(&prefix_kv_cache_report),
                                prompt_batch: Some(batch),
                            },
                            &cancel,
                            on_progress,
                        )?;
                        let facts = lane_controls::image_facts(
                            &warnings.take(),
                            &adapter_identities,
                            &generator.adapter_apply_reports(),
                            crate::execution_planner::prefix_kv_cache_record(
                                prefix_kv_cache_plan,
                                prefix_kv_cache_report.take().as_ref(),
                            ),
                        );
                        Ok((images, facts))
                    },
                );
            }
            drive_gen_items_scored_reported(
                tx,
                work,
                move |_index, (seed, prompt), preview, prompt_enhancement, on_progress| {
                    // This image's non-fatal engine warnings (sc-25681), reported with its asset.
                    let warnings = std::cell::RefCell::new(Vec::new());
                    // ...and the prefix-KV-cache path its render reports (epic 25899).
                    let prefix_kv_cache_report = std::cell::RefCell::new(None);
                    let render = |seed: i64, on_progress: &mut dyn FnMut(Progress)| {
                        generate_one_with_hires_on_surface(
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
                            hires_first_pass_generation_memory,
                            memory_strategy_context.as_ref(),
                            hires_first_pass_memory_context.as_ref(),
                            &enhance,
                            hires_fix,
                            preview.clone(),
                            prompt_enhancement.for_prompt(&prompt),
                            &LaneRgbaSurface {
                                multi_reference_alpha: &edit_ref_alpha,
                                flux3_references: flux3_references.as_ref(),
                                flux3_layout: flux3_layout.as_ref(),
                                output_channels: rgba_output_channels,
                                controls: lane_controls,
                                warnings: Some(&warnings),
                                prefix_kv_cache: Some(&prefix_kv_cache_report),
                                prompt_batch: None,
                            },
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
                    // The image's provenance facts (sc-25681): engine warnings, and each adapter's
                    // identity joined with the engine's own applied/skipped report.
                    let mut facts = lane_controls::image_facts(
                        &warnings.take(),
                        &adapter_identities,
                        &generator.adapter_apply_reports(),
                        crate::execution_planner::prefix_kv_cache_record(
                            prefix_kv_cache_plan,
                            prefix_kv_cache_report.take().as_ref(),
                        ),
                    )
                    .unwrap_or_default();
                    if let Some(references) = flux3_references.as_ref() {
                        facts.insert(
                            flux3_references::FACT_KEY.to_owned(),
                            references.image_fact((out_w, out_h)),
                        );
                    }
                    if let Some(layout) = flux3_layout.as_ref() {
                        facts.insert("flux3Layout".to_owned(), layout.image_fact());
                    }
                    if !facts.is_empty() {
                        prompt_enhancement.send_image_facts(facts);
                    }
                    Ok(Some((final_seed, out_w, out_h, pixels, face_likeness)))
                },
            )
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
        total,
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
    // MOSS edit loads its frozen BF16 transformer/text encoder contract with no quantization.
    // The generic resolver's omitted-setting Q8 default is not an executed tier for this lane.
    let (quant_label, quant_bits) =
        if sceneworks_core::moss_image::is_moss(&request.model) && request.mode == "edit_image" {
            (Some("bf16".to_owned()), None)
        } else {
            (quant_label, quant_bits)
        };
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
            .and_then(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_str()?.trim().parse().ok())
            })
            .and_then(serde_json::Number::from_f64)
    };
    let loras: Vec<String> = request.loras.iter().filter_map(lora_label).collect();
    // Two more facts beside the tier (sc-21484, epic 11037). `quant_label` answers "which variant
    // was selected"; it does NOT answer "what is this checkpoint stored in" or "what did this host
    // execute", and a stats row that reports only the tier lets a reader conclude that a
    // `nvfp4`-tier run executed NVFP4 natively on a box that in fact took the dense BF16 fallback.
    //
    // Both come from the verified source classification and the load receipt, never from
    // `quant_label` and never from the host's capability — see `checkpoint_weight_facts_host`.
    let (source_codec, execution_representation) =
        crate::checkpoint_weight_facts_host::manifest_entry_metrics_pair(
            &request.model_manifest_entry,
        );
    GenerationMetrics {
        model: (!request.model.is_empty()).then(|| request.model.clone()),
        quant_label,
        quant_bits: quant_bits.map(|bits| bits as u32),
        source_codec,
        execution_representation,
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
        pid_target: adv
            .get("pidTarget")
            .and_then(Value::as_str)
            .map(str::to_owned),
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
#[cfg(native_lane)]
fn effective_quant_label(
    request: &ImageRequest,
    tier_dir: Option<&Path>,
) -> (Option<String>, Option<i64>) {
    effective_quant_label_gated(request, nvfp4_host_eligible(), tier_dir)
}

/// [`effective_quant_label`] with the NVFP4 **host** gate passed in rather than probed (sc-11042).
/// Split out for testability, exactly like [`resolve_quant_gated`], which it delegates to. Production
/// has one caller.
#[cfg(native_lane)]
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
    // FLUX.1's generic Candle route accepts only an exact packed q4/q8 directory. Its omitted
    // selection defaults to q4 (rather than the shared resolver's q8 default), so derive the
    // LoadSpec/asset telemetry from that resolved directory instead of re-deriving a different
    // request default here.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if matches!(request.model.as_str(), "flux_schnell" | "flux_dev") {
        if let Some(tier) = tier_dir.and_then(tier_key_from_resolved_dir) {
            return match tier {
                "q4" => (Some("q4".to_owned()), Some(4)),
                "q8" => (Some("q8".to_owned()), Some(8)),
                _ => unreachable!("FLUX.1 generic Candle route resolves only q4/q8"),
            };
        }
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
        Quant::Fp8Rowwise => "fp8r",
    };
    let floors = active
        .iter()
        .map(|floor| {
            let resident = match floor.resident_tier {
                Quant::Q4 => "q4",
                Quant::Q8 => "q8",
                Quant::Nvfp4 => NVFP4_TIER,
                Quant::Fp8Rowwise => "fp8r",
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
#[cfg(native_lane)]
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
#[cfg(not(native_lane))]
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

/// [`slim_step_result`] plus the current preview frame, when one is live — the STEP-tick
/// result posted on every `GenEvent::Step`. Deliberately O(1): the accumulated
/// `assetWrites`/`generationSet`/`prefixKvCache` fact sets are omitted here (streamed on the
/// `Image` / `Decoding` / `Loading` arms instead) because the per-step cadence is
/// steps-per-image × images per batch and the accumulated set had begun OOM-killing the
/// WebView2 renderer mid-batch (see `streaming_result`'s RESOLVED TRADEOFF note, sc-8953).
/// The store restores any omitted accumulation into the durable row, so a slim step tick
/// can never regress it.
fn slim_step_result_with_preview(plan: &ImagePlan, preview: Option<&PreviewSlot>) -> JsonObject {
    let mut result = crate::image_jobs::slim_step_result(plan);
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
    // A native prompt batch (sc-25681) is ONE engine call over all `total` images: its steps
    // (the engine's `steps × count`) span the whole bar, not image 0's slice of it.
    // So is a native count batch (sc-24215, `image.nativeCountBatch`): one call over `count` images.
    let one_pass = !plan.request.prompts.is_empty()
        || sceneworks_core::image_sampling_controls::native_count_batch(
            &plan.request.model_manifest_entry,
        );
    let (step_slices, step_label) = if one_pass {
        (1, format!("Batch of {total}"))
    } else {
        (total_u32, String::new())
    };
    let step_text = |index: usize, detail: String| {
        if one_pass {
            format!("{step_label} — {detail}.")
        } else {
            format!("Image {}/{total} — {detail}.", index + 1)
        }
    };
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
    let prompt_enhancement_expected = raw_settings
        .get("enhancePrompt")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut prompt_enhancement_reports = PromptEnhancementReports::new();
    // The measured three-fact set of this job's load, once the producing lane sends one
    // (sc-11045). `None` for every lane that holds no runtime handle — those assets keep whatever
    // the route stamped pre-load (including the honest `no-runtime-receipt` materialization).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let mut measured_checkpoint_facts: Option<Value> = None;
    #[cfg(not(all(not(target_os = "macos"), feature = "backend-candle")))]
    let measured_checkpoint_facts: Option<Value> = None;
    // Per-image provenance facts (sc-25681), keyed by image index until that image is written.
    let mut pending_image_facts: std::collections::HashMap<usize, JsonObject> =
        std::collections::HashMap::new();
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
                        step_fraction(index, current, step_total, step_slices),
                        &progress_with_auto_tier_streaming(
                            &step_text(index, format!("step {current}/{step_total}")),
                            auto_tier_streaming,
                        ),
                        // Slim on purpose (sc-8953 follow-through): the fact set does not
                        // change on a step tick, so the accumulated asset writes must NOT
                        // ride it — see `slim_step_result_with_preview`.
                        Some(slim_step_result_with_preview(
                            plan,
                            latest_preview.as_ref(),
                        )),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Upsampling { index, tokens, max_tokens } => {
                mark_started(index);
                // This is before denoise: do not mark a sampling phase or invent effective_steps.
                // The select loop's independent interval still polls cancellation during prefill
                // and between coalesced token updates.
                if publish_upsampling_progress(tokens, max_tokens) {
                    update_job(
                        api,
                        &job.id,
                        image_progress(
                            JobStatus::Running,
                            ProgressStage::Generating,
                            step_fraction(index, 0, 1, step_slices),
                            &step_text(index, format!("enhancing prompt ({tokens}/{max_tokens} tokens)")),
                            Some(streaming_result(plan, asset_writes)),
                            backend,
                        ),
                    ).await?;
                }
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
                        step_fraction(index, 1, 1, step_slices),
                        &step_text(index, "decoding".to_owned()),
                        // Full result (asset writes CAN change on a decode), but the
                        // preview frame is the interim image's; the decode arm's own
                        // result carries no frame and `streaming_result` alone is right.
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Loading { index, phase } => {
                mark_started(index);
                let (status, stage) = crate::load_progress::generator_phase_stage(phase);
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        status,
                        stage,
                        step_fraction(index, 0, 1, total_u32),
                        &progress_with_auto_tier_streaming(
                            &format!("Image {}/{total} — {}", index + 1,
                                crate::load_progress::generator_phase(phase)),
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
                let encoded = crate::load_progress::spawn_blocking(move || {
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
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            GenEvent::CheckpointWeightFacts { facts } => {
                // Measured checkpoint facts from the loaded handle (sc-11045). Latest-wins on
                // purpose: the engine's own sink is last-write-wins per materialization, so the
                // newest event always describes the most recent read.
                measured_checkpoint_facts = Some(facts);
            }
            GenEvent::ImageFacts { index, facts } => {
                pending_image_facts.entry(index).or_default().extend(facts);
            }
            GenEvent::AssetWritten { index, fact } => {
                phase_timer.mark_item_done(Instant::now());
                asset_writes.push(Value::Object(fact));
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        0.1 + 0.85 * ((index + 1) as f64 / total as f64),
                        &format!("Generated output {}/{total}.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
            }
            GenEvent::PromptEnhancement {
                index,
                expected_prompt,
                report,
            } => {
                record_prompt_enhancement_report(
                    &mut prompt_enhancement_reports,
                    prompt_enhancement_expected,
                    total,
                    index,
                    expected_prompt,
                    report,
                )?;
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
                // This key is worker-owned even for legacy/raw jobs. Only the typed provider report
                // can add it back, after matching it to this image's exact requested prompt.
                image_raw_settings.remove(PROMPT_ENHANCEMENT_FACT_KEY);
                if prompt_enhancement_expected {
                    image_raw_settings.insert(
                        PROMPT_ENHANCEMENT_FACT_KEY.to_owned(),
                        take_prompt_enhancement_fact(&mut prompt_enhancement_reports, index, &plan.request.model)?,
                    );
                }
                if let Some(block) = face_likeness {
                    image_raw_settings.insert(
                        crate::face_likeness::FACE_LIKENESS_FACT_KEY.to_owned(),
                        Value::Object(block),
                    );
                }
                // Worker-owned like the keys above: a replayed `advanced` block can never forge
                // them, and only this image's own reported facts add them back (sc-25681). The
                // prefix-KV-cache record (epic 25899) is the same: it says which path RAN, so only
                // this image's own report may carry it onto the asset and the job result.
                image_raw_settings.remove(lane_controls::GENERATION_WARNINGS_KEY);
                image_raw_settings.remove(lane_controls::ADAPTER_APPLY_KEY);
                image_raw_settings.remove(lane_controls::PREFIX_KV_CACHE_KEY);
                flux3_references::merge_image_facts(
                    &mut image_raw_settings,
                    pending_image_facts.remove(&index),
                );
                // A measured fact set from the loaded handle supersedes the route's pre-load stamp
                // (whose materialization is honestly `no-runtime-receipt`): same three facts, now
                // with what the load actually materialized (sc-11045).
                if let Some(facts) = measured_checkpoint_facts.as_ref() {
                    image_raw_settings.insert(
                        crate::checkpoint_weight_facts_host::FACTS_RAW_SETTINGS_KEY.to_owned(),
                        facts.clone(),
                    );
                }
                // Encode + write the asset PNG off the async runtime thread (sc-8909 / F-107).
                let plan_for_task = plan.clone();
                let adapter_for_task = adapter_label.to_owned();
                let project_path_for_task = project_path.to_owned();
                let fact = crate::load_progress::spawn_blocking(move || {
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
    if !prompt_enhancement_reports.is_empty() {
        return Err(WorkerError::Engine(
            "provider emitted a prompt-enhancement report without a completed image".to_owned(),
        ));
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
    let tier_dir = resolve_completed_image_metrics_tier_dir(&plan.request, settings);
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
#[path = "tests/candle_label_tests.rs"]
mod candle_label_tests;

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/kolors_edit_worker_contract_tests.rs"]
mod kolors_edit_worker_contract_tests;

#[cfg(test)]
#[path = "tests/boogu_tier_tests.rs"]
mod boogu_tier_tests;

#[cfg(test)]
#[path = "tests/standard_tier_tests.rs"]
mod standard_tier_tests;

/// sc-8820: the recorded quant must reflect the tier subdir ACTUALLY resolved, not the one requested,
/// and a fallback must be surfaced (warn! + `quant_tier_downgraded` event) rather than silently
/// downgrading with lying telemetry.
///
/// macOS-only: exercises [`tier_quant_from_resolved_dir`] / [`reconcile_resolved_tier_quant`], which
/// only compile on the MLX `generate_stream` path. The candle lane has no quant-tier layout.
#[cfg(test)]
#[cfg(target_os = "macos")]
#[path = "tests/quant_tier_reconcile_tests.rs"]
mod quant_tier_reconcile_tests;

/// sc-10733: the shared capability-downtier chooser — walk installed tiers from the resolved default
/// down to the quality floor, pick the highest that fits, reject only when nothing fits. Pure decision,
/// compiled on both lanes (the candle vram gate + the MLX fit gate both feed it their own [`TierFit`]).
#[cfg(all(test, native_lane))]
#[path = "tests/capability_downtier_tests.rs"]
mod capability_downtier_tests;

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
#[path = "tests/mlx_downtier_emulation_tests.rs"]
mod mlx_downtier_emulation_tests;

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
#[path = "tests/candle_downtier_emulation_tests.rs"]
mod candle_downtier_emulation_tests;

// Krea "text style" gain gate (sc-12008): the slider lives at `.ui.textStyleGain` in the FULL model
// entry the worker receives, so the gate MUST read through `ui`. Reading the top level is a silent
// no-op end-to-end (the original bug). This drives a real resolved manifest entry through the shared
// `resolve_text_style_gain` seam — the coverage the engine-only A/B (sc-11878/sc-11884) never had.
#[cfg(all(test, native_lane))]
#[path = "tests/text_style_gain_gate_tests.rs"]
mod text_style_gain_gate_tests;

#[cfg(all(test, target_os = "macos"))]
#[path = "tests/declared_sequential_resident_tests.rs"]
mod declared_sequential_resident_tests;

include!("moss.rs");
