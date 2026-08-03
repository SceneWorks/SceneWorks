use super::{
    admit_conditioning_overlay, consume_gen_events, drive_gen_items_scored,
    ensure_depth_estimator_dir, parse_poses, preprocess_control_entry, requested_control_kind,
    resolve_control_identity_source, resolve_control_source, resolve_seed,
    resolve_user_control_map, stage_likeness, start_gen_stream, validate_control_kind, ApiClient,
    CancelFlag, ControlKind, Image, ImagePlan, JobSnapshot, JsonObject, Path, Progress, Settings,
    Value, WorkerResult,
};

use crate::conditioning_fit::ConditioningAdmission;

// Shared candle (Windows/CUDA) strict-control driver (sc-8304, epic 8236). The candle siblings of the
// MLX `strict_control.rs` driver: the three bespoke non-registry candle control providers
// (`ZImageControl` / `Flux2Control` / `QwenControl`) used to each carry a near-identical
// `start_gen_stream` lane that rendered a DWPose skeleton, fed it as a raw control `Image` to the
// provider, and scored the result. That scaffold — control-kind resolution + validation against the
// SAME [`STRICT_CONTROL_ENGINES`] table, pose/canny/depth/user-passthrough preprocessing via the
// cross-platform [`preprocess_control_entry`], source threading, the per-pose loop, identity-likeness
// scoring, and the `consume_gen_events` sink — is now ONE driver here. Each provider keeps only its
// genuinely-divergent bits (base/control weight resolution, its bespoke request struct + `generate`
// call, raw_settings keys, control-scale defaults, engine label) behind the [`CandleStrictControl`]
// trait.
//
// **Candle-only.** macOS keeps the MLX registry strict-control paths (driven by `strict_control.rs`
// directly); this driver is the off-Mac candle lane (`#[cfg(all(not(target_os = "macos"), feature =
// "backend-candle"))]`), a child module of the `image_jobs` module so it shares the unqualified imports.
//
// **Pose is byte-preserved (REGRESSION-CRITICAL).** A pose job sets no `advanced.controlMode`, so
// `requested_control_kind` returns [`ControlKind::Pose`] and `preprocess_control_entry` renders the
// exact same `draw_wholebody` skeleton at the same stickwidth the per-lane code did before — the
// control `Image` handed to each provider's `generate` is identical. canny / depth are purely additive
// (reached only when a future request sets `controlMode = canny | depth`), and the candle-gen control
// engines VAE-encode any RGB control image generically (no pose-only gate), so they flow without an
// engine change.

/// The genuinely per-provider half of a candle strict-control lane: build the bespoke request struct and
/// call the provider's `generate(&req, control_image, on_progress)`. Everything else (control-kind
/// resolution/validation, pose/canny/depth preprocessing, source threading, the per-pose loop, scoring,
/// streaming) is the shared [`run_candle_strict_control`] driver.
///
/// `Model` is the loaded `!Send` provider (`ZImageControl` / `Flux2Control` / `QwenControl`); it is built
/// and consumed entirely inside the one `spawn_blocking`, so it never needs to be `Send`. The implementor
/// (the small per-lane config struct) IS moved across the blocking boundary, so it must be `Send +
/// 'static`.
pub(super) trait CandleStrictControl: Send + 'static {
    /// The loaded candle control provider.
    type Model;

    /// The [`STRICT_CONTROL_ENGINES`] catalog id whose `supported_kinds` gates this lane's
    /// `advanced.controlMode` (`z_image_turbo_control` / `z_image_control` / `flux1_dev_control` /
    /// `flux2_dev_control` / `qwen_image_control`).
    fn engine_id(&self) -> &'static str;

    /// The engine label recorded on assets / `raw_settings` (`candle_zimage_control` /
    /// `candle_flux1_control` / `candle_flux2_control` / `candle_qwen_control`) — byte-preserved per the
    /// regression contract.
    fn engine_label(&self) -> &'static str;

    /// The short `start_gen_stream` tag (`zimage_control` / `flux2_control` / `qwen_control`).
    fn stream_tag(&self) -> &'static str;

    /// The output base dimensions for this generation: the geometry the control map (pose skeleton /
    /// canny / depth) is rendered at, and the size the provider generates. Normally the request's W×H;
    /// a PiD `2k` output tier (sc-10054) caps them so PiD's fixed 4× lands on ~2048. These MUST equal the
    /// provider's stored `width`/`height` (used inside `generate_one`), or the control map and the latent
    /// misalign — so each impl returns exactly those fields.
    fn out_width(&self) -> u32;
    fn out_height(&self) -> u32;

    /// How this lane is admitted before it allocates (sc-16069): either the resident on-disk footprint
    /// [`load`](Self::load) will hold — base weights plus the co-resident control overlay — for the shared
    /// weights floor, or a declaration that the lane runs its own MEASURED gate, named.
    ///
    /// **Required on purpose — there is no default.** A strict-control lane is by definition a second
    /// network overlaid on a base model, which is exactly the shape neither the `generate_candle_stream`
    /// `vram_gate` nor the `generator_cache` `apply_residency_policy` can see (the route is diverted before
    /// the first, and a bespoke `*Paths` load never produces the `LoadSpec` the second reads). Giving this
    /// a default would let a NEW lane inherit the sc-16069 defect silently — the very regression this
    /// method exists to make impossible — so choosing is part of the cost of adding a lane, enforced by
    /// the compiler.
    ///
    /// Build the floor arm with [`ConditioningFootprint::from_paths`] from the paths the lane already
    /// resolved. Unmeasurable weights are honest and safe: the gate treats them as no evidence and admits.
    /// Reach for [`ConditioningAdmission::MeasuredElsewhere`] only when the lane genuinely owns a measured
    /// gate — see that variant for why a measurement must REPLACE the floor rather than compose with it.
    fn conditioning_admission(&self) -> ConditioningAdmission;

    /// Load the provider on the blocking thread (download already happened in the async preamble).
    fn load(&self) -> WorkerResult<Self::Model>;

    /// Generate one image conditioned on the already-preprocessed `control` map (pose skeleton, canny
    /// edges, or depth map — the driver picked which). Builds the bespoke request struct internally; the
    /// driver supplies the shared seed + cancel + progress callback + preview sink. Returns the output
    /// [`Image`].
    ///
    /// `preview` is the job's **live** per-image [`gen_core::PreviewSink`] (epic 16948, sc-16962), built by
    /// [`preview_sink_for`](super::stream) and handed down by [`run_candle_strict_control`]. Every bespoke
    /// candle provider is invoked BY NAME rather than through `dyn Generator`, so the registry cannot thread
    /// the sink for it — this parameter is the only route a strict-control lane has to a live sink.
    ///
    /// **An impl whose request struct carries a `preview` field must set it from this argument.** Writing
    /// `preview: Default::default()` compiles, ships, and produces no previews on that lane — a silent
    /// narrowing that looks green. [`crate::candle_preview_wiring_tests`] reads these lanes' own source
    /// and fails on exactly that, on every platform, because both candle CI lanes are dispatch-only. An
    /// impl whose upstream request has no `preview` field yet binds `_preview`; its file is still swept,
    /// so adding the field upstream without wiring it here is caught the same way.
    fn generate_one(
        &self,
        model: &Self::Model,
        control: &Image,
        seed: u64,
        cancel: &CancelFlag,
        preview: &gen_core::PreviewSink,
        on_progress: &mut dyn FnMut(Progress),
    ) -> WorkerResult<Image>;
}

/// The shared async preamble + per-pose streaming driver for a candle strict-control lane (sc-8304).
///
/// Mirrors the MLX `strict_control.rs` flow, adapted to the candle providers' raw-`Image` control input
/// (vs the MLX `Conditioning::Control`): resolve + validate the requested [`ControlKind`] against the
/// engine's `supported_kinds`, resolve the user-supplied control-map passthrough and the canny/depth
/// source image, provision the depth estimator only for a depth job without a passthrough, then run one
/// generation per pose — each preprocessing its control map via [`preprocess_control_entry`] (pose =
/// `draw_wholebody`; canny = `canny`; depth = the backend depth estimator) and scoring the result
/// against the optional identity reference. `raw_settings` / `pose_count` / the engine label are passed
/// through to [`consume_gen_events`] unchanged.
///
/// `provider` is the per-lane [`CandleStrictControl`] config (it holds the resolved weight paths +
/// request numerics); it is moved into the blocking thread, loaded once, and drives every pose.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run_candle_strict_control<P: CandleStrictControl>(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    provider: P,
    raw_settings: JsonObject,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;

    // Conditioning-overlay VRAM admission (sc-16069, epic 15448) — the FIRST thing this driver does, so a
    // host that cannot hold the base + control branch is refused with an actionable message before any
    // allocation, rather than dying on a reactive CUDA OOM mid-render. This ONE call site covers all five
    // in-house strict-control lanes (qwen / kolors / z-image / flux2 / flux1), and `conditioning_admission`
    // is a REQUIRED trait method so a sixth cannot be added ungated.
    //
    // Placed before the depth-estimator provisioning below (a download, not an allocation) so the refusal
    // costs the user nothing. It is a weights FLOOR and never blocks a host that can hold the weights —
    // see `crate::conditioning_fit`.
    match provider.conditioning_admission() {
        crate::conditioning_fit::ConditioningAdmission::Floor(footprint) => {
            admit_conditioning_overlay(settings, &footprint).await?;
        }
        // The lane ALREADY gated itself, earlier, in its own preamble — today only Krea pose-control,
        // which runs the weights floor AND the measured `krea_control_fit` ladder there. It has to be
        // earlier: its preamble records the admitted peak with `vram_gate::note_loaded_peak`, and a
        // rejection arriving after that would leave a reclaimable high-water standing for a load that
        // never allocated. Gating again here would only re-probe the GPU and risk a second eviction.
        // See `ConditioningAdmission::GatedInPreamble`.
        crate::conditioning_fit::ConditioningAdmission::GatedInPreamble { gate } => {
            tracing::debug!(
                engine = provider.engine_label(),
                gate,
                "candle conditioning admission: already decided in this lane's preamble (it must run \
                 before its own note_loaded_peak), so the shared gate stands down (sc-16069)"
            );
        }
    }

    // Shared strict-control driver: validate the requested ControlKind against the engine's
    // supported_kinds + resolve an optional user-supplied control-map passthrough. A pose job sets no
    // `controlMode`, so `kind == Pose` and the skeleton preprocessor runs exactly as the per-lane code
    // did before (byte-identical pose path).
    let control_kind = requested_control_kind(request)?;
    validate_control_kind(provider.engine_id(), &control_kind)?;
    let user_control = resolve_user_control_map(request, settings, project_path)?;
    // Source threading: for canny/depth WITHOUT a user-supplied control map, the control map is
    // auto-derived from the input image (canny edges / Depth-Anything-V2). The pose tier never needs a
    // source (the skeleton is synthetic).
    let control_source = resolve_control_source(request, settings, project_path)?;
    // Auto depth-estimator weights: provisioned only when this is a depth job WITHOUT a user-supplied
    // depth map (the passthrough short-circuits estimation). Fetched once on the first depth job (the
    // off-Mac candle estimator, sc-8413).
    let depth_weights_dir = if control_kind == ControlKind::Depth && user_control.is_none() {
        Some(ensure_depth_estimator_dir(api, settings, job).await?)
    } else {
        None
    };

    let poses = parse_poses(request);
    let pose_count = poses.len();

    // Render the control map + generate at the provider's output dims (normally the request's W×H; a PiD
    // 2K output tier caps them — sc-10054). The provider's stored dims and these are the same value.
    let (width, height) = (provider.out_width(), provider.out_height());
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    // One shared seed across the pose set (the MLX `_generate_pose_set` convention) so noise-derived
    // attributes (hair, wardrobe, lighting) stay constant while only the pose changes. `resolve_seed`
    // returns `i64` (the asset-write tuple seed); the candle providers' request structs take `u64` (the
    // per-lane code cast the same at the `generate` call).
    let seed = resolve_seed(request, 0);

    // Identity-likeness scoring (epic 4406, sc-4410): when this candle strict-pose set carries a
    // character identity `referenceAssetId`, score every finished pose against that source identity face
    // through the SHARED generator-agnostic seam. Source decode + face-stack staging are non-fatal; the
    // `!Send` scorer is built ONCE in the load closure on the blocking thread (source embedded once,
    // reused across all poses).
    let likeness_source = resolve_control_identity_source(request, settings, project_path);
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        likeness_source.is_some(),
        "pose-set face-stack staging failed; likeness scores omitted",
    )
    .await;
    let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());

    let engine_label = provider.engine_label();
    let stream_tag = provider.stream_tag();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        stream_tag,
        0,
        move || {
            let model = provider.load()?;
            // Build the scorer once on the blocking thread (the `!Send` face stack lives here).
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            Ok((provider, model, poses, scorer))
        },
        move |(provider, model, poses, scorer), tx, cancel| {
            let user_control = user_control.as_ref();
            let control_source = control_source.as_ref();
            let depth_weights_dir = depth_weights_dir.as_deref();
            drive_gen_items_scored(tx, poses, move |_index, pose, preview, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                // Pose = the same `draw_wholebody` skeleton (byte-preserved); canny/depth derive from the
                // threaded source. A user-supplied control map short-circuits for any kind.
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
                // The job's live per-image preview sink, handed straight to the provider's bespoke request
                // (epic 16948, sc-16962). `drive_gen_items_scored` built it with `preview_sink_for`, so a
                // frame emitted on the denoise thread lands as a `GenEvent::Preview` and reaches
                // `WorkerProgressCard`. This closure used to bind `_preview` and drop it, which is why the
                // Krea control lane shipped an inert `Default::default()` sink.
                let out = match provider.generate_one(
                    &model,
                    &control,
                    seed as u64,
                    &cancel,
                    &preview,
                    on_progress,
                ) {
                    Ok(out) => out,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => return Err(error),
                };
                // Score this finished pose against the cached source embedding (sc-4410). The candle
                // strict-control lane produces the FINAL image directly (no face-restore pass). Clone
                // paid ONLY when a scorer exists; a full-body / turned pose with no reliable frontal
                // face → honest detected:false N/A.
                let face_likeness = scorer.as_ref().and_then(|scorer| {
                    crate::face_likeness::score_generated_image(
                        Some(scorer),
                        &Image {
                            width: out.width,
                            height: out.height,
                            pixels: out.pixels.clone(),
                        },
                        likeness_source_ref.as_deref(),
                    )
                });
                Ok(Some((
                    seed,
                    out.width,
                    out.height,
                    out.pixels,
                    face_likeness,
                )))
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
        engine_label,
        &raw_settings,
        pose_count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
