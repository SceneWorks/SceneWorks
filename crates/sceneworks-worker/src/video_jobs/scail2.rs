#[allow(unused_imports)]
use super::prelude::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::run_blocking_with_heartbeat;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::wan::scail2_sampling;
#[cfg(target_os = "macos")]
use super::wan::{
    generate_video, resolve_scail2_adapters, scail2_adapters_have_lightning, VideoGenInput,
};
#[cfg(target_os = "macos")]
use super::{
    ltx::extract_clip_frames,
    vace::{load_source_video_frames, replacement_status_value, resolve_character_references},
    wan::local_mlx_dir,
};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::video_request::MAX_SCAIL2_REFERENCE_CHARACTERS;

/// Cancel message shared by every SCAIL-2 person-segmentation pass (both backends).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SCAIL2_SEGMENT_CANCEL_MESSAGE: &str = "SCAIL-2 canceled during person segmentation.";

/// Run a SCAIL-2 person-segmentation-and-paint pass on the blocking pool under the heartbeat
/// keepalive (sc-8390 / sc-8807). The cold multi-GB SAM3 checkpoint parse + per-frame propagation
/// can exceed the API's 90s stale-sweep, so the keepalive drives progress and its cancel poll trips
/// the flag the engine's per-frame propagate contract observes between frames. Backend-neutral
/// (sc-8830): the caller's `segment` closure captures whichever SAM3 module the build links (MLX
/// `person_segment_sam3` vs `person_segment_sam3_candle`) plus the paint background, so the
/// heartbeat orchestration lives in exactly one place instead of a per-backend twin.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn scail2_segment_blocking<R, F>(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    task_label: &'static str,
    segment: F,
) -> WorkerResult<R>
where
    R: Send + 'static,
    F: FnOnce(gen_core::CancelFlag) -> WorkerResult<R> + Send + 'static,
{
    let cancel = gen_core::CancelFlag::new();
    let flag = cancel.clone();
    run_blocking_with_heartbeat(
        api,
        settings,
        job_id,
        Some(cancel),
        SCAIL2_SEGMENT_CANCEL_MESSAGE,
        task_label,
        crate::no_cancel_ack(),
        tokio::task::spawn_blocking(move || segment(flag)),
    )
    .await
}

// ---------------------------------------------------------------------------
// Real MLX SCAIL-2 generation (macOS, via mlx-gen-scail2, epic 5439 / sc-5448): end-to-end character
// animation — a reference character image + a driving video → an animated clip of the character
// performing the driving motion. The worker paints the color-coded segmentation masks the engine
// needs from native SAM3 (no user masks): the reference image and the driving frames are each
// segmented (every person → a distinct palette color, left-to-right) and painted onto the
// whose-world-to-keep background (animation: driving bg black, ref bg white). Conditioning =
// ordered `Reference` + `Mask` pairs (one per character) + `ControlClip{frames, mask}` (driving video +
// per-frame color masks). `replace_person` (cross-identity, replace_flag=true) is the same engine,
// wired in sc-5452.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX SCAIL-2 asset.
#[cfg(target_os = "macos")]
pub(super) const SCAIL2_ADAPTER: &str = "mlx_scail2";

/// SceneWorks SCAIL-2 model id → mlx-gen registry id, or `None` if `model` is not SCAIL-2.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn scail2_engine_id(model: &str) -> Option<&'static str> {
    (model == "scail2_14b").then_some("scail2_14b")
}

/// Whether the linked SCAIL-2 engine can serve this request now (resolvable weights).
#[cfg(target_os = "macos")]
pub(super) fn scail2_available(_request: &VideoRequest, settings: &Settings) -> bool {
    resolve_scail2_model_dir(settings).is_ok()
}

/// Resolve the SCAIL-2 MLX snapshot dir: env override (`SCENEWORKS_MLX_SCAIL2_DIR`) → app-managed
/// `<data>/models/mlx/scail2` → the turnkey download-on-first-use `SceneWorks/scail2-mlx` snapshot
/// (mirrors `resolve_bernini_model_dir`). Errors clearly if none is present (no stub fallback).
#[cfg(target_os = "macos")]
pub(super) fn resolve_scail2_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Some(dir) = local_mlx_dir(settings, "SCENEWORKS_MLX_SCAIL2_DIR", "scail2") {
        return Ok(dir);
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, "SceneWorks/scail2-mlx") {
        return Ok(dir);
    }
    Err(WorkerError::InvalidPayload(format!(
        "scail2: no MLX weights found. Download the turnkey SceneWorks/scail2-mlx snapshot via the \
         Model Manager, set $SCENEWORKS_MLX_SCAIL2_DIR, or place a converted snapshot at {}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join("scail2")
            .display(),
    )))
}

/// MLX quantization for a SCAIL-2 load: Q4 default, Q8 opt-in via the advanced `mlxQuantize: 8`
/// control, explicit `<= 0` ⇒ bf16 (power users with ample RAM). Delegates to the shared
/// [`resolve_mlx_dense_quant`] (sc-8830) — the byte-identical `resolve_bernini_quant` twin.
#[cfg(target_os = "macos")]
pub(super) fn resolve_scail2_quant(request: &VideoRequest) -> Option<Quant> {
    resolve_mlx_dense_quant(request)
}

/// The turnkey SceneWorks SCAIL-2 MLX repo (sc-9944, epic 8506). Hosts the quant matrix as
/// self-contained tier subdirs `q4/` (default) + `q8/` + `bf16/`, each a COMPLETE snapshot (the DiT
/// `dit.safetensors` at the tier's precision + the shared dense Wan2.1 z16 VAE + UMT5 T5 encoder +
/// open-CLIP ViT-H/14 visual tower + tokenizer + `config.json` carrying the quant manifest). This
/// augments the flat Q4 layout the repo shipped before (a single pre-quantized snapshot, sc-5445);
/// the worker now descends into the chosen tier so a pre-packed snapshot loads with no install-time
/// convert peak. The flat root files stay for back-compat with already-shipped workers that resolve
/// the repo root; a new worker only ever resolves the tier subdirs.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const SCAIL2_REPO: &str = "SceneWorks/scail2-mlx";

/// Pinned revision for [`SCAIL2_REPO`] (mirrors [`WAN_T2V_14B_REVISION`]). The repo is a hard-coded
/// const — no manifest/payload override reaches the on-demand `q8/*` + `bf16/*` fetches — so pulling
/// the mutable `main` branch would let an upstream re-push silently swap a checkpoint we load. This is
/// the commit that added the `q4/`/`q8/`/`bf16/` tier subdirs (sc-9944); the native downloader still verifies
/// each file's own hash on download.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const SCAIL2_REVISION: &str = "ce88cfdb1008f395e9c820e525e6db7b6695f7b3";

/// Map a SCAIL-2 model id to its `(quant-matrix repo, pinned revision)` for the on-demand tier fetch,
/// or `None` for a non-SCAIL-2 id. Keyed here so the whole tier-resolve/fetch path (mirroring the Wan
/// [`wan_tier_repo`] machinery) routes on `scail2_tier_repo(..).is_some()`.
#[cfg(target_os = "macos")]
pub(super) fn scail2_tier_repo(model: &str) -> Option<(&'static str, &'static str)> {
    (model == "scail2_14b").then_some((SCAIL2_REPO, SCAIL2_REVISION))
}

/// The SCAIL-2 quant-matrix tier search order for a request — preferred tier first, then the
/// always-smaller fallback tiers so a repo missing the preferred subdir still loads (mirrors
/// [`wan_tier_order`]): `<= 0` ⇒ `bf16`; `>= 8` ⇒ `q8`; and — with NO explicit `mlxQuantize` OR an
/// explicit `q4` — the **`q4`** default (q4-first). `bf16` stays OUT of the default order, so a default
/// job never pulls the huge dense tier by accident.
///
/// The video lane keeps the pre-sc-10726 q4-first default (sc-10859), NOT epic 10721's app-wide Q8 —
/// see [`wan_tier_order`] for the rationale (no MLX video Q8 lever ⇒ a silent Q8 only risks a
/// video-runtime OOM). This also lets the resolver drop its old `mlxQuantize`-presence check: the
/// no-explicit-pick default and an explicit `q4` now BOTH resolve q4-first, so [`resolve_scail2_quant`]
/// mapping both to `Some(Quant::Q4)` is exactly what we want.
#[cfg(target_os = "macos")]
pub(super) fn scail2_tier_order(request: &VideoRequest) -> &'static [&'static str] {
    match resolve_scail2_quant(request) {
        None => &["bf16", "q8", "q4"],
        Some(Quant::Q8) => &["q8", "q4"],
        // No explicit pick OR an explicit `q4` ⇒ q4-first (sc-10859 video carve-out).
        _ => &["q4", "q8"],
    }
}

/// Whether `dir` is a COMPLETE self-contained SCAIL-2 tier snapshot. The canonical six-file
/// inventory and exact q4/q8/bf16 config-marker contract live together in
/// `sceneworks_core::mlx_tier_completeness`, so the catalog and native resolvers cannot disagree
/// about an installed tier. A partial or mixed-marker tier fails this and [`scail2_tier_subdir`]
/// falls through to a smaller complete tier rather than half-loading.
#[cfg(target_os = "macos")]
pub(super) fn scail2_tier_is_complete(dir: &Path) -> bool {
    sceneworks_core::mlx_tier_completeness::scail2_tier_complete(dir)
}

/// Descend a resolved SCAIL-2 quant-matrix repo `root` into the requested quant tier subdir (sc-9944,
/// epic 8506), mirroring [`wan_tier_subdir`]. Returns the first COMPLETE tier in [`scail2_tier_order`],
/// or `None` when the repo has no complete tier subdir — a legacy flat snapshot, where the caller
/// keeps the root + load-time quant.
#[cfg(target_os = "macos")]
pub(super) fn scail2_tier_subdir(root: &Path, request: &VideoRequest) -> Option<PathBuf> {
    scail2_tier_order(request)
        .iter()
        .map(|tier| root.join(tier))
        .find(|dir| scail2_tier_is_complete(dir))
}

/// Resolve the SCAIL-2 `(model_dir, load-time quant)` for a generation, descending into the
/// quant-matrix tier subdir when the turnkey ships them (sc-9944). A pre-packed tier's `config.json`
/// is authoritative — [`Scail2Config::from_model_dir`] reads its `quantization` block and the DiT
/// loader builds each Linear from packed parts directly — so a resolved tier loads with `quant = None`
/// (`mlxQuantize` selects WHICH tier, never a load-time requant; the `bf16/` tier is dense, so `None`
/// ⇒ dense too). A legacy flat snapshot (no tier subdirs) keeps today's behavior: load the root and
/// quantize at load per [`resolve_scail2_quant`].
#[cfg(target_os = "macos")]
fn resolve_scail2_tier_dir_and_quant(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<(PathBuf, Option<Quant>)> {
    let root = resolve_scail2_model_dir(settings)?;
    match scail2_tier_subdir(&root, request) {
        Some(tier) => Ok((tier, None)),
        None => Ok((root, resolve_scail2_quant(request))),
    }
}

/// On-demand fetch of a non-default SCAIL-2 quant-matrix tier subdir (sc-9944, mirrors
/// [`ensure_wan_tier_present`]). The macOS default download is the lean `q4/` tier; a job that opts
/// into a heavier tier (`mlxQuantize <= 0` ⇒ `bf16`, `>= 8` ⇒ `q8`) pulls just that subdir from the
/// FIXED [`scail2_tier_repo`] revision the first time it is requested so [`scail2_tier_subdir`] can
/// resolve it. No-op for a non-SCAIL-2 model, a `q4` (default) job, when the repo snapshot isn't
/// downloaded yet (resolve surfaces the clear error), or when the tier is already complete. Fails loud
/// on a real download error — fast, before any compute; a tier that isn't published yet stays absent so
/// resolve falls back to a smaller complete tier.
#[cfg(target_os = "macos")]
async fn ensure_scail2_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    let Some((repo, revision)) = scail2_tier_repo(&request.model) else {
        return Ok(());
    };
    let tier = match resolve_scail2_quant(request) {
        None => "bf16",
        Some(Quant::Q8) => "q8",
        // q4 default — ships with the base install, nothing to fetch on demand.
        _ => return Ok(()),
    };
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, repo) else {
        return Ok(());
    };
    if scail2_tier_is_complete(&root.join(tier)) {
        return Ok(());
    }
    let files = vec![format!("{tier}/*")];
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, repo, revision, &files)
        .await
        .map(|_| ())
}

/// Raw-settings recorded on a real MLX SCAIL-2 asset (mirrors `bernini_raw_settings`). When the
/// lightx2v lightning LoRA is applied (`lightning`, sc-5700), records the effective step-distill recipe
/// the worker dispatched — so the chosen steps/CFG/shift is inspectable on the asset, not silent
/// (mirrors `wan_raw_settings`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn scail2_raw_settings(request: &VideoRequest, lightning: bool) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    // The engine task the SceneWorks mode resolved to (lineage / observability).
    raw.insert(
        "scail2Task".to_owned(),
        Value::String(scail2_engine_video_mode(&request.mode).to_owned()),
    );
    if lightning {
        let (steps, guidance, shift) = scail2_sampling(request, true);
        raw.insert("scail2Lightning".to_owned(), Value::Bool(true));
        if let Some(steps) = steps {
            raw.insert("effectiveSteps".to_owned(), json!(steps));
        }
        if let Some(guidance) = guidance {
            raw.insert("effectiveGuidanceScale".to_owned(), json!(guidance));
        }
        if let Some(shift) = shift {
            raw.insert("effectiveSchedulerShift".to_owned(), json!(shift));
        }
    }
    Value::Object(raw)
}

/// Map a SceneWorks video mode to the SCAIL-2 engine `video_mode` task string. `replace_person`
/// (cross-identity, sc-5452) flips the engine `replace_flag`; everything else (`animate_character`)
/// is plain animation.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn scail2_engine_video_mode(mode: &str) -> &'static str {
    match mode {
        "replace_person" => "replacement",
        _ => "animation",
    }
}

/// Segment every SCAIL-2 character separately, keeping the caller's order and checking the
/// heartbeat-owned cancellation flag between references. A single all-images SAM3 invocation
/// would blur reference association and could continue through five expensive encodes after a
/// cancellation; the engine instead receives a strict image/mask pair for each character.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn segment_scail2_references<F>(
    references: Vec<Image>,
    cancel: &gen_core::CancelFlag,
    mut segment_one: F,
) -> WorkerResult<Vec<(Image, Image)>>
where
    F: FnMut(&Image, &gen_core::CancelFlag) -> WorkerResult<Image>,
{
    if references.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "SCAIL-2 Animate Character requires at least one reference character image.".to_owned(),
        ));
    }
    if references.len() > MAX_SCAIL2_REFERENCE_CHARACTERS {
        return Err(WorkerError::InvalidPayload(format!(
            "SCAIL-2 Animate Character supports at most {MAX_SCAIL2_REFERENCE_CHARACTERS} reference characters."
        )));
    }

    let mut pairs = Vec::with_capacity(references.len());
    for reference in references {
        crate::person_segment_sam3_common::check_segment_canceled(Some(cancel))?;
        let mask = segment_one(&reference, cancel)?;
        crate::person_segment_sam3_common::check_segment_canceled(Some(cancel))?;
        pairs.push((reference, mask));
    }
    Ok(pairs)
}

/// Build the strict SCAIL-2 conditioned-input grammar: ordered `Reference`, `Mask` pairs for
/// every character, followed by the driving `ControlClip`. In particular, do not use the generic
/// image-only `MultiReference` carrier: it cannot associate each identity image with the color mask
/// SCAIL-2 must encode alongside it.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn scail2_animate_conditioning(
    reference_pairs: Vec<(Image, Image)>,
    driving: Vec<Image>,
    driving_mask: Vec<Image>,
) -> WorkerResult<Vec<Conditioning>> {
    if reference_pairs.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "SCAIL-2 Animate Character requires at least one reference character image.".to_owned(),
        ));
    }
    if reference_pairs.len() > MAX_SCAIL2_REFERENCE_CHARACTERS {
        return Err(WorkerError::InvalidPayload(format!(
            "SCAIL-2 Animate Character supports at most {MAX_SCAIL2_REFERENCE_CHARACTERS} reference characters."
        )));
    }

    let mut conditioning = Vec::with_capacity(reference_pairs.len() * 2 + 1);
    for (reference, mask) in reference_pairs {
        conditioning.push(Conditioning::Reference {
            image: reference,
            strength: None,
        });
        conditioning.push(Conditioning::Mask { image: mask });
    }
    conditioning.push(Conditioning::ControlClip {
        frames: driving,
        mask: driving_mask,
        masking_strength: 1.0,
        start_frame: 0,
        mode: ReplacementMode::default(),
    });
    Ok(conditioning)
}

/// Resolve a SCAIL-2 request into the engine conditioning: load the reference character image and the
/// driving clip, segment both with native SAM3 (every person → a palette color), paint the color-coded
/// masks (animation background convention), and assemble strict ordered `Reference`, `Mask` pairs
/// before the `ControlClip`. Every `referenceAssetIds` entry is independently loaded and segmented
/// in caller order; a legacy `sourceAssetId` remains the single-reference fallback. The segmentation
/// + painting run on the blocking pool (GPU inference).
#[cfg(target_os = "macos")]
pub(super) async fn resolve_scail2_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    // The ordered plural list wins. The legacy i2v source image remains only as a single-reference
    // fallback, never an unpaired extra character.
    let reference_ids: Vec<&str> = if request.reference_asset_ids.is_empty() {
        request.source_asset_id.as_deref().into_iter().collect()
    } else {
        request
            .reference_asset_ids
            .iter()
            .map(String::as_str)
            .collect()
    };
    if reference_ids.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "scail2 animate_character requires a reference character image (referenceAssetIds \
             or sourceAssetId)."
                .into(),
        ));
    }
    if reference_ids.len() > sceneworks_core::video_request::MAX_SCAIL2_REFERENCE_CHARACTERS {
        return Err(WorkerError::InvalidPayload(format!(
            "SCAIL-2 Animate Character supports at most {} reference characters.",
            sceneworks_core::video_request::MAX_SCAIL2_REFERENCE_CHARACTERS
        )));
    }
    let references = reference_ids
        .into_iter()
        .map(|reference_id| {
            load_reference_image(
                &settings.data_dir,
                &request.project_id,
                reference_id,
                project_path,
            )
        })
        .collect::<WorkerResult<Vec<_>>>()?;

    // The driving video → frames at the output size (the engine re-resizes internally).
    let clip_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "scail2 animate_character requires a driving video (sourceClipAssetId).".into(),
        )
    })?;
    let driving = extract_clip_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        clip_id,
        request.width,
        request.height,
        wan_frame_count(request.raw_frame_count()),
    )
    .await?;

    // SAM3 segmenter weights, resolved from the Model Manager install (sc-17629), shared by both
    // segmentation passes.
    let (sam_model, sam_tokenizer) =
        crate::person_segment_sam3::require_segmenter_weights(settings)?;

    // Segment + paint via the shared orchestrator (sc-8830). Animation keeps the reference's world
    // (ref bg white) and drops the driving world (driving bg black); the native SAM3 module is the
    // per-frame 1008² MLX twin.
    let (rm, rt) = (sam_model.clone(), sam_tokenizer.clone());
    assemble_scail2_animate_conditioning(
        api,
        settings,
        &job.id,
        move |flag| {
            segment_scail2_references(references, &flag, |reference, flag| {
                let masks = crate::person_segment_sam3::segment_all_persons_in_memory(
                    &rm,
                    &rt,
                    std::slice::from_ref(reference),
                    Some(flag.clone()),
                    None,
                )?;
                crate::scail2_masks::paint_reference_mask(&masks, crate::scail2_masks::BG_WHITE)
            })
        },
        move |flag| {
            let masks = crate::person_segment_sam3::segment_all_persons_in_memory(
                &sam_model,
                &sam_tokenizer,
                &driving,
                Some(flag),
                Some(Box::new(|frame, total| {
                    tracing::debug!(event = "scail2_sam3_propagate_progress", frame, total);
                })),
            )?;
            let masks =
                crate::scail2_masks::paint_driving_masks(&masks, crate::scail2_masks::BG_BLACK)?;
            Ok((driving, masks))
        },
    )
    .await
}

/// Real MLX SCAIL-2 generation (epic 5439 / sc-5448): build the `VideoGenInput` and run the shared
/// `generate_video` path. The SceneWorks mode resolves to the engine `video_mode` task
/// ([`scail2_engine_video_mode`]) and the source media into the SAM3-painted conditioning
/// ([`resolve_scail2_conditioning`]). A user-selected SCAIL-2 LoRA and the bundled Bias-Aware DPO
/// quality LoRA install as forward-time residuals over the Q4 base ([`resolve_scail2_adapters`],
/// sc-5451 / mlx-gen #462); steps/guidance stay at the engine defaults. Frame count uses the Wan
/// 1-mod-4 stride coercion (the
/// renderer is Wan2.1); the engine stitches > 81-frame clips into overlapping segments internally.
///
/// Returns the decoded clip plus the resolved `lightning` bool so the caller records the effective
/// recipe on the asset without re-resolving the adapters (which discarded errors via
/// `unwrap_or_default`, risking a lightning-flag inconsistency — F-118).
#[cfg(target_os = "macos")]
pub(super) async fn generate_scail2(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, bool)> {
    let negative_prompt = non_empty_negative_prompt(request);
    let conditioning =
        resolve_scail2_conditioning(api, settings, job, request, project_path).await?;
    // Selecting a lightx2v diff-patch "lightning" LoRA flips the worker to the step-distill recipe
    // (8 steps, CFG off, shift 1.0) so the toggle yields the speedup; otherwise steps/guidance/shift
    // stay `None` and the engine's quality defaults stand (sc-5700).
    let adapters = resolve_scail2_adapters(settings, request)?;
    let lightning = scail2_adapters_have_lightning(&adapters);
    let (steps, guidance, scheduler_shift) = scail2_sampling(request, lightning);
    // SCAIL-2 quant matrix (sc-9944, epic 8506): the macOS default install is the lean q4 tier; a
    // q8/bf16 job fetches that subdir on demand before resolving. No-op for a q4 job or an
    // already-present tier. Then descend into the chosen tier subdir when the turnkey ships them (a
    // pre-packed tier loads with quant=None, config.json authoritative); a legacy flat snapshot keeps
    // the root + load-time quant.
    ensure_scail2_tier_present(api, settings, job, request).await?;
    let (model_dir, quant) = resolve_scail2_tier_dir_and_quant(settings, request)?;
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps,
        guidance,
        scheduler_shift,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(scail2_engine_video_mode(&request.mode).to_owned()),
        adapters,
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((decoded, lightning))
}

/// Resolve a `replace_person` request into SCAIL-2 cross-identity replacement conditioning (sc-5452,
/// the **integrated** surface). Unlike the standalone `animate_character` path
/// ([`resolve_scail2_conditioning`], which segments a fresh driving clip), this reuses the masks
/// SceneWorks already computed: the saved person track (native YOLO11 → ByteTrack → SAM3,
/// corrections applied) supplies the per-frame driving masks, and the character's approved reference
/// image is the identity. Driving frames come from the source clip exactly as the Wan-VACE backend
/// loads them ([`load_source_video_frames`]), so the resampled track masks stay frame-aligned 1:1.
/// Replacement keeps the **driving** clip's world (driving mask bg white, reference mask bg black);
/// `video_mode = "replacement"` flips the engine `replace_flag`. SCAIL-2 is a full-character model —
/// it replaces the whole tracked person, so the face_only/full_person `replacementMode` knob and
/// `maskingStrength` are inert (the engine reads only the color masks). Multi-character (the extra
/// references) awaits the engine request-contract extension (sc-5583), so only the first reference
/// is used. Returns the conditioning plus the honest `replacementStatus` for the asset sidecar.
#[cfg(target_os = "macos")]
pub(super) async fn resolve_scail2_replace_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<(Vec<Conditioning>, Value)> {
    let track_id = request.person_track_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a person track (personTrackId).".to_owned(),
        )
    })?;
    let track = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_person_track(&request.project_id, track_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("person track {track_id}: {error}"))
        })?;

    // Driving frames + their per-frame binary person masks — the same source the Wan-VACE backend
    // consumes, loaded identically so the resampled masks align 1:1 with the frames.
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let driving =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let frame_total = driving.len();
    let (binary_masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frame_total,
    )?;
    // The tracked person → blue (person 0); replacement keeps the driving's world → white bg.
    let driving_masks = crate::scail2_masks::paint_track_driving_masks(
        &binary_masks,
        crate::scail2_masks::BG_WHITE,
    );

    // The character identity: the first approved reference image (multi-ref = sc-5583).
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let reference = references.into_iter().next().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Replace Person requires at least one approved character reference image.".to_owned(),
        )
    })?;

    // The reference color mask: a fresh native-SAM3 pass on the reference image → the primary person
    // painted blue on a black background (replacement discards the reference's surrounding world).
    let (sam_model, sam_tokenizer) =
        crate::person_segment_sam3::require_segmenter_weights(settings)?;
    // Heartbeat keepalive + user cancel across the cold SAM3 parse + single-frame propagate
    // (sc-8390 / sc-8807), via the shared blocking-segment helper (sc-8830).
    let ref_mask = scail2_segment_blocking(
        api,
        settings,
        &job.id,
        "scail2 reference segment task",
        move |flag| {
            let masks = crate::person_segment_sam3::segment_all_persons_in_memory(
                &sam_model,
                &sam_tokenizer,
                std::slice::from_ref(&reference),
                Some(flag),
                None,
            )?;
            let mask =
                crate::scail2_masks::paint_reference_mask(&masks, crate::scail2_masks::BG_BLACK)?;
            Ok((mask, reference))
        },
    )
    .await?;
    let (ref_mask, reference) = ref_mask;

    let conditioning = vec![
        Conditioning::Reference {
            image: reference,
            strength: None,
        },
        Conditioning::Mask { image: ref_mask },
        Conditioning::ControlClip {
            frames: driving,
            mask: driving_masks,
            // masking_strength / start_frame / mode are inert for SCAIL-2 (it reads only the color
            // masks); carried at neutral defaults for the shared ControlClip contract.
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        },
    ];
    // maskingStrength is recorded as 1.0 — SCAIL-2 always does a full-character replacement, so the
    // Wan-VACE partial-mask knob does not apply.
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        1.0,
        reference_count,
        frame_total,
        SCAIL2_ADAPTER,
    );
    Ok((conditioning, status))
}

/// Real MLX SCAIL-2 cross-identity replacement (epic 5439 / sc-5452): the integrated backend behind
/// the existing `replace_person` pipeline. Builds the replacement conditioning from the saved person
/// track + character reference ([`resolve_scail2_replace_conditioning`]) and runs the shared
/// `generate_video` path with `video_mode = "replacement"` (engine `replace_flag = true`). Returns
/// the decoded video plus the honest `replacementStatus` (adapter `mlx_scail2`). Mirrors
/// [`generate_wan_vace`]'s return shape so the dispatch folds the status identically.
#[cfg(target_os = "macos")]
pub(super) async fn generate_scail2_replace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value, bool)> {
    let negative_prompt = non_empty_negative_prompt(request);
    let (conditioning, status) =
        resolve_scail2_replace_conditioning(api, settings, job, request, project_path).await?;
    // Same quant-matrix tier resolution as the animate path (sc-9944): fetch a toggled q8/bf16 tier on
    // demand, then descend into the chosen tier subdir (pre-packed ⇒ quant=None) or keep the flat root
    // + load-time quant for a legacy snapshot.
    ensure_scail2_tier_present(api, settings, job, request).await?;
    let (model_dir, quant) = resolve_scail2_tier_dir_and_quant(settings, request)?;
    let adapters = resolve_scail2_adapters(settings, request)?;
    let lightning = scail2_adapters_have_lightning(&adapters);
    let (steps, guidance, scheduler_shift) = scail2_sampling(request, lightning);
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps,
        guidance,
        scheduler_shift,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(scail2_engine_video_mode(&request.mode).to_owned()),
        adapters,
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((decoded, status, lightning))
}
