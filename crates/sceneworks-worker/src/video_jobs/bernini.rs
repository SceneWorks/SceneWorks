#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::ltx::extract_clip_frames;
#[allow(unused_imports)]
use super::prelude::*;
#[cfg(target_os = "macos")]
use super::wan::local_mlx_dir;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::wan::resolve_wan_adapters;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::wan::{generate_video, VideoGenInput};

// ---------------------------------------------------------------------------
// Real MLX Bernini generation (macOS, via mlx-gen-bernini, epic 4699 / sc-4707 + sc-4703 + sc-5425):
// the full Qwen2.5-VL semantic planner + Wan2.2-T2V-A14B dual-expert renderer. Serves the planner
// video task surface: text_to_video (t2v), video_to_video (v2v — source-clip edit),
// reference_to_video (r2v — subject references → video), reference_video_to_video (rv2v — source clip
// + references), multi_video_to_video (mv2v — multiple source clips), ads2v (source video + reference
// video + references). The SceneWorks mode maps to the engine `video_mode` task string and the source
// media is resolved into the planner's `VideoClip` / `MultiReference` conditioning. Q4 default / Q8
// opt-in at load. The turnkey `SceneWorks/bernini-mlx` snapshot is self-contained. (t2i/i2i image
// companion = a separate image-typed catalog id, tracked under epic 4699.)
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Bernini asset.
#[cfg(target_os = "macos")]
pub(super) const BERNINI_ADAPTER: &str = "mlx_bernini";

/// SceneWorks Bernini model id → mlx-gen registry id, or `None` if `model` is not the Bernini family.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn bernini_engine_id(model: &str) -> Option<&'static str> {
    (model == "bernini").then_some("bernini")
}

/// Whether the linked Bernini engine can serve this request now (resolvable weights).
#[cfg(target_os = "macos")]
pub(super) fn bernini_available(_request: &VideoRequest, settings: &Settings) -> bool {
    resolve_bernini_model_dir(settings).is_ok()
}

/// Resolve the Bernini MLX snapshot dir: env override (`SCENEWORKS_MLX_BERNINI_DIR`) → app-managed
/// `<data>/models/mlx/bernini` → the turnkey download-on-first-use `SceneWorks/bernini-mlx` snapshot
/// (mirrors `resolve_wan_model_dir`). Errors clearly if none is present (no stub fallback).
#[cfg(target_os = "macos")]
pub(crate) fn resolve_bernini_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Some(dir) = local_mlx_dir(settings, "SCENEWORKS_MLX_BERNINI_DIR", "bernini") {
        return Ok(dir);
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, "SceneWorks/bernini-mlx") {
        return Ok(dir);
    }
    Err(WorkerError::InvalidPayload(format!(
        "bernini: no MLX weights found. Download the turnkey SceneWorks/bernini-mlx snapshot via the \
         Model Manager, set $SCENEWORKS_MLX_BERNINI_DIR, or place a converted snapshot at {}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join("bernini")
            .display(),
    )))
}

/// MLX quantization for a Bernini load: Q4 default (the validated 128 GB-fitting tier, sc-4709 ~44 GB
/// peak), Q8 opt-in via the advanced `mlxQuantize: 8` control, explicit `<= 0` ⇒ bf16 (power users
/// with ample RAM). Never defaults to bf16 — the snapshot is ~93 GB at bf16. Delegates to the shared
/// [`resolve_mlx_dense_quant`] (sc-8830) — the byte-identical Bernini/SCAIL-2 twin.
#[cfg(target_os = "macos")]
pub(super) fn resolve_bernini_quant(request: &VideoRequest) -> Option<Quant> {
    resolve_mlx_dense_quant(request)
}

// --- Bernini quant-matrix tiers (sc-9945, epic 8506) ---------------------------------------------
// The composite sibling of the Wan quant-matrix tiers. `SceneWorks/bernini-mlx` hosts self-contained
// `q4/` (default) + `q8/` + `bf16/` tier subdirs, each a COMPLETE composite snapshot (the Qwen2.5-VL
// planner components + both Wan renderer experts + the shared dense T5/VAE/tokenizer + config
// sidecars). The legacy flat dense-bf16 layout quantized at LOAD, staging the ~56 GB experts + ~14 GB
// planner backbone as bf16 first; a pre-packed tier loads with no dense-staging peak. Both the video
// (`bernini`) and image (`bernini_image`) load paths resolve through the same tier machinery. The flat
// root files stay for already-shipped workers; a new worker resolves the tier subdirs. Mirrors the
// `wan_tier_*` helpers, but keyed on raw `mlxQuantize` bits so it serves the image lane too.

/// The turnkey SceneWorks Bernini MLX repo (sc-9945). Hosts the `q4/`/`q8/`/`bf16/` tier subdirs.
#[cfg(target_os = "macos")]
const BERNINI_REPO: &str = "SceneWorks/bernini-mlx";

/// Pinned revision for [`BERNINI_REPO`] — the commit that adds the `q4/`/`q8/`/`bf16/` tier subdirs
/// (sc-9945). Pinning the exact commit (not the mutable `main`) stops an upstream re-push from silently
/// swapping a checkpoint the on-demand `q8/*` + `bf16/*` fetch loads (the native downloader still verifies each
/// file's own hash on download). This is the commit that added the `q4/`/`q8/`/`bf16/` tier subdirs
/// (sc-9945), with the exact hosted sizes: q4 37,815,703,819 / q8 55,129,270,617 / bf16 87,591,990,679.
#[cfg(target_os = "macos")]
pub(super) const BERNINI_REVISION: &str = "533d688f16c8f33dc832890c1e16c11921a2019a";

/// The runtime files that make a Bernini tier subdir COMPLETE for the load path: the planner
/// components + both renderer experts + the shared dense T5/VAE + tokenizer + config sidecars + the
/// planner's `mllm/tokenizer.json` (mirrors `bernini_tier_build::TIER_FILES`). The three packable
/// weights (`qwen2_5_vl.safetensors`, `high/low_noise_model.safetensors`) are present in every tier;
/// only their contents differ (packed vs dense).
#[cfg(target_os = "macos")]
pub(super) const BERNINI_TIER_FILES: &[&str] = &[
    "qwen2_5_vl.safetensors",
    "qwen2_5_vl_config.json",
    "connector.safetensors",
    "vit_decoder.safetensors",
    "mask_tokens.safetensors",
    "bernini_planner.json",
    "high_noise_model.safetensors",
    "low_noise_model.safetensors",
    "t5_encoder.safetensors",
    "vae.safetensors",
    "tokenizer.json",
    "config.json",
    "bernini_renderer.json",
    "mllm/tokenizer.json",
];

/// Bernini VIDEO-lane default tier order (no explicit `mlxQuantize`): **q4-first** (sc-10859). The MLX
/// video lane has no Q8 lever, so a silent Q8 default only risks a video-runtime OOM at heavy
/// res/frame counts — see [`wan_tier_order`]. Passed by the video generation path.
#[cfg(target_os = "macos")]
pub(crate) const BERNINI_VIDEO_DEFAULT_TIER_ORDER: &[&str] = &["q4", "q8"];

/// Bernini IMAGE-lane default tier order (no explicit `mlxQuantize`): **q8-first** — epic 10721 /
/// sc-10726's app-wide Q8 default, kept for the image lane (Image Studio has a Q8 picker and the
/// sc-8516 budget's 1024²-image transient is exactly what applies). Passed by `image_jobs::bernini`.
#[cfg(target_os = "macos")]
pub(crate) const BERNINI_IMAGE_DEFAULT_TIER_ORDER: &[&str] = &["q8", "q4"];

/// The Bernini quant-matrix tier search order for a request — preferred tier first, then the
/// always-smaller fallback tiers so a repo missing the preferred subdir still loads (mirrors
/// [`wan_tier_order`]): `mlxQuantize <= 0` ⇒ `bf16`; `>= 8` ⇒ `q8`; an explicit `q4` ⇒ q4-first; and —
/// with NO explicit pick — `default_order`. `bf16` stays OUT of the default order, so a default job
/// never pulls the huge dense tier by accident.
///
/// Bernini is SHARED by the video (`bernini`) and image (`bernini_image`) lanes, whose *default* tiers
/// diverge (sc-10859): the caller passes [`BERNINI_VIDEO_DEFAULT_TIER_ORDER`] (q4-first, video OOM
/// carve-out) or [`BERNINI_IMAGE_DEFAULT_TIER_ORDER`] (q8-first, epic-10721 app-wide default). Only the
/// no-explicit-pick (`None`) arm consults it; every explicit pick is lane-independent.
#[cfg(target_os = "macos")]
pub(super) fn bernini_tier_order(
    bits: Option<i64>,
    default_order: &'static [&'static str],
) -> &'static [&'static str] {
    match bits {
        None => default_order,
        Some(b) if b <= 0 => &["bf16", "q8", "q4"],
        Some(b) if b >= 8 => &["q8", "q4"],
        _ => &["q4", "q8"],
    }
}

/// Whether `dir` is a COMPLETE self-contained Bernini tier snapshot (all [`BERNINI_TIER_FILES`]). A
/// partially-downloaded tier fails this so [`bernini_tier_subdir`] falls through to a smaller complete
/// tier rather than half-loading.
#[cfg(target_os = "macos")]
fn bernini_tier_is_complete(dir: &Path) -> bool {
    BERNINI_TIER_FILES
        .iter()
        .all(|file| dir.join(file).is_file())
}

/// Descend a resolved Bernini repo `root` into the requested quant tier subdir (sc-9945), mirroring
/// [`wan_tier_subdir`]. Returns the first COMPLETE tier in [`bernini_tier_order`], or `None` when the
/// repo has no complete tier subdir — a legacy flat snapshot, where the caller keeps the root +
/// load-time quant.
#[cfg(target_os = "macos")]
pub(super) fn bernini_tier_subdir(
    root: &Path,
    bits: Option<i64>,
    default_order: &'static [&'static str],
) -> Option<PathBuf> {
    bernini_tier_order(bits, default_order)
        .iter()
        .map(|tier| root.join(tier))
        .find(|dir| bernini_tier_is_complete(dir))
}

/// Resolve the Bernini `(model_dir, load-time quant)` for a generation, descending into the
/// quant-matrix tier subdir when the turnkey ships them (sc-9945). A pre-packed tier's config sidecars
/// are authoritative — the planner (`Qwen25VlText::from_weights`, via the `quantization` block) and
/// both renderer experts (`WanTransformer::from_weights`) build packed — so a resolved tier loads with
/// `quant = None`: `mlxQuantize` selects WHICH tier, never a load-time requant (the `bf16/` tier is
/// dense, so `None` ⇒ dense too). A legacy flat snapshot (no tier subdirs) keeps today's behavior: load
/// the root and quantize at load per `legacy_quant`. Shared by the video + image lanes (each passes its
/// own parsed `bits` + `legacy_quant` + `default_order`: [`BERNINI_VIDEO_DEFAULT_TIER_ORDER`] for video,
/// [`BERNINI_IMAGE_DEFAULT_TIER_ORDER`] for image — the no-explicit-pick default diverges per sc-10859).
#[cfg(target_os = "macos")]
pub(crate) fn resolve_bernini_tier_dir_and_quant(
    settings: &Settings,
    bits: Option<i64>,
    legacy_quant: Option<Quant>,
    default_order: &'static [&'static str],
) -> WorkerResult<(PathBuf, Option<Quant>)> {
    let root = resolve_bernini_model_dir(settings)?;
    match bernini_tier_subdir(&root, bits, default_order) {
        Some(tier) => Ok((tier, None)),
        None => Ok((root, legacy_quant)),
    }
}

/// Parse `advanced.mlxQuantize` (int or numeric string) for the Bernini tier selector — the raw bits
/// the tier order keys on (the video lane's twin of the image lane's `resolve_bernini_image_quant`
/// bits). `resolve_bernini_quant` maps the same value to a `Quant`; this keeps the tier selection and
/// the legacy load-time quant in sync off one source.
#[cfg(target_os = "macos")]
fn bernini_quant_bits(request: &VideoRequest) -> Option<i64> {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
}

/// On-demand fetch of a non-default Bernini quant-matrix tier subdir (sc-9945, mirrors
/// [`ensure_wan_tier_present`]). The macOS default download is the lean `q4/` tier; a job that opts into
/// a heavier tier (`mlxQuantize <= 0` ⇒ `bf16`, `>= 8` ⇒ `q8`) pulls just that subdir from the FIXED
/// [`BERNINI_REVISION`] the first time it is requested so [`bernini_tier_subdir`] can resolve it. No-op
/// for a `q4` (default) job, when the repo snapshot isn't downloaded yet (resolve surfaces the clear
/// error), or when the tier is already complete. Fails loud on a real download error — fast, before any
/// compute; a tier that isn't published yet stays absent so resolve falls back to a smaller complete tier.
#[cfg(target_os = "macos")]
pub(crate) async fn ensure_bernini_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    bits: Option<i64>,
) -> WorkerResult<()> {
    let tier = match bits {
        Some(b) if b <= 0 => "bf16",
        Some(b) if b >= 8 => "q8",
        // q4 default — ships with the base install, nothing to fetch on demand.
        _ => return Ok(()),
    };
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, BERNINI_REPO) else {
        return Ok(());
    };
    if bernini_tier_is_complete(&root.join(tier)) {
        return Ok(());
    }
    let files = vec![format!("{tier}/*")];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        BERNINI_REPO,
        BERNINI_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// Raw-settings recorded on a real MLX Bernini asset (mirrors `wan_raw_settings`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn bernini_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    // The engine guidance task the SceneWorks mode resolved to (lineage / observability).
    raw.insert(
        "berniniTask".to_owned(),
        Value::String(bernini_engine_video_mode(&request.mode).to_owned()),
    );
    Value::Object(raw)
}

/// Map a SceneWorks video mode to the Bernini engine `video_mode` task string (which selects the
/// renderer guidance mode). The engine also infers the mode from the supplied conditioning, but the
/// explicit task keeps the mapping unambiguous. Unknown / `text_to_video` ⇒ plain `t2v`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn bernini_engine_video_mode(mode: &str) -> &'static str {
    match mode {
        "video_to_video" => "v2v",
        "reference_to_video" => "r2v",
        "reference_video_to_video" => "rv2v",
        // Multi-source-video modes (sc-5425): mv2v (multiple source clips) and ads2v
        // (source video + reference video + reference images). Both resolve to the
        // engine's `V2vApg` guidance via `task_to_vit_mode`; they differ only in the
        // supplied media.
        "multi_video_to_video" => "mv2v",
        "ads2v" => "ads2v",
        _ => "t2v",
    }
}

/// Resolve the source media for a Bernini editing/reference request into the planner conditioning:
/// source clips → [`Conditioning::VideoClip`] (the edit structure, VAE/ViT-encoded by the engine)
/// and subject reference images → [`Conditioning::MultiReference`]. `text_to_video` needs none.
/// The single-clip modes (v2v / rv2v / ads2v) use `sourceClipAssetId`; mv2v supplies several via
/// `sourceClipAssetIds`; ads2v additionally appends the reference video (`referenceClipAssetId`) as a
/// second clip after the source clip (sc-5425). Clips are emitted videos-first, in submission order,
/// then images — matching the engine's `collect_conditioning` / `assign_source_ids` ordering. Each
/// mode's required media is enforced here (defense in depth — the API validates the same), so a
/// mis-built request fails loudly instead of silently rendering an unconditioned clip. Every clip is
/// decoded to the output frame count (the engine resamples to its `target_fps` grid).
#[cfg(target_os = "macos")]
pub(super) async fn resolve_bernini_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let mode = request.mode.as_str();

    // Source video clips, in the order the engine assigns source ids (videos first; for ads2v the
    // source clip leads the reference video).
    let mut clip_ids: Vec<&str> = Vec::new();
    match mode {
        "video_to_video" | "reference_video_to_video" | "ads2v" => {
            let clip_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "bernini {} requires a source clip (sourceClipAssetId).",
                    request.mode.replace('_', " ")
                ))
            })?;
            clip_ids.push(clip_id);
        }
        "multi_video_to_video" => {
            if request.source_clip_asset_ids.len() < 2 {
                return Err(WorkerError::InvalidPayload(
                    "bernini multi video to video requires at least two source clips \
                     (sourceClipAssetIds)."
                        .to_owned(),
                ));
            }
            clip_ids.extend(request.source_clip_asset_ids.iter().map(String::as_str));
        }
        _ => {}
    }
    if mode == "ads2v" {
        let ref_clip_id = request.reference_clip_asset_id.as_deref().ok_or_else(|| {
            WorkerError::InvalidPayload(
                "bernini ads2v requires a reference video (referenceClipAssetId).".to_owned(),
            )
        })?;
        clip_ids.push(ref_clip_id);
    }

    let mut conditioning = Vec::new();
    for clip_id in clip_ids {
        let frames = extract_clip_frames(
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
        conditioning.push(Conditioning::VideoClip {
            frames,
            frame_idx: 0,
            strength: 1.0,
        });
    }

    // Subject reference images → MultiReference (r2v / rv2v / ads2v).
    let needs_refs = matches!(
        mode,
        "reference_to_video" | "reference_video_to_video" | "ads2v"
    );
    if needs_refs {
        if request.reference_asset_ids.is_empty() {
            return Err(WorkerError::InvalidPayload(format!(
                "bernini {} requires at least one reference image (referenceAssetIds).",
                request.mode.replace('_', " ")
            )));
        }
        let mut images = Vec::with_capacity(request.reference_asset_ids.len());
        for asset_id in &request.reference_asset_ids {
            images.push(load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?);
        }
        conditioning.push(Conditioning::MultiReference { images });
    }

    Ok(conditioning)
}

/// Real MLX Bernini video generation (epic 4699 / sc-4707 + sc-4703 + sc-5425): build the
/// `VideoGenInput` and run the shared `generate_video` path. The SceneWorks mode resolves to the
/// engine `video_mode` task ([`bernini_engine_video_mode`]) and the source media into the planner
/// conditioning ([`resolve_bernini_conditioning`]) — empty for t2v, one or more `VideoClip`s for
/// v2v/mv2v/rv2v/ads2v, and `MultiReference` for r2v/rv2v/ads2v. No LoRA (the engine reports
/// `supports_lora=false`); steps/guidance stay at the engine defaults. Frame count uses the Wan
/// 1-mod-4 stride coercion (the renderer is Wan).
#[cfg(target_os = "macos")]
pub(super) async fn generate_bernini(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let negative_prompt = non_empty_negative_prompt(request);
    let conditioning =
        resolve_bernini_conditioning(api, settings, job, request, project_path).await?;
    // sc-9945: fetch the requested quant tier subdir (q8/bf16) if it isn't the shipped q4 default, then
    // descend into it — a pre-packed tier loads with `quant = None` (config sidecars authoritative); a
    // legacy flat snapshot keeps load-time quant.
    let bits = bernini_quant_bits(request);
    ensure_bernini_tier_present(api, settings, job, bits).await?;
    // Video lane: no explicit pick defaults q4-first (sc-10859 OOM carve-out), unlike the image lane.
    let (model_dir, quant) = resolve_bernini_tier_dir_and_quant(
        settings,
        bits,
        resolve_bernini_quant(request),
        BERNINI_VIDEO_DEFAULT_TIER_ORDER,
    )?;
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
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(bernini_engine_video_mode(&request.mode).to_owned()),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}

// ---------------------------------------------------------------------------
// Real candle Bernini VIDEO generation (Windows/CUDA, via candle-gen-bernini, sc-10997 / epic 6562):
// the off-Mac sibling of the macOS `generate_bernini` above. The full Qwen2.5-VL planner + Wan2.2-T2V-
// A14B renderer registers under `bernini` (`Modality::Video`); the video path reaches it via
// `crate::inference_runtime::load("bernini")` through runtime-cuda's explicit catalog. Serves
// `text_to_video` + the editing/reference/multi-source video
// modes (v2v / r2v / rv2v / mv2v / ads2v); `generate_candle_bernini` maps the SceneWorks mode to the
// engine guidance task and resolves the source media into the planner conditioning. Loads the selected
// published `SceneWorks/bernini` bf16/q8/q4 tier and installs the resolved request LoRA/LoKr stack
// through the renderer's adapter-aware load path. A distinct candle engine — NO torch fallback (a
// missing snapshot fails loud at load, never a silent stub).
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real candle Bernini VIDEO asset — the `candle_<family>` sibling of the MLX
/// `mlx_bernini` label (same spelling as the still lane's `CANDLE_BERNINI_IMAGE_ADAPTER`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const CANDLE_BERNINI_ADAPTER: &str = "candle_bernini";

/// Resolve the source media for a candle Bernini editing/reference request into the planner
/// conditioning — the candle sibling of the macOS `resolve_bernini_conditioning`. Source clips →
/// [`Conditioning::VideoClip`] (the edit structure, VAE/ViT-encoded by the engine) and subject
/// reference images → [`Conditioning::MultiReference`]. `text_to_video` needs none. The single-clip
/// modes (v2v / rv2v / ads2v) use `sourceClipAssetId`; mv2v supplies several via `sourceClipAssetIds`;
/// ads2v additionally appends the reference video (`referenceClipAssetId`) as a second clip after the
/// source clip (sc-5425). Clips are emitted videos-first, in submission order, then images — matching
/// the engine's `collect_conditioning` / `assign_source_ids` ordering. Each mode's required media is
/// enforced here (defense in depth — the API validates the same), so a mis-built request fails loud
/// instead of silently rendering an unconditioned clip. Clips decode to the output frame count via the
/// candle-shared [`extract_clip_frames`]; reference images load via the shared `load_reference_image`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn resolve_candle_bernini_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let mode = request.mode.as_str();

    // Source video clips, in the order the engine assigns source ids (videos first; for ads2v the
    // source clip leads the reference video).
    let mut clip_ids: Vec<&str> = Vec::new();
    match mode {
        "video_to_video" | "reference_video_to_video" | "ads2v" => {
            let clip_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "bernini {} requires a source clip (sourceClipAssetId).",
                    request.mode.replace('_', " ")
                ))
            })?;
            clip_ids.push(clip_id);
        }
        "multi_video_to_video" => {
            if request.source_clip_asset_ids.len() < 2 {
                return Err(WorkerError::InvalidPayload(
                    "bernini multi video to video requires at least two source clips \
                     (sourceClipAssetIds)."
                        .to_owned(),
                ));
            }
            clip_ids.extend(request.source_clip_asset_ids.iter().map(String::as_str));
        }
        _ => {}
    }
    if mode == "ads2v" {
        let ref_clip_id = request.reference_clip_asset_id.as_deref().ok_or_else(|| {
            WorkerError::InvalidPayload(
                "bernini ads2v requires a reference video (referenceClipAssetId).".to_owned(),
            )
        })?;
        clip_ids.push(ref_clip_id);
    }

    let mut conditioning = Vec::new();
    for clip_id in clip_ids {
        let frames = extract_clip_frames(
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
        conditioning.push(Conditioning::VideoClip {
            frames,
            frame_idx: 0,
            strength: 1.0,
        });
    }

    // Subject reference images → MultiReference (r2v / rv2v / ads2v).
    let needs_refs = matches!(
        mode,
        "reference_to_video" | "reference_video_to_video" | "ads2v"
    );
    if needs_refs {
        if request.reference_asset_ids.is_empty() {
            return Err(WorkerError::InvalidPayload(format!(
                "bernini {} requires at least one reference image (referenceAssetIds).",
                request.mode.replace('_', " ")
            )));
        }
        let mut images = Vec::with_capacity(request.reference_asset_ids.len());
        for asset_id in &request.reference_asset_ids {
            images.push(crate::image_jobs::load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?);
        }
        conditioning.push(Conditioning::MultiReference { images });
    }

    Ok(conditioning)
}

/// Real candle Bernini video generation (sc-10997 / epic 6562): build the `VideoGenInput` and run the
/// shared [`generate_video`] path. The SceneWorks mode resolves to the engine `video_mode` task
/// ([`bernini_engine_video_mode`]) and the source media into the planner conditioning
/// ([`resolve_candle_bernini_conditioning`]) — empty for t2v, one or more `VideoClip`s for
/// v2v/mv2v/rv2v/ads2v, and `MultiReference` for r2v/rv2v/ads2v. The converted `SceneWorks/bernini`
/// snapshot descends into the requested quant tier subfolder (`bf16/`|`q8/`|`q4/`, sc-11003) via the
/// shared [`crate::image_jobs::resolve_candle_bernini_tier_dir_and_quant`] so video + still load from
/// the same tier: no explicit `mlxQuantize` ⇒ bf16 dense, `:4`|`:8` opt into the packed tiers. The
/// candle descriptor advertises LoRA/LoKr, and the resolved request adapter stack is installed at
/// load; steps / guidance stay at the engine defaults. Frame count uses the Wan 1-mod-4 stride (the
/// renderer is Wan).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn generate_candle_bernini(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let negative_prompt = non_empty_negative_prompt(request);
    let conditioning =
        resolve_candle_bernini_conditioning(api, settings, job, request, project_path).await?;
    // Select the published tier subfolder + matching load quant (sc-11003): parse `mlxQuantize` (int
    // or numeric string) the same way the still lane does, defaulting to bf16 dense.
    let tier_bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    let (model_dir, quant) =
        crate::image_jobs::resolve_candle_bernini_tier_dir_and_quant(settings, tier_bits)?;
    let adapters = resolve_wan_adapters(settings, request, engine_id)?;
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        adapters,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(bernini_engine_video_mode(&request.mode).to_owned()),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}
