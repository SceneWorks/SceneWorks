// ---------------------------------------------------------------------------
// Bernini still-image companion (macOS, epic 4699 / sc-5424): the full Qwen2.5-VL semantic planner +
// Wan2.2-A14B renderer producing a SINGLE image. The Bernini engine descriptor is `Modality::Both`,
// so the image-typed `bernini_image` catalog id maps to the SAME `engine_id:"bernini"` the video
// `bernini` id uses (mirroring `z_image_edit → z_image_turbo` — two ids, one engine), and the still
// surface is served here through Image Studio + `image_jobs` instead of Video Studio.
//
// Two tasks: t2i (text→image, no conditioning) and i2i (`edit_image` — the source image, resolved
// from `sourceAssetId`, is handed to the engine as a `Conditioning::Reference`; the planner ViT/VAE-
// encodes it at its own native resolution, so the worker does NOT pre-fit it to the output W×H). The
// worker forces `frames:1` + `video_mode:"t2i"|"i2i"` so the engine returns `GenerationOutput::Images`
// (a single still) rather than a clip. No LoRA (the descriptor reports `supports_lora:false`); steps +
// guidance flow through the standard resolvers (the descriptor advertises `supports_guidance:true` +
// `supports_negative_prompt:true`). Q4 default / Q8 opt-in at load. The turnkey `SceneWorks/bernini-mlx`
// snapshot is shared with the video id (resolved via [`crate::video_jobs::resolve_bernini_model_dir`]).
//
// Scope note: the engine's i2i is a planner-guided structural re-render (the source feeds the ViT/VAE
// conditioning), NOT a denoise-strength img2img — the `Conditioning::Reference` strength is ignored by
// the engine, so the worker passes `None` and does not surface a strength knob that would do nothing.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Bernini image asset (matches the `bernini_image` MODEL_TABLE
/// row's `adapter_label`, so the per-asset `adapter` + the generation-set `adapter` agree).
#[cfg(target_os = "macos")]
const BERNINI_IMAGE_ADAPTER: &str = "mlx_bernini";

/// True when this is a Bernini still-image job whose weights resolve: the `bernini_image` id + a
/// resolvable snapshot dir (env override → app-managed → turnkey download). Both t2i and i2i route
/// here (i2i adds the source conditioning); plain t2i is NOT served by the generic `mlx_available`
/// path because that path leaves `frames`/`video_mode` unset, which the engine would treat as a
/// (multi-frame) video request — so Bernini stills must use this dedicated path.
#[cfg(target_os = "macos")]
fn bernini_image_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "bernini_image"
        && crate::video_jobs::resolve_bernini_model_dir(settings).is_ok()
}

/// The Bernini engine task string for a SceneWorks image mode: `edit_image` → `i2i` (source-image
/// edit), everything else → `t2i` (text→image). Selects the engine guidance/conditioning path
/// (`resolve_vit_mode`/`task_to_vit_mode`); both still tasks resolve to `vae_txt_vit_wapg`. Shared by
/// the MLX and candle (sc-10996) still lanes — the engine task string is backend-neutral.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn bernini_image_engine_task(mode: &str) -> &'static str {
    if mode == "edit_image" {
        "i2i"
    } else {
        "t2i"
    }
}

/// MLX quantization for a Bernini image load: Q4 default (the validated 64 GB-fitting tier, sc-4709
/// ~44 GB peak), Q8 opt-in via the advanced `mlxQuantize:8` control, explicit `<= 0` ⇒ bf16 (power
/// users with ample RAM). Mirrors the video path's [`crate::video_jobs::resolve_bernini_quant`] (Q4
/// default, not the generic image `resolve_quant`'s Q8 default — the snapshot is ~93 GB at bf16).
#[cfg(target_os = "macos")]
fn resolve_bernini_image_quant(request: &ImageRequest) -> (Option<Quant>, Option<i64>) {
    match request.advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    }) {
        Some(bits) if bits <= 0 => (None, None),
        Some(bits) if bits <= 4 => (Some(Quant::Q4), Some(4)),
        Some(_) => (Some(Quant::Q8), Some(8)),
        None => (Some(Quant::Q4), Some(4)),
    }
}

/// Flat telemetry for a real Bernini image generation (parity with the other edit handlers +
/// `bernini_raw_settings`). Records the engine task so the lineage shows whether the planner ran t2i
/// or i2i, plus the standard repo/steps/guidance/quant knobs. Shared by the MLX and candle (sc-10996)
/// still lanes — the recorded knobs are backend-neutral (the candle lane records `candle` in `backend`
/// and `editEngine` stays `bernini`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn bernini_image_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    task: &str,
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
    raw.insert("berniniTask".to_owned(), Value::String(task.to_owned()));
    raw.insert("editEngine".to_owned(), Value::String("bernini".to_owned()));
    raw
}

/// Generate one Bernini still (RGB8) at `seed`. Builds the engine request with `frames:Some(1)` +
/// `video_mode:Some(task)` so the engine returns a single image, and the (optional) i2i source as the
/// shared `conditioning`. Standard guidance family (`guidance` carries the CFG scale, negative prompt
/// forwarded); no LoRA.
#[allow(clippy::too_many_arguments)]
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn bernini_image_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    task: &'static str,
    conditioning: Vec<Conditioning>,
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
        // A single still: `frames == 1` makes the engine return `GenerationOutput::Images`.
        frames: Some(1),
        video_mode: Some(task.to_owned()),
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::Engine(format!("Bernini image generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images
                .pop()
                .ok_or_else(|| WorkerError::Engine("Bernini image produced no image".to_owned()))?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "Bernini image returned non-image output".to_owned(),
        )),
    }
}

/// Real MLX Bernini still-image generation (epic 4699 / sc-5424): load the full planner+renderer once
/// (Q4 default), then one image per seed — t2i from the prompt alone, or i2i conditioned on the
/// `sourceAssetId` source. Mirrors [`generate_sensenova_edit_stream`]'s blocking-thread + streamed-
/// events shape; differs in forcing `frames:1` + the engine task string, no negative-prompt/CFG
/// special-casing (standard guidance family), and no reference-fit (the engine resizes the source
/// internally).
#[cfg(target_os = "macos")]
async fn generate_bernini_image_stream(
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
    let engine_id = model.engine_id();
    let backend = if model.backend().is_empty() {
        backend
    } else {
        model.backend()
    };
    let (quant, quant_bits) = resolve_bernini_image_quant(request);
    // sc-9945: raw `mlxQuantize` bits for the quant-matrix tier selector (distinct from `quant_bits`,
    // which is `None` for BOTH bf16 and the q4 default — the tier order must tell `<= 0` bf16 apart).
    // Fetch the requested tier subdir (q8/bf16) if not the shipped q4 default, then descend into it: a
    // pre-packed tier loads with `quant = None` (config sidecars authoritative); a legacy flat snapshot
    // keeps load-time quant (`quant` above).
    let tier_bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    crate::video_jobs::ensure_bernini_tier_present(api, settings, job, tier_bits).await?;
    // Image lane keeps the epic-10721 app-wide Q8 default on a no-explicit-pick (sc-10726); only the
    // video lane reverts to q4-first (sc-10859), so pass the image default order explicitly.
    let (weights_dir, quant) = crate::video_jobs::resolve_bernini_tier_dir_and_quant(
        settings,
        tier_bits,
        quant,
        crate::video_jobs::BERNINI_IMAGE_DEFAULT_TIER_ORDER,
    )?;
    let steps = resolve_steps(request, &model);
    // Standard guidance family: `guidance` carries the CFG scale (engine `omega_txt`); the negative
    // prompt is forwarded (descriptor advertises both). No true-CFG.
    let guidance = resolve_guidance(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);
    let repo = model_repo(request, &model);
    let task = bernini_image_engine_task(&request.mode);

    // i2i (`edit_image`): resolve the source image into the engine's `Conditioning::Reference`. The
    // engine ViT/VAE-encodes it at native resolution (no worker-side fit), and ignores the reference
    // strength (planner-guided structural re-render, not a denoise-strength img2img). t2i has no
    // conditioning. The routing gate (`bernini_image_mlx_eligible`) already requires a `sourceAssetId`
    // for `edit_image`, so this is defense in depth.
    let conditioning: Vec<Conditioning> = if task == "i2i" {
        let source_id = request
            .source_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "bernini_image edit_image requires a source image (sourceAssetId).".to_owned(),
                )
            })?;
        let image = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            source_id,
            project_path,
        )?;
        vec![Conditioning::Reference {
            image,
            strength: None,
        }]
    } else {
        Vec::new()
    };

    let raw_settings =
        bernini_image_raw_settings(request, &repo, steps, quant_bits, guidance, task);
    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count).map(|index| resolve_seed(request, index)).collect();
    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);

    let spec = load_spec(weights_dir, quant, Vec::new(), None);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        0,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(tx, seeds, move |_index, seed, on_progress| {
                let (out_w, out_h, pixels) = bernini_image_generate_one(
                    generator,
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    task,
                    conditioning.clone(),
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
        BERNINI_IMAGE_ADAPTER,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// Candle (Windows/CUDA) Bernini still-image lane (sc-10996, epic 6562)
//
// The off-Mac sibling of `generate_bernini_image_stream` above: the full candle Bernini planner+renderer
// `Generator` (candle-gen-bernini, engine id `bernini`, `Modality::Video`) is reached via
// `gen_core::load("bernini")` with `frames:1` + `video_mode:"t2i"|"i2i"` so it returns a SINGLE still —
// exactly the MLX shape, only the tensor backend + weights snapshot differ. Not `is_candle_engine` (the
// engine is video-modality), so it rides this dedicated stream rather than the generic
// `generate_candle_stream` txt2img lane. Shares the neutral harness (`load_spec` /
// `start_cached_gen_stream` / `drive_gen_items` / `bernini_image_generate_one` / `consume_gen_events`)
// and the backend-neutral resolvers (`resolve_steps` / `resolve_guidance` / `resolve_negative_prompt`)
// with the MLX path. GPU-val is BLOCKED on the converted `SceneWorks/bernini` weights (sc-11003,
// the 168 GB conversion, not yet published), so this delivers routing + lane wiring; a missing snapshot
// fails loud at load (never a silent stub).
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real candle Bernini image asset — the `candle_<family>` sibling of the MLX
/// `mlx_bernini` label (parity with `candle_adapter_label`, though Bernini's bespoke stream passes it
/// directly rather than through that generic map).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_BERNINI_IMAGE_ADAPTER: &str = "candle_bernini";

/// The turnkey candle Bernini snapshot repo (sc-11003): the converted full-Bernini tree (`transformer/`
/// `transformer_2/` `text_encoder/` `vae/` `tokenizer/` `mllm/` `connector/` `vit_decoder/`) the
/// `candle-gen-bernini` `load` reads. Distinct from the MLX `SceneWorks/bernini-mlx` turnkey (different
/// tensor layout). Published PUBLIC at `SceneWorks/bernini` with bf16/Q8/Q4 tiers (sc-11003).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_BERNINI_REPO: &str = "SceneWorks/bernini";

/// True when this is a candle Bernini still-image job: the `bernini_image` id. Routed on the model id
/// alone (like the sdxl candle txt2img arm) — NOT weight-gated — so a missing snapshot fails loud at
/// load rather than silently stubbing, and the routing decision needs no staged 168 GB weights. Both
/// t2i and i2i (`edit_image`) route here; the stream enforces the i2i source (`sourceAssetId`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn bernini_image_candle_available(request: &ImageRequest) -> bool {
    request.model == "bernini_image"
}

/// The candle Bernini quant-tier subfolders published under the turnkey `SceneWorks/bernini` root
/// (sc-11003): each holds a full component tree (`transformer/` `transformer_2/` `mllm/` `connector/`
/// `vit_decoder/` `vae/` …) at that precision. `bf16/` is the dense default; `q8/`/`q4/` are the
/// packed tiers (validated clean on sm_120, sc-11003). Mirrors the MLX `q4`/`q8`/`bf16` convention
/// ([`crate::video_jobs::bernini_tier_subdir`]).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_BERNINI_TIERS: &[&str] = &["bf16", "q8", "q4"];

/// True when `dir` carries a Bernini component tree — sentinel = the `transformer/` subdir (the
/// converted renderer's first DiT expert; the `candle-gen-bernini` `load` validates the full set and
/// reports the precise gap). Used both for a tier subfolder (`<root>/bf16/transformer`) and a legacy
/// flat snapshot (`<root>/transformer`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_bernini_tree_present(dir: &Path) -> bool {
    dir.join("transformer").is_dir()
}

/// True when `root` is a usable candle Bernini snapshot: it either nests the published tier
/// subfolders (`bf16/`|`q8/`|`q4/` each with a `transformer/` tree) or is a legacy flat tree
/// (`transformer/` AT root). The env/managed sentinels accept either shape.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_bernini_snapshot_ok(root: &Path) -> bool {
    candle_bernini_tree_present(root)
        || CANDLE_BERNINI_TIERS
            .iter()
            .any(|tier| candle_bernini_tree_present(&root.join(tier)))
}

/// Resolve the candle Bernini snapshot ROOT: `SCENEWORKS_CANDLE_BERNINI_DIR` override → app-managed
/// `<data>/models/candle/bernini` → the turnkey `SceneWorks/bernini` HF snapshot. Sentinel =
/// [`candle_bernini_snapshot_ok`] (a `transformer/` tree at root OR under a `bf16/`|`q8/`|`q4/` tier
/// subfolder — the published layout, sc-11003). Errors loudly when absent — like the candle SCAIL-2 /
/// Wan-VACE resolvers, a missing checkpoint surfaces a clear re-download error instead of degrading to
/// a stub. The candle sibling of the MLX `resolve_bernini_model_dir` (video_jobs.rs). Callers descend
/// into the requested quant tier via [`resolve_candle_bernini_tier_dir_and_quant`].
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn resolve_candle_bernini_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_CANDLE_BERNINI_DIR") {
        let path = PathBuf::from(dir.trim());
        if candle_bernini_snapshot_ok(&path) {
            return Ok(path);
        }
    }
    let managed = settings
        .data_dir
        .join("models")
        .join("candle")
        .join("bernini");
    if candle_bernini_snapshot_ok(&managed) {
        return Ok(managed);
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, CANDLE_BERNINI_REPO) {
        return Ok(dir);
    }
    Err(WorkerError::InvalidPayload(format!(
        "bernini (candle): no weights found. Download the turnkey {CANDLE_BERNINI_REPO} snapshot via \
         the Model Manager, set $SCENEWORKS_CANDLE_BERNINI_DIR, or place a converted full-Bernini \
         snapshot (bf16/|q8/|q4/ tier subfolders — or a flat tree — each with transformer/ + \
         transformer_2/ + text_encoder/ + vae/ + tokenizer/ + mllm/ + connector/ + vit_decoder/) at {}.",
        managed.display(),
    )))
}

/// The candle Bernini tier search order for a request's `mlxQuantize` bits — preferred tier first,
/// then the always-present fallbacks so a partial repo still loads (mirrors
/// [`crate::video_jobs::bernini_tier_order`]). The candle DEFAULT is **bf16 dense** (no explicit pick
/// ⇒ `bf16`), NOT the MLX q4-first default: off-Mac the box has ample VRAM and the dense path is the
/// validated baseline, so a packed tier is strictly opt-in (`mlxQuantize:4`|`:8`). `<= 0` also pins
/// bf16.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_bernini_tier_order(bits: Option<i64>) -> &'static [&'static str] {
    match bits {
        None => &["bf16", "q8", "q4"],
        Some(b) if b <= 0 => &["bf16", "q8", "q4"],
        Some(b) if b <= 4 => &["q4", "q8", "bf16"],
        Some(_) => &["q8", "q4", "bf16"],
    }
}

/// The load-time [`Quant`] a resolved candle Bernini tier subfolder loads at: `q4/` ⇒ [`Quant::Q4`],
/// `q8/` ⇒ [`Quant::Q8`], `bf16/` ⇒ `None` (dense). Passed to [`load_spec`] so the packed tiers
/// (validated clean on sm_120, sc-11003) build packed while the default stays dense.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_bernini_tier_quant(tier: &str) -> Option<Quant> {
    match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        _ => None,
    }
}

/// Resolve the candle Bernini `(weights_dir, load-time quant)` for a generation: descend the resolved
/// snapshot root into the requested quant-tier subfolder (`bf16/`|`q8/`|`q4/`, sc-11003) and pair it
/// with the matching load quant. The published `SceneWorks/bernini` layout nests each tier's component
/// tree under a tier subdir, so the dense/default path loads `bf16/` (quant `None`) and an explicit
/// Q4/Q8 pick loads `q4/`|`q8/` with [`Quant::Q4`]/[`Quant::Q8`]. Falls back through the smaller
/// complete tiers so a partial repo still loads, then to `root` itself when it is a legacy flat tree
/// (`transformer/` AT root, dense). Errors loud when neither a tier subfolder nor the root carries a
/// `transformer/` tree (defense in depth — the root resolver already gates on the same sentinel).
/// Shared by the still ([`generate_candle_bernini_image_stream`]) and video
/// ([`crate::video_jobs::generate_candle_bernini`]) lanes so both load the same tier.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn resolve_candle_bernini_tier_dir_and_quant(
    settings: &Settings,
    bits: Option<i64>,
) -> WorkerResult<(PathBuf, Option<Quant>)> {
    let root = resolve_candle_bernini_model_dir(settings)?;
    for tier in candle_bernini_tier_order(bits) {
        let dir = root.join(tier);
        if candle_bernini_tree_present(&dir) {
            return Ok((dir, candle_bernini_tier_quant(tier)));
        }
    }
    // Legacy flat snapshot: the component tree sits directly at the root (dense, no tier subdirs).
    if candle_bernini_tree_present(&root) {
        return Ok((root, None));
    }
    Err(WorkerError::InvalidPayload(format!(
        "bernini (candle): the resolved snapshot at {} has no tier subfolder (bf16/|q8/|q4/) with a \
         transformer/ tree, nor a flat transformer/ at root. Re-download the turnkey \
         {CANDLE_BERNINI_REPO} snapshot via the Model Manager.",
        root.display(),
    )))
}

/// Real candle Bernini still-image generation (sc-10996, epic 6562): the off-Mac sibling of
/// [`generate_bernini_image_stream`]. Loads the full candle planner+renderer once from the converted
/// `SceneWorks/bernini` snapshot (dense — the candle loader reads the converted tree as-is), then
/// one image per seed: t2i from the prompt alone, or i2i conditioned on the `sourceAssetId` source (the
/// engine ViT/VAE-encodes it at native resolution, no worker-side fit). Forces `frames:1` + the engine
/// task string so the video-modality engine returns a single still. Standard guidance family (no LoRA;
/// the descriptor reports `supports_lora:false`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_bernini_image_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    _device_backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    // Join the MODEL_TABLE `bernini_image` row with the linked candle `bernini` descriptor (the same
    // backend-neutral resolver the MLX path + `generate_candle_stream` use). `None` means the candle
    // provider crate wasn't linked/registered — fail loud rather than silently stubbing.
    let model = mlx_model(&request.model).ok_or_else(|| {
        WorkerError::Engine(format!(
            "candle backend not linked for model {} (no registered generator)",
            request.model
        ))
    })?;
    let engine_id = model.engine_id();
    // Report the descriptor's tensor backend ("candle") on the streamed events (parity with
    // `generate_candle_stream`), so the worker log + the UI architecture pill attribute the run to Candle.
    let backend = if model.backend().is_empty() {
        "candle"
    } else {
        model.backend()
    };
    let steps = resolve_steps(request, &model);
    // Standard guidance family: `guidance` carries the CFG scale (engine `omega_txt`); the negative
    // prompt is forwarded (the descriptor advertises both). No true-CFG.
    let guidance = resolve_guidance(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);
    let repo = model_repo(request, &model);
    let task = bernini_image_engine_task(&request.mode);
    // Requested tier bits (advanced `mlxQuantize`): selects WHICH published tier subfolder
    // (`bf16/`|`q8/`|`q4/`) the candle lane loads and the matching load-time quant (sc-11003). No
    // explicit pick ⇒ bf16 dense (the off-Mac validated baseline); `mlxQuantize:4`|`:8` opt into the
    // packed tiers (validated clean on sm_120, sc-11003).
    let tier_bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    let (weights_dir, quant) = resolve_candle_bernini_tier_dir_and_quant(settings, tier_bits)?;

    // i2i (`edit_image`): resolve the source image into the engine's `Conditioning::Reference` (the
    // engine ViT/VAE-encodes it at native resolution, no worker-side fit, and ignores the reference
    // strength — a planner-guided structural re-render). t2i has no conditioning. The routing gate
    // (`bernini_image_edit_candle_eligible`) already requires a `sourceAssetId` for `edit_image`, so
    // this is defense in depth. Mirrors the MLX path exactly.
    let conditioning: Vec<Conditioning> = if task == "i2i" {
        let source_id = request
            .source_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "bernini_image edit_image requires a source image (sourceAssetId).".to_owned(),
                )
            })?;
        let image = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            source_id,
            project_path,
        )?;
        vec![Conditioning::Reference {
            image,
            strength: None,
        }]
    } else {
        Vec::new()
    };

    let raw_settings =
        bernini_image_raw_settings(request, &repo, steps, tier_bits, guidance, task);
    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count).map(|index| resolve_seed(request, index)).collect();
    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);

    // Load the resolved tier subfolder at its matching quant: `bf16/` dense (quant `None`), or the
    // packed `q4/`|`q8/` tree with `Quant::Q4`|`Quant::Q8` (sc-11003).
    let spec = load_spec(weights_dir, quant, Vec::new(), None);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        0,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(tx, seeds, move |_index, seed, on_progress| {
                let (out_w, out_h, pixels) = bernini_image_generate_one(
                    generator,
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    task,
                    conditioning.clone(),
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
        CANDLE_BERNINI_IMAGE_ADAPTER,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
