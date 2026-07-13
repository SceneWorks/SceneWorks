// Candle (Windows/CUDA) Krea 2 pose-ControlNet route (sc-8464, epic 8459) — `krea_2_turbo` +
// `advanced.poses` off-Mac via `candle_gen_krea::Krea2Control`. The first Krea backbone control lane and
// the deployable form of the sc-8460 spike: a trained control-branch overlay loaded on the frozen Krea 2
// Turbo base (dense bf16), rendering one image per library pose, each conditioned on a full DWPose
// skeleton (rendered cross-platform by `openpose_skeleton::draw_wholebody`, the SAME renderer training
// used). True pose lock via a residual added to the single CFG-free guidance forward, scaled by
// `control_scale`; `control_scale = 0` is engine-proven byte-identical to base txt2img.
//
// **Candle-only.** There is no MLX Krea control twin yet (8459 S5 / sc-8465); this whole file is gated to
// the Windows/CUDA candle build (the `include!` in image_jobs.rs carries the cfg). It is `include!`d into
// the `image_jobs` module, so it shares that module's imports (`parse_poses`/`pose_entries`/`Settings`/
// `WorkerResult`/`huggingface_snapshot_dir`/`start_gen_stream`/… all in scope unqualified).
//
// The base is any complete Krea 2 Turbo diffusers snapshot (`transformer/ text_encoder/ vae/ tokenizer/`):
// the legacy dense `krea/Krea-2-Turbo`, OR — the common case now that the dense download is retired — the
// installed `SceneWorks/krea-2-turbo-mlx` tier the txt2img lane uses (q8 default / q4 / bf16). The control
// branch is a composable-forward overlay (`KreaTrainDit`) trained against the bf16 base; the packed q4/q8
// tiers are key-compatible and load via candle-gen's dequant-on-load (composable DiT reconstructs the dense
// grid from the packed triple — candle-gen #471, sc-11727), so q8 renders ≈ bf16 and q4 stays pose-locked.
// Peak VRAM ≈ dense (dequant-to-bf16 in VRAM), well within a single 96 GB card.

/// The dense Krea 2 Turbo diffusers repo when the manifest omits `repo` — a bring-your-own / legacy base
/// (the manifest download entry was retired in favor of the `SceneWorks/krea-2-turbo-mlx` tiers, sc-9092).
/// The control provider loads the dense bf16 composable base the overlay trained on; the packed mlx tiers
/// below are key-compatible (the bf16 tier mirrors this tree) and load via candle-gen's dequant-on-load.
const KREA_CONTROL_BASE_REPO: &str = "krea/Krea-2-Turbo";
/// The `SceneWorks/krea-2-turbo-mlx` turnkey (q8 default / q4 / bf16 self-contained subdirs) — the SAME
/// base the txt2img `krea_2_turbo` lane installs and loads. Now that the dense `krea/Krea-2-Turbo` download
/// is retired, this is what a user actually has on disk, so the control base resolves the installed tier
/// here (via the shared [`krea_model_subdir`]) when the legacy dense repo is absent. candle-gen packed-
/// detects the tier and the composable control DiT dequantizes it on load (candle-gen #471, sc-11727):
/// q8 renders ≈ bf16, q4 stays pose-locked (mild haze) — GPU-proven.
const KREA_CONTROL_MLX_REPO: &str = "SceneWorks/krea-2-turbo-mlx";
/// Pose ControlNet conditioning-scale default (candle-gen `Krea2Control::DEFAULT_CONTROL_SCALE`). The S0
/// spike found the usable band ~0.5–0.85 for the distilled CFG-free base; ship a comfortable mid.
const KREA_CONTROL_DEFAULT_SCALE: f32 = candle_gen_krea::DEFAULT_CONTROL_SCALE;
/// Hard cap on the exposed `control_scale` — above ~0.85 the frozen CFG-free base over-drives to halftone
/// (S0 finding: graceful soft-haze, never confetti, but not a usable range).
const KREA_CONTROL_SCALE_CAP: f32 = 0.85;
/// Denoise-steps default — the distilled Turbo schedule (8-step CFG-free).
const KREA_CONTROL_DEFAULT_STEPS: u32 = 8;
/// The adapter/engine id recorded on candle Krea control assets (distinct from the `candle_krea` txt2img
/// lane).
const KREA_CONTROL_ENGINE: &str = "candle_krea_control";
/// The [`STRICT_CONTROL_ENGINES`] catalog id this lane validates `advanced.controlMode` against (the Krea
/// pose-only row — `{Pose}`).
const KREA_CONTROL_ENGINE_ID: &str = "krea_2_turbo_control";
/// Env override pointing directly at a Krea 2 Turbo dense diffusers snapshot dir (validation / bring-your-
/// own base) — bypasses the HF-cache resolve.
const KREA_CONTROL_BASE_ENV: &str = "SCENEWORKS_KREA_CONTROL_BASE";
/// Env override pointing directly at a trained control-branch overlay `.safetensors` (validation against
/// the spike checkpoint / bring-your-own) — bypasses the hosted-overlay resolve + download.
const KREA_CONTROL_WEIGHTS_ENV: &str = "SCENEWORKS_CONTROLNET_KREA";
/// Default published Krea pose control-branch overlay repo (sc-8466) — the S0 spike (5,000-step)
/// checkpoint, hosted so the overlay downloads/provisions like the other control repos (the FLUX.2
/// `FLUX2_CONTROL_CANDLE_REPO` precedent) when the user hasn't selected a studio-trained overlay
/// (B4/sc-10165). EXPERIMENTAL / not-for-production: an 8-step CFG-free feasibility overlay, usable
/// pose-lock ~0.5–0.85 (S0). A studio-trained overlay (resolved to `controlWeights.path`) always overrides.
const KREA_CONTROL_OVERLAY_REPO: &str = "SceneWorks/krea2-pose-controlnet-beta";
/// The overlay weight file within [`KREA_CONTROL_OVERLAY_REPO`] (the final 5k-step checkpoint; the repo
/// also carries the 4.5k for comparison).
const KREA_CONTROL_OVERLAY_FILE: &str = "control_step5000.safetensors";
/// Pinned revision for the default overlay repo (defense-in-depth: `main` moving under us can't swap the
/// checkpoint we load — mirrors `FLUX2_CONTROL_CANDLE_REVISION` / sc-9879). Applied ONLY to the default
/// repo; a `controlWeights.repo` override keeps `main`. `ensure_hf_cached_file` still verifies the file's
/// `lfs.oid` from HF's tree API.
const KREA_CONTROL_OVERLAY_REVISION: &str = "cb3a0ac7590f5ec594a4eeb43b95ee1da0b5a0ac";

/// Model ids the candle Krea strict-pose control route accepts (the deployed base the overlay applies on).
fn is_krea_control_model(model: &str) -> bool {
    model == "krea_2_turbo"
}

/// Resolve the Krea 2 Turbo dense diffusers snapshot: the `SCENEWORKS_KREA_CONTROL_BASE` env → an explicit
/// `modelPath` (advanced or manifest) → the HF cache snapshot for the manifest `repo` (default
/// `krea/Krea-2-Turbo`). `None` ⇒ not present locally (the job is not candle-runnable). Mirrors
/// `resolve_flux2_control_base`.
fn resolve_krea_control_base(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if let Ok(env_dir) = std::env::var(KREA_CONTROL_BASE_ENV) {
        let p = PathBuf::from(env_dir.trim());
        if p.is_dir() {
            return Ok(Some(p));
        }
    }
    if let Some(path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
    {
        return resolve_app_managed_model_dir(settings, &path, "Krea control modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(KREA_CONTROL_BASE_REPO);
    // Legacy / bring-your-own dense diffusers base (`krea/Krea-2-Turbo`), if separately cached — keeps
    // existing dense-install behavior byte-identical.
    if let Some(dense) = huggingface_snapshot_dir(&settings.data_dir, repo) {
        return Ok(Some(dense));
    }
    // The installed `SceneWorks/krea-2-turbo-mlx` tier the user actually has (q8 default / q4 / bf16),
    // resolved EXACTLY like the txt2img lane (`krea_model_subdir` honours `advanced.mlxQuantize` and falls
    // back to any downloaded tier — so a q4-only or q8-only install resolves). candle-gen `from_dir`
    // packed-detects the tier; the composable control DiT dequantizes the packed base on load (candle-gen
    // #471, sc-11727). Gate on `transformer/` so a partial download surfaces "base not installed" rather
    // than half-loading.
    if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, KREA_CONTROL_MLX_REPO) {
        let tier = krea_model_subdir(&root, request);
        if tier.join("transformer").is_dir() {
            return Ok(Some(tier));
        }
    }
    Ok(None)
}

/// True when this is a candle-eligible Krea 2 strict-pose job: `krea_2_turbo` with a non-empty
/// `advanced.poses`, not edit mode, whose dense base resolves locally. Mirrors
/// `jobs_store::krea_control_candle_eligible` so the worker and router agree. The overlay weights are NOT
/// part of the gate: they are resolved on first use in the stream.
fn krea_control_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_krea_control_model(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_krea_control_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → default (8).
fn krea_control_candle_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", KREA_CONTROL_DEFAULT_STEPS, 1..=50)
}

/// The (repo, filename) of the hosted control overlay — `advanced.controlWeights.{repo,filename}`
/// overrides (a not-yet-cached registered/hosted overlay the API passed through), else the default
/// published beta overlay. Mirrors `flux2_control_candle_repo_file`; the filename must be a plain
/// component (sc-8821 / F-019).
fn krea_control_overlay_repo_file(request: &ImageRequest) -> WorkerResult<(String, String)> {
    let cw = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object);
    let pick = |key: &str, default: &str| {
        cw.and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    Ok((
        pick("repo", KREA_CONTROL_OVERLAY_REPO),
        safe_weight_filename(
            &pick("filename", KREA_CONTROL_OVERLAY_FILE),
            "advanced.controlWeights.filename",
        )?,
    ))
}

/// Confine a payload-supplied `advanced.controlWeights.path` to an app-managed root (sc-11168 / F-006).
/// The API writes this key for a studio-trained / registered LOCAL overlay (B4/sc-10165), but the value
/// arrives untrusted across the LAN boundary (epic 4484), so — like every other on-disk model input — it
/// must resolve under the app data dir / HF hub cache (or a declared external root) via the house
/// `normalize_app_managed_model_path`; without this a crafted job could point the loader at any file on
/// disk (an arbitrary-file-read primitive). Returns `Ok(None)` when the payload carries no path, `Ok(Some)`
/// for a confined path (whether or not it exists — the caller checks `is_file`), and the same
/// `InvalidPayload` rejection as the sibling lanes for an out-of-root path. Mirrors the MLX twin.
fn krea_control_payload_overlay_path(
    settings: &Settings,
    request: &ImageRequest,
) -> WorkerResult<Option<PathBuf>> {
    let Some(path) = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object)
        .and_then(|cw| cw.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    Ok(Some(crate::paths::normalize_app_managed_model_path(
        settings,
        path,
        "Krea control overlay",
    )?))
}

/// Resolve the control-branch overlay checkpoint the `Krea2Control` provider loads, downloading on first
/// use. Order (most specific wins): the `SCENEWORKS_CONTROLNET_KREA` env (validation / bring-your-own) → an
/// `advanced.controlWeights.path` (a studio-trained or registered LOCAL overlay the API resolved,
/// B4/sc-10165) → an `advanced.controlWeights.{repo,filename}` hosted override / the default published
/// overlay repo (`SceneWorks/krea2-pose-controlnet-beta`, sc-8466), fetched into the app cache. The
/// ~6.6 GB overlay is lazy-fetched only on the first Krea pose job (vs bloating the base download),
/// mirroring `ensure_flux2_control_candle_weights`.
async fn ensure_krea_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    // 1. Env override — a local overlay `.safetensors` (validation against the spike, bring-your-own).
    if let Ok(p) = std::env::var(KREA_CONTROL_WEIGHTS_ENV) {
        let p = PathBuf::from(p.trim());
        if p.is_file() {
            return Ok(p);
        }
    }
    // 2. A LOCAL overlay path the API resolved from a studio-trained / registered overlay selection
    //    (B4/sc-10165 `resolve_control_overlay_selection` writes `advanced.controlWeights.path`),
    //    confined to an app-managed root (sc-11168 / F-006).
    if let Some(p) = krea_control_payload_overlay_path(settings, request)? {
        if p.is_file() {
            return Ok(p);
        }
    }
    // 3. A hosted overlay: a `controlWeights.{repo,filename}` override (a not-yet-cached registered/hosted
    //    overlay the API passed through) or the default published beta overlay — HF cache, else download.
    let (repo, file) = krea_control_overlay_repo_file(request)?;
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, &repo) {
        let f = snapshot.join(&file);
        if f.is_file() {
            return Ok(f);
        }
    }
    let client = crate::downloads::streaming_download_client();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "Krea 2 strict-pose generation canceled while fetching control overlay.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-krea")
        .join(&file);
    // Pin the exact commit for the default overlay repo so `main` moving under us can't swap the
    // checkpoint (sc-8466 / sc-9879). A `controlWeights.repo` override may carry its own layout, so only
    // pin when we're on the default repo.
    let revision = if repo == KREA_CONTROL_OVERLAY_REPO {
        KREA_CONTROL_OVERLAY_REVISION
    } else {
        "main"
    };
    ensure_hf_cached_file(&context, &repo, revision, &file, &dst).await?;
    Ok(dst)
}

/// Flat telemetry recorded on candle Krea control assets.
fn krea_control_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(KREA_CONTROL_ENGINE.to_owned()),
    );
    // User LoRA labels applied on top of the pose control branch (sc-11721) — mirrors the
    // `image_settings_metrics` `loras` field so the control-lane asset records what rode alongside the
    // pose lock. Omitted when no LoRA was requested (the sc-4408 omit-when-absent contract).
    let loras: Vec<Value> = request
        .loras
        .iter()
        .filter_map(lora_label)
        .map(Value::String)
        .collect();
    if !loras.is_empty() {
        raw.insert("loras".to_owned(), Value::Array(loras));
    }
    raw
}

/// The per-lane half of the candle Krea strict-control [`CandleStrictControl`] driver: the resolved base +
/// overlay paths + request numerics. Krea 2 Turbo is CFG-free (no guidance / negative pass) and bf16
/// (no quant tier). Moved onto the blocking thread, loaded once, drives every pose.
struct KreaStrictControl {
    base: PathBuf,
    control: PathBuf,
    /// User LoRA/LoKr adapters applied additively to the frozen base DiT (sc-11721) — a character/style
    /// adapter reshapes the subject while the control branch keeps the pose lock. Empty ⇒ stock control.
    adapters: Vec<AdapterSpec>,
    /// Control-branch quant the VRAM fit ladder selected (sc-11754, candle-gen #483). `None` (bf16) is the
    /// big-card default; `Some(Q8)`/`Some(Q4)` is the last-resort rung engaged only when the predicted
    /// peak wouldn't otherwise fit the (possibly emulated) card. Folds the ~6.6 GB dense branch onto the
    /// GPU packed (dequant-on-forward) so it never lands dense in VRAM.
    branch_quant: Option<gen_core::Quant>,
    /// Force the seam-free tiled VAE decode (sc-11744) — the fit ladder's cheapest rung, engaged only
    /// when the predicted decode-phase peak exceeds free VRAM. `false` (the big-card default) is the
    /// monolithic full-speed decode. A *speed* cost, no quality cost.
    tile_vae_decode: bool,
    /// Engage sc-6217-style query-row attention chunking on the composable base stack + control branch
    /// (sc-11745, candle-gen #496) — the fit ladder's rung between VAE-decode tiling and branch quant,
    /// engaged only when the predicted denoise-phase activation peak exceeds free VRAM. `false` (the
    /// big-card default) is the unchunked full-speed forward. A *speed* cost (~+6%), byte-identical output.
    chunk_attention: bool,
    prompt: String,
    width: u32,
    height: u32,
    steps: u32,
    control_scale: f32,
}

impl CandleStrictControl for KreaStrictControl {
    type Model = candle_gen_krea::Krea2Control;

    fn engine_id(&self) -> &'static str {
        KREA_CONTROL_ENGINE_ID
    }

    fn engine_label(&self) -> &'static str {
        KREA_CONTROL_ENGINE
    }

    fn stream_tag(&self) -> &'static str {
        "krea_control"
    }

    fn out_width(&self) -> u32 {
        self.width
    }

    fn out_height(&self) -> u32 {
        self.height
    }

    fn load(&self) -> WorkerResult<Self::Model> {
        let paths = candle_gen_krea::Krea2ControlPaths {
            root: self.base.clone(),
            control: self.control.clone(),
            adapters: self.adapters.clone(),
            // bf16 by default; the fit ladder (sc-11754) sets q8/q4 only to fit a constrained card.
            branch_quant: self.branch_quant,
            // Unchunked (full speed) by default; the fit ladder (sc-11745) forces query-row attention
            // chunking only to bound the denoise activation peak on a constrained card — byte-identical.
            chunk_attention: self.chunk_attention,
        };
        candle_gen_krea::Krea2Control::load(&paths).map_err(|error| {
            WorkerError::Engine(format!("Krea 2 strict-pose control load failed: {error}"))
        })
    }

    fn generate_one(
        &self,
        model: &Self::Model,
        control: &Image,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> WorkerResult<Image> {
        let req = candle_gen_krea::Krea2ControlRequest {
            prompt: self.prompt.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps as usize,
            control_scale: self.control_scale,
            seed,
            tile_vae_decode: self.tile_vae_decode,
            cancel: cancel.clone(),
        };
        model.generate(&req, control, on_progress).map_err(|error| {
            WorkerError::Engine(format!("Krea 2 strict-pose generation failed: {error}"))
        })
    }
}

/// Real candle Krea 2 strict-pose generation: one image per pose, each conditioned on a full DWPose
/// skeleton (`controlMode` unset ⇒ pose) via a trained control-branch overlay on the frozen Turbo base
/// (sc-8464; engine sc-8462). Resolves the dense base + the overlay, then hands a [`KreaStrictControl`] to
/// the shared [`run_candle_strict_control`] driver (validation against `krea_2_turbo_control`'s
/// `supported_kinds` = {Pose}, per-pose skeleton rendering, scoring). Krea is CFG-free bf16. The pose path
/// is byte-preserved; `control_scale = 0` is byte-identical to base.
async fn generate_candle_krea_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let base = resolve_krea_control_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("Krea 2 Turbo base (krea/Krea-2-Turbo) weights not found".to_owned())
    })?;
    let control = ensure_krea_control_weights(api, settings, job, request).await?;
    // User LoRA/LoKr adapters ride additively on the frozen base DiT (sc-11721 / candle-gen sc-11720):
    // resolved + path-confined by the shared helper (enforces MAX_JOB_LORAS + `normalize_app_managed_
    // lora_path`), then installed on the base at load — the pose control branch is never adapted.
    let adapters = resolve_adapters(request, settings)?;

    let steps = krea_control_candle_steps(request);
    let control_scale = advanced::f32_clamped(
        &request.advanced,
        "controlScale",
        KREA_CONTROL_DEFAULT_SCALE,
        0.0..=KREA_CONTROL_SCALE_CAP,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(KREA_CONTROL_BASE_REPO)
        .to_owned();

    let pose_count = pose_entries(request).len();
    let raw_settings =
        krea_control_candle_raw_settings(request, &repo, steps, control_scale, pose_count);

    // VRAM fit ladder (sc-11754, epic 8459 → epic 10765). The control lane is diverted around the base.rs
    // `generate_candle_stream` fit-gate, so it gets its own here: predict the control-lane peak (base tier
    // + the ~6.6 GB bf16 control branch + activations + the end-of-render VAE-decode spike) and compare it
    // against the live/capped free VRAM. On a big card the bf16 branch fits — nothing engages, zero speed/
    // quality penalty. On a constrained card (or one emulated via `SCENEWORKS_CUDA_VRAM_CAP_GB`) the ladder
    // engages the last-resort branch-quant rung (q8 near-lossless, then q4 pose-locked) until it fits, else
    // rejects-before-OOM. The cheaper rungs (VAE-decode tiling sc-11744, activation chunking / res-cap
    // sc-11745) slot in ahead of branch-quant here once their candle-gen mechanism lands. NB: this lane uses
    // the UNcached `start_gen_stream` (it doesn't evict the single-slot generator cache), so budget against
    // raw live free VRAM — the cache's pages are NOT reclaimable by this load, unlike the base.rs gate.
    let tier =
        crate::vram_gate::requested_tier_key(&request.advanced, &request.model_manifest_entry);
    let budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    let (tile_vae_decode, chunk_attention, branch_quant) = match crate::krea_control_fit::fit_ladder(
        crate::krea_control_fit::predicted_control_peak_gb(&request.model_manifest_entry, tier),
        budget,
        crate::krea_control_fit::decode_tile_save_gb(&request.model_manifest_entry, tier),
        crate::krea_control_fit::chunk_attn_save_gb(&request.model_manifest_entry),
        crate::krea_control_fit::branch_quant_save_gb(&request.model_manifest_entry, "q8"),
        crate::krea_control_fit::branch_quant_save_gb(&request.model_manifest_entry, "q4"),
    ) {
        // Big-card fast path (or no signal): monolithic full-speed decode, unchunked attention, bf16 branch.
        crate::krea_control_fit::KreaControlFit::Unknown
        | crate::krea_control_fit::KreaControlFit::Fits {
            tile_vae_decode: false,
            chunk_attention: false,
            branch_quant: None,
        } => (false, false, None),
        // Constrained card: the fit ladder engaged the cheapest sufficient set of rungs to fit — the
        // seam-free tiled VAE decode (sc-11744) and/or query-row attention chunking (sc-11745), both
        // speed-only, and/or the last-resort branch quant (sc-11743, a quality cost).
        crate::krea_control_fit::KreaControlFit::Fits {
            tile_vae_decode: tile,
            chunk_attention: chunk,
            branch_quant: quant,
        } => {
            tracing::info!(
                model = %request.model,
                tier,
                tile_vae_decode = tile,
                chunk_attention = chunk,
                branch_quant = ?quant,
                "Krea control VRAM fit ladder: predicted peak exceeds free VRAM — engaging rungs \
                 (VAE-decode tiling, attention chunking, and/or control-branch quant) to fit"
            );
            (tile, chunk, quant)
        }
        // Won't fit even at the last rung ⇒ reject before the reactive CUDA OOM.
        crate::krea_control_fit::KreaControlFit::TooBig {
            needed_gb,
            available_gb,
        } => {
            return Err(WorkerError::InvalidPayload(format!(
                "Krea 2 pose-ControlNet at the {tier} base tier needs ~{needed} GB of VRAM (with \
                 headroom, tiled VAE decode + attention chunking + control branch quantized to q4) but \
                 GPU {gpu} has ~{available} GB available. Lower the output resolution or run on a card \
                 with more VRAM.",
                needed = needed_gb.round() as i64,
                available = available_gb.round() as i64,
                gpu = settings.gpu_id,
            )));
        }
    };

    let provider = KreaStrictControl {
        base,
        control,
        adapters,
        branch_quant,
        tile_vae_decode,
        chunk_attention,
        prompt: request.prompt.clone(),
        width: request.width,
        height: request.height,
        steps,
        control_scale,
    };

    run_candle_strict_control(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        provider,
        raw_settings,
        asset_writes,
    )
    .await
}
