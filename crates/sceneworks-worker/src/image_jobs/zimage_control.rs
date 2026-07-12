// Candle (Windows/CUDA) Z-Image Fun-ControlNet (strict pose) route (sc-5489, epic 5480) —
// `z_image_turbo` + `advanced.poses` off-Mac via `candle_gen_z_image::ZImageControl`. The LAST family
// of the sc-5489 3-family ControlNet port (after Qwen + Kolors). The candle sibling of the MLX Z-Image
// strict-pose path (zimage.rs `generate_zimage_control_stream`): one image per pose, each conditioned
// on a full DWPose skeleton (rendered cross-platform by `openpose_skeleton::draw_wholebody`) fed to the
// VACE-style `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` branch.
//
// **Candle-only.** macOS keeps the MLX `z_image_turbo_control` registry generator (zimage.rs); the
// candle `ZImageControl` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle
// build (the `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs`
// module, so it shares that module's imports (`parse_poses`/`pose_entries`/`Settings`/`WorkerResult`/
// `huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).

/// Default Turbo Fun-Controlnet-Union weights — the **8-step** variant the MLX path uses (zimage.rs
/// `ZIMAGE_CONTROL_FILE`); the candle `ZImageControl::generate` runs the matching 8-step schedule.
const ZIMAGE_CTRL_REPO: &str = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1";
const ZIMAGE_CTRL_FILE: &str = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors";
/// Pinned revision for the default Turbo `ZIMAGE_CTRL_REPO` (sc-9879, F-077 follow-up). Fetching the
/// mutable `main` branch means a re-push (or a compromised token) could silently swap the ControlNet
/// checkpoint we load; pin the exact commit for defense-in-depth (mirrors sc-8879/sc-9682). Applied ONLY
/// to the default repo — a manifest `controlWeights.repo` override keeps `main`. HF's tree API still
/// reports the file's `lfs.oid`, which `ensure_hf_cached_file` verifies against.
const ZIMAGE_CTRL_REVISION: &str = "5155fc56d17821007d6f62ac192c09e0f0e72016";
/// The Z-Image-Turbo base diffusers repo when the manifest omits `repo`.
const ZIMAGE_CTRL_DEFAULT_REPO: &str = "Tongyi-MAI/Z-Image-Turbo";
/// Base (non-distilled, real-CFG) Z-Image Fun-Controlnet-Union weights (sc-8379) — the same VACE
/// Fun-Union control branch as the Turbo variant, assembled from a base `Tongyi-MAI/Z-Image` snapshot +
/// the base control checkpoint. The base + Turbo Fun-Union safetensors are byte-structurally identical
/// (same key layout), so the candle `ZImageControl` loader handles both; the diffusers checkpoint ships a
/// single `.safetensors`. Mirrors the MLX `z_image_control` engine repo (`STRICT_CONTROL_ENGINES`).
const ZIMAGE_CTRL_BASE_REPO: &str = "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1";
const ZIMAGE_CTRL_BASE_FILE: &str = "diffusion_pytorch_model.safetensors";
/// Pinned revision for the default base `ZIMAGE_CTRL_BASE_REPO` (sc-9879, F-077 follow-up). Same
/// defense-in-depth rationale as `ZIMAGE_CTRL_REVISION`; applied ONLY to the default base repo, an
/// override keeps `main`, and the `lfs.oid` sha256 verify is retained.
const ZIMAGE_CTRL_BASE_REVISION: &str = "755999a934909bd5832e20718bb7c639d2a63eb9";
/// The base Z-Image diffusers repo when the manifest omits `repo` (sc-8379).
const ZIMAGE_CTRL_BASE_DEFAULT_REPO: &str = "Tongyi-MAI/Z-Image";
/// ControlNet conditioning-scale default (the strict-pose tier).
const ZIMAGE_CTRL_DEFAULT_SCALE: f32 = 1.0;
/// Base-mode (sc-8680) classifier-free guidance default — the undistilled base `z_image` runs real CFG;
/// the card recommends 3.0–5.0 (default 4.0), matching the `z_image` manifest `defaults.guidanceScale`
/// and `candle_gen_z_image`'s `BASE_DEFAULT_GUIDANCE`. Inert on Turbo (guidance-distilled, CFG-free).
const ZIMAGE_CTRL_BASE_DEFAULT_GUIDANCE: f32 = 4.0;
/// Denoise-steps default — the 8-step Turbo Fun-ControlNet variant (vs the 4-step distilled txt2img).
const ZIMAGE_CTRL_DEFAULT_STEPS: u32 = 8;
/// Denoise-steps default for the base (non-distilled) model — the undistilled foundation runs the full
/// ~50-step schedule (the base manifest `defaults.steps`). The candle `ZImageControl` engine takes a
/// `steps` count directly (it is guidance-distilled in shape but the schedule length is request-driven),
/// so the worker just feeds the higher default; an `advanced.steps` override still wins.
const ZIMAGE_CTRL_BASE_DEFAULT_STEPS: u32 = 50;
/// The adapter/engine id recorded on candle Z-Image control assets (distinct from the txt2img
/// `candle_z_image`/`candle_zimage` lane).
const ZIMAGE_CTRL_ENGINE: &str = "candle_zimage_control";
/// The [`STRICT_CONTROL_ENGINES`] catalog id the Turbo candle lane validates `advanced.controlMode`
/// against (the Turbo Fun-Controlnet-Union row — `{Pose, Canny, Depth}`). Mirrors the MLX
/// `z_image_turbo_control` registry engine's `supported_kinds` (sc-8304).
const ZIMAGE_CTRL_ENGINE_ID: &str = "z_image_turbo_control";
/// The [`STRICT_CONTROL_ENGINES`] catalog id the BASE candle lane validates against (the base Z-Image
/// Fun-Controlnet-Union row — `{Pose, Canny, Depth}`, sc-8379). Mirrors the MLX `z_image_control` engine.
const ZIMAGE_CTRL_BASE_ENGINE_ID: &str = "z_image_control";

/// Model ids the candle Z-Image ControlNet route accepts: the distilled `z_image_turbo` (8-step) and the
/// base `z_image` (full-CFG, ~50-step; sc-8379). Both ride the same candle `ZImageControl` engine — the
/// base + Turbo Fun-Union safetensors are byte-structurally identical — differing only in the
/// base/control repos + step count.
fn is_zimage_control_model(model: &str) -> bool {
    matches!(model, "z_image_turbo" | "z_image")
}

/// True when this is the base (non-distilled) Z-Image control model (sc-8379) — selects the base repos,
/// the base step default, and the `z_image_control` engine-id validation row.
fn is_zimage_base_model(model: &str) -> bool {
    model == "z_image"
}

/// The default base diffusers repo for this control job's model — Turbo (`Tongyi-MAI/Z-Image-Turbo`) or
/// the base undistilled `Tongyi-MAI/Z-Image` (sc-8379), selected by the request model id.
fn zimage_control_base_default_repo(model: &str) -> &'static str {
    if is_zimage_base_model(model) {
        ZIMAGE_CTRL_BASE_DEFAULT_REPO
    } else {
        ZIMAGE_CTRL_DEFAULT_REPO
    }
}

/// Resolve the Z-Image base (diffusers) snapshot: an explicit `modelPath` (advanced or manifest) → the
/// HF cache snapshot for the manifest `repo` (default `Tongyi-MAI/Z-Image-Turbo`, or `Tongyi-MAI/Z-Image`
/// for the base model, sc-8379). `None` ⇒ not present locally (the job is not candle-runnable, falls
/// through to torch). Mirrors `resolve_kolors_control_base`.
fn resolve_zimage_control_base(
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
        return resolve_app_managed_model_dir(settings, &path, "Z-Image control modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| zimage_control_base_default_repo(&request.model));
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible Z-Image strict-control job: `z_image_turbo` or the base `z_image`
/// (sc-8379) with a non-empty `advanced.poses`, not edit mode, whose base resolves locally. Mirrors
/// `jobs_store::zimage_control_candle_eligible` so the worker and router agree.
fn zimage_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_zimage_control_model(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_zimage_control_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → model default (Turbo 8,
/// base 50; sc-8379). The base undistilled model runs the longer schedule for its real-CFG quality.
fn zimage_control_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32_with(
        request,
        "steps",
        || {
            if is_zimage_base_model(&request.model) {
                ZIMAGE_CTRL_BASE_DEFAULT_STEPS
            } else {
                ZIMAGE_CTRL_DEFAULT_STEPS
            }
        },
        1..=50,
    )
}

/// Resolve the base-mode (sc-8680) classifier-free guidance scale: `advanced.guidanceScale` → manifest
/// `guidanceScale` → the base default (4.0), clamped to a sane CFG range. Consumed only by the base
/// `z_image` control path (real CFG); Turbo ignores it (guidance-distilled). Mirrors
/// `kolors_control_guidance`.
fn zimage_control_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        ZIMAGE_CTRL_BASE_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// The (repo, filename) of the ControlNet weights — `advanced.controlWeights.{repo,filename}` overrides,
/// else the model's Fun-Controlnet-Union default: the Turbo 8-step variant, or the base
/// (full-CFG) variant for the base `z_image` model (sc-8379). Parity with the MLX `resolve_control_weights`.
/// The payload filename must be a plain component (sc-8821 / F-019).
fn zimage_control_repo_file(request: &ImageRequest) -> WorkerResult<(String, String)> {
    let (default_repo, default_file) = if is_zimage_base_model(&request.model) {
        (ZIMAGE_CTRL_BASE_REPO, ZIMAGE_CTRL_BASE_FILE)
    } else {
        (ZIMAGE_CTRL_REPO, ZIMAGE_CTRL_FILE)
    };
    let cw = request.advanced.get("controlWeights").and_then(Value::as_object);
    let pick = |key: &str, default: &str| {
        cw.and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    Ok((
        pick("repo", default_repo),
        safe_weight_filename(
            &pick("filename", default_file),
            "advanced.controlWeights.filename",
        )?,
    ))
}

/// Resolve the Fun-Controlnet-Union weight **file** the `ZImageControl` provider loads, downloading on
/// first use. Order: an env-pinned file (`SCENEWORKS_CONTROLNET_ZIMAGE`) → a whole-repo HF cache
/// snapshot → download into the app cache. Mirrors `ensure_kolors_control_weights`.
async fn ensure_zimage_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file) = zimage_control_repo_file(request)?;
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_ZIMAGE") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
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
        cancel_message: "Z-Image strict-pose generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-zimage")
        .join(&file);
    // Pin the exact commit for whichever default control repo (Turbo or base) we resolved so `main`
    // moving under us can't swap the ControlNet checkpoint (sc-9879). A manifest `controlWeights.repo`
    // override may carry its own revision layout, so only pin when we're on a default repo.
    let revision = if repo == ZIMAGE_CTRL_REPO {
        ZIMAGE_CTRL_REVISION
    } else if repo == ZIMAGE_CTRL_BASE_REPO {
        ZIMAGE_CTRL_BASE_REVISION
    } else {
        "main"
    };
    ensure_hf_cached_file(&context, &repo, revision, &file, &dst).await?;
    Ok(dst)
}

/// Flat telemetry recorded on candle Z-Image control assets. `guidance` is recorded only for the base
/// `z_image` model (sc-8680, real CFG); Turbo is guidance-distilled (single cond forward), so it omits it.
fn zimage_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    control_scale: f32,
    pose_count: usize,
    is_base: bool,
    guidance: f32,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    if is_base {
        raw.insert("guidanceScale".to_owned(), json!(guidance));
    }
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(ZIMAGE_CTRL_ENGINE.to_owned()),
    );
    raw
}

/// The per-lane half of the candle Z-Image strict-control [`CandleStrictControl`] driver (sc-8304): the
/// resolved weight paths + the request numerics. Moved onto the blocking thread, loaded once, drives every
/// pose. Turbo is distilled (CFG-free, no negative prompt); the base model (sc-8680) runs the FAITHFUL
/// undistilled treatment — shift-6.0, ~50-step, and real classifier-free guidance — so `is_base` selects
/// the base control path in `ZImageControl` and threads `guidance` + `negative_prompt` (both inert on Turbo).
struct ZImageStrictControl {
    snapshot: PathBuf,
    controlnet: PathBuf,
    prompt: String,
    width: u32,
    height: u32,
    steps: u32,
    control_scale: f32,
    /// `true` for the base `z_image` model (sc-8680) — routes `ZImageControl::generate` to the faithful
    /// base path (shift-6.0 / ~50-step / real CFG); `false` for `z_image_turbo` (the distilled 8-step
    /// CFG-free path, byte-unchanged). Sets `ZImageControlPaths.base`.
    is_base: bool,
    /// Base-mode classifier-free guidance scale (sc-8680); ignored on Turbo. Threaded to
    /// `ZImageControlRequest.guidance`.
    guidance: f32,
    /// Base-mode negative prompt for the uncond CFG branch (sc-8680); ignored on Turbo. Threaded to
    /// `ZImageControlRequest.negative_prompt`.
    negative_prompt: String,
    /// The [`STRICT_CONTROL_ENGINES`] catalog id for this job's model — `z_image_turbo_control` (Turbo) or
    /// `z_image_control` (base, sc-8379) — the `advanced.controlMode` validation key.
    engine_id: &'static str,
    /// Per-generation PiD decoder weights (epic 7840, sc-8044): `Some` only when opted in (`advanced.usePid`)
    /// AND the PiD + Gemma snapshots are cached (Z-Image is the FLUX.1 latent space → `zimage-turbo` alias).
    /// Threaded into `with_pid` at load; `use_pid` on the request is `is_some()`. `None` ⇒ native VAE decode.
    pid: Option<gen_core::PidWeights>,
}

impl CandleStrictControl for ZImageStrictControl {
    type Model = ZImageControl;

    fn engine_id(&self) -> &'static str {
        self.engine_id
    }

    fn engine_label(&self) -> &'static str {
        ZIMAGE_CTRL_ENGINE
    }

    fn stream_tag(&self) -> &'static str {
        "zimage_control"
    }

    fn out_width(&self) -> u32 {
        self.width
    }

    fn out_height(&self) -> u32 {
        self.height
    }

    fn load(&self) -> WorkerResult<Self::Model> {
        let paths = ZImageControlPaths {
            snapshot: self.snapshot.clone(),
            control: self.controlnet.clone(),
            // Base `z_image` (sc-8680) → the faithful undistilled control path (shift-6.0, ~50-step,
            // real CFG); `z_image_turbo` → the distilled Turbo path (byte-unchanged).
            base: self.is_base,
        };
        let model = ZImageControl::load(&paths).map_err(|error| {
            WorkerError::Engine(format!("Z-Image strict-pose control load failed: {error}"))
        })?;
        // Attach the optional PiD decoder (sc-8044): `Some` only when opted in AND the snapshots are cached.
        match &self.pid {
            Some(pid) => model.with_pid(pid).map_err(|error| {
                WorkerError::Engine(format!("Z-Image control PiD decoder load failed: {error}"))
            }),
            None => Ok(model),
        }
    }

    fn generate_one(
        &self,
        model: &Self::Model,
        control: &Image,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> WorkerResult<Image> {
        // `guidance` + `negative_prompt` drive the base-mode real-CFG denoise; the distilled Turbo path
        // ignores both (single cond forward), so a `None`/value is inert there. We always forward the
        // resolved base values — they only take effect when `base = true` in the loaded `ZImageControl`.
        let req = ZImageControlRequest {
            prompt: self.prompt.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps as usize,
            control_scale: self.control_scale,
            guidance: self.is_base.then_some(self.guidance),
            negative_prompt: self
                .is_base
                .then(|| self.negative_prompt.clone())
                .filter(|value| !value.trim().is_empty()),
            seed,
            // PiD opt-in (sc-8044): in lockstep with the `with_pid` load — `is_some()` ⇒ decoder loaded.
            use_pid: self.pid.is_some(),
            cancel: cancel.clone(),
        };
        model.generate(&req, control, on_progress).map_err(|error| {
            WorkerError::Engine(format!("Z-Image strict-pose generation failed: {error}"))
        })
    }
}

/// Real candle Z-Image strict-control generation: one image per pose, each conditioned on a full DWPose
/// skeleton (`controlMode` unset) or a canny/depth control map. Serves both `z_image_turbo` (8-step
/// distilled) and the base `z_image` (full-CFG, ~50-step; sc-8379) — same candle `ZImageControl` engine,
/// model-selected base/control repos + step default + engine-id validation row. Resolves the base +
/// control weights, then hands a [`ZImageStrictControl`] to the shared [`run_candle_strict_control`]
/// driver, which validates the requested kind against the model's `supported_kinds`, preprocesses each
/// pose's control map, runs the per-pose loop, and scores against any identity reference. The pose path is
/// byte-preserved.
async fn generate_candle_zimage_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let is_base = is_zimage_base_model(&request.model);
    let base = resolve_zimage_control_base(request, settings)?.ok_or_else(|| {
        let label = if is_base { "Z-Image" } else { "Z-Image-Turbo" };
        WorkerError::InvalidPayload(format!("Z-Image base ({label}) weights not found"))
    })?;
    let controlnet = ensure_zimage_control_weights(api, settings, job, request).await?;

    let steps = zimage_control_steps(request);
    let control_scale = advanced::f32_clamped(
        &request.advanced,
        "controlScale",
        ZIMAGE_CTRL_DEFAULT_SCALE,
        0.0..=2.0,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| zimage_control_base_default_repo(&request.model))
        .to_owned();
    let engine_id = if is_base {
        ZIMAGE_CTRL_BASE_ENGINE_ID
    } else {
        ZIMAGE_CTRL_ENGINE_ID
    };
    // Base-mode real-CFG surface (sc-8680): resolve the guidance scale + negative prompt for the
    // undistilled base `z_image` control path. Inert on Turbo (the distilled path ignores both), so we
    // resolve them unconditionally and gate on `is_base` at request-build time.
    let guidance = zimage_control_guidance(request);
    let negative_prompt = request.negative_prompt.clone();

    let pose_count = pose_entries(request).len();
    // Per-generation PiD decode (epic 7840, sc-8044): resolve the `zimage-turbo`/`flux` PiD student + Gemma
    // when `advanced.usePid` is set and the snapshots are cached; else `None` → native Z-Image VAE.
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();
    // PiD output tier (sc-10054): 2K caps the effective base so PiD's fixed 4× lands on ~2048 (default
    // 4K/native leaves the requested dims untouched). The shared driver renders the control map at these
    // same dims (via `out_width`/`out_height`), keeping control + latent aligned.
    let (width, height) =
        pid_effective_dims(request.width, request.height, use_pid, pid_output_tier(request));
    let mut raw_settings = zimage_control_raw_settings(
        request,
        &repo,
        steps,
        control_scale,
        pose_count,
        is_base,
        guidance,
    );
    // Mark PiD output on the sidecar (NSCLv1 NC flows to PiD output); record whether PiD actually ran.
    raw_settings.insert("usePid".to_owned(), Value::Bool(use_pid));

    let provider = ZImageStrictControl {
        snapshot: base,
        controlnet,
        prompt: request.prompt.clone(),
        width,
        height,
        steps,
        control_scale,
        is_base,
        guidance,
        negative_prompt,
        engine_id,
        pid: pid_weights,
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
