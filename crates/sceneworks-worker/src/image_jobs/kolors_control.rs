// Candle (Windows/CUDA) Kolors ControlNet (strict pose) route (sc-5489, epic 5480; sc-8823 driver route)
// — `kolors` + `advanced.poses` off-Mac via `runtime_cuda::providers::kolors::KolorsControl`. The Kolors sibling of the
// candle Qwen strict-control path (qwen_control.rs): one image per pose, each conditioned on a full DWPose
// skeleton (rendered cross-platform by `openpose_skeleton::draw_wholebody`) fed to the
// `Kwai-Kolors/Kolors-ControlNet-Pose` branch, whose residuals are added into the vendored SDXL UNet the
// Kolors IP-Adapter lane (kolors_ipadapter.rs) already stands on.
//
// **Pose-only, routed through the shared driver (sc-8823).** Like the qwen/zimage/flux2/flux1 candle
// lanes, this lane implements [`CandleStrictControl`] and hands off to [`run_candle_strict_control`], so
// it inherits the SAME `advanced.controlMode` validation against [`STRICT_CONTROL_ENGINES`]. The Kolors
// `kolors_control` row lists `{Pose}` ONLY (the Kolors-ControlNet-Pose branch is a DWPose-skeleton
// ControlNet, not an input-agnostic Fun-Union VACE engine), so a `controlMode = canny | depth` job is
// REJECTED with a clear error instead of silently rendering a pose skeleton (the old bespoke scaffold
// never validated the mode). The pose path (no `controlMode`) is byte-preserved.
//
// **Candle-only.** macOS keeps the MLX Kolors ControlNet path; the candle `KolorsControl` is a bespoke
// provider, so this whole file is gated to the Windows/CUDA candle build (the `include!` in image_jobs.rs
// carries the cfg). It is `include!`d into the `image_jobs` module, so it shares that module's imports
// (`parse_poses`/`pose_entries`/`Settings`/`WorkerResult`/`huggingface_snapshot_dir`/
// `ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).

/// Default Kolors ControlNet (pose) weights — same repo/file the MLX path uses. The repo ships the
/// `.safetensors` on `main` (no `refs/pr` dance, unlike the Kolors IP-Adapter).
const KOLORS_CONTROL_REPO: &str = "Kwai-Kolors/Kolors-ControlNet-Pose";
const KOLORS_CONTROL_FILE: &str = "diffusion_pytorch_model.safetensors";
/// Pinned revision for the default `KOLORS_CONTROL_REPO` (sc-9879, F-077 follow-up). Fetching the
/// mutable `main` branch means a re-push (or a compromised token) could silently swap the ControlNet
/// checkpoint we load; pin the exact commit for defense-in-depth (mirrors sc-8879/sc-9682). Applied ONLY
/// to the default repo — a manifest `controlWeights.repo` override carries its own revision layout, so it
/// keeps `main`. HF's tree API still reports the file's `lfs.oid`, which `ensure_hf_cached_file` verifies.
const KOLORS_CONTROL_REVISION: &str = "83e35a8033a89d2e75044b412d0e2474111578f7";
/// The Kolors base diffusers repo when the manifest omits `repo`.
const KOLORS_CONTROL_DEFAULT_REPO: &str = "Kwai-Kolors/Kolors-diffusers";
/// ControlNet conditioning-scale default (the strict-pose tier).
const KOLORS_CONTROL_DEFAULT_SCALE: f32 = 1.0;
/// Denoise-steps default (Kolors production — diffusers `KolorsPipeline`).
const KOLORS_CONTROL_DEFAULT_STEPS: u32 = 50;
/// CFG default (Kolors production guidance).
const KOLORS_CONTROL_DEFAULT_GUIDANCE: f32 = 5.0;
/// The adapter/engine id recorded on candle Kolors control assets (distinct from the txt2img
/// `candle_kolors` lane and the `candle_kolors_ipadapter` reference lane).
const KOLORS_CONTROL_ENGINE: &str = "candle_kolors_control";

/// Model ids the candle Kolors ControlNet route accepts.
fn is_kolors_control_model(model: &str) -> bool {
    model == "kolors"
}

/// Resolve the Kolors base (diffusers) snapshot: an explicit `modelPath` (advanced or manifest) → the HF
/// cache snapshot for the manifest `repo` (default `Kwai-Kolors/Kolors-diffusers`). `None` ⇒ not present
/// locally (the job is not candle-runnable, falls through to torch). Mirrors `resolve_kolors_ipadapter_base`.
fn resolve_kolors_control_base(
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
        return resolve_app_managed_model_dir(settings, &path, "Kolors control modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(KOLORS_CONTROL_DEFAULT_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible Kolors strict-pose job: `kolors` with a non-empty
/// `advanced.poses`, not edit mode, whose base resolves locally. Mirrors
/// `jobs_store::kolors_control_candle_eligible` so the worker and router agree.
fn kolors_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_kolors_control_model(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_kolors_control_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=80) → manifest `steps` → default (50).
fn kolors_control_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", KOLORS_CONTROL_DEFAULT_STEPS, 1..=80)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (5.0), clamped.
fn kolors_control_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        KOLORS_CONTROL_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// The (repo, filename) of the ControlNet weights — `advanced.controlWeights.{repo,filename}` overrides,
/// else the Kolors-ControlNet-Pose default (parity with the MLX `resolve_control_weights_for`).
/// The payload filename must be a plain component (sc-8821 / F-019).
fn kolors_control_repo_file(request: &ImageRequest) -> WorkerResult<(String, String)> {
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
        pick("repo", KOLORS_CONTROL_REPO),
        safe_weight_filename(
            &pick("filename", KOLORS_CONTROL_FILE),
            "advanced.controlWeights.filename",
        )?,
    ))
}

/// Resolve the Kolors ControlNet weight **file** the `KolorsControl` provider loads, downloading on first
/// use. Order: an env-pinned file (`SCENEWORKS_CONTROLNET_KOLORS`) → a whole-repo HF cache snapshot →
/// download into the app cache. Mirrors `ensure_qwen_control_weights`.
async fn ensure_kolors_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file) = kolors_control_repo_file(request)?;
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_KOLORS") {
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
        cancel_message: "Kolors strict-pose generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-kolors")
        .join(&file);
    // Pin the exact commit for the default control repo so `main` moving under us can't swap the
    // ControlNet checkpoint (sc-9879). A manifest `controlWeights.repo` override may carry its own
    // revision layout, so only pin when we're on the default repo.
    let revision = if repo == KOLORS_CONTROL_REPO {
        KOLORS_CONTROL_REVISION
    } else {
        "main"
    };
    ensure_hf_cached_file(&context, &repo, revision, &file, &dst).await?;
    Ok(dst)
}

/// Flat telemetry recorded on candle Kolors control assets.
fn kolors_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(KOLORS_CONTROL_ENGINE.to_owned()),
    );
    raw
}

/// The [`STRICT_CONTROL_ENGINES`] catalog id this candle lane validates `advanced.controlMode` against —
/// the `kolors_control` row, whose `supported_kinds` is `{Pose}` ONLY (sc-8823). The Kolors lane rides the
/// `Kwai-Kolors/Kolors-ControlNet-Pose` DWPose-skeleton branch (NOT a Fun-Union input-agnostic VACE
/// engine), so a `controlMode = canny | depth` request is REJECTED by the shared driver rather than
/// silently rendered as a pose skeleton.
const KOLORS_CONTROL_ENGINE_ID: &str = "kolors_control";

/// The per-lane half of the candle Kolors strict-control [`CandleStrictControl`] driver (sc-8304 /
/// sc-8823): the resolved base + Kolors-ControlNet-Pose weight paths + the request numerics (Kolors runs
/// true CFG, so it carries a negative prompt + guidance, plus the curated unified-sampler selection). It
/// is moved onto the blocking thread, loaded once, and drives every pose. `KolorsControl::generate` takes
/// `&self`, so `generate_one` borrows the model immutably.
struct KolorsStrictControl {
    kolors_base: PathBuf,
    controlnet: PathBuf,
    prompt: String,
    negative: String,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    control_scale: f32,
    /// Curated unified-sampler selection (epic 7114). Already normalized against the curated menu in the
    /// async preamble; `None` keeps the native leading-Euler default byte-exact.
    sampler: Option<String>,
    scheduler: Option<String>,
    /// Per-generation PiD decoder weights (epic 7840, sc-8044): `Some` only when opted in (`advanced.usePid`)
    /// AND the `sdxl` PiD + Gemma snapshots are cached (Kolors composes the SDXL VAE). Threaded into
    /// `with_pid` at load; `use_pid` on the request is `is_some()`. `None` ⇒ native SDXL VAE decode.
    pid: Option<gen_core::PidWeights>,
}

impl CandleStrictControl for KolorsStrictControl {
    type Model = KolorsControl;

    fn engine_id(&self) -> &'static str {
        KOLORS_CONTROL_ENGINE_ID
    }

    fn engine_label(&self) -> &'static str {
        KOLORS_CONTROL_ENGINE
    }

    fn stream_tag(&self) -> &'static str {
        "kolors_control"
    }

    fn out_width(&self) -> u32 {
        self.width
    }

    fn out_height(&self) -> u32 {
        self.height
    }

    fn load(&self) -> WorkerResult<Self::Model> {
        let paths = KolorsControlPaths {
            kolors_base: self.kolors_base.clone(),
            controlnet: self.controlnet.clone(),
        };
        let model = KolorsControl::load(&paths).map_err(|error| {
            WorkerError::Engine(format!("Kolors strict-pose control load failed: {error}"))
        })?;
        // Attach the optional PiD decoder (sc-8044): `Some` only when opted in AND the snapshots are cached.
        match &self.pid {
            Some(pid) => model.with_pid(pid).map_err(|error| {
                WorkerError::Engine(format!("Kolors control PiD decoder load failed: {error}"))
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
        let req = KolorsControlRequest {
            prompt: self.prompt.clone(),
            negative: self.negative.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps as usize,
            guidance: self.guidance,
            control_scale: self.control_scale,
            seed,
            sampler: self.sampler.clone(),
            scheduler: self.scheduler.clone(),
            // PiD opt-in (sc-8044): in lockstep with the `with_pid` load — `is_some()` ⇒ decoder loaded.
            use_pid: self.pid.is_some(),
            cancel: cancel.clone(),
        };
        model.generate(&req, control, on_progress).map_err(|error| {
            WorkerError::Engine(format!("Kolors strict-pose generation failed: {error}"))
        })
    }
}

/// Real candle Kolors strict-pose generation: one image per pose, each conditioned on a full DWPose
/// skeleton. Resolves the Kolors base + Kolors-ControlNet-Pose weights, then hands a [`KolorsStrictControl`]
/// to the shared [`run_candle_strict_control`] driver (validation against the `kolors_control` pose-only
/// `supported_kinds`, per-pose preprocessing, identity-likeness scoring, streaming). A `controlMode =
/// canny | depth` request is REJECTED by the driver (Kolors is pose-only) instead of silently rendering a
/// pose skeleton (sc-8823). The pose path is byte-preserved.
async fn generate_candle_kolors_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let kolors_base = resolve_kolors_control_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("Kolors base (Kolors-diffusers) weights not found".to_owned())
    })?;
    let controlnet = ensure_kolors_control_weights(api, settings, job, request).await?;

    let steps = kolors_control_steps(request);
    let guidance = kolors_control_guidance(request);
    let control_scale = advanced::f32_clamped(
        &request.advanced,
        "controlScale",
        KOLORS_CONTROL_DEFAULT_SCALE,
        0.0..=2.0,
    );
    // Curated unified-sampler selection (epic 7114, sc-7432): the candle `KolorsControl` provider honors
    // a curated solver/scheduler via the shared `denoise_curated` primitive (#130). Read + N3-normalize
    // against the shared curated menu (an unknown name drops to the engine default + emits an event).
    // N1: unset ⇒ `None` ⇒ the native leading-Euler default loop runs byte-exact.
    let (curated_samplers, curated_schedulers) = curated_image_menu();
    let (sampler, scheduler, _shift) = read_advanced_sampling_knobs(&request.advanced);
    let sampler = normalize_sampling_knob(
        sampler,
        &curated_samplers,
        "sampler",
        &request.model,
        &job.id,
        backend,
    );
    let scheduler = normalize_sampling_knob(
        scheduler,
        &curated_schedulers,
        "scheduler",
        &request.model,
        &job.id,
        backend,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(KOLORS_CONTROL_DEFAULT_REPO)
        .to_owned();

    let pose_count = pose_entries(request).len();
    // Per-generation PiD decode (epic 7840, sc-8044): resolve the `sdxl` PiD student + Gemma when
    // `advanced.usePid` is set and the snapshots are cached; else `None` → native SDXL VAE.
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();
    // PiD output tier (sc-10054): 2K caps the effective base so PiD's fixed 4× lands on ~2048 (default
    // 4K/native leaves the requested dims untouched). The shared driver renders the control map at these
    // same dims (via `out_width`/`out_height`), keeping control + latent aligned.
    let (width, height) =
        pid_effective_dims(request.width, request.height, use_pid, pid_output_tier(request));
    let mut raw_settings =
        kolors_control_raw_settings(request, &repo, steps, guidance, control_scale, pose_count);
    // Mark PiD output on the sidecar (NSCLv1 NC flows to PiD output); record whether PiD actually ran.
    raw_settings.insert("usePid".to_owned(), Value::Bool(use_pid));

    let provider = KolorsStrictControl {
        kolors_base,
        controlnet,
        prompt: request.prompt.clone(),
        negative: request.negative_prompt.clone(),
        width,
        height,
        steps,
        guidance,
        control_scale,
        sampler,
        scheduler,
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
