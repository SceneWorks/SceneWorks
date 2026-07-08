// Candle (Windows/CUDA) FLUX.1-dev strict-control Fun-Controlnet-Union route (sc-8412, epic 8236) —
// `flux_dev` + `advanced.poses` off-Mac via `candle_gen_flux::Flux1DevControl`. The candle sibling of the
// MLX FLUX.1-dev strict-control path (flux1_control.rs `generate_flux1_dev_control_stream`, sc-8244 /
// engine sc-8239): one image per library pose (or, with `advanced.controlMode = canny|depth` + a source,
// an auto-derived canny / Depth-Anything-V2 map), each fed to the Shakker
// `FLUX.1-dev-ControlNet-Union-Pro-2.0` residual-emitter branch overlaid on the FLUX.1-dev base. True
// structural lock, not the best-effort reference tier.
//
// **Candle-only.** macOS keeps the MLX `flux1_dev_control` registry generator (flux1_control.rs); the
// candle `Flux1DevControl` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle
// build (the `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs`
// module, so it shares that module's imports (`parse_poses`/`pose_entries`/`Settings`/`WorkerResult`/
// `huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).
//
// The FLUX.1-dev base is HF-gated; the Shakker Union-Pro-2.0 control overlay is ungated. The candle
// `Flux1DevControl` provider loads BOTH dense bf16 (no on-the-fly quant seam — the 12B dev fits dense at
// bf16), so the worker passes no `Quant`. dev is guidance-distilled — a single embedded-guidance forward,
// no true-CFG / negative pass. Union-Pro-2.0 dropped the discrete mode index, so the control kind is
// input-agnostic: it only gates the accepted set (pose/canny/depth); it does not branch the forward.

/// Default Shakker Union-Pro-2.0 control-weights repo + filename — the single diffusers `.safetensors`
/// shipped in the repo. Parity with the MLX `flux1_control.rs` defaults (which reads the repo from the
/// shared `STRICT_CONTROL_ENGINES` table); the candle lane keeps its own constant (its own default-repo,
/// like the other candle control lanes).
const FLUX1_CONTROL_CANDLE_REPO: &str = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0";
const FLUX1_CONTROL_CANDLE_FILE: &str = "diffusion_pytorch_model.safetensors";
/// Pinned revision for the default `FLUX1_CONTROL_CANDLE_REPO` (sc-9879, F-077 follow-up). Fetching the
/// mutable `main` branch means a re-push (or a compromised token) could silently swap the ControlNet
/// checkpoint we load; pin the exact commit for defense-in-depth (mirrors sc-8879/sc-9682). Applied ONLY
/// to the default repo — a manifest `controlWeights.repo` override keeps `main`. HF's tree API still
/// reports the file's `lfs.oid`, which `ensure_hf_cached_file` verifies against.
const FLUX1_CONTROL_CANDLE_REVISION: &str = "5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5";
/// The FLUX.1-dev base diffusers repo when the manifest omits `repo` (HF-gated). The candle lane loads
/// the dense bf16 snapshot.
const FLUX1_CONTROL_CANDLE_BASE_REPO: &str = "black-forest-labs/FLUX.1-dev";
/// Control-conditioning-scale default — the Shakker Union-Pro-2.0 README sweet spot ≈ 0.7 (the engine
/// `DEFAULT_CONTROL_SCALE` too, and the MLX lane's default). Clamp [0, 2].
const FLUX1_CONTROL_CANDLE_DEFAULT_SCALE: f32 = 0.7;
/// Denoise-steps default — the guidance-distilled dev (~25 steps; the engine request default).
const FLUX1_CONTROL_CANDLE_DEFAULT_STEPS: u32 = 25;
/// Embedded-guidance default — distilled dev scalar (NOT true-CFG, no negative pass).
const FLUX1_CONTROL_CANDLE_DEFAULT_GUIDANCE: f32 = 3.5;
/// The adapter/engine id recorded on candle FLUX.1-dev control assets (distinct from the txt2img
/// `candle_flux` + FLUX.2 `candle_flux2_control` lanes).
const FLUX1_CONTROL_CANDLE_ENGINE: &str = "candle_flux1_control";
/// The [`STRICT_CONTROL_ENGINES`] catalog id this candle lane validates `advanced.controlMode` against
/// (the FLUX.1-dev Union-Pro-2.0 row — `{Pose, Canny, Depth}`). Mirrors the MLX `flux1_dev_control`
/// registry engine's `supported_kinds` (sc-8304).
const FLUX1_CONTROL_CANDLE_ENGINE_ID: &str = "flux1_dev_control";

/// Model ids the candle FLUX.1 strict-control route accepts (schnell has no control checkpoint).
fn is_flux1_control_model(model: &str) -> bool {
    model == "flux_dev"
}

/// Resolve the FLUX.1-dev base (diffusers) snapshot: an explicit `modelPath` (advanced or manifest) → the
/// HF cache snapshot for the manifest `repo` (default `black-forest-labs/FLUX.1-dev`). `None` ⇒ not
/// present locally (the job is not candle-runnable). Mirrors `resolve_flux2_control_base`.
fn resolve_flux1_control_base(
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
        return resolve_app_managed_model_dir(settings, &path, "FLUX.1 control modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(FLUX1_CONTROL_CANDLE_BASE_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible FLUX.1-dev strict-control job: `flux_dev` with a non-empty
/// `advanced.poses`, not edit mode, whose base resolves locally. Mirrors
/// `jobs_store::flux1_control_candle_eligible` so the worker and router agree. Control-weights presence is
/// NOT part of the gate: they are fetched on first use in the stream.
fn flux1_control_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_flux1_control_model(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_flux1_control_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → default (25).
fn flux1_control_candle_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", FLUX1_CONTROL_CANDLE_DEFAULT_STEPS, 1..=50)
}

/// Resolve embedded guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (3.5),
/// clamped. dev rides this scalar on the transformer's guidance embedder (no true-CFG).
fn flux1_control_candle_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        FLUX1_CONTROL_CANDLE_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// The (repo, filename) of the control weights — `advanced.controlWeights.{repo,filename}` overrides,
/// else the Shakker Union-Pro-2.0 default (parity with the MLX `flux1_control_repo_file`).
/// The payload filename must be a plain component (sc-8821 / F-019).
fn flux1_control_candle_repo_file(request: &ImageRequest) -> WorkerResult<(String, String)> {
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
        pick("repo", FLUX1_CONTROL_CANDLE_REPO),
        safe_weight_filename(
            &pick("filename", FLUX1_CONTROL_CANDLE_FILE),
            "advanced.controlWeights.filename",
        )?,
    ))
}

/// Resolve the Shakker Union-Pro-2.0 weight **file** the `Flux1DevControl` provider loads, downloading on
/// first use. Order: an env-pinned file (`SCENEWORKS_CONTROLNET_FLUX1`) → a whole-repo HF cache snapshot →
/// download into the app cache. Mirrors the MLX `ensure_flux1_control_weights` / candle
/// `ensure_flux2_control_candle_weights`. The control checkpoint is lazy-fetched only on the first pose
/// job (vs bloating the base download).
async fn ensure_flux1_control_candle_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file) = flux1_control_candle_repo_file(request)?;
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_FLUX1") {
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
    let client = reqwest::Client::new();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "FLUX.1-dev strict-control generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-flux1-candle")
        .join(&file);
    // Pin the exact commit for the default control repo so `main` moving under us can't swap the
    // ControlNet checkpoint (sc-9879). A manifest `controlWeights.repo` override may carry its own
    // revision layout, so only pin when we're on the default repo.
    let revision = if repo == FLUX1_CONTROL_CANDLE_REPO {
        FLUX1_CONTROL_CANDLE_REVISION
    } else {
        "main"
    };
    ensure_hf_cached_file(&context, &repo, revision, &file, &dst).await?;
    Ok(dst)
}

/// Flat telemetry recorded on candle FLUX.1-dev control assets (parity with the MLX
/// `flux1_control_raw_settings`). dev is dense bf16 (no quant) — no `mlxQuantize` key.
fn flux1_control_candle_raw_settings(
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
        Value::String(FLUX1_CONTROL_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// The per-lane half of the candle FLUX.1-dev strict-control [`CandleStrictControl`] driver (sc-8412):
/// the resolved base + Shakker control weight paths, the request numerics, and the resolved control-kind
/// label (input-agnostic — used only to satisfy the engine's accepted-set check, shared across the pose
/// set). dev keeps its embedded guidance (no true-CFG / negative pass). Moved onto the blocking thread,
/// loaded once (dense bf16), drives every pose.
struct Flux1StrictControl {
    base: PathBuf,
    control: PathBuf,
    prompt: String,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    control_scale: f32,
    /// The resolved control kind for this set (`pose` | `canny` | `depth`) — input-agnostic; the engine
    /// validates it against its accepted set but does NOT branch the forward (Union-Pro-2.0 dropped the
    /// discrete mode index). The whole pose set shares one `controlMode`, so a single label is correct.
    control_kind: String,
}

impl CandleStrictControl for Flux1StrictControl {
    type Model = Flux1DevControl;

    fn engine_id(&self) -> &'static str {
        FLUX1_CONTROL_CANDLE_ENGINE_ID
    }

    fn engine_label(&self) -> &'static str {
        FLUX1_CONTROL_CANDLE_ENGINE
    }

    fn stream_tag(&self) -> &'static str {
        "flux1_control"
    }

    fn out_width(&self) -> u32 {
        self.width
    }

    fn out_height(&self) -> u32 {
        self.height
    }

    fn load(&self) -> WorkerResult<Self::Model> {
        let paths = Flux1ControlPaths {
            flux_base: self.base.clone(),
            control: self.control.clone(),
        };
        Flux1DevControl::load(&paths).map_err(|error| {
            WorkerError::Engine(format!("FLUX.1-dev strict-control load failed: {error}"))
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
        let req = Flux1ControlRequest {
            prompt: self.prompt.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps as usize,
            guidance: self.guidance,
            // The worker resolves `controlScale` itself (user value or `FLUX1_CONTROL_CANDLE_DEFAULT_SCALE`)
            // before building the provider, so always pass the resolved value explicitly. `None` (engine
            // default) is only for callers that never resolved a scale; sc-9024 keeps explicit 0.0 = "control
            // off" honored verbatim.
            control_scale: Some(self.control_scale),
            control_kind: self.control_kind.clone(),
            seed,
            cancel: cancel.clone(),
        };
        model.generate(&req, control, on_progress).map_err(|error| {
            WorkerError::Engine(format!("FLUX.1-dev strict-control generation failed: {error}"))
        })
    }
}

/// Real candle FLUX.1-dev strict-control generation: one image per pose, each conditioned on a full DWPose
/// skeleton (`controlMode` unset) or a canny/depth control map via the Shakker Union-Pro-2.0 branch
/// (sc-8412; engine sc-8239). Resolves the base + control weights, then hands a [`Flux1StrictControl`] to
/// the shared [`run_candle_strict_control`] driver (validation against `flux1_dev_control`'s
/// `supported_kinds`, per-pose preprocessing, scoring). dev loads dense bf16 (no quant) and keeps its
/// embedded guidance (no CFG). The pose path is byte-preserved.
async fn generate_candle_flux1_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let base = resolve_flux1_control_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("FLUX.1-dev base (FLUX.1-dev) weights not found".to_owned())
    })?;
    let control = ensure_flux1_control_candle_weights(api, settings, job, request).await?;

    let steps = flux1_control_candle_steps(request);
    let guidance = flux1_control_candle_guidance(request);
    let control_scale = advanced::f32_clamped(
        &request.advanced,
        "controlScale",
        FLUX1_CONTROL_CANDLE_DEFAULT_SCALE,
        0.0..=2.0,
    );
    // Resolve the requested control kind up front (the whole pose set shares one `controlMode`). The
    // shared driver re-validates it against `flux1_dev_control`'s `supported_kinds`; here we just carry the
    // label into the request struct (input-agnostic — the engine validates the accepted set but does not
    // branch the forward). Defaults to `pose` when `controlMode` is unset (byte-preserved skeleton path).
    let control_kind = control_kind_label(&requested_control_kind(request)?);
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(FLUX1_CONTROL_CANDLE_BASE_REPO)
        .to_owned();

    let pose_count = pose_entries(request).len();
    let raw_settings =
        flux1_control_candle_raw_settings(request, &repo, steps, guidance, control_scale, pose_count);

    let provider = Flux1StrictControl {
        base,
        control,
        prompt: request.prompt.clone(),
        width: request.width,
        height: request.height,
        steps,
        guidance,
        control_scale,
        control_kind,
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
