// Candle (Windows/CUDA) FLUX.2-dev strict-pose Fun-Controlnet-Union route (sc-7736, epic 6564) —
// `flux2_dev` + `advanced.poses` off-Mac via `candle_gen_flux2::Flux2Control`. The candle sibling of the
// MLX FLUX.2-dev strict-pose path (flux2.rs `generate_flux2_dev_control_stream`, sc-6055 / engine
// sc-2292): one image per library pose, each conditioned on a full DWPose skeleton (rendered
// cross-platform by `openpose_skeleton::draw_wholebody`) fed to the VACE-style control branch overlaid on
// the dev DiT (`alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`). True pose lock, not the best-effort
// `MultiReference [skeleton, reference]` edit tier.
//
// **Candle-only.** macOS keeps the MLX `flux2_dev_control` registry generator (flux2.rs); the candle
// `Flux2Control` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build (the
// `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs` module, so it
// shares that module's imports (`parse_poses`/`pose_entries`/`Settings`/`WorkerResult`/`resolve_quant`/
// `huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).
//
// The dev base is the 32B flagship, so it loads via the Q4 CPU-stage → quantize-onto-GPU path
// (`resolve_quant` reads the manifest `mlx.quantize: 4`); the ~8 GB bf16 Fun-Controlnet-Union overlay
// loads dense on the device and quantizes in place. dev is guidance-distilled — a single embedded-
// guidance forward, no true-CFG / negative pass. `control_scale = 0` is engine-proven byte-identical to
// the base txt2img forward.

/// Default Fun-Controlnet-Union control-weights repo + the `-2602` CFG-distilled variant (the recommended
/// one — the previous version lost CFG distillation after control training). Parity with the MLX
/// `FLUX2_CONTROL_REPO` / `FLUX2_CONTROL_FILE`.
const FLUX2_CONTROL_CANDLE_REPO: &str = "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union";
const FLUX2_CONTROL_CANDLE_FILE: &str = "FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors";
/// Pinned revision for the default `FLUX2_CONTROL_CANDLE_REPO` (sc-9879, F-077 follow-up). Fetching the
/// mutable `main` branch means a re-push (or a compromised token) could silently swap the ControlNet
/// checkpoint we load; pin the exact commit for defense-in-depth (mirrors sc-8879/sc-9682). Applied ONLY
/// to the default repo — a manifest `controlWeights.repo` override keeps `main`. HF's tree API still
/// reports the file's `lfs.oid`, which `ensure_hf_cached_file` verifies against.
const FLUX2_CONTROL_CANDLE_REVISION: &str = "b3dcd7836a0e926248dac3ccba8fc0853495764b";
/// The FLUX.2-dev base diffusers repo when the manifest omits `repo` (the 32B flagship). The candle lane
/// loads the dense snapshot and Q4-quantizes it at load.
const FLUX2_CONTROL_CANDLE_BASE_REPO: &str = "black-forest-labs/FLUX.2-dev";
/// Pose ControlNet conditioning-scale default — the dev Fun-Controlnet-Union README sweet spot is
/// 0.65–0.80, the worker (and engine `DEFAULT_CONTROL_SCALE`) default 0.75. Clamp [0, 2].
const FLUX2_CONTROL_CANDLE_DEFAULT_SCALE: f32 = 0.75;
/// Denoise-steps default — the guidance-distilled dev (FLUX.1-dev pattern, ~28 steps).
const FLUX2_CONTROL_CANDLE_DEFAULT_STEPS: u32 = 28;
/// Embedded-guidance default — distilled dev scalar (NOT true-CFG, no negative pass).
const FLUX2_CONTROL_CANDLE_DEFAULT_GUIDANCE: f32 = 4.0;
/// The adapter/engine id recorded on candle FLUX.2-dev control assets (distinct from the txt2img
/// `candle_flux2` + edit `candle_flux2_edit` lanes).
const FLUX2_CONTROL_CANDLE_ENGINE: &str = "candle_flux2_control";
/// The [`STRICT_CONTROL_ENGINES`] catalog id this candle lane validates `advanced.controlMode` against
/// (the dev Fun-Controlnet-Union row — `{Pose, Canny, Depth}`). Mirrors the MLX `flux2_dev_control`
/// registry engine's `supported_kinds` (sc-8304).
const FLUX2_CONTROL_CANDLE_ENGINE_ID: &str = "flux2_dev_control";

/// Model ids the candle FLUX.2 strict-pose control route accepts (klein has no control checkpoint).
fn is_flux2_control_model(model: &str) -> bool {
    model == "flux2_dev"
}

/// Resolve the FLUX.2-dev base (diffusers) snapshot: an explicit `modelPath` (advanced or manifest) → the
/// HF cache snapshot for the manifest `repo` (default `black-forest-labs/FLUX.2-dev`). `None` ⇒ not
/// present locally (the job is not candle-runnable). Mirrors `resolve_zimage_control_base`.
fn resolve_flux2_control_base(
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
        return resolve_app_managed_model_dir(settings, &path, "FLUX.2 control modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(FLUX2_CONTROL_CANDLE_BASE_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible FLUX.2-dev strict-pose job: `flux2_dev` with a non-empty
/// `advanced.poses`, not edit mode, whose base resolves locally. Mirrors
/// `jobs_store::flux2_dev_control_candle_eligible` so the worker and router agree. Control-weights
/// presence is NOT part of the gate: they are fetched on first use in the stream.
fn flux2_control_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_flux2_control_model(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_flux2_control_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → default (28).
fn flux2_control_candle_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", FLUX2_CONTROL_CANDLE_DEFAULT_STEPS, 1..=50)
}

/// Resolve embedded guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (4.0),
/// clamped. dev rides this scalar on the transformer's guidance embedder (no true-CFG).
fn flux2_control_candle_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        FLUX2_CONTROL_CANDLE_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// The (repo, filename) of the control weights — `advanced.controlWeights.{repo,filename}` overrides,
/// else the Fun-Controlnet-Union `-2602` default (parity with the MLX `flux2_control_repo_file`).
/// The payload filename must be a plain component (sc-8821 / F-019).
fn flux2_control_candle_repo_file(request: &ImageRequest) -> WorkerResult<(String, String)> {
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
        pick("repo", FLUX2_CONTROL_CANDLE_REPO),
        safe_weight_filename(
            &pick("filename", FLUX2_CONTROL_CANDLE_FILE),
            "advanced.controlWeights.filename",
        )?,
    ))
}

/// Resolve the Fun-Controlnet-Union weight **file** the `Flux2Control` provider loads, downloading on
/// first use. Order: an env-pinned file (`SCENEWORKS_CONTROLNET_FLUX2`) → a whole-repo HF cache snapshot →
/// download into the app cache. Mirrors the MLX `ensure_flux2_control_weights` / candle
/// `ensure_zimage_control_weights`. The ~8 GB control checkpoint is lazy-fetched only on the first pose
/// job (vs bloating the base download).
async fn ensure_flux2_control_candle_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file) = flux2_control_candle_repo_file(request)?;
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_FLUX2") {
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
        cancel_message: "FLUX.2-dev strict-pose generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-flux2")
        .join(&file);
    // Pin the exact commit for the default control repo so `main` moving under us can't swap the
    // ControlNet checkpoint (sc-9879). A manifest `controlWeights.repo` override may carry its own
    // revision layout, so only pin when we're on the default repo.
    let revision = if repo == FLUX2_CONTROL_CANDLE_REPO {
        FLUX2_CONTROL_CANDLE_REVISION
    } else {
        "main"
    };
    ensure_hf_cached_file(&context, &repo, revision, &file, &dst).await?;
    Ok(dst)
}

/// Flat telemetry recorded on candle FLUX.2-dev control assets (parity with the MLX
/// `flux2_control_raw_settings`).
#[allow(clippy::too_many_arguments)]
fn flux2_control_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    quant_bits: Option<i64>,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(FLUX2_CONTROL_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// The per-lane half of the candle FLUX.2-dev strict-control [`CandleStrictControl`] driver (sc-8304):
/// the resolved base + control weight paths, the Q4 quant policy, and the request numerics. dev keeps its
/// embedded guidance (no true-CFG / negative pass). Moved onto the blocking thread, loaded once (Q4
/// CPU-stage → quantize-onto-GPU), drives every pose.
struct Flux2StrictControl {
    base: PathBuf,
    control: PathBuf,
    quant: Option<Quant>,
    prompt: String,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    control_scale: f32,
    /// Per-generation PiD decoder weights (epic 7840, sc-8044): `Some` only when this generation opted in
    /// (`advanced.usePid`) AND the `flux2` PiD + Gemma snapshots are cached. Threaded into `with_pid` at
    /// load; `use_pid` on the request is `is_some()` so the two stay in lockstep (the engine rejects a
    /// mismatch). `None` ⇒ native FLUX.2 VAE decode.
    pid: Option<gen_core::PidWeights>,
}

impl CandleStrictControl for Flux2StrictControl {
    type Model = Flux2Control;

    fn engine_id(&self) -> &'static str {
        FLUX2_CONTROL_CANDLE_ENGINE_ID
    }

    fn engine_label(&self) -> &'static str {
        FLUX2_CONTROL_CANDLE_ENGINE
    }

    fn stream_tag(&self) -> &'static str {
        "flux2_control"
    }

    fn out_width(&self) -> u32 {
        self.width
    }

    fn out_height(&self) -> u32 {
        self.height
    }

    fn load(&self) -> WorkerResult<Self::Model> {
        let paths = Flux2ControlPaths {
            root: self.base.clone(),
            control: self.control.clone(),
        };
        let model = Flux2Control::load(&paths, self.quant).map_err(|error| {
            WorkerError::Engine(format!("FLUX.2-dev strict-pose control load failed: {error}"))
        })?;
        // Attach the optional PiD decoder (sc-8044): `Some` only when opted in AND the snapshots are cached.
        match &self.pid {
            Some(pid) => model.with_pid(pid).map_err(|error| {
                WorkerError::Engine(format!("FLUX.2 control PiD decoder load failed: {error}"))
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
        let req = Flux2ControlRequest {
            prompt: self.prompt.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps as usize,
            guidance: self.guidance,
            control_scale: self.control_scale,
            seed,
            // PiD opt-in (sc-8044): in lockstep with the `with_pid` load — `is_some()` ⇒ decoder loaded.
            use_pid: self.pid.is_some(),
            cancel: cancel.clone(),
        };
        model.generate(&req, control, on_progress).map_err(|error| {
            WorkerError::Engine(format!("FLUX.2-dev strict-pose generation failed: {error}"))
        })
    }
}

/// Real candle FLUX.2-dev strict-pose generation: one image per pose, each conditioned on a full DWPose
/// skeleton (`controlMode` unset) or a canny/depth control map via the Fun-Controlnet-Union branch
/// (sc-7736; engine sc-7460). Resolves the base + control weights + Q4 quant, then hands a
/// [`Flux2StrictControl`] to the shared [`run_candle_strict_control`] driver (validation against
/// `flux2_dev_control`'s `supported_kinds`, per-pose preprocessing, scoring). dev (32B) loads Q4 (manifest
/// `mlx.quantize: 4` → `resolve_quant`); the control overlay quantizes in place. dev keeps its embedded
/// guidance (no CFG). The pose path is byte-preserved.
async fn generate_candle_flux2_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let base = resolve_flux2_control_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("FLUX.2-dev base (FLUX.2-dev) weights not found".to_owned())
    })?;
    let control = ensure_flux2_control_candle_weights(api, settings, job, request).await?;

    // dev (32B) loads Q4 (manifest `mlx.quantize: 4` → `resolve_quant`); the control overlay quantizes
    // in place. The control context is clean + constant across the denoise (encoded once).
    let (quant, quant_bits) = resolve_quant(request);
    let steps = flux2_control_candle_steps(request);
    let guidance = flux2_control_candle_guidance(request);
    let control_scale = advanced::f32_clamped(
        &request.advanced,
        "controlScale",
        FLUX2_CONTROL_CANDLE_DEFAULT_SCALE,
        0.0..=2.0,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(FLUX2_CONTROL_CANDLE_BASE_REPO)
        .to_owned();

    let pose_count = pose_entries(request).len();
    // Per-generation PiD decode (epic 7840, sc-8044): resolve the `flux2` PiD student + Gemma when
    // `advanced.usePid` is set and the snapshots are cached; else `None` → native FLUX.2 VAE.
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();
    // PiD output tier (sc-10054): 2K caps the effective base so PiD's fixed 4× lands on ~2048 (default
    // 4K/native leaves the requested dims untouched). The shared driver renders the control map at these
    // same dims (via `out_width`/`out_height`), keeping control + latent aligned.
    let (width, height) =
        pid_effective_dims(request.width, request.height, use_pid, pid_output_tier(request));
    let mut raw_settings = flux2_control_candle_raw_settings(
        request,
        &repo,
        steps,
        guidance,
        quant_bits,
        control_scale,
        pose_count,
    );
    // Mark PiD output on the sidecar (NSCLv1 NC flows to PiD output); record whether PiD actually ran.
    raw_settings.insert("usePid".to_owned(), Value::Bool(use_pid));

    let provider = Flux2StrictControl {
        base,
        control,
        quant,
        prompt: request.prompt.clone(),
        width,
        height,
        steps,
        guidance,
        control_scale,
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
