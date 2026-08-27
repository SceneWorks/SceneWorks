use super::{advanced, ensure_hf_cached_file, huggingface_snapshot_dir};
use super::{
    apply_candle_image_load_shape, attach_manifest_text_encoder, candle_certified_artifact_path,
    candle_certified_hf_artifact_path, candle_quant_for_resolved_tier, candle_resolved_tier_key,
    pid_effective_dims, pid_output_tier, pose_entries, resolve_adapters,
    resolve_advanced_or_manifest_f32, resolve_advanced_or_manifest_u32, resolve_pid_weights,
    run_candle_strict_control, trusted_control_weight_revision, ApiClient, CancelFlag,
    CandleStrictControl, Flux2Control, Flux2ControlPaths, Flux2ControlRequest, Image, ImagePlan,
    ImageRequest, JobSnapshot, JsonObject, Path, PathBuf, Progress, Quant, Settings, Value,
    WorkerError, WorkerResult,
};
use super::{
    resolve_app_managed_model_dir, safe_weight_filename, standard_tier_subdir, DownloadContext,
};
use crate::conditioning_fit::{ConditioningAdmission, ConditioningFootprint};
use serde_json::json;

pub(super) fn flux2_control_adapter_source_bytes(
    adapters: &[gen_core::AdapterSpec],
) -> WorkerResult<u64> {
    gen_core::adapter_stack_resident_bytes(adapters, gen_core::AdapterResidencyMode::Additive)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "FLUX.2 control cannot determine the resident size of the requested adapter stack."
                    .to_owned(),
            )
        })
}

// Candle (Windows/CUDA) FLUX.2-dev strict-pose Fun-Controlnet-Union route (sc-7736, epic 6564) —
// `flux2_dev` + `advanced.poses` off-Mac via `runtime_cuda::providers::flux2::Flux2Control`. The candle sibling of the
// MLX FLUX.2-dev strict-pose path (flux2.rs `generate_flux2_dev_control_stream`, sc-6055 / engine
// sc-2292): one image per library pose, each conditioned on a full DWPose skeleton (rendered
// cross-platform by `openpose_skeleton::draw_wholebody`) fed to the VACE-style control branch overlaid on
// the dev DiT (`alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`). True pose lock, not the best-effort
// `MultiReference [skeleton, reference]` edit tier.
//
// **Candle-only.** macOS keeps the MLX `flux2_dev_control` registry generator (flux2.rs); the candle
// `Flux2Control` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build (the
// the module declaration in image_jobs.rs carries the cfg). It is a child module of the `image_jobs` module, so it
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
/// checkpoint we load; pin the exact commit for defense-in-depth (mirrors sc-8879/sc-9682). Registered
/// overlays carry their own catalog-authorized immutable revision. HF's tree API still reports the
/// file's `lfs.oid`, which `ensure_hf_cached_file` verifies against.
#[cfg(test)]
pub(super) const FLUX2_CONTROL_CANDLE_REVISION: &str = "b3dcd7836a0e926248dac3ccba8fc0853495764b";
/// The FLUX.2-dev base diffusers repo when the manifest omits `repo` (the 32B flagship). The candle lane
/// loads the dense snapshot and Q4-quantizes it at load.
const FLUX2_CONTROL_CANDLE_BASE_REPO: &str = "black-forest-labs/FLUX.2-dev";
/// Pose ControlNet conditioning-scale default — the dev Fun-Controlnet-Union README sweet spot is
/// 0.65–0.80, the worker (and engine `DEFAULT_CONTROL_SCALE`) default 0.75. Clamp [0, 2].
pub(super) const FLUX2_CONTROL_CANDLE_DEFAULT_SCALE: f32 = 0.75;
/// Denoise-steps default — the guidance-distilled dev (FLUX.1-dev pattern, ~28 steps).
const FLUX2_CONTROL_CANDLE_DEFAULT_STEPS: u32 = 28;
/// Embedded-guidance default — distilled dev scalar (NOT true-CFG, no negative pass).
const FLUX2_CONTROL_CANDLE_DEFAULT_GUIDANCE: f32 = 4.0;
/// The adapter/engine id recorded on candle FLUX.2-dev control assets (distinct from the txt2img
/// `candle_flux2` + edit `candle_flux2_edit` lanes).
pub(super) const FLUX2_CONTROL_CANDLE_ENGINE: &str = "candle_flux2_control";
/// The [`STRICT_CONTROL_ENGINES`] catalog id this candle lane validates `advanced.controlMode` against
/// (the dev Fun-Controlnet-Union row — `{Pose, Canny, Depth}`). Mirrors the MLX `flux2_dev_control`
/// registry engine's `supported_kinds` (sc-8304).
pub(super) const FLUX2_CONTROL_CANDLE_ENGINE_ID: &str = "flux2_dev_control";

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
        return resolve_app_managed_model_dir(settings, &path, "FLUX.2 control modelPath")
            .map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            crate::engines::default_repo_for(&request.model)
                .unwrap_or(FLUX2_CONTROL_CANDLE_BASE_REPO)
        });
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo)
        .map(|root| standard_tier_subdir(&root, request)))
}

/// True when this is a candle-eligible FLUX.2-dev strict-pose job: `flux2_dev` with a non-empty
/// `advanced.poses`, not edit mode, whose base resolves locally. Mirrors
/// `jobs_store::flux2_dev_control_candle_eligible` so the worker and router agree. Control-weights
/// presence is NOT part of the gate: they are fetched on first use in the stream.
pub(super) fn flux2_control_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_flux2_control_model(&request.model)
        && request.mode != "edit_image"
        && flux2_control_candle_pose_count(request).is_some_and(|count| count > 0)
        && matches!(resolve_flux2_control_base(request, settings), Ok(Some(_)))
}

/// Strict scheduler/worker pose contract: no filtered-away malformed entries and no unbounded set.
fn flux2_control_candle_pose_count(request: &ImageRequest) -> Option<usize> {
    match request.advanced.get("poses") {
        None | Some(Value::Null) => Some(0),
        Some(Value::Array(poses))
            if poses.len() <= sceneworks_core::image_request::MAX_JOB_POSES
                && poses.iter().all(Value::is_object) =>
        {
            Some(poses.len())
        }
        Some(_) => None,
    }
}

/// Strict control consumes a control-map overlay, not an image-reference edit route. Character
/// Studio and style-variation entrypoints therefore share the provider's canonical text-to-image,
/// zero-reference memory behavior rather than minting unsupported catalog-mode contexts.
fn flux2_control_memory_request_mode(_catalog_mode: &str) -> &'static str {
    "text_to_image"
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
pub(super) fn flux2_control_candle_repo_file(
    request: &ImageRequest,
) -> WorkerResult<(String, String)> {
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
    let repo = pick("repo", FLUX2_CONTROL_CANDLE_REPO);
    let file = safe_weight_filename(
        &pick("filename", FLUX2_CONTROL_CANDLE_FILE),
        "advanced.controlWeights.filename",
    )?;
    trusted_control_weight_revision(request, FLUX2_CONTROL_CANDLE_ENGINE_ID, &repo, &file)?;
    Ok((repo, file))
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
    let revision =
        trusted_control_weight_revision(request, FLUX2_CONTROL_CANDLE_ENGINE_ID, &repo, &file)?;
    if let Some(snapshot) =
        crate::model_jobs::huggingface_pinned_snapshot_dir(&settings.data_dir, &repo, &revision)
    {
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
        cancel_message:
            "FLUX.2-dev strict-pose generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-flux2")
        .join(&file);
    // Pin the exact commit for the default control repo so `main` moving under us can't swap the
    // ControlNet checkpoint (sc-9879). Registered overlays carry their own immutable pin.
    ensure_hf_cached_file(&context, &repo, &revision, &file, &dst).await?;
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
pub(super) struct Flux2StrictControl {
    base: PathBuf,
    control: PathBuf,
    quant: Option<Quant>,
    prompt: String,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    control_scale: f32,
    memory: gen_core::GenerationMemory,
    memory_spec: gen_core::LoadSpec,
    memory_context: Option<gen_core::MemoryRunContext>,
    /// Per-generation PiD decoder weights (epic 7840, sc-8044): `Some` only when this generation opted in
    /// (`advanced.usePid`) AND the `flux2` PiD + Gemma snapshots are cached. Threaded into `with_pid` at
    /// load; `use_pid` on the request is `is_some()` so the two stay in lockstep (the engine rejects a
    /// mismatch). `None` ⇒ native FLUX.2 VAE decode.
    pid: Option<gen_core::PidWeights>,
    adapters: Vec<gen_core::AdapterSpec>,
}

#[cfg(test)]
pub(super) fn flux2_strict_control_test_fixture(path: PathBuf) -> Flux2StrictControl {
    let memory_spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(path.clone()));
    Flux2StrictControl {
        base: path.clone(),
        control: path,
        quant: None,
        prompt: "p".to_owned(),
        width: 512,
        height: 512,
        steps: 28,
        guidance: 4.0,
        control_scale: 0.75,
        memory: gen_core::GenerationMemory::default(),
        memory_spec,
        memory_context: None,
        pid: None,
        adapters: Vec::new(),
    }
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

    /// The FLUX.2-dev base tier dir + the Fun-Controlnet-Union overlay, plus the PiD decoder pair when
    /// this generation opted in — every path [`Self::load`] holds co-resident (sc-16069).
    fn conditioning_admission(&self) -> ConditioningAdmission {
        let mut overlays = vec![self.control.as_path()];
        overlays.extend(crate::conditioning_fit::pid_paths(self.pid.as_ref()));
        overlays.extend(self.adapters.iter().map(|adapter| adapter.path.as_path()));
        if let Some(text_encoder) = self.memory_spec.text_encoder.as_ref() {
            let transformer = self.base.join("transformer");
            let vae = self.base.join("vae");
            if transformer.is_dir() && vae.is_dir() {
                overlays.push(crate::conditioning_fit::weights_source_path(text_encoder));
                overlays.push(vae.as_path());
                return ConditioningAdmission::Floor(ConditioningFootprint::from_paths(
                    "FLUX.2-dev",
                    "strict-pose Fun-Controlnet-Union branch",
                    &transformer,
                    &overlays,
                ));
            }
        }
        ConditioningAdmission::Floor(ConditioningFootprint::from_paths(
            "FLUX.2-dev",
            "strict-pose Fun-Controlnet-Union branch",
            &self.base,
            &overlays,
        ))
    }

    fn load(&self) -> WorkerResult<Self::Model> {
        let paths = Flux2ControlPaths {
            root: self.base.clone(),
            control: self.control.clone(),
            adapters: self.adapters.clone(),
        };
        let loaded = match &self.memory_context {
            Some(context) => Flux2Control::load_with_memory_context(
                &paths,
                self.quant,
                &self.memory_spec,
                context,
            ),
            None => Flux2Control::load_with_memory_spec(
                &paths,
                self.quant,
                &self.memory_spec,
                self.memory,
            ),
        };
        let model = loaded.map_err(|error| {
            WorkerError::Engine(format!(
                "FLUX.2-dev strict-pose control load failed: {error}"
            ))
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
        preview: &gen_core::PreviewSink,
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
            preview: preview.clone(),
            cancel: cancel.clone(),
        };
        let generated = match self.memory_context.as_ref() {
            Some(context) => {
                model.generate_with_memory_context(context, &req, control, on_progress)
            }
            None => model.generate(&req, control, on_progress),
        };
        generated.map_err(|error| {
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
pub(super) async fn generate_candle_flux2_control_stream(
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

    // The resolved base tier drives both load quant and memory receipt. The control overlay quantizes
    // in place. The control context is clean + constant across the denoise (encoded once).
    let tier = candle_resolved_tier_key(request, &base, false);
    let (quant, quant_bits) = candle_quant_for_resolved_tier(request, tier, &base, true, false);
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
        .unwrap_or_else(|| {
            crate::engines::default_repo_for(&request.model)
                .unwrap_or(FLUX2_CONTROL_CANDLE_BASE_REPO)
        })
        .to_owned();

    let pose_count = pose_entries(request).len();
    // Per-generation PiD decode (epic 7840, sc-8044): resolve the `flux2` PiD student + Gemma when
    // `advanced.usePid` is set and the snapshots are cached; else `None` → native FLUX.2 VAE.
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();
    // PiD output tier (sc-10054): 2K caps the effective base so PiD's fixed 4× lands on ~2048 (default
    // 4K/native leaves the requested dims untouched). The shared driver renders the control map at these
    // same dims (via `out_width`/`out_height`), keeping control + latent aligned.
    let (width, height) = pid_effective_dims(
        request.width,
        request.height,
        use_pid,
        pid_output_tier(request),
    );
    let adapters = resolve_adapters(request, settings)?;
    let adapter_source_bytes = flux2_control_adapter_source_bytes(&adapters)?;
    let runtime_overlay_bytes = gen_core::weightsmeta::safetensors_path_bytes(&control)
        .saturating_add(adapter_source_bytes);
    let mut strategy_spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(base.clone()))
        .with_control(gen_core::WeightsSource::File(control.clone()))
        .with_adapters(adapters.clone())
        .with_offload_policy(gen_core::OffloadPolicy::Sequential);
    strategy_spec.quantize = quant;
    if let Some(pid) = pid_weights.as_ref() {
        strategy_spec = strategy_spec.with_pid(pid.checkpoint.clone(), pid.gemma.clone());
    }
    let strategy_spec = apply_candle_image_load_shape("flux2_dev", strategy_spec);
    let attached_strategy_spec = attach_manifest_text_encoder(
        strategy_spec,
        FLUX2_CONTROL_CANDLE_ENGINE_ID,
        request,
        settings,
    )?;
    let strategy_spec = attached_strategy_spec.into_load_spec();
    let raw_budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    let predicted_peak = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        &request.model_manifest_entry,
        tier,
        runtime_overlay_bytes,
    );
    let (control_repo, control_file) = flux2_control_candle_repo_file(request)?;
    let control_revision = trusted_control_weight_revision(
        request,
        FLUX2_CONTROL_CANDLE_ENGINE_ID,
        &control_repo,
        &control_file,
    )?;
    let artifact_is_certified = candle_certified_artifact_path("flux2_dev", settings, &base, tier)
        && candle_certified_hf_artifact_path(
            settings,
            &control_repo,
            &control_revision,
            Path::new(&control_file),
            &control,
        );
    let memory_evaluation = crate::candle_memory_strategy::evaluate_shared_image(
        "flux2_dev",
        &request.model,
        &strategy_spec,
        artifact_is_certified,
        &request.model_manifest_entry,
        tier,
        flux2_control_memory_request_mode(&request.mode),
        Some("control"),
        gen_core::MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            // The control map is an overlay input, not a token-concatenated image reference.
            reference_count: 0,
        },
        false,
        use_pid,
        false,
        false,
        raw_budget,
        predicted_peak,
        runtime_overlay_bytes,
        gen_core::MemoryCacheState::Cold,
    )?;
    let generation_memory = memory_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.memory)
        .unwrap_or_default();
    let memory_context = memory_evaluation
        .as_ref()
        .map(|evaluation| evaluation.context.clone());
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
    if let Some(evaluation) = &memory_evaluation {
        raw_settings.insert(
            "memoryStrategy".to_owned(),
            Value::String(format!("{:?}", evaluation.context.selection.strategy)),
        );
    }

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
        memory: generation_memory,
        memory_spec: strategy_spec,
        memory_context,
        pid: pid_weights,
        adapters,
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

#[cfg(test)]
mod memory_route_tests {
    use super::flux2_control_memory_request_mode;

    #[test]
    fn every_eligible_non_edit_pose_mode_uses_the_control_contract_route() {
        for mode in ["image_generation", "character_image", "style_variations"] {
            assert_eq!(flux2_control_memory_request_mode(mode), "text_to_image");
        }
    }
}
