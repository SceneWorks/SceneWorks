// macOS (Apple Silicon / MLX) Krea 2 pose-ControlNet route (sc-8465, epic 8459 S5) — the MLX twin of the
// candle `krea_control_candle.rs` lane (sc-8464). `krea_2_turbo` + `advanced.poses` on the Mac worker
// routes to the registry-backed `krea_2_turbo_control` engine (mlx-gen `KreaTurboControl`): the converted
// MLX control-branch overlay rides the frozen dense Krea 2 Turbo base, rendering one image per library
// pose, each conditioned on a full DWPose skeleton (rendered by `openpose_skeleton::draw_wholebody`, the
// SAME renderer training + the candle lane use). True pose lock via a residual on the single CFG-free
// guidance forward, scaled by `control_scale`; `control_scale = 0` is engine-proven byte-identical to base.
//
// **macOS-only.** The Windows/CUDA sibling is `krea_control_candle.rs`; the `include!` in image_jobs.rs
// carries the `cfg(target_os = "macos")`. This file is `include!`d into the `image_jobs` module, so it
// shares that module's imports (`parse_poses`/`pose_entries`/`Settings`/`WorkerResult`/`LoadSpec`/
// `start_cached_gen_stream`/… all in scope unqualified), exactly like `zimage.rs`.
//
// Registry-backed like the other MLX control lanes (Z-Image / Qwen / Kolors): resolve the base snapshot +
// the overlay checkpoint into a `LoadSpec`, then `start_cached_gen_stream(krea_2_turbo_control, …)` feeds
// `Conditioning::Control` per pose. The base is any complete Krea 2 Turbo snapshot: a legacy dense
// `krea/Krea-2-Turbo` diffusers repo when separately cached, else the installed `SceneWorks/krea-2-turbo-mlx`
// tier (q8 default / q4 / bf16) the txt2img lane uses. The dense `krea/Krea-2-Turbo` download entry was
// retired (sc-9092), so the installed tier is the base a current user actually has; the MLX
// `Krea2Transformer` packed-detects the tier and runs a true packed forward on the control base (mlx-gen
// sc-11730 / candle-gen #471, sc-11727) — pose-lock holds on q8/q4 (q8 ~ bf16, q4 mild haze).

/// The engine registry id — matches the mlx-gen `KreaTurboControl` registration and the shared
/// `STRICT_CONTROL_ENGINES` `krea_2_turbo_control` row (`supported_kinds = {Pose}`). One id, both
/// backends; the `cfg(target_os)` picks the MLX vs candle provider.
const KREA_CONTROL_ENGINE_ID: &str = "krea_2_turbo_control";
/// Pose ControlNet conditioning-scale default (mlx-gen `krea::control::DEFAULT_CONTROL_SCALE` / candle
/// parity). S0 usable band ~0.5–0.85 for the distilled CFG-free base; a comfortable mid.
const KREA_CONTROL_DEFAULT_SCALE: f32 = 0.6;
/// Hard cap on the exposed `control_scale` — above ~0.85 the frozen CFG-free base over-drives to halftone
/// (S0: graceful soft-haze, never confetti, but not a usable range). Matches the candle lane cap.
const KREA_CONTROL_SCALE_CAP: f32 = 0.85;
/// Denoise-steps default — the distilled Turbo schedule (8-step CFG-free).
const KREA_CONTROL_DEFAULT_STEPS: u32 = 8;
/// Env override → a Krea 2 Turbo dense snapshot dir (validation / bring-your-own base). Shared with the
/// candle lane so a single machine's env drives whichever backend it runs.
const KREA_CONTROL_BASE_ENV: &str = "SCENEWORKS_KREA_CONTROL_BASE";
/// Env override → a converted MLX control-branch overlay `.safetensors` (validation / bring-your-own).
const KREA_CONTROL_WEIGHTS_ENV: &str = "SCENEWORKS_CONTROLNET_KREA";
/// Default published Krea pose control-branch overlay repo (sc-8466) — the S0 spike (5,000-step)
/// checkpoint. EXPERIMENTAL / not-for-production.
const KREA_CONTROL_OVERLAY_REPO: &str = "SceneWorks/krea2-pose-controlnet-beta";
/// The overlay file within [`KREA_CONTROL_OVERLAY_REPO`] — the SAME candle `control_step5000.safetensors`
/// the candle lane loads. The MLX branch reads it DIRECTLY (mlx-gen `RmsScale` accepts the candle
/// `*.weight_p1` norm convention verbatim, sc-8465), so there is no separate MLX artifact to host.
const KREA_CONTROL_OVERLAY_FILE: &str = "control_step5000.safetensors";
/// Pinned revision for the default overlay repo (defense-in-depth, parity with the candle lane's
/// `KREA_CONTROL_OVERLAY_REVISION` — a repo re-push can't swap the checkpoint under us). Registered
/// overlays carry their own catalog-authorized immutable revision.
#[cfg(test)]
const KREA_CONTROL_OVERLAY_REVISION: &str = "cb3a0ac7590f5ec594a4eeb43b95ee1da0b5a0ac";
/// The `SceneWorks/krea-2-turbo-mlx` turnkey (q8 default / q4 / bf16 self-contained subdirs) — the SAME
/// base the txt2img lane loads. The pose-control lane falls back here (via the shared [`krea_model_subdir`])
/// when the legacy dense `krea/Krea-2-Turbo` repo is absent (retired, sc-9092). Mirrors the candle lane's
/// `KREA_CONTROL_MLX_REPO` (sc-11727).
const KREA_CONTROL_MLX_REPO: &str = "SceneWorks/krea-2-turbo-mlx";
const KREA_CONTROL_BASE_REVISION: &str = "d009674080cc1bccf2b629d834c34bf5eccdb723";
const KREA_CONTROL_OVERLAY_PIN: &str = "cb3a0ac7590f5ec594a4eeb43b95ee1da0b5a0ac";

/// The image-conditioning count one strict-pose render carries: the pose `Conditioning::Control`,
/// and nothing else (this lane never adds an identity `Reference` — it renders from noise).
///
/// This MUST equal what gen-core derives from the request the lane actually sends
/// (`GenerationRequest::image_reference_count`, which charges `Control`/`Depth`/`Mask` as one image
/// reference each), because the admitted geometry is re-checked against the live request twice:
/// gen-core's shared safety check rejects `has_reference != (reference_count > 0)`, and the MLX
/// request scope refuses any request whose geometry differs from the admitted one. Declaring zero
/// here refused every pose render with
/// `krea_2_turbo_control: request geometry WxHx1 references=1 does not fit admitted WxHx1
/// references=0` — the mlx-gen twins already agree on 1 (`mlx-gen-krea`'s pose-control behavior
/// fixture and the candle control provider's evidence probe both declare `reference_count: 1`).
const KREA_CONTROL_REFERENCE_COUNT: u32 = 1;

/// The per-request MLX memory declaration for one strict-pose render. Split out from the lane so the
/// declared geometry can be graded against the conditioning the lane really builds — see
/// [`KREA_CONTROL_REFERENCE_COUNT`].
fn krea_control_memory_inputs(
    width: u32,
    height: u32,
    source_mode: &str,
    adapter_count: usize,
) -> crate::mlx_fit_gate::MlxRequestInputs {
    // Character Studio labels this job `character_image`, while the ordinary image route labels it
    // `image_generation`. Neither label changes what this provider executes: Krea pose control
    // starts from noise and carries one Control image, not an img2img init. The real measurements,
    // promoted bindings, and calibration adapter therefore share the canonical `text_to_image` +
    // `control:1` identity. Normalize the UI/source label here so the live Character Studio path can
    // select those exact measured cells instead of silently falling back to estimate-backed
    // admission under `image_to_image`.
    debug_assert!(matches!(
        source_mode,
        "character_image" | "image_to_image" | "image_generation" | "text_to_image"
    ));
    crate::mlx_fit_gate::MlxRequestInputs {
        width,
        height,
        count: 1,
        mode: "text_to_image".to_owned(),
        overlay: Some("control:1".to_owned()),
        adapter_count,
        has_reference: KREA_CONTROL_REFERENCE_COUNT > 0,
        reference_count: KREA_CONTROL_REFERENCE_COUNT,
        use_pid: false,
        has_phases: false,
    }
}

fn krea_control_calibration_provenance(
    weights_dir: &Path,
    control_weights: &Path,
    verified_default_overlay: bool,
) -> Option<crate::model_jobs::ResolvedArtifactProvenance> {
    let weights_dir = std::fs::canonicalize(weights_dir).ok()?;
    let control_weights = std::fs::canonicalize(control_weights).ok()?;
    let tier = weights_dir.file_name()?.to_str()?;
    if tier != "q4" {
        return None;
    }
    let base_suffix = format!(
        "models--SceneWorks--krea-2-turbo-mlx/snapshots/{KREA_CONTROL_BASE_REVISION}/q4"
    );
    let overlay_suffix = format!(
        "models--SceneWorks--krea2-pose-controlnet-beta/snapshots/{KREA_CONTROL_OVERLAY_PIN}/{KREA_CONTROL_OVERLAY_FILE}"
    );
    if !weights_dir.to_string_lossy().ends_with(&base_suffix) {
        return None;
    }
    if !verified_default_overlay && !control_weights.to_string_lossy().ends_with(&overlay_suffix) {
        return None;
    }
    Some(crate::model_jobs::ResolvedArtifactProvenance {
        identity: crate::model_jobs::ResolvedArtifactIdentity {
            repository: KREA_CONTROL_MLX_REPO.to_owned(),
            revision: KREA_CONTROL_BASE_REVISION.to_owned(),
            variant: tier.to_owned(),
            fingerprint: format!(
                "{KREA_CONTROL_MLX_REPO}@{KREA_CONTROL_BASE_REVISION}:q4|\
                 {KREA_CONTROL_OVERLAY_REPO}@{KREA_CONTROL_OVERLAY_PIN}:{KREA_CONTROL_OVERLAY_FILE}"
            ),
        },
        fixed_artifact_tier: Some(tier.to_owned()),
    })
}

/// Model ids the MLX Krea strict-pose control route accepts (the deployed base the overlay applies on).
fn is_krea_control_model(model: &str) -> bool {
    model == "krea_2_turbo"
}

/// Resolve the Krea 2 Turbo base the MLX control provider loads: the `SCENEWORKS_KREA_CONTROL_BASE` env →
/// an explicit `modelPath` (advanced or manifest) → the legacy dense `krea/Krea-2-Turbo` HF-cache snapshot
/// when separately cached → the installed `SceneWorks/krea-2-turbo-mlx` tier (resolved EXACTLY like the
/// txt2img lane). `None` ⇒ not present locally (the job is not MLX-control-runnable). The MLX twin of the
/// candle `resolve_krea_control_base` (sc-11727): un-gates the dense-only assumption so a current user (who
/// installs the tier, not the retired dense repo) has a base.
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
        .unwrap_or(KREA_CONTROL_MLX_REPO);
    // The manifest `repo` (or the packed Krea MLX default) resolved to a cached snapshot.
    // Two on-disk shapes share this arm: a LEGACY dense diffusers tree (`krea/Krea-2-Turbo` — `tokenizer/
    // text_encoder/ transformer/ vae/` AT THE ROOT), or the current turnkey `SceneWorks/krea-2-turbo-mlx`
    // (q8/q4/bf16 tier SUBDIRS, nothing loadable at the root). CRUCIAL divergence from the candle twin: the
    // MLX `KREA_CONTROL_MLX_REPO` default is the TIERED repo, whereas candle's step-3 default is the
    // dense `krea/Krea-2-Turbo`. So the common MLX case lands the tiered
    // turnkey here — and returning its root un-descended made `KreaText::from_snapshot` look for
    // `<root>/tokenizer/tokenizer.json` on the tier-less root and fail with "tokenizer: No such file or
    // directory" (sc-11853). Only a genuine dense tree (tokenizer at the root) is the base as-is; a turnkey
    // root MUST be descended into the `mlxQuantize`-selected tier.
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, repo) {
        if snapshot.join("tokenizer").is_dir() {
            // Legacy / bring-your-own dense diffusers base (tokenizer at the root) — keep existing
            // dense-install behavior byte-identical.
            return Ok(Some(snapshot));
        }
        // A tiered turnkey root — descend EXACTLY like the txt2img lane (`krea_model_subdir` honours
        // `advanced.mlxQuantize` and falls back to any downloaded tier). The packed tier auto-detects its
        // quant and the `Krea2Transformer` runs a true packed forward on the base (sc-11727). Gate on
        // `transformer/` so a partial download surfaces "base not installed" rather than half-loading.
        let tier = krea_model_subdir(&snapshot, request);
        if tier.join("transformer").is_dir() {
            return Ok(Some(tier));
        }
    }
    // Explicit fallback to the installed `SceneWorks/krea-2-turbo-mlx` tier when the manifest `repo` pointed
    // at an absent legacy dense repo (so the arm above resolved nothing). Same tier descent.
    if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, KREA_CONTROL_MLX_REPO) {
        let tier = krea_model_subdir(&root, request);
        if tier.join("transformer").is_dir() {
            return Ok(Some(tier));
        }
    }
    Ok(None)
}

/// True when this is an MLX-eligible Krea 2 strict-pose job: `krea_2_turbo` with a non-empty
/// `advanced.poses`, not edit mode, whose dense base resolves locally. The MLX mirror of
/// `krea_control_candle_available`; the overlay weights are NOT part of the gate (resolved on first use).
fn krea_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_krea_control_model(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_krea_control_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → default (8).
fn krea_control_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", KREA_CONTROL_DEFAULT_STEPS, 1..=50)
}

/// The (repo, filename) of the hosted MLX overlay — `advanced.controlWeights.{repo,filename}` overrides
/// (a not-yet-cached registered/hosted overlay the API passed through), else the default published MLX
/// beta overlay. Mirrors the candle `krea_control_overlay_repo_file`; the filename must be a plain
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
    let repo = pick("repo", KREA_CONTROL_OVERLAY_REPO);
    let file = safe_weight_filename(
        &pick("filename", KREA_CONTROL_OVERLAY_FILE),
        "advanced.controlWeights.filename",
    )?;
    trusted_control_weight_revision(request, KREA_CONTROL_ENGINE_ID, &repo, &file)?;
    Ok((repo, file))
}

/// Confine a payload-supplied `advanced.controlWeights.path` to an app-managed root (sc-11168 / F-006).
/// The API writes this key for a studio-trained / registered LOCAL overlay (B4/sc-10165), but the value
/// arrives untrusted across the LAN boundary (epic 4484), so — like every other on-disk model input — it
/// must resolve under the app data dir / HF hub cache (or a declared external root) via the house
/// `normalize_app_managed_model_path`; without this a crafted job could point the loader at any file on
/// disk (an arbitrary-file-read primitive). Returns `Ok(None)` when the payload carries no path, `Ok(Some)`
/// for a confined path (whether or not it exists — the caller checks `is_file`), and the same
/// `InvalidPayload` rejection as the sibling lanes for an out-of-root path. Mirrors the candle twin.
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

/// Resolve the MLX control-branch overlay the `KreaTurboControl` provider loads, downloading on first use.
/// Order (most specific wins): the `SCENEWORKS_CONTROLNET_KREA` env → an `advanced.controlWeights.path`
/// (a studio-trained / registered LOCAL overlay the API resolved, B4/sc-10165) → an
/// `advanced.controlWeights.{repo,filename}` hosted override / the default published MLX overlay, fetched
/// into the app cache. The MLX twin of the candle `ensure_krea_control_weights`.
async fn ensure_krea_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<(PathBuf, bool)> {
    if let Ok(p) = std::env::var(KREA_CONTROL_WEIGHTS_ENV) {
        let p = PathBuf::from(p.trim());
        if p.is_file() {
            return Ok((p, false));
        }
    }
    if let Some(p) = krea_control_payload_overlay_path(settings, request)? {
        if p.is_file() {
            return Ok((p, false));
        }
    }
    let (repo, file) = krea_control_overlay_repo_file(request)?;
    let revision =
        trusted_control_weight_revision(request, KREA_CONTROL_ENGINE_ID, &repo, &file)?;
    let verified_default_overlay = repo == KREA_CONTROL_OVERLAY_REPO
        && revision == KREA_CONTROL_OVERLAY_PIN
        && file == KREA_CONTROL_OVERLAY_FILE;
    if let Some(snapshot) =
        crate::model_jobs::huggingface_pinned_snapshot_dir(&settings.data_dir, &repo, &revision)
    {
        let f = snapshot.join(&file);
        if f.is_file() {
            return Ok((f, verified_default_overlay));
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
    // checkpoint (parity with the candle lane). Registered overlays carry their own immutable pin.
    let path = ensure_hf_cached_file(&context, &repo, &revision, &file, &dst).await?;
    Ok((path, verified_default_overlay))
}

/// Flat telemetry recorded on MLX Krea control assets.
fn krea_control_raw_settings(
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
    // Krea 2 Turbo is CFG-free (distilled) — no guidance.
    raw.insert("guidanceScale".to_owned(), Value::Null);
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(KREA_CONTROL_ENGINE_ID.to_owned()),
    );
    // User LoRA labels applied on top of the pose control branch (mlx-gen sc-11720) — mirrors the candle
    // twin's `loras` field so the control-lane asset records what rode alongside the pose lock. Omitted
    // when no LoRA was requested (the sc-4408 omit-when-absent contract).
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

/// Load the MLX Krea pose-control generator: the base tier subdir + the control overlay (+ quant +
/// user adapters). The base runs at the `mlxQuantize`-selected tier (mlx-gen sc-11730): a pre-packed q4/q8
/// tier subdir loads packed and the matching `quant` is a no-op (`load_time_quant_bits` detects
/// already-packed), while a dense bf16 subdir with `quant` set quantizes the base DiT/TE at load.
/// Activation precision stays bf16 (the control provider requires it; weight packing is orthogonal) and the
/// pose overlay stays bf16. User LoRA/LoKr adapters ride additively on the frozen base DiT (mlx-gen
/// sc-11720): `spec.adapters` install BEFORE the optional quantize (mlx-gen `load_control_heavy`), so the
/// residual stacks over the possibly-already-packed base; the pose control branch is never an adapter
/// target. CFG-free, no identity img2img-init.
fn krea_control_spec(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::File(control_weights));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    spec
}

/// Generate one strict-pose image: the pre-built `conditioning` (the required pose `Control`) drives the
/// Krea control branch on the single CFG-free Turbo forward. No guidance / negative (distilled Turbo).
#[allow(clippy::too_many_arguments)]
fn krea_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    conditioning: Vec<Conditioning>,
    text_style_gain: Option<f32>,
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    memory_evaluation: Option<&crate::mlx_fit_gate::MlxRequestEvaluation>,
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let mut request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        conditioning,
        text_style_gain,
        preview,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = crate::memory_strategy::generate_with_scope(
        generator,
        &mut request,
        memory_evaluation.map(|evaluation| &evaluation.context),
        on_progress,
    )
    .map_err(|error| WorkerError::Engine(format!("Krea control generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("Krea control generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "Krea control generator returned non-image output".to_owned(),
        )),
    }
}

/// Real MLX Krea 2 strict-pose generation: one image per pose, each conditioned on a full DWPose skeleton
/// via the trained control-branch overlay on the frozen Turbo base (sc-8465; engine = mlx-gen
/// `krea_2_turbo_control`). The MLX twin of `generate_candle_krea_control_stream`; mirrors
/// `generate_zimage_control_stream`'s blocking-thread + streamed-events shape minus the identity
/// img2img-init and quant (Krea control is CFG-free dense bf16, pose-only). `control_scale = 0` is
/// byte-identical to base.
async fn generate_krea_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let weights_dir = resolve_krea_control_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Krea 2 Turbo base (krea/Krea-2-Turbo) weights not found".to_owned(),
        )
    })?;
    let (control_weights, verified_default_overlay) =
        ensure_krea_control_weights(api, settings, job, request).await?;
    // User LoRA/LoKr adapters ride additively on the frozen base DiT (mlx-gen sc-11720, the MLX twin of the
    // candle sc-11721 wiring): resolved + path-confined by the shared helper (enforces MAX_JOB_LORAS +
    // `normalize_app_managed_lora_path`), then installed on the base at load — the pose control branch is
    // never adapted. A character/style adapter reshapes the subject while the control branch keeps the pose
    // lock. Empty ⇒ stock control.
    let adapters = resolve_adapters(request, settings)?;

    let steps = krea_control_steps(request);
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
        .unwrap_or(KREA_CONTROL_MLX_REPO)
        .to_owned();

    // Shared strict-control driver: validate the requested ControlKind against the engine's
    // supported_kinds (krea_2_turbo_control = {Pose}) and resolve an optional user-supplied control-map
    // passthrough. A pose-only job sets no `controlMode`, so `kind == Pose` and the skeleton preprocessor
    // runs. Krea has no canny/depth tier, so `validate_control_kind` rejects anything but Pose.
    let control_kind = requested_control_kind(request)?;
    validate_control_kind(KREA_CONTROL_ENGINE_ID, &control_kind)?;
    let user_control = resolve_user_control_map(request, settings, project_path)?;
    let control_source = resolve_control_source(request, settings, project_path)?;

    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings = krea_control_raw_settings(request, &repo, steps, control_scale, count);
    // Strict pose shares one seed across the set so noise-derived attributes (hair, wardrobe, lighting)
    // stay constant while only the pose changes.
    let seed = resolve_seed(request, 0);

    // Identity-likeness scoring (epic 4406): a strict-control pose set is a Character-Studio pose-library
    // job; when it carries a character identity `referenceAssetId`, score every finished pose against that
    // source identity through the SHARED generator-agnostic seam (the z-image / candle Krea parity). All
    // non-fatal: a missing reference / staging failure → no scorer → scores omitted, the set still renders.
    let likeness_source = resolve_control_identity_source(request, settings, project_path);
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        likeness_source.is_some(),
        "pose-set face-stack staging failed; likeness scores omitted",
    )
    .await;

    let prompt = request.prompt.clone();
    // Krea "text style" tap-reweight gain (sc-12009) — self-gates on `ui.textStyleGain` (Krea only),
    // applied to the pose-control lane's CFG-free context by the engine (inference sc-12009).
    let text_style_gain = resolve_text_style_gain(request);
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    // The base runs at the `mlxQuantize`-selected tier (sc-11730); the pose overlay rides it bf16. User
    // LoRA/LoKr adapters (resolved above) install additively on the base DiT (mlx-gen sc-11720).
    let (quant, _quant_bits) = resolve_quant(request, Some(&weights_dir));
    let adapter_count = adapters.len();
    let spec = krea_control_spec(weights_dir, control_weights, quant, adapters);
    let calibration_provenance = krea_control_calibration_provenance(
        match &spec.weights {
            WeightsSource::Dir(path) => path,
            WeightsSource::File(_) => unreachable!("Krea control base is a directory"),
        },
        match spec.control.as_ref() {
            Some(WeightsSource::File(path)) => path,
            _ => unreachable!("Krea control overlay is a file"),
        },
        verified_default_overlay,
    );
    let memory_plan = crate::mlx_fit_gate::MlxRequestPlan::for_spec_and_manifest(
        KREA_CONTROL_ENGINE_ID,
        &request.model,
        &spec,
        Some(&request.model_manifest_entry),
        calibration_provenance,
    );
    let memory_inputs = krea_control_memory_inputs(width, height, &request.mode, adapter_count);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        KREA_CONTROL_ENGINE_ID,
        0,
        spec,
        "Krea control load failed".to_owned(),
        move |generator, tx, cancel| {
            let user_control = user_control.as_ref();
            let control_source = control_source.as_ref();
            // Build the per-job identity-likeness scorer ONCE on the generator-worker thread (the `!Send`
            // face stack lives here); the source identity is embedded once and reused across every pose.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            let mut cache_state = gen_core::MemoryCacheState::Cold;
            drive_gen_items_scored(tx, poses, move |_index, pose, preview, on_progress| {
                let control = preprocess_control_entry(
                    &control_kind,
                    user_control,
                    Some(&pose),
                    control_source,
                    width,
                    height,
                    stickwidth,
                    None,
                )?;
                // No identity img2img-init on the Krea control lane (pose renders from noise); the pose
                // `Control` is the only conditioning.
                let conditioning =
                    build_control_conditioning(control, control_kind.clone(), control_scale, None);
                let memory_evaluation = crate::mlx_fit_gate::evaluate_request(
                    generator,
                    &memory_plan,
                    &memory_inputs,
                    cache_state,
                    gen_core::OffloadPolicy::Resident,
                    0,
                )?;
                cache_state = gen_core::MemoryCacheState::Warm;
                let (out_w, out_h, pixels) = krea_control_generate_one(
                    generator,
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    conditioning,
                    text_style_gain,
                    preview,
                    &cancel,
                    on_progress,
                    Some(&memory_evaluation),
                )?;
                let face_likeness = scorer.as_ref().and_then(|scorer| {
                    crate::face_likeness::score_generated_image(
                        Some(scorer),
                        &Image {
                            width: out_w,
                            height: out_h,
                            pixels: pixels.clone(),
                        },
                        likeness_source_ref.as_deref(),
                    )
                });
                Ok(Some((seed, out_w, out_h, pixels, face_likeness)))
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
        KREA_CONTROL_ENGINE_ID,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
