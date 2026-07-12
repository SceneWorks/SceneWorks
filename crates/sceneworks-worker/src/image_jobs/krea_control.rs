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
// `Conditioning::Control` per pose. Krea 2 Turbo needs the DENSE `krea/Krea-2-Turbo` snapshot (the
// composable-forward overlay was trained on the bf16 base), NOT the packed q8 turnkey (`krea-2-turbo-mlx`)
// the plain `krea_2_turbo` txt2img lane loads — the same base the candle lane uses.

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
/// `KREA_CONTROL_OVERLAY_REVISION` — a repo re-push can't swap the checkpoint under us). Applied ONLY to
/// the default repo; a `controlWeights.repo` override keeps `main`.
const KREA_CONTROL_OVERLAY_REVISION: &str = "cb3a0ac7590f5ec594a4eeb43b95ee1da0b5a0ac";

/// Model ids the MLX Krea strict-pose control route accepts (the deployed base the overlay applies on).
fn is_krea_control_model(model: &str) -> bool {
    model == "krea_2_turbo"
}

/// Resolve the Krea 2 Turbo dense snapshot the MLX control provider loads: the
/// `SCENEWORKS_KREA_CONTROL_BASE` env → an explicit `modelPath` (advanced or manifest) → the HF-cache
/// snapshot for the manifest `repo` (default = the shared strict-control table's `krea_2_turbo_control`
/// base repo, `krea/Krea-2-Turbo`). `None` ⇒ not present locally (the job is not MLX-control-runnable).
/// The MLX twin of the candle `resolve_krea_control_base`.
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
        .unwrap_or_else(|| strict_control_default_repo(KREA_CONTROL_ENGINE_ID));
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
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
    Ok((
        pick("repo", KREA_CONTROL_OVERLAY_REPO),
        safe_weight_filename(
            &pick("filename", KREA_CONTROL_OVERLAY_FILE),
            "advanced.controlWeights.filename",
        )?,
    ))
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
) -> WorkerResult<PathBuf> {
    if let Ok(p) = std::env::var(KREA_CONTROL_WEIGHTS_ENV) {
        let p = PathBuf::from(p.trim());
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Some(path) = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object)
        .and_then(|cw| cw.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let p = PathBuf::from(path);
        if p.is_file() {
            return Ok(p);
        }
    }
    let (repo, file) = krea_control_overlay_repo_file(request)?;
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
        cancel_message: "Krea 2 strict-pose generation canceled while fetching control overlay.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-krea")
        .join(&file);
    // Pin the exact commit for the default overlay repo so `main` moving under us can't swap the
    // checkpoint (parity with the candle lane); a `controlWeights.repo` override carries its own layout.
    let revision = if repo == KREA_CONTROL_OVERLAY_REPO {
        KREA_CONTROL_OVERLAY_REVISION
    } else {
        "main"
    };
    ensure_hf_cached_file(&context, &repo, revision, &file, &dst).await?;
    Ok(dst)
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
    raw
}

/// Load the MLX Krea pose-control generator: the dense base snapshot + the control overlay. Krea control
/// is CFG-free dense bf16 — no quant, no adapters (the candle-lane parity), no identity img2img-init.
fn krea_control_spec(weights_dir: PathBuf, control_weights: PathBuf) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(weights_dir)).with_control(WeightsSource::File(control_weights))
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
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
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
    let control_weights = ensure_krea_control_weights(api, settings, job, request).await?;

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
        .unwrap_or_else(|| strict_control_default_repo(KREA_CONTROL_ENGINE_ID))
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
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    // Dense bf16, no adapters — the control overlay rides the frozen base (parity with the candle lane).
    let spec = krea_control_spec(weights_dir, control_weights);
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
            drive_gen_items_scored(tx, poses, move |_index, pose, on_progress| {
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
                let (out_w, out_h, pixels) = krea_control_generate_one(
                    generator,
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    conditioning,
                    &cancel,
                    on_progress,
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
