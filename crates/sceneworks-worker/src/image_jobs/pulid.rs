// ---------------------------------------------------------------------------
// PuLID-FLUX native face-identity character image (macOS, sc-3344 / epic 3069):
// the `pulid_flux_dev` SceneWorks target on the native `mlx-gen-pulid` engine,
// retiring the torch `_vendor/pulid_flux` + `PuLIDFluxAdapter`.
//
// Unlike InstantID (a bespoke provider), PuLID-FLUX is an inventory-registered
// `Generator` (engine id `pulid_flux`), so this rides the SAME cached registry
// path as the base MLX families (`start_cached_gen_stream` → `crate::inference_runtime::load`).
// What's bespoke is only the request mapping (a reference face → an
// id-embedding via `Conditioning::Reference`, plus the PuLID-specific
// `idWeight` / `timestepToStartCfg` knobs) and the weight provisioning: the
// engine resolves the PuLID adapter + EVA tower + native face stack through its
// env-var seam (`PULID_FLUX_WEIGHTS` / `PULID_EVA_WEIGHTS` /
// `PULID_FACE_WEIGHTS_DIR`), which the worker fills from its cache here.
// ---------------------------------------------------------------------------

/// SceneWorks model id for native PuLID-FLUX (FLUX.1-dev backbone + PuLID injection).
const PULID_MODEL: &str = "pulid_flux_dev";
/// The mlx-gen registry id the worker loads through `crate::inference_runtime::load`.
const PULID_ENGINE_ID: &str = "pulid_flux";
/// FLUX.1-dev backbone repo for the MLX path (sc-9947): the SAME ungated `SceneWorks/flux1-dev-mlx`
/// quant-matrix turnkey the base `flux_dev` built-in consumes — self-contained q4/q8/bf16 tier subdirs,
/// packed-detected by `runtime_macos::providers::flux::load_flux1` (which `mlx-gen-pulid` delegates the backbone to). This
/// de-gates PuLID on macOS (no HF token / license-accept, the sc-8669 `flux_dev` precedent) and drops the
/// install-time convert peak (the packed tier loads directly). The candle (Windows/Linux) lane keeps the
/// upstream gated dense BFL layout via its own `PULID_CANDLE_FLUX_REPO` const — its packed consumption is a
/// separate epic-9083 slice — so this const is the MLX-only lever, NOT a shared manifest `repo` (which both
/// resolvers would read).
const PULID_FLUX_REPO: &str = "SceneWorks/flux1-dev-mlx";
/// The PuLID-FLUX adapter checkpoint (IDFormer + PerceiverAttention CA blocks). Public repo,
/// downloaded directly (the torch path used the same `guozinan/PuLID` / `v0.9.1` weight).
const PULID_ADAPTER_REPO: &str = "guozinan/PuLID";
const PULID_ADAPTER_FILE: &str = "pulid_flux_v0.9.1.safetensors";
/// Converted-weights bundle (download-on-first-use): the EVA02-CLIP-L-336 tower
/// (`tools/convert_eva_clip.py`) + the BiSeNet face-parsing net (`bisenet_parsing.safetensors`)
/// the native face stack needs for `face_features_image`. Public repo, mirroring the
/// `SceneWorks/instantid-mlx` upload InstantID uses (sc-3633 / sc-3707).
///
/// PROVISIONING NOTE (sc-3344 / sc-5045): this repo holds the two converted safetensors named
/// below — `eva02_clip_l_336.safetensors` (f16, from `tools/convert_eva_clip.py`) +
/// `bisenet_parsing.safetensors`. PUBLISHED (sc-5045) + validated end-to-end (fresh HF download →
/// worker-path ArcFace identity 0.6815). The `SCENEWORKS_PULID_WEIGHTS` / explicit-env overrides
/// below still let an operator (or the parity gate) point at a pre-staged dir.
const PULID_MLX_REPO: &str = "SceneWorks/pulid-flux-mlx";
const PULID_EVA_FILE: &str = "eva02_clip_l_336.safetensors";
const PULID_BISENET_FILE: &str = "bisenet_parsing.safetensors";

/// Torch-parity defaults (the `pulid_flux_dev` MODEL_TARGETS + the sc-2012 "photoreal" preset):
/// 30 steps at guidance 4.0, id_weight 1.0, timestep_to_start_cfg 4.
const PULID_DEFAULT_STEPS: u32 = 30;
const PULID_DEFAULT_GUIDANCE: f32 = 4.0;
const PULID_DEFAULT_ID_WEIGHT: f32 = 1.0;
const PULID_DEFAULT_TIMESTEP_TO_START_CFG: u32 = 4;
/// T5 sequence length the torch path used; recorded for telemetry parity (the native FLUX
/// backbone owns its own T5 length — this is not a request knob).
const PULID_MAX_SEQUENCE_LENGTH: u32 = 128;
/// The adapter label recorded on every PuLID-FLUX asset (parity with the torch `pulidFlux`
/// recipe; mirrors `mlx_instantid` for the InstantID path).
const PULID_ADAPTER_LABEL: &str = "mlx_pulid_flux";

/// Resolve the FLUX.1-dev backbone snapshot for PuLID-FLUX: an explicit `modelPath` dir wins (a
/// pre-staged complete FLUX dir — used as-is, never tier-resolved), else the HF cache snapshot for the
/// manifest `repo` (default `SceneWorks/flux1-dev-mlx`, sc-9947). For the quant-matrix turnkey that root
/// holds `q4/`/`q8/`/`bf16/` tier subdirs, so pick the SELECTED tier via `standard_tier_subdir` — the SAME
/// resolver the base `flux_dev` MLX lane uses — exactly as `runtime_macos::providers::flux::load_flux1` (which
/// `mlx-gen-pulid` delegates the backbone to) packed-loads it. `None` means the base is not present, so the
/// job is not MLX-runnable. Mirrors `resolve_instantid_sdxl_base` + `base::snapshot_dir_for_request`.
fn resolve_pulid_flux_base(
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
        return resolve_app_managed_model_dir(settings, &path, "PuLID-FLUX modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(PULID_FLUX_REPO);
    let root = huggingface_snapshot_dir(&settings.data_dir, repo);
    // Quant-matrix turnkey (sc-9947): descend into the selected q4/q8/bf16 tier subdir. A non-turnkey
    // `repo`/root (no tier subdirs) falls through to the root untouched inside `standard_tier_subdir`.
    Ok(root.map(|root| {
        if uses_standard_tier_layout(request) {
            standard_tier_subdir(&root, request)
        } else {
            root
        }
    }))
}

/// True when this is a native-MLX-eligible PuLID-FLUX job: the production model in
/// `character_image` mode with a reference face whose FLUX.1-dev base resolves locally. Mirrors
/// `jobs_store::pulid_flux_mlx_eligible` so the router and worker agree (and mirrors the shape of
/// `instantid_available`). PuLID-FLUX is text-to-image-with-a-face only — no `edit_image`.
fn pulid_flux_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == PULID_MODEL
        && request.mode == "character_image"
        && non_empty(&request.reference_asset_id)
        && matches!(resolve_pulid_flux_base(request, settings), Ok(Some(_)))
}

/// Resolve PuLID denoise steps: `advanced.steps` (clamped 1..=80) → manifest `steps` → 30.
fn pulid_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", PULID_DEFAULT_STEPS, 1..=80)
}

/// Resolve PuLID guidance: `advanced.guidanceScale` → manifest `guidanceScale` → 4.0 (the FLUX.1-dev
/// guidance-distilled CFG; the engine's fake-CFG single forward consumes it).
fn pulid_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(request, "guidanceScale", PULID_DEFAULT_GUIDANCE, 0.0..=30.0)
}

/// The PuLID identity-strength knob → the reference conditioning's `strength` (the engine reads
/// it as `id_weight`). Torch clamp band 0.0–3.0 (the upstream gradio slider), default 1.0.
fn pulid_id_weight(request: &ImageRequest) -> f32 {
    advanced::f32_clamped(
        &request.advanced,
        "idWeight",
        PULID_DEFAULT_ID_WEIGHT,
        0.0..=3.0,
    )
}

/// `timestep_to_start_cfg` (higher = identity injected later = more editable / weaker identity).
/// Torch clamp 0..=20, default 4 ("photoreal"). NOTE: this gates the engine's real-CFG branch,
/// which the production path does NOT engage (fake-CFG, `true_cfg=1.0`), so it is currently a
/// no-op on output — but it is forwarded + recorded for parity and forward-compat with the
/// real-CFG path (sc-3075).
fn pulid_timestep_to_start_cfg(request: &ImageRequest) -> u32 {
    advanced::u32_clamped(
        &request.advanced,
        "timestepToStartCfg",
        PULID_DEFAULT_TIMESTEP_TO_START_CFG,
        0..=20,
    )
}

/// The resolved engine weight inputs: the three files/dirs the engine reads from its env-var seam.
struct PulidWeights {
    /// `pulid_flux_v0.9.1.safetensors` (the IDFormer + PerceiverAttention CA blocks).
    adapter: PathBuf,
    /// The converted EVA02-CLIP-L-336 safetensors.
    eva: PathBuf,
    /// A dir holding `scrfd_10g.safetensors` + `arcface_iresnet100.safetensors` +
    /// `bisenet_parsing.safetensors` (the native face stack `mlx-gen-face` loads).
    face_dir: PathBuf,
}

/// An operator-/parity-gate-supplied weight preset: if all three engine env vars are ALREADY set
/// and exist, the worker passes them through unchanged (no cache provisioning). This is the
/// pre-staged-weights escape hatch used by the e2e parity gate and lower-level validation, and it
/// keeps the worker functional before the `SceneWorks/pulid-flux-mlx` HF upload lands.
fn pulid_weights_env_preset() -> Option<PulidWeights> {
    let adapter = PathBuf::from(std::env::var_os("PULID_FLUX_WEIGHTS")?);
    let eva = PathBuf::from(std::env::var_os("PULID_EVA_WEIGHTS")?);
    let face_dir = PathBuf::from(std::env::var_os("PULID_FACE_WEIGHTS_DIR")?);
    (adapter.exists() && eva.exists() && face_dir.is_dir()).then_some(PulidWeights {
        adapter,
        eva,
        face_dir,
    })
}

/// Build the engine's identity-weight seam from the resolved cache paths (sc-8827). The `pulid_flux`
/// loader reads the PuLID adapter + EVA tower + native face stack from `LoadSpec::identity`, so the
/// worker threads the paths through the spec instead of mutating the process-global `PULID_*` env vars
/// at job time on the multithreaded tokio runtime (the old `set_var`/`remove_var` seam was unsound —
/// F-025). The paths travel with the `LoadSpec` into the spawned load task, no env involved.
fn pulid_identity_weights(weights: &PulidWeights) -> IdentityWeights {
    IdentityWeights {
        encoder: Some(WeightsSource::File(weights.adapter.clone())),
        eva: Some(WeightsSource::File(weights.eva.clone())),
        face_dir: Some(WeightsSource::Dir(weights.face_dir.clone())),
    }
}

/// Resolve all PuLID-FLUX engine weight inputs, downloading the converted bundle + the PuLID
/// adapter + the shared face stack on first use into ONE bundle dir (so it doubles as the engine's
/// `PULID_FACE_WEIGHTS_DIR`). Resolution order: a fully-set env preset → a `SCENEWORKS_PULID_WEIGHTS`
/// pre-staged dir → the app cache (`cache/pulid-flux-mlx`, downloading missing files). The face
/// detector + ArcFace embedder are the SAME converted files InstantID ships (reused from
/// `SceneWorks/instantid-mlx`); only EVA + BiSeNet + the PuLID adapter are PuLID-specific.
async fn ensure_pulid_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PulidWeights> {
    if let Some(preset) = pulid_weights_env_preset() {
        return Ok(preset);
    }
    let client = crate::downloads::streaming_download_client();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "PuLID-FLUX generation canceled while fetching weights.",
        fresh_download: false,
    };
    // One bundle dir holds every loose file (it IS the engine's PULID_FACE_WEIGHTS_DIR). An env
    // override (pre-staged for local validation) wins, else the app cache.
    let bundle = std::env::var("SCENEWORKS_PULID_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("pulid-flux-mlx"));

    // PuLID adapter (public guozinan/PuLID) + EVA/BiSeNet (SceneWorks bundle).
    let adapter =
        ensure_instantid_file(&context, PULID_ADAPTER_REPO, &bundle, PULID_ADAPTER_FILE).await?;
    let eva = ensure_instantid_file(&context, PULID_MLX_REPO, &bundle, PULID_EVA_FILE).await?;
    ensure_instantid_file(&context, PULID_MLX_REPO, &bundle, PULID_BISENET_FILE).await?;
    // Face detector + ArcFace embedder: reuse the InstantID converted bundle (same files), placed
    // into the SAME dir so PULID_FACE_WEIGHTS_DIR sees all three face inputs together.
    ensure_instantid_file(&context, INSTANTID_MLX_REPO, &bundle, INSTANTID_SCRFD_FILE).await?;
    ensure_instantid_file(&context, INSTANTID_MLX_REPO, &bundle, INSTANTID_ARCFACE_FILE).await?;

    Ok(PulidWeights {
        adapter,
        eva,
        face_dir: bundle,
    })
}

/// Flat telemetry recorded on PuLID-FLUX assets (parity with the torch `PuLIDFluxAdapter`
/// `raw_settings` recipe keys).
fn pulid_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: f32,
    id_weight: f32,
    timestep_to_start_cfg: u32,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("pulidFlux".to_owned(), Value::Bool(true));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("idWeight".to_owned(), json!(id_weight));
    raw.insert("timestepToStartCfg".to_owned(), json!(timestep_to_start_cfg));
    raw.insert("maxSequenceLength".to_owned(), json!(PULID_MAX_SEQUENCE_LENGTH));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert(
        "pulidFluxEngine".to_owned(),
        Value::String("mlx_pulid_flux".to_owned()),
    );
    raw
}

/// Real PuLID-FLUX generation: resolve the reference + weights on the async side, feed the engine's
/// env-var weight seam from the worker cache, then load the registry `pulid_flux` generator once
/// (cached) + generate each image on the blocking thread. Each image is a single-identity render at
/// the requested W×H from the reference face; `idWeight` rides the `Reference` conditioning's
/// strength and `timestepToStartCfg` rides `GenerationRequest.timestep_to_start_cfg`. Reuses the
/// shared streaming seam (`consume_gen_events`) so step/cancel/asset behavior matches every other
/// MLX family.
async fn generate_pulid_flux_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let flux_base = resolve_pulid_flux_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("PuLID-FLUX base (FLUX.1-dev) not found".to_owned())
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("PuLID-FLUX requires a reference face image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;

    let weights = ensure_pulid_weights(api, settings, job).await?;
    // Feed the engine's weight seam from the resolved cache paths on the `LoadSpec` (the `pulid_flux`
    // loader resolves the PuLID adapter + EVA + face stack from `LoadSpec::identity`, sc-8827). The
    // paths ride the spec into the spawned load task — no process-global env mutation (the old
    // `PULID_*` `set_var`/`remove_var` seam was unsound on the multithreaded runtime, F-025).
    let identity = pulid_identity_weights(&weights);

    let steps = pulid_steps(request);
    let guidance = pulid_guidance(request);
    let id_weight = pulid_id_weight(request);
    let start_cfg = pulid_timestep_to_start_cfg(request);
    // Quant-matrix backbone (sc-9947): the FLUX.1-dev backbone always supports quant, so resolve the
    // request quant the SAME way the base `flux_dev` MLX lane does and reconcile it against the tier
    // subdir actually resolved (`flux_base`) — record the precision that ran + emit `quant_tier_downgraded`
    // on a genuine fallback (requested tier absent). On an already-packed q4/q8 tier the load quant is a
    // harmless no-op; the bf16 tier resolves to `None`. The PuLID conditioning (EVA/IDFormer/CA) stays f32
    // in every case — `load_flux1` quantizes only the backbone linears (sc-3076).
    let (quant, recipe_bits) = reconcile_resolved_tier_quant(
        resolve_quant(request, Some(&flux_base)),
        &flux_base,
        true,
        &request.model,
        &job.id,
        backend,
    );
    // Curated unified-sampler selection (epic 7114, sc-7432): PuLID-FLUX delegates its denoise to the
    // FLUX backbone, which honors a curated solver/scheduler on the `GenerationRequest` (#537). Read +
    // N3-normalize against the shared curated menu (an unknown name drops to the engine default + emits
    // an event). N1: unset ⇒ `None` ⇒ the native flow-match default loop runs byte-exact.
    let (curated_samplers, curated_schedulers) = curated_image_menu();
    let (sampler, scheduler, scheduler_shift) = read_advanced_sampling_knobs(&request.advanced);
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
        .unwrap_or(PULID_FLUX_REPO)
        .to_owned();

    let raw_settings = pulid_raw_settings(
        request,
        &repo,
        steps,
        recipe_bits,
        guidance,
        id_weight,
        start_cfg,
    );

    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count).map(|index| resolve_seed(request, index)).collect();
    let (width, height) = (request.width, request.height);
    let prompt = request.prompt.clone();

    // Identity-likeness scoring (epic 4406, sc-4411 plain With-Character): PuLID-FLUX serves only the
    // single-identity `character_image` path (one identity image per seed, no angle/pose grouping), so
    // it is always a With-Character generation — score every output against the reference face through
    // the SHARED generator-agnostic seam (the same `build_face_likeness_scorer` InstantID / FLUX.2 /
    // Qwen / SenseNova use). Stage the antelopev2 face stack (shared bundle, no-op if cached); the
    // `!Send` scorer is built ONCE inside the generator-worker closure and reused across the N outputs
    // (source embedded once — the caching AC). The source is the CURRENT job's `referenceAssetId`, so
    // changing the reference changes the scored source. Staging is non-fatal (failure → no scorer →
    // scores omitted, generation still renders).
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        true,
        "PuLID-FLUX face-stack staging failed; likeness scores omitted",
    )
    .await;
    let likeness_source = face_stack_dir.as_ref().map(|_| reference.clone());
    let likeness_source_ref = reference_id.to_owned();

    let mut spec = load_spec(flux_base, quant, Vec::new(), None);
    // PuLID identity sub-model paths ride the spec (sc-8827) — the `pulid_flux` loader reads them from
    // `LoadSpec::identity` instead of the process-global `PULID_*` env vars.
    spec.identity = Some(identity);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        PULID_ENGINE_ID,
        0,
        spec,
        format!("{PULID_ENGINE_ID} load failed"),
        move |generator, tx, cancel| {
            // Per-job identity-likeness scorer built ONCE on the generator-worker thread (the `!Send`
            // face stack lives here); source embedded once, reused across every output (sc-4411).
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some(source)) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            drive_gen_items_scored(tx, seeds, move |_index, seed, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                // One reference face per image (the engine consumes it into the id-embedding +
                // CA injector). idWeight → strength; timestepToStartCfg → timestep_to_start_cfg.
                let req = GenerationRequest {
                    prompt: prompt.clone(),
                    negative_prompt: None,
                    width,
                    height,
                    count: 1,
                    seed: Some(seed as u64),
                    steps: Some(steps),
                    guidance: Some(guidance),
                    true_cfg: None,
                    timestep_to_start_cfg: Some(start_cfg),
                    sampler: sampler.clone(),
                    scheduler: scheduler.clone(),
                    scheduler_shift,
                    conditioning: vec![Conditioning::Reference {
                        image: reference.clone(),
                        strength: Some(id_weight),
                    }],
                    cancel: cancel.clone(),
                    ..Default::default()
                };
                let output = match generator.generate(&req, on_progress) {
                    Ok(output) => output,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "PuLID-FLUX generation failed: {error}"
                        )))
                    }
                };
                match output {
                    GenerationOutput::Images(mut images) => {
                        let image = images.pop().ok_or_else(|| {
                            WorkerError::Engine("PuLID-FLUX produced no image".to_owned())
                        })?;
                        // Score this finished image against the cached source embedding (sc-4411). The
                        // Image build + pixel clone is paid ONLY when a scorer exists; a non-frontal /
                        // no-face result records an honest detected:false N/A, `None` scorer ⇒ field
                        // omitted.
                        let face_likeness = scorer.as_ref().and_then(|scorer| {
                            crate::face_likeness::score_generated_image(
                                Some(scorer),
                                &Image {
                                    width: image.width,
                                    height: image.height,
                                    pixels: image.pixels.clone(),
                                },
                                Some(likeness_source_ref.as_str()),
                            )
                        });
                        Ok(Some((seed, image.width, image.height, image.pixels, face_likeness)))
                    }
                    _ => Err(WorkerError::Engine(
                        "PuLID-FLUX returned non-image output".to_owned(),
                    )),
                }
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
        PULID_ADAPTER_LABEL,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
