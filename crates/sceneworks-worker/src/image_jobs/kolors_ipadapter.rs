use super::{
    admit_conditioning_paths, candle_artifact_path_matches, candle_certified_hf_artifact_path,
    candle_resolved_tier_key, consume_gen_events, curated_image_menu, drive_gen_items_scored,
    load_reference_image, non_empty, normalize_sampling_knob, read_advanced_sampling_knobs,
    resolve_adapters, resolve_advanced_or_manifest_f32, resolve_advanced_or_manifest_u32,
    resolve_character_image_likeness_source, resolve_seed, stage_likeness, start_gen_stream,
    ApiClient, Image, ImagePlan, ImageRequest, IpAdapterKolors, IpAdapterKolorsPaths,
    IpAdapterKolorsRequest, JobSnapshot, JsonObject, Path, PathBuf, Settings, Value, WorkerError,
    WorkerResult,
};
use super::{advanced, ensure_hf_cached_file};
use super::{resolve_app_managed_model_dir, standard_tier_subdir, DownloadContext};
use serde_json::json;

// Candle (Windows/CUDA) Kolors IP-Adapter-Plus reference route (sc-5488, epic 5480) — reference-image
// (identity) conditioning on Kolors off-Mac via `runtime_cuda::providers::kolors::IpAdapterKolors`. The Kolors sibling
// of the candle SDXL IP-Adapter lane (sdxl_ipadapter.rs): CLIP ViT-L/14-336 image tokens injected into
// the vendored SDXL UNet alongside the encoder_hid_proj-projected ChatGLM3 text path, denoised with the
// Kolors leading-Euler sampler.
//
// **Candle-only.** macOS keeps the MLX Kolors IP path (the `Reference` conditioning the registry
// `kolors` generator handles once `with_ip_adapter` installs the K/V — kolors.rs, sc-4767); the candle
// `IpAdapterKolors` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build
// (the module declaration in image_jobs.rs carries the cfg). It is a child module of the `image_jobs` module, so
// it shares that module's imports (ImageRequest/Settings/WorkerResult/`advanced`/`load_reference_image`/
// `huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).

/// The Kolors IP-Adapter-Plus repo (CLIP ViT-L/14-336 `image_encoder/` + the `image_proj` Resampler +
/// `ip_adapter.*` K/V). Same repo the MLX path uses (`kolors.rs` `KOLORS_IP_ADAPTER_REPO`).
const KOLORS_IPADAPTER_REPO: &str = "Kwai-Kolors/Kolors-IP-Adapter-Plus";
/// The repo revision carrying the **safetensors**: the repo's `main` ships only `.bin` (a torch pickle
/// candle can't read); PR-4 adds `ip_adapter_plus_general.safetensors` + `image_encoder/model.safetensors`.
/// Pinned to the exact commit at the tip of `refs/pr/4` rather than the mutable `refs/pr/4` ref itself
/// (sc-11168 / F-007 — completes the sc-9879 rollout): a force-push to the PR could otherwise swap the
/// adapter/encoder weights we load. The native downloader still verifies each file's own hash on download.
pub(super) const KOLORS_IPADAPTER_REVISION: &str = "5c72aa86cd8d9d23ff406d293c5473820e09e1d9";
/// The IP-Adapter-Plus bundle file (root of the snapshot).
const KOLORS_IPADAPTER_BUNDLE: &str = "ip_adapter_plus_general.safetensors";
/// The CLIP ViT-L/14-336 image-encoder files inside the repo (config + weights).
const KOLORS_IPADAPTER_ENCODER_SRC: [&str; 2] = [
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
];
/// IP-Adapter scale default — the torch `KolorsDiffusersAdapter._ip_adapter_scale` 0.6 (matches the MLX
/// path's `KOLORS_IP_SCALE`, and the candle `IpAdapterKolors::DEFAULT_IP_ADAPTER_SCALE`).
const KOLORS_IPADAPTER_IP_SCALE: f32 = 0.6;
/// Denoise steps default (Kolors production — diffusers `KolorsPipeline`).
const KOLORS_IPADAPTER_DEFAULT_STEPS: u32 = 50;
/// CFG default (Kolors production guidance).
const KOLORS_IPADAPTER_DEFAULT_GUIDANCE: f32 = 5.0;
/// The Kolors base diffusers repo when the manifest omits `repo`.
const KOLORS_IPADAPTER_DEFAULT_REPO: &str = KOLORS_BASE_REPO;
const KOLORS_BASE_REPO: &str = "SceneWorks/kolors-mlx";
const KOLORS_BASE_REVISION: &str = "aadbd49f53b66a33ef1be09384eac409cbc44061";
/// The adapter/engine id recorded on candle Kolors IP-Adapter assets + telemetry (distinct from the
/// txt2img `candle_kolors` lane).
pub(super) const KOLORS_IPADAPTER_ENGINE: &str = "candle_kolors_ipadapter";

/// Model ids the candle Kolors IP-Adapter route accepts.
fn is_kolors_ipadapter_model(model: &str) -> bool {
    model == "kolors"
}

/// Resolve the Kolors base (diffusers) snapshot for the IP-Adapter route: an explicit `modelPath` dir
/// (advanced or manifest) wins, else the HF cache snapshot for the manifest `repo` (default
/// `Kwai-Kolors/Kolors-diffusers`). `None` means the base is not present locally, so the candle lane
/// refuses the job and no fallback is attempted. Mirrors `resolve_sdxl_ipadapter_base`.
fn resolve_kolors_ipadapter_base(
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
        return resolve_app_managed_model_dir(settings, &path, "Kolors IP-Adapter modelPath")
            .map(Some);
    }
    Ok(crate::model_jobs::huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        KOLORS_BASE_REPO,
        KOLORS_BASE_REVISION,
    )
    .map(|root| standard_tier_subdir(&root, request)))
}

/// True when this is a candle-eligible Kolors IP-Adapter job: the `kolors` model with a reference image
/// (and NOT an img2img/inpaint/edit shape — those are sc-5487) whose base resolves locally. Mirrors
/// `jobs_store::kolors_ipadapter_candle_eligible` so the worker and router agree.
pub(super) fn kolors_ipadapter_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_kolors_ipadapter_model(&request.model)
        && request.mode != "edit_image"
        && non_empty(&request.reference_asset_id)
        && !non_empty(&request.source_asset_id)
        && !non_empty(&request.mask_asset_id)
        && matches!(
            resolve_kolors_ipadapter_base(request, settings),
            Ok(Some(_))
        )
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=80) → manifest `steps` → default (50).
fn kolors_ipadapter_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", KOLORS_IPADAPTER_DEFAULT_STEPS, 1..=80)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (5.0), clamped.
fn kolors_ipadapter_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        KOLORS_IPADAPTER_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// Resolve the Kolors IP-Adapter-Plus snapshot **directory** (`image_encoder/` + the bundle file) the
/// `IpAdapterKolors` provider loads, downloading from the exact immutable IP-Adapter revision on first
/// use. Resolution order: that revision's HF snapshot → that revision's app cache. Mutable env/main/PR
/// sources are intentionally excluded from the production receipt boundary. Returns the directory
/// expected by [`IpAdapterKolorsPaths::ip_adapter`].
async fn ensure_kolors_ipadapter_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let has_layout = |dir: &Path| {
        dir.join(KOLORS_IPADAPTER_BUNDLE).is_file()
            && dir
                .join("image_encoder")
                .join("model.safetensors")
                .is_file()
    };
    // Only the immutable revision may authorize an optimized memory request. Mutable env/main
    // overrides formerly bypassed the receipt boundary and are deliberately no longer resolved.
    if let Some(snapshot) = crate::model_jobs::huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        KOLORS_IPADAPTER_REPO,
        KOLORS_IPADAPTER_REVISION,
    ) {
        if has_layout(&snapshot) {
            return Ok(snapshot);
        }
    }
    // Download-on-first-use into the app cache (the snapshot layout: bundle at root + image_encoder/).
    let client = crate::downloads::streaming_download_client();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "Kolors IP-Adapter generation canceled while fetching weights.",
        fresh_download: false,
    };
    let cache = settings
        .data_dir
        .join("cache")
        .join("ipadapter-kolors")
        .join(KOLORS_IPADAPTER_REVISION);
    ensure_hf_cached_file(
        &context,
        KOLORS_IPADAPTER_REPO,
        KOLORS_IPADAPTER_REVISION,
        KOLORS_IPADAPTER_BUNDLE,
        &cache.join(KOLORS_IPADAPTER_BUNDLE),
    )
    .await?;
    for source in KOLORS_IPADAPTER_ENCODER_SRC {
        let name = source.rsplit('/').next().unwrap_or(source);
        ensure_hf_cached_file(
            &context,
            KOLORS_IPADAPTER_REPO,
            KOLORS_IPADAPTER_REVISION,
            source,
            &cache.join("image_encoder").join(name),
        )
        .await?;
    }
    Ok(cache)
}

/// Flat telemetry recorded on candle Kolors IP-Adapter assets (parity with the SDXL/InstantID recipe-key
/// shape).
fn kolors_ipadapter_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    ip_scale: f32,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("ipAdapterScale".to_owned(), json!(ip_scale));
    raw.insert(
        "ipAdapterEngine".to_owned(),
        Value::String(KOLORS_IPADAPTER_ENGINE.to_owned()),
    );
    raw
}

/// Real candle Kolors IP-Adapter generation: resolve the reference + weights on the async side, then
/// load the `IpAdapterKolors` provider once + generate each image on the blocking thread. `request.count`
/// images, each its own seed; `generate` takes the per-job `CancelFlag` + a `Progress` callback, so
/// streaming is per-step and cancellation is honoured mid-denoise — same contract as the SDXL IP lane.
pub(super) async fn generate_candle_kolors_ipadapter_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let kolors_base = resolve_kolors_ipadapter_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Kolors IP-Adapter base (Kolors-diffusers) not found".to_owned(),
        )
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Kolors IP-Adapter requires a reference image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;
    let ip_adapter = ensure_kolors_ipadapter_weights(api, settings, job).await?;

    // Identity-likeness scoring (epic 4406, sc-4411 plain With-Character): the candle Kolors IP-Adapter
    // lane is the With-Character route for a Kolors model — score every output against the reference face
    // through the SHARED generator-agnostic seam. Eligibility goes through `resolve_character_image_
    // likeness_source` (the SAME gate the macOS lanes use), so the angle/pose/edit exclusion is explicit
    // and self-contained here — NOT dependent on dispatch order (an angle/pose job is excluded by the
    // helper even if it ever reached this lane, so it can never be double-scored). The helper's decode is
    // ignored: the already-decoded `reference` (this lane's generation input, the current job's
    // `referenceAssetId`) is the scorer source, so there is no second decode. Stage the antelopev2 SCRFD +
    // ArcFace bundle; the `!Send` scorer is built ONCE inside the load closure and reused across the N
    // outputs (source embedded once). Staging is non-fatal (failure → scores omitted).
    let score_likeness =
        resolve_character_image_likeness_source(request, settings, project_path).is_some();
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        score_likeness,
        "character_image face-stack staging failed; likeness scores omitted",
    )
    .await;
    let likeness_source = face_stack_dir.as_ref().map(|_| reference.clone());
    let likeness_source_ref = reference_id.to_owned();

    let steps = kolors_ipadapter_steps(request);
    let guidance = kolors_ipadapter_guidance(request);
    let ip_scale = advanced::f32_clamped(
        &request.advanced,
        "ipAdapterScale",
        KOLORS_IPADAPTER_IP_SCALE,
        0.0..=1.0,
    );
    // Curated unified-sampler selection (epic 7114, sc-7432): the candle `IpAdapterKolors` provider
    // honors a curated solver/scheduler via the shared `denoise_curated` primitive (#130). Read +
    // N3-normalize against the shared curated menu. N1: unset ⇒ `None` ⇒ the native default, byte-exact.
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
        .unwrap_or_else(|| {
            crate::engines::default_repo_for(&request.model)
                .unwrap_or(KOLORS_IPADAPTER_DEFAULT_REPO)
        })
        .to_owned();
    let mut raw_settings = kolors_ipadapter_raw_settings(request, &repo, steps, guidance, ip_scale);

    // Per-image work items: (seed, prompt) — `request.count` images at the reference identity.
    let (width, height) = (request.width, request.height);
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let negative_prompt = request.negative_prompt.clone();
    let adapters = resolve_adapters(request, settings)?;

    // SC-20790: the shared selector is authoritative before any heavyweight IP provider load.
    // Build its contract from the exact base, IP bundle/CLIP tower, and ordered adapter files that
    // the blocking closure receives; the provider recomputes the same contract at load.
    let contract_paths = IpAdapterKolorsPaths {
        kolors_base: kolors_base.clone(),
        ip_adapter: ip_adapter.clone(),
        adapters: adapters.clone(),
    };
    let tier = candle_resolved_tier_key(request, &kolors_base, false);
    let contract =
        runtime_cuda::providers::kolors::memory_strategy::provider_contract_for_ip(&contract_paths)
            .map_err(|error| {
                WorkerError::Engine(format!("Kolors IP memory contract failed: {error}"))
            })?;
    let overlay_receipt = crate::candle_memory_strategy::kolors_overlay_receipt_identity(
        KOLORS_IPADAPTER_ENGINE,
        &contract,
        false,
    )?
    .ok_or_else(|| {
        WorkerError::InvalidPayload("Kolors IP provider omitted its exact overlay receipt".into())
    })?;
    let strategy_spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(kolors_base.clone()))
        .with_adapters(adapters.clone())
        .with_ip_adapter(gen_core::WeightsSource::Dir(ip_adapter.clone()));
    let raw_budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    let predicted_peak = crate::vram_gate::predicted_peak_gb(&request.model_manifest_entry, tier);
    let pinned_ip = candle_certified_hf_artifact_path(
        settings,
        KOLORS_IPADAPTER_REPO,
        KOLORS_IPADAPTER_REVISION,
        Path::new("."),
        &ip_adapter,
    );
    let revision_cache = settings
        .data_dir
        .join("cache")
        .join("ipadapter-kolors")
        .join(KOLORS_IPADAPTER_REVISION);
    let artifact_is_certified = candle_certified_hf_artifact_path(
        settings,
        KOLORS_BASE_REPO,
        KOLORS_BASE_REVISION,
        Path::new(tier),
        &kolors_base,
    ) && (pinned_ip
        || candle_artifact_path_matches(&ip_adapter, &revision_cache));
    let memory_evaluation = crate::candle_memory_strategy::evaluate_shared_bespoke_image(
        KOLORS_IPADAPTER_ENGINE,
        "kolors",
        &strategy_spec,
        artifact_is_certified,
        &request.model_manifest_entry,
        tier,
        "character_image",
        Some("identity"),
        gen_core::MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 1,
        },
        true,
        false,
        false,
        raw_budget,
        raw_budget.map_or(0.0, crate::vram_gate::ladder_reserve_gb),
        predicted_peak,
        0,
        gen_core::MemoryCacheState::Cold,
        contract,
        crate::candle_memory_strategy::KOLORS_REQUEST_EVIDENCE_REVISION,
    )?
    .ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Kolors IP request has no exact authoritative memory candidate; refusing before load"
                .into(),
        )
    })?;
    raw_settings.insert(
        "memoryStrategy".to_owned(),
        Value::String(format!(
            "{:?}",
            memory_evaluation.context.selection.strategy
        )),
    );
    raw_settings.insert(
        "memoryEvidenceRevision".to_owned(),
        Value::String(memory_evaluation.context.evidence_revision.clone()),
    );
    raw_settings.insert("memoryReceipt".to_owned(), Value::String(overlay_receipt));
    tracing::info!(
        event = "image_memory_strategy_selected",
        route = KOLORS_IPADAPTER_ENGINE,
        actual_tier = tier,
        mode = "character_image",
        reference_count = 1,
        overlay = memory_evaluation.context.overlay.as_deref().unwrap_or("none"),
        strategy = ?memory_evaluation.context.selection.strategy,
        predicted_peak_bytes = memory_evaluation.context.predicted_peak_bytes,
        evidence_revision = %memory_evaluation.context.evidence_revision,
        "Kolors IP exact receipt-backed memory strategy selected"
    );
    crate::emit_event(
        "image_memory_strategy_selected",
        json!({
            "backend": "candle",
            "route": KOLORS_IPADAPTER_ENGINE,
            "actualTier": tier,
            "mode": "character_image",
            "referenceCount": 1,
            "geometry": { "width": width, "height": height, "batch": 1, "frames": 1 },
            "overlay": memory_evaluation.context.overlay.clone(),
            "strategy": format!("{:?}", memory_evaluation.context.selection.strategy),
            "predictedPeakBytes": memory_evaluation.context.predicted_peak_bytes,
            "evidenceRevision": memory_evaluation.context.evidence_revision.clone(),
        }),
    );
    let memory_context = memory_evaluation.context;
    let load_memory_context = memory_context.clone();

    // Conditioning-overlay VRAM admission (sc-16069, epic 15448) — the Kolors base held co-resident with
    // the IP-Adapter overlay. This lane loads through the UNcached `start_gen_stream` with a bespoke
    // `IpAdapterKolorsPaths`, so it reaches neither the `generate_candle_stream` `vram_gate` nor the
    // `generator_cache` `apply_residency_policy`; before this it allocated unchecked.
    let mut admission_overlays = vec![ip_adapter.as_path()];
    admission_overlays.extend(adapters.iter().map(|adapter| adapter.path.as_path()));
    admit_conditioning_paths(
        settings,
        "Kolors",
        "IP-Adapter",
        &kolors_base,
        &admission_overlays,
    )
    .await?;

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "kolors_ipadapter",
        0,
        move || {
            let paths = IpAdapterKolorsPaths {
                kolors_base,
                ip_adapter,
                adapters,
            };
            let model = IpAdapterKolors::load_with_memory_context(&paths, load_memory_context)
                .map_err(|error| {
                    WorkerError::Engine(format!("Kolors IP-Adapter load failed: {error}"))
                })?;
            // Per-job identity-likeness scorer built ONCE here (`!Send` face stack on the blocking
            // thread); source embedded once, reused across every output (sc-4411). `None` ⇒ non-fatal
            // staging / construction failure ⇒ scores omitted.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some(source)) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            Ok((model, reference, scorer))
        },
        move |(model, reference, scorer), tx, cancel| {
            // `IpAdapterKolors::generate` takes `&mut self` (it sets the IP image tokens on the UNet
            // before the denoise), so the per-item closure mutates `model`.
            let mut model = model;
            drive_gen_items_scored(
                tx,
                work,
                move |_index, (seed, prompt), preview, on_progress| {
                    if cancel.is_cancelled() {
                        return Ok(None);
                    }
                    let req = IpAdapterKolorsRequest {
                        prompt,
                        negative: negative_prompt.clone(),
                        width,
                        height,
                        steps: steps as usize,
                        guidance,
                        ip_adapter_scale: ip_scale,
                        seed: seed as u64,
                        sampler: sampler.clone(),
                        scheduler: scheduler.clone(),
                        preview,
                        cancel: cancel.clone(),
                    };
                    let out = match model.generate_with_memory_context(
                        &memory_context,
                        &req,
                        &reference,
                        &mut *on_progress,
                    ) {
                        Ok(out) => out,
                        Err(_) if cancel.is_cancelled() => return Ok(None),
                        Err(error) => {
                            return Err(WorkerError::Engine(format!(
                                "Kolors IP-Adapter generation failed: {error}"
                            )));
                        }
                    };
                    // Score this finished image against the cached source embedding (sc-4411). Clone paid
                    // ONLY when a scorer exists; non-frontal → honest detected:false N/A; `None` ⇒ omitted.
                    let face_likeness = scorer.as_ref().and_then(|scorer| {
                        crate::face_likeness::score_generated_image(
                            Some(scorer),
                            &Image {
                                width: out.width,
                                height: out.height,
                                pixels: out.pixels.clone(),
                            },
                            Some(likeness_source_ref.as_str()),
                        )
                    });
                    Ok(Some((
                        seed,
                        out.width,
                        out.height,
                        out.pixels,
                        face_likeness,
                    )))
                },
            )
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        KOLORS_IPADAPTER_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
