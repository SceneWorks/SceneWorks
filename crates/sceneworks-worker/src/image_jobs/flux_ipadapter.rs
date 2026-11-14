use super::admit_conditioning_paths;
use super::conditioning::load_reference_image;
use super::conditioning::non_empty;
use super::conditioning::resolve_character_image_likeness_source;
use super::conditioning::stage_likeness;
use super::events::consume_gen_events;
use super::load_policy::apply_candle_image_load_shape;
use super::load_spec::candle_artifact_path_matches;
use super::load_spec::candle_certified_artifact_path;
use super::load_spec::candle_pinned_hf_artifact_path;
use super::resolve_seed;
use super::sampling::resolve_advanced_or_manifest_f32_with;
use super::sampling::resolve_advanced_or_manifest_u32_with;
use super::standard_tier_subdir;
use super::stream::drive_gen_items_scored;
use super::stream::start_gen_stream_with_cancel;
use super::Image;
use super::ImagePlan;
use super::ImageRequest;
use super::IpAdapterFlux;
use super::IpAdapterFluxPaths;
use super::IpAdapterFluxRequest;
use crate::advanced;
use crate::ensure_hf_cached_file;
use crate::huggingface_snapshot_dir;
use crate::resolve_app_managed_model_dir;
use crate::ApiClient;
use crate::DownloadContext;
use crate::JobSnapshot;
use crate::JsonObject;
use crate::Path;
use crate::PathBuf;
use crate::Settings;
use crate::Value;
use crate::WorkerError;
use crate::WorkerResult;
use serde_json::json;

// Candle (Windows/CUDA) FLUX XLabs IP-Adapter reference route (sc-5872, epic 5480) — reference-image
// (identity) conditioning on FLUX.1 [dev]/[schnell] off-Mac via `runtime_cuda::providers::flux::IpAdapterFlux`. The
// FLUX sibling of the candle SDXL/Kolors IP-Adapter lanes (sdxl_ipadapter.rs / kolors_ipadapter.rs):
// the pooled CLIP-ViT-L image embedding is projected (XLabs `ImageProjModel`) into image-prompt tokens
// and injected as a decoupled cross-attention into the forked FLUX DiT's double blocks, denoised with
// the FLUX flow-match schedule (a single distilled forward per step).
//
// **Candle-only.** macOS keeps the MLX FLUX XLabs IP path (epic 3621, the registry `Reference` route
// the `flux1_*` generators handle via `with_ip_adapter`); the candle `IpAdapterFlux` is a bespoke
// provider, so this whole file is gated to the Windows/CUDA candle build (the module declaration in
// image_jobs.rs carries the cfg). It is a child module of the `image_jobs` module, so it shares that
// module's imports (ImageRequest/Settings/WorkerResult/`advanced`/`load_reference_image`/
// `huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).

/// The XLabs FLUX IP-Adapter repo + the single bundle file (`ip_adapter_proj_model` + the 19
/// per-double-block K/V projectors). Same repo the MLX path uses (`base.rs` `FLUX_IP_ADAPTER_REPO`).
const FLUX_IPADAPTER_ADAPTER_REPO: &str = "XLabs-AI/flux-ip-adapter";
const FLUX_IPADAPTER_ADAPTER_FILE: &str = "ip_adapter.safetensors";
/// The CLIP ViT-L/14 image encoder repo (the XLabs adapter conditions on `openai/clip-vit-large-patch14`
/// pooled image embeds) + the weight file the provider's `resolve_image_encoder` looks for in the dir.
const FLUX_IPADAPTER_ENCODER_REPO: &str = "openai/clip-vit-large-patch14";
const FLUX_IPADAPTER_ENCODER_FILE: &str = "model.safetensors";
/// Pinned revisions for the two IP-Adapter repos (sc-9879, F-077 follow-up). Both are fixed,
/// non-overridable consts, so fetching the mutable `main` branch means an upstream re-push could silently
/// swap the adapter / CLIP-image-encoder weights we load. Pin each to its exact commit for
/// defense-in-depth (mirrors sc-8879/sc-9682). HF's tree API still reports each file's `lfs.oid`, which
/// `ensure_hf_cached_file` verifies the downloaded content against. (Split per repo because a single sha
/// cannot address two different repos.)
pub(super) const FLUX_IPADAPTER_ADAPTER_REVISION: &str = "18f6940238ab5dc3744df7a8e30315892279d5f9";
pub(super) const FLUX_IPADAPTER_ENCODER_REVISION: &str = "32bd64288804d66eefd0ccbe215aa642df71cc41";
/// IP-Adapter scale default — the XLabs resemblance tier 0.7 (matches `base.rs` `FLUX_IP_SCALE`, the MLX
/// path, and the candle `IpAdapterFlux::DEFAULT_IP_SCALE`).
const FLUX_IPADAPTER_IP_SCALE: f32 = 0.7;
const FLUX_IPADAPTER_MEMORY_OVERLAY: &str = "identity";
/// The adapter/engine id recorded on candle FLUX IP-Adapter assets + telemetry (distinct from the
/// txt2img `candle_flux` lane).
pub(super) const FLUX_IPADAPTER_ENGINE: &str = "candle_flux_ipadapter";

fn flux_ipadapter_overlay_paths_are_certified(
    adapter_file: &Path,
    encoder_dir: &Path,
    expected_adapter: Option<PathBuf>,
    expected_encoder: Option<PathBuf>,
) -> bool {
    let (Some(expected_adapter), Some(expected_encoder)) = (expected_adapter, expected_encoder)
    else {
        return false;
    };
    candle_artifact_path_matches(adapter_file, &expected_adapter)
        && candle_artifact_path_matches(encoder_dir, &expected_encoder)
}

/// Model ids the candle FLUX IP-Adapter route accepts (both variants — the forked DiT injects the same
/// XLabs adapter for each; dev embeds the guidance scalar, schnell ignores it).
fn is_flux_ipadapter_model(model: &str) -> bool {
    matches!(model, "flux_dev" | "flux_schnell")
}

/// The installed base repo when the manifest omits `repo` (variant-keyed).
fn flux_ipadapter_default_repo(model: &str) -> &'static str {
    crate::engines::default_repo_for(model).unwrap_or("SceneWorks/flux1-dev-mlx")
}

/// Distilled default step count (FLUX parity): schnell 4, dev 25.
fn flux_ipadapter_default_steps(model: &str) -> u32 {
    match model {
        "flux_schnell" => 4,
        _ => 25,
    }
}

/// Default guidance: dev embeds 3.5; schnell is timestep-distilled (no guidance).
fn flux_ipadapter_default_guidance(model: &str) -> f32 {
    match model {
        "flux_schnell" => 0.0,
        _ => 3.5,
    }
}

/// Resolve the FLUX base (BFL snapshot) for the IP-Adapter route: an explicit `modelPath` dir
/// (advanced or manifest) wins, else the HF cache snapshot for the manifest `repo` (the installed
/// `MODEL_TABLE` turnkey by variant). `None` means the base is not present locally, so the candle lane
/// refuses the job and no fallback is attempted. Mirrors `resolve_kolors_ipadapter_base`.
fn resolve_flux_ipadapter_base(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if let Some(path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(crate::payload::trimmed_value_str)
        .map(str::to_owned)
    {
        return resolve_app_managed_model_dir(settings, &path, "FLUX IP-Adapter modelPath")
            .map(Some);
    }
    let repo = crate::payload::trimmed_string(request.model_manifest_entry.get("repo"))
        .unwrap_or_else(|| flux_ipadapter_default_repo(&request.model));
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo)
        .map(|root| standard_tier_subdir(&root, request)))
}

/// True when this is a candle-eligible FLUX IP-Adapter job: a `flux_dev`/`flux_schnell` model with a
/// reference image (and NOT an img2img/inpaint/edit shape — those are sc-5487) whose base resolves
/// locally. Mirrors `jobs_store::flux_ipadapter_candle_eligible` so the worker and router agree.
pub(super) fn flux_ipadapter_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_flux_ipadapter_model(&request.model)
        && request.mode != "edit_image"
        && non_empty(&request.reference_asset_id)
        && !non_empty(&request.source_asset_id)
        && !non_empty(&request.mask_asset_id)
        && matches!(resolve_flux_ipadapter_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=100) → manifest `steps` → variant default.
fn flux_ipadapter_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32_with(
        request,
        "steps",
        || flux_ipadapter_default_steps(&request.model),
        1..=100,
    )
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → variant default, clamped.
fn flux_ipadapter_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32_with(
        request,
        "guidanceScale",
        || flux_ipadapter_default_guidance(&request.model),
        0.0..=30.0,
    )
}

/// Resolve the XLabs adapter file + the CLIP-ViT-L encoder dir the `IpAdapterFlux` provider loads,
/// downloading on first use. Resolution order for each: an env-pinned root (`SCENEWORKS_IPADAPTER_FLUX`,
/// the validation-staging layout `<root>/ip_adapter.safetensors` + `<root>/image_encoder/`) → a
/// whole-repo HF cache snapshot → download into the app cache. Returns `(adapter_file, encoder_dir)`
/// (the latter is what [`IpAdapterFluxPaths::image_encoder`] wants — `resolve_image_encoder` finds
/// `model.safetensors` inside).
async fn ensure_flux_ipadapter_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<(PathBuf, PathBuf)> {
    // Env override: a staged dir laid out like the validation harness.
    if let Ok(root) = std::env::var("SCENEWORKS_IPADAPTER_FLUX") {
        let root = PathBuf::from(root);
        let adapter = root.join(FLUX_IPADAPTER_ADAPTER_FILE);
        let encoder = root.join("image_encoder");
        if adapter.is_file() && encoder.join(FLUX_IPADAPTER_ENCODER_FILE).is_file() {
            return Ok((adapter, encoder));
        }
    }

    let client = crate::downloads::streaming_download_client();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "FLUX IP-Adapter generation canceled while fetching weights.",
        fresh_download: false,
    };
    let cache = settings.data_dir.join("cache").join("ipadapter-flux");

    // XLabs adapter file: a whole-repo HF cache snapshot already carrying it, else download-on-first-use.
    let adapter = match huggingface_snapshot_dir(&settings.data_dir, FLUX_IPADAPTER_ADAPTER_REPO)
        .map(|snapshot| snapshot.join(FLUX_IPADAPTER_ADAPTER_FILE))
        .filter(|file| file.is_file())
    {
        Some(file) => file,
        None => {
            let dst = cache.join(FLUX_IPADAPTER_ADAPTER_FILE);
            ensure_hf_cached_file(
                &context,
                FLUX_IPADAPTER_ADAPTER_REPO,
                FLUX_IPADAPTER_ADAPTER_REVISION,
                FLUX_IPADAPTER_ADAPTER_FILE,
                &dst,
            )
            .await?;
            dst
        }
    };

    // CLIP-ViT-L encoder dir (must contain `model.safetensors`): the whole-repo snapshot, else download.
    let encoder = match huggingface_snapshot_dir(&settings.data_dir, FLUX_IPADAPTER_ENCODER_REPO)
        .filter(|snapshot| snapshot.join(FLUX_IPADAPTER_ENCODER_FILE).is_file())
    {
        Some(snapshot) => snapshot,
        None => {
            let dir = cache.join("image_encoder");
            ensure_hf_cached_file(
                &context,
                FLUX_IPADAPTER_ENCODER_REPO,
                FLUX_IPADAPTER_ENCODER_REVISION,
                FLUX_IPADAPTER_ENCODER_FILE,
                &dir.join(FLUX_IPADAPTER_ENCODER_FILE),
            )
            .await?;
            dir
        }
    };

    Ok((adapter, encoder))
}

/// Flat telemetry recorded on candle FLUX IP-Adapter assets (parity with the SDXL/Kolors recipe-key
/// shape).
fn flux_ipadapter_raw_settings(
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
        Value::String(FLUX_IPADAPTER_ENGINE.to_owned()),
    );
    raw
}

/// The per-output request numerics of one FLUX IP-Adapter job. The pre-load planner and the render
/// build each output's request from this one source, so the plan is compiled from the request it runs.
struct FluxIpRequests {
    width: u32,
    height: u32,
    steps: u32,
    sampling: gen_core::GenerationRequest,
    guidance: f32,
    ip_scale: f32,
    memory: gen_core::GenerationMemory,
}

impl FluxIpRequests {
    fn request(
        &self,
        seed: i64,
        prompt: String,
        preview: &gen_core::PreviewSink,
        cancel: &gen_core::CancelFlag,
    ) -> IpAdapterFluxRequest {
        IpAdapterFluxRequest {
            prompt,
            width: self.width,
            height: self.height,
            steps: self.steps as usize,
            sampler: self.sampling.sampler.clone(),
            scheduler: self.sampling.scheduler.clone(),
            scheduler_shift: self.sampling.scheduler_shift,
            guidance: self.guidance,
            ip_adapter_scale: self.ip_scale,
            seed: seed as u64,
            memory: self.memory,
            preview: preview.clone(),
            cancel: cancel.clone(),
        }
    }
}

/// Compile every output's render plan before any weights open (SW-6936): each plan binds the
/// reference, the request and the resolved load contract, so a request the provider refuses (a
/// load contract without the overlay it needs, a mixed overlay, an unsupported request) fails the job
/// here with a payload error instead of after the load.
fn plan_flux_ip_outputs(
    paths: &IpAdapterFluxPaths,
    spec: &gen_core::LoadSpec,
    requests: &FluxIpRequests,
    work: Vec<(i64, String)>,
    reference: &Image,
    cancel: &gen_core::CancelFlag,
) -> WorkerResult<Vec<gen_core::RenderPlan>> {
    work.into_iter()
        .map(|(seed, prompt)| {
            // Preview never shapes the trajectory; the live sink is bound at generation time.
            let req = requests.request(seed, prompt, &gen_core::PreviewSink::default(), cancel);
            runtime_cuda::providers::flux::ip_provider::compile_ip_plan(
                paths, spec, &req, reference,
            )
            .map_err(|error| super::render::preload_plan_error("FLUX IP-Adapter", error))
        })
        .collect()
}

/// Real candle FLUX IP-Adapter generation: resolve the reference + weights on the async side, then load
/// the `IpAdapterFlux` provider once + generate each image on the blocking thread. `request.count`
/// images, each its own seed; `generate` takes the per-job `CancelFlag` + a `Progress` callback, so
/// streaming is per-step and cancellation is honoured mid-denoise — same contract as the SDXL/Kolors IP
/// lanes.
pub(super) async fn generate_candle_flux_ipadapter_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let flux_base = resolve_flux_ipadapter_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("FLUX IP-Adapter base (FLUX.1 snapshot) not found".to_owned())
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("FLUX IP-Adapter requires a reference image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;
    let (adapter_file, encoder_dir) = ensure_flux_ipadapter_weights(api, settings, job).await?;

    // Identity-likeness scoring (epic 4406, sc-4411 plain With-Character): the candle FLUX XLabs
    // IP-Adapter lane is the With-Character route for a FLUX.1 model — score every output against the
    // reference face through the SHARED generator-agnostic seam. Eligibility goes through
    // `resolve_character_image_likeness_source` (the SAME gate the macOS lanes use), so the angle/pose/
    // edit exclusion is explicit and self-contained here — NOT dependent on dispatch order (an angle/pose
    // job is excluded by the helper even if it ever reached this lane, so it can never be double-scored).
    // The helper's decode is ignored: the already-decoded `reference` (this lane's generation input, the
    // current job's `referenceAssetId`) is the scorer source, so there is no second decode. Stage the
    // antelopev2 SCRFD + ArcFace bundle; the `!Send` scorer is built ONCE inside the load closure and
    // reused across the N outputs (source embedded once). Staging is non-fatal (failure → scores omitted).
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

    let sampling = super::sampling::explicit_sampling_request(request);
    let steps = flux_ipadapter_steps(request);
    let guidance = flux_ipadapter_guidance(request);
    let ip_scale = advanced::f32_clamped(
        &request.advanced,
        "ipAdapterScale",
        FLUX_IPADAPTER_IP_SCALE,
        0.0..=1.0,
    );
    let repo = crate::payload::trimmed_string(request.model_manifest_entry.get("repo"))
        .unwrap_or_else(|| flux_ipadapter_default_repo(&request.model))
        .to_owned();
    let mut raw_settings = flux_ipadapter_raw_settings(request, &repo, steps, guidance, ip_scale);

    // Per-image work items: (seed, prompt) — `request.count` images at the reference identity.
    let (width, height) = (request.width, request.height);
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let engine_id = if request.model == "flux_schnell" {
        "flux1_schnell"
    } else {
        "flux1_dev"
    };
    let tier = match flux_base.file_name().and_then(|name| name.to_str()) {
        Some("q4") => "q4",
        Some("q8") => "q8",
        _ => "bf16",
    };
    let runtime_overlay_bytes = gen_core::weightsmeta::safetensors_path_bytes(&adapter_file)
        .saturating_add(gen_core::weightsmeta::safetensors_path_bytes(&encoder_dir));
    let strategy_spec = apply_candle_image_load_shape(
        engine_id,
        gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(flux_base.clone()))
            .with_ip_adapter(gen_core::WeightsSource::File(adapter_file.clone()))
            .with_component(
                "flux_ip_image_encoder",
                gen_core::WeightsSource::Dir(encoder_dir.clone()),
            )
            .with_offload_policy(gen_core::OffloadPolicy::Sequential),
    );
    let raw_budget = crate::vram_gate::live_capped_budget(&settings.gpu_id).await;
    let predicted_peak = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        &request.model_manifest_entry,
        tier,
        runtime_overlay_bytes,
    );
    let expected_adapter = candle_pinned_hf_artifact_path(
        settings,
        FLUX_IPADAPTER_ADAPTER_REPO,
        FLUX_IPADAPTER_ADAPTER_REVISION,
        Path::new(FLUX_IPADAPTER_ADAPTER_FILE),
    );
    let expected_encoder = candle_pinned_hf_artifact_path(
        settings,
        FLUX_IPADAPTER_ENCODER_REPO,
        FLUX_IPADAPTER_ENCODER_REVISION,
        Path::new(""),
    );
    let artifact_is_certified =
        candle_certified_artifact_path(engine_id, settings, &flux_base, tier)
            && flux_ipadapter_overlay_paths_are_certified(
                &adapter_file,
                &encoder_dir,
                expected_adapter,
                expected_encoder,
            );
    let memory_evaluation = crate::candle_memory_strategy::evaluate_shared_image(
        engine_id,
        &request.model,
        &strategy_spec,
        artifact_is_certified,
        &request.model_manifest_entry,
        tier,
        &request.mode,
        Some(FLUX_IPADAPTER_MEMORY_OVERLAY),
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
        false,
        raw_budget,
        raw_budget.map_or(0.0, crate::vram_gate::ladder_reserve_gb),
        predicted_peak,
        runtime_overlay_bytes,
        gen_core::MemoryCacheState::Cold,
    )?;
    let generation_memory = memory_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.memory)
        .unwrap_or_default();
    if let Some(evaluation) = &memory_evaluation {
        raw_settings.insert(
            "memoryStrategy".to_owned(),
            Value::String(format!("{:?}", evaluation.context.selection.strategy)),
        );
    }
    let memory_context = memory_evaluation.map(|evaluation| evaluation.context);

    // Conditioning-overlay VRAM admission (sc-16069, epic 15448) — the FLUX.1 base held co-resident with
    // the XLabs IP-Adapter and its CLIP image encoder. This lane loads through the UNcached
    // `start_gen_stream` with a bespoke `IpAdapterFluxPaths`, so it reaches neither the
    // `generate_candle_stream` `vram_gate` nor the `generator_cache` `apply_residency_policy`; before this
    // it allocated unchecked.
    if generation_memory == gen_core::GenerationMemory::default() {
        admit_conditioning_paths(
            settings,
            "FLUX.1",
            "IP-Adapter",
            &flux_base,
            &[adapter_file.as_path(), encoder_dir.as_path()],
        )
        .await?;
    }

    let requests = FluxIpRequests {
        width,
        height,
        steps,
        sampling,
        guidance,
        ip_scale,
        memory: generation_memory,
    };
    let planning_work = work.clone();

    let (cancel, rx, blocking) = start_gen_stream_with_cancel(
        job.id.clone(),
        "flux_ipadapter",
        0,
        move |planning_cancel| {
            let paths = IpAdapterFluxPaths {
                flux_base,
                ip_adapter: adapter_file,
                image_encoder: encoder_dir,
                // This lane has no LoRA/LoKr plumbing; an empty stack is the load it has always
                // performed.
                adapters: Vec::new(),
            };
            let plans = plan_flux_ip_outputs(
                &paths,
                &strategy_spec,
                &requests,
                planning_work,
                &reference,
                planning_cancel,
            )?;
            if planning_cancel.is_cancelled() {
                return Err(WorkerError::Canceled(
                    "FLUX IP-Adapter generation canceled before load".into(),
                ));
            }
            let loaded = match memory_context.as_ref() {
                Some(context) => IpAdapterFlux::load_with_memory_context_spec(
                    &paths,
                    context.clone(),
                    &strategy_spec,
                ),
                None => {
                    IpAdapterFlux::load_with_memory_spec(&paths, generation_memory, &strategy_spec)
                }
            };
            let model = loaded.map_err(|error| {
                WorkerError::Engine(format!("FLUX IP-Adapter load failed: {error}"))
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
            Ok((model, reference, plans, scorer, memory_context, requests))
        },
        move |(model, reference, plans, scorer, memory_context, requests), tx, cancel| {
            // `IpAdapterFlux` renders through `&self` (the IP tokens live in a per-call injector, not on
            // the DiT), so — unlike the SDXL/Kolors lanes — the per-item closure needs no `mut model`.
            drive_gen_items_scored(
                tx,
                work,
                move |index, (seed, prompt), preview, on_progress| {
                    if cancel.is_cancelled() {
                        return Ok(None);
                    }
                    let req = requests.request(seed, prompt, &preview, &cancel);
                    let render_plan = plans.get(index).ok_or_else(|| {
                        WorkerError::Engine(format!(
                            "FLUX IP-Adapter: no pre-load render plan for output {index}"
                        ))
                    })?;
                    let generated = match memory_context.as_ref() {
                        Some(context) => model.generate_with_memory_context_planned(
                            context,
                            &req,
                            &reference,
                            render_plan,
                            &mut *on_progress,
                        ),
                        None => {
                            model.generate_planned(&req, &reference, render_plan, &mut *on_progress)
                        }
                    };
                    let out = match generated {
                        Ok(out) => out,
                        Err(_) if cancel.is_cancelled() => return Ok(None),
                        Err(error) => {
                            return Err(WorkerError::Engine(format!(
                                "FLUX IP-Adapter generation failed: {error}"
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
        FLUX_IPADAPTER_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

#[cfg(test)]
mod memory_artifact_tests {
    use super::*;

    #[test]
    fn conditional_base_prefers_external_model_path_then_the_selected_tier() {
        for (model, repo) in [
            ("flux_dev", "SceneWorks/flux1-dev-mlx"),
            ("flux_schnell", "SceneWorks/flux1-schnell-mlx"),
        ] {
            super::super::render::flux1_conditional_route_tests::assert_conditional_base_resolution(
                &format!("FLUX.1 XLabs IP-Adapter (Candle, {model})"),
                json!({
                    "projectId": "p", "model": model, "prompt": "a portrait", "count": 1,
                    "mode": "character_image", "referenceAssetId": "asset_1"
                }),
                repo,
                resolve_flux_ipadapter_base,
            );
        }
    }

    /// SW-6936: a load contract the provider refuses fails while compiling the pre-load plans, as a
    /// payload error the job reports, before any weights open.
    #[test]
    fn preload_plan_refusals_are_payload_errors_before_load() {
        let root = tempfile::tempdir().expect("ip fixture");
        let base = root.path().join("base");
        let adapter = root.path().join("ip_adapter.safetensors");
        let encoder = root.path().join("encoder");
        let paths = IpAdapterFluxPaths {
            flux_base: base.clone(),
            ip_adapter: adapter.clone(),
            image_encoder: encoder.clone(),
            adapters: Vec::new(),
        };
        let requests = FluxIpRequests {
            width: 512,
            height: 512,
            steps: 4,
            sampling: gen_core::GenerationRequest::default(),
            guidance: 3.5,
            ip_scale: 0.7,
            memory: gen_core::GenerationMemory::default(),
        };
        let reference = Image {
            width: 64,
            height: 64,
            pixels: vec![0; 64 * 64 * 3],
        };
        let base_spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(base.clone()));
        for (case, spec) in [
            ("missing IP-Adapter overlay", base_spec.clone()),
            (
                "mixed IP-Adapter + control",
                base_spec
                    .clone()
                    .with_ip_adapter(gen_core::WeightsSource::File(adapter.clone()))
                    .with_component(
                        "flux_ip_image_encoder",
                        gen_core::WeightsSource::Dir(encoder.clone()),
                    )
                    .with_control(gen_core::WeightsSource::File(adapter.clone())),
            ),
        ] {
            match plan_flux_ip_outputs(
                &paths,
                &spec,
                &requests,
                vec![(7, "a portrait".to_owned())],
                &reference,
                &gen_core::CancelFlag::new(),
            ) {
                Err(WorkerError::InvalidPayload(message)) => assert!(
                    message.contains("cannot compile render plan before load"),
                    "{case}: {message}"
                ),
                other => panic!("{case}: expected a pre-load payload refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn optimized_ip_evidence_requires_both_exact_overlay_artifacts() {
        let root = tempfile::tempdir().expect("overlay fixture");
        let expected_adapter = root.path().join("pinned-adapter.safetensors");
        let expected_encoder = root.path().join("pinned-encoder");
        std::fs::write(&expected_adapter, b"adapter").expect("adapter fixture");
        std::fs::create_dir_all(&expected_encoder).expect("encoder fixture");
        let alternate_adapter = root.path().join("alternate-adapter.safetensors");
        let alternate_encoder = root.path().join("alternate-encoder");
        std::fs::write(&alternate_adapter, b"adapter").expect("alternate adapter");
        std::fs::create_dir_all(&alternate_encoder).expect("alternate encoder");

        assert!(flux_ipadapter_overlay_paths_are_certified(
            &expected_adapter,
            &expected_encoder,
            Some(expected_adapter.clone()),
            Some(expected_encoder.clone()),
        ));
        assert_eq!(FLUX_IPADAPTER_MEMORY_OVERLAY, "identity");
        assert!(!flux_ipadapter_overlay_paths_are_certified(
            &alternate_adapter,
            &expected_encoder,
            Some(expected_adapter.clone()),
            Some(expected_encoder.clone()),
        ));
        assert!(!flux_ipadapter_overlay_paths_are_certified(
            &expected_adapter,
            &alternate_encoder,
            Some(expected_adapter.clone()),
            Some(expected_encoder.clone()),
        ));
        assert!(!flux_ipadapter_overlay_paths_are_certified(
            &alternate_adapter,
            &alternate_encoder,
            None,
            None,
        ));
    }
}
