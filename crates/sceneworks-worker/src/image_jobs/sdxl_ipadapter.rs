use super::{
    admit_conditioning_paths, consume_gen_events, curated_image_menu, dense_tier_subdir,
    drive_gen_items_scored, load_reference_image, non_empty, normalize_sampling_knob,
    pid_effective_dims, pid_output_tier, read_advanced_sampling_knobs,
    resolve_advanced_or_manifest_f32, resolve_advanced_or_manifest_u32,
    resolve_character_image_likeness_source, resolve_pid_weights, resolve_sdxl_components,
    resolve_seed, stage_likeness, standard_tier_subdir, start_gen_stream,
    uses_standard_tier_layout, ApiClient, Image, ImagePlan, ImageRequest, IpAdapterSdxl,
    IpAdapterSdxlPaths, IpAdapterSdxlRequest, JobSnapshot, JsonObject, Path, PathBuf, Settings,
    Value, WorkerError, WorkerResult,
};
use super::{advanced, ensure_hf_cached_file, huggingface_snapshot_dir};
use super::{resolve_app_managed_model_dir, DownloadContext};
use serde_json::json;

// Candle (Windows/CUDA) SDXL IP-Adapter-Plus reference route (sc-5488, epic 5480) â€” reference-image
// (identity) conditioning on SDXL/RealVisXL off-Mac via `runtime_cuda::providers::sdxl::IpAdapterSdxl`. The
// reference-conditioning sibling of the candle InstantID lane (instantid.rs), but plain SDXL: no face
// stack, no IdentityNet/OpenPose ControlNet â€” just the CLIP ViT-H image tokens â†’ pure-IP denoise.
//
// **Candle-only.** macOS keeps the MLX SDXL IP path (the registry `SdxlSubMode::Ip` in sdxl.rs); there
// is no MLX `IpAdapterSdxl`, so this whole file is gated to the Windows/CUDA candle build (the
// the module declaration in image_jobs.rs carries the cfg). It is a child module of the `image_jobs` module, so it
// shares that module's imports (ImageRequest/Settings/WorkerResult/`advanced`/`load_reference_image`/
// `huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/â€¦ all in scope unqualified).

/// h94 IP-Adapter repo (the ViT-H encoder + the plus/plus-face SDXL weights), matching the MLX SDXL IP
/// path's `SDXL_IP_ADAPTER_REPO`.
const SDXL_IPADAPTER_REPO: &str = "h94/IP-Adapter";
/// Pinned revision for the `h94/IP-Adapter` repo (sc-9879, F-077 follow-up). A fixed, non-overridable
/// repo (the env pin points at a local dir, not another HF repo), so fetching the mutable `main` branch
/// means an upstream re-push could silently swap the adapter / CLIP-encoder weights we load. Pin the
/// exact commit for defense-in-depth (mirrors sc-8879/sc-9682). HF's tree API still reports each file's
/// `lfs.oid`, which `ensure_hf_cached_file` verifies the downloaded content against.
pub(super) const SDXL_IPADAPTER_REVISION: &str = "018e402774aeeddd60609b4ecdb7e298259dc729";
/// The IP-Adapter-Plus (ViT-H) bundle inside the repo (`image_proj` Resampler + `ip_adapter.*` K/V).
const SDXL_IPADAPTER_BUNDLE_SRC: &str = "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors";
/// The CLIP ViT-H image-encoder files inside the repo (config + weights).
const SDXL_IPADAPTER_ENCODER_SRC: [&str; 2] = [
    "models/image_encoder/config.json",
    "models/image_encoder/model.safetensors",
];
/// IP-Adapter scale default â€” torch plus parity (matches the MLX SDXL path's `SDXL_IP_SCALE`).
const SDXL_IPADAPTER_IP_SCALE: f32 = 0.7;
/// Denoise steps default (SDXL production).
const SDXL_IPADAPTER_DEFAULT_STEPS: u32 = 30;
/// CFG default â€” the reference-conditioned envelope validated on GPU (sc-5488); base SDXL uses ~7.
const SDXL_IPADAPTER_DEFAULT_GUIDANCE: f32 = 5.0;
/// The adapter/engine id recorded on candle SDXL IP-Adapter assets + telemetry (distinct from the
/// txt2img `candle_sdxl` and the `candle_instantid` lanes).
pub(super) const SDXL_IPADAPTER_ENGINE: &str = "candle_sdxl_ipadapter";

/// SDXL model ids the candle IP-Adapter route accepts (the txt2img-eligible SDXL family). Must stay
/// in lockstep with `jobs_store::routing::candle`'s `sdxl_ipadapter_candle_eligible` guard.
fn is_sdxl_ipadapter_model(model: &str) -> bool {
    matches!(
        model,
        "sdxl" | "realvisxl" | "illustrious_xl_v1" | "illustrious_xl_v2"
    )
}

/// Default SDXL base repo for a model id when the manifest omits `repo` â€” which, in production, is
/// ALWAYS (sc-14463).
///
/// This reads `MODEL_TABLE` rather than carrying a private mapping, so it cannot drift from the
/// txt2img lane (`base.rs::model_repo`) or InstantID (`INSTANTID_SDXL_REPO`) again. It previously
/// hardcoded the flat upstream repos, and the comment here claimed "in practice the built-in SDXL
/// family points `repo` at the SceneWorks turnkey" â€” that was false. The lanes read a TOP-LEVEL
/// `repo` key, and no built-in model declares one (the manifest carries `downloads[].repo` and
/// `paths.model`, which are different keys; `resolve_model_manifest_entry` injects only `modelPath`).
/// So this is not a fallback, it is the only branch â€” and after the Group-B cutover (sc-8746) it
/// named repos the installer no longer stages, leaving every candle IP-Adapter job unserved.
///
/// The tier descent still applies on top: sc-10813 serves the request's q4/q8 tier via
/// `standard_tier_subdir`, and `dense_tier_subdir` (sc-10614) takes the non-standard branch.
pub(super) fn sdxl_ipadapter_default_repo(model: &str) -> &'static str {
    // Every id `is_sdxl_ipadapter_model` admits has a MODEL_TABLE row; the SDXL base turnkey is a
    // defensive floor rather than a reachable case.
    crate::engines::default_repo_for(model).unwrap_or("SceneWorks/sdxl-base-mlx")
}

/// Resolve the SDXL base snapshot for the IP-Adapter route: an explicit `modelPath` dir (advanced or
/// manifest) wins, else the HF cache snapshot for the manifest `repo` (default by model id). `None`
/// means the base is not present locally, so the candle lane refuses the job and no fallback is attempted.
fn resolve_sdxl_ipadapter_base(
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
        return resolve_app_managed_model_dir(settings, &path, "SDXL IP-Adapter modelPath")
            .map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| sdxl_ipadapter_default_repo(&request.model));
    // sc-10813: a standard-tier turnkey (`mlx.standardTierLayout`, e.g. `SceneWorks/sdxl-base-mlx`)
    // descends into the request's packed q4/q8 tier (or bf16) via `standard_tier_subdir` â€” the SAME
    // resolver the txt2img lane uses (base.rs `resolve_base_model_dir`) â€” now that `load_instantid_unet`
    // packed-detects the tier. A flat upstream diffusers snapshot (no q4/q8/bf16 subdirs) has no present
    // tier, so `standard_tier_subdir` returns its root untouched; a NON-standard tiered turnkey keeps the
    // dense `bf16/` descent (sc-10614). Both fall through `dense_tier_subdir` on the non-standard branch.
    Ok(
        huggingface_snapshot_dir(&settings.data_dir, repo).map(|root| {
            if uses_standard_tier_layout(request) {
                standard_tier_subdir(&root, request)
            } else {
                dense_tier_subdir(root)
            }
        }),
    )
}

/// True when this is a candle-eligible SDXL IP-Adapter job: an sdxl-family model with a reference image
/// (and NOT an img2img/inpaint/edit shape â€” those advanced SDXL modes are sc-5487) whose base resolves
/// locally. Mirrors `jobs_store::sdxl_ipadapter_candle_eligible` so the worker and router agree.
pub(super) fn sdxl_ipadapter_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_sdxl_ipadapter_model(&request.model)
        && request.mode != "edit_image"
        && non_empty(&request.reference_asset_id)
        && !non_empty(&request.source_asset_id)
        && !non_empty(&request.mask_asset_id)
        && matches!(resolve_sdxl_ipadapter_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=80) â†’ manifest `steps` â†’ default (30).
fn sdxl_ipadapter_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", SDXL_IPADAPTER_DEFAULT_STEPS, 1..=80)
}

/// Resolve guidance: `advanced.guidanceScale` â†’ manifest `guidanceScale` â†’ the reference-tuned default
/// (5.0), clamped to a sane CFG range.
fn sdxl_ipadapter_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        SDXL_IPADAPTER_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// Resolve the IP-Adapter bundle file + the CLIP ViT-H image-encoder dir, downloading from
/// `h94/IP-Adapter` on first use. Resolution order: an env-pinned root (pre-staged, the validation
/// path) â†’ a whole-repo HF cache snapshot â†’ download the bundle + encoder into the app cache. Returns
/// `(bundle_file, image_encoder_dir)` â€” what [`IpAdapterSdxlPaths`] wants.
async fn ensure_sdxl_ipadapter_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<(PathBuf, PathBuf)> {
    // Env override: a directory laid out like the h94 repo (or its HF snapshot), pre-staged for local
    // validation (`SCENEWORKS_IPADAPTER_SDXL`).
    if let Ok(root) = std::env::var("SCENEWORKS_IPADAPTER_SDXL") {
        let root = PathBuf::from(root);
        let bundle = root
            .join("sdxl_models")
            .join("ip-adapter-plus_sdxl_vit-h.safetensors");
        let encoder = root.join("models").join("image_encoder");
        if bundle.is_file() && encoder.is_dir() {
            return Ok((bundle, encoder));
        }
    }
    // Whole-repo HF cache snapshot already present (the model-download flow staged it).
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, SDXL_IPADAPTER_REPO) {
        let bundle = snapshot
            .join("sdxl_models")
            .join("ip-adapter-plus_sdxl_vit-h.safetensors");
        let encoder = snapshot.join("models").join("image_encoder");
        if bundle.is_file() && encoder.join("model.safetensors").is_file() {
            return Ok((bundle, encoder));
        }
    }
    // Download-on-first-use into the app cache (flat dest, nested source â€” the InstantID bundle pattern).
    let client = crate::downloads::streaming_download_client();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "SDXL IP-Adapter generation canceled while fetching weights.",
        fresh_download: false,
    };
    let cache = settings.data_dir.join("cache").join("ipadapter-sdxl");
    let bundle = ensure_hf_cached_file(
        &context,
        SDXL_IPADAPTER_REPO,
        SDXL_IPADAPTER_REVISION,
        SDXL_IPADAPTER_BUNDLE_SRC,
        &cache.join("ip-adapter-plus_sdxl_vit-h.safetensors"),
    )
    .await?;
    let encoder = cache.join("image_encoder");
    for source in SDXL_IPADAPTER_ENCODER_SRC {
        let name = source.rsplit('/').next().unwrap_or(source);
        ensure_hf_cached_file(
            &context,
            SDXL_IPADAPTER_REPO,
            SDXL_IPADAPTER_REVISION,
            source,
            &encoder.join(name),
        )
        .await?;
    }
    Ok((bundle, encoder))
}

/// Flat telemetry recorded on candle SDXL IP-Adapter assets (parity with the InstantID/`mlx_raw_settings`
/// recipe-key shape).
fn sdxl_ipadapter_raw_settings(
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
        Value::String(SDXL_IPADAPTER_ENGINE.to_owned()),
    );
    raw
}

/// Real candle SDXL IP-Adapter generation: resolve the reference + weights on the async side, then load
/// the `IpAdapterSdxl` provider once + generate each image on the blocking thread. `request.count`
/// images, each its own seed; the engine `generate` takes the per-job `CancelFlag` + a `Progress`
/// callback, so streaming is per-step and cancellation is honoured mid-denoise â€” same contract as the
/// registry families + the InstantID lane. Reuses [`consume_gen_events`] for the asset writes.
pub(super) async fn generate_candle_sdxl_ipadapter_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let sdxl_base = resolve_sdxl_ipadapter_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("SDXL IP-Adapter base (SDXL/RealVisXL) not found".to_owned())
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("SDXL IP-Adapter requires a reference image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;
    let (ip_bundle, image_encoder) = ensure_sdxl_ipadapter_weights(api, settings, job).await?;

    // Identity-likeness scoring (epic 4406, sc-4411 plain With-Character): the candle SDXL IP-Adapter
    // lane is the With-Character route for an SDXL-family model â€” score every output against the
    // reference face through the SHARED generator-agnostic seam. Eligibility goes through
    // `resolve_character_image_likeness_source` (the SAME gate the macOS lanes use), so the angle/pose/
    // edit exclusion is explicit and self-contained here â€” NOT dependent on dispatch order (an angle/pose
    // job is excluded by the helper even if it ever reached this lane, so it can never be double-scored).
    // The helper's decode is ignored: the already-decoded `reference` (this lane's generation input, the
    // current job's `referenceAssetId`) is the scorer source, so there is no second decode. Stage the
    // antelopev2 SCRFD + ArcFace bundle (the scorer's candle leg loads it); the `!Send` scorer is built
    // ONCE inside the load closure and reused across the N outputs (source embedded once â€” the caching
    // AC). Staging is non-fatal (failure â†’ no scorer â†’ scores omitted, generation still renders).
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

    let steps = sdxl_ipadapter_steps(request);
    let guidance = sdxl_ipadapter_guidance(request);
    let ip_scale = advanced::f32_clamped(
        &request.advanced,
        "ipAdapterScale",
        SDXL_IPADAPTER_IP_SCALE,
        0.0..=1.0,
    );
    // Curated unified-sampler selection (epic 7114, sc-7432): the candle `IpAdapterSdxl` provider honors
    // a curated solver/scheduler via the shared `denoise_curated` primitive (#130). Read + N3-normalize
    // against the shared curated menu (an unknown name drops to the engine default + emits an event). N1:
    // unset â‡’ `None` â‡’ the native ancestral default loop runs byte-exact. (`sdxl`/`realvisxl` are
    // MODEL_TABLE rows already advertising the curated menu â€” guarded by the existing drift guard.)
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
        .unwrap_or_else(|| sdxl_ipadapter_default_repo(&request.model))
        .to_owned();
    // Per-generation PiD decode (epic 7840, sc-8044): resolve the `sdxl` PiD student + Gemma when
    // `advanced.usePid` is set and the snapshots are cached; else `None` â†’ native VAE. The SDXL IP-Adapter
    // composes the SDXL VAE, so it shares the one `sdxl` student. `use_pid` and the engine's `with_pid`
    // load stay in lockstep (the engine rejects a mismatch).
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();

    let mut raw_settings = sdxl_ipadapter_raw_settings(request, &repo, steps, guidance, ip_scale);
    // Mark PiD output on the sidecar (epic 7840): the NSCLv1 NC restriction flows to PiD output. Record
    // whether PiD ACTUALLY ran (opted in AND snapshots cached), not merely whether it was requested.
    raw_settings.insert("usePid".to_owned(), Value::Bool(use_pid));

    // Per-image work items: (seed, prompt) â€” `request.count` images at the reference identity.
    // PiD output tier (sc-10054): 2K caps the effective base so PiD's fixed 4Ã— lands on ~2048 (default
    // 4K/native leaves the requested dims untouched).
    let (width, height) = pid_effective_dims(
        request.width,
        request.height,
        use_pid,
        pid_output_tier(request),
    );
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let negative_prompt = request.negative_prompt.clone();

    // SDXL's three caller-staged components (epic 13657, sc-13682): the CLIP-L/bigG tokenizers + fp16-fix
    // VAE the candle `IpAdapterSdxl` provider now REQUIRES (inference sc-13663 deleted its `hf_get`
    // self-fetch). Resolved before the engine load in the blocking closure so a missing one fails fast
    // with an actionable error naming the component id + repo, then moved into the closure with the paths.
    let (tokenizer_clip_l, tokenizer_clip_bigg, vae_fp16_fix) =
        resolve_sdxl_components(&request.model_manifest_entry, settings)?;

    // Conditioning-overlay VRAM admission (sc-16069, epic 15448). This lane loads through the UNcached
    // `start_gen_stream` with a bespoke `IpAdapterSdxlPaths`, so it is diverted around BOTH the
    // `generate_candle_stream` `vram_gate` and the `generator_cache` `apply_residency_policy` â€” before
    // this it allocated with no pre-flight check at all and died on a reactive CUDA OOM. Gated here, once
    // every resident path is resolved and before anything is moved into the load closure. The IP-Adapter
    // bundle + its CLIP image encoder are the overlay; the fp16-fix VAE and an opted-in PiD pair are
    // priced too, because they are held alongside the base (the tokenizers are JSON â€” no weights to sum).
    {
        let mut overlays = vec![
            ip_bundle.as_path(),
            image_encoder.as_path(),
            crate::conditioning_fit::weights_source_path(&vae_fp16_fix),
        ];
        overlays.extend(crate::conditioning_fit::pid_paths(pid_weights.as_ref()));
        admit_conditioning_paths(settings, "SDXL", "IP-Adapter", &sdxl_base, &overlays).await?;
    }

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "sdxl_ipadapter",
        0,
        move || {
            let paths = IpAdapterSdxlPaths {
                sdxl_base,
                ip_adapter: ip_bundle,
                image_encoder,
                tokenizer_clip_l,
                tokenizer_clip_bigg,
                vae_fp16_fix,
            };
            let model = IpAdapterSdxl::load(&paths).map_err(|error| {
                WorkerError::Engine(format!("SDXL IP-Adapter load failed: {error}"))
            })?;
            // Attach the optional PiD decoder (sc-8044): `Some` only when this generation opted in AND the
            // snapshots are cached, so a native-VAE generation is a no-op here.
            let model = match &pid_weights {
                Some(pid) => model.with_pid(pid).map_err(|error| {
                    WorkerError::Engine(format!("SDXL IP-Adapter PiD decoder load failed: {error}"))
                })?,
                None => model,
            };
            // Per-job identity-likeness scorer built ONCE here (on the blocking thread where the `!Send`
            // face stack is allowed); source embedded once, reused across every output (sc-4411 caching
            // AC). `None` â‡’ non-fatal staging / construction failure â‡’ scores omitted.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some(source)) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            Ok((model, reference, scorer))
        },
        move |(model, reference, scorer), tx, cancel| {
            // `IpAdapterSdxl::generate` takes `&mut self` (it sets the IP image tokens on the UNet before
            // the denoise), so the per-item closure mutates `model`.
            let mut model = model;
            drive_gen_items_scored(
                tx,
                work,
                move |_index, (seed, prompt), preview, on_progress| {
                    if cancel.is_cancelled() {
                        return Ok(None);
                    }
                    let req = IpAdapterSdxlRequest {
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
                        // PiD opt-in (sc-8044): in lockstep with the `with_pid` load above.
                        use_pid,
                        preview,
                        cancel: cancel.clone(),
                    };
                    let out = match model.generate(&req, &reference, &mut *on_progress) {
                        Ok(out) => out,
                        Err(_) if cancel.is_cancelled() => return Ok(None),
                        Err(error) => {
                            return Err(WorkerError::Engine(format!(
                                "SDXL IP-Adapter generation failed: {error}"
                            )));
                        }
                    };
                    // Score this finished image against the cached source embedding (sc-4411). Image build +
                    // pixel clone paid ONLY when a scorer exists; non-frontal â†’ honest detected:false N/A;
                    // `None` scorer â‡’ field omitted.
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
        SDXL_IPADAPTER_ENGINE,
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
mod sdxl_ipadapter_repo_tests {
    use super::*;

    /// sc-14463 regression guard. This lane must resolve the SAME repo the txt2img lane does, because
    /// `modelManifestEntry.repo` is ALWAYS absent in production (no built-in model declares a
    /// top-level `repo`), which makes `sdxl_ipadapter_default_repo` the only reachable branch.
    ///
    /// The `assert_ne!`s are the point: they name the exact pre-fix constants, so this test FAILS
    /// against the old code rather than merely restating whatever the function happens to return.
    /// Both named repos are absent from every manifest `downloads` block, so `huggingface_snapshot_dir`
    /// (a cache lookup, not a fetch) returned `None` and the lane declined every job to torch.
    #[test]
    fn ipadapter_default_repo_is_the_installed_turnkey_not_the_upstream_source() {
        assert_eq!(
            sdxl_ipadapter_default_repo("realvisxl"),
            "SceneWorks/realvisxl-mlx"
        );
        assert_ne!(
            sdxl_ipadapter_default_repo("realvisxl"),
            "SG161222/RealVisXL_V5.0",
            "the installer has not staged the flat upstream since the Group-B cutover (sc-8746)"
        );
        assert_eq!(
            sdxl_ipadapter_default_repo("sdxl"),
            "SceneWorks/sdxl-base-mlx"
        );
        assert_ne!(
            sdxl_ipadapter_default_repo("sdxl"),
            "stabilityai/stable-diffusion-xl-base-1.0",
            "same cutover moved the SDXL base to its turnkey"
        );
        // Illustrious was already correct; it must stay correct through the MODEL_TABLE rewire.
        assert_eq!(
            sdxl_ipadapter_default_repo("illustrious_xl_v1"),
            "SceneWorks/illustrious-xl-v1-mlx"
        );
        assert_eq!(
            sdxl_ipadapter_default_repo("illustrious_xl_v2"),
            "SceneWorks/illustrious-xl-v2-mlx"
        );
    }

    /// Every id this lane admits must have a MODEL_TABLE row â€” otherwise it silently takes the
    /// defensive `unwrap_or` floor and loads the WRONG model's weights instead of failing.
    #[test]
    fn every_admitted_model_has_a_model_table_row() {
        for model in [
            "sdxl",
            "realvisxl",
            "illustrious_xl_v1",
            "illustrious_xl_v2",
        ] {
            assert!(
                is_sdxl_ipadapter_model(model),
                "{model} must stay admitted by this lane"
            );
            assert!(
                crate::engines::default_repo_for(model).is_some(),
                "{model} is admitted but has no MODEL_TABLE row â€” it would silently take the floor"
            );
        }
    }
}
