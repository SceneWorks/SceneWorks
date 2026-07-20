// Candle (Windows/CUDA) Qwen-Image-Edit route (sc-5487, epic 5480) — reference-conditioned image
// editing on the Qwen-Image-Edit family off-Mac via `runtime_cuda::providers::qwen_image::QwenEdit`. The reference
// + edit prompt go through the Qwen2.5-VL vision-language encoder, the reference is VAE-encoded into
// the transformer's dual-latent sequence, and the MMDiT denoises a reference-respecting edit. Before
// this an off-Mac `edit_image` job on a Qwen-Image-Edit model fell back to the Python torch worker.
//
// **Candle-only.** macOS keeps the MLX `qwen_image_edit` registry path (qwen.rs). The candle `QwenEdit`
// is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build (the `include!`
// in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs` module, so it shares that
// module's imports (ImageRequest/Settings/WorkerResult/`load_reference_image`/`huggingface_snapshot_dir`/
// `resolve_app_managed_model_dir`/`resolve_seed`/`start_gen_stream`/`drive_gen_items`/
// `consume_gen_events`/`non_empty`/`gen_core`/… all in scope).
//
// Qwen-Image-Edit is a dual-latent reference concat (NOT strength-img2img + mask): the source is the
// reference, the prompt is the instruction. So this lane handles `edit_image` + `sourceAssetId` (no
// sub-modes / inpaint / outpaint — that masked shape is the SDXL edit lane's). The provider
// condition-resizes the reference internally (~384²), so — unlike the FLUX.2 lane — the source is NOT
// pre-fit to the render size.

/// Qwen-Image-Edit denoise steps default (the production, non-distilled variants).
const QWEN_EDIT_CANDLE_DEFAULT_STEPS: u32 = 30;
/// Qwen-Image-Edit-2511-Lightning few-step distill: 4-step default, matching the 4-step distill LoRA (sc-6220).
const QWEN_EDIT_CANDLE_LIGHTNING_STEPS: u32 = 4;
/// True-CFG guidance default.
const QWEN_EDIT_CANDLE_DEFAULT_GUIDANCE: f32 = 4.0;
/// The adapter/engine id recorded on candle Qwen-Image-Edit assets + telemetry.
const QWEN_EDIT_CANDLE_ENGINE: &str = "candle_qwen_edit";
/// The SceneWorks pre-packed Qwen-Image-Edit-2511 quant-matrix turnkey (sc-8669, epic 8506): self-
/// contained `q4/` (manifest default) + `q8/` + `bf16/` subdirs, only the transformer packed (the
/// Qwen2.5-VL TE + VAE stay dense bf16 in every tier). Shared by all four edit ids + the Lightning
/// distill — one checkpoint, and the ONLY repo the manifest `downloads` for those ids ever fetch.
///
/// The last-resort fallback for an id that is somehow not in [`MODEL_TABLE`]; the live ids all carry
/// this exact repo as their row `default_repo`, which [`qwen_edit_candle_repo`] reads first.
const QWEN_EDIT_CANDLE_TURNKEY_REPO: &str = "SceneWorks/qwen-image-edit-2511-mlx";
/// The lightx2v Qwen-Image-Edit-2511-Lightning distill LoRA (4-step bf16), fetched lazily into the HF
/// cache on first use — mirrors the MLX `qwen_edit_lightning` (sc-3398) repo/file.
const QWEN_EDIT_CANDLE_LIGHTNING_LORA_REPO: &str = "lightx2v/Qwen-Image-Edit-2511-Lightning";
const QWEN_EDIT_CANDLE_LIGHTNING_LORA_FILE: &str =
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors";
/// Pinned revision for the default candle Lightning distill-LoRA repo (sc-9879, F-077 follow-up).
/// Fetching the mutable `main` branch means an upstream re-push could silently swap the distill LoRA we
/// stack at load; pin the exact commit for defense-in-depth (mirrors sc-8879/sc-9682).
/// `HuggingFaceSnapshot::resolve` still verifies each file's `lfs.oid` sha256. Applied ONLY to the
/// default repo — a non-default repo keeps `main`. Matches the MLX `QWEN_LIGHTNING_LORA_REVISION`.
const QWEN_EDIT_CANDLE_LIGHTNING_LORA_REVISION: &str = "d74eba145674fd7e31b949324e148e21e7118abd";

/// Qwen-Image-Edit model ids the candle edit route accepts. The base variants map to the single edit
/// engine (the architecture is identical; `-2511` only flips `zero_cond_t`, which `QwenEdit` auto-detects
/// from `transformer/config.json`). The `-2511_lightning` distill is the same `-2511` base with the
/// lightx2v 4-step LoRA folded into the MMDiT at load + the CFG-off static-shift lightning schedule (sc-6220).
fn is_qwen_edit_candle_model(model: &str) -> bool {
    matches!(
        model,
        "qwen_image_edit"
            | "qwen_image_edit_2509"
            | "qwen_image_edit_2511"
            | "qwen_image_edit_2511_lightning"
    )
}

/// The Qwen-Image-Edit-2511-Lightning few-step distill variant (sc-6220): `QwenEdit` folds the lightx2v
/// LoRA into the MMDiT at load and runs the CFG-off lightning schedule (4 steps).
fn is_qwen_edit_lightning(model: &str) -> bool {
    model == "qwen_image_edit_2511_lightning"
}

/// True when this is a candle-eligible Qwen edit job: a Qwen-Image-Edit `edit_image` job with a source
/// image. Mirrors `jobs_store::qwen_edit_candle_eligible` so the worker and router agree on the lane.
fn qwen_edit_candle_mode(request: &ImageRequest) -> bool {
    request.mode == "edit_image" && non_empty(&request.source_asset_id)
}

/// The Qwen-Image-Edit base repo for this request: the manifest `repo` override, else this id's
/// [`MODEL_TABLE`] row `default_repo` — the SceneWorks Edit-2511 quant-matrix turnkey every edit id's
/// manifest `downloads` actually provision.
///
/// **sc-13534:** this used to default to the UPSTREAM `Qwen/Qwen-Image-Edit(-2511)` snapshot, which no
/// download flow ever fetches — the `downloads` for `qwen_image_edit_2511` + `_2511_lightning` list only
/// `SceneWorks/qwen-image-edit-2511-mlx`, and the manifest declares the turnkey under `paths.model`, NOT
/// the top-level `modelPath` [`resolve_qwen_edit_candle_base`] reads (`modelManifestEntry` passes through
/// verbatim — nothing flattens `paths.model` into `modelPath`). So the upstream default was reached on
/// every request and produced three distinct defects:
///
/// 1. **The lane was dead after a normal install** — the upstream repo is absent, so the base never
///    resolved and `qwen_edit_candle_available` was false. It only ever worked on a dev box that happened
///    to have the upstream snapshot cached from measurement work.
/// 2. **Wrong weights** — the non-Lightning ids defaulted to `Qwen/Qwen-Image-Edit`, the Aug-2025
///    ORIGINAL, not 2511. The manifest + `MODEL_TABLE` alias every edit id to the 2511 checkpoint, and
///    `zero_cond_t` is absent from the original's `transformer/config.json`, so a `qwen_image_edit_2511`
///    render would have silently taken the 2509-style single-timestep modulation.
/// 3. **The VRAM gate sized a tier that could not load** — see the fit-gate comment in
///    [`generate_candle_qwen_edit_stream`].
///
/// Reading the row keeps this catalog-driven: `MODEL_TABLE` already points all four edit ids at the
/// turnkey (the same repo the MLX lane loads), so the two lanes cannot drift apart again.
fn qwen_edit_candle_repo(request: &ImageRequest) -> String {
    request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            crate::engines::MODEL_TABLE
                .iter()
                .find(|row| row.sceneworks_id == request.model)
                .map(|row| row.default_repo.to_owned())
        })
        .unwrap_or_else(|| QWEN_EDIT_CANDLE_TURNKEY_REPO.to_owned())
}

/// Resolve the Qwen-Image-Edit base snapshot: an explicit `modelPath` dir (advanced or manifest) wins,
/// else the requested TIER SUBDIR of the [`qwen_edit_candle_repo`] turnkey snapshot. `None` means the
/// base is not present locally, so the job is not candle-runnable.
///
/// **sc-13534:** the tier descent is the fix's other half. `QwenEdit::load` takes no quant/tier argument —
/// it packed-detects per `Linear` from `transformer/config.json` (`transformer_group_size` /
/// `transformer_is_packed`), so the DIRECTORY alone decides the precision that loads. Pointing it at a
/// turnkey ROOT would hand the loader a dir holding only `q4/ q8/ bf16/` and no `transformer/`; pointing
/// it at the upstream dense snapshot (the old behavior) loaded DENSE bf16 while the fit-gate sized q4.
/// It resolves the tier through the SAME two-step chain the txt2img sibling uses in
/// [`resolve_weights_dir`] — the download RECEIPT first, then the tier descent — because those two steps
/// pick DIFFERENT tiers and only the pair reproduces the sibling's behavior. See
/// [`qwen_edit_candle_base_dir`] for why the receipt step is load-bearing.
///
/// `pub(crate)` so the real-weights GPU smoke can drive the exact seam it exercises in production
/// (`qwen_edit_worker_lane_gpu_smoke`), like `resolve_weights_dir` for the txt2img lanes.
pub(crate) fn resolve_qwen_edit_candle_base(
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
        return resolve_app_managed_model_dir(settings, &path, "Qwen edit modelPath").map(Some);
    }
    let repo = qwen_edit_candle_repo(request);
    let receipt = crate::model_jobs::huggingface_receipt_weights_dir(
        &settings.data_dir,
        &repo,
        Some(&request.model),
        requested_receipt_variant(request).as_deref(),
    );
    Ok(qwen_edit_candle_base_dir(
        receipt,
        huggingface_snapshot_dir(&settings.data_dir, &repo),
        request,
    ))
}

/// Pick the base dir from an already-resolved (`receipt`, `snapshot`) pair — the PURE half of
/// [`resolve_qwen_edit_candle_base`], split out so both steps are unit-testable (the resolver itself
/// needs a `Settings` plus the env-dependent HF cache lookup, which a test cannot drive without coupling
/// to the developer's real cache). Mirrors why [`gate_tier_key`] was extracted out of
/// `generate_candle_stream`. Production has exactly one caller, which passes the real lookups.
///
/// Order matches [`resolve_weights_dir`] exactly, and BOTH steps matter:
///
/// 1. **A tier RECEIPT wins outright.** [`requested_receipt_variant`] falls back to the manifest's
///    `default: true` download — **q4** for this family — so a stock install resolves the q4 receipt dir
///    and returns it as-is. This step is what honors the DECLARED default tier (sc-12155), and skipping it
///    was a real defect in the first cut of sc-13534: `standard_tier_subdir` reads ONLY
///    `advanced.mlxQuantize` and otherwise takes the app-wide **q8** default (sc-10726), so a box with
///    q4+q8 both installed loaded **q8 on edit while txt2img loaded q4** — same manifest, same checkpoint,
///    two different tiers, and ~12 GB more VRAM budgeted than the declared default needs.
/// 2. **Else the tier descent**, which clamps to what is INSTALLED (q8 → bf16 → q4) so a partial install
///    resolves a tier that exists instead of erroring on a missing one.
///
/// `QwenEdit::load` takes no quant/tier argument — it packed-detects per `Linear` from
/// `transformer/config.json` (`transformer_group_size` / `transformer_is_packed`), so the DIRECTORY alone
/// decides the precision. Both steps land on a self-contained tier subdir, which is EXACTLY what the
/// sc-11019 / sc-11666 `candle.vramGbByTier` rows were measured against.
///
/// Everything `None` ⇒ `None`: an absent turnkey stays "not candle-runnable" rather than becoming a bogus
/// path that fails deep inside the loader.
fn qwen_edit_candle_base_dir(
    receipt: Option<PathBuf>,
    snapshot: Option<PathBuf>,
    request: &ImageRequest,
) -> Option<PathBuf> {
    // A tier receipt already IS the exact self-contained tier dir — return it before the descent, so the
    // tier the download recorded is the tier that loads (and no second q4/q8/bf16 descent runs).
    if receipt
        .as_deref()
        .and_then(tier_key_from_resolved_dir)
        .is_some()
    {
        return receipt;
    }
    receipt
        .or(snapshot)
        .map(|root| standard_tier_subdir(&root, request))
}

/// True when this is a candle-eligible Qwen edit job (a Qwen-Image-Edit `edit_image` job with a source)
/// whose base resolves locally. Mirrors `jobs_store::qwen_edit_candle_eligible` (minus the weight-
/// resolve check).
fn qwen_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_qwen_edit_candle_model(&request.model)
        && qwen_edit_candle_mode(request)
        && matches!(
            resolve_qwen_edit_candle_base(request, settings),
            Ok(Some(_))
        )
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → family default
/// (Lightning → 4, else 30).
fn qwen_edit_candle_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32_with(
        request,
        "steps",
        || {
            if is_qwen_edit_lightning(&request.model) {
                QWEN_EDIT_CANDLE_LIGHTNING_STEPS
            } else {
                QWEN_EDIT_CANDLE_DEFAULT_STEPS
            }
        },
        1..=50,
    )
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (4.0), clamped.
fn qwen_edit_candle_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        QWEN_EDIT_CANDLE_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// Flat telemetry recorded on candle Qwen-Image-Edit assets. `base_dir` is the RESOLVED base
/// ([`resolve_qwen_edit_candle_base`]) — the tier label is read back off it, so the record names the
/// weights that actually loaded.
fn qwen_edit_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    base_dir: &Path,
    steps: u32,
    guidance: f32,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("referenceCount".to_owned(), json!(1));
    raw.insert(
        "editEngine".to_owned(),
        Value::String(QWEN_EDIT_CANDLE_ENGINE.to_owned()),
    );
    // sc-13534: record the TIER that ran, not just the repo. Every tier of this family lives in one
    // turnkey repo, so `repo` is constant while the tier is the thing that varies — without this the
    // asset record cannot say whether q4 or q8 produced the image, which is precisely the question this
    // story exists to answer. Read from the RESOLVED dir (the same basename the fit-gate budgets), so
    // the label describes what loaded rather than what was requested — mirroring `nvfp4_selected`'s
    // "the recorded label is gated on the resolver's own output" rule. `None` for a `modelPath` override
    // or any dir with no recognizable tier basename: omit the key rather than assert a tier we can't see.
    if let Some(tier) = tier_key_from_resolved_dir(base_dir) {
        raw.insert("quantTier".to_owned(), Value::String(tier.to_owned()));
    }
    raw
}

/// Load the Qwen edit source asset (the `sourceAssetId` is required) as an engine [`Image`].
fn load_qwen_edit_source(
    request: &ImageRequest,
    project_path: &Path,
    settings: &Settings,
) -> WorkerResult<Image> {
    let source_id = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Qwen edit requires a source image".to_owned())
        })?;
    load_reference_image(
        &settings.data_dir,
        &request.project_id,
        source_id,
        project_path,
    )
}

/// Ensure the lightx2v distill LoRA (`file` from HuggingFace `repo`) is materialized in the shared HF
/// hub cache, returning its absolute path (sc-6220). Fast-paths when already cached; else fetches just
/// that one file into the standard `models--<org>--<name>` layout (deduping with the Python loader +
/// other tools, sc-1904). The candle off-Mac twin of the MLX `qwen.rs::ensure_distill_lora_cached`
/// (sc-3398) — fully qualified because this file is `include!`d into the candle `image_jobs` build,
/// which does not import the MLX download helpers.
async fn ensure_qwen_lightning_lora_cached(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    repo: &str,
    file: &str,
) -> WorkerResult<PathBuf> {
    // Fast path: already materialized in the hub cache (the common case after first use).
    if let Some(snapshot_dir) =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, repo)
    {
        let candidate = snapshot_dir.join(file);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    let repo_dir = sceneworks_core::hf_home::huggingface_repo_cache_path(&settings.data_dir, repo)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Unable to resolve Hugging Face cache path for {repo}."
            ))
        })?;
    // Pin the exact commit for the default distill-LoRA repo so `main` moving under us can't swap the
    // LoRA (sc-9879). A non-default repo (none exists today, but the param is repo-agnostic) keeps `main`.
    let revision = if repo == QWEN_EDIT_CANDLE_LIGHTNING_LORA_REPO {
        QWEN_EDIT_CANDLE_LIGHTNING_LORA_REVISION
    } else {
        "main"
    };
    let client = crate::downloads::streaming_download_client();
    let snapshot = crate::downloads::HuggingFaceSnapshot::resolve(
        &client,
        settings,
        repo,
        revision,
        &[file.to_owned()],
    )
    .await?;
    if snapshot.files.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "Distill LoRA {file} not found in Hugging Face repo {repo}."
        )));
    }
    let mut progress = crate::downloads::DownloadProgress::new(
        repo,
        crate::directory_size(&repo_dir.join("blobs")).await,
        snapshot.total_bytes(),
        crate::progress_report_interval(settings),
    );
    crate::downloads::download_snapshot_into_cache(
        &crate::downloads::DownloadContext {
            api,
            client: &client,
            settings,
            job_id: &job.id,
            cancel_message: "Generation canceled while fetching the Lightning distill LoRA.",
            fresh_download: false,
        },
        &repo_dir,
        revision,
        &snapshot,
        &mut progress,
    )
    .await?;
    let snapshot_dir = crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, repo)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Hugging Face snapshot for {repo} missing after download."
            ))
        })?;
    let path = snapshot_dir.join(file);
    if !path.is_file() {
        return Err(WorkerError::InvalidPayload(format!(
            "Distill LoRA {file} missing from the {repo} snapshot after download."
        )));
    }
    Ok(path)
}

/// Real candle Qwen-Image-Edit generation: resolve the source + base on the async side, then load
/// `QwenEdit` once + generate each image on the blocking thread. The provider condition-resizes the
/// reference internally, so the source is passed as-is (no render-size pre-fit). `request.count` edits
/// of the same source, each its own seed. `generate` takes `&self`, so the per-item closure needs no
/// `mut`. Reuses [`consume_gen_events`].
async fn generate_candle_qwen_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let qwen_base = resolve_qwen_edit_candle_base(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Qwen-Image-Edit base not found".to_owned()))?;
    if !qwen_edit_candle_mode(request) {
        return Err(WorkerError::InvalidPayload(
            "Qwen edit requires edit_image mode + a source image".to_owned(),
        ));
    }
    let (width, height) = (request.width, request.height);
    let reference = load_qwen_edit_source(request, project_path, settings)?;

    let lightning = is_qwen_edit_lightning(&request.model);
    let steps = qwen_edit_candle_steps(request);
    // Lightning is CFG-distilled → run CFG-off (guidance 1.0); the provider forces a single forward when
    // `lightning` is set, so guidance is recorded for telemetry only there.
    let guidance = if lightning {
        1.0
    } else {
        qwen_edit_candle_guidance(request)
    };
    let repo = qwen_edit_candle_repo(request);
    // The lightx2v distill LoRA, lazily fetched into the HF cache — `QwenEdit` folds it into the MMDiT at
    // load (sc-6220). Empty for the production (multi-step true-CFG) variants.
    let mut adapters: Vec<AdapterSpec> = if lightning {
        let lora = ensure_qwen_lightning_lora_cached(
            api,
            settings,
            job,
            QWEN_EDIT_CANDLE_LIGHTNING_LORA_REPO,
            QWEN_EDIT_CANDLE_LIGHTNING_LORA_FILE,
        )
        .await?;
        vec![AdapterSpec::new(lora, 1.0, AdapterKind::Lora)]
    } else {
        Vec::new()
    };
    // User style/subject LoRAs (sc-10271): folded in alongside any built-in distill LoRA,
    // mirroring the MLX twin (qwen.rs) — `QwenEdit` applies the whole adapter list at load.
    // This closes the candle edit-LoRA gap for the Qwen-Image-Edit family; the SDXL / FLUX.2 /
    // Z-Image candle edit engines still need an `adapters` field in candle-gen (tracked).
    adapters.extend(resolve_adapters(request, settings)?);
    let mut raw_settings =
        qwen_edit_candle_raw_settings(request, &repo, &qwen_base, steps, guidance);
    // Record the Lightning recipe for telemetry / A-B parity (matches the MLX `distillLora` key format).
    if lightning {
        raw_settings.insert("sampler".to_owned(), Value::String("lightning".to_owned()));
        raw_settings.insert(
            "distillLora".to_owned(),
            Value::String(format!(
                "{QWEN_EDIT_CANDLE_LIGHTNING_LORA_REPO}/{QWEN_EDIT_CANDLE_LIGHTNING_LORA_FILE}"
            )),
        );
    }

    // Per-image work items: (seed, prompt) — `request.count` edits of the same source.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let negative = request.negative_prompt.clone();

    // VRAM fit-gate for the Qwen-Image-Edit lane (epic 10765 Phase 1c follow-up, sc-10968) — the edit
    // sibling of the txt2img gate (base.rs `generate_candle_stream`). The edit lane routes through THIS
    // function, not `generate_candle_stream`, so it needs its own gate: when the selected tier's predicted
    // resident peak won't fit the card, load with sequential component residency (`QwenEdit` loads the
    // VL encoder + VAE encoder → encodes + VAE-encodes the references → DROPS them before the DiT loads)
    // instead of OOMing; and if even the MEASURED sequential peak won't fit, reject-before-OOM. The edit
    // lane IS sequential-capable (sc-10968 wired `QwenEdit::generate_sequential`). Inert (Unknown →
    // resident) until the edit manifest tiers carry a `candle` block with measured peaks (the sc-10969 /
    // sc-10920 measure sibling for edit) — mirroring flux2 / qwen txt2img before their measure stories.
    let use_sequential = {
        let budget = crate::vram_gate::apply_vram_cap(
            crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
            crate::vram_gate::cuda_vram_cap_gb(),
        );
        // sc-13534: key the budget off the tier `resolve_qwen_edit_candle_base` ACTUALLY landed on, not
        // the bits the request asked for — this lane now grows the tier layout the old `nvfp4 = false`
        // note said to wire when it did. `gate_tier_key` is the shared txt2img helper (sc-12090 /
        // sc-12425), and the aliasing it exists to prevent was live here in two ways:
        //
        // * The lane loaded a DENSE upstream snapshot while the gate sized q4 — 56.7 GB budgeted
        //   (`vramGbByTier["q4"]`, measured on the PACKED turnkey q4) against a real bf16 peak of 81.7
        //   (base) / 87.4 (Lightning). A ~25-30 GB UNDER-prediction, i.e. permissive: exactly the
        //   reactive OOM the sc-10856 second-stage sequential gate exists to reject before.
        // * `standard_tier_subdir` clamps to the INSTALLED tier, so a q4 request on a box holding only
        //   `q8/` loads q8 (69.0) while the bits-derived key still said q4 (56.7) — the same
        //   under-prediction, now reachable through a perfectly ordinary partial install.
        //
        // Reading the resolved basename makes both cases self-correcting. `convrot_resolved = false`:
        // ConvRot is a Krea tier, not a Qwen-Image-Edit one. `nvfp4_selected` takes the resolved dir, so
        // an `nvfp4/` pick that fell back to an installed tier is budgeted as what it fell back TO.
        let tier = gate_tier_key(
            /* convrot_resolved */ false,
            &qwen_base,
            &request.advanced,
            &request.model_manifest_entry,
            nvfp4_selected(request, nvfp4_host_eligible(), Some(&qwen_base)),
        );
        let needed = crate::vram_gate::predicted_peak_gb(&request.model_manifest_entry, tier);
        match crate::vram_gate::resolve_offload(
            crate::vram_gate::fit_decision(needed, budget),
            /* sequential_capable */ true,
        ) {
            crate::vram_gate::FitDecision::Offload {
                needed_gb,
                available_gb,
            } => {
                // Second-stage gate (sc-10856): if this tier's MEASURED sequential peak is known and STILL
                // exceeds the budget, reject before load instead of a reactive OOM. Absent the number
                // (unmeasured edit tier) keep the best-effort sequential run.
                let sequential_needed = crate::vram_gate::predicted_sequential_peak_gb(
                    &request.model_manifest_entry,
                    tier,
                );
                if let Some(seq_gb) =
                    crate::vram_gate::sequential_overflow_gb(sequential_needed, budget)
                {
                    return Err(WorkerError::InvalidPayload(format!(
                        "{model} at the {tier} tier needs ~{seq} GB of VRAM even with sequential \
                         component residency (loading one component at a time), but GPU {gpu} has \
                         ~{available} GB available. Pick a lower tier (Q4/Q8), lower the output \
                         resolution, or run on a card with more VRAM.",
                        model = request.model,
                        seq = seq_gb.round() as i64,
                        available = available_gb.round() as i64,
                        gpu = settings.gpu_id,
                    )));
                }
                tracing::info!(
                    model = %request.model,
                    needed_gb = needed_gb.round() as i64,
                    available_gb = available_gb.round() as i64,
                    "candle Qwen-Edit VRAM fit-gate: resident peak exceeds free VRAM — loading with \
                     sequential component residency (VL encoder dropped before the DiT)"
                );
                true
            }
            crate::vram_gate::FitDecision::TooBig {
                needed_gb,
                available_gb,
            } => {
                return Err(WorkerError::InvalidPayload(format!(
                    "{model} at the {tier} tier needs ~{needed} GB of VRAM (with headroom) but GPU \
                     {gpu} has ~{available} GB available. Pick a lower tier (Q4/Q8), lower the output \
                     resolution, or run on a card with more VRAM.",
                    model = request.model,
                    needed = needed_gb.round() as i64,
                    available = available_gb.round() as i64,
                    gpu = settings.gpu_id,
                )));
            }
            _ => false,
        }
    };

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "qwen_edit",
        0,
        move || {
            let model = QwenEdit::load(&QwenEditPaths {
                root: qwen_base,
                adapters,
                // sc-10968: the fit-gate above picks sequential residency when the resident peak won't
                // fit; the provider then loads→encodes→drops the VL encoder before the DiT.
                offload_policy: if use_sequential {
                    gen_core::OffloadPolicy::Sequential
                } else {
                    gen_core::OffloadPolicy::Resident
                },
            })
            .map_err(|error| WorkerError::Engine(format!("Qwen edit load failed: {error}")))?;
            Ok((model, reference))
        },
        move |(model, reference), tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let req = QwenEditRequest {
                    prompt,
                    negative: negative.clone(),
                    width,
                    height,
                    steps: steps as usize,
                    guidance,
                    seed: seed as u64,
                    lightning,
                    cancel: cancel.clone(),
                };
                let result =
                    model.generate(&req, std::slice::from_ref(&reference), &mut *on_progress);
                let out = match result {
                    Ok(out) => out,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "Qwen edit generation failed: {error}"
                        )));
                    }
                };
                Ok(Some((seed, out.width, out.height, out.pixels)))
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
        QWEN_EDIT_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// sc-13534: the candle Qwen-Image-Edit lane must LOAD the tier its `candle.vramGbByTier` gate SIZES.
///
/// Every assertion pins a NON-default value, so "the repo is non-empty" / "the gate returns some tier"
/// style false greens are avoided. What is and is not actually mutation-killed, verified by running the
/// mutations rather than assuming:
///
/// * **Killed** — reverting [`qwen_edit_candle_repo`] to the upstream default, reverting the receipt step
///   or the tier descent inside [`qwen_edit_candle_base_dir`], or breaking [`gate_tier_key`]'s
///   resolved-dir mapping. (Confirmed RED: 4 of these tests fail with the pre-sc-13534 bodies restored.)
/// * **NOT killed** — rewiring the CALL SITES: `resolve_qwen_edit_candle_base` swapping back to a bare
///   `huggingface_snapshot_dir`, or the fit-gate swapping `gate_tier_key` back to `requested_tier_key`.
///   Both need an api/settings/job the unit layer cannot build, so they are held by compilation + review,
///   exactly as `gate_tier_key`'s own extraction note in base.rs describes. That is the residual gap here.
#[cfg(test)]
mod qwen_edit_tier_reconcile_tests {
    use super::*;
    use serde_json::json;

    /// The upstream repos this lane used to default to. No download flow ever fetches either — they are
    /// named here ONLY so the tests can assert we never resolve them again.
    const UPSTREAM_ORIGINAL: &str = "Qwen/Qwen-Image-Edit";
    const UPSTREAM_2511: &str = "Qwen/Qwen-Image-Edit-2511";

    fn request(model: &str, manifest: serde_json::Value, advanced: serde_json::Value) -> ImageRequest {
        ImageRequest::from_payload(
            json!({
                "model": model,
                "modelManifestEntry": manifest,
                "advanced": advanced,
            })
            .as_object()
            .unwrap(),
        )
    }

    /// The real manifest shape for both live edit ids (config/manifests/builtin.models.jsonc): the
    /// turnkey lives under `paths.model`, there is NO top-level `repo` and NO top-level `modelPath`, and
    /// `mlx.quantize: 4` (sc-12155) declares q4 the default tier. This exact shape is what fell through
    /// to the upstream default before sc-13534.
    fn live_manifest() -> serde_json::Value {
        json!({
            // The real `downloads` shape: ONE turnkey repo, three tier variants, q4 flagged `default`.
            // Load-bearing, not decoration — `requested_receipt_variant` derives the default tier from
            // the `default: true` entry, and there is no upstream-repo entry anywhere in this list.
            "downloads": [
                { "provider": "huggingface", "repo": "SceneWorks/qwen-image-edit-2511-mlx",
                  "variant": "q4", "default": true, "files": ["q4/*"] },
                { "provider": "huggingface", "repo": "SceneWorks/qwen-image-edit-2511-mlx",
                  "variant": "q8", "files": ["q8/*"] },
                { "provider": "huggingface", "repo": "SceneWorks/qwen-image-edit-2511-mlx",
                  "variant": "bf16", "files": ["bf16/*"] }
            ],
            "paths": { "model": "${HF_CACHE}/SceneWorks/qwen-image-edit-2511-mlx" },
            "mlx": { "quantize": 4 },
            "candle": {
                "minMemoryGb": 59,
                "vramGbByTier": { "q4": 56.7, "q8": 69.0, "bf16": 81.7 },
                "sequentialPeakGb": { "q4": 36.9, "q8": 39.3, "bf16": 52.2 },
                "measured": true
            }
        })
    }

    /// Defect 1 + 2: every edit id resolves the SceneWorks turnkey — the repo the manifest `downloads`
    /// actually provision — not an upstream snapshot nothing fetches.
    ///
    /// The `qwen_image_edit_2511` case is the wrong-WEIGHTS half: it used to fall to the family default
    /// `Qwen/Qwen-Image-Edit`, the Aug-2025 ORIGINAL rather than 2511.
    #[test]
    fn every_edit_id_resolves_the_packed_turnkey_not_an_upstream_snapshot() {
        for model in [
            "qwen_image_edit",
            "qwen_image_edit_2509",
            "qwen_image_edit_2511",
            "qwen_image_edit_2511_lightning",
        ] {
            let repo = qwen_edit_candle_repo(&request(model, live_manifest(), json!({})));
            assert_eq!(
                repo, QWEN_EDIT_CANDLE_TURNKEY_REPO,
                "{model} must load the turnkey its downloads provision"
            );
            assert_ne!(repo, UPSTREAM_ORIGINAL, "{model} regressed to the ORIGINAL upstream weights");
            assert_ne!(repo, UPSTREAM_2511, "{model} regressed to an upstream snapshot nothing downloads");
        }
    }

    /// `paths.model` is NOT a top-level `modelPath`, and nothing flattens it — the precise reason the
    /// old resolver never saw the turnkey. Pinned so a future "just read paths.model" refactor that
    /// removes the `MODEL_TABLE` lookup cannot quietly reintroduce the upstream fallthrough.
    #[test]
    fn the_manifest_declares_paths_model_not_a_top_level_model_path() {
        let request = request("qwen_image_edit_2511", live_manifest(), json!({}));
        assert!(
            request.model_manifest_entry.get("modelPath").is_none(),
            "if this ever becomes Some, the modelPath branch — not the MODEL_TABLE row — resolves the base"
        );
        assert!(request.model_manifest_entry.get("repo").is_none());
    }

    /// An explicit manifest `repo` still wins over the catalog row (the pre-existing override contract).
    #[test]
    fn an_explicit_manifest_repo_overrides_the_catalog_row() {
        let mut manifest = live_manifest();
        manifest["repo"] = json!("SceneWorks/some-other-edit-rehost");
        assert_eq!(
            qwen_edit_candle_repo(&request("qwen_image_edit_2511", manifest, json!({}))),
            "SceneWorks/some-other-edit-rehost"
        );
    }

    /// Build a turnkey root holding only the named tiers, each with a packed `transformer/`.
    fn turnkey_root(tiers: &[&str]) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        for tier in tiers {
            let dir = root.path().join(tier).join("transformer");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), b"packed").unwrap();
        }
        root
    }

    /// Defect 3, part 1: the lane descends into a TIER SUBDIR. `QwenEdit::load` packed-detects from
    /// `transformer/config.json`, so the directory alone decides the precision — and these subdirs are
    /// exactly what the `vramGbByTier` rows were measured against.
    ///
    /// Drives THIS LANE's own [`qwen_edit_candle_base_dir`] — not the shared helper underneath it — so
    /// deleting the descent from `resolve_qwen_edit_candle_base` (the pre-sc-13534 shape, which handed
    /// the loader a bare repo root) turns this RED. No receipt here: this is the descent step alone.
    #[test]
    fn the_base_resolves_a_tier_subdir_never_the_turnkey_root() {
        let root = turnkey_root(&["q4", "q8", "bf16"]);
        let at = |advanced: serde_json::Value| {
            qwen_edit_candle_base_dir(
                None,
                Some(root.path().to_path_buf()),
                &request("qwen_image_edit_2511", live_manifest(), advanced),
            )
            .expect("a present snapshot resolves a tier dir")
        };

        assert_eq!(at(json!({ "mlxQuantize": 4 })), root.path().join("q4"));
        assert_eq!(at(json!({ "mlxQuantize": 0 })), root.path().join("bf16"));

        // Whatever the default resolves to, it must be a real tier dir — the root has no `transformer/`,
        // so returning it (the old behavior) is not even loadable.
        let resolved = at(json!({}));
        assert_ne!(
            resolved,
            root.path(),
            "the turnkey root is not loadable — it holds no transformer/"
        );
        assert!(resolved.join("transformer").is_dir());

        // An absent turnkey stays absent — the lane reports "not candle-runnable" rather than
        // manufacturing a path that would fail deep inside `QwenEdit::load`.
        assert!(qwen_edit_candle_base_dir(
            None,
            None,
            &request("qwen_image_edit_2511", live_manifest(), json!({}))
        )
        .is_none());
    }

    /// Defect 3, part 2 — the under-prediction this story is named for.
    ///
    /// `gate_tier_key` must budget the tier that RESOLVED, not the bits requested. Both halves matter:
    ///
    /// * A q4 request that CLAMPS to an installed q8 (a perfectly ordinary partial install —
    ///   `standard_tier_subdir` falls back q8 → bf16 → q4) must budget q8's 69.0 GB, not q4's 56.7.
    ///   The bits-derived key said q4 and under-predicted by 12 GB.
    /// * A dir with no recognizable tier basename (the old upstream-snapshot shape) still falls back to
    ///   the request key — pinned so the fallback contract stays visible.
    #[test]
    fn the_gate_budgets_the_resolved_tier_not_the_requested_bits() {
        let manifest = live_manifest();
        let req = request("qwen_image_edit_2511", manifest.clone(), json!({ "mlxQuantize": 4 }));
        let entry = &req.model_manifest_entry;

        // Only q8 installed: the q4 REQUEST resolves q8 on disk, so the gate must size q8.
        let partial = turnkey_root(&["q8"]);
        let resolved = standard_tier_subdir_gated(partial.path(), &req, false);
        assert_eq!(resolved, partial.path().join("q8"));
        let tier = gate_tier_key(false, &resolved, &req.advanced, entry, false);
        assert_eq!(tier, "q8", "a q4 request that loaded q8 must be budgeted as q8");
        assert_eq!(
            crate::vram_gate::predicted_peak_gb(entry, tier),
            Some(69.0 + 2.0),
            "budgeting the requested q4 (56.7) for a q8 load under-predicts by ~12 GB"
        );

        // The bits-derived key — what the lane used before sc-13534 — disagrees. That disagreement IS
        // the defect; pin it so nobody "simplifies" the gate back to it.
        assert_eq!(
            crate::vram_gate::requested_tier_key(&req.advanced, entry, false),
            "q4"
        );

        // An unrecognizable basename (the old dense upstream snapshot dir) → fall back to the request key.
        let opaque = tempfile::tempdir().unwrap();
        assert_eq!(
            gate_tier_key(false, opaque.path(), &req.advanced, entry, false),
            "q4"
        );
    }

    /// The PRODUCTION default path, end to end: a stock install fetches only the `default: true` q4
    /// download, so a request with no `advanced.mlxQuantize` resolves `q4/` and the gate budgets q4's
    /// 56.7 + 2.0 GB — the manifest's declared default tier (sc-12155), the tier the sc-11019 row was
    /// measured on, and the tier that actually loads. All three finally name the same thing.
    ///
    /// This is the test that would have caught sc-13534: before the fix the load was a dense upstream
    /// bf16 snapshot (81.7 GB) while the gate said 56.7.
    #[test]
    fn a_stock_q4_only_install_loads_and_budgets_q4() {
        let stock = turnkey_root(&["q4"]);
        let req = request("qwen_image_edit_2511", live_manifest(), json!({}));

        let resolved = qwen_edit_candle_base_dir(None, Some(stock.path().to_path_buf()), &req)
            .expect("the stock turnkey resolves");
        assert_eq!(resolved, stock.path().join("q4"));

        let tier = gate_tier_key(false, &resolved, &req.advanced, &req.model_manifest_entry, false);
        assert_eq!(tier, "q4");
        assert_eq!(
            crate::vram_gate::predicted_peak_gb(&req.model_manifest_entry, tier),
            Some(56.7 + 2.0)
        );
    }

    /// **Sibling parity — the receipt step is what delivers it.** The txt2img `qwen_image` lane reaches
    /// its tier through [`resolve_weights_dir`], which resolves a download RECEIPT keyed by
    /// [`requested_receipt_variant`] and returns that tier dir BEFORE any `standard_tier_subdir` descent.
    /// With no `advanced.mlxQuantize`, that variant is the manifest's `default: true` download — **q4**.
    ///
    /// The first cut of sc-13534 wired only the descent, so on a box with q4 AND q8 installed the edit
    /// lane loaded **q8** while its txt2img sibling loaded **q4** — same checkpoint, same manifest, two
    /// tiers, and ~12 GB more VRAM budgeted than the declared default needs. Mirroring the receipt step
    /// fixes that; this test is the gate on it.
    #[test]
    fn a_tier_receipt_wins_over_the_descent_so_edit_matches_txt2img() {
        let all_tiers = turnkey_root(&["q4", "q8", "bf16"]);
        let req = request("qwen_image_edit_2511", live_manifest(), json!({}));

        // The declared default is q4 — both from the manifest `mlx.quantize` the gate reads …
        assert_eq!(
            crate::vram_gate::requested_tier_key(&req.advanced, &req.model_manifest_entry, false),
            "q4"
        );
        // … and from the `default: true` download the receipt variant resolves.
        assert_eq!(requested_receipt_variant(&req).as_deref(), Some("q4"));

        // A q4 receipt wins outright: the tier the download recorded is the tier that loads, even though
        // q8 is installed and the bare descent would have preferred it.
        let receipt = all_tiers.path().join("q4");
        let resolved = qwen_edit_candle_base_dir(
            Some(receipt.clone()),
            Some(all_tiers.path().to_path_buf()),
            &req,
        )
        .expect("the receipt resolves");
        assert_eq!(resolved, receipt);

        // Gate and load still agree, now on the DECLARED tier rather than the app-wide default.
        let tier = gate_tier_key(false, &resolved, &req.advanced, &req.model_manifest_entry, false);
        assert_eq!(tier, "q4");
        assert_eq!(
            crate::vram_gate::predicted_peak_gb(&req.model_manifest_entry, tier),
            Some(56.7 + 2.0)
        );

        // Without a receipt the descent takes the app-wide q8 default (sc-10726) — the residual
        // `mlx.quantize` vs `standard_tier_subdir` asymmetry tracked in sc-13542. Not an
        // under-prediction: the gate follows the load either way, asserted here so that stays true.
        let no_receipt = qwen_edit_candle_base_dir(None, Some(all_tiers.path().to_path_buf()), &req)
            .expect("the turnkey resolves");
        assert_eq!(no_receipt, all_tiers.path().join("q8"));
        let fallback_tier = gate_tier_key(
            false,
            &no_receipt,
            &req.advanced,
            &req.model_manifest_entry,
            false,
        );
        assert_eq!(fallback_tier, "q8");
        assert_eq!(
            crate::vram_gate::predicted_peak_gb(&req.model_manifest_entry, fallback_tier),
            Some(69.0 + 2.0)
        );
    }

    /// sc-13534: the asset record names the TIER, read off the resolved dir. Every tier of this family
    /// shares one turnkey repo, so `repo` alone cannot answer "which weights made this image".
    #[test]
    fn telemetry_records_the_tier_that_actually_loaded() {
        let root = turnkey_root(&["q4", "q8"]);
        let req = request("qwen_image_edit_2511", live_manifest(), json!({}));

        let q4 = qwen_edit_candle_raw_settings(
            &req,
            QWEN_EDIT_CANDLE_TURNKEY_REPO,
            &root.path().join("q4"),
            40,
            4.0,
        );
        assert_eq!(q4.get("quantTier").and_then(Value::as_str), Some("q4"));

        // A different tier from the SAME repo must record differently — the point of the field.
        let q8 = qwen_edit_candle_raw_settings(
            &req,
            QWEN_EDIT_CANDLE_TURNKEY_REPO,
            &root.path().join("q8"),
            40,
            4.0,
        );
        assert_eq!(q8.get("quantTier").and_then(Value::as_str), Some("q8"));
        assert_eq!(
            q4.get("repo"),
            q8.get("repo"),
            "repo is constant across tiers — which is why quantTier has to carry the signal"
        );

        // An unrecognizable dir (a `modelPath` override) omits the key rather than guessing.
        let opaque = tempfile::tempdir().unwrap();
        let unknown = qwen_edit_candle_raw_settings(
            &req,
            QWEN_EDIT_CANDLE_TURNKEY_REPO,
            opaque.path(),
            40,
            4.0,
        );
        assert!(unknown.get("quantTier").is_none());
    }
}
