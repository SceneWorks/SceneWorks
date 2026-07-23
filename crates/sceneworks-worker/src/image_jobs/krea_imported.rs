// macOS (MLX) in-place imported single-file Krea 2 checkpoint txt2img route (epic 14015 S0c, sc-14018).
// Renders a user-imported COMMUNITY checkpoint that is the Krea 2 **transformer only** (a bare DiT
// single file, e.g. a ComfyUI-exported `kreamania_variant5.safetensors`) — read in place, no copy, no
// re-download — by pairing it with a resident `krea_2` base tier that supplies the shared Qwen3-VL text
// encoder, Qwen VAE, tokenizer, and the DiT architecture config the single file omits. The assembly is
// the S0b MLX entrypoint `runtime_macos::providers::krea::load_from_native_dit_file(dit, base, descriptor)`
// — the sc-10670/10671 "read the DiT in place, source shared components from a resident tier" pattern, and
// the MLX twin of the candle z-image `load_from_comfyui_components` lane (`zimage_comfyui_candle.rs`).
//
// **macOS-only**, and a **bespoke provider**: the loaded generator is not registry-resolvable (its
// transformer is a single in-place file, not a diffusers snapshot dir), so it bypasses the registry
// snapshot-dir descriptor path and is loaded fresh per job through `start_gen_stream` rather than the
// cached registry path — like the z-image comfyui / Wan comfyui in-place lanes. This file is `include!`d
// into the `image_jobs` module, sharing its imports.
//
// Routing (S0d, sc-14019) already marks an imported/user image model whose declared `family` is `krea_2`
// as MLX-routable; this lane is what actually loads it. A builtin Krea model (`krea_2_turbo` /
// `krea_2_raw`, both in `MODEL_TABLE`) resolves through `mlx_model` and loads from its snapshot turnkey —
// `resolve_imported_krea_dit` returns `None` for it, so the existing snapshot-dir Krea path is untouched.
//
// Scope (S0c): dense bf16 single-file DiT, txt2img only (the imported checkpoint is a bare transformer,
// so pose / reference / edit conditioning is deliberately NOT claimed here — S0d did not claim those
// features for imported models either). The 26 GB load + render is validated on GPU in S0f.

/// The adapter/engine id recorded on imported-Krea assets + telemetry (distinct from the registry
/// `krea_2_turbo` / `krea_2_raw` builtins and their bespoke edit/control/multi-phase lanes).
#[cfg(target_os = "macos")]
const KREA_IMPORTED_ENGINE: &str = "mlx_krea_imported";
/// The base tier whose shared Qwen3-VL text encoder + Qwen VAE + tokenizer + DiT architecture config the
/// imported single-file transformer is paired with. The Turbo turnkey (`SceneWorks/krea-2-turbo-mlx`,
/// sc-7573) is the default base — its published Krea 2 architecture matches the community merges, and its
/// `bf16/` tier ships DENSE TE/VAE that pair correctly with the imported dense bf16 DiT. NOT configurable:
/// the single fixed default keeps the assembly deterministic (a per-request base override is a follow-up
/// if a Raw-base community checkpoint ever needs a different shared surface).
#[cfg(target_os = "macos")]
const KREA_IMPORTED_BASE_REPO: &str = "SceneWorks/krea-2-turbo-mlx";
/// The dense `bf16/` subdir of [`KREA_IMPORTED_BASE_REPO`] — the DENSE TE/VAE tier (the `q4/`/`q8/` tiers
/// ship a packed transformer, but their TE/VAE would not pair with a dense imported DiT). Same `bf16/`
/// surface the candle INT8-ConvRot base uses (`resolve_krea_convrot`).
#[cfg(target_os = "macos")]
const KREA_IMPORTED_BASE_TIER: &str = "bf16";
/// Denoise-steps fallback — the Krea 2 Turbo distilled default (the imported community merges are
/// distilled-Turbo dense merges, like variant5). The UI normally supplies `advanced.steps`; this only
/// applies when it does not.
#[cfg(target_os = "macos")]
const KREA_IMPORTED_DEFAULT_STEPS: u32 = 8;

/// A single-file checkpoint is one on-disk `.safetensors` FILE (the imported transformer), as opposed to
/// a diffusers snapshot DIRECTORY (a builtin turnkey tier). This is the single-file-vs-snapshot-dir
/// decision at the heart of S0c: a `true` here routes to the native single-file entrypoint; a directory
/// (`false`) keeps the registry snapshot-dir path. Pure (no settings / confinement), unit-testable alone.
#[cfg(target_os = "macos")]
fn is_single_file_checkpoint(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
}

/// A diffusers snapshot / turnkey tier directory — a `model_index.json` / `config.json` pipeline marker
/// or a `transformer/` component subtree. Such a dir is a SNAPSHOT (the registry path), never a
/// single-file import, so it is excluded from the native entrypoint even when it also holds loose
/// `.safetensors` shards.
#[cfg(target_os = "macos")]
fn is_diffusers_snapshot_dir(dir: &Path) -> bool {
    dir.join("model_index.json").is_file()
        || dir.join("config.json").is_file()
        || dir.join("transformer").is_dir()
}

/// The single-file DiT to load from a resolved weights location: the path itself when it is a single
/// `.safetensors` FILE, or the LONE top-level `.safetensors` inside a single-file install DIRECTORY (the
/// model-import job writes the imported checkpoint plus an install marker into
/// `<data>/models/imports/<name>/`, so the checkpoint is the one weight file there). `None` for a
/// diffusers snapshot dir (a builtin turnkey tier — [`is_diffusers_snapshot_dir`]), a dir with zero or
/// more than one top-level `.safetensors`, or a non-safetensors file — those are not a single-file import.
#[cfg(target_os = "macos")]
fn imported_dit_file(path: &Path) -> Option<PathBuf> {
    if is_single_file_checkpoint(path) {
        return Some(path.to_path_buf());
    }
    if !path.is_dir() || is_diffusers_snapshot_dir(path) {
        return None;
    }
    let mut found: Option<PathBuf> = None;
    for entry in std::fs::read_dir(path).ok()?.flatten() {
        let candidate = entry.path();
        if is_single_file_checkpoint(&candidate) {
            if found.is_some() {
                // More than one loose weight file → not the single-file shape the S0b loader takes.
                return None;
            }
            found = Some(candidate);
        }
    }
    found
}

/// Resolve the imported single-file Krea 2 DiT for `request`, or `None` when this is not an imported
/// single-file Krea job. `Some(file)` only when ALL hold:
///   - the model's declared `family` is `krea_2` (the S0d route-by-family family),
///   - the id is NOT a builtin engine model (`mlx_model` is `None`) — a builtin Krea loads from its
///     snapshot turnkey, never a single file, so this keeps the existing snapshot-dir path untouched,
///   - the model's weights location — an explicit `modelPath` (advanced or manifest) wins, else the
///     manifest entry's `paths.model` install dir the model-import job records — resolves, confined to
///     an app-managed root, to a single `.safetensors` DiT ([`imported_dit_file`]): the file directly,
///     or the lone weight file inside its single-file install dir, but NOT a diffusers snapshot dir.
///
/// Each path is confined by `normalize_app_managed_model_path` (a payload can never point the checkpoint
/// outside a declared root; LAN jobs API, epic 4484) — the same confinement `resolve_weights_dir` uses.
#[cfg(target_os = "macos")]
fn resolve_imported_krea_dit(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        != Some("krea_2")
    {
        return Ok(None);
    }
    // A builtin Krea engine id (in `MODEL_TABLE`) loads from its snapshot turnkey via the normal MLX
    // lane — never through the single-file entrypoint. Leaving those to the existing path is what keeps
    // builtin Krea rendering byte-identical (S0c requirement #3).
    if mlx_model(&request.model).is_some() {
        return Ok(None);
    }
    // An explicit `modelPath` (a future assembler could pin the file directly) wins; otherwise the
    // import job's recorded install dir (`paths.model`), which holds the single-file checkpoint.
    let Some(raw_path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .or_else(|| {
            request
                .model_manifest_entry
                .get("paths")
                .and_then(|paths| paths.get("model"))
        })
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let path = crate::paths::normalize_app_managed_model_path(
        settings,
        raw_path,
        "Imported Krea 2 checkpoint",
    )?;
    Ok(imported_dit_file(&path))
}

/// Resolve the resident `krea_2` base tier snapshot dir that supplies the shared text encoder, VAE,
/// tokenizer, and DiT architecture config the imported single-file transformer omits — the `base_snapshot_dir`
/// argument of the S0b entrypoint. The default base is the Turbo turnkey's dense `bf16/` tier
/// ([`KREA_IMPORTED_BASE_REPO`] / [`KREA_IMPORTED_BASE_TIER`]), resolved from the HF cache via the shared
/// repo→cache-path helper. REQUIRES it installed and complete (`transformer/config.json` for the arch
/// config, plus `text_encoder/ vae/ tokenizer/`); a clear typed error otherwise so the user knows to
/// install the Krea 2 base first, rather than a raw mid-load "No such file or directory".
#[cfg(target_os = "macos")]
fn resolve_krea_imported_base_tier(settings: &Settings) -> WorkerResult<PathBuf> {
    let base_missing = || {
        WorkerError::InvalidPayload(
            "Krea 2 base model is not installed — install the Krea 2 (Turbo) base model first. An \
             imported Krea 2 checkpoint is the transformer only; it is paired with the base model's \
             text encoder, VAE, and tokenizer to run."
                .to_owned(),
        )
    };
    let base = huggingface_snapshot_dir(&settings.data_dir, KREA_IMPORTED_BASE_REPO)
        .map(|root| root.join(KREA_IMPORTED_BASE_TIER))
        .filter(|dir| krea_imported_base_tier_complete(dir))
        .ok_or_else(base_missing)?;
    Ok(base)
}

/// The base tier is loadable when it carries the shared components the single-file DiT pairs with: the
/// transformer's `config.json` (the arch config `Krea2Config::from_snapshot` reads — the WEIGHTS are the
/// imported file, not this tier's), plus the `text_encoder/ vae/ tokenizer/` component trees.
#[cfg(target_os = "macos")]
fn krea_imported_base_tier_complete(dir: &Path) -> bool {
    dir.join("transformer").join("config.json").is_file()
        && dir.join("text_encoder").is_dir()
        && dir.join("vae").is_dir()
        && dir.join("tokenizer").is_dir()
}

/// True when this is an in-place imported single-file Krea 2 **txt2img** job: an imported `krea_2`-family
/// model whose `modelPath` resolves to a single `.safetensors` DiT, with no edit / pose / reference
/// (the imported checkpoint is a bare transformer — those conditioning modes need base-tier components
/// this lane does not stage). Deliberately does NOT gate on base-tier presence: a missing base surfaces
/// as the loud [`resolve_krea_imported_base_tier`] error in the handler rather than a silent fall-through
/// to the stub. Mirrors the shape of the other `…_available` predicates.
#[cfg(target_os = "macos")]
fn krea_imported_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.mode != "edit_image"
        && pose_entries(request).is_empty()
        && request.reference_asset_id.is_none()
        && request.reference_asset_ids.is_empty()
        && request.source_asset_id.is_none()
        && matches!(resolve_imported_krea_dit(request, settings), Ok(Some(_)))
}

/// Flat telemetry recorded on imported-Krea assets. No guidance — the imported distilled-Turbo merges
/// are CFG-free (the Turbo descriptor advertises `supports_guidance=false`).
#[cfg(target_os = "macos")]
fn krea_imported_raw_settings(request: &ImageRequest, steps: u32) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("mode".to_owned(), Value::String("text_to_image".to_owned()));
    raw.insert(
        "engine".to_owned(),
        Value::String(KREA_IMPORTED_ENGINE.to_owned()),
    );
    raw.insert(
        "importedCheckpoint".to_owned(),
        Value::String(request.model.clone()),
    );
    raw.insert(
        "kreaImportedBase".to_owned(),
        Value::String(format!("{KREA_IMPORTED_BASE_REPO}#{KREA_IMPORTED_BASE_TIER}")),
    );
    raw
}

/// Real MLX in-place imported single-file Krea 2 txt2img generation (epic 14015 S0c): resolve the
/// imported DiT + the resident base tier on the async side, then load the S0b native entrypoint once +
/// generate each image on the blocking thread. `request.count` images, each its own seed. The imported
/// merge is distilled Turbo (no CFG / negative prompt). The loaded `Box<dyn Generator>` is bespoke (not
/// registry-cached), driven like the z-image comfyui lane.
#[cfg(target_os = "macos")]
async fn generate_krea_imported_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let dit = resolve_imported_krea_dit(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Imported Krea 2 checkpoint could not be resolved (family/modelPath/single-file)"
                .to_owned(),
        )
    })?;
    // Require the resident base tier before any compute — a clear "install the Krea 2 base first" error.
    let base_dir = resolve_krea_imported_base_tier(settings)?;

    let (width, height) = (request.width, request.height);
    let steps =
        resolve_advanced_or_manifest_u32(request, "steps", KREA_IMPORTED_DEFAULT_STEPS, 1..=100);
    let raw_settings = krea_imported_raw_settings(request, steps);

    // Per-image work items: (seed, prompt) — `request.count` renders, each its own seed.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        KREA_IMPORTED_ENGINE,
        0,
        move || {
            // Turbo descriptor (`variant5` and its siblings are distilled-Turbo dense merges). The S0b
            // entrypoint reads the DiT from the single file, key-remaps native→diffusers, coverage/
            // shape-validates it against the base tier's Krea 2 geometry (fail-closed — the
            // architecture-compatibility check happens here, before pairing), and sources the shared
            // TE/VAE/tokenizer from `base_dir`.
            let descriptor = runtime_macos::providers::krea::descriptor();
            let model = runtime_macos::providers::krea::load_from_native_dit_file(
                &dit, &base_dir, descriptor,
            )
            .map_err(|error| {
                WorkerError::Engine(format!("Krea 2 imported checkpoint load failed: {error}"))
            })?;
            Ok(model)
        },
        move |model, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let request = GenerationRequest {
                    prompt,
                    width,
                    height,
                    count: 1,
                    seed: Some(seed as u64),
                    steps: Some(steps),
                    cancel: cancel.clone(),
                    ..Default::default()
                };
                let output = match model.generate(&request, &mut *on_progress) {
                    Ok(output) => output,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "Krea 2 imported checkpoint generation failed: {error}"
                        )));
                    }
                };
                match output {
                    GenerationOutput::Images(mut images) => {
                        let image = images.pop().ok_or_else(|| {
                            WorkerError::Engine(
                                "Krea 2 imported checkpoint produced no image".to_owned(),
                            )
                        })?;
                        Ok(Some((seed, image.width, image.height, image.pixels)))
                    }
                    _ => Err(WorkerError::Engine(
                        "Krea 2 imported checkpoint returned non-image output".to_owned(),
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
        KREA_IMPORTED_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
