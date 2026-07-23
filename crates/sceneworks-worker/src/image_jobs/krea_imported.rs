// Shared MLX/Candle in-place imported single-file Krea 2 checkpoint txt2img/img2img route
// (epic 14015 S0c, sc-14018/sc-14023/sc-14071).
// Renders a user-imported COMMUNITY checkpoint that is the Krea 2 **transformer only** (a bare DiT
// single file, e.g. a ComfyUI-exported `kreamania_variant5.safetensors`) — read in place, no copy, no
// re-download — by pairing it with a resident `krea_2` base tier that supplies the shared Qwen3-VL text
// encoder, Qwen VAE, tokenizer, and the DiT architecture config the single file omits. The assembly is
// the selected runtime's `providers::krea::load_from_native_dit_file(dit, base, descriptor)`
// — the sc-10670/10671 "read the DiT in place, source shared components from a resident tier" pattern, and
// following the candle z-image `load_from_comfyui_components` in-place assembly pattern.
//
// This is a **bespoke provider** on both backends: the loaded generator is not registry-resolvable (its
// transformer is a single in-place file, not a diffusers snapshot dir), so it bypasses the registry
// snapshot-dir descriptor path and is loaded fresh per job through `start_gen_stream` rather than the
// cached registry path — like the z-image comfyui / Wan comfyui in-place lanes. This file is `include!`d
// into the `image_jobs` module, sharing its imports.
//
// Routing (S0d, sc-14019) already marks an imported/user image model whose declared `family` is `krea_2`
// as same-family routable; this lane is what actually loads it. A builtin Krea model (`krea_2_turbo` /
// `krea_2_raw`, both in `MODEL_TABLE`) resolves through `mlx_model` and loads from its snapshot turnkey —
// `resolve_imported_krea_dit` returns `None` for it, so the existing snapshot-dir Krea path is untouched.
//
// Scope (S0c + sc-14023 + sc-14071): dense bf16 or descriptor-gated plain-int8-per-row single-file DiT,
// txt2img plus img2img (reference-guided latent-init off a single `referenceAssetId` + strength, resolved
// through the shared cross-platform `resolve_img2img_init_generic` on the SAME Turbo t2i descriptor — the
// engine keys img2img off a `Conditioning::Reference` on a non-edit descriptor, so BOTH the MLX and candle
// imported lanes get img2img). Pose / edit conditioning is still deliberately NOT claimed here (the
// imported checkpoint is a bare transformer; those need base-tier control/edit components this lane does
// not stage — imported edit is sc-14119). Descriptor contents and per-row scale shapes are validated by
// the inference loader before dequantization; ConvRot descriptors remain on their separate loader arm.

/// The adapter/engine id recorded on imported-Krea assets + telemetry (distinct from the registry
/// `krea_2_turbo` / `krea_2_raw` builtins and their bespoke edit/control/multi-phase lanes).
#[cfg(target_os = "macos")]
const KREA_IMPORTED_ENGINE: &str = "mlx_krea_imported";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_IMPORTED_ENGINE: &str = "candle_krea_imported";
/// Whether the selected backend's native single-file entrypoint accepts adapters — i.e. it serves job
/// LoRAs (sc-14111) and the Kontext edit surface (sc-14119, whose required `krea2_identity_edit` LoRA
/// IS an adapter). The MLX `load_from_native_dit_file` takes an `&[AdapterSpec]` (inference #211); the
/// candle one does NOT yet (it threads no load-time adapters — sc-14135, the candle follow-up), so the
/// candle imported lane stays **t2i / img2img only**. img2img (a `Conditioning::Reference` init) needs
/// no adapter, so it is served on both backends regardless of this flag. Read by
/// [`krea_imported_available`] (the claim gate mirrors the scheduler's
/// `imported_image_request_family_eligible(adapters_supported)`), so a candle host never routes a
/// LoRA/edit imported job into this lane.
#[cfg(target_os = "macos")]
const KREA_IMPORTED_SUPPORTS_ADAPTERS: bool = true;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_IMPORTED_SUPPORTS_ADAPTERS: bool = false;
/// The base tier whose shared Qwen3-VL text encoder + Qwen VAE + tokenizer + DiT architecture config the
/// imported single-file transformer is paired with. The Turbo turnkey (`SceneWorks/krea-2-turbo-mlx`,
/// sc-7573) is the default base — its published Krea 2 architecture matches the community merges, and its
/// `bf16/` tier ships DENSE TE/VAE that pair correctly with either supported imported DiT encoding. NOT
/// configurable:
/// the single fixed default keeps the assembly deterministic (a per-request base override is a follow-up
/// if a Raw-base community checkpoint ever needs a different shared surface).
const KREA_IMPORTED_BASE_REPO: &str = "SceneWorks/krea-2-turbo-mlx";
/// The dense `bf16/` subdir of [`KREA_IMPORTED_BASE_REPO`] — the DENSE TE/VAE tier (the `q4/`/`q8/` tiers
/// ship a packed transformer, but their TE/VAE would not pair with a dense imported DiT). Same `bf16/`
/// surface the candle INT8-ConvRot base uses (`resolve_krea_convrot`).
const KREA_IMPORTED_BASE_TIER: &str = "bf16";
/// Denoise-steps fallback — the Krea 2 Turbo distilled default (the imported community merges are
/// distilled-Turbo dense merges, like variant5). The UI normally supplies `advanced.steps`; this only
/// applies when it does not.
const KREA_IMPORTED_DEFAULT_STEPS: u32 = 8;

/// A single-file checkpoint is one on-disk `.safetensors` FILE (the imported transformer), as opposed to
/// a diffusers snapshot DIRECTORY (a builtin turnkey tier). This is the single-file-vs-snapshot-dir
/// decision at the heart of S0c: a `true` here routes to the native single-file entrypoint; a directory
/// (`false`) keeps the registry snapshot-dir path. Pure (no settings / confinement), unit-testable alone.
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
/// repo→cache-path helper. REQUIRES it installed and complete — `transformer/config.json` for the arch
/// config, plus POPULATED `text_encoder/ vae/ tokenizer/` component trees (weight files present, not
/// just the directories, so a torn base is caught here); a clear typed error otherwise so the user knows
/// to install the Krea 2 base first, rather than a raw mid-load "No such file or directory".
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
/// imported file, not this tier's), plus POPULATED `text_encoder/`, `vae/`, and `tokenizer/` component
/// trees. Each component dir is probed for an actual payload file — a `.safetensors` weight for the
/// dense TE/VAE, the `tokenizer.json` the tokenizer loads — not merely for the directory's existence:
/// a half-downloaded / torn base whose component dirs were created but never filled would otherwise pass
/// this gate and fail deep inside the S0b load with a generic Engine "load failed" instead of the
/// friendly [`resolve_krea_imported_base_tier`] "install the Krea 2 base first" typed error.
fn krea_imported_base_tier_complete(dir: &Path) -> bool {
    dir.join("transformer").join("config.json").is_file()
        && dir_has_safetensors(&dir.join("text_encoder"))
        && dir_has_safetensors(&dir.join("vae"))
        && dir.join("tokenizer").join("tokenizer.json").is_file()
}

/// True when `dir` holds at least one top-level `*.safetensors` weight file — the "is this component
/// dir actually populated, not just an empty shell left by a torn download" probe
/// [`krea_imported_base_tier_complete`] uses for the dense text encoder / VAE trees.
fn dir_has_safetensors(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        })
}

/// True when this is an in-place imported single-file Krea 2 job the selected backend can serve: an
/// imported `krea_2`-family model whose `modelPath` resolves to a single `.safetensors` DiT, in one of
/// these shapes:
///   - **txt2img** (plain), on every backend;
///   - **img2img** (mode NOT `edit_image` + a single `referenceAssetId`, sc-14071 — reference-guided
///     latent-init the shared [`resolve_img2img_init_generic`] resolves to one `Conditioning::Reference`
///     on the Turbo t2i descriptor), on every backend (no adapter needed);
///   - **LoRAs** on t2i / img2img (sc-14111) and the **Kontext edit** surface (mode `edit_image` + a
///     conditioning image, sc-14119) — ONLY on a backend whose native loader accepts adapters
///     ([`KREA_IMPORTED_SUPPORTS_ADAPTERS`]: MLX yes / candle not yet, sc-14135). This mirrors the
///     scheduler's `imported_image_request_family_eligible(adapters_supported)`, so the claim gate and
///     the router agree per backend and a candle host never routes a LoRA/edit imported job here.
///
/// Everything needing base-tier control/identity components this bare-transformer lane does NOT stage
/// stays rejected on EVERY backend: a pose set, a mask, a character / look, and a multi-phase
/// `advanced.phases` list. Outside edit mode a bare `sourceAssetId` and the plural
/// `reference_asset_ids` edit set also stay rejected — [`resolve_img2img_init_generic`] reads only
/// `reference_asset_id`, so admitting either would silently drop the source and render plain t2i.
///
/// Deliberately does NOT gate on base-tier presence: a missing base surfaces as the loud
/// [`resolve_krea_imported_base_tier`] error in the handler rather than a silent fall-through to the stub.
/// Mirrors the shape of the other `…_available` predicates.
fn krea_imported_available(request: &ImageRequest, settings: &Settings) -> bool {
    // Rejected on EVERY backend (bare-transformer lane): strict pose, inpaint mask, character / look,
    // multi-phase.
    if !pose_entries(request).is_empty()
        || request.mask_asset_id.is_some()
        || request.character_id.is_some()
        || request.character_look_id.is_some()
        || request
            .advanced
            .get("phases")
            .and_then(Value::as_array)
            .is_some_and(|phases| !phases.is_empty())
    {
        return false;
    }

    if request.mode == "edit_image" {
        // Kontext edit (sc-14119): an adapter-capable backend + a conditioning image (any of the
        // edit-reference fields, in `edit_reference_ids` priority). The required `krea2_identity_edit`
        // LoRA is enforced in the handler (R5). Inline the field probe (rather than the macOS-only
        // `edit_reference_ids`) so this shared predicate compiles on the candle lane too.
        let has_edit_reference = !request.reference_asset_ids.is_empty()
            || non_empty(&request.reference_asset_id)
            || non_empty(&request.source_asset_id);
        return KREA_IMPORTED_SUPPORTS_ADAPTERS
            && has_edit_reference
            && matches!(resolve_imported_krea_dit(request, settings), Ok(Some(_)));
    }

    // Non-edit t2i / img2img. img2img rides a single `referenceAssetId`; the plural edit set and a bare
    // `sourceAssetId` stay rejected here (the img2img resolve reads only `reference_asset_id`).
    if !request.reference_asset_ids.is_empty() || request.source_asset_id.is_some() {
        return false;
    }
    // LoRAs (sc-14111) ride the adapter path — adapter-capable backend only.
    if !request.loras.is_empty() && !KREA_IMPORTED_SUPPORTS_ADAPTERS {
        return false;
    }
    matches!(resolve_imported_krea_dit(request, settings), Ok(Some(_)))
}

/// Build the img2img conditioning for the imported Krea lane (sc-14071): a resolved reference + strength
/// becomes a single `Conditioning::Reference` — byte-identical to the generic lane's `identity_init`
/// path, which the engine routes to `generate_turbo_img2img` off a Reference on the (non-edit) Turbo t2i
/// descriptor. A plain txt2img job (`None`) yields the empty conditioning. Pure (no I/O), so the img2img
/// wiring is unit-testable without loading a real reference asset or a generator. Cross-platform (NOT
/// macOS-gated): the cross-platform [`generate_krea_imported_stream`] calls it, so the MLX and candle
/// imported lanes both thread img2img through this helper.
fn krea_imported_conditioning(img2img: Option<(Image, f32)>) -> Vec<Conditioning> {
    match img2img {
        Some((image, strength)) => vec![Conditioning::Reference {
            image,
            strength: Some(strength),
        }],
        None => Vec::new(),
    }
}

/// Flat telemetry recorded on imported-Krea assets. No guidance — the imported distilled-Turbo merges
/// are CFG-free (the Turbo descriptor advertises `supports_guidance=false`). `is_edit` records the
/// Kontext edit lane (sc-14119) vs plain t2i/img2img, and `adapter_count` the number of applied
/// LoRA/LoKr adapters (sc-14111 — the edit identity LoRA included).
fn krea_imported_raw_settings(
    request: &ImageRequest,
    steps: u32,
    is_edit: bool,
    adapter_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert(
        "mode".to_owned(),
        Value::String(
            if is_edit {
                "edit_image"
            } else {
                "text_to_image"
            }
            .to_owned(),
        ),
    );
    raw.insert("adapterCount".to_owned(), json!(adapter_count));
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

/// Resolve the adapter stack + edit conditioning for the imported lane on the adapter-capable (MLX)
/// backend (sc-14111 LoRAs + sc-14119 Kontext edit). Returns `(adapters, edit_conditioning)`:
///   - `adapters` — the job's LoRA/LoKr stack resolved into engine `AdapterSpec`s via the SHARED
///     builtin [`resolve_adapters`] (path confinement + `classify_adapter` LoKr detection + per-LoRA
///     weight); for an edit job this includes the required `krea2_identity_edit` LoRA the user selected.
///   - `edit_conditioning` — `Some(vec)` for an `edit_image` job: the fitted source reference(s) as a
///     single `Reference` or a scene+person `MultiReference`, built exactly like [`generate_krea_edit_stream`]
///     (`edit_reference_ids` → `load_reference_image` → `fit_edit_references` → `build_edit_conditioning`);
///     `None` for t2i / img2img (the caller uses [`krea_imported_conditioning`] instead).
///
/// macOS-only: the edit helpers (`edit_reference_ids` / `fit_edit_references` / `build_edit_conditioning`)
/// and the MLX native loader's adapter parameter live only on the MLX build, so the candle imported lane
/// (t2i / img2img only, sc-14135) never calls this.
#[cfg(target_os = "macos")]
fn resolve_krea_imported_adapters_and_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<(Vec<AdapterSpec>, Option<Vec<Conditioning>>)> {
    let adapters = resolve_adapters(request, settings)?;
    if request.mode != "edit_image" {
        return Ok((adapters, None));
    }
    // R5 (epic 10871): the bare transformer cannot edit without the `krea2_identity_edit` LoRA — the
    // in-context / grounded source conditioning is inert without the trained weights. Require it before
    // any compute, mirroring the builtin `generate_krea_edit_stream`.
    if !request_has_image_edit_lora(request) {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires the Krea 2 Identity Edit LoRA (or another image-edit LoRA): without \
             it the source-image conditioning is inert. Select it in the LoRA picker."
                .to_owned(),
        ));
    }
    let reference_ids = edit_reference_ids(request);
    if reference_ids.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires a source image.".to_owned(),
        ));
    }
    if reference_ids.len() > KREA_MAX_EDIT_REFERENCES {
        return Err(WorkerError::InvalidPayload(format!(
            "Krea 2 edit takes at most {KREA_MAX_EDIT_REFERENCES} images (image 1, then image 2)."
        )));
    }
    let mut sources = Vec::with_capacity(reference_ids.len());
    for id in &reference_ids {
        sources.push(load_reference_image(
            &settings.data_dir,
            &request.project_id,
            id,
            project_path,
        )?);
    }
    // Pre-fit each source to the target W×H (crop / pad / outpaint→pad), fixed order preserved — the same
    // shared edit-conditioning path the builtin lanes use.
    let sources = fit_edit_references(sources, request, request.width, request.height)?;
    Ok((adapters, Some(build_edit_conditioning(&sources))))
}

/// Real in-place imported single-file Krea 2 generation (epic 14015 S0c, sc-14023 + sc-14071 +
/// sc-14111 + sc-14119): resolve the imported DiT, the resident base tier, any img2img reference, and —
/// on the adapter-capable MLX backend — the job LoRA stack + Kontext edit conditioning, then load the
/// selected runtime's native entrypoint once and generate each image on the blocking thread.
///
/// Three shapes ride one lane: plain **t2i**, reference-guided **img2img** (one `Conditioning::Reference`
/// on the Turbo t2i descriptor, both backends), and — MLX only — **LoRA-adapted** t2i/img2img (sc-14111)
/// and the **Kontext edit** surface (sc-14119: the `turbo_edit_descriptor` + the fitted source
/// reference(s) as `Reference`/`MultiReference` + the `krea2_identity_edit` adapter). The merge is
/// distilled Turbo (no CFG / negative prompt). The `Box<dyn Generator>` is bespoke (not registry-cached).
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

    let is_edit = request.mode == "edit_image";

    // img2img reference-guided latent-init (sc-14071): the SAME generic seam the builtin Krea Turbo
    // img2img lane uses (`resolve_generic_lane_conditioning`'s generic arm), and it is CROSS-PLATFORM —
    // `model_supports_img2img` + `resolve_img2img_init_generic` are the shared candle/MLX helpers, so BOTH
    // the MLX and candle imported lanes get img2img. Resolved on the async side (decode → `Send` `Image`
    // moved into the worker thread). Only for a NON-edit job; an edit resolves its own conditioning below.
    let img2img = if model_supports_img2img(request) && !is_edit {
        resolve_img2img_init_generic(request, settings, project_path)?
    } else {
        None
    };

    // Adapter-capable backend (MLX, inference #211): resolve the job LoRA stack into engine `AdapterSpec`s
    // (sc-14111) and, for an `edit_image` job, the fitted source reference(s) + the required identity-edit
    // adapter (sc-14119). Candle takes no adapters (sc-14135), so it stays t2i/img2img with an empty stack.
    #[cfg(target_os = "macos")]
    let (adapters, edit_conditioning) =
        resolve_krea_imported_adapters_and_edit(request, settings, project_path)?;
    #[cfg(target_os = "macos")]
    let adapter_count = adapters.len();
    #[cfg(not(target_os = "macos"))]
    let adapter_count = 0usize;

    let (width, height) = (request.width, request.height);
    let steps =
        resolve_advanced_or_manifest_u32(request, "steps", KREA_IMPORTED_DEFAULT_STEPS, 1..=100);
    let raw_settings = krea_imported_raw_settings(request, steps, is_edit, adapter_count);

    // Per-image work items: (seed, prompt) — `request.count` renders, each its own seed.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        KREA_IMPORTED_ENGINE,
        adapter_count,
        move || {
            // The S0b entrypoint reads the DiT from the single file, key-remaps native→diffusers,
            // coverage/shape-validates it against the base tier's Krea 2 geometry (fail-closed — the
            // architecture-compatibility check happens here, before pairing), installs any adapters onto
            // the DiT (MLX), and sources the shared TE/VAE/tokenizer from `base_dir`. Descriptor: the
            // distilled-Turbo t2i surface (`variant5` dense / `variant4` plain-int8 are Turbo merges), or
            // the CFG-free Turbo **edit** surface (`turbo_edit_descriptor`) for an `edit_image` job.
            #[cfg(target_os = "macos")]
            let loaded = {
                let descriptor = if is_edit {
                    runtime_macos::providers::krea::turbo_edit_descriptor()
                } else {
                    runtime_macos::providers::krea::descriptor()
                };
                runtime_macos::providers::krea::load_from_native_dit_file(
                    &dit, &base_dir, &adapters, descriptor,
                )
            };
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            let loaded = runtime_cuda::providers::krea::load_from_native_dit_file(
                &dit,
                &base_dir,
                runtime_cuda::providers::krea::descriptor(),
            );
            let model = loaded
            .map_err(|error| {
                WorkerError::Engine(format!("Krea 2 imported checkpoint load failed: {error}"))
            })?;
            Ok(model)
        },
        move |model, tx, cancel| {
            // Build the conditioning once, then clone it per rendered image: the Kontext edit
            // `Reference`/`MultiReference` for an edit job (MLX), else the img2img `Reference` (or empty
            // for plain t2i).
            #[cfg(target_os = "macos")]
            let conditioning =
                edit_conditioning.unwrap_or_else(|| krea_imported_conditioning(img2img));
            #[cfg(not(target_os = "macos"))]
            let conditioning = krea_imported_conditioning(img2img);
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
                    conditioning: conditioning.clone(),
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
