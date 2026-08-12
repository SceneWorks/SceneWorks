// Shared MLX/Candle in-place imported single-file Krea 2 checkpoint txt2img/img2img route
// (epic 14015 S0c, sc-14018/sc-14023/sc-14071).
// Renders a user-imported COMMUNITY checkpoint that is the Krea 2 **transformer only** (a bare DiT
// single file, e.g. a ComfyUI-exported `kreamania_variant5.safetensors`) — read in place, no copy, no
// re-download — by pairing it with a resident `krea_2` base tier that supplies the shared Qwen3-VL text
// encoder, Qwen VAE, tokenizer, and the DiT architecture config the single file omits. The assembly is
// a normal registry `LoadSpec` whose primary source is `WeightsSource::File(dit)` and whose
// `base_snapshot` companion supplies the shared components.
//
// The imported and snapshot shapes now share the same provider ids, cache, fit gate, residency policy,
// and planner seam (sc-18306). The old inference entrypoints remain construction shims only.
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
// imported lanes get img2img). MLX also claims strict pose conditioning by composing the cached base-tier
// control branch with the imported DiT; candle still rejects pose because its native loader has no control
// parameter. Edit conditioning remains a separate surface (sc-14119). Descriptor contents and per-row scale
// shapes are validated by the inference loader before dequantization; ConvRot descriptors remain on their
// separate loader arm.

/// The adapter/engine id recorded on imported-Krea assets + telemetry (distinct from the registry
/// `krea_2_turbo` / `krea_2_raw` builtins and their bespoke edit/control/multi-phase lanes).
#[cfg(target_os = "macos")]
const KREA_IMPORTED_ENGINE: &str = "mlx_krea_imported";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_IMPORTED_ENGINE: &str = "candle_krea_imported";
fn krea_imported_operation(request: &ImageRequest) -> gen_core::ImportedModelOperation {
    if request_has_multiphase(request) {
        gen_core::ImportedModelOperation::MultiPhase
    } else if !pose_entries(request).is_empty() {
        gen_core::ImportedModelOperation::Pose
    } else if request.mode == "edit_image" {
        gen_core::ImportedModelOperation::Edit
    } else {
        gen_core::ImportedModelOperation::Generate
    }
}

/// Resolve the exact imported source shape and operation through the live provider registry. An
/// absent row is an explicit refusal; sibling routes for the family are never unioned.
fn krea_imported_descriptor(request: &ImageRequest) -> Option<gen_core::ModelDescriptor> {
    crate::inference_runtime::imported_model_descriptor(
        "krea_2",
        gen_core::ImportedModelSource::TransformerFile,
        krea_imported_operation(request),
    )
}
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
// The pinned Krea runtime's curated sampler falls back to Euler when `GenerationRequest.sampler` is
// absent. Name it explicitly in both the request and telemetry so execution and exported metadata
// cannot drift if that library default changes.
const KREA_IMPORTED_SAMPLER: &str = "euler";

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
/// Both the supplied location and the checkpoint selected from an install directory are confined by
/// `pin_app_managed_model_file` (a payload or child symlink can never point the checkpoint outside a
/// declared root; LAN jobs API, epic 4484), while retaining the lexical entry for lstat-pinned
/// re-opening (sc-18306).
#[cfg(all(target_os = "macos", test))]
fn resolve_imported_krea_dit(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    Ok(resolve_imported_krea_dit_pin(request, settings)?
        .map(|pin| pin.loader_path().to_path_buf()))
}

fn resolve_imported_krea_dit_pin(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<gen_core::PinnedWeightsFile>> {
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
    let confined = crate::paths::normalize_app_managed_model_path(
        settings,
        raw_path,
        "Imported Krea 2 checkpoint",
    )?;
    let candidate = if confined.is_dir() {
        // Select against the confined target, but retain the caller's lexical directory spelling in
        // the loader entry. Besides preserving extension/path dispatch, this lets the token fingerprint
        // every caller-visible parent component instead of silently switching to the canonical alias.
        imported_dit_file(&confined).and_then(|dit| {
            dit.file_name()
                .map(|file_name| Path::new(raw_path).join(file_name))
        })
    } else if imported_dit_file(Path::new(raw_path)).is_some() {
        Some(PathBuf::from(raw_path))
    } else {
        None
    };
    candidate
        .map(|dit| {
            crate::paths::pin_app_managed_model_file(
                settings,
                &dit,
                "Imported Krea 2 checkpoint",
            )
        })
        .transpose()
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
///   - a **strict-pose set** (a non-empty `advanced.poses` outside edit mode) — ONLY on a backend
///     whose native loader can assemble the pose control branch around the file-loaded DiT
///     ([`KREA_IMPORTED_SUPPORTS_POSE_CONTROL`]: MLX yes / candle no). The base-tier control overlay
///     is staged alongside the TE/VAE/tokenizer the lane already resolves; an optional single
///     `referenceAssetId` on a pose job is the identity-likeness scoring source (the builtin
///     `krea_control_available` semantics), never an img2img init.
///
/// Everything needing base-tier identity components this lane does NOT stage stays rejected on EVERY
/// backend: a mask, a character / look, and a multi-phase `advanced.phases` list. Outside edit mode a
/// bare `sourceAssetId` and the plural `reference_asset_ids` edit set also stay rejected —
/// [`resolve_img2img_init_generic`] reads only `reference_asset_id`, so admitting either would
/// silently drop the source and render plain t2i (and the pose lane reads neither).
///
/// Deliberately does NOT gate on base-tier presence: a missing base surfaces as the loud
/// [`resolve_krea_imported_base_tier`] error in the handler rather than a silent fall-through to the stub.
/// Mirrors the shape of the other `…_available` predicates.
fn krea_imported_request_shape_available(request: &ImageRequest) -> bool {
    let Some(descriptor) = krea_imported_descriptor(request) else {
        return false;
    };
    let caps = &descriptor.capabilities;
    if imported_model_quant(request, &descriptor, "Imported Krea 2").is_err() {
        return false;
    }
    // These identities need asset-resolution paths the imported handler does not own.
    if request.mask_asset_id.is_some()
        || request.character_id.is_some()
        || request.character_look_id.is_some()
    {
        return false;
    }
    if !request.loras.is_empty() && !(caps.supports_lora || caps.supports_lokr) {
        return false;
    }
    if request_has_multiphase(request) {
        return request.mode != "edit_image"
            && pose_entries(request).is_empty()
            && request.reference_asset_ids.is_empty()
            && request.reference_asset_id.is_none()
            && request.source_asset_id.is_none();
    }
    // A pose set inside edit mode is no lane anywhere (the builtin edit lane rejects it too), and
    // outside edit mode it requires a registered provider accepting Control conditioning.
    if !pose_entries(request).is_empty() {
        if request.mode == "edit_image"
            || !caps
                .conditioning
                .contains(&gen_core::ConditioningKind::Control)
        {
            return false;
        }
        // The plural edit set and a bare `sourceAssetId` would be silently dropped by the pose
        // render loop, so they stay rejected; a single `referenceAssetId` is the likeness source.
        if !request.reference_asset_ids.is_empty() || request.source_asset_id.is_some() {
            return false;
        }
        return true;
    }

    if request.mode == "edit_image" {
        // Kontext edit (sc-14119): an adapter-capable backend + a conditioning image (any of the
        // edit-reference fields, in `edit_reference_ids` priority). The required `krea2_identity_edit`
        // LoRA is enforced in the handler (R5). Inline the field probe (rather than the macOS-only
        // `edit_reference_ids`) so this shared predicate compiles on the candle lane too.
        let has_edit_reference = !request.reference_asset_ids.is_empty()
            || non_empty(&request.reference_asset_id)
            || non_empty(&request.source_asset_id);
        let accepts_reference = caps
            .conditioning
            .contains(&gen_core::ConditioningKind::Reference);
        let accepts_multi = caps
            .conditioning
            .contains(&gen_core::ConditioningKind::MultiReference);
        return has_edit_reference
            && accepts_reference
            && (request.reference_asset_ids.len() <= 1 || accepts_multi);
    }

    // Non-edit t2i / img2img. img2img rides a single `referenceAssetId`; the plural edit set and a bare
    // `sourceAssetId` stay rejected here (the img2img resolve reads only `reference_asset_id`).
    if !request.reference_asset_ids.is_empty() || request.source_asset_id.is_some() {
        return false;
    }
    if request.reference_asset_id.is_some()
        && !caps
            .conditioning
            .contains(&gen_core::ConditioningKind::Reference)
    {
        return false;
    }
    true
}

#[cfg(test)]
fn krea_imported_available(request: &ImageRequest, settings: &Settings) -> bool {
    krea_imported_request_shape_available(request)
        && matches!(resolve_imported_krea_dit_pin(request, settings), Ok(Some(_)))
}

/// True when this is an imported single-file Krea 2 **strict-pose** job the MLX backend serves: a
/// non-edit pose set on an imported `krea_2`-family checkpoint that the imported request-shape gate
/// admits (which already requires [`KREA_IMPORTED_SUPPORTS_POSE_CONTROL`] for a pose shape). Split
/// out so the router can claim the pose set into its own route arm ([`ImageRoute::KreaImportedControl`],
/// one image per pose) ahead of the plain per-image [`ImageRoute::KreaImported`] t2i/img2img arm.
#[cfg(all(target_os = "macos", test))]
fn krea_imported_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && krea_imported_available(request, settings)
}

#[derive(Debug)]
struct PreparedKreaImportedSources {
    dit_pin: gen_core::PinnedWeightsFile,
    prepared_adapters: PreparedAdapters,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct PreparedKreaImportedControlSources {
    dit_pin: gen_core::PinnedWeightsFile,
    control_pin: gen_core::PinnedWeightsFile,
    prepared_adapters: PreparedAdapters,
}

/// Resolve the pose control-branch overlay for the imported lane **cache-only** (epic 17625 AC9: a
/// render must never fetch weights). Resolution order mirrors the builtin
/// [`ensure_krea_control_weights`] minus its download tail: the `SCENEWORKS_CONTROLNET_KREA` env →
/// an `advanced.controlWeights.path` (a studio-trained / registered LOCAL overlay the API resolved,
/// already root-confined by [`krea_control_payload_overlay_path`]) → the installed hosted overlay
/// (an `advanced.controlWeights.{repo,filename}` override with its catalog-authorized revision, else
/// the default published beta pin) read out of the Model-Manager-populated cache via the shared
/// cache-only [`crate::downloads::resolve_hf_component_file`].
///
/// A missing install is an actionable "install it from the Model Manager" error naming the exact
/// component — the overlay is declared in `config/manifests/builtin.control_overlays.jsonc`, so it
/// installs like any other model — rather than a silent multi-hundred-megabyte fetch inside a job.
/// This is why the imported lane does NOT reuse the builtin lane's download-on-first-use helper:
/// that helper is a grandfathered job-time download site, and `job_time_download_guard` forbids new
/// ones.
#[cfg(all(target_os = "macos", test))]
fn resolve_krea_imported_control_overlay(
    settings: &Settings,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    Ok(resolve_krea_imported_control_overlay_pin(settings, request)?
        .loader_path()
        .to_path_buf())
}

/// Pin the payload-selected overlay from its lexical entry. A missing confined path preserves the
/// existing hosted-overlay fallback, while an existing path is pinned and two-sided-confined in one
/// operation so resolution cannot erase or race the entry identity.
#[cfg(target_os = "macos")]
fn pin_krea_imported_payload_control_overlay(
    settings: &Settings,
    request: &ImageRequest,
) -> WorkerResult<Option<gen_core::PinnedWeightsFile>> {
    let Some(path) = krea_control_payload_overlay_raw_path(request) else {
        return Ok(None);
    };
    if path.is_file() {
        return crate::paths::pin_app_managed_model_file(
            settings,
            &path,
            "Krea 2 pose ControlNet overlay",
        )
        .map(Some);
    }
    // Keep the historical missing-path fallback, but do not let an out-of-root missing payload avoid
    // the same confinement rejection an existing payload receives.
    crate::paths::normalize_app_managed_model_path(
        settings,
        path.to_string_lossy().as_ref(),
        "Krea 2 pose ControlNet overlay",
    )?;
    Ok(None)
}

#[cfg(target_os = "macos")]
fn resolve_krea_imported_control_overlay_pin(
    settings: &Settings,
    request: &ImageRequest,
) -> WorkerResult<gen_core::PinnedWeightsFile> {
    if let Ok(path) = std::env::var(KREA_CONTROL_WEIGHTS_ENV) {
        let path = PathBuf::from(path.trim());
        if path.is_file() {
            return crate::paths::pin_operator_model_file(
                &path,
                "SCENEWORKS_CONTROLNET_KREA overlay",
            );
        }
    }
    if let Some(pin) = pin_krea_imported_payload_control_overlay(settings, request)? {
        return Ok(pin);
    }
    let (repo, file) = krea_control_overlay_repo_file(request)?;
    let revision = trusted_control_weight_revision(request, KREA_CONTROL_ENGINE_ID, &repo, &file)?;
    let path = crate::downloads::resolve_hf_component_file(settings, &repo, &revision, &file)
        .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Krea 2 pose ControlNet overlay is not installed — install it from the Model Manager \
             ({repo} / {file}), then run the pose set again. An imported Krea 2 checkpoint is the \
             transformer only; the pose control branch is a separate trained overlay."
        ))
    })?;
    crate::paths::pin_app_managed_model_file(
        settings,
        &path,
        "Krea 2 pose ControlNet overlay",
    )
}

/// Prepare every File identity the ordinary imported route selects, once, before the async job
/// preamble. The returned tokens are moved through dispatch and finalized on the eventual `LoadSpec`;
/// handlers must not resolve the same source again.
fn prepare_krea_imported_sources(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedKreaImportedSources>> {
    if !krea_imported_request_shape_available(request) || !pose_entries(request).is_empty() {
        return Ok(None);
    }
    let Some(dit_pin) = resolve_imported_krea_dit_pin(request, settings)? else {
        return Ok(None);
    };
    let prepared_adapters = resolve_prepared_adapters(request, settings)?;
    Ok(Some(PreparedKreaImportedSources {
        dit_pin,
        prepared_adapters,
    }))
}

/// Prepared strict-pose counterpart: primary DiT, control overlay, and the complete ordered adapter
/// stack are all selected exactly once and carried through dispatch.
#[cfg(target_os = "macos")]
fn prepare_krea_imported_control_sources(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedKreaImportedControlSources>> {
    if request.mode == "edit_image"
        || pose_entries(request).is_empty()
        || !krea_imported_request_shape_available(request)
    {
        return Ok(None);
    }
    let Some(dit_pin) = resolve_imported_krea_dit_pin(request, settings)? else {
        return Ok(None);
    };
    let control_pin = resolve_krea_imported_control_overlay_pin(settings, request)?;
    let prepared_adapters = resolve_prepared_adapters(request, settings)?;
    Ok(Some(PreparedKreaImportedControlSources {
        dit_pin,
        control_pin,
        prepared_adapters,
    }))
}

/// The identity-likeness face stack (SCRFD + ArcFace) resolved **cache-only** — the scoring-side
/// twin of [`resolve_krea_imported_control_overlay`], and the reason this lane does not call the
/// shared `stage_likeness` (which downloads the stack on first use, a job-time download site).
///
/// Both weights are resolved out of the app-managed Hugging Face cache through the shared
/// [`crate::downloads::resolve_hf_component_file`] at the pinned `SceneWorks/instantid-mlx`
/// revision — NOT the legacy `<data_dir>/cache/instantid-mlx` dir the download-on-first-use helper
/// writes, which `job_time_download_guard`'s cache-destination rule reserves for the migrated
/// lanes. The scorer takes a directory, so this returns the snapshot dir the two files share,
/// having proved both are present.
///
/// `None` when the stack is not installed, which the caller treats as non-fatal exactly as
/// `stage_likeness` does on a staging failure: likeness scores are omitted and the pose set still
/// renders. So an uninstalled face stack costs the optional scores, never the render — and never a
/// mid-render download.
#[cfg(target_os = "macos")]
fn resolve_installed_face_stack_dir(settings: &Settings) -> Option<PathBuf> {
    let resolve = |file: &str| {
        crate::downloads::resolve_hf_component_file(
            settings,
            INSTANTID_MLX_REPO,
            INSTANTID_MLX_REVISION,
            file,
        )
    };
    let scrfd = resolve(INSTANTID_SCRFD_FILE);
    let arcface = resolve(INSTANTID_ARCFACE_FILE);
    match (scrfd, arcface) {
        // Both components resolved from the same snapshot; hand the scorer their shared parent.
        (Some(scrfd), Some(_)) => scrfd.parent().map(std::path::Path::to_path_buf),
        _ => {
            tracing::warn!(
                repo = INSTANTID_MLX_REPO,
                "identity face stack is not installed; imported-Krea pose likeness scores omitted \
                 (generation continues)"
            );
            None
        }
    }
}

/// Flat telemetry recorded on imported-Krea strict-pose assets: the imported-lane identity fields
/// (engine / checkpoint / base, mirroring [`krea_imported_raw_settings`]) plus the control-lane
/// fields the builtin `krea_control_raw_settings` records (control engine / scale / pose count; no
/// guidance — the distilled Turbo merge is CFG-free).
#[cfg(target_os = "macos")]
fn krea_imported_control_raw_settings(
    request: &ImageRequest,
    steps: u32,
    control_scale: f32,
    pose_count: usize,
    adapter_count: usize,
) -> JsonObject {
    let mut raw = krea_imported_raw_settings(request, steps, false, adapter_count);
    raw.insert("guidanceScale".to_owned(), Value::Null);
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(KREA_CONTROL_ENGINE_ID.to_owned()),
    );
    raw
}

/// Real MLX imported-checkpoint Krea 2 strict-pose generation: one image per pose, each conditioned
/// on a full DWPose skeleton via the trained control-branch overlay riding the FILE-LOADED imported
/// DiT (the imported twin of [`generate_krea_control_stream`], assembled through the registered
/// `krea_2_turbo_control` provider with a File primary). The
/// imported DiT is paired with the resident base tier's TE/VAE/tokenizer
/// ([`resolve_krea_imported_base_tier`] — the same staging the t2i lane uses) plus the pose control
/// overlay ([`resolve_krea_imported_control_overlay`] — the same env → payload-path → pinned
/// hosted-repo lookup as the builtin lane, but cache-only so jobs never download weights). Job
/// LoRA/LoKr adapters install on the
/// imported DiT (the branch is never an adapter target); `control_scale = 0` is engine-proven
/// byte-identical to the imported base. Shares one seed across the pose set (noise-derived
/// attributes constant, only the pose changes) and scores identity likeness against an optional
/// `referenceAssetId`, exactly like the builtin lane.
///
/// Memory. The loaded generator's contract carries the BUILTIN `krea_2_turbo_control` provider id
/// and calibration fingerprint (the native control contract deliberately does not mint an imported
/// identity), so provider-keyed machinery — the contract, the safety check, the request scope —
/// resolves identically to the builtin lane.
///
/// The promoted ENVELOPE, however, is deliberately NOT reused, and `None` is passed for the
/// resolved-artifact provenance on purpose rather than by omission. Promoted `krea_2_turbo_control`
/// evidence is artifact-bound, and every measured record in the corpus was captured on the **q4**
/// base tier (fixture `krea-pose-control-q4-seed16099`). This composition is DENSE: the native
/// loader materializes the imported single file bf16 (an int8-per-row file is dequantized at load,
/// and there is no quantize path on the native control assembly), so the imported 12B DiT is
/// ~24 GiB resident against the ~6 GiB the q4 record measured. Same architecture is NOT the same
/// footprint once the tier differs, and this gate's permissive-side failure mode is an OS Jetsam
/// SIGKILL — so the request runs on the conservative estimate path instead. The estimate is honest
/// about the real assembly: the spec points at the resolved DENSE `bf16` base tier (whose
/// `transformer/` ships the same-shape dense weights the imported file carries) plus the overlay
/// and adapter files.
///
/// Note what it would actually take for this lane to serve promoted numbers, so the estimate path
/// is not mistaken for a one-line gap: promoted evidence is ARTIFACT-bound, and an imported DiT is
/// a user file with no pinned repository/revision/variant to bind to — so a dense-tier capture
/// alone would not reach it. It would also need an explicit decision that a same-shape dense
/// imported DiT may INHERIT a dense record measured on the builtin base (defensible — identical
/// architecture and dtype means identical resident bytes, which is exactly the property the q4
/// record lacks), plus something that proves the imported file really is that shape and dtype
/// before the inheritance applies. That is an evidence-reuse policy change, not a binding.
///
/// Independently of the estimate, the generator's own measured, architecture-keyed feasibility
/// check (`control_geometry_fits`, sized off the arch config + branch block count + tiers) guards
/// the render exactly as it does the builtin lane.
#[cfg(target_os = "macos")]
async fn generate_krea_imported_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    dispatch: PreparedFileDispatch<'_, PreparedKreaImportedControlSources>,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let PreparedFileDispatch { plan, sources } = dispatch;
    let request = &plan.request;
    let PreparedKreaImportedControlSources {
        dit_pin,
        control_pin,
        prepared_adapters,
    } = sources;
    let dit = dit_pin.loader_path().to_path_buf();
    // Require the resident base tier before any compute — a clear "install the Krea 2 base first"
    // error (the shared TE/VAE/tokenizer + arch config the control assembly pairs the DiT with).
    let base_dir = resolve_krea_imported_base_tier(settings)?;
    let descriptor = krea_imported_descriptor(request).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "This runtime has no registered imported Krea pose provider.".to_owned(),
        )
    })?;
    let (quant, quant_bits) = imported_model_quant(request, &descriptor, "Imported Krea 2")?;
    let control_weights = control_pin.loader_path().to_path_buf();
    let PreparedAdapters {
        specs: adapters,
        pins: adapter_pins,
    } = prepared_adapters;

    let steps = krea_control_steps(request);
    let control_scale = advanced::f32_clamped(
        &request.advanced,
        "controlScale",
        KREA_CONTROL_DEFAULT_SCALE,
        0.0..=KREA_CONTROL_SCALE_CAP,
    );

    // Shared strict-control driver: validate the requested ControlKind against the engine's
    // supported_kinds (krea_2_turbo_control = {Pose}) and resolve an optional user-supplied
    // control-map passthrough — identical to the builtin lane.
    let control_kind = requested_control_kind(request)?;
    validate_control_kind(KREA_CONTROL_ENGINE_ID, &control_kind)?;
    let user_control = resolve_user_control_map(request, settings, project_path)?;
    let control_source = resolve_control_source(request, settings, project_path)?;

    let poses = parse_poses(request);
    let count = poses.len();
    let mut raw_settings =
        krea_imported_control_raw_settings(request, steps, control_scale, count, adapters.len());
    raw_settings.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(Value::from).unwrap_or(Value::Null),
    );
    // Strict pose shares one seed across the set so noise-derived attributes stay constant.
    let seed = resolve_seed(request, 0);

    // Identity-likeness scoring (epic 4406): same generator-agnostic seam as the builtin lane —
    // all non-fatal, the set still renders when the reference/staging is unavailable.
    let likeness_source = resolve_control_identity_source(request, settings, project_path);
    let face_stack_dir = likeness_source
        .is_some()
        .then(|| resolve_installed_face_stack_dir(settings))
        .flatten();

    let prompt = request.prompt.clone();
    let text_style_gain = resolve_text_style_gain(request);
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();

    // The actual imported assembly is the fit/cache/provider source of truth: primary File DiT, base
    // snapshot companion, control overlay, and adapters. The provider reports the File's own bytes;
    // it does not relabel a directory-tier rung-4 measurement as imported evidence.
    let mut spec = LoadSpec::new(WeightsSource::File(dit))
        .with_component(
            gen_core::BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(base_dir),
        )
        .with_control(WeightsSource::File(control_weights.clone()));
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    crate::paths::prepare_load_spec_with_file_pins(
        &mut spec,
        std::iter::once(dit_pin)
            .chain(std::iter::once(control_pin))
            .chain(adapter_pins),
        "Krea 2 imported pose source preparation failed",
    )?;
    let memory_plan = crate::mlx_fit_gate::MlxRequestPlan::try_for_spec_and_manifest(
        KREA_CONTROL_ENGINE_ID,
        &request.model,
        &spec,
        Some(&request.model_manifest_entry),
        None,
    )?;
    // Imported and builtin Krea pose routes send the same one-Control request. Keep their admitted
    // geometry and measured lane identity on the shared constructor so neither call site can drift
    // back to declaring zero references while gen-core derives one from `Conditioning::Control`.
    let memory_inputs =
        krea_control_memory_inputs(width, height, &request.mode, adapter_count);

    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state(
        job.id.clone(),
        KREA_CONTROL_ENGINE_ID,
        adapter_count,
        spec,
        "Krea 2 imported checkpoint pose-control load failed".to_owned(),
        move |model,
              initial_cache_state,
              loaded_policy,
              _requested_policy,
              external_committed_bytes,
              tx,
              cancel| {
            let user_control = user_control.as_ref();
            let control_source = control_source.as_ref();
            // Build the per-job identity-likeness scorer ONCE on the generator-worker thread (the
            // `!Send` face stack lives here); the source identity is embedded once, reused per pose.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            let mut cache_state = initial_cache_state;
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
                // The pose `Control` is the only conditioning (pose renders from noise; no
                // identity img2img-init on the Krea control lane).
                let conditioning =
                    build_control_conditioning(control, control_kind.clone(), control_scale, None);
                let memory_evaluation = crate::mlx_fit_gate::evaluate_request(
                    model,
                    &memory_plan,
                    &memory_inputs,
                    cache_state,
                    loaded_policy.offload_policy,
                    external_committed_bytes,
                )?;
                cache_state = gen_core::MemoryCacheState::Warm;
                let (out_w, out_h, pixels) = krea_control_generate_one(
                    model,
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
        KREA_IMPORTED_ENGINE,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
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

/// Explicit imported-checkpoint quant selection. Absence preserves the checkpoint's own effective
/// encoding; a selected load-time tier is accepted only when the exact provider advertises it.
fn imported_model_quant(
    request: &ImageRequest,
    descriptor: &gen_core::ModelDescriptor,
    label: &str,
) -> WorkerResult<(Option<Quant>, Option<i64>)> {
    let named = request
        .advanced
        .get("quantTier")
        .or_else(|| request.advanced.get("quant"))
        .and_then(Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_lowercase);
    let bits = request.advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    });
    let selected = if let Some(named) = named.as_deref() {
        match named {
            "nvfp4" => Some(Quant::Nvfp4),
            "q4" => Some(Quant::Q4),
            "q8" => Some(Quant::Q8),
            "bf16" | "dense" => None,
            other => {
                return Err(WorkerError::InvalidPayload(format!(
                    "{label} quant tier '{other}' is unknown; use bf16, q8, or q4."
                )))
            }
        }
    } else {
        match bits {
            Some(1..=4) => Some(Quant::Q4),
            Some(5..) => Some(Quant::Q8),
            Some(i64::MIN..=0) => None,
            None => return Ok((None, None)),
        }
    };
    if let Some(quant) = selected {
        if !descriptor.capabilities.supported_quants.contains(&quant) {
            return Err(WorkerError::InvalidPayload(format!(
                "The registered provider '{}' does not support imported {:?} quantization.",
                descriptor.id, quant
            )));
        }
    }
    let bits = match selected {
        Some(Quant::Q4) => Some(4),
        Some(Quant::Q8) => Some(8),
        Some(Quant::Nvfp4) | None => None,
    };
    Ok((selected, bits))
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
        "resolvedSampler".to_owned(),
        Value::String(KREA_IMPORTED_SAMPLER.to_owned()),
    );
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
    if request.hires_fix.enabled {
        raw.insert(
            "hiresFix".to_owned(),
            serde_json::to_value(&request.hires_fix).expect("HiresFixRequest is serializable"),
        );
    }
    raw
}

fn krea_imported_edit_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        return request.reference_asset_ids.clone();
    }
    request
        .reference_asset_id
        .as_deref()
        .or(request.source_asset_id.as_deref())
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(|id| vec![id.to_owned()])
        .unwrap_or_default()
}

fn krea_imported_has_edit_adapter(request: &ImageRequest) -> bool {
    request.loras.iter().any(|lora| {
        lora.get("conditioningRole")
            .and_then(Value::as_str)
            .map(|role| role.trim().to_ascii_lowercase().replace('-', "_") == "image_edit")
            .unwrap_or(false)
    })
}

fn krea_imported_edit_conditioning(references: Vec<Image>) -> Vec<Conditioning> {
    if references.len() == 1 {
        vec![Conditioning::Reference {
            image: references.into_iter().next().expect("one reference"),
            strength: None,
        }]
    } else {
        vec![Conditioning::MultiReference { images: references }]
    }
}

/// Resolve cross-platform edit conditioning after route selection pinned the adapter stack.
fn resolve_krea_imported_edit_conditioning(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<Vec<Conditioning>>> {
    if request.mode != "edit_image" {
        return Ok(None);
    }
    // R5 (epic 10871): the bare transformer cannot edit without the `krea2_identity_edit` LoRA — the
    // in-context / grounded source conditioning is inert without the trained weights. Require it before
    // any compute, mirroring the builtin `generate_krea_edit_stream`.
    if !krea_imported_has_edit_adapter(request) {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires the Krea 2 Identity Edit LoRA (or another image-edit LoRA): without \
             it the source-image conditioning is inert. Select it in the LoRA picker."
                .to_owned(),
        ));
    }
    let reference_ids = krea_imported_edit_reference_ids(request);
    if reference_ids.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires a source image.".to_owned(),
        ));
    }
    if reference_ids.len() > 2 {
        return Err(WorkerError::InvalidPayload(format!(
            "Krea 2 edit takes at most 2 images (image 1, then image 2)."
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
    let sources = if request.fit_mode == "stretch" {
        sources
    } else {
        sources
            .into_iter()
            .map(|source| {
                fit_engine_image(source, request.width, request.height, &request.fit_mode)
            })
            .collect::<WorkerResult<Vec<_>>>()?
    };
    Ok(Some(krea_imported_edit_conditioning(sources)))
}

fn krea_imported_reference_count(conditioning: &[Conditioning]) -> u32 {
    conditioning.iter().fold(0_u32, |count, item| {
        let increment = match item {
            Conditioning::Reference { .. } => 1,
            Conditioning::MultiReference { images } => {
                u32::try_from(images.len()).unwrap_or(u32::MAX)
            }
            _ => 0,
        };
        count.saturating_add(increment)
    })
}

/// Request identity handed to MLX admission for the imported Krea assembly. Hires admission names
/// the heavier final refinement (one generated-image reference); its first pass receives a derived
/// base-size context immediately before execution.
#[cfg(target_os = "macos")]
fn krea_imported_memory_inputs(
    request: &ImageRequest,
    conditioning: &[Conditioning],
    hires_fix: Option<HiresFixPlan>,
    adapter_count: usize,
) -> crate::mlx_fit_gate::MlxRequestInputs {
    let request_reference_count = krea_imported_reference_count(conditioning);
    let reference_count = hires_fix.map_or_else(
        || request_reference_count,
        |_| hires_fix_reference_count(),
    );
    let (width, height) =
        hires_fix.map_or((request.width, request.height), |hires| (hires.width, hires.height));
    let mut overlays = Vec::new();
    // Match the generic MLX lane: geometry describes the heaviest provider pass, while overlays
    // describe only caller-supplied request resources. A t2i hires refinement therefore has one
    // geometry reference but no external-reference overlay; edit hires retains its source count.
    if request_reference_count > 0 {
        overlays.push(format!("references:{request_reference_count}"));
    }
    if adapter_count > 0 {
        overlays.push(format!("adapters:{adapter_count}"));
    }
    crate::mlx_fit_gate::MlxRequestInputs {
        width,
        height,
        count: request.count,
        mode: request.mode.clone(),
        overlay: (!overlays.is_empty()).then(|| overlays.join("+")),
        adapter_count,
        has_reference: reference_count > 0,
        reference_count,
        use_pid: false,
        has_phases: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn krea_imported_generate_pass(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    conditioning: Vec<Conditioning>,
    text_style_gain: Option<f32>,
    memory: Option<gen_core::GenerationMemory>,
    memory_context: Option<&gen_core::MemoryRunContext>,
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let mut request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        sampler: Some(KREA_IMPORTED_SAMPLER.to_owned()),
        conditioning,
        text_style_gain,
        memory,
        preview,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = crate::memory_strategy::generate_with_scope(
        generator,
        &mut request,
        memory_context,
        on_progress,
    )
    .map_err(|error| {
        WorkerError::Engine(format!(
            "Krea 2 imported checkpoint generation failed: {error}"
        ))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("Krea 2 imported checkpoint produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "Krea 2 imported checkpoint returned non-image output".to_owned(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn krea_imported_generate_multiphase(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    phases: Vec<gen_core::GenerationPhase>,
    text_style_gain: Option<f32>,
    memory_evaluation: Option<&crate::mlx_fit_gate::MlxRequestEvaluation>,
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let mut request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        phases: Some(phases),
        text_style_gain,
        memory: memory_evaluation.map(|evaluation| evaluation.memory),
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
    .map_err(|error| {
        WorkerError::Engine(format!(
            "Krea 2 imported multi-phase generation failed: {error}"
        ))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine(
                    "Krea 2 imported multi-phase request produced no image".to_owned(),
                )
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "Krea 2 imported multi-phase request returned non-image output".to_owned(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn krea_imported_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    conditioning: &[Conditioning],
    text_style_gain: Option<f32>,
    hires_fix: Option<HiresFixPlan>,
    memory_evaluation: Option<&crate::mlx_fit_gate::MlxRequestEvaluation>,
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let Some(hires) = hires_fix else {
        return krea_imported_generate_pass(
            generator,
            prompt,
            width,
            height,
            seed,
            steps,
            conditioning.to_vec(),
            text_style_gain,
            memory_evaluation.map(|evaluation| evaluation.memory),
            memory_evaluation.map(|evaluation| &evaluation.context),
            preview,
            cancel,
            on_progress,
        );
    };

    let first_pass_context = memory_evaluation.map(|evaluation| {
        hires_first_pass_context(
            &evaluation.context,
            width,
            height,
            krea_imported_reference_count(conditioning),
        )
    });
    let combined_steps = steps.saturating_add(hires.steps);
    let mut first_progress = |progress| match progress {
        Progress::Step { current, .. } => on_progress(Progress::Step {
            current,
            total: combined_steps,
        }),
        Progress::Decoding => {}
        Progress::Loading(phase) => on_progress(Progress::Loading(phase)),
    };
    let (base_width, base_height, base_pixels) = krea_imported_generate_pass(
        generator,
        prompt,
        width,
        height,
        seed,
        steps,
        conditioning.to_vec(),
        text_style_gain,
        memory_evaluation.map(|evaluation| evaluation.memory),
        first_pass_context.as_ref(),
        preview.clone(),
        cancel,
        &mut first_progress,
    )?;
    if cancel.is_cancelled() {
        return Err(WorkerError::Engine("generation cancelled".to_owned()));
    }
    let high_res_reference = fit_engine_image(
        Image {
            width: base_width,
            height: base_height,
            pixels: base_pixels,
        },
        hires.width,
        hires.height,
        "stretch",
    )?;
    let final_conditioning = vec![Conditioning::Reference {
        image: high_res_reference,
        strength: Some(hires.provider_reference_strength),
    }];
    let mut second_progress = |progress| match progress {
        Progress::Step { current, .. } => on_progress(Progress::Step {
            current: steps.saturating_add(current),
            total: combined_steps,
        }),
        Progress::Decoding => on_progress(Progress::Decoding),
        Progress::Loading(phase) => on_progress(Progress::Loading(phase)),
    };
    krea_imported_generate_pass(
        generator,
        prompt,
        hires.width,
        hires.height,
        seed,
        hires.steps,
        final_conditioning,
        text_style_gain,
        memory_evaluation.map(|evaluation| evaluation.memory),
        memory_evaluation.map(|evaluation| &evaluation.context),
        preview,
        cancel,
        &mut second_progress,
    )
}

/// Exact request-state driver installed into the normal imported-Krea MLX cache callback.
///
/// Keeping the evaluator injectable makes the cache/admission handoff testable without loading a
/// multi-gigabyte checkpoint, while production passes [`crate::mlx_fit_gate::evaluate_request`]
/// directly. One evaluation is performed for every sequential image item. Hires remains one
/// admitted request with two provider passes, and [`krea_imported_generate_one`] opens the
/// appropriate request scope for each pass.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn drive_krea_imported_mlx_items<E>(
    generator: &dyn Generator,
    memory_plan: &crate::mlx_fit_gate::MlxRequestPlan,
    memory_inputs: &crate::mlx_fit_gate::MlxRequestInputs,
    initial_cache_state: gen_core::MemoryCacheState,
    loaded_offload_policy: gen_core::OffloadPolicy,
    external_committed_bytes: u64,
    work: Vec<(i64, String)>,
    width: u32,
    height: u32,
    steps: u32,
    conditioning: Vec<Conditioning>,
    negative_prompt: Option<String>,
    phases: Option<Vec<gen_core::GenerationPhase>>,
    text_style_gain: Option<f32>,
    hires_fix: Option<HiresFixPlan>,
    tx: tokio::sync::mpsc::Sender<GenEvent>,
    cancel: CancelFlag,
    mut evaluate_request: E,
) -> WorkerResult<()>
where
    E: FnMut(
        &dyn Generator,
        &crate::mlx_fit_gate::MlxRequestPlan,
        &crate::mlx_fit_gate::MlxRequestInputs,
        gen_core::MemoryCacheState,
        gen_core::OffloadPolicy,
        u64,
    ) -> WorkerResult<crate::mlx_fit_gate::MlxRequestEvaluation>,
{
    let mut cache_state = initial_cache_state;
    drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
        if cancel.is_cancelled() {
            return Ok(None);
        }
        let memory_evaluation = evaluate_request(
            generator,
            memory_plan,
            memory_inputs,
            cache_state,
            loaded_offload_policy,
            external_committed_bytes,
        )?;
        cache_state = gen_core::MemoryCacheState::Warm;
        let _request_memory_limit = memory_evaluation
            .process_limit_bytes
            .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
        let generated = if let Some(phases) = phases.clone() {
            krea_imported_generate_multiphase(
                generator,
                &prompt,
                negative_prompt.clone(),
                width,
                height,
                seed,
                phases,
                text_style_gain,
                Some(&memory_evaluation),
                preview,
                &cancel,
                on_progress,
            )
        } else {
            krea_imported_generate_one(
                generator,
                &prompt,
                width,
                height,
                seed,
                steps,
                &conditioning,
                text_style_gain,
                hires_fix,
                Some(&memory_evaluation),
                preview,
                &cancel,
                on_progress,
            )
        };
        let (out_width, out_height, pixels) = match generated {
            Ok(image) => image,
            Err(_) if cancel.is_cancelled() => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(Some((seed, out_width, out_height, pixels)))
    })
}

/// Real in-place imported single-file Krea 2 generation (epic 14015 S0c, sc-14023 + sc-14071 +
/// sc-14111 + sc-14119): resolve the imported DiT, the resident base tier, any img2img reference, and —
/// on the adapter-capable MLX backend — the job LoRA stack + Kontext edit conditioning, then load the
/// selected runtime's registered provider once and generate each image on the cache worker thread.
///
/// Three shapes ride one lane: plain **t2i**, reference-guided **img2img** (one `Conditioning::Reference`
/// on the Turbo t2i descriptor, both backends), and — MLX only — **LoRA-adapted** t2i/img2img (sc-14111)
/// and the **Kontext edit** surface (sc-14119: the `krea_2_turbo_edit` provider + the fitted source
/// reference(s) as `Reference`/`MultiReference` + the `krea2_identity_edit` adapter). The merge is
/// distilled Turbo (no CFG / negative prompt).
async fn generate_krea_imported_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    dispatch: PreparedFileDispatch<'_, PreparedKreaImportedSources>,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let PreparedFileDispatch { plan, sources } = dispatch;
    let request = &plan.request;
    let PreparedKreaImportedSources {
        dit_pin,
        prepared_adapters,
    } = sources;
    let dit = dit_pin.loader_path().to_path_buf();
    // Require the resident base tier before any compute — a clear "install the Krea 2 base first" error.
    let base_dir = resolve_krea_imported_base_tier(settings)?;
    let is_edit = request.mode == "edit_image";
    let descriptor = krea_imported_descriptor(request).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "This runtime has no registered imported Krea provider for the requested operation."
                .to_owned(),
        )
    })?;
    let (quant, quant_bits) = imported_model_quant(request, &descriptor, "Imported Krea 2")?;

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

    // Adapter File identities were prepared during route selection and survive every async preamble
    // await. Build only the edit conditioning here; the handler must never re-pin its selected stack.
    let edit_conditioning =
        resolve_krea_imported_edit_conditioning(request, settings, project_path)?;
    let adapter_count = prepared_adapters.specs.len();
    let conditioning = edit_conditioning.unwrap_or_else(|| krea_imported_conditioning(img2img));

    let phase_specs = request_has_multiphase(request)
        .then(|| {
            ensure_multiphase_job_shape(request)?;
            parse_multiphase_specs(request)
        })
        .transpose()?;
    let phases = phase_specs.as_deref().map(build_generation_phases);
    if phases.is_some() && request.hires_fix.enabled {
        return Err(WorkerError::InvalidPayload(
            "Krea multi-phase denoise cannot be combined with Hires.fix.".to_owned(),
        ));
    }

    let (width, height) = (request.width, request.height);
    let steps = phase_specs.as_ref().map_or_else(
        || resolve_advanced_or_manifest_u32(request, "steps", KREA_IMPORTED_DEFAULT_STEPS, 1..=100),
        |specs| specs.iter().map(|phase| phase.steps).sum(),
    );
    let hires_fix = resolve_hires_fix_plan(request, steps, None, None);
    let text_style_gain = resolve_text_style_gain(request);
    let mut raw_settings = krea_imported_raw_settings(request, steps, is_edit, adapter_count);
    raw_settings.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(Value::from).unwrap_or(Value::Null),
    );
    if let Some(specs) = phase_specs.as_ref() {
        raw_settings.insert("multiPhase".to_owned(), Value::Bool(true));
        raw_settings.insert(
            "phases".to_owned(),
            serde_json::to_value(
                specs
                    .iter()
                    .map(|phase| {
                        json!({
                            "steps": phase.steps,
                            "guidance": phase.guidance,
                            "loras": phase.loras.iter().map(|lora| json!({
                                "index": lora.index,
                                "weight": lora.weight,
                            })).collect::<Vec<_>>(),
                        })
                    })
                    .collect::<Vec<_>>(),
            )
            .expect("phase telemetry serializes"),
        );
    }

    // Per-image work items: (seed, prompt) — `request.count` renders, each its own seed.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let negative_prompt = (!request.negative_prompt.trim().is_empty())
        .then(|| request.negative_prompt.clone());

    let engine_id = descriptor.id;
    let mut spec = LoadSpec::new(WeightsSource::File(dit)).with_component(
        gen_core::BASE_SNAPSHOT_COMPONENT,
        WeightsSource::Dir(base_dir.clone()),
    );
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !prepared_adapters.specs.is_empty() {
        spec = spec.with_adapters(prepared_adapters.specs);
    }
    crate::paths::prepare_load_spec_with_file_pins(
        &mut spec,
        std::iter::once(dit_pin).chain(prepared_adapters.pins),
        "Krea 2 imported source preparation failed",
    )?;

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let cold_admission = {
        // The base snapshot is only a companion: its own transformer is replaced by `dit` and must
        // not be double-priced. Admit exactly the prepared primary plus the companion component dirs.
        let text_encoder = base_dir.join("text_encoder");
        let vae = base_dir.join("vae");
        prepare_cached_candle_base_floor(
            &request.model,
            "Krea imported",
            settings,
            &spec,
            &[text_encoder.as_path(), vae.as_path()],
        )?
    };

    #[cfg(target_os = "macos")]
    let memory_plan = crate::mlx_fit_gate::MlxRequestPlan::try_for_spec_and_manifest(
        engine_id,
        &request.model,
        &spec,
        Some(&request.model_manifest_entry),
        None,
    )?;
    #[cfg(target_os = "macos")]
    let memory_inputs =
        krea_imported_memory_inputs(request, &conditioning, hires_fix, adapter_count);

    #[cfg(target_os = "macos")]
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        "Krea 2 imported checkpoint load failed".to_owned(),
        move |model,
              initial_cache_state,
              loaded_policy,
              _requested_policy,
              external_committed_bytes,
              tx,
              cancel| {
            drive_krea_imported_mlx_items(
                model,
                &memory_plan,
                &memory_inputs,
                initial_cache_state,
                loaded_policy.offload_policy,
                external_committed_bytes,
                work,
                width,
                height,
                steps,
                conditioning,
                negative_prompt,
                phases,
                text_style_gain,
                hires_fix,
                tx,
                cancel,
                crate::mlx_fit_gate::evaluate_request,
            )
        },
    );

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let incoming_reclaimable_weight_bytes = cold_admission.reclaimable_weight_bytes();
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let (cancel, rx, blocking) = start_cached_gen_stream_after_cold_admission(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        "Krea 2 imported checkpoint load failed".to_owned(),
        ColdLoadAdmission::new(
            incoming_reclaimable_weight_bytes,
            move |resident_reclaimable_weight_bytes| {
                cold_admission.admit(resident_reclaimable_weight_bytes)
            },
        ),
        move |model, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let generated = if let Some(phases) = phases.clone() {
                    krea_imported_generate_multiphase(
                        model,
                        &prompt,
                        negative_prompt.clone(),
                        width,
                        height,
                        seed,
                        phases,
                        text_style_gain,
                        None,
                        preview,
                        &cancel,
                        on_progress,
                    )
                } else {
                    krea_imported_generate_one(
                        model,
                        &prompt,
                        width,
                        height,
                        seed,
                        steps,
                        &conditioning,
                        text_style_gain,
                        hires_fix,
                        None,
                        preview,
                        &cancel,
                        on_progress,
                    )
                };
                let (out_width, out_height, pixels) = match generated {
                    Ok(image) => image,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => return Err(error),
                };
                Ok(Some((seed, out_width, out_height, pixels)))
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
