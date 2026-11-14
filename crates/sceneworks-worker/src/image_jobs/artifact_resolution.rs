use super::min_quality_floor;
use super::mlx_model;
use super::nvfp4_host_eligible;
use super::nvfp4_requested;
use super::pick_loadable_tier;
use super::preferred_tier;
#[cfg(target_os = "macos")]
use super::quantization::fixed_mlx_artifact_tier;
use super::quantization::tier_key_from_resolved_dir;
use super::resolve_pinned_native_package;
use super::standard_tier_subdir;
use super::tier_components_present;
use super::uses_standard_tier_layout;
use super::ImageRequest;
use super::ResolvedModel;
use crate::huggingface_snapshot_dir;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use crate::model_jobs::huggingface_pinned_snapshot_dir;
use crate::resolve_app_managed_model_dir;
use crate::ApiClient;
use crate::JobSnapshot;
use crate::JsonObject;
use crate::Path;
use crate::PathBuf;
use crate::Settings;
use crate::Value;
use crate::WorkerError;
use crate::WorkerResult;
/// The HuggingFace repo for the model: the manifest entry's `repo` wins, else the
/// family default. Shared by the MLX path and the candle lane (sc-5096).
#[cfg(native_lane)]
pub(super) fn model_repo(request: &ImageRequest, model: &ResolvedModel) -> String {
    model_repo_with_default(request, model.default_repo())
}

#[cfg(native_lane)]
fn model_repo_with_default(request: &ImageRequest, default_repo: &str) -> String {
    crate::payload::trimmed_string(request.model_manifest_entry.get("repo"))
        .unwrap_or(default_repo)
        .to_owned()
}

/// Receipt variant selected by this request.  Prefer an explicit quant request; otherwise use the
/// manifest's default selectable download (or its only selectable download).  Co-requisites are
/// intentionally excluded because each repo resolves its own receipt independently.
#[cfg(native_lane)]
pub(super) fn requested_receipt_variant(request: &ImageRequest) -> Option<String> {
    if let Some(bits) = request.quant_bits() {
        return Some(crate::payload::tier_for_bits(Some(bits), "q8").to_owned());
    }
    let selectable = request
        .model_manifest_entry
        .get("downloads")
        .and_then(Value::as_array)?
        .iter()
        .filter(|entry| entry.get("coRequisite").and_then(Value::as_bool) != Some(true))
        .collect::<Vec<_>>();
    selectable
        .iter()
        .copied()
        .find(|entry| entry.get("default").and_then(Value::as_bool) == Some(true))
        .or_else(|| (selectable.len() == 1).then_some(selectable[0]))
        .and_then(|entry| entry.get("variant").and_then(Value::as_str))
        .map(str::to_owned)
}

/// The separate `SceneWorks/ideogram-4` repo that hosts Ideogram 4's bf16 tree under `bf16/`, the
/// selectable full-precision tier (sc-8513). The `q4/`/`q8/` packed turnkey lives in
/// `SceneWorks/ideogram-4-mlx`; bf16 is resolved from THIS repo rather than duplicated. sc-9650 wires it
/// on the candle lane too: the `bf16/` subdir is in the SAME single-file `transformer/model.safetensors`
/// layout the candle loader reads, and `linear_detect` takes its dense arm (no `.scales` sibling), so
/// candle dense-loads bf16 exactly like macOS/MLX — while the packed q4/q8 tiers still come from the
/// `-mlx` turnkey via `ideogram_model_subdir`.
#[cfg(native_lane)]
const IDEOGRAM_BF16_REPO: &str = "SceneWorks/ideogram-4";

/// Exact snapshots used by the on-demand non-default tier fetches.  The API replaces a submitted
/// `modelManifestEntry` with its resolved builtin/user manifest entry before enqueueing, so a repo
/// different from the engine default is an intentional configured override.  First-party turnkeys
/// never follow a mutable ref; configured override repos retain `main` because SceneWorks cannot know
/// their release commit in advance.
#[cfg(native_lane)]
pub(super) const BOOGU_MLX_TURNKEY_REVISION: &str = "a459e614d408bfdf57089c32cc3da706f5a017de";
#[cfg(native_lane)]
pub(super) const IDEOGRAM_MLX_TURNKEY_REVISION: &str = "a3095855b8819dc0d6b067cb1354aaa7da189ff8";
#[cfg(native_lane)]
pub(crate) const ZIMAGE_MLX_TURNKEY_REPO: &str = "SceneWorks/z-image-mlx";
#[cfg(native_lane)]
pub(crate) const ZIMAGE_MLX_TURNKEY_REVISION: &str = "c74f74c2ad193294fc9ff3f8a5be71daa00d22ab";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const FLUX1_SCHNELL_MLX_TURNKEY_REVISION: &str =
    "bba3ae01dfd94089f173c05edd4e1a4c551f2599";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const FLUX1_DEV_MLX_TURNKEY_REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const FLUX2_DEV_MLX_TURNKEY_REVISION: &str = "2868b1461b2b6e6e05d84e52534df3632b4c7d5d";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const FLUX2_KLEIN_9B_MLX_TURNKEY_REVISION: &str =
    "1902693279fcfb828919370dfac2b8922d99499a";
#[cfg(native_lane)]
pub(super) fn turnkey_tier_revision<'a>(
    repo: &str,
    default_repo: &str,
    default_revision: &'a str,
) -> &'a str {
    if repo == default_repo {
        default_revision
    } else {
        "main"
    }
}

#[cfg(native_lane)]
fn pinned_turnkey_snapshot_for_request(
    data_dir: &Path,
    request: &ImageRequest,
    repo: &str,
    default_repo: &str,
    revision: &str,
) -> Option<PathBuf> {
    if repo != default_repo {
        return None;
    }
    let root = crate::model_jobs::huggingface_pinned_snapshot_dir(data_dir, repo, revision)?;
    let selected = match request.model.as_str() {
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit" => {
            boogu_model_subdir(&root, request)
        }
        "ideogram_4" | "ideogram_4_turbo" => ideogram_model_subdir(&root, request),
        "z_image" => standard_tier_subdir(&root, request),
        _ => return None,
    };
    let complete = if request.model == "z_image" {
        tier_key_from_resolved_dir(&selected).is_some()
            && (selected
                .join("transformer/diffusion_pytorch_model.safetensors")
                .is_file()
                || selected
                    .join("transformer/diffusion_pytorch_model.safetensors.index.json")
                    .is_file())
    } else if request.model.starts_with("boogu_") {
        selected
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file()
            || selected
                .join("transformer/diffusion_pytorch_model.safetensors.index.json")
                .is_file()
    } else {
        selected.join("transformer/model.safetensors").is_file()
    };
    complete.then_some(root)
}

/// The whole-repo `Efficient-Large-Model/Sana_1600M_1024px_diffusers` HF snapshot the candle SANA
/// lane loads (sc-11780, epic 8485). The `candle-gen-sana` pipeline reads the diffusers-layout tree
/// (`transformer/` + `vae/` + `text_encoder/`) directly, so the off-Mac lane resolves this repo's
/// snapshot root — NOT the MLX-packed `SceneWorks/Sana_1600M_1024px_mlx` turnkey (the `MODEL_TABLE`
/// `default_repo`, which the macOS/MLX path loads) and NOT a `q4/q8/bf16` tier subdir. Matches the
/// manifest's windows/linux whole-repo download entry.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SANA_CANDLE_DIFFUSERS_REPO: &str = "Efficient-Large-Model/Sana_1600M_1024px_diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SANA_CANDLE_DIFFUSERS_REVISION: &str = "ac0da2ff55fbe434795be0dce883042e4d49e2fc";

/// The whole-repo `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` HF snapshot the candle
/// SANA-Sprint lane loads (sc-11781, epic 8485). The `candle-gen-sana` Sprint pipeline reads the same
/// diffusers-layout tree (`transformer/` Sprint Linear-DiT + guidance embedder + `vae/` + `text_encoder/`)
/// as base SANA, so the off-Mac lane resolves this repo's snapshot root — NOT the MLX-packed
/// `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` turnkey (the `MODEL_TABLE` `default_repo`, which the macOS/MLX
/// path loads) and NOT a `q4/q8/bf16` tier subdir. Matches the manifest's windows/linux whole-repo
/// download entry.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SANA_SPRINT_CANDLE_DIFFUSERS_REPO: &str =
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SANA_SPRINT_CANDLE_DIFFUSERS_REVISION: &str = "19683c58b7ea290e55cedd8950ae1d86ada7ef96";

#[cfg(native_lane)]
// Keep the explicit optional branch: the manifest audit recognizes this shape and proves every
// production modelPath reader preserves the normal repo-resolution fallback.
#[allow(clippy::question_mark)]
pub(super) fn model_path_override(request: &ImageRequest) -> Option<String> {
    let Some(raw_path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(crate::payload::trimmed_value_str)
    else {
        return None;
    };
    Some(raw_path.to_owned())
}

/// The confined directory a job's `modelPath` (the request's `advanced.modelPath`, else the entry's
/// injected one — [`model_path_override`]) names, through the app-managed resolver that refuses a
/// path outside the data dir. The ONE resolution the load ([`resolve_weights_dir`]) and the
/// failed-load diagnosis (`converted_install::explain_failed_load`) share, so a diagnosis judges —
/// and marks — exactly the bundle the load read. `Ok(None)` when the job names no `modelPath`.
#[cfg(native_lane)]
pub(crate) fn resolve_model_path_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    model_path_override(request)
        .map(|path| resolve_app_managed_model_dir(settings, &path, "Image modelPath"))
        .transpose()
}

/// Typed, actionable preflight for a **convert-at-install** model whose conversion has not produced
/// a local artifact yet (sc-20529).
///
/// Such a model is declared with `mlx.requiresConversion` + `mlx.converter`, and the API injects the
/// converted dir as `modelPath` (`inject_converted_model_path`) once, and only once, the conversion
/// has actually run. So a request with NO `modelPath` means there is no converted artifact — and the
/// source repo it would otherwise fall back to is a bare single-file transformer
/// with no `text_encoder/` / `vae/` / `tokenizer/`. Loading it produced a raw engine-level
/// `"<label>: snapshot missing the text_encoder/ component directory (at …)"` from candle-gen's
/// shared `loader.rs`, which names an internal path and tells the user nothing about what to do.
///
/// This turns that class into one typed [`WorkerError::InvalidPayload`] naming BOTH remedies: run
/// the conversion, and install the base model whose text encoder / VAE / tokenizer the conversion
/// borrows (`mlx.convertBaseRepo` + `mlx.convertBaseSubdir` — for `flux2_klein_9b_true_v2`,
/// `SceneWorks/flux2-klein-9b-mlx` `bf16`). Keyed on the MANIFEST CONTRACT, never on a model id, so
/// every present and future convert-at-install model inherits it.
///
/// **Platform-scoped on purpose.** The gate is [`convert_artifact_required_here`], not a bare
/// `requiresConversion` read: Anima (`anima_base` / `anima_aesthetic` / `anima_turbo`) declares
/// `requiresConversion` and ships windows/linux downloads, but its `anima_quant` converter is
/// macOS-only — off-Mac those models legitimately resolve the raw `circlestone-labs/Anima`
/// `split_files/` tree below with no `modelPath` at all. Refusing them here would break three
/// models that render fine today.
///
/// **Self-contained on purpose.** The `modelPath` test lives INSIDE this function rather than in the
/// caller, so the same verdict can be reached from two seams that see the request at different
/// points: [`resolve_weights_dir`] (which reaches the loader) and [`candle_weights_gap`] (the
/// stub fall-through, which never resolves weights at all — see its docs for why that second seam
/// is required).
#[cfg(native_lane)]
fn unconverted_model_preflight(request: &ImageRequest) -> WorkerResult<()> {
    // An injected `modelPath` IS the converted artifact — nothing outstanding.
    if model_path_override(request).is_some() {
        return Ok(());
    }
    let Some(mlx) = request.model_manifest_entry.get("mlx") else {
        return Ok(());
    };
    if mlx.get("requiresConversion").and_then(Value::as_bool) != Some(true) {
        return Ok(());
    }
    let converter = mlx
        .get("converter")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if !sceneworks_core::jobs_store::convert_artifact_required_here(converter) {
        return Ok(());
    }
    let name = crate::payload::trimmed_string(request.model_manifest_entry.get("name"))
        .unwrap_or(request.model.as_str());
    // The base whose text encoder / VAE / tokenizer the conversion borrows. Named explicitly when
    // the manifest declares it so the user knows the exact second install to make; the sentence is
    // dropped rather than faked when it does not (a dir-sourced converter borrows nothing).
    let base = crate::payload::trimmed_string(mlx.get("convertBaseRepo"))
        .map(|repo| {
            let tier = crate::payload::trimmed_string(mlx.get("convertBaseSubdir"));
            match tier {
                Some(tier) => format!(
                    " It also requires the base model {repo} ({tier}) to be installed — the \
                     conversion borrows that base's text encoder, VAE, and tokenizer, which the \
                     single-file checkpoint does not contain."
                ),
                None => format!(
                    " It also requires the base model {repo} to be installed — the conversion \
                     borrows that base's text encoder, VAE, and tokenizer, which the single-file \
                     checkpoint does not contain."
                ),
            }
        })
        .unwrap_or_default();
    Err(WorkerError::InvalidPayload(format!(
        "{name} has not been converted yet — convert it from the Model Manager before \
         generating. It downloads as a transformer-only checkpoint, so there is nothing to load \
         until the conversion assembles a complete local model.{base}"
    )))
}

/// Typed, actionable preflight for an injected converted artifact that would not load (sc-24216).
///
/// The twin of [`unconverted_model_preflight`] for the case where the conversion DID run but its
/// output is no longer what the engine accepts — truncated, edited, or written by an older build.
/// For a converter with an engine-owned pin contract
/// ([`sceneworks_core::converted_artifacts::CONTRACT_CONVERTERS`], e.g. `supra2_img`) the engine's
/// own CHEAP check runs here (manifest, pin contract, file sizes, and a verification failure a
/// previous failed load recorded), so a damaged bundle is refused before any load with a message
/// naming the remedy instead of the engine's raw refusal. Same-size corruption is caught by the
/// engine's own load-time digests and explained by [`crate::converted_install::explain_failed_load`]
/// — never by hashing here, so a healthy install is hashed once per load. Every other converter
/// passes untouched.
#[cfg(native_lane)]
fn converted_artifact_preflight(request: &ImageRequest, dir: &std::path::Path) -> WorkerResult<()> {
    let Some(mlx) = request.model_manifest_entry.get("mlx") else {
        return Ok(());
    };
    if mlx.get("requiresConversion").and_then(Value::as_bool) != Some(true) {
        return Ok(());
    }
    let converter = mlx
        .get("converter")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    let Some(problem) = sceneworks_core::converted_artifacts::converted_artifact_problem(
        converter,
        dir,
        sceneworks_core::converted_artifacts::ArtifactCheck::Sizes,
    ) else {
        return Ok(());
    };
    Err(WorkerError::InvalidPayload(
        crate::converted_install::converted_files_message(
            crate::converted_install::model_display_name(
                &request.model_manifest_entry,
                &request.model,
            ),
            &problem,
        ),
    ))
}

/// Fail-loud gate for the CANDLE stub fallback (sc-20529) — the off-Mac twin of [`mlx_weights_gap`],
/// consulted by `run_image_generate_job` immediately before it would call `generate_stub_stream`.
///
/// [`unconverted_model_preflight`] fires from [`resolve_weights_dir`], which only runs once a route
/// has CLAIMED the job. That is enough for `text_to_image` / `image_to_image` / `character_image` on
/// an unconverted convert-at-install model: those modes reach the ladder's terminal
/// `is_candle_engine(&request.model) && request.mode != "edit_image"` arm → `CandleTxt2Img` →
/// `generate_candle_stream`, which resolves weights and propagates the typed error.
///
/// `edit_image` does NOT. `flux2_edit_candle_available` (and every sibling `…_available` weight
/// gate) collapses a resolver `Err` to `false`, and every terminal arm of the ladder excludes
/// `mode == "edit_image"` — so an unconverted `edit_image` job is claimed by NO route,
/// `prepare_candle_image_route` returns `None`, and the job fell through to `generate_stub_stream`
/// and COMPLETED with a procedural gradient. A silent stub in place of a typed refusal is exactly
/// the no-silent-fallback posture sc-5099 forbids, and it is worse than the raw loader error it
/// replaced: the user gets a plausible-looking asset instead of an error.
///
/// Returning the preflight's own message keeps ONE wording for the class — the dispatch seam and
/// the loader seam cannot drift into two different explanations of the same missing conversion.
/// `None` for every other job, so a model that legitimately stubs (test ids, unported families)
/// stubs exactly as before.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_weights_gap(request: &ImageRequest) -> Option<String> {
    match unconverted_model_preflight(request) {
        Ok(()) => None,
        Err(WorkerError::InvalidPayload(message)) => Some(message),
        // The preflight raises exactly one class; anything else would be a new variant whose
        // wording has not been reviewed for this seam, so fall through to the existing behaviour
        // rather than surfacing it as a stub-gap message.
        Err(_) => None,
    }
}

/// Resolve the weights snapshot directory: an explicit `modelPath` dir wins, else the
/// HuggingFace cache snapshot for the model repo. `None` when the model is not a known
/// engine family or its snapshot is absent. Available on the candle lane too (sc-5501): the
/// off-Mac SenseNova-U1 VQA / interleave handlers resolve their snapshot through it.
#[cfg(native_lane)]
pub(crate) fn resolve_weights_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    // Canonical FLUX3 identities must be proved by the worker-owned exact-revision receipt.
    // Never let an arbitrary cache HEAD, a distilled snapshot, or a modelPath override impersonate
    // the training base an adapter declares. Other families retain their historical resolution.
    let flux3_identity = match request.model.as_str() {
        "flux3_klein" => Some((
            "black-forest-labs/flux-3-image-klein",
            "8454017f9ebaada30c998fccd9f67a80c7580f68",
        )),
        "flux3_klein_base" => Some((
            "black-forest-labs/flux-3-image-klein-base",
            "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4",
        )),
        _ => None,
    };
    if let Some((repo, revision)) = flux3_identity {
        if model_repo_with_default(request, repo) != repo {
            return Err(WorkerError::InvalidPayload(format!(
                "{} requires pinned {repo}@{revision}; a different repo is not this checkpoint",
                request.model
            )));
        }
        let explicit = model_path_override(request)
            .map(|path| resolve_app_managed_model_dir(settings, &path, "Image modelPath"))
            .transpose()?;
        return Ok(Some(resolve_pinned_native_package(
            &settings.data_dir,
            repo,
            revision,
            &request.model,
            explicit.as_deref(),
        )?));
    }
    if let Some(dir) = resolve_model_path_dir(request, settings)? {
        converted_artifact_preflight(request, &dir)?;
        // Anima (epic 10512) is convert-at-install with a q4/q8/bf16 MATRIX: unlike other convert
        // models (a single flat dir), its injected `modelPath` is the converted ROOT holding `bf16/`,
        // `q8/`, `q4/` tier subdirs (written by `convert_anima_prequant`). Descend into the requested
        // tier — bespoke, like Boogu/Ideogram — so the packed DiT loads at the chosen precision. Every
        // other `modelPath` model resolves to the flat dir unchanged.
        if is_anima_model(&request.model) {
            return Ok(Some(anima_tier_subdir(&dir, request)));
        }
        return Ok(Some(dir));
    }
    // No `modelPath` on a convert-at-install model means the conversion has not produced its local
    // artifact. Refuse HERE with an actionable error rather than falling through to the raw
    // single-file source snapshot, which loads no further than candle-gen's "snapshot missing the
    // text_encoder/ component directory" (sc-20529).
    unconverted_model_preflight(request)?;
    let Some(model) = mlx_model(&request.model) else {
        return Ok(None);
    };
    let repo = model_repo(request, &model);
    let receipt_variant = requested_receipt_variant(request);
    let receipt_snapshot = crate::model_jobs::huggingface_receipt_weights_dir(
        &settings.data_dir,
        &repo,
        Some(&request.model),
        receipt_variant.as_deref(),
    );
    let pinned_turnkey_snapshot = match request.model.as_str() {
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit" => {
            pinned_turnkey_snapshot_for_request(
                &settings.data_dir,
                request,
                &repo,
                model.default_repo(),
                BOOGU_MLX_TURNKEY_REVISION,
            )
        }
        "ideogram_4" | "ideogram_4_turbo" => pinned_turnkey_snapshot_for_request(
            &settings.data_dir,
            request,
            &repo,
            model.default_repo(),
            IDEOGRAM_MLX_TURNKEY_REVISION,
        ),
        "z_image" => pinned_turnkey_snapshot_for_request(
            &settings.data_dir,
            request,
            &repo,
            ZIMAGE_MLX_TURNKEY_REPO,
            ZIMAGE_MLX_TURNKEY_REVISION,
        ),
        _ => None,
    };
    let has_pinned_turnkey_snapshot = pinned_turnkey_snapshot.is_some();
    let snapshot = pinned_turnkey_snapshot
        .or_else(|| receipt_snapshot.clone())
        .or_else(|| huggingface_snapshot_dir(&settings.data_dir, &repo));
    // A tier receipt already resolves to the exact self-contained tier directory.  Returning it
    // before the family pickers prevents a second `q4/q8/bf16` descent and, more importantly, keeps
    // all load inputs on the receipt side of the all-receipt-or-all-current boundary.
    if !has_pinned_turnkey_snapshot
        && receipt_snapshot
            .as_deref()
            .and_then(tier_key_from_resolved_dir)
            .is_some()
    {
        return Ok(receipt_snapshot);
    }
    // Ideogram 4 ships a turnkey with packed `q4/` (default) + `q8/` self-contained subdirs; point
    // the engine at the chosen quant's subdir rather than the repo root (epic 4725 / sc-5992),
    // mirroring the LTX bundle pattern. The packed weights auto-detect their quant on load. The
    // turbo variant (mlx-gen #488) shares the same turnkey — each subdir also carries the bundled
    // `turbo_lora.safetensors` the `ideogram_4_turbo` engine installs at load.
    if request.model == "ideogram_4" || request.model == "ideogram_4_turbo" {
        // bf16 (sc-8513, epic 8506) is the SHARED `SceneWorks/ideogram-4` repo's `bf16/` subdir — NOT
        // duplicated into the MLX turnkey. When a request opts into bf16 (advanced mlxQuantize<=0) AND it
        // is downloaded, resolve there (the dense weights load with no quantize); else the q4 (default)/q8
        // turnkey subdir. A partial bf16 download falls back rather than half-loading. sc-9650: wired on
        // the candle lane too — the candle Ideogram loader reads the same `transformer/model.safetensors`
        // layout and dense-loads bf16 via `linear_detect` (`resolve_quant` returns None for mlxQuantize<=0,
        // so no on-the-fly quant runs). The packed q4/q8 tiers still resolve from the `-mlx` turnkey below.
        #[cfg(native_lane)]
        {
            let wants_bf16 = request.quant_bits().is_some_and(|bits| bits <= 0);
            if wants_bf16 {
                if let Some(bf16) = huggingface_snapshot_dir(&settings.data_dir, IDEOGRAM_BF16_REPO)
                    .map(|root| root.join("bf16"))
                    .filter(|dir| dir.join("transformer/model.safetensors").is_file())
                {
                    return Ok(Some(bf16));
                }
            }
        }
        return Ok(snapshot.map(|root| ideogram_model_subdir(&root, request)));
    }
    // Boogu (epic 6387) ships a turnkey with pre-packed Q8 `base/ turbo/ edit/` subfolders (default) +
    // full-precision `*-bf16/`; point the engine at the variant's subfolder rather than the repo root
    // (the packed weights auto-detect their quant on load).
    if matches!(
        request.model.as_str(),
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit"
    ) {
        return Ok(snapshot.map(|root| boogu_model_subdir(&root, request)));
    }
    // Krea 2 Turbo (epic 7565) + Krea 2 Raw (epic 9992) ship a turnkey with self-contained quant subdirs
    // (Turbo: packed `q8/` default + `q4/`; Raw: packed `q8/` default + `q4/` + dense `bf16/`); point the
    // engine at the chosen quant's subdir rather than the repo root. The packed weights auto-detect their
    // quant on load, so the resolved `spec.quantize` is a no-op on them. `krea_model_subdir` also falls
    // back to any downloaded tier when the preferred one is absent — so Raw generates off the `bf16/`
    // training-base tier when only that is present, instead of failing at the repo root (no `tokenizer/`
    // there). Without this branch Raw fell through to `Ok(snapshot)` (the repo root) and load errored
    // with "tokenizer: No such file or directory" (epic 9992 P5/P6 wiring gap — the `krea_2_raw` engine
    // row already documents this resolver, but the branch was never added).
    if request.model == "krea_2_turbo" || request.model == "krea_2_raw" {
        return Ok(snapshot.map(|root| krea_model_subdir(&root, request)));
    }
    // Anima off-Mac (candle, sc-10676): dense bf16, NOT convert-at-install. There is no converted tier
    // artifact off-Mac (the `anima_quant` converter is macOS-only), so point at the raw
    // `circlestone-labs/Anima` `split_files/` root the candle loader reads directly (its DiT +
    // `text_encoders/qwen_3_06b_base` + `vae/qwen_image_vae`, the exact dir the GPU-validated anima smoke
    // used) and SKIP `anima_tier_subdir` — there are no bf16/q8/q4 tier subdirs off-Mac. The candle
    // loader's `resolve_split_files` also accepts the snapshot parent, so fall back to the snapshot root
    // if `split_files/` is somehow absent (a partial download then surfaces a loud load error, not a
    // silently-wrong dir). macOS never reaches here: a converted Anima install injects `modelPath` and
    // returns early via the `is_anima_model` tier-descent branch at the top of this fn.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if is_anima_model(&request.model) {
        return Ok(snapshot.map(anima_dense_split_files_dir));
    }
    // SANA 1600M off-Mac (candle, sc-11780, epic 8485): the `candle-gen-sana` pipeline
    // (`from_diffusers_snapshot`) loads the WHOLE `Efficient-Large-Model/Sana_1600M_1024px_diffusers`
    // HF snapshot (diffusers layout: `transformer/` + `vae/` + `text_encoder/`) — NOT the MLX-packed
    // `SceneWorks/Sana_1600M_1024px_mlx` turnkey the macOS/MLX path loads (which has no diffusers tree
    // the candle pipeline can read) and NOT a `q4/q8/bf16` tier subdir. So resolve the diffusers repo's
    // snapshot ROOT directly, BYPASSING the `standard_tier_subdir` descent below
    // (`sana_1600m` is registered there for the MLX turnkey, which would otherwise append a nonexistent
    // `q4/` to the diffusers root). The whole-repo download (manifest windows/linux entry, empty files
    // list) provisions this snapshot; an unfetched repo surfaces as a loud "snapshot not found" load
    // error above. macOS never compiles this branch (it keeps the MLX turnkey path).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if request.model == "sana_1600m" {
        return Ok(huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            SANA_CANDLE_DIFFUSERS_REPO,
            SANA_CANDLE_DIFFUSERS_REVISION,
        ));
    }
    // SANA-Sprint 1.6B off-Mac (candle, sc-11781, epic 8485): identical treatment to base SANA above —
    // the `candle-gen-sana` Sprint pipeline loads the WHOLE `Efficient-Large-Model/
    // Sana_Sprint_1.6B_1024px_diffusers` HF snapshot (same diffusers layout: `transformer/` + `vae/` +
    // `text_encoder/`), NOT the MLX-packed turnkey and NOT a `q4/q8/bf16` tier subdir. Resolve the
    // diffusers repo's snapshot ROOT directly, BYPASSING the `standard_tier_subdir` descent below
    // (`sana_sprint_1600m` is registered there for the MLX turnkey, which would otherwise append a
    // nonexistent `q4/`). macOS never compiles this branch (it keeps the MLX turnkey path).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if request.model == "sana_sprint_1600m" {
        return Ok(huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            SANA_SPRINT_CANDLE_DIFFUSERS_REPO,
            SANA_SPRINT_CANDLE_DIFFUSERS_REVISION,
        ));
    }
    // SenseNova-U1 needs NO bespoke branch (sc-14249). It had one from sc-13817 to sc-14249:
    // `candle-gen-sensenova` mmapped its backbone at a hardcoded F32 and hard-rejected
    // `spec.quantize`, so it could read ONLY the dense `bf16/` tier — and `standard_tier_subdir`'s
    // app-wide q8 default handed it packed weights it could not load. That is fixed IN THE ENGINE:
    // every backbone projection now packed-detects its MLX triple, so all three tiers load and the
    // family goes through the ordinary `standard_tier_subdir` descent below like every other
    // quant-matrix model. The old `sensenova_candle_dense_tier` force is deliberately gone rather
    // than left as a harmless-looking guard: leaving it would pin every candle sensenova job to the
    // heaviest tier forever, which is the whole thing sc-14249 exists to undo.
    // Catalog-wide quant-matrix models (sc-8513, epic 8506) ship as SceneWorks pre-quantized
    // turnkeys with self-contained `q4/` (default) + `q8/` + `bf16/` subdirs (replacing any
    // install-time convert); point the engine at the chosen tier's subdir rather than the repo root.
    // FLUX.2-dev was the pilot; each builtin's compiled product now declares
    // `mlx.standardTierLayout: true` when this resolver applies.
    // Qwen-Image 2.1 (sc-24112) — a SPLIT-REPO tier layout, and the reason it cannot use
    // `standard_tier_subdir`: that resolver descends into `<root>/<tier>/` of ONE repo, and 2.1's
    // three tiers do not live in one repo. bf16 IS the released upstream snapshot at its own root
    // (the converter refuses to emit a `bf16/` copy, and re-hosting unmodified weights would be a
    // §3 redistribution SceneWorks does not need to make), while q8 and q4 are `q8/`/`q4/` subdirs
    // of the SceneWorks re-host. So the tier picks the REPO first, then the subdir.
    //
    // Exactly the shape the Ideogram-4 branch above already has, with the halves swapped: there
    // the packed tiers are the turnkey and bf16 lives in a separate shared repo; here bf16 is the
    // upstream tree and the packed pair is the re-host. Both fall back to the resolved default
    // rather than half-loading when the requested tier is not on disk.
    //
    // Repo and revision come from the MANIFEST's own download rows, not from consts here: the
    // catalog is the pin authority (F-029), and reading it means the terminal story's revision pin
    // reaches the loader by editing one place. A row still carrying the null-SHA placeholder
    // resolves to nothing, so a pending tier falls back to the installed default instead of
    // pointing the loader at a snapshot directory that cannot exist.
    #[cfg(native_lane)]
    // Both 2.1 ids (base and its distilled Turbo sibling, sc-25907) are split-repo tier families:
    // the upstream snapshot is bf16 and the SceneWorks re-host carries q8/q4.
    if sceneworks_core::jobs_store::is_qwen_image_2_1_model(&request.model) {
        if let Some((tier_dir, _tier)) = qwen_image_2_1_tier_dir(settings, request)? {
            return Ok(Some(tier_dir));
        }
        return Ok(snapshot);
    }
    // Catalog-wide quant-matrix models (sc-8513, epic 8506) ship as SceneWorks pre-quantized
    // turnkeys with self-contained `q4/` (default) + `q8/` + `bf16/` subdirs (replacing any
    // install-time convert); point the engine at the chosen tier's subdir rather than the repo root.
    // The compiled product's `mlx.standardTierLayout` flag selects this resolver.
    if uses_standard_tier_layout(request) {
        return Ok(snapshot.map(|root| standard_tier_subdir(&root, request)));
    }
    Ok(snapshot)
}

/// The installed tier directory for a `qwen_image_2_1` request AND the tier it is, or `None` to
/// fall back to the model's default snapshot (sc-24112).
///
/// The identity rides alongside the path because the path alone cannot carry it: bf16 is the
/// upstream snapshot ROOT, whose basename is a commit SHA rather than a tier token, so
/// [`tier_key_from_resolved_dir`] answers `None` for it. Every consumer that prices or loads the
/// resolved directory (the Candle VRAM gate via [`gate_tier_key`], the Candle load quant, the MLX
/// candidate quant and reconcile) reads it back through [`tier_key_for_resolved_dir`], which maps
/// that root to `bf16` from the same catalog rows this resolver descends. The two agree by
/// construction and `resolved_tier_identity_round_trips_through_the_directory` pins it.
///
/// Resolution:
///
/// * an EXPLICIT `advanced.mlxQuantize` (`<= 0` or `>= 9` ⇒ bf16, `1..=4` ⇒ q4, `5..=8` ⇒ q8) is
///   honoured exactly
///   or refused: a pick whose tier is not installed is a typed error naming the tier, never a
///   silent substitution of another one (the FLUX.1 Candle precedent,
///   `candle_flux1_packed_requested_tier`). The Studio only sends a tier it resolved from the
///   installed set, so this fires for a stale replay or a direct API call, not for the picker;
/// * with NO selection, the catalog default (`mlx.quantize`, q8) first, then the remaining
///   installed tiers DENSEST FIRST (bf16 → q8 → q4), so a partial install never lands on the washed
///   q4 while a denser tier is on disk.
///
/// A tier is "installed" when its resolved directory holds a loadable `transformer/`. With nothing
/// installed at all an UNSELECTED request answers `None`, so the caller's ordinary "install the
/// model" path owns that error; an EXPLICIT pick with nothing installed is the install error itself
/// (sc-24114) — answering `None` there let the caller fall back to the default snapshot directory,
/// which loads a bf16 root that may not hold a `transformer/` at all instead of saying "install".
#[cfg(native_lane)]
pub(super) fn qwen_image_2_1_tier_dir(
    settings: &Settings,
    request: &ImageRequest,
) -> WorkerResult<Option<(PathBuf, &'static str)>> {
    const TIERS: [&str; 3] = ["bf16", "q8", "q4"];
    let qwen_checkpoint_tier = |bits: i64| -> &'static str {
        match bits {
            // A dense bit width (16 = bf16, 32 = f32) is the unquantized tier, never q8 (sc-24114).
            bits if bits <= 0 || bits >= 9 => "bf16",
            bits if bits <= 4 => "q4",
            _ => "q8",
        }
    };
    let explicit = qwen_image_2_1_selected_bits(&request.advanced)
        .map_err(|value| {
            WorkerError::InvalidPayload(format!(
                "{} advanced.mlxQuantize must be an integer bit count, got {value}",
                request.model
            ))
        })?
        .map(qwen_checkpoint_tier);
    let installed = |tier: &'static str| -> Option<(PathBuf, &'static str)> {
        let dir = qwen_image_2_1_declared_tier_dir(settings, request, tier)?;
        qwen_image_2_1_has_loadable_transformer(&dir).then_some((dir, tier))
    };
    if let Some(tier) = explicit {
        if let Some(resolved) = installed(tier) {
            return Ok(Some(resolved));
        }
        if TIERS.into_iter().all(|other| installed(other).is_none()) {
            return Err(WorkerError::InvalidPayload(format!(
                "{} is not installed on this machine (no tier is on disk, so tier {tier} cannot \
                 load). Install it in Model Manager, then retry.",
                request.model
            )));
        }
        return Err(WorkerError::InvalidPayload(format!(
            "{} tier {tier} is not installed; install it or pick an installed tier \
             (an explicit tier selection is never substituted with a different tier)",
            request.model
        )));
    }
    let default = request
        .model_manifest_entry
        .get("mlx")
        .and_then(|mlx| mlx.get("quantize"))
        .and_then(Value::as_i64)
        .map_or("q8", qwen_checkpoint_tier);
    Ok(installed(default).or_else(|| {
        TIERS
            .into_iter()
            .filter(|tier| *tier != default)
            .find_map(installed)
    }))
}

/// The bits a Qwen-Image 2.1 request selects exactly: an explicit value, else the verified
/// binding's frozen tier (served in place, so it is never substituted by an installed sibling).
/// `Ok(None)` leaves the manifest default with its installed fallback; a malformed explicit value
/// is returned for refusal.
#[cfg(native_lane)]
fn qwen_image_2_1_selected_bits(advanced: &JsonObject) -> Result<Option<i64>, String> {
    use sceneworks_core::payload_util::QuantSelection;
    match sceneworks_core::payload_util::quant_selection(advanced) {
        QuantSelection::Explicit(bits) | QuantSelection::Bound(bits) => Ok(Some(bits)),
        // A present `null` has always meant "no tier named" here.
        QuantSelection::Unselected | QuantSelection::Null => Ok(None),
        QuantSelection::Malformed(value) => Err(value.to_string()),
    }
}

/// The "has a loadable backbone" probe the shared resolver applies, spelled for BOTH layouts this
/// family ships (sc-24114):
///
/// * the upstream diffusers bf16 snapshot — a sharded `transformer/` with its index, or one
///   `diffusion_pytorch_model.safetensors`;
/// * a packed q8/q4 tier as the engine's `convert::prequantize_turnkey` writes it — ONE
///   `transformer/model.safetensors` (the sharded index is deliberately absent) beside a
///   `config.json` carrying the `quantization` marker the engine's packed-detect reads.
#[cfg(native_lane)]
fn qwen_image_2_1_has_loadable_transformer(dir: &Path) -> bool {
    let transformer = dir.join("transformer");
    let dense = transformer
        .join("diffusion_pytorch_model.safetensors.index.json")
        .is_file()
        || transformer
            .join("diffusion_pytorch_model.safetensors")
            .is_file();
    let packed = transformer.join("model.safetensors").is_file()
        && std::fs::read_to_string(transformer.join("config.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .is_some_and(|config| config.get("quantization").is_some_and(Value::is_object));
    dense || packed
}

/// The tier a resolved directory IS, reading the request's catalog entry when the basename is not
/// a tier token (sc-24112).
///
/// Only a split-repo family reaches the second arm: `qwen_image_2_1`'s bf16 tier is a whole upstream
/// snapshot ROOT (`files` empty), resolved to `<library>/models--<org>--<name>/snapshots/<rev>`.
/// That directory is matched back to its catalog row by repo AND revision — the same pin
/// [`qwen_image_2_1_declared_tier_dir`] descends — so the gate, the Candle load quant and the MLX
/// reconcile see `bf16` instead of falling back to the request/manifest default (`q8`) and asking a
/// dense root for a packed load.
///
/// Scoped to [`SPLIT_REPO_TIER_MODELS`] on purpose: other catalog rows are also tier-tagged
/// whole-repo roots (the Candle SANA and Wan diffusers snapshots), and relabelling THEIR resolved
/// directories would move their gate pricing and load quant, which nothing here has validated.
#[cfg(native_lane)]
pub(crate) fn tier_key_for_resolved_dir(
    manifest_entry: &JsonObject,
    dir: &Path,
) -> Option<&'static str> {
    tier_key_from_resolved_dir(dir).or_else(|| split_repo_root_tier_key(manifest_entry, dir))
}

/// A dense snapshot root and packed q4/q8 subtrees in another repository require the root-tier
/// resolver. This is read from the compiled product downloads rather than a model-ID list.
#[cfg(native_lane)]
fn has_split_repo_root_tiers(model_id: &str) -> bool {
    let Some(spec) = sceneworks_core::registry::builtin_model_spec(model_id) else {
        return false;
    };
    let Some(downloads) = spec
        .product
        .fields
        .get("downloads")
        .and_then(Value::as_array)
    else {
        return false;
    };
    let tier = |name: &str, root: bool| {
        downloads.iter().find(|row| {
            row.get("variant").and_then(Value::as_str) == Some(name)
                && row
                    .get("files")
                    .and_then(Value::as_array)
                    .is_some_and(|files| {
                        if root {
                            files.is_empty()
                        } else {
                            files.contains(&Value::from(format!("{name}/*")))
                        }
                    })
        })
    };
    let (Some(dense), Some(q4), Some(q8)) =
        (tier("bf16", true), tier("q4", false), tier("q8", false))
    else {
        return false;
    };
    q4["repo"] == q8["repo"] && dense["repo"] != q4["repo"]
}

#[cfg(all(test, native_lane))]
#[test]
fn split_repo_root_layout_follows_compiled_download_shapes() {
    assert!(has_split_repo_root_tiers("qwen_image_2_1"));
    assert!(has_split_repo_root_tiers("qwen_image_2_1_turbo"));
    assert!(!has_split_repo_root_tiers("flux3_klein"));
    assert!(!has_split_repo_root_tiers("qwen_image"));
    assert!(!has_split_repo_root_tiers("unregistered_model"));
}

#[cfg(native_lane)]
pub(super) fn split_repo_root_tier_key(
    manifest_entry: &JsonObject,
    dir: &Path,
) -> Option<&'static str> {
    use sceneworks_core::model_artifacts::artifact_selection::{
        is_pending_artifact_download, model_download_for_variant,
    };
    let model_id = manifest_entry.get("id").and_then(Value::as_str)?;
    if !has_split_repo_root_tiers(model_id) {
        return None;
    }
    let revision = dir.file_name()?.to_str()?;
    let snapshots = dir.parent()?;
    if snapshots.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let repo_dir = snapshots.parent()?.file_name()?.to_str()?;
    let entry = Value::Object(manifest_entry.clone());
    ["bf16", "q8", "q4"].into_iter().find(|tier| {
        let Some(download) = model_download_for_variant(&entry, tier) else {
            return false;
        };
        let root_row = download
            .get("files")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty);
        root_row
            && !is_pending_artifact_download(&download)
            && download.get("revision").and_then(Value::as_str) == Some(revision)
            && download
                .get("repo")
                .and_then(Value::as_str)
                .is_some_and(|repo| repo_dir == format!("models--{}", repo.replace('/', "--")))
    })
}

/// Where `tier`'s snapshot directory would be, read off the request's OWN manifest entry.
///
/// `None` when the catalog declares no such tier, when its revision is still the sc-24112
/// null-SHA placeholder (the artifact is not published, so there is nothing to point at), or when
/// that pinned snapshot is not in the cache. The `files` glob's leading path component is what
/// names the subdir — `["q8/*"]` ⇒ `q8/` — and an empty `files` (the whole-repo upstream bf16 row)
/// means the snapshot ROOT, which is exactly the distinction that makes this family split-repo.
#[cfg(native_lane)]
fn qwen_image_2_1_declared_tier_dir(
    settings: &Settings,
    request: &ImageRequest,
    tier: &str,
) -> Option<PathBuf> {
    let entry = Value::Object(request.model_manifest_entry.clone());
    let download =
        sceneworks_core::model_artifacts::artifact_selection::model_download_for_variant(
            &entry, tier,
        )?;
    if sceneworks_core::model_artifacts::artifact_selection::is_pending_artifact_download(&download)
    {
        return None;
    }
    let repo = download.get("repo").and_then(Value::as_str)?;
    let revision = download.get("revision").and_then(Value::as_str)?;
    let files: Vec<String> = download
        .get("files")
        .and_then(Value::as_array)
        .map(|files| {
            files
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    // An install made at a revision the pin has since SUPERSEDED (sc-25909: the q8/q4 re-host
    // republished to add each tier's NOTICE) is carried forward cache-only before the exact lookup.
    let root = crate::model_jobs::huggingface_pinned_tier_snapshot_dir(
        &settings.data_dir,
        repo,
        revision,
        &files,
    )?;
    let subdir = files
        .first()
        .map(String::as_str)
        .and_then(|pattern| pattern.split('/').next())
        // Confined to a single plain directory name. The value is a checked-in catalog string
        // rather than user input, but a glob is still a string a future edit could widen, and a
        // traversal component here would point the LOADER outside the cache — cheap to refuse.
        .filter(|component| {
            !component.is_empty()
                && *component != "*"
                && !component.starts_with('.')
                && sceneworks_core::ids::is_safe_id(component)
        });
    match subdir {
        Some(component) => Some(root.join(component)),
        None => Some(root),
    }
}

/// FLUX.1's generic Candle route reads only the hosted packed q4/q8 turnkeys. The shared resolver
/// deliberately has a best-available fallback chain for ordinary matrix models; using it unchecked
/// here could turn an explicit FLUX q4/q8 selection into a different tier (or a dense root). Probe the
/// shared resolver with the exact requested tier, then accept its answer only when the resolved
/// directory names that same tier.
///
/// This is intentionally limited to generic Candle text-to-image. The existing FLUX IP-Adapter and
/// strict-control providers retain their own established weight-resolution and load contracts.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_flux1_packed_requested_tier(
    request: &ImageRequest,
) -> WorkerResult<&'static str> {
    candle_flux1_packed_tier_key(&request.advanced).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} on Candle supports only its hosted packed q4 or q8 tiers",
            request.model
        ))
    })
}

/// Portable core of [`candle_flux1_packed_requested_tier`]: an explicit value, else the verified
/// binding's frozen tier, else the q4 default; anything but q4/q8 is `None` (refused).
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(crate) fn candle_flux1_packed_tier_key(advanced: &JsonObject) -> Option<&'static str> {
    use sceneworks_core::payload_util::QuantSelection;
    match sceneworks_core::payload_util::quant_selection(advanced) {
        // A present `null` has always meant "no tier named" here: the q4 default.
        QuantSelection::Unselected | QuantSelection::Null => Some("q4"),
        QuantSelection::Explicit(4) | QuantSelection::Bound(4) => Some("q4"),
        QuantSelection::Explicit(8) | QuantSelection::Bound(8) => Some("q8"),
        _ => None,
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_flux1_packed_resolution_is_exact(
    requested_tier: &str,
    resolved: &Path,
) -> bool {
    resolved.file_name().and_then(|name| name.to_str()) == Some(requested_tier)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_candle_flux1_packed_weights_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    let requested_tier = candle_flux1_packed_requested_tier(request)?;
    let mut probe = request.clone();
    probe.advanced.insert(
        "mlxQuantize".to_owned(),
        Value::from(if requested_tier == "q4" { 4 } else { 8 }),
    );
    let resolved = resolve_weights_dir(&probe, settings)?;
    match resolved {
        Some(dir) if candle_flux1_packed_resolution_is_exact(requested_tier, &dir) => Ok(Some(dir)),
        Some(_) | None => Err(WorkerError::InvalidPayload(format!(
            "{} requested Candle packed {requested_tier}, but that exact artifact is not installed; \
             refusing to fall back to a different tier",
            request.model
        ))),
    }
}

/// Whether completed-image telemetry must replay the Candle-only FLUX.1 exact-tier resolver.
///
/// The metrics consumer is shared by MLX and Candle builds, but the direct packed FLUX.1 route is
/// not: macOS must keep using its ordinary MLX resolver and default-tier semantics. Keeping that
/// distinction in a pure cfg-sensitive predicate makes the build boundary testable without making
/// [`resolve_candle_flux1_packed_weights_dir`] (or any Candle production route) available on macOS.
#[cfg(native_lane)]
pub(super) fn completed_image_metrics_use_candle_flux1_exact_resolution(model: &str) -> bool {
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let exact = matches!(model, "flux_schnell" | "flux_dev");
    #[cfg(target_os = "macos")]
    let exact = {
        let _ = model;
        false
    };
    exact
}

/// Re-resolve the tier used by a completed image for telemetry only. The Candle-only FLUX.1 lane
/// replays its exact q4/q8 resolver; macOS and all other families replay the ordinary resolver that
/// selected their production weights. The cfg around the Candle call is intentional: a shared
/// metrics helper must never make the off-Mac Candle route part of a macOS build.
#[cfg(native_lane)]
pub(super) fn resolve_completed_image_metrics_tier_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> Option<PathBuf> {
    let _use_candle_exact =
        completed_image_metrics_use_candle_flux1_exact_resolution(&request.model);
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if _use_candle_exact {
        return resolve_candle_flux1_packed_weights_dir(request, settings)
            .ok()
            .flatten();
    }
    #[cfg(target_os = "macos")]
    debug_assert!(
        !_use_candle_exact,
        "macOS metrics must not select the Candle-only FLUX.1 resolver"
    );
    resolve_weights_dir(request, settings).ok().flatten()
}

#[cfg(target_os = "macos")]
pub(crate) fn resolved_mlx_artifact_tier(
    weights_dir: &Path,
    quant_bits: Option<i64>,
) -> Option<&'static str> {
    tier_key_from_resolved_dir(weights_dir)
        .or_else(|| quant_bits.map(|bits| if bits <= 4 { "q4" } else { "q8" }))
}

#[cfg(target_os = "macos")]
pub(crate) fn resolved_mlx_artifact_tier_for_model(
    model_id: &str,
    weights_dir: &Path,
    quant_bits: Option<i64>,
) -> Option<&'static str> {
    fixed_mlx_artifact_tier(model_id)
        .or_else(|| resolved_mlx_artifact_tier(weights_dir, quant_bits))
}

#[cfg(target_os = "macos")]
fn fixed_artifact_tier_matches(
    fixed_artifact_tier: Option<&str>,
    effective_tier: Option<&str>,
) -> bool {
    match (fixed_artifact_tier, effective_tier) {
        (Some(fixed), Some(actual)) => fixed == actual,
        _ => true,
    }
}

/// Resolve immutable provenance beside the exact artifact the MLX loader will open. HF identity is
/// supplied by a completed download receipt; local/converted identity is supplied only by the
/// worker-owned install receipt. Request and manifest provenance fields are intentionally ignored.
#[cfg(target_os = "macos")]
pub(crate) fn resolved_mlx_artifact_provenance(
    request: &ImageRequest,
    settings: &Settings,
    repo: &str,
    weights_dir: &Path,
    effective_tier: Option<&'static str>,
) -> WorkerResult<Option<crate::model_jobs::ResolvedArtifactProvenance>> {
    if let Some(path) = model_path_override(request) {
        let managed_root = resolve_app_managed_model_dir(settings, &path, "Image modelPath")?;
        let resolved_weights_dir = std::fs::canonicalize(weights_dir)?;
        if !resolved_weights_dir.starts_with(&managed_root) {
            return Ok(None);
        }
        let provenance = crate::model_jobs::app_managed_artifact_provenance(&resolved_weights_dir)?;
        return Ok(provenance.and_then(|mut provenance| {
            if provenance.fixed_artifact_tier.is_none() {
                provenance.fixed_artifact_tier =
                    tier_key_from_resolved_dir(&resolved_weights_dir).map(str::to_owned);
            }
            let matches = fixed_artifact_tier_matches(
                provenance.fixed_artifact_tier.as_deref(),
                effective_tier,
            );
            matches.then_some(provenance)
        }));
    }

    let requested_variant = requested_receipt_variant(request);
    let mut variants = vec![requested_variant, effective_tier.map(str::to_owned), None];
    variants.dedup();
    for variant in variants {
        // Probe WITHOUT repair first: this loop walks several candidate variants, and only the one
        // whose path actually holds the weights being loaded has earned a stamp write.
        let Some(mut resolved) = crate::model_jobs::huggingface_receipt_weights(
            &settings.data_dir,
            repo,
            Some(&request.model),
            variant.as_deref(),
            crate::model_jobs::ProvenanceRepair::Skip,
        ) else {
            continue;
        };
        if !weights_dir.starts_with(&resolved.path) {
            continue;
        }
        // This receipt names the artifact being loaded. If it never carried a tree-stamp baseline
        // (a backfilled or pre-stamping install), establish one now rather than leave the model
        // pinned to the legacy selector for good — see `ProvenanceRepair` (sc-16482). A receipt
        // whose stamp is present but MISMATCHED is drift and is left alone to fail closed.
        if resolved.provenance.is_none() {
            if let Some(repaired) = crate::model_jobs::huggingface_receipt_weights(
                &settings.data_dir,
                repo,
                Some(&request.model),
                variant.as_deref(),
                crate::model_jobs::ProvenanceRepair::Allow,
            ) {
                resolved = repaired;
            }
        }
        if let Some(mut provenance) = resolved.provenance {
            if provenance.fixed_artifact_tier.is_none() {
                provenance.fixed_artifact_tier =
                    tier_key_from_resolved_dir(weights_dir).map(str::to_owned);
            }
            if fixed_artifact_tier_matches(
                provenance.fixed_artifact_tier.as_deref(),
                effective_tier,
            ) {
                return Ok(Some(provenance));
            }
        }
    }
    Ok(None)
}

// The paragraph that used to sit here documented `DENSE_TE_TIER_MODELS` — a const that had already
// moved to `tier_resolver.rs` and is DELETED entirely by sc-15799, so its doc block was orphaned onto
// `dense_tier_subdir` below and read as that function's documentation. The rationale it carried
// (quantize the transformer, keep the text encoder bf16; `resolve_quant` must therefore return `None`)
// now lives with `tier_resolver::is_dense_te_tier`, beside the manifest flag that is the only way to
// request it.

/// The DENSE (`bf16/`) tier of a SceneWorks quant-matrix turnkey, or `root` unchanged for a flat
/// diffusers snapshot (sc-10614).
///
/// The FALLBACK tier resolver for the candle SDXL edit / IP-Adapter lanes (`sdxl_edit_candle.rs`,
/// `sdxl_ipadapter.rs`) on a NON-standard-layout repo. As of sc-10813 those lanes packed-detect and,
/// for a standard-tier turnkey (`mlx.standardTierLayout`), descend through [`standard_tier_subdir`]
/// (honouring `advanced.mlxQuantize`, exactly like the txt2img lane, sc-10767) — so the packed q4/q8
/// tiers now serve edit / inpaint / IP-Adapter, not just txt2img. This helper stays the else-branch:
/// a flat upstream diffusers repo (`stabilityai/stable-diffusion-xl-base-1.0`, `SG161222/RealVisXL_V5.0`)
/// roots its `unet/` and is returned untouched, and a non-standard tiered turnkey resolves its dense
/// `bf16/` (an SDXL turnkey has no component tree at its root, so the loader would find no `unet/`).
///
/// A flat snapshot roots its backbone dir — `unet/` for the SDXL family, `transformer/` for the DiTs
/// — and is returned untouched, so the existing two models keep resolving exactly as before. Same
/// backbone split the `apps/rust-api` training-readiness gate keys on (sc-10613).
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn dense_tier_subdir(root: PathBuf) -> PathBuf {
    let roots_a_component_tree = root.join("unet").is_dir() || root.join("transformer").is_dir();
    if !roots_a_component_tree && root.join("bf16").is_dir() {
        return root.join("bf16");
    }
    root
}

/// Whether `model` is one of the three Anima catalog ids (epic 10512). Anima is convert-at-install with
/// a bespoke tier resolver, so — like Ideogram/Boogu/Krea — it is NOT in [`STANDARD_TIER_MODELS`].
#[cfg(native_lane)]
pub(super) fn is_anima_model(model: &str) -> bool {
    matches!(model, "anima_base" | "anima_aesthetic" | "anima_turbo")
}

/// Pick the tier subdir of a converted Anima `root` (the injected `modelPath`, holding `bf16/ q8/ q4/`,
/// each a `diffusion_models/<variant>.safetensors` + dense `text_encoders/` + `vae/` tree the Anima
/// loader reads): `bf16/` when the request opts out of quantization (`advanced.mlxQuantize <= 0`), `q4/`
/// when it opts into Q4 (`1..=4`), else the default **`q8/`** (sc-10714). Falls back through q8 → bf16 →
/// q4 → `root` so a partially-written artifact surfaces as a load error rather than a silent half-load
/// onto the low-fidelity q4. A tier is "present" when its `diffusion_models/` holds a `.safetensors` DiT
/// (packed OR dense bf16). Mirrors [`standard_tier_subdir`], but keyed on Anima's `split_files` layout
/// rather than `transformer/`, and — unlike the standard q4-first convention — defaults to q8 (below).
#[cfg(native_lane)]
pub(super) fn anima_tier_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let bits = request.quant_bits();
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        // A hidden `._*.safetensors` AppleDouble sidecar is not a DiT (SceneWorks#1333).
        let has_dit = std::fs::read_dir(dir.join("diffusion_models"))
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|entry| !sceneworks_core::lora_family::is_hidden_file(&entry.path()))
                    .any(|entry| {
                        entry
                            .file_name()
                            .to_string_lossy()
                            .ends_with(".safetensors")
                    })
            })
            .unwrap_or(false);
        has_dit.then_some(dir)
    };
    // Default (no explicit `mlxQuantize`) → the app-wide Q8 default (epic 10721 / sc-10726), CLAMPED UP
    // to the model's per-model quality floor (`mlx.minQualityTier`, sc-10731). This REPLACES the sc-10714
    // anima-specific `None => "q8"` hardcode with the SHARED, floor-driven [`preferred_tier`]: Anima
    // base/aesthetic declare `minQualityTier: q8` in the manifest, so their default is now floor-derived
    // — the fix for base/aesthetic rendering WASHED at q4 (q4 weight-quant error amplified by CFG 4.5 over
    // 30 steps) is a general mechanism, not a resolver special-case. Turbo declares NO floor (it is
    // CFG-free, so q4 is acceptable there) and rides the plain q8 default. bf16 (explicit `<= 0`) is there
    // for max fidelity/speed on this small 2B DiT, and an explicit q4 pick is still honored (the web
    // flags a below-floor pick with an advisory). Fallback prefers the clean tiers (q8 → bf16 → q4) so a
    // partial install never silently lands on the washed q4.
    //
    // `nvfp4: false` (sc-11042): Anima is an MLX convert-at-install family and hosts no NVFP4 tier — the
    // lane is candle/Blackwell-only. Passing `false` keeps this resolver byte-identical to its pre-sc-11042
    // behavior; wire it here if an Anima NVFP4 tier is ever converted.
    let preferred = preferred_tier(bits, min_quality_floor(request), false);
    // sc-12279 generalized: Anima ships no `model_index.json`, so route the chain through the concrete
    // `anima_tier_complete` predicate — a torn tier (DiT present, text-encoder/VAE absent) now falls
    // through to a complete sibling instead of reaching the loader and dying on the missing file.
    pick_loadable_tier(
        &[preferred, "q8", "bf16", "q4"],
        &present,
        &sceneworks_core::mlx_tier_completeness::anima_tier_complete,
    )
    .unwrap_or_else(|| root.to_path_buf())
}

/// The candle off-Mac Anima weights dir (sc-10676): the `split_files/` subdir of the HF snapshot `root`
/// when it holds `diffusion_models/` (the dense DiT tree `runtime_cuda::providers::anima::loader::resolve_split_files`
/// reads — the exact dir the GPU-validated anima smoke used), else `root` itself. The candle loader also
/// accepts the snapshot parent, so falling back to `root` keeps a partial download a loud load error, not
/// a silently-wrong dir. Anima has NO convert-at-install tier off-Mac (the `anima_quant` converter is
/// macOS-only), so this deliberately SKIPS [`anima_tier_subdir`]'s bf16/q8/q4 tier descent — off-Mac is
/// always dense bf16.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn anima_dense_split_files_dir(root: PathBuf) -> PathBuf {
    let split = root.join("split_files");
    if split.join("diffusion_models").is_dir() {
        split
    } else {
        root
    }
}

/// The Ideogram 4 tier subdir a `mlxQuantize` request needs fetched ON DEMAND — `Some("q8")` when the
/// request opts into Q8 (`> 4`), else `None` (the shipped default `q4/`, which the catalog download
/// pulls; bf16 is a SEPARATE catalog repo the user opts into on the Models page, never an on-demand
/// fetch). The FETCH-side helper for [`ensure_ideogram_tier_present`] (which tier to pull). The LOAD-side
/// resolver [`ideogram_model_subdir`] no longer shares this: as of sc-10777 it routes its default through
/// the floor-aware [`preferred_tier`] (so a floored default clamps up to `mlx.minQualityTier`, capped by
/// installed), while this fetch helper stays keyed on the explicit pick only — no shipping Ideogram model
/// declares a floor, so the two still agree for every current model. Mirrors [`boogu_tier_subdir`].
pub(super) fn ideogram_tier_subdir(bits: Option<i64>) -> Option<&'static str> {
    match bits {
        Some(b) if b > 4 => Some("q8"),
        _ => None,
    }
}

/// Pick the engine-complete packed subdir of an Ideogram 4 turnkey `root`: `q8/` when the request
/// opts into Q8 (`advanced.mlxQuantize: 8`) AND it is downloaded, `q4/` for an explicit Q4 pick
/// (`1..=4`), else — with NO explicit `mlxQuantize` — the **`q8/`** default (epic 10721 / sc-10726),
/// CLAMPED UP to the model's per-model quality floor (`mlx.minQualityTier`, sc-10731 via the shared
/// [`preferred_tier`], sc-10777) and DOWN to what's installed (only `q4/` on disk ⇒ q4). Falls back to
/// `root` if neither subdir is present (a partially-downloaded bundle surfaces as a load error rather
/// than a silent half-load). The `q8/` tier is an on-demand download fetched by
/// [`ensure_ideogram_tier_present`] on an explicit opt-in (sc-9607); this resolver never triggers that
/// fetch for the plain default — it simply prefers q8 when it happens to be on disk. Ideogram's turnkey
/// carries only `q4/`/`q8/` (bf16 is a separate catalog repo), so the clean-tiers fallback here is
/// q8 ⇆ q4 and a bf16 floor has no in-turnkey tier to land on (it falls through to the best installed).
pub(super) fn ideogram_model_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let bits = request.quant_bits();
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        dir.join("transformer/model.safetensors")
            .is_file()
            .then_some(dir)
    };
    // The tier the request prefers: an explicit `mlxQuantize` pick maps directly (`<=0` → bf16, `>4` →
    // q8, `1..=4` → q4), else the app-wide q8 default (epic 10721 / sc-10726) CLAMPED UP to the model's
    // per-model quality floor (`mlx.minQualityTier`, sc-10731) — the SAME shared, floor-aware logic as
    // [`standard_tier_subdir`] / [`anima_tier_subdir`] (sc-10777, routing this resolver through
    // [`preferred_tier`] instead of its own non-floor-aware q8 default). No shipping Ideogram model
    // declares a floor, so this is byte-identical to the prior q8-first default for every current model —
    // it just keeps the worker default path from silently landing below a floor should one ever be
    // declared. Ideogram's turnkey carries only `q4/`/`q8/` (bf16 is the separate `SceneWorks/ideogram-4`
    // catalog repo, never an in-turnkey subdir), so a preferred `bf16` — an explicit `<=0` pick, or a
    // hypothetical bf16 floor — simply isn't present and falls through the clean-tiers chain (q8 → q4) to
    // the best installed tier, exactly as the prior resolver did.
    //
    // `nvfp4: false` (sc-11042): the Ideogram turnkey hosts no NVFP4 tier. Byte-identical to its
    // pre-sc-11042 behavior; wire it here if one is ever converted.
    let preferred = preferred_tier(bits, min_quality_floor(request), false);
    // sc-12279: the Ideogram turnkey ships a per-tier `model_index.json`, so the shared chain prefers
    // a tier whose declared component tree is fully on disk over a torn one.
    pick_loadable_tier(&[preferred, "q8", "q4"], &present, &tier_components_present)
        .unwrap_or_else(|| root.to_path_buf())
}

/// Requested Boogu download subfolder. Even the default Q8 variant can be absent when only
/// a sibling variant is installed. The load-side resolver separately chooses among complete tiers.
pub(super) fn boogu_tier_subdir(variant: &str, bits: Option<i64>) -> String {
    match crate::payload::tier_for_bits(bits, "q8") {
        "bf16" => format!("{variant}-bf16"),
        "q4" => format!("{variant}-q4"),
        _ => variant.to_owned(),
    }
}

/// Pick the engine-complete subfolder of a Boogu turnkey `root` for the requested variant. Each
/// catalog id maps to a variant folder: `boogu_image`→`base`, `boogu_image_turbo`→`turbo`,
/// `boogu_image_edit`→`edit`. **Q8 is the shipped default** (the pre-packed `<variant>/` folder),
/// CLAMPED UP to the model's per-model quality floor (`mlx.minQualityTier`, sc-10731 via the shared
/// [`preferred_tier`], sc-10777 — a bf16 floor raises the picker-less default to `<variant>-bf16/`); an
/// explicit advanced `mlxQuantize` selects another tier (sc-8513, epic 8506): `<=0` → the dense
/// `<variant>-bf16/`, `1..=4` → the packed `<variant>-q4/`. Falls back through Q8 → q4 → bf16 → `root`
/// when the requested tier isn't downloaded, so a partial bundle surfaces as a load error rather than
/// a silent half-load (a floor tier not on disk falls to the best installed). (The non-default tiers are
/// on-demand downloads fetched by [`ensure_boogu_tier_present`] before this resolves, sc-6568/sc-8513.)
pub(super) fn boogu_model_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let variant = match request.model.as_str() {
        "boogu_image_turbo" => "turbo",
        "boogu_image_edit" => "edit",
        _ => "base",
    };
    let bits = request.quant_bits();
    // q4/q8 ship a single packed transformer file; bf16 is the dense diffusers tree (SHARDED → only
    // the `.index.json`). Accept either shape.
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        let packed = dir
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file();
        let sharded = dir
            .join("transformer/diffusion_pytorch_model.safetensors.index.json")
            .is_file();
        (packed || sharded).then_some(dir)
    };
    let q4 = format!("{variant}-q4");
    let bf16 = format!("{variant}-bf16");
    // Boogu names its tiers `<variant>` (the packed Q8 shipped default), `<variant>-q4` and
    // `<variant>-bf16` — map a generic [`preferred_tier`] tier name onto that layout.
    let variant_folder = |tier: &str| -> String {
        match tier {
            "q8" => variant.to_owned(),
            other => format!("{variant}-{other}"),
        }
    };
    // The tier the request prefers: an explicit `mlxQuantize` pick maps directly (`<=0` → bf16, `1..=4` →
    // q4, else Q8 — the same mapping the old `boogu_tier_subdir` load-path branch used), else the app-wide
    // Q8 default (epic 10721 / sc-10726) CLAMPED UP to the model's per-model quality floor
    // (`mlx.minQualityTier`, sc-10731) — the SAME shared, floor-aware logic as [`standard_tier_subdir`] /
    // [`anima_tier_subdir`] (sc-10777, routing this resolver through [`preferred_tier`] instead of its own
    // non-floor-aware Q8 default). No shipping Boogu model declares a floor, so this is byte-identical to
    // the prior Q8-first default for every current variant; the routing keeps the resolver from silently
    // landing below a floor should one ever be declared — a bf16 floor would raise the picker-less default
    // from the packed Q8 up to the dense `<variant>-bf16`, capped by what's installed. The clean-tiers
    // fallback (Q8 → q4 → bf16) is unchanged, so a partial bundle still surfaces as a load error.
    //
    // `nvfp4: false` (sc-11042): Boogu's `<variant>-bf16` MLX layout hosts no NVFP4 tier. Byte-identical
    // to its pre-sc-11042 behavior; wire it here if one is ever converted.
    let preferred = preferred_tier(bits, min_quality_floor(request), false);
    // sc-12279 generalized: Boogu ships no `model_index.json`, so route the folder chain through the
    // concrete `boogu_tier_complete` predicate — a torn tier (transformer present, `mllm/tokenizer.json`
    // or VAE absent) now falls through to a complete sibling instead of crashing the loader on the
    // missing tokenizer. Chain is folder names (`<variant>`, `<variant>-q4`, `<variant>-bf16`).
    let chain = [variant_folder(preferred), variant.to_owned(), q4, bf16];
    let chain_refs: Vec<&str> = chain.iter().map(String::as_str).collect();
    pick_loadable_tier(
        &chain_refs,
        &present,
        &sceneworks_core::mlx_tier_completeness::boogu_tier_complete,
    )
    .unwrap_or_else(|| root.to_path_buf())
}

/// Pick the engine-complete packed subdir of a Krea 2 Turbo turnkey `root`: `q4/` when the request opts
/// into Q4 (`advanced.mlxQuantize <= 4`) AND it is downloaded, else the default `q8/` (the shipped
/// default — the P1-validated near-lossless quant), CLAMPED UP to the model's per-model quality floor
/// (`mlx.minQualityTier`, sc-10731 via the shared [`preferred_tier`], sc-10845 — a bf16 floor raises the
/// picker-less default to the dense `bf16/`). Falls back to whichever subdir is present, then `root`, so
/// a partially-downloaded bundle surfaces as a load error rather than a silent half-load (a floor tier not
/// on disk falls to the best installed). The turnkey (`SceneWorks/krea-2-turbo-mlx`, sc-7573) carries one
/// `from_snapshot`-loadable subdir per quant (each with a packed
/// `transformer/diffusion_pytorch_model.safetensors`); the loader auto-detects the packed quant, so the
/// resolved `spec.quantize` is a no-op on it. Mirrors [`ideogram_model_subdir`] (q4/q8 subdirs) with
/// Boogu's packed-transformer filename and a Q8-default selection.
pub(super) fn krea_model_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    krea_model_subdir_gated(root, request, nvfp4_host_eligible())
}

/// [`krea_model_subdir`] with the NVFP4 **host** gate passed in rather than probed (sc-11042). Split
/// out for testability, exactly like [`standard_tier_subdir_gated`]; production has one caller.
pub(super) fn krea_model_subdir_gated(
    root: &Path,
    request: &ImageRequest,
    nvfp4_host: bool,
) -> PathBuf {
    let bits = request.quant_bits();
    // q4/q8 ship a single packed transformer file; the bf16 tier (sc-8513) is the dense diffusers
    // tree (SHARDED → only the `.index.json`). Accept either shape.
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        let packed = dir
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file();
        let sharded = dir
            .join("transformer/diffusion_pytorch_model.safetensors.index.json")
            .is_file();
        (packed || sharded).then_some(dir)
    };
    // The tier the request prefers: an explicit `mlxQuantize` pick maps directly (`<=0` → bf16 the dense
    // base which krea's loader takes with no quantize, `>4` → q8, `1..=4` → q4 — the SAME mapping krea's
    // own match used), else the app-wide q8 default (epic 10721 / sc-10726, the P1-validated near-lossless
    // quant) CLAMPED UP to the model's per-model quality floor (`mlx.minQualityTier`, sc-10731) — the SAME
    // shared, floor-aware logic as `standard_tier_subdir` / `anima_tier_subdir` / `ideogram_model_subdir` /
    // `boogu_model_subdir` (sc-10845, routing this last bespoke resolver through [`preferred_tier`] instead
    // of its own non-floor-aware q8 default). No shipping Krea model declares a floor, so this is
    // byte-identical to the prior q8 default; the routing keeps the resolver from silently landing below a
    // floor should one ever be declared (a bf16 floor would raise the picker-less default from q8 to the
    // dense bf16, capped by what's installed). Fallback stays krea's q8 → q4 → bf16 (bf16 last — it is the
    // heaviest dense tree), so a partial bundle still surfaces as a load error.
    //
    // NVFP4 (sc-11042): wired here as well as in [`standard_tier_subdir`] because Krea 2 Turbo is epic
    // 11037's named SC#1/SC#2 validation vehicle (the 2026-07-15 Sana → Krea redirect, sc-12110) and its
    // per-tier subdir shape takes an `nvfp4/` dir with no new logic — the same packed
    // `transformer/diffusion_pytorch_model.safetensors` the q4/q8 tiers carry. No `nvfp4/` dir exists on
    // disk yet (sc-11043 owns the convert-at-install loop), so today this always falls through the
    // unchanged q8 → q4 → bf16 chain; wiring it now means the tier resolves the moment the converter
    // lands, rather than needing a second edit here.
    let preferred = preferred_tier(
        bits,
        min_quality_floor(request),
        nvfp4_requested(request) && nvfp4_host,
    );
    // sc-12279: a torn tier no longer short-circuits this chain — `pick_loadable_tier` prefers a tier
    // whose `model_index.json` component tree is fully on disk, so Raw with a torn `q8/` and a
    // complete `bf16/` training-base tier now generates off bf16 instead of dying on the absent
    // `q8/tokenizer/` (issue #850's symptom).
    pick_loadable_tier(
        &[preferred, "q8", "q4", "bf16"],
        &present,
        &tier_components_present,
    )
    .unwrap_or_else(|| root.to_path_buf())
}

/// The private HF repo hosting the Krea 2 INT8-ConvRot DiT single-file checkpoint (sc-9300, epic
/// 9083). Authed download with the SceneWorks HF token, like every private SceneWorks tier.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_CONVROT_REPO: &str = "SceneWorks/krea-2-turbo-int8-convrot";

/// The ConvRot DiT filename inside [`KREA_CONVROT_REPO`] (mirrors the manifest download `files`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_CONVROT_DIT_FILE: &str = "krea2_turbo_int8_convrot.safetensors";

/// Whether this Krea 2 request selected the candle-only INT8-ConvRot tier (sc-9300). The studio's tier
/// picker sends `advanced.convRot: true` for the `int8-convrot` variant (it has no `mlxQuantize` — the
/// online-rotation int8 DiT isn't a bits-based quant). Candle-lane only: the tier is `platforms`-scoped
/// off macOS and the worker only advertises the `int8_convrot` capability on the candle lane, so this
/// never fires on the MLX path even if a stray flag arrives. Confined to `krea_2_turbo`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn wants_krea_convrot(request: &ImageRequest) -> bool {
    sceneworks_core::portable_sources::krea_convrot_requested(
        &request.model,
        "candle",
        std::env::consts::OS,
        request.advanced.get("convRot"),
    )
}

/// Resolve the INT8-ConvRot LoadSpec inputs for a Krea 2 request (sc-9300): the canonical bf16 Krea 2
/// snapshot DIR (the LoadSpec `weights` root — tokenizer / Qwen3-VL TE / Qwen-Image VAE / config + the
/// non-quantized surface) and the downloaded ConvRot DiT single-file (the named
/// `krea_convrot_dit` component, leaving `LoadSpec::text_encoder` exclusively for
/// real encoder substitution).
///
/// `None` when the request didn't select ConvRot, OR either artifact isn't present yet (the bf16
/// `bf16/` subdir of the `krea-2-turbo-mlx` turnkey, or the ConvRot DiT `.safetensors`) — the caller
/// then falls back to the normal dense/packed path rather than half-loading. The bf16 base is fetched
/// on demand by [`ensure_krea_convrot_base_present`] before this resolves. The sm_89 compute-cap floor
/// is enforced ENGINE-side (`ensure_int8_floor` inside `load_components_convrot`) AND surfaced as the
/// worker's `int8_convrot` capability, so an ineligible card never reaches here (the picker hides it).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_krea_convrot(
    request: &ImageRequest,
    settings: &Settings,
) -> Option<(PathBuf, PathBuf)> {
    if !wants_krea_convrot(request) {
        return None;
    }
    // The bf16 base surface: the `bf16/` subdir of the shared `krea-2-turbo-mlx` turnkey (the SAME dir
    // the bf16 tier ships), a candle-readable `transformer/ text_encoder/ vae/ tokenizer/` root. The
    // ConvRot DiT replaces only the transformer at load, but the pipeline still reads the dense TE/VAE/
    // tokenizer/config from here.
    let base_dir = huggingface_snapshot_dir(&settings.data_dir, KREA_MLX_TURNKEY_REPO)
        .map(|root| root.join("bf16"))
        .filter(|dir| {
            dir.join("model_index.json").is_file()
                && dir.join("text_encoder").is_dir()
                && dir.join("vae").is_dir()
        })?;
    // The ConvRot DiT single-file inside the private repo's snapshot.
    let convrot_dit = resolve_krea_convrot_dit(settings)?;
    Some((base_dir, convrot_dit))
}

/// Resolve only the immutable ConvRot DiT single-file. Strict-pose routing uses this narrower probe so
/// it can claim the Krea control lane before the shared bf16 tokenizer/TE/VAE surface is fetched on
/// demand; generation then calls [`ensure_krea_convrot_base_present`] before resolving the full pair.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_krea_convrot_dit(settings: &Settings) -> Option<PathBuf> {
    huggingface_snapshot_dir(&settings.data_dir, KREA_CONVROT_REPO)
        .map(|root| root.join(KREA_CONVROT_DIT_FILE))
        .filter(|file| file.is_file())
}

/// The shared `krea-2-turbo-mlx` turnkey repo (its `bf16/` subdir supplies the ConvRot base surface).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const KREA_MLX_TURNKEY_REPO: &str = "SceneWorks/krea-2-turbo-mlx";
/// Pinned revision for the fixed [`KREA_MLX_TURNKEY_REPO`] (sc-9879, F-077 follow-up). The repo is a
/// hard-coded const (no manifest/payload override reaches this on-demand ConvRot-base fetch), so pulling
/// the mutable `main` branch would let an upstream re-push silently swap the bf16 DiT / Qwen3-VL TE /
/// Qwen-Image VAE we load. Pin the exact commit for defense-in-depth (mirrors the SeedVR2/Real-ESRGAN
/// pins, sc-8879/sc-9682). The native downloader still verifies each file's own hash on download.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const KREA_MLX_TURNKEY_REVISION: &str = "d009674080cc1bccf2b629d834c34bf5eccdb723";

/// On-demand fetch of the canonical bf16 Krea 2 base surface for the INT8-ConvRot tier (sc-9300),
/// the sibling of [`ensure_boogu_tier_present`]. The ConvRot catalog download pulls only the DiT
/// single-file; the bf16 `bf16/` subdir of the `krea-2-turbo-mlx` turnkey (tokenizer / Qwen3-VL TE /
/// Qwen-Image VAE / config) is fetched here when the ConvRot tier is selected and it isn't present —
/// so q4/q8 users are never forced to download the 35 GB bf16 base (it isn't a global co-requisite).
/// No-op when the request isn't a ConvRot job or the bf16 base is already complete; a real download
/// error fails loud (otherwise `resolve_krea_convrot` loads the freshly fetched base).
/// Fails loud on a real download error — fast, before any compute.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn ensure_krea_convrot_base_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<()> {
    if !wants_krea_convrot(request) {
        return Ok(());
    }
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, KREA_MLX_TURNKEY_REPO) else {
        // Turnkey never fetched → `hf` may still pull it below; probe the eventual bf16 subdir path.
        // (If the repo is entirely absent, the fetch below installs the requested bf16 leaf dirs.)
        return fetch_krea_convrot_base(api, settings, job).await;
    };
    let bf16 = root.join("bf16");
    // Present already (dense sharded transformer index + the dense TE/VAE) → no fetch.
    if bf16.join("model_index.json").is_file()
        && bf16.join("text_encoder").is_dir()
        && bf16.join("vae").is_dir()
    {
        return Ok(());
    }
    fetch_krea_convrot_base(api, settings, job).await
}

/// Pull the bf16 base leaf dirs of the `krea-2-turbo-mlx` turnkey into the HF cache (sc-9300). Same
/// leaf-glob shape as the manifest bf16 tier download; scratched marker dir keyed by job id.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn fetch_krea_convrot_base(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let files = vec![
        "bf16/transformer/*".to_owned(),
        "bf16/text_encoder/*".to_owned(),
        "bf16/vae/*".to_owned(),
        "bf16/tokenizer/*".to_owned(),
        "bf16/scheduler/*".to_owned(),
        "bf16/model_index.json".to_owned(),
    ];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        KREA_MLX_TURNKEY_REPO,
        KREA_MLX_TURNKEY_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// Ensure the requested Boogu variant/tier is complete in the shared snapshot before loading.
/// Base, Turbo and Edit share a repository but have separate component directories. Even Q8
/// may need fetching when only a sibling variant has been installed. Reuse the native cache
/// downloader for missing or partial tiers, then check tokenizer and component-weight presence.
#[cfg(native_lane)]
pub(super) async fn ensure_boogu_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<()> {
    let variant = match request.model.as_str() {
        "boogu_image" => "base",
        "boogu_image_turbo" => "turbo",
        "boogu_image_edit" => "edit",
        _ => return Ok(()),
    };
    let bits = request.quant_bits();
    let tier = boogu_tier_subdir(variant, bits);
    let Some(model) = mlx_model(&request.model) else {
        return Ok(());
    };
    let repo = model_repo(request, &model);
    let root = if repo == model.default_repo() {
        crate::model_jobs::huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            &repo,
            BOOGU_MLX_TURNKEY_REVISION,
        )
    } else {
        huggingface_snapshot_dir(&settings.data_dir, &repo)
    };
    let repo_is_installed = root.is_some()
        || (repo == model.default_repo()
            && huggingface_snapshot_dir(&settings.data_dir, &repo).is_some());
    if !repo_is_installed {
        // Turnkey not downloaded at all → leave it to the load path's "weights not found" error.
        return Ok(());
    }
    let tier_dir = root.as_ref().map(|root| root.join(&tier));
    // Base, Turbo and Edit share a repository, but installation of a sibling is not proof
    // that this variant exists. Repair missing default tiers and incomplete non-default tiers.
    if tier_dir
        .as_ref()
        .is_some_and(|dir| sceneworks_core::mlx_tier_completeness::boogu_tier_complete(dir))
    {
        return Ok(());
    }
    // The tier subfolder nests transformer/mllm/vae (leaf-dir globs, like the catalog Q8 entry).
    let files = vec![
        format!("{tier}/transformer/*"),
        format!("{tier}/mllm/*"),
        format!("{tier}/vae/*"),
    ];
    let revision = turnkey_tier_revision(&repo, model.default_repo(), BOOGU_MLX_TURNKEY_REVISION);
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, &repo, revision, &files).await?;
    let installed = if repo == model.default_repo() {
        crate::model_jobs::huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            &repo,
            BOOGU_MLX_TURNKEY_REVISION,
        )
    } else {
        huggingface_snapshot_dir(&settings.data_dir, &repo)
    };
    if !installed.as_ref().is_some_and(|root| {
        sceneworks_core::mlx_tier_completeness::boogu_tier_complete(&root.join(&tier))
    }) {
        return Err(WorkerError::InvalidPayload(format!(
            "{}: downloaded Boogu tier {tier} is incomplete; expected transformer, mllm tokenizer/weights, and VAE",
            request.model,
        )));
    }
    Ok(())
}

/// On-demand fetch of Ideogram 4's non-default `q8/` tier (sc-9607, epic 9083). The catalog download
/// pulls only the default `q4/` subdir (`files: ["q4/*"]`), so a job that opts into Q8
/// ([`ideogram_tier_subdir`]: `> 4` → `q8/`) needs the `q8/` subdir pulled into the HF cache before
/// [`ideogram_model_subdir`] can resolve it. No-op when the default q4 is requested, the model isn't
/// Ideogram, the turnkey snapshot isn't downloaded yet (`ideogram_model_subdir` then falls back to q4 /
/// surfaces the load error), or `q8/` is already complete. The `q8/*` glob is recursive (matches
/// `q8/transformer/…`, and for `ideogram_4_turbo` the bundled `q8/turbo_lora.safetensors`), mirroring
/// the catalog q4 entry and [`crate::video_jobs::ensure_ltx_q8_present`]. bf16 is NOT fetched here — it
/// lives in the separate `SceneWorks/ideogram-4` catalog repo the user opts into on the Models page
/// (and is macOS-only). This is the on-demand `q8/` download the [`ideogram_model_subdir`] docstring
/// flagged as a follow-up; it runs on BOTH the MLX (`generate_stream`) and candle
/// (`generate_candle_stream`) lanes, so off-Mac gets the same q4/q8 picker as macOS. Fails loud on a
/// real download error; a `q8/` tier that isn't published yet stays absent so the request falls back to q4.
#[cfg(native_lane)]
pub(super) async fn ensure_ideogram_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<()> {
    if request.model != "ideogram_4" && request.model != "ideogram_4_turbo" {
        return Ok(());
    }
    let bits = request.quant_bits();
    let Some(tier) = ideogram_tier_subdir(bits) else {
        // Default q4 ships in the catalog download — nothing to fetch.
        return Ok(());
    };
    let Some(model) = mlx_model(&request.model) else {
        return Ok(());
    };
    let repo = model_repo(request, &model);
    let root = if repo == model.default_repo() {
        crate::model_jobs::huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            &repo,
            IDEOGRAM_MLX_TURNKEY_REVISION,
        )
    } else {
        huggingface_snapshot_dir(&settings.data_dir, &repo)
    };
    let repo_is_installed = root.is_some()
        || (repo == model.default_repo()
            && huggingface_snapshot_dir(&settings.data_dir, &repo).is_some());
    if !repo_is_installed {
        // Turnkey not downloaded at all → leave it to the load path's "weights not found" error.
        return Ok(());
    }
    // Present already (the packed single-file transformer) → no fetch.
    if root.as_ref().is_some_and(|root| {
        root.join(tier)
            .join("transformer/model.safetensors")
            .is_file()
    }) {
        return Ok(());
    }
    let files = vec![format!("{tier}/*")];
    let revision =
        turnkey_tier_revision(&repo, model.default_repo(), IDEOGRAM_MLX_TURNKEY_REVISION);
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, &repo, revision, &files)
        .await
        .map(|_| ())
}

/// XLabs FLUX IP-Adapter repos (epic 3621). The torch `flux_dev` path already declares +
/// downloads these (the `ipAdapter` block in `image_adapters`); the MLX path reuses the same
/// HF-cache snapshots — there is no new weight to ship.
#[cfg(target_os = "macos")]
const FLUX_IP_ADAPTER_REPO: &str = "XLabs-AI/flux-ip-adapter";
#[cfg(target_os = "macos")]
const FLUX_IP_IMAGE_ENCODER_REPO: &str = "openai/clip-vit-large-patch14";
/// IP-Adapter scale when the request omits `ipAdapterScale` (XLabs resemblance tier 0.7, matching
/// the torch `FluxDiffusersAdapter`).
#[cfg(target_os = "macos")]
pub(super) const FLUX_IP_SCALE: f32 = 0.7;
/// `trueCfgScale` default for the FLUX.1-dev IP-Adapter path (real CFG; torch default ~4.0).
#[cfg(target_os = "macos")]
pub(super) const FLUX_IP_TRUE_CFG: f32 = 4.0;

/// The FLUX.1 engine families that carry the XLabs IP-Adapter (both variants — the Rust engine has
/// no diffusers `load_ip_adapter` schnell limitation).
#[cfg(target_os = "macos")]
pub(super) fn is_flux_model(model: &str) -> bool {
    matches!(model, "flux_schnell" | "flux_dev")
}

/// The SenseNova-U1 SceneWorks ids (base + 8-step distill), both served by the unified
/// `mlx-gen-sensenova` engine (sc-3900).
///
/// sc-13817 widened the gate from macOS-only so the off-Mac candle lane could force a dense tier.
/// sc-14249 retired that force (`candle-gen-sensenova` packed-detects all three tiers now), which
/// removed the only candle-side caller — so the gate is back to **macOS-only**. Its lone remaining
/// user is `image_jobs/sensenova.rs` (the MLX it2i / VQA / interleave routing), whose `include!` is
/// itself `cfg(target_os = "macos")`; leaving the wider cfg makes this dead code on the candle build
/// (`-D warnings` → `function is never used`).
///
/// The id list itself lives in [`sceneworks_core::mlx_tier_completeness::SENSENOVA_MODELS`], beside the
/// per-tier completeness predicates the tier resolver and rust-api's catalog both gate on — two
/// independently-maintained copies of a model-id list is exactly the drift that leaves one gate silently
/// un-wired (sc-14432).
#[cfg(target_os = "macos")]
pub(super) fn is_sensenova_model(model: &str) -> bool {
    sceneworks_core::mlx_tier_completeness::SENSENOVA_MODELS.contains(&model)
}

/// Stage the engine's IP-Adapter dir contract from the two cached HF snapshots:
/// `<staged>/ip_adapter.safetensors` (XLabs) + `<staged>/image_encoder/model.safetensors`
/// (openai CLIP-ViT-L). Errors loudly if either snapshot is missing — mirrors the SDXL IP path
/// (`resolve_ip_adapter_dir`); the repos reach the cache via the model-download flow / the torch
/// `flux_dev` path, not a new provisioning step.
#[cfg(target_os = "macos")]
pub(super) fn resolve_flux_ip_adapter_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    let missing = || {
        WorkerError::InvalidPayload(format!(
            "FLUX IP-Adapter weights not found (download {FLUX_IP_ADAPTER_REPO} + {FLUX_IP_IMAGE_ENCODER_REPO})."
        ))
    };
    let adapter_snap =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, FLUX_IP_ADAPTER_REPO)
            .ok_or_else(missing)?;
    let clip_snap =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, FLUX_IP_IMAGE_ENCODER_REPO)
            .ok_or_else(missing)?;
    let ip_file = adapter_snap.join("ip_adapter.safetensors");
    let clip_file = clip_snap.join("model.safetensors");
    if !ip_file.exists() || !clip_file.exists() {
        return Err(missing());
    }
    let staged = settings.data_dir.join("staged").join("flux-ip-adapter");
    let encoder_dir = staged.join("image_encoder");
    std::fs::create_dir_all(&encoder_dir)
        .map_err(|e| WorkerError::InvalidPayload(format!("stage flux ip-adapter dir: {e}")))?;
    // Re-link each call: the HF-cache targets are immutable, so a stable staged dir is reusable.
    let link = |src: &Path, dst: PathBuf| -> WorkerResult<()> {
        let _ = std::fs::remove_file(&dst);
        std::os::unix::fs::symlink(src, &dst)
            .map_err(|e| WorkerError::InvalidPayload(format!("stage flux ip-adapter link: {e}")))
    };
    link(&ip_file, staged.join("ip_adapter.safetensors"))?;
    link(&clip_file, encoder_dir.join("model.safetensors"))?;
    Ok(staged)
}

#[cfg(native_lane)]
#[cfg(all(test, target_os = "macos"))]
#[path = "tests/resolved_artifact_provenance_tests.rs"]
pub(crate) mod resolved_artifact_provenance_tests;

#[cfg(native_lane)]
#[cfg(test)]
#[path = "tests/boogu_tier_tests.rs"]
mod boogu_tier_tests;

#[cfg(native_lane)]
#[cfg(test)]
#[path = "tests/standard_tier_tests.rs"]
mod standard_tier_tests;
