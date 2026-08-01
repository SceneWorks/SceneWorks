use super::{huggingface_snapshot_dir, resolve_app_managed_model_dir, standard_tier_subdir};
use super::{
    non_empty, resolve_advanced_or_manifest_u32, ImageRequest, PathBuf, Settings, Value,
    WorkerResult,
};

// Candle (Windows/CUDA) Z-Image img2img / edit route (sc-6595, epic 5480) — pixel-conditioned editing
// on Z-Image-Turbo off-Mac via `runtime_cuda::providers::z_image::ZImageEdit`. The candle sibling of the MLX z-image
// img2img path (the registered `z_image_turbo` generator's `Conditioning::Reference` route, driven by
// `resolve_zimage_edit_init` in zimage.rs). Both `z_image_edit` and `z_image_turbo` (mode `edit_image`)
// reach this one lane — two ids, one engine (the Turbo weights with a source-latent init).
//
// **Candle-only.** macOS keeps the MLX `z_image_turbo` registry generator (img2img via the engine's
// `Reference` conditioning); the candle `z_image_turbo` descriptor is txt2img-only, so the candle
// `ZImageEdit` is a bespoke provider — this whole file is gated to the Windows/CUDA candle build (the
// the module declaration in image_jobs.rs carries the cfg). It is a child module of the `image_jobs` module, so it
// shares that module's imports (`ImageRequest`/`Settings`/`WorkerResult`/`advanced`/`load_reference_image`/
// `fit_engine_image`/`huggingface_snapshot_dir`/`resolve_app_managed_model_dir`/`resolve_seed`/
// `start_gen_stream`/`drive_gen_items`/`consume_gen_events`/`non_empty`/`gen_core`/… all in scope).

/// Denoise-steps default — the distilled 4-step Turbo schedule (the txt2img / MLX z-image default).
const ZIMAGE_EDIT_CANDLE_DEFAULT_STEPS: u32 = 4;
/// The Z-Image base diffusers repo when the manifest omits `repo`.
pub(super) const ZIMAGE_EDIT_CANDLE_DEFAULT_REPO: &str = "Tongyi-MAI/Z-Image-Turbo";

/// Model ids the candle Z-Image edit route accepts: the txt2img `z_image_turbo` (in `edit_image` mode)
/// and the dedicated `z_image_edit` id — both drive the Turbo weights' img2img path.
fn is_zimage_edit_candle_model(model: &str) -> bool {
    matches!(model, "z_image_turbo" | "z_image_edit")
}

/// Resolve the Z-Image base (diffusers) snapshot: an explicit `modelPath` (advanced or manifest) → the
/// HF cache snapshot for the manifest `repo` (default `Tongyi-MAI/Z-Image-Turbo`). `None` ⇒ not present
/// locally (the candle lane refuses the job; no fallback is attempted). Mirrors
/// `resolve_zimage_control_base`.
pub(super) fn resolve_zimage_edit_candle_base(
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
        return resolve_app_managed_model_dir(settings, &path, "Z-Image edit modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            crate::engines::default_repo_for(&request.model)
                .unwrap_or(ZIMAGE_EDIT_CANDLE_DEFAULT_REPO)
        });
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo)
        .map(|root| standard_tier_subdir(&root, request)))
}

/// True when this is a candle-eligible Z-Image edit job: a z-image-family `edit_image` job with a source
/// image whose base resolves locally. Mirrors `jobs_store::zimage_edit_candle_eligible` so the worker and
/// router agree.
pub(super) fn zimage_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_zimage_edit_candle_model(&request.model)
        && request.mode == "edit_image"
        && non_empty(&request.source_asset_id)
        && matches!(
            resolve_zimage_edit_candle_base(request, settings),
            Ok(Some(_))
        )
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → default (4, distilled).
pub(super) fn zimage_edit_candle_steps(request: &ImageRequest) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", ZIMAGE_EDIT_CANDLE_DEFAULT_STEPS, 1..=50)
}
