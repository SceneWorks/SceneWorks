use super::{huggingface_snapshot_dir, resolve_app_managed_model_dir, standard_tier_subdir};
use super::{non_empty, ImageRequest, PathBuf, Settings, Value, WorkerResult};

// Candle (Windows/CUDA) Z-Image img2img / edit route (sc-6595, epic 5480) — pixel-conditioned editing
// on Z-Image-Turbo off-Mac through the registered generator's `Conditioning::Reference` path. Both
// `z_image_edit` and `z_image_turbo` (mode `edit_image`) reach this one adapter-capable load path.
//
// **Candle-only gate.** macOS uses its own request router. This module now only resolves the off-Mac
// availability/base alias before dispatch joins the generic registered-generator stream.

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
