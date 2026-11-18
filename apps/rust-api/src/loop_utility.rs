use crate::{ApiError, AppState};
use sceneworks_core::loop_utility;

/// Validate a user-selected tier before enqueue. The worker repeats the check on its own route
/// against the frozen catalog and verifies the pinned files before loading.
pub(crate) async fn ensure_tier_installed(state: &AppState, tier: &str) -> Result<(), ApiError> {
    let mut entry =
        crate::models::resolve_model_manifest_entry(state, loop_utility::MODEL_ID).await?;
    let os = state.settings.host_os.as_str();
    let unsupported = || {
        ApiError::model_artifact_conflict(
            format!(
                "The {tier} tier of {} is unavailable on this platform.",
                loop_utility::MODEL_NAME
            ),
            "model_artifact_incomplete",
        )
    };
    loop_utility::download(&entry, os, tier).ok_or_else(unsupported)?;
    crate::models::retain_downloads_for_os(&mut entry, os);
    if !crate::models::installed_variant_keys(&entry, &state.settings.data_dir)
        .iter()
        .any(|installed| installed == tier)
    {
        return Err(ApiError::model_artifact_conflict(
            format!(
                "The {tier} tier of {} is not installed. Download this tier in the Model Manager.",
                loop_utility::MODEL_NAME
            ),
            "model_artifact_incomplete",
        ));
    }
    Ok(())
}
