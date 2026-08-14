use super::advanced;
use super::zimage_edit_candle::resolve_zimage_edit_candle_base;
use super::{pose_entries, ImageRequest, Settings};

// Candle (Windows/CUDA) Z-Image identity-init route for Image Studio "With Character" (sc-8409, epic
// 4406) — the off-Mac sibling of the macOS MLX generic lane's Z-Image identity img2img path
// (`resolve_zimage_identity_init` in zimage.rs, sc-3146). A `character_image` job with a chosen
// `referenceAssetId` (and `advanced.referenceStrength > 0`) seeds the Z-Image-Turbo denoise FROM the
// reference latents — carrying the character's identity into the variation — instead of falling through
// to plain txt2img (which drops the reference entirely, the pre-existing gap this story closes).
//
// The availability gate remains distinct because the request shape is `character_image` with a
// `referenceAssetId`; generation itself now uses the registered Z-Image Turbo provider so Reference
// conditioning and user adapters share one truthful load path.
//
// **Parity with macOS.** The engage condition mirrors the macOS `zimage_identity_strength` gate EXACTLY
// (`advanced.referenceStrength > 0` AND a non-empty `referenceAssetId`), so candle runs identity img2img
// precisely when the MLX generic lane does — a With-Character job WITHOUT a positive `referenceStrength`
// stays plain txt2img on both backends.
//
// **Face-likeness scoring (sc-4411 seam).** Once the route exists, each finished image is scored against
// the chosen reference face through the SHARED generator-agnostic seam (`build_face_likeness_scorer` +
// `score_generated_image`, source resolved by `resolve_character_image_likeness_source`), exactly as the
// macOS generic lane and the other identity lanes do — source embedded ONCE, reused across the N images,
// non-fatal, the hot-path pixel clone gated behind `scorer.is_some()`, non-frontal → honest N/A.
//
// This child module retains only the request gate and base-resolution helper shared with the generic
// route; face-likeness scoring stays in the common generation stream.

/// Model ids the candle Z-Image identity-init route accepts: only `z_image_turbo` (the With-Character
/// target — the candle z-image engine is the distilled Turbo). The dedicated `z_image_edit` id is an
/// edit-mode id (handled by `zimage_edit_candle`), not a character target.
fn is_zimage_identity_candle_model(model: &str) -> bool {
    model == "z_image_turbo"
}

/// The clamped identity img2img-init strength for a candle Z-Image With-Character job, or `None` when the
/// identity init does NOT engage. `Some(strength)` iff `advanced.referenceStrength > 0` AND a non-empty
/// `referenceAssetId` is present — the EXACT engage gate of the macOS `zimage_identity_strength`
/// (zimage.rs, sc-3146), so candle runs identity img2img precisely when the MLX generic lane does.
///
/// `strength` is the user value clamped to `[0.05, 1.0]` and carries the mflux `image_strength`
/// convention **verbatim** (no numeric inversion): higher strength → later denoise start
/// (`init_time_step`) → output stays closer to the reference. Pure (request only) so the parity-sensitive
/// gate + clamp are unit-testable without asset I/O. (Deliberately duplicates the macOS helper, which
/// lives in the macOS-only `zimage.rs` include — the same per-lane-helper pattern the candle siblings use,
/// kept in lockstep by the shared parity comment.)
pub(super) fn zimage_identity_candle_strength(request: &ImageRequest) -> Option<f32> {
    let strength = request
        .advanced
        .get("referenceStrength")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .filter(|strength| *strength > 0.0)?;
    let has_asset = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .is_some_and(|id| !id.is_empty());
    has_asset.then(|| (strength as f32).clamp(0.05, 1.0))
}

/// True when this is a candle-eligible Z-Image identity-init job: a `z_image_turbo` Image Studio
/// "With Character" generation (`mode == "character_image"`) with a chosen `referenceAssetId` and a
/// positive `referenceStrength`, that is NOT an angle set / pose-library set (those are already routed to
/// their own lanes — the candle InstantID angle/pose paths and the Z-Image strict-control lane — and
/// scored there), and whose Turbo base resolves locally. Mirrors `jobs_store::zimage_identity_candle_\
/// eligible` (minus the local weight-resolve check) so the worker and router agree.
pub(super) fn zimage_identity_candle_available(
    request: &ImageRequest,
    settings: &Settings,
) -> bool {
    is_zimage_identity_candle_model(&request.model)
        && request.mode == "character_image"
        && zimage_identity_candle_strength(request).is_some()
        // Angle / pose sets are `character_image` too, but route to (and score on) their own lanes —
        // exclude both so this plain With-Character lane never steals them (it sits BEFORE the Z-Image
        // strict-control lane in the dispatch). Mirrors `resolve_character_image_likeness_source`.
        && pose_entries(request).is_empty()
        && !advanced::flag(&request.advanced, "angleSet")
        && matches!(
            resolve_zimage_edit_candle_base(request, settings),
            Ok(Some(_))
        )
}
