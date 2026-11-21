//! video_admission scail2 owner.
use super::*;

/// Worker seal for both SCAIL-2 carriers (`animate_character` and `replace_person`), which share
/// one physical assembly built by `video_jobs::scail2`: one or more ordered `Reference` stills and their painted
/// `Mask`s, and a `ControlClip` of driving frames plus per-frame color masks. The engine TASK is a
/// separate axis because animation and replacement are different working sets over the same shapes.
pub(crate) fn scail2_carrier_receipt(
    lane: VideoLane,
    mode: &str,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    gen_core::generator::Scail2AnimationConditioningRef::catalog_receipt(
        backend_axis(lane),
        mode,
        width,
        height,
        conditioning,
    )
    .map_err(|error| error.to_string())
}
