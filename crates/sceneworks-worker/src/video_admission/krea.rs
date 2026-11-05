//! video_admission krea owner.
use super::*;

pub(crate) const KREA_V2V_RECEIPT_DOMAIN: &str = "krea-realtime-v2v-clip-v1";

pub(crate) const KREA_V2V_SEAL_DOMAIN: &str = "krea-realtime-v2v-request-seal-v1";

/// Worker seal for the Krea Realtime video-to-video carrier
/// (`video_jobs::krea_realtime::krea_realtime_conditioning`): one `VideoClip` whose frame count and
/// per-frame geometry set the autoregressive init working set, plus the v2v strength — the one knob
/// that distinguishes v2v from the i2v cache warm, which drops strength entirely.
pub(crate) fn krea_v2v_clip_receipt(
    lane: VideoLane,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    let [Conditioning::VideoClip {
        frames,
        frame_idx,
        strength,
    }] = conditioning
    else {
        return Err(
            "Krea Realtime video_to_video admission requires exactly one VideoClip".to_owned(),
        );
    };
    if *frame_idx != 0 || frames.is_empty() {
        return Err(format!(
            "Krea Realtime v2v admission requires a non-empty frame-0 clip, got {} frames at {frame_idx}",
            frames.len()
        ));
    }
    let mut seal = sha2::Sha256::new();
    seal.update(KREA_V2V_SEAL_DOMAIN.as_bytes());
    seal.update(strength.to_bits().to_le_bytes());
    let frames_axis = seal_images("clip", frames, &mut seal);
    Ok(format!(
        "{KREA_V2V_RECEIPT_DOMAIN}:backend-{}:composite-{width}x{height}:{frames_axis}:frame-0:strength-{:08x}+{KREA_V2V_SEAL_DOMAIN}-{:x}",
        backend_axis(lane),
        strength.to_bits(),
        seal.finalize()
    ))
}
