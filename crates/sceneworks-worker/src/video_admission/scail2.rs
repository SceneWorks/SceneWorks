//! video_admission scail2 owner.
use super::*;

pub(crate) const SCAIL2_CARRIER_RECEIPT_DOMAIN: &str = "scail2-carrier-v1";

pub(crate) const SCAIL2_CARRIER_SEAL_DOMAIN: &str = "scail2-carrier-request-seal-v1";

/// Worker seal for both SCAIL-2 carriers (`animate_character` and `replace_person`), which share
/// one physical assembly built by `video_jobs::scail2`: a `Reference` identity still, its painted
/// `Mask`, and a `ControlClip` of driving frames plus per-frame color masks. The engine TASK is a
/// separate axis because animation and replacement are different working sets over the same shapes.
pub(crate) fn scail2_carrier_receipt(
    lane: VideoLane,
    mode: &str,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    let task = match mode {
        "animate_character" => "animation",
        "replace_person" => "replacement",
        other => return Err(format!("SCAIL-2 admission has no carrier for mode {other}")),
    };
    let [Conditioning::Reference {
        image: reference,
        strength: reference_strength,
    }, Conditioning::Mask {
        image: reference_mask,
    }, Conditioning::ControlClip {
        frames,
        mask,
        masking_strength,
        start_frame,
        mode: replacement_mode,
    }] = conditioning
    else {
        return Err(
            "SCAIL-2 admission requires exactly one Reference, one Mask, and one ControlClip, in \
             that order"
                .to_owned(),
        );
    };
    if reference_strength.is_some() {
        return Err(
            "the SCAIL-2 identity reference carries no strength; a populated one is a different \
             workload"
                .to_owned(),
        );
    }
    if frames.len() != mask.len() || frames.is_empty() {
        return Err(format!(
            "SCAIL-2 admission requires equal non-empty driving frame/mask counts, got {} and {}",
            frames.len(),
            mask.len()
        ));
    }
    if *start_frame != 0 {
        return Err(format!(
            "SCAIL-2 admission requires a frame-0 driving clip, got {start_frame}"
        ));
    }
    let mut seal = sha2::Sha256::new();
    seal.update(SCAIL2_CARRIER_SEAL_DOMAIN.as_bytes());
    seal.update(task.as_bytes());
    seal.update(masking_strength.to_bits().to_le_bytes());
    let reference_axis = seal_images("reference", std::slice::from_ref(reference), &mut seal);
    let reference_mask_axis = seal_images(
        "reference-mask",
        std::slice::from_ref(reference_mask),
        &mut seal,
    );
    let frames_axis = seal_images("driving-frames", frames, &mut seal);
    let mask_axis = seal_images("driving-mask", mask, &mut seal);
    Ok(format!(
        "{SCAIL2_CARRIER_RECEIPT_DOMAIN}:backend-{}:task-{task}:composite-{width}x{height}:{reference_axis}:{reference_mask_axis}:{frames_axis}:{mask_axis}:strength-{:08x}:mode-{replacement_mode:?}:frame-0+{SCAIL2_CARRIER_SEAL_DOMAIN}-{:x}",
        backend_axis(lane),
        masking_strength.to_bits(),
        seal.finalize()
    ))
}
