//! video_admission wan vace owner.
use super::*;

// sc-20799. Outside Bernini and LTX the video carriers were keyed by SHAPE ALONE — a bare `"other"`
// plus `conditioning.len()` — so a one-image `MultiReference` and an eight-image one, or two
// different driving clips and masks, shared a single admission identity and could borrow each
// other's admitted peak. These seal the carriers the way Bernini's do: a content-INDEPENDENT
// evidence axis (order, counts, native shapes, and the exact knobs that change the working set)
// plus a request-only byte seal appended after `+`, which `video_curve_overlay` strips before curve
// lookup. Sealed BEFORE any fitted curve exists for these cells, so no curve can be fitted against
// a borrowable identity.
pub(crate) const WAN_VACE_REPLACE_RECEIPT_DOMAIN: &str = "wan-vace-replace-person-carrier-v1";

pub(crate) const WAN_VACE_REPLACE_SEAL_DOMAIN: &str = "wan-vace-replace-person-request-seal-v1";

/// Worker seal for the single-expert Wan-VACE `replace_person` carrier
/// (`video_jobs::vace::build_vace_conditioning`): one `ControlClip` of driving frames plus its
/// per-frame mask, then one `Reference` per character image. The frame/mask cardinality, the
/// per-frame geometry, the masking strength and the replacement mode all move the working set, and
/// the reference ORDER is load order, so all of them are evidence axes.
pub(crate) fn wan_vace_replace_person_receipt(
    lane: VideoLane,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    let [Conditioning::ControlClip {
        frames,
        mask,
        masking_strength,
        start_frame,
        mode,
    }, references @ ..] = conditioning
    else {
        return Err(
            "Wan-VACE replace_person admission requires a leading ControlClip carrier".to_owned(),
        );
    };
    if frames.len() != mask.len() || frames.is_empty() {
        return Err(format!(
            "Wan-VACE replace_person admission requires equal non-empty frame/mask counts, got {} and {}",
            frames.len(),
            mask.len()
        ));
    }
    if *start_frame != 0 {
        return Err(format!(
            "Wan-VACE replace_person admission requires a frame-0 control clip, got {start_frame}"
        ));
    }
    let mut reference_images: Vec<gen_core::Image> = Vec::with_capacity(references.len());
    for entry in references {
        let Conditioning::Reference { image, strength } = entry else {
            return Err(
                "Wan-VACE replace_person admission accepts only Reference entries after its \
                 ControlClip"
                    .to_owned(),
            );
        };
        if strength.is_some() {
            return Err(
                "Wan-VACE replace_person references carry no strength; a populated one is a \
                 different workload"
                    .to_owned(),
            );
        }
        reference_images.push(image.clone());
    }
    let mut seal = sha2::Sha256::new();
    seal.update(WAN_VACE_REPLACE_SEAL_DOMAIN.as_bytes());
    seal.update(masking_strength.to_bits().to_le_bytes());
    let frames_axis = seal_images("control-frames", frames, &mut seal);
    let mask_axis = seal_images("control-mask", mask, &mut seal);
    let reference_axis = seal_images("references", &reference_images, &mut seal);
    Ok(format!(
        "{WAN_VACE_REPLACE_RECEIPT_DOMAIN}:backend-{}:composite-{width}x{height}:{frames_axis}:{mask_axis}:strength-{:08x}:mode-{mode:?}:frame-0:{reference_axis}+{WAN_VACE_REPLACE_SEAL_DOMAIN}-{:x}",
        backend_axis(lane),
        masking_strength.to_bits(),
        seal.finalize()
    ))
}
