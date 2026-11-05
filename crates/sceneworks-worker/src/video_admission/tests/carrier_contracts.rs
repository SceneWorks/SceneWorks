//! Pure receipt contracts: byte buffers and SHA256 only; no provider or tensor execution.
use super::*;

fn image(width: u32, height: u32, pixel: u8) -> gen_core::Image {
    gen_core::Image {
        width,
        height,
        pixels: vec![pixel; width as usize * height as usize * 3],
    }
}
fn clip(pixel: u8) -> Conditioning {
    Conditioning::ControlClip {
        frames: vec![image(2, 3, pixel), image(3, 2, pixel + 1)],
        mask: vec![image(2, 3, 0), image(3, 2, 1)],
        masking_strength: 1.0,
        start_frame: 0,
        mode: gen_core::ReplacementMode::default(),
    }
}
fn scail() -> Vec<Conditioning> {
    vec![
        Conditioning::Reference {
            image: image(4, 2, 5),
            strength: None,
        },
        Conditioning::Mask {
            image: image(4, 2, 0),
        },
        clip(2),
    ]
}
fn vace() -> Vec<Conditioning> {
    vec![
        clip(2),
        Conditioning::Reference {
            image: image(4, 2, 5),
            strength: None,
        },
        Conditioning::Reference {
            image: image(2, 4, 6),
            strength: None,
        },
    ]
}
fn krea() -> Vec<Conditioning> {
    vec![Conditioning::VideoClip {
        frames: vec![image(2, 3, 2), image(3, 2, 3)],
        frame_idx: 0,
        strength: 0.75,
    }]
}
fn axis(receipt: &str) -> &str {
    receipt.split_once('+').expect("request-only byte seal").0
}

#[test]
fn scail2_receipt_separates_task_backend_shapes_and_request_bytes() {
    for lane in [VideoLane::Mlx, VideoLane::Candle] {
        let baseline = scail2_carrier_receipt(lane, "animate_character", 16, 24, &scail()).unwrap();
        assert!(baseline.starts_with("scail2-carrier-v1:backend-"));
        let replacement = scail2_carrier_receipt(lane, "replace_person", 16, 24, &scail()).unwrap();
        assert_ne!(axis(&baseline), axis(&replacement));
        let mut changed = scail();
        let Conditioning::ControlClip { frames, .. } = &mut changed[2] else {
            unreachable!()
        };
        frames[0].pixels[0] ^= 1;
        let changed_bytes =
            scail2_carrier_receipt(lane, "animate_character", 16, 24, &changed).unwrap();
        assert_eq!(
            axis(&baseline),
            axis(&changed_bytes),
            "capture evidence is content independent"
        );
        assert_ne!(
            baseline, changed_bytes,
            "request seals bind actual frame bytes"
        );
        let Conditioning::Reference { image, .. } = &mut changed[0] else {
            unreachable!()
        };
        *image = self::image(2, 4, 5);
        assert_ne!(
            axis(&baseline),
            axis(&scail2_carrier_receipt(lane, "animate_character", 16, 24, &changed).unwrap())
        );
        let other = if lane == VideoLane::Mlx {
            VideoLane::Candle
        } else {
            VideoLane::Mlx
        };
        assert_ne!(
            axis(&baseline),
            axis(&scail2_carrier_receipt(other, "animate_character", 16, 24, &scail()).unwrap())
        );
    }
}
#[test]
fn scail2_receipt_refuses_crossed_carriers_and_preserves_current_knob_axes() {
    let receipt = |conditioning: &[Conditioning]| {
        scail2_carrier_receipt(VideoLane::Mlx, "animate_character", 16, 24, conditioning)
    };
    let baseline = receipt(&scail()).unwrap();
    assert!(scail2_carrier_receipt(VideoLane::Mlx, "text_to_video", 16, 24, &scail()).is_err());
    for mutation in 0..5 {
        let mut changed = scail();
        match mutation {
            0 => changed.swap(0, 1),
            1 => {
                if let Conditioning::Reference { strength, .. } = &mut changed[0] {
                    *strength = Some(1.0);
                }
            }
            2 => {
                if let Conditioning::ControlClip { mask, .. } = &mut changed[2] {
                    mask.pop();
                }
            }
            3 => {
                if let Conditioning::ControlClip { frames, mask, .. } = &mut changed[2] {
                    frames.clear();
                    mask.clear();
                }
            }
            _ => {
                if let Conditioning::ControlClip { start_frame, .. } = &mut changed[2] {
                    *start_frame = 1;
                }
            }
        }
        assert!(receipt(&changed).is_err(), "mutation{mutation}");
    }
    let mut changed = scail();
    if let Conditioning::ControlClip {
        masking_strength, ..
    } = &mut changed[2]
    {
        *masking_strength = 0.5;
    }
    let changed = receipt(&changed).unwrap();
    assert_ne!(
        axis(&baseline),
        axis(&changed),
        "existing masking strength is an axis even when the engine treats it as neutral"
    );
}
#[test]
fn wan_vace_receipt_binds_order_and_masks_without_borrowing_another_request_seal() {
    let receipt = |lane, conditioning: &[Conditioning]| {
        wan_vace_replace_person_receipt(lane, 16, 24, conditioning)
    };
    let baseline = receipt(VideoLane::Mlx, &vace()).unwrap();
    assert!(baseline.starts_with("wan-vace-replace-person-carrier-v1:"));
    let mut changed = vace();
    if let Conditioning::ControlClip { mask, .. } = &mut changed[0] {
        mask[0].pixels[0] ^= 1;
    }
    let bytes = receipt(VideoLane::Mlx, &changed).unwrap();
    assert_eq!(axis(&baseline), axis(&bytes));
    assert_ne!(baseline, bytes);
    changed.swap(1, 2);
    assert_ne!(
        axis(&baseline),
        axis(&receipt(VideoLane::Mlx, &changed).unwrap()),
        "reference order is load order"
    );
    assert_ne!(
        axis(&baseline),
        axis(&receipt(VideoLane::Candle, &vace()).unwrap())
    );
    for mutation in 0..5 {
        let mut changed = vace();
        match mutation {
            0 => changed.swap(0, 1),
            1 => {
                if let Conditioning::Reference { strength, .. } = &mut changed[1] {
                    *strength = Some(1.0);
                }
            }
            2 => {
                if let Conditioning::ControlClip { mask, .. } = &mut changed[0] {
                    mask.pop();
                }
            }
            3 => {
                if let Conditioning::ControlClip { start_frame, .. } = &mut changed[0] {
                    *start_frame = 1;
                }
            }
            _ => changed.push(Conditioning::Mask {
                image: image(2, 2, 0),
            }),
        }
        assert!(
            receipt(VideoLane::Mlx, &changed).is_err(),
            "mutation{mutation}"
        );
    }
}
#[test]
fn krea_v2v_receipt_binds_strength_geometry_backend_and_exact_frame_zero_carrier() {
    let receipt =
        |lane, conditioning: &[Conditioning]| krea_v2v_clip_receipt(lane, 16, 24, conditioning);
    let baseline = receipt(VideoLane::Mlx, &krea()).unwrap();
    assert!(baseline.starts_with("krea-realtime-v2v-clip-v1:"));
    let mut changed = krea();
    if let Conditioning::VideoClip { frames, .. } = &mut changed[0] {
        frames[0].pixels[0] ^= 1;
    }
    let bytes = receipt(VideoLane::Mlx, &changed).unwrap();
    assert_eq!(axis(&baseline), axis(&bytes));
    assert_ne!(baseline, bytes);
    if let Conditioning::VideoClip { strength, .. } = &mut changed[0] {
        *strength = 0.5;
    }
    assert_ne!(
        axis(&baseline),
        axis(&receipt(VideoLane::Mlx, &changed).unwrap())
    );
    assert_ne!(
        axis(&baseline),
        axis(&receipt(VideoLane::Candle, &krea()).unwrap())
    );
    let mut changed = krea();
    if let Conditioning::VideoClip { frames, .. } = &mut changed[0] {
        frames.swap(0, 1);
    }
    assert_ne!(
        axis(&baseline),
        axis(&receipt(VideoLane::Mlx, &changed).unwrap())
    );
    for mutation in 0..3 {
        let mut changed = krea();
        match mutation {
            0 => {
                if let Conditioning::VideoClip { frames, .. } = &mut changed[0] {
                    frames.clear();
                }
            }
            1 => {
                if let Conditioning::VideoClip { frame_idx, .. } = &mut changed[0] {
                    *frame_idx = 1;
                }
            }
            _ => changed.push(Conditioning::Reference {
                image: image(2, 2, 0),
                strength: None,
            }),
        }
        assert!(
            receipt(VideoLane::Mlx, &changed).is_err(),
            "mutation{mutation}"
        );
    }
}
