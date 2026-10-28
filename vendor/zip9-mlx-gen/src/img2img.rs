//! Shared **img2img** host helpers — the small, byte-identical leaves of the fork's `LatentCreator`
//! img2img path that several image providers (Z-Image, Qwen-Image) duplicated verbatim: start-step
//! resolution, init-image preprocessing, and the noise-interpolation blend. The VAE-encode / pack
//! steps stay per-family (they depend on the family's VAE + latent packing), so this module owns only
//! the family-independent pieces (6939).

use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use crate::image::resize_lanczos_u8;
use crate::{Conditioning, Error, GenerationRequest, Image, Result};

/// Default strength for a supplied img2img reference when neither the reference nor request sets
/// one. A reference must not silently fall back to the pure-noise txt2img path.
pub const DEFAULT_IMG2IMG_STRENGTH: f32 = 0.5;

/// Resolve img2img strength with the common precedence: per-reference, request, then the positive
/// default. An explicit `Some(0.0)` is preserved as the deliberate no-op/txt2img selection.
pub fn resolve_strength(reference: Option<f32>, request: Option<f32>) -> f32 {
    reference.or(request).unwrap_or(DEFAULT_IMG2IMG_STRENGTH)
}

/// Resolve the single img2img init image from a generation request. The per-reference strength
/// takes precedence over `req.strength`; its absence remains `None` so families that have a
/// different default can apply it explicitly through [`resolve_strength`]. A `Reference` is a
/// single-image t2i/img2img surface: multi-image edit paths use [`Conditioning::MultiReference`]
/// and must resolve that policy independently.
pub fn resolve_reference<'a>(
    req: &'a GenerationRequest,
    id: &str,
) -> Result<Option<(&'a Image, Option<f32>)>> {
    let mut reference = None;
    for conditioning in &req.conditioning {
        if let Conditioning::Reference { image, strength } = conditioning {
            if reference.is_some() {
                return Err(Error::Msg(format!(
                    "{id}: multiple reference images are not supported (single img2img init only)"
                )));
            }
            reference = Some((image, strength.or(req.strength)));
        }
    }
    Ok(reference)
}

/// Resolve the img2img start step (the fork's `Config.init_time_step`): for a reference image with
/// `strength` in `(0, 1]`, `max(1, floor(num_steps · strength))`; otherwise `0` (pure txt2img).
/// Higher strength → later start → fewer denoise steps → output stays closer to the init image
/// (the fork's convention).
pub fn init_time_step(num_steps: usize, strength: Option<f32>) -> usize {
    match strength {
        Some(s) if s > 0.0 => {
            let s = s.clamp(0.0, 1.0);
            // Python `int(num_steps * strength)` truncates toward zero == floor for s >= 0.
            ((num_steps as f32 * s) as usize).max(1)
        }
        _ => 0,
    }
}

/// Scale an RGB8 init image to `target` dims with PIL LANCZOS (the fork's `scale_to_dimensions`,
/// a no-op when already sized), normalize `[0,255] → [-1,1]`, and lay out as NCHW `[1, 3, H, W]`
/// f32 — the input a VAE encoder expects. Port of `ImageUtil.to_array(scale_to_dimensions(...))`.
pub fn preprocess_init_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    if image.pixels.len()
        != gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX)
    {
        return Err(Error::Msg(format!(
            "init image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // PIL LANCZOS on the uint8 image (no-op when already at target size), matching the fork.
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)?
    };
    // /255 then [-1,1], as NHWC, then transpose to NCHW (the fork's `to_array` convention).
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let nhwc = Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]);
    Ok(nhwc.transpose_axes(&[0, 3, 1, 2])?)
}

/// Port of `LatentCreator.add_noise_by_interpolation`: `(1 - sigma) * clean + sigma * noise`. The
/// img2img blend that seeds the denoise loop at `sigma = sigmas[init_time_step]`.
pub fn add_noise_by_interpolation(clean: &Array, noise: &Array, sigma: f32) -> Result<Array> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    Ok(add(&multiply(clean, one_minus)?, &multiply(noise, s)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> Image {
        Image {
            width: 2,
            height: 2,
            pixels: vec![42; 12],
        }
    }

    #[test]
    fn resolve_reference_enforces_shared_single_image_policy() {
        let none = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(resolve_reference(&none, "test").unwrap().is_none());

        let per_reference = GenerationRequest {
            prompt: "a fox".into(),
            strength: Some(0.6),
            conditioning: vec![Conditioning::Reference {
                image: image(),
                strength: Some(0.3),
            }],
            ..Default::default()
        };
        assert_eq!(
            resolve_reference(&per_reference, "test")
                .unwrap()
                .unwrap()
                .1,
            Some(0.3)
        );

        let request_fallback = GenerationRequest {
            prompt: "a fox".into(),
            strength: Some(0.6),
            conditioning: vec![Conditioning::Reference {
                image: image(),
                strength: None,
            }],
            ..Default::default()
        };
        assert_eq!(
            resolve_reference(&request_fallback, "test")
                .unwrap()
                .unwrap()
                .1,
            Some(0.6)
        );

        let multiple = GenerationRequest {
            prompt: "a fox".into(),
            conditioning: vec![
                Conditioning::Reference {
                    image: image(),
                    strength: None,
                },
                Conditioning::Reference {
                    image: image(),
                    strength: None,
                },
            ],
            ..Default::default()
        };
        let error = resolve_reference(&multiple, "test")
            .unwrap_err()
            .to_string();
        assert!(error.contains("multiple reference images"), "{error}");
    }
}
