//! Image helpers split across the backend boundary (epic 3720, Appendix A):
//!
//! - The PIL-compatible resampling ([`resize_bicubic_u8`] & friends) and the host-side mask /
//!   geometry ops ([`contain_box`], [`outpaint_border_mask`], [`union_masks`]) are pure and now live
//!   in [`gen_core::imageops`]; they are re-exported here so the historical `mlx_gen::image::…`
//!   paths keep resolving.
//! - [`decoded_to_image`] — the VAE-decoded-tensor → [`Image`] denormalize/quantize step — operates
//!   on an `mlx_rs::Array` and stays here.

use mlx_rs::ops::{add, maximum, minimum, multiply, round};
use mlx_rs::Array;

use crate::array::scalar;
use crate::media::{Image, RgbaImage};
use crate::{Error, Result};

pub use gen_core::imageops::*;

/// Denormalize a VAE-decoded tensor to an RGB8 [`Image`]: `clip(x·0.5 + 0.5, 0, 1)` → drop the
/// singleton temporal axis (5-D → 4-D) → NCHW→NHWC → `(x·255).round()` → `u8` (batch must be 1).
/// Identical across the Z-Image and Qwen-Image pipelines (the decoded tensor must already be f32).
pub fn decoded_to_image(decoded: &Array) -> Result<Image> {
    let (width, height, channels, pixels) = decoded_to_interleaved_u8(decoded, "decoded_to_image")?;
    if channels != Image::CHANNELS {
        return Err(Error::Msg(format!(
            "decoded_to_image: expected a 3-channel (RGB) tensor, got {channels} channels; use \
             decoded_to_rgba_image for the four-channel decode"
        )));
    }
    Ok(Image {
        width,
        height,
        pixels,
    })
}

/// The four-channel sibling of [`decoded_to_image`] (sc-24111): denormalize a VAE-decoded **RGBA**
/// tensor to an [`RgbaImage`] with **straight (un-premultiplied)** alpha.
///
/// Bit-for-bit the same denormalize/quantize chain as [`decoded_to_image`], applied to four
/// channels instead of three — which is exactly upstream's own arithmetic. diffusers'
/// `VaeImageProcessor.postprocess` runs `(x / 2 + 0.5).clamp(0, 1)` per channel *independently*,
/// then `(v · 255).round().astype(uint8)` and `PIL.Image.fromarray(..., mode="RGBA")`; nothing
/// premultiplies and the alpha is clamped by the same clamp as the colour. So the RGB emission and
/// the RGBA emission of one decode differ only in whether the alpha is applied (over white) or
/// carried.
pub fn decoded_to_rgba_image(decoded: &Array) -> Result<RgbaImage> {
    let (width, height, channels, pixels) =
        decoded_to_interleaved_u8(decoded, "decoded_to_rgba_image")?;
    if channels != RgbaImage::CHANNELS {
        return Err(Error::Msg(format!(
            "decoded_to_rgba_image: expected a 4-channel (RGBA) tensor, got {channels} channels"
        )));
    }
    Ok(RgbaImage {
        width,
        height,
        pixels,
    })
}

/// The shared denormalize/quantize core of [`decoded_to_image`] and [`decoded_to_rgba_image`]:
/// `clip(x·0.5 + 0.5, 0, 1)` → drop the singleton temporal axis (5-D → 4-D) → NCHW→NHWC →
/// `(x·255).round()` → `u8`. Returns `(width, height, channels, interleaved pixels)`; the channel
/// count is whatever the tensor carried, and the callers above are what bind it to a type.
///
/// One body, so the two emissions can never drift in rounding: a provider that emits RGB for one
/// request and RGBA for the next must quantize both identically, or the "RGB is the RGBA decode
/// composited over white" claim in the provider `UPSTREAM.md`s stops being checkable.
fn decoded_to_interleaved_u8(decoded: &Array, who: &str) -> Result<(u32, u32, usize, Vec<u8>)> {
    // Rank + dtype hardening (F-064): the readback below is `as_slice::<f32>()`, which reinterprets
    // the raw buffer — a bf16/f16 tensor (one missed `.as_dtype(Float32)` in any of ~20 provider
    // decode paths) would mis-read the bytes and abort the process. Reject a wrong rank up front, and
    // defensively cast to f32 before the arithmetic (a no-op for the f32 tensors every caller passes).
    let rank = decoded.shape().len();
    if rank != 4 && rank != 5 {
        return Err(Error::Msg(format!(
            "{who}: expected a 4-D (NCHW) or 5-D (NCTHW) tensor, got rank {rank}"
        )));
    }
    let decoded = decoded.as_dtype(mlx_rs::Dtype::Float32)?;
    let half = scalar(0.5);
    // denormalize: clip(x*0.5 + 0.5, 0, 1)
    let x = add(&multiply(&decoded, &half)?, &half)?;
    let x = minimum(&maximum(&x, scalar(0.0))?, scalar(1.0))?;
    // drop the singleton temporal axis if present (5-D → 4-D)
    let x = if x.shape().len() == 5 {
        x.squeeze_axes(&[2])?
    } else {
        x
    };
    // NCHW → NHWC
    let x = x.transpose_axes(&[0, 2, 3, 1])?;
    // (x*255).round() to integer pixel values.
    let x = round(&multiply(&x, scalar(255.0))?, 0)?;

    let sh = x.shape();
    // Batch-correct + overflow-safe (F-082): this path emits a single image, so reject B>1 (the
    // per-image pipelines call with B==1) rather than silently keeping only batch 0, and size in
    // usize / flatten via -1 to avoid the u32/i32 product overflow at large resolutions.
    if sh[0] != 1 {
        return Err(Error::Msg(format!(
            "{who}: expected batch size 1, got {}",
            sh[0]
        )));
    }
    let (h, w, c) = (sh[1] as usize, sh[2] as usize, sh[3] as usize);
    let n = h * w * c;
    // `transpose_axes` yields a strided view; `reshape` re-materializes in C-order, so the slice is
    // logical NHWC.
    let flat = x.reshape(&[-1])?;
    let pixels: Vec<u8> = flat.as_slice::<f32>()[..n]
        .iter()
        .map(|&v| v as u8)
        .collect();
    Ok((w as u32, h as u32, c, pixels))
}

/// Reject pipeline dimensions that aren't multiples of `multiple` — the latent-pack requirement
/// shared by the mlx image pipelines that pack an /8 VAE latent with a 2×2 patchify (FLUX.1 / FLUX.2
/// / Boogu / Krea). `family` names the model in the error message so the single shared check reads
/// like the per-crate copies it replaces (F-083).
///
/// `multiple` is the caller's own family stride const — each family passes the SAME crate-root
/// `pub const` its `validate_request` enforces (`mlx_gen_flux::SIZE_MULTIPLE`,
/// `mlx_gen_boogu::RES_MULTIPLE`, …, all `= 16` today), so this pipeline-layer enforcement site can
/// never drift from the request-dimension stride sc-12612 tied. Passing the value rather than
/// hardcoding `16` is what makes "wired into every enforcement site" (sc-12612) true here (sc-12701).
pub fn validate_multiple_of(width: u32, height: u32, multiple: u32, family: &str) -> Result<()> {
    if !width.is_multiple_of(multiple) || !height.is_multiple_of(multiple) {
        return Err(Error::Msg(format!(
            "{family}: width and height must be multiples of {multiple}, got {width}x{height}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_rgb8_policy_rounds_midpoints_to_even() {
        // Pin native MLX/PyTorch midpoint semantics directly. A cast without round would truncate.
        let scaled = Array::from_slice(&[0.5f32, 1.5, 2.5, 3.5, 254.5], &[5]);
        let rounded = mlx_rs::ops::round(&scaled, 0).unwrap();
        assert_eq!(rounded.as_slice::<f32>(), &[0.0, 2.0, 2.0, 4.0, 254.0]);

        // Zero decoded is the exact midpoint in `(x + 1) * 127.5`: it maps to 128, not 127.
        let image = decoded_to_image(&Array::from_slice(&[0.0f32; 3], &[1, 3, 1, 1])).unwrap();
        assert_eq!(image.pixels, [128, 128, 128]);
    }

    /// `validate_multiple_of` must enforce the `multiple` it is PASSED, not a hardcoded literal
    /// (sc-12701). The discriminating case is 496×480: a multiple of 16 but not of 32 — it must be
    /// accepted at `multiple = 16` and rejected at `multiple = 32`. A regression that ignored the
    /// argument (e.g. the old body's literal `16`) would accept it in both, so this fails RED on the
    /// mutation the tie exists to prevent. The error text must also surface the actual `multiple`.
    #[test]
    fn validate_multiple_of_enforces_the_passed_stride() {
        // On-stride passes at its own stride.
        assert!(validate_multiple_of(512, 512, 16, "flux1").is_ok());
        assert!(validate_multiple_of(1024, 768, 32, "stride32_probe").is_ok());

        // Off-stride rejects, and the message names the stride it enforced (not a baked 16).
        let err =
            validate_multiple_of(500, 512, 16, "flux1").expect_err("500 is not a multiple of 16");
        let msg = err.to_string();
        assert!(
            msg.contains("multiples of 16"),
            "message must name the stride: {msg}"
        );
        assert!(msg.contains("500x512"), "message must echo the dims: {msg}");

        // Height alone off-stride also rejects (both axes are checked).
        assert!(validate_multiple_of(512, 500, 16, "flux1").is_err());

        // Discriminator: 496×480 is ÷16 but not ÷32 — the argument, not a literal, decides.
        assert!(
            validate_multiple_of(496, 480, 16, "flux1").is_ok(),
            "496×480 is a multiple of 16"
        );
        let stride32 = validate_multiple_of(496, 480, 32, "stride32_probe")
            .expect_err("496 is not a multiple of 32");
        assert!(
            stride32.to_string().contains("multiples of 32"),
            "a ÷32 caller must reject a ÷16-only size with a ÷32 message: {stride32}"
        );
    }

    /// `resize_u8` (now in gen-core) must be **bit-identical** to PIL `Image.BICUBIC` (the
    /// fixed-point integer path), not merely close — this is what gives the conditioning images
    /// pixel-parity with the fork (sc-2465: an f64-coefficient resampler diverged ±1–2 ULP at
    /// gradient cliffs → 24% e2e px>8). The golden tensor is read via `crate::weights::Weights`
    /// (MLX), so this test stays in mlx-gen. Golden via `tools/dump_pil_resize_golden.py`.
    #[test]
    #[ignore = "needs tools/golden/pil_resize_golden.safetensors (run tools/dump_pil_resize_golden.py)"]
    fn resize_bicubic_matches_pil_512_to_384() {
        let g = crate::weights::Weights::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/golden/pil_resize_golden.safetensors"
        ))
        .unwrap();
        // Sawtooth: `(x+y)%256`, ×2, ×3 — sharp 255→0 cliffs where bicubic implementations diverge.
        let mut saw = Vec::with_capacity(512 * 512 * 3);
        let mut smo = Vec::with_capacity(512 * 512 * 3);
        for y in 0..512u32 {
            for x in 0..512u32 {
                let b = (y + x) % 256;
                saw.push(b as u8);
                saw.push(((b * 2) % 256) as u8);
                saw.push(((b * 3) % 256) as u8);
                let v = ((x + y) / 4).min(255) as u8;
                smo.push(v);
                smo.push(v);
                smo.push(v);
            }
        }
        let cmp = |got: &[f32], pil: &[i32]| -> (usize, i32) {
            assert_eq!(got.len(), pil.len(), "len");
            got.iter()
                .zip(pil)
                .fold((0usize, 0i32), |(n, m), (&gv, &pv)| {
                    let d = (gv as i32 - pv).abs();
                    (n + (d != 0) as usize, m.max(d))
                })
        };
        let (saw_diff, saw_max) = cmp(
            &resize_bicubic_u8(&saw, 512, 512, 384, 384).unwrap(),
            g.require("pil384").unwrap().as_slice::<i32>(),
        );
        let (smo_diff, smo_max) = cmp(
            &resize_bicubic_u8(&smo, 512, 512, 384, 384).unwrap(),
            g.require("pil384_smooth").unwrap().as_slice::<i32>(),
        );
        println!("vs PIL 512->384: sawtooth {saw_diff} diff (max {saw_max}), smooth {smo_diff} diff (max {smo_max})");
        assert_eq!(
            saw_diff, 0,
            "resize_u8 must bit-match PIL BICUBIC on the cliff gradient"
        );
        assert_eq!(
            smo_diff, 0,
            "resize_u8 must bit-match PIL BICUBIC on the smooth ramp"
        );
    }
}
