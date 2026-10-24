//! Dataset preparation helpers for training (sc-3043) — family-agnostic and dependency-free.
//!
//! These operate on the core [`Image`] (decoded RGB8) so the core crate needs no image-decode
//! dependency: the family trainer (which already links the `image` crate for output encoding)
//! decodes the file into an [`Image`], then this module center-crops to a square and the family's
//! own `preprocess_init_image` resizes/normalises into the VAE input. Resolution is bucketed to a
//! multiple of 32 (the latent grid must tile cleanly), mirroring the Python kernel's
//! `bucket_resolution`.

use crate::media::Image;

/// Floor `resolution` to a multiple of 32 (the training-resolution bucket). Mirrors the Python
/// `bucket_resolution`: `0` → the `512` default; otherwise `(res/32)*32`, with a 32-px floor so a
/// tiny-but-nonzero input never collapses to 0 (the raw Python would yield 0 for `res < 32`; that
/// is never hit in practice but is a footgun, so we guard it).
pub fn bucket_resolution(resolution: u32) -> u32 {
    if resolution == 0 {
        return 512;
    }
    ((resolution / 32) * 32).max(32)
}

/// The bucketed training edges of `config` (epic 2123 sc-2127) — one per
/// [`TrainingConfig::training_buckets`](gen_core::TrainingConfig::training_buckets) entry, in order,
/// each floored by [`bucket_resolution`]. With no `resolution_buckets` this is exactly
/// `[bucket_resolution(config.resolution)]`, the single edge every trainer used before buckets
/// existed. A trainer caches each item once per edge (item-major) and walks the cache through a
/// [`gen_core::BucketSchedule`].
pub fn bucket_edges(config: &gen_core::TrainingConfig) -> Vec<u32> {
    config
        .training_buckets()
        .iter()
        .map(|b| bucket_resolution(b.resolution))
        .collect()
}

/// Center-crop `image` to its largest centered square (`side = min(w, h)`). A no-op when already
/// square. The family then resizes the square to the bucketed training edge.
pub fn center_crop_square(image: &Image) -> Image {
    let (w, h) = (image.width, image.height);
    if w == h {
        return image.clone();
    }
    let side = w.min(h);
    let x0 = (w - side) / 2;
    let y0 = (h - side) / 2;
    let mut pixels = Vec::with_capacity((side * side * 3) as usize);
    for y in y0..y0 + side {
        let row = ((y * w + x0) * 3) as usize;
        pixels.extend_from_slice(&image.pixels[row..row + (side * 3) as usize]);
    }
    Image {
        width: side,
        height: side,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_edges_floor_every_bucket_and_default_to_the_legacy_edge() {
        let mut cfg = gen_core::TrainingConfig {
            resolution: 1000,
            ..Default::default()
        };
        assert_eq!(bucket_edges(&cfg), vec![992]);
        cfg.resolution_buckets = vec![
            gen_core::ResolutionBucket {
                resolution: 512,
                repeats: 16,
            },
            gen_core::ResolutionBucket {
                resolution: 770,
                repeats: 4,
            },
            gen_core::ResolutionBucket {
                resolution: 1024,
                repeats: 1,
            },
        ];
        assert_eq!(bucket_edges(&cfg), vec![512, 768, 1024]);
    }

    #[test]
    fn bucket_floors_to_multiple_of_32() {
        assert_eq!(bucket_resolution(1024), 1024);
        assert_eq!(bucket_resolution(1000), 992); // 31*32
        assert_eq!(bucket_resolution(1023), 992);
        assert_eq!(bucket_resolution(0), 512); // default fallback
        assert_eq!(bucket_resolution(16), 32); // floored to the 32-px guard, not 0
        assert_eq!(bucket_resolution(512), 512);
    }

    #[test]
    fn center_crop_square_noop_when_square() {
        let img = Image {
            width: 4,
            height: 4,
            pixels: (0..4 * 4 * 3).map(|i| i as u8).collect(),
        };
        assert_eq!(center_crop_square(&img), img);
    }

    #[test]
    fn center_crop_landscape_takes_center() {
        // 4x2 RGB: each pixel's R channel = its column index, so we can check which columns survive.
        let mut pixels = Vec::new();
        for _y in 0..2 {
            for x in 0..4u8 {
                pixels.extend_from_slice(&[x, 0, 0]);
            }
        }
        let img = Image {
            width: 4,
            height: 2,
            pixels,
        };
        let out = center_crop_square(&img);
        assert_eq!((out.width, out.height), (2, 2));
        // x0 = (4-2)/2 = 1 → columns 1,2 survive in each of the 2 rows.
        let cols: Vec<u8> = out.pixels.chunks(3).map(|p| p[0]).collect();
        assert_eq!(cols, vec![1, 2, 1, 2]);
    }
}
