use image::RgbImage;

/// Bilinearly sample all three RGB channels with one neighbor fetch. Callers may
/// reorder the returned channels for BGR model inputs without re-reading pixels.
#[inline]
pub(crate) fn sample_rgb_bilinear(image: &RgbImage, x: f32, y: f32, border: f32) -> [f32; 3] {
    let (width, height) = (image.width() as i64, image.height() as i64);
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let get = |xi: i64, yi: i64| -> [f32; 3] {
        if xi < 0 || yi < 0 || xi >= width || yi >= height {
            [border; 3]
        } else {
            image.get_pixel(xi as u32, yi as u32).0.map(f32::from)
        }
    };
    let [v00, v10, v01, v11] = [
        get(x0, y0),
        get(x0 + 1, y0),
        get(x0, y0 + 1),
        get(x0 + 1, y0 + 1),
    ];
    std::array::from_fn(|channel| {
        let top = v00[channel] * (1.0 - fx) + v10[channel] * fx;
        let bottom = v01[channel] * (1.0 - fx) + v11[channel] * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn samples_all_channels_with_shared_geometry() {
        let image = RgbImage::from_fn(2, 2, |x, y| {
            let base = (x + y * 2) as u8 * 30;
            Rgb([base, base + 10, base + 20])
        });
        assert_eq!(
            sample_rgb_bilinear(&image, 0.5, 0.5, 0.0),
            [45.0, 55.0, 65.0]
        );
        assert_eq!(sample_rgb_bilinear(&image, -1.0, -1.0, 7.0), [7.0; 3]);
    }
}
