//! Shared test-support helpers for the real-weight GPU/MLX smoke harnesses (sc-8866, epic 8800).
//!
//! The per-lane smoke files (`*_mlx_smoke.rs` on macOS, `*_gpu_smoke.rs` on the off-Mac candle lane,
//! plus `footprint_measure.rs`) each drive `crate::inference_runtime::load(id)` + one real generation and then apply
//! the same handful of degenerate-decode floor checks and env plumbing. Those helpers were copy-pasted
//! verbatim across 10+ files; this module holds the byte-identical ones ONCE so a fix (or a stronger
//! gate) lands in a single place instead of drifting per-copy.
//!
//! Only the truly generic, model-agnostic helpers live here — `env_or` (the empty-filtering env read
//! unified in sc-8924), the RGB8 degenerate-decode floor checks (`image_mean` / `image_std` /
//! `is_all_zero`), and `save_png`. Each smoke keeps its own engine id, repo/tier/sentinel resolution,
//! and request recipe (those legitimately differ per model), so nothing about *what* a smoke asserts or
//! *which* tests run changes — this is a pure test-support reorg.
//!
//! Gated on the SUPERSET of the two smoke lanes (`test` AND (macOS OR the off-Mac candle build)) so the
//! module compiles exactly where at least one smoke that imports it does, and nowhere else.

use std::path::Path;

use gen_core::Image;

/// General degenerate-decode floor for the mean per-pixel std-dev sanity check (sc-9838, F-122).
///
/// This is the model-agnostic "is the decode alive?" guard: a NaN / all-black / flat collapse pulls
/// `image_std` toward 0, so any coherent render for the common t2i/edit/control lanes clears this by a
/// wide margin. Used by the lanes whose coherent output sits at a normal per-pixel std-dev; the lanes
/// whose coherent baseline runs materially hotter hold to the tighter [`DEGENERATE_STD_FLOOR_TIGHT`]
/// floor below instead. It is deliberately generous — it exists to catch a broken decode, not to grade
/// quality (the real quality call is the saved-PNG eyeball).
///
// Which smokes consume these floors is lane-conditional: several callers are MLX/macOS-gated, so on a
// given build lane (e.g. candle) a floor can have no consumer and read as unused. That per-lane "unused"
// is expected and benign — allow(dead_code) keeps clippy -D warnings green on every lane without
// cfg-gating the constants themselves. It is a no-op on lanes where the const IS used.
#[allow(dead_code)]
pub(crate) const DEGENERATE_STD_FLOOR_DEFAULT: f64 = 5.0;

/// Tighter degenerate-decode floor for the lanes whose coherent output runs hotter (sc-9838, F-122).
///
/// Some lanes render at a materially higher per-pixel std-dev than the common case, so a half-collapsed
/// decode can still clear the general [`DEGENERATE_STD_FLOOR_DEFAULT`] while stalling below the level a
/// live render on that lane actually reaches. Those lanes hold to this stricter per-model bar instead:
/// the Lens pair (transformer DiT alongside the heavy gpt-oss-20b MoE text encoder), chroma1_base, and
/// sdxl_base all sit here. This is a per-model tighter floor by design — it is NOT Lens-specific, and it
/// should NOT be collapsed back into the default.
///
// Lane-conditional consumer set (see DEGENERATE_STD_FLOOR_DEFAULT above): its callers are MLX/macOS-gated
// smokes, so on a lane where those are cfg'd out this const is unused. allow(dead_code) keeps clippy
// -D warnings green per-lane without cfg-gating; it is a no-op on lanes where the const IS used.
#[allow(dead_code)]
pub(crate) const DEGENERATE_STD_FLOOR_TIGHT: f64 = 20.0;

/// Read `key` from the environment, falling back to `default` when it is unset, empty, or
/// whitespace-only. Trims the value. Filtering set-but-empty values keeps a bare `FOO_STEPS=` (or a
/// whitespace-only value) from feeding `""` into a downstream `.parse()` and panicking (sc-8924).
pub(crate) fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Per-pixel mean over the RGB buffer — the "is it black?" floor check, reported for the record.
pub(crate) fn image_mean(img: &Image) -> f64 {
    let n = img.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n
}

/// Mean per-pixel std-dev across the RGB channels — a cheap "is the image non-degenerate" check. A
/// NaN / all-black / flat decode collapses the std toward 0; this guards that degenerate floor. The real
/// quality call is the saved-PNG eyeball.
pub(crate) fn image_std(img: &Image) -> f64 {
    let n = img.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = image_mean(img);
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt()
}

/// Mean absolute per-pixel difference between two same-sized RGB8 frames, on the native 0–255 scale
/// (sc-11992). The video lanes' "did the clip actually MOVE?" metric.
///
/// A FROZEN clip — the temporal path collapsed to N copies of one still — scores EXACTLY 0.0 here,
/// while passing every per-frame check a smoke makes (dimensions, non-all-zero, per-pixel std): each
/// individual frame is a perfectly good image, so the per-frame floors are structurally blind to it.
/// That is the same blind spot [`ghost_cepstrum_score`] exists for on the spatial axis.
///
/// Why the mean ABSOLUTE delta and not the difference of the two frames' [`image_mean`]s (the first
/// thing sc-11992 reached for): `image_mean` collapses a frame to ONE scalar, and motion routinely
/// preserves it — a pan, or any subject moving across a roughly uniform background, leaves the frame
/// mean essentially unchanged while every pixel moved. So `|mean(a) − mean(b)|` can read ~0 on a
/// perfectly live clip (false alarm) and, worse, is trivially cleared by a global brightness drift on
/// a frozen-geometry clip (false pass). The per-pixel mean-abs delta is 0 if and only if the frames
/// are bit-identical, which is exactly the property the guard needs.
///
/// Mismatched dimensions (or an empty frame) return 0.0 — no comparable signal. That fails SAFE for
/// the intended use: callers assert the delta is ABOVE a floor, so a mismatch trips the assertion
/// loudly instead of silently admitting.
pub(crate) fn mean_abs_frame_delta(a: &Image, b: &Image) -> f64 {
    if a.width != b.width || a.height != b.height || a.pixels.len() != b.pixels.len() {
        return 0.0;
    }
    let n = a.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    a.pixels
        .iter()
        .zip(b.pixels.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / n
}

/// Ceiling on the ghost/double-exposure cepstral score (sc-10852). See [`ghost_cepstrum_score`].
///
/// A coherent render leaves the score (a robust z-score of the strongest off-origin **cepstral** peak)
/// well under this; a ghosted double-exposure (the subject spatially translated and alpha-blended over
/// itself) is a textbook *echo*, which plants a sharp cepstral peak at the echo delay that clears it.
/// This exists because the [`DEGENERATE_STD_FLOOR_DEFAULT`] std floor is blind to ghosting: the sc-10826
/// double-exposure on the candle sdxl **default** sampler path scored per-pixel std 39.86 — far above the
/// floor — so std alone would NOT have caught it. This structural guard would.
///
/// Why the cepstrum and not plain spatial autocorrelation (the first thing sc-10852 tried): a real
/// coherent render is full of oriented / near-periodic structure (a forest of vertical trunks, hair,
/// foliage) whose raw autocorrelation has *large* off-origin peaks — on a real 1024² RealVisXL fox render
/// the autocorrelation peak (0.58) landed ABOVE several synthetic double-exposures, so no threshold
/// separated clean from ghost. The cepstrum (inverse FFT of the log power spectrum) takes the `log`, which
/// turns the echo's multiplicative `|1 + a·e^{-i·2π(u·dx+v·dy)}|²` ripple into an *additive* full-band
/// component that lands as one sharp peak at the delay, while the image's own smooth spectral envelope
/// stays at low quefrency (near the origin, excluded). On the same real fox render this scores ≈13 vs
/// ≈30–107 for pronounced double-exposures — a real, if not infinite, separation.
///
/// It is NOT a silver bullet: a *strongly periodic* legit texture (a full-frame brick grid) is
/// mathematically a sum of translated copies and can also peak here — so this is a golden-free guard tuned
/// to catch the pronounced double-exposure class (which sc-10826 was: a plainly-visible ghost), not to
/// grade quality. The [`GHOST_CEPSTRUM_Z_CEILING`] is tunable and the GPU smoke lets a maintainer relax it
/// (`RVXL_GHOST_GUARD=0`) when deliberately eyeballing a texture-heavy prompt. The saved-PNG eyeball
/// remains the quality call. Validated GPU-free by the unit tests below and, on the RTX PRO 6000 box, by
/// the `realvisxl_lightning_candle_gpu_smoke` engine-default run (real render passes; a synthesized ghost
/// of that same real render trips it).
//
// Lane-conditional consumer set (see the std floors above): the ghost guard is asserted by the off-Mac
// candle `realvisxl_lightning_gpu_smoke`, so on the macOS lane the smoke is cfg'd out — but the unit
// tests in this module reference it on both lanes, so it is not dead. allow(dead_code) is belt-and-braces
// consistency with the floors, a no-op where the const is used.
#[allow(dead_code)]
pub(crate) const GHOST_CEPSTRUM_Z_CEILING: f64 = 20.0;

/// Power-of-two working grid the luminance is area-averaged onto before the 2-D FFT (sc-10852). A
/// ghost/double-exposure is a large-scale structure that survives aggressive downsampling, and a small
/// fixed grid keeps the hand-rolled radix-2 FFT cheap enough to run inline in a GPU smoke in well under a
/// second even from a 1024²+ render.
#[allow(dead_code)]
const CEP_DIM: usize = 128;
/// L∞ radius of the low-quefrency exclusion zone: cepstral samples this close to the origin are the
/// image's own smooth spectral envelope, not an echo, so they are skipped in the peak search.
#[allow(dead_code)]
const CEP_EXCLUDE: usize = 3;

/// Structural ghost / double-exposure score: a robust z-score of the strongest off-origin peak of the
/// image's **cepstrum** (sc-10852). Higher = more echo-like; a coherent render sits low, a double-exposure
/// spikes. See [`GHOST_CEPSTRUM_Z_CEILING`] for the why-cepstrum-not-autocorrelation rationale.
///
/// Method (all on a small [`CEP_DIM`]² grid, hand-rolled FFT — no external dependency):
///  1. Rec.601 luminance, area-averaged down onto the grid, mean-subtracted, separable-Hann windowed
///     (the window kills the FFT's periodic-wrap edge seam that would otherwise read as a false echo).
///  2. 2-D FFT → `log(|F|² + ε)` → 2-D inverse FFT → real cepstrum.
///  3. over all quefrencies outside the [`CEP_EXCLUDE`] central zone, out to 3/8 of the grid, take the
///     max `(value − median) / MAD` — a median/MAD z-score so the bar is scale- and content-robust.
#[allow(dead_code)]
pub(crate) fn ghost_cepstrum_score(img: &Image) -> f64 {
    let (luma, w, h) = downsample_luma(img, CEP_DIM);
    if w < 2 * CEP_EXCLUDE + 4 || h < 2 * CEP_EXCLUDE + 4 {
        return 0.0;
    }
    let n = CEP_DIM;
    let mean = luma.iter().sum::<f64>() / luma.len() as f64;
    // Mean-subtract + separable Hann window, content centered in the N×N (zero-padded) grid.
    let hann = |i: usize, len: usize| -> f64 {
        if len <= 1 {
            1.0
        } else {
            0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (len as f64 - 1.0)).cos()
        }
    };
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];
    let (oy, ox) = ((n - h) / 2, (n - w) / 2);
    for y in 0..h {
        let wy = hann(y, h);
        for x in 0..w {
            re[(oy + y) * n + (ox + x)] = (luma[y * w + x] - mean) * wy * hann(x, w);
        }
    }
    fft_2d(&mut re, &mut im, n, false);
    // Real cepstrum = IFFT(log power spectrum).
    for i in 0..n * n {
        re[i] = (re[i] * re[i] + im[i] * im[i] + 1e-6).ln();
        im[i] = 0.0;
    }
    fft_2d(&mut re, &mut im, n, true);

    // Robust peak z-score over the off-origin quefrency search window (cepstrum is centered at index 0
    // and wraps, so negative quefrencies are indexed modulo n).
    let search = (n * 3 / 8) as isize;
    let excl = CEP_EXCLUDE as isize;
    let at = |dy: isize, dx: isize| -> f64 {
        re[dy.rem_euclid(n as isize) as usize * n + dx.rem_euclid(n as isize) as usize]
    };
    let mut vals: Vec<f64> = Vec::new();
    for dy in -search..=search {
        for dx in -search..=search {
            if dx.abs().max(dy.abs()) <= excl {
                continue;
            }
            vals.push(at(dy, dx));
        }
    }
    if vals.is_empty() {
        return 0.0;
    }
    let med = median(&mut vals.clone());
    let mut dev: Vec<f64> = vals.iter().map(|v| (v - med).abs()).collect();
    let mad = median(&mut dev) + 1e-9;
    let mut best = 0.0_f64;
    for dy in -search..=search {
        for dx in -search..=search {
            if dx.abs().max(dy.abs()) <= excl {
                continue;
            }
            let z = (at(dy, dx) - med) / mad;
            if z > best {
                best = z;
            }
        }
    }
    best
}

/// Median of `v` (mutates: sorts in place). Empty → 0.0.
#[allow(dead_code)]
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in cepstrum"));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

/// In-place iterative radix-2 Cooley-Tukey FFT of a power-of-two-length complex signal (`re`/`im`
/// parallel arrays). `inverse` runs the IFFT (conjugate twiddles + 1/N scaling).
#[allow(dead_code)]
fn fft_1d(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = if inverse { 1.0 } else { -1.0 } * 2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0_f64, 0.0_f64);
            for k in 0..half {
                let (a, b) = (i + k, i + k + half);
                let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let inv = 1.0 / n as f64;
        for x in re.iter_mut() {
            *x *= inv;
        }
        for x in im.iter_mut() {
            *x *= inv;
        }
    }
}

/// 2-D FFT of an `n × n` row-major complex grid, via 1-D FFTs over rows then columns.
#[allow(dead_code)]
fn fft_2d(re: &mut [f64], im: &mut [f64], n: usize, inverse: bool) {
    let (mut lr, mut li) = (vec![0.0_f64; n], vec![0.0_f64; n]);
    for y in 0..n {
        let base = y * n;
        lr.copy_from_slice(&re[base..base + n]);
        li.copy_from_slice(&im[base..base + n]);
        fft_1d(&mut lr, &mut li, inverse);
        re[base..base + n].copy_from_slice(&lr);
        im[base..base + n].copy_from_slice(&li);
    }
    for x in 0..n {
        for y in 0..n {
            lr[y] = re[y * n + x];
            li[y] = im[y * n + x];
        }
        fft_1d(&mut lr, &mut li, inverse);
        for y in 0..n {
            re[y * n + x] = lr[y];
            im[y * n + x] = li[y];
        }
    }
}

/// Rec.601 luminance area-averaged down so the long side is ≤ `max_dim` (never upscaled). Returns the
/// row-major `f64` luma plane and its `(width, height)`. When the source already fits, each target maps
/// to a single source pixel (identity).
#[allow(dead_code)]
fn downsample_luma(img: &Image, max_dim: usize) -> (Vec<f64>, usize, usize) {
    let (sw, sh) = (img.width as usize, img.height as usize);
    if sw == 0 || sh == 0 || img.pixels.len() < sw * sh * 3 {
        return (Vec::new(), 0, 0);
    }
    let long = sw.max(sh);
    let (tw, th) = if long <= max_dim {
        (sw, sh)
    } else {
        let scale = max_dim as f64 / long as f64;
        (
            ((sw as f64 * scale).round() as usize).max(1),
            ((sh as f64 * scale).round() as usize).max(1),
        )
    };
    let mut out = vec![0.0_f64; tw * th];
    for ty in 0..th {
        let sy0 = ty * sh / th;
        let sy1 = (((ty + 1) * sh / th).max(sy0 + 1)).min(sh);
        for tx in 0..tw {
            let sx0 = tx * sw / tw;
            let sx1 = (((tx + 1) * sw / tw).max(sx0 + 1)).min(sw);
            let mut acc = 0.0_f64;
            let mut n = 0.0_f64;
            for sy in sy0..sy1 {
                let row = sy * sw;
                for sx in sx0..sx1 {
                    let p = (row + sx) * 3;
                    acc += 0.299 * img.pixels[p] as f64
                        + 0.587 * img.pixels[p + 1] as f64
                        + 0.114 * img.pixels[p + 2] as f64;
                    n += 1.0;
                }
            }
            out[ty * tw + tx] = acc / n;
        }
    }
    (out, tw, th)
}

/// Whether EVERY pixel byte is exactly 0 — the precise degenerate signature of a broken decode.
// Only the macOS MLX packed-tier smokes (chroma1_base/lens*/sdxl) assert the all-zero floor; the
// off-Mac candle smokes don't, so this is legitimately unused on the candle-only lane. `allow` rather
// than a cfg gate keeps this shared helper available to any future smoke on either lane.
#[allow(dead_code)]
pub(crate) fn is_all_zero(img: &Image) -> bool {
    !img.pixels.is_empty() && img.pixels.iter().all(|&p| p == 0)
}

/// Write the RGB8 image to `path` as a PNG, panicking with the path on any failure.
pub(crate) fn save_png(img: &Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .expect("rgb buffer")
        .save(path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
}

/// GPU-free structural coverage for the video-lane motion metric (sc-11992). The smoke that consumes
/// [`mean_abs_frame_delta`] is `#[ignore]`d and needs ~50 GB of weights, so these tests pin the
/// metric's two load-bearing properties against synthetic frames: a frozen pair scores EXACTLY 0
/// (the failure the floor exists to catch), and a moved subject scores well clear of it — including
/// the case where the frame MEAN is preserved, which is precisely where the `image_mean`-difference
/// check the smoke used to print would have read ~0 on a live clip.
#[cfg(test)]
mod motion_metric_tests {
    use super::*;

    /// A `w`×`h` RGB8 frame: mid-gray with one white `size`×`size` square at (`x`, `y`). Moving the
    /// square translates content without changing how many white pixels exist — so the frame MEAN is
    /// identical between two positions, while every pixel under the square changed.
    fn frame_with_square(w: u32, h: u32, x: u32, y: u32, size: u32) -> Image {
        let mut pixels = vec![128u8; (w * h * 3) as usize];
        for row in y..(y + size).min(h) {
            for col in x..(x + size).min(w) {
                let i = ((row * w + col) * 3) as usize;
                pixels[i] = 255;
                pixels[i + 1] = 255;
                pixels[i + 2] = 255;
            }
        }
        Image {
            width: w,
            height: h,
            pixels,
        }
    }

    #[test]
    fn a_frozen_pair_scores_exactly_zero() {
        // The whole point: N copies of one good still. Every per-frame smoke check passes; this is the
        // only signal that separates it from a live clip.
        let frame = frame_with_square(64, 64, 8, 8, 16);
        assert_eq!(mean_abs_frame_delta(&frame, &frame), 0.0);
        let identical = frame_with_square(64, 64, 8, 8, 16);
        assert_eq!(mean_abs_frame_delta(&frame, &identical), 0.0);
    }

    #[test]
    fn a_moved_subject_scores_above_zero_even_when_the_frame_mean_is_unchanged() {
        // Translating the square leaves `image_mean` bit-identical (same pixel population), so the
        // old `|mean(first) − mean(last)|` check reads ~0 here — a live clip it would call frozen.
        // The per-pixel metric sees the motion.
        let a = frame_with_square(64, 64, 8, 8, 16);
        let b = frame_with_square(64, 64, 32, 32, 16);
        assert_eq!(
            image_mean(&a),
            image_mean(&b),
            "fixture invariant: translation preserves the frame mean"
        );
        let delta = mean_abs_frame_delta(&a, &b);
        assert!(
            delta > 1.0,
            "a fully displaced subject must score well above the liveness floor, got {delta:.3}"
        );
    }

    #[test]
    fn the_metric_scales_with_how_much_moved() {
        // Monotone in the amount of changed content — a partial overlap moves less than a full
        // displacement. Guards against a metric that saturates (e.g. any-pixel-changed booleans).
        let base = frame_with_square(64, 64, 8, 8, 16);
        let small = mean_abs_frame_delta(&base, &frame_with_square(64, 64, 12, 12, 16));
        let large = mean_abs_frame_delta(&base, &frame_with_square(64, 64, 40, 40, 16));
        assert!(
            large > small && small > 0.0,
            "expected 0 < small ({small:.3}) < large ({large:.3})"
        );
    }

    #[test]
    fn mismatched_or_empty_frames_score_zero_and_fail_safe() {
        // No comparable signal ⇒ 0.0, which is BELOW any motion floor ⇒ the caller's assertion trips
        // loudly rather than silently admitting an unverifiable clip.
        let a = frame_with_square(64, 64, 8, 8, 16);
        let smaller = frame_with_square(32, 32, 4, 4, 8);
        assert_eq!(mean_abs_frame_delta(&a, &smaller), 0.0);
        let empty = Image {
            width: 0,
            height: 0,
            pixels: Vec::new(),
        };
        assert_eq!(mean_abs_frame_delta(&empty, &empty), 0.0);
    }
}

/// GPU-free structural coverage for the ghost/double-exposure guard (sc-10852). The `*_gpu_smoke`
/// harness that consumes [`ghost_cepstrum_score`] is `#[ignore]`d and needs multi-GB weights, so these
/// tests lock the metric + the [`GHOST_CEPSTRUM_Z_CEILING`] threshold against *synthetic* clean vs.
/// double-exposure frames — a coherent frame must clear the ceiling with margin, and a
/// translated-and-blended copy of itself (the exact echo signature the std floor missed on sc-10826) must
/// trip it. A round-trip FFT identity test guards the hand-rolled transform. These run in the ordinary
/// `cargo test` pass, no hardware required.
#[cfg(test)]
mod ghost_guard_tests {
    use super::*;

    /// Deterministic value in `[0, 1)` — a cheap integer-hash noise field, so the synthetic subject
    /// carries broadband, non-periodic high-frequency content (no self-similarity of its own to bias the
    /// cepstral peak).
    fn hash01(x: usize, y: usize) -> f64 {
        let h = (x as u64).wrapping_mul(73_856_093) ^ (y as u64).wrapping_mul(19_349_663);
        let h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((h >> 40) & 0xFFFF) as f64 / 65_536.0
    }

    /// A single-subject luminance field `S(x, y)`: a smooth gradient base, a couple of Gaussian "subject"
    /// blobs (structure/edges), and broadband noise. Non-periodic — a faithful stand-in for a coherent
    /// single-subject render (the class the guard must clear).
    fn subject_luma(w: usize, h: usize) -> Vec<f64> {
        let mut s = vec![0.0_f64; w * h];
        let blob = |x: f64, y: f64, cx: f64, cy: f64, sig: f64, amp: f64| {
            let dx = x - cx;
            let dy = y - cy;
            amp * (-(dx * dx + dy * dy) / (2.0 * sig * sig)).exp()
        };
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64, y as f64);
                let base = 40.0 + 60.0 * fx / w as f64 + 20.0 * fy / h as f64;
                let structure = blob(
                    fx,
                    fy,
                    w as f64 * 0.35,
                    h as f64 * 0.40,
                    w as f64 * 0.12,
                    150.0,
                ) + blob(
                    fx,
                    fy,
                    w as f64 * 0.62,
                    h as f64 * 0.58,
                    w as f64 * 0.08,
                    120.0,
                );
                let noise = 40.0 * (hash01(x, y) - 0.5);
                s[y * w + x] = (base + structure + noise).clamp(0.0, 255.0);
            }
        }
        s
    }

    /// Wrap a luminance plane into an RGB8 [`Image`] (grayscale: luma → R = G = B).
    fn gray_image(luma: &[f64], w: usize, h: usize) -> Image {
        let mut pixels = vec![0u8; w * h * 3];
        for (i, &l) in luma.iter().enumerate() {
            let v = l.round().clamp(0.0, 255.0) as u8;
            pixels[i * 3] = v;
            pixels[i * 3 + 1] = v;
            pixels[i * 3 + 2] = v;
        }
        Image {
            width: w as u32,
            height: h as u32,
            pixels,
        }
    }

    /// A double-exposure: `alpha·S(x, y) + (1 − alpha)·S(x − dx, y − dy)` — the subject alpha-blended over
    /// a translated (wrap-around, so no seam) copy of itself. This is a textbook echo, the signature the
    /// cepstrum is built to flag.
    fn double_exposure(
        base: &[f64],
        w: usize,
        h: usize,
        dx: usize,
        dy: usize,
        alpha: f64,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let sx = (x + w - dx % w) % w;
                let sy = (y + h - dy % h) % h;
                out[y * w + x] = alpha * base[y * w + x] + (1.0 - alpha) * base[sy * w + sx];
            }
        }
        out
    }

    #[test]
    fn coherent_frame_clears_the_ghost_ceiling() {
        let clean = gray_image(&subject_luma(256, 256), 256, 256);
        let z = ghost_cepstrum_score(&clean);
        assert!(
            z < GHOST_CEPSTRUM_Z_CEILING,
            "coherent single-subject frame should clear the ceiling, got cepstral z {z:.2} \
             (ceiling {GHOST_CEPSTRUM_Z_CEILING})"
        );
    }

    #[test]
    fn double_exposure_trips_the_ghost_ceiling() {
        let s = subject_luma(256, 256);
        // Pronounced ghosts (the sc-10826 failure class was a plainly-visible double-exposure). A faint
        // near-invisible echo (alpha >= ~0.8) is intentionally NOT asserted — it blurs into the noise
        // floor and isn't the regression this guards.
        for (dx, dy, alpha) in [(40usize, 24usize, 0.5_f64), (30, 30, 0.6), (48, 20, 0.65)] {
            let ghost = gray_image(&double_exposure(&s, 256, 256, dx, dy, alpha), 256, 256);
            let z = ghost_cepstrum_score(&ghost);
            assert!(
                z > GHOST_CEPSTRUM_Z_CEILING,
                "double-exposure (dx={dx}, dy={dy}, alpha={alpha}) should trip the ceiling, got \
                 cepstral z {z:.2} (ceiling {GHOST_CEPSTRUM_Z_CEILING})"
            );
        }
    }

    #[test]
    fn ghost_separates_from_clean_with_margin() {
        // The whole point: clean and ghosted scores sit on opposite sides of the ceiling with a real
        // dead-band between them (not a hairline split).
        let s = subject_luma(256, 256);
        let clean_z = ghost_cepstrum_score(&gray_image(&s, 256, 256));
        let ghost_z = ghost_cepstrum_score(&gray_image(
            &double_exposure(&s, 256, 256, 36, 20, 0.55),
            256,
            256,
        ));
        eprintln!(
            "[ghost-guard] clean_z={clean_z:.2} ghost_z={ghost_z:.2} ceiling={GHOST_CEPSTRUM_Z_CEILING}"
        );
        assert!(
            ghost_z > clean_z + 10.0,
            "ghost/clean separation too thin: clean {clean_z:.2} vs ghost {ghost_z:.2}"
        );
    }

    #[test]
    fn flat_and_tiny_frames_are_safe() {
        // A flat frame has no echo → z stays finite and under the ceiling (no false ghost). A
        // sub-grid-size frame returns 0 rather than panicking.
        let flat_px = vec![128.0_f64; 256 * 256];
        let flat = gray_image(&flat_px, 256, 256);
        assert!(ghost_cepstrum_score(&flat) < GHOST_CEPSTRUM_Z_CEILING);
        let tiny_px = vec![10.0_f64; 3 * 3];
        let tiny = gray_image(&tiny_px, 3, 3);
        assert_eq!(ghost_cepstrum_score(&tiny), 0.0);
    }

    #[test]
    fn fft_round_trips() {
        // IFFT(FFT(x)) == x guards the hand-rolled radix-2 transform the cepstrum relies on.
        let n = 8;
        let mut re: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.37).sin() * 10.0).collect();
        let mut im = vec![0.0_f64; n * n];
        let orig = re.clone();
        fft_2d(&mut re, &mut im, n, false);
        fft_2d(&mut re, &mut im, n, true);
        for (a, b) in orig.iter().zip(re.iter()) {
            assert!((a - b).abs() < 1e-9, "fft round-trip drift: {a} vs {b}");
        }
    }
}
