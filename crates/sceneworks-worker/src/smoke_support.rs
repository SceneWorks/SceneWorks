//! Shared test-support helpers for the real-weight GPU/MLX smoke harnesses (sc-8866, epic 8800).
//!
//! The per-lane smoke files (`*_mlx_smoke.rs` on macOS, `*_gpu_smoke.rs` on the off-Mac candle lane,
//! plus `footprint_measure.rs`) each drive `gen_core::load(id)` + one real generation and then apply
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
