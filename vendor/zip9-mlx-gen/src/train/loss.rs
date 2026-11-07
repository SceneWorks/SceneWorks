//! The shared MLX training-loss reduction (epic 2123, sc-24828) — MSE / MAE with optional
//! **subject-mask loss weighting**.
//!
//! Every MLX trainer's traced loss closure reduces `diff = prediction − target` through
//! [`reduce_loss`]. With no weight it is exactly the historical `mean(diff²)` / `mean(|diff|)` (the
//! same ops, so a mask-off run is bit-identical). With a weight map `w` (built once per item at
//! cache time by [`subject_mask_weight`] from the backend-neutral
//! [`gen_core::train::subject_mask`](crate::train::subject_mask) map), the per-element loss is
//! multiplied by `w` **before** the mean — `mean(w ⊙ ℓ)`, the convention documented there — so a
//! latent element with `w = 0` contributes zero loss and zero gradient.

use mlx_rs::error::Result as MlxResult;
use mlx_rs::ops::{broadcast_to, multiply};
use mlx_rs::Array;

/// `mean(ℓ)` over every element, `ℓ = |diff|` (MAE) or `diff²` (MSE), optionally weighted per
/// element by `weight` (broadcastable to `diff`): `mean(weight ⊙ ℓ)`. Reduces to a 0-d scalar (the
/// grad needs a scalar cotangent).
pub fn reduce_loss(diff: &Array, weight: Option<&Array>, mae: bool) -> MlxResult<Array> {
    let per = if mae { diff.abs()? } else { diff.square()? };
    match weight {
        None => per.mean(None),
        Some(w) => multiply(&per, w)?.mean(None),
    }
}

/// Turn a row-major `[grid_h, grid_w]` latent weight map (from
/// [`subject_mask_latent_weights`](crate::train::subject_mask::subject_mask_latent_weights)) into an
/// f32 array broadcast to `latent_shape`, whose **last two axes** are the latent `(H, W)` grid (e.g.
/// `[C, F, H, W]` or `[B, C, H, W]`) — the same shape as the cached clean latent, so a trainer that
/// packs its latent into tokens can pack this array with the very same function.
pub fn subject_mask_weight(
    weights: &[f32],
    grid_h: usize,
    grid_w: usize,
    latent_shape: &[i32],
) -> crate::Result<Array> {
    let n = latent_shape.len();
    if n < 2
        || latent_shape[n - 2] as usize != grid_h
        || latent_shape[n - 1] as usize != grid_w
        || weights.len() != grid_h * grid_w
    {
        return Err(crate::Error::Msg(format!(
            "subject mask weight map {grid_h}x{grid_w} ({} values) does not match latent shape \
             {latent_shape:?}",
            weights.len()
        )));
    }
    let mut lead = vec![1i32; n - 2];
    lead.extend([grid_h as i32, grid_w as i32]);
    let w = Array::from_slice(weights, &lead);
    Ok(broadcast_to(&w, latent_shape)?)
}

/// The cache-time entry point every MLX trainer calls once per item (sc-24828): `None` when
/// subject-masked loss is off (no file is read), else the item's latent weight map — its mask
/// cropped with `crop_of(image_w, image_h)` (the trainer's own crop rule, e.g.
/// [`CropBox::center_square`](crate::train::subject_mask::CropBox::center_square)), area-averaged
/// onto the last two axes of `latent_shape` and broadcast to `latent_shape` (see
/// [`subject_mask_weight`]). Refusals (missing / mis-sized / empty mask) name the image.
pub fn item_subject_mask_weight(
    label: &str,
    item: &gen_core::TrainingItem,
    cfg: Option<&gen_core::SubjectMaskLoss>,
    crop_of: impl FnOnce(u32, u32) -> crate::train::subject_mask::CropBox,
    latent_shape: &[i32],
) -> crate::Result<Option<Array>> {
    let Some(cfg) = cfg else {
        return Ok(None);
    };
    let n = latent_shape.len();
    if n < 2 {
        return Err(crate::Error::Msg(format!(
            "{label}: subject mask needs a latent with a spatial grid, got shape {latent_shape:?}"
        )));
    }
    let (grid_h, grid_w) = (latent_shape[n - 2] as usize, latent_shape[n - 1] as usize);
    let weights = crate::train::subject_mask::subject_mask_latent_weights(
        label, item, cfg, crop_of, grid_w, grid_h,
    )?;
    subject_mask_weight(&weights, grid_h, grid_w, latent_shape).map(Some)
}

/// Per-bucket entry point (sc-24828 × sc-2127): the weight map of one cached latent from an item's
/// already-loaded [`PreparedSubjectMask`](crate::train::subject_mask::PreparedSubjectMask) — load
/// it once per item with
/// [`PreparedSubjectMask::load_if_enabled`](crate::train::subject_mask::PreparedSubjectMask::load_if_enabled),
/// then call this once per resolution bucket with that bucket's crop rule and clean-latent shape
/// (last two axes = the latent grid). `None` in, `None` out.
pub fn prepared_subject_mask_weight(
    label: &str,
    mask: Option<&crate::train::subject_mask::PreparedSubjectMask>,
    crop_of: impl FnOnce(u32, u32) -> crate::train::subject_mask::CropBox,
    latent_shape: &[i32],
) -> crate::Result<Option<Array>> {
    let Some(mask) = mask else {
        return Ok(None);
    };
    let n = latent_shape.len();
    if n < 2 {
        return Err(crate::Error::Msg(format!(
            "{label}: subject mask needs a latent with a spatial grid, got shape {latent_shape:?}"
        )));
    }
    let (grid_h, grid_w) = (latent_shape[n - 2] as usize, latent_shape[n - 1] as usize);
    let weights = mask.latent_weights(label, crop_of, grid_w, grid_h)?;
    subject_mask_weight(&weights, grid_h, grid_w, latent_shape).map(Some)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::rc::Rc;

    use mlx_rs::ops::subtract;
    use mlx_rs::transforms::keyed_value_and_grad;
    use mlx_rs::{random, Array};

    use super::*;
    use crate::train::lora::LoraParams;

    /// Host copy of `a`, materialised by a multiply with a dense ones array of its shape (a
    /// broadcast view — and an elementwise op with a scalar — keeps the small strided buffer).
    fn host(a: &Array) -> Vec<f32> {
        let ones = Array::ones::<f32>(a.shape()).unwrap();
        let dense = multiply(a, &ones).unwrap();
        dense.as_slice::<f32>().to_vec()
    }

    /// AC (sc-24828): with background weight 0, latent elements outside the mask contribute zero
    /// loss AND zero gradient w.r.t. the prediction. A 4x4 latent grid, 2 channels; the subject is
    /// the left two columns. The gradient tensor is checked element by element.
    #[test]
    fn zero_background_weight_zeroes_loss_and_gradient_outside_the_mask() {
        let (h, w) = (4usize, 4usize);
        let shape = [2i32, h as i32, w as i32];
        // Left half subject (weight 1), right half background (weight 0).
        let weights: Vec<f32> = (0..h * w)
            .map(|i| if i % w < 2 { 1.0 } else { 0.0 })
            .collect();
        let wmap = subject_mask_weight(&weights, h, w, &shape).unwrap();
        let pred =
            random::normal::<f32>(&shape, None, None, Some(&random::key(1).unwrap())).unwrap();
        let target =
            random::normal::<f32>(&shape, None, None, Some(&random::key(2).unwrap())).unwrap();

        for mae in [false, true] {
            let t = target.clone();
            let wm = wmap.clone();
            let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
                let diff = subtract(&p[&Rc::<str>::from("pred")], &t)?;
                Ok(vec![reduce_loss(&diff, Some(&wm), mae)?])
            };
            let mut params: LoraParams = HashMap::new();
            params.insert(Rc::from("pred"), pred.clone());
            let mut vg = keyed_value_and_grad(loss_fn);
            let (val, grads) = vg(params, 0).unwrap();
            let grad = host(&grads[&Rc::<str>::from("pred")]);
            let (p, tg) = (host(&pred), host(&target));
            let mut expected_loss = 0f32;
            for (i, g) in grad.iter().enumerate() {
                let col = i % w;
                let d = p[i] - tg[i];
                if col >= 2 {
                    assert_eq!(
                        *g, 0.0,
                        "mae={mae}: background element {i} has gradient {g}"
                    );
                } else {
                    assert_ne!(*g, 0.0, "mae={mae}: subject element {i} has no gradient");
                    expected_loss += if mae { d.abs() } else { d * d };
                }
            }
            // Weighted MEAN: divided by every element, not by the subject count.
            expected_loss /= grad.len() as f32;
            let loss = val[0].item::<f32>();
            assert!(
                (loss - expected_loss).abs() < 1e-5,
                "mae={mae}: loss {loss} != masked mean {expected_loss}"
            );
        }
    }

    /// Mask off ⇒ the reduction is exactly the historical `mean(diff²)` / `mean(|diff|)`, and an
    /// all-ones weight map gives the same value (the weighted mean shares the unweighted divisor).
    #[test]
    fn no_weight_is_the_plain_mean_and_ones_weight_matches_it() {
        let shape = [3i32, 2, 5];
        let diff =
            random::normal::<f32>(&shape, None, None, Some(&random::key(3).unwrap())).unwrap();
        let ones = subject_mask_weight(&[1.0; 10], 2, 5, &shape).unwrap();
        for mae in [false, true] {
            let legacy = if mae {
                diff.abs().unwrap().mean(None).unwrap()
            } else {
                diff.square().unwrap().mean(None).unwrap()
            };
            let plain = reduce_loss(&diff, None, mae).unwrap();
            assert_eq!(plain.item::<f32>(), legacy.item::<f32>());
            let weighted = reduce_loss(&diff, Some(&ones), mae).unwrap();
            assert!((weighted.item::<f32>() - legacy.item::<f32>()).abs() < 1e-6);
        }
    }

    #[test]
    fn weight_map_broadcasts_over_leading_axes_and_rejects_mismatch() {
        let w = subject_mask_weight(&[0.0, 1.0, 2.0, 3.0], 2, 2, &[3, 1, 2, 2]).unwrap();
        assert_eq!(w.shape(), &[3, 1, 2, 2]);
        let v = host(&w);
        assert_eq!(&v[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&v[8..], &[0.0, 1.0, 2.0, 3.0]);
        assert!(subject_mask_weight(&[0.0; 4], 2, 2, &[3, 2, 3]).is_err());
        assert!(subject_mask_weight(&[0.0; 3], 2, 2, &[3, 2, 2]).is_err());
    }
}
