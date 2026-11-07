//! F-040 regression guard: the implicit `ScaledDotProductAttentionMask::Causal` mode must match the
//! explicit additive causal mask the text encoders / LM decoders used to materialize host-side per
//! call. The CLIP encoder (mlx-gen-flux), JoyCaption LM (this crate) and prompt-refine LM all rely
//! on this equivalence after dropping their hand-built masks.
//!
//! Two shapes are covered:
//!   * prefill / full encode — `q_len == k_len` (CLIP, decode prefill),
//!   * cached decode — `q_len < k_len`, where MLX aligns the queries to the *last* `q_len` key
//!     positions (bottom-right), exactly reproducing the old `decode_mask(q_len, k_len, offset)`.

use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::random::{normal, seed};
use mlx_rs::Array;

/// The additive `[1, 1, q_len, k_len]` causal mask the decoders used to build by hand: query `r`
/// sits at absolute position `(k_len - q_len) + r` and may attend to keys `j <= pos`.
fn explicit_causal_mask(q_len: i32, k_len: i32) -> Array {
    let offset = (k_len - q_len) as usize;
    let (ql, kl) = (q_len as usize, k_len as usize);
    let mut data = vec![0f32; ql * kl];
    for r in 0..ql {
        let pos = offset + r;
        for j in 0..kl {
            if j > pos {
                data[r * kl + j] = f32::NEG_INFINITY;
            }
        }
    }
    Array::from_slice(&data, &[1, 1, q_len, k_len])
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    max(abs(subtract(a, b).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

fn assert_causal_matches_explicit(q_len: i32, k_len: i32) {
    seed(0).unwrap();
    let (h, d) = (4, 16);
    let scale = (d as f32).powf(-0.5);
    let q = normal::<f32>(&[1, h, q_len, d], None, None, None).unwrap();
    let k = normal::<f32>(&[1, h, k_len, d], None, None, None).unwrap();
    let v = normal::<f32>(&[1, h, k_len, d], None, None, None).unwrap();

    let mask = explicit_causal_mask(q_len, k_len);
    let explicit = scaled_dot_product_attention(&q, &k, &v, scale, &mask, None).unwrap();
    let implicit = scaled_dot_product_attention(
        &q,
        &k,
        &v,
        scale,
        ScaledDotProductAttentionMask::Causal,
        None,
    )
    .unwrap();

    let diff = max_abs_diff(&explicit, &implicit);
    assert!(
        diff < 1e-5,
        "implicit Causal diverged from explicit mask: q_len={q_len} k_len={k_len} max|Δ|={diff}"
    );
}

#[test]
fn causal_mode_matches_explicit_mask_prefill() {
    // CLIP / decode-prefill shape: square mask, offset 0.
    assert_causal_matches_explicit(8, 8);
}

#[test]
fn causal_mode_matches_explicit_mask_cached_decode() {
    // JoyCaption / prompt-refine cached-decode shape: 3 new queries against 8 cached keys (offset 5).
    assert_causal_matches_explicit(3, 8);
}
