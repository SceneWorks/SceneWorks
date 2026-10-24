//! Unit invariants for the SC-18319 primitive and the fused QKV projection.
//!
//! Per-family fixtures at real shapes live in `tests/qkv_family_fixtures.rs`; what is here is the
//! primitive's own arithmetic, the fusion safety predicate, and the two controls that stop the
//! bit-exactness assertions from being false greens:
//!
//! * [`fused_and_unfused_are_bit_exact_and_the_toggle_is_not_inert`] — the equivalent of
//!   `nn.rs`'s `compile_glue_helpers_are_bit_exact_and_modulate_dtype_policy` negative control. A
//!   fusion that never engaged would pass a bit-exactness test trivially, so the toggle is proved to
//!   change *which tensors are read*.
//! * [`a_live_adapter_forces_the_split_backing_and_is_never_dropped`] — the designed-against failure.
//! * [`fusion_is_opt_in_so_an_eligible_triple_is_split_by_default`] — the compiled-in default,
//!   pinned. Fusion is opt-in, so **every** test here that means to exercise the fused arm holds a
//!   [`FusedQkvGuard`] over the whole load (construction *and* every `quantize` / `cast_weights`,
//!   both of which re-pack). A test that forgot would not fail quietly: each fused arm asserts
//!   `fusion_engaged()` or `packed_shape()`, which observe the representation directly.

use super::*;
use mlx_rs::ops::array_eq;

fn seq(shape: &[i32], scale: f32, offset: f32) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * scale + offset).sin())
        .collect();
    Array::from_slice(&data, shape)
}

fn exact(a: &Array, b: &Array) -> bool {
    a.shape() == b.shape() && array_eq(a, b, None).unwrap().item::<bool>()
}

fn f32s(a: &Array) -> Vec<f32> {
    a.reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

// ── the rotation kernels ─────────────────────────────────────────────────────────────────────

/// Knob 2 — the two rotation conventions are genuinely different and each is pinned by a quarter
/// turn, not by agreement with the other. Under `AdjacentPair` a quarter turn maps `(a,b) → (−b,a)`
/// on *neighbouring* lanes; under `RotateHalf` lane `i` pairs with lane `i + half`, so the same
/// input yields `[−3, −4, 1, 2]`. A primitive that silently used one for the other is a running
/// model with garbage output and no shape error — exactly the Mage/Ideogram hazard.
#[test]
fn the_two_rotation_conventions_are_pinned_apart() {
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);

    let cos = Array::from_slice(&[0.0f32, 0.0], &[1, 2]);
    let sin = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);
    let adj = apply_rope(
        &x,
        RopeTables::new(&cos, &sin),
        RopeStyle::AdjacentPair,
        RotationAxes::TokenMajor,
        None,
        RopeDtype::RestoreInput,
    )
    .unwrap();
    assert_eq!(f32s(&adj), vec![-2.0, 1.0, -4.0, 3.0]);

    // `RotateHalf` consumes a FULL head_dim table.
    let cos = Array::from_slice(&[0.0f32; 4], &[1, 4]);
    let sin = Array::from_slice(&[1.0f32; 4], &[1, 4]);
    let half = apply_rope(
        &x,
        RopeTables::new(&cos, &sin),
        RopeStyle::RotateHalf,
        RotationAxes::TokenMajor,
        None,
        RopeDtype::RestoreInput,
    )
    .unwrap();
    assert_eq!(f32s(&half), vec![-3.0, -4.0, 1.0, 2.0]);

    assert_ne!(
        f32s(&adj),
        f32s(&half),
        "the two styles must not collapse onto one another"
    );
}

/// The rotation is an identity when the table is `(cos, sin) = (1, 0)` — the control that makes
/// every other rotation assertion meaningful.
#[test]
fn an_identity_table_leaves_the_stream_untouched() {
    let x = seq(&[2, 3, 2, 8], 0.13, 0.4);
    let cos = Array::from_slice(&[1.0f32; 3 * 4], &[3, 4]);
    let sin = Array::from_slice(&[0.0f32; 3 * 4], &[3, 4]);
    let out = apply_rope(
        &x,
        RopeTables::new(&cos, &sin),
        RopeStyle::AdjacentPair,
        RotationAxes::TokenMajor,
        None,
        RopeDtype::RestoreInput,
    )
    .unwrap();
    assert!(exact(&out, &x));
}

/// Knob 4 — partial RoPE rotates only the leading `rot_dims` channels and passes the tail through
/// **bit-identically**. Asserted against a hand-built expectation rather than against a full
/// rotation, so a `rot_dims` that was quietly ignored fails here.
#[test]
fn partial_rope_rotates_the_head_and_passes_the_tail_through() {
    // head_dim 8, rotate the leading 4 channels only.
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 1, 8]);
    let cos = Array::from_slice(&[0.0f32, 0.0], &[1, 2]);
    let sin = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);
    let out = apply_rope(
        &x,
        RopeTables::new(&cos, &sin),
        RopeStyle::AdjacentPair,
        RotationAxes::TokenMajor,
        Some(4),
        RopeDtype::RestoreInput,
    )
    .unwrap();
    // leading 4 quarter-turned as adjacent pairs; trailing 4 untouched.
    assert_eq!(f32s(&out), vec![-2.0, 1.0, -4.0, 3.0, 5.0, 6.0, 7.0, 8.0]);

    // `rot_dims == head_dim` must equal the unrestricted call, and `rot_dims == 0` a no-op.
    let full = apply_rope(
        &x,
        RopeTables::new(
            &Array::from_slice(&[0.0f32; 4], &[1, 4]),
            &Array::from_slice(&[1.0f32; 4], &[1, 4]),
        ),
        RopeStyle::AdjacentPair,
        RotationAxes::TokenMajor,
        Some(8),
        RopeDtype::RestoreInput,
    )
    .unwrap();
    assert_eq!(
        f32s(&full),
        vec![-2.0, 1.0, -4.0, 3.0, -6.0, 5.0, -8.0, 7.0]
    );
    let none = apply_rope(
        &x,
        RopeTables::new(&cos, &sin),
        RopeStyle::AdjacentPair,
        RotationAxes::TokenMajor,
        Some(0),
        RopeDtype::RestoreInput,
    )
    .unwrap();
    assert!(exact(&none, &x));
}

/// A wrong-length table is the classic silent-garbage RoPE bug (every token reads row 0). It must be
/// an error, not a broadcast.
#[test]
fn a_wrong_length_table_is_rejected() {
    let x = seq(&[1, 4, 2, 8], 0.1, 0.0);
    let cos = Array::from_slice(&[1.0f32; 8], &[2, 4]); // 2 tokens, stream has 4
    let err = apply_rope(
        &x,
        RopeTables::new(&cos, &cos),
        RopeStyle::AdjacentPair,
        RotationAxes::TokenMajor,
        None,
        RopeDtype::RestoreInput,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("elements but the stream needs"), "{err}");
}

/// A **per-batch** table (Ideogram's 3-D MRoPE is `[B, L, head_dim]`, one table per sample) must be
/// consumed per sample, not broadcast from sample 0. Two samples with deliberately different tables
/// must produce different rows; a permissive broadcast would make them identical.
#[test]
fn a_per_batch_rope_table_is_consumed_per_sample() {
    let (b, s, h, d) = (2, 3, 2, 8);
    let x = seq(&[b, h, s, d], 0.017, 0.2);
    // Sample 0 gets the identity rotation, sample 1 a quarter turn.
    let mut cos = vec![1.0f32; (b * s * d) as usize];
    let mut sin = vec![0.0f32; (b * s * d) as usize];
    for i in (s * d) as usize..(b * s * d) as usize {
        cos[i] = 0.0;
        sin[i] = 1.0;
    }
    let cos = Array::from_slice(&cos, &[b, s, d]);
    let sin = Array::from_slice(&sin, &[b, s, d]);
    let out = apply_rope(
        &x,
        RopeTables::new(&cos, &sin),
        RopeStyle::RotateHalf,
        RotationAxes::HeadMajor,
        None,
        RopeDtype::RestoreInput,
    )
    .unwrap();
    let (xs, os) = (f32s(&x), f32s(&out));
    let per_sample = (h * s * d) as usize;
    assert_eq!(
        &xs[0..per_sample],
        &os[0..per_sample],
        "sample 0's identity table must leave it untouched"
    );
    assert_ne!(
        &xs[per_sample..],
        &os[per_sample..],
        "sample 1's table must actually be applied — a broadcast from sample 0 would no-op it"
    );
}

/// The [`RotationAxes`] field is a *bit-exactness* field, not a semantic one: rotating before the
/// SDPA transpose and rotating after it must produce the identical BHSD result, because both the
/// rotation and the QK-norm act on the last axis and broadcast over the head axis. Pinning it means
/// a family migrated with the "wrong" axis order is still numerically right — and that the field
/// exists only to keep each migration a provable no-op.
#[test]
fn the_two_rotation_axis_orders_agree_bit_for_bit() {
    let (b, s, h, d) = (2, 5, 3, 8);
    let packed = seq(&[b, s, 3 * h * d], 0.011, 0.2);
    let nq = seq(&[d], 0.31, 1.0);
    let nk = seq(&[d], 0.17, 0.5);
    let cos = seq(&[s, d / 2], 0.07, 0.0);
    let sin = seq(&[s, d / 2], 0.05, 0.9);

    let build = |axes: RotationAxes| {
        let spec = AttnPrepSpec::new(h, d)
            .with_qk_norm(QkNormSpec::per_head(&nq, &nk, 1e-6))
            .with_rope(RopeSpec {
                style: RopeStyle::AdjacentPair,
                q: Some(RopeTables::new(&cos, &sin)),
                k: Some(RopeTables::new(&cos, &sin)),
                ..RopeSpec::default()
            })
            .with_rotation_axes(axes);
        prepare(QkvSource::Packed(&packed), &spec).unwrap()
    };
    let token_major = build(RotationAxes::TokenMajor);
    let head_major = build(RotationAxes::HeadMajor);
    assert!(exact(&token_major.q, &head_major.q));
    assert!(exact(&token_major.k, &head_major.k));
    assert!(exact(&token_major.v, &head_major.v));
}

// ── the prologue ─────────────────────────────────────────────────────────────────────────────

/// Knobs 5/6 — the q and k streams are rotated independently, and a `None` table leaves its stream
/// *untouched*. Mage rotates the image q/k and never the text stream; Anima's conditioner and LTX's
/// cross-modal path give q and k different tables entirely.
#[test]
fn rope_targets_and_separate_tables_are_honoured() {
    let (b, s, h, d) = (1, 4, 2, 8);
    let packed = seq(&[b, s, 3 * h * d], 0.019, 0.1);
    let cos_q = seq(&[s, d / 2], 0.03, 0.0);
    let sin_q = seq(&[s, d / 2], 0.09, 0.3);
    let cos_k = seq(&[s, d / 2], 0.13, 0.7);
    let sin_k = seq(&[s, d / 2], 0.02, 1.1);

    let bare = prepare(
        QkvSource::Packed(&packed),
        &AttnPrepSpec::new(h, d).with_qk_norm(QkNormSpec::none()),
    )
    .unwrap();

    // q only (Mage's "one stream only").
    let q_only = prepare(
        QkvSource::Packed(&packed),
        &AttnPrepSpec::new(h, d)
            .with_qk_norm(QkNormSpec::none())
            .with_rope(RopeSpec {
                style: RopeStyle::AdjacentPair,
                q: Some(RopeTables::new(&cos_q, &sin_q)),
                k: None,
                ..RopeSpec::default()
            }),
    )
    .unwrap();
    assert!(
        !exact(&q_only.q, &bare.q),
        "the query stream must actually be rotated"
    );
    assert!(
        exact(&q_only.k, &bare.k),
        "a `None` key table must leave the key stream untouched"
    );

    // separate q/k tables — the two streams must diverge from a shared-table run.
    let shared = prepare(
        QkvSource::Packed(&packed),
        &AttnPrepSpec::new(h, d)
            .with_qk_norm(QkNormSpec::none())
            .with_rope(RopeSpec {
                style: RopeStyle::AdjacentPair,
                q: Some(RopeTables::new(&cos_q, &sin_q)),
                k: Some(RopeTables::new(&cos_q, &sin_q)),
                ..RopeSpec::default()
            }),
    )
    .unwrap();
    let split_tables = prepare(
        QkvSource::Packed(&packed),
        &AttnPrepSpec::new(h, d)
            .with_qk_norm(QkNormSpec::none())
            .with_rope(RopeSpec {
                style: RopeStyle::AdjacentPair,
                q: Some(RopeTables::new(&cos_q, &sin_q)),
                k: Some(RopeTables::new(&cos_k, &sin_k)),
                ..RopeSpec::default()
            }),
    )
    .unwrap();
    assert!(exact(&shared.q, &split_tables.q), "q is table-independent");
    assert!(
        !exact(&shared.k, &split_tables.k),
        "a separate k table must actually be used"
    );
}

/// Knob 1 — the two QK-norm placements are not interchangeable. A per-head norm reduces over
/// `head_dim`; a full-dim pre-split norm reduces over `heads · head_dim`. Feeding the same
/// (broadcast) weight through both must give different numbers, or the placement knob is inert.
#[test]
fn the_qk_norm_placements_are_not_interchangeable() {
    let (b, s, h, d) = (1, 3, 4, 8);
    let packed = seq(&[b, s, 3 * h * d], 0.023, 0.4);
    let per_head_w = seq(&[d], 0.4, 0.2);
    // The same values tiled across the flat axis, so only the *reduction width* differs.
    let flat: Vec<f32> = (0..h).flat_map(|_| f32s(&per_head_w).into_iter()).collect();
    let full_w = Array::from_slice(&flat, &[h * d]);

    let per_head = prepare(
        QkvSource::Packed(&packed),
        &AttnPrepSpec::new(h, d).with_qk_norm(QkNormSpec::per_head(&per_head_w, &per_head_w, 1e-6)),
    )
    .unwrap();
    let full_dim = prepare(
        QkvSource::Packed(&packed),
        &AttnPrepSpec::new(h, d)
            .with_qk_norm(QkNormSpec::full_dim_pre_split(&full_w, &full_w, 1e-6)),
    )
    .unwrap();
    assert_eq!(per_head.q.shape(), full_dim.q.shape());
    assert!(
        !exact(&per_head.q, &full_dim.q),
        "the placement knob must change the result — otherwise LTX/Wan's deviation is not modelled"
    );
    assert!(
        exact(&per_head.v, &full_dim.v),
        "the value stream is never normalized under either placement"
    );
}

/// Knob 10 — GQA repeat-KV runs **after** the rotation, and repeating before it is a different
/// (wrong) model. Boogu is 28 q heads / 7 kv heads; the ratio here is the same 4.
#[test]
fn gqa_repeats_kv_after_the_rotation_and_that_ordering_matters() {
    let (b, s, h, kv, d) = (1, 4, 8, 2, 8);
    let q = seq(&[b, s, h * d], 0.017, 0.1);
    let k = seq(&[b, s, kv * d], 0.013, 0.6);
    let v = seq(&[b, s, kv * d], 0.011, 1.2);
    let cos = seq(&[s, d / 2], 0.07, 0.0);
    let sin = seq(&[s, d / 2], 0.05, 0.4);

    let out = prepare(
        QkvSource::Separate {
            q: &q,
            k: &k,
            v: &v,
        },
        &AttnPrepSpec::new(h, d)
            .with_kv_heads(kv)
            .with_qk_norm(QkNormSpec::none())
            .with_rope(RopeSpec {
                style: RopeStyle::AdjacentPair,
                q: Some(RopeTables::new(&cos, &sin)),
                k: Some(RopeTables::new(&cos, &sin)),
                ..RopeSpec::default()
            }),
    )
    .unwrap();
    assert_eq!(out.q.shape(), [b, h, s, d]);
    assert_eq!(out.k.shape(), [b, h, s, d], "kv heads repeated up to h");
    assert_eq!(out.v.shape(), [b, h, s, d]);

    // Repeat-then-rotate would rotate `groups` copies of the same table row — same shape, different
    // numbers only if the rotation is head-dependent, which it is not. What IS observable is that
    // the repeat is a genuine broadcast of each kv head across its group.
    let k_rows = f32s(&out.k);
    let per_head = (s * d) as usize;
    for g in 1..(h / kv) as usize {
        assert_eq!(
            &k_rows[0..per_head],
            &k_rows[g * per_head..(g + 1) * per_head],
            "each kv head must be repeated contiguously across its query group"
        );
    }
    assert_ne!(
        &k_rows[0..per_head],
        &k_rows[(h / kv) as usize * per_head..((h / kv) as usize + 1) * per_head],
        "adjacent groups must come from DIFFERENT kv heads"
    );
}

/// Knob 11 — the stream order is load-bearing and both orders are live in this tree.
#[test]
fn the_stream_order_actually_swaps_the_joint_sequence() {
    let mk = |scale: f32, tokens: i32| QkvHeads {
        q: seq(&[1, 2, tokens, 4], scale, 0.0),
        k: seq(&[1, 2, tokens, 4], scale, 0.5),
        v: seq(&[1, 2, tokens, 4], scale, 1.0),
    };
    let img = mk(0.11, 3);
    let txt = mk(0.29, 2);
    let image_first = StreamOrder::ImageFirst.join(&img, &txt).unwrap();
    let text_first = StreamOrder::TextFirst.join(&img, &txt).unwrap();
    assert_eq!(image_first.q.shape(), [1, 2, 5, 4]);
    assert!(!exact(&image_first.q, &text_first.q));
    assert_eq!(StreamOrder::ImageFirst.lead_tokens(3, 2), 3);
    assert_eq!(StreamOrder::TextFirst.lead_tokens(3, 2), 2);
}

/// Knob 8 — concat-then-RoPE (Chroma) and RoPE-then-concat (FLUX.1) are expressed as a call-order
/// choice over the same two building blocks, and they agree when the joined table is the
/// concatenation of the two per-stream tables. That agreement is what makes it a *choice* rather
/// than two different models.
#[test]
fn concat_then_rope_matches_rope_then_concat_on_a_concatenated_table() {
    let (b, h, d) = (1, 2, 8);
    let (si, st) = (3, 2);
    let mk = |tokens: i32, scale: f32| QkvHeads {
        q: seq(&[b, h, tokens, d], scale, 0.1),
        k: seq(&[b, h, tokens, d], scale, 0.7),
        v: seq(&[b, h, tokens, d], scale, 1.3),
    };
    let img = mk(si, 0.017);
    let txt = mk(st, 0.031);
    let cos_i = seq(&[si, d / 2], 0.05, 0.0);
    let sin_i = seq(&[si, d / 2], 0.03, 0.2);
    let cos_t = seq(&[st, d / 2], 0.09, 0.6);
    let sin_t = seq(&[st, d / 2], 0.07, 1.1);

    let rot = |x: &Array, c: &Array, s: &Array| {
        apply_rope(
            x,
            RopeTables::new(c, s),
            RopeStyle::AdjacentPair,
            RotationAxes::HeadMajor,
            None,
            RopeDtype::RestoreInput,
        )
        .unwrap()
    };

    // RoPE-then-concat.
    let a = StreamOrder::TextFirst
        .join(
            &QkvHeads {
                q: rot(&img.q, &cos_i, &sin_i),
                k: rot(&img.k, &cos_i, &sin_i),
                v: img.v.clone(),
            },
            &QkvHeads {
                q: rot(&txt.q, &cos_t, &sin_t),
                k: rot(&txt.k, &cos_t, &sin_t),
                v: txt.v.clone(),
            },
        )
        .unwrap();

    // Concat-then-RoPE over the concatenated `[txt, img]` table.
    let joined = StreamOrder::TextFirst.join(&img, &txt).unwrap();
    let cos = concatenate_axis(&[&cos_t, &cos_i], 0).unwrap();
    let sin = concatenate_axis(&[&sin_t, &sin_i], 0).unwrap();
    let b2 = QkvHeads {
        q: rot(&joined.q, &cos, &sin),
        k: rot(&joined.k, &cos, &sin),
        v: joined.v,
    };
    assert!(exact(&a.q, &b2.q));
    assert!(exact(&a.k, &b2.k));
    assert!(exact(&a.v, &b2.v));
}

/// A packed source whose width does not match the declared geometry must be an error, not a
/// mis-split — a mis-split produces a *running* model with three garbage streams.
#[test]
fn a_mis_sized_packed_projection_is_rejected() {
    let packed = seq(&[1, 4, 47], 0.1, 0.0);
    let err = prepare(
        QkvSource::Packed(&packed),
        &AttnPrepSpec::new(2, 8).with_qk_norm(QkNormSpec::none()),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("wide, expected"), "{err}");
}

// ── knob 2's third arm and the pre-broadcast table (LTX) ─────────────────────────────────────

/// Knob 2 — [`RopeStyle::HalvesPaired`] (LTX's `apply_split_rotary_emb`) is the SAME rotation as
/// [`RopeStyle::RotateHalf`], just driven by a half-width table through one fused `rope_rotate`
/// instead of the tiled-table `x·cos + rotate_half(x)·sin` expression. Pinning the agreement is what
/// makes them two arms of one knob rather than two different models — and pinning it against a
/// *tiled* `RotateHalf` table (rather than against itself) is what makes the assertion mean
/// something.
#[test]
fn halves_paired_and_rotate_half_are_the_same_rotation() {
    let (b, h, s, d) = (2, 3, 5, 8);
    let x = seq(&[b, h, s, d], 0.013, 0.3);
    let cos_half = seq(&[s, d / 2], 0.07, 0.1);
    let sin_half = seq(&[s, d / 2], 0.05, 0.8);
    // The full-width table `RotateHalf` consumes is this one tiled twice on the last axis.
    let cos_full = concatenate_axis(&[&cos_half, &cos_half], 1).unwrap();
    let sin_full = concatenate_axis(&[&sin_half, &sin_half], 1).unwrap();

    let paired = apply_rope(
        &x,
        RopeTables::new(&cos_half, &sin_half),
        RopeStyle::HalvesPaired,
        RotationAxes::HeadMajor,
        None,
        RopeDtype::RestoreInput,
    )
    .unwrap();
    let tiled = apply_rope(
        &x,
        RopeTables::new(&cos_full, &sin_full),
        RopeStyle::RotateHalf,
        RotationAxes::HeadMajor,
        None,
        RopeDtype::RestoreInput,
    )
    .unwrap();
    assert_eq!(paired.shape(), tiled.shape());
    // Different op sequences over f32, so agreement is to rounding, not bit-for-bit — the point is
    // that the two arms describe one rotation.
    let a = f32s(&paired);
    let c = f32s(&tiled);
    for (p, t) in a.iter().zip(c.iter()) {
        assert!((p - t).abs() < 1e-5, "{p} vs {t}");
    }

    // …and it is a real rotation, not a pass-through.
    assert_ne!(a, f32s(&x));
}

/// LTX ships `[B, H, T, head_dim/2]` tables — per batch **and** per head, which none of the rank-≤3
/// layouts can express. They must be consumed verbatim (no reshape), and a rank-4 table of the wrong
/// geometry must still be an error rather than a bad broadcast.
#[test]
fn a_pre_broadcast_rank4_table_is_used_verbatim_and_a_wrong_one_is_rejected() {
    let (b, h, s, d) = (2, 2, 3, 8);
    let x = seq(&[b, h, s, d], 0.011, 0.2);
    // Head 0 of every sample gets the identity; head 1 a quarter turn. A layout that collapsed the
    // head axis would give both heads the same treatment.
    let half = (d / 2) as usize;
    let n = (b * h * s) as usize * half;
    let mut cos = vec![1.0f32; n];
    let mut sin = vec![0.0f32; n];
    for sample in 0..b as usize {
        let base = (sample * h as usize + 1) * s as usize * half;
        for i in base..base + s as usize * half {
            cos[i] = 0.0;
            sin[i] = 1.0;
        }
    }
    let cos = Array::from_slice(&cos, &[b, h, s, d / 2]);
    let sin = Array::from_slice(&sin, &[b, h, s, d / 2]);
    let out = apply_rope(
        &x,
        RopeTables::new(&cos, &sin),
        RopeStyle::HalvesPaired,
        RotationAxes::HeadMajor,
        None,
        RopeDtype::RestoreInput,
    )
    .unwrap();
    let (xs, os) = (f32s(&x), f32s(&out));
    let per_head = (s * d) as usize;
    for sample in 0..b as usize {
        let h0 = (sample * h as usize) * per_head;
        let h1 = h0 + per_head;
        assert_eq!(
            &xs[h0..h0 + per_head],
            &os[h0..h0 + per_head],
            "sample {sample} head 0's identity table must leave it untouched"
        );
        assert_ne!(
            &xs[h1..h1 + per_head],
            &os[h1..h1 + per_head],
            "sample {sample} head 1's own table must be applied — a per-token-only broadcast \
             would give both heads the identity"
        );
    }

    // A rank-4 table that does not line up is an error.
    let bad = seq(&[b, h + 1, s, d / 2], 0.01, 0.0);
    let err = apply_rope(
        &x,
        RopeTables::new(&bad, &bad),
        RopeStyle::HalvesPaired,
        RotationAxes::HeadMajor,
        None,
        RopeDtype::RestoreInput,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("does not broadcast"), "{err}");
}

/// The primitive always runs the per-head QK-norm on `[B, S, H, D]` and only then transposes, while
/// SD3 and Chroma used to transpose first and norm on `[B, H, S, D]`. Both reduce over the last
/// axis and broadcast over the head axis, so the migration is only a no-op if the two orders agree
/// **bit-for-bit** at the real dtypes — a fused RMSNorm kernel reading a transposed view is not
/// obliged to. Pin it, at bf16 as well as f32, rather than assuming it.
#[test]
fn norming_before_and_after_the_sdpa_transpose_agrees_bit_for_bit() {
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let (b, s, h, d) = (2, 7, 6, 64);
        let x = seq(&[b, s, h, d], 0.0009, 0.3).as_dtype(dtype).unwrap();
        let w = seq(&[d], 0.31, 1.0).as_dtype(dtype).unwrap();

        let pre = mlx_rs::fast::rms_norm(&x, &w, 1e-6)
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap();
        let post =
            mlx_rs::fast::rms_norm(x.transpose_axes(&[0, 2, 1, 3]).unwrap(), &w, 1e-6).unwrap();
        assert!(
            exact(&pre, &post),
            "{dtype:?}: the QK-norm's position relative to the SDPA transpose is NOT free — \
             SD3/Chroma cannot be migrated without an explicit placement knob"
        );
    }
}

// ── the fused projection ─────────────────────────────────────────────────────────────────────

/// Compare two arrays that were produced by **differently shaped matmuls over the same numbers** —
/// the fused arm's one `[.., 3*out]` GEMM against the unfused arm's three `[.., out]` GEMMs.
///
/// Bit-exactness is asserted whenever it holds, and it does hold on the self-hosted macOS 26.2 NAX
/// build. It is **not** guaranteed in general: MLX picks its Metal GEMM kernel (and therefore its
/// tile shape and accumulation order) from `(M, N, K)`, and the two arms differ in `N` by
/// construction. Hosted CI (`macos-15-arm64`, `MACOSX_DEPLOYMENT_TARGET=15.0`) compiles a different
/// kernel set from the dev host's 26.2 NAX build, and float addition is not associative, so the two
/// arms can land a few ULP apart there.
///
/// The bound below is what separates the two failure modes, which is the whole point of this helper:
///
/// * **kernel variance** perturbs the last bits — a relative error at ULP scale;
/// * **a slicing bug** (a wrong row range, a stride misread, an off-by-one partition) returns
///   *different rows of the weight*, which is an O(1) relative error, not O(ulp).
///
/// So a genuine defect still fails this assertion loudly, and the message reports the magnitude so
/// the next reader can tell the two apart without re-deriving anything.
///
/// The byte-level claim this feature actually rests on — that packing concatenates on the output
/// axis and `unfuse` inverts it exactly — is asserted directly on the **tensors** by
/// [`unfusing_recovers_the_three_bases_exactly`], with no matmul in the way.
#[track_caller]
fn agree(a: &Array, b: &Array, what: &str) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    assert_eq!(a.dtype(), b.dtype(), "{what}: dtype");
    if array_eq(a, b, None).unwrap().item::<bool>() {
        return;
    }
    let af = a.as_dtype(Dtype::Float32).unwrap();
    let bf = b.as_dtype(Dtype::Float32).unwrap();
    let diff = mlx_rs::ops::abs(mlx_rs::ops::subtract(&af, &bf).unwrap()).unwrap();
    let max_abs = mlx_rs::ops::max(&diff, None).unwrap().item::<f32>();
    let scale = mlx_rs::ops::max(mlx_rs::ops::abs(&af).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .max(f32::MIN_POSITIVE);
    let rel = max_abs / scale;
    // One ULP of bf16 is ~2^-8 relative; of f32 ~2^-23. Allow a handful of accumulation steps'
    // worth, and nothing remotely close to "read the wrong rows".
    let tol = if a.dtype() == Dtype::Bfloat16 {
        1e-2
    } else {
        1e-5
    };
    assert!(
        rel <= tol,
        "{what}: the fused and unfused arms disagree by rel={rel:e} (max|delta|={max_abs:e}, \
         scale={scale:e}), far beyond Metal GEMM kernel variance — this is a SLICING defect (a \
         wrong row range or a stride misread), not accumulation order"
    );
}

fn dense_parts(inp: i32, q_out: i32, kv_out: i32, bias: bool) -> [AdaptableLinear; 3] {
    let mk = |out: i32, scale: f32| {
        AdaptableLinear::dense(
            seq(&[out, inp], scale, 0.3),
            bias.then(|| seq(&[out], scale * 3.0, 0.1)),
        )
    };
    [mk(q_out, 0.0013), mk(kv_out, 0.0017), mk(kv_out, 0.0019)]
}

fn dense_triple(inp: i32, q_out: i32, kv_out: i32, bias: bool) -> FusedQkvProjection {
    let [q, k, v] = dense_parts(inp, q_out, kv_out, bias);
    FusedQkvProjection::new(q, k, v)
}

/// **The bit-exactness invariant AND its negative control**, in the shape `nn.rs`'s compiled-glue
/// guard established:
///
/// * run the A/B twice — once unfused-vs-unfused (the eager-vs-eager sanity arm, asserted exactly)
///   and once fused-vs-unfused (the real invariant, asserted via [`agree`]: bit-exact where the
///   Metal GEMM kernel is shared across the two output widths, bounded at ULP scale otherwise);
/// * then prove the toggle is **not inert**. A bit-exactness test alone passes trivially if the
///   fused arm never engaged, which is the single most likely way this ships broken.
///
/// Three independent non-inertness proofs are asserted, because this is the assertion that carries
/// the whole story:
///
/// 1. the two arms hold **different representations** — [`FusedQkvProjection::packed_shape`] is
///    `Some([q+k+v, in])` on one and `None` on the other, and `forward` is a match on exactly that;
/// 2. the fused arm records the P4 activation receipt
///    (`FUSED_ATTENTION_PRIMITIVES` → `Applied`) and the unfused arm records `Fallback`, read back
///    off a real [`diagnostics`] scope — the same receipt the P6 matrix requires instead of
///    inferring the toggle from a flag or from timing;
/// 3. the packed matrix is a **single** `[q+k+v, in]` array, so the fused arm cannot be reading the
///    three bases: they no longer exist on that object.
#[test]
fn fused_and_unfused_are_bit_exact_and_the_toggle_is_not_inert() {
    use crate::diagnostics::{
        self, DiagnosticCounter, ToggleDisposition, FUSED_ATTENTION_PRIMITIVES,
    };

    let x = seq(&[2, 6, 32], 0.007, 0.2);
    for bias in [false, true] {
        let scope = diagnostics::begin_request("sc-18319", "qkv").expect("no scope is active");

        // Build both arms from the SAME three bases, flipping only the thread toggle — the
        // toggle-based A/B in one process.
        let unfused = {
            let _g = FusedQkvGuard::set(false);
            let [q, k, v] = dense_parts(32, 64, 64, bias);
            FusedQkvProjection::new(q, k, v)
        };
        let fused = {
            let _g = FusedQkvGuard::set(true);
            let [q, k, v] = dense_parts(32, 64, 64, bias);
            FusedQkvProjection::new(q, k, v)
        };
        let control = {
            let _g = FusedQkvGuard::set(false);
            let [q, k, v] = dense_parts(32, 64, 64, bias);
            FusedQkvProjection::new(q, k, v)
        };

        // Proof 1 — the arms are genuinely different objects.
        assert!(!unfused.fusion_engaged(), "bias={bias}");
        assert_eq!(unfused.refusal(), Some(&NoFusion::Disabled));
        assert_eq!(unfused.packed_shape(), None);
        assert!(fused.fusion_engaged(), "bias={bias}");
        assert_eq!(fused.refusal(), None);
        // Proof 3 — one packed `[q+k+v, in]` matrix, not three bases.
        assert_eq!(fused.packed_shape(), Some(vec![64 + 64 + 64, 32]));

        // The sanity arm: unfused == unfused.
        let (rq, rk, rv) = unfused.forward(&x).unwrap();
        let (cq, ck, cv) = control.forward(&x).unwrap();
        assert!(
            exact(&rq, &cq) && exact(&rk, &ck) && exact(&rv, &cv),
            "bias={bias}: control"
        );

        // The invariant: fused == unfused. Bit-exact where the Metal GEMM kernel is the same for
        // both output widths; see `agree` for why that is asserted with a bound rather than
        // assumed, and what a real slicing defect looks like against it.
        let (fq, fk, fv) = fused.forward(&x).unwrap();
        agree(&rq, &fq, &format!("bias={bias}: q"));
        agree(&rk, &fk, &format!("bias={bias}: k"));
        agree(&rv, &fv, &format!("bias={bias}: v"));

        // Proof 2 — the activation receipt distinguishes the arms.
        let report = scope.finish();
        let count = |want: ToggleDisposition| -> u64 {
            report
                .counters
                .iter()
                .filter_map(|c| match c {
                    DiagnosticCounter::Toggle {
                        toggle,
                        disposition,
                        count,
                    } if *toggle == FUSED_ATTENTION_PRIMITIVES && *disposition == want => {
                        Some(*count)
                    }
                    _ => None,
                })
                .sum()
        };
        assert_eq!(
            count(ToggleDisposition::Applied),
            1,
            "bias={bias}: exactly the fused arm must report the P4 toggle Applied"
        );
        assert_eq!(
            count(ToggleDisposition::Fallback),
            2,
            "bias={bias}: both unfused arms must report the P4 toggle Fallback"
        );
    }
}

/// **The compiled-in default, pinned: fusion is OPT-IN** (Michael, 2026-08-16).
///
/// The decision this asserts is a reproducibility one, not a taste one: the fused arm's GEMM has a
/// different `N`, MLX picks its Metal tile shape (hence accumulation order) from `(M, N, K)`, and
/// off the self-hosted NAX build the two arms can land a few ULP apart — which a chaos sampler turns
/// into a different image at the same seed. See [`set_fused_qkv`].
///
/// The assertion has to survive the obvious way of faking it — "it is not fused because it *cannot*
/// be fused" — so the test carries its own **eligibility control**: the identical triple is built
/// once *with* an explicit opt-in and shown to pack. Only then does the no-opt-in arm's split
/// backing mean what it claims.
///
/// Three further arms, each of which would break if the default flipped back to on:
///
/// * the refusal **names the toggle** ([`NoFusion::Disabled`]) rather than a shape reason, so a
///   future relaxation of the safety predicate cannot make this pass for the wrong reason;
/// * the default survives the load path's *re-packs* — `quantize` calls
///   [`FusedQkvProjection::repack`], which re-reads the toggle, so a default-off model does not
///   quietly fuse itself the first time a tier is applied;
/// * [`FusedQkvGuard`] restores the **default**, not `true`, on drop — which is what keeps every
///   other test in this file from leaking an opt-in into this one.
#[test]
fn fusion_is_opt_in_so_an_eligible_triple_is_split_by_default() {
    // The harness is single-threaded (`.cargo/config.toml` forces `RUST_TEST_THREADS=1`, so libtest
    // runs every test on this one thread) and nothing in this crate calls `set_fused_qkv` outside a
    // `FusedQkvGuard`. The value observed here is therefore the compiled-in default itself; if some
    // future test leaks an opt-in, it fails HERE with a specific message instead of silently
    // invalidating the claim below.
    assert!(
        !fused_qkv(),
        "the compiled-in default must be OFF (and no test may leak an opt-in — use FusedQkvGuard, \
         never a bare set_fused_qkv)"
    );

    // The eligibility control: this exact triple DOES pack when a caller opts in. Without it, the
    // split backing below could be an unsafe-shape refusal wearing the default's clothes.
    let eligible = {
        let _g = FusedQkvGuard::set(true);
        dense_triple(32, 64, 64, true)
    };
    assert!(
        eligible.fusion_engaged(),
        "the control triple must be fusable, or this test proves nothing: {:?}",
        eligible.refusal()
    );
    assert_eq!(eligible.packed_shape(), Some(vec![192, 32]));
    assert_eq!(eligible.refusal(), None);

    // …and with NO opt-in anywhere, the same triple is split.
    let by_default = dense_triple(32, 64, 64, true);
    assert!(
        !by_default.fusion_engaged(),
        "fusion is opt-in: a triple built with no explicit set_fused_qkv(true) must stay split"
    );
    assert_eq!(by_default.packed_shape(), None);
    assert_eq!(
        by_default.refusal(),
        Some(&NoFusion::Disabled),
        "the refusal must name the toggle, not a shape reason"
    );

    // The default survives the load path's re-packs — `quantize` re-reads the toggle.
    let mut quantized = dense_triple(128, 128, 64, true);
    quantized.quantize(8, Some(64)).unwrap();
    assert!(!quantized.fusion_engaged());
    assert_eq!(
        quantized.refusal(),
        Some(&NoFusion::Disabled),
        "quantizing must not silently opt a default-off model into fusion"
    );

    // Split is not "broken": the default arm still computes the model's output, and it is exactly
    // the concatenation of the three separate forwards.
    let x = seq(&[1, 3, 32], 0.013, 0.4);
    let (q, k, v) = by_default.forward(&x).unwrap();
    assert_eq!(q.shape(), [1, 3, 64]);
    let reference = concatenate_axis(&[&q, &k, &v], 2).unwrap();
    assert!(exact(&by_default.forward_packed(&x).unwrap(), &reference));

    // The guard restores the DEFAULT on drop, which is the mechanism the isolation above relies on.
    {
        let _g = FusedQkvGuard::set(true);
        assert!(
            fused_qkv(),
            "the guard must engage its opt-in for its scope"
        );
    }
    assert!(
        !fused_qkv(),
        "the guard must restore the opt-in default (off), not leave the thread fused"
    );
}

/// `forward_packed` (the single `[.., q+k+v]` array the primitive's [`QkvSource::Packed`] arm
/// consumes) must be the concatenation of the three separate forwards, fused or not.
#[test]
fn forward_packed_matches_the_concatenated_separate_forwards() {
    let x = seq(&[1, 5, 32], 0.011, 0.5);
    let unfused = {
        let _g = FusedQkvGuard::set(false);
        dense_triple(32, 64, 32, true)
    };
    let fused = {
        let _g = FusedQkvGuard::set(true);
        dense_triple(32, 64, 32, true)
    };
    let (q, k, v) = unfused.forward(&x).unwrap();
    let reference = concatenate_axis(&[&q, &k, &v], 2).unwrap();
    // The unfused arm concatenates its own three forwards, so this half IS bit-exact.
    assert!(exact(&unfused.forward_packed(&x).unwrap(), &reference));
    assert!(fused.fusion_engaged());
    // The fused arm runs ONE wider GEMM — see `agree`.
    agree(
        &fused.forward_packed(&x).unwrap(),
        &reference,
        "forward_packed",
    );
}

/// The packed matrix is the SOLE retained copy of the three bases, so `unfuse` must recover them
/// **exactly** — otherwise fusing would be a lossy, one-way transformation and installing a LoRA
/// after load would silently change the base model. Asserted for dense and for Q4/Q8, where the
/// row split has to carry `weight`, `scales`, `biases` and the dense `bias` together.
#[test]
fn unfusing_recovers_the_three_bases_exactly() {
    let x = seq(&[1, 4, 128], 0.005, 0.35);
    for quant in [None, Some(4), Some(8)] {
        // Fusion is opt-in, and the toggle is read at construction AND at every `repack` — which
        // `quantize` (below) and the explicit `repack` at the end of this loop both run. The guard
        // therefore spans the whole body, not just the constructor: scoping it to `dense_triple`
        // alone would leave the later re-packs refusing with `NoFusion::Disabled`.
        let _g = FusedQkvGuard::set(true);
        let mut fused = dense_triple(128, 128, 64, true);
        if let Some(bits) = quant {
            fused.quantize(bits, Some(64)).unwrap();
        }
        assert!(fused.fusion_engaged(), "quant={quant:?}");
        let packed = fused.forward(&x).unwrap();

        fused.unfuse().unwrap();
        assert_eq!(fused.packed_shape(), None);

        // **The claim, asserted on the TENSORS.** `try_pack` concatenates on the output axis and
        // `split_parts` slices at exactly those row ranges, so every recovered base must be
        // byte-identical to the one that went in — dense weight and bias, and on a packed tier the
        // codes, the scales, the affine biases and the declared `(bits, group)` too. This is the
        // property the effective-bits argument rests on, and unlike an output comparison it has no
        // matmul in the way, so it is exact on every host.
        let reference = dense_parts(128, 128, 64, true);
        let mut reference: Vec<AdaptableLinear> = reference.into_iter().collect();
        if let Some(bits) = quant {
            for r in reference.iter_mut() {
                r.quantize(bits, Some(64)).unwrap();
            }
        }
        for (i, name) in ["q", "k", "v"].iter().enumerate() {
            let got = fused.part_mut(match i {
                0 => QkvPart::Q,
                1 => QkvPart::K,
                _ => QkvPart::V,
            });
            let got = got.unwrap();
            let want = &reference[i];
            assert_eq!(
                got.base_shape(),
                want.base_shape(),
                "quant={quant:?}: {name} shape"
            );
            assert_eq!(
                got.is_quantized(),
                want.is_quantized(),
                "quant={quant:?}: {name} tier"
            );
            match (got.dense_weight(), want.dense_weight()) {
                (Some((gw, gb)), Some((ww, wb))) => {
                    assert!(exact(gw, ww), "quant={quant:?}: {name} dense weight");
                    match (gb, wb) {
                        (Some(gb), Some(wb)) => {
                            assert!(exact(gb, wb), "quant={quant:?}: {name} dense bias")
                        }
                        (None, None) => {}
                        _ => panic!("quant={quant:?}: {name} bias presence"),
                    }
                }
                (None, None) => {
                    let (gw, gs, gbi, gb, ggrp, gbits) = got.quantized_params().unwrap();
                    let (ww, ws, wbi, wb, wgrp, wbits) = want.quantized_params().unwrap();
                    assert!(exact(gw, ww), "quant={quant:?}: {name} codes");
                    assert!(exact(gs, ws), "quant={quant:?}: {name} scales");
                    assert!(exact(gbi, wbi), "quant={quant:?}: {name} affine biases");
                    assert!(
                        exact(gb.unwrap(), wb.unwrap()),
                        "quant={quant:?}: {name} dense bias"
                    );
                    assert_eq!((ggrp, gbits), (wgrp, wbits), "quant={quant:?}: {name} spec");
                }
                _ => panic!("quant={quant:?}: {name} changed tier across the unfuse"),
            }
        }

        // The forwards agree too — see `agree` for why that comparison is bounded rather than
        // bit-exact while the tensor comparison above is not.
        let split = fused.forward(&x).unwrap();
        agree(&packed.0, &split.0, &format!("quant={quant:?}: q"));
        agree(&packed.1, &split.1, &format!("quant={quant:?}: k"));
        agree(&packed.2, &split.2, &format!("quant={quant:?}: v"));

        // …and re-packing returns to the identical numbers, so the round trip is closed. This arm
        // IS bit-exact by construction: both sides are the same packed representation, hence the
        // same matmul shape.
        fused.repack();
        assert!(fused.fusion_engaged(), "quant={quant:?}: repack");
        let again = fused.forward(&x).unwrap();
        assert!(exact(&packed.0, &again.0), "quant={quant:?}: q round trip");
        assert!(exact(&packed.1, &again.1), "quant={quant:?}: k round trip");
        assert!(exact(&packed.2, &again.2), "quant={quant:?}: v round trip");
    }
}

/// **The designed-against failure.** P4 is the first read-side fusion that must consult
/// `AdaptableLinear::adapters()`; no precedent guard existed. Four claims:
///
/// 1. reaching the parts to install an adapter **unpacks** — a packed matrix and a live adapter
///    cannot coexist, so there is no ordering that strands a residual behind a stale pack;
/// 2. a subsequent `repack` refuses ([`NoFusion::AdaptersInstalled`]) rather than dropping it;
/// 3. the adapted output is **bit-identical** to the same adapter on three bare `AdaptableLinear`s
///    — the LoRA is applied, not merely "not crashed";
/// 4. the adapted output actually **differs** from the unadapted one, so claim 3 is not the vacuous
///    "both sides dropped it".
///
/// Claim 4 is the one that catches a silent drop: without it, a fusion that ignored the adapter
/// stack would satisfy claims 1–3 by returning the base output twice.
#[test]
fn a_live_adapter_forces_the_split_backing_and_is_never_dropped() {
    use crate::adapters::Adapter;

    let x = seq(&[1, 4, 32], 0.009, 0.15);
    let lora = || Adapter::Lora {
        a: seq(&[32, 4], 0.021, 0.3),
        b: seq(&[4, 64], 0.037, 0.8),
        scale: 1.0,
    };

    let _g = FusedQkvGuard::set(true);
    let mut proj = dense_triple(32, 64, 64, true);
    assert!(proj.fusion_engaged());
    let unadapted = proj.forward(&x).unwrap().0;

    // Claim 1 — reaching the parts unpacks.
    proj.parts_mut().unwrap().0.set_adapters(vec![lora()]);
    assert!(
        !proj.fusion_engaged(),
        "parts_mut must unpack: a live adapter behind a packed matrix is the failure mode"
    );

    // Claim 2 — repack refuses while the adapter is live, even with the toggle on.
    proj.repack();
    assert_eq!(
        proj.refusal(),
        Some(&NoFusion::AdaptersInstalled),
        "a live adapter must refuse the pack, not silently drop the residual"
    );
    assert!(!proj.fusion_engaged());

    // Claim 3 — identical to the same LoRA on a bare AdaptableLinear.
    let adapted = proj.forward(&x).unwrap().0;
    let mut bare = dense_parts(32, 64, 64, true)[0].clone();
    bare.set_adapters(vec![lora()]);
    assert!(
        exact(&adapted, &bare.forward(&x).unwrap()),
        "the LoRA-active output must match the unfused reference exactly"
    );

    // Claim 4 — and it is genuinely different from the unadapted output.
    assert!(
        !exact(&adapted, &unadapted),
        "the LoRA residual was dropped — this is the failure the guard exists to prevent"
    );

    // A *disabled* adapter (scale 0) is byte-identically a no-op, so it must NOT cost the fusion.
    proj.parts_mut()
        .unwrap()
        .0
        .set_adapters(vec![Adapter::Lora {
            a: seq(&[32, 4], 0.021, 0.3),
            b: seq(&[4, 64], 0.037, 0.8),
            scale: 0.0,
        }]);
    proj.repack();
    assert!(
        proj.fusion_engaged(),
        "an adapter installed at scale 0 is a no-op and must not disable fusion"
    );
    assert!(exact(&proj.forward(&x).unwrap().0, &unadapted));
}

/// Quantized fusion must be **bit-exact** and must not change effective bits: the packed matrix
/// carries the same `bits`/`group_size` and each row keeps the scales it was quantized with, because
/// the concatenation is on the OUTPUT axis while the affine groups run along the input axis.
#[test]
fn quantized_fusion_is_bit_exact_and_preserves_effective_bits() {
    for bits in [4, 8] {
        let x = seq(&[1, 4, 128], 0.003, 0.25);
        // The toggle is read at construction AND at every `repack` — and `quantize` re-packs — so
        // each arm holds its guard across both steps.
        let unfused = {
            let _g = FusedQkvGuard::set(false);
            let mut p = dense_triple(128, 128, 128, true);
            p.quantize(bits, Some(64)).unwrap();
            p
        };
        let fused = {
            let _g = FusedQkvGuard::set(true);
            let mut p = dense_triple(128, 128, 128, true);
            p.quantize(bits, Some(64)).unwrap();
            p
        };
        assert!(!unfused.fusion_engaged());
        assert!(
            fused.fusion_engaged(),
            "group-aligned Q{bits} must pack: {:?}",
            fused.refusal()
        );

        let off = unfused.forward(&x).unwrap();
        let on = fused.forward(&x).unwrap();
        agree(&off.0, &on.0, &format!("Q{bits}: q"));
        agree(&off.1, &on.1, &format!("Q{bits}: k"));
        agree(&off.2, &on.2, &format!("Q{bits}: v"));

        // Effective bits: the packed matrix declares the same (bits, group) as its parts, and its
        // logical shape has exactly the summed rows — no re-quantization, no regrouping.
        assert_eq!(fused.packed_shape(), Some(vec![384, 128]));
        assert_eq!(
            fused.packed_quant_spec(),
            Some((bits, 64)),
            "the packed matrix must declare its parts' bits and group size verbatim"
        );
    }
}

/// **Boogu, at its real numbers.** `mlx-gen-boogu/src/quant.rs` sets `GROUP_SIZE = 32` — explicit
/// and non-default, because the DiT hidden size 3360 = 32·105 is not divisible by 64. With 28 query
/// heads / 7 kv heads at head_dim 120, the kv projections are `out = 840` and `840 % 32 = 8`, so
/// they are **not** output-axis group-aligned and the pack is refused by construction rather than
/// papered over. The query projection alone (`out = 3360 = 105·32`) would align — the triple still
/// cannot pack, because a triple packs or it does not.
///
/// Boogu also can never share a packed matrix with another family: every other family here resolves
/// to `DEFAULT_GROUP_SIZE = 64`, and nothing in [`FusedQkvProjection`] builds a cross-family matrix
/// at all.
#[test]
fn boogus_group_32_kv_projections_are_refused_not_papered_over() {
    const GROUP: i32 = 32;

    // Full-size divisibility, asserted as arithmetic so the constants cannot silently drift.
    assert_eq!(3360 % GROUP, 0, "boogu q out is group-aligned");
    assert_eq!(840 % GROUP, 8, "boogu kv out is NOT group-aligned");
    assert_ne!(120 % GROUP, 0, "boogu head_dim is not group-aligned either");
    assert_eq!(
        3360 % 64,
        32,
        "…and 3360 is why boogu cannot use the default 64"
    );

    // And the predicate refuses a triple built at exactly those out widths. `in` is kept at a
    // group-aligned 320 so the ONLY reason for refusal is the kv `out`.
    let _g = FusedQkvGuard::set(true);
    let mk = |out: i32, scale: f32| {
        let mut l = AdaptableLinear::dense(seq(&[out, 320], scale, 0.2), None);
        l.quantize(8, Some(GROUP)).unwrap();
        l
    };
    let proj = FusedQkvProjection::new(mk(3360, 0.001), mk(840, 0.002), mk(840, 0.003));
    assert_eq!(
        proj.refusal(),
        Some(&NoFusion::OutFeaturesNotGroupAligned {
            out: 840,
            group: GROUP
        }),
        "boogu's kv projections must be refused with their real numbers"
    );
    assert!(!proj.fusion_engaged());
    assert_eq!(proj.packed_shape(), None);

    // Refused means *correct*, not broken: the unfused path still produces the model's output.
    let x = seq(&[1, 2, 320], 0.005, 0.4);
    let (q, k, v) = proj.forward(&x).unwrap();
    assert_eq!(q.shape(), [1, 2, 3360]);
    assert_eq!(k.shape(), [1, 2, 840]);
    assert_eq!(v.shape(), [1, 2, 840]);
}

/// The remaining refusal reasons, each pinned so a future relaxation is a deliberate edit.
#[test]
fn the_fusion_safety_predicate_refuses_every_unsafe_shape() {
    let _g = FusedQkvGuard::set(true);

    // Mixed dense/quantized bases.
    let dense = AdaptableLinear::dense(seq(&[64, 64], 0.01, 0.1), None);
    let mut quant = AdaptableLinear::dense(seq(&[64, 64], 0.01, 0.2), None);
    quant.quantize(8, Some(64)).unwrap();
    let mixed = FusedQkvProjection::new(
        dense,
        quant.clone(),
        AdaptableLinear::dense(seq(&[64, 64], 0.01, 0.3), None),
    );
    assert_eq!(mixed.refusal(), Some(&NoFusion::MixedBases));

    // Differing bits.
    let mut q4 = AdaptableLinear::dense(seq(&[64, 64], 0.01, 0.4), None);
    q4.quantize(4, Some(64)).unwrap();
    let mismatch = FusedQkvProjection::new(quant.clone(), q4, quant.clone());
    assert!(matches!(
        mismatch.refusal(),
        Some(NoFusion::QuantMismatch { .. })
    ));

    // NOTE: `NoFusion::InFeaturesNotGroupAligned` is deliberately NOT exercised here. It is
    // unreachable by construction — MLX's `quantize` rejects a misaligned last dimension outright
    // (`[quantize] The last dimension … needs to be divisible by the quantization group size`), and
    // a pre-quantized base derives `in` from the scales grid as `scales_cols · group_size`. A test
    // that had to fake a state neither path can produce would be asserting the test harness, not the
    // guard; the guard's own doc records why it is kept.

    // Differing input widths — the three cannot share one matmul at all.
    let wide = AdaptableLinear::dense(seq(&[64, 128], 0.01, 0.5), None);
    let narrow = AdaptableLinear::dense(seq(&[64, 64], 0.01, 0.6), None);
    let widths = FusedQkvProjection::new(wide, narrow.clone(), narrow.clone());
    assert_eq!(widths.refusal(), Some(&NoFusion::InputWidthMismatch));

    // Bias on some parts only.
    let biased = AdaptableLinear::dense(seq(&[64, 64], 0.01, 0.7), Some(seq(&[64], 0.02, 0.1)));
    let bias_mix = FusedQkvProjection::new(biased, narrow.clone(), narrow);
    assert_eq!(bias_mix.refusal(), Some(&NoFusion::BiasMismatch));
}
// ── the exemption table ──────────────────────────────────────────────────────────────────────

/// The story permits leaving a family on its native path **provided the deviation is documented**.
/// [`EXEMPTIONS`] is that documentation; this asserts it is present and specific, so an exemption
/// cannot degrade into a bare name.
#[test]
fn every_structural_exemption_carries_a_specific_justification() {
    assert!(
        EXEMPTIONS.len() >= 11,
        "the survey found 11 structural exemptions, the table lists {}",
        EXEMPTIONS.len()
    );
    for e in EXEMPTIONS {
        assert!(
            e.path.contains("mlx-gen"),
            "an exemption must name the crate/file it exempts: {:?}",
            e.path
        );
        assert!(
            e.deviation.len() > 80,
            "exemption {} has no substantive deviation recorded",
            e.path
        );
    }
    let mut paths: Vec<&str> = EXEMPTIONS.iter().map(|e| e.path).collect();
    paths.sort_unstable();
    let before = paths.len();
    paths.dedup();
    assert_eq!(before, paths.len(), "duplicate exemption entries");
}

// ── the newly adopted families (SC-18319 P4) ─────────────────────────────────────────────────

/// One adopted family's **production** projection geometry, as surveyed from its own config and
/// offline packer. `inp`/`out` are the weight's real `[out, in]`; `group` is the family's own
/// quantization group size.
struct FamilyGeometry {
    family: &'static str,
    /// Which triple — a family with two joint streams contributes two rows.
    stream: &'static str,
    inp: i32,
    out: i32,
    group: i32,
    /// Whether the family's q/k/v carry a dense bias (SD3/Mage/Chroma do; SANA/Anima do not).
    bias: bool,
}

/// Every triple this story moved onto [`FusedQkvProjection`], at the width its checkpoint actually
/// ships. The arithmetic in the table's own doc is asserted by
/// [`every_adopted_family_geometry_is_group_aligned`] rather than trusted.
const ADOPTED: &[FamilyGeometry] = &[
    // SD3.5 Large: hidden = 38 heads x 64 = 2432, group 64 (`mlx-gen-sd3/src/transformer.rs`).
    FamilyGeometry {
        family: "sd3-large",
        stream: "image",
        inp: 2432,
        out: 2432,
        group: 64,
        bias: true,
    },
    FamilyGeometry {
        family: "sd3-large",
        stream: "text",
        inp: 2432,
        out: 2432,
        group: 64,
        bias: true,
    },
    // SD3.5 Medium: hidden = 24 x 64 = 1536; `attn2` (MMDiT-X dual attention) is the same width.
    FamilyGeometry {
        family: "sd3-medium",
        stream: "attn2",
        inp: 1536,
        out: 1536,
        group: 64,
        bias: true,
    },
    // Mage-Flow: hidden 3072 = 24 heads x 128, group 64 (`mlx-gen-mage/src/quant.rs`).
    FamilyGeometry {
        family: "mage",
        stream: "image",
        inp: 3072,
        out: 3072,
        group: 64,
        bias: true,
    },
    FamilyGeometry {
        family: "mage",
        stream: "text",
        inp: 3072,
        out: 3072,
        group: 64,
        bias: true,
    },
    // Chroma: inner 3072 = 24 x 128, group 64. Double blocks carry two streams, single blocks one.
    FamilyGeometry {
        family: "chroma",
        stream: "double-image",
        inp: 3072,
        out: 3072,
        group: 64,
        bias: true,
    },
    FamilyGeometry {
        family: "chroma",
        stream: "single",
        inp: 3072,
        out: 3072,
        group: 64,
        bias: true,
    },
    // Anima DiT self-attention: hidden 2048 = 16 x 128, group 64, bias-free.
    FamilyGeometry {
        family: "anima-dit",
        stream: "self_attn",
        inp: 2048,
        out: 2048,
        group: 64,
        bias: false,
    },
    // Anima conditioner self-attention: model_dim 1024 = 16 x 64, bias-free (and always dense).
    FamilyGeometry {
        family: "anima-cond",
        stream: "self_attn",
        inp: 1024,
        out: 1024,
        group: 64,
        bias: false,
    },
    // SANA 1600M attn1: inner 2240 = 70 heads x 32, group 64 (`mlx-gen-sana/src/quant.rs`).
    // 2240 = 64 * 35 -- aligned on both axes despite the unusual head_dim.
    FamilyGeometry {
        family: "sana",
        stream: "attn1",
        inp: 2240,
        out: 2240,
        group: 64,
        bias: false,
    },
];

/// The group-alignment arithmetic each family's adoption rests on, checked rather than asserted in
/// prose — including the **fused** row count, which is the number `try_pack` will actually see and
/// the one a family could get wrong while each individual projection still looked fine.
///
/// The negative control is Boogu, whose kv `out = 840` is deliberately *not* group-aligned at its
/// own `GROUP_SIZE = 32`; it must fail the very predicate every row above passes.
#[test]
fn every_adopted_family_geometry_is_group_aligned() {
    for g in ADOPTED {
        let who = format!("{}/{}", g.family, g.stream);
        assert_eq!(g.inp % g.group, 0, "{who}: in {} % {}", g.inp, g.group);
        assert_eq!(g.out % g.group, 0, "{who}: out {} % {}", g.out, g.group);
        assert_eq!(
            (3 * g.out) % g.group,
            0,
            "{who}: the FUSED out {} must stay group-aligned",
            3 * g.out
        );
    }
    // Negative control -- Boogu's real numbers, which must NOT satisfy the same predicate.
    assert_ne!(840 % 32, 0, "boogu kv out 840 is the misaligned case");
    assert_ne!(120 % 32, 0, "boogu head_dim 120 is the misaligned case");
}

/// **The per-family bit-exactness fixture**, at each adopted family's production projection width,
/// run at **both** bf16 and f32.
///
/// Three claims per row, per dtype:
///
/// 1. agreement via [`agree`] — bit-exact where the Metal GEMM kernel is shared across the two
///    output widths, bounded far below a slicing defect otherwise — between the fused and the
///    unfused arm, built from byte-identical bases with only [`set_fused_qkv`] flipped between them;
/// 2. **dtype equality** -- a fusion that silently promoted would still compare equal under a
///    tolerance-based check and under `array_eq`'s own broadcasting, so the dtype is asserted
///    separately;
/// 3. a **negative control** that the arms are genuinely different objects: the fused arm holds one
///    `[3*out, in]` packed matrix (`packed_shape()`) and the unfused arm holds none. Without it a
///    fusion that never engaged passes claims 1 and 2 trivially -- which is the single most likely
///    way this ships broken.
#[test]
fn every_adopted_family_is_bit_exact_at_production_geometry_in_both_dtypes() {
    for g in ADOPTED {
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let who = format!("{}/{} @ {dtype:?}", g.family, g.stream);
            let x = seq(&[1, 4, g.inp], 0.0007, 0.2).as_dtype(dtype).unwrap();

            let build = |on: bool| {
                let _g = FusedQkvGuard::set(on);
                let [q, k, v] = dense_parts(g.inp, g.out, g.out, g.bias);
                let mut p = FusedQkvProjection::new(q, k, v);
                p.cast_weights(dtype).unwrap();
                p
            };
            let fused = build(true);
            let unfused = build(false);

            // Claim 3 first -- if this fails the other two are meaningless.
            assert_eq!(
                fused.packed_shape(),
                Some(vec![3 * g.out, g.inp]),
                "{who}: the fused arm must hold ONE packed matrix"
            );
            assert_eq!(
                unfused.packed_shape(),
                None,
                "{who}: the unfused arm must hold no packed matrix"
            );
            assert_eq!(unfused.refusal(), Some(&NoFusion::Disabled), "{who}");

            // Claim 1 -- exact, per projection and over the concatenated form.
            let (fq, fk, fv) = fused.forward(&x).unwrap();
            let (uq, uk, uv) = unfused.forward(&x).unwrap();
            for (name, a, b) in [("q", &fq, &uq), ("k", &fk, &uk), ("v", &fv, &uv)] {
                // Claim 1 -- see `agree`: exact where the Metal GEMM kernel is shared across the
                // two output widths, bounded far below a slicing defect otherwise.
                agree(a, b, &format!("{who}: {name}"));
                // Claim 2 -- dtype, separately from value. `agree` checks the two arms against each
                // other; this pins them to the family's own dtype so a silent promotion is caught.
                assert_eq!(a.dtype(), dtype, "{who}: {name} left the family's dtype");
            }
            agree(
                &fused.forward_packed(&x).unwrap(),
                &unfused.forward_packed(&x).unwrap(),
                &format!("{who}: forward_packed"),
            );
        }
    }
}

/// The same fixture on a **quantized** base at each family's own group size, at both compute
/// dtypes. The extra claim is the effective-bits receipt: packing must carry `bits`/`group_size`
/// verbatim, never re-quantize and never regroup.
///
/// The negative control here is the packed spec itself -- if `packed_quant_spec()` came back `None`
/// the arm was not packed and the equality proves nothing. (`anima-cond` is excluded: its
/// conditioner is dense on every tier, `convert.rs`'s predicate skips `llm_adapter.*` outright.)
#[test]
fn every_adopted_family_is_bit_exact_when_quantized_at_its_own_group_size() {
    for g in ADOPTED.iter().filter(|g| g.family != "anima-cond") {
        for bits in [4, 8] {
            let who = format!("{}/{} @ q{bits}", g.family, g.stream);
            let x = seq(&[1, 4, g.inp], 0.0007, 0.2)
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let build = |on: bool| {
                let _guard = FusedQkvGuard::set(on);
                let [q, k, v] = dense_parts(g.inp, g.out, g.out, g.bias);
                let mut p = FusedQkvProjection::new(q, k, v);
                p.quantize(bits, Some(g.group)).unwrap();
                p
            };
            let fused = build(true);
            let unfused = build(false);

            assert_eq!(
                fused.packed_quant_spec(),
                Some((bits, g.group)),
                "{who}: effective bits/group must survive the pack verbatim"
            );
            assert_eq!(unfused.packed_quant_spec(), None, "{who}");

            let (fq, fk, fv) = fused.forward(&x).unwrap();
            let (uq, uk, uv) = unfused.forward(&x).unwrap();
            for (name, a, b) in [("q", &fq, &uq), ("k", &fk, &uk), ("v", &fv, &uv)] {
                agree(a, b, &format!("{who}: {name}"));
            }
        }
    }
}

/// **The LoRA-active fixture**, per adopted family, at production width and at both dtypes.
///
/// Two assertions, and the second is the load-bearing one:
///
/// 1. with a live adapter installed, the output is **identical** with fusion on and with fusion off
///    -- because a live adapter refuses the pack, so "fusion on" resolves to the same split backing;
/// 2. and that output **differs from the unadapted output**. Without this second assertion a fusion
///    that silently dropped the residual would satisfy the first one perfectly: both arms would
///    return the base output, equal to each other and wrong.
#[test]
fn a_live_lora_survives_fusion_at_every_adopted_family_geometry() {
    use crate::adapters::Adapter;

    for g in ADOPTED {
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let who = format!("{}/{} @ {dtype:?}", g.family, g.stream);
            let x = seq(&[1, 4, g.inp], 0.0007, 0.2).as_dtype(dtype).unwrap();
            let lora = || Adapter::Lora {
                a: seq(&[g.inp, 4], 0.0021, 0.3).as_dtype(dtype).unwrap(),
                b: seq(&[4, g.out], 0.0037, 0.8).as_dtype(dtype).unwrap(),
                scale: 1.0,
            };

            let build = |on: bool| {
                let _guard = FusedQkvGuard::set(on);
                let [q, k, v] = dense_parts(g.inp, g.out, g.out, g.bias);
                let mut p = FusedQkvProjection::new(q, k, v);
                p.cast_weights(dtype).unwrap();
                p
            };

            // The unadapted reference, taken while the pack IS engaged.
            let fused_clean = build(true);
            assert!(fused_clean.fusion_engaged(), "{who}");
            let unadapted = fused_clean.forward(&x).unwrap().0;

            let adapted = |on: bool| {
                let _guard = FusedQkvGuard::set(on);
                let mut p = build(on);
                p.part_mut(QkvPart::Q).unwrap().set_adapters(vec![lora()]);
                // Re-attempt the pack: it must refuse while the adapter is live.
                p.repack();
                // Whichever refusal fires, the pack MUST be refused. With the toggle on that
                // refusal is the adapter guard itself; with it off the toggle short-circuits first,
                // which is why the two arms can be compared at all.
                assert_eq!(
                    p.refusal(),
                    Some(if on {
                        &NoFusion::AdaptersInstalled
                    } else {
                        &NoFusion::Disabled
                    }),
                    "{who}: a live adapter must refuse the pack, not silently drop the residual"
                );
                assert!(!p.fusion_engaged(), "{who}");
                p.forward(&x).unwrap().0
            };

            let on = adapted(true);
            let off = adapted(false);

            // 1 -- fusion-on and fusion-off agree exactly.
            assert!(
                exact(&on, &off),
                "{who}: LoRA output differs across the toggle"
            );
            assert_eq!(on.dtype(), off.dtype(), "{who}: LoRA output dtype");
            assert_eq!(
                on.dtype(),
                dtype,
                "{who}: LoRA output left the family's dtype"
            );

            // 2 -- and it is genuinely NOT the unadapted output. Load-bearing: without this a
            // fusion that dropped the residual would pass assertion 1.
            assert!(
                !exact(&on, &unadapted),
                "{who}: the LoRA residual was dropped -- both arms returned the base output"
            );
        }
    }
}

// ── the probe/mutate split ───────────────────────────────────────────────────────────────────

/// A three-hop host tree with the same shape every adopted family has -- transformer -> block ->
/// attention -- so the probe contract can be exercised end to end without a family's weights.
///
/// `forward_probe` models the one thing a family can get wrong: an intermediate node that does NOT
/// override `adaptable_facts` falls back to the trait default, which resolves through
/// `adaptable_mut` and unfuses everything below it.
struct ProbeAttention {
    qkv: FusedQkvProjection,
    out: AdaptableLinear,
}

struct ProbeBlock {
    attn: ProbeAttention,
    /// Whether this block forwards the probe to its attention. `false` reproduces the bug.
    forward_probe: bool,
}

struct ProbeTransformer {
    block: ProbeBlock,
    forward_probe: bool,
}

impl crate::adapters::AdaptableHost for ProbeAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["to_q"] => self.qkv.part_mut(QkvPart::Q).ok(),
            ["to_k"] => self.qkv.part_mut(QkvPart::K).ok(),
            ["to_v"] => self.qkv.part_mut(QkvPart::V).ok(),
            ["to_out"] => Some(&mut self.out),
            _ => None,
        }
    }

    fn adaptable_facts(&mut self, path: &[&str]) -> Option<crate::adapters::LinearFacts> {
        match path {
            ["to_q"] => Some(self.qkv.part_facts(QkvPart::Q)),
            ["to_k"] => Some(self.qkv.part_facts(QkvPart::K)),
            ["to_v"] => Some(self.qkv.part_facts(QkvPart::V)),
            _ => self
                .adaptable_mut(path)
                .map(|l| crate::adapters::LinearFacts::of(l)),
        }
    }
}

impl crate::adapters::AdaptableHost for ProbeBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_facts(&mut self, path: &[&str]) -> Option<crate::adapters::LinearFacts> {
        if !self.forward_probe {
            // The bug: fall through to the trait default's `adaptable_mut` delegation.
            return self
                .adaptable_mut(path)
                .map(|l| crate::adapters::LinearFacts::of(l));
        }
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_facts(rest),
            _ => None,
        }
    }
}

impl crate::adapters::AdaptableHost for ProbeTransformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["blocks", "0", rest @ ..] => self.block.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_facts(&mut self, path: &[&str]) -> Option<crate::adapters::LinearFacts> {
        if !self.forward_probe {
            return self
                .adaptable_mut(path)
                .map(|l| crate::adapters::LinearFacts::of(l));
        }
        match path {
            ["blocks", "0", rest @ ..] => self.block.adaptable_facts(rest),
            _ => None,
        }
    }
}

fn probe_host(block_forwards: bool, top_forwards: bool) -> ProbeTransformer {
    let _g = FusedQkvGuard::set(true);
    ProbeTransformer {
        block: ProbeBlock {
            attn: ProbeAttention {
                qkv: dense_triple(64, 64, 64, true),
                out: dense_parts(64, 64, 64, true)[0].clone(),
            },
            forward_probe: block_forwards,
        },
        forward_probe: top_forwards,
    }
}

/// **A probe must not unfuse** -- the whole point of the SC-18319 split, and the assertion without
/// which the rest of this change is unverified.
///
/// Claims:
///
/// 1. a full-depth `adaptable_facts` walk over every q/k/v path leaves the projection **packed**
///    (`packed_shape()` observes the representation directly, so this cannot be faked);
/// 2. the facts a probe returns are the same ones the split representation reports, so the probe is
///    invisible to its caller -- a probe that answered "no such path" while packed would trivially
///    satisfy claim 1;
/// 3. **the negative controls**: a genuine mutation (`adaptable_mut`) DOES unfuse, and an
///    intermediate hop that forgets to forward the probe DOES unfuse. Without those two, claim 1
///    would pass on a host that was never packed in the first place, or on a test that never
///    reached the fused leaf.
#[test]
fn a_probe_does_not_unfuse_but_a_mutation_and_a_missing_forward_do() {
    use crate::adapters::AdaptableHost;

    let paths = [
        ["blocks", "0", "attn", "to_q"],
        ["blocks", "0", "attn", "to_k"],
        ["blocks", "0", "attn", "to_v"],
    ];

    // Claim 1 -- probe every path, at full depth, and stay packed throughout.
    let mut host = probe_host(true, true);
    let packed_before = host.block.attn.qkv.packed_shape();
    assert_eq!(
        packed_before,
        Some(vec![192, 64]),
        "the fixture must start packed or the test proves nothing"
    );
    for p in &paths {
        let facts = host.adaptable_facts(p).expect("path resolves");
        // Claim 2 -- the probe answers with the SPLIT projection's own facts.
        assert_eq!(facts.base_shape, vec![64, 64], "{p:?}");
        assert!(!facts.is_quantized, "{p:?}");
        assert!(!facts.has_live_adapters, "{p:?}");
        assert_eq!(facts.adapter_count, 0, "{p:?}");
        assert_eq!(facts.bias_shape, Some(vec![64]), "{p:?}");
        assert_eq!(
            host.block.attn.qkv.packed_shape(),
            packed_before,
            "probing {p:?} unfused the projection"
        );
    }
    // A probe for a path that does not exist must also leave it alone.
    assert!(host
        .adaptable_facts(&["blocks", "0", "attn", "nope"])
        .is_none());
    assert_eq!(host.block.attn.qkv.packed_shape(), packed_before);

    // Claim 3a -- negative control: the MUTATION half does unfuse, so claim 1 is not vacuous.
    let mut host = probe_host(true, true);
    assert!(host.block.attn.qkv.fusion_engaged());
    assert!(host
        .adaptable_mut(&["blocks", "0", "attn", "to_q"])
        .is_some());
    assert!(
        !host.block.attn.qkv.fusion_engaged(),
        "adaptable_mut must unfuse -- it is the half that may install an adapter"
    );
    assert_eq!(host.block.attn.qkv.refusal(), Some(&NoFusion::Unfused));

    // Claim 3b -- negative control: a block that forgets to forward the probe unfuses, which is
    // exactly the trap the trait doc warns about.
    let mut host = probe_host(false, true);
    assert!(host.block.attn.qkv.fusion_engaged());
    let _ = host.adaptable_facts(&["blocks", "0", "attn", "to_q"]);
    assert!(
        !host.block.attn.qkv.fusion_engaged(),
        "a missing intermediate `adaptable_facts` MUST fall through to `adaptable_mut` -- if this \
         assertion fails the test cannot detect a family that forgot to forward the probe"
    );

    // Claim 3c -- and the same trap one hop higher up.
    let mut host = probe_host(true, false);
    assert!(host.block.attn.qkv.fusion_engaged());
    let _ = host.adaptable_facts(&["blocks", "0", "attn", "to_q"]);
    assert!(!host.block.attn.qkv.fusion_engaged());
}

/// A probe must report a live adapter honestly -- the second half of the question the probe surface
/// exists to answer ("does this projection currently carry adapters?").
///
/// The fixture is asserted **packed on arrival** before anything touches it. Fusion is opt-in, so
/// without that the "installing unfused it" assertion below would pass on a host that was never
/// packed -- the exact false green this suite exists to avoid.
#[test]
fn a_probe_reports_adapter_state_without_touching_the_representation() {
    use crate::adapters::{AdaptableHost, Adapter};

    let mut host = probe_host(true, true);
    assert_eq!(
        host.block.attn.qkv.packed_shape(),
        Some(vec![192, 64]),
        "the fixture must start packed, or 'installing unfused it' proves nothing"
    );
    // Installing goes through the mutation half, which unfuses (as it must).
    host.adaptable_mut(&["blocks", "0", "attn", "to_q"])
        .unwrap()
        .set_adapters(vec![Adapter::Lora {
            a: seq(&[64, 4], 0.021, 0.3),
            b: seq(&[4, 64], 0.037, 0.8),
            scale: 1.0,
        }]);
    assert!(!host.block.attn.qkv.fusion_engaged());

    let facts = host
        .adaptable_facts(&["blocks", "0", "attn", "to_q"])
        .unwrap();
    assert!(
        facts.has_live_adapters,
        "a live adapter must be visible to a probe"
    );
    assert_eq!(facts.adapter_count, 1);
    // ... and the siblings are still clean.
    let k = host
        .adaptable_facts(&["blocks", "0", "attn", "to_k"])
        .unwrap();
    assert!(!k.has_live_adapters);
    assert_eq!(k.adapter_count, 0);

    // A disabled adapter is NOT "live" but IS "installed" -- the two fields answer different
    // questions, and `adapter_count` is the one a capture/replay walk needs.
    host.adaptable_mut(&["blocks", "0", "attn", "to_k"])
        .unwrap()
        .set_adapters(vec![Adapter::Lora {
            a: seq(&[64, 4], 0.021, 0.3),
            b: seq(&[4, 64], 0.037, 0.8),
            scale: 0.0,
        }]);
    let k = host
        .adaptable_facts(&["blocks", "0", "attn", "to_k"])
        .unwrap();
    assert!(!k.has_live_adapters, "scale-0 is not live");
    assert_eq!(k.adapter_count, 1, "but it IS installed");
}
