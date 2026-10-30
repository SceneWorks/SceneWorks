//! **Per-family bit-exactness fixtures for the SC-18319 attention-prologue migration.**
//!
//! Ten families were moved off hand-written QK-norm/RoPE/layout prologues and onto
//! [`mlx_gen::qkv`]. That migration is only acceptable if it is a *provable* no-op, so each test
//! here does four things:
//!
//! 1. builds deterministic inputs at the family's **real** geometry and dtype (the head counts,
//!    head_dim, GQA ratio and eps are cited per test against the family's own source);
//! 2. transcribes the family's **old** op sequence — the `-` lines of the migration diff — verbatim
//!    into a local reference closure;
//! 3. runs `qkv::prepare` (plus `StreamOrder::join` / `qkv::merge_heads` where the family uses
//!    them) with the **same** spec the migrated family now builds;
//! 4. asserts **exact** equality — same shape, same dtype, `array_eq` true. No tolerance.
//!
//! Every test also carries a **negative control**: the reference is re-run against a deliberately
//! wrong spec (a flipped [`StreamOrder`], a dropped k-table, a dropped QK-norm, a flipped
//! [`RopeStyle`], a flipped [`RopeDtype`]) and must *disagree*. Without that, a bit-exactness
//! assertion passes trivially whenever both sides computed the same nothing.
//!
//! **Dtypes.** Every family in this set loads bf16 weights in production, and bf16 is where a
//! reordered op sequence actually diverges, so every fixture runs at `Bfloat16` as well as
//! `Float32`. The RoPE tables stay f32 in both arms because every table in this tree is built in
//! f32, while the QK-norm weights follow the stream — that is the real production pairing.
//!
//! That pairing is what makes [`RopeDtype`] (knob 12) observable at all: an f32 table over a bf16
//! stream promotes, and whether the family casts the result back decides whether SDPA runs in bf16
//! or f32. The first draft of these fixtures ran with that knob absent — the primitive restored the
//! input dtype unconditionally — and Ideogram, Anima and PiD failed here, because none of their
//! removed rotations cast back. Each family's arm is pinned below and, wherever the stream is
//! genuinely bf16 at the rotation, flipping it is asserted to change the result.
//!
//! **Compiled glue.** `nn::rope_rotate` is eager by default (`COMPILE_GLUE` starts `false`) and
//! `nn.rs`'s `compile_glue_helpers_are_bit_exact_and_modulate_dtype_policy` already pins
//! compiled == eager bit-for-bit, so these fixtures run at the default and do not re-litigate it.
//!
//! Sizes are shrunk on the *token* axis only. Head counts, head_dim, the GQA ratio and every
//! divisibility fact are the production values.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{
    add, array_eq, broadcast_to, concatenate_axis, mean_axis, multiply, rsqrt, split, stack_axis,
    subtract,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{apply_text_rope, rope_rotate};
use mlx_gen::qkv::{
    self, AttnPrepSpec, NormDtype, QkNormSpec, QkvHeads, QkvSource, RopeDtype, RopeSpec, RopeStyle,
    RopeTables, RotationAxes, StreamOrder,
};

// ── shared helpers ───────────────────────────────────────────────────────────────────────────

/// The two dtype arms every fixture runs. bf16 is the production arm and the one where a reordered
/// op sequence actually diverges; f32 is the parity arm.
const DTYPES: [Dtype; 2] = [Dtype::Float32, Dtype::Bfloat16];

/// Deterministic pseudo-random f32 fill — a closed-form function of the flat index, never a seeded
/// RNG, so a failure is reproducible from the test source alone.
fn seq(shape: &[i32], scale: f32, offset: f32) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * scale + offset).sin())
        .collect();
    Array::from_slice(&data, shape)
}

/// [`seq`] at a given dtype — the stream/weight arms; RoPE tables stay f32.
fn seq_at(shape: &[i32], scale: f32, offset: f32, dt: Dtype) -> Array {
    seq(shape, scale, offset).as_dtype(dt).unwrap()
}

/// Bit-exactness: same shape, same **dtype** (a silent widening from bf16 to f32 is not a no-op —
/// it changes what SDPA consumes), and every element identical. No tolerance.
fn same(a: &Array, b: &Array) -> bool {
    a.shape() == b.shape() && a.dtype() == b.dtype() && array_eq(a, b, None).unwrap().item::<bool>()
}

/// Assert the migrated stream reproduces the reference **exactly**, reporting which of the three
/// claims failed.
///
/// The dtype claim is not pedantry: it is [`RopeDtype`] (knob 12). On a bf16 stream with f32 RoPE
/// tables, `Promoted` hands SDPA f32 queries/keys and `RestoreInput` hands it bf16 ones — a real
/// numeric change with no shape error, so a mismatch is reported with a note on whether the
/// *values* also moved.
fn assert_same(what: &str, dt: Dtype, reference: &Array, migrated: &Array) {
    assert_eq!(
        reference.shape(),
        migrated.shape(),
        "{what} @ {dt:?}: shape changed"
    );
    if reference.dtype() != migrated.dtype() {
        let demoted = reference.as_dtype(migrated.dtype()).unwrap();
        let demoted_agree = array_eq(&demoted, migrated, None).unwrap().item::<bool>();
        // The plain `array_eq` the story asks for, with MLX's own promotion applied.
        let promoted_agree = array_eq(reference, migrated, None).unwrap().item::<bool>();
        panic!(
            "{what} @ {dt:?}: the migrated prologue yields {:?} but the pre-migration op sequence \
             yielded {:?} — SDPA no longer consumes the same tensor. Values agree after a demoting \
             cast: {demoted_agree}; values agree under MLX's own promotion (the bare `array_eq`): \
             {promoted_agree}. This is knob 12 (`RopeDtype`): check the arm this family's spec \
             selects against what its removed rotation actually did with the f32 promotion its \
             f32 RoPE tables introduce.",
            migrated.dtype(),
            reference.dtype()
        );
    }
    assert!(
        array_eq(reference, migrated, None).unwrap().item::<bool>(),
        "{what} @ {dt:?}: not bit-identical to the pre-migration op sequence"
    );
}

/// The negative control: a deliberately wrong spec must *disagree* with the reference, or the
/// fixture is vacuous.
fn assert_differs(what: &str, dt: Dtype, reference: &Array, wrong: &Array) {
    assert!(
        !same(reference, wrong),
        "{what} @ {dt:?}: the deliberately-wrong spec agreed with the reference — this fixture \
         proves nothing"
    );
}

// ── Lens ─────────────────────────────────────────────────────────────────────────────────────

/// **Lens** — `mlx-gen-lens/src/dit/attention.rs`.
///
/// Geometry: `num_heads = 24`, `head_dim = 64` (`mlx-gen-lens/src/dit/transformer.rs:43-44`),
/// `RMS_EPS = 1e-6` (`mlx-gen-lens/src/dit/attention.rs:25`), per-stream RoPE tables
/// `[seq, head_dim/2]` (`mlx-gen-lens/src/dit/rope.rs:16-17`), fused packed QKV, `[img, txt]` join.
#[test]
fn lens_joint_attention_prologue_is_bit_exact() {
    const HEADS: i32 = 24;
    const HEAD_DIM: i32 = 64;
    const RMS_EPS: f32 = 1e-6;
    let (b, img_seq, txt_seq) = (2, 6, 4);
    let half = HEAD_DIM / 2;

    // The removed `apply_rope` (interleaved complex), verbatim.
    let old_rope = |x: &Array, cos: &Array, sin: &Array| -> Array {
        let sh = x.shape();
        let (b, seq, heads, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let half = hd / 2;
        let x5 = x.reshape(&[b, seq, heads, half, 2]).unwrap();
        let parts = split(&x5, 2, 4).unwrap();
        let xr = parts[0].reshape(&[b, seq, heads, half]).unwrap();
        let xi = parts[1].reshape(&[b, seq, heads, half]).unwrap();
        let cos = cos.reshape(&[1, seq, 1, half]).unwrap();
        let sin = sin.reshape(&[1, seq, 1, half]).unwrap();
        let out_r = subtract(multiply(&xr, &cos).unwrap(), multiply(&xi, &sin).unwrap()).unwrap();
        let out_i = add(multiply(&xr, &sin).unwrap(), multiply(&xi, &cos).unwrap()).unwrap();
        let stacked = concatenate_axis(
            &[
                &out_r.expand_dims(4).unwrap(),
                &out_i.expand_dims(4).unwrap(),
            ],
            4,
        )
        .unwrap();
        stacked
            .reshape(&[b, seq, heads, hd])
            .unwrap()
            .as_dtype(x.dtype())
            .unwrap()
    };

    for dt in DTYPES {
        let img_packed = seq_at(&[b, img_seq, 3 * HEADS * HEAD_DIM], 0.0007, 0.2, dt);
        let txt_packed = seq_at(&[b, txt_seq, 3 * HEADS * HEAD_DIM], 0.0011, 0.7, dt);
        let nq = seq_at(&[HEAD_DIM], 0.31, 1.0, dt);
        let nk = seq_at(&[HEAD_DIM], 0.17, 0.5, dt);
        let anq = seq_at(&[HEAD_DIM], 0.23, 0.1, dt);
        let ank = seq_at(&[HEAD_DIM], 0.13, 0.9, dt);
        let img_cos = seq(&[img_seq, half], 0.07, 0.0);
        let img_sin = seq(&[img_seq, half], 0.05, 0.9);
        let txt_cos = seq(&[txt_seq, half], 0.03, 0.4);
        let txt_sin = seq(&[txt_seq, half], 0.09, 1.3);

        // ── reference: the removed `-` lines of `LensJointAttention::forward`.
        let old_stream = |packed: &Array,
                          s: i32,
                          nq: &Array,
                          nk: &Array,
                          cos: &Array,
                          sin: &Array|
         -> (Array, Array, Array) {
            let t = packed.reshape(&[b, s, 3, HEADS, HEAD_DIM]).unwrap();
            let parts = split(&t, 3, 2).unwrap();
            let q = parts[0].reshape(&[b, s, HEADS, HEAD_DIM]).unwrap();
            let k = parts[1].reshape(&[b, s, HEADS, HEAD_DIM]).unwrap();
            let v = parts[2].reshape(&[b, s, HEADS, HEAD_DIM]).unwrap();
            let q = rms_norm(&q, nq, RMS_EPS).unwrap();
            let k = rms_norm(&k, nk, RMS_EPS).unwrap();
            (old_rope(&q, cos, sin), old_rope(&k, cos, sin), v)
        };
        let (iq, ik, iv) = old_stream(&img_packed, img_seq, &nq, &nk, &img_cos, &img_sin);
        let (tq, tk, tv) = old_stream(&txt_packed, txt_seq, &anq, &ank, &txt_cos, &txt_sin);
        let cat = |a: &Array, b: &Array| {
            concatenate_axis(&[a, b], 1)
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap()
        };
        let (ref_q, ref_k, ref_v) = (cat(&iq, &tq), cat(&ik, &tk), cat(&iv, &tv));

        // ── migrated: the `+` lines.
        let stream = |packed: &Array, nq: &Array, nk: &Array, cos: &Array, sin: &Array| {
            let spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(QkNormSpec::per_head(nq, nk, RMS_EPS))
                .with_rope(RopeSpec {
                    style: RopeStyle::AdjacentPair,
                    q: Some(RopeTables::new(cos, sin)),
                    k: Some(RopeTables::new(cos, sin)),
                    // Knob 12 — the removed rotation ended in `.as_dtype(x.dtype())`.
                    dtype: RopeDtype::RestoreInput,
                    ..RopeSpec::default()
                });
            qkv::prepare(QkvSource::Packed(packed), &spec).unwrap()
        };
        let img = stream(&img_packed, &nq, &nk, &img_cos, &img_sin);
        let txt = stream(&txt_packed, &anq, &ank, &txt_cos, &txt_sin);
        let joint = StreamOrder::ImageFirst.join(&img, &txt).unwrap();

        assert_eq!(joint.q.shape(), [b, HEADS, img_seq + txt_seq, HEAD_DIM]);
        assert_same("lens q", dt, &ref_q, &joint.q);
        assert_same("lens k", dt, &ref_k, &joint.k);
        assert_same("lens v", dt, &ref_v, &joint.v);

        // ── negative controls.
        // Knob 11 flipped: `[txt, img]` is a running model with garbage output and no shape error.
        let wrong_order = StreamOrder::TextFirst.join(&img, &txt).unwrap();
        assert_differs("lens stream order", dt, &ref_q, &wrong_order.q);
        // Knob 5: dropping the k table must change the key stream.
        let no_k_table = {
            let spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(QkNormSpec::per_head(&nq, &nk, RMS_EPS))
                .with_rope(RopeSpec {
                    style: RopeStyle::AdjacentPair,
                    q: Some(RopeTables::new(&img_cos, &img_sin)),
                    k: None,
                    dtype: RopeDtype::RestoreInput,
                    ..RopeSpec::default()
                });
            qkv::prepare(QkvSource::Packed(&img_packed), &spec).unwrap()
        };
        assert_differs("lens dropped k table", dt, &img.k, &no_k_table.k);
        // Knob 12: leaving the f32 promotion standing feeds SDPA a different tensor.
        if dt != Dtype::Float32 {
            let promoted = {
                let spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
                    .with_qk_norm(QkNormSpec::per_head(&nq, &nk, RMS_EPS))
                    .with_rope(RopeSpec {
                        style: RopeStyle::AdjacentPair,
                        q: Some(RopeTables::new(&img_cos, &img_sin)),
                        k: Some(RopeTables::new(&img_cos, &img_sin)),
                        dtype: RopeDtype::Promoted,
                        ..RopeSpec::default()
                    });
                qkv::prepare(QkvSource::Packed(&img_packed), &spec).unwrap()
            };
            assert_differs("lens rope dtype", dt, &img.q, &promoted.q);
        }
    }
}

// ── Ideogram ─────────────────────────────────────────────────────────────────────────────────

/// **Ideogram 4** — `mlx-gen-ideogram/src/transformer/block.rs`.
///
/// Geometry: `num_heads = 18`, `head_dim = 256` (`mlx-gen-ideogram/src/config.rs:83-84`,
/// `emb_dim = 4608 = 18 · 256`), `eps = 1e-5` (`.../transformer/block.rs:47`), **per-batch**
/// full-width MRoPE tables `[B, L, head_dim]` (`.../transformer/mrope.rs:53`, built f32 at `:55-63`),
/// fused packed QKV, half-split `rotate_half` applied head-major, single stream.
#[test]
fn ideogram_attention_prologue_is_bit_exact() {
    const HEADS: i32 = 18;
    const HEAD_DIM: i32 = 256;
    const EPS: f32 = 1e-5;
    let (b, s) = (2, 5);

    // The removed `apply_rope` (HF half-split, `[B, H, L, hd]` layout), verbatim.
    let old_rope = |x: &Array, cos: &Array, sin: &Array| -> Array {
        let cos = cos.expand_dims(1).unwrap(); // [B,1,L,hd]
        let sin = sin.expand_dims(1).unwrap();
        let parts = split(x, 2, 3).unwrap();
        let rot = concatenate_axis(&[&parts[1].negative().unwrap(), &parts[0]], 3).unwrap();
        add(multiply(x, &cos).unwrap(), multiply(&rot, &sin).unwrap()).unwrap()
    };

    for dt in DTYPES {
        let packed = seq_at(&[b, s, 3 * HEADS * HEAD_DIM], 0.0003, 0.15, dt);
        let nq = seq_at(&[HEAD_DIM], 0.011, 0.3, dt);
        let nk = seq_at(&[HEAD_DIM], 0.013, 0.8, dt);
        let cos = seq(&[b, s, HEAD_DIM], 0.017, 0.0);
        let sin = seq(&[b, s, HEAD_DIM], 0.019, 0.6);

        // ── reference.
        let t = packed.reshape(&[b, s, 3, HEADS, HEAD_DIM]).unwrap();
        let parts = split(&t, 3, 2).unwrap();
        let q = parts[0].reshape(&[b, s, HEADS, HEAD_DIM]).unwrap();
        let k = parts[1].reshape(&[b, s, HEADS, HEAD_DIM]).unwrap();
        let v = parts[2].reshape(&[b, s, HEADS, HEAD_DIM]).unwrap();
        let q = rms_norm(&q, &nq, EPS).unwrap();
        let k = rms_norm(&k, &nk, EPS).unwrap();
        let q = q.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let k = k.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let ref_v = v.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let ref_q = old_rope(&q, &cos, &sin);
        let ref_k = old_rope(&k, &cos, &sin);

        // ── migrated.
        let spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
            .with_qk_norm(QkNormSpec::per_head(&nq, &nk, EPS))
            .with_rope(RopeSpec {
                style: RopeStyle::RotateHalf,
                q: Some(RopeTables::new(&cos, &sin)),
                k: Some(RopeTables::new(&cos, &sin)),
                // Knob 12 — the removed rotation was a bare `x·cos + rotate_half(x)·sin` with no
                // cast back, so the f32 tables' promotion reaches SDPA.
                dtype: RopeDtype::Promoted,
                ..RopeSpec::default()
            })
            .with_rotation_axes(RotationAxes::HeadMajor);
        let heads = qkv::prepare(QkvSource::Packed(&packed), &spec).unwrap();

        assert_eq!(heads.q.shape(), [b, HEADS, s, HEAD_DIM]);
        assert_same("ideogram q", dt, &ref_q, &heads.q);
        assert_same("ideogram k", dt, &ref_k, &heads.k);
        assert_same("ideogram v", dt, &ref_v, &heads.v);

        // ── negative controls: no rotation at all, and a dropped k table.
        let no_rope = qkv::prepare(
            QkvSource::Packed(&packed),
            &AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(QkNormSpec::per_head(&nq, &nk, EPS))
                .with_rotation_axes(RotationAxes::HeadMajor),
        )
        .unwrap();
        assert_differs("ideogram rope disabled", dt, &ref_q, &no_rope.q);
        let no_k_table = qkv::prepare(
            QkvSource::Packed(&packed),
            &AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(QkNormSpec::per_head(&nq, &nk, EPS))
                .with_rope(RopeSpec {
                    style: RopeStyle::RotateHalf,
                    q: Some(RopeTables::new(&cos, &sin)),
                    k: None,
                    dtype: RopeDtype::Promoted,
                    ..RopeSpec::default()
                })
                .with_rotation_axes(RotationAxes::HeadMajor),
        )
        .unwrap();
        assert_differs("ideogram dropped k table", dt, &ref_k, &no_k_table.k);
        // Knob 12: casting back to bf16 would hand SDPA a different tensor. This is the control
        // for the divergence these fixtures found before `RopeDtype` existed.
        if dt != Dtype::Float32 {
            let restored = qkv::prepare(
                QkvSource::Packed(&packed),
                &AttnPrepSpec::new(HEADS, HEAD_DIM)
                    .with_qk_norm(QkNormSpec::per_head(&nq, &nk, EPS))
                    .with_rope(RopeSpec {
                        style: RopeStyle::RotateHalf,
                        q: Some(RopeTables::new(&cos, &sin)),
                        k: Some(RopeTables::new(&cos, &sin)),
                        dtype: RopeDtype::RestoreInput,
                        ..RopeSpec::default()
                    })
                    .with_rotation_axes(RotationAxes::HeadMajor),
            )
            .unwrap();
            assert_differs("ideogram rope dtype", dt, &ref_q, &restored.q);
        }
    }
}

// ── PiD ──────────────────────────────────────────────────────────────────────────────────────

/// **PiD** — `mlx-gen-pid/src/backbone/blocks.rs`.
///
/// Geometry: `num_groups = 24` heads, `hidden_size = 1536` ⇒ `head_dim = 64`
/// (`mlx-gen-pid/src/config.rs:128-129`, `:96-98`), `RMS_EPS = 1e-6`
/// (`mlx-gen-pid/src/backbone/layers.rs:18`), the f32-round-trip RMSNorm at `layers.rs:32-36`
/// ([`NormDtype::F32RoundTrip`]), interleaved RoPE tables `[N, head_dim/2]` applied **head-major**,
/// fused packed QKV, `[txt, img]` join.
#[test]
fn pid_backbone_prologue_is_bit_exact() {
    const HEADS: i32 = 24;
    const HEAD_DIM: i32 = 64;
    const RMS_EPS: f32 = 1e-6;
    let (b, nx, ny) = (2, 6, 4);
    let half = HEAD_DIM / 2;

    // `backbone::layers::rms`, verbatim.
    let old_rms = |x: &Array, w: &Array| -> Array {
        let xf = x.as_dtype(Dtype::Float32).unwrap();
        let wf = w.as_dtype(Dtype::Float32).unwrap();
        rms_norm(&xf, &wf, RMS_EPS)
            .unwrap()
            .as_dtype(x.dtype())
            .unwrap()
    };
    // The removed `backbone::rope::apply_rope`, verbatim (one stream).
    let old_rope = |x: &Array, cos: &Array, sin: &Array| -> Array {
        let s = cos.shape()[0];
        let half = cos.shape()[1];
        let cos = cos.reshape(&[1, 1, s, half]).unwrap();
        let sin = sin.reshape(&[1, 1, s, half]).unwrap();
        let sh = x.shape();
        let (b, h, seq, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let x5 = x.reshape(&[b, h, seq, hd / 2, 2]).unwrap();
        let p = split(&x5, 2, 4).unwrap();
        let real = p[0].reshape(&[b, h, seq, hd / 2]).unwrap();
        let imag = p[1].reshape(&[b, h, seq, hd / 2]).unwrap();
        let (out0, out1) = rope_rotate(&real, &imag, &cos, &sin).unwrap();
        concatenate_axis(
            &[&out0.expand_dims(4).unwrap(), &out1.expand_dims(4).unwrap()],
            4,
        )
        .unwrap()
        .reshape(&[b, h, seq, hd])
        .unwrap()
    };

    for dt in DTYPES {
        let x_packed = seq_at(&[b, nx, 3 * HEADS * HEAD_DIM], 0.0009, 0.25, dt);
        let y_packed = seq_at(&[b, ny, 3 * HEADS * HEAD_DIM], 0.0013, 0.85, dt);
        let qnx = seq_at(&[HEAD_DIM], 0.29, 0.4, dt);
        let knx = seq_at(&[HEAD_DIM], 0.19, 1.1, dt);
        let qny = seq_at(&[HEAD_DIM], 0.37, 0.2, dt);
        let kny = seq_at(&[HEAD_DIM], 0.41, 0.7, dt);
        let cos_img = seq(&[nx, half], 0.05, 0.0);
        let sin_img = seq(&[nx, half], 0.03, 0.5);
        let cos_txt = seq(&[ny, half], 0.11, 0.9);
        let sin_txt = seq(&[ny, half], 0.07, 1.4);

        // ── reference: `split_qkv` → `rms` → `to_bhsd` → `apply_rope`.
        let old_stream = |packed: &Array,
                          s: i32,
                          qn: &Array,
                          kn: &Array,
                          cos: &Array,
                          sin: &Array|
         -> (Array, Array, Array) {
            let q5 = packed.reshape(&[b, s, 3, HEADS, HEAD_DIM]).unwrap();
            let parts = split(&q5, 3, 2).unwrap();
            let take = |a: &Array| a.reshape(&[b, s, HEADS, HEAD_DIM]).unwrap();
            let (q, k, v) = (take(&parts[0]), take(&parts[1]), take(&parts[2]));
            let q = old_rms(&q, qn);
            let k = old_rms(&k, kn);
            let to_bhsd = |a: &Array| a.transpose_axes(&[0, 2, 1, 3]).unwrap();
            let (q, k, v) = (to_bhsd(&q), to_bhsd(&k), to_bhsd(&v));
            (old_rope(&q, cos, sin), old_rope(&k, cos, sin), v)
        };
        let (qx, kx, vx) = old_stream(&x_packed, nx, &qnx, &knx, &cos_img, &sin_img);
        let (qy, ky, vy) = old_stream(&y_packed, ny, &qny, &kny, &cos_txt, &sin_txt);
        // joint `[txt, img]` on axis 2.
        let cat = |y: &Array, x: &Array| concatenate_axis(&[y, x], 2).unwrap();
        let (ref_q, ref_k, ref_v) = (cat(&qy, &qx), cat(&ky, &kx), cat(&vy, &vx));

        // ── migrated: `pid_prepare` per stream, then `StreamOrder::TextFirst`.
        let prep = |packed: &Array, qn: &Array, kn: &Array, cos: &Array, sin: &Array| {
            let spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(
                    QkNormSpec::per_head(qn, kn, RMS_EPS).with_dtype(NormDtype::F32RoundTrip),
                )
                .with_rope(RopeSpec {
                    style: RopeStyle::AdjacentPair,
                    q: Some(RopeTables::new(cos, sin)),
                    k: Some(RopeTables::new(cos, sin)),
                    // Knob 12 — `backbone::rope::apply_rope` never cast back, so the f32 tables'
                    // promotion reaches `flash_sdpa`.
                    dtype: RopeDtype::Promoted,
                    ..RopeSpec::default()
                })
                .with_rotation_axes(RotationAxes::HeadMajor);
            qkv::prepare(QkvSource::Packed(packed), &spec).unwrap()
        };
        let img = prep(&x_packed, &qnx, &knx, &cos_img, &sin_img);
        let txt = prep(&y_packed, &qny, &kny, &cos_txt, &sin_txt);
        let joint = StreamOrder::TextFirst.join(&img, &txt).unwrap();

        assert_eq!(joint.q.shape(), [b, HEADS, ny + nx, HEAD_DIM]);
        assert_same("pid q", dt, &ref_q, &joint.q);
        assert_same("pid k", dt, &ref_k, &joint.k);
        assert_same("pid v", dt, &ref_v, &joint.v);

        // ── negative controls: the `[txt, img]` order, and the f32-round-trip norm policy.
        let wrong_order = StreamOrder::ImageFirst.join(&img, &txt).unwrap();
        assert_differs("pid stream order", dt, &ref_q, &wrong_order.q);
        let native_norm = qkv::prepare(
            QkvSource::Packed(&x_packed),
            &AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(QkNormSpec::none())
                .with_rope(RopeSpec {
                    style: RopeStyle::AdjacentPair,
                    q: Some(RopeTables::new(&cos_img, &sin_img)),
                    k: Some(RopeTables::new(&cos_img, &sin_img)),
                    dtype: RopeDtype::Promoted,
                    ..RopeSpec::default()
                })
                .with_rotation_axes(RotationAxes::HeadMajor),
        )
        .unwrap();
        assert_differs("pid dropped qk-norm", dt, &qx, &native_norm.q);
        // Knob 12 — PiD's `F32RoundTrip` norm hands the rotation a bf16 stream, so restoring that
        // dtype afterwards (which the removed code did NOT do) is observable.
        if dt != Dtype::Float32 {
            let restored = qkv::prepare(
                QkvSource::Packed(&x_packed),
                &AttnPrepSpec::new(HEADS, HEAD_DIM)
                    .with_qk_norm(
                        QkNormSpec::per_head(&qnx, &knx, RMS_EPS)
                            .with_dtype(NormDtype::F32RoundTrip),
                    )
                    .with_rope(RopeSpec {
                        style: RopeStyle::AdjacentPair,
                        q: Some(RopeTables::new(&cos_img, &sin_img)),
                        k: Some(RopeTables::new(&cos_img, &sin_img)),
                        dtype: RopeDtype::RestoreInput,
                        ..RopeSpec::default()
                    })
                    .with_rotation_axes(RotationAxes::HeadMajor),
            )
            .unwrap();
            assert_differs("pid rope dtype", dt, &qx, &restored.q);
        }
    }
}

// ── Chroma ───────────────────────────────────────────────────────────────────────────────────

/// **Chroma** — `mlx-gen-chroma/src/transformer.rs`.
///
/// Geometry: `num_attention_heads = 24`, `attention_head_dim = 128`
/// (`mlx-gen-chroma/src/config.rs:186-187`), `QK_RMS_EPS = 1e-6`
/// (`mlx-gen-chroma/src/transformer.rs:45`), FluxPosEmbed tables `[N, head_dim/2]` (`:121`),
/// separate q/k/v, the f32-promoting `proj_heads` policy ([`NormDtype::PromoteToF32`], which
/// promotes `v` too), head-major adjacent-pair rotation, and **knob 8's concat-then-RoPE arm** with
/// a `[txt, img]` join.
#[test]
fn chroma_double_and_single_attention_prologues_are_bit_exact() {
    const HEADS: i32 = 24;
    const HEAD_DIM: i32 = 128;
    const QK_RMS_EPS: f32 = 1e-6;
    let (b, img_seq, txt_seq) = (2, 6, 4);
    let half = HEAD_DIM / 2;

    // The removed `proj_heads` (post-projection half), verbatim.
    let old_proj_heads = |y: &Array, norm: Option<&Array>| -> Array {
        let (b, s) = (y.shape()[0], y.shape()[1]);
        let y = y
            .reshape(&[b, s, HEADS, HEAD_DIM])
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap();
        match norm {
            Some(w) => rms_norm(y.as_dtype(Dtype::Float32).unwrap(), w, QK_RMS_EPS).unwrap(),
            None => y.as_dtype(Dtype::Float32).unwrap(),
        }
    };
    // The removed `apply_rope_one`, verbatim.
    let old_rope_one = |x: &Array, cos: &Array, sin: &Array| -> Array {
        let sh = x.shape();
        let (b, heads, seq, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let half = hd / 2;
        let x5 = x
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[b, heads, seq, half, 2])
            .unwrap();
        let p = split(&x5, 2, 4).unwrap();
        let real = p[0].reshape(&[b, heads, seq, half]).unwrap();
        let imag = p[1].reshape(&[b, heads, seq, half]).unwrap();
        let c = cos.reshape(&[1, 1, seq, half]).unwrap();
        let s = sin.reshape(&[1, 1, seq, half]).unwrap();
        let (out0, out1) = rope_rotate(&real, &imag, &c, &s).unwrap();
        concatenate_axis(
            &[&out0.expand_dims(4).unwrap(), &out1.expand_dims(4).unwrap()],
            4,
        )
        .unwrap()
        .reshape(&[b, heads, seq, hd])
        .unwrap()
    };

    for dt in DTYPES {
        let inner = HEADS * HEAD_DIM;
        let hid_q = seq_at(&[b, img_seq, inner], 0.0007, 0.1, dt);
        let hid_k = seq_at(&[b, img_seq, inner], 0.0009, 0.5, dt);
        let hid_v = seq_at(&[b, img_seq, inner], 0.0011, 0.9, dt);
        let enc_q = seq_at(&[b, txt_seq, inner], 0.0013, 0.3, dt);
        let enc_k = seq_at(&[b, txt_seq, inner], 0.0017, 0.7, dt);
        let enc_v = seq_at(&[b, txt_seq, inner], 0.0019, 1.1, dt);
        let nq = seq_at(&[HEAD_DIM], 0.23, 0.2, dt);
        let nk = seq_at(&[HEAD_DIM], 0.29, 0.6, dt);
        let anq = seq_at(&[HEAD_DIM], 0.31, 1.0, dt);
        let ank = seq_at(&[HEAD_DIM], 0.37, 1.4, dt);
        // The joint table covers `[txt, img]` in that order.
        let joint_cos = seq(&[txt_seq + img_seq, half], 0.013, 0.0);
        let joint_sin = seq(&[txt_seq + img_seq, half], 0.011, 0.4);

        // ── reference: DoubleAttn (`proj_heads` ×6 → concat `[txt, img]` → `apply_rope_one`).
        let iq = old_proj_heads(&hid_q, Some(&nq));
        let ik = old_proj_heads(&hid_k, Some(&nk));
        let iv = old_proj_heads(&hid_v, None);
        let eq = old_proj_heads(&enc_q, Some(&anq));
        let ek = old_proj_heads(&enc_k, Some(&ank));
        let ev = old_proj_heads(&enc_v, None);
        let cat = |t: &Array, i: &Array| concatenate_axis(&[t, i], 2).unwrap();
        let ref_v = cat(&ev, &iv);
        let ref_q = old_rope_one(&cat(&eq, &iq), &joint_cos, &joint_sin);
        let ref_k = old_rope_one(&cat(&ek, &ik), &joint_cos, &joint_sin);

        // ── migrated: `chroma_spec` (rope `None`) per stream → `join` → `rotate_joint`.
        let unrotated = |q: &Array, k: &Array, v: &Array, nq: &Array, nk: &Array| {
            let spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(
                    QkNormSpec::per_head(nq, nk, QK_RMS_EPS).with_dtype(NormDtype::PromoteToF32),
                )
                .with_rotation_axes(RotationAxes::HeadMajor);
            qkv::prepare(QkvSource::Separate { q, k, v }, &spec).unwrap()
        };
        let img = unrotated(&hid_q, &hid_k, &hid_v, &nq, &nk);
        let txt = unrotated(&enc_q, &enc_k, &enc_v, &anq, &ank);
        let joint = StreamOrder::TextFirst.join(&img, &txt).unwrap();
        let rotate_joint = |x: &Array| {
            qkv::apply_rope(
                x,
                RopeTables::new(&joint_cos, &joint_sin),
                RopeStyle::AdjacentPair,
                RotationAxes::HeadMajor,
                None,
                RopeDtype::Promoted,
            )
            .unwrap()
        };
        let got_q = rotate_joint(&joint.q);
        let got_k = rotate_joint(&joint.k);

        assert_eq!(got_q.shape(), [b, HEADS, txt_seq + img_seq, HEAD_DIM]);
        assert_same("chroma double q", dt, &ref_q, &got_q);
        assert_same("chroma double k", dt, &ref_k, &got_k);
        assert_same("chroma double v", dt, &ref_v, &joint.v);

        // ── reference: SingleAttn (per-stream rotation, knob 8's other arm).
        let single_cos = seq(&[img_seq, half], 0.013, 0.0);
        let single_sin = seq(&[img_seq, half], 0.011, 0.4);
        let single_ref_q =
            old_rope_one(&old_proj_heads(&hid_q, Some(&nq)), &single_cos, &single_sin);
        let single = qkv::prepare(
            QkvSource::Separate {
                q: &hid_q,
                k: &hid_k,
                v: &hid_v,
            },
            &AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(
                    QkNormSpec::per_head(&nq, &nk, QK_RMS_EPS).with_dtype(NormDtype::PromoteToF32),
                )
                .with_rope(RopeSpec {
                    style: RopeStyle::AdjacentPair,
                    q: Some(RopeTables::new(&single_cos, &single_sin)),
                    k: Some(RopeTables::new(&single_cos, &single_sin)),
                    // Knob 12 — Chroma is `Promoted`, and it is a no-op here because
                    // `NormDtype::PromoteToF32` already carried the whole stream to f32.
                    dtype: RopeDtype::Promoted,
                    ..RopeSpec::default()
                })
                .with_rotation_axes(RotationAxes::HeadMajor),
        )
        .unwrap();
        assert_same("chroma single q", dt, &single_ref_q, &single.q);

        // ── negative controls: the `[txt, img]` order, and the f32-promoting norm policy (which is
        // what carries `v` to f32 even though `v` is never normalized).
        let wrong_order = StreamOrder::ImageFirst.join(&img, &txt).unwrap();
        assert_differs("chroma stream order", dt, &ref_v, &wrong_order.v);
        if dt != Dtype::Float32 {
            let native = qkv::prepare(
                QkvSource::Separate {
                    q: &hid_q,
                    k: &hid_k,
                    v: &hid_v,
                },
                &AttnPrepSpec::new(HEADS, HEAD_DIM)
                    .with_qk_norm(QkNormSpec::per_head(&nq, &nk, QK_RMS_EPS))
                    .with_rotation_axes(RotationAxes::HeadMajor),
            )
            .unwrap();
            assert_differs("chroma native norm dtype", dt, &iv, &native.v);
        }
    }
}

// ── SD3 ──────────────────────────────────────────────────────────────────────────────────────

/// **SD3.5-Large** — `mlx-gen-sd3/src/transformer.rs`.
///
/// Geometry: `num_attention_heads = 38`, `attention_head_dim = 64`
/// (`mlx-gen-sd3/src/config.rs:56` — `inner_dim = 38 × 64 = 2432`), `RMS_EPS = 1e-6`
/// (`mlx-gen-sd3/src/config.rs:81`). The sparsest knob selection in the tree: separate q/k/v,
/// per-head q/k RMSNorm, **no RoPE at all** (knob 7), head-major axes, `[img, txt]` join.
#[test]
fn sd3_joint_attention_prologue_is_bit_exact() {
    const HEADS: i32 = 38;
    const HEAD_DIM: i32 = 64;
    const RMS_EPS: f32 = 1e-6;
    let (b, img_seq, txt_seq) = (2, 6, 4);

    for dt in DTYPES {
        let inner = HEADS * HEAD_DIM;
        let mk = |s: i32, scale: f32, off: f32| seq_at(&[b, s, inner], scale, off, dt);
        let (iq, ik, iv) = (
            mk(img_seq, 0.0007, 0.1),
            mk(img_seq, 0.0009, 0.5),
            mk(img_seq, 0.0011, 0.9),
        );
        let (tq, tk, tv) = (
            mk(txt_seq, 0.0013, 0.3),
            mk(txt_seq, 0.0017, 0.7),
            mk(txt_seq, 0.0019, 1.1),
        );
        let nq = seq_at(&[HEAD_DIM], 0.23, 0.2, dt);
        let nk = seq_at(&[HEAD_DIM], 0.29, 0.6, dt);
        let anq = seq_at(&[HEAD_DIM], 0.31, 1.0, dt);
        let ank = seq_at(&[HEAD_DIM], 0.37, 1.4, dt);

        // ── reference: the removed body of `process_qkv` (reshape → transpose → rms_norm).
        let old_process =
            |q: &Array, k: &Array, v: &Array, nq: &Array, nk: &Array| -> (Array, Array, Array) {
                let s = q.shape()[1];
                let to_bhsd = |a: &Array| {
                    a.reshape(&[b, s, HEADS, HEAD_DIM])
                        .unwrap()
                        .transpose_axes(&[0, 2, 1, 3])
                        .unwrap()
                };
                let (q, k, v) = (to_bhsd(q), to_bhsd(k), to_bhsd(v));
                (
                    rms_norm(&q, nq, RMS_EPS).unwrap(),
                    rms_norm(&k, nk, RMS_EPS).unwrap(),
                    v,
                )
            };
        let (riq, rik, riv) = old_process(&iq, &ik, &iv, &nq, &nk);
        let (rtq, rtk, rtv) = old_process(&tq, &tk, &tv, &anq, &ank);
        let cat = |i: &Array, t: &Array| concatenate_axis(&[i, t], 2).unwrap();
        let (ref_q, ref_k, ref_v) = (cat(&riq, &rtq), cat(&rik, &rtk), cat(&riv, &rtv));

        // ── migrated.
        let process = |q: &Array, k: &Array, v: &Array, nq: &Array, nk: &Array| {
            let spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(QkNormSpec::per_head(nq, nk, RMS_EPS))
                .with_rotation_axes(RotationAxes::HeadMajor);
            qkv::prepare(QkvSource::Separate { q, k, v }, &spec).unwrap()
        };
        let image = process(&iq, &ik, &iv, &nq, &nk);
        let text = process(&tq, &tk, &tv, &anq, &ank);
        let joint = StreamOrder::ImageFirst.join(&image, &text).unwrap();

        assert_eq!(joint.q.shape(), [b, HEADS, img_seq + txt_seq, HEAD_DIM]);
        assert_same("sd3 q", dt, &ref_q, &joint.q);
        assert_same("sd3 k", dt, &ref_k, &joint.k);
        assert_same("sd3 v", dt, &ref_v, &joint.v);

        // The post-SDPA merge SD3 now shares.
        assert_same(
            "sd3 merge_heads",
            dt,
            &joint
                .v
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap()
                .reshape(&[b, img_seq + txt_seq, HEADS * HEAD_DIM])
                .unwrap(),
            &qkv::merge_heads(&joint.v).unwrap(),
        );

        // ── negative controls: `[txt, img]`, and a dropped QK-norm.
        let wrong_order = StreamOrder::TextFirst.join(&image, &text).unwrap();
        assert_differs("sd3 stream order", dt, &ref_q, &wrong_order.q);
        let unnormed = qkv::prepare(
            QkvSource::Separate {
                q: &iq,
                k: &ik,
                v: &iv,
            },
            &AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(QkNormSpec::none())
                .with_rotation_axes(RotationAxes::HeadMajor),
        )
        .unwrap();
        assert_differs("sd3 dropped qk-norm", dt, &riq, &unnormed.q);
    }
}

// ── Mochi ────────────────────────────────────────────────────────────────────────────────────

/// **Mochi-1** — `mlx-gen-mochi/src/transformer.rs`.
///
/// Geometry: `num_heads = 24`, `head_dim = 128` (`mlx-gen-mochi/src/transformer.rs:129-130`,
/// `inner_dim = 3072`), `QK_NORM_EPS = 1e-5` (`:150`), the **eager** f32 RMSNorm formulation
/// (`rms_weightless` at `:275-279` + a weight multiply at `:283-286` ⇒ [`NormDtype::EagerF32`]),
/// **per-head** RoPE tables `[seq, heads, head_dim/2]` (`mlx-gen-mochi/src/rope.rs:27-33`) on the
/// **visual stream only**, `[visual, text]` join, and a final cast to the compute dtype.
#[test]
fn mochi_asymm_attention_prologue_is_bit_exact() {
    const HEADS: i32 = 24;
    const HEAD_DIM: i32 = 128;
    const QK_NORM_EPS: f32 = 1e-5;
    let (b, sv, st) = (2, 6, 4);
    let half = HEAD_DIM / 2;

    // `rms_weightless` + the weight multiply (`MochiRMSNorm(dim_head, eps, True)`), verbatim.
    let old_rms_weighted = |x: &Array, weight: &Array| -> Array {
        let xf = x.as_dtype(Dtype::Float32).unwrap();
        let ms = mean_axis(xf.square().unwrap(), -1, true).unwrap();
        let normed = multiply(
            &xf,
            rsqrt(add(&ms, Array::from_f32(QK_NORM_EPS)).unwrap()).unwrap(),
        )
        .unwrap();
        multiply(&normed, weight.as_dtype(Dtype::Float32).unwrap()).unwrap()
    };
    // `MochiRope::apply`, verbatim.
    let old_rope = |x: &Array, cos: &Array, sin: &Array| -> Array {
        let sh = x.shape();
        let (b, s, n, d) = (sh[0], sh[1], sh[2], sh[3]);
        let half = d / 2;
        let x5 = x
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[b, s, n, half, 2])
            .unwrap();
        let parts = split(&x5, 2, 4).unwrap();
        let x_even = parts[0].reshape(&[b, s, n, half]).unwrap();
        let x_odd = parts[1].reshape(&[b, s, n, half]).unwrap();
        let (out_even, out_odd) = rope_rotate(&x_even, &x_odd, cos, sin).unwrap();
        let e5 = out_even.reshape(&[b, s, n, half, 1]).unwrap();
        let o5 = out_odd.reshape(&[b, s, n, half, 1]).unwrap();
        concatenate_axis(&[&e5, &o5], 4)
            .unwrap()
            .reshape(&[b, s, n, d])
            .unwrap()
    };

    for dt in DTYPES {
        let inner = HEADS * HEAD_DIM;
        let mk = |s: i32, scale: f32, off: f32| seq_at(&[b, s, inner], scale, off, dt);
        let (vq, vk, vv) = (
            mk(sv, 0.0007, 0.1),
            mk(sv, 0.0009, 0.5),
            mk(sv, 0.0011, 0.9),
        );
        let (tq, tk, tv) = (
            mk(st, 0.0013, 0.3),
            mk(st, 0.0017, 0.7),
            mk(st, 0.0019, 1.1),
        );
        let nq = seq_at(&[HEAD_DIM], 0.23, 0.2, dt);
        let nk = seq_at(&[HEAD_DIM], 0.29, 0.6, dt);
        let anq = seq_at(&[HEAD_DIM], 0.31, 1.0, dt);
        let ank = seq_at(&[HEAD_DIM], 0.37, 1.4, dt);
        // Per-head tables — `pos_frequencies` is `[3, heads, head_dim/2]`, so every head rotates by
        // its own angles.
        let cos = seq(&[sv, HEADS, half], 0.007, 0.0);
        let sin = seq(&[sv, HEADS, half], 0.005, 0.6);
        // The compute dtype `to_v.compute_dtype()` casts to before SDPA.
        let cd = dt;

        // ── reference.
        let to_heads = |x: &Array| {
            let s = x.shape()[1];
            x.reshape(&[b, s, HEADS, HEAD_DIM]).unwrap()
        };
        let q = old_rope(&old_rms_weighted(&to_heads(&vq), &nq), &cos, &sin);
        let k = old_rope(&old_rms_weighted(&to_heads(&vk), &nk), &cos, &sin);
        let v = to_heads(&vv);
        let eq = old_rms_weighted(&to_heads(&tq), &anq);
        let ek = old_rms_weighted(&to_heads(&tk), &ank);
        let ev = to_heads(&tv);
        let t = |a: &Array| {
            a.transpose_axes(&[0, 2, 1, 3])
                .unwrap()
                .as_dtype(cd)
                .unwrap()
        };
        let ref_q = concatenate_axis(&[&t(&q), &t(&eq)], 2).unwrap();
        let ref_k = concatenate_axis(&[&t(&k), &t(&ek)], 2).unwrap();
        let ref_v = concatenate_axis(&[&t(&v), &t(&ev)], 2).unwrap();

        // ── migrated.
        let visual_spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
            .with_qk_norm(
                QkNormSpec::per_head(&nq, &nk, QK_NORM_EPS).with_dtype(NormDtype::EagerF32),
            )
            .with_rope(RopeSpec {
                style: RopeStyle::AdjacentPair,
                q: Some(RopeTables::new(&cos, &sin)),
                k: Some(RopeTables::new(&cos, &sin)),
                // Knob 12 — Mochi is `Promoted`, a no-op here because `NormDtype::EagerF32` has
                // already left q/k in f32 before the rotation.
                dtype: RopeDtype::Promoted,
                ..RopeSpec::default()
            });
        let vis_heads = qkv::prepare(
            QkvSource::Separate {
                q: &vq,
                k: &vk,
                v: &vv,
            },
            &visual_spec,
        )
        .unwrap();
        let text_spec = AttnPrepSpec::new(HEADS, HEAD_DIM).with_qk_norm(
            QkNormSpec::per_head(&anq, &ank, QK_NORM_EPS).with_dtype(NormDtype::EagerF32),
        );
        let txt_heads = qkv::prepare(
            QkvSource::Separate {
                q: &tq,
                k: &tk,
                v: &tv,
            },
            &text_spec,
        )
        .unwrap();
        let cast = |s: QkvHeads| QkvHeads {
            q: s.q.as_dtype(cd).unwrap(),
            k: s.k.as_dtype(cd).unwrap(),
            v: s.v.as_dtype(cd).unwrap(),
        };
        let joint = StreamOrder::ImageFirst
            .join(&cast(vis_heads), &cast(txt_heads))
            .unwrap();

        assert_eq!(joint.q.shape(), [b, HEADS, sv + st, HEAD_DIM]);
        assert_same("mochi q", dt, &ref_q, &joint.q);
        assert_same("mochi k", dt, &ref_k, &joint.k);
        assert_same("mochi v", dt, &ref_v, &joint.v);

        // ── negative controls: the text stream must stay unrotated (knob 5), and the eager f32
        // norm formulation is not interchangeable with MLX's fused kernel on a bf16 stream.
        let rotated_text = qkv::prepare(
            QkvSource::Separate {
                q: &tq,
                k: &tk,
                v: &tv,
            },
            &AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(
                    QkNormSpec::per_head(&anq, &ank, QK_NORM_EPS).with_dtype(NormDtype::EagerF32),
                )
                .with_rope(RopeSpec {
                    style: RopeStyle::AdjacentPair,
                    q: Some(RopeTables::new(
                        &seq(&[st, HEADS, half], 0.007, 0.0),
                        &seq(&[st, HEADS, half], 0.005, 0.6),
                    )),
                    k: None,
                    dtype: RopeDtype::Promoted,
                    ..RopeSpec::default()
                }),
        )
        .unwrap();
        assert_differs(
            "mochi text stream rotated",
            dt,
            &eq.transpose_axes(&[0, 2, 1, 3]).unwrap(),
            &rotated_text.q,
        );
    }
}

// ── Mage ─────────────────────────────────────────────────────────────────────────────────────

/// **Mage-Flow** — `mlx-gen-mage/src/attention.rs`.
///
/// Geometry: `num_heads = 24`, `head_dim = hidden_size / num_heads = 128`
/// (`mlx-gen-mage/src/config.rs:385`, `:393-396` — hidden 3072), `NORM_EPS = 1e-6`
/// (`mlx-gen-mage/src/config.rs:118`), msrope table `[img_tokens, head_dim/2]`
/// (`mlx-gen-mage/src/rope_embedder.rs:298-307`) applied to the **image stream only** (knob 5), and
/// the `[text, image]` join (knob 11). The packed streams are `[1, tokens, dim]`.
#[test]
fn mage_joint_attention_prologue_is_bit_exact() {
    const HEADS: i32 = 24;
    const HEAD_DIM: i32 = 128;
    const NORM_EPS: f32 = 1e-6;
    let (img_tokens, txt_tokens) = (6, 4);
    let half = HEAD_DIM / 2;

    // The removed `apply_rope` (`apply_rotary_emb_mageflow`), verbatim — rank-3
    // `[tokens, heads, head_dim]`.
    let old_rope = |x: &Array, cos: &Array, sin: &Array| -> Array {
        let sh = x.shape();
        let (tokens, heads, head_dim) = (sh[0], sh[1], sh[2]);
        let half = head_dim / 2;
        let dtype = x.dtype();
        let pairs = x
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[tokens, heads, half, 2])
            .unwrap();
        let parts = split(&pairs, 2, 3).unwrap();
        let real = parts[0].reshape(&[tokens, heads, half]).unwrap();
        let imag = parts[1].reshape(&[tokens, heads, half]).unwrap();
        // `freqs_cis.unsqueeze(1)` — one table row per token, broadcast across heads.
        let cos = cos.reshape(&[tokens, 1, half]).unwrap();
        let sin = sin.reshape(&[tokens, 1, half]).unwrap();
        let (out_real, out_imag) = rope_rotate(&real, &imag, &cos, &sin).unwrap();
        stack_axis(&[out_real, out_imag], 3)
            .unwrap()
            .reshape(&[tokens, heads, head_dim])
            .unwrap()
            .as_dtype(dtype)
            .unwrap()
    };

    for dt in DTYPES {
        let dim = HEADS * HEAD_DIM;
        let mk = |tokens: i32, scale: f32, off: f32| seq_at(&[1, tokens, dim], scale, off, dt);
        let (iq, ik, iv) = (
            mk(img_tokens, 0.0007, 0.1),
            mk(img_tokens, 0.0009, 0.5),
            mk(img_tokens, 0.0011, 0.9),
        );
        let (tq, tk, tv) = (
            mk(txt_tokens, 0.0013, 0.3),
            mk(txt_tokens, 0.0017, 0.7),
            mk(txt_tokens, 0.0019, 1.1),
        );
        let nq = seq_at(&[HEAD_DIM], 0.23, 0.2, dt);
        let nk = seq_at(&[HEAD_DIM], 0.29, 0.6, dt);
        let anq = seq_at(&[HEAD_DIM], 0.31, 1.0, dt);
        let ank = seq_at(&[HEAD_DIM], 0.37, 1.4, dt);
        let cos = seq(&[img_tokens, half], 0.013, 0.0);
        let sin = seq(&[img_tokens, half], 0.011, 0.5);

        // ── reference: `[1, S, D]` → `[S, heads, head_dim]` → QK-norm → msrope (image only), then
        // one segment's `[text, image]` join to `[1, heads, L, head_dim]`.
        let flat = |x: &Array, tokens: i32| x.reshape(&[tokens, HEADS, HEAD_DIM]).unwrap();
        let rimg_q = old_rope(
            &rms_norm(flat(&iq, img_tokens), &nq, NORM_EPS).unwrap(),
            &cos,
            &sin,
        );
        let rimg_k = old_rope(
            &rms_norm(flat(&ik, img_tokens), &nk, NORM_EPS).unwrap(),
            &cos,
            &sin,
        );
        let rimg_v = flat(&iv, img_tokens);
        let rtxt_q = rms_norm(flat(&tq, txt_tokens), &anq, NORM_EPS).unwrap();
        let rtxt_k = rms_norm(flat(&tk, txt_tokens), &ank, NORM_EPS).unwrap();
        let rtxt_v = flat(&tv, txt_tokens);
        let joint_ref = |t: &Array, i: &Array| -> Array {
            let cat = concatenate_axis(&[t, i], 0).unwrap();
            let sh = cat.shape();
            cat.reshape(&[1, sh[0], sh[1], sh[2]])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap()
        };
        let ref_q = joint_ref(&rtxt_q, &rimg_q);
        let ref_k = joint_ref(&rtxt_k, &rimg_k);
        let ref_v = joint_ref(&rtxt_v, &rimg_v);

        // ── migrated.
        let img_spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
            .with_qk_norm(QkNormSpec::per_head(&nq, &nk, NORM_EPS))
            .with_rope(RopeSpec {
                style: RopeStyle::AdjacentPair,
                q: Some(RopeTables::new(&cos, &sin)),
                k: Some(RopeTables::new(&cos, &sin)),
                dtype: RopeDtype::RestoreInput,
                ..RopeSpec::default()
            });
        let img = qkv::prepare(
            QkvSource::Separate {
                q: &iq,
                k: &ik,
                v: &iv,
            },
            &img_spec,
        )
        .unwrap();
        let txt_spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
            .with_qk_norm(QkNormSpec::per_head(&anq, &ank, NORM_EPS));
        let txt = qkv::prepare(
            QkvSource::Separate {
                q: &tq,
                k: &tk,
                v: &tv,
            },
            &txt_spec,
        )
        .unwrap();
        let joint = StreamOrder::TextFirst.join(&img, &txt).unwrap();

        assert_eq!(
            joint.q.shape(),
            [1, HEADS, txt_tokens + img_tokens, HEAD_DIM]
        );
        assert_same("mage q", dt, &ref_q, &joint.q);
        assert_same("mage k", dt, &ref_k, &joint.k);
        assert_same("mage v", dt, &ref_v, &joint.v);

        // ── negative controls: the `[text, image]` order, and rotating the text stream (which the
        // reference deliberately never does — `apply_text_rotary_emb: false`).
        let wrong_order = StreamOrder::ImageFirst.join(&img, &txt).unwrap();
        assert_differs("mage stream order", dt, &ref_q, &wrong_order.q);
        let rotated_txt = qkv::prepare(
            QkvSource::Separate {
                q: &tq,
                k: &tk,
                v: &tv,
            },
            &AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_qk_norm(QkNormSpec::per_head(&anq, &ank, NORM_EPS))
                .with_rope(RopeSpec {
                    style: RopeStyle::AdjacentPair,
                    q: Some(RopeTables::new(
                        &seq(&[txt_tokens, half], 0.013, 0.0),
                        &seq(&[txt_tokens, half], 0.011, 0.5),
                    )),
                    k: None,
                    dtype: RopeDtype::RestoreInput,
                    ..RopeSpec::default()
                }),
        )
        .unwrap();
        assert_differs("mage text rotated", dt, &txt.q, &rotated_txt.q);
        // Knob 12 — Mage's `.type_as(x)` is load-bearing; leaving the promotion standing changes
        // what SDPA consumes.
        if dt != Dtype::Float32 {
            let promoted = qkv::prepare(
                QkvSource::Separate {
                    q: &iq,
                    k: &ik,
                    v: &iv,
                },
                &AttnPrepSpec::new(HEADS, HEAD_DIM)
                    .with_qk_norm(QkNormSpec::per_head(&nq, &nk, NORM_EPS))
                    .with_rope(RopeSpec {
                        style: RopeStyle::AdjacentPair,
                        q: Some(RopeTables::new(&cos, &sin)),
                        k: Some(RopeTables::new(&cos, &sin)),
                        dtype: RopeDtype::Promoted,
                        ..RopeSpec::default()
                    }),
            )
            .unwrap();
            assert_differs("mage rope dtype", dt, &img.q, &promoted.q);
        }
    }
}

// ── Boogu ────────────────────────────────────────────────────────────────────────────────────

/// **Boogu** — `mlx-gen-boogu/src/transformer/block.rs`.
///
/// Geometry: `num_attention_heads = 28`, `num_kv_heads = 7` (GQA ratio 4),
/// `head_dim = hidden_size / heads = 3360 / 28 = 120` (`mlx-gen-boogu/src/config.rs:53-54`,
/// `:177-179`, asserted at `:245`), `QK_EPS = 1e-5`
/// (`mlx-gen-boogu/src/transformer/block.rs:26`), interleaved tables `[1, s, head_dim/2]` (`:125`),
/// token-major, and — knob 10 — `repeat_interleave` placed **after** the rotation.
#[test]
fn boogu_gqa_prologue_is_bit_exact() {
    const HEADS: i32 = 28;
    const KV_HEADS: i32 = 7;
    const HEAD_DIM: i32 = 120;
    const QK_EPS: f32 = 1e-5;
    let (b, s) = (1, 8);
    let half = HEAD_DIM / 2;
    let groups = HEADS / KV_HEADS;
    assert_eq!(groups, 4, "boogu's GQA ratio is 28/7");

    // The removed `apply_interleaved_rope`, verbatim.
    let old_rope = |x: &Array, cos: &Array, sin: &Array| -> Array {
        let dt = x.dtype();
        let sh = x.shape();
        let (b, s, h, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let half = hd / 2;
        let cos = cos
            .as_dtype(Dtype::Float32)
            .unwrap()
            .expand_dims(2)
            .unwrap();
        let sin = sin
            .as_dtype(Dtype::Float32)
            .unwrap()
            .expand_dims(2)
            .unwrap();
        let xr = x
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[b, s, h, half, 2])
            .unwrap();
        let parts = split(&xr, 2, 4).unwrap();
        let xe = parts[0].reshape(&[b, s, h, half]).unwrap();
        let xo = parts[1].reshape(&[b, s, h, half]).unwrap();
        let out_e = subtract(multiply(&xe, &cos).unwrap(), multiply(&xo, &sin).unwrap()).unwrap();
        let out_o = add(multiply(&xe, &sin).unwrap(), multiply(&xo, &cos).unwrap()).unwrap();
        let out = concatenate_axis(
            &[
                &out_e.expand_dims(4).unwrap(),
                &out_o.expand_dims(4).unwrap(),
            ],
            4,
        )
        .unwrap();
        out.reshape(&[b, s, h, hd]).unwrap().as_dtype(dt).unwrap()
    };
    // The removed `transformer::repeat_kv`, verbatim.
    let old_repeat_kv = |x: &Array, groups: i32| -> Array {
        if groups == 1 {
            return x.clone();
        }
        let sh = x.shape();
        let (b, s, hkv, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let x = x.expand_dims(3).unwrap();
        let x = broadcast_to(&x, &[b, s, hkv, groups, hd]).unwrap();
        x.reshape(&[b, s, hkv * groups, hd]).unwrap()
    };

    for dt in DTYPES {
        let q_proj = seq_at(&[b, s, HEADS * HEAD_DIM], 0.0007, 0.1, dt);
        let k_proj = seq_at(&[b, s, KV_HEADS * HEAD_DIM], 0.0013, 0.6, dt);
        let v_proj = seq_at(&[b, s, KV_HEADS * HEAD_DIM], 0.0017, 1.2, dt);
        let nq = seq_at(&[HEAD_DIM], 0.23, 0.2, dt);
        let nk = seq_at(&[HEAD_DIM], 0.29, 0.6, dt);
        let cos = seq(&[1, s, half], 0.019, 0.0);
        let sin = seq(&[1, s, half], 0.023, 0.4);

        // ── reference.
        let q = q_proj.reshape(&[b, s, HEADS, HEAD_DIM]).unwrap();
        let k = k_proj.reshape(&[b, s, KV_HEADS, HEAD_DIM]).unwrap();
        let v = v_proj.reshape(&[b, s, KV_HEADS, HEAD_DIM]).unwrap();
        let q = rms_norm(&q, &nq, QK_EPS).unwrap();
        let k = rms_norm(&k, &nk, QK_EPS).unwrap();
        let q = old_rope(&q, &cos, &sin);
        let k = old_rope(&k, &cos, &sin);
        let k = old_repeat_kv(&k, groups);
        let v = old_repeat_kv(&v, groups);
        let t = |a: &Array| a.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let (ref_q, ref_k, ref_v) = (t(&q), t(&k), t(&v));

        // ── migrated: `boogu_spec`.
        let spec = AttnPrepSpec::new(HEADS, HEAD_DIM)
            .with_kv_heads(KV_HEADS)
            .with_qk_norm(QkNormSpec::per_head(&nq, &nk, QK_EPS))
            .with_rope(RopeSpec {
                style: RopeStyle::AdjacentPair,
                q: Some(RopeTables::new(&cos, &sin)),
                k: Some(RopeTables::new(&cos, &sin)),
                dtype: RopeDtype::RestoreInput,
                ..RopeSpec::default()
            });
        let heads = qkv::prepare(
            QkvSource::Separate {
                q: &q_proj,
                k: &k_proj,
                v: &v_proj,
            },
            &spec,
        )
        .unwrap();

        assert_eq!(heads.k.shape(), [b, HEADS, s, HEAD_DIM]);
        assert_same("boogu q", dt, &ref_q, &heads.q);
        assert_same("boogu k", dt, &ref_k, &heads.k);
        assert_same("boogu v", dt, &ref_v, &heads.v);

        // ── negative control: repeat-KV placed BEFORE the rotation is a different (wrong) model —
        // the RoPE table is indexed per kv head, so repeating first rotates `groups` copies. Build
        // it by repeating the projection and declaring `kv_heads == heads`.
        let pre_repeated = {
            let k_wide = old_repeat_kv(
                &k_proj.reshape(&[b, s, KV_HEADS, HEAD_DIM]).unwrap(),
                groups,
            )
            .reshape(&[b, s, HEADS * HEAD_DIM])
            .unwrap();
            let v_wide = old_repeat_kv(
                &v_proj.reshape(&[b, s, KV_HEADS, HEAD_DIM]).unwrap(),
                groups,
            )
            .reshape(&[b, s, HEADS * HEAD_DIM])
            .unwrap();
            qkv::prepare(
                QkvSource::Separate {
                    q: &q_proj,
                    k: &k_wide,
                    v: &v_wide,
                },
                &AttnPrepSpec::new(HEADS, HEAD_DIM)
                    .with_qk_norm(QkNormSpec::per_head(&nq, &nk, QK_EPS))
                    .with_rope(RopeSpec {
                        style: RopeStyle::AdjacentPair,
                        q: Some(RopeTables::new(&cos, &sin)),
                        k: None,
                        dtype: RopeDtype::RestoreInput,
                        ..RopeSpec::default()
                    }),
            )
            .unwrap()
        };
        assert_differs("boogu repeat-then-rotate", dt, &ref_k, &pre_repeated.k);
        // …and a dropped k table.
        let no_k_table = qkv::prepare(
            QkvSource::Separate {
                q: &q_proj,
                k: &k_proj,
                v: &v_proj,
            },
            &AttnPrepSpec::new(HEADS, HEAD_DIM)
                .with_kv_heads(KV_HEADS)
                .with_qk_norm(QkNormSpec::per_head(&nq, &nk, QK_EPS))
                .with_rope(RopeSpec {
                    style: RopeStyle::AdjacentPair,
                    q: Some(RopeTables::new(&cos, &sin)),
                    k: None,
                    dtype: RopeDtype::RestoreInput,
                    ..RopeSpec::default()
                }),
        )
        .unwrap();
        assert_differs("boogu dropped k table", dt, &ref_k, &no_k_table.k);
        // Knob 12 — `rope.rs`'s `.as_dtype(dt)` is load-bearing.
        if dt != Dtype::Float32 {
            let promoted = qkv::prepare(
                QkvSource::Separate {
                    q: &q_proj,
                    k: &k_proj,
                    v: &v_proj,
                },
                &AttnPrepSpec::new(HEADS, HEAD_DIM)
                    .with_kv_heads(KV_HEADS)
                    .with_qk_norm(QkNormSpec::per_head(&nq, &nk, QK_EPS))
                    .with_rope(RopeSpec {
                        style: RopeStyle::AdjacentPair,
                        q: Some(RopeTables::new(&cos, &sin)),
                        k: Some(RopeTables::new(&cos, &sin)),
                        dtype: RopeDtype::Promoted,
                        ..RopeSpec::default()
                    }),
            )
            .unwrap();
            assert_differs("boogu rope dtype", dt, &ref_q, &promoted.q);
        }
    }
}

// ── Anima ────────────────────────────────────────────────────────────────────────────────────

/// **Anima** — `mlx-gen-anima/src/transformer.rs` (DiT) and `.../conditioner.rs`.
///
/// DiT geometry: `num_attention_heads = 16`, `attention_head_dim = 128`
/// (`mlx-gen-anima/src/config.rs:33-34`, hidden 2048), `ATTN_QK_NORM_EPS = 1e-5`
/// (`mlx-gen-anima/src/transformer.rs:32`), half-split `rotate_half` over full-width `[1, S, hd]`
/// tables (`mlx_gen::nn::apply_text_rope`), token-major.
/// Conditioner geometry: `model_dim = 1024` / `num_attention_heads = 16` ⇒ `head_dim = 64`,
/// `norm_eps = 1e-6` (`mlx-gen-anima/src/config.rs:85`, `:90`, `:94-96`) — and **knob 6's reference
/// case**: `q_rope` and `k_rope` are genuinely different tables.
#[test]
fn anima_dit_and_conditioner_prologues_are_bit_exact() {
    const DIT_HEADS: i32 = 16;
    const DIT_HEAD_DIM: i32 = 128;
    const DIT_EPS: f32 = 1e-5;
    const COND_HEADS: i32 = 16;
    const COND_HEAD_DIM: i32 = 64;
    const COND_EPS: f32 = 1e-6;
    let (b, sq, sk) = (2, 6, 4);

    for dt in DTYPES {
        // ── DiT self-attention (RoPE on) ───────────────────────────────────────────────────
        let inner = DIT_HEADS * DIT_HEAD_DIM;
        let q_proj = seq_at(&[b, sq, inner], 0.0007, 0.1, dt);
        let k_proj = seq_at(&[b, sq, inner], 0.0009, 0.5, dt);
        let v_proj = seq_at(&[b, sq, inner], 0.0011, 0.9, dt);
        let nq = seq_at(&[DIT_HEAD_DIM], 0.23, 0.2, dt);
        let nk = seq_at(&[DIT_HEAD_DIM], 0.29, 0.6, dt);
        let cos = seq(&[1, sq, DIT_HEAD_DIM], 0.013, 0.0);
        let sin = seq(&[1, sq, DIT_HEAD_DIM], 0.011, 0.7);

        // reference: reshape → per-head rms_norm → `apply_text_rope` → transpose.
        let split_h = |x: &Array, s: i32, h: i32, hd: i32| x.reshape(&[b, s, h, hd]).unwrap();
        let rq = rms_norm(split_h(&q_proj, sq, DIT_HEADS, DIT_HEAD_DIM), &nq, DIT_EPS).unwrap();
        let rk = rms_norm(split_h(&k_proj, sq, DIT_HEADS, DIT_HEAD_DIM), &nk, DIT_EPS).unwrap();
        let rv = split_h(&v_proj, sq, DIT_HEADS, DIT_HEAD_DIM);
        let rq = apply_text_rope(&rq, &cos, &sin).unwrap();
        let rk = apply_text_rope(&rk, &cos, &sin).unwrap();
        let t = |a: &Array| a.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let (ref_q, ref_k, ref_v) = (t(&rq), t(&rk), t(&rv));

        // migrated.
        let spec = AttnPrepSpec::new(DIT_HEADS, DIT_HEAD_DIM)
            .with_qk_norm(QkNormSpec::per_head(&nq, &nk, DIT_EPS))
            .with_rope(RopeSpec {
                style: RopeStyle::RotateHalf,
                q: Some(RopeTables::new(&cos, &sin)),
                k: Some(RopeTables::new(&cos, &sin)),
                // Knob 12 — `nn::apply_text_rope` is a bare `x·cos + rotate_half(x)·sin` with no
                // cast back, so the f32 Cosmos tables' promotion reaches SDPA.
                dtype: RopeDtype::Promoted,
                ..RopeSpec::default()
            });
        let heads = qkv::prepare(
            QkvSource::Separate {
                q: &q_proj,
                k: &k_proj,
                v: &v_proj,
            },
            &spec,
        )
        .unwrap();
        assert_eq!(heads.q.shape(), [b, DIT_HEADS, sq, DIT_HEAD_DIM]);
        assert_same("anima dit q", dt, &ref_q, &heads.q);
        assert_same("anima dit k", dt, &ref_k, &heads.k);
        assert_same("anima dit v", dt, &ref_v, &heads.v);

        // ── DiT cross-attention (knob 3: `pe == None` ⇒ RoPE skipped outright) ─────────────
        let no_rope_ref_q =
            t(&rms_norm(split_h(&q_proj, sq, DIT_HEADS, DIT_HEAD_DIM), &nq, DIT_EPS).unwrap());
        let no_rope = qkv::prepare(
            QkvSource::Separate {
                q: &q_proj,
                k: &k_proj,
                v: &v_proj,
            },
            &AttnPrepSpec::new(DIT_HEADS, DIT_HEAD_DIM)
                .with_qk_norm(QkNormSpec::per_head(&nq, &nk, DIT_EPS)),
        )
        .unwrap();
        assert_same("anima dit cross-attn q", dt, &no_rope_ref_q, &no_rope.q);
        // Negative control — the RoPE arm must genuinely differ from the no-RoPE arm.
        assert_differs("anima rope disabled", dt, &ref_q, &no_rope.q);

        // ── Conditioner cross-attention (knob 6: separate q/k tables) ─────────────────────
        let c_inner = COND_HEADS * COND_HEAD_DIM;
        let cq = seq_at(&[b, sq, c_inner], 0.0021, 0.15, dt);
        let ck = seq_at(&[b, sk, c_inner], 0.0023, 0.55, dt);
        let cv = seq_at(&[b, sk, c_inner], 0.0027, 0.95, dt);
        let cnq = seq_at(&[COND_HEAD_DIM], 0.31, 1.0, dt);
        let cnk = seq_at(&[COND_HEAD_DIM], 0.37, 1.4, dt);
        let q_cos = seq(&[1, sq, COND_HEAD_DIM], 0.017, 0.0);
        let q_sin = seq(&[1, sq, COND_HEAD_DIM], 0.019, 0.3);
        let k_cos = seq(&[1, sk, COND_HEAD_DIM], 0.029, 0.8);
        let k_sin = seq(&[1, sk, COND_HEAD_DIM], 0.031, 1.2);

        let rcq = rms_norm(split_h(&cq, sq, COND_HEADS, COND_HEAD_DIM), &cnq, COND_EPS).unwrap();
        let rck = rms_norm(split_h(&ck, sk, COND_HEADS, COND_HEAD_DIM), &cnk, COND_EPS).unwrap();
        let rcv = split_h(&cv, sk, COND_HEADS, COND_HEAD_DIM);
        let rcq = apply_text_rope(&rcq, &q_cos, &q_sin).unwrap();
        let rck = apply_text_rope(&rck, &k_cos, &k_sin).unwrap();
        let (cref_q, cref_k, cref_v) = (t(&rcq), t(&rck), t(&rcv));

        let cond = qkv::prepare(
            QkvSource::Separate {
                q: &cq,
                k: &ck,
                v: &cv,
            },
            &AttnPrepSpec::new(COND_HEADS, COND_HEAD_DIM)
                .with_qk_norm(QkNormSpec::per_head(&cnq, &cnk, COND_EPS))
                .with_rope(RopeSpec {
                    style: RopeStyle::RotateHalf,
                    q: Some(RopeTables::new(&q_cos, &q_sin)),
                    k: Some(RopeTables::new(&k_cos, &k_sin)),
                    dtype: RopeDtype::Promoted,
                    ..RopeSpec::default()
                }),
        )
        .unwrap();
        assert_same("anima conditioner q", dt, &cref_q, &cond.q);
        assert_same("anima conditioner k", dt, &cref_k, &cond.k);
        assert_same("anima conditioner v", dt, &cref_v, &cond.v);

        // Negative control — reusing the query's table on the key stream (knob 6 collapsed) must
        // change the key stream. The two tables are different lengths, so this also proves the
        // primitive is indexing the key stream by its OWN token count.
        let shared_table = qkv::prepare(
            QkvSource::Separate {
                q: &ck,
                k: &ck,
                v: &cv,
            },
            &AttnPrepSpec::new(COND_HEADS, COND_HEAD_DIM)
                .with_qk_norm(QkNormSpec::per_head(&cnq, &cnk, COND_EPS))
                .with_rope(RopeSpec {
                    style: RopeStyle::RotateHalf,
                    q: Some(RopeTables::new(&k_cos, &k_sin)),
                    k: Some(RopeTables::new(
                        &seq(&[1, sk, COND_HEAD_DIM], 0.017, 0.0),
                        &seq(&[1, sk, COND_HEAD_DIM], 0.019, 0.3),
                    )),
                    dtype: RopeDtype::Promoted,
                    ..RopeSpec::default()
                }),
        )
        .unwrap();
        assert_differs("anima shared q/k table", dt, &cref_k, &shared_table.k);

        // Knob 12 — casting back to bf16 would hand SDPA a different tensor. This is the control
        // for the divergence these fixtures found before `RopeDtype` existed.
        if dt != Dtype::Float32 {
            let restored = qkv::prepare(
                QkvSource::Separate {
                    q: &q_proj,
                    k: &k_proj,
                    v: &v_proj,
                },
                &AttnPrepSpec::new(DIT_HEADS, DIT_HEAD_DIM)
                    .with_qk_norm(QkNormSpec::per_head(&nq, &nk, DIT_EPS))
                    .with_rope(RopeSpec {
                        style: RopeStyle::RotateHalf,
                        q: Some(RopeTables::new(&cos, &sin)),
                        k: Some(RopeTables::new(&cos, &sin)),
                        dtype: RopeDtype::RestoreInput,
                        ..RopeSpec::default()
                    }),
            )
            .unwrap();
            assert_differs("anima rope dtype", dt, &ref_q, &restored.q);
        }
    }
}

// ── LTX ──────────────────────────────────────────────────────────────────────────────────────

/// **LTX-2** — `mlx-gen-ltx/src/transformer.rs`.
///
/// Geometry: `num_attention_heads = 32`, `attention_head_dim = 128`
/// (`mlx-gen-ltx/src/config.rs:118-119` — video inner 4096), `norm_eps = 1e-6` (`:130`, passed as
/// the q/k-RMSNorm epsilon at `mlx-gen-ltx/src/transformer.rs:937-943`). LTX forced two knobs:
/// **`FullDimPreSplit`** (the RMSNorm reduces over the whole `heads · dim_head` projection, before
/// the head split) and **knob 6** (`k_pe` is an independent key table). Plus knob 3 (`pe == None`)
/// and knob 2's `HalvesPaired` arm over pre-broadcast `[B, H, T, dim_head/2]` tables
/// (`mlx-gen-ltx/src/rope.rs:208-228`, `:241-242`).
#[test]
fn ltx_attention_prologue_is_bit_exact() {
    const HEADS: i32 = 32;
    const DIM_HEAD: i32 = 128;
    const EPS: f32 = 1e-6;
    let (b, s, ctx) = (2, 6, 4);
    let half = DIM_HEAD / 2;

    // `rope::apply_split_rotary_emb`, verbatim.
    let old_split_rope = |x: &Array, cos: &Array, sin: &Array| -> Array {
        let in_dtype = x.dtype();
        let x = x.as_dtype(Dtype::Float32).unwrap();
        let cos = cos.as_dtype(Dtype::Float32).unwrap();
        let sin = sin.as_dtype(Dtype::Float32).unwrap();
        let axis = (x.ndim() - 1) as i32;
        let halves = split(&x, 2, axis).unwrap();
        let (out_first, out_second) = rope_rotate(&halves[0], &halves[1], &cos, &sin).unwrap();
        concatenate_axis(&[&out_first, &out_second], axis)
            .unwrap()
            .as_dtype(in_dtype)
            .unwrap()
    };

    for dt in DTYPES {
        let inner = HEADS * DIM_HEAD;
        let q_proj = seq_at(&[b, s, inner], 0.0005, 0.1, dt);
        let k_proj = seq_at(&[b, ctx, inner], 0.0007, 0.5, dt);
        let v_proj = seq_at(&[b, ctx, inner], 0.0009, 0.9, dt);
        // Full-dim QK-norm weights: `[heads · dim_head]`, NOT `[dim_head]`.
        let q_norm = seq_at(&[inner], 0.0003, 0.2, dt);
        let k_norm = seq_at(&[inner], 0.0004, 0.6, dt);
        // Pre-broadcast per-batch AND per-head tables.
        let q_cos = seq(&[b, HEADS, s, half], 0.0021, 0.0);
        let q_sin = seq(&[b, HEADS, s, half], 0.0023, 0.4);
        let k_cos = seq(&[b, HEADS, ctx, half], 0.0031, 0.8);
        let k_sin = seq(&[b, HEADS, ctx, half], 0.0033, 1.2);

        // ── reference: full-dim RMSNorm → `to_heads` → `apply_split_rotary_emb`.
        let to_heads = |x: &Array| {
            let sh = x.shape();
            x.reshape(&[sh[0], sh[1], HEADS, DIM_HEAD])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap()
        };
        let q = rms_norm(&q_proj, &q_norm, EPS).unwrap();
        let k = rms_norm(&k_proj, &k_norm, EPS).unwrap();
        let ref_v = to_heads(&v_proj);
        let ref_q = old_split_rope(&to_heads(&q), &q_cos, &q_sin);
        let ref_k = old_split_rope(&to_heads(&k), &k_cos, &k_sin);

        // ── migrated.
        let spec = AttnPrepSpec::new(HEADS, DIM_HEAD)
            .with_qk_norm(QkNormSpec::full_dim_pre_split(&q_norm, &k_norm, EPS))
            .with_rope(RopeSpec {
                style: RopeStyle::HalvesPaired,
                q: Some(RopeTables::new(&q_cos, &q_sin)),
                k: Some(RopeTables::new(&k_cos, &k_sin)),
                dtype: RopeDtype::RestoreInput,
                ..RopeSpec::default()
            })
            .with_rotation_axes(RotationAxes::HeadMajor);
        let heads = qkv::prepare(
            QkvSource::Separate {
                q: &q_proj,
                k: &k_proj,
                v: &v_proj,
            },
            &spec,
        )
        .unwrap();

        assert_eq!(heads.q.shape(), [b, HEADS, s, DIM_HEAD]);
        assert_eq!(heads.k.shape(), [b, HEADS, ctx, DIM_HEAD]);
        assert_same("ltx q", dt, &ref_q, &heads.q);
        assert_same("ltx k", dt, &ref_k, &heads.k);
        assert_same("ltx v", dt, &ref_v, &heads.v);

        // ── knob 3: `pe == None` ⇒ no rotation on either stream (text cross-attention).
        let no_rope_ref_q = to_heads(&rms_norm(&q_proj, &q_norm, EPS).unwrap());
        let no_rope = qkv::prepare(
            QkvSource::Separate {
                q: &q_proj,
                k: &k_proj,
                v: &v_proj,
            },
            &AttnPrepSpec::new(HEADS, DIM_HEAD)
                .with_qk_norm(QkNormSpec::full_dim_pre_split(&q_norm, &k_norm, EPS))
                .with_rotation_axes(RotationAxes::HeadMajor),
        )
        .unwrap();
        assert_same("ltx no-rope q", dt, &no_rope_ref_q, &no_rope.q);

        // ── negative controls: the rotation is real, and `HalvesPaired` is not `AdjacentPair`.
        assert_differs("ltx rope disabled", dt, &ref_q, &no_rope.q);
        let adjacent = qkv::prepare(
            QkvSource::Separate {
                q: &q_proj,
                k: &k_proj,
                v: &v_proj,
            },
            &AttnPrepSpec::new(HEADS, DIM_HEAD)
                .with_qk_norm(QkNormSpec::full_dim_pre_split(&q_norm, &k_norm, EPS))
                .with_rope(RopeSpec {
                    style: RopeStyle::AdjacentPair,
                    q: Some(RopeTables::new(&q_cos, &q_sin)),
                    k: Some(RopeTables::new(&k_cos, &k_sin)),
                    dtype: RopeDtype::RestoreInput,
                    ..RopeSpec::default()
                })
                .with_rotation_axes(RotationAxes::HeadMajor),
        )
        .unwrap();
        assert_differs("ltx rotation style", dt, &ref_q, &adjacent.q);
        // Knob 12 — `apply_split_rotary_emb`'s `.astype(input_dtype)` is load-bearing: it is what
        // keeps a bf16 DiT (and the bf16 connector) bf16 through SDPA.
        if dt != Dtype::Float32 {
            let promoted = qkv::prepare(
                QkvSource::Separate {
                    q: &q_proj,
                    k: &k_proj,
                    v: &v_proj,
                },
                &AttnPrepSpec::new(HEADS, DIM_HEAD)
                    .with_qk_norm(QkNormSpec::full_dim_pre_split(&q_norm, &k_norm, EPS))
                    .with_rope(RopeSpec {
                        style: RopeStyle::HalvesPaired,
                        q: Some(RopeTables::new(&q_cos, &q_sin)),
                        k: Some(RopeTables::new(&k_cos, &k_sin)),
                        dtype: RopeDtype::Promoted,
                        ..RopeSpec::default()
                    })
                    .with_rotation_axes(RotationAxes::HeadMajor),
            )
            .unwrap();
            assert_differs("ltx rope dtype", dt, &ref_q, &promoted.q);
        }
    }
}
