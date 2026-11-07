//! sc-12438 / sc-12746 — empirical MLX large-output write probes (RUN ON METAL, MLX 0.32.0).
//!
//! `#[ignore]`d: each allocates multi-GB single-buffer arrays straddling the `i32::MAX`-element
//! boundary. They exist to establish — on *this* pinned runtime (MLX core 0.32.0 via pmetal-mlx-rs
//! `eb76c4ba`, re-probed 2026-08-11) — the exact per-operation behaviour the tiling fix depends on.
//! On this pin the
//! `pad`/`concat` copy-gate overflow is FIXED (sc-12746, pad-copy-int64.patch): those probes now
//! assert EXACT writes above 2^31, not merely below. **sc-12748 (slice 6):**
//! `conv3d_8to128_output_across_i32max` now asserts the FIXED behaviour too — MLX 0.32.0 fixes the
//! conv output-offset overflow (upstream #3524 promoted the per-thread offset to `size_t`), so on
//! this pin conv3d is EXACT above 2^31 (measured `first_bad_offset=-1`, 216 positions checked below
//! and above the bound). The old 0.31.2 *corruption* claim inverted here; the conv-stage write-guard
//! retirement it justified is done in sc-12748. Every op the tiled/untiled decode touches above the
//! bound (conv3d, pad, concat, multi-dimensional reshape, as_slice, elementwise) is now
//! probe-verified int64-safe; a single `reshape(-1)` dimension remains rejected as documented below.
//! **sc-12926:** `conv3d_128ch_input_across_i32max` closes the read-side gap — a conv whose INPUT
//! (not just output) crosses 2^31, Mochi's real over-bound geometry (`cin=128` padded full-res).
//! Run explicitly:
//!
//! ```text
//! cargo test -p mlx-gen --test integration mlx_write_bound_probe:: -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Probe rules (avoid the false greens that hid this): **position-dependent** data (`i % 251`);
//! compare only sub-bound slices / individual elements (never a reduction or an op *over* the oversized
//! tensor); read back with `as_slice` (the one large-array operation MLX does correctly) and compare on
//! the host after a single `eval`.
//!
//! Findings feed the design of `mlx_gen::vae_tiling::tiled_decode`'s safe path and the doc rewrite.

use mlx_rs::ops::{add, conv3d, multiply, pad};
use mlx_rs::Array;

const I32_MAX: i64 = i32::MAX as i64; // 2_147_483_647

/// Position-dependent value for flat index `i`. Range 0..=250, exact in f32, non-constant along every
/// axis so a scrambled output index lands on a different value with prob. ~250/251.
fn pv(i: i64) -> f32 {
    (i % 251) as f32
}

/// **The core claim.** `conv3d` 8→128 whose OUTPUT crosses `i32::MAX` (the story's exact
/// `128×D×480×848` geometry, D=41 below / D=42 above). A pointwise (1×1×1) conv with a permutation
/// weight makes `out[..,co] = in[..,co%8]`, so every output element is predictable. On MLX 0.31.2 the
/// per-thread output offset overflowed int32 and this conv corrupted above the bound; MLX 0.32.0's
/// upstream #3524 promotes that offset to `size_t`, so **on this pin (0.32.0 fork `eb76c4ba`) the conv
/// is EXACT both below AND above the bound** (sc-12748). This probe therefore now asserts the *fixed*
/// behaviour — the inversion of the 0.31.2 corruption claim — which is what retires the conv-stage
/// write guard (`writable_frame_cap` / Mochi's decode guard).
#[test]
#[ignore = "sc-12438/sc-12748 heavy MLX conv write probe (~9 GB); run with --ignored on Metal"]
fn conv3d_8to128_output_across_i32max() {
    let (h, w): (i64, i64) = (480, 848);
    let (cin, cout): (i64, i64) = (8, 128);
    // weight [cout,1,1,1,cin]: W[co,ci] = 1 iff ci == co%cin → out[..,co] = in[..,co%cin].
    let mut wbuf = vec![0f32; (cout * cin) as usize];
    for co in 0..cout {
        wbuf[(co * cin + co % cin) as usize] = 1.0;
    }
    let weight = Array::from_slice(&wbuf, &[cout as i32, 1, 1, 1, cin as i32]);
    weight.eval().unwrap();

    let run = |d: i64| -> (i64, i64, i64) {
        let in_elems = d * h * w * cin;
        let xhost: Vec<f32> = (0..in_elems).map(pv).collect();
        let x = Array::from_slice(&xhost, &[1, d as i32, h as i32, w as i32, cin as i32]);
        x.eval().unwrap();
        drop(xhost);
        let y = conv3d(&x, &weight, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1).unwrap();
        y.eval().unwrap();
        assert_eq!(
            y.shape(),
            &[1, d as i32, h as i32, w as i32, cout as i32],
            "conv output shape"
        );
        let ys = y.as_slice::<f32>();
        let out_elems = d * h * w * cout;
        let mut first_bad = -1i64;
        let mut checked = 0i64;
        for dd in [0i64, d / 2, d - 2, d - 1] {
            for hh in [0i64, h / 2, h - 1] {
                for ww in [0i64, w / 2, w - 1] {
                    for co in [0i64, 1, 7, 63, 64, 127] {
                        let off = ((dd * h + hh) * w + ww) * cout + co;
                        let want = pv(((dd * h + hh) * w + ww) * cin + co % cin);
                        let got = ys[off as usize];
                        checked += 1;
                        if (got - want).abs() > 1e-3 && first_bad < 0 {
                            first_bad = off;
                        }
                    }
                }
            }
        }
        (out_elems, first_bad, checked)
    };

    let (be, bf, bc) = run(41); // 2_136_145_920 < 2^31
    eprintln!(
        "[conv D=41 below] out_elems={be} (={:.4}× 2^31)  checked={bc}  first_bad_offset={bf}",
        be as f64 / I32_MAX as f64
    );
    let (ae, af, ac) = run(42); // 2_188_247_040 > 2^31
    eprintln!(
        "[conv D=42 above] out_elems={ae} (={:.4}× 2^31)  checked={ac}  first_bad_offset={af} \
         (={:.4}× 2^31)",
        ae as f64 / I32_MAX as f64,
        if af < 0 {
            0.0
        } else {
            af as f64 / I32_MAX as f64
        }
    );

    assert!(be < I32_MAX && ae > I32_MAX, "geometry must bracket 2^31");
    assert!(
        bf < 0,
        "conv BELOW the bound must be exact — corruption at offset {bf} breaks the premise"
    );
    // sc-12748: on MLX 0.32.0 (#3524) the conv is EXACT above the bound too. `first_bad < 0` means no
    // corruption was found at any sampled above-2^31 position (measured `af == -1`, 216 positions).
    // A regression to 0.31.2's behaviour — a future MLX bump re-breaking the size_t offset — would
    // set `af >= 0` and re-fail this, the signal to re-instate the conv-stage write guard.
    assert!(
        af < 0,
        "conv ABOVE the bound CORRUPTED (first bad offset {af}) — MLX 0.32.0's #3524 conv fix has \
         regressed on this pin. Re-instate the conv-stage write guard (writable_frame_cap / Mochi \
         decode guard) and re-open sc-12748."
    );
}

/// sc-12926 — the READ-side complement to [`conv3d_8to128_output_across_i32max`]: a `conv3d` whose
/// **INPUT crosses `i32::MAX`**. That sibling probe drives an over-bound OUTPUT from an under-bound
/// `cin=8` input, but Mochi's real over-bound conv reads a padded full-resolution `cin=128`
/// activation — input AND output both straddle the bound — previously validated only by the
/// end-to-end Mochi acceptance render, not at op level.
///
/// Geometry: `[1,41,480,848,128]` = 2_136_145_920 elems (just under 2^31, `from_slice`-able), then
/// `pad` the frame axis at the FRONT (`(1,0)`) so the LAST real frame's data sits at flat offsets
/// crossing 2^31 (a back-pad would leave only pad zeros above the bound — the read side would go
/// unexercised). Building the over-bound input uses `pad`, whose above-bound exactness is itself
/// probe-verified by [`pad_story_geometry_128x42x480x848`]. A pointwise 128→128 **shifted**
/// permutation weight (`out[..,co] = in[..,(co+1)%128]`, not the identity) keeps every element
/// predictable while catching a channel-index misread. Probe rules as the module doc: position-
/// dependent data, per-element host comparison at sampled offsets, `as_slice` read-back.
#[test]
#[ignore = "sc-12926 heavy MLX conv over-bound-INPUT probe (~27 GB); run with --ignored on Metal"]
fn conv3d_128ch_input_across_i32max() {
    let (h, w): (i64, i64) = (480, 848);
    let ch: i64 = 128;
    let d: i64 = 41;
    let in_elems = d * h * w * ch; // 2_136_145_920
    assert!(
        in_elems < I32_MAX,
        "pre-pad input must stay under the bound (from_slice)"
    );
    let xhost: Vec<f32> = (0..in_elems).map(pv).collect();
    let x = Array::from_slice(&xhost, &[1, d as i32, h as i32, w as i32, ch as i32]);
    x.eval().unwrap();
    drop(xhost);
    // Front-pad the frame axis: XP[0,0,..] = 0, XP[0,dd,..] = X[0,dd-1,..] for dd ≥ 1. The padded
    // input is 42·480·848·128 = 2_188_247_040 > 2^31, with REAL data (frames 40→dd=41) above it.
    let xp = pad(
        &x,
        &[(0, 0), (1, 0), (0, 0), (0, 0), (0, 0)][..],
        None,
        None,
    )
    .unwrap();
    xp.eval().unwrap();
    drop(x);
    let din = d + 1;
    assert!(
        din * h * w * ch > I32_MAX,
        "padded conv INPUT must cross the bound"
    );

    // weight [cout,1,1,1,cin]: W[co,ci] = 1 iff ci == (co+1)%ch → out[..,co] = in[..,(co+1)%ch].
    let mut wbuf = vec![0f32; (ch * ch) as usize];
    for co in 0..ch {
        wbuf[(co * ch + (co + 1) % ch) as usize] = 1.0;
    }
    let weight = Array::from_slice(&wbuf, &[ch as i32, 1, 1, 1, ch as i32]);
    weight.eval().unwrap();

    let y = conv3d(&xp, &weight, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1).unwrap();
    y.eval().unwrap();
    assert_eq!(
        y.shape(),
        &[1, din as i32, h as i32, w as i32, ch as i32],
        "conv output shape"
    );
    let ys = y.as_slice::<f32>();

    // Expected: Y[0,dd,hh,ww,co] = XP[0,dd,hh,ww,(co+1)%ch] = pv((((dd−1)·h+hh)·w+ww)·ch + (co+1)%ch)
    // for dd ≥ 1, else 0 (the front pad frame). At dd=41 (the last real frame) the input read offsets
    // pass 2^31 from hh ≈ 104 — those samples are the read-side proof this probe exists for.
    let mut first_bad = -1i64;
    let mut checked = 0i64;
    let mut above_input_reads = 0i64;
    let mut max_abs = 0f32;
    for dd in [0i64, 1, din / 2, din - 2, din - 1] {
        for hh in [0i64, h / 2, h - 1] {
            for ww in [0i64, w / 2, w - 1] {
                for co in [0i64, 1, 63, 126, 127] {
                    let off = ((dd * h + hh) * w + ww) * ch + co;
                    let in_off = ((dd * h + hh) * w + ww) * ch + (co + 1) % ch;
                    let want = if dd >= 1 {
                        pv((((dd - 1) * h + hh) * w + ww) * ch + (co + 1) % ch)
                    } else {
                        0.0
                    };
                    if dd >= 1 && in_off > I32_MAX {
                        above_input_reads += 1;
                    }
                    let got = ys[off as usize];
                    checked += 1;
                    let err = (got - want).abs();
                    if err > max_abs {
                        max_abs = err;
                    }
                    if err > 1e-3 && first_bad < 0 {
                        first_bad = off;
                    }
                }
            }
        }
    }
    eprintln!(
        "[conv cin=128 over-bound INPUT] in_elems={} (={:.4}× 2^31)  checked={checked}  \
         above_2^31_real_input_reads={above_input_reads}  max_abs={max_abs:e}  first_bad_offset={first_bad}",
        din * h * w * ch,
        (din * h * w * ch) as f64 / I32_MAX as f64,
    );
    assert!(
        above_input_reads > 0,
        "probe must sample REAL-data conv reads at input offsets above 2^31, or the read side is untested"
    );
    assert!(
        first_bad < 0,
        "conv reading an over-bound INPUT corrupted (first bad output offset {first_bad}, \
         max_abs={max_abs:e}) — the #3524 int64 conv fix does not cover the input/read side on this \
         pin. Re-instate the conv-stage guards and re-open sc-12748/sc-12926."
    );
}

/// The story's exact `pad` geometry: `[128,41,480,848]` → pad the frame axis `+1` → `[128,42,480,848]`
/// (output 2.188e9 > 2^31). Front data at `c≈126,127` lands above 2^31. Reports whether pad corrupts
/// the above-boundary region **on this runtime** — an earlier `[2200,1024,1024]` pad at 1.074× was
/// exact, so this checks the specific geometry the story flagged rather than assuming universality.
#[test]
#[ignore = "sc-12438 heavy MLX pad write probe (~18 GB); run with --ignored on Metal"]
fn pad_story_geometry_128x42x480x848() {
    let (c, d, h, w): (i64, i64, i64, i64) = (128, 41, 480, 848);
    let in_elems = c * d * h * w;
    assert!(
        in_elems < I32_MAX,
        "input must stay under the bound (from_slice)"
    );
    let xhost: Vec<f32> = (0..in_elems).map(pv).collect();
    let x = Array::from_slice(&xhost, &[c as i32, d as i32, h as i32, w as i32]);
    x.eval().unwrap();
    drop(xhost);

    // Pad the D (frame-like) axis by (0,1) → zeros at d=41.
    let y = pad(&x, &[(0, 0), (0, 1), (0, 0), (0, 0)][..], None, None).unwrap();
    y.eval().unwrap();
    let dout = d + 1;
    assert_eq!(y.shape(), &[c as i32, dout as i32, h as i32, w as i32]);
    let out_elems = c * dout * h * w;
    assert!(out_elems > I32_MAX, "padded output must cross the bound");
    let ys = y.as_slice::<f32>();

    // expected Y[cc,dd,hh,ww] = X[cc,dd,hh,ww] if dd<d else 0. sc-12746: on the copy-gate-fixed pin
    // (pmetal-mlx-rs eb76c4ba, pad-copy-int64.patch) pad must be EXACT above 2^31 too — not merely
    // below — so this loop now also tracks the above-bound region (first_bad_above / above_checked)
    // and max_abs.
    let mut first_bad = -1i64;
    let mut first_bad_above = -1i64;
    let mut bad = 0i64;
    let mut above_checked = 0i64;
    let mut max_abs = 0f32;
    let mut checked = 0i64;
    for cc in [0i64, 64, 120, 125, 126, 127] {
        for dd in [0i64, d / 2, d - 1, d /* = pad row (zeros) */] {
            for &(hh, ww) in &[(0i64, 0i64), (h / 2, w / 2), (h - 1, w - 1)] {
                let off = ((cc * dout + dd) * h + hh) * w + ww;
                let want = if dd < d {
                    pv(((cc * d + dd) * h + hh) * w + ww)
                } else {
                    0.0
                };
                let got = ys[off as usize];
                checked += 1;
                let err = (got - want).abs();
                if err > max_abs {
                    max_abs = err;
                }
                if off >= I32_MAX {
                    above_checked += 1;
                }
                if err > 1e-3 {
                    bad += 1;
                    if first_bad < 0 {
                        first_bad = off;
                    }
                    if off >= I32_MAX && first_bad_above < 0 {
                        first_bad_above = off;
                    }
                }
            }
        }
    }
    eprintln!(
        "[pad 128×42×480×848] out_elems={out_elems} (={:.4}× 2^31)  checked={checked}  \
         above_2^31_checked={above_checked}  bad={bad}  max_abs={max_abs:e}  \
         first_bad_offset={first_bad}  first_bad_above={first_bad_above}",
        out_elems as f64 / I32_MAX as f64,
    );
    // sc-12746: on the copy-gate-fixed pin, pad is exact across the WHOLE output. Below 2^31 was always
    // safe (check_output_writable's premise). Above 2^31 is now also exact — pad-copy-int64.patch bases
    // the copy dispatch on the addressable span (offset+Σ(shape-1)·|stride|), turning the int32
    // dst-offset overflow (unpatched 0.32.0 here: bad=24, first_bad_above=2_153_241_600) into an int64
    // copy. Both invariants must hold.
    assert!(
        above_checked > 0,
        "pad probe must sample positions above 2^31 to prove the fix"
    );
    assert!(
        !(0..I32_MAX).contains(&first_bad),
        "pad corrupted a BELOW-2^31 position (offset {first_bad}) — the write-index boundary is not at \
         2^31 and check_output_writable's 'pad is safe under the bound' premise is broken"
    );
    assert_eq!(
        first_bad_above, -1,
        "pad corrupted an ABOVE-2^31 position (offset {first_bad_above}); max_abs={max_abs:e} — the \
         pad-copy-int64 gate did not cover this geometry"
    );
    assert!(
        bad == 0,
        "pad had {bad} corrupted sample(s); max_abs={max_abs:e}"
    );
}

/// sc-12746 companion to the pad probe: `concatenate` hits the SAME `copy_gpu_inplace`
/// slice-VIEW bug. Concatenate two `[128,21,480,848]` arrays along the frame axis (1) →
/// `[128,42,480,848]` (2.188e9 > 2^31). Each input is copied into a strided slice-view of the
/// output whose true dst span is 2_179_699_199 > 2^31 while `out_slice.data_size()` = 1_094_123_520
/// < 2^31 — so the unpatched int32 `gg` kernel overflows above 2^31, and `pad-copy-int64.patch`
/// (true-span gate → int64 kernel) makes it exact. Same probe rules as the pad probe.
#[test]
#[ignore = "sc-12746 heavy MLX concat write probe (~18 GB); run with --ignored on Metal"]
fn concat_story_geometry_128x42x480x848() {
    use mlx_rs::ops::concatenate_axis;
    let (c, dh, h, w): (i64, i64, i64, i64) = (128, 21, 480, 848);
    let in_elems = c * dh * h * w;
    assert!(
        in_elems < I32_MAX,
        "each input must stay under the bound (from_slice)"
    );
    // Position-dependent, and DISTINCT per input so a scrambled index is caught: input k's flat
    // element j holds pv(j + k*OFFSET). We reconstruct the same value on the host for comparison.
    let off_b: i64 = 500_000_009; // shift so the two inputs don't alias mod 251
    let mk = |k: i64| -> Array {
        let host: Vec<f32> = (0..in_elems).map(|j| pv(j + k * off_b)).collect();
        let a = Array::from_slice(&host, &[c as i32, dh as i32, h as i32, w as i32]);
        a.eval().unwrap();
        a
    };
    let a0 = mk(0);
    let a1 = mk(1);
    let y = concatenate_axis(&[a0, a1], 1).unwrap();
    y.eval().unwrap();
    let dout = 2 * dh; // 42
    assert_eq!(y.shape(), &[c as i32, dout as i32, h as i32, w as i32]);
    let out_elems = c * dout * h * w;
    assert!(
        out_elems > I32_MAX,
        "concatenated output must cross the bound"
    );
    let ys = y.as_slice::<f32>();

    // Y[cc,dd,hh,ww] = input_k[cc, dd-k*dh, hh, ww] with k = dd/dh; its flat index within input k
    // is j = ((cc*dh + (dd - k*dh))*h + hh)*w + ww, value = pv(j + k*off_b).
    let mut first_bad = -1i64;
    let mut first_bad_above = -1i64;
    let mut bad = 0i64;
    let mut above_checked = 0i64;
    let mut max_abs = 0f32;
    let mut checked = 0i64;
    for cc in [0i64, 64, 120, 126, 127] {
        for dd in [0i64, dh - 1, dh, dh + 1, dout - 1] {
            for &(hh, ww) in &[(0i64, 0i64), (h / 2, w / 2), (h - 1, w - 1)] {
                let off = ((cc * dout + dd) * h + hh) * w + ww;
                let k = dd / dh;
                let din = dd - k * dh;
                let j = ((cc * dh + din) * h + hh) * w + ww;
                let want = pv(j + k * off_b);
                let got = ys[off as usize];
                checked += 1;
                let err = (got - want).abs();
                if err > max_abs {
                    max_abs = err;
                }
                if off >= I32_MAX {
                    above_checked += 1;
                }
                if err > 1e-3 {
                    bad += 1;
                    if first_bad < 0 {
                        first_bad = off;
                    }
                    if off >= I32_MAX && first_bad_above < 0 {
                        first_bad_above = off;
                    }
                }
            }
        }
    }
    eprintln!(
        "[concat 128×42×480×848] out_elems={out_elems} (={:.4}× 2^31)  checked={checked}  \
         above_2^31_checked={above_checked}  bad={bad}  max_abs={max_abs:e}  \
         first_bad_offset={first_bad}  first_bad_above={first_bad_above}",
        out_elems as f64 / I32_MAX as f64,
    );
    assert!(
        above_checked > 0,
        "probe must sample positions above 2^31 to prove the fix"
    );
    assert!(
        !(0..I32_MAX).contains(&first_bad),
        "concat corrupted a BELOW-2^31 position (offset {first_bad})"
    );
    assert_eq!(
        first_bad_above, -1,
        "concat corrupted an ABOVE-2^31 position (offset {first_bad_above})"
    );
    assert!(
        bad == 0,
        "concat had {bad} corrupted sample(s); max_abs={max_abs:e}"
    );
}

/// Which reshape/read ops on a >i32::MAX array survive on this pin. **sc-12748 corrects the earlier
/// weak check here** (which used `catch_unwind(...).is_ok()` and so only saw *panics*, silently passing
/// an MLX `Err`). The real ground truth, which `contiguous`'s materialization depends on:
///  - `as_slice` and elementwise over the oversized array: **correct**.
///  - `reshape(&[-1])` — flatten to a single >i32::MAX dimension: **RAISES** (`Err`) — MLX's
///    `check_shape_dim` (#3524) rejects any single dimension outside the `i32` range. This is why
///    `contiguous` must NOT flatten the full output.
///  - a **multi-dimensional** reshape whose total > i32::MAX but every dimension ≤ i32::MAX (e.g.
///    `[c, d·h·w]`): **works** — the per-dimension check passes and the size product is computed wide.
///    This is the int64-safe materialization the decode read-back uses.
///  - `from_slice` (host `Vec` → over-bound `Array`) stays i32-capped (mlx-rs assert), untested here.
#[test]
#[ignore = "sc-12438/sc-12748 MLX >i32::MAX reshape/contiguous probe (~9 GB); run with --ignored on Metal"]
fn reshape_and_contiguous_on_oversized_array() {
    let (c, d, h, w): (i64, i64, i64, i64) = (128, 41, 480, 848);
    let xhost: Vec<f32> = (0..c * d * h * w).map(pv).collect();
    let x = Array::from_slice(&xhost, &[c as i32, d as i32, h as i32, w as i32]);
    x.eval().unwrap();
    drop(xhost);
    let y = pad(&x, &[(0, 0), (0, 1), (0, 0), (0, 0)][..], None, None).unwrap();
    y.eval().unwrap();
    let dout = d + 1;
    let out_elems = c * dout * h * w;
    assert!(out_elems > I32_MAX);

    // (a) as_slice on the oversized array — read side is fine. Touch the last (>2^31) element.
    let read_ok = std::panic::catch_unwind(|| {
        let s = y.as_slice::<f32>();
        s[out_elems as usize - 1]
    })
    .is_ok();

    // (b) elementwise 2·y+1 over the oversized array — elementwise is safe.
    let two = Array::from_slice(&[2.0f32], &[1]);
    let one = Array::from_slice(&[1.0f32], &[1]);
    let elementwise_ok = std::panic::catch_unwind(|| {
        let z = add(multiply(&y, &two).unwrap(), &one).unwrap();
        z.eval().unwrap();
        let zs = z.as_slice::<f32>();
        let off = (127i64 * dout * h * w) as usize; // above 2^31
        (zs[off] - (2.0 * pv(127 * d * h * w) + 1.0)).abs() < 1e-3
    })
    .unwrap_or(false);

    // (c) reshape(&[-1]) — flatten to one >i32::MAX dim. Check the Result (not just panic-freedom).
    let reshape_flat_ok = y.reshape(&[-1]).is_ok();

    // (d) multi-dim reshape [c, dout·h·w] — total > i32::MAX, each dim ≤ i32::MAX. This is the
    //     materialization `contiguous` uses. Verify it returns Ok AND reads back correctly at an
    //     above-2^31 flat offset (position-dependent), then reshape back to the 4-D shape.
    let per = dout * h * w; // 42·480·848 = 17,111,040 < i32::MAX
    assert!(per < I32_MAX && c < I32_MAX && c * per > I32_MAX);
    let r2 = y.reshape(&[c as i32, per as i32]);
    let reshape_2d_ok = r2.is_ok();
    let mut reshape_2d_value_ok = false;
    let mut reshape_back_ok = false;
    if let Ok(r2) = r2 {
        r2.eval().unwrap();
        // Element [c=127, k] flat offset = 127·per + k (> 2^31). y[127,dd,hh,ww] = pv(orig) for dd<d
        // else 0; pick dd=0,hh=0,ww=0 → k=0 → pv(127·d·h·w).
        let rs = r2.as_slice::<f32>();
        let off = (127 * per) as usize;
        reshape_2d_value_ok = (rs[off] - pv(127 * d * h * w)).abs() < 1e-3;
        reshape_back_ok = r2
            .reshape(&[c as i32, dout as i32, h as i32, w as i32])
            .is_ok();
    }

    eprintln!(
        "[oversized array ops] as_slice_read_ok={read_ok}  elementwise_ok={elementwise_ok}  \
         reshape(-1)_ok={reshape_flat_ok}  reshape_2d_ok={reshape_2d_ok}  \
         reshape_2d_value_ok={reshape_2d_value_ok}  reshape_back_ok={reshape_back_ok}  \
         (out_elems={out_elems} = {:.4}× 2^31)",
        out_elems as f64 / I32_MAX as f64
    );
    assert!(read_ok, "reading back an oversized array must work");
    assert!(
        elementwise_ok,
        "elementwise over an oversized array must be correct (the safe path relies on this)"
    );
    assert!(
        !reshape_flat_ok,
        "reshape(-1) to a single >i32::MAX dim is expected to RAISE on this pin (#3524 \
         check_shape_dim) — if it now works, contiguous() can be simplified"
    );
    assert!(
        reshape_2d_ok && reshape_2d_value_ok && reshape_back_ok,
        "the multi-dim reshape contiguous() relies on must work AND read back correctly above 2^31 \
         (2d_ok={reshape_2d_ok} value_ok={reshape_2d_value_ok} back_ok={reshape_back_ok})"
    );
}
