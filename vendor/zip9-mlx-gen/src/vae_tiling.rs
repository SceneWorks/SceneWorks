//! Array-level **tiled VAE decode** (sc-11747) — the MLX/tensor half of the gen-core tiling seam.
//!
//! [`crate::tiling`] (re-exported from gen-core) is the **pure** half: tiling presets, the per-axis
//! interval split, the 1-D trapezoidal blend mask, and the [`TilePlan`] for a latent — no tensor dep,
//! Linux-buildable. This module is the tensor half: given a [`TilePlan`] and a per-tile decode closure,
//! it slices each overlapping tile out of the (already-denormalized) latent, decodes it, trapezoidally
//! blends the results, and folds them into the full output while keeping the peak bounded by one tile's
//! decode. sc-18320 replaced the pad-every-tile-to-the-full-output accumulator with a per-axis fold —
//! see [`tiled_decode_with_hooks`] for the mechanics and what they deliberately leave unchanged.
//!
//! It is **layout-agnostic** (the caller passes the `[t, h, w]` axis indices for NCTHW vs channels-last
//! and a decode closure that reaches its own VAE's decoder), so every VAE that tiles a whole decode tail
//! shares it: the Wan z16/z48 video VAEs (`mlx-gen-wan`, via a thin `vae_common` delegator preserving
//! their call sites), the LTX video VAE (`mlx-gen-ltx`, via [`tiled_decode_with_hooks`] for its
//! per-tile cache release), the Qwen-Image still-image VAE (`mlx-gen-qwen-image`, the Krea 2
//! pose-control decode sc-11747 bounds), and Sana's and Mage's image VAEs. Lifting it here removes the
//! divergence hazard of a per-crate copy of this subtle slice/blend/place/accumulate loop (the Wan
//! sc-4998/sc-5690 seam-artifact history).
//!
//! The AutoencoderKL/Z-Image/FLUX.2 families do **not** come through here: sc-19753 converted them to
//! layer-wise tiling ([`GlobalGroupNorm`] + [`tiled_conv2d_3x3_nhwc`]), which partitions each 3×3
//! convolution into halo-expanded cores and needs no blend at all.

use crate::array::scalar;
use crate::tiling::{AxisTile, TilePlan, MAX_WRITABLE_ELEMS};
use crate::{CancelFlag, Error, Result};
use mlx_rs::ops::{add, concatenate_axis, divide, maximum, multiply, pad, rsqrt, subtract};
use mlx_rs::Array;

/// Global GroupNorm statistics for an NHWC activation.
///
/// Whole-tail VAE tiling is not normalization-correct: every tile computes GroupNorm against its
/// own crop, while dense decode normalizes against the whole image. This context reduces the full
/// activation once to per-batch/per-group mean and inverse standard deviation, then applies those
/// same statistics to halo-expanded convolution tiles. Only the tiny statistics arrays are kept;
/// the full normalized activation is never materialized.
pub struct GlobalGroupNorm {
    mean: Array,
    inv_std: Array,
    weight: Array,
    bias: Array,
    groups: i32,
    channels: i32,
}

impl GlobalGroupNorm {
    /// Capture dense GroupNorm statistics for an NHWC rank-4 activation.
    pub fn new(x: &Array, weight: &Array, bias: &Array, groups: i32, eps: f32) -> Result<Self> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(Error::Msg(format!(
                "global group norm expects NHWC rank 4, got {shape:?}"
            )));
        }
        let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
        if groups <= 0 || channels % groups != 0 {
            return Err(Error::Msg(format!(
                "global group norm: channel count {channels} not divisible by groups {groups}"
            )));
        }
        if weight.shape() != [channels] || bias.shape() != [channels] {
            return Err(Error::Msg(format!(
                "global group norm: affine shapes must both be [{channels}], got {:?} and {:?}",
                weight.shape(),
                bias.shape()
            )));
        }

        let group_size = channels / groups;
        let grouped = x
            .reshape(&[batch, height, width, groups, group_size])?
            .transpose_axes(&[0, 3, 1, 2, 4])?;
        let mean = grouped.mean_axes(&[2, 3, 4], Some(true))?;
        let variance = grouped.var_axes(&[2, 3, 4], Some(true), None)?;
        let inv_std = rsqrt(&add(&variance, scalar(eps))?)?;
        mean.eval()?;
        inv_std.eval()?;

        Ok(Self {
            mean,
            inv_std,
            weight: weight.clone(),
            bias: bias.clone(),
            groups,
            channels,
        })
    }

    /// Normalize one NHWC crop with the full activation's statistics and affine parameters.
    pub fn apply(&self, tile: &Array) -> Result<Array> {
        let shape = tile.shape();
        if shape.len() != 4 || shape[3] != self.channels {
            return Err(Error::Msg(format!(
                "global group norm crop must be NHWC with {} channels, got {shape:?}",
                self.channels
            )));
        }
        let (batch, height, width) = (shape[0], shape[1], shape[2]);
        let group_size = self.channels / self.groups;
        let grouped = tile
            .reshape(&[batch, height, width, self.groups, group_size])?
            .transpose_axes(&[0, 3, 1, 2, 4])?;
        let normalized = multiply(&subtract(&grouped, &self.mean)?, &self.inv_std)?
            .transpose_axes(&[0, 2, 3, 1, 4])?
            .reshape(&[batch, height, width, self.channels])?;
        Ok(add(&multiply(&normalized, &self.weight)?, &self.bias)?)
    }
}

/// Apply one NHWC 3×3/pad-1 convolution with bounded spatial work and exact convolution halos.
///
/// The output is partitioned into non-overlapping cores. Each input crop expands by one pixel on
/// every available side, `preprocess` runs on that halo-expanded crop (typically global-stat
/// GroupNorm + SiLU), and the convolution's halo is cropped away before accumulation. Internal tile
/// boundaries therefore never observe synthetic padding and never require a blend. `max_tile_edge`
/// bounds the expanded convolution input, not merely the written core.
pub fn tiled_conv2d_3x3_nhwc(
    x: &Array,
    weight: &Array,
    bias: Option<&Array>,
    max_tile_edge: i32,
    cancel: Option<&CancelFlag>,
    preprocess: impl Fn(&Array) -> Result<Array>,
) -> Result<Array> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(Error::Msg(format!(
            "tiled conv2d expects NHWC rank 4, got {shape:?}"
        )));
    }
    let weight_shape = weight.shape();
    if weight_shape.len() != 4 || weight_shape[1] != 3 || weight_shape[2] != 3 {
        return Err(Error::Msg(format!(
            "tiled conv2d expects [out,3,3,in] weights, got {weight_shape:?}"
        )));
    }
    if max_tile_edge < 3 {
        return Err(Error::Msg(format!(
            "tiled conv2d needs max_tile_edge >= 3, got {max_tile_edge}"
        )));
    }
    let (batch, height, width) = (shape[0], shape[1], shape[2]);
    let out_channels = weight_shape[0];
    if height <= max_tile_edge && width <= max_tile_edge {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        return crate::nn::conv2d(&preprocess(x)?, weight, bias, 1, 1);
    }

    const HALO: i32 = 1;
    let core_edge = max_tile_edge - 2 * HALO;
    let mut rows = Vec::new();
    let mut y0 = 0;
    while y0 < height {
        let y1 = (y0 + core_edge).min(height);
        let input_y0 = (y0 - HALO).max(0);
        let input_y1 = (y1 + HALO).min(height);
        let mut cores = Vec::new();
        let mut x0 = 0;
        while x0 < width {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(Error::Canceled);
            }
            let x1 = (x0 + core_edge).min(width);
            let input_x0 = (x0 - HALO).max(0);
            let input_x1 = (x1 + HALO).min(width);

            let tile = slice_axis(x, 1, input_y0, input_y1)?;
            let tile = slice_axis(&tile, 2, input_x0, input_x1)?;
            let tile = preprocess(&tile)?;
            let convolved = crate::nn::conv2d(&tile, weight, bias, 1, 1)?;
            let crop_y0 = y0 - input_y0;
            let crop_x0 = x0 - input_x0;
            let core = slice_axis(&convolved, 1, crop_y0, crop_y0 + (y1 - y0))?;
            let core = slice_axis(&core, 2, crop_x0, crop_x0 + (x1 - x0))?;
            core.eval()?;
            cores.push(core);
            x0 = x1;
        }
        let core_refs = cores.iter().collect::<Vec<_>>();
        let row = concatenate_axis(&core_refs, 2)?;
        row.eval()?;
        rows.push(row);
        y0 = y1;
    }

    if rows.is_empty() {
        return Err(Error::Msg("tiled conv2d produced no tiles".into()));
    }
    let row_refs = rows.iter().collect::<Vec<_>>();
    let output = concatenate_axis(&row_refs, 1)?;
    output.eval()?;
    debug_assert_eq!(output.shape(), &[batch, height, width, out_channels]);
    crate::array::contiguous(&output)
}

/// Refuse — with a catchable error — building an over-[`MAX_WRITABLE_ELEMS`] array **from a host
/// buffer via `from_slice`** (the one write path still `i32`-capped on this pin). `full_elems` is the
/// element count; `out_*` are for the message.
///
/// **sc-12748 — this is now a narrow backstop, not the tiled-decode gate.** sc-12438 added this as an
/// up-front refusal on MLX 0.31.2, where every way to *produce* an over-bound assembled output was
/// broken: `pad` corrupted (~1.003× the bound), `conv3d` corrupted, and `from_slice`/`reshape(-1)`
/// overflowed the flat `i32` size (MLX #3327). On this pin (0.32.0 fork `eb76c4ba` + the sc-12746
/// copy-gate patch), the 2026-08-11 re-probe verifies `pad`/`concat`/`conv3d`/multi-dimensional
/// reshape/`as_slice`/elementwise EXACT above `i32::MAX`, so the tiled `pad`-and-accumulate decode now
/// renders past the bound (the refusal was lifted from [`tiled_decode`]). A single `reshape(-1)`
/// dimension still raises, but `contiguous` deliberately uses the verified multi-dimensional path.
/// `Array::from_slice` also remains i32-capped: it still asserts `len == shape.product::<i32>()` (a
/// fork-side bug).
///
/// **sc-12926 — status: retained latent tripwire, NO production caller.** When sc-12748 lifted the
/// refusal it removed both call sites, so nothing in the decode paths invokes this today (they read
/// back via `as_slice` and never `from_slice` a full output — there is currently no host→Array
/// materialization of output-scale buffers anywhere to guard). It is deliberately kept, with its
/// boundary unit test, as the ready-made guard for any **future** code that builds an over-bound
/// array from a host `Vec`: call it before that `from_slice` (the mlx-rs assert would still abort
/// loudly without it, but as a panic rather than a catchable [`Error`]). There is no separate
/// blend-weight buffer to guard: sc-18320's separable fold retired the full-output-sized weight
/// accumulator — the normalizer is carried as three 1-D host vectors (`AxisCoverage`) and
/// materialized once as a rank-1 broadcast product, strictly smaller than the output.
pub fn check_output_writable(full_elems: i64, out_f: i32, out_h: i32, out_w: i32) -> Result<()> {
    if full_elems > MAX_WRITABLE_ELEMS {
        return Err(Error::Msg(format!(
            "vae output materialization: a {out_f}×{out_h}×{out_w} = {full_elems}-element buffer is \
             over the {MAX_WRITABLE_ELEMS}-element ceiling above which mlx-rs `from_slice` overflows \
             its i32 length assert. Read back via `as_slice` / tile the host build — do not `from_slice` \
             the full buffer."
        )));
    }
    Ok(())
}

/// Gather the contiguous range `[start, end)` along `axis` (mlx-rs has no slice op). Layout-agnostic.
fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), axis)?)
}

/// `[1; rank]` with `len` placed at `axis` — a 1-D vector reshaped to broadcast along its own axis.
fn axis_broadcast_shape(rank: usize, axis: i32, len: i32) -> Vec<i32> {
    let mut shape = vec![1i32; rank];
    shape[axis as usize] = len;
    shape
}

/// Add `x` into `acc` with `x`'s `extent` elements placed at `offset` along `axis` of an
/// `out_len`-long axis (sc-18320).
///
/// The zero-pad spans **one** axis, so the placement intermediate is `x` grown along that single
/// axis — not the full output — and an axis the caller already covers end-to-end skips the pad
/// altogether. `pad`/`add` are the two assembly ops `tests/mlx_write_bound_probe.rs` verifies exact
/// above `i32::MAX` on this pin, which is why the accumulator stays on them instead of the
/// `scatter`/`slice_update` family (unprobed there, and `slice_update` is not even public in this
/// mlx-rs revision).
pub fn accumulate_along_axis(
    acc: Option<Array>,
    x: &Array,
    axis: i32,
    offset: i32,
    extent: i32,
    out_len: i32,
) -> Result<Array> {
    // `extent` is what the caller claims it is placing. The fully-covered fast path below returns `x`
    // verbatim, so a wrong `extent` would otherwise mis-shape the stage silently instead of erroring.
    let actual = x.shape()[axis as usize];
    if actual != extent {
        return Err(Error::Msg(format!(
            "vae tiled decode: placing {extent} elements along axis {axis} of an array whose extent \
             there is {actual}"
        )));
    }
    let placed = if offset == 0 && extent == out_len {
        x.clone()
    } else {
        let mut pads = vec![(0, 0); x.shape().len()];
        pads[axis as usize] = (offset, out_len - (offset + extent));
        pad(x, &pads[..], None, None)?
    };
    match acc {
        None => Ok(placed),
        Some(acc) => Ok(add(&acc, &placed)?),
    }
}

/// The blend's accumulated weight, kept as one summed mask **per tiled axis** (sc-18320).
///
/// The tile grid is a product grid and the per-tile blend weight is the separable outer product
/// `tm_i ⊗ hm_j ⊗ wm_k`, so the accumulated weight factorizes exactly:
/// `Σ_{i,j,k} tm_i[t]·hm_j[h]·wm_k[w] = (Σ_i tm_i[t])·(Σ_j hm_j[h])·(Σ_k wm_k[w])`. Three `out_f` /
/// `out_h` / `out_w`-long host vectors therefore carry what the pad-to-full route accumulated as a
/// second full-output-sized MLX array, one `pad`+`add` per tile.
///
/// Factorizing requires each axis tile to contribute **one** written extent — the decoder's output
/// extent along an axis must depend only on that axis's tile, which is what a spatially/temporally
/// local decode tail gives. [`Self::bind`] enforces it with a catchable error rather than trusting
/// it: a decoder that returned a different extent for the same axis tile would silently break the
/// factorization (and the fold's shapes) otherwise.
struct AxisCoverage {
    /// Summed masks, indexed by position in `axes`: `[t, h, w]`.
    sums: [Vec<f32>; 3],
    /// The one written extent bound to each axis tile, indexed the same way.
    extents: [Vec<Option<i32>>; 3],
}

impl AxisCoverage {
    fn new(plan: &TilePlan) -> Self {
        Self {
            sums: [
                vec![0.0; plan.out_f.max(0) as usize],
                vec![0.0; plan.out_h.max(0) as usize],
                vec![0.0; plan.out_w.max(0) as usize],
            ],
            extents: [
                vec![None; plan.t.len()],
                vec![None; plan.h.len()],
                vec![None; plan.w.len()],
            ],
        }
    }

    /// Bind `extent` to axis tile `index`, summing its mask into the coverage the first time. Later
    /// visits of the same axis tile must agree — the mask is added exactly once, so a disagreement
    /// would leave the normalizer describing a different blend than the accumulation performed.
    fn bind(&mut self, slot: usize, index: usize, tile: &AxisTile, extent: i32) -> Result<()> {
        const NAMES: [&str; 3] = ["temporal", "height", "width"];
        if let Some(bound) = self.extents[slot][index] {
            if bound != extent {
                return Err(Error::Msg(format!(
                    "vae tiled decode: the decode closure returned {extent} {} output elements for \
                     {} tile {index} after returning {bound}. The separable blend normalizer needs \
                     one written extent per axis tile.",
                    NAMES[slot], NAMES[slot]
                )));
            }
            return Ok(());
        }
        let sum = &mut self.sums[slot];
        let start = tile.out_start;
        if start < 0 || (start + extent) as usize > sum.len() {
            return Err(Error::Msg(format!(
                "vae tiled decode: {} tile {index} writes [{start}, {}) outside the {}-long output \
                 axis",
                NAMES[slot],
                start + extent,
                sum.len()
            )));
        }
        for (offset, weight) in tile.mask[..extent as usize].iter().enumerate() {
            sum[start as usize + offset] += weight;
        }
        self.extents[slot][index] = Some(extent);
        Ok(())
    }

    /// The rank-1 blend normalizer, broadcast-shaped over `axes` and materialized once — the exact
    /// array the pad-to-full route rebuilt incrementally in its `weights` accumulator.
    fn normalizer(&self, rank: usize, axes: [i32; 3]) -> Result<Array> {
        let mut normalizer: Option<Array> = None;
        for (slot, axis) in axes.into_iter().enumerate() {
            let sum = &self.sums[slot];
            let factor =
                Array::from_slice(sum, &axis_broadcast_shape(rank, axis, sum.len() as i32));
            normalizer = Some(match normalizer {
                None => factor,
                Some(product) => multiply(&product, &factor)?,
            });
        }
        normalizer.ok_or_else(|| Error::Msg("vae tiled decode: no tiled axes".into()))
    }
}

/// The trapezoidally-blended tile-accumulate loop shared by every tiled `decode`. Slices each
/// overlapping tile out of `denorm` (the already-denormalized latent), decodes it via the
/// layout-specific `decode_tile` closure, trapezoidally blends along the three tiled axes, and
/// accumulates into the full output. `axes` are the `[t, h, w]` axis indices for the layout (`[2, 3, 4]`
/// for NCTHW, `[1, 2, 3]` for channels-last); the mask shapes and placements derive from those
/// indices, so the only per-layout input is the closure.
///
/// `plan` comes from [`TilingConfig::plan`](crate::tiling::TilingConfig::plan). The reference's per-tile
/// `mx.eval` (bounding the lazy graph + peak memory) is preserved — without it the whole tiled graph
/// would materialize at once, defeating the point of tiling.
///
/// `cancel` is the cooperative cancellation handle: the decode is a dominant fraction of a render's
/// wall-clock, so a cancel is checked between tiles and returns [`Error::Canceled`]. The per-tile `eval`
/// forces materialization, so the check observes the trip promptly.
///
/// See [`tiled_decode_with_hooks`] for the accumulation mechanics, and for the variant that lets a
/// caller inject its own materialization and buffer-release points.
pub fn tiled_decode(
    denorm: &Array,
    plan: &TilePlan,
    axes: [i32; 3],
    cancel: Option<&CancelFlag>,
    decode_tile: impl FnMut(&Array) -> Result<Array>,
) -> Result<Array> {
    tiled_decode_with_hooks(
        denorm,
        plan,
        axes,
        cancel,
        decode_tile,
        |accumulators| {
            for accumulator in accumulators {
                accumulator.eval()?;
            }
            Ok(())
        },
        || {},
    )
}

/// [`tiled_decode`] with the per-tile **materialization** and **buffer-release** points injected.
///
/// `materialize` receives every accumulator that is live after the tile has been folded in and must
/// force it (the reference's per-tile `mx.eval`); it is called exactly once per tile, after the
/// tile-local handles have dropped. `release` then runs with only those accumulators live, so a
/// caller that returns dead buffers to the OS (`mlx_rs::memory::clear_cache`) evicts the tile's
/// decoder and assembly buffers without evicting the assembly in progress. It is called exactly once
/// per tile, and once more on the failure path after the accumulators are dropped — a pre-tripped
/// cancel, which runs no tile, calls neither hook.
///
/// # Accumulation mechanics (sc-18320)
///
/// The tiles form a **product grid** and the blend weight is the separable outer product
/// `tm_i ⊗ hm_j ⊗ wm_k`, so the blended sum re-associates by axis:
///
/// ```text
/// out[t,h,w] = Σ_i tm_i[t] · ( Σ_j hm_j[h] · ( Σ_k wm_k[w] · dec_ijk[t,h,w] ) )
/// ```
///
/// This function folds in exactly that order — innermost over `w` into a `w`-full strip, then over
/// `h` into an `h`-full slab, then over `t` into the output. Each placement zero-pads along **one**
/// axis, so a tile's placement intermediate is its own decode grown along a single axis. The
/// predecessor route padded every weighted tile *and* its blend mask to the full output shape, so it
/// touched full-output-sized buffers `|t|·|h|·|w|` times; the fold touches them `|t|` times, and the
/// intermediate axis strips are smaller than the output by the ratio the untiled axes contribute.
///
/// Re-association means the accumulated blend weight factorizes exactly
/// (`Σ_{i,j,k} tm_i·hm_j·wm_k = (Σ_i tm_i)(Σ_j hm_j)(Σ_k wm_k)`), so the fold carries the
/// normalizer as three 1-D host vectors and materializes it once at the end — replacing a second
/// full-output-sized accumulator that took a `pad`+`add` of its own per tile.
///
/// **What does not change.** The per-tile blend weights, the tile geometry, and the final
/// `Σwd / max(Σw, 1e-8)` normalization are identical, so the *effective* weight each tile's decode
/// carries at each output coordinate is bit-for-bit the same profile as before: the conv-halo seam
/// term an overlap narrower than the decoder's receptive field leaves behind is attenuated exactly as
/// much as it was, neither more nor less. Only the summation's association changes, which moves
/// results by float rounding (~1 ULP over the ≤8 tiles that cover any coordinate), not by blend
/// characteristic. Globally-scoped decode work stays where its caller put it — this function only
/// ever sees the tile closure it is handed, so a head that runs once on the whole latent (sc-19753)
/// keeps running once.
pub fn tiled_decode_with_hooks(
    denorm: &Array,
    plan: &TilePlan,
    axes: [i32; 3],
    cancel: Option<&CancelFlag>,
    mut decode_tile: impl FnMut(&Array) -> Result<Array>,
    mut materialize: impl FnMut(&[&Array]) -> Result<()>,
    mut release: impl FnMut(),
) -> Result<Array> {
    let [t_ax, h_ax, w_ax] = axes;
    let mut coverage = AxisCoverage::new(plan);
    // One accumulator per fold stage: `w_acc` spans the output's w axis, `h_acc` also its h axis,
    // `t_acc` also its t axis (the full output). Only `t_acc` is ever full-output-sized.
    let mut t_acc: Option<Array> = None;
    let mut h_acc: Option<Array> = None;
    let mut w_acc: Option<Array> = None;

    for (ti, t) in plan.t.iter().enumerate() {
        let last_t = ti + 1 == plan.t.len();
        for (hj, hh) in plan.h.iter().enumerate() {
            let last_h = hj + 1 == plan.h.len();
            for (wk, ww) in plan.w.iter().enumerate() {
                let last_w = wk + 1 == plan.w.len();
                if cancel.is_some_and(CancelFlag::is_cancelled) {
                    let live = w_acc.is_some() || h_acc.is_some() || t_acc.is_some();
                    drop(w_acc.take());
                    drop(h_acc.take());
                    drop(t_acc.take());
                    if live {
                        release();
                    }
                    return Err(Error::Canceled);
                }
                // Fold one tile in. The inner scope owns every tile-local handle so they are dropped
                // before `materialize`/`release` run — the accumulators are all that stay live.
                let folded = (|| -> Result<()> {
                    let tile = slice_axis(denorm, t_ax, t.start, t.end)?;
                    let tile = slice_axis(&tile, h_ax, hh.start, hh.end)?;
                    let tile = slice_axis(&tile, w_ax, ww.start, ww.end)?;
                    let dec = decode_tile(&tile)?;

                    let ds = dec.shape();
                    let rank = ds.len();

                    // sc-12748: the sc-12438 over-bound REFUSAL is RETIRED here. This assembly builds
                    // the full output only with `pad` (+`add`/`multiply`/`divide`/`maximum`) and reads
                    // it back through `contiguous`'s multi-dimensional `reshape` + `as_slice` — and
                    // every operation in that path is probe-verified int64-safe above `i32::MAX` on
                    // this pin (`mlx-gen/tests/mlx_write_bound_probe.rs`: pad & concat EXACT via the
                    // sc-12746 copy-gate patch; multi-dimensional reshape/as_slice/elementwise all
                    // correct; a single reshape(-1) dimension still raises). So a tiled decode whose
                    // *assembled* output now crosses the bound RENDERS correctly instead of erroring
                    // (validated end-to-end vs a below-bound reference in
                    // `tiled_decode_renders_over_bound_output` and the LTX real-weights render). The
                    // one path still i32-capped is a `from_slice` host→Array materialization, which
                    // this loop never takes for output-scale data (`check_output_writable` is retained
                    // as an uncalled latent tripwire for future code that would — sc-12926).
                    // sc-18320 keeps the assembly on this same probed op set: `scatter`/`slice_update`
                    // are outside it (and `slice_update` is not public in this mlx-rs revision), so a
                    // literal scatter accumulator would put unverified ops on the over-bound path.
                    let at = ds[t_ax as usize].min(t.out_stop - t.out_start);
                    let ah = ds[h_ax as usize].min(hh.out_stop - hh.out_start);
                    let aw = ds[w_ax as usize].min(ww.out_stop - ww.out_start);
                    coverage.bind(0, ti, t, at)?;
                    coverage.bind(1, hj, hh, ah)?;
                    coverage.bind(2, wk, ww, aw)?;

                    // 1-D masks → outer product, each broadcasting along its own (t/h/w) axis.
                    let tm = Array::from_slice(
                        &t.mask[..at as usize],
                        &axis_broadcast_shape(rank, t_ax, at),
                    );
                    let hm = Array::from_slice(
                        &hh.mask[..ah as usize],
                        &axis_broadcast_shape(rank, h_ax, ah),
                    );
                    let wm = Array::from_slice(
                        &ww.mask[..aw as usize],
                        &axis_broadcast_shape(rank, w_ax, aw),
                    );
                    let blend = multiply(&multiply(&tm, &hm)?, &wm)?;

                    let dec = slice_axis(&dec, t_ax, 0, at)?;
                    let dec = slice_axis(&dec, h_ax, 0, ah)?;
                    let dec = slice_axis(&dec, w_ax, 0, aw)?;
                    let weighted = multiply(&dec, &blend)?;

                    // Stage 1 — place along w only, into the w-full strip.
                    w_acc = Some(accumulate_along_axis(
                        w_acc.take(),
                        &weighted,
                        w_ax,
                        ww.out_start,
                        aw,
                        plan.out_w,
                    )?);
                    // Stage 2/3 — close the strip into the slab, and the slab into the output, as
                    // soon as their rows finish. Doing it here (rather than after the loop) keeps
                    // this tile's `materialize` the single point that forces every live accumulator.
                    if last_w {
                        let strip = w_acc
                            .take()
                            .ok_or_else(|| Error::Msg("vae tiled decode: empty w strip".into()))?;
                        h_acc = Some(accumulate_along_axis(
                            h_acc.take(),
                            &strip,
                            h_ax,
                            hh.out_start,
                            ah,
                            plan.out_h,
                        )?);
                    }
                    if last_w && last_h {
                        let slab = h_acc
                            .take()
                            .ok_or_else(|| Error::Msg("vae tiled decode: empty h slab".into()))?;
                        t_acc = Some(accumulate_along_axis(
                            t_acc.take(),
                            &slab,
                            t_ax,
                            t.out_start,
                            at,
                            plan.out_f,
                        )?);
                    }
                    let live: Vec<&Array> = [w_acc.as_ref(), h_acc.as_ref(), t_acc.as_ref()]
                        .into_iter()
                        .flatten()
                        .collect();
                    materialize(&live)
                })();
                match folded {
                    Ok(()) => release(),
                    Err(error) => {
                        drop(w_acc.take());
                        drop(h_acc.take());
                        drop(t_acc.take());
                        release();
                        return Err(error);
                    }
                }
                debug_assert!(
                    !(last_w && last_h && last_t) || (w_acc.is_none() && h_acc.is_none()),
                    "the fold must close every strip and slab it opened"
                );
            }
        }
    }

    let output = t_acc.ok_or_else(|| Error::Msg("vae tiled decode: plan had no tiles".into()))?;
    let normalizer = coverage.normalizer(output.shape().len(), axes)?;
    // sc-12748: int64-safe contiguity (the assembled output can exceed i32::MAX — a single-dim
    // reshape(-1) would raise; `array::contiguous` flattens via a 2-D split instead).
    crate::array::contiguous(&divide(&output, &maximum(&normalizer, scalar(1e-8))?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiling::{AxisTile, SpatialTiling, TemporalTiling, TilingConfig, VaeTiling};

    /// Two non-overlapping tiles along the temporal axis with all-ones masks and an identity decode must
    /// exactly reconstruct the input — exercising slice/mask/pad placement and accumulation for a given
    /// axis layout.
    fn roundtrip(denorm: &Array, axes: [i32; 3], t_full: i32) -> Vec<f32> {
        let half = t_full / 2;
        let tile = |start, out_start| AxisTile {
            start,
            end: start + half,
            out_start,
            out_stop: out_start + half,
            mask: vec![1.0; half as usize],
        };
        let unit = AxisTile {
            start: 0,
            end: 2,
            out_start: 0,
            out_stop: 2,
            mask: vec![1.0; 2],
        };
        let plan = TilePlan {
            t: vec![tile(0, 0), tile(half, half)],
            h: vec![unit.clone()],
            w: vec![unit],
            out_f: t_full,
            out_h: 2,
            out_w: 2,
        };
        let out = tiled_decode(denorm, &plan, axes, None, |tile| Ok(tile.clone())).unwrap();
        out.eval().unwrap();
        out.as_slice::<f32>().to_vec()
    }

    #[test]
    fn identity_roundtrip_ncthw() {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 1, 4, 2, 2]);
        assert_eq!(roundtrip(&denorm, [2, 3, 4], 4), vals);
    }

    #[test]
    fn identity_roundtrip_channels_last() {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 4, 2, 2, 1]);
        assert_eq!(roundtrip(&denorm, [1, 2, 3], 4), vals);
    }

    #[test]
    fn honors_pretripped_cancel() {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 1, 4, 2, 2]);
        let half = 2;
        let tile = |start, out_start| AxisTile {
            start,
            end: start + half,
            out_start,
            out_stop: out_start + half,
            mask: vec![1.0; half as usize],
        };
        let unit = AxisTile {
            start: 0,
            end: 2,
            out_start: 0,
            out_stop: 2,
            mask: vec![1.0; 2],
        };
        let plan = TilePlan {
            t: vec![tile(0, 0), tile(half, half)],
            h: vec![unit.clone()],
            w: vec![unit],
            out_f: 4,
            out_h: 2,
            out_w: 2,
        };
        let cancel = CancelFlag::new();
        cancel.cancel();
        let res = tiled_decode(&denorm, &plan, [2, 3, 4], Some(&cancel), |t| Ok(t.clone()));
        assert!(matches!(res, Err(Error::Canceled)));
    }

    /// Block (nearest-neighbour) spatial upsample by `scale` along `axes[1..]` — tile-consistent, so a
    /// correct partition-of-unity blend reconstructs the full upsample exactly. Isolates the
    /// slice/mask/pad/accumulate/normalize machinery for the **image** (T=1) case this story adds.
    fn upsample_spatial(x: &Array, axes: [i32; 3], scale: i32) -> Array {
        let mut y = x.clone();
        for &ax in &axes[1..] {
            y = Array::repeat_axis::<f32>(y, scale, ax).unwrap();
        }
        y
    }

    /// sc-11747: the Qwen-Image case — a single temporal frame (T=1), spatial ×8, tiled on H and W.
    /// A tile-consistent block-upsample decode blended through the real [`TilingConfig::plan`] geometry
    /// must reconstruct the full upsample exactly (no seam), proving the image path of the shared loop.
    #[test]
    fn image_spatial_tiles_reconstruct() {
        let vae = VaeTiling::QWEN_IMAGE; // spatial ×8, temporal ×1
        let cfg = TilingConfig {
            spatial: Some(SpatialTiling {
                tile_px: 4 * vae.spatial_scale, // 4-latent tiles
                overlap_px: 2 * vae.spatial_scale,
            }),
            temporal: None,
        };
        // NCTHW latent [1, 16→2 (tiny), 1, 13, 13]: ragged 3×3 spatial tiling, T=1.
        let (f, h, w) = (1, 13, 13);
        assert!(cfg.needs_tiling(vae, f, h, w));
        let plan = cfg.plan(vae, f, h, w);
        let shape = [1, 2, f, h, w];
        let n: i32 = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.19).sin()).collect();
        let denorm = Array::from_slice(&vals, &shape);

        let expected = upsample_spatial(&denorm, [2, 3, 4], vae.spatial_scale);
        let got = tiled_decode(&denorm, &plan, [2, 3, 4], None, |t| {
            Ok(upsample_spatial(t, [2, 3, 4], vae.spatial_scale))
        })
        .unwrap();
        got.eval().unwrap();
        assert_eq!(got.shape(), expected.shape());
        let (g, e) = (got.as_slice::<f32>(), expected.as_slice::<f32>());
        let max = g
            .iter()
            .zip(e)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max < 1e-4,
            "image tiled blend did not reconstruct: max|Δ|={max:.3e}"
        );
    }

    /// sc-18318 reachability through a **production decode seam**, not a hand-built scope.
    ///
    /// [`tiled_conv2d_3x3_nhwc`] is the shipped bounded VAE-decode loop: it calls
    /// [`crate::nn::conv2d`] per halo-expanded tile and runs the caller's GroupNorm→SiLU preprocess
    /// inside it — three of the six P3 operations, on the real code path a bounded decode executes.
    /// Driving it under an observation scope (which records without overriding) shows a production
    /// decode taking the native epilogues; before the promotion every one of these was composed
    /// because nothing in production ever installed the toggle. The bounded-versus-dense tracking
    /// assertion doubles as the numeric-safety check: the fused epilogues must not move the decode.
    #[test]
    fn a_production_bounded_decode_reaches_the_exact_epilogues() {
        use crate::diagnostics::{self, DiagnosticCounter, ToggleDisposition};

        let (batch, height, width, channels, out_channels) = (1, 7, 9, 32, 3);
        let values = (0..batch * height * width * channels)
            .map(|i| (i as f32 * 0.071).sin())
            .collect::<Vec<_>>();
        let x = Array::from_slice(&values, &[batch, height, width, channels]);
        let norm_weight = Array::from_slice(
            &(0..channels)
                .map(|i| 0.7 + i as f32 * 0.013)
                .collect::<Vec<_>>(),
            &[channels],
        );
        let norm_bias = Array::from_slice(
            &(0..channels)
                .map(|i| (i as f32 * 0.31).cos() * 0.08)
                .collect::<Vec<_>>(),
            &[channels],
        );
        let conv_weight = Array::from_slice(
            &(0..out_channels * 3 * 3 * channels)
                .map(|i| (i as f32 * 0.037).sin() * 0.025)
                .collect::<Vec<_>>(),
            &[out_channels, 3, 3, channels],
        );
        let conv_bias = Array::from_slice(&[0.02f32, -0.03, 0.01], &[out_channels]);

        let decode = || {
            let global = GlobalGroupNorm::new(&x, &norm_weight, &norm_bias, 4, 1e-5).unwrap();
            let tiled =
                tiled_conv2d_3x3_nhwc(&x, &conv_weight, Some(&conv_bias), 5, None, |tile| {
                    crate::nn::silu(&global.apply(tile)?)
                })
                .unwrap();
            let dense_norm = crate::nn::group_norm(&x, &norm_weight, &norm_bias, 4, 1e-5).unwrap();
            let dense = crate::nn::conv2d(
                &crate::nn::silu(&dense_norm).unwrap(),
                &conv_weight,
                Some(&conv_bias),
                1,
                1,
            )
            .unwrap();
            tiled.eval().unwrap();
            dense.eval().unwrap();
            (tiled, dense)
        };

        let composed = {
            let _opt_out = crate::capability::CapabilityOptOut::install(
                crate::capability::PipelineCapability::ExactEpilogues,
            );
            decode()
        };

        let scope =
            diagnostics::begin_observed_request("vae-tiling-p3-production", "test").unwrap();
        let (tiled, dense) = decode();
        let report = scope.finish();

        // The bounded decode still tracks the dense layer, and the fused decode is bit-identical to
        // the composed one it replaces — both tiled and dense.
        let max_diff = |a: &Array, b: &Array| {
            a.as_slice::<f32>()
                .iter()
                .zip(b.as_slice::<f32>())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            max_diff(&tiled, &dense) < 2e-4,
            "fused bounded decode diverged from the dense layer"
        );
        assert_eq!(
            max_diff(&tiled, &composed.0),
            0.0,
            "the P3 epilogues changed a bounded decode's output"
        );
        assert_eq!(
            max_diff(&dense, &composed.1),
            0.0,
            "the P3 epilogues changed a dense decode's output"
        );

        for operation in [
            diagnostics::EXACT_CONV2D_BIAS,
            diagnostics::EXACT_SILU,
            diagnostics::EXACT_GROUP_NORM_AFFINE,
        ] {
            assert!(
                report.counters.iter().any(|counter| matches!(
                    counter,
                    DiagnosticCounter::ExactEpilogue {
                        operation: recorded,
                        disposition: ToggleDisposition::Applied,
                        reason: None,
                        count,
                    } if *recorded == operation && *count > 0
                )),
                "a production decode must apply {operation}; receipts were {:?}",
                report.counters
            );
        }
        assert!(report.counters.contains(&DiagnosticCounter::Toggle {
            toggle: diagnostics::EXACT_EPILOGUES,
            disposition: ToggleDisposition::Applied,
            count: 1,
        }));
    }

    /// sc-19753: layer-wise convolution tiling must use the dense activation's GroupNorm
    /// statistics. A whole-tail tiled decoder instead normalizes every crop independently and is
    /// exactly the bug this regression guards against. Position-dependent values make those local
    /// statistics observably different; the halo/core convolution path must still track the dense
    /// GroupNorm→SiLU→3×3 convolution.
    #[test]
    fn global_group_norm_tiled_conv_tracks_dense_layer() {
        let (batch, height, width, channels, out_channels) = (1, 7, 9, 32, 3);
        let values = (0..batch * height * width * channels)
            .map(|i| {
                let y = (i / channels / width) as f32;
                let x = (i / channels % width) as f32;
                (i as f32 * 0.071).sin() + y * 0.17 - x * 0.09
            })
            .collect::<Vec<_>>();
        let x = Array::from_slice(&values, &[batch, height, width, channels]);
        let norm_weight = Array::from_slice(
            &(0..channels)
                .map(|i| 0.7 + i as f32 * 0.013)
                .collect::<Vec<_>>(),
            &[channels],
        );
        let norm_bias = Array::from_slice(
            &(0..channels)
                .map(|i| (i as f32 * 0.31).cos() * 0.08)
                .collect::<Vec<_>>(),
            &[channels],
        );
        let conv_weight = Array::from_slice(
            &(0..out_channels * 3 * 3 * channels)
                .map(|i| (i as f32 * 0.037).sin() * 0.025)
                .collect::<Vec<_>>(),
            &[out_channels, 3, 3, channels],
        );
        let conv_bias = Array::from_slice(&[0.02f32, -0.03, 0.01], &[out_channels]);

        let dense_norm = crate::nn::group_norm(&x, &norm_weight, &norm_bias, 4, 1e-5).unwrap();
        let dense = crate::nn::conv2d(
            &crate::nn::silu(&dense_norm).unwrap(),
            &conv_weight,
            Some(&conv_bias),
            1,
            1,
        )
        .unwrap();

        let global = GlobalGroupNorm::new(&x, &norm_weight, &norm_bias, 4, 1e-5).unwrap();
        let tiled = tiled_conv2d_3x3_nhwc(&x, &conv_weight, Some(&conv_bias), 5, None, |tile| {
            crate::nn::silu(&global.apply(tile)?)
        })
        .unwrap();
        dense.eval().unwrap();
        tiled.eval().unwrap();
        assert_eq!(tiled.shape(), dense.shape());
        let max = tiled
            .as_slice::<f32>()
            .iter()
            .zip(dense.as_slice::<f32>())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 2e-4,
            "global-stat tiled layer diverged from dense GroupNorm+conv: max|Δ|={max:.3e}"
        );
    }

    // --- sc-18320: the separable fold vs the pad-to-full accumulator it replaced ----------------

    /// The **predecessor accumulator**, verbatim: every weighted tile *and* its blend mask padded to
    /// the full output shape, added into two full-output-sized accumulators, then normalized once.
    ///
    /// This is the reference the fold is graded against. It is deliberately a copy rather than a
    /// refactor: grading the new mechanics against a paraphrase of themselves would prove nothing.
    fn pad_to_full_reference(
        denorm: &Array,
        plan: &TilePlan,
        axes: [i32; 3],
        decode_tile: impl Fn(&Array) -> Result<Array>,
    ) -> Result<Array> {
        let [t_ax, h_ax, w_ax] = axes;
        let mut output: Option<Array> = None;
        let mut weights: Option<Array> = None;
        for t in &plan.t {
            for hh in &plan.h {
                for ww in &plan.w {
                    let tile = slice_axis(denorm, t_ax, t.start, t.end)?;
                    let tile = slice_axis(&tile, h_ax, hh.start, hh.end)?;
                    let tile = slice_axis(&tile, w_ax, ww.start, ww.end)?;
                    let dec = decode_tile(&tile)?;
                    let ds = dec.shape();
                    let rank = ds.len();
                    let at = ds[t_ax as usize].min(t.out_stop - t.out_start);
                    let ah = ds[h_ax as usize].min(hh.out_stop - hh.out_start);
                    let aw = ds[w_ax as usize].min(ww.out_stop - ww.out_start);
                    let tm = Array::from_slice(
                        &t.mask[..at as usize],
                        &axis_broadcast_shape(rank, t_ax, at),
                    );
                    let hm = Array::from_slice(
                        &hh.mask[..ah as usize],
                        &axis_broadcast_shape(rank, h_ax, ah),
                    );
                    let wm = Array::from_slice(
                        &ww.mask[..aw as usize],
                        &axis_broadcast_shape(rank, w_ax, aw),
                    );
                    let blend = multiply(&multiply(&tm, &hm)?, &wm)?;
                    let dec = slice_axis(&dec, t_ax, 0, at)?;
                    let dec = slice_axis(&dec, h_ax, 0, ah)?;
                    let dec = slice_axis(&dec, w_ax, 0, aw)?;
                    let weighted = multiply(&dec, &blend)?;
                    let mut pads = vec![(0, 0); rank];
                    pads[t_ax as usize] = (t.out_start, plan.out_f - (t.out_start + at));
                    pads[h_ax as usize] = (hh.out_start, plan.out_h - (hh.out_start + ah));
                    pads[w_ax as usize] = (ww.out_start, plan.out_w - (ww.out_start + aw));
                    let weighted_full = pad(&weighted, &pads[..], None, None)?;
                    let blend_full = pad(&blend, &pads[..], None, None)?;
                    output = Some(match output {
                        None => weighted_full,
                        Some(acc) => add(&acc, &weighted_full)?,
                    });
                    weights = Some(match weights {
                        None => blend_full,
                        Some(acc) => add(&acc, &blend_full)?,
                    });
                    output.as_ref().unwrap().eval()?;
                    weights.as_ref().unwrap().eval()?;
                }
            }
        }
        let output = output.ok_or_else(|| Error::Msg("reference: no tiles".into()))?;
        let weights = weights.ok_or_else(|| Error::Msg("reference: no tiles".into()))?;
        crate::array::contiguous(&divide(&output, &maximum(&weights, scalar(1e-8))?)?)
    }

    /// A ragged, overlapping three-axis **video** plan for `vae`, built through the real
    /// [`TilingConfig::plan`] so every axis carries the shipped trapezoidal mask. The spatial axes are
    /// ragged (their last tile is short); whether the TEMPORAL axis is ragged, and which output
    /// mapping it uses, is `vae`'s to decide — see [`ragged_video_plan`] and
    /// [`ragged_causal_video_plan`].
    fn ragged_plan_for(vae: VaeTiling, f: i32, h: i32, w: i32) -> (Array, TilePlan, [i32; 3]) {
        let cfg = TilingConfig {
            spatial: Some(SpatialTiling {
                tile_px: 4 * vae.spatial_scale,
                overlap_px: vae.spatial_scale,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 3 * vae.temporal_scale,
                overlap_frames: vae.temporal_scale,
            }),
        };
        assert!(cfg.needs_tiling(vae, f, h, w));
        let plan = cfg.plan(vae, f, h, w);
        assert!(
            plan.t.len() > 1 && plan.h.len() > 1 && plan.w.len() > 1,
            "the fixture must tile all three axes"
        );
        assert!(
            plan.h.last().unwrap().out_stop - plan.h.last().unwrap().out_start
                < plan.h[0].out_stop - plan.h[0].out_start,
            "the spatial axes must be ragged"
        );
        // NCTHW, channel axis 1, tiled axes [2, 3, 4].
        let shape = [1, 2, f, h, w];
        let count: i32 = shape.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|i| {
                let (frame, row, col) =
                    ((i / (h * w) % f) as f32, (i / w % h) as f32, (i % w) as f32);
                (i as f32 * 0.037).sin() * 0.4 + frame * 0.03 - row * 0.021 + col * 0.017
            })
            .collect();
        (Array::from_slice(&values, &shape), plan, [2, 3, 4])
    }

    /// The **non-causal** temporal geometry — Wan 2.1 z16, whose temporal axis tiles exactly like a
    /// spatial one (`out_f = f·4`, `left_from_0 = false`). Its three temporal tiles are equal-length;
    /// only the spatial axes are ragged here.
    fn ragged_video_plan() -> (Array, TilePlan, [i32; 3]) {
        let (denorm, plan, axes) = ragged_plan_for(VaeTiling::WAN, 7, 9, 11);
        const { assert!(!VaeTiling::WAN.causal_temporal) };
        assert_eq!(plan.out_f, 7 * VaeTiling::WAN.temporal_scale);
        (denorm, plan, axes)
    }

    /// The **causal** temporal geometry — Wan 2.2 z48, the mapping LTX and WAN22 actually ship
    /// (`out_f = 1 + (f−1)·scale`, every tile after the first starting one latent frame early, and the
    /// `left_from_0` mask convention that fades a tile in from exactly 0 rather than from the first
    /// interior step).
    ///
    /// This geometry is structurally different from the non-causal one in three ways the fold has to
    /// survive: temporal tiles are RAGGED in output length even when equal in latent length, their
    /// output spans are not `latent·scale`, and the leading mask sample is a true zero. It is also the
    /// only fixture where the synthetic decode's `f·scale` frames exceed the plan's declared span, so
    /// it drives the `at = min(decoded, declared)` truncation on the temporal axis for real.
    fn ragged_causal_video_plan() -> (Array, TilePlan, [i32; 3]) {
        let (denorm, plan, axes) = ragged_plan_for(VaeTiling::WAN22, 7, 9, 11);
        const { assert!(VaeTiling::WAN22.causal_temporal) };
        assert_eq!(plan.out_f, 1 + (7 - 1) * VaeTiling::WAN22.temporal_scale);
        assert!(
            plan.t.iter().skip(1).any(|t| t.mask[0] == 0.0),
            "a causal temporal tile must fade in from exactly 0"
        );
        assert!(
            plan.t
                .iter()
                .any(|t| t.out_stop - t.out_start != plan.t[0].out_stop - plan.t[0].out_start),
            "the causal temporal axis must be ragged in output length"
        );
        (denorm, plan, axes)
    }

    /// A decode closure that is **not** tile-consistent: it upsamples to the plan's scales and then
    /// offsets by a statistic of the tile it was handed, so neighbouring tiles genuinely disagree in
    /// their overlap and the blend weights plus their normalizer are load-bearing. A fold that
    /// mis-places a tile, drops a mask, or mis-normalizes cannot pass by reconstructing a
    /// partition-of-unity identity.
    fn seam_bearing_decode(tile: &Array) -> Result<Array> {
        seam_bearing_decode_for(VaeTiling::WAN)(tile)
    }

    /// Block-upsample an NCTHW tile to `vae`'s temporal and spatial scales.
    fn block_upsample(tile: &Array, vae: VaeTiling) -> Result<Array> {
        let up = Array::repeat_axis::<f32>(tile.clone(), vae.temporal_scale, 2)?;
        let up = Array::repeat_axis::<f32>(up, vae.spatial_scale, 3)?;
        Ok(Array::repeat_axis::<f32>(up, vae.spatial_scale, 4)?)
    }

    /// [`seam_bearing_decode`] at an arbitrary VAE's scales, so the causal and non-causal temporal
    /// geometries can be driven by the same decode.
    fn seam_bearing_decode_for(vae: VaeTiling) -> impl Fn(&Array) -> Result<Array> + Copy {
        move |tile: &Array| {
            let up = block_upsample(tile, vae)?;
            let bias = tile.mean(None)?;
            add(&up, &bias).map_err(Into::into)
        }
    }

    /// `max_abs_rgb_u8` — the shipped decode-quality corpus's metric: `clip(x·0.5 + 0.5, 0, 1)·255`
    /// rounded to `u8` (the [`crate::image::decoded_to_image`] mapping), then the max absolute
    /// per-byte difference. Never a scale-invariant cosine.
    fn max_abs_rgb_u8(left: &Array, right: &Array) -> u32 {
        let quantize = |x: &Array| -> Vec<u8> {
            x.as_slice::<f32>()
                .iter()
                .map(|v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect()
        };
        let (a, b) = (quantize(left), quantize(right));
        assert_eq!(a.len(), b.len(), "pixel buffers differ in length");
        a.iter()
            .zip(&b)
            .map(|(l, r)| u32::from(l.abs_diff(*r)))
            .max()
            .unwrap_or_default()
    }

    /// sc-18320 acceptance (b): the fold must not move behaviour the pad-to-full accumulator already
    /// produced. Graded with the corpus metric (`max_abs_rgb_u8`) on a ragged three-axis plan whose
    /// decode is deliberately seam-bearing, so the blend weights and the separable normalizer both
    /// matter. Re-association of the same products moves values by float rounding only, which the
    /// u8 quantization must not see at all.
    #[test]
    fn fold_matches_the_pad_to_full_accumulator() {
        let (denorm, plan, axes) = ragged_video_plan();
        let reference = pad_to_full_reference(&denorm, &plan, axes, seam_bearing_decode).unwrap();
        let folded = tiled_decode(&denorm, &plan, axes, None, seam_bearing_decode).unwrap();
        reference.eval().unwrap();
        folded.eval().unwrap();
        assert_eq!(folded.shape(), reference.shape());
        assert_eq!(
            max_abs_rgb_u8(&folded, &reference),
            0,
            "the fold moved an admitted pixel"
        );
        let max = folded
            .as_slice::<f32>()
            .iter()
            .zip(reference.as_slice::<f32>())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-5,
            "re-association must stay at float-rounding scale: max|Δ|={max:.3e}"
        );
    }

    /// sc-18320 acceptance (b), on the **causal** temporal mapping LTX and WAN22 ship.
    ///
    /// [`fold_matches_the_pad_to_full_accumulator`] drives Wan 2.1's non-causal temporal axis, which
    /// tiles exactly like a spatial one. The causal mapping is a different geometry — `out_stop =
    /// 1 + (end−1)·scale`, every tile after the first starting one latent frame early, a `left_from_0`
    /// mask whose first sample is a true zero, and output spans that differ between tiles even when
    /// their latent spans do not. It also makes the synthetic decode overshoot the plan's declared
    /// temporal span, so this is the fixture that drives the `at = min(decoded, declared)` truncation
    /// on the temporal axis. The fold must reproduce the predecessor there too.
    #[test]
    fn fold_matches_the_pad_to_full_accumulator_on_a_causal_temporal_plan() {
        let (denorm, plan, axes) = ragged_causal_video_plan();
        let decode = seam_bearing_decode_for(VaeTiling::WAN22);
        let reference = pad_to_full_reference(&denorm, &plan, axes, decode).unwrap();
        let folded = tiled_decode(&denorm, &plan, axes, None, decode).unwrap();
        reference.eval().unwrap();
        folded.eval().unwrap();
        assert_eq!(folded.shape(), reference.shape());
        assert_eq!(
            folded.shape()[axes[0] as usize],
            plan.out_f,
            "the causal temporal axis must assemble to 1 + (f−1)·scale"
        );
        assert_eq!(
            max_abs_rgb_u8(&folded, &reference),
            0,
            "the fold moved an admitted pixel on the causal geometry"
        );
        let max = folded
            .as_slice::<f32>()
            .iter()
            .zip(reference.as_slice::<f32>())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-5,
            "re-association must stay at float-rounding scale: max|Δ|={max:.3e}"
        );
    }

    /// sc-18320: the *effective* weight the blend applies to each tile at each output coordinate is
    /// unchanged, so an overlap narrower than the decoder's receptive field still attenuates the
    /// conv-halo seam by exactly as much as before — the accumulator must not quietly widen or
    /// narrow that. Pinned by driving a decode that returns a constant per tile: the result is then
    /// literally the normalized weight profile, and it must match the predecessor's to f32 exactness.
    #[test]
    fn fold_preserves_the_effective_blend_weight_profile() {
        let (denorm, plan, axes) = ragged_video_plan();
        // Tile index → a distinct constant, so the normalized output at each coordinate is the
        // convex combination of tile indices — i.e. the effective weight profile itself.
        let stamp = std::cell::Cell::new(0f32);
        let stamped = |tile: &Array| -> Result<Array> {
            stamp.set(stamp.get() + 1.0);
            let up = Array::repeat_axis::<f32>(tile.clone(), 4, 2)?;
            let up = Array::repeat_axis::<f32>(up, 8, 3)?;
            let up = Array::repeat_axis::<f32>(up, 8, 4)?;
            let zeroed = multiply(&up, scalar(0.0))?;
            add(&zeroed, scalar(stamp.get())).map_err(Into::into)
        };
        let reference = pad_to_full_reference(&denorm, &plan, axes, stamped).unwrap();
        stamp.set(0.0);
        let folded = tiled_decode(&denorm, &plan, axes, None, &stamped).unwrap();
        reference.eval().unwrap();
        folded.eval().unwrap();
        let max = folded
            .as_slice::<f32>()
            .iter()
            .zip(reference.as_slice::<f32>())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-4,
            "the effective per-tile weight profile moved: max|Δ|={max:.3e}"
        );
    }

    /// sc-18320: the placement primitive grows its input along **one** axis, and skips the pad
    /// entirely for an axis the tile already covers end-to-end (the untiled-axis case every image
    /// decode hits on its singleton temporal axis). This is what bounds a fold stage's intermediate
    /// to a strip rather than the full output.
    #[test]
    fn accumulate_along_axis_grows_one_axis_and_skips_a_covered_one() {
        let x = Array::from_slice(
            &(0..24).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 2, 3, 4],
        );
        // Placed at offset 1 of a 7-long axis 2: only axis 2 grows.
        let grown = accumulate_along_axis(None, &x, 2, 1, 3, 7).unwrap();
        assert_eq!(grown.shape(), &[1, 2, 7, 4]);
        // A claimed extent that does not match the array is refused — the fully-covered fast path
        // below returns the input verbatim, so an unchecked mismatch would mis-shape a stage silently.
        let mismatch = accumulate_along_axis(None, &x, 2, 0, 5, 7);
        assert!(
            matches!(mismatch, Err(Error::Msg(ref m)) if m.contains("whose extent there is 3")),
            "a wrong extent must be refused, got {mismatch:?}"
        );
        // A fully covered axis is returned as-is — no padded copy at all.
        let untouched = accumulate_along_axis(None, &x, 2, 0, 3, 3).unwrap();
        assert_eq!(untouched.shape(), x.shape());
        // Accumulating into an existing stage keeps the stage's shape.
        let summed = accumulate_along_axis(Some(grown), &x, 2, 4, 3, 7).unwrap();
        assert_eq!(summed.shape(), &[1, 2, 7, 4]);
        summed.eval().unwrap();
        // Both placements landed where they were asked to, and nowhere else.
        let read = summed.as_slice::<f32>();
        assert_eq!(read[0..4], [0.0; 4], "row 0 of the padded axis stays zero");
        assert_eq!(
            read[4..8],
            x.as_slice::<f32>()[0..4],
            "offset 1 holds x row 0"
        );
        assert_eq!(
            read[16..20],
            x.as_slice::<f32>()[0..4],
            "offset 4 holds x row 0 again"
        );
    }

    /// sc-18320: the point of the fold. A full-output-shaped accumulator must not exist until the
    /// first **temporal** tile closes — every tile before that is assembled in strips and slabs. The
    /// predecessor route created two full-output-sized arrays on its very first tile and on every
    /// tile after. Asserts SHAPES and plan-derived structure only, never a corpus population.
    #[test]
    fn fold_reaches_full_output_shape_only_when_a_temporal_tile_closes() {
        let (denorm, plan, axes) = ragged_video_plan();
        let full = [plan.out_f, plan.out_h, plan.out_w];
        let observed = std::cell::RefCell::new(Vec::<Vec<Vec<i32>>>::new());
        tiled_decode_with_hooks(
            &denorm,
            &plan,
            axes,
            None,
            seam_bearing_decode,
            |accumulators| {
                observed
                    .borrow_mut()
                    .push(accumulators.iter().map(|a| a.shape().to_vec()).collect());
                for accumulator in accumulators {
                    accumulator.eval()?;
                }
                Ok(())
            },
            || {},
        )
        .unwrap();
        let observed = observed.into_inner();
        let extents = |shape: &[i32]| {
            [
                shape[axes[0] as usize],
                shape[axes[1] as usize],
                shape[axes[2] as usize],
            ]
        };
        assert_eq!(
            observed.len(),
            plan.t.len() * plan.h.len() * plan.w.len(),
            "materialization runs exactly once per tile"
        );
        let first_full = observed
            .iter()
            .position(|live| live.iter().any(|shape| extents(shape) == full))
            .expect("the assembly must reach the full output shape");
        assert_eq!(
            first_full,
            plan.h.len() * plan.w.len() - 1,
            "no full-output-shaped array may exist before the first temporal tile closes"
        );
        assert!(
            observed
                .iter()
                .all(|live| live.iter().filter(|shape| extents(shape) == full).count() <= 1),
            "at most one full-output-shaped accumulator may be live at a time"
        );
        // Every other live accumulator is short along at least one tiled axis: the w-stage strip is
        // short along t and h, the h-stage slab short along t.
        assert!(
            observed
                .iter()
                .flatten()
                .filter(|shape| extents(shape) != full)
                .all(|shape| {
                    let [t, h, w] = extents(shape);
                    w == plan.out_w && (t < plan.out_f || h < plan.out_h)
                }),
            "the fold's inner stages must span only the w axis (strip) or w and h (slab)"
        );
    }

    /// sc-18320: a cancel tripped *during* the assembly is honored, the accumulators are dropped, and
    /// the release hook still runs — the discipline the LTX seam (sc-19655) contributed, now on the
    /// shared loop. Lazy eval cannot mask it: `materialize` forces every live accumulator first.
    #[test]
    fn fold_honors_a_cancel_tripped_mid_assembly() {
        let (denorm, plan, axes) = ragged_video_plan();
        let cancel = CancelFlag::new();
        let decodes = std::cell::Cell::new(0usize);
        let releases = std::cell::Cell::new(0usize);
        let after_first = cancel.clone();
        let result = tiled_decode_with_hooks(
            &denorm,
            &plan,
            axes,
            Some(&cancel),
            |tile| {
                decodes.set(decodes.get() + 1);
                seam_bearing_decode(tile)
            },
            |accumulators| {
                for accumulator in accumulators {
                    accumulator.eval()?;
                }
                Ok(())
            },
            || {
                releases.set(releases.get() + 1);
                after_first.cancel();
            },
        );
        assert!(matches!(result, Err(Error::Canceled)));
        assert_eq!(
            decodes.get(),
            1,
            "the cancel must land after exactly one tile"
        );
        assert_eq!(
            releases.get(),
            2,
            "the completed tile releases, and so does the cancel path once accumulators are dropped"
        );
    }

    /// sc-18320: a pre-tripped cancel runs no tile and therefore no hook — there is nothing dead to
    /// release, so calling the release hook anyway would be a false signal to a caller that returns
    /// buffers to the OS.
    #[test]
    fn fold_pretripped_cancel_runs_neither_hook() {
        let (denorm, plan, axes) = ragged_video_plan();
        let cancel = CancelFlag::new();
        cancel.cancel();
        let releases = std::cell::Cell::new(0usize);
        let result = tiled_decode_with_hooks(
            &denorm,
            &plan,
            axes,
            Some(&cancel),
            seam_bearing_decode,
            |_| unreachable!("a pre-tripped cancel materializes nothing"),
            || releases.set(releases.get() + 1),
        );
        assert!(matches!(result, Err(Error::Canceled)));
        assert_eq!(releases.get(), 0);
    }

    /// sc-18320: a failing tile releases exactly once, after its accumulators are dropped.
    #[test]
    fn fold_failed_tile_releases_after_dropping_accumulators() {
        let (denorm, plan, axes) = ragged_video_plan();
        let releases = std::cell::Cell::new(0usize);
        let result = tiled_decode_with_hooks(
            &denorm,
            &plan,
            axes,
            None,
            |_| Err(Error::Msg("synthetic decoder failure".into())),
            |_| unreachable!("a failed tile produces no accumulator to materialize"),
            || releases.set(releases.get() + 1),
        );
        assert!(matches!(result, Err(Error::Msg(m)) if m == "synthetic decoder failure"));
        assert_eq!(releases.get(), 1);
    }

    /// sc-18320: the separable normalizer is only valid if each axis tile writes one extent. A
    /// decoder that returns a different extent for the same axis tile must be REFUSED with a
    /// catchable error, not silently normalized against a blend it did not perform.
    #[test]
    fn fold_refuses_a_non_separable_decode_extent() {
        let (denorm, plan, axes) = ragged_video_plan();
        let calls = std::cell::Cell::new(0usize);
        let result = tiled_decode(&denorm, &plan, axes, None, |tile| {
            calls.set(calls.get() + 1);
            let decoded = seam_bearing_decode(tile)?;
            // Shorten the SECOND tile's width output. Its w tile is a different index, so the first
            // offence is the third tile revisiting w tile 0 with a full extent — either way the
            // per-axis extent bookkeeping must catch a decoder that is not separable.
            if calls.get() == 2 {
                let ds = decoded.shape();
                return slice_axis(&decoded, axes[2], 0, ds[axes[2] as usize] - 1);
            }
            Ok(decoded)
        });
        let message = match result {
            Err(Error::Msg(message)) => message,
            other => panic!("expected a refusal, got {other:?}"),
        };
        assert!(
            message.contains("one written extent per axis tile"),
            "the refusal must name the separability contract: {message}"
        );
    }

    /// sc-18320: the blend normalizer must carry **every** tiled axis.
    ///
    /// The shipped trapezoidal masks are a partition of unity, so each axis's coverage is ≈1 and a
    /// normalizer missing a whole factor is invisible — the equivalence fixtures above cannot see it.
    /// This plan deliberately breaks that: two overlapping tiles per axis with all-ones masks, so
    /// coverage is `[1, 2, 2, 1]` on each of t, h and w and the normalizer is the only thing standing
    /// between the accumulation and a 2×/4×/8× over-count. An identity decode must therefore
    /// reconstruct the input exactly, and must match the pad-to-full accumulator's own normalization.
    ///
    /// Executed mutation: dropping any one of the three factors from
    /// `AxisCoverage::normalizer` turns this RED (and only this) — 2× too large in the doubly-covered
    /// band of that axis.
    #[test]
    fn fold_normalizer_carries_every_tiled_axis() {
        let (extent, edge) = (4i32, 3i32);
        let count: i32 = extent * extent * extent;
        // Position-dependent: value = 100·t + 10·h + w, distinct at every coordinate.
        let values: Vec<f32> = (0..count)
            .map(|i| {
                let t = i / (extent * extent);
                let h = i / extent % extent;
                let w = i % extent;
                (t * 100 + h * 10 + w) as f32
            })
            .collect();
        let denorm = Array::from_slice(&values, &[1, 1, extent, extent, extent]);
        let overlapping = || {
            vec![
                AxisTile {
                    start: 0,
                    end: edge,
                    out_start: 0,
                    out_stop: edge,
                    mask: vec![1.0; edge as usize],
                },
                AxisTile {
                    start: extent - edge,
                    end: extent,
                    out_start: extent - edge,
                    out_stop: extent,
                    mask: vec![1.0; edge as usize],
                },
            ]
        };
        let plan = TilePlan {
            t: overlapping(),
            h: overlapping(),
            w: overlapping(),
            out_f: extent,
            out_h: extent,
            out_w: extent,
        };

        let folded =
            tiled_decode(&denorm, &plan, [2, 3, 4], None, |tile| Ok(tile.clone())).unwrap();
        let reference =
            pad_to_full_reference(&denorm, &plan, [2, 3, 4], |tile| Ok(tile.clone())).unwrap();
        folded.eval().unwrap();
        reference.eval().unwrap();
        assert_eq!(folded.shape(), denorm.shape());
        let max_vs_input = folded
            .as_slice::<f32>()
            .iter()
            .zip(denorm.as_slice::<f32>())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_vs_input < 1e-3,
            "a coverage-normalized identity decode must reconstruct the input: max|Δ|={max_vs_input:.3e}"
        );
        let max_vs_reference = folded
            .as_slice::<f32>()
            .iter()
            .zip(reference.as_slice::<f32>())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_vs_reference < 1e-3,
            "the fold's normalizer diverged from the accumulated one: max|Δ|={max_vs_reference:.3e}"
        );
    }

    /// sc-18320: coverage must count only the extent a tile actually **wrote**.
    ///
    /// `at`/`ah`/`aw` clamp to the decoder's real output when it is shorter than the plan's declared
    /// span, and the blend then uses `mask[..extent]`. The coverage vector has to be summed over the
    /// same prefix — counting the whole declared mask would normalize positions the tile never wrote
    /// against a weight it never applied. Here the first w tile spans 2 latent columns but declares 3
    /// output columns, so its written extent is 2 while its mask is 3 long, and output column 2 is
    /// covered by the second tile alone.
    ///
    /// Executed mutation: summing `tile.mask` instead of `tile.mask[..extent]` halves column 2 and
    /// turns this RED.
    #[test]
    fn fold_coverage_counts_only_the_written_extent() {
        let values: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let denorm = Array::from_slice(&values, &[1, 1, 2, 2, 4]);
        let whole = AxisTile {
            start: 0,
            end: 2,
            out_start: 0,
            out_stop: 2,
            mask: vec![1.0; 2],
        };
        let plan = TilePlan {
            t: vec![whole.clone()],
            h: vec![whole],
            w: vec![
                // Spans 2 latent columns but declares 3 output columns: written extent 2, mask 3.
                AxisTile {
                    start: 0,
                    end: 2,
                    out_start: 0,
                    out_stop: 3,
                    mask: vec![1.0; 3],
                },
                AxisTile {
                    start: 1,
                    end: 4,
                    out_start: 1,
                    out_stop: 4,
                    mask: vec![1.0; 3],
                },
            ],
            out_f: 2,
            out_h: 2,
            out_w: 4,
        };

        let folded =
            tiled_decode(&denorm, &plan, [2, 3, 4], None, |tile| Ok(tile.clone())).unwrap();
        let reference =
            pad_to_full_reference(&denorm, &plan, [2, 3, 4], |tile| Ok(tile.clone())).unwrap();
        folded.eval().unwrap();
        reference.eval().unwrap();
        assert_eq!(folded.shape(), &[1, 1, 2, 2, 4]);
        let max = folded
            .as_slice::<f32>()
            .iter()
            .zip(reference.as_slice::<f32>())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-4,
            "the truncated-extent blend diverged from the accumulated one: max|Δ|={max:.3e}"
        );
        // Column 2 is written by the second tile only, so it must come back at unit weight — the
        // input value, not half of it.
        let read = folded.as_slice::<f32>();
        let input = denorm.as_slice::<f32>();
        for row in 0..4usize {
            assert!(
                (read[row * 4 + 2] - input[row * 4 + 2]).abs() < 1e-4,
                "singly-covered column 2 of row {row} was normalized against an unwritten weight: \
                 got {}, want {}",
                read[row * 4 + 2],
                input[row * 4 + 2]
            );
        }
    }

    /// sc-18320: an axis tile that writes past the output extent is REFUSED before any placement.
    /// Without the bounds check the coverage vector would be indexed out of range (a panic, not a
    /// catchable error) or — with the check weakened to a clamp — silently normalized against a blend
    /// the accumulation never performed.
    #[test]
    fn fold_refuses_an_axis_tile_that_writes_past_the_output() {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 1, 4, 2, 2]);
        let unit = AxisTile {
            start: 0,
            end: 2,
            out_start: 0,
            out_stop: 2,
            mask: vec![1.0; 2],
        };
        let plan = TilePlan {
            // The temporal tile claims output [3, 7) of a 4-long axis.
            t: vec![AxisTile {
                start: 0,
                end: 4,
                out_start: 3,
                out_stop: 7,
                mask: vec![1.0; 4],
            }],
            h: vec![unit.clone()],
            w: vec![unit],
            out_f: 4,
            out_h: 2,
            out_w: 2,
        };
        let result = tiled_decode(&denorm, &plan, [2, 3, 4], None, |tile| Ok(tile.clone()));
        let message = match result {
            Err(Error::Msg(message)) => message,
            other => panic!("expected a refusal, got {other:?}"),
        };
        assert!(
            message.contains("outside the 4-long output axis"),
            "the refusal must name the axis it overran: {message}"
        );
    }

    /// sc-12438: `check_output_writable` allows an output exactly AT the bound and refuses the first
    /// element past it — the sharp `> MAX_WRITABLE_ELEMS` boundary, not `>=`. Mutation-discriminating:
    /// flipping the comparison, the constant, or an off-by-one turns one of these two assertions red.
    #[test]
    fn check_output_writable_boundary_is_sharp() {
        assert!(
            check_output_writable(MAX_WRITABLE_ELEMS, 1, 1, 1).is_ok(),
            "an output exactly at the bound must be allowed"
        );
        assert!(
            check_output_writable(MAX_WRITABLE_ELEMS + 1, 1, 1, 1).is_err(),
            "one element past the bound must be refused"
        );
        // A realistic RGB video geometry just over the bound (LTX 1280²·441f class): 3·441·1280·1280.
        let over = 3i64 * 441 * 1280 * 1280;
        assert!(over > MAX_WRITABLE_ELEMS);
        assert!(check_output_writable(over, 441, 1280, 1280).is_err());
    }

    /// sc-12748: a tiled decode whose **assembled output crosses `i32::MAX`** now RENDERS (the sc-12438
    /// refusal is retired) and reads back correctly — the payoff of this slice, on the shared loop. Drives
    /// the real `pad`-and-accumulate + `contiguous`(multi-dimensional reshape)+`as_slice` path with a tiny
    /// position-dependent latent placed into an over-bound `out_h·out_w·3 > i32::MAX` output, and checks
    /// the placed voxels (sub-bound offsets) hold the identity-decoded values while the rest is zero.
    /// `#[ignore]`d — it allocates a ~2.19e9-element (8.7 GiB) output accumulator.
    #[test]
    #[ignore = "sc-12748 heavy over-bound tiled-decode render (~12 GiB); run with --ignored on Metal"]
    fn tiled_decode_renders_over_bound_output() {
        const I32_MAX: i64 = i32::MAX as i64;
        // Small NCTHW latent, channel axis 1, tiled axes [2,3,4]. Position-dependent so a scrambled
        // read-back is caught: latent[0,c,0,h,w] = c*100 + h*10 + w (distinct over c,h,w ∈ 0..4).
        let mut vals = vec![0f32; 3 * 4 * 4];
        for c in 0..3 {
            for h in 0..4 {
                for w in 0..4 {
                    vals[(c * 4 + h) * 4 + w] = (c * 100 + h * 10 + w) as f32;
                }
            }
        }
        let denorm = Array::from_slice(&vals, &[1, 3, 1, 4, 4]);
        let axis = |out_stop: i32| AxisTile {
            start: 0,
            end: 4,
            out_start: 0,
            out_stop,
            mask: vec![1.0; out_stop as usize],
        };
        // out_f=1, out_h=out_w=27_000 → 3·1·27000·27000 = 2.187e9 = 1.019× i32::MAX (in the probed band).
        // The h/w tiles place the 4-wide identity-decoded tile at offset 0; the rest is zero-padded.
        let out_hw = 27_000i32;
        assert!(
            3 * (out_hw as i64) * (out_hw as i64) > I32_MAX,
            "geometry must cross the bound"
        );
        let plan = TilePlan {
            t: vec![AxisTile {
                start: 0,
                end: 1,
                out_start: 0,
                out_stop: 1,
                mask: vec![1.0; 1],
            }],
            h: vec![axis(out_hw)],
            w: vec![axis(out_hw)],
            out_f: 1,
            out_h: out_hw,
            out_w: out_hw,
        };
        // Must NOT refuse — it renders the over-bound accumulator.
        let out = tiled_decode(&denorm, &plan, [2, 3, 4], None, |tile| Ok(tile.clone()))
            .expect("over-bound tiled decode must render, not refuse (sc-12748)");
        out.eval().unwrap();
        assert_eq!(out.shape(), &[1, 3, 1, out_hw, out_hw]);
        let flat = out.as_slice::<f32>();
        assert_eq!(flat.len() as i64, 3 * out_hw as i64 * out_hw as i64);
        // Placed region: [0,c,0,h,w] at flat ((c*out_hw + h)*out_hw + w) must equal the identity latent.
        let at =
            |c: i64, h: i64, w: i64| flat[((c * out_hw as i64 + h) * out_hw as i64 + w) as usize];
        for c in 0..3i64 {
            for h in 0..4i64 {
                for w in 0..4i64 {
                    let want = (c * 100 + h * 10 + w) as f32;
                    let got = at(c, h, w);
                    assert!(
                        (got - want).abs() < 1e-3,
                        "over-bound read-back scrambled at (c={c},h={h},w={w}): got {got}, want {want}"
                    );
                }
            }
        }
        // An above-2^31 flat offset (c=2 is beyond ~1.46e9; h=w=13000 → offset ≈ 1.81e9; and the very
        // last element at ≈2.187e9) must read back as the zero pad, proving the >i32::MAX region is
        // addressed correctly, not aliased onto the placed tile.
        assert_eq!(
            at(2, 13_000, 13_000),
            0.0,
            "over-bound zero-pad region must read 0"
        );
        assert_eq!(
            flat[(3 * out_hw as i64 * out_hw as i64 - 1) as usize],
            0.0,
            "last (>2^31) elem"
        );
    }
}
