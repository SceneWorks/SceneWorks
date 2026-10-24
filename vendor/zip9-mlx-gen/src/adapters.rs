//! Adapter framework — LoRA + LoKr applied as forward-time residuals over a shared
//! base. Quantized-safe: the base is never fused/mutated. Ported from the sc-2338
//! spike; mirrors the Python mflux fork's `LoKrLinear` / `FusedLoRALinear` (sc-2216).
//!
//! The base is a real `nn::Linear` *or* `nn::QuantizedLinear` (sc-2342), so quantization
//! and adapters compose: `base(x) + Σ adapter.residual(x)`. Forward is taken by `&self`
//! (we call the underlying ops directly rather than the `&mut self` `Module` trait), so a
//! whole model tree can be evaluated through shared references.
//!
//! Adapters are installed by dotted path via [`AdaptableHost`] / [`install_adapter`] — the
//! Rust stand-in for Python's dynamic `getattr`-swap, since mlx-rs flattens module params to
//! `Array` leaves and cannot replace a submodule in place.

use mlx_rs::{
    module::Param,
    nn::{Linear, QuantizedLinear},
    ops::{add, addmm, einsum, kron, matmul, multiply},
    Array, Dtype,
};

use crate::array::scalar;
use crate::nn::quantized_matmul_with_bias;
use crate::Result;

pub mod loader;

/// Reconstruct a LoKr weight delta `ΔW = (alpha/rank) · kron(w1, w2)`, reshaped to the
/// base weight's logical `[out, in]` and cast to `out_dtype`. Each Kronecker factor is either
/// full (`w1` / `w2`) or a low-rank product (`w1_a @ w1_b` / `w2_a @ w2_b`). Mirrors
/// PEFT/LyCORIS `LoKrLayer.get_delta_weight` (pending the sc-2324 cross-impl parity check).
///
/// `out_dtype` is `Bfloat16` for the fork-parity residual path (Z-Image/Qwen — PARITY-BF16,
/// sc-2609) and `Float32` for the SDXL merge path (f32-everywhere, no fork to match — sc-2640).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_lokr_delta(
    alpha: f32,
    rank: f32,
    base_shape: &[i32],
    w1: Option<&Array>,
    w1_a: Option<&Array>,
    w1_b: Option<&Array>,
    w2: Option<&Array>,
    w2_a: Option<&Array>,
    w2_b: Option<&Array>,
    out_dtype: Dtype,
) -> Result<Array> {
    // Guard the metadata-derived `alpha`/`rank` at the shared seam so EVERY consumer inherits it
    // (F-141, sc-11129). A file that stamps `rank = 0` (alpha then defaults to rank) makes the scale
    // `0/0 = NaN`, baked into the reconstructed delta and merged as NaN — NaN-poisoning every
    // subsequent render while the load reports success. `lora_delta` already rejects rank 0 (sc-5252),
    // but this middle path was missed; hoisting the guard here retires the class rather than patching
    // each callsite.
    if !rank.is_finite() || rank <= 0.0 || !alpha.is_finite() {
        return Err(format!(
            "LoKr: invalid metadata scale (rank = {rank}, alpha = {alpha}); rank must be > 0 and \
             alpha finite — a `rank = 0` / non-finite `alpha` would make the delta NaN"
        )
        .into());
    }
    // The SceneWorks peft path bakes `alpha/rank` as the scale and is always linear (no tucker
    // factor, equal-rank factors) — delegate to the general scaled form, which is then byte-identical.
    reconstruct_lokr_delta_scaled(
        alpha / rank,
        base_shape,
        w1,
        w1_a,
        w1_b,
        w2,
        None,
        w2_a,
        w2_b,
        out_dtype,
    )
}

/// Reconstruct a LoKr weight delta `ΔW = scale · kron(w1, w2)`, reshaped to `base_shape` and cast to
/// `out_dtype`. Generalizes [`reconstruct_lokr_delta`] for **third-party LyCORIS** LoKr (sc-3642):
/// the caller passes the final `scale` directly (lycoris derives it per module — `alpha/rank`, or a
/// forced `1.0` when both factors are full), and `w2` may be a **tucker/CP** factor
/// (`w2_t2` + `w2_a` + `w2_b`, lycoris `use_cp`). Each Kronecker factor is full (`w1`/`w2`), a
/// low-rank product (`w1_a@w1_b` / `w2_a@w2_b`), or — for `w2` only — the tucker rebuild
/// `einsum("ij…,ip,jr->pr…", t2, w2_a, w2_b)`. Mirrors LyCORIS `LokrModule.get_weight` + `make_kron`
/// (w1's trailing dims are unsqueezed to w2's rank before the product — the conv case).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_lokr_delta_scaled(
    scale: f32,
    base_shape: &[i32],
    w1: Option<&Array>,
    w1_a: Option<&Array>,
    w1_b: Option<&Array>,
    w2: Option<&Array>,
    w2_t2: Option<&Array>,
    w2_a: Option<&Array>,
    w2_b: Option<&Array>,
    out_dtype: Dtype,
) -> Result<Array> {
    let factor1 = match (w1, w1_a, w1_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w1 missing (need full w1 or w1_a@w1_b)".into()),
    };
    let factor2 = match (w2, w2_t2, w2_a, w2_b) {
        (Some(w), _, _, _) => w.clone(),
        (_, Some(t2), Some(a), Some(b)) => rebuild_tucker(t2, a, b)?,
        (_, None, Some(a), Some(b)) => matmul(a, b)?,
        _ => {
            return Err("LoKr: w2 missing (need full w2, tucker t2+w2_a+w2_b, or w2_a@w2_b)".into())
        }
    };
    // LyCORIS `make_kron` unsqueezes w1's TRAILING dims to w2's rank before `torch.kron` (the conv
    // case: w1 `[a,c]` → `[a,c,1,1]` against w2 `[b,d,kH,kW]`). Linear factors share rank → no-op.
    let factor1 = match_trailing_rank(&factor1, factor2.shape().len())?;
    let delta = multiply(&kron(&factor1, &factor2)?, scalar(scale))?;
    Ok(delta.reshape(base_shape)?.as_dtype(out_dtype)?)
}

/// Reshape `a` to append trailing length-1 dims until it has `ndim` dimensions (LyCORIS `make_kron`'s
/// `w1.unsqueeze(-1)` loop). A no-op when `a` already has ≥ `ndim` dims.
fn match_trailing_rank(a: &Array, ndim: usize) -> Result<Array> {
    let mut shape = a.shape().to_vec();
    if shape.len() >= ndim {
        return Ok(a.clone());
    }
    shape.resize(ndim, 1);
    Ok(a.reshape(&shape)?)
}

/// LyCORIS `rebuild_tucker(t, wa, wb) = einsum("i j …, i p, j r -> p r …", t, wa, wb)` — the CP/tucker
/// rebuild for a conv `w2` (`use_cp`). `t2` is `[i, j, kH, kW]`, `wa` is `[i, p]`, `wb` is `[j, r]`,
/// yielding `[p, r, kH, kW]`. Only the 4-D conv form lycoris emits is supported (others error loudly).
fn rebuild_tucker(t2: &Array, wa: &Array, wb: &Array) -> Result<Array> {
    if t2.shape().len() != 4 {
        return Err(format!(
            "LoKr tucker: expected a 4-D lokr_t2 [in,out,kH,kW], got shape {:?}",
            t2.shape()
        )
        .into());
    }
    Ok(einsum("ijhw,ip,jr->prhw", [t2, wa, wb])?)
}

/// The two small Kronecker factors of a LoKr delta, kept UNMATERIALIZED for a deferred
/// structured forward (sc-10050). `ΔW = scale · kron(w1, w2)` reshapes to the base's `[out, in]`,
/// but the Kronecker–vector identity lets us apply it WITHOUT ever forming that `[out, in]` tensor:
/// with `w1` `[a, c]` (LyCORIS `shape[0][0] × shape[1][0]`) and `w2` `[b, d]`
/// (`shape[0][1] × shape[1][1]`), so `out = a·b` and `in = c·d`, the residual `y = x · ΔWᵀ` is
///   `Y = w1 · X · w2ᵀ`  (then flatten row-major `[.., a, b] → [.., out]`),
/// where `X = reshape(x, [.., c, d])`. Two small matmuls (`[a,c]·[..,c,d]` then `·[d,b]`) touch only
/// the factor shapes — never `[out, in]` — so a LoKr applies on a packed Q4/Q8 base at the same
/// memory profile as a plain LoRA (sc-10050 / epic 10043). `w1`/`w2` are the *small* Kronecker
/// factors, materialized from their low-rank `w_a·w_b` inner products if decomposed (that product is
/// bounded by the factor dims, NOT `out×in`), so every linear LoKr variant is deferrable this way.
#[derive(Clone)]
pub struct LokrFactors {
    /// `[a, c]` — the left Kronecker factor (`out = a·b`, `in = c·d`).
    pub w1: Array,
    /// `[b, d]` — the right Kronecker factor.
    pub w2: Array,
    /// `a` — leading (row) count of `w1`, so the flattened output index is `p·b + q`.
    pub a: i32,
    /// `b` — leading (row) count of `w2`.
    pub b: i32,
    /// `c` — trailing (col) count of `w1`, so the flattened input index is `r·d + s`.
    pub c: i32,
    /// `d` — trailing (col) count of `w2`.
    pub d: i32,
    /// The **pre-bake** `scale` (sc-15265). `scale` is folded into [`w2`](Self::w2) at build time and
    /// is not otherwise recoverable from the factors — but [`Adapter::is_disabled`] needs it to
    /// short-circuit a disabled structured LoKr exactly as it short-circuits LoRA/LoKr. Keeping it
    /// here is what makes the short-circuit *universal* instead of a two-of-three exemption: at
    /// `scale == 0` the residual is never computed, so a `w2` factor carrying a non-finite value (an
    /// `Inf` in a third-party checkpoint — [`build_lokr_factors`] guards the *scale* for finiteness
    /// but cannot vouch for the factors) can no longer turn `Inf · 0.0` into a `NaN` that poisons the
    /// output. It is otherwise unused: the residual math still reads the already-baked `w2`.
    pub scale: f32,
}

/// Build the small [`LokrFactors`] from a LoKr module's factors (full `w1`/`w2` or low-rank
/// `w_a·w_b`), baking `scale` into `w2` and casting to `out_dtype` — the deferred, allocation-free
/// counterpart to [`reconstruct_lokr_delta_scaled`] (which materializes the full `[out, in]` delta).
/// Only the **linear** (2-D-factor) LoKr forms are deferrable via the vec-trick; a **tucker/CP** `w2`
/// (`w2_t2`, conv-only in LyCORIS) has no 2-D matrix form, so this returns `Ok(None)` and the caller
/// falls back to materialization (dense) or a clear error (packed). `base_shape` is the target's
/// logical `[out, in]`, used to confirm `a·b == out` and `c·d == in`.
#[allow(clippy::too_many_arguments)]
pub fn build_lokr_factors(
    scale: f32,
    base_shape: &[i32],
    w1: Option<&Array>,
    w1_a: Option<&Array>,
    w1_b: Option<&Array>,
    w2: Option<&Array>,
    w2_t2: Option<&Array>,
    w2_a: Option<&Array>,
    w2_b: Option<&Array>,
    out_dtype: Dtype,
) -> Result<Option<LokrFactors>> {
    // Guard the caller-derived `scale` at the shared seam so the packed/deferred path inherits the
    // F-141 protection too (sc-11129): the SDXL/Wan callers compute `(alpha/rank)·strength`, so a
    // `rank = 0` metadata makes `scale = NaN`, which would bake into the structured `w2` and
    // NaN-poison the deferred residual. Refuse it here rather than silently install a NaN factor.
    if !scale.is_finite() {
        return Err(format!("LoKr: non-finite scale ({scale}) — rank = 0 / invalid alpha").into());
    }
    // Tucker/CP `w2` is a 4-D conv factor with no 2-D matrix form — not deferrable via the vec-trick.
    // (LyCORIS only emits `lokr_t2` for conv layers; the Wan DiT adapter surface is all Linear, so
    // this never fires there, but guard it so a conv LoKr falls back rather than silently mis-applies.)
    if w2_t2.is_some() {
        return Ok(None);
    }
    // The two small Kronecker factors — full, or the low-rank product (bounded by the factor dims,
    // NEVER `out×in`). Building `w1_a @ w1_b` yields the small `[a, c]`, not the packed delta.
    let factor1 = match (w1, w1_a, w1_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w1 missing (need full w1 or w1_a@w1_b)".into()),
    };
    let factor2 = match (w2, w2_a, w2_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w2 missing (need full w2 or w2_a@w2_b)".into()),
    };
    // A conv-shaped factor (>2-D) is likewise not a plain matrix — defer to materialization.
    if factor1.shape().len() != 2 || factor2.shape().len() != 2 {
        return Ok(None);
    }
    let (a, c) = (factor1.shape()[0], factor1.shape()[1]);
    let (b, d) = (factor2.shape()[0], factor2.shape()[1]);
    // The base must factor as `out = a·b`, `in = c·d` (a 2-element logical shape). Anything else
    // (a conv weight with kernel dims, or a factor/base mismatch) is not this linear vec-trick.
    if base_shape.len() != 2 || a * b != base_shape[0] || c * d != base_shape[1] {
        return Ok(None);
    }
    // Bake `scale` into `w2` (either factor works; `w2` keeps `w1` a clean copy) and match dtype.
    let factor2 = multiply(&factor2, scalar(scale))?.as_dtype(out_dtype)?;
    let factor1 = factor1.as_dtype(out_dtype)?;
    Ok(Some(LokrFactors {
        w1: factor1,
        w2: factor2,
        a,
        b,
        c,
        d,
        // Retained so a `scale == 0` structured LoKr is recoverably disabled (sc-15265) — see the
        // field doc. The value baked into `w2` above is the authoritative one for the math.
        scale,
    }))
}

/// Reconstruct a **LoHa** (Hadamard-product) weight delta `ΔW = scale · ((w1_a·w1_b) ⊙ (w2_a·w2_b))`,
/// reshaped to `base_shape` and cast to `out_dtype` (sc-3643). Third-party LyCORIS LoHa decomposes a
/// delta as the elementwise product of TWO low-rank products (vs LoKr's Kronecker). With tucker
/// factors (`t1`/`t2`, lycoris `use_cp` for conv) each side is a CP rebuild
/// `einsum("ij…,jr,ip->pr…", t, w_b, w_a)`. `scale` is `alpha/rank` (the caller derives it per
/// module). Mirrors LyCORIS `LohaModule.get_weight` / `loha_diff_weight` (`HadaWeight`/`HadaWeightTucker`):
/// the saved factors map as `w1d=hada_w1_b, w1u=hada_w1_a, w2d=hada_w2_b, w2u=hada_w2_a`, so the
/// non-tucker product is `(w1_a @ w1_b) ⊙ (w2_a @ w2_b)` (conv folds the kernel into the factors' dim 1,
/// then `reshape(base_shape)` restores `[out,in,kH,kW]`).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_loha_delta(
    scale: f32,
    base_shape: &[i32],
    w1_a: &Array,
    w1_b: &Array,
    w2_a: &Array,
    w2_b: &Array,
    t1: Option<&Array>,
    t2: Option<&Array>,
    out_dtype: Dtype,
) -> Result<Array> {
    let (m1, m2) = match (t1, t2) {
        (Some(t1), Some(t2)) => (
            loha_rebuild_tucker(t1, w1_b, w1_a)?,
            loha_rebuild_tucker(t2, w2_b, w2_a)?,
        ),
        _ => (matmul(w1_a, w1_b)?, matmul(w2_a, w2_b)?),
    };
    let delta = multiply(&multiply(&m1, &m2)?, scalar(scale))?;
    Ok(delta.reshape(base_shape)?.as_dtype(out_dtype)?)
}

/// LoHa tucker rebuild: `einsum("i j …, j r, i p -> p r …", t, w_b, w_a)` (lycoris `HadaWeightTucker`).
/// `t` is `[dim, dim, kH, kW]`, `w_b` is `[dim, in]`, `w_a` is `[dim, out]` → `[out, in, kH, kW]`.
fn loha_rebuild_tucker(t: &Array, w_b: &Array, w_a: &Array) -> Result<Array> {
    if t.shape().len() != 4 {
        return Err(format!(
            "LoHa tucker: expected a 4-D hada_t [dim,dim,kH,kW], got shape {:?}",
            t.shape()
        )
        .into());
    }
    Ok(einsum("ijhw,jr,ip->prhw", [t, w_b, w_a])?)
}

/// Fuse a **conv-layer** LoRA pair into a single conv-weight delta, returned in the trained-file
/// NCHW `[out, in, kH, kW]` layout (sc-2919). Conv LoRAs decompose a conv into a spatial `down`
/// (`lora_down`, `[rank, in, kH, kW]`) followed by a 1×1 `up` (`lora_up`, `[out, rank, 1, 1]`); the
/// fused weight is the composition of those two convs:
///   `δ[o, i, y, x] = Σ_r up[o, r] · down[r, i, y, x]`,
/// which is exactly `up[out, rank] @ down[rank, in·kH·kW]` reshaped back to `[out, in, kH, kW]` —
/// bit-identical to PEFT/diffusers' `Conv2d` LoRA fusion (`F.conv2d(down.permute(1,0,2,3), up)`),
/// and uniform across 1×1 and k×k kernels. Then scaled by `(alpha/rank)·scale`.
///
/// Precision mirrors the SDXL Linear `lora_delta`: the `up @ down` matmul runs in **f32** (correct,
/// and avoids the former NAX 16-bit-GEMM bug — now fixed at the toolchain level, sc-2772) and is
/// rounded back through the factors' source dtype (f16 for community/accel LoRAs), so the result is
/// the same value an f16 reference fusion would produce, returned as f32 for the caller to cast to
/// the conv weight's dtype on merge. The merge itself is the chaos-safe `W += δ` (the SDXL ancestral
/// sampler needs a merged weight, not a forward-time residual — cf. [`AdaptableLinear::merge_dense_delta`]).
pub fn conv_lora_delta(
    down: &Array,
    up: &Array,
    alpha: f32,
    rank: f32,
    scale: f32,
) -> Result<Array> {
    let src = up.dtype(); // f16 for kohya/community LoRAs; f32 makes the round-trip a no-op.
    let ds = down.shape(); // [rank, in, kH, kW]
    let us = up.shape(); // [out, rank, 1, 1]
                         // A malformed conv LoRA with 2-D factors would panic on the `ds[2]`/`ds[3]` slice below; surface a
                         // typed error up front instead, matching the tucker reconstructors' style (F-006).
    if ds.len() != 4 || us.len() != 4 {
        return Err(format!(
            "conv LoRA: expected 4-D factors (down [rank,in,kH,kW], up [out,rank,1,1]), got down \
             {ds:?} up {us:?}"
        )
        .into());
    }
    if us[1] != ds[0] {
        return Err(format!(
            "conv LoRA: rank mismatch between factors — down[0]={} but up[1]={} (down {ds:?} up {us:?})",
            ds[0], us[1]
        )
        .into());
    }
    let (r, cin, kh, kw) = (ds[0], ds[1], ds[2], ds[3]);
    let out = us[0];
    let down2 = down.reshape(&[r, cin * kh * kw])?; // [rank, in·kH·kW]
    let up2 = up.reshape(&[out, r])?; // [out, rank]
    let ba = matmul(
        &up2.as_dtype(Dtype::Float32)?,
        &down2.as_dtype(Dtype::Float32)?,
    )?;
    let ba = ba.as_dtype(src)?.as_dtype(Dtype::Float32)?;
    // effective_scale in f64 then f32, matching a reference's Python-float arithmetic.
    let eff = ((alpha as f64 / rank as f64) * scale as f64) as f32;
    Ok(multiply(&ba, scalar(eff))?.reshape(&[out, cin, kh, kw])?)
}

/// One adapter's contribution WITHOUT the base, so a host can sum stacked adapters over
/// a single base application.
#[derive(Clone)]
pub enum Adapter {
    /// LoRA: `residual = scale · x·A·B`.
    Lora { a: Array, b: Array, scale: f32 },
    /// LoKr: `residual = scale · x·ΔWᵀ`; `delta` stored bf16 (see [`reconstruct_lokr_delta`]).
    Lokr { delta: Array, scale: f32 },
    /// LoKr applied as a **structured, deferred Kronecker** residual (sc-10050 / epic 10043) — the
    /// full `[out, in]` delta is NEVER materialized. `scale` is baked into `factors.w2`, so the
    /// residual is `vec(w1 · reshape(x) · w2ᵀ)` (see [`LokrFactors`]) — two small matmuls in compute
    /// precision over the packed (or dense) base, matching the folded delta within tolerance. This is
    /// the packed-tier path so a LoKr works on q4/q8 at plain-LoRA memory cost.
    LokrStructured { factors: LokrFactors },
}

impl Adapter {
    /// Evaluate only this residual's adapter-owned payload arrays.
    ///
    /// Safetensors loads and the transpose/reconstruction graph layered over them are lazy. Provider
    /// loaders call this while the adapter file's immutable token is still guarded, so no LoRA/LoKr
    /// payload can first touch disk after the pin's post-check. Base Linear weights are intentionally
    /// absent from this evaluation set.
    pub fn materialize(&self) -> Result<()> {
        match self {
            Adapter::Lora { a, b, .. } => mlx_rs::transforms::eval([a, b])?,
            Adapter::Lokr { delta, .. } => mlx_rs::transforms::eval([delta])?,
            Adapter::LokrStructured { factors } => {
                mlx_rs::transforms::eval([&factors.w1, &factors.w2])?
            }
        }
        Ok(())
    }

    /// One adapter's forward-time contribution `scale · …`, replicating the fork's `LoRALinear`
    /// / `LoKrLinear` `.residual` **byte-for-byte** (sc-2718). No dtype is forced: the earlier f32
    /// upcast (sc-2602/2719) was a workaround for the NAX 16-bit dense GEMM returning garbage on the
    /// low-rank `[seq,r]·[r,out]` matmul (`K = rank ≤ 512`, `M ≥ 2`); that GEMM is now correct at the
    /// toolchain level (sc-2772 — Metal target ≥ 26.2), so the math runs in the natural promoted
    /// dtype exactly as the fork does — restoring parity (the f32 forcing was the DEVIATION):
    ///   * LoRA — `scale · (x·A)·B` with `A`/`B` kept at their loaded (file) dtype. The fork never
    ///     casts the factors, so a bf16 `x` against f32 factors (the goldens ship f32) promotes to
    ///     f32; a bf16-factor file runs bf16 (the formerly-buggy shape, now safe).
    ///   * LoKr — `scale · x·ΔWᵀ` with `ΔW` (stored bf16) cast to the **activation dtype** — bf16 on
    ///     the bf16 path — mirroring the fork's `delta.astype(x.dtype)`.
    ///
    /// The result is NOT cast back **here** — this returns the residual in its natural promoted
    /// dtype, byte-for-byte as the fork's `.residual` does. The hand-off to the host is a separate
    /// concern and lives in `AdaptableLinear`'s adapter accumulation, which narrows the residual to the
    /// host's output dtype before the add (sc-15265) so installing an adapter cannot widen the host
    /// Linear — and therefore the whole downstream chain — from bf16 to f32. An f32-activation
    /// target is unchanged either way (FLUX.2; Qwen's f32 image stream; SDXL merges
    /// instead) — the residual was f32 before and stays f32. A bf16-activation target now runs the
    /// residual in bf16 like the fork (Z-Image's latents; Qwen's bf16 text stream); validated against
    /// the fork goldens (Z-Image / Qwen LoRA+LoKr) — px>8 byte-identical to the old forced-f32 path,
    /// i.e. the dtype change is sub-threshold while restoring fork-faithfulness (sc-2718). `scale` is
    /// applied through a dtype-matched scalar so the multiply preserves the residual's dtype, matching
    /// the fork's weak Python-float `scale * …` (a strong f32 scalar would wrongly promote a bf16
    /// residual to f32; verified against MLX).
    pub fn residual(&self, x: &Array) -> Result<Array> {
        let (r, scale) = match self {
            Adapter::Lora { a, b, scale } => (matmul(&matmul(x, a)?, b)?, *scale),
            Adapter::Lokr { delta, scale } => {
                let d = delta.as_dtype(x.dtype())?;
                (matmul(x, d.t())?, *scale)
            }
            // Structured LoKr (sc-10050): the vec-trick, `scale` already baked into `factors.w2`, so
            // there is nothing left to multiply — return directly. `y = w1 · X · w2ᵀ` reshaped back,
            // with `X = reshape(x, [.., c, d])` and no `[out, in]` delta ever formed.
            Adapter::LokrStructured { factors } => return factors.residual(x),
        };
        // Dtype-matched scalar → preserves the residual's dtype (the fork's weak-float `scale * …`).
        Ok(multiply(&r, &scalar(scale).as_dtype(r.dtype())?)?)
    }

    /// `true` when this adapter contributes nothing — a `scale` of exactly zero (sc-15265).
    ///
    /// [`AdaptableLinear::forward`] skips a disabled adapter entirely rather than adding its
    /// (mathematically zero) residual, so **"install an adapter at scale 0" is byte-identical to
    /// "install no adapter at all"**: no residual is formed, nothing is added, and the host's
    /// output array is returned untouched. That is what makes scale 0 usable both as the
    /// "disabled LoRA" model a UI/worker reaches for and as an A/B control when validating an
    /// adapter. (It is also strictly cheaper: two matmuls per adapted Linear are skipped.)
    ///
    /// [`Adapter::LokrStructured`] bakes its `scale` into [`LokrFactors::w2`], so it carries the
    /// pre-bake value in [`LokrFactors::scale`] purely to answer this question (sc-15265). It
    /// deliberately is NOT exempted: a zero `scale` does multiply `w2` to exact zeros, and adding
    /// exact zeros would already be bit-exact — but only for *finite* factors. `Inf · 0.0 = NaN`,
    /// so a checkpoint carrying a non-finite `w2` entry would otherwise NaN-poison the output at
    /// exactly the setting the user chose to turn the adapter OFF. Skipping the residual outright
    /// makes that unreachable rather than merely unlikely.
    pub fn is_disabled(&self) -> bool {
        match self {
            Adapter::Lora { scale, .. } | Adapter::Lokr { scale, .. } => *scale == 0.0,
            Adapter::LokrStructured { factors } => factors.scale == 0.0,
        }
    }
}

impl LokrFactors {
    /// The deferred, allocation-free LoKr residual `scale · x·ΔWᵀ` via the Kronecker–vector identity
    /// (`scale` is baked into [`w2`](Self::w2)). For an activation `x` of shape `[.., in]` (`in = c·d`):
    /// reshape to `[.., c, d]`, compute `Y = w1 · X · w2ᵀ` (`[.., a, b]`), and flatten row-major to
    /// `[.., out]` (`out = a·b`). The two matmuls (`w1 · X` then `· w2ᵀ`) touch only the small factor
    /// shapes `[a,c]`/`[b,d]` — the full `[out, in]` delta is NEVER materialized, so this holds the
    /// same memory profile on a packed Q4/Q8 base as a plain LoRA. Factors are cast to the activation
    /// dtype (mirroring [`Adapter::Lokr`]'s `delta.astype(x.dtype)`) so a bf16 stream runs bf16.
    pub fn residual(&self, x: &Array) -> Result<Array> {
        let w1 = self.w1.as_dtype(x.dtype())?;
        let w2t = self.w2.as_dtype(x.dtype())?.t();
        let lead: Vec<i32> = x.shape()[..x.shape().len() - 1].to_vec();
        let n: i32 = lead.iter().product::<i32>().max(1);
        // Collapse every leading dim into ONE batch axis, then reshape `[N, in] → [N, c, d]`. Doing the
        // batched matmul against an explicit `[N, …]` batch (rather than broadcasting a bare 2-D `w1`
        // over `[.., c, d]`) avoids the numpy/MLX matmul ambiguity when a leading batch dim happens to
        // equal `a`/`b`. `w1`/`w2ᵀ` are prepended a length-1 batch so they broadcast cleanly over `N`.
        let xr = x.reshape(&[n, self.c, self.d])?;
        let w1b = w1.reshape(&[1, self.a, self.c])?;
        let w2tb = w2t.reshape(&[1, self.d, self.b])?;
        // Y = w1 · X · w2ᵀ  → [N, a, b].
        let y = matmul(&matmul(&w1b, &xr)?, &w2tb)?;
        // [N, a, b] → [.., out] (out = a·b), restoring the original leading dims.
        let mut ys = lead;
        ys.push(self.a * self.b);
        Ok(y.reshape(&ys)?)
    }
}

/// A linear base — dense or quantized — evaluated through a shared reference. Mirrors the
/// `forward` of mlx-rs's `nn::Linear` / `nn::QuantizedLinear` but without requiring `&mut`.
#[derive(Clone)]
pub enum LinearBase {
    Dense(Linear),
    Quantized(QuantizedLinear),
}

impl LinearBase {
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(match self {
            LinearBase::Dense(l) => {
                // Mirror MLX `nn.Linear` exactly: the biased case is a FUSED `addmm(bias, x, Wᵀ)`
                // — accumulate `x·Wᵀ`, add bias, round to the output dtype ONCE. A separate
                // `matmul` then `add` rounds the matmul, *then* rounds the bias add again — a
                // ~1.4e-3 double-rounding error per biased Linear in bf16 that compounds over a
                // deep net (sc-2779; localized in the Wan DiT, q_proj 1.4e-3 → ~4e-7 with addmm).
                // f32-INVISIBLE and therefore safe for every crate today: with f32 activations
                // (the current Z-Image/Qwen/FLUX path, even with bf16 weights) `addmm == matmul+add`
                // bit-for-bit, because nothing rounds to bf16 mid-op (verified, sc-2779). It bites
                // only once a path runs bf16 activations (the sc-2718–2721 reverts). The unbiased
                // case stays a plain `matmul`, as mlx-rs's own `Linear::forward` does.
                match l.bias.value.as_ref() {
                    Some(b) => addmm(b, x, l.weight.value.t(), 1.0, 1.0)?,
                    None => matmul(x, l.weight.value.t())?,
                }
            }
            LinearBase::Quantized(q) => {
                // Activations are fed to `quantized_matmul` AS-IS — no dtype upcast. `quantized_matmul`
                // accumulates in fp32 (mlx#963) and is correct at every activation shape/dtype, so it
                // was never the buggy op: the NAX 16-bit-GEMM bug lived in the *dense* 16-bit×16-bit
                // Metal GEMM, and that is now fixed at the toolchain level (sc-2772 — metal target ≥26.2).
                // The former bf16→f32 upcast here (sc-2719) guarded a proven non-bug and is removed:
                // feeding bf16 activations straight in matches the fork's own quantized compute dtype
                // (bf16 latents → `quantized_matmul` → bf16), so it is strictly *more* faithful, not less.
                // Weights stay Q4/Q8 throughout. (`q8_smoke.rs` exercises the bf16-activation path.)
                quantized_matmul_with_bias(
                    x,
                    &q.inner.weight.value,
                    &q.scales.value,
                    &q.biases.value,
                    q.inner.bias.value.as_ref(),
                    q.group_size,
                    q.bits,
                )?
            }
        })
    }
}

/// A linear base plus a stack of adapters, applied as `base(x) + Σ adapter.residual(x)`.
/// Quantized-safe: the base weight is never mutated.
#[derive(Clone)]
pub struct AdaptableLinear {
    base: LinearBase,
    adapters: Vec<Adapter>,
    /// `true` only for a **training** adapter stack (installed via
    /// [`set_training_adapters`](Self::set_training_adapters)), which opts OUT of both sc-15265
    /// rules in `apply_adapters` — see that method's doc for why.
    training_residual: bool,
}

impl AdaptableLinear {
    /// Build from a raw `[out, in]` weight (and optional bias) — the common path when
    /// loading dense (bf16/fp16/fp32) checkpoints via the `weights` module.
    pub fn dense(weight: Array, bias: Option<Array>) -> Self {
        Self::from_linear(Linear {
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }

    /// Wrap an existing dense `nn::Linear`.
    pub fn from_linear(linear: Linear) -> Self {
        Self {
            base: LinearBase::Dense(linear),
            adapters: Vec::new(),
            training_residual: false,
        }
    }

    /// Wrap an existing `nn::QuantizedLinear` (sc-2342 quantized weights).
    pub fn from_quantized(q: QuantizedLinear) -> Self {
        Self {
            base: LinearBase::Quantized(q),
            adapters: Vec::new(),
            training_residual: false,
        }
    }

    /// Build a quantized base from **already-packed** parts read off disk — a *pre-quantized*
    /// checkpoint (group-wise affine `weight` u32 codes + `scales` + `biases`, optional dense
    /// `bias`). The consume-side counterpart to [`quantize`](Self::quantize): no re-quantization
    /// happens, the on-disk scales are used as-is. Mirrors the fork's `loading.py` — `nn.quantize`
    /// stubs then `load_weights` of the packed tensors — but as a direct construction. `group_size`
    /// and `bits` come from the checkpoint's manifest (e.g. Wan's `config.json` `quantization` block).
    pub fn from_quantized_parts(
        weight: Array,
        scales: Array,
        biases: Array,
        bias: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self::from_quantized(QuantizedLinear {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner: Linear {
                weight: Param::new(weight),
                bias: Param::new(bias),
            },
        })
    }

    /// Stack a new adapter (LoRA or LoKr) on top of any already installed.
    pub fn push(&mut self, adapter: Adapter) {
        self.adapters.push(adapter);
    }

    /// Replace the entire adapter stack under the normal (inference) dtype rules — see
    /// `apply_adapters`. An empty `Vec` clears the stack back to the bare
    /// frozen base, which is what every non-training caller uses this for.
    pub fn set_adapters(&mut self, adapters: Vec<Adapter>) {
        self.adapters = adapters;
        self.training_residual = false;
    }

    /// Replace the entire adapter stack as a **training** stack. The forward-time counterpart to
    /// [`push`](Self::push) used by training (sc-3042/3039): each optimizer step produces new
    /// trainable LoRA factor arrays, so the trainer re-injects a single fresh `Adapter::Lora` per
    /// target every step rather than accumulating residuals. Setting the SAME
    /// `(transpose, alpha/rank fold, scale)` an inference reload applies
    /// (`adapters::loader::install_lora_groups`) makes the trained adapter round-trip bit-for-bit.
    ///
    /// The stack is flagged so `apply_adapters` leaves the trainer's
    /// numerics **exactly as they shipped** — no scale-0 skip, no narrowing cast (sc-15265). Both
    /// the traced `loss_fn` install (`train::lora::install_training_lora_as` /
    /// `install_training_lokr`) and every family's gradient-checkpoint recompute go through this
    /// method, so the train/recompute consistency invariant is preserved by construction.
    pub fn set_training_adapters(&mut self, adapters: Vec<Adapter>) {
        self.adapters = adapters;
        self.training_residual = true;
    }

    pub fn adapters(&self) -> &[Adapter] {
        &self.adapters
    }

    /// Evaluate every adapter residual attached to this Linear without touching its base weight.
    pub fn materialize_adapters(&self) -> Result<()> {
        for adapter in &self.adapters {
            adapter.materialize()?;
        }
        Ok(())
    }

    /// Evaluate this linear's retained base and adapter arrays.  Each call is deliberately scoped to
    /// one projection: load-time quantizers can invoke it while walking a model so MLX never has to
    /// materialize the complete dense source model before retaining the packed Q4/Q8 result.
    pub fn materialize_weights(&self) -> Result<()> {
        match &self.base {
            LinearBase::Dense(linear) => {
                mlx_rs::transforms::eval(
                    std::iter::once(&linear.weight.value).chain(linear.bias.value.iter()),
                )?;
            }
            LinearBase::Quantized(quantized) => {
                mlx_rs::transforms::eval(
                    [
                        &quantized.inner.weight.value,
                        &quantized.scales.value,
                        &quantized.biases.value,
                    ]
                    .into_iter()
                    .chain(quantized.inner.bias.value.iter()),
                )?;
            }
        }
        self.materialize_adapters()
    }

    /// Merge a precomputed `[out, in]` delta into the dense base weight (`W += δ`) — the in-place
    /// LoRA/LoKr *merge*, distinct from the forward-time [`Adapter::residual`] stack. The merge
    /// reproduces a reference's merged-weight forward (`(W+δ)·x`) bit-for-bit, where a residual
    /// (`W·x + δ·x`) differs by ~1 ULP; on a chaos-sensitive sampler (SDXL's ancestral) that 1-ULP
    /// cascades to a visible whole-image divergence, so the SDXL provider merges (matching the
    /// vendored `lora.py` `module.weight += delta`) rather than stacking residuals. `delta` is cast
    /// to the base weight's dtype before the add. Errors on a quantized base — a LoRA must be merged
    /// into the dense (e.g. f32) weight BEFORE quantization (the fork merges pre-quantize too).
    pub fn merge_dense_delta(&mut self, delta: &Array) -> Result<()> {
        match &mut self.base {
            LinearBase::Dense(l) => {
                let merged = add(&l.weight.value, &delta.as_dtype(l.weight.value.dtype())?)?;
                l.weight = Param::new(merged);
                Ok(())
            }
            LinearBase::Quantized(_) => Err(
                "merge_dense_delta: base is quantized; a LoRA must be merged before quantization"
                    .into(),
            ),
        }
    }

    /// Merge a precomputed bias delta into the base **bias** (`b += δ`) — the bias-channel
    /// analog of [`merge_dense_delta`](Self::merge_dense_delta), for a ComfyUI/lightx2v `.diff_b`
    /// diff-patch (a full bias delta a low-rank adapter cannot express). `delta` is cast to the base
    /// bias's dtype before the add. Errors only on a base with **no** bias — the caller (the
    /// diff-patch fold) treats that as a surfaced skip, never inventing a bias the reference module
    /// doesn't have.
    ///
    /// **Tier-independent in *coverage*, unlike [`merge_dense_delta`](Self::merge_dense_delta)
    /// (sc-15326).** A quantized base packs only its *weight*: `QuantizedLinear` keeps the bias as a
    /// plain dense vector and its forward is `quantized_matmul(x, wq, …) + b`, so `b += δ_b` is exactly
    /// as correct over a Q4/Q8 base as over a bf16 one. That is what lets Krea Realtime apply a
    /// lightx2v step-distill LoRA's 407 `.diff_b` bias deltas on **every** tier instead of only on the
    /// dense one. Quant tier is a *creative* choice in this product, so an adapter result that varied
    /// with it would be a real defect.
    ///
    /// *Coverage*, not bit-for-bit: every `.diff_b` lands on every tier, but the fold dtype can still
    /// differ, because [`quantize`](Self::quantize) casts the bias to bf16 along with the weight
    /// (PARITY-BF16, sc-2609). A base that was f32 on disk therefore folds in f32 when dense and in
    /// bf16 when packed. Immaterial for the bf16-native Krea DiT, where both sides are bf16 anyway.
    pub fn merge_bias_delta(&mut self, delta: &Array) -> Result<()> {
        let bias = match &mut self.base {
            LinearBase::Dense(l) => &mut l.bias,
            LinearBase::Quantized(q) => &mut q.inner.bias,
        };
        match bias.value.as_ref() {
            Some(b) => {
                let merged = add(b, &delta.as_dtype(b.dtype())?)?;
                *bias = Param::new(Some(merged));
                Ok(())
            }
            None => Err(
                "merge_bias_delta: base linear has no bias; a `.diff_b` cannot fold into a bias-free module"
                    .into(),
            ),
        }
    }

    /// The base's dense bias, packed or not — the tier-independent half of a diff-patch target.
    /// `Some` for both a dense and a quantized base (a `QuantizedLinear` never packs its bias), so a
    /// `.diff_b` shape check does not have to first ask whether the weight is packed (sc-15326).
    pub fn bias(&self) -> Option<&Array> {
        match &self.base {
            LinearBase::Dense(l) => l.bias.value.as_ref(),
            LinearBase::Quantized(q) => q.inner.bias.value.as_ref(),
        }
    }

    /// Cast the dense base weight (and bias) to `dtype` in place — the training-time compute-dtype
    /// switch (sc-4887: bf16 mixed-precision training over an f32-on-disk checkpoint). Quantized
    /// bases are left untouched (their compute dtype is fixed by the packed format). Destructive for
    /// a narrowing cast (f32→bf16 drops mantissa bits): reload the model to get the f32 weights back.
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        if let LinearBase::Dense(l) = &mut self.base {
            if l.weight.value.dtype() != dtype {
                l.weight = Param::new(l.weight.value.as_dtype(dtype)?);
            }
            if let Some(b) = l.bias.value.as_ref() {
                if b.dtype() != dtype {
                    l.bias = Param::new(Some(b.as_dtype(dtype)?));
                }
            }
        }
        Ok(())
    }

    /// The dense base weight's dtype, or `None` for a quantized base. Lets a forward stay
    /// dtype-following (cast its inputs to the weight's compute dtype) without assuming f32.
    pub fn weight_dtype(&self) -> Option<Dtype> {
        match &self.base {
            LinearBase::Dense(l) => Some(l.weight.value.dtype()),
            LinearBase::Quantized(_) => None,
        }
    }

    /// `true` once the base has been quantized (Q4/Q8).
    pub fn is_quantized(&self) -> bool {
        matches!(self.base, LinearBase::Quantized(_))
    }

    /// Diagnostic accessor: the dense base's `(weight, bias)`, or `None` if already quantized.
    /// Used by the sc-2604 diagnostic to inspect the loaded weight dtype before quantization.
    pub fn dense_weight(&self) -> Option<(&Array, Option<&Array>)> {
        match &self.base {
            LinearBase::Dense(l) => Some((&l.weight.value, l.bias.value.as_ref())),
            LinearBase::Quantized(_) => None,
        }
    }

    /// Diagnostic accessor: the quantized base's `(packed_weight, scales, biases, bias, group_size,
    /// bits)`, or `None` if the base is still dense. Used by the sc-2604 Q8 root-cause diagnostic to
    /// byte-compare the *loaded* model's quantization against the fork's `mx.quantize` (the
    /// `qmm_smallk` probe only exercised the free `quantize` op, not `try_from_linear`).
    #[allow(clippy::type_complexity)]
    pub fn quantized_params(&self) -> Option<(&Array, &Array, &Array, Option<&Array>, i32, i32)> {
        match &self.base {
            LinearBase::Quantized(q) => Some((
                &q.inner.weight.value,
                &q.scales.value,
                &q.biases.value,
                q.inner.bias.value.as_ref(),
                q.group_size,
                q.bits,
            )),
            LinearBase::Dense(_) => None,
        }
    }

    /// The base weight's logical `[out, in]` shape — what a LoKr delta must reshape to.
    /// For a quantized base the packed weight is opaque, so recover it from the scales grid
    /// (`[out, in/group_size]`) times the group size.
    pub fn base_shape(&self) -> Vec<i32> {
        match &self.base {
            LinearBase::Dense(l) => l.weight.value.shape().to_vec(),
            LinearBase::Quantized(q) => {
                // Recover `in = scales_cols · group_size`. Exact only when `in % group_size == 0`,
                // which always holds for the group-quantized linears here (in is a multiple of the
                // group size by construction) (F-089).
                let s = q.scales.value.shape();
                vec![s[0], s[1] * q.group_size]
            }
        }
    }

    /// Quantize the dense base in place to Q4/Q8 (`group_size` defaults to 64), the mlx-rs
    /// equivalent of `nn.quantize` over this Linear. No-op if already quantized. Adapters are
    /// forward-time residuals over the (now quantized) base, so they are unaffected — this is
    /// why the base is never fused: fusing would force re-quantization on every adapter swap.
    pub fn quantize(&mut self, bits: i32, group_size: Option<i32>) -> Result<()> {
        if let LinearBase::Dense(l) = &self.base {
            // PARITY-BF16 (sc-2609): downcast for fork parity. f32 quantization (f32 group scales)
            // is *more* accurate; we cast to bf16 only to byte-match the fork's golden. Flip to f32
            // for quality once parity is no longer the goal — f32 is safe (the qmm path never hits
            // the bf16-GEMM bug). Rationale below.
            //
            // The fork (mflux) loads every weight at bf16 — its compute dtype — and quantizes THAT.
            // Some checkpoints (e.g. Z-Image-Turbo's transformer) ship f32 on disk; quantizing the
            // as-loaded f32 weight yields group `scales` that differ from the fork's bf16 scales by
            // ~0.13% (the integer `wq` codes and `biases` survive the perturbation, the scales do
            // not), which compounds into the base-model Q8/Q4 e2e residual (sc-2604). Cast weight +
            // bias to bf16 first so the packing is byte-identical to the fork. No-op when already
            // bf16 (e.g. Qwen, whose checkpoint is bf16-native — which is why its Q8 already matched).
            let weight = l.weight.value.as_dtype(Dtype::Bfloat16)?;
            let bias = l
                .bias
                .value
                .as_ref()
                .map(|b| b.as_dtype(Dtype::Bfloat16))
                .transpose()?;
            let linear = Linear {
                weight: Param::new(weight),
                bias: Param::new(bias),
            };
            let q = QuantizedLinear::try_from_linear(
                linear,
                group_size.unwrap_or(crate::quant::DEFAULT_GROUP_SIZE),
                bits,
            )?;
            self.base = LinearBase::Quantized(q);
        }
        Ok(())
    }

    /// Accumulate the adapter stack over an already-computed base output, in the **host's** dtype
    /// (sc-15265). Two rules, both about the host staying the host:
    ///
    /// 1. A [disabled](Adapter::is_disabled) adapter (`scale == 0`) is skipped outright, so `out`
    ///    is returned as the base produced it — installing an adapter and setting its scale to 0
    ///    is byte-identical to never installing one.
    /// 2. A live adapter's residual is cast to `out`'s dtype **before** the add.
    ///
    /// Rule 2 is the consequential one and is not confined to scale 0. Adapter factors carry their
    /// on-disk dtype and [`Adapter::residual`] deliberately does not force one (sc-2718,
    /// fork-faithful); the goldens ship **f32** factors, so on a **bf16** host `add(bf16, f32)`
    /// promoted the Linear's output to f32 — and from there the whole downstream chain, since
    /// every subsequent op inherits the widened activation. That is a global precision/allocation
    /// change triggered by merely installing a LoRA, measured e2e at ≈1.9e-4 (dense) / ≈2.9e-4
    /// (Q4) versus the unadapted model. Training already had to work around it per-call-site
    /// (`install_training_lora_as`'s compute-dtype cast, sc-4887, whose doc names exactly this
    /// "silently re-promotes the whole chain to f32"); this fixes it once at the shared seam.
    ///
    /// The residual itself is still computed in its natural promoted dtype — only the *hand-off*
    /// to the host is narrowed — so an f32-activation host (FLUX.2, Qwen's image stream) is
    /// completely unaffected: `out` is already f32 and the cast is a no-op.
    ///
    /// **What rule 2 costs, stated honestly.** It is not free and it is not an accuracy *gain* — it
    /// moves the adapted Linear AWAY from a fully-f32 reference, not toward it. Two independent
    /// seam-level measurements (bf16 host, f32 rank-4 factors, relative L2 vs an all-f32 reference):
    ///
    /// | fixture   | unadapted bf16 base floor | err before | err after |
    /// |-----------|---------------------------|------------|-----------|
    /// | 64-wide   | 6.3014e-4                 | 6.3066e-4  | 7.4336e-4 |
    /// | 1024-wide | 1.95376e-3                | 1.95376e-3 | 2.0238e-3 |
    ///
    /// The load-bearing observation is that `err(before)` equals the unadapted base's own bf16 floor
    /// to five/six digits: the promoted residual was essentially exact and contributed ~nothing on
    /// top of that floor. Narrowing it costs a further `≈4.7e-4` / `≈5.3e-4` relative. Expressed as a
    /// *fraction of the adapter's own effect* at that Linear the number is NOT stable — ~19% at the
    /// first fixture, ~92% at the second — because it depends entirely on how large the residual is
    /// relative to the host's bf16 ulp; quote it and you will mislead yourself. The claim that IS
    /// stable is the absolute bound. The narrowing path rounds **twice** — once casting the residual
    /// to the host dtype, and once in the host-dtype `add` — each at most half an ulp, so the
    /// constructive bound is **at most about one ulp of the host dtype**, not half. (The measured
    /// extra `≈4.7e-4` / `≈5.3e-4` relative above is consistent with ~1 ulp: bf16's relative ulp is
    /// ≈3.9e-3.) That is still **a bounded accuracy loss of the same order as the host's own bf16
    /// rounding, with a non-systematic end-to-end sign** — the qualitative conclusion and every
    /// measured number are unaffected by the correction. The krea e2e tier sweep
    /// bears the second half out: −0.17% (bf16), −1.1% (Q8), **+**1.5% (Q4) — non-monotonic,
    /// consistent with chaotic propagation of rounding rather than a directional bias.
    ///
    /// **User-visible consequence at inference.** Rule 2's dead zone is not confined to training:
    /// on a bf16 host a sufficiently low user LoRA strength is now quantized to the host dtype
    /// rather than accumulated in f32, so it can round away *entirely* where pre-fix it had a small
    /// but visible f32 effect. The measured threshold is roughly half a bf16 ulp of the host output
    /// — i.e. a residual/output ratio of about **2e-3** or less contributes nothing. This is the
    /// accepted diffusers convention and is not a defect, but it is the answer to "my LoRA at very
    /// low strength does nothing now": raise the strength, or run an f32-activation host.
    ///
    /// This is the diffusers convention (LoRA cast to the model dtype); it diverges from the frozen
    /// fork, which adds the promoted residual. **No image-level A/B on a shipped LoRA was run**, and
    /// the Z-Image / Qwen LoRA+LoKr **fork-parity goldens named in [`Adapter::residual`] were not
    /// re-run** against this narrowing — they need a licensed host. Both are open follow-ups, not
    /// claims. What is bought is the invariant above: installing an adapter no longer silently
    /// re-dtypes the model, and "installed at scale 0" is genuinely nothing.
    ///
    /// `candle-gen`'s `quant::adapt` carries the same defect at its own seam and is NOT fixed here
    /// (tracked as sc-15444).
    fn apply_adapters(&self, x: &Array, mut out: Array) -> Result<Array> {
        for adapter in &self.adapters {
            // sc-15265 rule 0 — the TRAINING exemption. A training stack
            // ([`set_training_adapters`]) opts out of both rules and runs the pre-sc-15265 add
            // verbatim, so this change is inference-only and the shipped trainers are
            // bit-identical. Reasons, in order:
            //   * Neither rule is *for* training. Rule 1 exists so "installed at scale 0" is a
            //     no-op; a trainer always installs at `scale = 1` and has no disabled case.
            //   * Rule 2 would change shipped trainer numerics, and this is an inference-scoped
            //     fix. Wan installs its **f32 master** factors unchanged (`install_training_lora`
            //     passes `dtype = None`; see
            //     `mlx_gen_wan::transformer::WanTransformer::forward_train_checkpointed`) over a
            //     bf16 block stream. `b` initializes to zeros and the residual grows *from* zero,
            //     so for an initial stretch it sits below the bf16 ulp of the base and narrowing
            //     `base + r` to bf16 would round the adapter contribution away entirely. That
            //     FORWARD dead zone is real and measured at the seam: at `|b|` up to 1e-4 the
            //     residual peaks at 3.3e-7 and the narrowed output is bit-identical to the
            //     unadapted base.
            //     It does NOT, however, stall the optimizer — that would be an overclaim. The
            //     `astype`/`add` VJP is straight-through, so gradient still reaches the LoRA
            //     parameters through the narrowing path: `grad` of a squared-sum loss at `b == 0`
            //     measures max|g| = 4.42e-2, and `b` grows out of the dead zone. The justification
            //     for the exemption is therefore the narrower and sufficient one — it keeps the
            //     shipped trainers **bit-identical** while this change stays inference-only, and
            //     avoids a forward dead zone whose end-to-end training effect was never measured.
            //     The wider accumulation is the master-weights pattern working as intended.
            //   * The alternative — casting the factors to bf16 at the install site — moves the
            //     low-rank matmul itself into bf16, which is a LARGER departure from shipped
            //     trainer numerics than doing nothing, so it cannot be justified as conservative.
            // No before/after Wan training run was measured (a 14B run is not a CI-scale
            // experiment); this exemption is what makes that measurement unnecessary.
            if self.training_residual {
                out = add(&out, &adapter.residual(x)?)?;
                continue;
            }
            if adapter.is_disabled() {
                continue;
            }
            let residual = adapter.residual(x)?;
            let residual = if residual.dtype() == out.dtype() {
                residual
            } else {
                residual.as_dtype(out.dtype())?
            };
            out = add(&out, &residual)?;
        }
        Ok(out)
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let out = self.base.forward(x)?;
        self.apply_adapters(x, out)
    }

    /// Evaluate with dense base weights widened to the activation dtype for this operation.
    /// Packed bases already compute from the activation dtype and are left packed. This supports
    /// bf16-resident/f32-compute text encoders without permanently widening their weight store.
    pub fn forward_upcast(&self, x: &Array) -> Result<Array> {
        let out = match &self.base {
            LinearBase::Dense(l) => {
                let weight = l.weight.value.as_dtype(x.dtype())?;
                match l.bias.value.as_ref() {
                    Some(bias) => addmm(&bias.as_dtype(x.dtype())?, x, weight.t(), 1.0, 1.0)?,
                    None => matmul(x, weight.t())?,
                }
            }
            LinearBase::Quantized(_) => self.base.forward(x)?,
        };
        self.apply_adapters(x, out)
    }
}

/// A dense Conv2d weight (mlx NHWC `[out, kH, kW, in]`) plus its optional bias, that can have a
/// conv-layer LoRA delta merged into it (sc-2919). Convs in this codebase are **merge-only**: they
/// are never quantized and never carry a forward-time residual, so unlike [`AdaptableLinear`] there
/// is no adapter stack or quantized variant — just the mergeable weight and the accessors a forward
/// pass needs. The merge takes a delta in the trained-file NCHW layout and folds it in chaos-safely
/// (`W += δ`), the conv analog of [`AdaptableLinear::merge_dense_delta`].
#[derive(Clone)]
pub struct AdaptableConv2d {
    /// NHWC `[out, kH, kW, in]` — the layout `mlx_gen::nn::conv2d` expects.
    weight: Array,
    bias: Option<Array>,
}

impl AdaptableConv2d {
    /// Wrap an already-NHWC conv weight (`[out, kH, kW, in]`) and optional bias.
    pub fn new(weight_nhwc: Array, bias: Option<Array>) -> Self {
        Self {
            weight: weight_nhwc,
            bias,
        }
    }

    /// The NHWC `[out, kH, kW, in]` weight, to feed `mlx_gen::nn::conv2d`.
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// The optional conv bias.
    pub fn bias(&self) -> Option<&Array> {
        self.bias.as_ref()
    }

    /// Merge a conv LoRA `delta` — given in the **trained-file NCHW** `[out, in, kH, kW]` layout (what
    /// [`conv_lora_delta`] returns) — into the stored NHWC weight: transpose NCHW→NHWC, cast to the
    /// weight's dtype, and add (`W += δ`). Reproduces a reference's merged-weight conv forward
    /// bit-for-bit (a residual would differ by ~1 ULP and cascade on the chaos sampler — see
    /// [`AdaptableLinear::merge_dense_delta`]). A zero delta is a bit-exact no-op (`W + 0 == W`).
    pub fn merge_conv_delta(&mut self, delta_nchw: &Array) -> Result<()> {
        // [out, in, kH, kW] → [out, kH, kW, in] to match the stored NHWC weight.
        let delta_nhwc = delta_nchw.transpose_axes(&[0, 2, 3, 1])?;
        self.weight = add(&self.weight, &delta_nhwc.as_dtype(self.weight.dtype())?)?;
        Ok(())
    }

    /// Cast the conv weight (and bias) to `dtype` in place — the conv analog of
    /// [`AdaptableLinear::cast_weights`], for bf16 mixed-precision training (sc-4878/sc-4941). Convs
    /// are never quantized, so this always applies. Destructive for a narrowing cast (reload to get
    /// the f32 weights back).
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        if self.weight.dtype() != dtype {
            self.weight = self.weight.as_dtype(dtype)?;
        }
        if let Some(b) = &self.bias {
            if b.dtype() != dtype {
                self.bias = Some(b.as_dtype(dtype)?);
            }
        }
        Ok(())
    }

    /// The conv weight's dtype — lets a forward stay dtype-following without assuming f32.
    pub fn weight_dtype(&self) -> Dtype {
        self.weight.dtype()
    }

    /// Evaluate this dense convolution without evaluating neighboring model weights.
    pub fn materialize_weights(&self) -> Result<()> {
        mlx_rs::transforms::eval(std::iter::once(&self.weight).chain(self.bias.iter()))?;
        Ok(())
    }
}

/// A read-only snapshot of one adaptable projection — **everything the probe half of the
/// adapter-resolution surface can answer**, carried by value.
///
/// This type is the whole point of [`AdaptableHost::adaptable_facts`] (SC-18319). Before it existed,
/// the only way to ask "is there a projection here, how big is it, is it packed, does it already
/// carry an adapter?" was to take a `&mut AdaptableLinear` through
/// [`adaptable_mut`](AdaptableHost::adaptable_mut) — and once a family's `to_q`/`to_k`/`to_v` live
/// behind a [`FusedQkvProjection`](crate::qkv::FusedQkvProjection), *handing out that `&mut` is
/// exactly what unfuses the block*. A mere scan (the LoRA/LyCORIS/BFL installers' pass 1, a
/// `.diff`-patch existence check, a trainer's target-shape read) would then dismantle the fusion it
/// just built, over every block, for nothing.
///
/// A probe therefore gets facts, never a handle: **mutation is not expressible through this type**,
/// so a probe cannot install, clear, merge, quantize or unfuse anything, and a fused host is free to
/// answer from its packed representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearFacts {
    /// The logical `[out, in]` base shape — [`AdaptableLinear::base_shape`].
    pub base_shape: Vec<i32>,
    /// Whether the base is group-quantized (packed) rather than dense.
    pub is_quantized: bool,
    /// Whether **any** installed adapter is live (`scale != 0`). A disabled adapter reads `false`,
    /// matching `apply_adapters` rule 1 and [`FusedQkvProjection`](crate::qkv::FusedQkvProjection)'s
    /// pack predicate, which both treat "installed at scale 0" as "never installed".
    pub has_live_adapters: bool,
    /// How many adapters are on the stack **including disabled ones** — the "is there anything here
    /// at all to read" question, which is what a capture/replay walk needs (anima's block stream) and
    /// which [`has_live_adapters`](Self::has_live_adapters) deliberately does not answer.
    pub adapter_count: usize,
    /// The dense bias's shape, when the projection carries one.
    pub bias_shape: Option<Vec<i32>>,
}

impl LinearFacts {
    /// Snapshot a projection. The `&` receiver is the point: facts are readable without the `&mut`
    /// that would unfuse a packed QKV triple.
    pub fn of(lin: &AdaptableLinear) -> Self {
        Self {
            base_shape: lin.base_shape(),
            is_quantized: lin.is_quantized(),
            has_live_adapters: lin.adapters().iter().any(|a| !a.is_disabled()),
            adapter_count: lin.adapters().len(),
            bias_shape: lin.bias().map(|b| b.shape().to_vec()),
        }
    }
}

/// A module tree that can resolve a dotted parameter path (split into segments) to the
/// [`AdaptableLinear`] living there, so an adapter can be installed onto it. This is the
/// hand-written form of the macro the full adapter framework (sc-2343) will generate.
///
/// # Probe vs mutate (SC-18319)
///
/// The surface is deliberately split in two, and a caller must pick the half that matches its
/// intent:
///
/// * [`adaptable_facts`](Self::adaptable_facts) — **the probe.** "Is there a projection at this
///   path, and what is it like?" Returns [`LinearFacts`] by value.
/// * [`adaptable_mut`](Self::adaptable_mut) — **the mutation.** "Give me the projection so I can
///   change it." Returns `&mut AdaptableLinear`.
///
/// Only the mutation half may unfuse a [`FusedQkvProjection`](crate::qkv::FusedQkvProjection).
/// **A host holding one MUST override `adaptable_facts`** for the paths that resolve into it, and
/// answer from the packed representation
/// ([`FusedQkvProjection::part_facts`](crate::qkv::FusedQkvProjection::part_facts)); the default
/// below routes through `adaptable_mut`, which unfuses. Because the routing is a hand-written tree,
/// every intermediate node on the way down to a fused leaf must forward `adaptable_facts` too — the
/// default's `adaptable_mut` delegation would otherwise take over at the first node that forgot.
/// Each adopting family pins this with a `probe_does_not_unfuse` test.
pub trait AdaptableHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear>;

    /// **The probe half** — resolve a dotted path to a read-only [`LinearFacts`] snapshot, without
    /// taking the `&mut AdaptableLinear` that unfuses a packed QKV triple. See the trait doc.
    ///
    /// The receiver is `&mut self` rather than `&self` only because the routing tree is: a host with
    /// lazily-materialized blocks (anima's block stream) resolves a path by touching that tree. The
    /// *return type* is what makes this half non-mutating — no handle escapes, so no probe can
    /// install, merge, clear, quantize or unfuse anything.
    ///
    /// The default answers through [`adaptable_mut`](Self::adaptable_mut), which is correct for
    /// every host whose projections are plain [`AdaptableLinear`]s and wrong for one holding a
    /// [`FusedQkvProjection`](crate::qkv::FusedQkvProjection) — see the trait doc.
    fn adaptable_facts(&mut self, path: &[&str]) -> Option<LinearFacts> {
        self.adaptable_mut(path).map(|lin| LinearFacts::of(lin))
    }

    /// The dtype a LyCORIS (LoKr / LoHa) install builds its residual in — the reconstructed
    /// `[out, in]` delta and the structured Kronecker factors. The default is bf16 (PARITY-BF16,
    /// sc-2609), which every fork-parity family relies on; a host computing in f32 (e.g. Iris under
    /// `Precision::Fp32`) overrides it so the adapter is not rounded below its own compute.
    fn lycoris_dtype(&self) -> Dtype {
        Dtype::Bfloat16
    }

    /// Resolve a dotted path to the [`AdaptableConv2d`] living there, for conv-layer LoRA merging
    /// (sc-2919) — the conv analog of [`adaptable_mut`](Self::adaptable_mut). The default is empty:
    /// only the SDXL U-Net (the one conv-bearing adapter host) overrides it; DiT/MMDiT families
    /// (Z-Image, Qwen, FLUX) have no conv adapter targets (their only convs live in the un-adapted
    /// VAE / patch-embed), so a conv-shaped key applied to them surfaces as skipped, never merged.
    fn adaptable_conv_mut(&mut self, _path: &[&str]) -> Option<&mut AdaptableConv2d> {
        None
    }

    /// Enumerate every adapter target reachable through the kohya `lora_unet_` convention, as
    /// dotted paths in the trained-file (diffusers) naming that [`adaptable_mut`](Self::adaptable_mut)
    /// accepts. Used to build the `flattened → dotted` lookup that disambiguates kohya keys (whose
    /// `.`→`_` flattening cannot be re-split blindly — module names like `to_out.0` / `feed_forward.w1`
    /// already contain underscores). Mirrors the fork's explicit per-target `lora_unet_…` patterns
    /// (sc-2618): block-indexed layer targets only — the families' fork mappings carry no `lora_unet_`
    /// pattern for global targets, which stay reachable via the diffusers/peft dotted form.
    ///
    /// Every returned path MUST resolve via [`adaptable_mut`](Self::adaptable_mut) and the set MUST be
    /// collision-free once flattened (both guarded by tests). The default is empty — a host that does
    /// not override it has no kohya support and a kohya file applied to it surfaces every key as
    /// unmatched (loud), never silently dropped.
    fn adaptable_paths(&self) -> Vec<String> {
        Vec::new()
    }

    /// Enumerate the host's **BFL / ComfyUI** fused→split adapter targets (sc-2743), the orthogonal
    /// axis to the kohya `lora_unet_` flattening of [`adaptable_paths`](Self::adaptable_paths). A
    /// [`BflTarget`](loader::BflTarget) maps one source key spelling (in any of the BFL prefix
    /// conventions — `lora_unet_…`, `diffusion_model.…`, `base_model.model.…`) to a diffusers module
    /// path, optionally row-slicing the up/down factor so a *fused* source linear (BFL `…img_attn.qkv`,
    /// `…linear1`) fans out into the model's *split* targets (`attn.to_q/to_k/to_v`, …). Mirrors the
    /// fork's `Flux2LoRAMapping._get_bfl_*` + the `base_model.model.` global renames.
    ///
    /// The default is empty — only FLUX.2/FLUX.1 expose a BFL surface (Z-Image/Qwen/SDXL have none),
    /// so a BFL file applied to a host without one surfaces every key as unmatched (loud), never
    /// silently dropped. The per-target slices MUST be byte-faithful to `LoraTransforms` (guarded by
    /// tests).
    fn bfl_targets(&self) -> Vec<loader::BflTarget> {
        Vec::new()
    }

    /// Resolve a dotted path to a **non-Linear dense parameter** a ComfyUI/lightx2v diff-patch delta
    /// can fold into — the norm-layer analog of [`adaptable_mut`](Self::adaptable_mut) (sc-15326).
    ///
    /// A step-distill / lightning LoRA for a Wan-family backbone patches its **norms** as well as its
    /// Linears: the lightx2v `Wan2.1-T2V-14B` cfg-step-distill file carries 200 `.diff` weight deltas
    /// on `self_attn.norm_q`/`norm_k`, `cross_attn.norm_q`/`norm_k` and `norm3`, plus 40 `norm3.diff_b`
    /// bias deltas. Those are RMSNorm/LayerNorm parameters, not `AdaptableLinear`s, so they are
    /// unreachable through the Linear surface at any width — and before this existed they were dropped
    /// without a word.
    ///
    /// Every parameter reachable here MUST be dense on **every quant tier** (norm weights and biases
    /// are: the Wan `_quantize_predicate` packs only the per-block attention/FFN Linears), which is
    /// what makes a fold through this surface tier-independent — the property that distinguishes it
    /// from a `.diff` weight fold into a Linear, which cannot happen over a packed base at all.
    ///
    /// The default is `None` for every host: a family with no norm-adaptable surface reports such a
    /// delta as unmatched (loud), never silently dropped.
    fn diff_patch_param_mut(&mut self, _path: &[&str], _part: DiffPatchPart) -> Option<&mut Array> {
        None
    }
}

/// Which half of a module a ComfyUI/lightx2v diff-patch delta addresses: `‹module›.diff` patches the
/// [`Weight`](DiffPatchPart::Weight), `‹module›.diff_b` the [`Bias`](DiffPatchPart::Bias). The
/// selector [`AdaptableHost::diff_patch_param_mut`] takes, so one path resolves both halves of e.g.
/// an affine LayerNorm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffPatchPart {
    /// `‹module›.diff` — the weight/gain parameter.
    Weight,
    /// `‹module›.diff_b` — the bias/offset parameter.
    Bias,
}

/// Prefix each of `host`'s [`AdaptableHost::adaptable_paths`] with `‹prefix›.` — the enumeration
/// analog of a parent's `["‹prefix›", rest @ ..] => sub.adaptable_mut(rest)` delegation, so a
/// composite host can build its full path list from its children's relative ones (sc-2618 kohya).
pub fn prefixed_paths(prefix: &str, host: &impl AdaptableHost) -> Vec<String> {
    host.adaptable_paths()
        .iter()
        .map(|p| format!("{prefix}.{p}"))
        .collect()
}

/// Install an adapter onto the [`AdaptableLinear`] addressed by `dotted` (e.g.
/// `"attention.to_q"`). Errors if the path resolves to no adaptable linear.
pub fn install_adapter(
    host: &mut impl AdaptableHost,
    dotted: &str,
    adapter: Adapter,
) -> Result<()> {
    let parts: Vec<&str> = dotted.split('.').collect();
    host.adaptable_mut(&parts)
        .ok_or_else(|| format!("no adaptable linear at path: {dotted}"))?
        .push(adapter);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{all_close, array_eq};

    #[derive(Clone, Copy, Debug)]
    enum ResidualPayloadKind {
        Lora,
        MaterializedLokr,
        StructuredLokr,
    }

    fn write_lazy_adapter_payload(path: &std::path::Path, value: f32) {
        let mut header = br#"{"a":{"dtype":"F32","shape":[8,2],"data_offsets":[0,64]},"b":{"dtype":"F32","shape":[2,8],"data_offsets":[64,128]},"delta":{"dtype":"F32","shape":[8,8],"data_offsets":[128,384]},"w1":{"dtype":"F32","shape":[2,2],"data_offsets":[384,400]},"w2":{"dtype":"F32","shape":[4,4],"data_offsets":[400,464]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = Vec::with_capacity(8 + header.len() + 464);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        for _ in 0..116 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn adapter_from_lazy_payload(path: &std::path::Path, kind: ResidualPayloadKind) -> Adapter {
        let weights = crate::weights::Weights::from_file(path).unwrap();
        match kind {
            ResidualPayloadKind::Lora => Adapter::Lora {
                a: weights.require("a").unwrap().clone(),
                b: weights.require("b").unwrap().clone(),
                scale: 1.0,
            },
            ResidualPayloadKind::MaterializedLokr => Adapter::Lokr {
                delta: weights.require("delta").unwrap().clone(),
                scale: 1.0,
            },
            ResidualPayloadKind::StructuredLokr => Adapter::LokrStructured {
                factors: LokrFactors {
                    w1: weights.require("w1").unwrap().clone(),
                    w2: weights.require("w2").unwrap().clone(),
                    a: 2,
                    b: 4,
                    c: 2,
                    d: 4,
                    scale: 1.0,
                },
            },
        }
    }

    #[test]
    fn adapter_payload_materialization_stays_inside_pin_for_every_residual_form() {
        for kind in [
            ResidualPayloadKind::Lora,
            ResidualPayloadKind::MaterializedLokr,
            ResidualPayloadKind::StructuredLokr,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let source = dir.path().join("adapter.safetensors");
            let replacement = dir.path().join("replacement.safetensors");
            write_lazy_adapter_payload(&source, 0.25);
            write_lazy_adapter_payload(&replacement, -0.75);
            let pin = crate::PinnedWeightsFile::pin(&source).unwrap();

            let result: crate::Result<()> = pin.read_unchanged(|path| {
                let adapter = adapter_from_lazy_payload(path, kind);
                std::fs::rename(&replacement, &source).unwrap();
                adapter.materialize()
            });
            let error = result.expect_err("A to B replacement must invalidate the adapter pin");
            match error {
                crate::Error::Unsupported(reason)
                    if reason.starts_with("artifact seal mismatch after load: ") => {}
                crate::Error::Unsupported(reason) => {
                    panic!("{kind:?}: unexpected artifact-seal reason: {reason}")
                }
                other => {
                    panic!("{kind:?}: expected a typed artifact-seal rejection, got: {other:?}")
                }
            }
        }
    }

    fn lokr_2x2() -> Array {
        reconstruct_lokr_delta(
            8.0,
            4.0,
            &[2, 2],
            Some(&Array::from_slice(&[0.5f32, 0.6], &[2, 1])),
            None,
            None,
            Some(&Array::from_slice(&[0.7f32, 0.8], &[1, 2])),
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap()
    }

    #[test]
    fn lokr_delta_stored_bf16() {
        assert_eq!(lokr_2x2().dtype(), Dtype::Bfloat16);
    }

    // ── sc-15265: adapter installation must not change the host Linear's output dtype ──────────
    //
    // A bf16 base with f32 LoRA/LoKr factors used to promote the WHOLE forward to f32, so
    // "install an adapter and set its scale to 0" was NOT the same as "don't install it"
    // (measured drift ≈1.9e-4 dense / 2.9e-4 Q4 e2e), and a *live* adapter silently widened
    // every downstream activation. The helpers below build a 64-wide (group-size-aligned, so the
    // same base can be quantized) bf16 host and f32 rank-4 factors — the exact shape of the bug.

    /// Deterministic `[rows, cols]` f32 array from a cheap analytic pattern.
    fn synth(rows: i32, cols: i32, seed: f32) -> Array {
        let n = (rows * cols) as usize;
        let v: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.017 + seed).sin() * 0.05)
            .collect();
        Array::from_slice(&v, &[rows, cols])
    }

    /// A bf16 host Linear (`[64, 64]`, no bias), bf16 activations `[4, 64]`, and f32 rank-4
    /// LoRA factors `a: [64, 4]`, `b: [4, 64]` — the promoting combination.
    fn bf16_host() -> (AdaptableLinear, Array, Array, Array) {
        let w = synth(64, 64, 0.3).as_dtype(Dtype::Bfloat16).unwrap();
        let x = synth(4, 64, 1.1).as_dtype(Dtype::Bfloat16).unwrap();
        (
            AdaptableLinear::dense(w, None),
            x,
            synth(64, 4, 2.7),
            synth(4, 64, 3.9),
        )
    }

    /// `true` iff the two arrays are bit-identical in dtype, shape, and every element's **raw bit
    /// pattern**.
    ///
    /// This deliberately does NOT use `array_eq`, which is *value* equality and therefore accepts
    /// `-0.0 == +0.0` (and, for that matter, rejects `NaN == NaN`). The contract these tests pin is
    /// byte identity — "installed at scale 0" produces the same bytes as "not installed" — and the
    /// sign bit is exactly where a value comparison would let a real regression through: a LoKr
    /// residual is built as `multiply(&factor2, scalar(0.0))`, which yields **signed** zeros, so
    /// `base(-0.0) + (+0.0) = +0.0` flips a sign bit that `array_eq` cannot see. Reinterpret through
    /// `view` (an unsigned integer of the same width) and compare those instead.
    fn bit_exact(got: &Array, want: &Array) -> bool {
        if got.dtype() != want.dtype() || got.shape() != want.shape() {
            return false;
        }
        let bits = |a: &Array| match a.dtype() {
            Dtype::Bfloat16 | Dtype::Float16 => a.view::<u16>().unwrap(),
            Dtype::Float32 => a.view::<u32>().unwrap(),
            Dtype::Float64 => a.view::<u64>().unwrap(),
            // Integer/bool dtypes have no redundant encodings — value equality IS bit equality.
            _ => a.clone(),
        };
        array_eq(bits(got), bits(want), false)
            .unwrap()
            .item::<bool>()
    }

    #[test]
    fn bit_exact_rejects_a_signed_zero_flip_that_value_equality_accepts() {
        // Pins the helper above (MINOR 5): the gates below are only as strong as this comparison.
        let neg = Array::from_slice(&[-0.0f32, 1.0], &[2]);
        let pos = Array::from_slice(&[0.0f32, 1.0], &[2]);
        assert!(
            array_eq(&neg, &pos, false).unwrap().item::<bool>(),
            "value equality cannot see the sign bit — which is why bit_exact must not use it"
        );
        assert!(!bit_exact(&neg, &pos), "bit_exact must see the sign bit");
        assert!(bit_exact(&neg, &neg.clone()));
    }

    #[test]
    fn scale_zero_lora_is_bit_exact_noop_on_a_bf16_dense_base() {
        let (mut lin, x, a, b) = bf16_host();
        let base = lin.forward(&x).unwrap();
        assert_eq!(base.dtype(), Dtype::Bfloat16, "host output is bf16");

        lin.push(Adapter::Lora {
            a: a.clone(),
            b: b.clone(),
            scale: 0.0,
        });
        let zero = lin.forward(&x).unwrap();
        assert_eq!(
            zero.dtype(),
            Dtype::Bfloat16,
            "installing an adapter must not widen the host's output dtype"
        );
        assert!(
            bit_exact(&zero, &base),
            "a scale-0 adapter must be bit-exact to no adapter at all"
        );

        // MUTATION: the same adapter at a live scale must actually change the output — otherwise
        // the bit-exactness above would be a false green from a dead residual.
        let mut live =
            AdaptableLinear::dense(synth(64, 64, 0.3).as_dtype(Dtype::Bfloat16).unwrap(), None);
        live.push(Adapter::Lora { a, b, scale: 1.0 });
        assert!(
            !bit_exact(&live.forward(&x).unwrap(), &base),
            "a live adapter must change the forward (residual is not identically zero)"
        );
    }

    /// A `[rows, cols]` f32 array whose every entry is `+INFINITY`.
    fn inf(rows: i32, cols: i32) -> Array {
        Array::from_slice(&vec![f32::INFINITY; (rows * cols) as usize], &[rows, cols])
    }

    #[test]
    fn a_disabled_adapters_residual_is_never_evaluated() {
        // sc-15265, MAJOR-1 gate. This is the ONLY test that discriminates the `is_disabled()`
        // short-circuit in `apply_adapters` from the narrowing cast beside it. Every other
        // scale-0 gate here (and the krea e2e tier sweep) stays green with the short-circuit
        // deleted, because `0 · finite = 0` and casting an exact-zero residual to the host dtype is
        // *already* bit-exact — the cast alone carries them.
        //
        // The discriminator is a residual that is NOT zero when you compute it: a factor of
        // `+INFINITY` makes the low-rank product non-finite, and `Inf · 0.0 = NaN`. So a scale-0
        // adapter whose forward is still bit-exact to the unadapted base proves the residual was
        // never formed at all — no arithmetic identity can rescue a NaN. Delete the
        // `if adapter.is_disabled() { continue; }` in `apply_adapters` and every assertion below
        // fails with a NaN output.
        //
        // This is not a synthetic-only worry: it is exactly the failure mode a third-party
        // checkpoint carrying an `Inf` weight would hit at the moment a user turns the adapter OFF.
        let (lin, x, _, b) = bf16_host();
        let base = lin.forward(&x).unwrap();

        // (1) LoRA — an `Inf` in `a`.
        let poison_lora = Adapter::Lora {
            a: inf(64, 4),
            b: b.clone(),
            scale: 0.0,
        };
        // The fixture must actually be a trap, or this whole test is vacuous.
        let raw = poison_lora.residual(&x).unwrap();
        assert!(
            !raw.is_finite().unwrap().all(None).unwrap().item::<bool>(),
            "fixture check: evaluating this residual must produce a non-finite value"
        );
        let mut poisoned = lin.clone();
        poisoned.push(poison_lora);
        assert!(
            bit_exact(&poisoned.forward(&x).unwrap(), &base),
            "a disabled LoRA's residual must never be evaluated (short-circuit deleted?)"
        );

        // (2) Materialized LoKr — an `Inf` in the delta.
        let mut poisoned = lin.clone();
        poisoned.push(Adapter::Lokr {
            delta: inf(64, 64).as_dtype(Dtype::Bfloat16).unwrap(),
            scale: 0.0,
        });
        assert!(
            bit_exact(&poisoned.forward(&x).unwrap(), &base),
            "a disabled LoKr's residual must never be evaluated"
        );

        // (3) Structured LoKr — an `Inf` in `w2`. This arm is why `LokrFactors` retains its
        // pre-bake `scale` (sc-15265): the scale is folded into `w2` at build time, so `Inf · 0.0`
        // bakes a NaN straight into the factor. Without a recoverable disabled flag there is no
        // way to skip it, and "turn the adapter off" would output NaN.
        let factors = build_lokr_factors(
            0.0,
            &[64, 64],
            Some(&synth(8, 8, 6.1)),
            None,
            None,
            Some(&inf(8, 8)),
            None,
            None,
            None,
            Dtype::Float32,
        )
        .unwrap()
        .expect("8×8 ⊗ 8×8 factors a 64×64 base");
        assert!(
            !factors
                .w2
                .is_finite()
                .unwrap()
                .all(None)
                .unwrap()
                .item::<bool>(),
            "fixture check: Inf · 0.0 must bake a NaN into w2"
        );
        let mut poisoned = lin.clone();
        poisoned.push(Adapter::LokrStructured { factors });
        assert!(
            bit_exact(&poisoned.forward(&x).unwrap(), &base),
            "a disabled structured LoKr's residual must never be evaluated"
        );

        // And `forward_upcast` runs the same accumulation, so it inherits the same guarantee.
        let upcast_base = lin.forward_upcast(&x).unwrap();
        let mut poisoned = lin.clone();
        poisoned.push(Adapter::Lora {
            a: inf(64, 4),
            b,
            scale: 0.0,
        });
        assert!(
            bit_exact(&poisoned.forward_upcast(&x).unwrap(), &upcast_base),
            "forward_upcast must short-circuit a disabled adapter too"
        );
    }

    #[test]
    fn scale_zero_lora_is_bit_exact_noop_on_a_packed_base() {
        for bits in [4, 8] {
            let (mut lin, x, a, b) = bf16_host();
            lin.quantize(bits, None).unwrap();
            let base = lin.forward(&x).unwrap();
            assert_eq!(base.dtype(), Dtype::Bfloat16, "q{bits} host output is bf16");

            lin.push(Adapter::Lora { a, b, scale: 0.0 });
            let zero = lin.forward(&x).unwrap();
            assert_eq!(
                zero.dtype(),
                Dtype::Bfloat16,
                "q{bits}: installing an adapter must not widen the host's output dtype"
            );
            assert!(
                bit_exact(&zero, &base),
                "q{bits}: a scale-0 adapter must be bit-exact to no adapter at all"
            );
        }
    }

    #[test]
    fn scale_zero_lokr_variants_are_bit_exact_noops_on_a_bf16_base() {
        // The two LoKr arms, both of which are pure regression pins rather than reproductions of
        // the sc-15265 promotion:
        //   * materialized LoKr casts its delta to the ACTIVATION dtype in `Adapter::residual`, so
        //     its residual was already bf16 on a bf16 host — this arm never promoted;
        //   * structured LoKr builds its factors at an explicit `out_dtype` (f32 here), but
        //     `LokrFactors::residual` likewise casts them to the activation dtype, so this arm
        //     never promoted either. Building it f32 over a bf16 host does NOT reproduce the bug —
        //     the arm is exercised only to pin that it *stays* a bit-exact no-op at `scale = 0`.
        // (The arm that actually promoted is LoRA, whose factors are used at their file dtype; see
        // `live_adapter_keeps_the_host_output_dtype_and_matches_a_cast_residual`.)
        let (lin, x, _, _) = bf16_host();
        let base = lin.forward(&x).unwrap();

        let mut mat = lin.clone();
        mat.push(Adapter::Lokr {
            delta: synth(64, 64, 5.5).as_dtype(Dtype::Bfloat16).unwrap(),
            scale: 0.0,
        });
        assert!(
            bit_exact(&mat.forward(&x).unwrap(), &base),
            "a scale-0 materialized LoKr must be bit-exact to no adapter"
        );

        let factors = build_lokr_factors(
            0.0,
            &[64, 64],
            Some(&synth(8, 8, 6.1)),
            None,
            None,
            Some(&synth(8, 8, 7.3)),
            None,
            None,
            None,
            Dtype::Float32,
        )
        .unwrap()
        .expect("8×8 ⊗ 8×8 factors a 64×64 base");
        let mut structured = lin.clone();
        structured.push(Adapter::LokrStructured { factors });
        let got = structured.forward(&x).unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Bfloat16,
            "a structured LoKr must not widen the host's output dtype"
        );
        assert!(
            bit_exact(&got, &base),
            "a scale-0 structured LoKr must be bit-exact to no adapter"
        );
    }

    #[test]
    fn live_adapter_keeps_the_host_output_dtype_and_matches_a_cast_residual() {
        // Half 2 (sc-15265): the widening is NOT confined to scale 0. A live f32-factor adapter
        // over a bf16 host must land its residual in the host dtype — `bf16(base) + bf16(residual)`
        // — rather than promoting the whole downstream chain to f32.
        let (mut lin, x, a, b) = bf16_host();
        let base = lin.forward(&x).unwrap();
        let adapter = Adapter::Lora { a, b, scale: 0.75 };
        let residual = adapter.residual(&x).unwrap();
        assert_eq!(
            residual.dtype(),
            Dtype::Float32,
            "f32 factors still promote the RESIDUAL itself (Adapter::residual is unchanged)"
        );

        lin.push(adapter);
        let got = lin.forward(&x).unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Bfloat16,
            "a live adapter must not widen the host's output dtype"
        );
        let want = add(&base, residual.as_dtype(Dtype::Bfloat16).unwrap()).unwrap();
        assert!(
            bit_exact(&got, &want),
            "the forward must be base + host-dtype(residual)"
        );
    }

    #[test]
    fn a_training_stack_keeps_its_pre_sc15265_numerics() {
        // sc-15265 rule 0. The trainers ship f32 master factors over bf16 block streams
        // (`install_training_lora` passes `dtype = None`); narrowing that residual would round the
        // adapter's whole contribution away while it is still small, so a training stack opts out
        // of BOTH rules. This pins that: `set_training_adapters` must reproduce the pre-fix
        // `add(base, residual)` exactly — promotion and all — while the same adapters installed the
        // inference way must not.
        let (lin, x, a, b) = bf16_host();
        let base = lin.forward(&x).unwrap();
        let adapter = Adapter::Lora {
            a: a.clone(),
            b: b.clone(),
            scale: 1.0,
        };
        let residual = adapter.residual(&x).unwrap();
        assert_eq!(residual.dtype(), Dtype::Float32);

        let mut train = lin.clone();
        train.set_training_adapters(vec![adapter]);
        let got = train.forward(&x).unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Float32,
            "a training stack must keep the promoted (pre-fix) accumulation dtype"
        );
        assert!(
            bit_exact(&got, &add(&base, &residual).unwrap()),
            "a training stack must be bit-identical to the pre-sc-15265 add"
        );

        // Rule 1 is opted out of too, so a training install is untouched even at scale 0 — an f32
        // zero residual still promotes, exactly as it did before. No partial exemption.
        let mut train0 = lin.clone();
        train0.set_training_adapters(vec![Adapter::Lora { a, b, scale: 0.0 }]);
        assert_eq!(
            train0.forward(&x).unwrap().dtype(),
            Dtype::Float32,
            "the training exemption covers the scale-0 skip as well"
        );

        // And `set_adapters` (the inference / clear path) must NOT inherit the flag.
        let mut back = train;
        back.set_adapters(vec![Adapter::Lora {
            a: synth(64, 4, 2.7),
            b: synth(4, 64, 3.9),
            scale: 1.0,
        }]);
        assert_eq!(
            back.forward(&x).unwrap().dtype(),
            Dtype::Bfloat16,
            "set_adapters must clear the training flag, not leave it latched"
        );
    }

    #[test]
    fn forward_upcast_also_keeps_its_own_output_dtype() {
        // `forward_upcast` widens the BASE to the activation dtype on purpose; the adapter must
        // still not widen it any further than that.
        let w = synth(64, 64, 0.3).as_dtype(Dtype::Bfloat16).unwrap();
        let x = synth(4, 64, 1.1).as_dtype(Dtype::Bfloat16).unwrap();
        let mut lin = AdaptableLinear::dense(w, None);
        let base = lin.forward_upcast(&x).unwrap();
        assert_eq!(base.dtype(), Dtype::Bfloat16);
        lin.push(Adapter::Lora {
            a: synth(64, 4, 2.7),
            b: synth(4, 64, 3.9),
            scale: 0.0,
        });
        let zero = lin.forward_upcast(&x).unwrap();
        assert!(
            bit_exact(&zero, &base),
            "forward_upcast: a scale-0 adapter must be a bit-exact no-op too"
        );
    }

    #[test]
    fn forward_upcast_uses_activation_dtype_without_mutating_storage() {
        let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let lin = AdaptableLinear::dense(weight, None);
        let x = Array::from_slice(&[0.25f32, 0.5], &[1, 2]);

        let got = lin.forward_upcast(&x).unwrap();
        let want = Array::from_slice(&[1.25f32, 2.75], &[1, 2]);

        assert_eq!(got.dtype(), Dtype::Float32);
        assert!(all_close(&got, &want, 1e-6, 1e-6, false)
            .unwrap()
            .item::<bool>());
        let LinearBase::Dense(base) = &lin.base else {
            panic!("expected dense base");
        };
        assert_eq!(base.weight.value.dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn reconstruct_lokr_delta_rejects_zero_rank() {
        // F-141 (sc-11129): a `rank = 0` metadata (alpha then defaults to rank) makes the scale
        // `0/0 = NaN`, which would bake into the reconstructed delta and NaN-poison every render. The
        // shared seam must reject it with a typed error so every consumer (SDXL/Kolors/InstantID/Wan)
        // inherits the guard, retiring the fixes-don't-travel class.
        let err = reconstruct_lokr_delta(
            0.0,
            0.0,
            &[2, 2],
            Some(&Array::from_slice(&[0.5f32, 0.6], &[2, 1])),
            None,
            None,
            Some(&Array::from_slice(&[0.7f32, 0.8], &[1, 2])),
            None,
            None,
            Dtype::Bfloat16,
        )
        .expect_err("rank 0 must be rejected, not produce a NaN delta");
        assert!(
            matches!(err, crate::Error::Msg(_)),
            "expected a typed Msg error"
        );
        // Regression: a well-formed alpha/rank still reconstructs (the happy path is unaffected).
        assert!(lokr_2x2().sum(None).unwrap().item::<f32>().is_finite());
    }

    #[test]
    fn build_lokr_factors_rejects_non_finite_scale() {
        // F-141 (sc-11129): the packed/deferred path receives the pre-derived `(alpha/rank)·strength`
        // scale, so a `rank = 0` yields a NaN scale that would bake into the structured `w2`. The seam
        // must reject a non-finite scale rather than install a NaN factor. (`LokrFactors` is not Debug,
        // so match rather than `expect_err`.)
        let result = build_lokr_factors(
            f32::NAN,
            &[2, 2],
            Some(&Array::from_slice(&[0.5f32, 0.6], &[2, 1])),
            None,
            None,
            Some(&Array::from_slice(&[0.7f32, 0.8], &[1, 2])),
            None,
            None,
            None,
            Dtype::Bfloat16,
        );
        assert!(
            matches!(result, Err(crate::Error::Msg(_))),
            "a non-finite scale must be rejected with a typed Msg error"
        );
    }

    #[test]
    fn scale_zero_lokr_is_bit_exact_noop() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let mut lin = AdaptableLinear::dense(w, None);
        let base = lin.forward(&x).unwrap();
        lin.push(Adapter::Lokr {
            delta: lokr_2x2(),
            scale: 0.0,
        });
        let out = lin.forward(&x).unwrap();
        assert!(array_eq(&out, &base, false).unwrap().item::<bool>());
    }

    #[test]
    fn lokr_residual_runs_in_activation_dtype() {
        // sc-2718: the f32 bug-workaround is gone (NAX 16-bit dense GEMM fixed at the toolchain
        // level, sc-2772). A LoKr residual now runs in the ACTIVATION dtype — bf16 on the bf16 path
        // — mirroring the fork's `scale · matmul(x, delta.astype(x.dtype).T)`. So a bf16-input LoKr
        // residual must (a) return bf16 and (b) match the f32 reference within bf16 rounding — NOT
        // diverge (which is what the old buggy bf16 GEMM produced and the f32 detour avoided).
        let delta = lokr_2x2(); // bf16
        let x32 = Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25, -1.0, 2.0], &[3, 2]);
        let lokr = Adapter::Lokr {
            delta: delta.clone(),
            scale: 0.5,
        };

        let got = lokr
            .residual(&x32.as_dtype(Dtype::Bfloat16).unwrap())
            .unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Bfloat16,
            "bf16-input LoKr residual runs in the activation dtype"
        );

        let want = multiply(
            matmul(&x32, delta.as_dtype(Dtype::Float32).unwrap().t()).unwrap(),
            scalar(0.5),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        assert!(
            all_close(&got, &want, 5e-2, 5e-2, false)
                .unwrap()
                .item::<bool>(),
            "bf16 LoKr residual diverged from the f32 reference (bf16 GEMM bug?)"
        );
    }

    #[test]
    fn lora_residual_is_fork_faithful_no_forced_dtype() {
        // sc-2718: LoRA factors keep their loaded dtype and the result is NOT cast back, replicating
        // the fork's `scale · matmul(matmul(x, lora_A), lora_B)` byte-for-byte.
        let a32 = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.1 - 0.4).collect::<Vec<_>>(),
            &[2, 4],
        );
        let b32 = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.05).collect::<Vec<_>>(),
            &[4, 2],
        );
        let x_bf16 = Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25, -1.0, 2.0], &[3, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        // f32 factors (the goldens' dtype): a bf16 `x` promotes the residual to f32 — and it is
        // byte-exact to the fork's `scale · (x·A)·B` (no forced dtype, no cast-back).
        let lora_f32 = Adapter::Lora {
            a: a32.clone(),
            b: b32.clone(),
            scale: 0.5,
        };
        let got_f32 = lora_f32.residual(&x_bf16).unwrap();
        assert_eq!(
            got_f32.dtype(),
            Dtype::Float32,
            "f32 factors promote the residual to f32 (fork-faithful, not forced)"
        );
        let want_f32 = multiply(
            matmul(matmul(&x_bf16, &a32).unwrap(), &b32).unwrap(),
            scalar(0.5),
        )
        .unwrap();
        assert!(
            array_eq(&got_f32, &want_f32, false).unwrap().item::<bool>(),
            "LoRA residual must be byte-exact to the fork's scale·(x·A)·B"
        );

        // bf16 factors: the residual runs bf16 — the `[seq,r]·[r,out]` (K=rank=4≤512, M=seq=3≥2)
        // shape the NAX build mis-ran before sc-2772 — and matches the f32 reference within bf16
        // rounding (NOT garbage), proving the GEMM bug is gone so the f32 detour is unneeded.
        let lora_bf16 = Adapter::Lora {
            a: a32.as_dtype(Dtype::Bfloat16).unwrap(),
            b: b32.as_dtype(Dtype::Bfloat16).unwrap(),
            scale: 0.5,
        };
        let got_bf16 = lora_bf16.residual(&x_bf16).unwrap();
        assert_eq!(
            got_bf16.dtype(),
            Dtype::Bfloat16,
            "bf16 factors keep the residual in the activation dtype"
        );
        let want_bf16 = multiply(
            matmul(
                matmul(x_bf16.as_dtype(Dtype::Float32).unwrap(), &a32).unwrap(),
                &b32,
            )
            .unwrap(),
            scalar(0.5),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        assert!(
            all_close(&got_bf16, &want_bf16, 5e-2, 5e-2, false)
                .unwrap()
                .item::<bool>(),
            "bf16 LoRA residual diverged from the f32 reference (bf16 GEMM bug?)"
        );
    }

    #[test]
    fn biased_dense_forward_is_fused_addmm() {
        // sc-2779: the biased dense base must be a FUSED `addmm(bias, x, Wᵀ)`, not `matmul`+`add`.
        // In bf16 the two differ (double-rounding), so feed bf16 activations and assert the forward
        // is bit-exact to `addmm` and bit-distinct from `matmul`+`add` — i.e. the fusion is real.
        let n = 4 * 64;
        let w = Array::from_slice(
            &(0..64 * 64)
                .map(|i| (i as f32 * 0.013).sin() * 0.05)
                .collect::<Vec<_>>(),
            &[64, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let bias = Array::from_slice(
            &(0..64)
                .map(|i| (i as f32 * 0.7).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let x = Array::from_slice(
            &(0..n)
                .map(|i| (i as f32 * 0.031).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[4, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        let lin = AdaptableLinear::dense(w.clone(), Some(bias.clone()));
        let got = lin.forward(&x).unwrap();

        let want_addmm = addmm(&bias, &x, w.t(), 1.0, 1.0).unwrap();
        assert!(
            array_eq(&got, &want_addmm, false).unwrap().item::<bool>(),
            "biased dense forward must be bit-exact to addmm(bias, x, Wᵀ)"
        );

        // And it must NOT be the double-rounding matmul+add (which is what the bug looked like).
        let matmul_add = add(matmul(&x, w.t()).unwrap(), &bias).unwrap();
        assert!(
            !array_eq(&got, &matmul_add, false).unwrap().item::<bool>(),
            "bf16 addmm should differ from matmul+add (double-rounding) — fusion not applied?"
        );
    }

    #[test]
    fn biased_dense_forward_f32_acts_match_matmul_add_bit_exact() {
        // sc-2779 golden-safety invariant: with f32 activations (the current Z-Image/Qwen/FLUX path,
        // even over bf16 weights), `addmm == matmul+add` bit-for-bit — nothing rounds to bf16
        // mid-op. This is why lifting the core to addmm cannot regress any current f32-act golden.
        let w = Array::from_slice(
            &(0..64 * 64)
                .map(|i| (i as f32 * 0.013).sin() * 0.05)
                .collect::<Vec<_>>(),
            &[64, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap(); // bf16 weights
        let bias = Array::from_slice(
            &(0..64)
                .map(|i| (i as f32 * 0.7).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let x = Array::from_slice(
            &(0..4 * 64)
                .map(|i| (i as f32 * 0.031).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[4, 64],
        ); // f32 activations

        let got = AdaptableLinear::dense(w.clone(), Some(bias.clone()))
            .forward(&x)
            .unwrap();
        let matmul_add = add(matmul(&x, w.t()).unwrap(), &bias).unwrap();
        assert!(
            array_eq(&got, &matmul_add, false).unwrap().item::<bool>(),
            "f32-activation addmm must be bit-exact to matmul+add (no golden regression)"
        );
    }

    #[test]
    fn merge_dense_delta_adds_to_weight_and_zero_is_noop() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);

        // A zero delta is a bit-exact no-op (`W + 0 == W`) — the scale-0 LoRA invariant.
        let mut lin = AdaptableLinear::dense(w.clone(), None);
        let base = lin.forward(&x).unwrap();
        lin.merge_dense_delta(&Array::from_slice(&[0.0f32; 4], &[2, 2]))
            .unwrap();
        assert!(array_eq(lin.forward(&x).unwrap(), &base, false)
            .unwrap()
            .item::<bool>());

        // A nonzero delta is exactly `(W + δ)·x`.
        let delta = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
        let mut lin2 = AdaptableLinear::dense(w.clone(), None);
        lin2.merge_dense_delta(&delta).unwrap();
        let want = AdaptableLinear::dense(add(&w, &delta).unwrap(), None)
            .forward(&x)
            .unwrap();
        assert!(array_eq(lin2.forward(&x).unwrap(), &want, false)
            .unwrap()
            .item::<bool>());

        // Merging into a quantized base is rejected (must merge before quantization).
        let mut lin3 = AdaptableLinear::dense(
            Array::from_slice(
                &(0..4096).map(|i| i as f32 * 1e-3).collect::<Vec<_>>(),
                &[64, 64],
            ),
            None,
        );
        lin3.quantize(8, None).unwrap();
        assert!(lin3
            .merge_dense_delta(&Array::from_slice(&[0.0f32; 4096], &[64, 64]))
            .is_err());
    }

    #[test]
    fn conv_lora_delta_one_by_one_matches_hand_fold() {
        // sc-2919: a 1×1 conv LoRA (rank 2, in 2, out 2). down/up are `[*, *, 1, 1]`; the fused
        // delta is `Σ_r up[o,r]·down[r,i]`, scaled by alpha/rank. Hand-computed independently:
        //   down2 = [[1,2],[3,4]] (rank,in); up2 = [[5,6],[7,8]] (out,rank)
        //   δ[0,0]=5·1+6·3=23  δ[0,1]=5·2+6·4=34  δ[1,0]=7·1+8·3=31  δ[1,1]=7·2+8·4=46
        //   eff = alpha/rank = 4/2 = 2 → [[46,68],[62,92]]
        let down = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2, 1, 1]);
        let up = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2, 1, 1]);
        let delta = conv_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
        assert_eq!(delta.shape(), &[2, 2, 1, 1]);
        let want = Array::from_slice(&[46.0f32, 68.0, 62.0, 92.0], &[2, 2, 1, 1]);
        assert!(all_close(&delta, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn conv_lora_delta_kxk_rank1_broadcasts_spatial_kernel() {
        // sc-2919: a 3×3-shaped (here 2×2) conv LoRA with rank 1 reduces to `δ[o,i,y,x] =
        // up[o]·down[0,i,y,x]` — proving the spatial kernel is preserved (not collapsed). in=1, out=2.
        //   down[0,0,:,:] = [[1,2],[3,4]]; up = [10, 20]
        //   δ[0] = 10·[1,2,3,4] = [10,20,30,40];  δ[1] = 20·[...] = [20,40,60,80]
        let down = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let up = Array::from_slice(&[10.0f32, 20.0], &[2, 1, 1, 1]);
        let delta = conv_lora_delta(&down, &up, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(delta.shape(), &[2, 1, 2, 2]);
        let want = Array::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 20.0, 40.0, 60.0, 80.0],
            &[2, 1, 2, 2],
        );
        assert!(all_close(&delta, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
        // The user scale composes multiplicatively (scale 0 ⇒ a zero delta ⇒ no-op merge).
        let zero = conv_lora_delta(&down, &up, 1.0, 1.0, 0.0).unwrap();
        assert!(
            array_eq(&zero, Array::zeros::<f32>(&[2, 1, 2, 2]).unwrap(), false)
                .unwrap()
                .item::<bool>()
        );
    }

    #[test]
    fn conv_lora_delta_rejects_non_4d_factors() {
        // F-006: a malformed conv LoRA with 2-D factors must surface a typed error, not panic on the
        // `ds[2]`/`ds[3]` slice. And a rank mismatch between the factors is rejected too.
        let down2d = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let up = Array::from_slice(&[10.0f32, 20.0], &[2, 1, 1, 1]);
        let err = conv_lora_delta(&down2d, &up, 1.0, 1.0, 1.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("4-D factors"), "got: {err}");

        // 4-D but mismatched rank: down rank 1, up rank 2.
        let down = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
        let up_bad = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2, 1, 1]);
        let err = conv_lora_delta(&down, &up_bad, 1.0, 1.0, 1.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("rank mismatch"), "got: {err}");
    }

    #[test]
    fn merge_conv_delta_transposes_nchw_and_adds() {
        // sc-2919: NHWC weight `[out=1, kH=1, kW=1, in=2]` = [1, 2]; a 1×1 NCHW delta
        // `[out=1, in=2, kH=1, kW=1]` = [0.5, 0.25] transposes to NHWC [0.5, 0.25] and adds → [1.5, 2.25].
        let w = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
        let mut conv = AdaptableConv2d::new(w.clone(), None);
        let delta = Array::from_slice(&[0.5f32, 0.25], &[1, 2, 1, 1]);
        conv.merge_conv_delta(&delta).unwrap();
        let want = Array::from_slice(&[1.5f32, 2.25], &[1, 1, 1, 2]);
        assert!(array_eq(conv.weight(), &want, false)
            .unwrap()
            .item::<bool>());

        // A zero delta is a bit-exact no-op.
        let mut conv2 = AdaptableConv2d::new(w.clone(), None);
        conv2
            .merge_conv_delta(&Array::zeros::<f32>(&[1, 2, 1, 1]).unwrap())
            .unwrap();
        assert!(array_eq(conv2.weight(), &w, false).unwrap().item::<bool>());
    }

    #[test]
    fn merge_conv_delta_kxk_zero_is_noop_and_nonzero_lands_in_nhwc() {
        // sc-2919 regression: with a k×k (here 3×3, out≠in) kernel the NCHW→NHWC transpose is a real
        // permutation (not the trivial 1×1 case), so a zero delta must STILL be a bit-exact no-op — i.e.
        // the merge must not permute/scramble the weight. NHWC weight [out=2, kH=3, kW=3, in=4].
        let n = 2 * 3 * 3 * 4;
        let wv: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 0.3).collect();
        let w = Array::from_slice(&wv, &[2, 3, 3, 4]);

        let mut conv = AdaptableConv2d::new(w.clone(), None);
        conv.merge_conv_delta(&Array::zeros::<f32>(&[2, 4, 3, 3]).unwrap())
            .unwrap();
        assert_eq!(
            conv.weight().as_slice::<f32>(),
            w.as_slice::<f32>(),
            "a zero k×k conv delta must be a bit-exact no-op (no permutation/scramble)"
        );

        // A nonzero NCHW delta must land at the matching NHWC position: δ_nchw[o,i,y,x] adds to
        // weight_nhwc[o,y,x,i]. Put a single spike at nchw (o=1,i=2,y=0,x=2) → nhwc (1,0,2,2).
        let mut spike = vec![0f32; n];
        // nchw flat index for [2,4,3,3] at (1,2,0,2) = ((1*4+2)*3+0)*3+2 = 56.
        spike[56] = 5.0;
        let mut conv2 = AdaptableConv2d::new(w.clone(), None);
        conv2
            .merge_conv_delta(&Array::from_slice(&spike, &[2, 4, 3, 3]))
            .unwrap();
        // nhwc flat index for [2,3,3,4] at (1,0,2,2) = ((1*3+0)*3+2)*4+2 = 46.
        let nhwc_idx = 46usize;
        let got = conv2.weight().as_slice::<f32>();
        for (j, (&g, &b)) in got.iter().zip(&wv).enumerate() {
            let want = if j == nhwc_idx { b + 5.0 } else { b };
            assert_eq!(
                g, want,
                "conv delta landed at wrong NHWC index (got change at {j})"
            );
        }
    }

    #[test]
    fn stacks_mixed_lora_and_lokr_summing_residuals() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let mut lin = AdaptableLinear::dense(w, None);
        let base = lin.forward(&x).unwrap();
        let lora = Adapter::Lora {
            a: Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]),
            b: Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]),
            scale: 0.5,
        };
        let lokr = Adapter::Lokr {
            delta: lokr_2x2(),
            scale: 0.7,
        };
        let lora_r = lora.residual(&x).unwrap();
        let lokr_r = lokr.residual(&x).unwrap();
        lin.push(lora);
        lin.push(lokr);
        assert_eq!(lin.adapters().len(), 2);
        let expected = add(add(&base, &lora_r).unwrap(), &lokr_r).unwrap();
        assert!(
            all_close(lin.forward(&x).unwrap(), &expected, 1e-4, 1e-2, false)
                .unwrap()
                .item::<bool>()
        );
    }

    // ---- Structured (deferred) LoKr — the vec-trick (sc-10050) ------------------------------------

    /// The vec-trick `residual` must equal the materialized-delta `residual` for a **full** `w1⊗w2`
    /// LoKr, proving `vec(w1·X·w2ᵀ) == x·(w1⊗w2)ᵀ` under row-major kron ordering — the core identity
    /// this story rests on. `out = a·b = 2·3 = 6`, `in = c·d = 4·5 = 20`.
    #[test]
    fn structured_lokr_full_matches_materialized_delta() {
        let (a, b, c, d) = (2i32, 3, 4, 5);
        let (out, inp) = (a * b, c * d);
        let w1 = Array::from_slice(
            &(0..a * c)
                .map(|i| (i as f32 * 0.11).sin())
                .collect::<Vec<_>>(),
            &[a, c],
        );
        let w2 = Array::from_slice(
            &(0..b * d)
                .map(|i| (i as f32 * 0.07).cos())
                .collect::<Vec<_>>(),
            &[b, d],
        );
        let scale = 0.9f32;
        // Materialized reference delta and its residual `x·ΔWᵀ`.
        let delta = reconstruct_lokr_delta_scaled(
            scale,
            &[out, inp],
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
            Dtype::Float32,
        )
        .unwrap();
        // Structured factors — no [out,in] tensor built.
        let factors = build_lokr_factors(
            scale,
            &[out, inp],
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
            Dtype::Float32,
        )
        .unwrap()
        .expect("a plain linear LoKr is deferrable");
        // The built factors are the SMALL Kronecker factors, never the [out,in] delta.
        assert_eq!(factors.w1.shape(), &[a, c]);
        assert_eq!(factors.w2.shape(), &[b, d]);

        let x = Array::from_slice(
            &(0..2 * inp)
                .map(|i| (i as f32 * 0.013 - 0.5).sin())
                .collect::<Vec<_>>(),
            &[2, inp],
        );
        let want = Adapter::Lokr { delta, scale: 1.0 }.residual(&x).unwrap();
        let got = factors.residual(&x).unwrap();
        assert_eq!(got.shape(), &[2, out]);
        // Metal reduced-precision matmul (~1e-3 relative) — the vec-trick and the materialized delta
        // take different matmul shapes, so compare at the Metal tolerance, not bit-exact.
        assert!(
            all_close(&got, &want, 2e-2, 2e-3, false)
                .unwrap()
                .item::<bool>(),
            "structured LoKr residual must match the materialized-delta residual"
        );
    }

    /// Same identity for a **decomposed** LoKr (`w1_a·w1_b`, `w2_a·w2_b`) — the low-rank inner factors
    /// are materialized only as the SMALL `[a,c]`/`[b,d]` Kronecker factors, never the `[out,in]` delta.
    #[test]
    fn structured_lokr_decomposed_matches_materialized_delta() {
        let (a, b, c, d, r) = (3i32, 2, 5, 4, 2);
        let (out, inp) = (a * b, c * d);
        let mk = |rows: i32, cols: i32, seed: f32| {
            Array::from_slice(
                &(0..rows * cols)
                    .map(|i| (i as f32 * 0.09 + seed).sin() * 0.3)
                    .collect::<Vec<_>>(),
                &[rows, cols],
            )
        };
        let (w1a, w1b) = (mk(a, r, 0.1), mk(r, c, 0.2)); // w1 = [a,c]
        let (w2a, w2b) = (mk(b, r, 0.3), mk(r, d, 0.4)); // w2 = [b,d]
        let scale = 1.3f32;
        let delta = reconstruct_lokr_delta_scaled(
            scale,
            &[out, inp],
            None,
            Some(&w1a),
            Some(&w1b),
            None,
            None,
            Some(&w2a),
            Some(&w2b),
            Dtype::Float32,
        )
        .unwrap();
        let factors = build_lokr_factors(
            scale,
            &[out, inp],
            None,
            Some(&w1a),
            Some(&w1b),
            None,
            None,
            Some(&w2a),
            Some(&w2b),
            Dtype::Float32,
        )
        .unwrap()
        .expect("a decomposed linear LoKr is deferrable");
        assert_eq!(factors.w1.shape(), &[a, c]);
        assert_eq!(factors.w2.shape(), &[b, d]);

        let x = Array::from_slice(
            &(0..inp)
                .map(|i| (i as f32 * 0.02).cos())
                .collect::<Vec<_>>(),
            &[1, inp],
        );
        let want = Adapter::Lokr { delta, scale: 1.0 }.residual(&x).unwrap();
        let got = factors.residual(&x).unwrap();
        assert!(
            all_close(&got, &want, 2e-2, 2e-3, false)
                .unwrap()
                .item::<bool>(),
            "decomposed structured LoKr must match the materialized delta"
        );
    }

    /// A **tucker/CP** `w2` (`w2_t2`, conv-only) is NOT deferrable via the 2-D vec-trick → `Ok(None)`,
    /// so the caller can fall back to materialization (dense) or a clear error (packed). Never a panic.
    #[test]
    fn structured_lokr_tucker_is_not_deferrable() {
        let t2 = Array::from_slice(&[0.1f32; 2 * 2 * 3 * 3], &[2, 2, 3, 3]);
        let w2a = Array::from_slice(&[0.2f32; 2 * 4], &[2, 4]);
        let w2b = Array::from_slice(&[0.3f32; 2 * 5], &[2, 5]);
        let w1 = Array::from_slice(&[0.4f32; 3 * 4], &[3, 4]);
        let got = build_lokr_factors(
            1.0,
            &[24, 180], // conv-ish; the tucker guard fires before any shape check
            Some(&w1),
            None,
            None,
            None,
            Some(&t2),
            Some(&w2a),
            Some(&w2b),
            Dtype::Float32,
        )
        .unwrap();
        assert!(
            got.is_none(),
            "tucker/CP LoKr must be reported non-deferrable"
        );
    }

    /// End-to-end on a **quantized (Q8)** base: a structured LoKr applies with a non-degenerate,
    /// finite output and the base stays packed — the memory-safe packed-tier path (sc-10050). The
    /// residual is computed only from the small factors (asserted above), so no `[out,in]` is built.
    #[test]
    fn structured_lokr_applies_on_quantized_base() {
        let (a, b, c, d) = (4i32, 8, 8, 8); // out=32, in=64 (in % group_size(64)==0)
        let (out, inp) = (a * b, c * d);
        let wv: Vec<f32> = (0..out * inp).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut lin = AdaptableLinear::dense(Array::from_slice(&wv, &[out, inp]), None);
        lin.quantize(8, Some(64)).unwrap();
        assert!(lin.is_quantized());

        let w1 = Array::from_slice(
            &(0..a * c)
                .map(|i| (i as f32 * 0.05).sin() * 0.1)
                .collect::<Vec<_>>(),
            &[a, c],
        );
        let w2 = Array::from_slice(
            &(0..b * d)
                .map(|i| (i as f32 * 0.03).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[b, d],
        );
        let factors = build_lokr_factors(
            0.8,
            &lin.base_shape(),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap()
        .unwrap();
        lin.push(Adapter::LokrStructured { factors });

        let x = Array::from_slice(
            &(0..inp)
                .map(|i| (i as f32 * 0.01).sin())
                .collect::<Vec<_>>(),
            &[1, inp],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, out]);
        assert!(lin.is_quantized(), "base must stay packed after LoKr apply");
        // Non-degenerate + finite.
        let ys = y.as_dtype(Dtype::Float32).unwrap();
        let sum_abs = ys.abs().unwrap().sum(None).unwrap().item::<f32>();
        assert!(
            sum_abs.is_finite() && sum_abs > 0.0,
            "output must be finite and non-zero"
        );
    }
}
